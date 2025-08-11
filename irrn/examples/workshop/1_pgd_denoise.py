from typing import Tuple

import numpy as np
from torch import nn
import torch as th

from irrn.modules.layers import ImplicitLayer
from irrn.solvers import BatchedRecurrentSolver


def signum(input):
    out = - th.ones_like(input)
    out[input > 0] = 1
    return out


class SphericalProjection(th.autograd.Function):
    r""" Y = SphericalProjection(X, Z, ALPHA, STDN) computes the projection
    of the input X to a sphere which is centered at Z and has a radius equal
    to EPSILON. This projection is mathematically defined as the solution to
    the following optimization problem:

    Y = prox_IC(Z, EPSILON){X} = argmin ||Y-X||^2
                               ||(Y-Z)|| <= EPSILON

                               = argmin ||Y-X||^2 + i_C(Z, EPSILON){Y}
                                   Y

                       { 0 if ||Y-Z|| <= EPSILON
    i_C(Z,EPSILON){Y}= {
                       { +inf if ||Y-Z|| > EPSILON

    X, Z and Y are tensors of size N x C x H x W, ALPHA is a scalar tensor,
    STDN is either a scalar tensor or a tensor with N elements and
    EPSILON = exp(ALPHA)*V*STDN, where V = sqrt(H*W*C-1).

    Y = Z + K* (X-Z) where K = EPSILON / max(||X-Z||, EPSILON);

    DLDX, DLDA = SphericalProjection.backward(DLDY) computes the derivatives of
    the layer projected onto DLDY. DLDX has the same dimensions as X and DLDA
    the same dimensions as ALPHA.

    DLDX = K ( I - (X-Z)*(X-Z)^T/ max(||X-Z||, EPSILON)^2) * R) * DLDY

    where R = (sign(||X-Z||-EPSILON)+1)/2. For the sign function we use the
    convention that it holds: sign(0) = -1.

    DLDA = B*(X-Z)^T*DLDY

    where B = [ EPSILON *{ 2*max(||X-Z||, EPSILON)-
    EPSILON*(1-sgn(||X-Z||-EPSILON)) } ] / [ 2*max(||X-Z||, EPSILON)^2 ]"""

    @staticmethod
    def forward(ctx, x, z, alpha, stdn):

        assert(x.dim() == 4 and z.dim() == 4), \
            "Input and other are expected to be 4-D tensors."

        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
            "either None or a tensor of size 1."

        N = np.sqrt(x[0].numel()-1)
        batch = x.size(0)

        assert(stdn.numel() == 1 or stdn.numel() == batch), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number."

        assert(all(stdn.view(-1) > 0)), \
            "The noise standard deviations must be positive."

        stdn = stdn.view(-1, 1, 1, 1)

        if alpha is None:
            alpha = th.Tensor([0]).type_as(stdn)

        epsilon = stdn.mul(alpha.exp())*N

        diff = x.add(-z)
        diff_norm = diff.view(batch, -1).norm(p=2, dim=1).view(batch, 1, 1, 1)
        max_norm = diff_norm.max(epsilon)

        # No need to save the variables during inference
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(diff, diff_norm, max_norm, epsilon)

        return z + diff.mul(epsilon).div(max_norm)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_alpha = None

        diff, diff_norm, max_norm, epsilon = ctx.saved_variables
        batch = grad_output.size(0)

        if ctx.needs_input_grad[2]:
            k = epsilon.div(diff_norm)
            k[k >= 1] = 0
            grad_alpha = grad_output.mul(diff).mul(k).sum().view(1)

        if ctx.needs_input_grad[0]:
            r = (signum(diff_norm-epsilon)+1)/2
            r = r.div(max_norm.pow(2))
            grad_input = grad_output.mul(epsilon.div(max_norm))
            ip = grad_input.mul(diff).view(batch, -1).sum(1).view(-1, 1, 1, 1)
            ip = ip.mul(r)
            grad_input -= diff.mul(ip)

        return grad_input, None, grad_alpha, None


class CNNStepModule(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(CNNStepModule, self).__init__()
        self.backbone = backbone
        self.projection_fn = SphericalProjection.apply
        self.sigma = None
        self.alpha = nn.Parameter(th.ones(1)*1e-2)
        self.solution_previous = None
        self.solution_list = []

    def extrapolate(self, solution_new, solution_previous, step_idx):
        return solution_new

    def forward(self, degraded: th.Tensor, latent: th.Tensor, call_id=None) -> Tuple[th.Tensor]:
        res = self.backbone(latent)
        res = self.projection_fn(res, degraded, self.alpha, self.sigma)
        res = self.extrapolate(res, latent, call_id)
        self.solution_previous = res
        return (res, )

    def prepare_for_restoration(self, noise_std: th.Tensor = None, **other_kwargs) -> None:
        self.sigma = noise_std

    @property
    def tensors_for_grad(self):
        return tuple(self.parameters())


MAX_STEP_NUM = 10
RTOL = 1e-3
ATOL = 1e-5


def main():
    backbone = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.ReLU(), nn.Conv2d(3, 3, 3, padding=1), nn.ReLU())
    step_module = CNNStepModule(backbone)
    forward_solver = BatchedRecurrentSolver(step_module, max_steps_num=MAX_STEP_NUM, atol=ATOL, rtol=RTOL,
                                            initialization_fn=lambda x: (x, ), verbose=True)
    module = ImplicitLayer(forward_solver, solver_backward=None, jacobian_free_backward=True)
    degraded = th.rand(1, 3, 16, 16)
    noise_std = th.rand(1, 1)
    result = module(degraded, noise_std=noise_std)
    result[0].sum().backward()
    return


if __name__ == '__main__':
    main()
