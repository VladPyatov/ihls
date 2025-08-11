from typing import Tuple, Union

import torch as th

from irrn.modules.backbones import KPBackbone, ConvBackbone, L2NormBackbone
from irrn.modules.layers import LinearSystemStepLayer, ImplicitLayer
from irrn.operators import LearnableKPNConvOperator, LinearOperatorBase, IdentityOperator, LearnableCNNOperator, \
    LearnableNumberOperator, IRLSSystemOperatorHandler
from irrn.solvers import BatchedConjugateGradientSolver
from irrn.solvers.recurrent import BatchedRecurrentSolver
from irrn.utils import MultiVector

IMAGES_NUM_CHANNELS = 1
CG_MAX_ITER = 250
CG_RTOL = 1e-3
CG_ATOL = 1e-5
CG_RESTARTS_ITER = 25
CG_MAX_ITER_BACKWARD = 500
CG_RTOL_BACKWARD = 1e-3
CG_ATOL_BACKWARD = 1e-5
CG_RESTARTS_ITER_BACKWARD = 25
REG_CONV_BACKBONE_NUM_OUT_FEATURES = [8, 16, 32, 64, 128]
REG_CONV_BACKBONE_KERNEL_SIZE = [5, 3, 3, 3, 3]
REG_CONV_BACKBONE_PADDING_SIZE = [0] * 5
REG_CONV_BACKBONE_STRIDES = [1] * 5
REG_CONV_BACKBONE_PADDING_TYPE = None
USE_CIRCULANT_PRECOND = False
USE_CIRCULANT_PRECOND_BACKWARD = False
MAX_STEPS_NUM = 30
DEVICE = 'cuda'


class RegFidWithNoiseNoiseKPNOperator(LearnableKPNConvOperator):
    def prepare_for_restoration(self, noise_std=None, input=None, **other_kwargs) -> 'LearnableKPNConvOperator':
        vector = input / noise_std[..., None, None]
        self.update_effective_kernel(vector)
        return self


class IRLSRecurrentStepLayer(LinearSystemStepLayer):
    def residual_with_grad_residual_operator(self, degraded: th.Tensor, *solution: th.Tensor
                                             ) -> Tuple[Union[th.Tensor, MultiVector], LinearOperatorBase]:
        return self.system.residual_with_grad_residual_operator(degraded, *solution)


class IRLSRecurrentSolver(BatchedRecurrentSolver):
    def residual_with_grad_residual_operator(self, degraded: th.Tensor, *solution: th.Tensor
                                             ) -> Tuple[Union[th.Tensor, MultiVector], LinearOperatorBase]:
        return self.step_module.residual_with_grad_residual_operator(degraded, *solution)


class ImplicitLayerBoundsExtension(ImplicitLayer):
    def forward(self, degraded_images: th.Tensor, **fwd_solver_kwargs) -> Union[th.Tensor, Tuple[th.Tensor]]:
        pad_op = self.solver_forward.step_module.system.reg_images.pad_operator
        degraded_images_pad = pad_op.symmetric_pad(degraded_images)
        ret = super(ImplicitLayerBoundsExtension, self).forward(degraded_images_pad, **fwd_solver_kwargs)
        ret = (pad_op._zero_pad_transpose(ret[0]), )
        return ret


def get_implicit_irls_denoiser():
    degradation_operator = IdentityOperator()

    kpn_bbne = KPBackbone(num_in_channels=IMAGES_NUM_CHANNELS, num_out_channels=8, filter_size=3)
    reg_fidelity_operator = RegFidWithNoiseNoiseKPNOperator(kpn_bbne, observation_keyword='input', latent_index=None,
                                                            padding_mode=None, learnable=True)

    reg_images_backbone = ConvBackbone(out_features_per_layer=REG_CONV_BACKBONE_NUM_OUT_FEATURES,
                                       kernel_size_per_layer=REG_CONV_BACKBONE_KERNEL_SIZE,
                                       padding_per_layer=REG_CONV_BACKBONE_PADDING_SIZE,
                                       strides_per_layer=REG_CONV_BACKBONE_STRIDES,
                                       num_in_features=IMAGES_NUM_CHANNELS)
    features_extractor_operator = LearnableCNNOperator(reg_images_backbone, padding_mode=None, learnable=True)

    weight_decay_operator = LearnableNumberOperator(th.ones(1) * 1e-3, learnable=False, function=lambda x: x)

    noise_backbone = None
    reg_backbone = L2NormBackbone()

    system = IRLSSystemOperatorHandler(degradation_operator, reg_fidelity_operator, features_extractor_operator,
                                       weight_decay_operator, reg_backbone, weight_noise_backbone=noise_backbone,
                                       zero_step_system=None, use_weight_noise_scaling_coef=False,
                                       use_circulant_precond=USE_CIRCULANT_PRECOND,
                                       use_circulant_precond_backward=USE_CIRCULANT_PRECOND_BACKWARD)
    linsys_solver = BatchedConjugateGradientSolver(rtol=CG_RTOL, atol=CG_ATOL,
                                                   rtol_backward=CG_RTOL_BACKWARD, atol_backward=CG_ATOL_BACKWARD,
                                                   max_iter=CG_MAX_ITER, max_iter_backward=CG_MAX_ITER_BACKWARD,
                                                   restarts_iter=CG_RESTARTS_ITER,
                                                   restarts_iter_backward=CG_RESTARTS_ITER_BACKWARD, verbose=False)

    system_layer = IRLSRecurrentStepLayer(system, linsys_solver)

    solver_forward = IRLSRecurrentSolver(system_layer, max_steps_num=MAX_STEPS_NUM, initialization_fn=lambda x: (x,))
    irls_impl_module = ImplicitLayerBoundsExtension(solver_forward, linsys_solver)
    return irls_impl_module


def main():
    irls_impl_module = get_implicit_irls_denoiser().to(device=DEVICE)
    noisy_images = th.rand(4, 1, 128, 128).to(device=DEVICE)
    stds = th.rand(4, 1).to(device=DEVICE)
    res = irls_impl_module(noisy_images, input=noisy_images, noise_std=stds)
    loss = res[0].sum()
    loss.backward()


if __name__ == "__main__":
    main()
