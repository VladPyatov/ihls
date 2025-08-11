from typing import Callable

import pytest
import torch as th
from torch import nn

from irrn.functional import SymEigFunction
from irrn.modules.backbones import ConvBackbone, RegBackbone
from irrn.modules.layers import QMRecurrentLayer
from irrn.operators import LearnableConvolutionOperator, LearnableNumberOperator, LearnableCNNOperator, \
    AutogradGradLinearOperator, LearnableMatMulOperator, LearnableLinearOperatorBase
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator
from irrn.operators.linsys.irls import IRLSSystemOperatorHandler
from irrn.solvers import BatchedConjugateGradientSolver


class IRLSMatMulOperator(LearnableMatMulOperator):
    def reshape_before(self, vector: th.Tensor) -> th.Tensor:  # B, Q, C, H, W
        return vector.permute(0, 3, 4, 2, 1)  # B, H, W, C, Q

    def reshape_after(self, vector: th.Tensor) -> th.Tensor:  # B, H, W, C, Q
        return vector.permute(0, 4, 3, 1, 2)  # B, Q, C, H, W


class IRLSSystemOperatorMatMulHandler(IRLSSystemOperatorHandler):
    @staticmethod
    def linearized_reg_operator(matrix: th.Tensor) -> LearnableLinearOperatorBase:
        """
        This method creates linear operator parameterized by the output of regularization backbone.
        By default returns diagonal operator, which assumes incoming tensor as a vector.

        :param matrix: output from a backbone to parametrize operator
        :return: linear operator parametrized by backbone output
        """
        reg_backbone_linear = IRLSMatMulOperator(matrix, side='left')
        return reg_backbone_linear


class SpectralSquaredFunctionWrapper(nn.Module):
    def __init__(self, spectral_function: Callable, symeig_tol: float = 1e-7, symeig_approx_order: int = 100,
                 min_gamma_value: float = 1e-6) -> None:
        super(SpectralSquaredFunctionWrapper, self).__init__()
        self.spectral_function = spectral_function
        self.symeig_tol = symeig_tol
        self.symeig_approx_order = symeig_approx_order
        self.eps = min_gamma_value
        self.identity = None

    def forward(self, x: th.Tensor, step_num: int = 0) -> th.Tensor:
        assert x.dim() == 5  # [B, Q, C, H, W]
        matrices = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, Q]
        sym_matrices = th.matmul(matrices, matrices.transpose(-1, -2))  # [B, H, W, C, C]
        gamma = self.eps
        if self.identity is None:
            dim = sym_matrices.shape[-1]
            self.identity = th.eye(dim, dtype=sym_matrices.dtype, device=sym_matrices.device)[None, None, None, ...]
        sym_matrices = sym_matrices + gamma*self.identity
        eigvals, eigvecs = SymEigFunction.apply(sym_matrices, self.symeig_tol)  # [B, H, W, C], [B, H, W, C, C]
        eigvals_processed = self.spectral_function(eigvals.permute(0, 3, 1, 2))  # [B, C/1, H, W]
        if eigvals_processed.shape[1] == 1:
            eigvals_processed = eigvals_processed.repeat(1, eigvecs.shape[-1], 1, 1)
        eigvals_processed = eigvals_processed.permute(0, 2, 3, 1)  # [B, H, W, C]
        sym_matrices = th.matmul(eigvecs, eigvals_processed.unsqueeze(-1) * eigvecs.transpose(-1, -2))  # [B, H, W, C, C]
        return sym_matrices


class TestIRLSSystem:
    batch_size = 2
    num_channels = 1
    image_size = 128
    kernel_size = 17
    pad_type = None
    solver_maxiter = 5000
    restarts_iter = 10
    device = 'cuda'
    dtype = th.float64
    irls_noise_std = 1
    reg_images_kernel_size = 11
    atol = 1e-7
    rtol = 1e-5

    def get_toy_irls_system(self, degradation_operator, use_precond, num_in_channels):
        reg_fidelity = LearnableConvolutionOperator(filter_size=3, filter_num_in_channels=num_in_channels,
                                                    learnable=False, mix_in_channels=False)
        reg_images = LearnableCNNOperator(backbone=ConvBackbone(out_features_per_layer=[8],
                                                                kernel_size_per_layer=[self.reg_images_kernel_size],
                                                                padding_per_layer=[0], strides_per_layer=[1],
                                                                num_in_features=num_in_channels), learnable=False,
                                          mix_in_channels=True)
        weight_decay = LearnableNumberOperator(th.ones(1) * 1e-4, function=lambda x: x, learnable=False)
        reg_backbone = self.get_reg_backbone()

        system = IRLSSystemOperatorHandler(degradation_operator, reg_fidelity, reg_images, weight_decay, reg_backbone,
                                           zero_step_system=None, use_weight_noise_scaling_coef=True,
                                           use_circulant_precond=use_precond)
        system = system.to(device=self.device, dtype=self.dtype)
        return system

    def get_toy_matmul_irls_system(self, degradation_operator, use_precond, num_in_channels):
        reg_fidelity = LearnableConvolutionOperator(filter_size=3, filter_num_in_channels=num_in_channels,
                                                    learnable=False, mix_in_channels=False)
        reg_images = LearnableCNNOperator(backbone=ConvBackbone(out_features_per_layer=[8],
                                                                kernel_size_per_layer=[self.reg_images_kernel_size],
                                                                padding_per_layer=[0], strides_per_layer=[1],
                                                                num_in_features=num_in_channels), learnable=False,
                                          mix_in_channels=False)
        weight_decay = LearnableNumberOperator(th.ones(1) * 1e-4, function=lambda x: x, learnable=False)
        reg_backbone = self.get_matmul_reg_backbone(num_in_channels)

        system = IRLSSystemOperatorMatMulHandler(degradation_operator, reg_fidelity, reg_images, weight_decay,
                                                 reg_backbone, zero_step_system=None,
                                                 use_weight_noise_scaling_coef=True,
                                                 use_circulant_precond=False)
        system = system.to(device=self.device, dtype=self.dtype)
        return system

    def init_matmul(self, use_precond, num_in_channels=None):
        if num_in_channels is None:
            num_in_channels = self.num_channels
        degradation_operator = ConvDecimateLinearDegradationOperator(padding_mode=None)
        kernels = \
            degradation_operator._init_kernels_gaussian(self.batch_size, 3).to(device=self.device, dtype=self.dtype)
        degradation_operator.init_with_parameters_(kernel=kernels)
        solver = BatchedConjugateGradientSolver(max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter,
                                                atol=1e-9, rtol=1e-6, verbose=True)
        layer = QMRecurrentLayer(self.get_toy_matmul_irls_system(degradation_operator, use_precond, num_in_channels),
                                 solver, num_steps=5)
        degraded = th.rand(self.batch_size, num_in_channels, self.image_size, self.image_size, dtype=self.dtype,
                           device=self.device)
        noise_std = th.ones(self.batch_size, 1, dtype=self.dtype, device=self.device) * self.irls_noise_std
        data_dict = {'input': degraded, 'kernel': kernels, 'noise_std': noise_std}
        return layer, data_dict

    def init(self, use_precond, num_in_channels=None):
        if num_in_channels is None:
            num_in_channels = self.num_channels
        degradation_operator = ConvDecimateLinearDegradationOperator(padding_mode=None)
        kernels = \
            degradation_operator._init_kernels_gaussian(self.batch_size, 3).to(device=self.device, dtype=self.dtype)
        degradation_operator.init_with_parameters_(kernel=kernels)
        solver = BatchedConjugateGradientSolver(max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter,
                                                atol=1e-9, rtol=1e-6, verbose=True)
        layer = QMRecurrentLayer(self.get_toy_irls_system(degradation_operator, use_precond, num_in_channels),
                                 solver, num_steps=5)
        degraded = th.rand(self.batch_size, num_in_channels, self.image_size, self.image_size, dtype=self.dtype,
                           device=self.device)
        noise_std = th.ones(self.batch_size, 1, dtype=self.dtype, device=self.device) * self.irls_noise_std
        data_dict = {'input': degraded, 'kernel': kernels, 'noise_std': noise_std}
        return layer, data_dict

    def get_reg_backbone(self):
        bbne = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(16, 8, kernel_size=3, padding=1),
                             nn.ReLU())
        reg_backbone = RegBackbone(bbne)
        return reg_backbone

    def get_matmul_reg_backbone(self, num_in_channels):
        bbne = nn.Sequential(nn.Conv2d(num_in_channels, 16, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(16, num_in_channels, kernel_size=3, padding=1),
                             nn.ReLU())
        reg_backbone = RegBackbone(bbne)
        ret = SpectralSquaredFunctionWrapper(reg_backbone)
        return ret

    def test_dummy(self):
        s = th.rand(10).to(device=self.device)

    @pytest.mark.parametrize('use_precond', [True, False])
    def test_precond_time(self, use_precond):
        layer, data_dict = self.init(use_precond)
        degraded = data_dict['input']
        res = layer(degraded, None, **data_dict)
        print(res[-1][0])

    def test_precond_values(self):
        layer, data_dict = self.init(False)
        degraded = data_dict['input']
        res = layer(degraded, None, **data_dict)
        res_without_precond = res[1][0]
        layer.system_layer.system.use_circulant_precond = True
        res = layer(degraded, None, **data_dict)
        res_with_precond = res[1][0]
        assert th.all(th.isclose(res_without_precond, res_with_precond)), \
            f'Diff: {((res_without_precond - res_with_precond)/res_without_precond).abs()}'

    @pytest.mark.parametrize('num_channels', [1, 3])
    def test_grad_residual_operator_diagonal(self, num_channels):
        layer, data_dict = self.init(False, num_in_channels=num_channels)
        image = data_dict['input']
        kernel = data_dict['kernel']
        system = layer.system_layer.system
        system.prepare_for_restoration(**data_dict)
        solution = th.rand(*image.shape[:-2],
                           image.shape[-2] + kernel.shape[-2] - 1,
                           image.shape[-1] + kernel.shape[-1] - 1).to(image)
        solution.requires_grad_(True)

        reg_fidelity = system.reg_fidelity.prepare_for_step(-1, *solution)
        reg_images = system.reg_images.prepare_for_step(-1, *solution)
        weight_noise = system.weight_noise_operator.prepare_for_step(-1, *solution)

        reg_images_solution = reg_images(solution)
        reg_backbone_linear = system.linear_reg_backbone(reg_images_solution)

        residual = system.degradation(solution) - image
        residual = weight_noise(system.degradation.T(reg_fidelity.transpose_apply(residual))) + \
                   reg_images.T(reg_backbone_linear(reg_images_solution))

        grad_r_op_autograd = AutogradGradLinearOperator((residual,), (solution, ))

        r, grad_r_op = system.residual_with_grad_residual_operator(image, solution)
        vec = th.rand_like(solution)
        assert th.allclose(grad_r_op(vec), grad_r_op_autograd(vec), atol=self.atol, rtol=self.rtol)
        tensors_for_grad = tuple([t for t in system.tensors_for_grad if t.requires_grad])
        grads_residual = th.autograd.grad(residual, tensors_for_grad, grad_outputs=vec,
                                          retain_graph=True, create_graph=False, allow_unused=False, only_inputs=True)
        grads_r = th.autograd.grad(r, tensors_for_grad, grad_outputs=vec,
                                   retain_graph=True, create_graph=False, allow_unused=False, only_inputs=True)
        assert len(grads_residual) == len(grads_r)
        for g_old, g_new in zip(grads_residual, grads_r):
            assert th.allclose(g_old, g_new, atol=self.atol, rtol=self.rtol)

    @pytest.mark.parametrize('num_channels', [2, 3])
    def test_grad_residual_operator_matmul(self, num_channels):
        layer, data_dict = self.init_matmul(False, num_in_channels=num_channels)
        image = data_dict['input']
        kernel = data_dict['kernel']
        system = layer.system_layer.system
        system.prepare_for_restoration(**data_dict)
        solution = th.rand(*image.shape[:-2],
                           image.shape[-2] + kernel.shape[-2] - 1,
                           image.shape[-1] + kernel.shape[-1] - 1).to(image)
        solution.requires_grad_(True)

        reg_fidelity = system.reg_fidelity.prepare_for_step(-1, *solution)
        reg_images = system.reg_images.prepare_for_step(-1, *solution)
        weight_noise = system.weight_noise_operator.prepare_for_step(-1, *solution)

        reg_images_solution = reg_images(solution)
        reg_backbone_linear = system.linear_reg_backbone(reg_images_solution)

        residual = system.degradation(solution) - image
        residual = weight_noise(system.degradation.T(reg_fidelity.transpose_apply(residual))) + \
                   reg_images.T(reg_backbone_linear(reg_images_solution))

        grad_r_op_autograd = AutogradGradLinearOperator((residual,), (solution,))

        r, grad_r_op = system.residual_with_grad_residual_operator(image, solution)
        vec = th.rand_like(solution)
        assert th.allclose(grad_r_op(vec), grad_r_op_autograd(vec), atol=self.atol, rtol=self.rtol)
        tensors_for_grad = tuple([t for t in system.tensors_for_grad if t.requires_grad])
        grads_residual = th.autograd.grad(residual, tensors_for_grad, grad_outputs=vec,
                                          retain_graph=True, create_graph=False, allow_unused=False, only_inputs=True)
        grads_r = th.autograd.grad(r, tensors_for_grad, grad_outputs=vec,
                                   retain_graph=True, create_graph=False, allow_unused=False, only_inputs=True)
        assert len(grads_residual) == len(grads_r)
        for g_old, g_new in zip(grads_residual, grads_r):
            assert th.allclose(g_old, g_new, atol=self.atol, rtol=self.rtol)
