import itertools
from typing import Tuple, Optional

import cv2
import pytest
import torch as th

from irrn.modules.backbones import ConvBackbone, L2NormBackbone, WeightedL2NormBackbone, L1NormBackbone, \
    WeightedL1NormBackbone, KPBackbone
from irrn.operators import LearnableCNNOperator, IRLSSystemOperatorHandler, LearnableNumberOperator, IdentityOperator, \
    LearnableKPNConvOperator, LinearDegradationOperatorBase
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator
from irrn.solvers import BatchedConjugateGradientSolver

SIZES_PARAMS = {'batch_size': [1, 2],
                'channels_number': [3],
                'spatial_dims': ['32-32', '128-128']}
SOLVER_ATOL = 1e-7
SOLVER_RTOL = 1e-5
SOLVER_MAXITER = 250


class RegFidNoiseKPNOperator(LearnableKPNConvOperator):
    noise_std: Optional[th.Tensor]
    degraded: Optional[th.Tensor]
    degraded_operator: Optional[LinearDegradationOperatorBase]

    def prepare_for_restoration(self, noise_std=None, input=None, degradation_operator=None, **other_kwargs
                                ) -> 'LearnableKPNConvOperator':
        # vector = self.degradation_operator.T(input) / noise_std[..., None, None]
        vector = input / noise_std[..., None, None]
        self.update_effective_kernel(vector)
        return self

    def prepare_for_step(self, step_idx: int, latent: th.Tensor) -> 'RegFidNoiseKPNOperator':
        return self


class TestIRLSL2NoRegFid:
    __test__ = True
    solver = BatchedConjugateGradientSolver
    device = 'cuda'
    dtype = th.float64
    solver_init_params = {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'verbose': False, 'max_iter': SOLVER_MAXITER,
                          'restarts_iter': 25}
    solution_check_atol = 1e-5
    solution_check_rtol = 1e-3
    irls_conv_kernel_size = 15
    noise_std = 2/255
    irls_reg_images_kernel_size = 13
    weight_noise_scale = False

    def get_regfid_operator(self, batch_size, num_channels):
        return IdentityOperator()

    def get_reg_backbone(self):
        reg_backbone = L2NormBackbone()
        return reg_backbone

    def get_toy_irls_system(self, batch_size, num_channels, precond):
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device) * self.noise_std}
        degradation_operator = IdentityOperator()#ConvDecimateLinearDegradationOperator(padding_mode=None)
        # degradation_operator = degradation_operator.init_with_parameters(kernel=kernels)
        # input_params['kernel'] = \
        #     degradation_operator._init_kernels_gaussian(batch_size, 3).to(device=self.device, dtype=self.dtype)

        reg_fidelity = self.get_regfid_operator(batch_size, num_channels)
        reg_images = LearnableCNNOperator(
            backbone=ConvBackbone(out_features_per_layer=[128], kernel_size_per_layer=[self.irls_reg_images_kernel_size],
                                  padding_per_layer=[0], strides_per_layer=[1],
                                  num_in_features=num_channels), learnable=False, mix_in_channels=False)
        weight_decay = LearnableNumberOperator(th.ones(1) * 1e-4, function=lambda x: x, learnable=False)
        reg_backbone = self.get_reg_backbone()

        system = IRLSSystemOperatorHandler(degradation_operator, reg_fidelity, reg_images, weight_decay, reg_backbone,
                                           zero_step_system=None, use_circulant_precond=precond,
                                           use_weight_noise_scaling_coef=self.weight_noise_scale)
        system = system.to(device=self.device, dtype=self.dtype)
        for p in system.parameters():
            p.requires_grad = False
        return system, input_params

    def get_solver_callable(self):
        solver = self.solver(**self.solver_init_params)
        return solver

    def get_random_solution(self, batch_size, num_channels, height, width):
        return th.rand(batch_size, num_channels, height, width, dtype=self.dtype,
                       device=self.device)

    @staticmethod
    def eval_solver(solver, matrix_callable, rhs, *matrix_args):
        return solver.solve(matrix_callable, *matrix_args, right_hand_side=rhs)

    def solve_using_operator(self, solver, matrix_op, rhs, *matrix_args):
        return self.eval_solver(solver, matrix_op, rhs, *matrix_args)

    def get_and_solve_system_grad_matrix(self, batch_size, channels_num, spatial_dims, use_precond):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num, use_precond)
        solver = self.get_solver_callable()

        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)

        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * self.noise_std
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})
        latents = (th.clamp(solution_true + th.randn_like(solution_true) * 0.5, 0, 1).requires_grad_(True),)
        return self.solve_system_grad_matrix(system, degraded, inputs, latents, solver, solution_true)

    def solve_system_grad_matrix(self, system, degraded, inputs, latents, solver, solution_true):
        system = system.prepare_for_restoration(**inputs)
        system.prepare_for_step(1, degraded, latents)
        r, grad_r_operator = system.residual_with_grad_residual_operator(degraded, *latents)
        rhs = grad_r_operator(solution_true)
        solver.clear_state()
        solution_solver = solver.solve(grad_r_operator, right_hand_side=rhs, initialization=None, solution_id=-1)
        num_iter = solver.convergence_stats_forward[0]['num_iter']
        return num_iter, solution_true, solution_solver

    # @pytest.mark.parametrize("batch_size,channels_num,spatial_dims,use_precond",
    #                          list(itertools.product(SIZES_PARAMS['batch_size'],
    #                                                 SIZES_PARAMS['channels_number'],
    #                                                 SIZES_PARAMS['spatial_dims'],
    #                                                 [False, True])))
    # def test_with_irls_grad_matrix(self, batch_size, channels_num, spatial_dims, use_precond):
    #     self.compare_solutions(*self.get_and_solve_system_grad_matrix(batch_size, channels_num,
    #                                                                   spatial_dims, use_precond))

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims",
                             list(itertools.product(SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'])))
    def test_compare_with_irls_grad_matrix(self, batch_size, channels_num, spatial_dims):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num, False)
        solver = self.get_solver_callable()
        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)
        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * self.noise_std
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})
        latents = (th.clamp(solution_true + th.randn_like(solution_true) * 0.5, 0, 1).requires_grad_(True),)

        num_iter, solution_true, solution_solver = \
            self.solve_system_grad_matrix(system, degraded, inputs, latents, solver, solution_true)
        max_atol = th.abs(solution_solver - solution_true).max()
        max_rtol = th.abs((solution_solver - solution_true) / solution_solver).max()
        solver.clear_state()
        system.use_circulant_precond_backward = True
        num_iter_precond, solution_true, solution_solver = \
            self.solve_system_grad_matrix(system, degraded, inputs, latents, solver, solution_true)
        max_atol_precond = th.abs(solution_solver - solution_true).max()
        max_rtol_precond = th.abs((solution_solver - solution_true) / solution_solver).max()
        print(f'\nWith  precond (num_iter/rtol/atol): {num_iter_precond}/{max_rtol_precond}/{max_atol_precond}\n' \
              f'W/out precond (num_iter/rtol/atol): {num_iter}/{max_rtol}/{max_atol}.')
        assert num_iter_precond <= num_iter, f'With precond: {num_iter_precond}, without precond: {num_iter}'
        if num_iter_precond == num_iter:
            assert max_rtol_precond <= max_rtol and max_atol_precond <= max_atol

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims",
                             list(itertools.product(SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'])))
    def test_compare_with_irls_matrix(self, batch_size, channels_num, spatial_dims):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num, False)
        solver = self.get_solver_callable()

        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)

        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * self.noise_std
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})
        latents = (th.clamp(solution_true + th.randn_like(solution_true) * 0.5, 0, 1),)

        num_iter, solution_true, solution_solver = \
            self.solve_system_matrix(system, degraded, inputs, latents, solver, solution_true)
        max_atol = th.abs(solution_solver - solution_true).max()
        max_rtol = th.abs((solution_solver - solution_true) / solution_solver).max()
        solver.clear_state()
        system.use_circulant_precond = True
        num_iter_precond, solution_true, solution_solver = \
            self.solve_system_matrix(system, degraded, inputs, latents, solver, solution_true)
        max_atol_precond = th.abs(solution_solver - solution_true).max()
        max_rtol_precond = th.abs((solution_solver - solution_true) / solution_solver).max()
        print(f'\nWith  precond (num_iter/rtol/atol): {num_iter_precond}/{max_rtol_precond}/{max_atol_precond}\n' \
              f'W/out precond (num_iter/rtol/atol): {num_iter}/{max_rtol}/{max_atol}.')
        assert num_iter_precond <= num_iter, f'With precond: {num_iter_precond}, without precond: {num_iter}'
        if num_iter_precond == num_iter:
            assert max_rtol_precond <= max_rtol and max_atol_precond <= max_atol

    def get_and_solve_system_matrix(self, batch_size, channels_num, spatial_dims, use_precond):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num, use_precond)
        solver = self.get_solver_callable()

        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)

        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * self.noise_std
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})
        latents = (th.clamp(solution_true + th.randn_like(solution_true) * 0.5, 0, 1),)

        return self.solve_system_matrix(system, degraded, inputs, latents, solver, solution_true)

    def solve_system_matrix(self, system, degraded, inputs, latents, solver, solution_true):
        system = system.prepare_for_restoration(**inputs)
        system = system.prepare_for_step(1, degraded, latents)
        rhs = system(solution_true, degraded, *latents)
        solver.clear_state()
        solution_solver = solver.solve(system, degraded, *latents, right_hand_side=rhs,
                                       initialization=latents[0], solution_id=-1)
        num_iter = solver.convergence_stats_forward[0]['num_iter']
        return num_iter, solution_true, solution_solver

    # @pytest.mark.parametrize("batch_size,channels_num,spatial_dims,use_precond",
    #                          list(itertools.product(SIZES_PARAMS['batch_size'],
    #                                                 SIZES_PARAMS['channels_number'],
    #                                                 SIZES_PARAMS['spatial_dims'],
    #                                                 [False, True])))
    # def test_with_irls_system(self, batch_size, channels_num, spatial_dims, use_precond):
    #     self.compare_solutions(*self.get_and_solve_system_matrix(batch_size, channels_num,
    #                                                                   spatial_dims, use_precond))

    def compare_solutions(self, num_iter, solution_solver, solution_true):
        print(f'\nNum iter: {num_iter}')
        if not th.all(th.isclose(solution_solver, solution_true, atol=self.solution_check_atol,
                                 rtol=self.solution_check_rtol)):
            max_atol = th.abs(solution_solver - solution_true).max()
            max_rtol = th.abs((solution_solver - solution_true) / solution_solver).max()
            raise ValueError(
                f'Solutions do not match after {num_iter} iters.\nMax atol: {max_atol}\nMax rtol: {max_rtol}\n'
                f'True solution:\n{solution_true}\nSolution from solver:\n{solution_solver}'
                f'\nDiff:\n{solution_solver - solution_true}.')

    @staticmethod
    def get_test_image(is_color: bool = True, size: Tuple[int, int] = (512, 512)):
        image = cv2.imread('files/test_image_512.png')
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        image = th.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255
        return image


class TestIRLSL2Regfid(TestIRLSL2NoRegFid):
    weight_noise_scale = False

    def get_regfid_operator(self, batch_size, num_channels):
        kpb = KPBackbone(num_in_channels=num_channels, num_out_channels=8, filter_size=3)
        op = RegFidNoiseKPNOperator(kpb, observation_keyword='input', learnable=False, mix_in_channels=False)
        return op


class TestIRLSWeightedL2NoRegfid(TestIRLSL2NoRegFid):
    def get_reg_backbone(self):
        return WeightedL2NormBackbone(8)


class TestIRLSWeightedL2Regfid(TestIRLSWeightedL2NoRegfid):
    weight_noise_scale = False

    def get_regfid_operator(self, batch_size, num_channels):
        kpb = KPBackbone(num_in_channels=num_channels, num_out_channels=8, filter_size=3)
        op = RegFidNoiseKPNOperator(kpb, observation_keyword='input', learnable=False)
        return op


class TestIRLSL1NoRegFid(TestIRLSL2NoRegFid):
    def get_reg_backbone(self):
        return L1NormBackbone()


class TestIRLSL1RegFid(TestIRLSL1NoRegFid):
    weight_noise_scale = False

    def get_regfid_operator(self, batch_size, num_channels):
        kpb = KPBackbone(num_in_channels=num_channels, num_out_channels=8, filter_size=3)
        op = RegFidNoiseKPNOperator(kpb, observation_keyword='input', learnable=False)
        return op


class TestIRLSWeightedL1NoRegFid(TestIRLSL2NoRegFid):
    def get_reg_backbone(self):
        return WeightedL1NormBackbone(8)


class TestIRLSWeightedL1RegFid(TestIRLSWeightedL1NoRegFid):
    weight_noise_scale = False

    def get_regfid_operator(self, batch_size, num_channels):
        kpb = KPBackbone(num_in_channels=num_channels, num_out_channels=8, filter_size=3)
        op = RegFidNoiseKPNOperator(kpb, observation_keyword='input', learnable=False)
        return op
