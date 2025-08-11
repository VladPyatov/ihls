import itertools
from typing import Optional

import pytest
import torch as th
from torch import nn

from irrn.functional import ImplicitLinearSystemFunction, SVDFunction, SymEigFunction, RobustSVDFunction, \
    RobustSymPSDEigFunction, PatchGroupTransform, PatchGroupTransformTranspose
from irrn.modules.backbones import ConvBackbone, L2NormBackbone
from irrn.modules.layers import QMRecurrentLayer, LinearSystemStepLayer, ImplicitLayer
from irrn.operators import ImageKernelJacobian, LearnableConvolutionOperator, LearnableDiagonalOperator, \
    LearnableNumberOperator, IdentityOperator, LearnableCNNOperator, LearnableLinearSystemOperatorBase, \
    LinearOperatorBase
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator
from irrn.operators.linsys import WienerFilteringSystemOperator
from irrn.operators.linsys.irgn import QMImageKernelSystemOperator
from irrn.operators.linsys.irls import IRLSSystemOperatorHandler
from irrn.solvers import BatchedConjugateGradientSolver, BatchedBiCGStabSolver
from irrn.solvers.linsys.inverse_call import BatchedInverseCallSolver
from irrn.solvers.recurrent import BatchedRecurrentSolver


def identity(x, *args):
    return x


class QMCorrelatedImageKernelSystemOperator(QMImageKernelSystemOperator):
    def prepare_for_step(self, step_idx, degraded, latents, parts_scheduler=None):
        scale_weight = degraded.mean()
        for latent in latents:
            scale_weight += latent.mean()
        for t in self.tensors_for_grad:
            scale_weight += t.mean()
        scale_weight = scale_weight.unsqueeze(0)/9.5759607213672
        step_size_image = LearnableNumberOperator(scale_weight, function=self.step_size_image.function, learnable=False)
        system = QMCorrelatedImageKernelSystemOperator(
            self.jacobian, self.reg_fidelity, self.reg_image, self.diag_image, self.reg_kernel, self.diag_kernel,
            self.weight_decay_image, self.weight_decay_kernel, step_size_image, self.step_size_kernel)
        if isinstance(parts_scheduler, (list, tuple)):
            optim_part = parts_scheduler[step_idx]
        else:
            optim_part = parts_scheduler(step_idx)
        return system._choose_system_operator(optim_part)


class DummyConjugateGradientSolver(BatchedConjugateGradientSolver):
    def solve(self, linsys: LinearOperatorBase, *linsys_call_params: th.Tensor,
              right_hand_side: Optional[th.Tensor] = None, initialization: Optional[th.Tensor] = None,
              solution_id: int = None) -> th.Tensor:
        def matrix_callable(vector):
            return linsys.apply(vector, *linsys_call_params)

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        if initialization is None:
            initialization = self.initialization_forward if self.initialization_forward is not None \
                else th.zeros_like(right_hand_side)
        solution, stats = self.solve_given_params(matrix_callable, right_hand_side, initialization,
                                                  *self.solver_forward_params,
                                                  precond_left_inv=linsys.preconditioner_left_inv,
                                                  precond_right_inv=linsys.preconditioner_right_inv)
        return solution

    def solve_in_backward(self, linsys: LinearOperatorBase, *linsys_call_params: th.Tensor,
                          right_hand_side: Optional[th.Tensor] = None, initialization: Optional[th.Tensor] = None,
                          solution_id: int = None) -> th.Tensor:
        def matrix_callable(vector):
            return linsys.apply(vector, *linsys_call_params)

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        if initialization is None:
            initialization = th.zeros_like(right_hand_side)
        solution, stats = self.solve_given_params(matrix_callable, right_hand_side, initialization,
                                                  *self.solver_backward_params,
                                                  precond_left_inv=linsys.preconditioner_left_inv,
                                                  precond_right_inv=linsys.preconditioner_right_inv)
        return solution


class TestQMLGradsBase:
    __test__ = False
    solver_maxiter = 250
    restarts_iter = 10
    grads_atol = 1e-5
    grads_rtol = 1e-3
    to_optimize = 'both'
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 2
    batch_size = 2
    num_channels = 3
    image_size = 16
    kernel_size = 5
    pad_type = None
    force_num_reg_filters_to = None
    system_class = QMImageKernelSystemOperator
    solver_class = BatchedConjugateGradientSolver

    def init(self):
        degraded = th.rand(self.batch_size, self.num_channels, self.image_size, self.image_size)
        jacobian = ImageKernelJacobian(self.sr_scale, padding_mode=self.pad_type)
        images, kernels = jacobian.init_parameters(degraded, self.kernel_size)
        reg_fidelity = LearnableConvolutionOperator(filter_size=3, learnable=False)
        reg_image = LearnableConvolutionOperator(filter_size=3, learnable=False)
        reg_kernel = LearnableConvolutionOperator(filter_size=3, learnable=False)
        if self.force_num_reg_filters_to:
            reg_fidelity.effective_kernel = reg_fidelity.effective_kernel[:self.force_num_reg_filters_to]
            reg_image.effective_kernel = reg_image.effective_kernel[:self.force_num_reg_filters_to]
            reg_kernel.effective_kernel = reg_kernel.effective_kernel[:self.force_num_reg_filters_to]
        diag_image = LearnableDiagonalOperator(diagonal_vector=1 + th.rand(1, reg_image.effective_kernel.shape[0], 1, 1),
                                               function=self.func, learnable=False)
        diag_kernel = LearnableDiagonalOperator(diagonal_vector=1 + th.rand(1, reg_kernel.effective_kernel.shape[0], 1, 1),
                                                function=self.func, learnable=False)
        weight_decay_image = LearnableNumberOperator(1 + th.rand(1), function=self.func, learnable=False)
        weight_decay_kernel = LearnableNumberOperator(1 + th.rand(1), function=self.func, learnable=False)
        step_size_image = LearnableNumberOperator(1 + th.rand(1), function=self.func, learnable=False)
        step_size_kernel = LearnableNumberOperator(1 + th.rand(1), function=self.func, learnable=False)
        system = self.system_class(jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel,
                                   weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel)
        system = system.to(device=self.device, dtype=self.dtype)
        degraded = degraded.to(device=self.device, dtype=self.dtype)
        images = images.to(device=self.device, dtype=self.dtype)
        kernels = kernels.to(device=self.device, dtype=self.dtype)
        solver = self.solver_class(
            verbose=False, rtol=1e-6, atol=1e-8, max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter)
        return degraded, (images, kernels), system, solver

    def check_grads(self, grads_1, grads_2, grads_1_name='Autograd', grads_2_name='Custom'):
        if grads_1 is None and grads_2 is None:
            return
        elif grads_1 is None:
            grads_1 = th.zeros_like(grads_2)
        elif grads_2 is None:
            grads_2 = th.zeros_like(grads_1)
        if not th.all(th.isclose(grads_1, grads_2, atol=self.grads_atol, rtol=self.grads_rtol)):
            max_atol = th.abs(grads_1 - grads_2).max()
            max_rtol = th.abs((grads_1 - grads_2)/grads_1).max()
            raise ValueError(
                f'Gradients do not match.\nMax atol: {max_atol}\nMax rtol: {max_rtol}\n'
                f'{grads_1_name} gradients:\n{grads_1}\n{grads_2_name} gradients:\n{grads_2}'
                f'\nDiff:\n{grads_1 - grads_2}.')


# Vulnerable to wrap-around numerical errors
class TestQMLGradsGradCheck(TestQMLGradsBase):
    __test__ = True
    solver_maxiter = 250
    restarts_iter = 50
    grads_atol = 1e-5
    grads_rtol = 1e-3
    to_optimize = 'both'
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 1
    batch_size = 1
    num_channels = 1
    image_size = 3
    kernel_size = 3
    pad_type = None
    force_num_reg_filters_to = 1
    solver_class = DummyConjugateGradientSolver

    def function_to_check(self, system, solver, num_latents, degraded, *tensor_args):
        latents = tensor_args[:num_latents]
        system_step = system.prepare_for_step(
            0, degraded, latents, parts_scheduler=[self.to_optimize])
        return ImplicitLinearSystemFunction.apply(system_step, solver, 0, num_latents, degraded, *latents,
                                                  *system_step.tensors_for_grad)

    @pytest.mark.parametrize("which_grad", ["degraded", "images", "kernels", "reg_fidelity", "reg_image", "diag_image",
                                            "reg_kernel", "diag_kernel", "weight_decay_image", "weight_decay_kernel",
                                            "step_size_image", "step_size_kernel"], scope="class")
    def test_grads_separate(self, which_grad):
        degraded, latents, system, solver = self.init()
        if which_grad == "degraded":
            degraded.requires_grad = True
        elif which_grad == "images":
            latents[0].requires_grad = True
        elif which_grad == "kernels":
            latents[1].requires_grad = True
        elif which_grad == "reg_fidelity":
            system.reg_fidelity.cast_parameters_to_requires_grad()
        elif which_grad == "reg_image":
            system.reg_image.cast_parameters_to_requires_grad()
        elif which_grad == "diag_image":
            system.diag_image.cast_parameters_to_requires_grad()
        elif which_grad == "reg_kernel":
            system.reg_kernel.cast_parameters_to_requires_grad()
        elif which_grad == "diag_kernel":
            system.diag_kernel.cast_parameters_to_requires_grad()
        elif which_grad == "weight_decay_image":
            system.weight_decay_image.cast_parameters_to_requires_grad()
        elif which_grad == "weight_decay_kernel":
            system.weight_decay_kernel.cast_parameters_to_requires_grad()
        elif which_grad == "step_size_image":
            system.step_size_image.cast_parameters_to_requires_grad()
        elif which_grad == "step_size_kernel":
            system.step_size_kernel.cast_parameters_to_requires_grad()

        th.autograd.gradcheck(self.function_to_check,
                              (system, solver, len(latents), degraded, *latents, *system.tensors_for_grad),
                              nondet_tol=1e-4, atol=self.grads_atol, rtol=self.grads_rtol)


class TestQMLGradsGradCheckCorrelated(TestQMLGradsGradCheck):
    system_class = QMCorrelatedImageKernelSystemOperator


class TestQMLGradsGradCheckSeveralSteps(TestQMLGradsGradCheck):
    __test__ = True
    solver_maxiter = 250
    restarts_iter = 50
    grads_atol = 1e-5
    grads_rtol = 1e-3
    to_optimize = ['both', 'both']
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 1
    batch_size = 1
    num_channels = 1
    image_size = 3
    kernel_size = 3
    pad_type = None
    force_num_reg_filters_to = 1
    num_steps = 2

    def function_to_check(self, system, solver, num_latents, degraded, *tensor_args):
        latents = tensor_args[:num_latents]
        updates_history = [latents]
        for step in range(self.num_steps):
            latents_step = updates_history[-1]
            system_step = system.prepare_for_step(step, degraded, latents_step, parts_scheduler=self.to_optimize)
            solution = ImplicitLinearSystemFunction.apply(
                system_step, solver, 0, num_latents, degraded, *latents_step, *system_step.tensors_for_grad)
            updates_history.append(system_step.perform_step(latents_step, solution))
        return updates_history[-1]


class TestQMLGradsGradCheckSeveralStepsCorrelated(TestQMLGradsGradCheckSeveralSteps):
    system_class = QMCorrelatedImageKernelSystemOperator


class TestWienerGradsGradCheck(TestQMLGradsBase):
    __test__ = True
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 1
    batch_size = 1
    num_channels = 1
    image_size = 3
    kernel_size = 3
    pad_type = 'periodic'
    solver_class = BatchedInverseCallSolver
    force_num_reg_filters_to = 1

    def function_to_check(self, system, solver, *tensors_for_grad):
        return ImplicitLinearSystemFunction.apply(system, solver, 0, 0, *tensors_for_grad)

    def init(self):
        degradation_operator = \
            ConvDecimateLinearDegradationOperator(scale_factor=self.sr_scale, padding_mode=self.pad_type)
        kernels = degradation_operator._init_kernels_gaussian(self.batch_size, 3).to(device=self.device,
                                                                                     dtype=self.dtype)
        degradation_operator = degradation_operator.init_with_parameters(kernel=kernels)

        reg_fidelity = LearnableConvolutionOperator(filter_size=3, padding_mode=self.pad_type, learnable=False)
        reg_images = LearnableConvolutionOperator(filter_size=3,
                                                  filter_num_in_channels=self.num_channels,
                                                  padding_mode=self.pad_type, learnable=False)
        reg_images.effective_kernel = th.rand_like(reg_images.effective_kernel)[:self.force_num_reg_filters_to]
        reg_fidelity.effective_kernel = th.rand_like(reg_fidelity.effective_kernel)[:self.force_num_reg_filters_to]
        system = WienerFilteringSystemOperator(
            degradation_operator, reg_fidelity, reg_images, weight_noise=IdentityOperator())
        system = system.to(device=self.device, dtype=self.dtype)
        degraded = th.rand(self.batch_size, self.num_channels, self.image_size, self.image_size, dtype=self.dtype,
                           device=self.device)
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device) * 2/255,
                        'kernel': kernels, 'input': degraded}
        for p in system.parameters():
            p.requires_grad = False
        solver = self.solver_class()
        return system.prepare_for_restoration(**input_params), degraded, solver

    @pytest.mark.parametrize("which_grad", ["degraded", "reg_fidelity", "reg_images"], scope="class")
    def test_grads_separate(self, which_grad):
        system, degraded, solver = self.init()
        if which_grad == "degraded":
            degraded.requires_grad = True
        elif which_grad == "reg_fidelity":
            system.reg_fidelity.cast_parameters_to_requires_grad()
        elif which_grad == "reg_images":
            system.reg_images.cast_parameters_to_requires_grad()
        elif which_grad == "diag_images":
            system.diag_image.cast_parameters_to_requires_grad()

        th.autograd.gradcheck(self.function_to_check,
                              (system, solver, degraded, *system.tensors_for_grad),
                              nondet_tol=1e-4, atol=self.grads_atol, rtol=self.grads_rtol)


class PowerIterationDummy(nn.Module):
    def __init__(self, n_c=3):
        super(PowerIterationDummy, self).__init__()
        self.matrix = nn.Conv2d(n_c, n_c, 3, bias=True)

    def forward(self, degraded, x, **kwargs):
        ret = self.matrix(x)
        ret = th.conv_transpose2d(ret, self.matrix.weight)
        return (ret/(ret.pow(2).sum(dim=(-1, -2, -3), keepdim=True).pow(0.5)), )

    def initialize(self, degraded):
        return (degraded.clone(), )

    def residual(self, degraded, *solution):
        return (self.forward(degraded, *solution)[0] - solution[0], )

    @property
    def tensors_for_grad(self):
        return tuple(self.parameters())

    def prepare_for_restoration(self):
        pass


class TestImplicitGradsDummy(TestQMLGradsBase):
    __test__ = True
    device = 'cuda'

    def init(self, jacobian_free=False):
        degraded = th.rand(self.batch_size, self.num_channels, self.image_size, self.image_size, dtype=self.dtype,
                           device=self.device)
        step_layer = PowerIterationDummy(n_c=self.num_channels).to(dtype=self.dtype, device=self.device)
        solver_forward = BatchedRecurrentSolver(step_layer, max_steps_num=50, verbose=True, atol=1e-7, rtol=1e-5)
        solver_backward = BatchedBiCGStabSolver(max_iter=1000, verbose=True, atol=1e-8, rtol=1e-6)
        model = ImplicitLayer(solver_forward, solver_backward, jacobian_free_backward=jacobian_free)
        return degraded, model

    def grads_custom(self, degraded, model):
        print('Recurrent restoration with gradients computed with implicit function theorem...')
        solution = model(degraded)
        loss = solution[0].sum()
        loss.backward()
        grad = model.solver_forward.step_module.matrix.weight.grad.clone()
        model.solver_forward.step_module.matrix.weight.grad = None
        return grad

    def grads_autograd(self, degraded, model):
        print('Recurrent restoration with gradients computed with autograd...')
        solution = model.solver_forward.solve(degraded)
        loss = solution[0].sum()
        loss.backward()
        grad = model.solver_forward.step_module.matrix.weight.grad.clone()
        model.solver_forward.step_module.matrix.weight.grad = None
        return grad

    @pytest.mark.parametrize("jacobian_free", [False, True])
    def test_on_poweriter(self, jacobian_free):
        degraded, model = self.init()
        self.check_grads(self.grads_autograd(degraded, model), self.grads_custom(degraded, model))


class TestImplicitGradsIRLSL2(TestQMLGradsBase):
    __test__ = True
    system_class = IRLSSystemOperatorHandler
    solver_class = BatchedConjugateGradientSolver
    solver_maxiter = 1000

    def init(self):
        degraded = th.rand(self.batch_size, self.num_channels, self.image_size, self.image_size)
        degradation_operator = \
            ConvDecimateLinearDegradationOperator(scale_factor=self.sr_scale, padding_mode=self.pad_type)
        kernels = degradation_operator._init_kernels_gaussian(self.batch_size, self.kernel_size)
        degradation_operator = degradation_operator.init_with_parameters(kernel=kernels)

        reg_fidelity = LearnableConvolutionOperator(filter_size=3, learnable=False,
                                                    filter_num_in_channels=self.num_channels)
        reg_images = LearnableCNNOperator(backbone=ConvBackbone(out_features_per_layer=[8], kernel_size_per_layer=[3],
                                                                padding_per_layer=[0], strides_per_layer=[1],
                                                                num_in_features=self.num_channels), learnable=False)
        weight_decay = LearnableNumberOperator(th.ones(1)*1e-4, function=lambda x: x, learnable=False)
        reg_backbone = L2NormBackbone()
        reg_fidelity_wiener = LearnableConvolutionOperator(filter_size=3, learnable=False,
                                                           filter_num_in_channels=self.num_channels)
        reg_images_wiener = LearnableConvolutionOperator(filter_size=3, learnable=False,
                                                         filter_num_in_channels=self.num_channels)
        zero_step_system = WienerFilteringSystemOperator(degradation_operator, reg_fidelity_wiener, reg_images_wiener,
                                                         IdentityOperator())

        if self.force_num_reg_filters_to:
            reg_fidelity.effective_kernel = reg_fidelity.effective_kernel[:self.force_num_reg_filters_to]
            reg_images.effective_kernel = reg_images.effective_kernel[:self.force_num_reg_filters_to]
        system = self.system_class(degradation_operator, reg_fidelity, reg_images, weight_decay, reg_backbone,
                                   zero_step_system=zero_step_system, use_circulant_precond=False)
        system = system.to(device=self.device, dtype=self.dtype)
        degraded = degraded.to(device=self.device, dtype=self.dtype)
        kernels = kernels.to(device=self.device, dtype=self.dtype)
        noise_std = th.ones(1, 1, device=self.device, dtype=self.dtype)
        solver = self.solver_class(
            verbose=False, rtol=1e-6, atol=1e-8, max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter)

        solver_backward = BatchedConjugateGradientSolver(max_iter=1000, verbose=True)
        return degraded, kernels, noise_std, system, solver, solver_backward

    @pytest.mark.parametrize("which_grad,jacobian_free",
                             list(itertools.product(["degraded", "reg_fidelity", "reg_images"], [False, True])),
                             scope="class")
    def test_grads_separate(self, which_grad, jacobian_free):
        degraded, kernels, noise_std, system, solver, solver_backward = self.init()
        link_to_tensor_with_grad = None
        if which_grad == "degraded":
            link_to_tensor_with_grad = degraded
        elif which_grad == "reg_fidelity":
            link_to_tensor_with_grad = system.reg_fidelity.effective_kernel
        elif which_grad == "reg_images":
            for param in system.reg_images._backbone.parameters():
                param.requires_grad = True
            link_to_tensor_with_grad = next(system.reg_images._backbone.parameters())
        elif which_grad == "reg_backbone":
            link_to_tensor_with_grad = system.backbone.weight

        link_to_tensor_with_grad.requires_grad = True
        n_steps = self.grad_implicit(degraded, kernels, noise_std, system, solver, solver_backward,
                                     jacobian_free=jacobian_free)
        grad_implicit = link_to_tensor_with_grad.grad.clone()
        link_to_tensor_with_grad.grad = None
        self.grad_explicit(n_steps, degraded, kernels, noise_std, system, solver)
        grad_explicit = link_to_tensor_with_grad.grad
        self.check_grads(grad_implicit, grad_explicit, grads_1_name='Implicit', grads_2_name='Explicit')

    def grad_implicit(self, degraded, kernels, noise_std, system, linsys_solver, solver_backward, jacobian_free=False):
        class IRLSRecurrentStepLayer(LinearSystemStepLayer):
            def residual_with_grad_residual_operator(self_inner, degraded: th.Tensor, *solution: th.Tensor):
                return self_inner.system.residual_with_grad_residual_operator(degraded, *solution)

        class IRLSRecurrentSolver(BatchedRecurrentSolver):
            def residual_with_grad_residual_operator(self_inner, degraded: th.Tensor, *solution: th.Tensor):
                return self_inner.step_module.residual_with_grad_residual_operator(degraded, *solution)

        system_layer = IRLSRecurrentStepLayer(system, linsys_solver)
        solver_forward = IRLSRecurrentSolver(system_layer, max_steps_num=50, atol=0.,
                                             initialization_fn=lambda x: system_layer(x, None, call_id=0),
                                             verbose=True)
        model = ImplicitLayer(solver_forward, solver_backward=solver_backward, jacobian_free_backward=jacobian_free)
        solution = model(degraded, noise_std=noise_std, kernel=kernels)
        loss = solution[0].sum()
        loss.backward()
        n_steps = solver_forward.convergence_stats['num_steps']
        return n_steps

    def grad_explicit(self, num_steps, degraded, kernels, noise_std, system, linsys_solver):
        model = QMRecurrentLayer(system, linsys_solver, num_steps=num_steps, track_updates_history=False)
        solution = model(degraded, None, noise_std=noise_std, kernel=kernels)
        loss = solution[0].sum()
        loss.backward()


# legacy: not working now, since linsys solver does inplace operations
class TestQMLGradsAutograd(TestQMLGradsBase):
    __test__ = False
    solver_maxiter = 250
    restarts_iter = 50
    grads_atol = 1e-4
    grads_rtol = 1e-3
    to_optimize = 'both'
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 2
    batch_size = 2
    num_channels = 3
    image_size = 16
    kernel_size = 5
    pad_type = None

    def grad_autograd(self, degraded, latents, system, solver):
        system_step = system.prepare_for_step(0, degraded, latents, [self.to_optimize])
        solver = self.solver_class(
            verbose=False, rtol=1e-6, atol=1e-8, max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter)
        solution = solver.solve(
            lambda vector: system_step(vector, *latents), system_step.right_hand_side(degraded, *latents))
        loss = solution.sum()
        loss.backward()

    def grad_custom(self, degraded, latents, system, solver):
        system_step = system.prepare_for_step(0, degraded, latents, [self.to_optimize])
        solver = self.solver_class(
            verbose=False, rtol=1e-6, atol=1e-8, max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter)

        solution = ImplicitLinearSystemFunction.apply(
            system_step, solver, 0, len(latents), degraded, *latents, *system_step.tensors_for_grad)
        loss = 0
        for component_loss in solution:
            loss += component_loss.sum()
        if loss.requires_grad:
            loss.backward()

    @pytest.mark.parametrize("which_grad", ["degraded", "images", "kernels", "reg_fidelity", "reg_image", "diag_image",
                                            "reg_kernel", "diag_kernel", "weight_decay_image", "weight_decay_kernel",
                                            "step_size_image", "step_size_kernel"], scope="class")
    def test_grads_separate(self, which_grad):
        degraded, latents, system, solver = self.init()
        link_to_tensor_with_grad = None
        if which_grad == "degraded":
            link_to_tensor_with_grad = degraded
        elif which_grad == "images":
            link_to_tensor_with_grad = latents[0]
        elif which_grad == "kernels":
            link_to_tensor_with_grad = latents[1]
        elif which_grad == "reg_fidelity":
            link_to_tensor_with_grad = system.reg_fidelity.effective_kernel
        elif which_grad == "reg_image":
            link_to_tensor_with_grad = system.reg_image.effective_kernel
        elif which_grad == "diag_image":
            link_to_tensor_with_grad = system.diag_image.diagonal_vector
        elif which_grad == "reg_kernel":
            link_to_tensor_with_grad = system.reg_kernel.effective_kernel
        elif which_grad == "diag_kernel":
            link_to_tensor_with_grad = system.diag_kernel.diagonal_vector
        elif which_grad == "weight_decay_image":
            link_to_tensor_with_grad = system.weight_decay_image.diagonal_vector
        elif which_grad == "weight_decay_kernel":
            link_to_tensor_with_grad = system.weight_decay_kernel.diagonal_vector
        elif which_grad == "step_size_image":
            link_to_tensor_with_grad = system.step_size_image.diagonal_vector
        elif which_grad == "step_size_kernel":
            link_to_tensor_with_grad = system.step_size_kernel.diagonal_vector

        link_to_tensor_with_grad.requires_grad = True
        self.grad_autograd(degraded, latents, system, solver)
        grad_autograd = link_to_tensor_with_grad.grad.clone()
        link_to_tensor_with_grad.grad = None
        self.grad_custom(degraded, latents, system, solver)
        grad_custom = link_to_tensor_with_grad.grad
        self.check_grads(grad_autograd, grad_custom)

    @pytest.mark.parametrize("which_grad", ["degraded", "images", "kernels", "reg_fidelity", "reg_image", "diag_image",
                                            "reg_kernel", "diag_kernel", "weight_decay_image", "weight_decay_kernel",
                                            "step_size_image", "step_size_kernel"], scope="class")
    def test_grads_from_all(self, which_grad):
        if not pytest.grads_autograd:
            assert pytest.grads_custom is None
            pytest.grads_autograd, pytest.grads_custom = self.calculate_all_grads(*self.init())
        if which_grad == "degraded":
            self.check_grads(pytest.grads_autograd[0], pytest.grads_custom[0])
        elif which_grad == "images":
            self.check_grads(pytest.grads_autograd[1], pytest.grads_custom[1])
        elif which_grad == "kernels":
            self.check_grads(pytest.grads_autograd[2], pytest.grads_custom[2])
        elif which_grad == "reg_fidelity":
            self.check_grads(pytest.grads_autograd[3], pytest.grads_custom[3])
        elif which_grad == "reg_image":
            self.check_grads(pytest.grads_autograd[4], pytest.grads_custom[4])
        elif which_grad == "diag_image":
            self.check_grads(pytest.grads_autograd[5], pytest.grads_custom[5])
        elif which_grad == "reg_kernel":
            self.check_grads(pytest.grads_autograd[6], pytest.grads_custom[6])
        elif which_grad == "diag_kernel":
            self.check_grads(pytest.grads_autograd[7], pytest.grads_custom[7])
        elif which_grad == "weight_decay_image":
            self.check_grads(pytest.grads_autograd[8], pytest.grads_custom[8])
        elif which_grad == "weight_decay_kernel":
            self.check_grads(pytest.grads_autograd[9], pytest.grads_custom[9])
        elif which_grad == "step_size_image":
            self.check_grads(pytest.grads_autograd[10], pytest.grads_custom[10])
        elif which_grad == "step_size_kernel":
            self.check_grads(pytest.grads_autograd[11], pytest.grads_custom[11])

    def calculate_all_grads(self, degraded, latents, system, solver):

        images, kernels = latents
        degraded.requires_grad = True
        images.requires_grad = True
        kernels.requires_grad = True

        reg_fidelity_filters = system.reg_fidelity.effective_kernel
        reg_fidelity_filters.requires_grad = True
        system.reg_fidelity.effective_kernel = reg_fidelity_filters

        reg_image_filters = system.reg_image.effective_kernel
        reg_image_filters.requires_grad = True
        system.reg_image.effective_kernel = reg_image_filters

        diag_image_vector = system.diag_image.diagonal_vector
        diag_image_vector.requires_grad = True
        system.diag_image.diagonal_vector = diag_image_vector

        reg_kernel_filters = system.reg_kernel.effective_kernel
        reg_kernel_filters.requires_grad = True
        system.reg_kernel.effective_kernel = reg_kernel_filters

        diag_kernel_vector = system.diag_kernel.diagonal_vector
        diag_kernel_vector.requires_grad = True
        system.diag_kernel.diagonal_vector = diag_kernel_vector

        wdecay_image_vector = system.weight_decay_image.diagonal_vector
        wdecay_image_vector.requires_grad = True
        system.weight_decay_image.diagonal_vector = wdecay_image_vector

        wdecay_kernel_vector = system.weight_decay_kernel.diagonal_vector
        wdecay_kernel_vector.requires_grad = True
        system.weight_decay_kernel.diagonal_vector = wdecay_kernel_vector

        ssize_image_vector = system.step_size_image.diagonal_vector
        ssize_image_vector.requires_grad = True
        system.step_size_image.diagonal_vector = ssize_image_vector

        ssize_kernel_vector = system.step_size_kernel.diagonal_vector
        ssize_kernel_vector.requires_grad = True
        system.step_size_kernel.diagonal_vector = ssize_kernel_vector

        self.grad_autograd(degraded, latents, system, solver)
        degraded_grad_autograd = degraded.grad.clone()
        images_grad_autograd = images.grad.clone()
        kernels_grad_autograd = kernels.grad.clone()
        reg_fidelity_filters_grad_autograd = reg_fidelity_filters.grad.clone()
        reg_image_filters_grad_autograd = reg_image_filters.grad.clone()
        diag_image_vector_grad_autograd = diag_image_vector.grad.clone()
        reg_kernel_filters_grad_autograd = reg_kernel_filters.grad.clone()
        diag_kernel_vector_grad_autograd = diag_kernel_vector.grad.clone()
        wdecay_image_vector_grad_autograd = wdecay_image_vector.grad.clone()
        wdecay_kernel_vector_grad_autograd = wdecay_kernel_vector.grad.clone()
        ssize_image_vector_grad_autograd = ssize_image_vector.grad.clone()
        ssize_kernel_vector_grad_autograd = ssize_kernel_vector.grad.clone()
        degraded.grad = None
        images.grad = None
        kernels.grad = None
        reg_fidelity_filters.grad = None
        reg_image_filters.grad = None
        diag_image_vector.grad = None
        reg_kernel_filters.grad = None
        diag_kernel_vector.grad = None
        wdecay_image_vector.grad = None
        wdecay_kernel_vector.grad = None
        ssize_image_vector.grad = None
        ssize_kernel_vector.grad = None

        self.grad_custom(degraded, latents, system, solver)
        degraded_grad_custom = degraded.grad
        images_grad_custom = images.grad
        kernels_grad_custom = kernels.grad
        reg_fidelity_filters_grad_custom = reg_fidelity_filters.grad
        reg_image_filters_grad_custom = reg_image_filters.grad
        diag_image_vector_grad_custom = diag_image_vector.grad
        reg_kernel_filters_grad_custom = reg_kernel_filters.grad
        diag_kernel_vector_grad_custom = diag_kernel_vector.grad
        wdecay_image_vector_grad_custom = wdecay_image_vector.grad
        wdecay_kernel_vector_grad_custom = wdecay_kernel_vector.grad
        ssize_image_vector_grad_custom = ssize_image_vector.grad
        ssize_kernel_vector_grad_custom = ssize_kernel_vector.grad

        return (degraded_grad_autograd,
                images_grad_autograd,
                kernels_grad_autograd,
                reg_fidelity_filters_grad_autograd,
                reg_image_filters_grad_autograd,
                diag_image_vector_grad_autograd,
                reg_kernel_filters_grad_autograd,
                diag_kernel_vector_grad_autograd,
                wdecay_image_vector_grad_autograd,
                wdecay_kernel_vector_grad_autograd,
                ssize_image_vector_grad_autograd,
                ssize_kernel_vector_grad_autograd), \
               (degraded_grad_custom,
                images_grad_custom,
                kernels_grad_custom,
                reg_fidelity_filters_grad_custom,
                reg_image_filters_grad_custom,
                diag_image_vector_grad_custom,
                reg_kernel_filters_grad_custom,
                diag_kernel_vector_grad_custom,
                wdecay_image_vector_grad_custom,
                wdecay_kernel_vector_grad_custom,
                ssize_image_vector_grad_custom,
                ssize_kernel_vector_grad_custom)


# legacy: not working now, since linsys solver does inplace operations
class TestQMLGradsAutogradCorrelated(TestQMLGradsAutograd):
    system_class = QMCorrelatedImageKernelSystemOperator


# legacy: not working now, since linsys solver does inplace operations
class TestQMLGradsAutogradTwoSteps(TestQMLGradsAutograd):
    solver_maxiter = 250
    restarts_iter = 50
    grads_atol = 1e-4
    grads_rtol = 1e-3
    to_optimize = 'both'
    device = 'cuda'
    dtype = th.float64
    func = th.sin
    sr_scale = 2
    batch_size = 2
    num_channels = 3
    image_size = 16
    kernel_size = 5
    pad_type = None
    num_steps = 2
    on_history = False

    def solve_autograd(self, degraded, latents, system, solver):
        updates_history = [latents]
        for step in range(self.num_steps):
            latents_step = updates_history[-1]
            system_step = system.prepare_for_step(step, degraded, latents_step, [self.to_optimize]*self.num_steps)
            solution = solver.solve(
                lambda vector: system_step(vector, *latents_step), system_step.right_hand_side(degraded, *latents_step))
            if solution.__class__ is th.Tensor:
                solution = (solution,)
            else:
                solution = tuple(solution)
            updates_history.append(system_step.perform_step(latents_step, solution))
        if not self.on_history:
            updates_history = [updates_history[-1]]
        return updates_history

    def calculate_loss_and_perform_backward(self, solution):
        loss = 0
        for pair in solution:
            for elem in pair:
                loss += elem.sum()
        if loss.requires_grad:
            loss.backward()

    def grad_autograd(self, degraded, latents, system, solver):
        solution = self.solve_autograd(degraded, latents, system, solver)
        self.calculate_loss_and_perform_backward(solution)

    def grad_custom(self, degraded, latents, system, solver):
        layer = QMRecurrentLayer(system, solver, num_steps=self.num_steps, track_updates_history=self.on_history)
        solution = layer(degraded, *latents, parts_scheduler=[self.to_optimize]*self.num_steps)
        if not self.on_history:
            solution = [solution]
        self.calculate_loss_and_perform_backward(solution)


# legacy: not working now, since linsys solver does inplace operations
class TestQMLGradsAutogradTwoStepsCorrelated(TestQMLGradsAutogradTwoSteps):
    system_class = QMCorrelatedImageKernelSystemOperator


class TestFBMD:
    matrix_shape = (2, 1, 4, 4)

    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_symeig(self, device):
        def symmetrize_and_eig(matrix, tol):
            matrix = th.matmul(matrix.transpose(-1, -2), matrix)
            return SymEigFunction.apply(matrix, tol)
        matrices = th.rand(*self.matrix_shape, device=device, dtype=th.float64, requires_grad=True)
        th.autograd.gradcheck(symmetrize_and_eig, (matrices, 1e-9))

    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_svd(self, device):
        matrices = th.rand(*self.matrix_shape, device=device, dtype=th.float64, requires_grad=True)
        th.autograd.gradcheck(SVDFunction.apply, (matrices, 1e-9))

    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_symeig_robust(self, device):
        def symmetrize_and_eig(matrix, tol):
            matrix = th.matmul(matrix.transpose(-1, -2), matrix)
            return RobustSymPSDEigFunction.apply(matrix, tol, 100, None)
        matrices = th.rand(*self.matrix_shape, device=device, dtype=th.float64, requires_grad=True)
        th.autograd.gradcheck(symmetrize_and_eig, (matrices, 1e-9))

    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_svd_robust(self, device):
        matrices = th.rand(*self.matrix_shape, device=device, dtype=th.float64, requires_grad=True)
        th.autograd.gradcheck(RobustSVDFunction.apply, (matrices, 1e-9, 100))


class TestPatchGroup:
    @pytest.mark.parametrize('device,batch_size,img_size,img_num_channels,patch_size,stride,group_size,kernel',
                             list(itertools.product(['cpu', 'cuda'],
                                                    [1, 2],
                                                    [8, (9, 8)],
                                                    [1, 2],
                                                    [3, (2, 3)],
                                                    [1, (1, 2)],
                                                    [4, 5],
                                                    [None, '4d', '5d'])))
    def test_patch_group_transform(
            self, device, batch_size, img_size, img_num_channels, patch_size, stride, group_size, kernel):
        th.use_deterministic_algorithms(False)
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
        if isinstance(patch_size, int):
            p_h, p_w = patch_size, patch_size
        else:
            p_h, p_w = patch_size
        if isinstance(stride, int):
            s_h, s_w = stride, stride
        else:
            s_h, s_w = stride
        n_h = max((h - p_h) // s_h + 1, 0)
        n_w = max((w - p_w) // s_w + 1, 0)

        if kernel is not None:
            if kernel == '4d':
                kernel = th.rand(2, img_num_channels, p_h, p_w,
                                 dtype=th.float64, device=device, requires_grad=True)
            elif kernel == '5d':
                kernel = th.rand(batch_size, 2, img_num_channels, p_h, p_w,
                                 dtype=th.float64, device=device, requires_grad=True)
            else:
                raise RuntimeError

        images = th.rand(batch_size, img_num_channels, h, w, dtype=th.float64, device=device, requires_grad=True)
        indices = th.randint(0, n_h*n_w, size=(batch_size, group_size, n_h, n_w), dtype=th.int64, device=device)

        th.autograd.gradcheck(PatchGroupTransform.apply, (images,  kernel, indices, patch_size, stride))

    @pytest.mark.parametrize('device,batch_size,img_size,img_num_channels,patch_size,stride,group_size,kernel',
                             list(itertools.product(['cpu', 'cuda'],
                                                    [1, 2],
                                                    [8, (9, 8)],
                                                    [1, 2],
                                                    [3, (2, 3)],
                                                    [1, (1, 2)],
                                                    [4, 5],
                                                    [None, '4d', '5d'])))
    def test_patch_group_transform_transpose(
            self, device, batch_size, img_size, img_num_channels, patch_size, stride, group_size, kernel):
        th.use_deterministic_algorithms(False)
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
        if isinstance(patch_size, int):
            p_h, p_w = patch_size, patch_size
        else:
            p_h, p_w = patch_size
        if isinstance(stride, int):
            s_h, s_w = stride, stride
        else:
            s_h, s_w = stride
        n_h = max((h - p_h) // s_h + 1, 0)
        n_w = max((w - p_w) // s_w + 1, 0)

        if kernel is not None:
            if kernel == '4d':
                kernel = th.rand(2, img_num_channels, p_h, p_w,
                                 dtype=th.float64, device=device, requires_grad=True)
            elif kernel == '5d':
                kernel = th.rand(batch_size, 2, img_num_channels, p_h, p_w,
                                 dtype=th.float64, device=device, requires_grad=True)
            else:
                raise RuntimeError
            groups = th.rand(batch_size, 2, group_size, n_h, n_w,
                             dtype=th.float64, device=device, requires_grad=True)
        else:
            groups = th.rand(batch_size, p_h * p_w * img_num_channels, group_size, n_h, n_w,
                             dtype=th.float64, device=device, requires_grad=True)

        indices = th.randint(0, n_h * n_w, size=(batch_size, group_size, n_h, n_w), dtype=th.int64, device=device)

        th.autograd.gradcheck(PatchGroupTransformTranspose.apply,
                              (groups, kernel, indices, img_num_channels, patch_size, stride, None))
