import pytest
import torch as th

from irrn.functional.implicit_linear import ImplicitLinearSystemFunction
from irrn.solvers import BatchedConjugateGradientSolver
from irrn.operators import ImageKernelJacobian, LearnableConvolutionOperator, LearnableDiagonalOperator, \
    LearnableNumberOperator
from irrn.operators.linsys.irgn import QMImageKernelSystemOperator
import os

def identity(x):
    return x


class TestGradsCompare:
    solver_maxiter = 500
    restarts_iter = 10
    grads_atol = 1e-8
    grads_rtol = 1e-5
    to_optimize = 'both'
    device = 'cuda'
    data_path = '/cache/koshelev/grads_test'

    @pytest.mark.parametrize("which_grad", ["degraded", "images", "kernels", "reg_fidelity", "reg_image", "diag_image",
                                            "reg_kernel", "diag_kernel", "weight_decay_image", "weight_decay_kernel",
                                            "step_size_image", "step_size_kernel"], scope="class")
    def test(self, which_grad):
        if not pytest.grads_autograd:
            assert pytest.grads_custom is None
            print('Calculating grads...')
            degraded, latents, jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel, \
            weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel = self.init()
            grads_new = self.calculate_all_grads(degraded, latents, jacobian, reg_fidelity, reg_image, diag_image,
                                                 reg_kernel, diag_kernel, weight_decay_image, weight_decay_kernel,
                                                 step_size_image, step_size_kernel)
            grads_closed = self.load_grads()
            pytest.grads_autograd, pytest.grads_custom = grads_closed, grads_new
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

    def init(self, sr_scale=2, batch_size=2, num_channels=3, image_size=128, kernel_size=13, pad_type=None):
        th.set_deterministic(True)
        degraded = th.load(os.path.join(self.data_path, 'degraded_images'))
        jacobian = ImageKernelJacobian(sr_scale, padding_mode=pad_type)

        images = th.load(os.path.join(self.data_path, 'images'))
        kernels = th.load(os.path.join(self.data_path, 'kernels'))

        reg_fidelity = LearnableConvolutionOperator(filter_size=3, learnable=False)
        reg_fidelity.filters = th.load(os.path.join(self.data_path, 'regfid_filters'))

        reg_image = LearnableConvolutionOperator(filter_size=3, learnable=False)
        reg_image.filters = th.load(os.path.join(self.data_path, 'regimg_filters'))

        diag_image = LearnableDiagonalOperator(diagonal_vector=1 + th.rand(1, reg_image.filters.shape[0], 1, 1, 1),
                                               function=identity, learnable=False)
        diag_image.diagonal_vector = th.load(os.path.join(self.data_path, 'diagimg_vector'))

        reg_kernel = LearnableConvolutionOperator(filter_size=3, learnable=False)
        reg_kernel.filters = th.load(os.path.join(self.data_path, 'regkern_filters'))

        diag_kernel = LearnableDiagonalOperator(diagonal_vector=1 + th.rand(1, reg_kernel.filters.shape[0], 1, 1, 1),
                                                function=identity, learnable=False)
        diag_kernel.diagonal_vector = th.load(os.path.join(self.data_path, 'diagkern_vector'))

        weight_decay_image = LearnableNumberOperator(1 + th.rand(1), function=identity, learnable=False)
        weight_decay_image.diagonal_vector = th.load(os.path.join(self.data_path, 'wdecimg_vector'))

        weight_decay_kernel = LearnableNumberOperator(1 + th.rand(1), function=identity, learnable=False)
        weight_decay_kernel.diagonal_vector = th.load(os.path.join(self.data_path, 'wdeckern_vector'))

        step_size_image = LearnableNumberOperator(1 + th.rand(1), function=identity, learnable=False)
        step_size_image.diagonal_vector = th.load(os.path.join(self.data_path, 'ssizeimg_vector'))

        step_size_kernel = LearnableNumberOperator(1 + th.rand(1), function=identity, learnable=False)
        step_size_kernel.diagonal_vector = th.load(os.path.join(self.data_path, 'ssizekern_vector'))

        return degraded, (images, kernels), jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel, \
               weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel

    def load_grads(self):
        degraded_grad_closed = th.load(os.path.join(self.data_path, 'degraded_images_grads'))
        images_grad_closed = th.load(os.path.join(self.data_path, 'images_grads'))
        kernels_grad_closed = th.load(os.path.join(self.data_path, 'kernels_grads'))
        reg_fidelity_filters_grad_closed = th.load(os.path.join(self.data_path, 'regfid_filters_grads'))
        reg_image_filters_grad_closed = th.load(os.path.join(self.data_path, 'regimg_filters_grads'))
        diag_image_vector_grad_closed = th.load(os.path.join(self.data_path, 'diagimg_vector_grads'))
        reg_kernel_filters_grad_closed = th.load(os.path.join(self.data_path, 'regkern_filters_grads'))
        diag_kernel_vector_grad_closed = th.load(os.path.join(self.data_path, 'diagkern_vector_grads'))
        wdecay_image_vector_grad_closed = th.load(os.path.join(self.data_path, 'wdecimg_vector_grads'))
        wdecay_kernel_vector_grad_closed = th.load(os.path.join(self.data_path, 'wdeckern_vector_grads'))
        ssize_image_vector_grad_closed = th.load(os.path.join(self.data_path, 'ssizeimg_vector_grads'))
        ssize_kernel_vector_grad_closed = th.load(os.path.join(self.data_path, 'ssizekern_vector_grads'))
        return degraded_grad_closed, \
               images_grad_closed, \
               kernels_grad_closed, \
               reg_fidelity_filters_grad_closed, \
               reg_image_filters_grad_closed, \
               diag_image_vector_grad_closed, \
               reg_kernel_filters_grad_closed, \
               diag_kernel_vector_grad_closed, \
               wdecay_image_vector_grad_closed, \
               wdecay_kernel_vector_grad_closed, \
               ssize_image_vector_grad_closed, \
               ssize_kernel_vector_grad_closed

    def grad_custom(self, degraded, latents, jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel,
                    weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel):
        system = QMImageKernelSystemOperator(jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel,
                                             weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel)
        degraded = degraded.to(device=self.device, dtype=th.float64)
        latents = tuple([latent.to(device=self.device, dtype=th.float64) for latent in latents])
        system = system.to(device=self.device, dtype=th.float64)
        system = system.prepare_for_step(0, [self.to_optimize])
        solver = BatchedConjugateGradientSolver(
            verbose=False, rtol=1e-6, max_iter=self.solver_maxiter, restarts_iter=self.restarts_iter)

        solution = ImplicitLinearSystemFunction.apply(system, solver, 0, len(latents), degraded, *latents, *system.tensors_for_grad)

        loss = 0
        for component_loss in solution:
            loss += component_loss.sum()
        if loss.requires_grad:
            loss.backward()

        solution_image = th.load(os.path.join(self.data_path, 'sol_image'))
        solution_kernel = th.load(os.path.join(self.data_path, 'sol_kernel'))
        with th.no_grad():
            assert th.all(th.isclose(solution[0], solution_image))
            assert th.all(th.isclose(solution[1], solution_kernel))

    def check_grads(self, grad_closed, grad_new):
        if grad_closed is None and grad_new is None:
            return
        elif grad_closed is None:
            grad_closed = th.zeros_like(grad_new)
        elif grad_new is None:
            grad_new = th.zeros_like(grad_closed)
        if not th.all(th.isclose(grad_closed, grad_new, atol=self.grads_atol, rtol=self.grads_rtol)):
            max_atol = th.abs(grad_closed - grad_new).max()
            max_rtol = th.abs((grad_closed - grad_new)/grad_closed).max()
            raise ValueError(
                f'Values do not match.\nMax atol: {max_atol}\nMax rtol: {max_rtol}\n'
                f'Closed form gradients:\n{grad_closed}\nNew scheme gradients:\n{grad_new}'
                f'\nDiff:\n{grad_closed - grad_new}.')

    def calculate_all_grads(self, degraded, latents, jacobian, reg_fidelity, reg_image, diag_image, reg_kernel,
                            diag_kernel, weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel):
        th.autograd.set_detect_anomaly(True)

        images, kernels = latents
        degraded.requires_grad = True
        images.requires_grad = True
        kernels.requires_grad = True

        reg_fidelity_filters = reg_fidelity.filters
        reg_fidelity_filters.requires_grad = True
        reg_fidelity.filters = reg_fidelity_filters

        reg_image_filters = reg_image.filters
        reg_image_filters.requires_grad = True
        reg_image.filters = reg_image_filters

        diag_image_vector = diag_image.diagonal_vector
        diag_image_vector.requires_grad = True
        diag_image.diagonal_vector = diag_image_vector

        reg_kernel_filters = reg_kernel.filters
        reg_kernel_filters.requires_grad = True
        reg_kernel.filters = reg_kernel_filters

        diag_kernel_vector = diag_kernel.diagonal_vector
        diag_kernel_vector.requires_grad = True
        diag_kernel.diagonal_vector = diag_kernel_vector

        wdecay_image_vector = weight_decay_image.diagonal_vector
        wdecay_image_vector.requires_grad = True
        weight_decay_image.diagonal_vector = wdecay_image_vector

        wdecay_kernel_vector = weight_decay_kernel.diagonal_vector
        wdecay_kernel_vector.requires_grad = True
        weight_decay_kernel.diagonal_vector = wdecay_kernel_vector

        ssize_image_vector = step_size_image.diagonal_vector
        ssize_image_vector.requires_grad = True
        step_size_image.diagonal_vector = ssize_image_vector

        ssize_kernel_vector = step_size_kernel.diagonal_vector
        ssize_kernel_vector.requires_grad = True
        step_size_kernel.diagonal_vector = ssize_kernel_vector

        self.grad_custom(degraded, latents, jacobian, reg_fidelity, reg_image, diag_image, reg_kernel, diag_kernel,
                         weight_decay_image, weight_decay_kernel, step_size_image, step_size_kernel)
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

        return degraded_grad_custom,\
               images_grad_custom,\
               kernels_grad_custom,\
               reg_fidelity_filters_grad_custom,\
               reg_image_filters_grad_custom,\
               diag_image_vector_grad_custom,\
               reg_kernel_filters_grad_custom,\
               diag_kernel_vector_grad_custom,\
               wdecay_image_vector_grad_custom,\
               wdecay_kernel_vector_grad_custom,\
               ssize_image_vector_grad_custom,\
               ssize_kernel_vector_grad_custom
