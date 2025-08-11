import warnings
from typing import Tuple, Union, Iterable, Optional, Dict, Callable

import torch as th
from torch import nn

from irrn.operators import LearnableLinearSystemOperatorBase, LearnableLinearOperatorBase, LinearOperatorBase, \
    LearnableNumberOperator, LearnableDiagonalOperator, IdentityOperator, LinearDegradationOperatorBase, \
    LearnableConvolutionOperator, LearnableMatMulOperator
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator
from irrn.operators.linsys.irgn import assert_is_tensor
from irrn.utils import MultiVector
from .wiener import WienerFilteringSystemOperator, CirculantPreconditionOperator


def handler_called_error():
    raise RuntimeError('Method corresponding to quadratic problem was called in handler class, '
                       'which is unexpected behaviour. Call prepare_for_step before solving the system.')


class IRLSSystemOperatorHandler(LearnableLinearSystemOperatorBase):
    """
    This class implements a handler for IRLS, and can not be used directly for solving the system.
    """
    def __init__(self, degradation_operator: LinearDegradationOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
                 reg_images: LearnableLinearOperatorBase,
                 weight_decay: Union[LearnableNumberOperator, Iterable[LearnableNumberOperator]],
                 reg_backbone: nn.Module, weight_noise_backbone: Optional[nn.Module] = None,
                 weight_noise_operator: Optional[LearnableNumberOperator] = None,
                 zero_step_system: Optional[LearnableLinearSystemOperatorBase] = None,
                 use_weight_noise_scaling_coef: bool = True, use_circulant_precond: bool = False,
                 use_circulant_precond_backward: bool = False,
                 use_equilibration_precond: bool = False,
                 use_equilibration_precond_backward: bool = False,
                 grad_residual_stability_const: float = 0.) -> None:
        """
        Initializing operators

        :param degradation_operator:  (H)           linear operator, representing degradation matrix
        :param reg_fidelity:          (\Phi)        learned linear operators, representing data fidelity regularization
        :param reg_images:            (G)           learned linear operator, representing regularization operator
        :param weight_decay:          (\psi(\beta)) learned number, representing decay regularization weight
        :param reg_backbone:          (f())         neural network, representing gradient of regularization function
        :param weight_noise_backbone: (\alpha)      neural network, representing noise-specific scaler
        :param weight_noise_operator: (\alpha)      noise-specific scaler prepared for restoration
        :param zero_step_system: system, which is independent of previous solution and may be used for initialization
        :param use_weight_noise_scaling_coef: wheter to use scaling of data fildelity with 1/\sigma^2
        :param use_circulant_precond: flag which determines whether to use preconditioners based on closest circulant
                                      approximation of system matrix
        :param use_circulant_precond_backward: flag which determines whether to use preconditioners based on closest
                                               circulant approximation of system matrix during backward
        """
        super(IRLSSystemOperatorHandler, self).__init__()
        assert isinstance(degradation_operator, LinearDegradationOperatorBase)
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_images, LearnableLinearOperatorBase)
        assert isinstance(weight_decay, (list, tuple, LearnableNumberOperator, nn.ModuleList))
        assert isinstance(zero_step_system, (LearnableLinearSystemOperatorBase, type(None)))
        self.degradation = degradation_operator
        self.reg_fidelity = reg_fidelity
        self.reg_images = reg_images
        if isinstance(weight_decay, (list, tuple)):
            weight_decay = nn.ModuleList(weight_decay)
        self.weight_decay = weight_decay
        self.backbone = reg_backbone
        self.weight_noise_backbone = weight_noise_backbone
        self.weight_noise_operator = weight_noise_operator
        self.zero_step_system = zero_step_system
        self.use_weight_noise_scaling_coef = use_weight_noise_scaling_coef
        self.use_circulant_precond = use_circulant_precond
        self.use_circulant_precond_backward = use_circulant_precond_backward
        self.use_equilibration_precond = use_equilibration_precond
        self.use_equilibration_precond_backward = use_equilibration_precond_backward
        self.grad_residual_stablility_const = grad_residual_stability_const
        assert not (use_circulant_precond and use_equilibration_precond)
        assert not (use_circulant_precond_backward and use_equilibration_precond_backward)
        if use_circulant_precond or use_circulant_precond_backward:
            assert isinstance(reg_fidelity, (IdentityOperator, LearnableConvolutionOperator))
            assert isinstance(reg_images, (IdentityOperator, LearnableConvolutionOperator))
            assert isinstance(degradation_operator, (IdentityOperator, ConvDecimateLinearDegradationOperator))
        elif use_circulant_precond_backward:
            assert hasattr(reg_backbone, 'diag_of_grad_matrix')
        if use_equilibration_precond or use_equilibration_precond_backward:
            assert isinstance(reg_fidelity, (IdentityOperator, LearnableConvolutionOperator))
        if zero_step_system is not None:
            zero_step_system.use_circulant_precond = use_circulant_precond
            zero_step_system.use_equilibration_precond = use_equilibration_precond

    def prepare_for_restoration(self, noise_std: Optional[th.Tensor] = None, **other_kwargs
                                ) -> 'LearnableLinearSystemOperatorBase':
        """
        This method initializes the system based on kernels (if any available) and noise standard deviation

        :param noise_std: standard deviation of noise (if known) for restoration
        :param other_kwargs: other arguments which are required for system parametrization
        :return: system prepared for restoration
        """
        if self.use_weight_noise_scaling_coef and noise_std is not None:
            assert_is_tensor(noise_std)
            weight_noise = 1/(noise_std**2)
            if self.weight_noise_backbone is not None:
                weight_noise = self.weight_noise_backbone(weight_noise)
            weight_noise = weight_noise.unsqueeze(-1).unsqueeze(-1)
            weight_noise = LearnableNumberOperator(scale_weight=weight_noise, function=lambda x: x, learnable=False)
        else:
            weight_noise = IdentityOperator()
        self.degradation = self.degradation.init_with_parameters(noise_std=noise_std, **other_kwargs)

        self.reg_fidelity.prepare_for_restoration(noise_std=noise_std,
                                                  degradation_operator=self.degradation, **other_kwargs)
        self.reg_images.prepare_for_restoration(noise_std=noise_std,
                                                degradation_operator=self.degradation, **other_kwargs)
        self.weight_noise_operator = weight_noise

        if self.zero_step_system is not None:
            self.zero_step_system.prepare_for_restoration(noise_std=noise_std, degradation_operator=self.degradation,
                                                          **other_kwargs)
        self.precomputed_precond: Optional[Dict[str, th.Tensor]] = None
        return self

    def prepare_for_step(self, step_idx: int, degraded: th.Tensor, latents: Tuple[th.Tensor, ...],
                         **system_kwargs) -> Union['IRLSSystemOperator', 'WienerFilteringSystemOperator']:
        """
        This method creates linear system with all necessary parameters for the next optimization step.
        Since each IRLS step is solved as a quadratic problem, this method performs regularizer linearization and
        returns corresponding system object for optimization.

        :param step_idx: index of current step (step number)
        :param degraded: batch of observations required for restoration
        :param latents: tuple with latent images of shape [B, C, H, W], which parametrize linear system at current step
        :param system_kwargs: arguments which are required for system parametrization
        :return: quadratic system for next optimization step
        """
        assert len(latents) == 1
        images = latents[0]
        if step_idx != 0:
            assert_is_tensor(images)
        assert_is_tensor(degraded)
        assert self.weight_noise_operator is not None

        if step_idx == 0:
            if self.zero_step_system is not None:
                system = self.zero_step_system.prepare_for_step(step_idx, degraded, *latents)
            else:
                warnings.warn('You are using zero step Wiener system with the same operators, as in IRLS system. '
                              'This is not a best initialization and may lead to sub-par restoration quality.')
                system = WienerFilteringSystemOperator(self.degradation,
                                                       self.reg_fidelity.prepare_for_step(step_idx, degraded),
                                                       self.reg_images.prepare_for_step(step_idx, degraded),
                                                       self.weight_noise_operator.prepare_for_step(step_idx, degraded)
                                                       ).prepare_for_step(step_idx, degraded, *latents)
        else:
            reg_fidelity = self.reg_fidelity.prepare_for_step(step_idx, *latents)
            reg_images = self.reg_images.prepare_for_step(step_idx, *latents)
            weight_noise = self.weight_noise_operator.prepare_for_step(step_idx, *latents)

            reg_backbone_linear = self.linear_reg_backbone(reg_images(images), step_num=step_idx)

            if isinstance(self.weight_decay, nn.ModuleList):
                if step_idx >= len(self.weight_decay):
                    weight_decay = self.weight_decay[-1]
                else:
                    weight_decay = self.weight_decay[step_idx - 1]
            else:
                weight_decay = self.weight_decay
            weight_decay = weight_decay.prepare_for_step(step_idx, *latents)

            if self.use_circulant_precond:
                precond_op = \
                    self.get_circulant_approx_precond(self.degradation, reg_fidelity, reg_images,
                                                      reg_backbone_linear, weight_noise, weight_decay, latents[0].shape)
                flag_precond_bkwrd = self.use_circulant_precond_backward
            elif self.use_equilibration_precond:
                precond_op = \
                    self.get_equilibration_precond(self.degradation, reg_fidelity, reg_images, reg_backbone_linear,
                                                   weight_noise, weight_decay, degraded, *latents)
                flag_precond_bkwrd = self.use_equilibration_precond_backward
            else:
                precond_op = IdentityOperator()
                flag_precond_bkwrd = False
            system = IRLSSystemOperator(self.degradation, reg_fidelity, reg_images, weight_decay,
                                        reg_backbone_linear, weight_noise, step_idx,
                                        preconditioner_right_inv=precond_op, preconditioner_left_inv=precond_op,
                                        transpose_op_with_precond=flag_precond_bkwrd)
        return system

    @staticmethod
    @th.no_grad()
    def get_data_fidelity_circulant_approx(degradation: Union[IdentityOperator, ConvDecimateLinearDegradationOperator],
                                           reg_fidelity: Union[IdentityOperator, LearnableConvolutionOperator],
                                           shape_of_incoming_vector: Tuple[int]) -> Union[float, th.Tensor]:
        """
        This method finds circulant approximation for data fidelity related term of IRLS system, which involves
        degradation and data fidelity regularization linear operators.

        :param degradation: degradation operator in data fildelity term
        :param reg_fidelity: regularization operator in data fildelity term
        :param shape_of_incoming_vector: dimensions of approximation operator
        :return: closest circulant approximation of data fidelity related term of IRLS system at current step
        """
        num_in_channels = shape_of_incoming_vector[-3]
        if isinstance(degradation, ConvDecimateLinearDegradationOperator) and \
                isinstance(reg_fidelity, LearnableConvolutionOperator):
            assert not reg_fidelity.mix_in_channels
            assert degradation.scale_factor == 1
            h_h, w_h = degradation.kernel.shape[-2:]
            h_p, w_p = reg_fidelity.effective_kernel.shape[-2:]
            h_e = h_h + h_p - 1
            w_e = w_h + w_p - 1
            pad_left = w_e // 2
            pad_right = pad_left - 1 + w_e % 2
            pad_top = h_e // 2
            pad_bottom = pad_top - 1 + h_e % 2

            kernel = reg_fidelity.get_dirac_kernel(1 + 2*(pad_top + pad_bottom), 1 + 2*(pad_left + pad_right), 1)
            kernel = kernel.to(degradation.kernel_unified_shape)
            kernel = kernel.repeat(degradation.kernel_unified_shape.shape[0], num_in_channels, 1, 1)
            kernel = reg_fidelity(degradation(kernel))
            if kernel.dim() == 4:
                assert num_in_channels == 1
                kernel = kernel.unsqueeze(2)
            ret = degradation.get_circulant_abs_squared_eigvals_with_kernels(kernel, shape_of_incoming_vector)
            if degradation.pad_operator.pad_mode is None and reg_fidelity.pad_operator.pad_mode is None:
                h_in, w_in = shape_of_incoming_vector[-2:]
                ones_rate = (h_in - kernel.shape[-2] + 1) * (w_in - kernel.shape[-2] + 1) / (h_in * w_in)
                ret *= ones_rate
        elif isinstance(reg_fidelity, LearnableConvolutionOperator):
            assert isinstance(degradation, IdentityOperator)
            assert not reg_fidelity.mix_in_channels
            ret = reg_fidelity.eigvals_of_transpose_apply_circulant_approx(shape_of_incoming_vector)
        elif isinstance(degradation, ConvDecimateLinearDegradationOperator):
            assert isinstance(reg_fidelity, IdentityOperator)
            ret = degradation.eigvals_of_transpose_apply_circulant_approx(shape_of_incoming_vector)
        else:
            assert isinstance(degradation, IdentityOperator)
            assert isinstance(reg_fidelity, IdentityOperator)
            ret = 1.
        return ret

    @th.no_grad()
    def get_circulant_approx_precond(
            self, degradation: Union[IdentityOperator, ConvDecimateLinearDegradationOperator],
            reg_fidelity: Union[IdentityOperator, LearnableConvolutionOperator],
            reg_images: LearnableConvolutionOperator, reg_backbone_linear: LearnableDiagonalOperator,
            weight_noise: Union[IdentityOperator, LearnableNumberOperator], weight_decay: LearnableNumberOperator,
            shape_of_incoming_vector: Tuple[int]):
        """
        This method finds operator representing closest approximation of linear system matrix by circulant matrix
        and returns operator, corresponding to this closest approximation.

        :param step_idx: current step index, allows to reuse computations
        :param degradation: degradation operator of IRLS system
        :param reg_fidelity: data fidelity regulariser of IRLS system
        :param reg_images: images regularizer of IRLS system
        :param reg_backbone_linear: diagonal operator, corresponding to linearization at current step
        :param weight_noise: operator, corresponding to noise injection
        :param weight_decay: operator, penalizing difference of current and previous solutions
        :param shape_of_incoming_vector: dimensions of approximation operator
        :return: eigvals of preconditioner, representing inverse square root of closest circulant approximation
                 of IRLS system at current step
        """
        assert not reg_images.mix_in_channels
        precond_vec = self.get_data_fidelity_circulant_approx(degradation, reg_fidelity, shape_of_incoming_vector)
        precond_vec = weight_noise(precond_vec) + reg_images.eigvals_of_transpose_apply_circulant_approx(
            shape_of_incoming_vector, diagonal_vector_between=reg_backbone_linear.rescaled_diagonal_vector) + \
                      weight_decay.rescaled_diagonal_vector
        precond_op = LearnableDiagonalOperator(precond_vec.pow_(-0.5), function=lambda x: x, learnable=False)
        precond_op = CirculantPreconditionOperator(precond_op)
        return precond_op

    @th.no_grad()
    def get_equilibration_precond(
            self, degradation: Union[IdentityOperator, ConvDecimateLinearDegradationOperator],
            reg_fidelity: Union[IdentityOperator, LearnableConvolutionOperator],
            reg_images: LearnableConvolutionOperator, reg_backbone_linear: LearnableDiagonalOperator,
            weight_noise: Union[IdentityOperator, LearnableNumberOperator], weight_decay: LearnableNumberOperator,
            degraded: th.Tensor, latent: th.Tensor) -> LearnableDiagonalOperator:
        """
        This method finds diagonal operator, which equilibrates columns of normal equations matrix when multiplied by
        from the right. Each column is then normalized by the scalar containing inverse Euclidean distance of this
        column.

        :param degradation: degradation operator of IRLS system
        :param reg_fidelity: data fidelity regulariser of IRLS system
        :param reg_images: images regularizer of IRLS system
        :param reg_backbone_linear: diagonal operator, corresponding to linearization at current step
        :param weight_noise: operator, corresponding to noise injection
        :param weight_decay: operator, penalizing difference of current and previous solutions
        :param degraded: degraded samples, required to extract shape, dtype and device information
        :param latent: latent estimates, required to extract shape, dtype and device information
        :return: diagonal operator, which normalizes the columns of normal equations matrix when multiplied by from the left
        """
        if isinstance(reg_fidelity, LearnableConvolutionOperator):
            if isinstance(degradation, ConvDecimateLinearDegradationOperator) and degradation.kernel is not None:
                assert degradation.scale_factor == 1
                assert degradation.pad_operator.pad_mode == reg_fidelity.pad_operator.pad_mode
                dirac_kernel = degradation.kernel_flipped
                if dirac_kernel.shape[1] == 1:
                    dirac_kernel = dirac_kernel.expand(-1, reg_fidelity.kernel_unified_shape.shape[2], -1, -1)
                dirac_kernel = reg_fidelity.pad_operator.zero_pad(dirac_kernel, n=2)
                kernel = reg_fidelity(dirac_kernel).flip(dims=(-1, -2))
                if kernel.ndim == 4:
                    kernel = kernel.unsqueeze(2).expand(-1, -1, reg_fidelity.kernel_unified_shape.shape[2], -1, -1)
                deg_equilib = weight_noise(reg_fidelity.get_cols_squared_norms(reg_fidelity(degraded), override_kernel=kernel))
            elif isinstance(degradation, IdentityOperator):
                deg_equilib = weight_noise(reg_fidelity(degradation.get_cols_squared_norms(degraded)))
            else:
                raise NotImplementedError
        elif isinstance(reg_fidelity, IdentityOperator):
            deg_equilib = weight_noise(reg_fidelity(degradation.get_cols_squared_norms(degraded)))  # 1/s^2*phi*c_h^2
        elif isinstance(reg_fidelity, LearnableNumberOperator):
            deg_equilib = weight_noise(reg_fidelity(degradation.get_cols_squared_norms(degraded), n=2))  # 1/s^2*phi*c_h^2
        elif isinstance(reg_fidelity, LearnableDiagonalOperator):
            deg_equilib = weight_noise(degradation.get_cols_squared_norms(
                degraded, diag_at_left=reg_fidelity.rescaled_diagonal_vector))  # 1/s^2*phi*c_h^2
        else:
            raise NotImplementedError
        if isinstance(reg_backbone_linear, LearnableDiagonalOperator):
            wg_equilib = reg_images.get_cols_squared_norms(
                latent, diag_at_left=reg_backbone_linear.rescaled_diagonal_vector)  # w * c_g^2
        elif isinstance(reg_backbone_linear, LearnableMatMulOperator):
            res_w_squared = reg_backbone_linear.sqrt_operator.elementwise_squared_operator.T(
                th.ones_like(reg_images(latent)))
            wg_equilib = reg_images.get_cols_squared_norms(latent, diag_at_left=res_w_squared)  # w * c_g^2
        alpha_equilib = weight_decay(th.ones_like(latent))  # \alpha
        precond_op = LearnableDiagonalOperator((deg_equilib.clamp(0) + wg_equilib.clamp(0) +
                                                alpha_equilib).pow_(-0.5),
                                               function=lambda x: x, learnable=False)
        return precond_op

    def linear_reg_backbone(self, images: th.Tensor, step_num: Optional[int] = None) -> \
            Union[LearnableDiagonalOperator, LearnableMatMulOperator]:
        """
        Auxilary method, which returns diagonal approximation (Q matrix) of regularizer function in given point.

        :param images: vector, at which linearization should be performed
        :param step_num: index of current recurrent step
        :return: diagonal approximation of regularizer function
        """
        diag_vec = self.call_backbone(images, step_num)
        reg_backbone_linear = self.linearized_reg_operator(diag_vec)
        return reg_backbone_linear

    def call_backbone(self, vector: th.Tensor, step_idx: int) -> th.Tensor:
        """
        Auxiliary method, which calls IRLS backbone with additional parameters

        :param vector: vector to call backbone on
        :param step_idx: index of current step (step number)
        :return: result of called backbone
        """
        return self.backbone(vector)

    @staticmethod
    def linearized_reg_operator(backbone_output: th.Tensor) -> LearnableLinearOperatorBase:
        """
        This method creates linear operator parameterized by the output of regularization backbone.
        By default returns diagonal operator, which assumes incoming tensor as a vector.

        :param backbone_output: output from a backbone to parametrize operator
        :return: linear operator parametrized by backbone output
        """
        assert th.all(backbone_output >= 0), \
            "Linearization of regularization function (Q matrix) should be positive definite"
        reg_backbone_linear = \
            LearnableDiagonalOperator(diagonal_vector=backbone_output, function=lambda x: x, learnable=False)
        return reg_backbone_linear

    def to(self, *args, **kwargs) -> 'IRLSSystemOperatorHandler':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self.degradation = self.degradation.to(*args, **kwargs)
        self.reg_fidelity = self.reg_fidelity.to(*args, **kwargs)
        self.reg_images = self.reg_images.to(*args, **kwargs)
        if isinstance(self.weight_decay, (list, tuple)):
            self.weight_decay = [weight_decay.to(*args, **kwargs) for weight_decay in self.weight_decay]
        else:
            self.weight_decay = self.weight_decay.to(*args, **kwargs)
        self.backbone = self.backbone.to(*args, **kwargs)
        if self.weight_noise_backbone is not None:
            self.weight_noise_backbone = self.weight_noise_backbone.to(*args, **kwargs)
        if self.zero_step_system is not None:
            self.zero_step_system = self.zero_step_system.to(*args, **kwargs)
        return self

    def perform_step(self, inputs: Tuple[th.Tensor], solutions: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        handler_called_error()

    def apply(self, *args, **kwargs) -> th.Tensor:
        handler_called_error()

    def right_hand_side(self,  *args, **kwargs) -> th.Tensor:
        handler_called_error()

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        ret = (*self.reg_fidelity.tensors_for_grad, *self.reg_images.tensors_for_grad)
        if hasattr(self.backbone, 'tensors_for_grad'):
            ret += self.backbone.tensors_for_grad
        else:
            ret += tuple(self.backbone.parameters())
        if self.weight_noise_operator is not None:
            ret += self.weight_noise_operator.tensors_for_grad
        elif self.weight_noise_backbone is not None:
            ret += tuple(self.weight_noise_backbone.parameters())
        return ret

    def residual(self, degraded: th.Tensor, *solution: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method computes residual of IRLS problem at solution point.

        :param degraded: observations which are restored
        :param solution: solution point, at which residual should be evaluated
        :return: value of residual in solution point
        """
        solution_vector = MultiVector(solution)

        reg_fidelity = self.reg_fidelity.prepare_for_step(-1, *solution)
        reg_images = self.reg_images.prepare_for_step(-1, *solution)
        weight_noise = self.weight_noise_operator.prepare_for_step(-1, *solution)
        reg_images_solution = reg_images(solution_vector)

        ret = self.degradation(solution_vector) - degraded
        ret = weight_noise(self.degradation.T(reg_fidelity.transpose_apply(ret))) + \
              reg_images.T(self.linear_reg_backbone(reg_images_solution)(reg_images_solution), solution_vector.shape)
        return (ret, )

    def grad_residual_operator(self, *solution) -> 'IRLSGradResidualOperator':
        """
        This method returns grad residual operator as an instance of IRLSGradResidualOperator.

        :param solution: solution point w.r.t. which to calculate gradient in grad residual operator
        :return: instance of IRLSGradResidualOperator, representing gradient of residual w.r.t. solution vector
        """
        return IRLSGradResidualOperator(self, *solution, stability_const=self.grad_residual_stablility_const)

    def residual_with_grad_residual_operator(self, degraded: th.Tensor, *solution: th.Tensor
                                             ) -> Tuple[Union[th.Tensor, MultiVector], LinearOperatorBase]:
        """
        This method computes residual based on given solution and constructs operator which represents gradient of
        residual w.r.t. solution. Both residual and grad residual operator are used in
        backpropagation based on implicit function theorem. Grad residual operator is returned based on
        autograd. If it is possible, returned operator is preconditioned by its closest circulant approximation.

        :param degraded: batch of observations under restoration
        :param solution: solution, w.r.t. which the gradient of residual should be computed
        :return: linear operator, which represents gradient of residual w.r.t. given solution
        """
        grad_r_op = self.grad_residual_operator(*solution)

        solution_vector = grad_r_op.solution_vector
        reg_fidelity = grad_r_op.reg_fidelity
        reg_images = grad_r_op.reg_images
        weight_noise = grad_r_op.weight_noise
        reg_images_solution = grad_r_op.reg_images_solution

        reg_backbone_linear = self.linear_reg_backbone(reg_images_solution, -1)
        residual = self.degradation(solution_vector) - degraded
        residual = weight_noise(self.degradation.T(reg_fidelity.transpose_apply(residual))) + \
              reg_images.T(reg_backbone_linear(reg_images_solution), solution_vector.shape)

        if self.use_circulant_precond_backward:
            precond_op = self.get_precond_operator_backward_circulant_approx(weight_noise, reg_fidelity, reg_images,
                                                                             reg_backbone_linear, reg_images_solution,
                                                                             *solution)
            grad_r_op.preconditioner_left_inv = precond_op
            grad_r_op.preconditioner_right_inv = precond_op
        return residual, grad_r_op

    @th.no_grad()
    def get_precond_operator_backward_circulant_approx(
            self, weight_noise: LinearOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
            reg_images: LearnableLinearOperatorBase,
            reg_backbone_linear: LearnableDiagonalOperator,
            reg_images_solution: th.Tensor, *solution: th.Tensor):
        """
        This method returns preconditioner operator for IRLS backward system using closest
        circulant approximation of it.

        :param weight_noise: noise-dependent operator in data fidelity term
        :param reg_fidelity: learned linear operator, representing data fidelity regularization
        :param reg_images: learned linear operator, representing regularization operator (features extractor)
        :param reg_backbone_linear: regularization backbone, linearized in solution point
        :param reg_images_solution: features extracted by reg_images features extractor from solution point
        :param solution: solution point
        :return: preconditioner for IRLS residual grad linear system based on its closest circulant approximation
        """
        assert isinstance(reg_backbone_linear, LearnableDiagonalOperator)
        diag_to_precond = \
            self.backbone.diag_of_grad_matrix(reg_backbone_linear.rescaled_diagonal_vector, reg_images_solution)
        precond_vec = self.get_data_fidelity_circulant_approx(self.degradation, reg_fidelity, solution[0].shape)
        precond_vec = \
            weight_noise(precond_vec) + \
            reg_images.eigvals_of_transpose_apply_circulant_approx(solution[0].shape,
                                                                   diagonal_vector_between=diag_to_precond)
        precond_op = LearnableDiagonalOperator(precond_vec.pow_(-0.5), function=lambda x: x, learnable=False)
        precond_op = CirculantPreconditionOperator(precond_op)
        return precond_op


class IRLSSystemOperator(WienerFilteringSystemOperator):
    """
    This class implements linear system corresponding to quadratic subproblem involved in IRLS:
    (\frac{1}{\alpha(\sigma)} H^T \Phi^T \Phi H + G^T Q(x^k) G + \psi(\beta)I_n)x^{k+1} =
    \frac{1}{\alpha(\sigma)} H^T \Phi^T \Phi y + \psi(\beta)x^k
    """
    def __init__(self, degradation_operator: LinearOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
                 reg_images: LearnableLinearOperatorBase, weight_decay: LearnableNumberOperator,
                 reg_backbone_linear: LearnableLinearOperatorBase, weight_noise: LearnableNumberOperator, step_idx: int,
                 preconditioner_right_inv: LinearOperatorBase = IdentityOperator(),
                 preconditioner_left_inv: LinearOperatorBase = IdentityOperator(),
                 transpose_op_with_precond: bool = True) -> None:
        """
        Initializing operators

        :param degradation_operator: (H)           linear operator, representing degradation matrix
        :param reg_fidelity:         (\Phi)        learned linear operators, representing data fidelity regularization
        :param reg_images:           (G)           learned linear operator, representing regularization operator
        :param weight_decay:         (\psi(\beta)) learned number, representing decay regularization weight
        :param reg_backbone_linear:  (Q)           neural network related operator - linearized regularizer
        :param weight_noise:         (\alpha)      operator, representing noise-specific scaler
        :param step_idx: index of current recurrent step
        :param preconditioner_left_inv: inverse of left preconditioner matrix
        :param preconditioner_right_inv: inverse of right preconditioner matrix
        :param transpose_op_with_precond: flag which determines whether to use preconditioners for transpose operator
        """
        super(IRLSSystemOperator, self).__init__(degradation_operator, reg_fidelity, reg_images, weight_noise)
        assert isinstance(weight_decay, LearnableNumberOperator)
        assert isinstance(reg_backbone_linear, (IdentityOperator, LearnableDiagonalOperator, LearnableMatMulOperator))
        self.weight_decay = weight_decay
        self.reg_backbone = reg_backbone_linear
        self.preconditioner_right_inv = preconditioner_right_inv
        self.preconditioner_left_inv = preconditioner_left_inv
        self.step_idx = step_idx
        self.transpose_op_with_precond = transpose_op_with_precond

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched left hand side linear transformation of input vector.

        :param vector: input vector of shape [B, C, H, W] to be transformed by linear operator
        :return: vector of shape [B, C, H, W], transformed by linear operator
        """
        assert_is_tensor(vector)
        if self.reg_fidelity.__class__ == IdentityOperator:
            operator_between = None
        else:
            operator_between = self.reg_fidelity.transpose_apply
        lhs = self.weight_noise(
            self.degradation.transpose_apply(vector, operator_between=operator_between)) + \
              self.reg_images.transpose_apply(vector, operator_between=self.reg_backbone) + self.weight_decay(vector)

        return lhs

    @staticmethod
    def _clamp(x: th.Tensor, min_value: float = -1., max_value: float = 1.) -> th.Tensor:
        """
        This auxiliary method performs differentiable clamping operation using reparameterization trick.

        :param x: input tensor to be clamped
        :param min_value: lower bound to clip values
        :param max_value: upper bound to clip values
        :return: clamped input
        """
        x_detached = x.detach()
        return x + x_detached.clamp(min_value, max_value) - x_detached

    def right_hand_side(self, degraded: th.Tensor, latent: th.Tensor) -> th.Tensor:
        """
        This method implements right hand side of linear system, given parametrization arguments

        :param degraded: batch of observations required for restoration
        :param latent: batch of latent images of shape [B, C, H, W], which parametrize linear system at current step
        :return: right hand side of linear system
        """
        assert_is_tensor(degraded)
        assert_is_tensor(latent)

        rhs = super(IRLSSystemOperator, self).right_hand_side(degraded) + self.weight_decay(latent)

        return rhs

    @staticmethod
    def norm(vector: th.Tensor, dims: Tuple[int]) -> th.Tensor:
        """
        Auxiliary method, which computes L2 norm of vector in dimensions, specified by self.vector_dims.

        :param vector: batch of vectors, which norm should be computed
        :param dims at which dimensions to compute norm
        :return: norm of input batch of vectors
        """
        return vector.pow(2).sum(dim=dims, keepdim=True).pow(0.5)

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        """
        This method should return all tensors, which may require gradients computation

        :return: tuple with tensors, that may require gradients computation
        """
        return (*super(IRLSSystemOperator, self).tensors_for_grad, *self.weight_decay.tensors_for_grad,
                *self.reg_backbone.tensors_for_grad)

    def to(self, *args, **kwargs) -> 'IRLSSystemOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        super(IRLSSystemOperator, self).to(*args, **kwargs)
        self.weight_decay = self.weight_decay.to(*args, **kwargs)
        self.reg_backbone = self.reg_backbone.to(*args, **kwargs)
        return self

    @property
    def T_operator(self) -> LinearOperatorBase:
        return IRLSSystemOperator(self.degradation, self.reg_fidelity, self.reg_images, self.weight_decay,
                                  self.reg_backbone, self.weight_noise, self.step_idx,
                                  preconditioner_left_inv=self.preconditioner_right_inv.T_operator,
                                  preconditioner_right_inv=self.preconditioner_left_inv.T_operator)

    @property
    def normal_equations_system(self):
        return IRLSNormalEquationsSystemOperator(
            self.degradation, self.reg_fidelity, self.reg_images, self.weight_decay, self.reg_backbone,
            self.weight_noise, self.step_idx, weight_decay_sqrt=None, reg_backbone_linear_sqrt=None,
            weight_noise_sqrt=None, preconditioner_sym_inv=self.preconditioner_right_inv)


class IRLSNormalEquationsSystemOperator(IRLSSystemOperator):
    """
    This class implements linear system corresponding to normal equations of quadratic subproblem involved in IRLS:
    C^T Cx = C^T b, where:
    C = [\frac{1}{\sqrt(\alpha(\sigma))} \Phi H, \sqrt(Q(x^k)) G, \sqrt(\psi(\beta)) I_n]^T
    b = [y, 0, \sqrt(\psi(\beta))]^T
    """
    def __init__(self, degradation_operator: LinearOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
                 reg_images: LearnableLinearOperatorBase, weight_decay: LearnableNumberOperator,
                 reg_backbone_linear: Optional[LearnableLinearOperatorBase],
                 weight_noise: Optional[LearnableNumberOperator], step_idx: int,
                 weight_decay_sqrt: Optional[LearnableNumberOperator] = None,
                 reg_backbone_linear_sqrt: Optional[LearnableLinearOperatorBase] = None,
                 weight_noise_sqrt: Optional[LearnableNumberOperator] = None,
                 preconditioner_sym_inv: LinearOperatorBase = IdentityOperator()) -> None:
        """
        Initializing operators. If square roots for any of Q (reg_backbone_linear), \psi and \alpha

        :param degradation_operator: (H) linear operator, representing degradation matrix
        :param reg_fidelity: (\Phi) learned linear operators, representing data fidelity regularization
        :param reg_images: (G) learned linear operator, representing regularization operator
        :param weight_decay: (\psi(\beta)) learned number, representing decay regularization weight
        :param reg_backbone_linear: (Q) neural network related operator - linearized regularizer
        :param weight_noise: (\alpha) operator, representing noise-specific scaler
        :param step_idx: index of current recurrent step
        :param weight_decay_sqrt: (\sqrt((\psi(\beta)))) learned number, representing square root of decay
                                  regularization weight. If not available, is calculated from weight_decay
        :param reg_backbone_linear_sqrt: \sqrt(Q(x^k)) neural network related operator - square root of linearized
                                         regularizer. If not available, is calculated from reg_backbone_linear
        :param weight_noise_sqrt: \sqrt(\alpha(\sigma)) learned number, representing square root of
                                  noise-specific scaler. If not available, is calculated from weight_noise
        :param preconditioner_sym_inv: linear operator $P^{-1}$, determining inverse of symmmetric preconditioner:
                                       $P^{-T} C^T C P^{-1} Px = P^{-T} b$
        """
        nn.Module.__init__(self)
        assert isinstance(degradation_operator, LinearOperatorBase)
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_images, LearnableLinearOperatorBase)
        if weight_decay_sqrt is not None:
            assert isinstance(weight_decay_sqrt, (LearnableNumberOperator, IdentityOperator))
            self.weight_decay_sqrt = weight_decay_sqrt
            self.weight_decay = weight_decay_sqrt.quadratic_operator
        else:
            assert isinstance(weight_decay, (LearnableNumberOperator, IdentityOperator))
            self.weight_decay = weight_decay
            with th.no_grad():
                self.weight_decay_sqrt = weight_decay.sqrt_operator
        if weight_noise_sqrt is not None:
            assert isinstance(weight_noise_sqrt, (LearnableNumberOperator, IdentityOperator))
            self.weight_noise_sqrt = weight_noise_sqrt
            self.weight_noise = weight_noise_sqrt.quadratic_operator
        else:
            assert isinstance(weight_noise, (LearnableNumberOperator, IdentityOperator))
            self.weight_noise = weight_noise
            with th.no_grad():
                self.weight_noise_sqrt = weight_noise.sqrt_operator
        if reg_backbone_linear_sqrt is not None:
            assert isinstance(reg_backbone_linear_sqrt,
                              (LearnableDiagonalOperator, LearnableMatMulOperator, IdentityOperator))
            self.reg_backbone_sqrt = reg_backbone_linear_sqrt
            self.reg_backbone = reg_backbone_linear_sqrt.quadratic_operator
        else:
            assert isinstance(reg_backbone_linear,
                              (LearnableDiagonalOperator, LearnableMatMulOperator, IdentityOperator))
            self.reg_backbone = reg_backbone_linear
            with th.no_grad():
                self.reg_backbone_sqrt = reg_backbone_linear.sqrt_operator
        self.degradation = degradation_operator
        self.reg_fidelity = reg_fidelity
        self.reg_images = reg_images
        self.use_weight_noise_scaling_coef = True
        self.step_idx = step_idx
        self.preconditioner_sym_inv = preconditioner_sym_inv

    def apply(self, vector: th.Tensor, *args, **kwargs) -> MultiVector:
        """
        This method performs batched left hand side linear transformation of input vector.

        :param vector: input vector of shape [B, C, H, W] to be transformed by linear operator
        :return: MultiVector of shape [B, ...], transformed by linear operator
        """
        elem1 = self.weight_noise_sqrt(self.reg_fidelity(self.degradation(vector)))  # \Phi H x
        elem2 = self.reg_backbone_sqrt(self.reg_images(vector))  # \sqrt(W) G x
        elem3 = self.weight_decay_sqrt(vector)
        return MultiVector((elem1, elem2, elem3))

    def _transpose(self, vector: MultiVector, *args, **kwargs) -> th.Tensor:
        """
        This method applies transposed linear operation on input vector.

        :param vector: input vector of shape [B, ...] to apply transposed operation
        :param args, kwargs: aux parameters
        :return: result of transposed linear operation, applied to input vector
        """
        assert isinstance(vector, MultiVector)
        assert vector.num_elements == 3
        elem1, elem2, elem3 = tuple(vector)
        res = \
            self.weight_noise_sqrt.T(
                self.degradation.T(self.reg_fidelity.T(elem1, *args, **kwargs), *args, **kwargs)) + \
            self.reg_images.T(self.reg_backbone_sqrt.T(elem2, *args, **kwargs), *args, **kwargs) + \
            self.weight_decay_sqrt.T(elem3, *args, **kwargs)
        return res

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs) -> th.Tensor:
        if operator_between is None:
            return super(IRLSNormalEquationsSystemOperator, self).apply(vector, *args, **kwargs)
        else:
            return super(IRLSNormalEquationsSystemOperator, self).transpose_apply(
                vector, *args, operator_between=operator_between, **kwargs)

    def right_hand_side(self, degraded: th.Tensor, latent: th.Tensor) -> MultiVector:
        # TODO: replace with meta-tensor when conv operation is supported
        with th.no_grad():
            mid_shape = self.reg_images(latent)
        return MultiVector((self.weight_noise_sqrt(degraded),
                            th.zeros(*mid_shape.shape, dtype=latent.dtype, device=latent.device),
                            self.weight_decay_sqrt(latent)))
    
    def right_hand_side_full(self, degraded: th.Tensor, latent: th.Tensor) -> MultiVector:
        return super(IRLSNormalEquationsSystemOperator, self).right_hand_side(degraded, latent)
        
    def right_hand_side_extension_vector(self, *args) -> type(None):
        return None

    def embed_right_hand_side_for_backward(self, right_hand_side: th.Tensor, degraded: th.Tensor, latent: th.Tensor
                                           ) -> None:
        with th.no_grad():
            mid_shape = self.reg_images(latent)
        return MultiVector((th.zeros_like(degraded),
                            th.zeros(*mid_shape.shape, dtype=latent.dtype, device=latent.device),
                            self.weight_decay_sqrt.inv(right_hand_side)))

    @property
    def T_operator(self) -> LinearOperatorBase:
        return self

    def residual(self, solution: th.Tensor, degraded: th.Tensor, latent: th.Tensor) -> th.Tensor:
        return self.right_hand_side_full(degraded, latent) - \
               self.transpose_apply(solution, operator_between=None, inplace=False)
    
    def to(self, *args, **kwargs) -> 'IRLSNormalEquationsSystemOperator':
        super(IRLSNormalEquationsSystemOperator, self).to(*args, **kwargs)
        self.weight_decay_sqrt = self.weight_decay_sqrt.to(*args, **kwargs)
        self.weight_noise_sqrt = self.weight_noise_sqrt.to(*args, **kwargs)
        self.reg_backbone_sqrt = self.reg_backbone_sqrt.to(*args, **kwargs)
        return self


class IRLSGradResidualOperator(LinearOperatorBase):
    """
    This class represents gradient of residual for IRLS.

    IT WORKS ONLY UNDER MANDATORY ASSUMPTION:
    1. All linear operators involved in IRLS residual do not depend on solution point.
    2. For matmul backbone case matrices are constructed independently for each pixel without mixing them.
    These things are not checked internally, IT IS UP TO USER TO CONTROL THIS.
    """
    def __init__(self, irls_handler: IRLSSystemOperatorHandler, *solution_point: th.Tensor,
                 stability_const: float = 0.) -> None:
        self.preconditioner_right_inv = IdentityOperator()
        self.preconditioner_left_inv = IdentityOperator()
        self.solution_vector = MultiVector(solution_point)
        self.reg_fidelity = irls_handler.reg_fidelity.prepare_for_step(-1, *solution_point)
        self.reg_images = irls_handler.reg_images.prepare_for_step(-1, *solution_point)
        self.weight_noise = irls_handler.weight_noise_operator.prepare_for_step(-1, *solution_point)
        self.reg_images_solution = self.reg_images(self.solution_vector)
        self.input_for_grad = self.reg_images_solution.detach().requires_grad_(True)
        self.output_for_grad = irls_handler.call_backbone(self.input_for_grad, -1)
        self.reg_backbone_linearized_operator = irls_handler.linearized_reg_operator(self.output_for_grad.detach())
        assert isinstance(self.reg_backbone_linearized_operator, (LearnableDiagonalOperator, LearnableMatMulOperator))
        self.degradation = irls_handler.degradation
        assert isinstance(stability_const, float)
        self.stability_const = stability_const

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs multiplication of IRLS residual gradient matrix with incoming vector.
        Here only the gradient of regularizer (diagonal of Q matrix) is computed using autograd,
        the rest is computed from operators stored in IRLS system handler.

        :param vector: incoming vector to multiply with IRLS resudial gradient matrix
        :param args, kwargs: all other arguments
        :return: gradient of IRLS residual multiplied by input vector
        """
        ret = self.grad_of_data_fidelity_system_term(vector) + self.grad_of_regularizer_system_term(vector)
        if self.stability_const != 0.:
            ret = ret + vector*self.stability_const
        return ret

    def grad_of_data_fidelity_system_term(self, input_vector: th.Tensor) -> th.Tensor:
        """
        This method performs multiplication of gradient of IRLS data fidelity system term with incoming vector.
        Under assumptions stated in header operators in data fidelity term do not depend on solution point,
        so the required gradient is computed by calling data fidelity part of the system with the input vector.

        :param input_vector: incoming vector to multiply with IRLS data fidelity system term gradient matrix
        :return: gradient of IRLS data fidelity system term multiplied by input vector
        """
        ret = self.degradation.transpose_apply(
            input_vector, operator_between=self.reg_fidelity.transpose_apply)
        ret = self.weight_noise(ret)
        return ret

    def grad_of_regularizer_system_term(self, input_vector: th.Tensor) -> th.Tensor:
        """
        This method performs multiplication of gradient of IRLS regularization system term with incoming vector.
        Under assumptions stated in header images regularization linear operator G does not depend on solution point,
        so the required gradient of G^T Q G is computed by calling only the gradient of Q.

        :param input_vector: incoming vector to multiply with IRLS regularization system term gradient matrix
        :return: gradient of IRLS regularization system term multiplied by input vector
        """
        ret = self.reg_images.transpose_apply(
            input_vector, operator_between=self.grad_inner_part_of_regularizer_system_term)
        return ret

    def grad_inner_part_of_regularizer_system_term(self, input_vector: th.Tensor) -> th.Tensor:
        """
        This method performs multiplication of inner gradient of IRLS regularization system term with incoming vector.

        :param input_vector: incoming vector to multiply with IRLS regularization system term inner gradient matrix
        :return: inner gradient of IRLS regularization system term multiplied by input vector
        """
        #  out_vec: either Q(x*)Gv or W(x*)Gv
        out_vec = self.reg_backbone_linearized_operator(input_vector)
        if isinstance(self.reg_backbone_linearized_operator, LearnableDiagonalOperator):
            #  out_vec_part: diag(Gx*)Gv
            out_vec_part = self.reg_images_solution * input_vector
        elif isinstance(self.reg_backbone_linearized_operator, LearnableMatMulOperator):
            assert input_vector.dim() == self.reg_images_solution.dim() == 5
            #  out_vec_part: Gg Z^T
            # out_vec_part = th.matmul(input_vector.permute(0, 3, 4, 1, 2),
            #                          self.reg_images_solution.permute(0, 3, 4, 2, 1))
            out_vec_part = th.matmul(input_vector.permute(0, 3, 4, 2, 1),
                                     self.reg_images_solution.permute(0, 3, 4, 1, 2))
        else:
            raise NotImplementedError(
                'Grad residual operator is implemented only for cases when linearization of regularizer is either '
                'diagonal operator, or matmul operator. '
                f'Given operator of class {self.reg_backbone_linearized_operator.__class__}.')
        out_vec_part = th.autograd.grad(self.output_for_grad, self.input_for_grad, grad_outputs=out_vec_part,
                                        retain_graph=True, create_graph=False, only_inputs=True, allow_unused=False)[0]
        out_vec = out_vec + out_vec_part
        return out_vec
