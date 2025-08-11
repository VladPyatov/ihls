from typing import Union, Tuple, Optional, Callable

import torch as th
from torch import nn

from irrn.operators import LearnableLinearSystemOperatorBase, LinearDegradationOperatorBase, \
    LearnableLinearOperatorBase, LearnableNumberOperator, IdentityOperator, LinearOperatorBase, \
    LearnableConvolutionOperator, LearnableDiagonalOperator
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator, \
    ConvMosaicLinearDegradationOperator
from irrn.operators.linsys.irgn import assert_is_tensor
from irrn.utils import MultiVector
from irrn.utils.fft import rfft2, irfft2, fft2, ifft2


class CirculantPreconditionOperator(LinearOperatorBase):
    def __init__(self, diag_operator_between: LearnableDiagonalOperator):
        self.diag_operator_between = diag_operator_between

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return irfft2(self.diag_operator_between(rfft2(vector)), signal_sizes=vector.shape[-2:])

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.apply(vector, *args, **kwargs)

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return irfft2(self.diag_operator_between.inv(rfft2(vector)), signal_sizes=vector.shape[-2:])

    def inv_T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.inv(vector, *args, **kwargs)

    @property
    def T_operator(self) -> LinearOperatorBase:
        return self


class WienerFilteringSystemOperator(LearnableLinearSystemOperatorBase):
    """
    This class implements linear system corresponding to quadratic problem of Wiener filtering:
    argmin_x \frac{1}{\sigma^2} ||\Phi(H x - y)||_2^2 + ||Gx||_2^2,
    where one-shot inversion is implemented for denoising and deblurring tasks.
    """
    def __init__(self, degradation_operator: LinearDegradationOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
                 reg_images: LearnableLinearOperatorBase,
                 weight_noise: Union[LearnableNumberOperator, IdentityOperator],
                 use_weight_noise_scaling_coef: bool = True, use_circulant_precond: bool = False,
                 use_equilibration_precond: bool = False) -> None:
        """
        Initializing operators

        :param degradation_operator: (H)           linear operator, representing degradation matrix
        :param reg_fidelity:         (\Phi)        learned linear operators, representing data fidelity regularization
        :param reg_images:           (G)           learned linear operator, representing regularization operator
        :param weight_noise:         (\alpha)      operator, representing noise-specific scaler
        :param use_weight_noise_scaling_coef: whether to use scaling of data fildelity with 1/\sigma^2
        """
        super(WienerFilteringSystemOperator, self).__init__()
        assert isinstance(degradation_operator, LinearOperatorBase)
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_images, LearnableLinearOperatorBase)
        assert isinstance(weight_noise, (LearnableNumberOperator, IdentityOperator))
        assert isinstance(use_weight_noise_scaling_coef, bool)
        self.degradation = degradation_operator
        self.reg_fidelity = reg_fidelity
        self.reg_images = reg_images
        self.weight_noise = weight_noise
        self.use_weight_noise_scaling_coef = use_weight_noise_scaling_coef
        self.use_circulant_precond = use_circulant_precond
        self.use_equilibration_precond = use_equilibration_precond

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
        lhs = self.degradation.transpose_apply(vector, operator_between=operator_between)
        lhs = self.weight_noise(lhs) + self.reg_images.transpose_apply(vector)

        return lhs

    def right_hand_side(self, degraded: th.Tensor, *args) -> th.Tensor:
        """
        This method implements right hand side of linear system, given parametrization arguments

        :param degraded: batch of observations required for restoration
        :return: right hand side of linear system
        """
        assert_is_tensor(degraded)

        rhs = self.weight_noise(self.degradation.T(self.reg_fidelity.transpose_apply(degraded)))

        return rhs

    def perform_step(self, inputs: Tuple[th.Tensor], solutions: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        """
        This method makes a final update on a current solution based on a previous one

        :param inputs: batch of previous solutions to update
        :param solutions: batch of IRLS solutions for images at current step
        :return: updated solutions
        """
        return solutions

    def prepare_for_restoration(self, noise_std: Optional[th.Tensor] = None, **other_kwargs,
                                ) -> 'LearnableLinearSystemOperatorBase':
        """
        This method initializes the system based on kernels (if any available) and noise standard deviation

        :param noise_std: standard deviation of noise (if known) for restoration
        :param other_kwargs: other arguments which are required for system parametrization
        :return: system prepared for restoration
        """
        if self.use_weight_noise_scaling_coef and noise_std is not None:
            assert_is_tensor(noise_std)
            weight_noise = 1 / (noise_std * noise_std)
            weight_noise = weight_noise.unsqueeze(-1).unsqueeze(-1)
            self.weight_noise = \
                LearnableNumberOperator(scale_weight=weight_noise, function=lambda x: x, learnable=False)
        else:
            self.weight_noise = IdentityOperator()
        self.degradation.init_with_parameters_(noise_std=noise_std, **other_kwargs)
        self.reg_fidelity.prepare_for_restoration(noise_std=noise_std, **other_kwargs)
        self.reg_images.prepare_for_restoration(noise_std=noise_std, **other_kwargs)
        return self

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        """
        This method should return all tensors, which may require gradients computation

        :return: tuple with tensors, that may require gradients computation
        """
        return (*self.reg_fidelity.tensors_for_grad, *self.reg_images.tensors_for_grad,
                *self.weight_noise.tensors_for_grad)

    def to(self, *args, **kwargs) -> 'WienerFilteringSystemOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self.degradation = self.degradation.to(*args, **kwargs)
        self.reg_fidelity = self.reg_fidelity.to(*args, **kwargs)
        self.reg_images = self.reg_images.to(*args, **kwargs)
        self.weight_noise = self.weight_noise.to(*args, **kwargs)
        return self

    @staticmethod
    def check_operator_circulant(operator: LinearOperatorBase) -> None:
        """
        This method checks if operator represents a circulant matrix, and raises error if it is not the case

        :param operator: operator for which a check is required
        :return: Nothing
        """
        if isinstance(operator, (IdentityOperator, type(None), LearnableNumberOperator)):
            return
        assert isinstance(operator, (ConvDecimateLinearDegradationOperator, LearnableConvolutionOperator))
        assert operator.pad_operator.pad_mode == 'periodic'
        if isinstance(operator, ConvDecimateLinearDegradationOperator):
            assert operator.scale_factor == 1
        else:
            assert not operator.mix_in_channels

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements inversion of wiener filtering linear system matrix if it is circulant.
        Inversion is based on eigen decomposition of circulant matrix with FFTs and is performed in frequency domain.

        :param vector: incoming vector to be multiplied by inverse matrix
        :return: vector, multiplied by inverse of matrix represented by operator
        """
        self.check_operator_circulant(self.degradation)
        self.check_operator_circulant(self.reg_fidelity)
        self.check_operator_circulant(self.reg_images)
        ret = rfft2(vector)
        system_eigvals = self.degradation.get_circulant_abs_squared_eigvals(vector.shape)
        if not isinstance(self.reg_fidelity, IdentityOperator):
            system_eigvals = system_eigvals * self.reg_fidelity.get_circulant_abs_squared_eigvals(vector.shape)
        system_eigvals = self.weight_noise(system_eigvals)
        system_eigvals = system_eigvals + self.reg_images.get_circulant_abs_squared_eigvals(vector.shape)
        ret /= system_eigvals
        return irfft2(ret, signal_sizes=vector.shape[-2:])

    def inv_T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements transpose inversion of wiener filtering linear system matrix if it is circulant.
        Inversion is based on eigen decomposition of circulant matrix with FFTs and is performed in frequency domain.
        Since matrix of this linear system is symmetric, then operation is performed by calling .inv method.

        :param vector: incoming vector to be multiplied by inverse matrix
        :return: vector, multiplied by inverse of matrix represented by operator
        """
        return self.inv(vector, *args, **kwargs)

    @property
    def T_operator(self) -> LinearOperatorBase:
        return self

    def prepare_for_step(self, step_idx: int, degraded: th.Tensor, *latents, **kwargs
                         ) -> 'WienerFilteringSystemOperator':
        if self.use_equilibration_precond:
            latent = self.reg_images(self.degradation.T(degraded))
            self.preconditioner_right_inv = self.preconditioner_left_inv = self.get_equilibration_precond(
                self.degradation, self.reg_fidelity, self.reg_images, self.weight_noise, degraded, latent)
        elif self.use_circulant_precond:
            latent = self.reg_images(self.degradation.T(degraded))
            self.preconditioner_right_inv = self.preconditioner_left_inv = self.get_circulant_approx_precond(
                self.degradation, self.reg_fidelity, self.reg_images, self.weight_noise, latent.shape)
        return self

    @th.no_grad()
    def get_circulant_approx_precond(
            self, degradation: Union[IdentityOperator, ConvDecimateLinearDegradationOperator],
            reg_fidelity: Union[IdentityOperator, LearnableConvolutionOperator],
            reg_images: LearnableConvolutionOperator,
            weight_noise: Union[IdentityOperator, LearnableNumberOperator], shape_of_incoming_vector: Tuple[int]):
        """
        This method finds operator representing closest approximation of linear system matrix by circulant matrix
        and returns operator, corresponding to this closest approximation.

        :param degradation: degradation operator of IRLS system
        :param reg_fidelity: data fidelity regulariser of IRLS system
        :param reg_images: images regularizer of IRLS system
        :param weight_noise: operator, corresponding to noise injection
        :param shape_of_incoming_vector: dimensions of approximation operator
        :return: eigvals of preconditioner, representing inverse square root of closest circulant approximation
                 of IRLS system at current step
        """
        assert not reg_images.mix_in_channels
        precond_vec = self.get_data_fidelity_circulant_approx(degradation, reg_fidelity, shape_of_incoming_vector)
        precond_vec = weight_noise(precond_vec) + reg_images.eigvals_of_transpose_apply_circulant_approx(
            shape_of_incoming_vector, diagonal_vector_between=None)
        precond_op = LearnableDiagonalOperator(precond_vec.pow_(-0.5), function=lambda x: x, learnable=False)
        precond_op = CirculantPreconditionOperator(precond_op)
        return precond_op

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

            kernel = reg_fidelity.get_dirac_kernel(1 + 2 * (pad_top + pad_bottom), 1 + 2 * (pad_left + pad_right), 1)
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
    def get_equilibration_precond(
            self, degradation: Union[IdentityOperator, ConvDecimateLinearDegradationOperator],
            reg_fidelity: Union[IdentityOperator, LearnableConvolutionOperator],
            reg_images: LearnableConvolutionOperator,
            weight_noise: Union[IdentityOperator, LearnableNumberOperator],
            degraded: th.Tensor, latent: th.Tensor) -> LearnableDiagonalOperator:
        """
        This method finds diagonal operator, which equilibrates columns of normal equations matrix when multiplied by
        from the right. Each column is then normalized by the scalar containing inverse Euclidean distance of this
        column.

        :param degradation: degradation operator of IRLS system
        :param reg_fidelity: data fidelity regulariser of IRLS system
        :param reg_images: images regularizer of IRLS system
        :param weight_noise: operator, corresponding to noise injection
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
                deg_equilib = weight_noise(
                    reg_fidelity.get_cols_squared_norms(reg_fidelity(degraded), override_kernel=kernel))
            elif isinstance(degradation, IdentityOperator):
                deg_equilib = weight_noise(reg_fidelity(degradation.get_cols_squared_norms(degraded)))
            else:
                raise NotImplementedError
        elif isinstance(reg_fidelity, IdentityOperator):
            deg_equilib = weight_noise(reg_fidelity(degradation.get_cols_squared_norms(degraded)))  # 1/s^2*phi*c_h^2
        elif isinstance(reg_fidelity, LearnableNumberOperator):
            deg_equilib = weight_noise(
                reg_fidelity(degradation.get_cols_squared_norms(degraded), n=2))  # 1/s^2*phi*c_h^2
        elif isinstance(reg_fidelity, LearnableDiagonalOperator):
            deg_equilib = weight_noise(degradation.get_cols_squared_norms(
                degraded, diag_at_left=reg_fidelity.rescaled_diagonal_vector))  # 1/s^2*phi*c_h^2
        else:
            raise NotImplementedError

        wg_equilib = reg_images.get_cols_squared_norms(latent)  # c_g^2
        precond_op = LearnableDiagonalOperator((deg_equilib.clamp(0) + wg_equilib.clamp(0)).pow_(-0.5),
                                               function=lambda x: x, learnable=False)
        return precond_op

    @property
    def normal_equations_system(self):
        return WienerNormalEquationsSystem(self.degradation, self.reg_fidelity, self.reg_images, self.weight_noise,
                                           self.use_weight_noise_scaling_coef,
                                           precond_sym_inv=self.preconditioner_right_inv)


class WienerNormalEquationsSystem(WienerFilteringSystemOperator):
    def __init__(self, degradation_operator: LinearDegradationOperatorBase, reg_fidelity: LearnableLinearOperatorBase,
                 reg_images: LearnableLinearOperatorBase,
                 weight_noise: Union[LearnableNumberOperator, IdentityOperator],
                 use_weight_noise_scaling_coef: bool = True, precond_sym_inv=IdentityOperator(),
                 weight_noise_sqrt=None) -> None:
        """
        Initializing linear system.

        :param degradation_operator: (H) linear operator, representing degradation matrix
        :param reg_fidelity: (\Phi) learned linear operators, representing data fidelity regularization
        :param reg_images: (G) learned linear operator, representing regularization operator
        :param weight_noise: (\alpha) operator, representing noise-specific scaler
        :param weight_noise_sqrt: \sqrt(\alpha(\sigma)) learned number, representing square root of
                                  noise-specific scaler. If not available, is calculated from weight_noise
        :param preconditioner_sym_inv: linear operator $P^{-1}$, determining inverse of symmmetric preconditioner:
                                       $P^{-T} C^T C P^{-1} Px = P^{-T} b$
        """
        nn.Module.__init__(self)
        assert isinstance(degradation_operator, LinearOperatorBase)
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_images, LearnableLinearOperatorBase)
        if weight_noise_sqrt is not None:
            assert isinstance(weight_noise_sqrt, (LearnableNumberOperator, IdentityOperator))
            self.weight_noise_sqrt = weight_noise_sqrt
            self.weight_noise = weight_noise_sqrt.quadratic_operator
        else:
            assert isinstance(weight_noise, (LearnableNumberOperator, IdentityOperator))
            self.weight_noise = weight_noise
            with th.no_grad():
                self.weight_noise_sqrt = weight_noise.sqrt_operator
        self.degradation = degradation_operator
        self.reg_fidelity = reg_fidelity
        self.reg_images = reg_images
        self.use_weight_noise_scaling_coef = use_weight_noise_scaling_coef
        self.preconditioner_sym_inv = precond_sym_inv

    def apply(self, vector: th.Tensor, *args, **kwargs) -> MultiVector:
        """
        This method performs batched left hand side linear transformation of input vector.

        :param vector: input vector of shape [B, C, H, W] to be transformed by linear operator
        :return: MultiVector of shape [B, ...], transformed by linear operator
        """
        elem1 = self.weight_noise_sqrt(self.reg_fidelity(self.degradation(vector)))  # \Phi H x
        elem2 = self.reg_images(vector)  # \sqrt(W) G x
        return MultiVector((elem1, elem2))

    def _transpose(self, vector: MultiVector, *args, inplace: bool = True, **kwargs) -> th.Tensor:
        """
        This method applies transposed linear operation on input vector.

        :param vector: input vector of shape [B, ...] to apply transposed operation
        :param args, kwargs: aux parameters
        :return: result of transposed linear operation, applied to input vector
        """
        assert isinstance(vector, MultiVector)
        assert vector.num_elements == 2
        elem1, elem2 = tuple(vector)
        # reusing the same tensor storage
        res = self.reg_images.T(elem2, *args, **kwargs) + \
              self.weight_noise_sqrt.T(self.degradation.T(self.reg_fidelity.T(elem1, *args, **kwargs), *args, **kwargs),
                                       inplace=inplace)
        return res

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs) -> th.Tensor:
        if operator_between is None:
            return super(WienerNormalEquationsSystem, self).apply(vector, *args, **kwargs)
        else:
            return super(WienerNormalEquationsSystem, self).transpose_apply(
                vector, *args, operator_between=operator_between, **kwargs)

    def right_hand_side(self, degraded: th.Tensor, *args) -> MultiVector:
        # TODO: replace with meta-tensor when conv operation is supported
        with th.no_grad():
            mid_shape = self.reg_images(self.degradation.T(degraded))
        return MultiVector((self.weight_noise_sqrt(self.reg_fidelity(degraded)), th.zeros_like(mid_shape)))

    def right_hand_side_full(self, degraded: th.Tensor, *args) -> MultiVector:
        return super(WienerNormalEquationsSystem, self).right_hand_side(degraded, *args)

    def right_hand_side_extension_vector(self, *args) -> type(None):
        return None

    @property
    def T_operator(self) -> LinearOperatorBase:
        return self

    def residual(self, solution: th.Tensor, degraded: th.Tensor, latent: th.Tensor) -> th.Tensor:
        return self.right_hand_side_full(degraded, latent) - \
               self.transpose_apply(solution, operator_between=None, inplace=False)

    def to(self, *args, **kwargs) -> 'WienerNormalEquationsSystem':
        super(WienerNormalEquationsSystem, self).to(*args, **kwargs)
        self.weight_noise_sqrt = self.weight_noise_sqrt.to(*args, **kwargs)
        return self


class SRWienerFilteringSystemOperator(WienerFilteringSystemOperator):
    """
    This class implements linear system corresponding to quadratic problem of Wiener filtering:
    argmin_x \frac{1}{\sigma^2} ||\Phi(H x - y)||_2^2 + ||Gx||_2^2,
    where one-shot inversion is implemented for super-resolution task.
    Inversion is based on the method proposed in
        Zhao, N., Wei, Q., Basarab, A., Dobigeon, N., Kouamé, D., & Tourneret, J. Y. (2016).
        Fast Single Image Super-Resolution Using a New Analytical Solution for $\ell_{2}$–$\ell_{2}$ Problems.
        IEEE Transactions on Image Processing, 25(8), 3683-3697.
        https://hal.archives-ouvertes.fr/hal-01373784/document
    """
    @staticmethod
    def check_operator_circulant(operator: LinearOperatorBase) -> None:
        """
        This method checks if operator represents a circulant matrix, and raises error if it is not the case

        :param operator: operator for which a check is required
        :return: Nothing
        """
        if isinstance(operator, (IdentityOperator, type(None), LearnableNumberOperator)):
            return
        if isinstance(operator, ConvDecimateLinearDegradationOperator):
            assert operator.pad_operator.pad_mode == 'periodic' or operator.kernel is None
        elif isinstance(operator, LearnableConvolutionOperator):
            assert operator.pad_operator.pad_mode == 'periodic'
        else:
            raise AssertionError

    def _check_inv_conditions_and_define_unfold_fn(self, vector: th.Tensor) -> Tuple[Callable, int]:
        """
        This method checks whether the one-shot inversion could be applied and constructs unfolding function for it.

        :param vector: incoming vector which has to be multiplied by inverse matrix
        :return: unfolding function and corresponding scale factor value
        """
        # checking conditions under which the implemented inversion holds and preparing parameters
        self.check_operator_circulant(self.degradation)
        self.check_operator_circulant(self.reg_images)
        assert isinstance(self.reg_fidelity, IdentityOperator)
        scale_factor = self.degradation.scale_factor
        assert vector.shape[-2] % scale_factor == 0
        assert vector.shape[-1] % scale_factor == 0
        nl_h = vector.shape[-2] // scale_factor
        nl_w = vector.shape[-1] // scale_factor

        # operation, which performs splitting of eigenvalues based on scale factor
        def unfold(tensor):
            tensor_folded = tensor.unfold(-2, nl_h, nl_h).unfold(-2, nl_w, nl_w)
            return tensor_folded
        return unfold, scale_factor

    def _get_eigenvalues_for_inv(self, vector: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        This method returns eigenvalues of degradation and regularization operators to perform one-shot inversion.

        :param vector: incoming vector which has to be multiplied by inverse matrix
        :return: eigenvalues of degradation operator and abs squared eigenvalues of regularization operator
        """
        # eigenvalues of regularization and degradation operators
        reg_eigvals_sq = \
            self.reg_images.get_circulant_eigvals_doublesided(vector.shape).abs().pow_(2).sum(dim=1, keepdim=True)
        deg_eigvals = self.degradation.get_circulant_eigvals_doublesided(vector.shape)
        if isinstance(deg_eigvals, float):
            deg_eigvals = th.ones(*((1,) * reg_eigvals_sq.dim()), dtype=vector.dtype, device=vector.device)
        deg_eigvals = deg_eigvals.conj()
        return deg_eigvals, reg_eigvals_sq

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements inversion of wiener filtering linear system matrix for super-resolution problem if
        both downscale and regularization convolutions are circulant.
        Inversion is based on the method proposed in
            Zhao, N., Wei, Q., Basarab, A., Dobigeon, N., Kouamé, D., & Tourneret, J. Y. (2016).
            Fast Single Image Super-Resolution Using a New Analytical Solution for $\ell_{2}$–$\ell_{2}$ Problems.
            IEEE Transactions on Image Processing, 25(8), 3683-3697.
            https://hal.archives-ouvertes.fr/hal-01373784/document

        :param vector: incoming vector to be multiplied by inverse matrix
        :return: vector, multiplied by inverse of matrix represented by operator
        """
        unfold_fn, scale_factor = self._check_inv_conditions_and_define_unfold_fn(vector)
        deg_eigvals, reg_eigvals_sq = self._get_eigenvalues_for_inv(vector)

        vector = fft2(vector).unsqueeze(1)
        vector /= reg_eigvals_sq

        diag = \
            unfold_fn(deg_eigvals.abs().pow_(2).sum(dim=1, keepdim=True) / reg_eigvals_sq).sum(dim=(-3, -4))
        identity_d = self.weight_noise.inv(scale_factor ** 2)
        if isinstance(identity_d, th.Tensor):
            for i in range(diag.dim() - identity_d.dim()):
                identity_d = identity_d.unsqueeze(-1)
        diag += identity_d
        ret = unfold_fn(vector * deg_eigvals).sum(dim=(-3, -4)) / diag

        ret = deg_eigvals.conj() * (ret.repeat(1, 1, 1, scale_factor, scale_factor))
        ret = ret.sum(dim=1)
        ret = vector[:, 0, ...] - ret / reg_eigvals_sq.squeeze(1)
        return ifft2(ret).real


class DemosaicWienerFilteringSystemOperator(SRWienerFilteringSystemOperator):
    """
    This class implements linear system corresponding to quadratic problem of Wiener filtering:
    argmin_x \frac{1}{\sigma^2} ||\Phi(H x - y)||_2^2 + ||Gx||_2^2,
    where one-shot inversion is implemented for demosaicing task.
    Inversion is derived as an extension of a method proposed in
        Zhao, N., Wei, Q., Basarab, A., Dobigeon, N., Kouamé, D., & Tourneret, J. Y. (2016).
        Fast Single Image Super-Resolution Using a New Analytical Solution for $\ell_{2}$–$\ell_{2}$ Problems.
        IEEE Transactions on Image Processing, 25(8), 3683-3697.
        https://hal.archives-ouvertes.fr/hal-01373784/document
    """
    @staticmethod
    def check_operator_circulant(operator: LinearOperatorBase) -> None:
        """
        This method checks if operator represents a circulant matrix, and raises error if it is not the case

        :param operator: operator for which a check is required
        :return: Nothing
        """
        if isinstance(operator, (IdentityOperator, type(None), LearnableNumberOperator)):
            return
        if isinstance(operator, ConvMosaicLinearDegradationOperator):
            assert operator.pad_operator.pad_mode == 'periodic' or operator.kernel is None
            assert operator.pattern in ('rggb', 'bggr', 'gbrg', 'grbg'), \
                'Only rggb, bggr, gbrg and grbg patterns are supported.'
        elif isinstance(operator, LearnableConvolutionOperator):
            assert operator.pad_operator.pad_mode == 'periodic'
        else:
            raise AssertionError

    def get_shift_eigvals(self, pattern: str, vector: th.Tensor) -> th.Tensor:
        """
        This method returns eigenvalues of shifting operators for red, blue and two green elements in Bayer
        pattern given its pixels order. Corresponding shifting operators perform circular shifts for corresponding
        elements in order for them to move to the top-left location, i.e. location of the red color elements.

        :param pattern: pixels order in mosaic pattern
        :param vector: vector of shape [..., 3, H, W] for which the bayer pattern is applied
        :return: tensor of shape [..., 4, H, W], where the first two channels provide eigenvalues for
                 red and blue pixels, and the last two channels provide eigenvalues for two green pixels.
        """
        if pattern == 'rggb':
            return self.get_shift_eigvals_rggb(vector)
        elif pattern == 'bggr':
            return self.get_shift_eigvals_bggr(vector)
        elif pattern == 'gbrg':
            return self.get_shift_eigvals_gbrg(vector)
        elif pattern == 'grbg':
            return self.get_shift_eigvals_grbg(vector)
        else:
            raise ValueError

    @staticmethod
    def get_shift_eigvals_rggb(vector: th.Tensor) -> th.Tensor:
        """
        This method returns eigenvalues of shifting operators for red, blue and two green elements in rggb Bayer
        pattern. Corresponding shifting operators perform circular shifts for corresponding elements in
        order for them to move to the top-left location, i.e. location of the red color elements.

        :param vector: vector of shape [..., 3, H, W] for which the bayer pattern is applied
        :return: tensor of shape [..., 4, H, W], where the first two channels provide eigenvalues for
                 red and blue pixels, and the last two channels provide eigenvalues for two green pixels.
        """
        shape = list(vector.shape)
        shape[-3] = 4
        shift_eigvals = th.zeros(*shape, dtype=vector.dtype, device=vector.device)
        shift_eigvals[..., 0, 0, 0] = 1  # r
        shift_eigvals[..., 1, -1, -1] = 1  # b
        shift_eigvals[..., 2, 0, -1] = 1  # g1
        shift_eigvals[..., 3, -1, 0] = 1  # g2
        return fft2(shift_eigvals)

    @staticmethod
    def get_shift_eigvals_bggr(vector: th.Tensor) -> th.Tensor:
        """
        This method returns eigenvalues of shifting operators for red, blue and two green elements in bggr Bayer
        pattern. Corresponding shifting operators perform circular shifts for corresponding elements in
        order for them to move to the top-left location, i.e. location of the red color elements.

        :param vector: vector of shape [..., 3, H, W] for which the bayer pattern is applied
        :return: tensor of shape [..., 4, H, W], where the first two channels provide eigenvalues for
                 red and blue pixels, and the last two channels provide eigenvalues for two green pixels.
        """
        shape = list(vector.shape)
        shape[-3] = 4
        shift_eigvals = th.zeros(*shape, dtype=vector.dtype, device=vector.device)
        shift_eigvals[..., 0, -1, -1] = 1  # r
        shift_eigvals[..., 1, 0, 0] = 1  # b
        shift_eigvals[..., 2, 0, -1] = 1  # g1
        shift_eigvals[..., 3, -1, 0] = 1  # g2
        return fft2(shift_eigvals)

    @staticmethod
    def get_shift_eigvals_gbrg(vector: th.Tensor) -> th.Tensor:
        """
        This method returns eigenvalues of shifting operators for red, blue and two green elements in gbrg Bayer
        pattern. Corresponding shifting operators perform circular shifts for corresponding elements in
        order for them to move to the top-left location, i.e. location of the red color elements.

        :param vector: vector of shape [..., 3, H, W] for which the bayer pattern is applied
        :return: tensor of shape [..., 4, H, W], where the first two channels provide eigenvalues for
                 red and blue pixels, and the last two channels provide eigenvalues for two green pixels.
        """
        shape = list(vector.shape)
        shape[-3] = 4
        shift_eigvals = th.zeros(*shape, dtype=vector.dtype, device=vector.device)

        shift_eigvals[..., 0, -1, 0] = 1  # r
        shift_eigvals[..., 1, 0, -1] = 1  # b
        shift_eigvals[..., 2, 0, 0] = 1  # g1
        shift_eigvals[..., 3, -1, -1] = 1  # g2
        return fft2(shift_eigvals)

    @staticmethod
    def get_shift_eigvals_grbg(vector: th.Tensor) -> th.Tensor:
        """
        This method returns eigenvalues of shifting operators for red, blue and two green elements in grbg Bayer
        pattern. Corresponding shifting operators perform circular shifts for corresponding elements in
        order for them to move to the top-left location, i.e. location of the red color elements.

        :param vector: vector of shape [..., 3, H, W] for which the bayer pattern is applied
        :return: tensor of shape [..., 4, H, W], where the first two channels provide eigenvalues for
                 red and blue pixels, and the last two channels provide eigenvalues for two green pixels.
        """
        shape = list(vector.shape)
        shape[-3] = 4
        shift_eigvals = th.zeros(*shape, dtype=vector.dtype, device=vector.device)

        shift_eigvals[..., 0, 0, -1] = 1  # r
        shift_eigvals[..., 1, -1, 0] = 1  # b
        shift_eigvals[..., 2, 0, 0] = 1  # g1
        shift_eigvals[..., 3, -1, -1] = 1  # g2
        return fft2(shift_eigvals)

    @staticmethod
    def _inv_hermitian_bwdb_2(upper_left: th.Tensor, upper_right: th.Tensor, bottom_right: th.Tensor
                              ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        This method performs a closed-form inversion of block 2x2 matrix, where all blocks are diagonal matrices.
        Inversion is performed under assumption (is not checked explicitly), that the matrix is Hermitian.
        Elements of the matrix are organised as follows:
        | upper_left        upper_right  |
        | upper_right^*     bottom_right |
        Method is expecting three inputs, representing upper triangular part of the matrix.

        :param upper_left: tensor representing diagonal of upper-left block of block 2x2 matrix
        :param upper_right: tensor representing diagonal of upper-right block of block 2x2 matrix
        :param bottom_right: tensor representing diagonal of bottom-right block of block 2x2 matrix
        :return:
        """
        denom = (upper_left*bottom_right - upper_right.abs().pow_(2))
        ret_a = bottom_right/denom
        ret_c = upper_left/denom
        ret_b = -upper_right/denom
        return ret_a, ret_b, ret_c

    def inv_term_rb(self, vector: th.Tensor, deg_eigvals: th.Tensor, reg_eigvals_sq: th.Tensor, scale_factor: int,
                    unfold_fn: Callable, identity_d: Union[th.Tensor, int, float]) -> th.Tensor:
        """
        This method performs color related processing for system matrix inversion for red and blue channels.

        :param vector: stacked red and blue parts of incoming vector which has to be multiplied by inverse matrix
        :param deg_eigvals: stacked red and blue parts of eigenvalues for fused shift and convolution operators
        :param reg_eigvals_sq: red and blue parts of abs squared eigenvalues of regularization operator
        :param scale_factor: scale factor related to the bayer pattern (2 for rggb case)
        :param unfold_fn: unfolding function corresponding to the scale factor (2 for rggb case) and vector size
        :param identity_d: noise-related term required for inversion
        :return: red and blue channels result of color-related processing in system matrix inversion
        """
        diag = \
            unfold_fn(deg_eigvals.abs().pow_(2).sum(dim=1, keepdim=True) / reg_eigvals_sq).sum(dim=(-3, -4))
        if isinstance(identity_d, th.Tensor):
            for i in range(diag.dim() - identity_d.dim()):
                identity_d = identity_d.unsqueeze(-1)

        diag += identity_d
        ret = unfold_fn(vector * deg_eigvals).sum(dim=(-3, -4)) / diag

        th.conj(deg_eigvals, out=deg_eigvals)

        ret = deg_eigvals * (ret.repeat(1, 1, 1, scale_factor, scale_factor))
        ret = ret.sum(dim=1)
        return ret

    def inv_term_g(self, vector: th.Tensor, deg_eigvals: th.Tensor, reg_eigvals_sq: th.Tensor, scale_factor: int,
                    unfold_fn: Callable, identity_d: Union[th.Tensor, int, float]) -> th.Tensor:
        """
        This method performs color related processing for system matrix inversion for green channel.

        :param vector: green part of incoming vector which has to be multiplied by inverse matrix
        :param deg_eigvals: stacked green parts of eigenvalues for fused shift and convolution operators
        :param reg_eigvals_sq: green part of abs squared eigenvalues of regularization operator
        :param scale_factor: scale factor related to the bayer pattern (2 for rggb case)
        :param unfold_fn: unfolding function corresponding to the scale factor (2 for rggb case) and vector size
        :param identity_d: noise-related term required for inversion
        :return: green channel result of color-related processing in system matrix inversion
        """
        g1 = deg_eigvals[..., 0, None, :, :]
        g2 = deg_eigvals[..., 1, None, :, :]
        a = unfold_fn(g1.abs().pow_(2).sum(dim=1, keepdim=True) / reg_eigvals_sq).sum(dim=(-3, -4))
        b = unfold_fn((g1 * g2.conj()).sum(dim=1, keepdim=True) / reg_eigvals_sq).sum(dim=(-3, -4))
        c = unfold_fn(g2.abs().pow_(2).sum(dim=1, keepdim=True) / reg_eigvals_sq).sum(dim=(-3, -4))
        if isinstance(identity_d, th.Tensor):
            for i in range(a.dim() - identity_d.dim()):
                identity_d = identity_d.unsqueeze(-1)

        a += identity_d
        c += identity_d
        a, b, c = self._inv_hermitian_bwdb_2(a, b, c)

        ret = unfold_fn(vector * deg_eigvals).sum(dim=(-3, -4))
        ret_g1 = ret[..., :1, :, :]
        ret_g2 = ret[..., 1:, :, :]
        ret = th.cat([a * ret_g1 + b * ret_g2, b.conj() * ret_g1 + c * ret_g2], dim=-3)
        ret_g1 = ret[..., :1, :, :]
        ret_g2 = ret[..., 1:, :, :]

        ret_g1 = g1.conj() * (ret_g1.repeat(1, 1, 1, scale_factor, scale_factor))
        ret_g1 = ret_g1.sum(dim=1)

        ret_g2 = g2.conj() * (ret_g2.repeat(1, 1, 1, scale_factor, scale_factor))
        ret_g2 = ret_g2.sum(dim=1)
        ret = ret_g1 + ret_g2
        return ret

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements inversion of wiener filtering linear system matrix for demosaic problem if
        both downscale and regularization convolutions are circulant.
        Inversion is derived as an extension of a method proposed in
            Zhao, N., Wei, Q., Basarab, A., Dobigeon, N., Kouamé, D., & Tourneret, J. Y. (2016).
            Fast Single Image Super-Resolution Using a New Analytical Solution for $\ell_{2}$–$\ell_{2}$ Problems.
            IEEE Transactions on Image Processing, 25(8), 3683-3697.
            https://hal.archives-ouvertes.fr/hal-01373784/document

        :param vector: incoming vector to be multiplied by inverse matrix
        :return: vector, multiplied by inverse of matrix represented by operator
        """
        unfold_fn, scale_factor = self._check_inv_conditions_and_define_unfold_fn(vector)
        deg_eigvals, reg_eigvals_sq = self._get_eigenvalues_for_inv(vector)

        # eigenvalues of shifting operators which represent the cyclic shift based on pixel locations
        shifted_deg_eigvals = self.get_shift_eigvals(self.degradation.pattern, vector).unsqueeze(1)
        shifted_deg_eigvals_rb = shifted_deg_eigvals[..., :2, :, :]
        shifted_deg_eigvals_gg = shifted_deg_eigvals[..., -2:, :, :]

        # merging shifting eigenvalues with possibly channel-dependent degradation conv eigenvalues
        if deg_eigvals.shape[-3] == 1:
            shifted_deg_eigvals *= deg_eigvals
        elif deg_eigvals.shape[-3] == 3:
            shifted_deg_eigvals_rb[..., 0, :, :] *= deg_eigvals[..., 0, :, :]
            shifted_deg_eigvals_rb[..., 1, :, :] *= deg_eigvals[..., -1, :, :]
            shifted_deg_eigvals_gg *= deg_eigvals[..., 1, None, :, :]
        else:
            raise ValueError

        # splitting and regrouping possibly channel-dependent regularization operator eigenvalues
        if reg_eigvals_sq.shape[-3] == 1:
            reg_eigvals_sq_rb = reg_eigvals_sq
            reg_eigvals_sq_g = reg_eigvals_sq
        elif reg_eigvals_sq.shape[-3] == 3:
            reg_eigvals_sq_rb = th.stack([reg_eigvals_sq[..., 0, :, :], reg_eigvals_sq[..., -1, :, :]], dim=-3)
            reg_eigvals_sq_g = reg_eigvals_sq[..., 1, None, :, :]
        else:
            raise ValueError

        # initial channel-independent processing
        vector = fft2(vector).unsqueeze(1)
        vector /= reg_eigvals_sq
        identity_d = self.weight_noise.inv(scale_factor ** 2)

        # regrouping channels part of vector to process independently
        vector_rb = th.stack([vector[..., 0, :, :], vector[..., -1, :, :]], dim=-3)
        vector_g = vector[..., 1, None, :, :]

        # processing inversion for red+blue channels independently of green
        ret_rb = \
            self.inv_term_rb(vector_rb, shifted_deg_eigvals_rb, reg_eigvals_sq_rb, scale_factor, unfold_fn, identity_d)
        ret_g = self.inv_term_g(vector_g, shifted_deg_eigvals_gg, reg_eigvals_sq_g, scale_factor, unfold_fn, identity_d)

        # merging separately processed channels and do final channel-independent processing
        ret = th.cat([ret_rb[..., :1, :, :], ret_g, ret_rb[..., -1:, :, :]], dim=1)
        ret = vector[:, 0, ...] - ret / reg_eigvals_sq.squeeze(1)
        return ifft2(ret).real
