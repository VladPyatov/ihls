import abc
import warnings
from abc import ABC
from typing import Tuple, List, Callable, Union, Optional

import torch as th
import torch.nn as nn


class LinearOperatorBase:
    """
    This is a base class, which implements parametric linear operator. In general, it can depend on several parameters,
    stored in different tensors.
    """
    has_sqrt_implemented = False

    @abc.abstractmethod
    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        pass

    def set_parameters(self, *parameters, **kwparameters) -> None:
        """
        This method updates parameters (class attributes) needed to compute operator output and initializes
        corresponding latent ones, if they are not presented in inputs.

        :param parameters, kwparameters: input parameters, which should replace the operator's existing ones
        :return: Nothing
        """
        pass

    def init_parameters(self, *args, **kwargs) -> None:
        """
        This method initializes all latent parameters needed to compute operator output.

        :return: Nothing
        """
        pass

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method applies transposed linear operation on input vector, using the following differential property
        (denominator layout): y = Ax => J = (dy/dx)^T = A, so x^T J = J^T x = A^T x.

        :param vector: input vector of shape [B, ...] to apply transposed operation
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of transposed linear operation, applied to input vector
        """
        if vector.requires_grad:
            create_graph = True
        else:
            create_graph = False
        grad = th.autograd.functional.vjp(self.apply, vector, create_graph=create_graph)[1]
        return grad

    def T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method overrides transposed by calling ._transpose, so operator could be used in a convenient way.

        :param vector: input vector of shape [B, ...] to apply transpose operation
        :return: result of transposed linear operation, applied to input vector
        """
        return self._transpose(vector, *args, **kwargs)

    @property
    def T_operator(self) -> 'LinearOperatorBase':
        """
        This property returns operator, corresponding to transpose of the current one.

        :return: linear operator, transpose to the current one
        """
        class TransposeLinearOperator(self.__class__):
            def __init__(self_t):
                self_t.parent = self

            @staticmethod
            def get_transpose_if_exists(operator):
                if isinstance(operator, LinearOperatorBase):
                    return operator.T_operator
                else:
                    return operator

            @property
            def preconditioner_right_inv(self_t) -> Callable:
                return self_t.get_transpose_if_exists(self_t.parent.preconditioner_left_inv)

            @property
            def preconditioner_left_inv(self_t) -> Callable:
                return self_t.get_transpose_if_exists(self_t.parent.preconditioner_right_inv)

            def __getattr__(self_t, item: str):
                if item == 'parent':
                    return self
                return getattr(self_t.parent, item)

            def __setattr__(self_t, key, value):
                object.__setattr__(self_t, key, value)

            def apply(self_t, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_t.parent.T(vector, *args, **kwargs)

            def _transpose(self_t, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_t.parent.apply(vector, *args, **kwargs)

            def T_operator(self_t):
                return self_t.parent

            def inv(self_t, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_t.parent.inv_T(vector, *args, **kwargs)

            def inv_T(self_t, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_t.parent.inv(vector, *args, **kwargs)
        raise NotImplementedError

    @property
    def quadratic_operator(self) -> 'LinearOperatorBase':
        """
        This property returns operator, corresponding to square of the current one.
        By square the quadratic form is assumed: B = A^T A.

        :return: linear operator, square to the current one
        """
        class QuadraticLinearOperator(self.__class__):
            def __init__(self_q):
                self_q.parent = self

            @staticmethod
            def get_quadratic_if_exists(operator):
                if isinstance(operator, IdentityOperator) or operator is None or \
                        not isinstance(operator, LinearOperatorBase):
                    return operator
                else:
                    raise NotImplementedError

            @property
            def preconditioner_right_inv(self_q) -> Callable:
                return self_q.get_quadratic_if_exists(self_q.parent.preconditioner_left_inv)

            @property
            def preconditioner_left_inv(self_q) -> Callable:
                return self_q.get_quadratic_if_exists(self_q.parent.preconditioner_right_inv)

            def __getattr__(self_q, item: str):
                if item == 'parent':
                    return self
                return getattr(self_q.parent, item)

            def __setattr__(self_q, key, value):
                object.__setattr__(self_q, key, value)

            def apply(self_q, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_q.parent.transpose_apply(vector, *args, operator_between=None, **kwargs)

            def _transpose(self_q, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_q.apply(vector, *args, **kwargs)

            def T_operator(self_q):
                return self_q

            def inv(self_q, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_q.parent.inv(self_q.parent.inv_T(vector, *args, **kwargs), *args, **kwargs)

            def inv_T(self_q, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
                return self_q.inv(vector, *args, **kwargs)

            def sqrt(self_q, vector: th.Tensor) -> th.Tensor:
                return self_q.parent(vector)

            def sqrt_operator(self_q) -> 'LinearOperatorBase':
                return self_q.parent
        raise NotImplementedError

    def __call__(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method overrides call by calling .apply, so operator could be used in a convenient way.

        :param vector: input vector to be transformed by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector, transformed by linear operator
        """
        return self.apply(vector, *args, **kwargs)

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs) -> th.Tensor:
        """
        This method performs linear transformation followed by transpose transformation with the same operator.

        :param vector: input vector of shape [B, ...] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector, transformed by linear operator and its transpose
        """
        output_shape = vector.shape
        if operator_between is not None:
            ret = self._transpose(operator_between(self.apply(vector, *args, **kwargs)), *args,
                                  output_shape=output_shape, **kwargs)
        else:
            ret = self._transpose(self.apply(vector, *args, **kwargs), *args, output_shape=output_shape, **kwargs)
        return ret

    def sqrt(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of square root of operator.

        :param vector: input vector to apply operator's square root on
        :return: result of square root application to vector
        """
        raise NotImplementedError('Square root is not implemented for this operator.')

    @property
    def sqrt_operator(self) -> 'LinearOperatorBase':
        """
        This property returns operator B, corresponding to square root of initial operator A = B^T B.

        :return: linear operator, which transpose_apply corresponds to the current one
        """
        raise NotImplementedError('Square root is not implemented for this operator.')

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of an inverse of operator.

        :param vector: input vector to apply operator's inverse on
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of operator's inverse application to vector
        """
        raise AttributeError('Inverse operation is not implemented for this operator.')

    def inv_T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of inverse transpose of operator.

        :param vector: input vector to apply operator's inverse on
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of operator's inverse application to vector
        """
        raise AttributeError('Inverse transpose operation is not implemented for this operator.')


class LinearDegradationOperatorBase(LinearOperatorBase):
    @abc.abstractmethod
    def init_with_parameters(self, *args, **kwargs) -> 'LinearDegradationOperatorBase':
        """
        This method sets all parameters of degradation and returns corresponding degradation operator

        :param args: everything that might be needed to parametrize degradation operator
        :param kwargs: everything that might be needed to parametrize degradation operator
        :return: prepared degradation operator
        """
        pass

    @abc.abstractmethod
    def init_with_parameters_(self, *args, **kwargs) -> 'LinearDegradationOperatorBase':
        """
        This method sets all parameters of degradation inplace

        :param args: everything that might be needed to parametrize degradation operator
        :param kwargs: everything that might be needed to parametrize degradation operator
        :return: self
        """
        pass


class JacobianOperatorBase(LinearDegradationOperatorBase):
    @abc.abstractmethod
    def apply_nonlinear(self, input_vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs nonlinear transformation of input vector using degradation model corresponding to Jacobian.

        :param input_vector: vector to be transformed by nonlinear degradation
        :param args, kwargs: altering parameters to parametrize operator
        :return: degraded output, produced by nonlinear degradation model
        """
        pass

    def init_with_parameters(self, *args, **kwargs) -> 'LinearDegradationOperatorBase':
        """
        Typically Jacobian is parametrized by latent variables, so return self

        :param args: everything that might be needed to parametrize degradation operator
        :param kwargs: everything that might be needed to parametrize degradation operator
        :return:
        """
        return self

    def project(self, inputs: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        """
        Project current estimates to some constraint space.

        :param inputs: parameters under restoration to project
        :return: projected parameters
        """
        pass


class LearnableLinearOperatorBase(LinearOperatorBase, nn.Module, ABC):
    """
    This is a base class, which implements learned parametric linear operator, i.e. any linear operator,
    which has learnable parameters. In general, it can depend on several parameters, stored in different tensors.
    """
    learnable: bool = False

    @th.enable_grad()
    def _transpose(self, vector: th.Tensor, output_shape: Union[Tuple[int], List[int]], *args, **kwargs) -> th.Tensor:
        """
        This method implements quazi-transpose forward pass.
        The implementation uses gradient trick, in which the transpose is calculated as a vector jacobian product.

        :param vector: input vector of shape [B, Q, H, W] to be transformed
        :param output_shape: shape of the output
        :return: transformed vector of shape [B, C, H, W]
        """
        assert len(output_shape) >= 3
        should_create_graph = vector.requires_grad
        for param in self.tensors_for_grad:
            if param.requires_grad:
                should_create_graph = True
                break
        tmp_vec = th.rand(*vector.shape[:-3], *tuple(output_shape[-3:]), device=vector.device, dtype=vector.dtype,
                          requires_grad=True)
        ret = th.autograd.grad(self.apply(tmp_vec, *args, **kwargs), tmp_vec, vector,
                               only_inputs=True, create_graph=should_create_graph)[0]
        return ret

    def transpose_apply_gradient_trick(self, vector: th.Tensor, *args,
                                       operator_between: Optional[Union[LinearOperatorBase, Callable]] = None,
                                       norm_dims: Optional[Tuple[int, ...]] = (-1, -2, -3),
                                       **kwargs) -> th.Tensor:
        """
        This method performs linear transformation followed by transpose transformation with the same operator.
        For CNN, this operator can be evaluated efficiently using L2 norm gradient trick if operator_between has .sqrt
        method. If not, sequential calculation is used.

        :param vector: input vector of shape [B, C, H, W] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param norm_dims: tuple of summation indexes for grad norm computation
        :return: vector of shape [B, C, H, W], transformed by linear operator and its transpose
        """
        assert vector.ndim == 4
        if operator_between is None or (hasattr(operator_between, 'sqrt') and operator_between.has_sqrt_implemented):
            should_create_graph = vector.requires_grad
            if not should_create_graph:
                for p in self.tensors_for_grad:
                    if p.requires_grad:
                        should_create_graph = True
                        break
            with th.enable_grad():
                vector_requires_grad_flag = vector.requires_grad
                ret = self.apply(vector.requires_grad_(True), *args, **kwargs)
                if operator_between is not None:
                    ret = operator_between.sqrt(ret)
                    if not should_create_graph and hasattr(operator_between, 'tensors_for_grad'):
                        for p in operator_between.tensors_for_grad:
                            if p.requires_grad:
                                should_create_graph = True
                                break
            ret = 0.5 * self.l2_norm_squared_grad(ret, vector, should_create_graph, norm_dims)
        else:
            ret = self._transpose(operator_between(self.apply(vector, *args, **kwargs)), vector.shape[-3:])
        vector.requires_grad_(vector_requires_grad_flag)
        return ret

    @staticmethod
    @th.enable_grad()
    def l2_norm_squared_grad(tensor_for_norm: th.Tensor, argument: th.Tensor, create_graph: bool,
                             norm_dims: Tuple[int, ...] = (-1, -2, -3)) -> th.Tensor:
        """
        This method calculates gradient of L2 squared norm of input batch w.r.t. some argument using autograd mechanics.

        :param tensor_for_norm: tensor for which the gradient of L2 squared norm is computed
        :param argument: tensor w.r.t. which the gradient is computed
        :param create_graph: flag which specifies whether the graph of the derivative will be created
        :param norm_dims: dimensions to reduce in norm calculation
        :return: gradient of L2 squared norm of tensor_for_norm w.r.t. argument
        """
        norm = tensor_for_norm.pow(2).sum(dim=norm_dims)
        assert norm.ndim == 1
        ret = th.autograd.grad(norm, argument, th.ones_like(norm), only_inputs=True, create_graph=create_graph)[0]
        return ret

    @property
    def parameters_names_list(self) -> List[str]:
        """
        This method returns parameters names list, which should be learned.

        :return: list or tuple with parameters names to be learned
        """
        names = []
        for n, _ in self.named_parameters():
            names.append(n)
        return names

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor, ...]:
        """
        This method returns tuple with tensors (parameters), which should be passed to handcrafted autograd function.
        Implementation for the most common case, when parameters are passed as is.

        :return: tuple with parameters to be passed to handcrafted autograd function
        """
        out_list = []
        for name in self.parameters_names_list:
            out_list.append(getattr(self, name))
        return tuple(out_list)

    def cast_parameters_to_nn_param(self) -> None:
        """
        Auxiliary method, which casts operator parameters to nn.Parameter in order to learn them.
        :return: Nothing
        """
        for param in self.parameters_names_list:
            setattr(self, param, nn.Parameter(getattr(self, param)))
        self.learnable = True

    def cast_parameters_to_requires_grad(self) -> None:
        """
        Auxiliary method, which casts operator parameters to tensors which require grad.
        :return: Nothing
        """
        for param in self.parameters_names_list:
            setattr(getattr(self, param), 'requires_grad', True)

    def cast_parameters_to_tensor(self) -> None:
        """
        Auxiliary method, which casts operator parameters to tensors without gradients in order not to learn them.
        :return: Nothing
        """
        for param in self.parameters_names_list:
            assert isinstance(param, nn.Parameter)
            setattr(self, param, getattr(self, param).data)
        self.learnable = False

    def to(self, *args, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        if self.learnable:
            return super(LearnableLinearOperatorBase, self).to(*args, **kwargs)
        else:
            return self._nonlearnable_to(*args, **kwargs)

    def _nonlearnable_to(self, *args, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        Moves and/or casts the parameters.
        :return: self
        """
        for param in self.parameters_names_list:
            setattr(self, param, getattr(self, param).to(*args, **kwargs))
        return self

    def prepare_for_step(self, step_idx: int, *tensor_params: th.Tensor, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        This method prepares operator and its parameters for the next optimization step.

        :param step_idx: index of current step (step number)
        :param tensor_params: parameters, which operator can be dependent to
        :param kwargs: other parameters, which are needed for step-specific parametrization
        :return: operator prepared for next optimization step
        """
        return self

    def prepare_for_restoration(self, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        This method prepares operator for recurrent restoration based on some parameters

        :param kwargs: other parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        return self

    def __setattr__(self, key, value):
        try:
            super(LearnableLinearOperatorBase, self).__setattr__(key, value)
        except TypeError:
            warnings.warn(f'Assigning {value} to attribute {key} against nn.Module lodic.')
            self.__delattr__(key)
            super(LearnableLinearOperatorBase, self).__setattr__(key, value)
    

class IdentityOperator(LearnableLinearOperatorBase, LinearDegradationOperatorBase):
    has_sqrt_implemented = True
    """
    This class implements identity operator, i.e. operator which does absolutely nothing and thus can be used as a dummy
    to avoid unnecessary checks.
    """
    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements forward pass through an identity operator.

        :param vector: input vector of shape [B, C, H, W] to be transformed
        :param args, kwargs: other params, which may be passed to the operator
        :return: transformed vector the same as input
        """
        return vector

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method implements transpose pass through an identity operator.

        :param vector: input vector of shape [B, C, H, W] to be transformed
        :param args, kwargs: other params, which may be passed to the operator
        :return: transformed vector the same as input
        """
        return vector

    @property
    def parameters_names_list(self) -> List[str]:
        """
        This method returns empty list, since identity is independent of parameters.

        :return: list or tuple with parameters names to be learned
        """
        return []

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs):
        """
        This method performs identity transformation followed by transpose transformation with the same identity.

        :param vector: input vector of shape [B, ...] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector, transformed by linear operator and its transpose
        """
        if operator_between is not None:
            return operator_between(vector)
        else:
            return vector

    def sqrt(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of square root of identity, which is identity itself.

        :param vector: input vector to apply operator's square root on
        :return: result of square root application to vector
        """
        return vector

    @property
    def T_operator(self) -> 'LinearOperatorBase':
        """
        This property returns operator, corresponding to transpose of the current one, which is identity.

        :return: linear operator, transpose to the current one
        """
        return self

    @property
    def sqrt_operator(self):
        """
        This property returns identity operator, corresponding to square root of initial identity.

        :return: identity linear operator
        """
        return IdentityOperator()

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of an inverse of operator. For identity operator inverse is identity itself.

        :param vector: input vector to apply operator's inverse on
        :return: result of operator's inverse application to vector
        """
        return vector

    def inv_T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of inverse transpose of operator.
        For identity operator inverse transpose is identity itself.

        :param vector: input vector to apply operator's inverse on
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of operator's inverse application to vector
        """
        return vector

    def init_with_parameters(self, *args, **kwargs) -> 'IdentityOperator':
        """
        This method sets all parameters of degradation and returns corresponding degradation operator

        :param args: everything that might be needed to parametrize degradation operator
        :param kwargs: everything that might be needed to parametrize degradation operator
        :return: prepared degradation operator
        """
        return self

    def init_with_parameters_(self, *args, **kwargs) -> 'IdentityOperator':
        """
        This method sets all parameters of degradation inplace

        :param args: everything that might be needed to parametrize degradation operator
        :param kwargs: everything that might be needed to parametrize degradation operator
        :return: self
        """
        return self

    def get_circulant_eigvals(self, shape_of_incoming_vector: Tuple[int]) -> th.Tensor:
        """
        This method returns eigenvalues of identity operator, which is just a vector of ones.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :return: eigenvalues of operator considering periodic boundary conditions for conv
        """
        return 1.

    def get_circulant_eigvals_doublesided(self, shape_of_incoming_vector: Tuple[int]) -> th.Tensor:
        """
        This method returns eigenvalues of identity operator, which is just a vector of ones.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :return: eigenvalues of operator considering periodic boundary conditions for conv
        """
        return 1.

    def get_circulant_abs_squared_eigvals(self, shape_of_incoming_vector: Tuple[int],
                                          mul_between: Union[float, th.Tensor] = 1.) -> th.Tensor:
        """
        This method returns squared absolute eigenvalues. This can be used as eigenvalues of circulant approximation to
        transpose_apply operator.
        More precisely processing is going in the following way:
        \sum_{i=1}^{C_{out}} |otf|_i^2 * mul_between_i, i.e. otf^T mul_between otf, where otf - eigenvalues of
        circulant convolution with the same kernel, as stored in operator.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :param mul_between: multiplier between transpose conv and conv operators which scales responses of each
                            output channel. May be either single float or tensor of shape [B, C_out, 1, 1]
        :return: squared absolute eigenvalues of operator considering periodic boundary conditions for conv
        """
        return mul_between


class LearnableLinearSystemOperatorBase(LearnableLinearOperatorBase):
    """
    This class implements learned preconditioned linear system of the form P_l{^-1} A P_r{^-1} P_r x = P_l{^-1}b, where:
    - left preconditioner P_l^{-1} is given by self.preconditioner_left_inv linear operator;
    - right preconditioner P_r^{-1} is given by self.preconditioner_right_inv linear operator;
    - matrix call A v with given vector v is implemented by self.apply method;
    - right hand side b is implemented by self.right_hand_side method.
    By default identity matrix is used for both left and right preconditioners (no preconditioners at all).
    """
    def __init__(self):
        super(LearnableLinearSystemOperatorBase, self).__init__()
        self.preconditioner_right_inv: LinearOperatorBase = IdentityOperator()
        self.preconditioner_left_inv: LinearOperatorBase = IdentityOperator()

    @abc.abstractmethod
    def right_hand_side(self, *args, **kwargs) -> th.Tensor:
        """
        This method implements right hand side of linear system, given parametrization arguments

        :param args, kwargs: altering parameters to parametrize operator
        :return: right hand side of linear system
        """
        pass

    @abc.abstractmethod
    def perform_step(self, previous_point: Tuple[th.Tensor], solution: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        """
        This method makes a step for latent variables under restoration.

        :param previous_point: latent variables under restoration to update
        :param solution: obtained solution at current, which is raw output from linear system solver
        :return: updated parameters
        """
        pass

    @property
    @abc.abstractmethod
    def tensors_for_grad(self) -> Tuple[th.Tensor, ...]:
        """
        This method should return all tensors, which may require gradients computation
        :return: tuple with tensors, that may require gradients computation
        """
        pass

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method applies transposed linear operation on input vector.
        This implementation is correct only if operator is symmetric, so fell free to override if it is not the case.

        :param vector: input vector of shape [B, ...] to apply transposed operation
        :return: result of transposed linear operation, applied to input vector
        """
        return self.apply(vector, *args, **kwargs)

    def _degraded_self(self, *args) -> th.Tensor:
        """
        This method provides nonlinear degradation of input vector using jacobian.

        :param args: required inputs and params to calculate degraded output
        :return: degraded output, produced assuming degradation model stored in jacobian
        """
        return self.jacobian.apply_nonlinear(*args)

    def residual(self, solution: th.Tensor, degraded: th.Tensor, *latent: th.Tensor) -> th.Tensor:
        """
        This method returns residual r = Ax - b of linear system Ax = b based on its solution.
        This residual is used in backpropagation through implicit function theorem.

        :param solution: solution point, in which the residual (condition) should be computed
        :param degraded: batch of observations required for restoration
        :param latent: latent variable (possibly several parts), to parametrise linear system
        :return: tuple with tensors, representing residual
        """
        r = self.right_hand_side(degraded, *latent) - self.apply(solution, degraded, *latent)
        return r
