from typing import Union, List, Tuple, Callable, Optional

import torch as th

from .base import LearnableLinearOperatorBase


class LearnableDiagonalOperator(LearnableLinearOperatorBase):
    """
    This class implements operator in a form of diagonal matrix
    """
    diagonal_vector: th.Tensor
    function: Callable
    has_sqrt_implemented = True

    def __init__(self, diagonal_vector: th.Tensor = None, function: Callable = th.exp,
                 diagonal_vector_shape: Union[List[int], Tuple[int], th.Size] = None, learnable: bool = True) -> None:
        """
        Initializing operator parameters

        :param diagonal_vector: vector, which is placed on diagonal of operator matrix
        :param function: function to rescale diagonal elements
        :param diagonal_vector_shape: shape of diagonal vector to be initialized; is used, if diagonal_vector is None
        :param learnable: flag, which determines whether to make filters learnable
        """
        super(LearnableDiagonalOperator, self).__init__()
        self.set_parameters(diagonal_vector, function, diagonal_vector_shape)
        if learnable:
            self.cast_parameters_to_nn_param()

    def apply(self, vector: th.Tensor, *args, n: int = 1, inplace: bool = False, **kwargs) -> th.Tensor:
        """
        This method performs batched point-wise multiplication of input vector with operator's diagonal vector.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param n: how many times to apply operator
        :param inplace: if true, operation is performed inplace
        :return: vector of shape [B, ...], transformed by linear operator
        """
        assert vector.ndim == self.diagonal_vector.ndim, f'Expected input vector and diagonal vector to have the ' \
                                                         f'same number of dimensions, but diagonal vector has ' \
                                                         f'shape {self.diagonal_vector.shape}, and received input ' \
                                                         f'vector has shape {vector.shape}.'
        assert isinstance(n, int)
        shape = vector.shape
        diag_vec = self.rescaled_diagonal_vector
        if pow != 1:
            if inplace:
                diag_vec = th.pow(diag_vec, n, out=diag_vec)
            else:
                diag_vec = th.pow(diag_vec, n)
        if inplace:
            vector *= diag_vec
        else:
            vector = vector * diag_vec
        return vector.reshape(shape)

    def _transpose(self, vector: th.Tensor, *args, n: int = 1, inplace: bool = False, **kwargs) -> th.Tensor:
        """
        This method performs transpose batched point-wise multiplication of input vector with operator's diagonal vector
        For diagonal operator transpose operation is exactly the same, as operation itself.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param n: how many times to apply operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        return self.apply(vector, *args, n=n, inplace=inplace, **kwargs)

    @property
    def rescaled_diagonal_vector(self) -> th.Tensor:
        """
        This method returns diagonal vector, scaled by function

        :return: rescaled diagonal vector
        """
        return self.function(self.diagonal_vector)

    @property
    def parameters_names_list(self) -> List[str]:
        return ['diagonal_vector']

    def set_parameters(self, diagonal_vector: th.Tensor, function: Callable,
                       diagonal_vector_shape: Union[List[int], Tuple[int], th.Size] = None) -> None:
        """
        Initializing operator parameters

        :param diagonal_vector: vector, which is placed on diagonal of operator matrix
        :param function: function to rescale diagonal elements
        :param diagonal_vector_shape: shape of diagonal vector to be initialized; is used, if diagonal_vector is None
        """
        if diagonal_vector is None:
            assert diagonal_vector_shape is not None, 'Either diagonal_vector or diagonal_vector_shape should not be ' \
                                                      'None. Given both as None.'
            self.init_parameters(diagonal_vector_shape)
        else:
            assert isinstance(diagonal_vector, th.Tensor), f'Input vector should be torch.Tensor, given ' \
                                                           f'{diagonal_vector.__class__}.'
            self.diagonal_vector = diagonal_vector
        self.function = function

    def init_parameters(self, diagonal_vector_shape: Union[List[int], Tuple[int], th.Size]) -> None:
        """
        This method initializes and sets all parameters needed to compute operator output.

        :param diagonal_vector_shape: shape of diagonal vector to be initialized
        :return: Nothing
        """
        self.diagonal_vector = th.zeros(diagonal_vector_shape)

    def sqrt(self, vector: th.Tensor) -> th.Tensor:
        """
        This method performs point-wise multiplication of input vector with square root of operator's diagonal vector.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        rescaled_diag_vector = self.rescaled_diagonal_vector
        return (rescaled_diag_vector.pow(0.5)*vector).reshape_as(vector)

    @property
    def sqrt_operator(self) -> 'LearnableDiagonalOperator':
        """
        This property returns diagonal operator B, corresponding to square root of current diagonal operator
        A = B*B = B^T B.

        :return: linear operator, representing square root of the current one
        """
        return self.__class__(diagonal_vector=self.rescaled_diagonal_vector.pow(0.5), function=lambda x: x,
                              learnable=False)

    @property
    def quadratic_operator(self) -> 'LearnableDiagonalOperator':
        """
        This property returns diagonal operator B, corresponding to square of current diagonal operator
        A = = B^T B.

        :return: linear operator, representing square of the current one
        """
        return self.__class__(diagonal_vector=self.rescaled_diagonal_vector.abs().pow(2), function=lambda x: x,
                              learnable=False)

    def inv(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of an inverse diagonal of operator.

        :param vector: input vector to apply diagonal operator inverse
        :return: result of diagonal operator inverse applied to vector
        """
        rescaled_diag_vector = self.rescaled_diagonal_vector
        assert th.all(rescaled_diag_vector != 0)
        return (vector/rescaled_diag_vector).reshape_as(vector)

    def inv_T(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of an inverse transpose diagonal of operator, which is the same as its inverse.

        :param vector: input vector to apply diagonal operator inverse transpose
        :return: result of diagonal operator inverse applied to vector
        """
        return self.inv(vector)

    def get_rows_norms(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes Euclidean norm of each row in a current matrix, which is the absolute value of each
        diagonal element. In order to broadcast shape of incoming vector, this operation is performed by calling .apply
        on the corresponding vector of ones and taking abs value.

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :return: vector of shape [B, ...], each element of which contains Euclidean norm of corresponding matrix row
        """
        return self.apply(th.ones_like(vector), *args, **kwargs).abs()

    def get_cols_squared_norms(self, vector: th.Tensor, *args, diag_at_left: Optional[th.Tensor] = None, **kwargs
                               ) -> th.Tensor:
        """
        This method computes squared Euclidean norm of each column in a current matrix, which is the squared value
        of each diagonal element. In order to properly broadcast shape of incoming vector, this operation is performed
        by calling .apply(..., n=2) on the corresponding vector of ones (or diag_at_left).

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :param diag_at_left: diagonal of a diagonal matrix applied from the left to compute norm of columns jointly for
                             the product
        :return: vector of shape [B, ...], with each element containing squared Euclidean norm of corresponding matrix
                 column
        """
        if diag_at_left is not None:
            vec = diag_at_left
        else:
            vec = th.ones_like(vector)
        ret = self.apply(vec, *args, n=2, **kwargs)
        return ret

    @property
    def T_operator(self) -> 'LearnableDiagonalOperator':
        return self

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs) -> th.Tensor:
        if operator_between is None:
            return self.quadratic_operator(vector, *args, **kwargs)
        else:
            return super(LearnableDiagonalOperator, self).transpose_apply(
                vector, *args, operator_between=operator_between, **kwargs)


class LearnableNumberOperator(LearnableDiagonalOperator):
    """
    This operator represents a learnable weight in QM paradigm, which is just a single number
    """
    def __init__(self, scale_weight: th.Tensor = None, function: Callable = th.exp, learnable: bool = True) -> None:
        """
        Initializing operator parameters

        :param scale_weight: vector, which is placed on diagonal of operator matrix
        :param function: function to rescale diagonal elements
        :param learnable: flag, which determines whether to make filters learnable
        """
        super(LearnableLinearOperatorBase, self).__init__()
        self.set_parameters(scale_weight, function, diagonal_vector_shape=(1, ))
        if learnable:
            self.cast_parameters_to_nn_param()

    def apply(self, vector: th.Tensor, *args, n: int = 1, **kwargs) -> th.Tensor:
        """
        This method performs batched point-wise multiplication of input vector with operator's number.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param n: how many times to apply operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        diag = self.rescaled_diagonal_vector if n == 1 else self.rescaled_diagonal_vector.pow(n)
        return diag*vector

    def inv(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of an inverse diagonal of operator.

        :param vector: input vector to apply diagonal operator inverse
        :return: result of diagonal operator inverse applied to vector
        """
        rescaled_diag_vector = self.rescaled_diagonal_vector
        assert th.all(rescaled_diag_vector != 0)
        return vector/rescaled_diag_vector

    @property
    def sqrt_operator(self) -> 'LearnableDiagonalOperator':
        """
        This property returns diagonal operator B, corresponding to square root of current diagonal operator
        A = B*B = B^T B.

        :return: linear operator, representing square root of the current one
        """
        return self.__class__(scale_weight=self.rescaled_diagonal_vector.pow(0.5), function=lambda x: x,
                              learnable=False)

    @property
    def quadratic_operator(self) -> 'LinearOperatorBase':
        """
        This property returns diagonal operator B, corresponding to square of current diagonal operator
        A = = B^T B.

        :return: linear operator, representing square of the current one
        """
        return self.__class__(scale_weight=self.rescaled_diagonal_vector.abs().pow(2), function=lambda x: x,
                              learnable=False)
