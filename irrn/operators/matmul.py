from typing import List

import torch as th

from irrn.operators import LearnableLinearOperatorBase


class LearnableMatMulOperator(LearnableLinearOperatorBase):
    """
    This operator represents batched matrix multiplication with a batch of matrices acting as learned parameters:
    Ax or xA for matrix A and incoming vector x.
    """
    def __init__(self, batch_of_matrices: th.Tensor, side: str = 'left') -> None:
        """
        Initializing linear operator with a batch of matrices.

        :param batch_of_matrices: matrices, which parametrize linear operator. The last two dims are treated as matrix,
                                  all other - batches.
        :param side: at which side to multiply vector with matrix, should be either 'left' or 'right'. If 'left' side
                     is selected, then operator will multiply incoming vector from the left: Ax. If 'right'
                     side is selected, then operator will multiply incoming vector from the left: xA.
        """
        super(LearnableMatMulOperator, self).__init__()
        assert isinstance(batch_of_matrices, th.Tensor)
        assert batch_of_matrices.dim() > 2
        assert isinstance(side, str)
        if side not in ('left', 'right'):
            self._raise_side_error(side)
        self.matrices = batch_of_matrices
        self.side = side

    @staticmethod
    def reshape_before(vector: th.Tensor) -> th.Tensor:
        """
        This method reshapes incoming vector before performing batched matrix multiplication. It should reshape vector
        such that the matrix/vector dimensions are placed at the end.
        By default returns input vector without reshaping.

        :param vector: vector to be reshaped
        :return: reshaped vector prepared for batched matmul
        """
        return vector

    @staticmethod
    def reshape_after(vector: th.Tensor) -> th.Tensor:
        """
        This method reshapes result of batched matrix multiplication.
        By default returns input vector without reshaping.

        :param vector: vector to be reshaped
        :return: reshaped vector prepared for batched matmul
        """
        return vector

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method multiplies incoming batched vector with batch of matrices, stored in linear operator.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        return self._matmul(vector, self.matrices)

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method multiplies incoming batched vector with batch of transpose matrices, stored in linear operator.

        :param vector: input vector of shape [B, ...] to be transformed by transpose linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        return self._matmul(vector, self.matrices.transpose(-1, -2))

    def sqrt(self, vector: th.Tensor) -> th.Tensor:
        """
        This method computes an output of square root of operator. Cholesky decomposition is used to find the square
        root of matrices.

        :param vector: input vector to apply operator's square root on
        :return: result of square root application to vector
        """
        assert self.matrices.shape[-1] == self.matrices.shape[-2], 'Square root is defined only for squared matrices'
        return self._matmul(vector, th.linalg.cholesky(self.matrices.transpose(-2, -1).conj()).transpose(-2, -1).conj())

    @property
    def sqrt_operator(self) -> 'LearnableMatMulOperator':
        """
        This property returns operator B, corresponding to square root of initial operator A = B^T B.
        Cholesky decomposition is used to find the square root of matrices.

        :return: linear operator, which transpose_apply corresponds to the current one
        """
        assert self.matrices.shape[-1] == self.matrices.shape[-2], 'Square root is defined only for squared matrices'
        return self.__class__(th.linalg.cholesky(self.matrices.transpose(-2, -1).conj()).transpose(-2, -1).conj(),
                              side=self.side)

    def inv(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of an inverse of operator.

        :param vector: input vector to apply operator's inverse on
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of operator's inverse application to vector
        """
        return self._matmul(vector, th.linalg.inv(self.matrices))

    def inv_T(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes an output of transpose inverse of operator.

        :param vector: input vector to apply operator's transpose inverse on
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of operator's inverse application to vector
        """
        return self._matmul(vector, th.linalg.inv(self.matrices).transpose(-1, -2))

    def _matmul(self, vector: th.Tensor, matrix: th.Tensor) -> th.Tensor:
        """
        Auxilary method, which performs batched matrix multiplication of vector given matrices.

        :param vector: input vector of shape [B, ...] to be multiplied with matrix by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        vector = self.reshape_before(vector)
        assert vector.dim() == matrix.dim()
        if self.side == 'left':
            res = th.matmul(matrix, vector)
        elif self.side == 'right':
            res = th.matmul(vector, matrix)
        else:
            self._raise_side_error(self.side)
        return self.reshape_after(res)

    @staticmethod
    def _raise_side_error(side) -> str:
        raise RuntimeError("Expected 'side' argument to be either 'left' (matmul from left) or "
                           f"'right' (matmul from right), but given {side}.")

    @property
    def parameters_names_list(self) -> List[str]:
        return ['matrices']

    @property
    def elementwise_squared_operator(self) -> 'LearnableMatMulOperator':
        """
        This method returns Hadamard squared operator, representing element-wise squared matrices.
        :return: new operator with element-wise squared matrices
        """
        return self.__class__(self.matrices**2, side=self.side)
