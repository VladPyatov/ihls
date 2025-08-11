from typing import Tuple, Union

import torch as th

from .base import LinearOperatorBase, IdentityOperator
from irrn.utils.multivector import MultiVector


class AutogradGradLinearOperator(LinearOperatorBase):
    """
    This class implements linear operator in a form of a gradient of some tensors w.r.t. other tensors.
    """
    def __init__(self, function_outputs: Tuple[th.Tensor], function_arguments: Tuple[th.Tensor]) -> None:
        assert isinstance(function_outputs, tuple)
        assert isinstance(function_arguments, tuple)
        self.outputs = function_outputs
        self.arguments = function_arguments
        self.preconditioner_left_inv = IdentityOperator()
        self.preconditioner_right_inv = IdentityOperator()

    def apply(self, vector: Union[th.Tensor, MultiVector], *args, **kwargs) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector, given by gradient matrix.

        :param vector: input vector of shape [B, ...] to be transformed by gradient matrix
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by gradient matrix
        """
        vector_tuple = MultiVector.to_tuple(vector)
        with th.enable_grad():
            result = th.autograd.grad(self.outputs, self.arguments,
                                      grad_outputs=vector_tuple, retain_graph=True, create_graph=False,
                                      only_inputs=True, allow_unused=False)
        result = MultiVector(result)
        return result

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method applies transposed linear operation on input vector.

        :param vector: input vector of shape [B, ...] to apply transpose operation
        :param args, kwargs: altering parameters to parametrize operator
        :return: result of transposed linear operation, applied to input vector
        """
        raise NotImplementedError

    def T_operator(self) -> LinearOperatorBase:
        """
        This property returns operator, corresponding to transpose of the current one.

        :return: linear operator, transpose to the current one
        """
        raise NotImplementedError
