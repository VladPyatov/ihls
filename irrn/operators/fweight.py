from typing import Tuple, Union, List, Callable

import torch as th
from torch.fft import irfftn, rfftn

from .base import LearnableLinearOperatorBase


class LearnableFourierWeightOperator(LearnableLinearOperatorBase):
    """
    This method implements Fourier weighting operator in QML paradigm.
    """
    scale_vectors: th.Tensor
    signal_size: List[int]

    def __init__(self, num_weights: int = 1, signal_size: Union[int, List[int], Tuple[int]] = None, learnable: bool = True
                 ) -> None:
        """
        Initializing operator parameters

        :param num_weights: which number of different weights to use
        :param signal_size: two integers, which define zero-padding size, hence amount of frequencies involved in FFT
        :param learnable: flag, which determines whether to make filters learnable
        """
        super(LearnableFourierWeightOperator, self).__init__()
        self.init_parameters(num_weights, signal_size)
        if learnable:
            self.cast_parameters_to_nn_param()

    def apply(self, vector: th.Tensor) -> th.Tensor:
        """
        This method performs batched weighting of input vector in frequency domain with a bank of weights.

        :param vector: input vector of shape [B, C, H, W] to be weighted with weights bank
        :return: complex-valued vector of shape [B, N, C, H_out, W_out]
        """
        ret = self.scale_vectors*(self.rfft2(vector, self.signal_size).unsqueeze(1))
        return ret

    def _transpose(self, vector: th.Tensor, output_size: Union[List[int], Tuple[int]] = None) -> th.Tensor:
        """
        This method performs transpose batched weighting in frequency domain of input vector with weights bank

        :param vector: input vector of shape [B, N, C, H, W] to be transformed by linear operator
        :param output_size: spatial sizes of output tensor
        :return: vector of shape [B, C, output_size[0], output_size[1]], transformed by linear operator
        """
        assert len(output_size) >= 2
        ret = self.irfft2(self.scale_vectors*vector, self.signal_size).sum(dim=1)
        return ret[..., :output_size[-2], :output_size[-1]]

    def transpose_apply(self, vector: th.Tensor, operator_between: Callable = None):
        """
        This method performs operation then operator_between followed by transpose operation,
        where operation is weighting on input vector in frequency domain with weights bank

        :param vector: input vector of shape [B, N, C, H, W] to be transformed
        :param operator_between: operator, which should be applied between transformation and its transpose
        :return: vector of shape [B, C, H, W]
        """
        ret = self.apply(vector)
        if operator_between is not None:
            ret = operator_between(ret)
        ret = self._transpose(ret, vector.shape[-2:])
        return ret

    def init_parameters(self, num_weights: int, scale_vector_size: Union[int, List[int], Tuple[int], th.Size]) -> None:
        """
        This method initializes and sets all parameters needed to compute operator output.

        :param num_weights: which number of weights to return
        :param scale_vector_size: shape of diagonal vector to be initialized
        :return: Nothing
        """
        if isinstance(scale_vector_size, int):
            self.signal_size = [scale_vector_size]*2
        else:
            assert len(scale_vector_size) == 2, f'Scale vector should have two dimensions. ' \
                                                f'Given {scale_vector_size} as input.'
            self.signal_size = list(scale_vector_size)
        scale_vector_size_onesided = (self.signal_size[0], self.signal_size[1]//2 + 1)
        self.scale_vectors = th.ones(1, num_weights, 1, *scale_vector_size_onesided)

    @property
    def parameters_names_list(self) -> List[str]:
        return ['scale_vectors']

    @staticmethod
    def rfft2(signal: th.Tensor, signal_sizes: Union[Tuple[int], List[int]] = None) -> th.Tensor:
        return rfftn(signal, s=signal_sizes, dim=(-2, -1), norm="ortho")

    @staticmethod
    def irfft2(signal: th.Tensor, signal_sizes: Union[Tuple[int], List[int]] = None) -> th.Tensor:
        return irfftn(signal, s=signal_sizes, dim=(-2, -1), norm="ortho")
