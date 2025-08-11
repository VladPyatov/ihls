from .base import LinearOperatorBase
import torch as th
from typing import Tuple, List, Union
import torch.nn.functional as F


class Pad2DOperator(LinearOperatorBase):
    """
    This method implements padding linear operator
    """
    pad_size: Union[int, Tuple[int], List[int]]
    pad_mode: str

    def __init__(self, padding_size: Union[int, Tuple[int], List[int]] = 0, padding_mode: str = None) -> None:
        """
        Initializing padding size and type

        :param padding_size: the size of the padding;
            If a 4-tuple, uses (padding_top, padding_bottom, padding_left, padding_right).
            If a 2-tuple, uses (padding_vertical, padding_horizontal) - the same padding in in both sides of each axis.
            If is int, uses the same padding in all boundaries.
        :param padding_mode: which padding to use: 'zero', 'symmetric', 'periodic', None padding types are supported;
            If None - no padding is applied
        """
        self.set_parameters(padding_size, padding_mode)

    def set_padding_size(self, padding_size: Union[int, Tuple[int], List[int]]) -> None:
        """
        This method sets the size of padding to be applied by operator

        :param padding_size: the size of the padding;
            If a 4-tuple, uses (padding_top, padding_bottom, padding_left, padding_right).
            If a 2-tuple, uses (padding_vertical, padding_horizontal) - the same padding in in both sides of each axis.
            If is int, uses the same padding in all boundaries.
        :return: Nothing
        """
        assert isinstance(padding_size, int) or hasattr(padding_size, '__len__')
        if isinstance(padding_size, int):
            assert (padding_size >= 0), 'Pad must be either a non-negative integer'
            self.pad_size = (padding_size, padding_size, padding_size, padding_size)
        elif len(padding_size) == 2:
            for elem in padding_size:
                assert isinstance(elem, int)
                assert (elem >= 0), 'Pad must be either a non-negative integer'
            self.pad_size = (padding_size[0], padding_size[0], padding_size[1], padding_size[1])
        elif len(padding_size) == 4:
            for elem in padding_size:
                assert isinstance(elem, int)
                assert (elem >= 0), 'Pad must be either a non-negative integer'
            self.pad_size = tuple(padding_size)
        else:
            raise ValueError(f'Expected padding_size to be single int, two ints, or four ints. '
                             f'Given padding_size={padding_size}')

    def set_padding_mode(self, padding_mode: str) -> None:
        """
        This method sets the mode of padding to be applied by operator

        :param padding_mode: which padding to use: 'zero', 'symmetric', 'periodic', None padding types are supported;
            If None - no padding is applied
        """
        if padding_mode not in (None, 'zero', 'symmetric', 'periodic'):
            raise NotImplementedError(f'Padding mode should be either zero, symmetric, periodic, or None. '
                                      f'Given padding mode {padding_mode} as input.')
        self.pad_mode = padding_mode

    def set_parameters(self, padding_size: Union[int, Tuple[int], List[int]], padding_mode: str) -> None:
        """
        This method sets both the size of padding and its mode to be applied by operator

        :param padding_size: the size of the padding;
            If a 4-tuple, uses (padding_top, padding_bottom, padding_left, padding_right).
            If a 2-tuple, uses (padding_vertical, padding_horizontal) - the same padding in in both sides of each axis.
            If is int, uses the same padding in all boundaries.
        :param padding_mode: which padding to use: 'zero', 'symmetric', 'periodic', None padding types are supported;
            If None - no padding is applied
        """
        self.set_padding_size(padding_size)
        self.set_padding_mode(padding_mode)

    def apply(self, vector: th.Tensor) -> th.Tensor:
        """
        This method performs batched 2D padding of input vector.

        :param vector: input vector of shape [..., H, W] to be padded
        :return: padded vector of shape [..., H, W]
        """
        assert isinstance(vector, th.Tensor)
        if self.pad_mode is None or not any(self.pad_size):
            return vector

        assert (self.pad_size[0] + self.pad_size[1] <= vector.shape[-2] and
                self.pad_size[2] + self.pad_size[3] <= vector.shape[-1]), "Pad size is bigger, than input vector size."
        if self.pad_mode == 'zero':
            return self.zero_pad(vector)
        elif self.pad_mode == 'periodic':
            return self.periodic_pad(vector)
        elif self.pad_mode == 'symmetric':
            return self.symmetric_pad(vector)
        else:
            raise NotImplementedError(f'Padding mode should be either zero, symmetric, periodic, or None. '
                                      f'Given padding mode {self.padding_mode} as input.')

    def _transpose(self, vector: th.Tensor, **kwargs) -> th.Tensor:
        """
        This method performs batched 2D cropping (transpose padding) of input vector.

        :param vector: input vector of shape [..., H, W] to be cropped
        :return: cropped vector of shape [..., H, W]
        """
        assert isinstance(vector, th.Tensor)
        if self.pad_mode is None or not any(self.pad_size):
            return vector

        assert (self.pad_size[0] + self.pad_size[1] <= vector.shape[-2] and
                self.pad_size[2] + self.pad_size[3] <= vector.shape[-1]), "Crop size is bigger, than input vector size."
        if self.pad_mode == 'zero':
            return self._zero_pad_transpose(vector)
        elif self.pad_mode == 'periodic':
            return self._periodic_pad_transpose(vector)
        elif self.pad_mode == 'symmetric':
            return self._symmetric_pad_transpose(vector)
        else:
            raise NotImplementedError(f'Padding mode should be either zero, symmetric, periodic, or None. '
                                      f'Given padding mode {self.padding_mode} as input.')

    def output_shape(self, images: th.Tensor) -> th.Size:
        """
        This method calculates padded images output shape, based on current padding characteristics without performing
        the exact padding operation

        :param images: input images of shape [..., H, W], for which the padded size should be calculated
        :return: shape of self.apply output, if evaluated on input images
        """
        assert images.ndim >= 2
        if self.pad_mode is None or self.pad_size == (0, 0, 0, 0):
            return images.shape

        h, w = images.shape[-2:]
        t, b, l, r = self.pad_size
        h += t + b
        w += l + r
        return th.Size((*images.shape[:-2], h, w))

    @property
    def pad_size_th(self) -> Tuple[int, int, int, int]:
        """
        This method returns padding sizes in an order (padding_left, padding_right, padding_top, padding_bottom),
        suitable for torch.nn.functional.pad

        :return: 4-tuple with padding sizes in PyTorch format
        """
        return (self.pad_size[2], self.pad_size[3], self.pad_size[0], self.pad_size[1])

    def zero_pad(self, tensor: th.Tensor, n: int = 1) -> th.Tensor:
        """
        Pads the spatial dimensions (last two dimensions) of the input tensor by inserting zeros

        :param tensor: input of shape [..., H, W] to be padded
        :return: padded tensor of shape [..., H + pad_h, W + pad_w]
        """
        pad_size = tuple(p*n for p in self.pad_size_th)
        return F.pad(tensor, pad=pad_size, mode='constant', value=0)

    def _zero_pad_transpose(self, tensor: th.Tensor) -> th.Tensor:
        """
        Adjoint of the zero padding operation which amounts to an ordinary cropping.

        :param tensor: input of shape [..., H, W] to be cropped
        :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
        """
        out = tensor[..., self.pad_size[0]:tensor.shape[-2] - self.pad_size[1],
              self.pad_size[2]:tensor.shape[-1] - self.pad_size[3]]
        return out

    def periodic_pad(self, tensor: th.Tensor) -> th.Tensor:
        """
        Pads circularly the spatial dimensions (last two dimensions) of the input tensor

        :param tensor: input of shape [..., H, W] to be padded
        :return: padded tensor of shape [..., H + pad_h, W + pad_w]
        """
        return F.pad(tensor, pad=self.pad_size_th, mode='circular')

    def _periodic_pad_transpose(self, tensor: th.Tensor) -> th.Tensor:
        """
        Adjoint of the periodic padding operation which amounts to a special type of cropping.

        :param tensor: input of shape [..., H, W] to be cropped
        :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
        """
        sz = tensor.size()
        out = tensor.clone()
        # Top
        if self.pad_size[1] != 0:
            out[..., self.pad_size[0]:self.pad_size[0] + self.pad_size[1], :] += out[..., sz[-2]-self.pad_size[1]::, :]
        # Bottom
        if self.pad_size[0] != 0:
            out[..., -self.pad_size[0] - self.pad_size[1]:sz[-2]-self.pad_size[1], :] += out[..., 0:self.pad_size[0], :]
        # Left
        if self.pad_size[3] != 0:
            out[..., self.pad_size[2]:self.pad_size[2] + self.pad_size[3]] += out[..., sz[-1]-self.pad_size[3]::]
        # Right
        if self.pad_size[2] != 0:
            out[..., -self.pad_size[2] - self.pad_size[3]:sz[-1]-self.pad_size[3]] += out[..., 0:self.pad_size[2]]
        if self.pad_size[1] == 0:
            end_h = sz[-2] + 1
        else:
            end_h = sz[-2] - self.pad_size[1]
        if self.pad_size[3] == 0:
            end_w = sz[-1] + 1
        else:
            end_w = sz[-1] - self.pad_size[3]
        out = out[..., self.pad_size[0]:end_h, self.pad_size[2]:end_w]
        return out

    def symmetric_pad(self, tensor: th.Tensor) -> th.Tensor:
        """
        Pads symmetrically the spatial dimensions (last two dimensions) of the input tensor.

        :param tensor: input of shape [..., H, W] to be padded
        :return: padded tensor of shape [..., H + pad_h, W + pad_w]
        """
        sz = list(tensor.size())
        sz[-1] = sz[-1] + sum(self.pad_size[2::])
        sz[-2] = sz[-2] + sum(self.pad_size[0:2])
        out = th.empty(sz, dtype=tensor.dtype, device=tensor.device)
        # Copy the original tensor to the central part
        out[..., self.pad_size[0]:out.size(-2) - self.pad_size[1], self.pad_size[2]:out.size(-1) - self.pad_size[3]] = \
            tensor
        # Pad Top
        if self.pad_size[0] != 0:
            out[..., 0:self.pad_size[0], :] = th.flip(out[..., self.pad_size[0]:2 * self.pad_size[0], :], (-2,))
        # Pad Bottom
        if self.pad_size[1] != 0:
            out[..., out.size(-2) - self.pad_size[1]::, :] = \
                th.flip(out[..., out.size(-2) - 2*self.pad_size[1]:out.size(-2) - self.pad_size[1], :], (-2,))
        # Pad Left
        if self.pad_size[2] != 0:
            out[..., :, 0:self.pad_size[2]] = th.flip(out[..., :, self.pad_size[2]:2 * self.pad_size[2]], (-1,))
        # Pad Right
        if self.pad_size[3] != 0:
            out[..., :, out.size(-1) - self.pad_size[3]::] = \
                th.flip(out[..., :, out.size(-1) - 2 * self.pad_size[3]:out.size(-1) - self.pad_size[3]], (-1,))
        return out

    def _symmetric_pad_transpose(self, tensor: th.Tensor) -> th.Tensor:
        """
        Adjoint of the SymmetricPad2D operation which amounts to a special type of cropping.

        :param tensor: input of shape [..., H, W] to be cropped
        :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
        """
        sz = list(tensor.size())
        out = tensor.clone()
        # Top
        if self.pad_size[0] != 0:
            out[..., self.pad_size[0]:2 * self.pad_size[0], :] += th.flip(out[..., 0:self.pad_size[0], :], (-2,))
        # Bottom
        if self.pad_size[1] != 0:
            out[..., -2 * self.pad_size[1]:-self.pad_size[1], :] += th.flip(out[..., -self.pad_size[1]::, :], (-2,))
            # Left
        if self.pad_size[2] != 0:
            out[..., self.pad_size[2]:2 * self.pad_size[2]] += th.flip(out[..., 0:self.pad_size[2]], (-1,))
        # Right
        if self.pad_size[3] != 0:
            out[..., -2 * self.pad_size[3]:-self.pad_size[3]] += th.flip(out[..., -self.pad_size[3]::], (-1,))
        if self.pad_size[1] == 0:
            end_h = sz[-2] + 1
        else:
            end_h = sz[-2] - self.pad_size[1]

        if self.pad_size[3] == 0:
            end_w = sz[-1] + 1
        else:
            end_w = sz[-1] - self.pad_size[3]
        out = out[..., self.pad_size[0]:end_h, self.pad_size[2]:end_w]
        return out
