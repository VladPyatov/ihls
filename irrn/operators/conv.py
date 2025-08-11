from typing import List, Tuple, Union, Optional, Callable

import torch as th
from scipy.fftpack import dctn
from torch import nn as nn
from torch.nn import functional as F

from irrn.modules.backbones import KPBackbone
from irrn.utils.fft import ConvOperatorFFTHelper
from .base import LearnableLinearOperatorBase
from .pad import Pad2DOperator


class LearnableConvolutionOperator(LearnableLinearOperatorBase, ConvOperatorFFTHelper):
    """
    This method implements convolution operator with padding, where convolution kernel is stored and learned directly.
    """
    effective_kernel: th.Tensor
    num_in_channels: int
    num_out_channels: int
    mix_in_channels: bool

    def __init__(self, filter_size: int = 7, filter_num_in_channels: int = 1,
                 filter_num_out_channels: Optional[int] = None, padding_mode: str = None, filters: th.Tensor = None,
                 learnable: bool = True, mix_in_channels: bool = False) -> None:
        """
        Initializing convolution filters and padding type

        :param filter_size: which filter size to use to initialize filter bank
        :param filter_num_in_channels: which number of channels to use in filter bank
            this quantity should be either 1, or number of channels of operator's inputs
        :param padding_mode: which padding to use in convolution
            'zero', 'symmetric', 'periodic', None padding types are supported if None - no padding is applied
            and valid convolution is used, else - same convolution is used with selected padding type
        :param filters: filter bank of shape [N, C, h, w] to initialize operator's filters
            if None, then filters will be initialized based on filter_size and filter_num_in_channels params
        :param learnable: flag, which determines whether to make filters learnable
        :param mix_in_channels: flag, which determines whether operator mixes input channels or convolves each input
                                channel with its own set of filters
        """
        super(LearnableConvolutionOperator, self).__init__()
        if filters is not None:
            self.effective_kernel = filters
            self.num_out_channels = filters.shape[0]
            self.num_in_channels = filters.shape[1]
        else:
            self.init_parameters(filter_size, filter_num_in_channels)
            self.num_in_channels = filter_num_in_channels
            if filter_num_out_channels is not None:
                assert filter_num_out_channels <= filter_size ** 2 - 1
                self.num_out_channels = filter_num_out_channels
                self.effective_kernel = self.effective_kernel[-filter_num_out_channels:]
            else:
                self.num_out_channels = filter_size ** 2 - 1
        padding_size = self._compute_padding_size(self.effective_kernel.shape)
        self.pad_operator = Pad2DOperator(padding_size, padding_mode)
        if learnable:
            self.cast_parameters_to_nn_param()
        self.mix_in_channels = mix_in_channels
        if self.num_in_channels == 1:
            self.mix_in_channels = False

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched linear convolution of input vector with some filterbank.

        :param vector: input vector of shape [B, C, H_in, W_in] to be convolved with filterbank
        :return: vector of shape [B, N, C, H_out, W_out] or [B, N, H_out, W_out] if C = 1 or channels mixing
                 enabled
        """
        return self._conv(self.pad_operator(vector))

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method applies transpose linear convolution with some padding on input vector

        :param vector: input vector of shape [B, N, C, H, W] or [B, N, H_out, W_out] if C = 1 or channels mixing
                       enabled to apply transpose convolution
        :return: result of transpose convolution, applied to input vector of shape [B, C, H_out, W_out]
        """
        return self.pad_operator.T(self._conv_transpose(vector))

    def init_parameters(self, filter_size: int, filter_num_in_channels: int) -> None:
        """
        Initializing convolution filters, using DCT basis

        :param filter_size: which filter size to use to initialize filter bank
        :param filter_num_in_channels: which number of channels to use in filter bank
        :return: Nothing
        """
        assert isinstance(filter_size, int) and filter_size > 0
        assert isinstance(filter_num_in_channels, int) and filter_num_in_channels > 0
        filters = self._get_dct2_filters(filter_size, filter_size)[1:, ...]
        self.effective_kernel = filters.repeat(1, filter_num_in_channels, 1, 1)

    @staticmethod
    def _compute_padding_size(kernel_size: Union[int, List[int]]) -> Tuple[int, int, int, int]:
        """
        Auxilary method, which computes the amount of padding needed for 'same' convolution.

        :param kernel_size: size of kernel [h, w], used in convolution
        :return: padding sizes in pixels given as (top, bottom, left, right)
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        k_size_h, k_size_w = tuple(kernel_size[-2:])
        pad_left = k_size_w // 2
        pad_right = pad_left - 1 + k_size_w % 2
        pad_top = k_size_h // 2
        pad_bottom = pad_top - 1 + k_size_h % 2
        return pad_top, pad_bottom, pad_left, pad_right

    def _conv(self, images: th.Tensor) -> th.Tensor:
        """
        Image convolution with operator's filterbank: Gu = (G1 u, ..., Gn u)^T

        :param images: batch with input images of shape [B, C, H_in, W_in]
        :return: tensor of shape either [B, N, C, H_out, W_out], or [B, N, H_out, W_out] if C = 1 or channels mixing
                 enabled - input convolved with filters from filter bank
        """
        if self.mix_in_channels or images.shape[1] == 1:
            return self._conv_with_mixing(images, self.effective_kernel)
        else:
            return self._conv_without_mixing(images, self.effective_kernel)

    def _conv_transpose(self, images: th.Tensor) -> th.Tensor:
        """
        Image transpose convolution with operator's filter bank: G^T (u1, ..., un) = G1 u1 + ... + Gn un

        :param images: batch with input images of shape either [B, N, C, H_out, W_out], or [B, N, H_out, W_out] if C = 1
                       or channels mixing enabled
        :return: tensor of shape [B, C, H_in, W_in] - input transpose convolved with filters from filter bank
        """
        if images.dim() == 4:
            return self._conv_transpose_with_mixing(images, self.effective_kernel)
        else:
            return self._conv_transpose_without_mixing(images, self.effective_kernel)

    @staticmethod
    def _conv_with_mixing(images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image 'valid' convolution with some filter bank: Gu = (G1 u, ..., Gn u)^T, where input channels are mixed by
        summation after convolution with a filter bank

        :param images: input images of shape [B, C, H_in, W_in]
        :param filters: bank of convolution filters of shape [N, C, h, w]
        :return: tensor of shape [B, N, H_out, W_out] - input convolved with filters from filter bank
        """
        return F.conv2d(images, filters)

    @staticmethod
    def _conv_transpose_with_mixing(images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image transpose 'valid' convolution with some filter bank: Gu = (G1 u, ..., Gn u)^T, where input channels
        are mixed by sumamtion after convolution with a filter bank

        :param images: input images of shape [B, C, H_in, W_in]
        :param filters: bank of convolution filters of shape [N, C, h, w]
        :return: tensor of shape [B, N, H_out, W_out] - input convolved with filters from filter bank
        """
        return F.conv_transpose2d(images, filters)

    @staticmethod
    def _conv_without_mixing(images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image 'valid' convolution with some filter bank: Gu = (G1 u, ..., Gn u)^T, where each input channel is convolved
        with its own corresponding filterbank, so there is no input channels intermixing.

        :param images: input images of shape [B, C, H_in, W_in]
        :param filters: bank of convolution filters of shape [N, C, h, w]
        :return: tensor of shape [B, N, C, H_out, W_out] - input convolved with filters from filter bank
        """
        assert images.dim() == 4
        assert filters.dim() == 4
        b, c_in = images.shape[:2]
        num_filters, num_filter_channels = filters.shape[:2]
        if num_filter_channels == 1:
            images = images.flatten(start_dim=0, end_dim=1).unsqueeze(1).expand(-1, num_filters, -1, -1)
            ret = F.conv2d(images, filters, groups=num_filters)
            ret = ret.view(b, c_in, num_filters, *ret.shape[-2:]).permute(0, 2, 1, 3, 4)
        else:
            images = images.repeat(1, num_filters, 1, 1)
            ret = F.conv2d(images, filters.flatten(start_dim=0, end_dim=1).unsqueeze(1), groups=num_filters*c_in)
            ret = ret.view(b, num_filters, c_in, *ret.shape[-2:])
        return ret

    @staticmethod
    def _conv_transpose_without_mixing(images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image transpose 'valid' convolution with some filter bank: G^T (u1, ..., un) = G1 u1 + ... + Gn un, where each
        input channel is transpose convolved with its own corresponding filterbank, so there is no input channels
        intermixing.

        :param images: input images of shape [B, N, C, H_out, W_out]
        :param filters: bank of convolution filters of shape [N, C, h, w]
        :return: tensor of shape [B, C, H_in, W_in] - input transpose convolved with filters from filter bank
        """
        assert images.dim() == 5
        assert filters.dim() == 4
        b, _, c_in = images.shape[:3]
        num_filters, num_filter_channels = filters.shape[:2]
        if num_filter_channels == 1:
            images = images.permute(0, 2, 1, 3, 4).flatten(start_dim=0, end_dim=1)
            ret = F.conv_transpose2d(images, filters)
            ret = ret.squeeze(1).view(b, c_in,  *ret.shape[-2:])
        else:
            ret = F.conv_transpose2d(images.flatten(start_dim=1, end_dim=2),
                                     filters.flatten(start_dim=0, end_dim=1).unsqueeze(1), groups=num_filters*c_in)
            ret = ret.reshape(b, num_filters, c_in, *ret.shape[-2:]).sum(dim=1)
        return ret

    def _batchwise_conv_without_mixing(self, images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image 'valid' convolution with a batch of some filter banks: Gu = (G1 u, ..., Gn u)^T,
        where each input channel of each element in batch is convolved with its own corresponding filterbank,
        so there is no input channels intermixing.

        :param images: input images of shape [B, C, H_in, W_in]
        :param filters: bank of convolution filters of shape [B, N, C, h, w]
        :return: tensor of shape [B, N, C, H_out, W_out] - input convolved with filters from filter bank
        """
        assert images.dim() == 4
        assert filters.dim() == 5
        images_prepared = images.flatten(end_dim=1).unsqueeze(0)
        filters_prepared = filters.permute(1, 0, 2, 3, 4).flatten(start_dim=1, end_dim=2)
        ret = self._conv_without_mixing(images_prepared, filters_prepared).squeeze(0)
        ret = ret.view(ret.shape[0], *images.shape[:2], *ret.shape[-2:]).transpose(0, 1)
        return ret

    def _batchwise_conv_transpose_without_mixing(self, images: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image transpose 'valid' convolution with a batch of some filter banks: Gu = (G1 u, ..., Gn u)^T,
        where each input channel of each element in batch is transpose convolved with its own corresponding filterbank,
        so there is no input channels intermixing.

        :param images: input images of shape [B, N, C, H_in, W_in]
        :param filters: bank of convolution filters of shape [B, N, C, h, w]
        :return: tensor of shape [B, N, C, H_out, W_out] - input convolved with filters from filter bank
        """
        assert images.dim() == 5
        assert filters.dim() == 5
        images_prepared = images.transpose(0, 1).flatten(start_dim=1, end_dim=2).unsqueeze(0)
        filters_prepared = filters.permute(1, 0, 2, 3, 4).flatten(start_dim=1, end_dim=2)
        ret = self._conv_transpose_without_mixing(images_prepared, filters_prepared).squeeze(0)
        ret = ret.view(images.shape[0], images.shape[2], *ret.shape[-2:])
        return ret

    @staticmethod
    def _batchwise_conv_with_mixing(vector: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image 'valid' convolution with some batch of filter banks: Gu = (G1 u, ..., Gn u)^T, where input channels
        are mixed by summation after convolution.

        :param vector: input vector of shape [B, C_in, H_in, W_in] to be convolved
        :param filters: bank of convolution filters of shape [B, C_out, C_in, h, w]
        :return: convolved vector of shape [B, C_out, H_out, W_out]
        """
        assert vector.dim() == 4
        assert filters.dim() == 5
        b, c_out = filters.shape[:2]
        assert vector.shape[0] == b
        ret = th.conv2d(vector.flatten(start_dim=0, end_dim=1).unsqueeze(0),
                        filters.flatten(start_dim=0, end_dim=1), groups=b)
        ret = ret.view(b, c_out, *ret.shape[-2:])
        return ret

    @staticmethod
    def _batchwise_conv_transpose_with_mixing(vector: th.Tensor, filters: th.Tensor) -> th.Tensor:
        """
        Image transpose 'valid' convolution with some batch of filter banks: Gu = (G1 u, ..., Gn u)^T, where
        input channels are mixed by summation after convolution

        :param vector: input vector of shape [B, C_out, H_out, W_out] to be transpose convolved
        :param filters: bank of convolution filters of shape [B, C_out, C_in, h, w]
        :return: transpose convolved vector of shape [B, C_in, H_in, W_in]
        """
        assert vector.ndim == 4
        assert filters.dim() == 5
        b, _, c_in = filters.shape[:3]
        assert vector.shape[0] == b

        ret = th.conv_transpose2d(vector.flatten(start_dim=0, end_dim=1).unsqueeze(0),
                                  filters.flatten(start_dim=0, end_dim=1), groups=b)
        ret = ret.view(b, c_in, *ret.shape[-2:])
        return ret

    @staticmethod
    def _get_dct2_filters(size_h: int, size_w: int) -> th.Tensor:
        """
        This method returns a 2D DCT basis in a form of convolution filter bank.
        Full basis of filters is returned except for a first basis component (constant).

        :param size_h:  vertical size of filters in filter bank
        :param size_w: horizontal size of filters in filter bank
        :return: filter bank of shape [size_h*size_w - 1, 1, size_h, size_w] with DCT basis
        """
        kernel = th.zeros(size_h * size_w, 1, size_h, size_w)
        dirac = th.zeros(size_h, size_w).numpy()
        for h in range(size_h):
            for w in range(size_w):
                dirac[h, w] = 1
                kernel[:, 0, h, w] = th.from_numpy(dctn(dirac, norm='ortho')).flatten()
                dirac[h, w] = 0
        return kernel

    @staticmethod
    def get_dirac_kernel(size_h: int, size_w: int, num_filters: int) -> th.Tensor:
        """
        This method returns a 2D identity filters in a form of convolution filter bank.

        :param size_h:  vertical size of filters in filter bank
        :param size_w: horizontal size of filters in filter bank
        :param num_filters: number of filters to create
        :return: filter bank of shape [num_filters, 1, size_h, size_w] with identity filters
        """
        kernel = th.zeros(num_filters, 1, size_h, size_w)
        kernel[:, :, size_h//2, size_w//2] = 1
        return kernel

    @staticmethod
    def crop_for_valid_conv(tensor: th.Tensor, kernel_size: Union[List[int], Tuple[int], th.Size]) -> th.Tensor:
        """
        This method crops a batch of 2D images using corresponding PSF size to produce a valid convolution

        :param tensor: batch of images of shape [..., H + hk - 1, W + wk - 1]
        :param kernel_size: size of kernel used in convolution in an order hk,wk
        :return: resulted cropped image of shape [..., H, W]
        """
        p_h = (kernel_size[-2] - 1) // 2
        a_h = (kernel_size[-2] - 1) % 2
        p_w = (kernel_size[-1] - 1) // 2
        a_w = (kernel_size[-1] - 1) % 2
        # extracting the center part
        ret = tensor[..., (p_h + a_h):-p_h, (p_w + a_w):-p_w]
        return ret

    @property
    def parameters_names_list(self) -> List[str]:
        return ['effective_kernel']

    @property
    def kernel_unified_shape(self) -> th.Tensor:
        """
        This property returns kernel, casted to the unified shape [?, C_out, C_in, h, w], where ? is either 1 or
        batch size.
        :return: convolution filters used to parametrize operator with shape [?, C_out, C_in, h, w]
        """
        return self.effective_kernel.unsqueeze(0)

    def get_rows_norms(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes Euclidean norm of each row in a current convolution matrix using the trick described
        below: $||C_i||^2_2 = (C_i^{\circ 2}) e$.
        Here $e$ is a vector of ones, $C_i$ is the i-th row of matrix C and $C^{\circ 2}$ is a Hadamard square of
        matrix C. Since C represents convolution matrix, each of its element is either zero or some value from a
        corresponding convolution kernel, so one can construct operator corresponding to $C^{\circ 2}$ by element-wise
        squaring the convolution kernel and considering convolution with this kernel afterwards.

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :return: vector of shape [B, ...], each element of which contains Euclidean norm of corresponding matrix row
        """
        effective_kernel_old = self.effective_kernel
        self.effective_kernel = effective_kernel_old**2
        ret = self.apply(th.ones_like(vector), *args, **kwargs)
        self.effective_kernel = effective_kernel_old
        return th.sqrt(ret)

    def get_cols_squared_norms(self, vector: th.Tensor, *args, diag_at_left: Optional[th.Tensor] = None,
                               override_kernel: Optional[th.Tensor] = None, **kwargs
                               ) -> th.Tensor:
        """
        This method computes Euclidean squared norm of each column in a current convolution matrix using the trick
        described below: $||C^T_i||^2_2 = (C^{\circ 2})^T_i e$.
        Here $e$ is a vector of ones, $C^T_i$ is the i-th column of matrix C and $C^{\circ 2}$ is a Hadamard square of
        matrix C. Since C represents convolution matrix, each of its element is either zero or some value from a
        corresponding convolution kernel, so one can construct operator corresponding to $(C^{\circ 2})^T$ by
        element-wise squaring the convolution kernel and considering transpose convolution with this kernel afterwards.
        If matrix $C$ is multiplied by some diagonal matrix $D = diag(d)$ from the left, this method can compute
        Euclidean squared norm for the product $DC$ by applying (C^{\circ 2})^T_i to the vector $d^2$ instead of
        $e$: $||(DC)^T_i||^2_2 = (C^{\circ 2})^T_i d^2$.

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :param diag_at_left: diagonal of a diagonal matrix applied from the left to compute norm of columns jointly for
                             the product
        :param override_kernel: override operator's convolution kernel with a custom one, which is useful when several
                                convolution operators are stacked
        :return: vector of shape [B, ...], with each element containing squared Euclidean norm of corresponding matrix
                 column
        """
        effective_kernel_orig = self.effective_kernel
        if diag_at_left is not None:
            vec = diag_at_left
        else:
            vec = th.ones_like(vector)

        if override_kernel is None:
            self.effective_kernel = effective_kernel_orig ** 2
            ret = self._transpose(vec, *args, **kwargs)
            self.effective_kernel = effective_kernel_orig
        else:
            assert isinstance(override_kernel, th.Tensor)
            assert override_kernel.dim() == 5
            self.effective_kernel = override_kernel ** 2
            self.pad_operator = Pad2DOperator(self._compute_padding_size(self.effective_kernel.shape),
                                              self.pad_operator.pad_mode)
            ret = self.pad_operator.T(LearnableKPNConvOperator._conv_transpose(self, vec))
            self.effective_kernel = effective_kernel_orig
            self.pad_operator = Pad2DOperator(self._compute_padding_size(self.effective_kernel.shape),
                                              self.pad_operator.pad_mode)
        return ret

    def output_shape(self, vector_shape: Union[th.Tensor, Tuple[int]] = None) -> Tuple[int]:
        """
        This method calculates the size of the output, when operator acts to the given vector.
        :param vector_shape: input vector or its shape, for which the output shape will be calculated
        :return: shape of the output vector
        """
        if isinstance(vector_shape, th.Tensor):
            assert vector_shape.dim() == 4
            b, c_in, h, w = vector_shape.shape
        else:
            assert len(vector_shape) == 4
            b, c_in, h, w = vector_shape
        c_out = self.kernel_unified_shape.shape[1]
        if self.mix_in_channels or c_in == 1:
            out_shape = (b, c_out)
        else:
            out_shape = (b, c_out, c_in)
        if self.pad_operator.pad_mode is not None:
            out_shape = out_shape + (h, w)
        else:
            pad_top, pad_bottom, pad_left, pad_right = self.pad_operator.pad_size
            out_shape = out_shape + (h - pad_top - pad_bottom, w - pad_left - pad_right)
        return out_shape


class LearnableCNNOperator(LearnableConvolutionOperator):
    """
    This class implements linear operator in a form of some linear convolution neural network (CNN)
    """
    _backbone: nn.Module

    def __init__(self, backbone: nn.Module, padding_mode: Optional[str] = None, learnable: bool = True,
                 mix_in_channels: bool = True) -> None:
        """
        Initializes convolution operator overparametrized by linear CNN.

        :param backbone: linear CNN, implemented as nn.Sequential in .backbone attribute
        :param padding_mode: which padding to use in convolution;
            'zero', 'symmetric', 'periodic', None padding types are supported. If None - no padding is applied
            and 'valid' convolution is used, else - 'same' convolution is used with selected padding type
        :param learnable: flag, which determines whether to make operator learnable
        :param mix_in_channels: flag, which determines whether operator mixes input channels or convolves each input
                                channel with its own set of filters
        """
        assert isinstance(backbone, nn.Module)
        assert isinstance(backbone.backbone, nn.Sequential)
        assert isinstance(learnable, bool)
        assert isinstance(mix_in_channels, bool)

        effective_kernel_size = [1, 1]
        for layer in backbone.backbone:
            self._check_layer(layer)
            effective_kernel_size[0] += layer.kernel_size[0] - 1
            effective_kernel_size[1] += layer.kernel_size[1] - 1
        super(LearnableCNNOperator, self).__init__(
            filters=th.empty(backbone.backbone[-1].out_channels, backbone.backbone[0].in_channels,
                             *tuple(effective_kernel_size)),
            padding_mode=padding_mode, learnable=False, mix_in_channels=mix_in_channels)
        self.effective_kernel_padding_size = tuple([2*p for p in self.pad_operator.pad_size_th])
        self._backbone = backbone
        self.update_effective_kernel()

    def prepare_for_restoration(self, **kwargs) -> 'LearnableCNNOperator':
        """
        This method prepares CNN operator for recurrent restoration by updating a merged kernel

        :param kwargs: parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        self.update_effective_kernel(**kwargs)
        return self

    def get_effective_kernel(self, **kwargs) -> th.Tensor:
        """
        This method calculates merged convolution kernel

        :param kwargs: parameters which may be required for kernel update
        :return: Nothing
        """
        kernel = self._get_dirac_kernel_with_mixing()
        kernel = self._backbone(kernel)
        effective_kernel = kernel.permute(1, 0, 2, 3).flip([-1, -2])
        return effective_kernel

    def update_effective_kernel(self, **kwargs) -> None:
        """
        This method calculates merged convolution kernel to replace linear CNN and writes it to
        .effective_kernel attribute inplace

        :param kwargs: parameters which may be required for kernel update
        :return: Nothing
        """
        kernel = self.get_effective_kernel(**kwargs)
        self.effective_kernel = kernel

    def _get_dirac_kernel_with_mixing(self) -> th.Tensor:
        """
        This method initializes multichannel dirac convolution kernel

        :return: multichannel dirac convolution kernel
        """
        kernel = th.eye(self.num_in_channels, dtype=next(self._backbone.backbone.parameters()).dtype,
                        device=next(self._backbone.backbone.parameters()).device)[:, :, None, None]
        kernel = F.pad(kernel, self.effective_kernel_padding_size, mode='constant', value=0)
        return kernel

    @staticmethod
    def _check_layer(layer: nn.Conv2d) -> None:
        """
        Checking whether convolutional layer satisfies all conditions necessary for kernel replacement trick.

        :param layer: layer to be checked
        :return: Nothing
        """
        assert isinstance(layer, nn.Conv2d), f'Expected layer to be of Conv2D instance, got {layer}.'
        assert layer.stride == (1, 1), f'Expected Conv2D layer to have stride=(1, 1), got {layer.stride}.'
        assert layer.dilation == (1, 1), f'Expected Conv2D layer to have dilation=(1, 1), got {layer.dilation}.'
        assert layer.padding == (0, 0), f'Expected Conv2D layer to have padding=(0, 0), got {layer.padding}.'
        assert layer.bias is None

    def to(self, *args, **kwargs) -> 'LearnableCNNOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self._backbone.to(*args, **kwargs)
        self.update_effective_kernel()
        return self

    @property
    def kernel_unified_shape(self) -> th.Tensor:
        """
        This property returns kernel, casted to the unified shape [?, C_out, C_in, h, w], where ? is either 1 or
        batch size.
        :return: convolution filters used to parametrize operator with shape [?, C_out, C_in, h, w]
        """
        return self.effective_kernel.unsqueeze(0)

    def cnn_to_conv_operator(self, learnable: Optional[bool] = None) -> LearnableConvolutionOperator:
        """
        This method converts current deep linear cnn into a corresponding convolution operator, mostly for
        inference purposes.

        :param learnable: whether to make a new operator learnable (i.e. assign filters as nn.Parameter);
                          if not provided, self.learnable value is used
        :return: copy of the current operator as a LearnableConvolutionOperator instance
        """
        if learnable is None:
            learnable = self.learnable
        train_state = self.training
        self.eval()
        self.update_effective_kernel()
        self.train(train_state)
        return LearnableConvolutionOperator(filters=self.effective_kernel, padding_mode=self.pad_operator.pad_mode,
                                            mix_in_channels=self.mix_in_channels, learnable=learnable)


class LearnableKPNConvOperator(LearnableCNNOperator):
    batch_size: Optional[int] = None
    num_out_channels: Optional[int] = None

    def __init__(self, backbone: KPBackbone, observation_keyword: Optional[str] = None,
                 latent_index: Optional[int] = None, padding_mode: Optional[str] = None, learnable: bool = True,
                 mix_in_channels: bool = True) -> None:
        """
        Initializes convolution operator which filters are predicted by deep network. Filters can be predicted either
        for the whole restoration based on observed signal, or from latent estimates of the true signal appearing in
        iterative restoration pipeline. For the first case, observation_keyword should be given, for the latter case
        latent_index should be specified.

        :param backbone: kernel prediction network
        :param observation_keyword: name of keyword argument which should be used for kernel prediction;
                                use this if you want to perform kernel prediction based on observation
        :param latent_index: index to choose the specific latent tensor to be used for kernel prediction;
                                   use this if you want to perform kernel prediction based on some latent variable
        :param padding_mode: which padding to use in convolution;
            'zero', 'symmetric', 'periodic', None padding types are supported. If None - no padding is applied
            and 'valid' convolution is used, else - 'same' convolution is used with selected padding type
        :param learnable: flag, which determines whether to make operator learnable
        :param mix_in_channels: flag, which determines whether operator mixes input channels or convolves each input
                                channel with its own set of filters
        """
        assert isinstance(backbone, KPBackbone)
        super(LearnableCNNOperator, self).__init__(filters=th.empty(*backbone.kernel_shape), padding_mode=padding_mode,
                                                   learnable=False, mix_in_channels=mix_in_channels)
        assert observation_keyword is not None or latent_index is not None, \
            'Either observation_keyword or latent_index should be given.'
        assert observation_keyword is None or latent_index is None, \
            'Both observation_keyword and latent_index are given, which is unsupported. Please keep only one of those.'
        if observation_keyword is not None:
            assert isinstance(observation_keyword, str)
        if latent_index is not None:
            assert isinstance(latent_index, int)

        self.observation_keyword = observation_keyword
        self.latent_index = latent_index

        self._backbone = backbone
        for param in self._backbone.parameters():
            param.requires_grad = learnable
        self.learnable = learnable

    def _conv(self, images: th.Tensor) -> th.Tensor:
        """
        Image convolution with operator's filterbank: Gu = (G1 u, ..., Gn u)^T

        :param images: batch with input images of shape [B, C, H_in, W_in]
        :return: tensor of shape [B, N, C, H_out, W_out] - input convolved with filters from filter bank
        """
        if self.mix_in_channels or images.shape[1] == 1:
            return self._batchwise_conv_with_mixing(images, self.kernel_unified_shape)
        else:
            return self._batchwise_conv_without_mixing(images, self.kernel_unified_shape)

    def _conv_transpose(self, images: th.Tensor) -> th.Tensor:
        """
        Image transpose convolution with operator's filter bank: G^T (u1, ..., un) = G1 u1 + ... + Gn un

        :param images: batch with input images of shape [B, N, C, H_out, W_out]
        :return: tensor of shape [B, C, H_in, W_in] - input transpose convolved with filters from filter bank
        """
        if images.dim() == 4:
            return self._batchwise_conv_transpose_with_mixing(images, self.kernel_unified_shape)
        else:
            return self._batchwise_conv_transpose_without_mixing(images, self.kernel_unified_shape)

    def prepare_for_restoration(self, **kwargs) -> 'LearnableKPNConvOperator':
        """
        This method prepares CNN operator for recurrent restoration by predicting kernels from observation

        :param kwargs: parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        if self.observation_keyword is None:
            self.clear_effective_kernel()
        else:
            assert self.observation_keyword in kwargs.keys()
            self.update_effective_kernel(kwargs[self.observation_keyword])
        return self

    def prepare_for_step(self, step_idx: int, *tensor_params: th.Tensor, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        This method prepares operator and its parameters for the next optimization step based on latent estimates.

        :param step_idx: index of current step (step number)
        :param tensor_params: parameters, which operator can be dependent to
        :param kwargs: other parameters, which are needed for step-specific parametrization
        :return: operator prepared for next optimization step
        """
        if self.latent_index is not None:
            assert self.latent_index < len(tensor_params)
            tensor = tensor_params[self.latent_index]
            self.update_effective_kernel(tensor)
        return self

    def update_effective_kernel(self, tensor: th.Tensor) -> None:
        """
        This method predicts convolution kernel and writes it to .effective_kernel attribute

        :param tensor: tensor to use for kernel prediction
        :return: Nothing
        """
        kernel = self._backbone(tensor)
        self.batch_size, self.num_out_channels, self.num_in_channels = kernel.shape[:3]
        self.effective_kernel = kernel

    def clear_effective_kernel(self) -> None:
        """
        This method clears predicted kernel and all parameters, associated with it.
        :return: None
        """
        self.effective_kernel = None
        self.batch_size = None
        self.num_out_channels = None
        self.num_in_channels = None

    def to(self, *args, **kwargs) -> 'LearnableKPNConvOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self._backbone.to(*args, **kwargs)
        if self.effective_kernel is not None:
            self.effective_kernel = self.effective_kernel.to(*args, **kwargs)
        return self

    @property
    def kernel_unified_shape(self) -> th.Tensor:
        """
        This property returns kernel, casted to the unified shape [?, C_out, C_in, h, w], where ? is either 1 or
        batch size.
        :return: convolution filters used to parametrize operator with shape [?, C_out, C_in, h, w]
        """
        return self.effective_kernel

    def cnn_to_conv_operator(self, learnable: Optional[bool] = None) -> LearnableConvolutionOperator:
        raise NotImplementedError


class LearnableKPNSAConvOperator(LearnableKPNConvOperator):
    batch_size: Optional[int] = None
    num_out_channels: Optional[int] = None

    def __init__(self, backbone: KPBackbone, observation_keyword: Optional[str] = None,
                 latent_index: Optional[int] = None, padding_mode: Optional[str] = None, learnable: bool = True,
                 mix_in_channels: bool = True) -> None:
        super(LearnableKPNSAConvOperator, self).__init__(backbone=backbone,
                                                         observation_keyword=observation_keyword,
                                                         latent_index=latent_index,
                                                         padding_mode=padding_mode,
                                                         learnable=learnable,
                                                         mix_in_channels=mix_in_channels)
        self.padding_mode = padding_mode
        self.mix_in_channels = mix_in_channels

    @staticmethod
    def _batchwise_conv(vector: th.Tensor, kernel: th.Tensor) -> th.Tensor:
        """
        Preparing for image folded convolution with a batch of some kernel banks for mixing and non-mixing modes

        :param vector: input vector of images of shape [B, C_in, H, W]
        :param kernel: bank of convolution kernels of shape [B, C_out, C_in, H - kH + 1, W - kW + 1, kH, kW]
        :return: tensor of shape [B, C_out, C_in, H - kH + 1, W - kW + 1, kH, kW] - input convolved with kernels
        from kernel bank
        """
        assert vector.dim() == 4
        assert kernel.dim() == 7
        kH = kernel.shape[-2]
        kW = kernel.shape[-1]
        x_folded = vector.unfold(-2, kH, 1).unfold(-2, kW, 1).unsqueeze(1)
        return x_folded * kernel

    def _batchwise_conv_with_mixing(self, vector: th.Tensor, kernel: th.Tensor) -> th.Tensor:
        """
        Image folded convolution with a batch of some kernel banks,
        where input channels are mixed by summation after convolution with a kernel bank

        :param vector: input vector of images of shape [B, C_in, H, W]
        :param kernel: bank of convolution kernels of shape [B, C_out, C_in, H - kH + 1, W - kW + 1, kH, kW]
        :return: tensor of shape [B, C_out, H - kH + 1, W - kW + 1] - input convolved with kernels
        from kernel bank
        """
        sum_dims = (-1, -2, 2)
        res = self._batchwise_conv(vector, kernel).sum(sum_dims)
        return res

    def _batchwise_conv_without_mixing(self, vector: th.Tensor, kernel: th.Tensor) -> th.Tensor:
        """
        Image folded convolution with a batch of some kernel banks,
        where each input channel is convolved with its own corresponding filterbank,
        so there is no input channels intermixing.

        :param vector: input vector of images of shape [B, C_in, H, W]
        :param kernel: bank of convolution kernels of shape [B, C_out, C_in, H - kH + 1, W - kW + 1, kH, kW]
        :return: tensor of shape [B, C_out, C_in, H - kH + 1, W - kW + 1] - input convolved with kernels
        from kernel bank
        """
        sum_dims = (-1, -2)
        res = self._batchwise_conv(vector, kernel).sum(sum_dims)
        return res

    def _transpose(self, vector: th.Tensor, *args, output_size: Union[List[int], Tuple[int]] = None, **kwargs
                   ) -> th.Tensor:
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

        grad = th.autograd.functional.vjp(self.apply,
                                          th.rand(vector.shape[0], *output_size[-3:],
                                                  dtype=vector.dtype, device=vector.device),
                                          vector, create_graph=create_graph)[1]
        return grad

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs):
        """
        This method performs linear transformation followed by transpose transformation with the same operator.

        :param vector: input vector of shape [B, ...] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector, transformed by linear operator and its transpose
        """
        if self.mix_in_channels or self.num_in_channels == 1:
            norm_dims = (-1, -2, -3)
        else:
            norm_dims = (-1, -2, -3, -4)
        ret = self.transpose_apply_gradient_trick(vector, *args, operator_between, norm_dims=norm_dims, **kwargs)
        return ret
