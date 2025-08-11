from typing import Union, Tuple, List, Optional, Callable

import torch as th
from torch.nn import functional as F

from irrn.operators import Pad2DOperator, LearnableConvolutionOperator, JacobianOperatorBase, \
    LinearDegradationOperatorBase
from irrn.utils import MultiVector
from irrn.utils.fft import ConvOperatorFFTHelper, rfft2, irfft2


class ConvDecimateDegradationHelper:
    pad_operator: Pad2DOperator
    scale_factor: int

    def apply_nonlinear(self, images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        This method performs transformation of input vector using bilinear degradation model corresponding to Jacobian.

        :param images: batch of images to be transformed by bilinear degradation
        :param kernels: batch of kernels to be transformed by bilinear degradation
        :return: degraded output, produced by bilinear degradation model
        """
        assert isinstance(images, th.Tensor)
        assert isinstance(kernels, th.Tensor)
        self.prepare_padding(kernels)
        return self._decimate(self.valid_convolve(self.pad_operator(images), kernels))

    def prepare_padding(self, kernels: th.Tensor) -> None:
        """
        Set the amount of padding needed for 'same' convolution with given kernels.

        :param kernels: filters bank of shape [B, 1, h, w], used in convolution
        :return: Nothing
        """
        if kernels is None:
            padding = 0
        else:
            padding = self._compute_padding_size(kernels.shape[-2:])
        self.pad_operator.set_padding_size(padding)

    @staticmethod
    def valid_convolve(images: th.Tensor, kernels: Union[th.Tensor, type(None)]) -> th.Tensor:
        """
        Method, which performs a valid convolution of batch of images with batch of kernels. If kernels
        provided is None, then input tensor is returned as-is, simulating convolution with dirac.

        :param images: batch of images of shape [B, C, H, W]
        :param kernels: batch of kernels of shape [B, 1/C, h, w] or None
        :return: convolved images of shape [B, C, H-h, W-w]
        """
        if kernels is None:
            return images
        if kernels.shape[1] == 1:
            ret = th.conv2d(images.transpose(1, 0), kernels, groups=kernels.shape[0]).transpose(1, 0)
        else:
            # merge batch and channel dimensions before applying convolution
            ret = th.conv2d(images.view(-1, *images.shape[-2:]).unsqueeze(1).transpose(1, 0),
                            kernels.view(-1, *kernels.shape[-2:]).unsqueeze(1),
                            groups=images.shape[:2].numel()).transpose(1, 0)
            # unmerge batch and channel dimensions back
            ret = ret.view(*images.shape[:2], *ret.shape[-2:])
        return ret

    @staticmethod
    def valid_convolve_transpose(images: th.Tensor, kernels: Union[th.Tensor, type(None)]) -> th.Tensor:
        """
        Method, which performs a transpose valid convolution of batch of images with batch of kernels. If kernels
        provided is None, then input tensor is returned as-is, simulating transpose convolution with dirac.

        :param images: batch of images of shape [B, C, H, W]
        :param kernels: batch of kernels of shape [B, 1/C, h, w] or None
        :return: transpose convolved images of shape [B, C, H-h, W-w]
        """
        if kernels is None:
            return images
        if kernels.shape[1] == 1:
            ret = th.conv_transpose2d(images.transpose(1, 0), kernels, groups=kernels.shape[0]).transpose(1, 0)
        else:
            # merge batch and channel dimensions before applying transpose convolution
            ret = th.conv_transpose2d(images.view(-1, *images.shape[-2:]).unsqueeze(1).transpose(1, 0),
                                      kernels.view(-1, *kernels.shape[-2:]).unsqueeze(1),
                                      groups=images.shape[:2].numel()).transpose(1, 0)
            # unmerge batch and channel dimensions back
            ret = ret.view(*images.shape[:2], *ret.shape[-2:])
        return ret

    def decimate(self, images: th.Tensor, scale_factor: Union[int, th.Tensor]) -> th.Tensor:
        """
        Method that performs downscaling of input image by decimation operation.
        Example of such operation for 1/2 downscaling is given below:
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        -------------------------
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        Here  '+' - pixels, from which the downscaled image will be constructed.
        We can think about this operation as firstly dividing image by superpixels and secondly selecting one pixel per
        each superpixel.

        :param images: batch of input images with shape [B, C, H, W] to downscale
        :param scale_factor: downscaling factor
        :return: batch of downscaled images with shape [B, C, H//self.scale_factor, W//self.scale_factor]
        """
        scale_factor = int(scale_factor)
        return images[..., 0::scale_factor, 0::scale_factor]

    def decimate_transpose(self, images: th.Tensor, scale_factor: Union[int, th.Tensor]) -> th.Tensor:
        """
        Method that performs upscale of input image by transpose decimation operation.
        Example of such operation for 2x upscale is given below:
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        -------------------------
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        Here  '+' - pixels from input image, which are padded with zeros to construct image of higher resolution.

        :param images: batch of input images with shape [B, C, H, W] to upscale
        :param scale_factor: upscaling factor
        :return: batch of scaled images with shape [B, C, H*self.scale_factor, W*self.scale_factor]
        """
        scale_factor = int(scale_factor)
        upscaled = th.zeros(*images.shape[:-2], images.shape[-2] * scale_factor, images.shape[-1] * scale_factor,
                            dtype=images.dtype, device=images.device)
        upscaled[..., 0::scale_factor, 0::scale_factor].copy_(images)
        return upscaled

    def _decimate(self, images: th.Tensor) -> th.Tensor:
        """
        Method that performs downscaling of input image by decimation operation.
        Example of such operation for 1/2 downscaling is given below:
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        -------------------------
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        Here  '+' - pixels, from which the downscaled image will be constructed.
        We can think about this operation as firstly dividing image by superpixels and secondly selecting one pixel per
        each superpixel.

        :param images: batch of input images with shape [B, C, H, W] to downscale
        :return: batch of downscaled images with shape [B, C, H//self.scale_factor, W//self.scale_factor]
        """
        self.check_dims(images.shape, self.scale_factor)
        return self.decimate(images, self.scale_factor)

    def _decimate_transpose(self, images: th.Tensor) -> th.Tensor:
        """
        Method that performs upscale of input image by transpose decimation operation.
        Example of such operation for 2x upscale is given below:
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        -------------------------
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        Here  '+' - pixels from input image, which are padded with zeros to construct image of higher resolution.

        :param images: batch of input images with shape [B, C, H, W] to upscale
        :return: batch of scaled images with shape [B, C, H*self.scale_factor, W*self.scale_factor]
        """
        return self.decimate_transpose(images, self.scale_factor)

    def _decimate_transpose_apply(self, images: th.Tensor, inplace: bool = False) -> th.Tensor:
        """
        Method that performs upscale of input image by transpose decimation operation.
        Example of such operation for 2x upscale is given below:
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        -------------------------
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        Here  '+' - corresponding pixels from input image, the rest are replaced with zeros.

        :param images: batch of input images with shape [B, C, H, W] to upscale
        :param inplace: whether to perform transpose_apply operation inplace or not
        :return: batch of scaled images with shape [B, C, H*self.scale_factor, W*self.scale_factor]
        """
        self.check_dims(images.shape, self.scale_factor)
        return self.decimate_transpose_apply(images, self.scale_factor, inplace)

    def decimate_transpose_apply(self, images: th.Tensor, scale_factor: Union[int, th.Tensor], inplace: bool
                                 ) -> th.Tensor:
        """
        Method that performs decimation followed by its transpose of input image .
        Example of such operation for 2x upscale is given below:
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        -------------------------
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        Here  '+' - corresponding pixels from input image, the rest are replaced with zeros.

        :param images: batch of input images with shape [B, C, H, W] to process
        :param inplace: whether to perform transpose_apply operation inplace or not
        :return: batch of scaled images with shape [B, C, H, W]
        """
        scale_factor = int(scale_factor)
        decimated = th.zeros_like(images)
        decimated[..., 0::scale_factor, 0::scale_factor].copy_(images[..., 0::scale_factor, 0::scale_factor])
        return decimated

    def _transpose_kernel_part_fft(self, vector: th.Tensor, images: th.Tensor) -> th.Tensor:
        """
        Auxiliary method, which performs fft-based ops in kernel part of transpose Jacobian

        :param vector: input vector after transpose decimation
        :param images: images after padding
        :return: fft-based result of transpose Jacobian kernel part
        """
        kernel_part = rfft2(images) * \
                      rfft2(vector, images.shape[-2:]).conj()
        kernel_part = irfft2(kernel_part, images.shape[-2:])
        kernel_part = th.roll(kernel_part, (vector.shape[-2] // 2, vector.shape[-1] // 2), dims=(-2, -1))
        kernel_part = LearnableConvolutionOperator.crop_for_valid_conv(kernel_part, vector.shape[-2:])
        kernel_part = kernel_part.sum(1).unsqueeze(1)
        return kernel_part

    @staticmethod
    def _compute_padding_size(filters_size: Union[int, Tuple[int], List[int]]) -> Tuple[int, int, int, int]:
        """
        Auxiliary method, which computes the amount of padding needed for 'same' convolution.

        :param filters_size: filter bank of shape [B, 1, h, w], used in convolution
        :return: padding sizes in pixels given as (left, right, top, bottom)
        """
        if isinstance(filters_size, int):
            k_size_h = k_size_w = filters_size
        else:
            assert len(filters_size) >= 2
            k_size_h, k_size_w = filters_size[-2], filters_size[-1]
        pad_left = k_size_w // 2
        pad_right = pad_left - 1 + k_size_w % 2
        pad_top = k_size_h // 2
        pad_bottom = pad_top - 1 + k_size_h % 2
        return pad_top, pad_bottom, pad_left, pad_right

    def _init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.
        For linear downscale degradation nearest neighbours upscale suits better for initialization.
        This is similar to transpose decimation, but zeros are filled with neighbouring pixels intensity values.
        Borders are treated in a way to smoothly continue image, independently of convolution type.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        hw = [i * self.scale_factor for i in degraded_images.shape[-2:]]
        upscaled = th.zeros(*degraded_images.shape[:2], *hw, dtype=degraded_images.dtype, device=degraded_images.device)
        upscaled = upscaled.unfold(-2, self.scale_factor, self.scale_factor).unfold(-2, self.scale_factor,
                                                                                    self.scale_factor)
        upscaled[:, :, :, :, :, :] = degraded_images.unsqueeze(-1).unsqueeze(-1)
        upscaled = upscaled.permute(0, 1, 4, 5, 2, 3).flatten(start_dim=-2).flatten(start_dim=1, end_dim=-2)
        upscaled = F.fold(upscaled, hw, self.scale_factor, stride=self.scale_factor)
        if self.pad_operator.pad_mode is None:
            upscaled = self.pad_operator.symmetric_pad(upscaled)
        #upscaled = edgetaper(upscaled, self.kernels)
        return upscaled.to(degraded_images)

    def _init_latent_kernels(
            self, degraded_images: th.Tensor, kernels_size: Union[int, Tuple[int], List[int]]) -> th.Tensor:
        """
        This method is used to init latent kernels for the first restoration step.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :param kernels_size: size of blur or downscale kernels to initialize
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        num_kernels = degraded_images.shape[0]
        if self.scale_factor == 1:
            # here may be different kernels initialization for deblurring problem
            latent = self._init_kernels_gaussian(num_kernels, kernels_size)
        else:
            latent = self._init_kernels_gaussian(num_kernels, kernels_size)
        return latent.to(degraded_images)

    @staticmethod
    def _init_kernels_dirac(batch_size: int, kernels_size: Union[int, Tuple[int], List[int]]) -> th.Tensor:
        """
        Initializes kernels as Dirac delta functions.

        :param batch_size: number of kernels to initialize
        :param kernels_size: size of kernels to initialize
        :return: batch with initialized kernels
        """
        if isinstance(kernels_size, int):
            kernels_size = (kernels_size, kernels_size)
        assert len(kernels_size) == 2
        kernels = th.zeros(batch_size, 1, *kernels_size)
        kernels[:, :, kernels_size[0] // 2, kernels_size[1] // 2] = 1
        return kernels

    @staticmethod
    def _init_kernels_gaussian(batch_size: int, kernels_size: Union[int, Tuple[int], List[int]]) -> th.Tensor:
        """
        Initializes kernels as Gaussians.

        :param batch_size: number of kernels to initialize
        :param kernels_size: size of kernels to initialize
        :return: batch with initialized kernels
        """

        def gkern(size, sigma=0.5):
            x, y = th.meshgrid(th.linspace(-1, 1, size[-2]), th.linspace(-1, 1, size[-1]))
            d = x * x + y * y
            g = th.exp(-d / (2.0 * sigma ** 2))
            g /= g.sum()
            return g

        if isinstance(kernels_size, int):
            kernels_size = (kernels_size, kernels_size)
        kernels = gkern(kernels_size, 0.5).unsqueeze(0).unsqueeze(0)
        kernels = kernels.expand(batch_size, -1, -1, -1).clone()
        return kernels

    def _init_pad_operator(self, pad_operator: Pad2DOperator, padding_mode: str) -> None:
        """
        Auxilary
        :param pad_operator:
        :param padding_mode:
        :return:
        """
        if pad_operator:
            self.pad_operator = pad_operator
        else:
            self.pad_operator = Pad2DOperator(padding_mode=padding_mode)

    @staticmethod
    def check_dims(dims: Tuple[int], scale_factor: int) -> None:
        assert dims[-1] % scale_factor == 0 and dims[-2] % scale_factor == 0, \
            'Dims of vector should be divisible by scale factor. ' \
            f'Given scale factor {scale_factor} and vector of shape {dims}'


class ConvDecimateLinearDegradationOperator(ConvDecimateDegradationHelper, ConvOperatorFFTHelper,
                                            LinearDegradationOperatorBase):
    kernel_flipped: th.Tensor
    """
    This class represents general conv->decimate linear degradation. Depending on scale factor this can be used for
    the following problems:
    1 (default). scale factor = 1: decimate is identity, so degradation operator is convolution => deblurring problem
    2. scale factor > 1: degradation operator is downscale with some kernel => super-resolution problem
    Kernel and scale factor are supposed to be batch-dependent, padding mode is not.
    """
    def __init__(self, scale_factor: Optional[int] = 1, kernel: Optional[th.Tensor] = None,
                 pad_operator: Optional[Pad2DOperator] = None, padding_mode: Optional[str] = None) -> None:
        if pad_operator is not None:
            assert isinstance(pad_operator, Pad2DOperator)
            self.pad_operator = pad_operator
        else:
            self.pad_operator = Pad2DOperator(padding_size=0, padding_mode=padding_mode)
        self.kernel = kernel
        self.prepare_padding(kernel)
        self.scale_factor = scale_factor
        self.update_flipped_kernel(kernel)

    def init_with_parameters(self, scale_factor: Optional[int] = None, kernel: th.Tensor = None, **kwargs
                             ) -> 'ConvDecimateLinearDegradationOperator':
        """
        This method sets data-dependent volatile parameters (scale factor and kernel) of degradation and returns
        corresponding degradation operator as a new instance. Padding mode is supposed to be the same for every data.
        If scale_factor is not provided, using already set scale factor.
        If kernel is not provided, identity (dirac) is used.

        :param scale_factor: new scale factor to reparametrize operator
        :param kernel: new degradation kernel to reparametrize operator
        :param kwargs: everything else that might arrive from new data
        :return: prepared degradation operator
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
        else:
            assert isinstance(scale_factor, int), 'Only integer scale factors are supported.'
        if kernel is not None:
            assert isinstance(kernel, th.Tensor)
        self.kernel = kernel
        ret = ConvDecimateLinearDegradationOperator(scale_factor=scale_factor,
                                                    kernel=kernel, pad_operator=self.pad_operator)
        return ret

    def init_with_parameters_(self, scale_factor: Optional[int] = None, kernel: th.Tensor = None,
                              **kwargs) -> 'ConvDecimateLinearDegradationOperator':
        """
        This method sets data-dependent volatile parameters (scale factor and kernel) of degradation operator inplace.
        Padding mode is supposed to be the same for every data.
        If scale_factor is not provided, using already set scale factor.
        If kernel is not provided, identity (dirac) is used.

        :param scale_factor: new scale factor to reparametrize operator
        :param kernel: new degradation kernel to reparametrize operator
        :param kwargs: everything else that might arrive from new data
        :return: self
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
        else:
            assert isinstance(scale_factor, int), 'Only integer scale factors are supported.'
        if kernel is not None:
            assert isinstance(kernel, th.Tensor)
        self.kernel = kernel
        self.scale_factor = scale_factor
        self.prepare_padding(kernel)
        self.update_flipped_kernel(kernel)
        return self

    def update_flipped_kernel(self, kernel: Union[th.Tensor, type(None)]) -> None:
        """
        This method computes flipped kernel and writes it to 'kernel_flipped' in order for it to be used in
        accelerated transpose convolutions.

        :param kernel: tensor of shape [..., h, w] representing the original kernel
        :return: Nothing
        """
        self.kernel_flipped = None if kernel is None else th.flip(kernel, dims=(-1, -2))

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched linear degradation of input vector by convolution with blur kernel possibly
        followed by decimation (if scale factor != 1).

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        assert isinstance(vector, th.Tensor)
        if vector.dim() == 4:
            return self._decimate(self.valid_convolve(self.pad_operator(vector), self.kernel))
        elif vector.dim() == 5:
            ret = []
            for i in range(vector.shape[1]):
                ret.append(self.apply(vector[:, i], *args, **kwargs))
            ret = th.stack(ret, dim=1)
            return ret
        else:
            raise NotImplementedError

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched transpose conv+decimate degradation of input vector.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        assert isinstance(vector, th.Tensor)
        if vector.dim() == 4:
            # accelerate inference on GPU by using cudnn::detail::implicit_convolve_sgemm (th.conv2d) with flipped
            # kernel and a proper padding accounted instead of using cudnn::detail::dgrad_engine (th.conv_transpose2d)
            ret = self.pad_operator.zero_pad(self._decimate_transpose(vector), n=2)
            ret = self.valid_convolve(ret, self.kernel_flipped)
            return self.pad_operator.T(ret)
        elif vector.dim() == 5:
            ret = []
            for i in range(vector.shape[1]):
                ret.append(self._transpose(vector[:, i], *args, **kwargs))
            ret = th.stack(ret, dim=1)
            return ret
        else:
            raise NotImplementedError

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs) -> th.Tensor:
        """
        This method performs linear transformation followed by transpose transformation with the same operator.
        If there is no operator in between, then the operation is accelerated by fusing transpose decimation and
        decimation into a single operation.

        :param vector: input vector of shape [B, ...] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector, transformed by linear operator and its transpose
        """
        if operator_between is None:
            ret = self.valid_convolve(self.pad_operator(vector), self.kernel)
            ret = self._decimate_transpose_apply(ret, inplace=True)
            ret = self.pad_operator.zero_pad(ret, n=2)
            ret = self.pad_operator.T(self.valid_convolve(ret, self.kernel_flipped))
            return ret
        else:
            return super(ConvDecimateLinearDegradationOperator, self).transpose_apply(
                vector, *args, operator_between=operator_between, **kwargs)

    @property
    def kernel_unified_shape(self) -> Union[th.Tensor, type(None)]:
        """
        This property returns kernel, casted to the unified shape [B, C_out, C_in, h, w], where ? is either 1 or
        batch size.
        :return: convolution filters used to parametrize operator with shape [B, 1, 1, h, w]
        """
        if self.kernel is None:
            return None
        return self.kernel.unsqueeze(1)

    def _rate_of_ones_on_diag_between(self, h_in: int, w_in: int) -> float:
        if self.pad_operator.pad_mode == 'periodic':
            ones_rate = 1.0/(self.scale_factor**2)
        else:
            top, bot, left, right = self._compute_padding_size(self.kernel.shape)
            h_after_crop = (h_in - top - bot)
            w_after_crop = (w_in - left - right)
            self.check_dims((h_after_crop, w_after_crop), self.scale_factor)
            num_ones_on_diag = (h_after_crop * w_after_crop) // (self.scale_factor ** 2)
            ones_rate = num_ones_on_diag / (h_in * w_in)
        return ones_rate

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        if self.kernel is not None:
            self.kernel = self.kernel.to(*args, **kwargs)
        if self.kernel_flipped is not None:
            self.kernel_flipped = self.kernel_flipped.to(*args, **kwargs)
        return self

    def get_rows_norms(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method computes Euclidean norm of each row in a current matrix using the trick described
        below: $||C_i||^2_2 = (C_i^{\circ 2}) e$.
        Here $e$ is a vector of ones, $C_i$ is the i-th row of matrix C and $C^{\circ 2}$ is a Hadamard square of
        matrix C. Since C represents convolution+decimation matrix, each of its element is either zero or some value
        from a corresponding convolution kernel, so one can construct operator corresponding to
        $C^{\circ 2}$ by element-wise squaring the convolution kernel and considering convolution+decimation with this
        kernel afterwards.

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :return: vector of shape [B, ...], each element of which contains Euclidean norm of corresponding matrix row
        """
        kernel_old = self.kernel
        if self.kernel is not None:
            self.kernel = kernel_old**2
        ret = self.apply(th.ones_like(vector), *args, **kwargs)
        self.kernel = kernel_old
        return th.sqrt(ret)

    def get_cols_squared_norms(self, vector: th.Tensor, *args, diag_at_left: Optional[th.Tensor] = None, **kwargs
                               ) -> th.Tensor:
        """
        This method computes Euclidean squared norm of each column in a current matrix using the trick
        described below: $||C^T_i||^2_2 = (C^{\circ 2})^T_i e$.
        Here $e$ is a vector of ones, $C^T_i$ is the i-th column of matrix C and $C^{\circ 2}$ is a Hadamard square of
        matrix C. Since C represents convolution+decimation matrix, each of its element is either zero or some value
        from a corresponding convolution kernel, so one can construct operator corresponding to $(C^{\circ 2})^T$ by
        element-wise squaring the convolution kernel and considering transpose convolution+decimation with this kernel
        afterwards.
        If matrix $C$ is multiplied by some diagonal matrix $D = diag(d)$ from the left, this method can compute
        Euclidean squared norm for the product $DC$ by applying (C^{\circ 2})^T_i to the vector $d^2$ instead of
        $e$: $||(DC)^T_i||^2_2 = (C^{\circ 2})^T_i d^2$.

        :param vector: example of vector of shape [B, ...], which the matrix is supposed to be applied to;
                       only the shape and dtype of this argument are used by the method
        :param args, kwargs: auxiliary arguments which might be needed to perform matrix call
        :param diag_at_left: diagonal of a diagonal matrix applied from the left to compute norm of columns jointly for
                             the product
        :return: vector of shape [B, ...], with each element containing squared Euclidean norm of corresponding matrix
                 column
        """
        kernel_flipped_orig = self.kernel_flipped
        if self.kernel_flipped is not None:
            self.kernel_flipped = kernel_flipped_orig**2
        if diag_at_left is not None:
            vec = diag_at_left
        else:
            vec = th.ones_like(vector)
        ret = self._transpose(vec, *args, **kwargs)
        self.kernel_flipped = kernel_flipped_orig
        return ret


class ConvMosaicLinearDegradationOperator(ConvDecimateLinearDegradationOperator):
    def __init__(self, kernel: Optional[th.Tensor] = None, pad_operator: Optional[Pad2DOperator] = None,
                 padding_mode: Optional[str] = None, pattern='rggb') -> None:
        super(ConvMosaicLinearDegradationOperator, self).__init__(
            scale_factor=2, kernel=kernel, pad_operator=pad_operator, padding_mode=padding_mode)
        self._check_pattern_type(pattern)
        self.pattern = pattern

    def init_with_parameters(self, kernel: th.Tensor = None, pattern=None, **kwargs
                             ) -> 'ConvMosaicLinearDegradationOperator':
        """
        This method sets data-dependent volatile parameters (scale factor and pattern) of degradation and returns
        corresponding degradation operator as a new instance. Padding mode is supposed to be the same for every data.
        If pattern is not provided, using already set pattern.
        If kernel is not provided, identity (dirac) is used.

        :param kernel: new degradation kernel to reparametrize operator
        :param pattern: the pattern for mosaic, by default is rggb
        :param kwargs: everything else that might arrive from new data
        """
        if pattern is None:
            pattern = self.pattern
        return ConvMosaicLinearDegradationOperator(kernel=kernel, pattern=pattern, pad_operator=self.pad_operator)

    def init_with_parameters_(self, kernel: th.Tensor = None, pattern=None, **kwargs
                              ) -> 'ConvMosaicLinearDegradationOperator':
        """
        This method sets data-dependent volatile parameters (scale factor and pattern) of degradation inplace and
        returns the degradation operator with updated parameters.
        Padding mode is supposed to be the same for every data.
        If pattern is not provided, using already set pattern.
        If kernel is not provided, identity (dirac) is used.

        :param kernel: new degradation kernel to reparametrize operator
        :param pattern: the pattern for mosaic, by default is rggb
        :param kwargs: everything else that might arrive from new data
        """
        if pattern is None:
            pattern = self.pattern
        else:
            self._check_pattern_type(pattern)
        if kernel is not None:
            assert isinstance(kernel, th.Tensor)
        self.kernel = kernel
        self.pattern = pattern
        self.prepare_padding(kernel)
        return self

    def _decimate(self, images: th.Tensor) -> th.Tensor:
        """
        This method performs batched mosaicing according to the pattern.

        :param images: input vector of shape [B, ...] to be transformed by linear operator
        :return: vector of shape [B, ...], transformed by linear operator
        """
        self.check_dims(images.shape, self.scale_factor)
        return self.decimate(images, self.pattern)

    def _decimate_transpose(self, images: th.Tensor) -> th.Tensor:
        """
        Method that performs 'demosaic' of input image by transpose decimation operation.

        :param images: batch of input images with shape [B, 4, H // 2, W // 2] to upscale
        :return: batch of scaled images with shape [B, 3, H, W]
        """
        return self.decimate_transpose(images, self.pattern)

    def _decimate_transpose_apply(self, images: th.Tensor, inplace: bool = False) -> th.Tensor:
        """
        Method that performs mosaicing followed by its transpose of input image.

        :param images: batch of input images with shape [B, 3, H, W] to process
        :param inplace: whether to perform transpose_apply operation inplace or not
        :return: batch of images with shape [B, 3, H, W]
        """
        self.check_dims(images.shape, self.scale_factor)
        return self.decimate_transpose_apply(images, self.pattern, inplace)

    @staticmethod
    def _check_pattern_type(pattern: str) -> None:
        """This class checks whether the supported pattern is given and raises an error if it is not the case."""
        assert pattern in ('rggb', 'bggr', 'gbrg', 'grbg'), 'Only rggb, bggr, gbrg and grbg patterns are supported.'

    @staticmethod
    def get_pattern_slices(pattern) -> List[Tuple[slice, slice]]:
        """
        This property returns vertical and horizontal tensor slices to select r, g1, g2, b pixels according to the
        pattern being used.
        :return: horizontal and vertical slices for the r, g1, g2, b pixels
        """
        if pattern == 'rggb':
            return [(slice(0, None, 2), slice(0, None, 2)), (slice(0, None, 2), slice(1, None, 2)),
                    (slice(1, None, 2), slice(0, None, 2)), (slice(1, None, 2), slice(1, None, 2))]
        elif pattern == 'bggr':
            return [(slice(1, None, 2), slice(1, None, 2)), (slice(0, None, 2), slice(1, None, 2)),
                    (slice(1, None, 2), slice(0, None, 2)), (slice(0, None, 2), slice(0, None, 2))]
        elif pattern == 'gbrg':
            return [(slice(1, None, 2), slice(0, None, 2)), (slice(0, None, 2), slice(0, None, 2)),
                    (slice(1, None, 2), slice(1, None, 2)), (slice(0, None, 2), slice(1, None, 2))]
        elif pattern == 'grbg':
            return [(slice(0, None, 2), slice(1, None, 2)), (slice(0, None, 2), slice(0, None, 2)),
                    (slice(1, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2))]
        else:
            raise ValueError

    def decimate(self, vector: th.Tensor, pattern: str) -> th.Tensor:
        """
        This method performs batched mosaicing according to the pattern.

        :param vector: input vector of shape [B, ...] to be transformed by linear operator
        :param pattern: mosaic pattern
        :return: vector of shape [B, ...], transformed by linear operator
        """
        self._check_pattern_type(pattern)
        b, c, h, w = vector.shape
        assert c == 3, "The number of channels should be 3"
        assert h % 2 == 0 and w % 2 == 0, "We consider even image sizes"
        (slices_r_h, slices_r_w), (slices_g1_h, slices_g1_w), (slices_g2_h, slices_g2_w), (slices_b_h, slices_b_w) = \
            self.get_pattern_slices(pattern)
        r = vector[:, 0:1, slices_r_h, slices_r_w]
        g1 = vector[:, 1:2, slices_g1_h, slices_g1_w]
        g2 = vector[:, 1:2, slices_g2_h, slices_g2_w]
        b = vector[:, 2:3, slices_b_h, slices_b_w]
        return th.cat([r, g1, g2, b], dim=1)

    def decimate_transpose(self, images: th.Tensor, pattern: str) -> th.Tensor:
        """
        Method that performs 'demosaic' of input image by transpose decimation operation.

        :param images: batch of input images with shape [B, 4, H // 2, W // 2] to upscale
        :param pattern: mosaic pattern
        :return: batch of scaled images with shape [B, 3, H, W]
        """
        b, c, h, w = images.shape
        self._check_pattern_type(pattern)
        assert c == 4
        result = th.zeros(*images.shape[:-3], 3, images.shape[-2] * 2, images.shape[-1] * 2, dtype=images.dtype,
                          device=images.device)
        (slices_r_h, slices_r_w), (slices_g1_h, slices_g1_w), (slices_g2_h, slices_g2_w), (slices_b_h, slices_b_w) = \
            self.get_pattern_slices(pattern)
        result[..., 0, slices_r_h,  slices_r_w].copy_(images[..., 0, :, :])  # r
        result[..., 1, slices_g1_h, slices_g1_w].copy_(images[..., 1, :, :])  # g1
        result[..., 1, slices_g2_h, slices_g2_w].copy_(images[..., 2, :, :])  # g2
        result[..., 2, slices_b_h,  slices_b_w].copy_(images[..., 3, :, :])  # b
        return result

    def decimate_transpose_apply(self, images: th.Tensor, pattern: str, inplace: bool) -> th.Tensor:
        """
        Method that performs mosaicing followed by its transpose of input image.

        :param images: batch of input images with shape [B, 3, H, W] to process
        :param pattern: mosaic pattern
        :param inplace: whether to perform transpose_apply operation inplace or not
        :return: batch of images with shape [B, 3, H, W]
        """
        b, c, h, w = images.shape
        self._check_pattern_type(pattern)
        assert c == 3
        result = th.zeros_like(images)
        (slices_r_h, slices_r_w), (slices_g1_h, slices_g1_w), (slices_g2_h, slices_g2_w), (slices_b_h, slices_b_w) = \
            self.get_pattern_slices(pattern)
        result[..., 0, slices_r_h,  slices_r_w].copy_(images[..., 0, slices_r_h,  slices_r_w])  # r
        result[..., 1, slices_g1_h, slices_g1_w].copy_(images[..., 1, slices_g1_h, slices_g1_w])  # g1
        result[..., 1, slices_g2_h, slices_g2_w].copy_(images[..., 1, slices_g2_h, slices_g2_w])  # g2
        result[..., 2, slices_b_h,  slices_b_w].copy_(images[..., 2, slices_b_h,  slices_b_w])  # b
        return result


class ImageJacobian(JacobianOperatorBase, ConvDecimateLinearDegradationOperator):
    """
    This class implements Jacobian for degradations, managing image (blur, downscale):
    y = G(x, h) + n,
    where x - image (batch of images), h - known kernel (batch of kernels)
    """
    def apply(self, vector: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector by Jacobian matrix.

        :param vector: input vector of shape [B, ...] to be transformed by Jacobian
        :param kernels: batch of kernels of shape [B, 1, h, w], which parametrize Jacobian
        :return: vector of shape [B, ...], transformed by Jacobian
        """
        assert isinstance(vector, th.Tensor)
        assert isinstance(kernels, th.Tensor)
        self.prepare_padding(kernels)
        ret = self._decimate(self.valid_convolve(self.pad_operator(vector), kernels))
        return ret

    def _transpose(self, vector: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        This method applies transpose Jacobian operation on input vector, using custom linearization point.

        :param vector: input vector of shape [B, ...] to apply transpose Jacobian
        :param kernels: batch of kernels of shape [B, 1, h, w], which parametrize Jacobian
        :return: result of transpose Jacobian, applied to input vector
        """
        self.prepare_padding(kernels)
        vector = self._decimate_transpose(vector)
        image = self.pad_operator.T(self.valid_convolve_transpose(vector, kernels))
        return image


class KernelJacobian(ConvDecimateDegradationHelper, JacobianOperatorBase):
    """
    This class implements Jacobian for degradations, managing kernel (blur, downscale):
    y = G(h, x) + n,
    where h - kernel (batch of kernels), x - image (batch of images)
    """
    def __init__(self, scale_factor: int, pad_operator: Pad2DOperator, padding_mode: str = None) -> None:
        """
        Initializing Jacobian parameters for kernel linear problem.

        :param scale_factor: downscaling factor, used in degradation
        :param pad_operator: operator to be used for padding
        :param padding_mode: which padding to use in convolution, if no pad_operator was given:
            'zero', 'symmetric', 'periodic', None padding types are supported if None - no padding is applied
            and valid convolution is used, else - same convolution is used with selected padding type
        """
        assert isinstance(scale_factor, int)
        self.scale_factor = scale_factor
        if pad_operator:
            assert isinstance(pad_operator, Pad2DOperator)
            self.pad_operator = pad_operator
        else:
            self.pad_operator = Pad2DOperator(padding_mode=padding_mode)

    def apply(self, vector: th.Tensor, images: th.Tensor) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector by Jacobian matrix.

        :param vector: input vector of shape [B, ...] to be transformed by Jacobian
        :param images: batch of images of shape [B, C, H, W], which parametrize Jacobian
        :return: vector of shape [B, ...], transformed by Jacobian
        """
        assert isinstance(vector, th.Tensor)
        self.prepare_padding(vector)
        ret = self._decimate(self.valid_convolve(self.pad_operator(images), vector))
        return ret

    def _transpose(self, vector: th.Tensor, images: th.Tensor) -> th.Tensor:
        """
        This method applies transpose Jacobian operation on input vector, using custom linearization point
        (images + kernels pair).

        :param vector: input vector of shape [B, ...] to apply transpose Jacobian
        :param images: batch of images of shape [B, C, H, W], which parametrize Jacobian
        :return: result of transpose Jacobian, applied to input vector
        """
        self.prepare_padding(vector)
        vector = self._decimate_transpose(vector)
        images = self.pad_operator(images)

        kernel_part = self._transpose_kernel_part_fft(vector, images)
        return kernel_part

    def apply_nonlinear(self, vector: th.Tensor, images: th.Tensor) -> th.Tensor:
        """
        This method performs transformation of input vector using bilinear degradation model corresponding to Jacobian.

        :param vector: batch of kernels to be transformed by bilinear degradation
        :param images: batch of images to be transformed by bilinear degradation
        :return: degraded output, produced by bilinear degradation model
        """
        return super(KernelJacobian, self).apply_nonlinear(images, vector)


class ImageKernelJacobian(ConvDecimateDegradationHelper, JacobianOperatorBase):
    """
    This class implements Jacobian for degradations, managing both image and kernel (blur, downscale):
    y = G(z) + n,
    where z = (x, h)^T, x - image (batch of images), h - kernel (batch of kernels)
    """
    kernels_size: Tuple[int]
    images_jacobian: ImageJacobian
    kernels_jacobian: KernelJacobian

    def __init__(self, scale_factor: int, padding_mode: str = None) -> None:
        """
        Initializing Jacobian parameters for image+kernel bilinear problem.

        :param scale_factor: downscaling factor, used in degradation
        :param padding_mode: which padding to use in convolution
            'zero', 'symmetric', 'periodic', None padding types are supported if None - no padding is applied
            and valid convolution is used, else - same convolution is used with selected padding type
        """
        self.scale_factor = scale_factor
        self.padding_mode = padding_mode
        self.pad_operator = Pad2DOperator(padding_mode=padding_mode)
        self.images_jacobian = ImageJacobian(scale_factor=scale_factor, pad_operator=self.pad_operator)
        self.kernels_jacobian = KernelJacobian(scale_factor, self.pad_operator)

    def init_parameters(self, degraded_images: th.Tensor,
                        kernels_size: Union[int, Tuple[int], List[int]]) -> Tuple[th.Tensor]:
        """
        This method initializes all parameters needed to compute operator output.

        :param degraded_images: batch of images of shape [B, C, H_d, W_d] required for latent images initialization
        :param kernels_size: size of blur or downscale kernels to initialize
        :return: Nothing
        """
        self.pad_operator.set_padding_size(self._compute_padding_size(kernels_size))
        images = self._init_latent_images(degraded_images)
        kernels = self._init_latent_kernels(degraded_images, kernels_size)
        return images, kernels

    def apply(self, vector: MultiVector, images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector by Jacobian matrix.

        :param vector: input vector of shape [B, ...] to be transformed by Jacobian
        :param images: batch of images of shape [B, C, H, W], which parametrize Jacobian
        :param kernels: batch of kernels of shape [B, 1, h, w], which parametrize Jacobian
        :return: vector of shape [B, ...], transformed by Jacobian
        """
        assert isinstance(vector, MultiVector)
        ret = self.images_jacobian(vector.elements[0], kernels) + self.kernels_jacobian(vector.elements[1], images)
        return ret

    def _transpose(self, vector: th.Tensor, images: th.Tensor, kernels: th.Tensor) -> MultiVector:
        """
        This method applies transpose Jacobian operation on input vector, using custom linearization point
        (images + kernels pair).

        :param vector: input vector of shape [B, ...] to apply transpose Jacobian
        :param images: batch of images of shape [B, C, H, W], which parametrize Jacobian
        :param kernels: batch of kernels of shape [B, 1, h, w], which parametrize Jacobian
        :return: result of transpose Jacobian, applied to input vector
        """
        vector = self._decimate_transpose(vector)
        images = self.pad_operator(images)

        image_part = self.pad_operator.T(self.valid_convolve_transpose(vector, kernels))
        kernel_part = self._transpose_kernel_part_fft(vector, images)

        return MultiVector((image_part, kernel_part))
