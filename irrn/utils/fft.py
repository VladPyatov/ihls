from typing import Union, Tuple, List, Optional

import torch as th
from torch.fft import rfft, irfft, rfftn, irfftn, fftn, ifftn


def fft2(signal: th.Tensor, signal_sizes: Optional[Union[Tuple[int], List[int]]] = None) -> th.Tensor:
    return fftn(signal, s=signal_sizes, dim=(-2, -1))


def ifft2(signal: th.Tensor, signal_sizes: Optional[Union[Tuple[int], List[int]]] = None) -> th.Tensor:
    return ifftn(signal, s=signal_sizes, dim=(-2, -1))


def rfft2(signal: th.Tensor, signal_sizes: Optional[Union[Tuple[int], List[int]]] = None) -> th.Tensor:
    return rfftn(signal, s=signal_sizes, dim=(-2, -1))


def irfft2(signal: th.Tensor, signal_sizes: Union[Tuple[int], List[int]]) -> th.Tensor:
    return irfftn(signal, s=signal_sizes, dim=(-2, -1))


def psf2otf(psf: th.Tensor, out_shape) -> th.Tensor:
    """
    Convert point-spread function to optical transfer function.

    Computes the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.

    To ensure that the OTF is not altered due to PSF off-centering, psf2otf
    post-pads the PSF array with zeros in the way to disperse both halves of
    the signal to different borders in both vertical and horizontal directions.

    :param psf: batch of Point Spread Functions, real-valued tensor of shape [..., H, W]
    :param out_shape: Shape of the output Optical Transfer Function, last 2 numbers in the order of H_out,W_out
    :return: Optical Transfer Function, complex-valued tensor of shape [..., H_out, W_out]
    """
    otf = psf.new_zeros(psf.shape[:-2] + out_shape[-2:])
    otf[..., :psf.shape[-2], :psf.shape[-1]] = psf

    s = th.div(th.tensor(psf.shape[-2:]), 2, rounding_mode='floor') + th.tensor(psf.shape[-2:]) % 2 - 1
    otf = th.roll(otf, tuple(-s), dims=(-2, -1))
    otf = rfft2(otf)
    return otf


def psf2otf_doublesided(psf: th.Tensor, out_shape) -> th.Tensor:
    """
    Convert point-spread function to optical transfer function.

    Computes the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.

    To ensure that the OTF is not altered due to PSF off-centering, psf2otf
    post-pads the PSF array with zeros in the way to disperse both halves of
    the signal to different borders in both vertical and horizontal directions.

    :param psf: batch of Point Spread Functions, real-valued tensor of shape [..., H, W]
    :param out_shape: Shape of the output Optical Transfer Function, last 2 numbers in the order of H_out,W_out
    :return: Optical Transfer Function, complex-valued tensor of shape [..., H_out, W_out]
    """
    otf = psf.new(psf.shape[:-2] + out_shape[-2:]).fill_(0)
    otf[..., :psf.shape[-2], :psf.shape[-1]] = psf

    s = th.tensor(psf.shape[-2:]) // 2 + th.tensor(psf.shape[-2:]) % 2 - 1
    otf = th.roll(otf, tuple(-s), dims=(-2, -1))
    otf = fft2(otf)
    return otf


def otf2psf(otf: th.Tensor, out_shape, canvas_shape) -> th.Tensor:
    """
    Convert optical transfer function to point-spread function.

    Computes the inverse Fast Fourier Transform (IFFT)
    of the optical transfer function (OTF) array and creates a point spread
    function (PSF), centered at the origin.

    To center the PSF at the origin, otf2psf circularly shifts the values
    of the output array down (or to the right) until the (1,1) element
    reaches the central position, then it crops the result to match
    dimensions specified by out_shape.

    :param otf: batch Optical Transfer Functions of shape [..., H, W]
    :param out_shape: Shape of the output Point Spread Function, last 2 numbers in the order of H,W
    :param canvas_shape: Shape of the padded Point Spread Function, last 2 numbers in the order of H,W
    :return: Point Spread Function, real-valued tensor of shape [..., out_shape[-2], out_shape[-1]]
    """
    psf = irfft2(otf, canvas_shape[-2:])
    s = th.tensor(out_shape[-2:]) // 2 + th.tensor(out_shape[-2:]) % 2 - 1
    psf = th.roll(psf, tuple(s), dims=(-2, -1))
    psf = psf[..., :out_shape[-2], :out_shape[-1]]
    return psf


def circular_conv_fft(images: th.Tensor, kernels: th.Tensor):
    """
    This function performs cicrulant convolution in frequency domain using convolution theorem.

    :param images: batch of images of shape [..., C, H, W] to be convolved
    :param kernels: batch of kernels of shape either [..., 1, H, W] or [..., C, H, W] to convolve with
    :return: batch of convolved images of shape [..., C, H, W]
    """
    assert images.dim() == kernels.dim()
    assert kernels.shape[-3] in (1, images.shape[-3])
    kernel_size = kernels.shape[-2:]
    image_size = images.shape[-2:]
    ret = rfft2(images) * rfft2(kernels, signal_sizes=image_size).conj()
    ret = irfft2(ret, signal_sizes=image_size)
    ret = th.roll(ret, (kernel_size[-2] // 2, kernel_size[-1] // 2), dims=(-2, -1))
    return ret


def edgetaper(images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
    """
    Function which performs smoothing of image boundaries to construct a better image initialization.

    :param images: images with bounds to be smoothed
    :param kernels: kernels, which are used for smoothing
    :return:
    """
    assert images.ndim == kernels.ndim == 4
    beta = {}
    if kernels.size(-1) != 1:
        kernels_projection = kernels.sum(dim=-1)
        z = th.zeros(*kernels_projection.shape[:-1], images.size(-2) - 1).type_as(kernels)
        z[..., :kernels.size(-1)] = kernels_projection
        z = rfft(z)
        z = irfft(z.abs() ** 2, n=images.size(-2) - 1)
        z = th.cat((z, z[..., 0:1]), dim=-1).div(z.max())
        beta['dim0'] = z.unsqueeze(-1)

    if kernels.size(-2) != 1:
        kernels_projection = kernels.sum(dim=-2)
        z = th.zeros(*kernels_projection.shape[:-1], images.size(-1) - 1).type_as(kernels)
        z[..., 0:kernels.size(-2)] = kernels_projection
        z = rfft(z)
        z = irfft(z.abs() ** 2, n=images.size(-1) - 1)
        z = th.cat((z, z[..., 0:1]), dim=-1).div(z.max())
        beta['dim1'] = z.unsqueeze(-2)

    if len(beta.keys()) == 1:
        alpha = 1 - beta[list(beta.keys())[0]]
    else:
        alpha = (1 - beta['dim0']) * (1 - beta['dim1'])
    while alpha.dim() < images.dim():
        alpha = alpha.unsqueeze(0)
    blurred_input = circular_conv_fft(images, kernels)
    output = alpha * images + (1 - alpha) * blurred_input

    return output.clamp(images.min(), images.max())


class ConvOperatorFFTHelper:
    @staticmethod
    def _prepare_tensor_mul_between(mul_between: th.Tensor) -> None:
        """
        Auxiliary method, which checks dimensions of tensor passed as mul_between to get_circulant_abs_squared_eigvals
        and casts it to the desired shape.
        :param mul_between: tensor to check, should be of shape [B, C_out, 1, 1]
        """
        assert mul_between.dim() in (4, 5)
        if mul_between.dim() == 4:
            mul_between = mul_between.unsqueeze(2)
        h, w = mul_between.shape[-2:]
        assert h == 1
        assert w == 1
        return mul_between

    def get_circulant_eigvals(self, shape_of_incoming_vector: Tuple[int]) -> Union[th.Tensor, float]:
        """
        This method returns eigenvalues, considering that conv operation is performed with periodic
        boundaries.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :return: eigenvalues of operator considering periodic boundary conditions for conv
        """
        psf = self.kernel_unified_shape
        if psf is None:
            return 1.
        assert psf.dim() == 5
        eigvals = psf2otf(psf, shape_of_incoming_vector)
        return eigvals

    def get_circulant_eigvals_doublesided(self, shape_of_incoming_vector: Tuple[int]) -> Union[th.Tensor, float]:
        """
        This method returns eigenvalues, considering that conv operation is performed with periodic
        boundaries.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :return: eigenvalues of operator considering periodic boundary conditions for conv
        """
        psf = self.kernel_unified_shape
        if psf is None:
            return 1.
        assert psf.dim() == 5
        eigvals = psf2otf_doublesided(psf, shape_of_incoming_vector)
        return eigvals

    def get_circulant_abs_squared_eigvals(self, shape_of_incoming_vector: Tuple[int],
                                          mul_between: Union[float, th.Tensor] = 1.) -> Union[th.Tensor, float]:
        """
        This method returns squared absolute eigenvalues, considering that conv operation is performed with periodic
        boundaries. This can be used as eigenvalues of circulant approximation to transpose_apply operator.
        More precisely processing is going in the following way:
        \sum_{i=1}^{C_{out}} |otf|_i^2 * mul_between_i, i.e. otf^T mul_between otf, where otf - eigenvalues of
        circulant convolution with the same kernel, as stored in operator.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :param mul_between: multiplier between transpose conv and conv operators which scales responses of each
                            output channel. May be either single float or tensor of shape [B, C_out, 1, 1]
        :return: squared absolute eigenvalues of operator considering periodic boundary conditions for conv
        """
        if self.kernel_unified_shape is None:
            return mul_between
        eigvals = self.get_circulant_eigvals(shape_of_incoming_vector)
        eigvals = eigvals.abs().pow_(2)  # [1, C_out, C_in, h, w]
        if isinstance(mul_between, float):
            eigvals *= mul_between  # [1, C_out, C_in, h, w]
        elif isinstance(mul_between, th.Tensor):
            mul_between = self._prepare_tensor_mul_between(mul_between)
            eigvals = eigvals * mul_between  # [B, C_out, C_in, h, w]
        else:
            raise ValueError('Expected argument mul_between to be either float or th.Tensor, '
                             f'received {mul_between.__class__} object.')
        eigvals = eigvals.sum(dim=1)  # [B, C_in, h, w]
        return eigvals

    @staticmethod
    def check_pad_suitable_for_circulant_approx(pad_type: Union[str, type(None)]):
        assert pad_type in ('periodic', None), \
            'Circulant approximation is implemented either for valid convolution or for convolution with periodic ' \
            f'boundaries. Given f{pad_type} boundaries type.'

    def _rate_of_ones_on_diag_between(self, h_in: int, w_in: int) -> float:
        if self.pad_operator.pad_mode == 'periodic' or self.kernel_unified_shape is None:
            ones_rate = 1.0
        else:
            top, bot, left, right = self._compute_padding_size(self.kernel_unified_shape.shape)
            ones_rate = (h_in - top - bot) * (w_in - left - right) / (h_in * w_in)
        return ones_rate

    def eigvals_of_transpose_apply_circulant_approx(self, shape_of_incoming_vector: Tuple[int],
                                                    diagonal_vector_between: Optional[th.Tensor] = None
                                                    ) -> Union[th.Tensor, float]:
        """
        This method returns eigenvalues of circulant approximation to transpose_apply operator.
        In general convolution operator can be represented as K = CHP, where P - padding operator, CH - 'valid'
        convolution, for which H = F^H diag(a) F - cicrulant convolution with eigenvalues a, C - cropping operator.
        In this case for some diagonal matrix B=diag(b) .transpose_apply has the form: K^TBK = P^TH^TC^T B CHP.

        Three choices are available:
        1. If P - periodic padding, then K is already circulant, and its circulant approx is the same matrix. Then:
           K^T diag(b) K = F^H diag(a^*) F diag(b) F^H diag(a) F \approx F^H diag(a^*) F mean(b)I F^H diag(a) F =
           = F^H diag(mean(b) * |a|^2) F, so eigenvalues of circulant approximation are mean(b) * |a|^2.
        2. If P - identity operator, then K is a valid convolution and:
           K^T diag(b) K = F^H diag(a^*) F C^T diag(b) C F^H diag(a) F \approx
           \approx F^H diag(a^*) F mean([0_n1 b^T 0_n2])I F^H diag(a) F = F^H diag(mean([0_n1 b^T 0_n2]) * |a|^2) F,
           so eigenvalues of circulant approximation are mean([0_n1 b^T 0_n2]) * |a|^2. Here n1 and n2 are amount of
           pixels cropped from each border of a vector, hence n = n1 + n2 is a total amount of pixels discarded by crop
           operator C. mean([0_n1 b^T 0_n2]) = sum(b) / (n + dim(b)).
        3. P - any other padding operator (zero padding or symmetric padding), then straightforward circulant
           approximation for K^T diag(b) K does not exist.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :param diagonal_vector_between: diagonal of diagonal operator, which stays between operator and its transpose;
                                        by default using diagonal of identity operator, hence vector consisting of ones
        :return: eigenvalues of transpose_apply approximated with circulant matrix
        """
        self.check_pad_suitable_for_circulant_approx(self.pad_operator.pad_mode)
        h, w = shape_of_incoming_vector[-2:]
        if diagonal_vector_between is not None:
            eigvals_mul_between = diagonal_vector_between.sum(dim=(-1, -2), keepdim=True) / (h * w)
        else:
            eigvals_mul_between = self._rate_of_ones_on_diag_between(h, w)
        eigvals = self.get_circulant_abs_squared_eigvals(shape_of_incoming_vector, mul_between=eigvals_mul_between)
        return eigvals

    @staticmethod
    def get_circulant_abs_squared_eigvals_with_kernels(kernel: th.Tensor,
                                                       shape_of_incoming_vector: Tuple[int]) -> th.Tensor:
        """
        This method returns squared absolute eigenvalues, considering that conv operation is performed with periodic
        boundaries.

        :param shape_of_incoming_vector: output shape of eigenvalues, which is the same as incoming vector
        :return: squared absolute eigenvalues of operator considering periodic boundary conditions for conv
        """
        eigvals = psf2otf(kernel, shape_of_incoming_vector)
        eigvals = eigvals.abs().pow_(2)  # [1, C_out, C_in, h, w]
        eigvals = eigvals.sum(dim=1)  # [?, C_in, h, w]
        return eigvals