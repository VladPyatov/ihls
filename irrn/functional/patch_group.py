import warnings
from typing import Tuple, Union, Optional, Sequence

import torch as th

from irrn.utils import deep_detach

try:
    from nlop_cpu import group_transpose
    from nlop_cuda import group_transpose as group_transpose_cuda
except (ModuleNotFoundError, ImportError):
    warnings.warn('You do not have nlop properly compiled. Consider compiling/recompiling it from '
                  'irrn/extensions/nlop. Continuing run in reduced functionality mode.')

from torch.autograd import Function


def idx_to_tuple(idx: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Auxilary method which converts index or indices to a unified format of 2 element tuple.

    :param idx: input index or indices to convert
    :return: two elements tuple with indices
    """
    if isinstance(idx, int):
        idx = (idx, idx)
    else:
        assert len(idx) == 2
        assert isinstance(idx[0], int)
        assert isinstance(idx[1], int)
    return idx


def patch_extract_transform(images: th.Tensor, kernel: Union[th.Tensor, type(None)],
                            patch_size: Tuple[int, int], stride: Tuple[int, int]) -> th.Tensor:
    """
    This function performs patches transformation and extraction.

    :param images: a 4-D tensor of shape [B, C, H, W]
    :param kernel: tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] that is used to transform
           the extracted patches which are of shape [C, patch_size[0], patch_size[1]]. patch_num_features is the number
           of coefficients in the transform domain. If the kernel is set to None then no transformation of the patches
           takes place and the patch elements are equal to patch_numel = C*patch_size[0]*patch_size[1].
    :param patch_size: indicates the height and width of the extracted patches from the input images.
    :param stride: indicates the overlap in the y, x-axis of the extracted patches.
    :return: tensor of shape [B, patch_num_features, n_h, n_w] which holds the batch of transformed and extracted
             patches from images batch.
    """
    assert images.ndim == 4, "A 4-d tensor is expected."
    patch_size = idx_to_tuple(patch_size)
    stride = idx_to_tuple(stride)

    if kernel is not None:
        assert kernel.ndim == 4 or kernel.ndim == 5, "Invalid dimensions for input 'kernel'."
        assert kernel.shape[-2:] == th.Size(patch_size), "Invalid dimensions for input 'kernel'."
        if kernel.dim() == 4:
            kernel_c_in = kernel.size(1)
        else:
            assert images.shape[0] == kernel.shape[0]
            kernel_c_in = kernel.size(2)
        assert kernel_c_in in (1, images.size(1)), "Invalid dimensions for input 'kernel'."

    patch_numel = patch_size[0] * patch_size[1]
    img_num_channels = images.shape[1]
    # Patch extraction and transform.
    if kernel is None:
        kernel = th.eye(patch_numel).type_as(images)
        kernel = kernel.view(patch_numel, 1, patch_size[0], patch_size[1])
        kernel = kernel.repeat(img_num_channels, 1, 1, 1)
        groups = img_num_channels
        pf = th.conv2d(images, kernel, stride=stride, groups=groups)
    elif kernel.dim() == 4:
        groups = 1
        if kernel.size(1) == 1:
            assert kernel.size(0) % img_num_channels == 0, "Dimensions mismatch between input images and kernel."
            groups = img_num_channels
        else:
            assert kernel.size(1) == img_num_channels, "Dimensions mismatch between input images and kernel."
        pf = th.conv2d(images, kernel, stride=stride, groups=groups)
    else:
        b, c_out, c_in = kernel.shape[:3]
        if c_in == 1:
            kernel = kernel.expand(-1, -1, images.shape[1], -1, -1)
        images = images.flatten(start_dim=0, end_dim=1).unsqueeze(0)
        kernel = kernel.flatten(start_dim=0, end_dim=1)
        pf = th.conv2d(images, kernel, stride=stride, groups=b)
        pf = pf.view(b, c_out, *pf.shape[-2:])
    return pf


def check_sizes(images_size: Sequence[int], indices_size: Sequence[int], patch_size: Sequence[int],
                stride: Sequence[int]) -> None:
    """
    This function checks whether sizes of input images and indices correspond to input patch size and stride.

    :param images_size: size of input images
    :param indices_size: size of indices with coords for grouping
    :param patch_size: size of patches to extract
    :param stride: stride, which is used for patches extraction
    :return: Nothing
    """
    n_h = max((images_size[-2] - patch_size[0]) // stride[0] + 1, 0)
    n_w = max((images_size[-1] - patch_size[1]) // stride[1] + 1, 0)
    assert indices_size[-2:] == th.Size([n_h, n_w]), "Dimensions mismatch between inputs 'indices', 'patch_size' " \
                                                     "and 'stride'."


def patch_group(input_patches: th.Tensor, indices: th.Tensor) -> th.Tensor:
    """
    This function performs grouping of input patches based on given indices representing centers of each patch
    in a group.

    :param input_patches: tensor of shape [B, patch_num_features, n_h, n_w] which holds patches to group
    :param indices: tensor of shape [B, nbrs, n_h, n_w] (typically output of patch match), which holds the coordinates
           of the centers of the valid patches extracted from images (in a single index format).
    :return: tensor of shape [B, patch_num_features, nbrs, n_h, n_w] which holds the patch groups for each patch in the
             image batch.
    """
    input_patches = input_patches.view(input_patches.shape[0:-2] + (-1,))
    batch, nbrs, n_h, n_w = indices.shape
    patch_num_features = input_patches.size(1)
    # Patch Grouping
    indices = indices.view(batch, nbrs, -1)
    # indices holds the single index coordinates of the center of each patch
    # for every image in the batch. Those indices lie in the range
    # (0, patch_num-1). In order to transform these coordinates to account for
    # all the elements of the patch and the batch dimension we need to add the
    # correct offset which will allow us to do the patch grouping.
    batch_offset = patch_num_features*n_h*n_w*th.arange(0, batch, dtype=th.int64, device=input_patches.device)
    batch_offset = batch_offset.view(batch, 1, 1, 1)
    patch_offset = n_h*n_w*th.arange(0, patch_num_features, dtype=th.int64, device=input_patches.device)
    patch_offset = patch_offset.view(1, patch_num_features, 1, 1)
    idx = batch_offset+patch_offset+indices.reshape(batch, 1, nbrs, -1)
    groups = input_patches.take(idx).view(batch, patch_num_features, nbrs, n_h, n_w)
    return groups


def patch_group_transpose(groups: th.Tensor, indices: th.Tensor) -> th.Tensor:
    """
    This function performs transpose operation to patches grouping based on given indices representing centers of each
    patch in a group.

    :param groups: tensor of shape [B, patch_num_features, nbrs, n_h, n_w] which holds the groups for each patch in the
           images batch.
    :param indices: tensor of shape [B, nbrs, n_h, n_w] (typically output of patch match), which holds the coordinates
           of the centers of the valid patches extracted from images (in a single index format).
    :returns: tensor of shape [B, patch_num_features, n_h, n_w] with patches put back according to their indices
              as a result of patch group transpose operation.
    """
    batch, pn, nbrs, n_h, n_w = groups.shape
    if groups.is_cuda:
        groups = group_transpose_cuda(groups.view(groups.shape[0:-2]+(-1,)).contiguous(),
                                      indices.view(batch, nbrs, -1).contiguous())
    else:
        groups = group_transpose(groups.view(groups.shape[0:-2]+(-1,)).contiguous(),
                                 indices.view(batch, nbrs, -1).contiguous())
    groups = groups.view(groups.shape[0:-1]+(n_h, n_w))
    return groups


def patch_extract_transform_transpose(groups: th.Tensor, num_channels: int,
                                      patch_size: Optional[Tuple[int, int]] = (8, 8),
                                      stride: Optional[Tuple[int, int]] = (1, 1),
                                      kernel: Optional[th.Tensor] = None,
                                      output_size: Optional[Union[int, Tuple[int, int]]] = None) -> th.Tensor:
    """
    This function performs transpose operation to patches transformation and extraction.

    :param groups: input of shape [B, patch_num_features, n_h, n_w] to perform transpose operation
    :param num_channels: specifies the number of channels of the output.
    :param patch_size: indicates the height and width of the extracted patches from the input images.
    :param stride: indicates the overlap in the y, x-axis of the extracted patches.
    :param kernel: tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] that is used to transform
           the extracted patches which are of shape [C, patch_size[0], patch_size[1]]. patch_num_features is the number
           of coefficients in the transform domain. If the kernel is set to None then no transformation of the patches
           takes place and the patch elements are equal to patch_numel = C*patch_size[0]*patch_size[1].
    :param output_size: if specified, output tensor is extended to this size
    :returns: tensor of shape [B, C, H, W] as a result of patch transformation and extraction transpose operation.
    """
    batch, pn, n_h, n_w = groups.shape
    pn //= num_channels

    if kernel is not None:
        assert kernel.ndim == 4 or kernel.ndim == 5, "Invalid dimensions for input 'kernel'."
        assert kernel.shape[-2:] == th.Size(patch_size), "Invalid dimensions for input 'kernel'."
        if kernel.dim() == 4:
            kernel_c_in = kernel.size(1)
        else:
            assert groups.shape[0] == kernel.shape[0]
            kernel_c_in = kernel.size(2)
        assert kernel_c_in in (1, num_channels), "Invalid dimensions for input 'kernel'."

    if output_size is not None:
        h_out, w_out = idx_to_tuple(output_size)
        h_conv_out = (n_h - 1) * stride[0] + (patch_size[-2] - 1) + 1
        w_conv_out = (n_w - 1) * stride[1] + (patch_size[-1] - 1) + 1
        assert h_conv_out <= h_out
        assert w_conv_out <= w_out
        pad_size_out = (h_out - h_conv_out, w_out - w_conv_out)
    else:
        pad_size_out = (0, 0)

    if kernel is None:
        kernel = th.eye(patch_size[0]*patch_size[1]).type_as(groups)
        kernel = kernel.view(pn, 1, patch_size[0], patch_size[1])
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        num_conv_groups = num_channels
        y = th.conv_transpose2d(groups, kernel, stride=stride, groups=num_conv_groups,
                                output_padding=pad_size_out)
    elif kernel.dim() == 4:
        num_conv_groups = 1
        if kernel.size(1) == 1:
            assert kernel.size(0) % num_channels == 0, "Dimensions mismatch between input image 'groups' and kernel."
            num_conv_groups = num_channels
        else:
            assert kernel.size(1) == num_channels, "Dimensions mismatch between input image 'groups' and kernel."
        y = th.conv_transpose2d(groups, kernel, stride=stride, groups=num_conv_groups,
                                output_padding=pad_size_out)
    else:
        b, _, c_in = kernel.shape[:3]
        if c_in == 1:
            kernel = kernel.expand(-1, -1, num_channels, -1, -1)
        y = th.conv_transpose2d(
            groups.flatten(start_dim=0, end_dim=1).unsqueeze(0), kernel.flatten(start_dim=0, end_dim=1),
            groups=b, stride=stride, output_padding=pad_size_out)
        y = y.view(b, num_channels, *y.shape[-2:])
    return y


class PatchGroupTransform(Function):
    """
    This class implements patches extraction and grouping based on given indices representing centers of each patch in
    a group and a backward pass of this operation w.r.t. input tensor. Since such operation is
    linear w.r.t. input, its gradient is implemented by patch group transform transpose.
    """
    @staticmethod
    def forward(ctx, images: th.Tensor, kernel: Optional[th.Tensor], indices: th.Tensor,
                patch_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]) -> th.Tensor:
        """
        Implements forward pass through patches extraction and grouping operation.

        :param images: a 4-D tensor of shape [B, C, H, W]
        :param indices: tensor of shape [B, nbrs, n_h, n_w] (typically output of patch match), which holds the
               coordinates of the centers of the valid patches extracted from images (in a single index format).
        :param kernel: tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] that is used to transform
               the extracted patches which are of shape [C, patch_size[0], patch_size[1]]. patch_num_features is the
               number of coefficients in the transform domain. If the kernel is set to None then no transformation of
               the patches takes place and the patch elements are equal to patch_numel = C*patch_size[0]*patch_size[1].
        :param patch_size: indicates the height and width of the extracted patches from the input images.
        :param stride: indicates the overlap in the y, x-axis of the extracted patches.
        :return: tensor of shape [B, patch_num_features, nbrs, n_h, n_w] which holds the groups for each patch in the
                 images batch.
        """
        patch_size = idx_to_tuple(patch_size)
        stride = idx_to_tuple(stride)
        check_sizes(images.shape, indices.shape, patch_size, stride)

        if ctx.needs_input_grad[1]:
            images_detached, kernel_detached = deep_detach((images, kernel))
            with th.enable_grad():
                patches = patch_extract_transform(images_detached, kernel_detached, patch_size, stride)
                ctx.patches = patches
                ctx.kernel = kernel_detached
        else:
            patches = patch_extract_transform(images, kernel, patch_size, stride)
        ret = patch_group(patches, indices)

        if any(ctx.needs_input_grad):
            ctx.patch_size = patch_size
            ctx.stride = stride
            ctx.input_shape = images.shape
            # gradient is independent of argument 'images', so there is no need to store and track inplace modifications
            ctx.save_for_backward(indices, kernel)
        return ret

    @staticmethod
    def backward(ctx, grad_output: th.Tensor) -> Tuple[Union[th.Tensor, type(None)]]:
        """
        Implements backward pass through patches extraction and grouping operation, i.e. its transpose.

        :param grad_output: vector of shape [B, patch_num_features, nbrs, n_h, n_w] to be multiplied by gradient matrix
        :return: tensor of shape [B, C, H, W] which represents gradient of patch group transform operation w.r.t. its
                 input tensor;
                 tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] which represents gradient of
                 patch group transform operation w.r.t. transformation kernel if kernel is given, either None;
                 gradients w.r.t. the rest arguments (including indices) are not computed and returned as None
        """
        indices, kernel = ctx.saved_tensors
        grad_images = grad_kernel = grad_indices = None
        ret = patch_group_transpose(grad_output, indices)
        if ctx.needs_input_grad[1]:
            grad_kernel = th.autograd.grad(ctx.patches, ctx.kernel, grad_outputs=ret, retain_graph=False,
                                           create_graph=False, only_inputs=True, allow_unused=False)[0]
        if ctx.needs_input_grad[0]:
            grad_images = patch_extract_transform_transpose(ret, ctx.input_shape[1], ctx.patch_size, ctx.stride, kernel,
                                                            ctx.input_shape[-2:])
        assert not any(ctx.needs_input_grad[2:])
        return grad_images, grad_kernel, grad_indices, None, None


class PatchGroupTransformTranspose(Function):
    """
    This class performs transpose patches extraction and grouping based on given indices representing
    centers of each patch in a group and a backward pass of this operation w.r.t. input tensor. Since such operation is
    linear w.r.t. input, its gradient is implemented by patch group transform.
    """
    @staticmethod
    def forward(ctx, groups: th.Tensor, kernel: Optional[th.Tensor], indices: th.Tensor, num_channels: int,
                patch_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]],
                output_size: Optional[Union[int, Tuple[int, int]]]) -> th.Tensor:
        """"
        Implements forward pass through transpose patches extraction and grouping operation.

        :param groups: tensor of shape [B, patch_num_features, nbrs, n_h, n_w] which holds the groups for each patch in
               the images batch.
        :param kernel: tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] that is used to transform
               the extracted patches which are of shape [C, patch_size[0], patch_size[1]]. patch_num_features is the
               number of coefficients in the transform domain. If the kernel is set to None then no transformation of
               the patches takes place and the patch elements are equal to patch_numel = C*patch_size[0]*patch_size[1].
        :param num_channels: specifies the number of channels of the output.
        :param indices: tensor of shape [B, nbrs, n_h, n_w] (typically output of patch match), which holds the
               coordinates of the centers of the valid patches extracted from images (in a single index format).
        :param patch_size: indicates the height and width of the extracted patches from the input images.
        :param stride: indicates the overlap in the y, x-axis of the extracted patches.
        :param output_size: if specified, output tensor is extended to this size
        :returns: tensor of shape [B, C, H, W] as a result of patch group transpose operation.
        """
        patch_size = idx_to_tuple(patch_size)
        stride = idx_to_tuple(stride)
        if output_size is not None:
            check_sizes(output_size, indices.shape, patch_size, stride)
        patches = patch_group_transpose(groups, indices)
        if ctx.needs_input_grad[1]:
            with th.enable_grad():
                kernel_detached = kernel.detach().requires_grad_(True)
                output = patch_extract_transform_transpose(patches, num_channels, patch_size=patch_size, stride=stride,
                                                           kernel=kernel_detached, output_size=output_size)
                output_to_save = output
                output = output.detach()
                ctx.kernel = kernel_detached
        else:
            output = patch_extract_transform_transpose(patches, num_channels, patch_size=patch_size, stride=stride,
                                                       kernel=kernel, output_size=output_size)
            output_to_save = None
        if any(ctx.needs_input_grad):
            ctx.patch_size = patch_size
            ctx.stride = stride
            # gradient is independent of argument 'groups', so there is no need to store and track inplace modifications
            ctx.save_for_backward(indices, kernel, output_to_save)
        return output

    @staticmethod
    def backward(ctx, grad_output: th.Tensor) -> Tuple[Union[th.Tensor, type(None)]]:
        """
        Implements backward pass through transpose patches extraction and grouping operation, i.e. its transpose.

        :param grad_output: vector of shape [B, C, H, W] to be multiplied by gradient matrix
        :return: tensor of shape [B, patch_num_features, nbrs, n_h, n_w] which represents gradient of patch group
                 transform transpose operation w.r.t. its input tensor;
                 tensor of shape [patch_num_features, C|1, patch_size[0], patch_size[1]] which represents gradient of
                 patch group transform operation w.r.t. transformation kernel if kernel is given, either None;
                 gradients w.r.t. the rest arguments (including indices) are not computed and returned as None
        """
        indices, kernel, output = ctx.saved_tensors
        grad_groups = grad_kernel = grad_indices = None
        if ctx.needs_input_grad[0]:
            grad_groups = patch_extract_transform(grad_output, kernel, ctx.patch_size, ctx.stride)
            grad_groups = patch_group(grad_groups, indices)
        if ctx.needs_input_grad[1]:
            grad_kernel = th.autograd.grad(output, ctx.kernel, grad_outputs=grad_output, retain_graph=False,
                                           create_graph=False, only_inputs=True, allow_unused=False)[0]
        assert not any(ctx.needs_input_grad[2:])
        return grad_groups, grad_kernel, grad_indices, None, None, None, None
