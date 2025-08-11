from math import ceil
from typing import List, Tuple, Union, Optional, Sequence, Callable

import numpy as np
import torch as th
from torch import nn

from irrn.functional.patch_group import PatchGroupTransform, PatchGroupTransformTranspose, idx_to_tuple
from irrn.modules.backbones import KPBackbone
from . import LinearOperatorBase, Pad2DOperator, LearnableConvolutionOperator, LearnableCNNOperator, \
    LearnableKPNConvOperator


class PatchGroupOperator(LinearOperatorBase):
    group_indices: Optional[th.Tensor] = None
    pad_operator: Pad2DOperator = None
    effective_kernel: type(None) = None
    """
    This class implements patch grouping operator with padding.
    Indices for patches extraction and grouping are computed using knn search from either observation, or
    latent image.
    """
    def __init__(self, group_size: int = 5, patch_size: Union[int, Sequence[int]] = 8,
                 search_window: Union[int, Tuple[int, int]] = 15, exclude_from_search: Union[int, Tuple[int, int]] = 0,
                 distance_weights: Optional[th.Tensor] = None, distance_type: str = 'euclidean',
                 stride: Union[int, Tuple[int, int]] = 1, padding_mode: str = None,
                 observation_keyword: Optional[str] = None, latent_index: Optional[int] = None) -> None:
        """
        Initializing operator parameters.

        :param group_size: amount of closest patches to use for extraction and grouping
        :param patch_size: size of patches to use for extraction and grouping
        :param search_window: region to perform knn patches search
        :param exclude_from_search: which indices in window to exclude from knn search
        :param distance_weights: symmetric weights to calculate weighted distance
        :param distance_type: 'euclidean' or 'abs' indicates the type of the patch distance
        :param stride: step-sizes in x- and y-directions
        :param padding_mode: which padding to use in convolution 'zero', 'symmetric', 'periodic', None padding types
               are supported. if None - no padding is applied.
        :param observation_keyword: name of keyword argument which should be used for knn search;
               this argument is used if one wants to perform search based on observation
        :param latent_index: index to choose the specific latent tensor to be used for knn search;
               this argument is used if one wants to perform search based on some latent variable
        """
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

        self.patch_size = idx_to_tuple(patch_size)
        self.num_out_channels = self.patch_numel = self.patch_size[0] * self.patch_size[1]
        self.stride = idx_to_tuple(stride)
        self.search_window = idx_to_tuple(search_window)
        self.exclude_from_search = idx_to_tuple(exclude_from_search)
        max_group_size = ceil(self.search_window[0] / self.stride[0]) * ceil(self.search_window[1] / self.stride[1])
        assert group_size < max_group_size, f'For selected parameters group size cannot be more than {max_group_size}.'
        self.group_size = group_size
        self.distance_weights = distance_weights
        self.distance_type = distance_type
        if self.pad_operator is None:
            padding_size = LearnableConvolutionOperator._compute_padding_size(self.patch_size)
            self.pad_operator = Pad2DOperator(padding_size, padding_mode)

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        This method performs batched linear patches extraction and grouping of input vector
        followed by convolution with a filterbank.

        :param vector: input vector of shape [B, C, H_in, W_in] to extract and group patches from
        :return: vector of shape [B, self.patch_numel, self.group_size, H_out, W_out]
        """
        return PatchGroupTransform.apply(self.pad_operator(vector), self.effective_kernel,
                                         self.group_indices, self.patch_size, self.stride)

    def _transpose(self, vector: th.Tensor, output_shape: Union[Tuple[int], List[int]], *args, **kwargs) -> th.Tensor:
        """
        This method performs batched transpose patches extraction and grouping of input vector
        followed by transpose convolution with a filterbank.

        :param vector: input vector of shape [B, self.patch_numel, self.group_size, H_in, W_in]
        :return: vector of shape [B, C, H_out, W_out]
        """
        assert len(output_shape) >= 3
        return self.pad_operator.T(PatchGroupTransformTranspose.apply(
            vector, self.effective_kernel, self.group_indices, output_shape[-3], self.patch_size, self.stride,
            output_shape[-2:]))

    def transpose_apply(self, vector: th.Tensor, *args, operator_between: Callable = None, **kwargs):
        """
        This method performs linear transformation followed by transpose transformation with the same operator.

        :param vector: input vector of shape [B, C, H_in, W_in] to apply transformation
        :param operator_between: operator, which should be applied between transformation and its transpose
        :param args, kwargs: altering parameters to parametrize operator
        :return: vector of shape [B, C, H_in, W_in], transformed by linear operator and its transpose
        """
        output_shape = self.pad_operator.output_shape(vector)
        if operator_between:
            ret = self._transpose(operator_between(self.apply(vector, *args, **kwargs)), output_shape, *args, **kwargs)
        else:
            ret = self._transpose(self.apply(vector, *args, **kwargs), output_shape, *args, **kwargs)
        return ret

    def update_group_indices(self, tensor: th.Tensor) -> None:
        """
        This method updates grouping indices based on KNN search of patches in input tensor.

        :param tensor: input tensor of shape [B, C, H, W] to perform KNN search
        :return:
        """
        img_padded = self.pad_operator(tensor)
        knn_idx, knn_dist = self.knn_patch(img_padded)
        n_h = max((img_padded.size(-2) - self.patch_size[0]) // self.stride[0] + 1, 0)
        n_w = max((img_padded.size(-1) - self.patch_size[1]) // self.stride[1] + 1, 0)

        px_ = ((self.patch_size[0] + 1) // 2) - 1
        py_ = ((self.patch_size[1] + 1) // 2) - 1

        knn_idx = (knn_idx // img_padded.size(-1) - px_) // self.stride[0] * n_w + \
                  (knn_idx % img_padded.size(-1) - py_) // self.stride[1]
        self.group_indices = knn_idx.view(knn_idx.size(0), self.group_size, n_h, n_w)

    def prepare_for_restoration(self, **kwargs) -> 'LearnableConvPatchGroupLinearOperator':
        """
        This method prepares patch grouping operator for recurrent restoration by predicting grouping indices from
        observation.

        :param kwargs: parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        if self.observation_keyword is None:
            self.group_indices = None
        else:
            assert self.observation_keyword in kwargs.keys()
            self.update_group_indices(kwargs[self.observation_keyword])
        return self

    def prepare_for_step(self, step_idx: int, *tensor_params: th.Tensor, **kwargs
                         ) -> 'LearnableConvPatchGroupLinearOperator':
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
            self.update_group_indices(tensor)
        return self

    def knn_patch(self, img: th.Tensor, sorted_output: bool = True):
        """
        This method performs search for closest patches in input image.

        :param img: input image to perform patch nn search
        :param sorted_output: whether to sort output based on distance value
        :return: indices of found nearest patches and distances for them
        """
        while img.dim() < 4:
            img = img.unsqueeze(0)

        const = img.view(img.shape[0:-2] + (-1,)).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        img = img.div(const)  # normalize the values to avoid overflow

        if self.distance_weights is None:
            distance_weights = img.new_ones(1, img.size(1), self.patch_size[0], self.patch_size[1])
            distance_weights = distance_weights.div(np.prod(self.patch_size) * img.size(1))
        else:
            distance_weights = self.distance_weights

        assert distance_weights.shape == (1, img.size(1), self.patch_size[0], self.patch_size[1]), \
            "The weight kernels have invalid dimensions."

        n_h = max((img.size(2) - distance_weights.size(2)) // self.stride[0] + 1, 0)
        n_w = max((img.size(3) - distance_weights.size(3)) // self.stride[1] + 1, 0)
        patch_numel = n_h * n_w
        batch = img.size(0)

        num_neighbours = self.group_size - 1
        knn = th.zeros(batch, num_neighbours, patch_numel, dtype=th.int64, device=img.device)
        knn_dist = img.new_zeros(batch, num_neighbours, patch_numel)

        patch_center = ((self.patch_size[0] + 1) // 2 - 1, (self.patch_size[1] + 1) // 2 - 1)
        # vector of size equal to patch_numel in which we store the image coordinates
        # of the centers of the valid image patches.
        center_coords = th.arange(0, img.size(2) * img.size(3), dtype=th.int64, device=img.device)
        center_coords = center_coords.reshape(img.size(2), img.size(3))
        center_coords = center_coords[
                        patch_center[0]:img.size(2) + 1 - (self.patch_size[0] - patch_center[0]): self.stride[0],
                        patch_center[1]:img.size(3) + 1 - (self.patch_size[1] - patch_center[1]):self.stride[1]]
        center_coords = center_coords.flatten()

        # if Nbrs == 1 we don't need to perform any search
        if num_neighbours == 0:
            knn = center_coords.repeat(batch, 1, 1)
            return knn, knn_dist

        tmp = th.arange(0, self.search_window[0] + 1, self.stride[0], dtype=th.int32, device=img.device)
        tmp = tmp[1:]
        offsets_x = th.cat((-tmp.flip(0), tmp.new_tensor([0]), tmp), dim=0)

        tmp = th.arange(0, self.search_window[1] + 1, self.stride[1], dtype=th.int32, device=img.device)
        tmp = tmp[1:]
        offsets_y = th.cat((-tmp.flip(0), tmp.new_tensor([0]), tmp), dim=0)
        del tmp

        index = th.arange(0, batch, dtype=th.int64, device=img.device).view(batch, 1, 1)*num_neighbours*patch_numel + \
                th.arange(0, patch_numel, dtype=th.int64, device=img.device).view(1, 1, -1)
        ctr = 0
        for kx in offsets_x:
            for ky in offsets_y:
                if kx == 0 and ky == 0:  # We don't check the (0,0)-offset from
                    # the patch center since in this case each patch is compared to
                    # itself and the distance should be zero. We add this weight at
                    # the end. (This is why we redefine the Nbrs as Nbrs-1.)
                    continue
                elif abs(kx) <= self.exclude_from_search[0] and \
                        abs(ky) <= self.exclude_from_search[1]:
                    continue
                else:
                    # img_s = shift_inf(f,-(kx,ky)); # img_s[m][n]=img[m+kx, n+ky]
                    # Note that we use the shift function with inf boundary
                    # conditions so that the comparison between valid and non-valid
                    # patches (those which do not exist in the image domain) to
                    # give us an inf distance measure.
                    if self.distance_type == "euclidean":
                        E = (img - self.shift_inf(img, (0, 0, -kx.item(), -ky.item()))).pow(2)
                    elif self.distance_type == 'abs':
                        E = th.abs(img - self.shift_inf(img, (0, 0, -kx.item(), -ky.item())))
                    else:
                        raise NotImplementedError("Unknown distance.")

                    dist = th.nn.functional.conv2d(E, distance_weights, stride=self.stride)
                    dist = dist.reshape(img.size(0), patch_numel)

                    if ctr < num_neighbours:
                        knn[:, ctr, :] = center_coords + kx * img.size(3) + ky
                        # If we consider the offset (kx,ky) then the spatial
                        # coordinates are (i+kx,j+ky) = (center_coords//W+kx,center_coords%W+ky) where
                        # W=img.size(3). In single index notation this becomes :
                        # ind=(center_coords//W+kx)*W+(center_coords%W)+ky =
                        # (center_coords//W)*W + center_coords%W + kx*W+ky

                        # Note that center_coords = i*W+j => center_coords//W = i and center_coords%W = j. Therefore
                        # (center_coords//W)*W+(center_coords%W) = center_coords. Consequently:
                        # ind = center_coords + kx*W + ky
                        knn_dist[:, ctr, :] = dist
                    else:
                        max_dist, idx = knn_dist.max(dim=1)
                        idx = index + idx.view(batch, 1, patch_numel) * patch_numel
                        idx.squeeze_(1)
                        # Find which patches have a distance larger than the
                        # distance computed for the current offset
                        mask = (max_dist > dist)
                        # Replace those distances which are larger than the
                        # current ones
                        knn_dist.put_(idx[mask], dist[mask])

                        # Now we also need to correctly change the corresponding
                        # coordinates
                        R = (center_coords + kx * img.size(3) + ky).repeat(batch, 1)
                        knn.put_(idx[mask], R[mask])
                    ctr += 1

        # Finally we add the distances for the (0,0)-offset and the corresponding
        # coordinates
        knn_dist = th.cat((knn_dist.new_zeros(batch, 1, patch_numel), knn_dist), dim=1)
        knn = th.cat((center_coords.repeat(batch, 1, 1), knn), dim=1)

        if sorted_output:
            knn_dist, idx = th.sort(knn_dist, dim=1)
            idx = (th.arange(0, batch, dtype=th.int64, device=img.device).view(
                batch, 1, 1) * (num_neighbours + 1) + idx) * patch_numel + th.arange(
                0, patch_numel, dtype=th.int64, device=img.device).view(
                1, 1, patch_numel)
            knn = knn.take(idx)

        return knn, knn_dist

    @staticmethod
    def shift_inf(x: th.Tensor, s: Sequence[int]) -> th.Tensor:
        """
        Shift operator that assumes constant infinite boundary conditions and applies to a tensor of arbitrary
        dimensions. Usage: xs = shift_inf(x, (0, 1, -3, 3)).

        :param x: The tensor to be shifted
        :param s: sequence that matches the dimensions of x, with the corresponding shifts
        :return: input tensor, shifted according to given sequence with constant infinite boundary conditions
        """
        # use a list sequence instead of a tuple since the latter is an
        # immutable sequence and cannot be altered
        indices = [slice(0, x.shape[k]) for k in range(x.dim())]
        xs = x.new_ones(x.shape) * float('inf')
        idx_x = indices[:]
        idx_xs = indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                if s[i] > 0:  # right shift
                    idx_x[i] = slice(0, x.shape[i] - s[i])
                    idx_xs[i] = slice(s[i], x.shape[i])
                else:  # left shift
                    idx_x[i] = slice(-s[i], x.shape[i])
                    idx_xs[i] = slice(0, x.shape[i] + s[i])

        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        return xs

    def to(self, *args, **kwargs) -> 'LearnableConvPatchGroupLinearOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        if self.effective_kernel is not None:
            self.effective_kernel = self.effective_kernel.to(*args, **kwargs)
        if self.group_indices is not None:
            self.group_indices = self.group_indices.to(*args, **kwargs)
        return self


class LearnableConvPatchGroupOperator(PatchGroupOperator, LearnableConvolutionOperator):
    """
    This method implements conv patch grouping operator with padding, where convolution kernel is stored and learned
    directly.
    Indices for patches extraction and grouping are computed using knn search from either observation, or
    latent image.
    """
    def __init__(self, group_size: int = 5, patch_size: int = 8, search_window: Union[int, Tuple[int, int]] = 15,
                 exclude_from_search: Union[int, Tuple[int, int]] = 0, distance_weights: Optional[th.Tensor] = None,
                 distance_type: str = 'euclidean', stride: Union[int, Tuple[int, int]] = 1, padding_mode: str = None,
                 observation_keyword: Optional[str] = None, latent_index: Optional[int] = None,
                 kernel_num_in_channels: int = 1, kernel_num_out_channels: Optional[int] = None,
                 kernel: Optional[th.Tensor] = None, learnable: bool = False) -> None:
        """
        Initializing operator parameters.

        :param group_size: amount of closest patches to use for extraction and grouping
        :param patch_size: size of patches to use for extraction and grouping
        :param search_window: region to perform knn patches search
        :param exclude_from_search: which indices in window to exclude from knn search
        :param distance_weights: symmetric weights to calculate weighted distance
        :param distance_type: 'euclidean' or 'abs' indicates the type of the patch distance
        :param stride: step-sizes in x- and y-directions
        :param padding_mode: which padding to use in convolution 'zero', 'symmetric', 'periodic', None padding types
               are supported. if None - no padding is applied
        :param observation_keyword: name of keyword argument which should be used for knn search;
               this argument is used if one wants to perform search based on observation
        :param latent_index: index to choose the specific latent tensor to be used for knn search;
               this argument is used if one wants to perform search based on some latent variable
        :param kernel_num_in_channels: which number of channels to use in filter bank
               this quantity should be either 1, or number of channels of operator's inputs
        :param kernel_num_out_channels: number of output channels (features) to use in filter bank
        :param kernel: filter bank of shape [N, C, h, w] to initialize operator's filters
            if None, then filters will be initialized based on filter_size and filter_num_in_channels params
        :param learnable: flag, which determines whether to make filters learnable
        """
        LearnableConvolutionOperator.__init__(
            self, filter_size=patch_size, filter_num_in_channels=kernel_num_in_channels,
            filter_num_out_channels=kernel_num_out_channels, padding_mode=padding_mode, filters=kernel,
            mix_in_channels=True, learnable=learnable)
        PatchGroupOperator.__init__(
            self, group_size=group_size, patch_size=self.effective_kernel.shape[-2:], search_window=search_window,
            exclude_from_search=exclude_from_search, distance_weights=distance_weights, distance_type=distance_type,
            stride=stride, padding_mode=padding_mode, observation_keyword=observation_keyword,
            latent_index=latent_index)


class LearnableCNNPatchGroupOperator(PatchGroupOperator, LearnableCNNOperator):
    """
    This method implements conv patch grouping operator with padding, where convolution kernel is implemented a form of
    some linear convolution neural network (CNN).
    Indices for patches extraction and grouping are computed using knn search from either observation, or latent image.
    """
    def __init__(self, backbone: nn.Module, group_size: int = 5, search_window: Union[int, Tuple[int, int]] = 15,
                 exclude_from_search: Union[int, Tuple[int, int]] = 0, distance_weights: Optional[th.Tensor] = None,
                 distance_type: str = 'euclidean', stride: Union[int, Tuple[int, int]] = 1, padding_mode: str = None,
                 observation_keyword: Optional[str] = None, latent_index: Optional[int] = None,
                 learnable: bool = False) -> None:
        """
        Initializing operator parameters.

        :param backbone: linear CNN, implemented as nn.Sequential in .backbone attribute
        :param group_size: amount of closest patches to use for extraction and grouping
        :param search_window: region to perform knn patches search
        :param exclude_from_search: which indices in window to exclude from knn search
        :param distance_weights: symmetric weights to calculate weighted distance
        :param distance_type: 'euclidean' or 'abs' indicates the type of the patch distance
        :param stride: step-sizes in x- and y-directions
        :param padding_mode: which padding to use in convolution 'zero', 'symmetric', 'periodic', None padding types
               are supported. if None - no padding is applied.
        :param observation_keyword: name of keyword argument which should be used for knn search;
               this argument is used if one wants to perform search based on observation
        :param latent_index: index to choose the specific latent tensor to be used for knn search;
               this argument is used if one wants to perform search based on some latent variable
        :param learnable: flag, which determines whether to make filters learnable
        """
        LearnableCNNOperator.__init__(
            self, backbone, padding_mode=padding_mode, learnable=learnable, mix_in_channels=True)
        PatchGroupOperator.__init__(
            self, group_size=group_size, patch_size=self.effective_kernel.shape[-2:], search_window=search_window,
            exclude_from_search=exclude_from_search, distance_weights=distance_weights, distance_type=distance_type,
            stride=stride, padding_mode=padding_mode, observation_keyword=observation_keyword,
            latent_index=latent_index)

    def prepare_for_restoration(self, **kwargs) -> 'LearnableCNNPatchGroupOperator':
        """
        This method prepares patch grouping operator for recurrent restoration by predicting grouping indices from
        observation.

        :param kwargs: parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        super(LearnableCNNPatchGroupOperator, self).prepare_for_restoration(**kwargs)
        self.update_effective_kernel(**kwargs)
        return self

    def to(self, *args, **kwargs) -> 'LearnableCNNPatchGroupOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        if self.group_indices is not None:
            self.group_indices = self.group_indices.to(*args, **kwargs)
        LearnableCNNOperator.to(self, *args, **kwargs)
        return self


class LearnableKPNConvPatchGroupOperator(PatchGroupOperator, LearnableKPNConvOperator):
    """
    This method implements conv patch grouping operator with padding, where convolution kernel is implemented a  form of
    some linear convolution neural network (CNN).
    Indices for patches extraction and grouping are computed using knn search from either observation, or latent image.
    """
    def __init__(self, backbone: KPBackbone, group_size: int = 5, search_window: Union[int, Tuple[int, int]] = 15,
                 exclude_from_search: Union[int, Tuple[int, int]] = 0, distance_weights: Optional[th.Tensor] = None,
                 distance_type: str = 'euclidean', stride: Union[int, Tuple[int, int]] = 1, padding_mode: str = None,
                 observation_keyword: Optional[str] = None, latent_index: Optional[int] = None, learnable: bool = False
                 ) -> None:
        """
        Initializing operator parameters.

        :param backbone: linear CNN, implemented as nn.Sequential in .backbone attribute
        :param group_size: amount of closest patches to use for extraction and grouping
        :param search_window: region to perform knn patches search
        :param exclude_from_search: which indices in window to exclude from knn search
        :param distance_weights: symmetric weights to calculate weighted distance
        :param distance_type: 'euclidean' or 'abs' indicates the type of the patch distance
        :param stride: step-sizes in x- and y-directions
        :param padding_mode: which padding to use in convolution 'zero', 'symmetric', 'periodic', None padding types
               are supported. if None - no padding is applied.
        :param observation_keyword: name of keyword argument which should be used for knn search;
               this argument is used if one wants to perform search based on observation
        :param latent_index: index to choose the specific latent tensor to be used for knn search;
               this argument is used if one wants to perform search based on some latent variable
        :param learnable: flag, which determines whether to make filters learnable
        """
        LearnableKPNConvOperator.__init__(
            self, backbone, observation_keyword=observation_keyword, latent_index=latent_index,
            padding_mode=padding_mode, learnable=learnable, mix_in_channels=True)
        PatchGroupOperator.__init__(
            self, group_size=group_size, patch_size=self._backbone.filter_size, search_window=search_window,
            exclude_from_search=exclude_from_search, distance_weights=distance_weights, distance_type=distance_type,
            stride=stride, padding_mode=padding_mode, observation_keyword=observation_keyword,
            latent_index=latent_index)

    def prepare_for_restoration(self, **kwargs) -> 'LearnableKPNConvPatchGroupOperator':
        """
        This method prepares patch grouping operator for recurrent restoration by predicting grouping indices from
        observation.

        :param kwargs: parameters for operator initialization
        :return: operator prepared for recurrent restoration
        """
        if self.observation_keyword is None:
            self.group_indices = None
            self.clear_effective_kernel()
        else:
            assert self.observation_keyword in kwargs.keys()
            self.update_group_indices(kwargs[self.observation_keyword])
            self.update_effective_kernel(kwargs[self.observation_keyword])
        return self

    def prepare_for_step(self, step_idx: int, *tensor_params: th.Tensor, **kwargs
                         ) -> 'LearnableKPNConvPatchGroupOperator':
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
            self.update_group_indices(tensor)
            self.update_effective_kernel(tensor)
        return self

    def to(self, *args, **kwargs) -> 'LearnableKPNConvPatchGroupOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        if self.group_indices is not None:
            self.group_indices = self.group_indices.to(*args, **kwargs)
        LearnableKPNConvOperator.to(self, *args, **kwargs)
        return self
