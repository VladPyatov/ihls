from itertools import product
from typing import Generator, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import pytest
import torch as th
import torch.nn as nn
from scipy.signal import correlate2d

from irrn.operators import LinearOperatorBase, LearnableConvolutionOperator, Pad2DOperator, LearnableDiagonalOperator, \
    ImageKernelJacobian, LearnableFourierWeightOperator, LearnableCNNOperator, LearnableKPNConvOperator, \
    LearnableKPNSAConvOperator, LearnableMatMulOperator, PatchGroupOperator, LearnableConvPatchGroupOperator, \
    LearnableCNNPatchGroupOperator, LearnableKPNConvPatchGroupOperator, IRLSSystemOperator, LearnableNumberOperator
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator, \
    ConvMosaicLinearDegradationOperator
from irrn.operators.linsys.irls import IRLSNormalEquationsSystemOperator
from irrn.utils import MultiVector
from irrn.modules.backbones import KPBackbone


class TestLinearOperatorBase:
    degradation_class: LinearOperatorBase
    __test__ = False
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)]}
    should_fail_with = None
    args_name = 'parametric_args'

    def init_test_class(self, *args, **kwargs) -> Any:
        """
        SHOULD ASSIGN TO self.linear_operator_class: A DEGRADATION CLASS WITH DOUBLE DTYPES
        """
        pass

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'],
                      kwargs['spatial_dims'][0], kwargs['spatial_dims'][1]).double()
        return vec

    def iterate_through_test(self, kwargs, test_fn):
        if kwargs is None:
            for kwargs in self.params_iterator:
                err = self.get_fail_error(**kwargs)
                if self.should_fail_with is not None:
                    with pytest.raises(self.should_fail_with):
                        test_fn(kwargs)
                elif err is not None:
                    with pytest.raises(err):
                        test_fn(kwargs)
                else:
                    test_fn(kwargs)
        else:
            if self.should_fail_with is not None:
                with pytest.raises(self.should_fail_with):
                    test_fn(kwargs)
            else:
                test_fn(kwargs)

    def test_transform_transpose(self, kwargs=None):
        """
        Checks correctness of implementation of linear transpose operator, using property of dot product:
        (A x, y) = (x, A^T y)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if parametric_args is None:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            Avec_a = self.linear_operator_class.apply(vec_a, *parametric_args)
            vec_b = th.rand_like(Avec_a)
            A_Tvec_b = self.linear_operator_class.T(vec_b, *parametric_args)
            prod_1 = self._prod(Avec_a, vec_b)
            prod_2 = self._prod(vec_a, A_Tvec_b)
            assert th.all(th.isclose(prod_1, prod_2))
        self.iterate_through_test(kwargs, pass_test)

    def test_transpose_apply_symmetry(self, kwargs=None):
        """
        Checks correctness of implementation of linear transpose operator, using property of dot product:
        (A^TA x, y) = (x, A^TA y)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if parametric_args is None:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            vec_b = th.rand_like(vec_a)
            ATAvec_a = self.linear_operator_class.transpose_apply(vec_a, *parametric_args, operator_between=None)
            ATAvec_b = self.linear_operator_class.transpose_apply(vec_b, *parametric_args, operator_between=None)

            prod_1 = self._prod(ATAvec_a, vec_b)
            prod_2 = self._prod(vec_a, ATAvec_b)
            assert th.all(th.isclose(prod_1, prod_2))
        self.iterate_through_test(kwargs, pass_test)

    def prepare_for_test(self, kwargs):
        pass

    @property
    def params_iterator(self) -> Generator:
        keys, values = zip(*self.sizes_params.items())
        for bundle in product(*values):
            d = dict(zip(keys, bundle))
            yield d

    def _prod(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        assert x.shape == y.shape
        d = tuple(range(1, x.dim()))
        return (x*y).sum(dim=d)

    def get_test_image(self, is_color: bool = True, size: Tuple[int, int] = (512, 512)):
        image = cv2.imread('files/test_image_512.png')
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        image = th.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)/255
        return image

    def get_fail_error(self, **kwargs):
        return None


class TestPatchGroupOperator(TestLinearOperatorBase):
    linear_operator_class: PatchGroupOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'group_size': [4, 5],
                    'patch_size': [3, 4],
                    'stride': [1, 2],
                    'search_window': [7, 8],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'distance_type': ['euclidean', 'abs'],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 17)],
                    'exclude_from_search': [(0, 0), (1, 1)]}

    def init_test_class(self, batch_size=None, group_size=None, patch_size=None, stride=None, search_window=None,
                        distance_type=None, padding_mode=None, channels_number=None, spatial_dims=None,
                        exclude_from_search=None, **kwargs) -> LinearOperatorBase:
        self.linear_operator_class = PatchGroupOperator(
            group_size=group_size, patch_size=patch_size, search_window=search_window,
            exclude_from_search=exclude_from_search, distance_weights=None, distance_type=distance_type, stride=stride,
            padding_mode=padding_mode, observation_keyword='degraded', latent_index=None).to(th.float64)
        degraded = th.rand(batch_size, channels_number, *spatial_dims, dtype=th.float64)
        self.linear_operator_class.prepare_for_restoration(degraded=degraded)
        return (self.linear_operator_class.pad_operator.output_shape(degraded), )

    def test_visual(self):
        image = self.get_test_image(is_color=True, size=(256, 256))
        patch_size = 15
        group_size = 5
        op = PatchGroupOperator(group_size=group_size, patch_size=patch_size, search_window=100, distance_weights=None,
                                distance_type='euclidean', stride=5, padding_mode=None, observation_keyword='degraded',
                                latent_index=None)
        op.prepare_for_restoration(degraded=image)
        groups = op(image)
        patches = groups[0, :, :, 36, 10].view(3, patch_size, patch_size, group_size)
        fig, ax = plt.subplots(nrows=1, ncols=group_size, figsize=(3 * group_size, 3))
        for i in range(group_size):
            ax[i].imshow(patches[..., i].permute(1, 2, 0))
            ax[i].axis('off')
        plt.show()


class TestConvPatchGroupOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableConvPatchGroupOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'group_size': [4, 5],
                    'patch_size': [3, 4],
                    'stride': [1, 2],
                    'search_window': [7, 8],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'distance_type': ['euclidean', 'abs'],
                    'channels_number': [1, 3],
                    'out_channels_number': [7, 8],
                    'spatial_dims': [(16, 17)],
                    'exclude_from_search': [(0, 0), (1, 1)]}

    def init_test_class(self, batch_size=None, group_size=None, patch_size=None, stride=None, search_window=None,
                        distance_type=None, padding_mode=None, channels_number=None, spatial_dims=None,
                        out_channels_number=None, exclude_from_search=None, **kwargs) -> LinearOperatorBase:
        self.linear_operator_class = LearnableConvPatchGroupOperator(
            group_size=group_size, patch_size=patch_size, search_window=search_window,
            exclude_from_search=exclude_from_search, distance_weights=None, distance_type=distance_type, stride=stride,
            padding_mode=padding_mode, observation_keyword='degraded', latent_index=None,
            kernel_num_in_channels=channels_number, kernel_num_out_channels=out_channels_number, kernel=None,
            learnable=False).to(th.float64)
        degraded = th.rand(batch_size, channels_number, *spatial_dims, dtype=th.float64)
        self.linear_operator_class.prepare_for_restoration(degraded=degraded)
        return (self.linear_operator_class.pad_operator.output_shape(degraded), )


class TestCNNPatchGroupOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableCNNPatchGroupOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'group_size': [4, 5],
                    'patch_size': [3, 4],
                    'stride': [1, 2],
                    'search_window': [7, 8],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'distance_type': ['euclidean', 'abs'],
                    'channels_number': [1, 3],
                    'out_channels_number': [7, 8],
                    'spatial_dims': [(16, 17)],
                    'exclude_from_search': [(0, 0), (1, 1)]}

    def init_test_class(self, batch_size=None, group_size=None, patch_size=None, stride=None, search_window=None,
                        distance_type=None, padding_mode=None, channels_number=None, out_channels_number=None,
                        spatial_dims=None, exclude_from_search=None, **kwargs) -> LinearOperatorBase:
        class TestLinearCNN(nn.Module):
            def __init__(self):
                super(TestLinearCNN, self).__init__()
                num_hidden_layers = (patch_size - 3)
                if patch_size % 2 == 0:
                    k_size = 2
                else:
                    k_size = 3
                hidden_layers = tuple([nn.Conv2d(out_channels_number, out_channels_number, k_size, bias=False)
                                       for i in range(num_hidden_layers)])
                self.backbone = nn.Sequential(nn.Conv2d(channels_number, out_channels_number, 3, bias=False),
                                              *hidden_layers)

            def forward(self, x):
                return self.backbone(x)
        cnn = TestLinearCNN()
        self.linear_operator_class = LearnableCNNPatchGroupOperator(
            cnn, group_size=group_size, search_window=search_window, exclude_from_search=exclude_from_search,
            distance_weights=None, distance_type=distance_type, stride=stride, padding_mode=padding_mode,
            observation_keyword='degraded', latent_index=None, learnable=False).to(th.float64)
        degraded = th.rand(batch_size, channels_number, *spatial_dims, dtype=th.float64)
        self.linear_operator_class.prepare_for_restoration(degraded=degraded)
        return (self.linear_operator_class.pad_operator.output_shape(degraded), )


class TestKPNConvPatchGroupOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableKPNConvPatchGroupOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'group_size': [4, 5],
                    'patch_size': [3, 4],
                    'stride': [1, 2],
                    'search_window': [7, 8],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'distance_type': ['euclidean', 'abs'],
                    'channels_number': [1, 3],
                    'out_channels_number': [7, 8],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)]}

    def init_test_class(self, batch_size=None, group_size=None, patch_size=None, stride=None, search_window=None,
                        distance_type=None, padding_mode=None, channels_number=None, out_channels_number=None,
                        spatial_dims=None, **kwargs) -> LinearOperatorBase:
        class KPCNN(KPBackbone):
            def __init__(self):
                super(KPBackbone, self).__init__()
                self.net = nn.Conv2d(channels_number, channels_number * out_channels_number * patch_size * patch_size,
                                     3, bias=False)
                self.num_in_channels = channels_number
                self.num_out_channels = out_channels_number
                self.filter_size = patch_size

            def forward(self, x):
                res = self.net(x)
                res = res.mean(dim=(-1, -2))
                res = res.view(-1, self.num_out_channels, self.num_in_channels, self.filter_size, self.filter_size)
                return res

        cnn = KPCNN()
        self.linear_operator_class = LearnableKPNConvPatchGroupOperator(
            cnn, group_size=group_size, search_window=search_window, distance_weights=None,
            distance_type=distance_type, stride=stride, padding_mode=padding_mode, observation_keyword='degraded',
            latent_index=None, learnable=False).to(th.float64)
        degraded = th.rand(batch_size, channels_number, *spatial_dims, dtype=th.float64)
        self.linear_operator_class.prepare_for_restoration(degraded=degraded)
        return (self.linear_operator_class.pad_operator.output_shape(degraded), )


class TestMatMulLinearOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableMatMulOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'matrix_size': [16, 17],
                    'side': ['left', 'right']}
    should_fail_with = None

    def init_test_class(self, batch_size, channels_number, spatial_dims, matrix_size, side, **kwargs
                        ) -> LinearOperatorBase:
        if side == 'left':
            matrix = th.rand(batch_size, channels_number, matrix_size, spatial_dims[0])
        else:
            matrix = th.rand(batch_size, channels_number, spatial_dims[1], matrix_size)
        self.linear_operator_class = LearnableMatMulOperator(matrix, side)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)


class TestConvolutionOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableConvolutionOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'filters_size': [3, 2],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'channels_mixing': [True, False],
                    'filters_color': [True, False]}
    should_fail_with = None

    def return_test_class(self, filters_size=3, padding_mode=None,
                          channels_number=3, **kwargs) -> LinearOperatorBase:
        return LearnableConvolutionOperator(
            filter_size=filters_size, filter_num_in_channels=channels_number,
            padding_mode=padding_mode, learnable=False)

    def init_test_class(self, filters_size=3, filters_color=True, padding_mode=None,
                        channels_number=3, channels_mixing=False, **kwargs) -> LinearOperatorBase:
        if channels_number == 1 or not filters_color:
            filters_num_in_channels = 1
        else:
            filters_num_in_channels = channels_number
        self.linear_operator_class = self.return_test_class(
            filters_size=filters_size, padding_mode=padding_mode,
            channels_number=filters_num_in_channels, **kwargs)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)

    def test_convolution(self, kwargs=None):
        """
        Checks whether batched convolution is computed correctly - compare it with scipy.signal.correlate2d.
        """
        def pass_test(kwargs):
            self.init_test_class(**kwargs)
            image = th.rand(kwargs['batch_size'], kwargs['channels_number'],
                            kwargs['spatial_dims'][0], kwargs['spatial_dims'][1]).double()
            convolved_th = self.linear_operator_class._conv(image)
            filters = self.linear_operator_class.effective_kernel
            if image.shape[1] != 1 and filters.shape == 1:
                filters = filters.expand(-1, kwargs['channels_number'], -1, -1)
            convolved_true = []
            for batch in range(kwargs['batch_size']):
                filtered = []
                for num in range(filters.shape[0]):
                    colored_image = []
                    for channel in range(kwargs['channels_number']):
                        if filters.shape[1] == 1:
                            c = 0
                        else:
                            c = channel
                        conv = correlate2d(image[batch, channel, :, :].numpy(), filters[num, c, :, :].numpy(),
                                           mode='valid')
                        colored_image.append(th.from_numpy(conv))
                    filtered.append(th.stack(colored_image, dim=0))
                convolved_true.append(th.stack(filtered, dim=0))
            convolved_true = th.stack(convolved_true, dim=0)
            if self.linear_operator_class.mix_in_channels:
                convolved_true = convolved_true.sum(dim=2)
            assert th.all(th.isclose(convolved_true.squeeze(), convolved_th.squeeze()))
        if kwargs is None:
            for kwargs in self.params_iterator:
                pass_test(kwargs)
        else:
            pass_test(kwargs)


class TestCNNOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableCNNOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'out_channels_number': [1, 5],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'channels_mixing': [True, False]}
    should_fail_with = None

    def init_test_class(self, batch_size, channels_number, spatial_dims, out_channels_number, padding_mode,
                        channels_mixing, **kwargs) -> LinearOperatorBase:
        class TestLinearCNN(nn.Module):
            def __init__(self):
                super(TestLinearCNN, self).__init__()
                self.backbone = nn.Sequential(nn.Conv2d(channels_number, channels_number*2, 3, bias=False),
                                              nn.Conv2d(channels_number*2, out_channels_number, 3, bias=False))

            def forward(self, x):
                return self.backbone(x)
        cnn = TestLinearCNN()
        self.linear_operator_class = LearnableCNNOperator(cnn, padding_mode=padding_mode, learnable=False,
                                                          mix_in_channels=channels_mixing)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)

    def test_parametrization(self):
        def pass_test(kwargs):
            if kwargs['padding_mode'] is not None:
                return
            if not kwargs['channels_mixing']:
                return
            self.init_test_class(**kwargs)
            rnd_vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
            with th.no_grad():
                res_true = self.linear_operator_class(rnd_vec)
                res_merged = self.linear_operator_class._backbone(rnd_vec)
            assert th.all(th.isclose(res_true, res_merged))

        for kwargs in self.params_iterator:
            pass_test(kwargs)
        

class TestKPNConvOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableKPNConvOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'filter_size': [3, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'out_channels_number': [1, 5],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic'],
                    'channels_mixing': [True, False]}
    should_fail_with = None

    def init_test_class(self, batch_size, filter_size, channels_number, spatial_dims, out_channels_number, padding_mode,
                        channels_mixing, **kwargs) -> LinearOperatorBase:
        class KPCNN(KPBackbone):
            def __init__(self):
                super(KPBackbone, self).__init__()
                self.net = nn.Conv2d(channels_number, channels_number*out_channels_number*filter_size*filter_size,
                                     3, bias=False)
                self.num_in_channels = channels_number
                self.num_out_channels = out_channels_number
                self.filter_size = filter_size

            def forward(self, x):
                res = self.net(x)
                res = res.mean(dim=(-1, -2))
                res = res.view(-1, self.num_out_channels, self.num_in_channels, self.filter_size, self.filter_size)
                return res
        cnn = KPCNN()
        self.linear_operator_class = LearnableKPNConvOperator(cnn, 'latent', padding_mode=padding_mode,
                                                              learnable=False, mix_in_channels=channels_mixing)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)
        latent = th.rand(batch_size, channels_number, *spatial_dims, dtype=th.float64)
        self.linear_operator_class.prepare_for_restoration(latent=latent)

    def test_convolution(self, kwargs=None):
        """
        Checks whether batched convolution is computed correctly - compare it with scipy.signal.correlate2d.
        """
        def pass_test(kwargs):
            self.init_test_class(**kwargs)
            image = th.rand(kwargs['batch_size'], kwargs['channels_number'],
                            kwargs['spatial_dims'][0], kwargs['spatial_dims'][1]).double()
            filters = self.linear_operator_class.effective_kernel
            assert filters.dim() == 5
            convolved_op = self.linear_operator_class._batchwise_conv_with_mixing(image, filters)
            convolved_true = []
            for x, f in zip(image, filters):
                convolved_true.append(th.conv2d(x.unsqueeze(0), f))
            convolved_true = th.cat(convolved_true, dim=0)
            assert th.all(th.isclose(convolved_true.squeeze(), convolved_op.squeeze()))
        if kwargs is None:
            for kwargs in self.params_iterator:
                pass_test(kwargs)
        else:
            pass_test(kwargs)


class TestKPNSAConvOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableKPNSAConvOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'filter_size': [3, 2],
                    'channels_number': [1, 3, 5],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'out_channels_number': [1, 3, 5],
                    'padding_mode': [None, 'zero', 'periodic', 'symmetric'],
                    'channels_mixing': [True, False]}
    should_fail_with = None

    def init_test_class(self, batch_size, filter_size, channels_number, spatial_dims, out_channels_number, padding_mode,
                        channels_mixing, **kwargs) -> LinearOperatorBase:
        class KPCNN(KPBackbone):
            def __init__(self):
                super(KPBackbone, self).__init__()
                self.net = nn.Conv2d(channels_number, channels_number * out_channels_number * filter_size * filter_size,
                                     3, bias=False)
                self.num_in_channels = channels_number
                self.num_out_channels = out_channels_number
                self.filter_size = filter_size

            def forward(self, x):
                res = self.net(x)
                res = res.mean(dim=(-1, -2))
                res = res.view(-1, self.num_out_channels, self.num_in_channels, self.filter_size, self.filter_size)
                return res

        cnn = KPCNN()
        self.linear_operator_class = LearnableKPNSAConvOperator(cnn, observation_keyword='latent',
                                                                padding_mode=padding_mode,
                                                                learnable=False, mix_in_channels=channels_mixing)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)
        kernel_spatial_dim_1 = spatial_dims[0]
        kernel_spatial_dim_2 = spatial_dims[1]
        if padding_mode is None:
            kernel_spatial_dim_1 -= (filter_size - 1)
            kernel_spatial_dim_2 -= (filter_size - 1)
        if channels_mixing or channels_number == 1:
            kernel_channel_number = channels_number
        else:
            kernel_channel_number = 1

        self.linear_operator_class.effective_kernel = th.rand(batch_size,
                                                              out_channels_number, kernel_channel_number,
                                                              kernel_spatial_dim_1, kernel_spatial_dim_2,
                                                              filter_size, filter_size,
                                                              dtype=th.float64)

    def test_transform_transpose(self, kwargs=None):
        """
        Checks correctness of implementation of linear transpose operator, using property of dot product:
        (A x, y) = (x, A^T y)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if not parametric_args:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            Avec_a = self.linear_operator_class.apply(vec_a, *parametric_args)
            vec_b = th.rand_like(Avec_a)
            A_Tvec_b = self.linear_operator_class.T(vec_b, *parametric_args, output_size=vec_a.shape[-3:])
            prod_1 = self._prod(Avec_a, vec_b)
            prod_2 = self._prod(vec_a, A_Tvec_b)
            assert th.all(th.isclose(prod_1, prod_2))

        if kwargs is None:
            for kwargs in self.params_iterator:
                if self.should_fail_with is not None:
                    with pytest.raises(self.should_fail_with):
                        pass_test(kwargs)
                else:
                    pass_test(kwargs)
        else:
            print(self.should_fail_with)
            if self.should_fail_with is not None:
                with pytest.raises(self.should_fail_with):
                    pass_test(kwargs)
            else:
                pass_test(kwargs)


class TestFourierWeightOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableFourierWeightOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'num_weights': [1, 2],
                    'signal_size': [17, (20, 21)]}

    def init_test_class(self, num_weights=1, signal_size=20, **kwargs) -> LinearOperatorBase:
        self.linear_operator_class = LearnableFourierWeightOperator(
            num_weights=num_weights, signal_size=signal_size, learnable=False).to(th.float64)

    def test_transform_transpose(self, kwargs=None):
        """
        Checks correctness of implementation of linear transpose operator, using property of dot product:
        (A x, y) = (x, A^T y)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if not parametric_args:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            Avec_a = self.linear_operator_class.apply(vec_a, *parametric_args)
            vec_b = th.rand_like(Avec_a)
            A_Tvec_b = self.linear_operator_class.T(vec_b, *parametric_args, output_size=vec_a.shape[-2:])
            prod_1 = self._prod(Avec_a, vec_b, self.linear_operator_class.signal_size[-1])
            prod_2 = self._prod(vec_a, A_Tvec_b)
            assert th.all(th.isclose(prod_1, prod_2))
        if kwargs is None:
            for kwargs in self.params_iterator:
                if self.should_fail_with is not None:
                    with pytest.raises(self.should_fail_with):
                        pass_test(kwargs)
                else:
                    pass_test(kwargs)
        else:
            print(self.should_fail_with)
            if self.should_fail_with is not None:
                with pytest.raises(self.should_fail_with):
                    pass_test(kwargs)
            else:
                pass_test(kwargs)

    def _prod(self, x: th.Tensor, y: th.Tensor, last_dim=None) -> th.Tensor:
        assert x.shape == y.shape
        if x.dtype not in (th.complex128, th.complex64, th.complex32):
            return super(TestFourierWeightOperator, self)._prod(x, y)
        else:
            ret = x.conj()*y
            ret = self.onesided_sum(ret, last_dim)
            d = tuple(range(1, x.dim()-2))
            return ret.sum(d).real

    @staticmethod
    def onesided_sum(tensor, dim):
        add_dim = dim - (dim // 2 + 1)
        if dim % 2 == 0:
            return tensor.sum((-1, -2)) + tensor[..., -(add_dim + 1):-1].sum((-1, -2))
        else:
            return tensor.sum((-1, -2)) + tensor[..., -add_dim:].sum((-1, -2))


class TestPaddingOperator(TestLinearOperatorBase):
    linear_operator_class: Pad2DOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'padding_size': [0, 3, (2, 3), (1, 2, 3, 4), (1, 0, 1, 0)],
                    'padding_mode': [None, 'zero', 'symmetric', 'periodic']}
    should_fail_with = None

    def init_test_class(self, padding_size=5, padding_mode=None, **kwargs) -> LinearOperatorBase:
        self.linear_operator_class = Pad2DOperator(padding_size, padding_mode)

    def test_errors(self):
        with pytest.raises(NotImplementedError):
            Pad2DOperator(5, 'unknown')
        with pytest.raises(AssertionError):
            Pad2DOperator(5.5, 'zero')
        with pytest.raises(AssertionError):
            Pad2DOperator((5, 5.5), 'symmetric')
        with pytest.raises(AssertionError):
            Pad2DOperator((5, 5.5, 5, 5), 'periodic')
        with pytest.raises(ValueError):
            Pad2DOperator((1, 2, 3), None)
        with pytest.raises(AssertionError):
            pad_op = Pad2DOperator((10, 10, 10, 10), 'zero')
            pad_op(th.rand(1, 3, 5, 5))

    def test_visual(self):
        image = self.get_test_image(is_color=True, size=(512, 512))
        pad_h = image.shape[-2]//3
        pad_w = image.shape[-1]//3
        num_padding_modes = len(self.sizes_params['padding_mode'])
        fig, ax = plt.subplots(nrows=1, ncols=num_padding_modes, figsize=(3*num_padding_modes, 3))
        for i in range(num_padding_modes):
            padding_mode = self.sizes_params['padding_mode'][i]
            pad_op = Pad2DOperator((pad_h, pad_w), padding_mode)
            ax[i].imshow(pad_op(image)[0].permute(1, 2, 0))
            ax[i].axis('off')
            ax[i].set_title(str(padding_mode))
        plt.show()


class TestDiagonalOperator(TestLinearOperatorBase):
    linear_operator_class: LearnableDiagonalOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'not_reduce_dims': [(1,), (1, 2)]}
    should_fail_with = None

    def init_test_class(self, batch_size, channels_number,
                        spatial_dims, not_reduce_dims, **kwargs) -> LinearOperatorBase:
        shape = (batch_size, channels_number, *spatial_dims)
        diag_shape = [1, 1, 1, 1]
        for dim in not_reduce_dims:
            diag_shape[dim] = shape[dim]
        self.linear_operator_class = LearnableDiagonalOperator(function=th.cos, diagonal_vector_shape=diag_shape)
        self.linear_operator_class = self.linear_operator_class.to(th.float64)


class TestImageKernelJacobian(TestLinearOperatorBase):
    linear_operator_class: ImageKernelJacobian
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'kernel_size': [5, (3, 5)],
                    'scale_factor': [1, 2, 3],
                    'padding_type': ['zero', None, 'symmetric', 'periodic']}
    should_fail_with = None

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), kernel_size=5, scale_factor=1,
                        padding_type=None, **kwargs) -> Any:
        degraded_images = th.rand(batch_size, channels_number, *spatial_dims).to(th.float64)
        self.linear_operator_class = ImageKernelJacobian(scale_factor, padding_mode=padding_type)
        images, kernels = self.linear_operator_class.init_parameters(degraded_images, kernel_size)
        return images, kernels

    def init_input_vector(self, kwargs=None):
        images, kernels = kwargs[self.args_name]
        vec = MultiVector((th.rand_like(images),
                          th.rand_like(kernels))).double()
        return vec

    def test_visual(self):
        image = self.get_test_image(is_color=True, size=(128, 128))
        sr_scale = [1, 2]
        kernels_size = [13, (5, 13)]
        iterator = product(sr_scale, kernels_size)
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(3*4, 6))
        for i, (scale, kernel_size) in enumerate(iterator):
            jacobian = ImageKernelJacobian(scale)
            img, kern = jacobian.init_parameters(image, kernel_size)
            ax[0][i].imshow(kern[0, 0])
            ax[0][i].axis('off')
            ax[1][i].imshow(img[0].permute(1, 2, 0))
            ax[1][i].axis('off')
        plt.show()


class TestConvDecimateDegradationOperator(TestLinearOperatorBase):
    linear_operator_class: ConvDecimateLinearDegradationOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'kernel_color': [False, True],
                    'spatial_dims': [(18, 18), (17, 17), (18, 21)],
                    'kernel_size': [None, (5, 5), (3, 5)],
                    'scale_factor': [1, 2, 3],
                    'padding_type': ['zero', None, 'symmetric', 'periodic']}
    should_fail_with = None

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), kernel_size=(5, 5),
                        scale_factor=1, padding_type=None, kernel_color=False, **kwargs) -> Any:
        if channels_number != 1 and kernel_color:
            kernel_nc = channels_number
        else:
            kernel_nc = 1
        if kernel_size is None:
            kernel = None
        else:
            kernel = th.rand(batch_size, kernel_nc, *kernel_size, dtype=th.float64)
        self.linear_operator_class = ConvDecimateLinearDegradationOperator(
            scale_factor=scale_factor, padding_mode=padding_type, kernel=kernel)

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
        return vec

    def get_fail_error(self, padding_type=None, spatial_dims=(18, 18), kernel_size=(5, 5), scale_factor=1, **kwargs):
        if kernel_size is None:
            kernel_size = (1, 1)
        if padding_type is None:
            shape_before_decimation = (spatial_dims[0] - kernel_size[0] + 1, spatial_dims[1] - kernel_size[1] + 1)
        else:
            shape_before_decimation = spatial_dims
        if shape_before_decimation[0] % scale_factor != 0 or shape_before_decimation[1] % scale_factor != 0:
            return AssertionError


class TestConvMosaicLinearDegradationOperator(TestLinearOperatorBase):
    linear_operator_class: ConvMosaicLinearDegradationOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'pattern': ['rggb', 'bggr', 'gbrg', 'grbg'],
                    'channels_number': [1, 3],
                    'kernel_color': [False, True],
                    'spatial_dims': [(18, 18), (17, 17), (18, 21)],
                    'kernel_size': [None, (5, 5), (3, 5)],
                    'padding_type': ['zero', None, 'symmetric', 'periodic']}
    should_fail_with = None

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), kernel_size=(5, 5),
                        padding_type=None, kernel_color=False, pattern='rggb', **kwargs) -> Any:
        if channels_number != 1 and kernel_color:
            kernel_nc = channels_number
        else:
            kernel_nc = 1
        if kernel_size is None:
            kernel = None
        else:
            kernel = th.rand(batch_size, kernel_nc, *kernel_size, dtype=th.float64)
        self.linear_operator_class = ConvMosaicLinearDegradationOperator(padding_mode=padding_type, kernel=kernel,
                                                                   pattern=pattern)

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
        return vec

    def get_fail_error(self, padding_type=None, spatial_dims=(18, 18), kernel_size=(5, 5), channels_number=3,
                       pattern='rggb', **kwargs):
        if channels_number != 3:
            return AssertionError
        if pattern in ('rggb', 'bggr', 'gbrg', 'grbg'):
            scale_factor = 2
        else:
            raise NotImplementedError
        if kernel_size is None:
            kernel_size = (1, 1)
        if padding_type is None:
            shape_before_decimation = (spatial_dims[0] - kernel_size[0] + 1, spatial_dims[1] - kernel_size[1] + 1)
        else:
            shape_before_decimation = spatial_dims
        if shape_before_decimation[0] % scale_factor != 0 or shape_before_decimation[1] % scale_factor != 0:
            return AssertionError


class TestIRLSOperator(TestLinearOperatorBase):
    linear_operator_class: IRLSSystemOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(18, 18), (17, 17), (18, 21)]}
    should_fail_with = None

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
        return vec

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), **kwargs) -> Any:
        degradation = ConvDecimateLinearDegradationOperator(scale_factor=1, padding_mode=None,
                                                            kernel=th.rand(batch_size, channels_number, 5, 5))
        reg_images = LearnableConvolutionOperator(filter_size=3, filter_num_in_channels=channels_number,
                                                  padding_mode=None, learnable=False, mix_in_channels=False)
        reg_fidelity = LearnableConvolutionOperator(filter_size=2, filter_num_in_channels=channels_number,
                                                    padding_mode=None, learnable=False, mix_in_channels=False)
        weight_decay = LearnableNumberOperator(th.ones(1)*1e1, function=lambda x: x)
        reg_backbone_linear = LearnableDiagonalOperator(reg_images(
            self.init_input_vector({'batch_size': batch_size,
                                    'channels_number': channels_number, 'spatial_dims': spatial_dims}).float()))
        weight_noise = LearnableNumberOperator(th.ones(1)*1e-1)
        self.linear_operator_class = IRLSSystemOperator(degradation, reg_fidelity, reg_images, weight_decay,
                                                        reg_backbone_linear, weight_noise, 1).to(dtype=th.float64)


class TestIRLSormalEquationsOperator(TestLinearOperatorBase):
    linear_operator_class: IRLSNormalEquationsSystemOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(18, 18), (17, 17), (18, 21)],
                    'mix_in_channels': [True]}
    should_fail_with = None

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
        return vec

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), mix_in_channels=True,
                        **kwargs) -> Any:
        degradation = ConvDecimateLinearDegradationOperator(scale_factor=1, padding_mode=None,
                                                            kernel=th.rand(batch_size, channels_number, 5, 5))
        reg_images = LearnableConvolutionOperator(filter_size=3, filter_num_in_channels=channels_number,
                                                  padding_mode=None, learnable=False, mix_in_channels=True)
        reg_fidelity = LearnableConvolutionOperator(filter_size=2, filter_num_in_channels=channels_number,
                                                    padding_mode=None, learnable=False, mix_in_channels=mix_in_channels)
        weight_decay = LearnableNumberOperator(th.ones(1)*1e1, function=lambda x: x)
        reg_backbone_linear = LearnableDiagonalOperator(reg_images(
            self.init_input_vector({'batch_size': batch_size,
                                    'channels_number': channels_number, 'spatial_dims': spatial_dims}).float()).abs())
        weight_noise = LearnableNumberOperator(th.ones(1)*1e-1)
        self.linear_operator_class = IRLSNormalEquationsSystemOperator(
            degradation, reg_fidelity, reg_images, weight_decay, reg_backbone_linear,
            weight_noise, 1).to(dtype=th.float64)

    def test_transpose_apply(self, kwargs=None):
        """
        Checks correctness of implementation by checking, that A.T(A(x)) matches A.transpose_apply(x)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if parametric_args is None:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            Avec_a = self.linear_operator_class.apply(vec_a, *parametric_args)
            A_TAvec_a = self.linear_operator_class.T(Avec_a, *parametric_args)
            A_TAvec_a_true = self.linear_operator_class.transpose_apply(vec_a, *parametric_args)

            try:
                assert th.allclose(A_TAvec_a, A_TAvec_a_true, rtol=1e-4, atol=1e-7)
            except AssertionError:
                print(A_TAvec_a - A_TAvec_a_true)
        self.iterate_through_test(kwargs, pass_test)


class TestIRLSormalEquationsOperatorSqrtInit(TestLinearOperatorBase):
    linear_operator_class: IRLSNormalEquationsSystemOperator
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(18, 18), (17, 17), (18, 21)],
                    'mix_in_channels': [True]}
    should_fail_with = None

    def init_input_vector(self, kwargs=None):
        vec = th.rand(kwargs['batch_size'], kwargs['channels_number'], *kwargs['spatial_dims']).double()
        return vec

    def init_test_class(self, batch_size=3, channels_number=3, spatial_dims=(16, 16), mix_in_channels=True,
                        **kwargs) -> Any:
        degradation = ConvDecimateLinearDegradationOperator(scale_factor=1, padding_mode=None,
                                                            kernel=th.rand(batch_size, channels_number, 5, 5))
        reg_images = LearnableConvolutionOperator(filter_size=3, filter_num_in_channels=channels_number,
                                                  padding_mode=None, learnable=False, mix_in_channels=True)
        reg_fidelity = LearnableConvolutionOperator(filter_size=2, filter_num_in_channels=channels_number,
                                                    padding_mode=None, learnable=False, mix_in_channels=mix_in_channels)
        weight_decay = LearnableNumberOperator(th.ones(1)*1e1, function=lambda x: x)
        reg_backbone_linear = LearnableDiagonalOperator(reg_images(
            self.init_input_vector({'batch_size': batch_size,
                                    'channels_number': channels_number, 'spatial_dims': spatial_dims}).float()).abs())
        weight_noise = LearnableNumberOperator(th.ones(1)*1e-1)
        self.linear_operator_class = IRLSNormalEquationsSystemOperator(
            degradation, reg_fidelity, reg_images, None, None,
            None, 1, weight_decay_sqrt=weight_decay, reg_backbone_linear_sqrt=reg_backbone_linear,
            weight_noise_sqrt=weight_noise).to(dtype=th.float64)

    def test_transpose_apply(self, kwargs=None):
        """
        Checks correctness of implementation by checking, that A.T(A(x)) matches A.transpose_apply(x)
        """
        def pass_test(kwargs):
            parametric_args = self.init_test_class(**kwargs)
            if parametric_args is None:
                parametric_args = ()
            kwargs[self.args_name] = parametric_args
            vec_a = self.init_input_vector(kwargs).double()
            Avec_a = self.linear_operator_class.apply(vec_a, *parametric_args)
            A_TAvec_a = self.linear_operator_class.T(Avec_a, *parametric_args)
            A_TAvec_a_true = self.linear_operator_class.transpose_apply(vec_a, *parametric_args)

            try:
                assert th.allclose(A_TAvec_a, A_TAvec_a_true)
            except AssertionError:
                print(A_TAvec_a - A_TAvec_a_true)
        self.iterate_through_test(kwargs, pass_test)
