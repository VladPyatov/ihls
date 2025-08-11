from typing import List, Optional, Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F

#from asgrad.torch_tools import add_spectral_norm
#from asgrad.torch_tools import init_network_orthogonal


class KPBackbone(nn.Module):
    """
    KPN model with global average pooling in the last layer
    """
    net: nn.Module
    num_in_channels: int
    num_out_channels: int
    filter_size: int

    def __init__(self, mode='instance', num_in_channels=1, num_out_channels=8, filter_size=3,
                 predictor_num_in_channels=None, pool_kernel_sizes=(2, 2, 2, 2)):
        super(KPBackbone, self).__init__()
        if predictor_num_in_channels is None:
            predictor_num_in_channels = num_in_channels
        self.net = nn.Sequential(self.inconv(predictor_num_in_channels, 12, mode),
                                 self.downconv(12, 24, mode, pool_kernel_size=pool_kernel_sizes[0]),
                                 self.downconv(24, 48, mode, pool_kernel_size=pool_kernel_sizes[1]),
                                 self.downconv(48, 96, mode, pool_kernel_size=pool_kernel_sizes[2]),
                                 self.downconv(96, 96, mode, pool_kernel_size=pool_kernel_sizes[3]),
                                 self.double_conv(96, 48, mode),
                                 self.double_conv(48, 32, mode))
        self.outc = self.outconv(32, num_in_channels * num_out_channels * filter_size * filter_size)
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.filter_size = filter_size

    def forward(self, x):
        batch_num = x.shape[0]
        ret = self.net(x).mean(dim=(-1, -2), keepdim=True)  # B x 32 x 1 x 1
        ret = self.outc(ret)
        out_shape = (batch_num, self.num_out_channels, self.num_in_channels, self.filter_size, self.filter_size)
        ret = ret.reshape(*out_shape)  # B x C_out x C_in x H_g x W_g
        return ret

    def downconv(self, in_ch, out_ch, mode, pool_kernel_size=2):
        mpconv = nn.Sequential(
            nn.MaxPool2d(pool_kernel_size),
            self.double_conv(in_ch, out_ch, mode)
        )
        return mpconv

    @staticmethod
    def outconv(in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, 1)

    def inconv(self, in_ch, out_ch, mode):
        return self.double_conv(in_ch, out_ch, mode)

    @staticmethod
    def double_conv(in_ch, out_ch, mode):
        if mode == 'instance':
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif mode == 'batch':
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif mode == 'none':
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError('Wrong norm type is specified. Expected instance, batch or none.')
        return conv

    @property
    def kernel_shape(self):
        return self.num_out_channels, self.num_in_channels, self.filter_size, self.filter_size


from irrn.functional import SymEigFunction


class NoiseBackbone(nn.Module):
    def __init__(self, out_features_per_layer: List[int] = [2, 4, 8, 16, 16, 8, 4, 2],
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        self.backbone = nn.Sequential()
        num_in_features = 1
        for i, num_feat in enumerate(out_features_per_layer):
            self.backbone.add_module(str(2*i), nn.Linear(num_in_features, num_feat))
            self.backbone.add_module(str(2*i + 1), activation)
            num_in_features = num_feat
        self.backbone.add_module(str(2*(i+1)), nn.Linear(num_in_features, 1))
        self.final_activation = nn.Softplus()

    def forward(self, x: th.Tensor) -> th.Tensor:
        ret = self.backbone(x) + x
        ret = self.final_activation(ret)
        return ret


class RegBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, enable_abs: bool = False, final_activation: Callable = F.relu) -> None:
        super().__init__()
        self.backbone = backbone
        self.enable_abs = enable_abs
        self.final_activation = final_activation

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.enable_abs:
            x = x.abs()
        return self.final_activation(self.backbone(x))


class RegBackboneChannelwise(nn.Module):
    def __init__(self, in_features_per_layer: List[int] = [128, 256, 256, 128],
                 activation: nn.Module = nn.ReLU(inplace=True)) -> None:
        super().__init__()
        assert in_features_per_layer[0] == in_features_per_layer[-1]
        self.backbone = nn.Sequential()
        for i in range(len(in_features_per_layer) - 1):
            num_in_features = in_features_per_layer[i]
            num_out_features = in_features_per_layer[i + 1]
            self.backbone.add_module(nn.Linear(num_in_features, num_out_features))
            self.backbone.add_module(activation)

    def forward(self, x):
        # x: B, 1, H, W
        # Gx: B, Q, H, W
        ret = x.permute(0, 2, 3, 1)  # B, H, W, Q
        permute_shape = ret.shape
        ret = self.backbone(ret.flatten(end_dim=2))  # BxHxW, Q
        ret = ret.view(permute_shape)  # B, H, W, Q
        ret = ret.permute(0, 3, 1, 2)  # B, Q, H, W
        return F.relu(x + ret)


class ConvBackbone(nn.Module):
    def __init__(self, out_features_per_layer: List[int] = [32, 64, 128],
                 kernel_size_per_layer: List[int] = [5, 3, 3],
                 padding_per_layer: List[int] = [0, 0, 0],
                 strides_per_layer: List[int] = [1, 1, 1],
                 num_in_features: int = 1, rescale_output: bool = False, use_spectral_norm: bool = True) -> None:
        super().__init__()
        self.backbone = nn.Sequential()
        params_iterator = zip(out_features_per_layer, kernel_size_per_layer, padding_per_layer, strides_per_layer)
        for i, (num_out_features, kernel_size, padding_size, stride) in enumerate(params_iterator):
            self.backbone.add_module(str(i), nn.Conv2d(num_in_features, num_out_features, kernel_size,
                                                       padding=padding_size, stride=stride, bias=False))
            num_in_features = num_out_features
        init_network_orthogonal(self.backbone)
        if use_spectral_norm:
            add_spectral_norm(self.backbone)
        self.num_out_features = out_features_per_layer[-1]
        self.rescale_output = rescale_output

    def forward(self, x: th.Tensor) -> th.Tensor:
        ret = self.backbone(x)
        if self.rescale_output:
            ret = ret*self.num_out_features
        return ret


class WeightedNormBackboneHelper:
    weights: th.Tensor
    use_cumsum: bool
    norm_const: float

    def __init__(self, num_features: int, eps: float = 1e-6, use_cumsum: bool = False,
                 norm_const: Optional[float] = None) -> None:
        super().__init__(eps)
        self.weights = nn.Parameter(th.rand(num_features))
        self.use_cumsum = use_cumsum
        if norm_const is None:
            self.norm_const = float(num_features)
        else:
            self.norm_const = norm_const

    def sorted_weights(self, x: th.Tensor) -> th.Tensor:
        if self.use_cumsum:
            const = th.arange(self.weights.shape[1], 0, -1, dtype=self.weights.dtype, device=self.weights.device)
            wt = self.weights + th.log(const)
            wt = th.softmax(wt, dim=0) / const
            wt = th.cumsum(wt, 0)
            wt = th.flip(wt, dims=(0,))
        else:
            wt = th.softmax(self.weights, dim=0)
        wt = wt * self.norm_const

        b, q, h, w = x.shape
        idx = th.argsort(th.argsort(x, dim=1, descending=True), dim=1, descending=False)
        wt = wt.repeat(b, h, w, 1).permute(0, 3, 1, 2)
        wt = th.gather(wt, 1, idx)
        return wt

    def sorted_weights_old(self, x: th.Tensor) -> th.Tensor:
        b, q, h, w = x.shape
        idx = th.argsort(th.argsort(x, dim=1, descending=True), dim=1, descending=False)
        wt = th.exp(self.weights)
        if self.use_cumsum:
            wt = th.cumsum(wt, 0)
            assert wt.dim() == 1
            wt = th.flip(wt, dims=(0,))
        wt = wt / wt.sum() * self.norm_const
        wt = wt.repeat(b, h, w, 1).permute(0, 3, 1, 2)
        wt = th.gather(wt, 1, idx)
        return wt


class SpectralFunctionWrapper(nn.Module):
    def __init__(self, spectral_function: Callable, symeig_tol: float = 1e-7) -> None:
        super(SpectralFunctionWrapper, self).__init__()
        self.spectral_function = spectral_function
        self.symeig_tol = symeig_tol

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.dim() == 5  # [B, Q, C, H, W]
        matrices = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, Q]
        sym_matrices = th.matmul(matrices, matrices.transpose(-1, -2))  # [B, H, W, C, C]
        eigvals, eigvecs = SymEigFunction.apply(sym_matrices, self.symeig_tol)  # [B, H, W, C], [B, H, W, C, C]
        eigvals = self.spectral_function(eigvals.permute(0, 3, 1, 2).pow(0.5))  # [B, C/1, H, W]
        if eigvals.shape[1] == 1:
            eigvals = eigvals.repeat(1, eigvecs.shape[-1], 1, 1)
        eigvals = eigvals.permute(0, 2, 3, 1)  # [B, H, W, C]
        sym_matrices = th.matmul(eigvecs, eigvals.unsqueeze(-1) * eigvecs.transpose(-1, -2))  # [B, H, W, C, C]
        return sym_matrices


class LpNormBackbone(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(th.ones(1)*5)
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        p = th.sigmoid(self.p)*2
        denom = (x*x + self.eps).pow(p/2).sum(dim=1, keepdim=True)
        denom = denom.pow((1 - p)/p)
        ret = (x*x + self.eps).pow((p - 2)/2)*denom
        return ret


class LppNormBackbone(nn.Module):
    def __init__(self, eps: float = 1e-6, max_p: float = 1., min_p: float = 0., init_p_param_value: float = 3.) -> None:
        super().__init__()
        self.p = nn.Parameter(th.ones(1)*init_p_param_value)
        self.max_p = max_p
        self.min_p = min_p
        self.eps = eps

    def get_p_value(self):
        return th.sigmoid(self.p)*(self.max_p - self.min_p) + self.min_p

    def forward(self, z: th.Tensor) -> th.Tensor:
        p = self.get_p_value()
        ret = p*(z*z + self.eps).pow((p - 2)/2)
        return ret

    def grad_diagonal(self, z: th.Tensor) -> th.Tensor:
        z_sq = z*z
        p = self.get_p_value()
        z_sq_smooth = z_sq + self.eps
        return z_sq_smooth.pow((p - 2) / 2) * p * (1 + (p - 2) * z_sq / z_sq_smooth)


class WeightedLppNormBackboneWithContinuation(WeightedNormBackboneHelper, LppNormBackbone):
    start_gamma: th.Tensor

    def __init__(self, num_features: int, eps: float = 1e-6, use_cumsum: bool = False,
                 norm_const: Optional[float] = None, max_p: float = 1.0, start_gamma_mul: float = 1,
                 decay_gamma: float = 1/1.15) -> None:
        super(WeightedLppNormBackboneWithContinuation, self).__init__(num_features, eps=eps, use_cumsum=use_cumsum,
                                                                      norm_const=norm_const)
        self.p = nn.Parameter(th.zeros(1))
        self.max_p = max_p
        self.start_gamma_mul = start_gamma_mul
        self.decay_gamma = decay_gamma

    def forward(self, x: th.Tensor, step_num: int = 0) -> th.Tensor:
        wt = self.sorted_weights(x)
        p = th.sigmoid(self.p)*self.max_p
        assert step_num != 0
        if step_num is not None and step_num > 0:
            if step_num == 1:
                self.start_gamma = self.start_gamma_mul*th.norm(x.flatten(start_dim=1), dim=-1)
                for i in range(x.dim() - 1):
                    self.start_gamma = self.start_gamma.unsqueeze(-1)
            gamma = self.start_gamma*(self.decay_gamma**step_num)
            gamma = th.clamp(gamma, min=0, max=self.eps)
        else:
            gamma = self.eps
        ret = (x*x + gamma).pow((p - 2)/2)
        return wt*ret


class L0NormBackbone(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = 0
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        return 1/(x*x + self.eps)


class L1NormBackbone(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = 1
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        return 1/th.sqrt(x*x + self.eps)

    def diag_of_grad_matrix(self, backbone_output: th.Tensor, backbone_input: th.Tensor):
        return th.zeros_like(backbone_input)


class WeightedL1NormBackbone(WeightedNormBackboneHelper, L1NormBackbone):
    def forward(self, x: th.Tensor) -> th.Tensor:  # x: B x Q x H x W
        wt = self.sorted_weights(x)
        ret = super(WeightedL1NormBackbone, self).forward(x)
        return wt*ret


class L2NormBackbone(nn.Module):
    def __init__(self, eps: float = 1e-6, dim: int = 1) -> None:
        super().__init__()
        self.p = 2
        self.eps = eps
        self.dim = dim

    def forward(self, z: th.Tensor) -> th.Tensor:  # z: B x Q x H x W
        assert z.dim() == 4
        ret = 1/(z.norm(p=2, dim=self.dim, keepdim=True) + self.eps).expand(z.shape)
        return ret

    def diag_of_grad_matrix(self, backbone_output: th.Tensor, backbone_input: th.Tensor):
        diag = backbone_output - backbone_output.pow(3)*backbone_input.pow(2)
        return diag


class SmoothL2NormBackbone(L2NormBackbone):
    def forward(self, z: th.Tensor) -> th.Tensor:
        assert z.dim() == 4
        norm = self.smoothed_l2_norm(z, self.dim)
        ret = (1 / norm).expand(z.shape)
        return ret

    def smoothed_l2_norm(self, z: th.Tensor, dim: int) -> th.Tensor:
        norm = ((z*z).sum(dim=dim, keepdim=True) + self.eps).sqrt()
        return norm

    def grad_operator(self, z: th.Tensor, *args):
        raise NotImplementedError


class WeightedL2NormBackbone(WeightedNormBackboneHelper, L2NormBackbone):
    def forward(self, x: th.Tensor) -> th.Tensor:  # x: B x Q x H x W
        wt = self.sorted_weights(x)
        ret = super(WeightedL2NormBackbone, self).forward(wt * x)
        ret = ret * (wt.pow(2))
        return ret

    def diag_of_grad_matrix(self, backbone_output: th.Tensor, backbone_input: th.Tensor):
        wt = self.sorted_weights(backbone_input)
        wt.pow_(2)
        wt += self.eps
        diag = backbone_input.pow(2)
        diag *= backbone_output.pow(3)
        diag /= wt
        diag -= backbone_output
        return diag.neg_()

