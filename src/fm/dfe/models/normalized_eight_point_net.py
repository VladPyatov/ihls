"""NormalizedEightPointNet with subnets.
"""
from kornia.geometry.conversions import convert_points_to_homogeneous
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import robust_symmetric_epipolar_distance, robust_sampson_distance
from src.utils.svd import RobustEighFunction, matrix_3x3_rank2_avg_projection, matrix_3x3_rank2_projection
from src.fm.utils.irls import FundamentalMatrixEstimator
from src.fm.utils.essential import find_essential

AUGMENTATION_CONST = 1e-4
EIGH9x9_BWD_ORDER = 100
SVD3x3_BWD_ORDER = 20
DEBUG = False


def rank2_projection(F_mat: torch.Tensor):
    FFt = torch.bmm(F_mat, F_mat.transpose(-2, -1))
    identity = torch.eye(3, dtype=FFt.dtype, device=FFt.device).unsqueeze(0)
    FFt += AUGMENTATION_CONST * identity
    S, U = RobustEighFunction.apply(FFt, SVD3x3_BWD_ORDER, AUGMENTATION_CONST)
    rank_mask = torch.tensor([0., 1., 1.], dtype=U.dtype, device=U.device).unsqueeze(0)
    F_proj = torch.bmm(torch.bmm(U, rank_mask.unsqueeze(-1)*U.transpose(-2, -1)), F_mat)
    return F_proj


class IRLSEstimator(FundamentalMatrixEstimator):
    def __init__(self, p: float, num_irls_steps: int = 100, atol: float = 1e-8, rtol: float = -1, 
                 eps: float = 1e-6, learn_p: bool = False, initial_p: float = -1, 
                 initial_num_irls_steps: int = 100, matrix_type: str = 'F'):
        super().__init__(p, num_irls_steps, atol, rtol, eps, learn_p, initial_p, initial_num_irls_steps)
        # if matrix_type == 'E':
        #     self.rank2_projection = matrix_3x3_rank2_avg_projection
        self.matrix_type = matrix_type
    def normalize(self, pts, weights):
        """Normalize points based on weights.

        Args:
            pts (tensor): points
            weights (tensor): estimated weights

        Returns:
            tensor: normalized points
        """

        denom = weights.sum(1)

        center = torch.sum(pts * weights, 1) / denom
        dist = pts - center.unsqueeze(1)
        meandist = (
            (weights * (dist[:, :, :2].pow(2).sum(2).sqrt().unsqueeze(2))).sum(1)
            / denom
        ).squeeze(1)

        scale = 1.4142 / meandist

        transform = torch.zeros((pts.size(0), 3, 3), device=pts.device, dtype=pts.dtype)

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

        return pts_out, transform
    
    def forward(self, points1, points2, weights=1):
        # cast inputs to double
        input_dtype = points1.dtype
        if input_dtype is not torch.float64:
            points1, points2, weights = points1.double(), points2.double(), weights.double()
        weights = weights.squeeze(1).unsqueeze(2)

        pts1n, transform1 = self.normalize(points1, weights)
        pts2n, transform2 = self.normalize(points2, weights)
        if self.matrix_type == 'E' and not self.training and DEBUG:
            pts1 = points1.transpose(-1, -2)
            pts2 = points2.transpose(-1, -2)
            A = torch.cat((pts1[:, 0].unsqueeze(1) * pts2, pts1[:, 1].unsqueeze(1) * pts2, pts2), 1).permute(0, 2, 1) * weights
            extra_info = {'p1': pts1[:, :2].transpose(-1, -2), 'p2': pts2[:, :2].transpose(-1, -2), 'conf': weights.squeeze(-1)}
        else:
            A = torch.cat((pts1n[:, 0].unsqueeze(1) * pts2n, pts1n[:, 1].unsqueeze(1) * pts2n, pts2n), 1).permute(0, 2, 1) * weights
            extra_info = None
        # compute initial solution if required
        if self.initial_p != -1:
            initial_solution = self.irls(A, self.initial_p, self.initial_num_irls_steps, extra_info=extra_info)
        else:
            initial_solution = None
        # compute general solution
        F_mat, statistics, last_weights = self.irls(A, self.p(), self.num_irls_steps, initial_solution, return_stats=True, extra_info=extra_info)
        F_mat = F_mat.view(-1, 3, 3)
        if F_mat.isnan().any():
            statistics['converged'] = False
        else:
            # make a projection onto a rank-2 matrix space
            F_mat = self.rank2_projection(F_mat)
        # do a back transformation and normalize result
        if self.matrix_type == 'E' and not self.training and DEBUG:
            F_est = F_mat
        else:
            F_est = transform1.permute(0, 2, 1).bmm(F_mat).bmm(transform2)
        return F_est.to(input_dtype), statistics, last_weights.to(input_dtype)


class ResNetBlock(nn.Module):
    def __init__(self, inplace=True, has_bias=True, learn_affine=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=has_bias)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=has_bias)
        self.inorm1 = nn.InstanceNorm1d(128)
        self.bnorm1 = nn.BatchNorm1d(128)
        self.inorm2 = nn.InstanceNorm1d(128)
        self.bnorm2 = nn.BatchNorm1d(128)

    def forward(self, data):
        x = self.bnorm1(self.inorm1(self.conv1(data)))
        x = F.relu(self.bnorm2(self.inorm2(self.conv2(x))))
        return data + x


class LDFWeightEstimatorNet(nn.Module):
    """Network for weight estimation. - Architecture described in "Learning to find good corrspondences"
    """

    def __init__(self, input_size, inplace=True, has_bias=True, learn_affine=True):
        """Init.

        Args:
            input_size (float): size of input
            inplace (bool, optional): Defaults to True. LeakyReLU inplace?
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """

        super(LDFWeightEstimatorNet, self).__init__()

        track = False
        self.conv_in = nn.Conv1d(input_size, 128, kernel_size=1, bias=has_bias)

        blocks = []
        for i in range(12):
            blocks.append(ResNetBlock())

        self.backbone = nn.Sequential(*blocks)

        self.conv_out = nn.Conv1d(128, 1, kernel_size=1, bias=has_bias)

    def forward(self, data):
        """Forward pass.

        Args:
            data (tensor): input data

        Returns:
            tensor: forward pass
        """

        out = self.conv_in(data)
        out = self.backbone(out)
        out = self.conv_out(out)

        return out


class WeightEstimatorNet(nn.Module):
    """Network for weight estimation.
    """

    def __init__(self, input_size, inplace=True, has_bias=True, learn_affine=True):
        """Init.

        Args:
            input_size (float): size of input
            inplace (bool, optional): Defaults to True. LeakyReLU inplace?
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """

        super(WeightEstimatorNet, self).__init__()

        track = False
        has_bias = True
        learn_affine = True
        self.model = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(64, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(64, 128, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(128, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(128, 1024, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(1024, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(1024, 512, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(512, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(512, 256, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(256, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256, 1, kernel_size=1, bias=has_bias),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data (tensor): input data

        Returns:
            tensor: forward pass
        """

        return self.model(data)


class RescaleAndExpand(nn.Module):
    """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.
    """

    def __init__(self):
        """Init.
        """

        super(RescaleAndExpand, self).__init__()

        self.register_buffer("ones", torch.ones((1, 1, 1)))

    def normalize(self, pts):
        """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.

        Args:
            pts (tensor): input points

        Returns:
            tensor: transformed points
            tensor: transformation
        """

        ones = self.ones.expand(pts.size(0), pts.size(1), 1).to(pts.dtype)

        pts = torch.cat((pts, ones), 2)

        center = torch.mean(pts, 1)
        dist = pts - center.unsqueeze(1)
        meandist = dist[:, :, :2].pow(2).sum(2).sqrt().mean(1)

        scale = 1.0 / meandist

        transform = torch.zeros((pts.size(0), 3, 3), device=pts.device, dtype=pts.dtype)

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

        return pts_out, transform

    def forward(self, pts):
        """Forward pass.

        Args:
            pts (tensor): point correspondences

        Returns:
            tensor: transformed points in first image
            tensor: transformed points in second image
            tensor: transformtion (first image)
            tensor: transformtion (second image)
        """

        pts1, transform1 = self.normalize(pts[:, :, :2])
        pts2, transform2 = self.normalize(pts[:, :, 2:])

        return pts1, pts2, transform1, transform2


class ModelEstimator(nn.Module):
    """Esimator for model.
    """

    def __init__(self, matrix_type='F', robust_computation=False):
        """Init.
        """

        super(ModelEstimator, self).__init__()

        self.robust_computation = robust_computation
        self.register_buffer("mask", torch.ones(3))
        self.mask[-1] = 0
        self.register_buffer("augmentation_diag",
                             torch.diag(torch.ones(9)).unsqueeze(0)*AUGMENTATION_CONST)
        self.rank2_projection = matrix_3x3_rank2_projection
        self.matrix_type = matrix_type
    def normalize(self, pts, weights):
        """Normalize points based on weights.

        Args:
            pts (tensor): points
            weights (tensor): estimated weights

        Returns:
            tensor: normalized points
        """

        denom = weights.sum(1)

        center = torch.sum(pts * weights, 1) / denom
        dist = pts - center.unsqueeze(1)
        meandist = (
            (weights * (dist[:, :, :2].pow(2).sum(2).sqrt().unsqueeze(2))).sum(1)
            / denom
        ).squeeze(1)

        scale = 1.4142 / meandist

        transform = torch.zeros((pts.size(0), 3, 3), device=pts.device, dtype=pts.dtype)

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

        return pts_out, transform
    
    def get_solution(self, X, weights):
        matrix = torch.bmm(X.transpose(-2, -1), weights * X)
        matrix = matrix + self.augmentation_diag
        _, v = RobustEighFunction.apply(matrix.double(), EIGH9x9_BWD_ORDER, AUGMENTATION_CONST)
        solution = v.to(X)[..., 0]
        return solution.view(v.shape[0], 3, 3)

    def weighted_svd(self, pts1, pts2, weights):
        """Solve homogeneous least squares problem and extract model.

        Args:
            pts1 (tensor): points in first image
            pts2 (tensor): points in second image
            weights (tensor): estimated weights

        Returns:
            tensor: estimated fundamental matrix
        """

        weights = weights.squeeze(1).unsqueeze(2)

        pts1n, transform1 = self.normalize(pts1, weights)
        pts2n, transform2 = self.normalize(pts2, weights)

        p = torch.cat(
            (pts1n[:, 0].unsqueeze(1) * pts2n, pts1n[:, 1].unsqueeze(1) * pts2n, pts2n),
            1,
        ).permute(0, 2, 1)

        if self.robust_computation:
            # get constrained least squares solution
            F = self.get_solution(p, weights*weights)
            # project fundamental matrix to the rank-2 space
            # U, S, Vh = robust_reduced_svd(F, SVD3x3_BWD_ORDER, AUGMENTATION_CONST)
            # out = torch.bmm(U, (S*self.mask.unsqueeze(0)).unsqueeze(-1)*Vh)
            out = rank2_projection(F)
            out = transform1.permute(0, 2, 1).bmm(out).bmm(transform2)
        # legacy code
        elif self.matrix_type == 'F' or self.matrix_type == 'E' and (self.training or not DEBUG):
            X = p * weights
            out_batch = []
            for batch in range(X.size(0)):
                # solve homogeneous least squares problem
                assert not X[batch].isnan().any()
                _, _, V = torch.svd(X[batch])
                F = V[:, -1].view(3, 3)
                
                # model extractor
                assert not F.isnan().any()
                # U, S, V = torch.svd(F)
                # F_projected = U.mm((S * self.mask).diag()).mm(V.t())
                F_projected = self.rank2_projection(F[None])[0]
                out_batch.append(F_projected.unsqueeze(0))
            out = torch.cat(out_batch, 0)
            # back-transformation
            out = transform1.permute(0, 2, 1).bmm(out).bmm(transform2)
        elif self.matrix_type == 'E' and not self.training and DEBUG: # Note, this option is only for debug purposes
            out = find_essential(
                pts1[..., :2],
                pts2[..., :2],
                weights[..., 0]).transpose(-1, -2)
            #out = transform1.permute(0, 2, 1).bmm(out).bmm(transform2)
            #out = out.transpose(-1, -2)
        return out

    def forward(self, pts1, pts2, weights):
        """Forward pass.

        Args:
            pts1 (tensor): points in first image
            pts2 (tensor): points in second image
            weights (tensor): estimated weights

        Returns:
            tensor: estimated fundamental matrix
        """

        out = self.weighted_svd(pts1, pts2, weights)

        return out


class NormalizedEightPointNet(nn.Module):
    """NormalizedEightPointNet for fundamental matrix estimation.

    The output of the forward pass is the fundamental matrix and the rescaling matrices that
    transform the input points to [-1, 1]^2.

    The input are the point correspondences as well as the associated  side information.
    """

    def __init__(self, depth=1, side_info_size=0):
        """Init.
            depth (int, optional): Defaults to 1. [description]
            side_info_size (int, optional): Defaults to 0. [description]
        """

        super(NormalizedEightPointNet, self).__init__()

        self.depth = depth

        # data processing
        self.rescale_and_expand = RescaleAndExpand()

        # model estimator
        self.model = ModelEstimator()

        # weight estimator
        self.weights_init = WeightEstimatorNet(4 + side_info_size)
        self.weights_iter = WeightEstimatorNet(6 + side_info_size)

    def forward(self, pts, side_info):
        """Forward pass.

        Args:
            pts (tensor): point correspondences
            side_info (tensor): side information

        Returns:
            tensor: fundamental matrix, transformation of points in first and second image
        """

        # recale points to [-1, 1]^2 and expand with 1
        pts1, pts2, rescaling_1, rescaling_2 = self.rescale_and_expand(pts)

        pts1 = pts1.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)

        # init weights
        input_p_s = torch.cat(
            ((pts1[:, :, :2] + 1) / 2, (pts2[:, :, :2] + 1) / 2, side_info), 2
        ).permute(0, 2, 1)
        weights = F.softmax(self.weights_init(input_p_s), dim=2)

        out_depth = self.model(pts1, pts2, weights)
        out = [out_depth]

        # iter weights
        for _ in range(1, self.depth):
            residual = robust_symmetric_epipolar_distance(pts1, pts2, out_depth)

            input_p_s_w_r = torch.cat((input_p_s, weights, residual.unsqueeze(1)), 1)
            weights = F.softmax(self.weights_iter(input_p_s_w_r), dim=2)

            out_depth = self.model(pts1, pts2, weights)
            out.append(out_depth)

        return out, rescaling_1, rescaling_2, weights


class EightPointNet(nn.Module):
    """NormalizedEightPointNet for fundamental matrix estimation.

    The output of the forward pass is the fundamental matrix and the rescaling matrices that
    transform the input points to [-1, 1]^2.

    The input are the point correspondences as well as the associated  side information.
    """

    def __init__(self, depth=1, side_info_size=0, return_last=False):
        """Init.
            depth (int, optional): Defaults to 1. [description]
            side_info_size (int, optional): Defaults to 0. [description]
        """

        super().__init__()

        self.depth = depth
        self.return_last = return_last

        # data processing
        self.rescale_and_expand = RescaleAndExpand()

        # model estimator
        self.model = ModelEstimator()

        # weight estimator
        self.weights_init = WeightEstimatorNet(4 + side_info_size)
        self.weights_iter = WeightEstimatorNet(6 + side_info_size)

        # post processing of the estimated fundamental matrix
        self.post_processing = post_processing

    def forward(self, pts1, pts2, side_info):
        """Forward pass.

        Args:
            pts (tensor): point correspondences
            side_info (tensor): side information

        Returns:
            tensor: fundamental matrix, transformation of points in first and second image
        """
        assert pts1.shape[1] >= 8, f'The number of correspondences should be >= 8, but given {pts1.shape[1]}.'
        # recale points to [-1, 1]^2 and expand with 1
        pts1, pts2, rescaling_1, rescaling_2 = self.rescale_and_expand(torch.cat([pts1, pts2], -1))

        pts1 = pts1.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)

        # init weights
        input_p_s = torch.cat(
            ((pts1[:, :, :2] + 1) / 2, (pts2[:, :, :2] + 1) / 2, side_info), 2
        ).permute(0, 2, 1)
        weights = F.softmax(self.weights_init(input_p_s), dim=2)

        out_depth = self.model(pts1, pts2, weights)
        out = [self.post_processing(out_depth, rescaling_1, rescaling_2)]

        # iter weights
        for _ in range(1, self.depth):
            residual = robust_symmetric_epipolar_distance(pts1, pts2, out_depth)

            input_p_s_w_r = torch.cat((input_p_s, weights, residual.unsqueeze(1)), 1)
            weights = F.softmax(self.weights_iter(input_p_s_w_r), dim=2)

            out_depth = self.model(pts1, pts2, weights)
            out.append(self.post_processing(out_depth, rescaling_1, rescaling_2))

        # return fundamental matrix and statistics (there are no statistics yet, so return empty dict)
        out = torch.stack(out, 1)
        stats = {
            'converged': torch.tensor(1., dtype=torch.float64),
            'num_steps': torch.tensor(1., dtype=torch.float64),
            'atol': torch.tensor(0., dtype=torch.float64),
            'rtol': torch.tensor(0., dtype=torch.float64),
        }
        if self.return_last:
            return out[:, [-1]], stats
        else:
            return out, stats


def post_processing(F_est: torch.Tensor, rescaling_1: torch.Tensor, rescaling_2: torch.Tensor):
    # go to the image coords space
    if rescaling_1 is not None and rescaling_2 is not None:
        F_est = rescaling_1.permute(0, 2, 1).bmm(F_est.bmm(rescaling_2))
    # transpose solution (TODO: check whether this is the case)
    F_est = F_est.transpose(-1, -2)
    # normalization
    F_est = F_est / F_est[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
    return F_est


class IRLSEightPointNet(nn.Module):
    """IRLSEightPointNet for fundamental matrix estimation.

    The output of the forward pass is the fundamental matrix estimates from each/last step.

    The input are the point correspondences as well as the associated  side information.
    """

    def __init__(self, depth=1, side_info_size=0, weight_normalization='sigmoid', return_last=False, matrix_type='F'):
        """Init.
            depth (int, optional): Defaults to 1. [description]
            side_info_size (int, optional): Defaults to 0. [description]
        """

        super().__init__()

        self.depth = depth
        self.return_last = return_last
        self.matrix_type = matrix_type
        if weight_normalization == 'softmax':
            self.weight_normalization = lambda x: torch.softmax(x, dim=2)
        elif weight_normalization == 'sigmoid':
            self.weight_normalization = lambda x: torch.sigmoid(x)
        else:
            raise NotImplementedError('Unknown weight normalization type')

        # data processing
        self.rescale_and_expand = RescaleAndExpand()

        # model estimator
        self.model = ModelEstimator(matrix_type=matrix_type).double()
        # for fast inference: num_irls_steps=30, atol=1e0, rtol=-1., eps=1e-8
        self.fm_irls = IRLSEstimator(p=0.5, num_irls_steps=500, atol=1e-8, rtol=-1., 
                                     eps=1e-6, learn_p=False, matrix_type=matrix_type) 
        # weight estimator
        self.weights_init = WeightEstimatorNet(4 + side_info_size).double()
        self.weights_iter = WeightEstimatorNet(6 + side_info_size).double()

        # post processing of the estimated fundamental matrix
        self.post_processing = post_processing

    def forward(self, pts1, pts2, side_info):
        """Forward pass.

        Args:
            pts (tensor): point correspondences
            side_info (tensor): side information

        Returns:
            tensor: fundamental matrix, transformation of points in first and second image
        """
        assert pts1.shape[1] >= 5, f'The number of correspondences should be >= 5, but given {pts1.shape[1]}.'
        input_dtype = pts1.dtype
        if input_dtype is not torch.float64:
            pts1, pts2, side_info = pts1.double(), pts2.double(), side_info.double()
        # recale points to [-1, 1]^2 and expand with 1
        if self.matrix_type == 'F':
            pts1, pts2, rescaling_1, rescaling_2 = self.rescale_and_expand(torch.cat([pts1, pts2], -1))
        else:
            pts1 = convert_points_to_homogeneous(pts1).transpose(-2, -1)
            pts2 = convert_points_to_homogeneous(pts2).transpose(-2, -1)
            rescaling_1, rescaling_2 = None, None

        pts1 = pts1.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)

        # init weights
        input_p_s = torch.cat(
            ((pts1[:, :, :2] + 1) / 2, (pts2[:, :, :2] + 1) / 2, side_info), 2
        ).permute(0, 2, 1)
        weights = self.weight_normalization(self.weights_init(input_p_s))
        # compute initial solution
        out_depth = self.model(pts1, pts2, weights)
        out = [self.post_processing(out_depth, rescaling_1, rescaling_2)]

        # iter weights
        for _ in range(1, self.depth):
            residual = robust_symmetric_epipolar_distance(pts1, pts2, out_depth)
            input_p_s_w_r = torch.cat((input_p_s, weights, residual.unsqueeze(1)), 1)
            weights = self.weight_normalization(self.weights_iter(input_p_s_w_r))

            out_depth, statistics, irls_weights = self.fm_irls(pts1, pts2, weights)
            out.append(self.post_processing(out_depth, rescaling_1, rescaling_2))

        # return fundamental matrix and statistics (there are no statistics yet, so return empty dict)
        out = torch.stack(out, 1).to(input_dtype)
        weights = (weights[:, 0] * irls_weights).to(input_dtype)
        #weights, statistics = weights.to(input_dtype), {}
        if self.return_last:
            return out[:, [-1]], statistics, weights
        else:
            return out, statistics, weights
