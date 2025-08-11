from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import sampson_epipolar_distance
import torch


def epipolar_loss(b_idx, data, epipolar_geometry_matrix, epipolar_geometry_type='F'):
    num_of_matrices = epipolar_geometry_matrix.shape[0]
    valid_mask = data['gt_valid_mask'][b_idx]
    b_kpts0_gt, b_kpts1_gt = data['gt_pts0'][b_idx, valid_mask][None], data['gt_pts1'][b_idx, valid_mask][None]
    b_K0, b_K1 = data['K0'][b_idx][None], data['K1'][b_idx][None]
    # backproject points
    if epipolar_geometry_type == 'E':
        intr0i = torch.linalg.inv(b_K0)
        intr1i = torch.linalg.inv(b_K1)
        pts0h = convert_points_to_homogeneous(b_kpts0_gt)
        pts1h = convert_points_to_homogeneous(b_kpts1_gt)
        b_kpts0_gt = (intr0i @ pts0h.transpose(-1, -2)).transpose(-1, -2)[..., :2]
        b_kpts1_gt = (intr1i @ pts1h.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    loss = sampson_epipolar_distance(
        b_kpts0_gt.repeat(num_of_matrices, 1, 1), 
        b_kpts1_gt.repeat(num_of_matrices, 1, 1), 
        epipolar_geometry_matrix).mean()
    return loss