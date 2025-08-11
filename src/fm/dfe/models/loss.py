"""Loss functions.
"""
import torch


def reciprocal(x):
    x = x.where(x < 1e-8, 1e-8, x)
    x = x.where(x > 1e8, 1e8, x)
    return 1/x

def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """
    CRITICAL_REPLACE = 10
    # compute epipolar lines
    line_1 = torch.bmm(pts1, fundamental_mat)
    line_2 = torch.bmm(pts2, fundamental_mat.permute(0, 2, 1))
    # compute line norms
    line_1_norm = line_1[:, :, :2].norm(2, 2)
    line_2_norm = line_2[:, :, :2].norm(2, 2)
    scalar_product = (pts2 * line_1).sum(2)
    denominator = 1 / line_1_norm + 1 / line_2_norm
    mask_nan, mask_inf = denominator.isnan(), denominator.isinf()
    denominator = torch.where(mask_nan | mask_inf, CRITICAL_REPLACE, denominator)
    ret = scalar_product.abs() * denominator

    return ret


# def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=1.0):
def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=0.5):
    """Robust symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust symmetric epipolar distance
    """

    sed = symmetric_epipolar_distance(pts1, pts2, fundamental_mat)
    ret = torch.clamp(sed, max=gamma)

    return ret


def sampson_distance(pts1, pts2, fundamental_mat):
    """Sampson distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: sampson distance
    """
    CRITICAL_REPLACE = 10
    line_1 = torch.bmm(pts1, fundamental_mat)
    line_2 = torch.bmm(pts2, fundamental_mat.permute(0, 2, 1))

    scalar_product = (pts2 * line_1).sum(2)
    denominator = 1/ (line_1[:, :, :2].norm(2, 2).pow(2) + line_2[:, :, :2].norm(2, 2).pow(2))
    mask_nan, mask_inf = denominator.isnan(), denominator.isinf()
    denominator = torch.where(mask_nan | mask_inf, CRITICAL_REPLACE, denominator)
    ret = scalar_product.pow(2) * denominator

    return ret

def robust_sampson_distance(pts1, pts2, fundamental_mat, gamma=1):
    """Robust sampson distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust sampson distance
    """

    sed = sampson_distance(pts1, pts2, fundamental_mat)
    ret = torch.clamp(sed, max=gamma)

    return ret