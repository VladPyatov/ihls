"""Module containing the functionalities for computing the Fundamental Matrix."""

from typing import Tuple

import torch
import kornia

from src.utils.svd import RobustSymPSDEigFunction, RobustSVDFunction, AUGMENTATION_CONST, EIGH3x3_BWD_ORDER


def normalize_points(points: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Normalizes points (isotropic).

    Computes the transformation matrix such that the two principal moments of the set of points
    are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
    about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points (torch.Tensor): Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps (float): epsilon value to avoid numerical unstabilities. Default: 1e-8.

    Returns:
       Tuple[torch.Tensor, torch.Tensor]: tuple containing the normalized points in the
       shape :math:`(B, N, 2)` and the transformation matrix in the shape :math:`(B, 3, 3)`.

    """
    assert len(points.shape) == 3, points.shape
    assert points.shape[-1] == 2, points.shape

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1x2

    scale = (points - x_mean).norm(dim=-1).mean(dim=-1)   # B
    scale = torch.sqrt(torch.tensor(2.)) / (scale + eps)  # B

    ones, zeros = torch.ones_like(scale), torch.zeros_like(scale)

    transform = torch.stack([
        scale, zeros, -scale * x_mean[..., 0, 0],
        zeros, scale, -scale * x_mean[..., 0, 1],
        zeros, zeros, ones], dim=-1)  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = kornia.geometry.linalg.transform_points(transform, points)  # BxNx2

    return (points_norm, transform)


def normalize_transformation(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Normalizes a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M (torch.Tensor): The transformation to be normalized of any shape with a minimum size of 2x2.
        eps (float): small value to avoid unstabilities during the backpropagation.

    Returns:
        torch.Tensor: the normalized transformation matrix with same shape as the input.

    """
    assert len(M.shape) >= 2, M.shape
    norm_val: torch.Tensor = M[..., -1:, -1:]
    return torch.where(norm_val.abs() > eps, M / (norm_val + eps), M)


def construct_points_matrix(points1: torch.Tensor, points2: torch.Tensor):
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    ones = torch.ones_like(x1)
    A = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9
    return A


def find_fundamental(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor, mode: int = 1) -> torch.Tensor:
    r"""Computes the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1 (torch.Tensor): A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2 (torch.Tensor): A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights (torch.Tensor): Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        torch.Tensor: the computed fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    assert points1.shape == points2.shape, (points1.shape, points2.shape)
    assert len(weights.shape) == 2 and weights.shape[1] == points1.shape[1], weights.shape

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)
    X = construct_points_matrix(points1_norm, points2_norm)

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    # (WX)^T @ WX - another option
    X = w_diag @ X

    # compute eigevectors and retrieve the one with the smallest eigenvalue
    _, _, V = torch.svd(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    if mode == 0:
        F_est = transform2.transpose(-2, -1) @ (F_mat @ transform1)
    elif mode == 1:
        U, S, V = torch.svd(F_mat)
        rank_mask = torch.tensor([1., 1., 0]).to(F_mat.device)
        F_proj = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
        F_est = transform2.transpose(-2, -1) @ (F_proj @ transform1)
    elif mode == 2:
        S, U = torch.linalg.eigh(F_mat @ F_mat.transpose(-2, -1))
        rank_mask = torch.tensor([0., 1., 1.]).to(F_mat.device)
        F_proj2 = U @ (torch.diag_embed(rank_mask) @ U.transpose(-2, -1)) @ F_mat
        F_est = transform2.transpose(-2, -1) @ (F_proj2 @ transform1)
    elif mode == 3:
        FFt = F_mat @ F_mat.transpose(-2, -1)
        min_sing_value = 1e-4
        identity = torch.eye(3)[None].repeat(FFt.shape[0], 1, 1).to(FFt.device)
        FFt += min_sing_value * identity
        S, U = RobustSymPSDEigFunction.apply(FFt, 1e-7, 20, min_sing_value)
        rank_mask = torch.tensor([0., 1., 1.]).to(U.device)
        F_proj2 = U @ (torch.diag_embed(rank_mask) @ U.transpose(-2, -1)) @ F_mat
        F_est = transform2.transpose(-2, -1) @ (F_proj2 @ transform1)
    
    return normalize_transformation(F_est)
