"""Module containing functionalities for the Essential matrix."""
from typing import Optional

import torch

import kornia.geometry.epipolar as epi
from kornia.core import eye, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.geometry import solvers
from kornia.utils.helpers import _torch_svd_cast

from src.utils.svd import RobustEighFunction

AUGMENTATION_CONST = 1e-4
EIGH9x9_BWD_ORDER = 100

def run_5point(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized].

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")
    if weights is not None:
        KORNIA_CHECK_SAME_SHAPE(points1[:, :, 0], weights)

    batch_size, _, _ = points1.shape
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
    ones = ones_like(x1)

    # build equations system and find null space.
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    # BxNx9
    X = torch.cat([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1)

    # apply the weights to the linear system
    if weights is not None:
        w_diag = torch.diag_embed(weights)
        X = w_diag @ X
    X = X.transpose(-2, -1) @ X
    # compute eigenvectors and retrieve the one with the smallest eigenvalue, using SVD
    # turn off the grad check due to the unstable gradients from SVD.
    # several close to zero values of eigenvalues.
    # X = X + torch.diag(torch.ones(9)).unsqueeze(0).to(X)*AUGMENTATION_CONST
    # _, V = RobustEighFunction.apply(X.double(), EIGH9x9_BWD_ORDER, AUGMENTATION_CONST)
    # V = V.to(X)
    _, _, V = _torch_svd_cast(X)  # torch.svd
    null_ = V[:, :, -4:]  # the last four rows
    nullSpace = V.transpose(-1, -2)[:, -4:, :]

    coeffs = zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)

    def fun(i: int, j: int) -> torch.Tensor:
        return null_[:, 3 * j + i]

    # Determinant constraint
    coeffs[:, 9] = (
        solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 1), fun(1, 2)) - solvers.multiply_deg_one_poly(fun(0, 2), fun(1, 1)),
            fun(2, 0),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 2), fun(1, 0)) - solvers.multiply_deg_one_poly(fun(0, 0), fun(1, 2)),
            fun(2, 1),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 0), fun(1, 1)) - solvers.multiply_deg_one_poly(fun(0, 1), fun(1, 0)),
            fun(2, 2),
        )
    )

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (Eqn. 20 in the paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j] : indices[i, j] + 10] = (
                solvers.multiply_deg_one_poly(fun(i, 0), fun(j, 0))
                + solvers.multiply_deg_one_poly(fun(i, 1), fun(j, 1))
                + solvers.multiply_deg_one_poly(fun(i, 2), fun(j, 2))
            )

    for i in range(10):
        t = 0.5 * (d[:, indices[0, 0] + i] + d[:, indices[1, 1] + i] + d[:, indices[2, 2] + i])
        d[:, indices[0, 0] + i] -= t
        d[:, indices[1, 1] + i] -= t
        d[:, indices[2, 2] + i] -= t

    cnt = 0
    for i in range(3):
        for j in range(3):
            row = (
                solvers.multiply_deg_two_one_poly(d[:, indices[i, 0] : indices[i, 0] + 10], fun(0, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 1] : indices[i, 1] + 10], fun(1, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 2] : indices[i, 2] + 10], fun(2, j))
            )
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs), ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10])) * 10
    )

    eliminated_mat = torch.linalg.solve(coeffs[singular_filter, :, :10], b[singular_filter])

    coeffs_ = torch.cat((coeffs[singular_filter, :, :10], eliminated_mat), dim=-1)

    A = zeros(coeffs_.shape[0], 3, 13, device=coeffs_.device, dtype=coeffs_.dtype)

    for i in range(3):
        A[:, i, 0] = 0.0
        A[:, i : i + 1, 1:4] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 10:13]
        A[:, i : i + 1, 0:3] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 10:13]
        A[:, i, 4] = 0.0
        A[:, i : i + 1, 5:8] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 13:16]
        A[:, i : i + 1, 4:7] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 13:16]
        A[:, i, 8] = 0.0
        A[:, i : i + 1, 9:13] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 16:20]
        A[:, i : i + 1, 8:12] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 16:20]

    cs = solvers.determinant_to_polynomial(A)
    E_models = []

    # for loop because of different numbers of solutions
    for bi in range(A.shape[0]):
        A_i = A[bi]
        null_i = nullSpace[bi]

        # companion matrix solver for polynomial
        C = zeros((10, 10), device=cs.device, dtype=cs.dtype)
        C[0:-1, 1:] = eye(C[0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
        C[-1, :] = -cs[bi][:-1] / cs[bi][-1]

        roots = torch.real(torch.linalg.eigvals(C))

        if roots is None:
            continue
        n_sols = roots.size()
        if n_sols == 0:
            continue
        Bs = stack(
            (
                A_i[:3, :1] * (roots**3) + A_i[:3, 1:2] * roots.square() + A_i[0:3, 2:3] * roots + A_i[0:3, 3:4],
                A_i[0:3, 4:5] * (roots**3) + A_i[0:3, 5:6] * roots.square() + A_i[0:3, 6:7] * roots + A_i[0:3, 7:8],
            ),
            dim=0,
        ).transpose(0, -1)

        bs = (
            A_i[0:3, 8:9] * (roots**4)
            + A_i[0:3, 9:10] * (roots**3)
            + A_i[0:3, 10:11] * roots.square()
            + A_i[0:3, 11:12] * roots
            + A_i[0:3, 12:13]
        ).T.unsqueeze(-1)

        # We try to solve using top two rows,
        xzs = Bs[:, 0:2, 0:2].inverse() @ (bs[:, 0:2])

        mask = (abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3).flatten()
        if torch.sum(mask) != 0:
            q, r = torch.linalg.qr(Bs[mask].clone())  #
            xzs[mask] = torch.linalg.solve(r, q.transpose(-1, -2) @ bs[mask])  # [mask]

        # models
        Es = null_i[0] * (-xzs[:, 0]) + null_i[1] * (-xzs[:, 1]) + null_i[2] * roots.unsqueeze(-1) + null_i[3]

        # Since the rows of N are orthogonal unit vectors, we can normalize the coefficients instead
        inv = 1.0 / torch.sqrt((-xzs[:, 0]) ** 2 + (-xzs[:, 1]) ** 2 + roots.unsqueeze(-1) ** 2 + 1.0)
        Es *= inv
        if Es.shape[0] < 10:
            Es = torch.cat(
                (Es.clone(), eye(3, device=Es.device, dtype=Es.dtype).repeat(10 - Es.shape[0], 1).reshape(-1, 9))
            )
        E_models.append(Es)

    # if not E_models:
    #     return torch.eye(3, device=cs.device, dtype=cs.dtype).unsqueeze(0)
    # else:
    return torch.cat(E_models).view(-1, 3, 3).transpose(-1, -2)

def find_essential(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=5`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=5`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(5, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`,
        one model for each batch selected out of ten solutions by Sampson distances.

    """
    E = run_5point(points1, points2, weights).to(points1.dtype)

    # select one out of 10 possible solutions from 5PC Nister solver.
    solution_num = 10
    batch_size = points1.shape[0]

    error = zeros((batch_size, solution_num))

    for b in range(batch_size):
        b_weights = weights[[b]].repeat(10, 1)
        b_error = epi.sampson_epipolar_distance(points1[b], points2[b], E.view(batch_size, solution_num, 3, 3)[b])
        error[b] = (b_weights * b_error).sum(-1)

    KORNIA_CHECK_SHAPE(error, ["f{batch_size}", "10"])

    chosen_indices = torch.argmin(error, dim=-1)
    result = stack([(E.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :] for i in range(batch_size)])

    return result