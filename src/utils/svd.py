import warnings

import torch as th

try:
    from fbmd_cuda import batch_svd_cuda, batch_svd_backward, batch_symeig_cuda, batch_symeig_backward, \
        batch_robust_symeig_backward, batch_robust_svd_backward
except (ModuleNotFoundError, ImportError):
    warnings.warn('You do not have fbmd_cuda properly compiled. Consider compiling/recompiling it from '
                  'irrn/extensions/fbmd_cuda. Continuing run in reduced functionality mode.')
from torch.autograd import Function
from typing import Tuple, Union
MAX_SWEEPS = 20  # maximum number of sweeps to use in Jacobi eigenvalue algorithm
AUGMENTATION_CONST = 1e-4
EIGH9x9_BWD_ORDER = 100
EIGH3x3_BWD_ORDER = 20


class SVDFunction(Function):
    """
    This function wraps fast cuda batched singular values decomposition of small matrices from fbmd_cuda
    to be used with PyTorch.
    """
    @staticmethod
    def forward(ctx, matrix: th.Tensor, tol: float):
        if matrix.is_cuda:
            u, sigma, v = batch_svd_cuda(matrix, True, tol, MAX_SWEEPS)
        else:
            u, sigma, v = th.svd(matrix, True, True)
            # u, sigma, v = th.linalg.svd(matrix, False)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(matrix, u, sigma, v)
        return u, sigma, v

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            matrix, u, sigma, v = ctx.saved_tensors
            grad_matrix = batch_svd_backward(grad_outputs, matrix, True, True, u, sigma, v)
        return grad_matrix, None


class RobustSVDFunction(Function):
    """
    This function wraps fast robust cuda batched singular values decomposition of small matrices from fbmd_cuda
    to be used with PyTorch.
    """
    @staticmethod
    def forward(ctx, matrix: th.Tensor, tol: float, order: int, min_eigval_value: Union[float, th.Tensor] = 0) -> Tuple[th.Tensor]:
        if matrix.is_cuda:
            u, sigma, v = batch_svd_cuda(matrix, True, tol, MAX_SWEEPS)
            sigma.clamp_(min=min_eigval_value)
        else:
            u, sigma, v = th.svd(matrix, True, True)
            # u, sigma, v = th.linalg.svd(matrix, False)
        if any(ctx.needs_input_grad):
            ctx.order = order
            ctx.save_for_backward(matrix, u, sigma, v)
        return u, sigma, v

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            matrix, u, sigma, v = ctx.saved_tensors
            grad_matrix = batch_robust_svd_backward(grad_outputs, matrix, True, True, u, sigma, v, ctx.order)
        return grad_matrix, None, None, None


class SymEigFunction(Function):
    """
    This function wraps fast cuda batched symmetric eigen decomposition of small matrices from fbmd_cuda to be used
    with PyTorch.
    """
    @staticmethod
    def forward(ctx, matrix: th.Tensor, tol: float):
        if matrix.is_cuda:
            eigvals, eigvecs = batch_symeig_cuda(matrix, True, True, tol, MAX_SWEEPS)
        else:
            eigvals, eigvecs = th.linalg.eigh(matrix, UPLO='U')
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(matrix, eigvals, eigvecs)
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            matrix, eigvals, eigvecs = ctx.saved_tensors
            grad_matrix = batch_symeig_backward(grad_outputs, matrix, True, True, eigvals, eigvecs)
        return grad_matrix, None


class RobustSymPSDEigFunction(Function):
    """
    This function wraps fast robust cuda batched symmetric eigen decomposition of small positive semidefinite matrices
    from fbmd_cuda in a robust way to be used with PyTorch. To stabilize decomposition and avoid round-off errors
    leading to negative eigenvalues, they are explicitly clamped to min_eigval_value value.
    """
    @staticmethod
    def forward(ctx, matrix: th.Tensor, tol: float, order: int, min_eigval_value: Union[type(None), float, th.Tensor]
                ) -> Tuple[th.Tensor]:
        if min_eigval_value is None:
            min_eigval_value = 0.
        if matrix.is_cuda:
            eigvals, eigvecs = batch_symeig_cuda(matrix, True, True, tol, MAX_SWEEPS)
        else:
            eigvals, eigvecs = th.linalg.eigh(matrix, UPLO='U')
        if isinstance(min_eigval_value, th.Tensor):
            # TODO: replace with inplace version when it is ready: https://github.com/pytorch/pytorch/issues/28329
            eigvals = th.where(eigvals < min_eigval_value, min_eigval_value, eigvals)
        else:
            eigvals.clamp_(min=min_eigval_value)
        if any(ctx.needs_input_grad):
            ctx.order = order
            ctx.save_for_backward(matrix, eigvals, eigvecs)
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            matrix, eigvals, eigvecs = ctx.saved_tensors
            grad_matrix = batch_robust_symeig_backward(grad_outputs, matrix, True, True, eigvals, eigvecs, ctx.order)
        return grad_matrix, None, None, None


class RobustEighFunction(Function):
    """
    This function wraps fast robust cuda batched symmetric eigen decomposition of positive semidefinite matrices
    with a backward implemented in a robust way. To stabilize decomposition and avoid round-off errors
    leading to negative eigenvalues, they are explicitly clamped to min_eigval_value value.
    """
    @staticmethod
    def forward(ctx, matrix: th.Tensor, order: int, min_eigval_value: Union[type(None), float, th.Tensor]
                ) -> Tuple[th.Tensor]:
        if min_eigval_value is None:
            min_eigval_value = 0.
        assert not matrix.isnan().any()
        eigvals, eigvecs = th.linalg.eigh(matrix)
        if isinstance(min_eigval_value, th.Tensor):
            # TODO: replace with inplace version when it is ready: https://github.com/pytorch/pytorch/issues/28329
            eigvals = th.where(eigvals < min_eigval_value, min_eigval_value, eigvals)
        else:
            eigvals.clamp_(min=min_eigval_value)
        if any(ctx.needs_input_grad):
            ctx.order = order
            ctx.save_for_backward(matrix, eigvals, eigvecs)
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            matrix, eigvals, eigvecs = ctx.saved_tensors
            grad_matrix = batch_robust_symeig_backward(grad_outputs, matrix, True, True, eigvals, eigvecs, ctx.order)
        return grad_matrix, None, None, None


def matrix_3x3_rank2_projection(F_mat: th.Tensor):
    FFt = th.bmm(F_mat, F_mat.transpose(-2, -1))
    identity = th.eye(3, dtype=FFt.dtype, device=FFt.device).unsqueeze(0)
    FFt += AUGMENTATION_CONST * identity
    S, U = RobustEighFunction.apply(FFt, EIGH3x3_BWD_ORDER, AUGMENTATION_CONST)
    rank_mask = th.tensor([0., 1., 1.], dtype=U.dtype, device=U.device).unsqueeze(0)
    F_proj = th.bmm(th.bmm(U, rank_mask.unsqueeze(-1)*U.transpose(-2, -1)), F_mat)
    return F_proj

def matrix_3x3_rank2_avg_projection(mat: th.Tensor):
    U, S, V_t = th.linalg.svd(mat.double())
    new_S = th.zeros_like(S)
    new_S[..., 0] = new_S[..., 1] = (S[..., 0] + S[..., 1]) / 2
    mat_proj = th.bmm(th.bmm(U, th.diag_embed(new_S)), V_t).to(mat)
    return mat_proj


class DiagonalPinv(Function):
    r"""Inverts all the elements of the input tensor whose magnitude is less than epsilon.
    If epsilon is set to None then machine resolution (approximate decimal resolution of data type) is used."""
    @staticmethod
    def forward(ctx, x: th.Tensor, epsilon: Union[float, None]) -> th.Tensor:
        if epsilon is None:
            epsilon = th.finfo(x.dtype).resolution

        mask = x.abs() <= epsilon
        y = th.empty_like(x)

        y[mask] = 0.
        y[~mask] = 1/x[~mask]
        ctx.mask = mask
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: th.Tensor) -> th.Tensor:
        mask = ctx.mask
        x, = ctx.saved_tensors
        grad_input = th.empty_like(grad_output)

        grad_input[mask] = grad_output[mask]
        grad_input[~mask] = -grad_output[~mask]/x[~mask]**2

        return grad_input, None


# WIP: everything below
def robust_reduced_svd(matrices: th.Tensor, order: int = 100, augmentation_const: Union[float, th.Tensor] = 1e-4) -> \
        Tuple[th.Tensor, th.Tensor, th.Tensor]:
    if isinstance(augmentation_const, float):
        assert augmentation_const >= 0.
    elif isinstance(augmentation_const, th.Tensor):
        assert augmentation_const.dim() == 1
        assert matrices.dtype == augmentation_const.dtype
        augmentation_const = augmentation_const.unsqueeze(-1)  # B, 1

    b, m, n = matrices.shape
    min_size = min(m, n)
    identity_aug = th.diag_embed(
        th.ones(b, min_size, dtype=matrices.dtype, device=matrices.device) * augmentation_const)

    if n <= m:  # v
        matrix = th.bmm(matrices.transpose(-1, -2).conj(), matrices)  # TODO: replace with .adjoint() for PyTorch>=1.11
    else:  # u
        matrix = th.bmm(matrices, matrices.transpose(-1, -2).conj())  # TODO: replace with .adjoint() for PyTorch>=1.11

    vals, uv = RobustEighFunction.apply(matrix + identity_aug, order, augmentation_const)
    vals = th.sqrt(th.clamp(vals - augmentation_const, th.finfo(vals.dtype).eps))
    vals_inv = DiagonalPinv.apply(vals, None)

    if n <= m:  # v
        vh = uv.transpose(-1, -2).conj()  # TODO: replace with .adjoint() for PyTorch>=1.11
        u = th.bmm(matrices, uv) * vals_inv.unsqueeze(-2)
    else:  # u
        u = uv
        vh = th.bmm(u.transpose(-1, -2).conj(), matrices) * vals_inv.unsqueeze(-1)  # TODO: replace with .adjoint() for PyTorch>=1.11

    return u.flip(-1), vals.flip(-1), vh.flip(-2)


class Test:
    def get_random_matrices(self, b: int, n: int, m: int, dtype):
        return th.rand(b, n, m, dtype=dtype)

    def get_random_matrices_lowrank(self, b, n, m, rank, dtype):
        matrices = self.get_random_matrices(b, n, m, th.float64)
        min_size = min(m, n)
        assert rank < min_size
        mask = th.ones(1, min_size, dtype=matrices.dtype, device=matrices.device)
        mask[:, -(min_size - rank):] = 0.
        u, s, vh = th.linalg.svd(matrices)
        matrices = th.bmm(u, (s*mask).unsqueeze(-1)*vh)
        return matrices.to(dtype=dtype)

    def test_RobustEighFunction(self):
        def symmetrize_and_eigh(matrix):
            matrix = th.matmul(matrix.transpose(-1, -2), matrix)
            return RobustEighFunction.apply(matrix, 100, None)
        x = self.get_random_matrices(2, 9, 9, th.float64)
        x.requires_grad = True
        th.autograd.gradcheck(symmetrize_and_eigh, (x, ))

    def test_robust_reduced_svd(self):
        a = self.get_random_matrices(2, 3, 5, th.float64)
        b = a.clone()
        a.requires_grad = True
        b.requires_grad = True
        ut, st, vht = th.linalg.svd(a, full_matrices=False)
        (ut.abs().sum() + st.abs().sum() + vht.abs().sum()).backward()
        u, s, vh = robust_reduced_svd(b, 100, 1e-4)
        (u.abs().sum() + s.abs().sum() + vh.abs().sum()).backward()
        assert th.allclose(a.grad, b.grad, atol=max(th.finfo(a.dtype).resolution, 1e-8))

    def test_robust_reduced_svd_already_lowrank(self):
        a = self.get_random_matrices_lowrank(2, 3, 3, 2, th.float32)
        b = a.clone()
        a.requires_grad = True
        b.requires_grad = True
        ut, st, vht = th.linalg.svd(a, full_matrices=False)
        (ut.abs().sum() + st.abs().sum() + vht.abs().sum()).backward()
        u, s, vh = robust_reduced_svd(b, 20, 1e-4)
        (u.abs().sum() + s.abs().sum() + vh.abs().sum()).backward()
        assert th.allclose(a.grad, b.grad, atol=max(th.finfo(a.dtype).resolution, 1e-8))
