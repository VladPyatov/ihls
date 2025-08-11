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
    def forward(ctx, matrix: th.Tensor, tol: float, order: int):
        if matrix.is_cuda:
            u, sigma, v = batch_svd_cuda(matrix, True, tol, MAX_SWEEPS)
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
        return grad_matrix, None, None


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
