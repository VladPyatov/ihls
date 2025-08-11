import abc
from typing import Tuple, Union, Optional, Callable, Dict

from torch import nn
import numpy as np
import torch
import torch as th

from src.utils.svd import (RobustEighFunction, matrix_3x3_rank2_projection, EIGH9x9_BWD_ORDER, AUGMENTATION_CONST,
                           MAX_SWEEPS)

from irrn.functional import ImplicitFunction
from irrn.operators import AutogradGradLinearOperator, LinearOperatorBase
from irrn.solvers import BatchedRecurrentSolver
from irrn.solvers.linsys.exact_inv import ExactMatrixInverseSolver
from irrn.utils.deepdetach import NonDetachable

from .fundamental import normalize_points, normalize_transformation, construct_points_matrix
from .essential import find_essential


def get_masked_tensor(tensor: Union[th.Tensor, type(None)], mask: Union[th.Tensor, type(None)]) -> th.Tensor:
    """
    This method takes tensor elements from its first dimension (supposed to be batches) according to a 1D mask
    :param tensor: input of shape [B, ...] to take the required batch elements
    :param mask: 1D boolean mask of shape [B] which is True for elements of batch that should be taken
    :return: masked tensor of shape [B', ...]
    """
    if mask is not None and tensor is not None:
        b = len(tensor)
        if b != 1:
            assert len(mask) == b
            if th.sum(mask) != b:
                tensor = tensor[mask]
    return tensor


class L0NormWeightsModule(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        if x is None:
            return th.ones(1, 1)
        return (x*x + self.eps).pow(-1)

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = ()
        if isinstance(self.eps, th.Tensor):
            tensors += (self.eps, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        return th.sum(th.log(x*x + self.eps), dim=-1)


class WeightedL0NormWeightsModule(L0NormWeightsModule):
    def __init__(self, norm_weights: Union[float, th.Tensor] = 1., eps: float = 1e-6) -> None:
        super().__init__(eps=eps)
        self.weights = norm_weights

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        if x is None:
            return weights
        return ((x*x + self.eps / weights).pow(-1))

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = super().tensors_for_grad
        if isinstance(self.weights, th.Tensor):
            tensors += (self.weights, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        return th.sum(th.log(x*x + self.eps / weights), dim=-1)


class L1NormWeightsModule(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        if x is None:
            return th.ones(1, 1)
        return 1/th.sqrt(x*x + self.eps)

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = ()
        if isinstance(self.eps, th.Tensor):
            tensors += (self.eps, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        return th.sum(th.sqrt(x*x + self.eps), dim=-1)


class WeightedL1NormWeightsModule(L1NormWeightsModule):
    def __init__(self, norm_weights: Union[float, th.Tensor] = 1., eps: float = 1e-6) -> None:
        super().__init__(eps=eps)
        self.weights = norm_weights

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        ret = super().forward(x, batch_mask=batch_mask).to(weights)
        return weights*ret

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = super().tensors_for_grad
        if isinstance(self.weights, th.Tensor):
            tensors += (self.weights, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        return th.sum(weights*th.sqrt(x*x + self.eps), dim=-1)


class LppNormWeightsModule(nn.Module):
    def __init__(self, p: float = 0.8, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.p = p
        self.last_weights = NonDetachable([None]) # FIXME: always returns None
        self.max_weight = p * (eps ** ((p-2)/2))

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        if x is None:
            self.last_weights[0] = th.ones(1, 1) # TODO: check why (1, 1)
        else:
            self.last_weights[0] = self.p*((x*x + self.eps).pow((self.p - 2)/2))
            # normalization trick
            self.last_weights[0] /= self.max_weight
        return self.last_weights[0]

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = ()
        if isinstance(self.p, th.Tensor):
            tensors += (self.p, )
        if isinstance(self.eps, th.Tensor):
            tensors += (self.eps, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        return th.sum(th.pow(x*x + self.eps, self.p/2), dim=-1)


class GMWeightsModule(nn.Module):
    def __init__(self, p: float = 1, k: float = 1) -> None:
        super().__init__()
        self.p = p
        self.k = k
        self.last_weights = NonDetachable([None]) # FIXME: always returns None
        self.max_weight = 2 / (p * k * k)

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        if x is None:
            self.last_weights[0] = th.ones(1, 1) # TODO: check why (1, 1)
        else:
            denom = (x/self.k)*(x/self.k) + self.p
            self.last_weights[0] = 2 * self.p / (self.k * self.k * denom * denom)
        return self.last_weights[0]

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = ()
        if isinstance(self.p, th.Tensor):
            tensors += (self.p, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        return th.sum((x*x)/(x*x + self.p), dim=-1)


class WeightedLppNormWeightsModule(LppNormWeightsModule):
    def __init__(self, p: float = 0.8, norm_weights: Union[float, th.Tensor] = 1., eps: float = 1e-6) -> None:
        super().__init__(p=p, eps=eps)
        self.weights = norm_weights

    def forward(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        ret = super().forward(x, batch_mask=batch_mask).to(weights)
        return weights*ret

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        tensors = super().tensors_for_grad
        if isinstance(self.weights, th.Tensor):
            tensors += (self.weights, )
        return tensors

    def penalty_function_value(self, x: th.Tensor, batch_mask: Optional[th.Tensor] = None) -> th.Tensor:
        weights = get_masked_tensor(self.weights, batch_mask)
        return th.sum(weights*th.pow(x*x + self.eps, self.p/2), dim=-1)


class BackwardLinearSystemSolver(ExactMatrixInverseSolver):
    """
    This solver constructs the dense matrix out of the corresponding callable using passes of all basis vectors.
    Only applicable for very small matrices (up to 64x64). If any matrix in a batch is singular, uses pseudo-inverse
    for all matrix within the batch. This realization is valid for a 4-dim batched tensors of shape [B, C, H, W],
    where [C, H, W] are considered as vector dimensions.
    """
    def solve_given_params(self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor,
                           precond_left_inv: Callable = lambda x: x, precond_right_inv: Callable = lambda x: x
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of linear systems with any matrices using the th.linalg.solve call.

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right-hand sides
        :param initialization: initial guess for solution
        :param precond_left_inv: inverse of left preconditioner for the system
        :param precond_right_inv: inverse of right preconditioner for the system
        """

        def preconditioned_matrix(vector):
            return precond_left_inv(matrix(precond_right_inv(vector)))

        rhs_norm = self.norm(right_hand_side)
        matrices = self.get_matrix_from_callable(preconditioned_matrix, right_hand_side)
        rhs = precond_left_inv(right_hand_side)
        rhs_flat = rhs.flatten(start_dim=1)
        try:
            x = th.linalg.solve(matrices, rhs_flat)
            stats = {'rank': rhs_flat.shape[-1],
                     'is_singular': False}
        except RuntimeError:
            x, residuals, rank, singular_values = th.linalg.lstsq(matrices.cpu(), rhs_flat.cpu(), driver='gelsd')
            stats = {'rank': th.min(rank),
                     'is_singular': True}
            x = x.to(device=right_hand_side.device)
        x = precond_right_inv(x.view_as(right_hand_side))
        atol = self.norm(matrix(x) - right_hand_side)
        rtol = atol / rhs_norm
        stats.update({'num_iter': rhs_flat.shape[-1], 'atol': th.max(atol), 'rtol': th.max(rtol)})
        if self.verbose:
            print(stats)
        return x, stats


class IRLSStepModule(nn.Module):
    eigh_eps: float = AUGMENTATION_CONST
    eigh_bwd_order: int = EIGH9x9_BWD_ORDER

    def __init__(self, weights_module: nn.Module, initial_solution: th.Tensor = None, extra_info: dict = None):
        super().__init__()
        self.weights_module = weights_module
        self.initial_solution = initial_solution
        self.normal_equation_matrices = None
        self.extra_info = extra_info

    def forward(self, coordinates_matrices: th.Tensor, fundamental_matrix_estimate: th.Tensor, call_id: int = 0,
                batch_mask: Optional[th.Tensor] = None) -> Tuple[th.Tensor]:
        if call_id < 0:
            matrices = self.compute_normal_equation_matrices(coordinates_matrices, fundamental_matrix_estimate)
            solution = self.get_min_eigvec_hermitian_robust(get_masked_tensor(matrices, batch_mask))
        else:
            if self.extra_info is None:
                solution = self.get_min_eigvec_hermitian(get_masked_tensor(self.normal_equation_matrices, batch_mask))
            else:
                solution = find_essential(
                    self.extra_info['p1'], self.extra_info['p2'], self.extra_info['conf'] * self.extra_info['conf'] *
                    self.irls_weights_diagonal.squeeze(-1)).transpose(-1, -2).reshape(-1, 9)
        return (solution, )

    def get_min_eigvec_hermitian(self, matrix: th.Tensor) -> th.Tensor:
        _, v = torch.linalg.eigh(matrix)
        return v[..., 0]

    def get_min_eigvec_hermitian_robust(self, matrix: th.Tensor) -> th.Tensor:
        assert matrix.dim() == 3
        mat_size = matrix.shape[-1]
        identity = th.eye(mat_size, dtype=matrix.dtype, device=matrix.device)
        vals, vecs = RobustEighFunction.apply(matrix + self.eigh_eps * identity, self.eigh_bwd_order, self.eigh_eps)
        return vecs[..., 0]

    def get_min_eigval_hermitian_robust(self, matrix: th.Tensor) -> th.Tensor:
        assert matrix.dim() == 3
        mat_size = matrix.shape[-1]
        identity = th.eye(mat_size, dtype=matrix.dtype, device=matrix.device)
        vals, vecs = RobustEighFunction.apply(matrix + self.eigh_eps * identity, self.eigh_bwd_order, self.eigh_eps)
        return vals[..., 0] - self.eigh_eps

    def clear(self) -> None:
        self.normal_equation_matrices = None

    def compute_normal_equation_matrices(self, coordinates_matrices: th.Tensor, fundamental_matrix_estimate: th.Tensor
                                         ) -> th.Tensor:
        residuals = torch.abs(th.bmm(coordinates_matrices, fundamental_matrix_estimate[..., None])[..., 0])  # |Af|
        self.irls_weights_diagonal = self.weights_module(residuals).unsqueeze(-1)  # W(Af)
        return th.bmm(coordinates_matrices.transpose(-2, -1), self.irls_weights_diagonal * coordinates_matrices)

    def update_normal_equation_matrices(self, coordinates_matrices: th.Tensor, fundamental_matrix_estimate: th.Tensor
                                         ) -> None:
        self.normal_equation_matrices = self.compute_normal_equation_matrices(coordinates_matrices,
                                                                              fundamental_matrix_estimate)

    def initialize(self, coordinates_matrices: th.Tensor) -> Tuple[th.Tensor]:
        self.clear()
        if self.initial_solution is not None:
            solution = (self.initial_solution, )
        else:
            weights = self.weights_module(None).to(coordinates_matrices)
            normal_equation_matrices = \
                th.bmm(coordinates_matrices.transpose(-2, -1), weights.unsqueeze(-1) * coordinates_matrices)
            if self.extra_info is None:
                solution = self.get_min_eigvec_hermitian(normal_equation_matrices)
            else:
                solution = find_essential(self.extra_info['p1'], self.extra_info['p2'], 
                self.extra_info['conf'] * self.extra_info['conf'] * weights).transpose(-1, -2).reshape(-1, 9)
            self.update_normal_equation_matrices(coordinates_matrices, solution)
        return (solution, )

    @property
    def tensors_for_grad(self):
        return self.weights_module.tensors_for_grad

    @th.no_grad()
    def tolerance(self, coordinates_matrices: th.Tensor, fundamental_matrix: th.Tensor):
        self.update_normal_equation_matrices(coordinates_matrices, fundamental_matrix)
        residual = th.bmm(self.normal_equation_matrices, fundamental_matrix[..., None]).squeeze(-1)
        lagrange_multipliers = (fundamental_matrix * residual).sum(dim=-1) / 2
        residual = residual - 2. * lagrange_multipliers.unsqueeze(-1) * fundamental_matrix
        return residual

    def residual(self, coordinates_matrices: th.Tensor, fundamental_matrix: th.Tensor) -> th.Tensor:
        residuals = torch.abs(th.bmm(coordinates_matrices, fundamental_matrix[..., None]))  # |Af|
        irls_weights_diagonal = self.weights_module(residuals[..., 0]).unsqueeze(-1)  # W(Af)
        residual = th.bmm(coordinates_matrices.transpose(-2, -1), irls_weights_diagonal * residuals)[..., 0]
        lagrange_multipliers = (fundamental_matrix * residual).sum(dim=-1) / 2
        residual = residual - 2. * lagrange_multipliers.unsqueeze(-1) * fundamental_matrix
        return residual

    def residual_with_grad_residual_operator(self, coordinates_matrices: th.Tensor, fundamental_matrix: th.Tensor
                                             ) -> Tuple[th.Tensor, LinearOperatorBase]:
        residual = self.residual(coordinates_matrices, fundamental_matrix)
        grad_residual_operator = AutogradGradLinearOperator((residual, ), (fundamental_matrix, ))
        return residual, grad_residual_operator


class SoftRankIRLSStepModuleBase(IRLSStepModule):
    @abc.abstractmethod
    def get_softrank_weight(self, coordinates_matrices: th.Tensor) -> Union[float, th.Tensor, nn.Parameter]:
        """
        :param coordinates_matrices: [B, N, 9]
        :return: float/scalar_tensor/parameter or tensor/parameter with shape [B, 1, 1]
        """
        pass

    @th.no_grad()
    def tolerance(self, G: th.Tensor, f_k: th.Tensor):
        f_mat_size = int(np.round(f_k.shape[1] ** 0.5))
        r_k = th.bmm(G, f_k.unsqueeze(-1)).squeeze(-1)  # alpha_k: [b, N]
        B_k = self.weights_module(r_k)  # [b, N]
        B_k = th.bmm(G.transpose(-2, -1), B_k.unsqueeze(-1) * G)

        F = self.vec2mat(f_k, f_mat_size, n=f_mat_size)
        u_k = self.get_min_eigvec_hermitian(th.bmm(F.transpose(-1, -2), F))
        C_k = th.bmm(u_k.unsqueeze(-1), u_k.unsqueeze(1))  # [b, 9, 1] @ [b, 1, 9] = [b, 9, 9]
        C_k = self.kron(C_k, th.eye(f_mat_size, dtype=C_k.dtype, device=C_k.device).unsqueeze(0))
        beta = self.get_softrank_weight(G)

        self.normal_equation_matrices = B_k + beta*C_k
        residual = th.bmm(self.normal_equation_matrices, f_k.unsqueeze(-1)).squeeze(-1)
        lambda_k = (f_k * residual).sum(dim=-1, keepdim=True) # [B, 1]
        residual = residual - lambda_k * f_k
        return residual

    def initialize(self, G: th.Tensor) -> Tuple[th.Tensor]:
        self.clear()
        if self.initial_solution is not None:
            solution = (self.initial_solution, )
        else:
            weights = self.weights_module(None).to(G)
            B = th.bmm(G.transpose(-2, -1), weights.unsqueeze(-1) * G)
            f_0 = self.get_min_eigvec_hermitian(B)
            solution = (f_0, )

            f_mat_size = int(np.round(f_0.shape[1] ** 0.5))
            r_0 = th.bmm(G, f_0.unsqueeze(-1)).squeeze(-1)  # alpha_k: [b, N]
            B_0 = self.weights_module(r_0)
            B_0 = th.bmm(G.transpose(-2, -1), B_0.unsqueeze(-1) * G)

            F_0 = self.vec2mat(f_0, f_mat_size, n=f_mat_size)
            u_0 = self.get_min_eigvec_hermitian(th.bmm(F_0.transpose(-1, -2), F_0))
            C_0 = th.bmm(u_0.unsqueeze(-1), u_0.unsqueeze(1))  # [b, 9, 1] @ [b, 1, 9] = [b, 9, 9]
            C_0 = self.kron(C_0, th.eye(f_mat_size, dtype=C_0.dtype, device=C_0.device).unsqueeze(0))
            beta = self.get_softrank_weight(G)

            self.normal_equation_matrices = B_0 + beta * C_0

        return solution

    @property
    def tensors_for_grad(self):
        return self.weights_module.tensors_for_grad + tuple(p for p in self.parameters(recurse=False))

    def residual(self, G: th.Tensor, f_star: th.Tensor) -> th.Tensor:
        f_mat_size = int(np.round(f_star.shape[1] ** 0.5))

        r_star = th.bmm(G, f_star.unsqueeze(-1)).squeeze(-1)   # alpha_star: [B, N]
        W_star = self.weights_module(r_star)  # [B, N]
        B_f_star = th.bmm(G.transpose(-2, -1), (W_star * r_star).unsqueeze(-1)).squeeze(-1) # G^T @ W(x_star) @ G @ x_star
        F_star = self.vec2mat(f_star, f_mat_size)

        u_star = self.get_min_eigvec_hermitian_robust(th.bmm(F_star.transpose(-1, -2), F_star))
        C_f_star = th.bmm(u_star.unsqueeze(-1), u_star.unsqueeze(1))  # [B, 9, 1] @ [B, 1, 9] = [B, 9, 9]
        C_f_star = self.kron(C_f_star, th.eye(f_mat_size, dtype=C_f_star.dtype, device=C_f_star.device).unsqueeze(0))
        C_f_star = th.bmm(C_f_star, f_star.unsqueeze(-1)).squeeze(-1)  # [B, 9]
        beta = self.get_softrank_weight(G)
        residual = B_f_star + beta*C_f_star
        lambda_star = (f_star * residual).sum(dim=-1, keepdim=True) # [B, 1]
        residual = residual - lambda_star * f_star
        return residual

    @staticmethod
    def vec2mat(vector: th.Tensor, m: int, n: Optional[int] = None) -> th.Tensor:
        b = vector.shape[0]
        if n is None:
            n = m
        if vector.ndim != 2:
            raise Exception(f"Not a valid batch of vectors, expected input of the shape [{b}, vec_size], but got the "
                            f"input with the shape {vector.shape}")
        return vector.view(b, n, m).transpose(-2, -1)  # column-wise stacking

    @staticmethod
    def kron(a: th.Tensor, b: th.Tensor) -> th.Tensor:
        """
        Kronecker product of matrices a and b with leading batch dimensions.
        Batch dimensions are broadcast. Source: https://gist.github.com/yulkang/4a597bcc5e9ccf8c7291f8ecb776382d
        :param a: torch.Tensor
        :param b: torch.Tensor
        :param: torch.Tensor
        """
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out_size = siz0 + siz1
        return res.view(*out_size)


class SoftRankLearnedWeightIRLSStepModule(SoftRankIRLSStepModuleBase):
    def __init__(self, weights_module: nn.Module, softrank_weight_initial_value: float = 1.) -> None:
        super().__init__(weights_module)
        self.softrank_weight = nn.Parameter(th.ones(1)[0] * softrank_weight_initial_value)

    def get_softrank_weight(self, coordinates_matrices: th.Tensor) -> Union[float, th.Tensor, nn.Parameter]:
        return self.softrank_weight


class AlphasFirstIRLSStepModule(nn.Module):
    eigh_eps: float = AUGMENTATION_CONST
    eigh_bwd_order: int = EIGH9x9_BWD_ORDER

    def __init__(self, weights_module: nn.Module, initial_solution: th.Tensor = None):
        super().__init__()
        self.weights_module = weights_module
        self.initial_solution = initial_solution
        self.normal_equation_matrices = None

    def forward(self, G: th.Tensor, alpha_k: th.Tensor, call_id: int = 0,
                batch_mask: Optional[th.Tensor] = None) -> Tuple[th.Tensor]:
        B_k = self.get_normal_equation_matrix(G, alpha_k, batch_mask=batch_mask)
        u_k = self.get_min_eigvec_hermitian_robust(B_k) if call_id < 0 else self.get_min_eigvec_hermitian(B_k)
        solution = th.bmm(G, u_k.unsqueeze(-1)).squeeze(-1).abs()
        return (solution, )

    def get_final_solution(self, G: th.Tensor, alpha_star: th.Tensor) -> th.Tensor:
        B_star = self.get_normal_equation_matrix(G, alpha_star)
        x_star = self.get_min_eigvec_hermitian_robust(B_star)
        return x_star

    def get_normal_equation_matrix(self, G: th.Tensor, alpha_k: th.Tensor, batch_mask: Optional[th.Tensor] = None
                                   ) -> th.Tensor:
        G = get_masked_tensor(G, batch_mask)
        W_k = self.weights_module(alpha_k, batch_mask=batch_mask)
        B_k = th.bmm(G.transpose(-2, -1), W_k.unsqueeze(-1) * G)
        return B_k

    def get_min_eigvec_hermitian(self, matrix: th.Tensor) -> th.Tensor:
        _, v = th.linalg.eigh(matrix)
        return v[..., 0]

    def get_min_eigvec_hermitian_robust(self, matrix: th.Tensor) -> th.Tensor:
        assert matrix.dim() == 3
        mat_size = matrix.shape[-1]
        identity = th.eye(mat_size, dtype=matrix.dtype, device=matrix.device)
        vals, vecs = RobustEighFunction.apply(matrix + self.eigh_eps * identity, self.eigh_bwd_order, self.eigh_eps)
        return vecs[..., 0]

    def initialize(self, G: th.Tensor) -> Tuple[th.Tensor]:
        if self.initial_solution is not None:
            solution = (self.initial_solution, )
        else:
            # solution = th.rand(2, G.shape[-2], dtype=G.dtype, device=G.device)/2 + 0.5
            solution = th.ones(1, 1, dtype=G.dtype, device=G.device)
        return (solution, )

    @property
    def tensors_for_grad(self):
        return self.weights_module.tensors_for_grad


class BatchedIRLSMMSolver(BatchedRecurrentSolver):
    step_module: IRLSStepModule

    def get_initial_solution(self, degraded: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method finds starting point to run recurrent solver for given observation.

        :param degraded: batch of observations required for restoration
        :return: initial point for given observation
        """
        return self.step_module.initialize(degraded)

    @th.no_grad()
    def get_tolerances(self, coordinates_matrices: th.Tensor, solution_prev: Tuple[th.Tensor],
                       solution: Tuple[th.Tensor], batch_mask: Optional[th.Tensor] = None
                       ) -> Tuple[th.Tensor, th.Tensor]:
        assert isinstance(solution_prev, tuple)
        assert isinstance(solution, tuple)
        assert len(solution_prev) == len(solution)
        solution = solution[0]
        with th.no_grad():
            r = self.step_module.tolerance(coordinates_matrices, solution)
        atol = th.linalg.norm(r, dim=-1, keepdim=True)
        rtol = atol
        return atol, rtol

    def residual_with_grad_residual_operator(self, coordinates_matrices: th.Tensor, fundamental_matrix: th.Tensor
                                             ) -> Tuple[th.Tensor, th.Tensor]:
        return self.step_module.residual_with_grad_residual_operator(coordinates_matrices, fundamental_matrix)


def find_fundamental_irls(
        points1: th.Tensor, points2: th.Tensor, weights: th.Tensor = 1., num_irls_steps: int = 100, p: float = 1.,
        atol=1e-8, rtol=-1., softrank_weight=0.
        ):
    assert points1.shape == points2.shape, (points1.shape, points2.shape)
    assert len(weights.shape) == 2 and weights.shape[1] == points1.shape[1], weights.shape

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    X = construct_points_matrix(points1_norm, points2_norm)  # BxNx9
    ############################################################################################################
    irls_weights_module = WeightedLppNormWeightsModule(p=0.5, norm_weights=weights, eps=1e-6).to(dtype=X.dtype)
    # irls_weights_module = WeightedL1NormWeightsModule(norm_weights=weights, eps=1e-10)
    step_module = IRLSStepModule(irls_weights_module) if softrank_weight == 0. else (
        SoftRankLearnedWeightIRLSStepModule(irls_weights_module, softrank_weight_initial_value=softrank_weight))
    solver = BatchedIRLSMMSolver(step_module, max_steps_num=num_irls_steps,
                                 atol=atol, rtol=rtol, convergence_num_checks=3,
                                 vector_dims=(-1, ), verbose=False, keep_steps_history=False)
    linsys_solver_backward = BackwardLinearSystemSolver(vector_dims=(-1, ), verbose=False)
    F_mat = ImplicitFunction.apply(solver, linsys_solver_backward, X, *solver.tensors_for_grad)[0]
    ############################################################################################################
    irls_weights_module = WeightedL0NormWeightsModule(norm_weights=weights, eps=1e-6).to(dtype=X.dtype)
    # irls_weights_module = WeightedL1NormWeightsModule(norm_weights=weights, eps=1e-10)
    step_module = IRLSStepModule(irls_weights_module) if softrank_weight == 0. else (
        SoftRankLearnedWeightIRLSStepModule(irls_weights_module, softrank_weight_initial_value=softrank_weight))
    solver = BatchedIRLSMMSolver(step_module, max_steps_num=100,
                                 atol=atol, rtol=rtol, convergence_num_checks=3,
                                 vector_dims=(-1, ), verbose=False, keep_steps_history=False)
    linsys_solver_backward = BackwardLinearSystemSolver(vector_dims=(-1, ), verbose=False)
    F_mat = ImplicitFunction.apply(solver, linsys_solver_backward, X, *solver.tensors_for_grad)[0]
    ############################################################################################################
    # make a projection to the "rank 2" space
    F_mat = F_mat.view(-1, 3, 3)
    F_proj = matrix_3x3_rank2_projection(F_mat)
    # perform denormalization
    F_est = transform2.transpose(-2, -1) @ (F_proj @ transform1)
    return normalize_transformation(F_est)


class P_param(torch.nn.Module):
    def __init__(self, p: Union[float, th.Tensor] = 1., dtype: th.dtype = th.float32):
        super().__init__()
        if not isinstance(p, th.Tensor):
            p = th.tensor(p, dtype=dtype)
        self.p = th.nn.Parameter(th.logit(p))

    def forward(self):
        return th.nn.functional.sigmoid(self.p)


class FundamentalMatrixEstimator(nn.Module):
    def __init__(self, p: Union[float, th.Tensor], num_irls_steps: int = 100, atol: float = 1e-8, rtol: float = -1.,
                 eps: float = 1e-6, learn_p: bool = False, initial_p: float = -1, initial_num_irls_steps: int = 100):
        super().__init__()
        self.learn_p = learn_p
        self.p = P_param(p) if self.learn_p else lambda: p
        self.num_irls_steps = num_irls_steps
        self.atol = atol
        self.rtol = rtol
        self.eps = eps
        self.initial_p = initial_p
        self.initial_num_irls_steps = initial_num_irls_steps
        self.rank2_projection = matrix_3x3_rank2_projection

    def irls(self, G: th.Tensor, p: Union[float, th.Tensor], num_steps: int = 100,
             initial_solution: th.Tensor = None, return_stats: bool = False, extra_info: Optional[th.Tensor] = None):
        if p < 0:
            irls_weights_module = GMWeightsModule(p=1e-4, k=1)
        elif p == 0:
            irls_weights_module = L0NormWeightsModule(eps=self.eps)
        else:
            irls_weights_module = LppNormWeightsModule(p=p, eps=self.eps)
        #
        irls_weights_module = irls_weights_module.to(dtype=G.dtype)
        #
        step_module = IRLSStepModule(irls_weights_module, initial_solution, extra_info)
        #
        solver = BatchedIRLSMMSolver(step_module, max_steps_num=num_steps,
                                     atol=self.atol, rtol=self.rtol, convergence_num_checks=3,
                                     vector_dims=(-1, ), verbose=False, keep_steps_history=False)
        #
        linsys_solver_backward = BackwardLinearSystemSolver(vector_dims=(-1, ), verbose=False)
        #
        F_mat = ImplicitFunction.apply(solver, linsys_solver_backward, G, *solver.tensors_for_grad)[0]
        if return_stats:
            return F_mat, solver.convergence_stats, irls_weights_module.last_weights[0]
        else:
            return F_mat

    def forward(self, points1: th.Tensor, points2: th.Tensor, weights: th.Tensor = 1.):
        assert points1.shape == points2.shape, (points1.shape, points2.shape)
        assert len(weights.shape) == 2 and weights.shape[1] == points1.shape[1], weights.shape
        # cast inputs to double
        points1, points2, weights = points1.double(), points2.double(), weights.double()
        # normalize points and estimate observation matrix
        points1_norm, transform1 = normalize_points(points1)
        points2_norm, transform2 = normalize_points(points2)
        A = construct_points_matrix(points1_norm, points2_norm)
        G = weights.unsqueeze(-1) * A
        # compute initial solution if required
        if self.initial_p != -1:
            initial_solution = self.irls(G, self.initial_p, self.initial_num_irls_steps)
        else:
            initial_solution = None
        # compute general solution
        F_mat, statistics, last_weights = self.irls(G, self.p(), self.num_irls_steps, initial_solution, return_stats=True)
        F_mat = F_mat.view(-1, 3, 3)
        # make a projection onto a rank-2 matrix space
        F_proj = self.rank2_projection(F_mat)
        # do a back transformation and normalize result
        F_est = transform2.transpose(-2, -1) @ (F_proj @ transform1)
        F_normalized = normalize_transformation(F_est)
        final_weights = weights * last_weights
        return F_normalized[:, None].float(), statistics, final_weights.float()
