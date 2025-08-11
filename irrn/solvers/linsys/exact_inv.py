from typing import Tuple, Union, Callable, Dict

import torch as th

from irrn.solvers import BatchedLinSysSolverBase


class ExactMatrixInverseSolver(BatchedLinSysSolverBase):
    """
    This solver constructs the dense matrix out of the corresponding callable using passes of all basis vectors.
    Only applicable for very small matrices (up to 64x64). If any matrix in a batch is singular, uses pseudo-inverse
    for all matrix within the batch. This realization is valid for a 4-dim batched tensors of shape [B, C, H, W],
    where [C, H, W] are considered as vector dimensions.
    """
    def get_matrix_from_callable(self, matrix_callable: Callable, reference_vec: th.Tensor) -> th.Tensor:
        """
        This method constructs a batch of dense matrices using passes of all basis vectors.

        :param matrix_callable: function, that provides matvec operation with its input
        :param reference_vec: reference vector to be used to extract shape, device and dtype data
        :return: batch of matrices of shape [B, CHW**2, CHW**2]
        """
        shape = reference_vec.shape
        vec = th.zeros(*shape, dtype=reference_vec.dtype, device=reference_vec.device)
        vec_flat = vec.flatten(start_dim=1)
        matrix = []
        for col in range(vec_flat.shape[1]):
            vec_flat[:, col] = 1
            matrix.append(matrix_callable(vec).flatten(start_dim=1))
            vec_flat[:, col] = 0
        matrix = th.stack(matrix, dim=-1)
        return matrix

    def solve_given_params(self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor,
                           precond_left_inv: Callable = lambda x: x, precond_right_inv: Callable = lambda x: x
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of linear systems with any matrices using the th.linalg.solve call.

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: initial guess for solution
        :param precond_left_inv: inverse of left preconditioner for the system
        :param precond_right_inv: inverse of right preconditioner for the system
        """
        assert right_hand_side.dim() == 4
        assert initialization.dim() == 4

        def preconditioned_matrix(vector):
            return precond_left_inv(matrix(precond_right_inv(vector)))

        rhs_norm = self.norm(right_hand_side)
        matrices = self.get_matrix_from_callable(preconditioned_matrix, initialization)
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
            x = x.to(device=initialization.device)
        x = precond_right_inv(x.view_as(initialization))
        atol = self.norm(matrix(x) - right_hand_side)
        rtol = atol / rhs_norm
        stats.update({'num_iter': rhs_flat.shape[-1],
                 'atol': th.max(atol),
                 'rtol': th.max(rtol)})
        if self.verbose:
            print(stats)
        return x, stats

    @property
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in forward step
        :return: dict with forward step parameters for solver
        """
        return ()

    @property
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        return ()
