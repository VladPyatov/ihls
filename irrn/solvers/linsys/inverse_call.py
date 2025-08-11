from typing import Callable, Tuple, Dict, Union, Optional

import torch as th

from irrn.operators import LinearOperatorBase, LearnableLinearSystemOperatorBase
from irrn.solvers.base import BatchedLinSysSolverBase


class BatchedInverseCallSolver(BatchedLinSysSolverBase):
    """
    This solver solves batch of linear systems Ax = b with .inv attribute implemented for operator A. One-shot solution
    is obtained by x = A^{-1} b = A.inv(b).
    """
    def solve(self, linsys: LinearOperatorBase, *linsys_call_params: th.Tensor,
              right_hand_side: Optional[th.Tensor] = None, initialization: Optional[th.Tensor] = None,
              solution_id: int = None) -> th.Tensor:
        """
        This method performs batched solution of linear system of equations

        :param linsys: linear system object, which implements batch matrix multiplication of linear systems matrices and
                       possibly batch of right hand sides for linear system we opt to solve
        :param linsys_call_params: parameters which are used to call a linear system
        :param right_hand_side: custom right hand side to solve a system with; by default linsys.right_hand_side is used
        :param initialization: initialization point to start the solver from
        :param solution_id: id of current solution to distinguish it from other solutions performed by the same solver
        :return: solution of linear system as well as dict with convergence results
        """
        if self.verbose:
            print('Solving in forward step')

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)

        solution, stats = self.solve_given_params(linsys, right_hand_side, *linsys_call_params)
        stats['id'] = solution_id
        self.convergence_stats_forward.append(stats)
        return solution

    def solve_in_backward(self, linsys: LinearOperatorBase, *linsys_call_params: th.Tensor,
                          right_hand_side: Optional[th.Tensor] = None, initialization: Optional[th.Tensor] = None,
                          solution_id: int = None) -> th.Tensor:
        """
        This method performs batched solution of linear system of equations during backward pass.

        :param linsys: linear system object, which implements batch matrix multiplication of linear systems matrices and
                       possibly batch of right hand sides for linear system we opt to solve
        :param linsys_call_params: parameters which are used to call a linear system
        :param right_hand_side: custom right hand side to solve a system with; by default linsys.right_hand_side is used
        :param initialization: initialization point to start the solver from
        :param solution_id: id of current solution to distinguish it from other solutions performed by the same solver
        :return: solution of linear system as well as dict with convergence results
        """
        if self.verbose:
            print('Solving in backward step')

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)

        solution, stats = self.solve_given_params(linsys, right_hand_side, *linsys_call_params)
        stats['id'] = solution_id
        self.convergence_stats_backward.append(stats)
        return solution

    def solve_given_params(self, linsys_matrix: LinearOperatorBase, right_hand_side: th.Tensor,
                           *linsys_call_params: th.Tensor, initialization: Optional[th.Tensor] = None,
                           precond_left_inv: Optional[Callable] = None, precond_right_inv: Optional[Callable] = None
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of linear systems.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.

        :param linsys_matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param linsys_call_params: parameters with which linear system matrix object should be called
        :param initialization: initialization point to start solution from
        :param precond_left_inv: left preconditioner for the system
        :param precond_right_inv: right preconditioner for the system
        """
        solution = linsys_matrix.inv(right_hand_side, *linsys_call_params)
        r = right_hand_side - linsys_matrix(solution, *linsys_call_params)
        residual_norm = self.norm(r)
        rhs_norm = self.norm(right_hand_side)
        atol = th.max(residual_norm)
        rtol = th.max(residual_norm / rhs_norm)
        if self.verbose:
            print(f"{1} | "
                  f"atol: {str.format('{:.3e}', atol)}, "
                  f"rtol: {str.format('{:.3e}', rtol)} ")
        stats = {'converged': True, 'num_iter': 1,
                 'atol': atol,
                 'rtol': rtol}
        return solution, stats

    @property
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in forward step
        :return: dict with forward step parameters for solver
        """
        return ()

    @property
    def solver_backward_params(self) -> Dict[str, Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        return ()
