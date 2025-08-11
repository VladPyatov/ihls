from abc import ABC
from typing import Callable, Tuple, Dict, Union, Optional

import torch as th

from irrn.solvers.base import BatchedLinSysSolverBase
from irrn.utils.deepdetach import NonDetachable


# TODO: check whether unified system converges faster
# TODO: Implement wrapper which solves using Woodburry identity trick
class BatchedConjugateGradientSquaredSolver(BatchedLinSysSolverBase, ABC):
    def __init__(self, rtol: float = 1e-3, atol: float = 0., max_iter: Optional[int] = None,
                 restarts_iter: Optional[int] = None, restarts_tol: Optional[float] = None,
                 rtol_backward: Optional[float] = None, atol_backward: Optional[float] = None,
                 max_iter_backward: Optional[int] = None, restarts_iter_backward: Optional[int] = None,
                 restarts_tol_backward: Optional[float] = None, initialization_forward: Optional[th.Tensor] = None,
                 vector_dims: Tuple[int] = (-1, -2, -3), verbose: bool = False) -> None:
        """
        Initializing Conjugate Gradient solver and its parameters.
        IN THIS REALIZATION IT IS ASSUMED, THAT DIMENSIONS SPECIFIED BY vector_dims REPRESENT VECTOR,
        AND ALL OTHER - BATCHES.

        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_iter: frequency of computing the true residual instead of its recursive approximation
        :param restarts_tol: tolerance which is used to detect "serious breakdown" in convergence and perform a restart
        :param rtol_backward: relative tolerance for early exit in backward, by default the same as rtol
        :param atol_backward: absolute tolerance for early exit in backward, by default the same as atol
        :param max_iter_backward: maximum number of iterations to perform in backward, by default the same as max_iter
        :param restarts_iter_backward: frequency of computing the true residual instead of its recursive approximation
                                       in backward, by default the same as restarts_iter
        :param restarts_tol_backward: tolerance which is used to detect "serious breakdown" in convergence and perform
                                      a restart in backward, by default the same as restarts_tol
        :param initialization_forward: which initialization is used in forward step at first iteration;
            by default else right hand side is used
        :param vector_dims: dimensions, which represent vector; all other dimensions are considered as batches
        :param verbose: flag, which determines whether or not to print status messages
        """
        assert rtol > 0 or atol > 0
        assert isinstance(max_iter, int) or max_iter is None
        assert isinstance(restarts_tol, float) or restarts_tol is None
        assert isinstance(vector_dims, tuple) and len(vector_dims) != 0
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        if restarts_tol is not None:
            self.restarts_tol = restarts_tol
        else:
            self.restarts_tol = 0.
        if restarts_iter:
            self.restarts_iter = restarts_iter
        else:
            self.restarts_iter = float('nan')
        self.vector_dims = vector_dims
        self.verbose = verbose
        self.rtol_backward = rtol_backward if rtol_backward is not None else self.rtol
        self.atol_backward = atol_backward if atol_backward is not None else self.atol
        self.max_iter_backward = max_iter_backward if max_iter_backward is not None else self.max_iter
        self.restarts_iter_backward = restarts_iter_backward if restarts_iter_backward is not None \
            else self.restarts_iter
        self.restarts_tol_backward = restarts_tol_backward if restarts_tol_backward is not None \
            else self.restarts_tol
        self.initialization_forward = initialization_forward
        self.convergence_stats_forward = NonDetachable([])
        self.convergence_stats_backward = NonDetachable([])

    @property
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in forward step
        :return: dict with forward step parameters for solver
        """
        return self.atol, self.rtol, self.max_iter, self.restarts_iter, self.restarts_tol

    @property
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        return self.atol_backward, self.rtol_backward, self.max_iter_backward, \
               self.restarts_iter_backward, self.restarts_tol_backward

    def solve_given_params(
            self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor,
            atol: float, rtol: float, max_iter: int, restarts_iter: int,
            restarts_tol: float, precond_left_inv: Callable = lambda x: x, precond_right_inv: Callable = lambda x: x
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of generic matrix linear systems using the preconditioned CGSquared algorithm.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.
        credit: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_tol: tolerance which is used to detect "serious breakdown" in convergence and perform a restart
        :param precond_matrix: a callable that performs a batch matrix multiplication of the preconditioning matrices
        :param initialization: initial guess for solution, defaults to precond_matrix(right_hand_side)
        """
        # if precond_matrix is None:
        #    def precond_matrix(tensor): return tensor
        if initialization is None:
            x = th.zeros_like(right_hand_side)
            r = right_hand_side.clone()
        else:
            x = initialization.clone()
            r = right_hand_side - matrix(initialization)
        max_iter = self._get_max_iter(max_iter, right_hand_side)
        assert right_hand_side.shape == x.shape

        rhs_norm = self.norm(right_hand_side)
        stopping_matrix = th.max(rtol * rhs_norm, atol * th.ones_like(rhs_norm))
        restarting_matrix = restarts_tol * rhs_norm

        residual_norm = self.norm(r)
        r_tilde = r / residual_norm  # init base vector thus it has a unit norm
        u = r.clone()
        p = r.clone()
        q = r.clone()

        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))
        optimal = False
        num_iter = max_iter
        for k in range(1, max_iter + 1):
            rho = self.prod(r, r_tilde)

            if th.min(th.abs(rho)) == 0:
                if (residual_norm <= stopping_matrix).all():
                    optimal = True
                    break
                else:
                    r_tilde = r / residual_norm

            if k == 1:
                pass
            else:
                beta = rho / rho_prev
                th.addcmul(r, beta, q, out=u)
                th.addcmul(q, beta, p, out=p)
                th.addcmul(u, beta, p, out=p)

            p_hat = th.clone(p)
            v_hat = matrix(p_hat)

            alpha = rho / self.prod(r_tilde, v_hat)
            th.nan_to_num_(alpha, nan=0., posinf=0., neginf=0.)

            th.addcmul(u, alpha, v_hat, value=-1, out=q)

            u_hat = u + q
            th.addcmul(x, alpha, u_hat, out=x)
            q_hat = matrix(u_hat)

            if (k % restarts_iter) == 0:
                r = right_hand_side - matrix(x)
            else:
                th.addcmul(r, alpha, q_hat, value=-1, out=r)

            residual_norm = self.norm(r)
            if self.verbose:
                with th.no_grad():
                    atol_current = th.max(residual_norm)
                    rtol_current = th.max(residual_norm / rhs_norm)
                print(f"{str(k)} | "
                      f"atol: {str.format('{:.3e}', atol_current)}, "
                      f"rtol: {str.format('{:.3e}', rtol_current)} ")

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                num_iter = k
                break

            # check whether we should restart optimization for some elements in the batch
            mask = th.sign(residual_norm - restarting_matrix)
            if (mask > 0).any():
                x = x * (1 - mask) / 2 + th.randn(right_hand_side.shape).to(x.device) * (1 + mask) / 2
                r = right_hand_side - matrix(x)

            rho_prev = rho

        stats = {'converged': optimal, 'num_iter': num_iter,
                 'atol': th.max(residual_norm),
                 'rtol': th.max(residual_norm / rhs_norm)}

        if self.verbose:
            if not optimal:
                print(f"Terminated in {num_iter} steps (reached max_iter).")
            else:
                print(f"Terminated in {num_iter} steps (optimal).")

        print(stats)
        return x, stats

    @staticmethod
    def _print_stats(k: int, tag: str, residual_norm: th.Tensor, rhs_norm: th.Tensor) -> None:
        with th.no_grad():
            atol_current = th.max(residual_norm)
            rtol_current = th.max(residual_norm / rhs_norm)
        print(f"{str(k)}_{tag} | "
              f"atol: {str.format('{:.3e}', atol_current)}, "
              f"rtol: {str.format('{:.3e}', rtol_current)} ")
