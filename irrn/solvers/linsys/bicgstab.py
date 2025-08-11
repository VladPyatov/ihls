from typing import Callable, Tuple, Dict, Union, Optional

import torch as th

from irrn.solvers.linsys.cg import BatchedConjugateGradientSolver
from irrn.utils.deepdetach import NonDetachable


# TODO: check whether unified system converges faster
# TODO: Implement wrapper which solves using Woodburry identity trick
class BatchedBiCGStabSolver(BatchedConjugateGradientSolver):
    def __init__(self, rtol: float = 1e-3, atol: float = 0., max_iter: Optional[int] = None,
                 restarts_tol: Optional[float] = None, rtol_backward: Optional[float] = None,
                 atol_backward: Optional[float] = None, max_iter_backward: Optional[int] = None,
                 restarts_tol_backward: Optional[float] = None, initialization_forward: Optional[th.Tensor] = None,
                 vector_dims: Tuple[int] = (-1, -2, -3), verbose: bool = False) -> None:
        """
        Initializing Conjugate Gradient solver and its parameters.
        IN THIS REALIZATION IT IS ASSUMED, THAT DIMENSIONS SPECIFIED BY vector_dims REPRESENT VECTOR,
        AND ALL OTHER - BATCHES.

        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_tol: tolerance which is used to detect "serious breakdown" in convergence and perform a restart
        :param rtol_backward: relative tolerance for early exit in backward, by default the same as rtol
        :param atol_backward: absolute tolerance for early exit in backward, by default the same as atol
        :param max_iter_backward: maximum number of iterations to perform in backward, by default the same as max_iter
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
        self.vector_dims = vector_dims
        self.verbose = verbose
        self.rtol_backward = rtol_backward if rtol_backward is not None else self.rtol
        self.atol_backward = atol_backward if atol_backward is not None else self.atol
        self.max_iter_backward = max_iter_backward if max_iter_backward is not None else self.max_iter
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
        return self.atol, self.rtol, self.max_iter, self.restarts_tol

    @property
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        return self.atol_backward, self.rtol_backward, self.max_iter_backward, self.restarts_tol_backward

    def solve_given_params(self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor, atol: float,
                           rtol: float, max_iter: int, restarts_tol: float, precond_left_inv: Callable = lambda x: x,
                           precond_right_inv: Callable = lambda x: x
                           ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, bool, int]]]:
        """
        Solves a batch of generic matrix linear systems using the preconditioned BiCGStab algorithm.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.
        credit: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
        credit: http://www.math.utep.edu/Faculty/xzeng/2017spring_math5330/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: initialization point to start solution from
        :param atol: absolute tolerance of residual norm for early exit
        :param rtol: relative tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_tol: tolerance which is used to detect "serious breakdown" in convergence and perform a restart
        """
        max_iter = self._get_max_iter(max_iter, right_hand_side)

        def preconditioned_matrix(vector):
            return precond_left_inv(matrix(precond_right_inv(vector)))

        right_hand_side = precond_left_inv(right_hand_side)
        r0 = right_hand_side - preconditioned_matrix(initialization)
        x = initialization.clone()
        if hasattr(precond_right_inv, 'inv'):
            x = precond_right_inv.inv(x)
        else:
            self._warn_about_precond('right')
        assert right_hand_side.shape == x.shape

        rhs_norm = self.norm(right_hand_side)
        stopping_matrix = th.max(rtol * rhs_norm, atol * th.ones_like(rhs_norm))

        s = th.zeros_like(right_hand_side)
        p = r0.clone()
        r = r0.clone()
        numerator = self.prod(r, r0)
        restarts_abs = th.zeros_like(numerator)
        restarts_bool = th.zeros_like(numerator, dtype=th.bool)

        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))
        optimal = False
        for k in range(1, max_iter + 1):
            matrix_p = preconditioned_matrix(p)
            alpha = numerator/self.prod(matrix_p, r0)
            th.nan_to_num_(alpha, nan=0., posinf=0., neginf=0.)
            th.addcmul(r, alpha, matrix_p, value=-1., out=s)
            th.addcmul(x, alpha, p, value=1., out=x)

            residual_norm = self.norm(s)
            if self.verbose:
                self._print_stats(k, 's', residual_norm, rhs_norm)
            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break

            matrix_s = preconditioned_matrix(s)
            omega = self.prod(matrix_s, s)/self.prod(matrix_s, matrix_s)
            th.nan_to_num_(omega, nan=0., posinf=0., neginf=0.)
            th.addcmul(x, omega, s, value=1., out=x)
            th.addcmul(s, omega, matrix_s, value=-1., out=r)

            residual_norm = self.norm(r)
            if self.verbose:
                self._print_stats(k, 'r', residual_norm, rhs_norm)
            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break

            denominator = numerator
            numerator = self.prod(r, r0)
            beta = numerator/denominator
            beta *= alpha
            beta /= omega
            th.nan_to_num_(beta, nan=0., posinf=0., neginf=0.)
            th.addcmul(p, omega, matrix_p, value=-1., out=p)
            th.addcmul(r, beta, p, value=1., out=p)

            th.abs(numerator, out=restarts_abs)
            th.le(restarts_abs, restarts_tol, out=restarts_bool)
            indices_to_restart = th.nonzero(restarts_bool)[:, 0]
            for idx in indices_to_restart:
                r0[idx] = r[idx]
                p[idx] = r[idx]
                numerator[idx] = self.prod(r[idx], r0[idx])
        x = precond_right_inv(x)
        stats = {'converged': optimal, 'num_iter': k,
                 'atol': th.max(residual_norm),
                 'rtol': th.max(residual_norm/rhs_norm)}
        if self.verbose:
            if not optimal:
                print(f"Terminated in {k} steps (reached max_iter).")
            else:
                print(f"Terminated in {k} steps (optimal).")
        return x, stats

    @staticmethod
    def _print_stats(k: int, tag: str, residual_norm: th.Tensor, rhs_norm: th.Tensor) -> None:
        with th.no_grad():
            atol_current = th.max(residual_norm)
            rtol_current = th.max(residual_norm / rhs_norm)
        print(f"{str(k)}_{tag} | "
              f"atol: {str.format('{:.3e}', atol_current)}, "
              f"rtol: {str.format('{:.3e}', rtol_current)} ")
