from typing import Callable, Tuple, Dict, Union, Optional

import torch as th

from irrn.operators import LinearOperatorBase, LearnableLinearSystemOperatorBase, IdentityOperator
from irrn.solvers.base import BatchedLinSysSolverBase


# TODO: check whether unified system converges faster
# TODO: Implement wrapper which solves using Woodburry identity trick
class BatchedConjugateGradientSolver(BatchedLinSysSolverBase):
    def __init__(self, rtol: float = 1e-3, atol: float = 0., max_iter: Optional[int] = None,
                 restarts_iter: Optional[int] = None, rtol_backward: Optional[float] = None,
                 atol_backward: Optional[float] = None, max_iter_backward: Optional[int] = None,
                 restarts_iter_backward: Optional[int] = None, initialization_forward: Optional[th.Tensor] = None,
                 vector_dims: Tuple[int] = (-1, -2, -3), verbose: bool = False) -> None:
        """
        Initializing Conjugate Gradient solver and its parameters.
        IN THIS REALIZATION IT IS ASSUMED, THAT DIMENSIONS SPECIFIED BY vector_dims REPRESENT VECTOR,
        AND ALL OTHER - BATCHES.

        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_iter: frequency of computing the true residual instead of its recursive approximation
        :param rtol_backward: relative tolerance for early exit in backward, by default the same as rtol
        :param atol_backward: absolute tolerance for early exit in backward, by default the same as atol
        :param max_iter_backward: maximum number of iterations to perform in backward, by default the same as max_iter
        :param restarts_iter_backward: frequency of computing the true residual instead of its recursive approximation
            in backward, by default the same as restarts_iter
        :param initialization_forward: which initialization is used in forward step at first iteration;
            by default else right hand side is used
        :param vector_dims: dimensions, which represent vector; all other dimensions are considered as batches
        :param verbose: flag, which determines whether or not to print status messages
        """
        super(BatchedConjugateGradientSolver, self).__init__(vector_dims=vector_dims, verbose=verbose)
        assert rtol >= 0 or atol >= 0
        assert isinstance(max_iter, int) or max_iter is None
        assert isinstance(restarts_iter, int) or restarts_iter is None
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        if restarts_iter:
            self.restarts_iter = restarts_iter
        else:
            self.restarts_iter = float('nan')
        self.rtol_backward = rtol_backward if rtol_backward is not None else self.rtol
        self.atol_backward = atol_backward if atol_backward is not None else self.atol
        self.max_iter_backward = max_iter_backward if max_iter_backward is not None else self.max_iter
        self.restarts_iter_backward = restarts_iter_backward if restarts_iter_backward is not None \
            else self.restarts_iter
        self.initialization_forward = initialization_forward

    @property
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in forward step
        :return: dict with forward step parameters for solver
        """
        return self.atol, self.rtol, self.max_iter, self.restarts_iter

    @property
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        return self.atol_backward, self.rtol_backward, self.max_iter_backward, self.restarts_iter_backward

    def solve_given_params(self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor,
                           atol: float, rtol: float, max_iter: int, restarts_iter: int,
                           precond_left_inv: LinearOperatorBase = IdentityOperator(),
                           precond_right_inv: LinearOperatorBase = IdentityOperator()
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of PD matrix linear systems using the CG algorithm.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of positive definite matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.
        credit: https://github.com/sbarratt/torch_cg

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: initial guess for solution
        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param restarts_iter: frequency of computing the true residual instead of its recursive approximation
        """
        max_iter = self._get_max_iter(max_iter, right_hand_side)

        def preconditioned_matrix(vector):
            return precond_left_inv(matrix(precond_right_inv(vector)), inplace=True)

        rhs_norm = self.norm(right_hand_side)
        stopping_matrix = th.max(rtol * rhs_norm, atol * th.ones_like(rhs_norm))

        right_hand_side = precond_left_inv(right_hand_side)
        x = precond_right_inv.inv(initialization.clone())
        precond_left = precond_left_inv.inv
        assert right_hand_side.shape == x.shape
        r = right_hand_side - preconditioned_matrix(x)
        p = r.clone()
        numerator = self.prod(r, r)
        alpha = th.empty_like(numerator)
        beta = th.empty_like(numerator)
        converged_mask = th.zeros(*rhs_norm.shape, dtype=th.bool, device=r.device)
        zero_scalar = th.zeros(1, dtype=initialization.dtype, device=initialization.device)[0]

        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))
        optimal = False
        for k in range(1, max_iter + 1):
            matrix_p_k = preconditioned_matrix(p)
            denominator = self.prod(p, matrix_p_k)
            # numerator = self.prod(p, r)
            th.div(numerator, denominator, out=alpha)
            th.nan_to_num_(alpha, nan=0., posinf=0., neginf=0.)
            alpha = th.where(converged_mask, zero_scalar, alpha)  # do not update elements that are already converged
            th.addcmul(x, alpha, p, out=x)
            if (k % restarts_iter) == 0:
                r = right_hand_side - preconditioned_matrix(x)
            else:
                th.addcmul(r, matrix_p_k, alpha, value=-1, out=r)

            residual_norm = self.norm(precond_left(r))
            if self.verbose:
                with th.no_grad():
                    atol_current = th.max(residual_norm)
                    rtol_current = th.max(residual_norm/rhs_norm)
                stats_step = f"{str(k)} | atol: {str.format('{:.3e}', atol_current)}, " \
                             f"rtol: {str.format('{:.3e}', rtol_current)}"
                if self.true_solution_for_tests is not None:
                    sol = precond_right_inv(x)
                    norm_true = self.norm(sol - self.true_solution_for_tests)
                    atol_true = th.max(norm_true)
                    rtol_true = th.max(norm_true/self.norm(self.true_solution_for_tests))
                    stats_step = f"{stats_step}, true atol: {str.format('{:.3e}', atol_true)}, " \
                                 f"true rtol: {str.format('{:.3e}', rtol_true)}"
                print(stats_step)

            mask = residual_norm <= stopping_matrix  # keeping track of batch elements, that are already converged
            converged_mask = th.where(mask, mask, converged_mask)
            if (converged_mask).all():
                optimal = True
                break

            denominator = numerator
            numerator = self.prod(r, r)
            # numerator = -self.prod(r, matrix_p_k)
            th.div(numerator, denominator, out=beta)
            th.nan_to_num_(beta, nan=0., posinf=0., neginf=0.)
            th.addcmul(r, beta, p, out=p)

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


class BatchedExtendedConjugateGradientLeastSquaresSolver(BatchedConjugateGradientSolver):
    """
    This solver is designed to solve the extended normal equations of least squares problem of the form:
    A^TA x = A^T b + c
    as it was proposed in https://doi.org/10.1137/19M1288644
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

        def matrix_callable(vector):
            return linsys.apply(vector, *linsys_call_params)

        def matrix_transpose_callable(vector):
            return linsys.T(vector, *linsys_call_params)

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        if initialization is None:
            initialization = self.initialization_forward if self.initialization_forward is not None \
                else th.zeros_like(matrix_transpose_callable(right_hand_side))
        extension_vector = linsys.right_hand_side_extension_vector(*linsys_call_params) if \
            hasattr(linsys, 'right_hand_side_extension_vector') else None
        solution, stats = self.solve_given_params(matrix_callable, matrix_transpose_callable, right_hand_side,
                                                  initialization, extension_vector, *self.solver_forward_params,
                                                  precond_sym_inv=linsys.preconditioner_sym_inv)
        self.initialization_forward = solution.detach()
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

        def matrix_callable(vector):
            return linsys.apply(vector, *linsys_call_params)

        def matrix_transpose_callable(vector):
            return linsys.T(vector, *linsys_call_params)

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        elif hasattr(linsys, 'embed_right_hand_side_for_backward'):
            right_hand_side = linsys.embed_right_hand_side_for_backward(right_hand_side, *linsys_call_params)

        if initialization is None:
            initialization = th.zeros_like(matrix_transpose_callable(right_hand_side))
        extension_vector = linsys.right_hand_side_extension_vector(*linsys_call_params) if \
            hasattr(linsys, 'right_hand_side_extension_vector') else None
        solution, stats = self.solve_given_params(matrix_callable, matrix_transpose_callable, right_hand_side,
                                                  initialization, extension_vector, *self.solver_backward_params,
                                                  precond_sym_inv=linsys.preconditioner_sym_inv)
        stats['id'] = solution_id
        self.convergence_stats_backward.append(stats)
        return solution

    def solve_given_params(self, matrix: Callable, matrix_transpose: Callable, right_hand_side: th.Tensor,
                           initialization: th.Tensor, extension_vector: Union[th.Tensor, type(None)], atol: float,
                           rtol: float, max_iter: int, restarts_iter: int,
                           precond_sym_inv: LinearOperatorBase = IdentityOperator()
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of extended least squares linear systems using the CGLS algorithm.
        This function solves a batch of matrix linear systems of the form
            A_i^T A_i x_i = A_i^T b_i + c_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.
        credit: https://doi.org/10.1137/19M1288644

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param matrix_transpose: a callable that performs a batch transpose matrix multiplication of
                                 linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: tensor of shape [B, ...] representing initial guess for solutions
        :param extension_vector: vector, representing 'c' in extended normal equations A^TA x = A^T b + c
        :param rtol: relative tolerance of full residual norm for early exit
        :param atol: absolute tolerance of full residual norm for early exit
        :param max_iter: upper limit for number of iterations to perform
        :param restarts_iter: frequency of computing the true residual instead of its recursive approximation
        :param precond_sym_inv: preconditioner P^-1, that is applied as symmetric preconditioner:
                                P^-T A^T A P^-1 Px = P^-T A^T b + P^-T c,
        :return: batch of solutions with convergence stats
        """
        def precond_matrix(vector: th.Tensor) -> th.Tensor:
            return matrix(precond_sym_inv(vector))

        def precond_matrix_transpose(vector: th.Tensor) -> th.Tensor:
            return precond_sym_inv.T(matrix_transpose(vector))

        max_iter = self._get_max_iter(max_iter, right_hand_side)  # max number of iterations if not specified
        c = extension_vector if extension_vector is not None else 0.
        rhs_norm_full = self.norm(matrix_transpose(right_hand_side) + c)  # A^T
        stopping_matrix = th.max(rtol * rhs_norm_full, atol * th.ones_like(rhs_norm_full))
        if isinstance(c, th.Tensor):
            c = precond_sym_inv.T(c)  # c = P^-T c_orig

        residual_norm_prev = th.ones_like(rhs_norm_full)*float('inf')
        residual_norm_min = residual_norm_prev.clone()
        converged_mask = residual_norm_prev <= stopping_matrix
        zero_scalar = th.zeros(1, dtype=initialization.dtype, device=initialization.device)[0]

        r = right_hand_side - matrix(initialization)  # r_0
        s = precond_matrix_transpose(r) + c  # s_0
        p = s.clone()  # p_1
        numerator = self.prod(s, s)  # ||s_{k-1}||^2
        alpha = th.empty_like(numerator)
        beta = th.empty_like(numerator)
        k = 0

        x = precond_sym_inv.inv(initialization.clone())  # x_0 = P x_init
        x_min_tol = th.empty_like(x)

        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))
        optimal = False
        for k in range(1, max_iter + 1):
            t = precond_matrix(p)  # t_k = Ap_k
            denominator_t = self.prod(t, t)  # ||t_k||^2 = ||Ap_k||^2
            th.div(numerator, denominator_t, out=alpha)  # alpha_k = ||s_{k-1}||^2 / ||t_k||^2
            th.nan_to_num_(alpha, nan=0., posinf=0., neginf=0.)  # account for instability caused by division
            alpha = th.where(converged_mask, zero_scalar, alpha)  # do not update elements that are already converged
            th.addcmul(x, alpha, p, out=x)  # x_k = x_{k-1} + alpha * p_k
            if (k % restarts_iter) == 0:  # recompute true residual to avoid roundoff errors
                r = right_hand_side - precond_matrix(x)    # in its recurrent updates
                residual_norm_prev = th.ones_like(residual_norm_prev)*float('inf')
            else:
                th.addcmul(r, t, alpha, value=-1, out=r)  # r_k = r_{k-1} - alpha_k * t_k
            s = precond_matrix_transpose(r)
            s += c  # s = A^T r_k + c - true residual of quadratic system

            residual_norm = self.norm(r)
            residual_norm_full = self.norm(precond_sym_inv.inv_T(s))

            if self.verbose:
                with th.no_grad():
                    atol_current = th.max(residual_norm_full)
                    rtol_current = th.max(residual_norm_full / rhs_norm_full)
                stats_step = f"{str(k)} | atol: {str.format('{:.3e}', atol_current)}, " \
                             f"rtol: {str.format('{:.3e}', rtol_current)}, " \
                             f"rnorm diff: {str.format('{:.3e}', th.max(residual_norm_prev - residual_norm))}"
                if self.true_solution_for_tests is not None:
                    sol = precond_sym_inv(x)
                    norm_true = self.norm(sol - self.true_solution_for_tests)
                    atol_true = th.max(norm_true)
                    rtol_true = th.max(norm_true / self.norm(self.true_solution_for_tests))
                    stats_step = f"{stats_step}, true atol: {str.format('{:.3e}', atol_true)}, " \
                                 f"true rtol: {str.format('{:.3e}', rtol_true)}"
                print(stats_step)

            mask = residual_norm < residual_norm_min  # account for cases when iterative process starts to diverge
            residual_norm_min = th.where(mask, residual_norm, residual_norm_min)
            x_min_tol = th.where(mask, x, x_min_tol)

            mask = residual_norm_full <= stopping_matrix  # keeping track of batch elements, that are already converged
            converged_mask = th.where(mask, mask, converged_mask)
            if (converged_mask).all():
                optimal = True
                break

            residual_norm_prev = residual_norm
            denominator = numerator  # ||s_{k-1}||^2
            numerator = self.prod(s, s)  # ||s_{k}||^2
            th.div(numerator, denominator, out=beta)  # beta_k = |s_{k}||^2 / ||s_{k-1}||^2
            th.nan_to_num_(beta, nan=0., posinf=0., neginf=0.)
            th.addcmul(s, beta, p, out=p)  # p_{k+1} = s_k + beta_k*p_k

        x = th.where(converged_mask, x, x_min_tol)
        x = precond_sym_inv(x)
        r = matrix_transpose(right_hand_side - matrix(x))  # calculate final tolerances for stats
        if extension_vector is not None:
            r += extension_vector
        residual_norm_full = self.norm(r)
        atol = th.max(residual_norm_full)
        rtol = th.max(residual_norm_full / rhs_norm_full)

        stats = {'converged': optimal, 'num_iter': k,
                 'atol': atol,
                 'rtol': rtol}
        if self.verbose:
            string = f"Terminated in {k} steps (optimal)" if optimal else f"Terminated in {k} steps (reached max_iter)"
            string = f"{string} | atol: {str.format('{:.3e}', atol)}, rtol: {str.format('{:.3e}', rtol)}"
            print(string)

        return x, stats
