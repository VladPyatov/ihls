from typing import Callable, Tuple, Dict, Union, Optional

import torch as th

from irrn.solvers.base import BatchedLinSysSolverBase


class BatchedMinResSolver(BatchedLinSysSolverBase):
    def __init__(self, rtol: float = 1e-3, atol: float = 0., max_iter: Optional[int] = None,
                 rtol_backward: Optional[float] = None,
                 atol_backward: Optional[float] = None,
                 max_iter_backward: Optional[int] = None,
                 initialization_forward: Optional[th.Tensor] = None,
                 vector_dims: Tuple[int] = (-1, -2, -3), verbose: bool = False) -> None:
        """
        Initializing MinRes solver and its parameters.
        IN THIS REALIZATION IT IS ASSUMED, THAT DIMENSIONS SPECIFIED BY vector_dims REPRESENT VECTOR,
        AND ALL OTHER - BATCHES.

        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param max_iter: maximum number of iterations to perform
        :param rtol_backward: relative tolerance for early exit in backward, by default the same as rtol
        :param atol_backward: absolute tolerance for early exit in backward, by default the same as atol
        :param max_iter_backward: maximum number of iterations to perform in backward, by default the same as max_iter
        :param initialization_forward: which initialization is used in forward step at first iteration;
            by default else right hand side is used
        :param vector_dims: dimensions, which represent vector; all other dimensions are considered as batches
        :param verbose: flag, which determines whether or not to print status messages
        """
        super(BatchedMinResSolver, self).__init__(vector_dims=vector_dims, verbose=verbose)
        assert rtol >= 0 or atol >= 0
        assert isinstance(max_iter, int) or max_iter is None
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

        self.rtol_backward = rtol_backward if rtol_backward is not None else self.rtol
        self.atol_backward = atol_backward if atol_backward is not None else self.atol
        self.max_iter_backward = max_iter_backward if max_iter_backward is not None else self.max_iter

        self.initialization_forward = initialization_forward
        self.verbose = verbose

    @property
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property returns parameters which are used to call .solve_given_params in forward step
        :return: tuple with forward step parameters for solver
        """
        return self.max_iter, self.atol, self.rtol

    @property
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property returns parameters which are used to call .solve_given_params in backward step
        :return: tuple with backward step parameters for solver
        """
        return self.max_iter_backward, self.atol_backward, self.rtol_backward

    def solve_given_params(self, matrix: Callable,
                           right_hand_side: th.Tensor,
                           initialization: th.Tensor,
                           max_iter: int,
                           atol: float,
                           rtol: float,
                           precond_left_inv: Callable = lambda x: x,
                           precond_right_inv: Callable = lambda x: x
                           ) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of linear systems with symmetric and possibly indefinite matrices using the MinRes algorithm.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of symmetric matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.
        credit https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/minres/minres.py

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: initial guess for solution
        :param max_iter: maximum number of iterations to perform
        :param rtol: relative tolerance of residual norm for early exit
        :param atol: absolute tolerance of residual norm for early exit
        :param precond_left_inv: inverse of left preconditioner for the system
        :param precond_right_inv: inverse of right preconditioner for the system
        """
        def preconditioned_matrix(vector):
            return precond_left_inv(matrix(precond_right_inv(vector)))

        rhs = precond_left_inv(right_hand_side)  # M b - preconditioned right hand side
        rhs_norm = self.norm(rhs)  # norm of preconditioned right hand side
        rhs_norm_noprecond = self.norm(right_hand_side)  # norm of original right hand side
        x = initialization.clone()
        if hasattr(precond_right_inv, 'inv'):
            x = precond_right_inv.inv(x)
        else:
            self._warn_about_precond('right')

        max_iter = self._get_max_iter(max_iter, rhs)
        assert rhs.shape == x.shape

        atol_true = atol
        rtol_true = rtol
        atol_current_true_max = None
        rtol_current_true_max = None
        num_reduced_rtol = 0

        itn = 0
        converged = False

        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))

        # ------------------------------------------------------------------
        # Set up y and v for the first Lanczos vector v1.
        # y  =  beta P' v1,  where  P = C**(-1).
        # v is really P' v1.
        # ------------------------------------------------------------------
        r2 = rhs - preconditioned_matrix(x)
        y = r2.clone()
        beta = self.prod(r2, y)

        #  Test for an indefinite preconditioner.
        #  If b = 0 exactly, stop with x = 0.
        if (beta == 0.0).all():
            converged = True

        if (beta > 0).all():
            beta = th.sqrt(beta)  # Normalize y to get v1 later.

        # -------------------------------------------------------------------
        # Initialize memory containers.
        # ------------------------------------------------------------------
        v = th.empty_like(y)
        beta_oldb = th.empty_like(beta)
        alpha_beta = th.empty_like(beta)
        z = th.empty_like(beta)
        phi = th.empty_like(beta)
        rtol_current = th.empty_like(beta)
        delta = th.empty_like(beta)
        neg_epsln = th.empty_like(beta)
        gbar = th.empty_like(beta)

        # -------------------------------------------------------------------
        # Initialize other quantities.
        # ------------------------------------------------------------------
        oldb = 0.0
        rhs1 = beta.clone()
        rhs2 = th.zeros_like(beta)
        cs = -th.ones_like(beta)
        sn = th.zeros_like(beta)
        epsln = th.zeros_like(beta)
        dbar = th.zeros_like(beta)
        phibar = beta.clone()
        w = th.zeros_like(rhs)
        w_prev = th.zeros_like(rhs)

        # ---------------------------------------------------------------------
        # Main iteration loop.
        # ---------------------------------------------------------------------
        if not converged:            # k = itn = 1 first time through
            while itn < max_iter:
                r1 = r2
                r2 = y

                itn += 1
                th.div(y, beta, out=v)
                y = preconditioned_matrix(v)

                if itn >= 2:
                    th.div(beta, oldb, out=beta_oldb)
                    th.addcmul(y, beta_oldb, r1, value=-1, out=y)
                oldb = beta

                alpha = self.prod(v, y)
                th.div(alpha, beta, out=alpha_beta)
                th.addcmul(y, alpha_beta, r2, value=-1, out=y)

                beta = self.prod(y, y)
                th.sqrt(beta, out=beta)

                # Apply previous rotation Qk-1 to get
                #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
                #   [gbar k dbar k+1]   [sn -cs][alphak betak+1].
                th.mul(cs, dbar, out=delta)
                th.addcmul(delta, sn, alpha, value=1, out=delta)  # delta1 = 0         deltak

                # gbar 1 = alpha1     gbar k
                th.mul(sn, dbar, out=gbar)
                th.addcmul(gbar, cs, alpha, value=-1, out=gbar)
                th.mul(th.neg_(cs), beta, out=dbar)                       # dbar 2 = beta2     dbar k+1

                # Compute the next plane rotation Qk
                gamma = gbar**2
                th.addcmul(gamma, beta, beta, out=gamma)
                th.sqrt_(gamma)                  # gammak
                th.div(gbar, gamma, out=cs)      # ck
                th.mul(cs, phibar, out=phi)      # phik

                th.addcmul(v, w_prev, epsln, value=-1, out=v)
                th.mul(sn, beta, out=epsln)      # epsln2 = 0         epslnk+1
                th.div(beta, gamma, out=sn)      # sk
                th.mul(sn, phibar, out=phibar)   # phibark+1

                # Update  x.
                th.addcmul(v, w, delta, value=-1, out=v)

                w_prev = w.clone()
                th.div(v, gamma, out=w)
                th.nan_to_num_(w, nan=0., posinf=0., neginf=0.)
                th.nan_to_num_(phi, nan=0., posinf=0., neginf=0.)
                th.addcmul(x, phi, w, out=x)

                # Go round again.
                th.div(rhs1, gamma, out=z)

                th.addcmul(rhs2, delta, z, value=-1, out=rhs1)
                th.neg(epsln, out=neg_epsln)
                th.mul(neg_epsln, z, out=rhs2)

                with th.no_grad():
                    atol_current_max = th.max(phibar)
                    th.div(phibar, rhs_norm, out=rtol_current)
                    rtol_current_max = th.max(rtol_current)

                if self.verbose:
                    print(f"{str(itn)} | "
                          f"atol: {str.format('{:.3e}', atol_current_max)}, "
                          f"rtol: {str.format('{:.3e}', rtol_current_max)} ")

                converged = atol_current_max <= atol or rtol_current_max <= rtol
                if converged:
                    atol_current_true = self.norm(right_hand_side - matrix(precond_right_inv(x)))
                    atol_current_true_max = th.max(atol_current_true)
                    rtol_current_true_max = th.max(atol_current_true / rhs_norm_noprecond)
                    converged = atol_current_true_max <= atol_true or \
                                rtol_current_true_max <= rtol_true
                    if converged:
                        break
                    if num_reduced_rtol < 5:
                        atol = atol / 10
                        rtol = rtol / 10
                        if self.verbose:
                            print(f"Actual tols | atol: {str.format('{:.3e}', atol_current_true_max)}, "
                                  f"rtol: {str.format('{:.3e}', rtol_current_true_max)}\n"
                                  f"Tols reduced | new atol: {str.format('{:.3e}', atol)}, "
                                  f"new rtol: {str.format('{:.3e}', rtol)}")
                    num_reduced_rtol += 1

                if itn >= max_iter:
                    break

        x = precond_right_inv(x)
        if not converged:
            atol_current_true = self.norm(right_hand_side - matrix(x))
            atol_current_true_max = th.max(atol_current_true)
            rtol_current_true_max = th.max(atol_current_true / rhs_norm_noprecond)
        stats = {'converged': converged, 'num_iter': itn,
                 'atol': atol_current_true_max,
                 'rtol': rtol_current_true_max}
        if self.verbose:
            print(stats)
        return x, stats
