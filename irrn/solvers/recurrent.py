from typing import Optional, Callable, Tuple, Union

import torch as th
from torch import nn

from irrn.solvers.base import BatchedSolverBase
from irrn.utils.deepdetach import NonDetachable


class BatchedRecurrentSolver(BatchedSolverBase):
    """
    This solver implements a forward pass which finds solution point by recurrent calls of some module.
    """
    def __init__(self, step_module: nn.Module, max_steps_num: Optional[int] = 10, atol: Optional[float] = 1e-5,
                 rtol: Optional[float] = 1e-3, convergence_num_checks: int = 1,
                 initialization_fn: Optional[Callable] = None, vector_dims: Tuple[int] = (-1, -2, -3),
                 verbose: Optional[bool] = False, keep_steps_history: Optional[bool] = False) -> None:
        """
        This method initializes recurrent solver.

        :param step_module: module, which will be called in a recurrent fashion as a recurrent step
        :param max_steps_num: maximum number of recurrent steps (step_module calls)
        :param atol: absolute tolerance of step update for early exit
        :param rtol: relative tolerance of step update for early exit
        :param convergence_num_checks: number of continuous convergence checks that should be satisfied in order
                                       to consider solver to converge for each batch element
        :param initialization_fn: function, that may be used for initialization; if this function is not specified,
                                  initialization will be obtained by calling step_module.initialize
        :param vector_dims: dimensions, which represent vector; all other dimensions are considered as batches
        :param verbose: flag, which determines whether or not to print status messages
        :param keep_steps_history: if True history of all recurrent steps performed for incoming batch is saved and
                                   stored in self.steps_history attribute, which is cleared when a new batch arrives
        """
        super(BatchedRecurrentSolver, self).__init__()
        assert isinstance(step_module, nn.Module)
        assert hasattr(step_module, 'tensors_for_grad'), 'Step module should have method .tensors_for_grad implemented.'
        assert isinstance(max_steps_num, int)
        assert max_steps_num > 0
        assert isinstance(atol, float)
        assert isinstance(rtol, float)
        assert isinstance(verbose, bool)
        assert isinstance(keep_steps_history, bool)
        self.step_module = step_module
        self.max_steps_num = max_steps_num
        self.atol = atol  # threshold for early exit based on absolute tolerance  ||x_k - x_{k+1}||
        self.rtol = rtol  # threshold for early exit based on relative tolerance  ||x_k - x_{k+1}||/||x_{k+1}||

        self.vector_dims = vector_dims
        self.verbose = verbose
        self.keep_steps_history = keep_steps_history
        if initialization_fn is None:
            assert hasattr(step_module, 'initialize'), 'Either function for initialization should be given, ' \
                                                       'or this function should be implemented as a .initialize ' \
                                                       'method of recurrent layer.'
            self.initialization_fn = step_module.initialize
        else:
            self.initialization_fn = initialization_fn
        self.convergence_num_checks = convergence_num_checks
        self.convergence_stats = NonDetachable({})
        self.steps_history = NonDetachable([])

    def solve(self, degraded: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method performs forward batched solution for given observation using iterative calls of underlying module.

        :param degraded: batch of observations required for restoration
        :return: fixed point (solution) found by recurrent solver
        """
        solution = self.get_initial_solution(degraded)
        optimal = False
        step_num = 0
        atols = None
        rtols = None
        converged_counter = None  # counter tracking the number of successive satisfactions of convergence criterion
        converged_global = None  # mask tracking already converged elements of the batch
        atols_final = None  # for each already converged element of a batch
        rtols_final = None  # for each already converged element of a batch
        solution_final = tuple(th.empty_like(elem) for elem in solution)

        if self.keep_steps_history:
            self.steps_history.append(solution)
        if self.verbose:
            print("%03s | %010s" % ("it", "dist"))
        for step_num in range(1, self.max_steps_num + 1):
            solution_next = self.perform_iteration(step_num, degraded, *solution)
            atols, rtols = self.get_tolerances(degraded, solution, solution_next)
            converged_step = (rtols <= self.rtol) | (atols <= self.atol)  # convergence check per step for each element

            if step_num == 1:
                converged_counter = th.zeros(*converged_step.shape, dtype=th.int16, device=converged_step.device)
                converged_global = th.zeros_like(converged_step)
                atols_final = th.ones_like(atols)*float('inf')
                rtols_final = th.ones_like(rtols)*float('inf')

            converged_step |= converged_global
            converged_counter = converged_counter*converged_step + converged_step
            converged = converged_counter >= self.convergence_num_checks
            if (converged_global != converged).any():
                diff = (converged_global == 0) & (converged == 1)
                solution_final = tuple(th.where(diff, solution_next_elem, solution_final_elem) for
                                       (solution_next_elem, solution_final_elem) in zip(solution_next, solution_final))
                atols_final = th.where(diff, atols, atols_final)
                rtols_final = th.where(diff, rtols, rtols_final)
            converged_global = converged

            if self.verbose:
                print(f"{str(step_num)} | "
                      f"atol: {str.format('{:.3e}', th.max(th.where(converged_global, atols_final, atols)))}, "
                      f"rtol: {str.format('{:.3e}', th.max(th.where(converged_global, rtols_final, rtols)))}")
            if self.keep_steps_history:
                self.steps_history.append(solution_next)
            if th.all(converged_global):
                optimal = True
                break
            solution = solution_next
        if not optimal:
            solution_final = tuple(th.where(converged_global, solution_final_elem, solution_elem) for
                                   (solution_final_elem, solution_elem) in zip(solution_final, solution))
            atols_final = th.where(converged_global, atols_final, atols)
            rtols_final = th.where(converged_global, rtols_final, rtols)
        self.convergence_stats.update({'converged': optimal, 'num_steps': step_num, 'atol': th.max(atols_final),
                                       'rtol': th.max(rtols_final)})
        return solution_final

    def get_initial_solution(self, degraded: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method finds starting point to run recurrent solver for given observation.

        :param degraded: batch of observations required for restoration
        :return: initial point for given observation
        """
        return self.initialization_fn(degraded)

    def perform_iteration(self, step_idx: int, degraded: th.Tensor, *latent_args: th.Tensor, **kwargs
                          ) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        This method performs single iteration of solver with given latent estimates

        :param step_idx: step index, which usually represents iteration number during restoration
        :param degraded: batch of observations required for restoration
        :param latent_args: latent arguments to perform next iteration on them
        :param kwargs: other arguments which may be required for iteration
        :return:
        """
        return self.step_module(degraded, *latent_args, call_id=step_idx)

    @th.no_grad()
    def get_tolerances(self, degraded: th.Tensor, solution_prev: Tuple[th.Tensor],
                       solution: Tuple[th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        This method calculates absolute and relative tolerances of current solution estimate.

        :param solution_prev: solution estimate at the previous step
        :param solution: solution estimate at the current step
        :return: absolute and relative tolerances of solution estimate at the current step
        """
        assert isinstance(solution_prev, tuple)
        assert isinstance(solution, tuple)
        assert len(solution_prev) == len(solution)
        residual_norm = 0
        solution_norm = 0
        for elem, elem_next in zip(solution_prev, solution):
            assert isinstance(elem, th.Tensor)
            assert isinstance(elem_next, th.Tensor)
            residual_norm += self.l2_norm_squared(elem - elem_next)
            solution_norm += self.l2_norm_squared(elem_next)
        residual_norm.pow_(0.5)
        solution_norm.pow_(0.5)
        th.div(residual_norm, solution_norm, out=solution_norm)
        return residual_norm, solution_norm

    def residual(self, degraded: th.Tensor, *solution: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method should return residual of solution based on some condition, which is used in backpropagation
        through implicit function theorem.
        By default the difference between solution and self.step_module(solution) is returned as residual, feel free to
        override this.

        :param degraded: batch of observations required for restoration
        :param solution: solution (tuple of tensors), for which the residual (condition) should be computed
        :return: tuple with tensors, that may require gradients computation
        """
        fwd_res = self.perform_iteration(-1, degraded, *solution)
        ret = []
        for r, s in zip(fwd_res, solution):
            ret.append(r - s)
        return tuple(ret)

    def l2_norm_squared(self, vector: th.Tensor) -> th.Tensor:
        """
        Auxiliary method, which computes L2 norm of vector in dimensions, specified by self.vector_dims.

        :param vector: batch of vectors, which norm should be computed
        :return: norm of input batch of vectors
        """
        return vector.pow(2).sum(dim=self.vector_dims, keepdim=True)

    def prepare_for_restoration(self, **kwargs) -> None:
        """
        This method prepares solver for restoration of a specific data batch by assigning some parameters
        based on this data

        :param kwargs: other parameters, which may be required for restoration
        :return: Nothing
        """
        self.clear_state()
        self.step_module.prepare_for_restoration(**kwargs)
        if hasattr(self.initialization_fn, 'prepare_for_restoration'):
            self.initialization_fn.prepare_for_restoration(**kwargs)

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor, ...]:
        """
        This method should return all tensors, which may require gradients computation
        :return: tuple with tensors, that may require gradients computation
        """
        return self.step_module.tensors_for_grad

    def clear_state(self) -> None:
        """
        Clear initializations before new restoration
        :return:
        """
        self.convergence_stats.clear()
        self.steps_history.clear()

    def to(self, *args, **kwargs) -> 'BatchedRecurrentSolver':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        super(BatchedRecurrentSolver, self).to(*args, **kwargs)
        self.step_module.to(*args, **kwargs)
        return self
