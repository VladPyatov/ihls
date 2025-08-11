from typing import Union, Tuple, Callable, Optional

import torch as th
import torch.nn as nn

from irrn.functional import ImplicitLinearSystemFunction
from irrn.functional.implicit_nonlinear import ImplicitFunction, ImplicitFunctionJacobianFree
from irrn.operators import LearnableLinearSystemOperatorBase
from irrn.solvers.base import BatchedSolverBase, BatchedLinSysSolverBase
from irrn.utils import MultiVector


class LinearSystemLayer(nn.Module):
    def __init__(self, system: LearnableLinearSystemOperatorBase, solver: BatchedLinSysSolverBase) -> None:
        super(LinearSystemLayer, self).__init__()
        assert isinstance(system, LearnableLinearSystemOperatorBase)
        assert isinstance(solver, BatchedLinSysSolverBase)
        self.system = system
        self.solver = solver

    def forward(self, degraded_images: th.Tensor, *other_args, **system_kwargs) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        Method, which performs a recurrent restoration via iterative regularized quadratic minimization

        :param degraded_images: batch of observations required for restoration
        :param system_kwargs: all other parameters needed to control restoration at each step
        :return: solution of linear system
        """
        self.prepare_for_restoration(**system_kwargs)
        solution = ImplicitLinearSystemFunction.apply(self.system, self.solver, 0, 0,
                                                      degraded_images, *self.system.tensors_for_grad)
        ret = self.system.perform_step(solution, solution)
        return ret

    def prepare_for_restoration(self, **system_kwargs) -> 'LinearSystemLayer':
        """
        This method prepares layer for restoration bases on input kwargs provided to system.

        :param system_kwargs: kwargs required for system preparation for batch with data
        :return: self
        """
        self.solver.clear_state()
        self.system.prepare_for_restoration(**system_kwargs)
        return self

    def residual(self, degraded: th.Tensor, *solution: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method returns residual of linear system in solution point, which is then used to perform backpropagation
        based on implicit function theorem. By default for system Ax = b this method returns r = Ax - b, feel free to
        override this if you need other behaviour.

        :param degraded: batch of observations required for restoration
        :param solution: solution (possibly several tensors), for which the residual (condition) should be computed
        :return: tuple with tensors, that may require gradients computation
        """
        return self.system(MultiVector(solution), degraded) - self.system.right_hand_side(degraded, *solution)

    @property
    def tensors_for_grad(self):
        return self.system.tensors_for_grad

    def to(self, *args, **kwargs) -> 'LinearSystemLayer':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        super(LinearSystemLayer, self).to(*args, **kwargs)
        self.system.to(*args, **kwargs)
        return self


class LinearSystemStepLayer(LinearSystemLayer):
    """
    This layer implements forward step pass through a linear system of equations.
    """
    def forward(self, degraded_images: th.Tensor, *latents: th.Tensor, call_id: Optional[int] = 0, **system_kwargs
                ) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        Method, which performs solution of linear system on some data, assuming that this data was earlier passed to
        .prepare_for_restoration method of that system

        :param degraded_images: batch of observations which need to be restored
        :param latents: latent parameters which may be required for system parametrization in recurrent restoration
        :param call_id: individual call index to distinguish between different layer calls,
                        usually represents step number during restoration
        :param system_kwargs: all other parameters needed to control restoration at each step
        :return: solution of batch of linear systems
        """
        assert isinstance(call_id, int)
        system_step = self.system.prepare_for_step(call_id, degraded_images, latents, **system_kwargs)
        solution = ImplicitLinearSystemFunction.apply(system_step, self.solver, call_id, len(latents),
                                                      degraded_images, *latents, *system_step.tensors_for_grad)
        ret = system_step.perform_step(latents, solution)
        return ret


class ProjectedLinearSystemStepLayer(LinearSystemStepLayer):
    def __init__(self, system: LearnableLinearSystemOperatorBase, solver: BatchedLinSysSolverBase,
                 projection_fn: Callable) -> None:
        super(ProjectedLinearSystemStepLayer, self).__init__(system, solver)
        self.projection_fn = projection_fn

    def forward(self, degraded_images: th.Tensor, *latents: th.Tensor, call_id: Optional[int] = 0, **system_kwargs
                ) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        Method, which performs solution of linear system on some data and then projects it to some manifold.
        It is  assumed that this data was earlier passed to .prepare_for_restoration method of that system

        :param degraded_images: batch of observations which need to be restored
        :param latents: latent parameters which may be required for system parametrization in recurrent restoration
        :param call_id: individual call index to distinguish between different layer calls,
                        usually represents step number during restoration
        :param system_kwargs: all other parameters needed to control restoration at each step
        :return: solution of batch of linear systems
        """
        ret = super(ProjectedLinearSystemStepLayer, self).forward(degraded_images, *latents, call_id=call_id,
                                                                  **system_kwargs)
        ret = self.projection_fn(call_id, ret)
        return ret


class QMRecurrentLayer(nn.Module):
    system_layer: LinearSystemStepLayer

    def __init__(self, system: LearnableLinearSystemOperatorBase, solver: BatchedLinSysSolverBase, num_steps: int = 10,
                 track_updates_history: bool = True, project_function: Callable = None, start_step_id: int = 0) -> None:
        super(QMRecurrentLayer, self).__init__()
        if project_function is None:
            self.system_layer = LinearSystemStepLayer(system, solver)
        else:
            self.system_layer = ProjectedLinearSystemStepLayer(system, solver, project_function)
        self.num_steps = num_steps
        self.track_updates_history = track_updates_history
        self.start_step_id = start_step_id

    def forward(self, degraded_images: th.Tensor, *latents: th.Tensor, **system_kwargs
                ) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        Method, which performs a recurrent restoration via iterative regularized quadratic minimization

        :param degraded_images: batch of observations required for restoration
        :param latents: latent parameters for first step
        :param system_kwargs: all other parameters needed to control restoration at each step
        :return: restored parameters or their history at all restoration steps
        """
        updates_history = []
        latents_step = latents
        self.system_layer.prepare_for_restoration(**system_kwargs)
        for step in range(self.start_step_id, self.start_step_id + self.num_steps):
            latents_step = self.system_layer(degraded_images, *latents_step, call_id=step, **system_kwargs)
            if self.track_updates_history:
                updates_history.append(latents_step)
        if not self.track_updates_history:
            updates_history = latents_step
        return updates_history

    def to(self, *args, **kwargs) -> 'QMRecurrentLayer':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        super(QMRecurrentLayer, self).to(*args, **kwargs)
        self.system_layer.to(*args, **kwargs)
        return self


class ImplicitLayer(nn.Module):
    def __init__(self, solver_forward: BatchedSolverBase, solver_backward: Optional[BatchedLinSysSolverBase] = None,
                 jacobian_free_backward: bool = False) -> None:
        super(ImplicitLayer, self).__init__()
        assert isinstance(solver_forward, BatchedSolverBase)
        self.solver_forward = solver_forward
        assert isinstance(jacobian_free_backward, bool)
        self.jacobian_free_backward = jacobian_free_backward
        if not self.jacobian_free_backward:
            assert isinstance(solver_backward, BatchedLinSysSolverBase)
            self.solver_backward = solver_backward

    def forward(self, degraded_images: th.Tensor, **fwd_solver_kwargs) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """
        Method, which performs a recurrent restoration via iterative regularized quadratic minimization

        :param degraded_images: batch of observations required for restoration
        :param fwd_solver_kwargs: all other parameters needed to control restoration at each step
        :return: restored parameters or their history at all restoration steps
        """
        self.solver_forward.prepare_for_restoration(**fwd_solver_kwargs)
        if self.jacobian_free_backward:
            solution = ImplicitFunctionJacobianFree.apply(self.solver_forward, degraded_images,
                                                          *self.solver_forward.tensors_for_grad)
        else:
            self.solver_backward.clear_state()
            solution = ImplicitFunction.apply(self.solver_forward, self.solver_backward, degraded_images,
                                              *self.solver_forward.tensors_for_grad)
        return solution

    def to(self, *args, **kwargs) -> 'ImplicitLayer':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        super(ImplicitLayer, self).to(*args, **kwargs)
        self.solver_forward.to(*args, **kwargs)
        return self
