from typing import Tuple, Any

import torch as th
from torch.autograd import Function

from irrn.utils import deep_detach, MultiVector
from irrn.operators import LearnableLinearSystemOperatorBase
#from irrn.solvers import BatchedLinSysSolverBase

BatchedLinSysSolverBase = Any

class ImplicitLinearSystemFunction(Function):
    """
    This class implements forward and implicit backward passes through linear system solution.
    """
    @staticmethod
    def forward(ctx, linsys_operator: LearnableLinearSystemOperatorBase, solver: BatchedLinSysSolverBase,
                call_id: int, num_latent_args: int, degraded: th.Tensor, *tensor_args) -> Tuple[th.Tensor]:
        """
        This method performs a forward step, i.e. a solution of linear system.

        :param ctx: context of PyTorch Function
        :param linsys_operator: operator which represents linear system (symmetric matrix and right hand side)
        :param solver: class which is used to solve batched linear system
        :param call_id: individual call index to distinguish between different function calls,
                        usually represents step number during restoration
        :param degraded: degraded object, that we opt to restore
        :param num_latent_args: number of latent arguments, presented in *tensor_args
        :param tensor_args: tensors, for which gradients will be computed, first num_latent_args should be latents
        :return: solution of linear system at current step
        """
        assert len(tensor_args) >= num_latent_args
        if any(ctx.needs_input_grad):
            linsys_operator_detached, degraded_detached, latents_detached = \
                deep_detach((linsys_operator, degraded, tensor_args[:num_latent_args]))
        else:
            linsys_operator_detached, degraded_detached, latents_detached = \
                linsys_operator, degraded, tensor_args[:num_latent_args]

        if len(latents_detached) == 0:
            solver_initialization = None
        else:
            solver_initialization = MultiVector(latents_detached)
        right_hand_side = linsys_operator_detached.right_hand_side(degraded_detached, *latents_detached)
        solution = solver.solve(linsys_operator_detached, degraded, *latents_detached, right_hand_side=right_hand_side,
                                solution_id=call_id, initialization=solver_initialization)
        solution_tuple = MultiVector.to_tuple(solution)

        if any(ctx.needs_input_grad):
            ctx.linsys_operator = linsys_operator_detached
            ctx.degraded = degraded_detached
            ctx.latents = latents_detached
            ctx.solver = solver
            ctx.call_id = call_id
            ctx.num_solution_elements = len(solution_tuple)
            ctx.save_for_backward(*solution_tuple, degraded, *tensor_args)
        return solution_tuple

    @staticmethod
    def backward(ctx, *grad_outputs: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method performs backward step for tensors, given in forward as tensor_args.

        :param ctx: context of PyTorch Function
        :param grad_outputs: gradients, coming from the right during backward
        :return: gradients of tensors, given as args in forward
        """
        solution = ctx.saved_tensors[:ctx.num_solution_elements]
        degraded = ctx.degraded
        latents = ctx.latents
        solver = ctx.solver
        linsys_operator = ctx.linsys_operator

        needs_grad_state = ctx.needs_input_grad[4:]
        ret = (None,) * 4

        tensors_for_grad = (degraded,)
        none_latents_counter = 0
        for latent in latents:
            if latent is not None:
                tensors_for_grad += (latent, )
            else:
                none_latents_counter += 1
        tensors_for_grad += linsys_operator.tensors_for_grad
        tensors_require_grad = tuple([param for param in tensors_for_grad if param.requires_grad])
        assert len(needs_grad_state) - none_latents_counter == len(tensors_for_grad), \
            'Number of input tensors in forward which may need gradients computation is not equal to ' \
            'number of latent and linear system params. It means, that backpropagation may be missed ' \
            'for some linear system parameters, which is unsupported.'
        assert sum(needs_grad_state) == len(tensors_require_grad), \
            'Number of input tensors which need gradients computation is not equal to number of tensors with ' \
            'requires_grad tag in latents and linear system operator. It means, that for some tensors this tag ' \
            'was changed somewhere in between forward and backward passes, which is unsupported behaviour.'
        assert isinstance(grad_outputs, tuple)

        if any(needs_grad_state):
            rhs_grad_outputs = MultiVector(grad_outputs)
            g = solver.solve_in_backward(linsys_operator.T_operator, degraded, *latents,
                                         right_hand_side=rhs_grad_outputs, solution_id=ctx.call_id)
            solution_detached = MultiVector(solution).detach()
            with th.enable_grad():
                r = linsys_operator.residual(solution_detached, degraded, *latents)
                r = MultiVector.to_tuple(r)
                g = MultiVector.to_tuple(g)
                r_for_grad = ()
                g_for_grad = ()
                for r_elem, g_elem in zip(r, g):
                    if r_elem.requires_grad:
                        r_for_grad += (r_elem,)
                        g_for_grad += (g_elem,)

                grads = th.autograd.grad(r_for_grad, tensors_require_grad, grad_outputs=g_for_grad,
                                         retain_graph=False, allow_unused=False, only_inputs=True)

            i = 0
            for flag in needs_grad_state:
                if flag:
                    ret += (grads[i],)
                    i += 1
                else:
                    ret += (None,)
        else:
            ret += (None, ) * len(needs_grad_state)
        return ret
