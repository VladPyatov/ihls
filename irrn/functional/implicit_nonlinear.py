from typing import Tuple, Any

import torch as th
from torch.autograd import Function

#from irrn.solvers.base import BatchedSolverBase, BatchedLinSysSolverBase
from irrn.utils import deep_detach
from irrn.utils.multivector import MultiVector

BatchedSolverBase = Any
BatchedLinSysSolverBase = Any


class ImplicitFunction(Function):
    """
    This class implements forward and backward pass through Implicit Layer
    """
    @staticmethod
    def forward(ctx, forward_solver: BatchedSolverBase, backward_linsys_solver: BatchedLinSysSolverBase,
                degraded: th.Tensor, *tensor_args) -> Tuple[th.Tensor]:
        """
        This method performs a forward step, i.e. a solution based on some solver.

        :param ctx: context of PyTorch Function
        :param forward_solver: object which represents solver of a problem;
                               should have .residual method implemented in order to perform backward
        :param backward_linsys_solver: class which is used to solve batched linear system in backward step
        :param degraded: degraded object, that we opt to restore
        :param tensor_args: tensors, for which gradients will be computed
        :return: solution of linear system at current step
        """
        if any(ctx.needs_input_grad):
            forward_solver_detached, degraded_detached = deep_detach((forward_solver, degraded))
        else:
            forward_solver_detached, degraded_detached = forward_solver, degraded

        solution_tuple = forward_solver_detached.solve(degraded_detached)

        if any(ctx.needs_input_grad):
            ctx.forward_solver = forward_solver_detached
            ctx.degraded = degraded_detached
            ctx.backward_linsys_solver = backward_linsys_solver
            ctx.num_solution_elements = len(solution_tuple)
            ctx.save_for_backward(*solution_tuple, degraded, *tensor_args)
        return solution_tuple

    @staticmethod
    def backward(ctx, *grad_outputs: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method performs backward step for tensors, given in forward as degraded and tensor_args.

        :param ctx: context of PyTorch Function
        :param grad_outputs: gradients, coming from the right during backward
        :return: gradients of tensors, given as args in forward
        """
        solution = ctx.saved_tensors[:ctx.num_solution_elements]
        degraded = ctx.degraded
        forward_solver = ctx.forward_solver
        linsys_solver = ctx.backward_linsys_solver

        needs_grad_state = ctx.needs_input_grad[2:]
        ret = (None,) * 2

        tensors_for_grad = (degraded, *forward_solver.tensors_for_grad)
        tensors_require_grad = tuple([param for param in tensors_for_grad if param.requires_grad])
        assert len(needs_grad_state) == len(tensors_for_grad), \
            'Number of input tensors in forward which may need gradients computation is not equal to ' \
            'number of problem solver params. It means, that backpropagation may be missed ' \
            'for some problem solver parameters, which is unsupported.'
        assert sum(needs_grad_state) == len(tensors_require_grad), \
            'Number of input tensors which need gradients computation is not equal to number of tensors with ' \
            'requires_grad tag in degraded and problem solver object. It means, that for some tensors this tag ' \
            'was changed somewhere in between forward and backward passes, which is unsupported behaviour.'
        assert isinstance(grad_outputs, tuple)

        if any(needs_grad_state):
            solution_with_grads_enabled = tuple([elem.detach().requires_grad_(True) for elem in solution])
            with th.enable_grad():
                residual, grad_r_operator = \
                    forward_solver.residual_with_grad_residual_operator(degraded, *solution_with_grads_enabled)

            rhs = MultiVector(grad_outputs)
            grads = linsys_solver.solve_in_backward(grad_r_operator, right_hand_side=rhs, solution_id=0)

            grads = th.autograd.grad(residual, tensors_require_grad, grad_outputs=grads,
                                     retain_graph=False, create_graph=False, allow_unused=False, only_inputs=True)
            i = 0
            for flag in needs_grad_state:
                if flag:
                    ret += (-grads[i],)
                    i += 1
                else:
                    ret += (None,)
        else:
            ret += (None, ) * len(needs_grad_state)
        return ret


class ImplicitFunctionJacobianFree(Function):
    """
    This class implements forward and backward pass through Implicit Layer using jacobian-free backprop described in
    https://arxiv.org/pdf/2103.12803.pdf
    """
    @staticmethod
    def forward(ctx, forward_solver: BatchedSolverBase, degraded: th.Tensor, *tensor_args) -> Tuple[th.Tensor]:
        """
        This method performs a forward step, i.e. a solution based on some solver.

        :param ctx: context of PyTorch Function
        :param forward_solver: object which represents solver of a problem;
                               should have .perform_iteration method implemented in order to perform backward
        :param degraded: degraded object, that we opt to restore
        :param tensor_args: tensors, for which gradients will be computed
        :return: solution of linear system at current step
        """
        if any(ctx.needs_input_grad):
            forward_solver_detached, degraded_detached = deep_detach((forward_solver, degraded))
        else:
            forward_solver_detached, degraded_detached = forward_solver, degraded

        solution_tuple = forward_solver_detached.solve(degraded_detached)

        if any(ctx.needs_input_grad):
            ctx.forward_solver = forward_solver_detached
            ctx.degraded = degraded_detached
            ctx.num_solution_elements = len(solution_tuple)
            ctx.save_for_backward(*solution_tuple, degraded, *tensor_args)
        return solution_tuple

    @staticmethod
    def backward(ctx, *grad_outputs: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method performs backward step for tensors, given in forward as degraded and tensor_args.

        :param ctx: context of PyTorch Function
        :param grad_outputs: gradients, coming from the right during backward
        :return: gradients of tensors, given as args in forward
        """
        solution = ctx.saved_tensors[:ctx.num_solution_elements]
        degraded = ctx.degraded
        forward_solver: BatchedSolverBase = ctx.forward_solver

        needs_grad_state = ctx.needs_input_grad[1:]
        ret = (None,)

        tensors_for_grad = (degraded, *forward_solver.tensors_for_grad)
        tensors_require_grad = tuple([param for param in tensors_for_grad if param.requires_grad])
        assert len(needs_grad_state) == len(tensors_for_grad), \
            'Number of input tensors in forward which may need gradients computation is not equal to ' \
            'number of problem solver params. It means, that backpropagation may be missed ' \
            'for some problem solver parameters, which is unsupported.'
        assert sum(needs_grad_state) == len(tensors_require_grad), \
            'Number of input tensors which need gradients computation is not equal to number of tensors with ' \
            'requires_grad tag in degraded and problem solver object. It means, that for some tensors this tag ' \
            'was changed somewhere in between forward and backward passes, which is unsupported behaviour.'
        assert isinstance(grad_outputs, tuple)

        if any(needs_grad_state):
            solution_detached = tuple([elem.detach() for elem in solution])
            with th.enable_grad():
                solution_with_grad = forward_solver.perform_iteration(-1, degraded, *solution_detached)
                grads = th.autograd.grad(solution_with_grad, tensors_require_grad, grad_outputs=grad_outputs,
                                         retain_graph=False, create_graph=False, allow_unused=False, only_inputs=True)
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
