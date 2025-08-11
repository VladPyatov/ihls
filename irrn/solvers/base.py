import abc
import warnings
from typing import Tuple, Optional, List, Dict, Callable, Union

import torch as th
from torch import nn

from irrn.utils.multivector import MultiVector
from irrn.operators import LearnableLinearSystemOperatorBase, LinearOperatorBase, AutogradGradLinearOperator
from irrn.utils.deepdetach import NonDetachable


class BatchedSolverBase(nn.Module):
    """
    This class defines the structure, containing both problem as a condition to be satisfied in forward and some solver,
    which solves this problem in forward step. All learned parameters should be presented in problem condition,
    but not in the solver.
    """
    convergence_stats: dict

    @abc.abstractmethod
    def solve(self, degraded: th.Tensor) -> Tuple[th.Tensor]:
        """
        This method performs forward batched solution for given observation.

        :param degraded: batch of observations required for restoration
        :return: solution of linear system as well as dict with convergence results
        """
        pass

    @abc.abstractmethod
    def prepare_for_restoration(self, **kwargs) -> None:
        """
        This method prepares solver for restoration of a specific data batch by assigning some parameters
        based on this data

        :param kwargs: other parameters, which may be required for restoration
        :return: Nothing
        """
        pass

    @abc.abstractmethod
    def residual(self, degraded: th.Tensor, *solution: th.Tensor) -> th.Tensor:
        """
        This method should return residual of solution based on some condition, which is used in backpropagation
        through implicit function theorem

        :param degraded: batch of observations required for restoration
        :param solution: solution, for which the residual (condition) should be computed
        :return: tuple with tensors, that may require gradients computation
        """
        pass

    def residual_with_grad_residual_operator(self, degraded: th.Tensor, *solution: th.Tensor
                                             ) -> Tuple[Union[th.Tensor, MultiVector], LinearOperatorBase]:
        """
        This method computes residual based on given solution and constructs operator (instance of LinearOperatorBase)
        which represents gradient of residual w.r.t. solution. Both residual and grad residual operator are used in
        backpropagation based on implicit function theorem. By default grad residual operator is returned based on
        autograd with identity preconditioners, feel free to override this method if you need other behaviour.

        :param degraded: batch of observations required for restoration
        :param solution: solution, w.r.t. which the gradient of residual should be computed
        :return: linear operator, which represents gradient of residual w.r.t. given solution
        """
        residual = self.residual(degraded, *solution)
        grad_operator = AutogradGradLinearOperator(residual, solution)
        return residual, grad_operator

    @property
    @abc.abstractmethod
    def tensors_for_grad(self) -> Tuple[th.Tensor, ...]:
        """
        This method should return all tensors, which may require gradients computation
        :return: tuple with tensors, that may require gradients computation
        """
        pass

    def clear_state(self) -> None:
        """
        Clears solver state before new restoration.

        :return: Nothing
        """
        pass

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
        raise NotImplementedError


# TODO: implement AlwaysDoubleSolverWrapper to solve always on double
class BatchedLinSysSolverBase:
    initialization_forward: Optional[th.Tensor]
    convergence_stats_forward: List[Dict[str, float]]
    convergence_stats_backward: List[Dict[str, float]]
    vector_dims: Tuple[int]
    verbose: bool
    true_solution_for_tests: Optional[th.Tensor] = None

    def __init__(self, vector_dims: Tuple[int] = (-1, -2, -3), verbose: bool = False) -> None:
        """
        :param vector_dims: dimensions, which represent vector; all other dimensions are considered as batches
        :param verbose: flag, which determines whether or not to print status messages
        """
        assert isinstance(vector_dims, tuple) and len(vector_dims) != 0
        self.verbose = verbose
        self.vector_dims = vector_dims
        self.initialization_forward = None
        self.convergence_stats_forward = NonDetachable([])
        self.convergence_stats_backward = NonDetachable([])

    @property
    @abc.abstractmethod
    def solver_forward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in forward step
        :return: dict with forward step parameters for solver
        """
        pass

    @property
    @abc.abstractmethod
    def solver_backward_params(self) -> Tuple[Union[int, float]]:
        """
        This property should return parameters which are used to call .solve_given_params in backward step
        :return: dict with backward step parameters for solver
        """
        pass

    @abc.abstractmethod
    def solve_given_params(self, matrix: Callable, right_hand_side: th.Tensor, initialization: th.Tensor,
                           *solver_params, precond_left_inv: Callable = lambda x: x,
                           precond_right_inv: Callable = lambda x: x) -> Tuple[th.Tensor, Dict[str, Union[float, int]]]:
        """
        Solves a batch of linear systems.
        This function solves a batch of matrix linear systems of the form
            A_i x_i = b_i,  i=1,...,K,
        where A is a method, which performs batched matrix multiplication of matrices with input
        vectors, x and b are batches of some vectors representing solutions and right hand sides respectively.

        :param matrix: a callable that performs a batch matrix multiplication of linear systems matrices
        :param right_hand_side: tensor of shape [B, ...] representing the right hand sides
        :param initialization: initialization point to start solution from
        :param solver_params: parameters required by the solver, which are provided by either
                              .solver_forward_params or .solver_backward_params
        :param precond_left_inv: left preconditioner for the system
        :param precond_right_inv: right preconditioner for the system
        """
        pass

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
        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        if initialization is None:
            initialization = self.initialization_forward if self.initialization_forward is not None \
                else th.zeros_like(right_hand_side)
        solution, stats = self.solve_given_params(matrix_callable, right_hand_side, initialization,
                                                  *self.solver_forward_params,
                                                  precond_left_inv=linsys.preconditioner_left_inv,
                                                  precond_right_inv=linsys.preconditioner_right_inv)
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

        if right_hand_side is None:
            assert isinstance(linsys, LearnableLinearSystemOperatorBase)
            right_hand_side = linsys.right_hand_side(*linsys_call_params)
        if initialization is None:
            initialization = th.zeros_like(right_hand_side)
        solution, stats = self.solve_given_params(matrix_callable, right_hand_side, initialization,
                                                  *self.solver_backward_params,
                                                  precond_left_inv=linsys.preconditioner_left_inv,
                                                  precond_right_inv=linsys.preconditioner_right_inv)
        stats['id'] = solution_id
        self.convergence_stats_backward.append(stats)
        return solution

    def clear_state(self) -> None:
        """
        Clear initializations before new restoration
        :return:
        """
        self.initialization_forward = None
        self.convergence_stats_forward.clear()
        self.convergence_stats_backward.clear()

    @staticmethod
    def _get_max_iter(max_iter: Optional[int], right_hand_side: th.Tensor) -> int:
        """
        This auxiliary method returns maximum number of iterations to perform to achieve solution.
        If this number was specified by user, it is returned as-is. Else the dimension of right hand side is used.

        :param max_iter: maximum number of iterations which may be was specified by user
        :param right_hand_side: right hand side of a system to get its dimensionality
        :return: maximum number of iterations to perform in iterative method
        """
        if max_iter is None:
            shape = right_hand_side.shape
            if isinstance(shape, th.Size):
                max_iter = shape[-3:].numel()
            else:
                max_iter = 0
                for s in shape:
                    max_iter += s[-3:].numel()
        return max_iter

    def prod(self, vector_1: th.Tensor, vector_2: th.Tensor) -> th.Tensor:
        """
        Auxiliary method, which computes dot product between two vectors in dimensions, specified by self.vector_dims.

        :param vector_1: first (left) component of dot product
        :param vector_2: second (right) component of dot product
        :return: result of dot product between input vectors batches
        """
        return (vector_1 * vector_2).sum(dim=self.vector_dims, keepdim=True)

    def norm(self, vector: th.Tensor) -> th.Tensor:
        """
        Auxiliary method, which computes L2 norm of vector in dimensions, specified by self.vector_dims.

        :param vector: batch of vectors, which norm should be computed
        :return: norm of input batch of vectors
        """
        return vector.pow(2).sum(dim=self.vector_dims, keepdim=True).pow_(0.5)

    @staticmethod
    def _warn_about_precond(side: str):
        warnings.warn(f'Operator representing inverse {side} preconditioner does not have inverse operation implemented. '
                      'It means, that for preconditioned system P_l^{-1} A P_r^{-1} \hat{x} = \hat{b} wrong '
                      'initialization \hat{x}_{init} = x_{init} might be used instead of true one '
                      '\hat{x}_{init} = P_r x_{init} = (P_r^{-1})^{-1} x_{init}.')
