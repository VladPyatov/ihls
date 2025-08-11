import itertools
from typing import Tuple

import cv2
import numpy as np
import pytest
import torch as th
from scipy.sparse.linalg import bicgstab, bicg, LinearOperator, cgs, gcrotmk, lsqr, gmres, lgmres
from torch import nn

from irrn.modules.backbones import ConvBackbone, RegBackbone, L2NormBackbone, KPBackbone
from irrn.operators import LearnableConvolutionOperator, LearnableCNNOperator, LearnableNumberOperator, \
    LinearOperatorBase, IdentityOperator, LearnableKPNConvOperator, LearnableDiagonalOperator
from irrn.operators.degradation.conv_decimate import ConvDecimateLinearDegradationOperator, \
    ConvMosaicLinearDegradationOperator
from irrn.operators.linsys import WienerFilteringSystemOperator
from irrn.operators.linsys.irls import IRLSSystemOperatorHandler
from irrn.operators.linsys.wiener import SRWienerFilteringSystemOperator, DemosaicWienerFilteringSystemOperator
from irrn.solvers import BatchedConjugateGradientSolver, BatchedBiCGStabSolver, BatchedConjugateGradientSquaredSolver
from irrn.solvers.linsys import BatchedMinResSolver
from irrn.solvers.linsys.cg import BatchedExtendedConjugateGradientLeastSquaresSolver
from irrn.solvers.linsys.inverse_call import BatchedInverseCallSolver
from irrn.tests.pykrylov import KrylovMethod, TFQMR, CGS
from irrn.utils import MultiVector
from nl_denoise.modules.linop.composition import LearnableLinearOperatorComposition

SIZES_PARAMS = {'batch_size': [1, 2],
                'channels_number': [1, 2, 6],
                'spatial_dims': ['7-5', '7-7', '8-7', '8-8', '32-32']}
SOLVER_ATOL = 1e-10
SOLVER_RTOL = 1e-9
SOLVER_MAXITER = 250


class CallableLinearOperator(LinearOperatorBase):
    def __init__(self, callable):
        self.callable = callable

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.callable(vector)


class MatrixLinearOperator(LinearOperatorBase):
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.matmul(self.matrix, vector)

    def _transpose(self, vector: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.matmul(self.matrix.transpose(-1, -2), vector)

    @staticmethod
    def matmul(matrix, vector):
        assert vector.ndim == 4
        s = vector.shape
        vector_flat = vector.flatten(start_dim=1).unsqueeze(-1)
        ret = th.bmm(matrix, vector_flat)
        ret = ret.reshape(s)
        return ret


class TestSolverBase:
    __test__ = False
    solver = None
    device = 'cuda'
    dtype = th.float64
    solver_init_params = None
    solution_check_atol = 1e-5
    solution_check_rtol = 1e-3
    noise_scalar = 1e-1
    irls_conv_kernel_size = 3
    irls_noise_std = 1
    irls_circ_precond = False
    irls_reg_images_kernel_size = 3

    def check_matrix_dimensions(self, matrix_size):
        torch_dtype_sizes_dict = {
            th.bool: 1,
            th.uint8: 8,
            th.int8: 8,
            th.int16: 16,
            th.int32: 32,
            th.int64: 64,
            th.float16: 16,
            th.float32: 32,
            th.float64: 64,
            th.complex64: 64,
            th.complex128: 128
        }
        mem = (matrix_size**2)*torch_dtype_sizes_dict[self.dtype]/8/1024/1024/1024
        if mem > 100:
            raise RuntimeError(f'You are trying to allocate {mem} GB of memory. Buy a new RAM!')

    def get_random_matrix(self, batch_size, num_channels, height, width):
        matrix_size = num_channels*(height*width)
        self.check_matrix_dimensions(matrix_size)
        matrix = \
            th.eye(matrix_size, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1) + \
            th.randn(batch_size, matrix_size, matrix_size, dtype=self.dtype, device=self.device)*self.noise_scalar
        return matrix

    def get_random_solution(self, batch_size, num_channels, height, width):
        return th.rand(batch_size, num_channels, height, width, dtype=self.dtype,
                       device=self.device)

    def compare_solutions(self, solution_solver, solution_true, matrix):
        if not th.all(th.isclose(solution_solver, solution_true, atol=self.solution_check_atol,
                                 rtol=self.solution_check_rtol)):
            if matrix is not None:
                cond_number = th.linalg.cond(matrix)
            else:
                cond_number = 'n/a'
            max_atol = th.abs(solution_solver - solution_true).max()
            max_rtol = th.abs((solution_solver - solution_true) / solution_solver).max()
            raise ValueError(
                f'Solutions do not match. Condition number: {cond_number}\nMax atol: {max_atol}\nMax rtol: {max_rtol}\n'
                f'True solution:\n{solution_true}\nSolution from solver:\n{solution_solver}'
                f'\nDiff:\n{solution_solver - solution_true}.')

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims",
                             list(itertools.product(SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'])))
    def test_with_random_matrix(self, batch_size, channels_num, spatial_dims):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        matrix = self.get_random_matrix(batch_size, channels_num, *spatial_dims)
        solution_true = self.get_random_solution(batch_size, channels_num, *spatial_dims)
        solver = self.get_solver_callable()
        system = MatrixLinearOperator(matrix)
        rhs = system(solution_true)
        solution_solver = self.solve_using_operator(solver, system, rhs)
        self.compare_solutions(solution_solver, solution_true, matrix)

    def get_matrix_from_callable(self, matrix_callable, shape):
        matrix_size = shape.numel()
        self.check_matrix_dimensions(matrix_size)
        vec = th.zeros(*shape, dtype=self.dtype, device=self.device)
        vec_flat = vec.flatten(start_dim=1)
        matrix = []
        for col in range(vec_flat.shape[1]):
            vec_flat[:, col] = 1
            matrix.append(matrix_callable(vec).flatten(start_dim=1))
            vec_flat[:, col] = 0
        matrix = th.stack(matrix, dim=-1)
        return matrix

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims,use_precond",
                             list(itertools.product(SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'],
                                                    [False, True])))
    def test_with_irls_grad_matrix(self, batch_size, channels_num, spatial_dims, use_precond):
        solver = self.get_solver_callable()
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        solution_true = self.get_random_solution(batch_size, channels_num, *spatial_dims)
        assert self.irls_conv_kernel_size % 2 == 1
        crop = self.irls_conv_kernel_size // 2
        degraded = th.rand_like(solution_true)[..., crop:-crop, crop:-crop]
        fixed_point = th.rand_like(solution_true, requires_grad=True)
        system, params = self.get_toy_irls_system(batch_size, channels_num)
        system.use_circulant_precond_backward = use_precond
        system = system.prepare_for_restoration(**params)
        system.prepare_for_step(1, degraded, (fixed_point, ))
        r, grad_r_operator = system.residual_with_grad_residual_operator(degraded, fixed_point)
        rhs = grad_r_operator(solution_true)
        solution_solver = self.solve_using_operator(solver, grad_r_operator, rhs)
        matrix_exact = self.get_matrix_from_callable(grad_r_operator.apply, rhs.shape)
        self.compare_solutions(solution_solver, solution_true, matrix_exact)

    @staticmethod
    def get_test_image(is_color: bool = True, size: Tuple[int, int] = (512, 512)):
        image = cv2.imread('files/test_image_512.png')
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        image = th.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)/255
        return image

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims,precond_type",
                             list(itertools.product([1, 2],
                                                    [1],
                                                    ['32-32', '64-64'],
                                                    [None, 'circulant', 'equilibrating'])))
    def test_with_irls_system(self, batch_size, channels_num, spatial_dims, precond_type):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num)
        if precond_type is not None and precond_type[0] == 'c':
            system.use_circulant_precond = True
        elif precond_type is not None and precond_type[0] == 'e':
            system.use_equilibration_precond = True
        solver = self.get_solver_callable()

        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)

        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded)*0.5
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})

        system = system.prepare_for_restoration(**inputs)
        latents = (th.clamp(solution_true + th.randn_like(solution_true)*0.5, 0, 1), )
        system = system.prepare_for_step(1, degraded, latents)
        rhs = system(solution_true, degraded, *latents)
        solver.true_solution_for_tests = solution_true
        solution_solver = self.solve_using_operator(solver, system, rhs, degraded, *latents)
        matrix_exact = self.get_matrix_from_callable(
            lambda x: system.preconditioner_left_inv(system(system.preconditioner_right_inv(x))), rhs.shape)
        # matrix_exact = None
        self.compare_solutions(solution_solver, solution_true, matrix_exact)

    @staticmethod
    def get_reg_backbone():
        bbne = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(16, 8, kernel_size=3, padding=1),
                             nn.ReLU())
        reg_backbone = RegBackbone(bbne)
        return reg_backbone

    def get_toy_irls_system(self, batch_size, num_channels):
        degradation_operator = ConvDecimateLinearDegradationOperator(padding_mode=None)
        kernels = degradation_operator._init_kernels_gaussian(batch_size, 3).to(device=self.device, dtype=self.dtype)
        degradation_operator = degradation_operator.init_with_parameters(kernel=kernels)

        # reg_fidelity = LearnableCNNOperator(
        #     backbone=ConvBackbone(out_features_per_layer=[3], kernel_size_per_layer=[3],
        #                           padding_per_layer=[0], strides_per_layer=[1],
        #                           num_in_features=num_channels), learnable=False, mix_in_channels=False)
        reg_fidelity = IdentityOperator()
        reg_images = LearnableCNNOperator(
            backbone=ConvBackbone(out_features_per_layer=[8], kernel_size_per_layer=[self.irls_reg_images_kernel_size],
                                  padding_per_layer=[0], strides_per_layer=[1],
                                  num_in_features=num_channels), learnable=False, mix_in_channels=False)
        weight_decay = LearnableNumberOperator(th.ones(1) * 1e-4, function=lambda x: x, learnable=False)
        reg_backbone = self.get_reg_backbone()

        system = IRLSSystemOperatorHandler(degradation_operator, reg_fidelity, reg_images, weight_decay, reg_backbone,
                                           zero_step_system=None, use_circulant_precond=self.irls_circ_precond)
        system = system.to(device=self.device, dtype=self.dtype)
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device)*self.irls_noise_std,
                        'kernel': kernels}
        for p in system.parameters():
            p.requires_grad = False
        return system, input_params

    def get_solver_callable(self):
        solver = self.solver(**self.solver_init_params)
        return solver

    @staticmethod
    def eval_solver(solver, matrix_callable, rhs, *matrix_args):
        return solver.solve(matrix_callable, *matrix_args, right_hand_side=rhs)

    def solve_using_operator(self, solver, matrix_op, rhs, *matrix_args):
        return self.eval_solver(solver, matrix_op, rhs, *matrix_args)

    @staticmethod
    def matmul_diag_dummy(x):
        b = x.shape[0][0]
        ret = x.clone()
        for i in range(b):
            ret[i] *= (i + 1)
        return ret

    @staticmethod
    def matmul_diag_dummy_inv(x):
        b = x.shape[0][0]
        ret = x.clone()
        for i in range(b):
            ret[i] /= (i + 1)
        return ret


class TestSciPySolverBase(TestSolverBase):
    __test__ = False
    solver = staticmethod(None)
    device = 'cuda'
    dtype = th.float64
    solver_init_params = {'tol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'maxiter': SOLVER_MAXITER}

    def get_solver_callable(self):
        return lambda A, b: self.solver(A, b, **self.solver_init_params)

    @staticmethod
    def matmul_callable(matrix):
        class LinOpFromMatrix(LinearOperator):
            def __init__(self, mat):
                super(LinOpFromMatrix, self).__init__(mat.dtype, mat.shape)
                self.mat = mat

            def _matvec(self, x):
                return self.mat @ x
        return LinOpFromMatrix(matrix)

    def torch_callable_to_scipy(self, matrix_callable_th, rhs_th):
        torch_to_np_dtype_dict = {
            th.bool: np.bool,
            th.uint8: np.uint8,
            th.int8: np.int8,
            th.int16: np.int16,
            th.int32: np.int32,
            th.int64: np.int64,
            th.float16: np.float16,
            th.float32: np.float32,
            th.float64: np.float64,
            th.complex64: np.complex64,
            th.complex128: np.complex128
        }

        class LinOpFromOperator(LinearOperator):
            def __init__(self, mat, shape_th, dtype_th, device):
                self.vec_shape = shape_th[1:]
                self.batch_size = shape_th[0]
                n = self.vec_shape.numel()
                super(LinOpFromOperator, self).__init__(torch_to_np_dtype_dict[dtype_th], (n, n))
                self.mat = mat
                self.dtype_th = dtype_th
                self.device = device
                self.item_num = None

            def _matvec(self, vector):
                vector_th = \
                    th.from_numpy(vector).to(device=self.device, dtype=self.dtype_th).reshape(1, *self.vec_shape)
                result = matrix_callable_th(vector_th.expand(self.batch_size, -1, -1, -1))[self.item_num]
                return result.flatten().cpu().numpy()

            def __matmul__(self, other):
                return self._matvec(other)

            def prepare_for_item(self, item_num):
                self.item_num = item_num
                return self

        call_obj = LinOpFromOperator(matrix_callable_th, rhs_th.shape, rhs_th.dtype, rhs_th.device)
        return call_obj

    def solve_using_operator(self, solver, matrix_op, rhs, *matrix_args):
        assert rhs.ndim == 4
        rhs_np = rhs.flatten(start_dim=1).cpu().numpy()
        solution_solver = []
        linop_sp = self.torch_callable_to_scipy(matrix_op.apply, rhs)
        for idx, b in enumerate(rhs_np):
            solution = self.eval_solver(solver, linop_sp.prepare_for_item(idx), b)
            solution_solver.append(solution)
        solution_solver = np.stack(solution_solver)
        solution_solver = th.from_numpy(solution_solver).to(dtype=self.dtype, device=self.device)
        solution_solver = solution_solver.reshape(*rhs.shape)
        return solution_solver

    @staticmethod
    def eval_solver(solver, matrix_callable, rhs):
        solution, stats = solver(matrix_callable, rhs)
        print(f'Convergence info: {stats}')
        return solution


class TestCGSolver(TestSolverBase):
    __test__ = True
    solver = BatchedConjugateGradientSolver
    device = 'cuda:7'
    dtype = th.float32
    irls_noise_std = 1/255
    irls_circ_precond = False
    solver_init_params = {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'verbose': True, 'max_iter': SOLVER_MAXITER,
                          'restarts_iter': 50}

    def get_random_matrix(self, batch_size, num_channels, height, width):
        matrix = super(TestCGSolver, self).get_random_matrix(batch_size, num_channels, height, width)
        matrix = matrix.transpose(-1, -2) @ matrix
        return matrix

    @staticmethod
    def get_reg_backbone():
        reg_backbone = L2NormBackbone()
        return reg_backbone

    def test_diag_dummy(self):
        rhs = MultiVector((th.rand(10, 3, 256, 256), th.rand(10, 1, 13, 13)))
        rhs[0] = th.ones_like(rhs[0])
        solver = self.solver(rtol=1e-43, atol=1e-43, verbose=True, max_iter=100)

        solution_true = self.matmul_diag_dummy_inv(rhs)
        init = th.zeros_like(rhs)
        init[0] = solution_true[0]
        system = CallableLinearOperator(self.matmul_diag_dummy)
        solution_cg = solver.solve(system, right_hand_side=rhs, initialization=init)
        assert th.isclose(solution_cg, solution_true).all()


class TestCGLSSolver(TestCGSolver):
    __test__ = True
    solver = BatchedExtendedConjugateGradientLeastSquaresSolver

    def get_random_matrix(self, batch_size, num_channels, height, width):
        matrix = super(TestCGSolver, self).get_random_matrix(batch_size, num_channels, height, width)
        return matrix

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims",
                             list(itertools.product(SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'])))
    def test_with_random_matrix(self, batch_size, channels_num, spatial_dims):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        matrix = self.get_random_matrix(batch_size, channels_num, *spatial_dims)
        solution_true = self.get_random_solution(batch_size, channels_num, *spatial_dims)
        solver = self.get_solver_callable()
        system = MatrixLinearOperator(matrix)
        rhs = system(solution_true)
        solution_solver = self.solve_using_operator(solver, system, rhs)
        self.compare_solutions(solution_solver, solution_true, matrix.transpose(-1, -2) @ matrix)

    def test_with_irls_grad_matrix(self):
        pass

    @pytest.mark.parametrize("batch_size,channels_num,spatial_dims,precond_type",
                             list(itertools.product([1, 2],
                                                    [1],
                                                    ['32-32', '64-64'],
                                                    [None, 'circulant', 'equilibrating'])))
    def test_with_irls_system(self, batch_size, channels_num, spatial_dims, precond_type):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))
        system, inputs = self.get_toy_irls_system(batch_size, channels_num)
        if precond_type is not None and precond_type[0] == 'c':
            system.use_circulant_precond = True
        elif precond_type is not None and precond_type[0] == 'e':
            system.use_equilibration_precond = True
        solver = self.get_solver_callable()

        solution_true = self.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)

        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * 0.5
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})

        system = system.prepare_for_restoration(**inputs)
        latents = (th.clamp(solution_true + th.randn_like(solution_true) * 0.5, 0, 1),)
        system = system.prepare_for_step(1, degraded, latents)
        system_normal = system.normal_equations_system
        system_normal.preconditioner_sym_inv = system.preconditioner_right_inv
        rhs_normal = system_normal(solution_true, degraded, *latents)
        solver.true_solution_for_tests = solution_true
        solution_solver = self.solve_using_operator(solver, system_normal, rhs_normal, degraded, *latents)
        rhs = system(solution_true, degraded, *latents)
        matrix_exact = self.get_matrix_from_callable(
            lambda x: system.preconditioner_left_inv(system(system.preconditioner_right_inv(x))), rhs.shape)
        # matrix_exact = None
        self.compare_solutions(solution_solver, solution_true, matrix_exact)


class TestMinResSolver(TestCGSolver):
    solver = BatchedMinResSolver
    solver_init_params = {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'max_iter': SOLVER_MAXITER}


class TestBiCGStabSolver(TestSolverBase):
    __test__ = True
    solver = BatchedBiCGStabSolver
    device = 'cuda'
    dtype = th.float64
    solver_init_params = {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'verbose': False, 'max_iter': SOLVER_MAXITER,
                          'restarts_tol': 0.0}

    def test_diag_dummy(self):
        TestCGSolver.test_diag_dummy(self)


class TestCGSSolver(TestSolverBase):
    __test__ = True
    solver = BatchedConjugateGradientSquaredSolver
    device = 'cuda'
    dtype = th.float64
    solver_init_params = {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'verbose': True, 'max_iter': SOLVER_MAXITER,
                          'restarts_iter': 25, 'restarts_tol': 1e-5}

    def test_diag_dummy(self):
        TestCGSolver.test_diag_dummy(self)


class TestSciPyBiCGStabSolver(TestSciPySolverBase):
    __test__ = True
    solver = staticmethod(bicgstab)


class TestSciPyBiCGSolver(TestSciPySolverBase):
    __test__ = True
    solver = staticmethod(bicg)


class TestSciPyGMRESSolver(TestSciPySolverBase):
    __test__ = False
    solver = staticmethod(gmres)


class TestSciPyLGMRESSolver(TestSciPySolverBase):
    __test__ = False
    solver = staticmethod(lgmres)


class TestSciPyCGSSolver(TestSciPySolverBase):
    __test__ = True
    solver = staticmethod(cgs)


class TestSciPyGCROTSolver(TestSciPySolverBase):
    __test__ = True
    solver = staticmethod(gcrotmk)


class TestSciPyLSQRSolver(TestSciPySolverBase):
    __test__ = True
    solver = staticmethod(lsqr)
    solver_init_params = {'btol': SOLVER_RTOL, 'atol': SOLVER_ATOL, 'iter_lim': SOLVER_MAXITER}


class TestPyKrylovTFQMRSolver(TestSciPySolverBase):
    __test__ = True
    solver: KrylovMethod = TFQMR
    device = 'cuda'
    dtype = th.float64
    solver_init_params = {'abstol': SOLVER_ATOL, 'reltol': SOLVER_RTOL, 'precon': None, 'matvec_max': SOLVER_MAXITER*2}

    def get_solver_callable(self):
        return None

    def solve_using_operator(self, solver, matrix_op, rhs, *matrix_args):
        assert rhs.ndim == 4
        rhs_np = rhs.flatten(start_dim=1).cpu().numpy()
        solution_solver = []
        linop_sp = self.torch_callable_to_scipy(matrix_op.apply, rhs)
        for idx, b in enumerate(rhs_np):
            solver = self.solver(linop_sp.prepare_for_item(idx), **self.solver_init_params)
            solver.solve(b, **self.solver_init_params)
            solution = solver.bestSolution
            solution_solver.append(solution)
        solution_solver = np.stack(solution_solver)
        solution_solver = th.from_numpy(solution_solver).to(dtype=self.dtype, device=self.device)
        solution_solver = solution_solver.reshape(*rhs.shape)
        return solution_solver


class TestPyKrylovCGSSolver(TestPyKrylovTFQMRSolver):
    solver: KrylovMethod = CGS


class TestInvSolver:
    __test__ = True
    solver = None
    device = 'cpu'
    dtype = th.float64
    solution_check_atol = 1e-7
    solution_check_rtol = 1e-5
    noise_std = 3/255
    reg_images_kernel_size = 3
    degradation_kernel_size = (5, 3)

    @staticmethod
    def get_reg_backbone():
        return TestSciPySolverBase.get_reg_backbone()

    @pytest.mark.parametrize("conv_op,batch_size,channels_num,spatial_dims",
                             list(itertools.product(['conv', 'cnn', 'kpn'],
                                                    SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    SIZES_PARAMS['spatial_dims'])))
    def test_with_wiener_system(self, conv_op, batch_size, channels_num, spatial_dims):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))

        system, inputs = self.get_toy_wiener_system(conv_op, batch_size, channels_num)
        solution_true = TestSolverBase.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)
        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * 0.5
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})

        solver = BatchedInverseCallSolver()
        system = system.prepare_for_restoration(**inputs)
        rhs = system(solution_true, degraded)
        solution_solver = solver.solve(system, right_hand_side=rhs)
        TestSolverBase.compare_solutions(self, solution_solver, solution_true, None)

    def get_reg_images_cnn(self, num_channels, padding_mode):
        reg_images = LearnableCNNOperator(
            backbone=ConvBackbone(out_features_per_layer=[8], kernel_size_per_layer=[self.reg_images_kernel_size],
                                  padding_per_layer=[0], strides_per_layer=[1],
                                  num_in_features=num_channels), padding_mode=padding_mode, learnable=False,
            mix_in_channels=False)
        return reg_images

    def get_reg_images_conv(self, num_channels, padding_mode):
        reg_images = LearnableConvolutionOperator(filter_size=self.reg_images_kernel_size,
                                                  filter_num_out_channels=num_channels,
                                                  padding_mode=padding_mode, learnable=False, mix_in_channels=False)
        reg_images.effective_kernel = th.rand_like(reg_images.effective_kernel)
        return reg_images

    def get_reg_images_kpn(self, num_channels, padding_mode, kpn_in_channels=None):
        if kpn_in_channels is None:
            kpn_in_channels = num_channels
        class KPCNN(KPBackbone):
            def __init__(self_in, kernel_size):
                super(KPBackbone, self_in).__init__()
                self_in.net = nn.Conv2d(kpn_in_channels, num_channels*8*(kernel_size**2), 3, bias=False)
                self_in.num_in_channels = num_channels
                self_in.num_out_channels = 8
                self_in.filter_size = kernel_size

            def forward(self_in, x):
                res = self_in.net(x)
                res = res.mean(dim=(-1, -2))
                res = res.view(-1, self_in.num_out_channels, self_in.num_in_channels, self_in.filter_size,
                               self_in.filter_size)
                return res
        cnn = KPCNN(self.reg_images_kernel_size)
        reg_images = LearnableKPNConvOperator(cnn, observation_keyword='input', padding_mode=padding_mode,
                                              learnable=False, mix_in_channels=False)
        return reg_images

    def get_conv_operator(self, name, num_channels, padding_mode, kpn_in_channels=None):
        if name == 'conv':
            return self.get_reg_images_conv(num_channels, padding_mode)
        elif name == 'cnn':
            return self.get_reg_images_cnn(num_channels, padding_mode)
        elif name == 'kpn':
            return self.get_reg_images_kpn(num_channels, padding_mode, kpn_in_channels=kpn_in_channels)
        else:
            raise ValueError

    def get_toy_wiener_system(self, conv_op, batch_size, num_channels, padding_mode='periodic'):
        degradation_operator = ConvDecimateLinearDegradationOperator(padding_mode=padding_mode)
        kernels = degradation_operator._init_kernels_gaussian(batch_size, 3).to(device=self.device, dtype=self.dtype)
        degradation_operator = degradation_operator.init_with_parameters(kernel=kernels)

        # reg_fidelity = LearnableConvolutionOperator(filter_size=3, padding_mode=padding_mode, learnable=False)
        reg_images = self.get_conv_operator(conv_op, num_channels, padding_mode)
        reg_fidelity = IdentityOperator()
        system = WienerFilteringSystemOperator(
            degradation_operator, reg_fidelity, reg_images, weight_noise=IdentityOperator())
        system = system.to(device=self.device, dtype=self.dtype)
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device)*self.noise_std,
                        'kernel': kernels}
        for p in system.parameters():
            p.requires_grad = False
        return system, input_params

    def get_toy_sr_wiener_system(self, conv_op, scale_factor, batch_size, num_channels, kernel, use_noise,
                                 padding_mode='periodic'):
        degradation_operator = ConvDecimateLinearDegradationOperator(scale_factor=scale_factor,
                                                                     padding_mode=padding_mode)
        if kernel == 'none':
            kernels = None
        elif kernel == 'single_channel':
            kernels = th.rand(batch_size, 1, *self.degradation_kernel_size, device=self.device, dtype=self.dtype)
            kernels /= kernels.sum(dim=(-1, -2), keepdim=True)
        elif kernel == 'multi_channel':
            kernels = th.rand(batch_size, num_channels, *self.degradation_kernel_size,
                              device=self.device, dtype=self.dtype)
            kernels /= kernels.sum(dim=(-1, -2), keepdim=True)
        else:
            raise RuntimeError
        degradation_operator = degradation_operator.init_with_parameters(scale_factor=scale_factor, kernel=kernels)
        reg_fidelity = IdentityOperator()
        reg_images = self.get_conv_operator(conv_op, num_channels, padding_mode)
        system = SRWienerFilteringSystemOperator(
            degradation_operator, reg_fidelity, reg_images,
            weight_noise=IdentityOperator(), use_weight_noise_scaling_coef=use_noise)
        system = system.to(device=self.device, dtype=self.dtype)
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device)*self.noise_std,
                        'kernel': kernels,
                        'scale_factor': scale_factor}
        for p in system.parameters():
            p.requires_grad = False
        return system, input_params

    @pytest.mark.parametrize("conv_op,scale_factor,batch_size,channels_num,spatial_dims,kernel,use_noise",
                             list(itertools.product(['conv', 'cnn', 'kpn'],
                                                    [2, 3],
                                                    SIZES_PARAMS['batch_size'],
                                                    SIZES_PARAMS['channels_number'],
                                                    ['132-132', '12-24'],
                                                    ['none', 'single_channel', 'multi_channel'],
                                                    [True, False])))
    def test_with_sr_wiener_system(self, conv_op, scale_factor, batch_size, channels_num, spatial_dims, kernel,
                                   use_noise):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))

        system, inputs = self.get_toy_sr_wiener_system(conv_op, scale_factor, batch_size, channels_num, kernel,
                                                       use_noise)
        solution_true = TestSolverBase.get_test_image(is_color=False, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, channels_num, 1, 1).to(device=self.device, dtype=self.dtype)
        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * 0.5
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})

        solver = BatchedInverseCallSolver()
        system = system.prepare_for_restoration(**inputs)
        rhs = system(solution_true, degraded)
        solution_solver = solver.solve(system, right_hand_side=rhs)
        TestSolverBase.compare_solutions(self, solution_solver, solution_true, None)

    def get_toy_demosaic_wiener_system(self, conv_op, pattern, batch_size, kernel, use_noise, padding_mode='periodic'):
        degradation_operator = ConvMosaicLinearDegradationOperator(pattern=pattern, padding_mode=padding_mode)
        if kernel == 'none':
            kernels = None
        elif kernel == 'single_channel':
            kernels = th.rand(batch_size, 1, *self.degradation_kernel_size, device=self.device, dtype=self.dtype)
            kernels /= kernels.sum(dim=(-1, -2), keepdim=True)
        elif kernel == 'multi_channel':
            kernels = th.rand(batch_size, 3, *self.degradation_kernel_size,
                              device=self.device, dtype=self.dtype)
            kernels /= kernels.sum(dim=(-1, -2), keepdim=True)
        else:
            raise RuntimeError
        degradation_operator = degradation_operator.init_with_parameters(pattern=pattern, kernel=kernels)
        reg_fidelity = IdentityOperator()
        reg_images = self.get_conv_operator(conv_op, 3, padding_mode, kpn_in_channels=4)
        system = DemosaicWienerFilteringSystemOperator(
            degradation_operator, reg_fidelity, reg_images,
            weight_noise=IdentityOperator(), use_weight_noise_scaling_coef=use_noise)
        system = system.to(device=self.device, dtype=self.dtype)
        input_params = {'noise_std': th.ones(1, 1, dtype=self.dtype, device=self.device)*self.noise_std,
                        'kernel': kernels,
                        'pattern': pattern}
        for p in system.parameters():
            p.requires_grad = False
        return system, input_params

    @pytest.mark.parametrize("conv_op,pattern,batch_size,spatial_dims,kernel,use_noise",
                             list(itertools.product(['conv', 'cnn', 'kpn'],
                                                    ['rggb', 'bggr', 'gbrg', 'grbg'],
                                                    SIZES_PARAMS['batch_size'],
                                                    ['132-132', '12-24'],
                                                    ['none', 'single_channel', 'multi_channel'],
                                                    [True, False])))
    def test_with_demosaic_wiener_system(self, conv_op, pattern, batch_size, spatial_dims, kernel, use_noise):
        if isinstance(spatial_dims, str):
            h, w = spatial_dims.split('-')
            spatial_dims = (int(h), int(w))

        system, inputs = self.get_toy_demosaic_wiener_system(conv_op, pattern, batch_size, kernel, use_noise)
        solution_true = TestSolverBase.get_test_image(is_color=True, size=spatial_dims)
        solution_true = solution_true.repeat(batch_size, 1, 1, 1).to(device=self.device, dtype=self.dtype)
        degraded = system.degradation(solution_true)
        degraded += th.randn_like(degraded) * 0.5
        degraded.clamp_(0, 1)
        inputs.update({'input': degraded})

        solver = BatchedInverseCallSolver()
        system = system.prepare_for_restoration(**inputs)
        rhs = system(solution_true, degraded)
        solution_solver = solver.solve(system, right_hand_side=rhs)
        TestSolverBase.compare_solutions(self, solution_solver, solution_true, None)
