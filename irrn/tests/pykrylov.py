import logging
import numpy as np


# Default (null) logger.
null_log = logging.getLogger('krylov')
null_log.setLevel(logging.INFO)
null_log.addHandler(logging.NullHandler())


class KrylovMethod(object):
    """
    A general template for implementing iterative Krylov methods. This module
    defines the `KrylovMethod` generic class. Other modules subclass
    `KrylovMethod` to implement specific algorithms.

    :parameters:

            :op:  an operator describing the coefficient matrix `A`.
                  `y = op * x` must return the operator-vector product
                  `y = Ax` for any given vector `x`. If required by
                  the method, `y = op.T * x` must return the operator-vector
                  product with the adjoint operator.

    :keywords:

        :atol:    absolute stopping tolerance. Default: 1.0e-8.

        :rtol:    relative stopping tolerance. Default: 1.0e-6.

        :precon:  optional preconditioner. If not `None`, `y = precon(x)`
                  returns the vector `y` solution of the linear system
                  `M y = x`.

        :logger:  a `logging.logger` instance. If none is supplied, a default
                  null logger will be used.


    For general references on Krylov methods, see [Demmel]_, [Greenbaum]_,
    [Kelley]_, [Saad]_ and [Templates]_.

    References:

    .. [Demmel] J. W. Demmel, *Applied Numerical Linear Algebra*, SIAM,
                Philadelphia, 1997.

    .. [Greenbaum] A. Greenbaum, *Iterative Methods for Solving Linear Systems*,
                   number 17 in *Frontiers in Applied Mathematics*, SIAM,
                   Philadelphia, 1997.

    .. [Kelley] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
                Equations*, number 16 in *Frontiers in Applied Mathematics*,
                SIAM, Philadelphia, 1995.

    .. [Saad] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
              SIAM, Philadelphia, 2003.

    .. [Templates] R. Barrett, M. Berry, T. F. Chan, J. Demmel, J. M. Donato,
                   J. Dongarra, V. Eijkhout, R. Pozo, C. Romine and
                   H. Van der Vorst, *Templates for the Solution of Linear
                   Systems: Building Blocks for Iterative Methods*, SIAM,
                   Philadelphia, 1993.
    """

    def __init__(self, op, **kwargs):

        self.prefix = 'Generic: '
        self.name   = 'Generic Krylov Method (must be subclassed)'

        # Mandatory arguments
        self.op = op

        # Optional keyword arguments
        self.abstol = kwargs.get('abstol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.precon = kwargs.get('precon', None)
        self.logger = kwargs.get('logger', null_log)

        self.residNorm  = None
        self.residNorm0 = None
        self.residHistory = []

        self.nMatvec = 0
        self.nIter = 0
        self.converged = False
        self.bestSolution = None
        self.x = self.bestSolution

    def _write(self, msg):
        # If levels other than info are needed they should be used explicitly.
        self.logger.info(msg)

    def solve(self, rhs, **kwargs):
        """
        This is the :meth:`solve` method of the abstract `KrylovMethod` class.
        The class must be specialized and this method overridden.
        """
        raise NotImplementedError('This method must be subclassed')


class TFQMR(KrylovMethod):
    """
    A pure Python implementation of the transpose-free quasi-minimal residual
    (TFQMR) algorithm. TFQMR may be used to solve unsymmetric systems of linear
    equations, i.e., systems of the form

        `A x = b`

    where the operator `A` may be unsymmetric.

    TFQMR requires 2 operator-vector products with `A`, 4 dot products and
    10 daxpys per iteration. It does not require products with the adjoint
    of `A`.

    If a preconditioner is supplied, TFQMR needs to solve 2 preconditioning
    systems per iteration. Our implementation is inspired by the original
    description in [Freund]_ and that of [Kelley]_.

    References:

    .. [Freund] R. W. Freund, *A Transpose-Free Quasi-Minimal Residual Method
                for Non-Hermitian Linear Systems*, SIAM Journal on Scientific
                Computing, **14** (2), pp. 470--482, 1993.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Transpose-Free Quasi-Minimum Residual'
        self.acronym = 'TFQMR'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the TFQMR method.
        The vector `rhs` should be a Numpy array.

        :keywords:
            :guess:      Initial guess (Numpy array, default: 0)
            :matvec_max: Max. number of matrix-vector produts (2n)
        """
        n = rhs.shape[0]
        nMatvec = 0

        # Initial guess is zero unless one is supplied
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n)).astype(result_type)
        matvec_max = kwargs.get('matvec_max', 2*n)

        r0 = rhs  # Fixed vector throughout
        if guess_supplied:
            r0 = rhs - self.op @ x

        rho = np.dot(r0,r0)
        residNorm = np.abs(np.sqrt(rho))
        self.residNorm0 = residNorm
        threshold = max( self.abstol, self.reltol * self.residNorm0 )
        self.logger.info('Initial residual = %8.2e' % self.residNorm0)
        self.logger.info('Threshold = %8.2e' % threshold)

        finished = (residNorm <= threshold or nMatvec >= matvec_max)

        if not finished:
            y = r0.copy()   # Initial residual vector
            w = r0.copy()
            d = np.zeros(n, dtype=result_type)
            theta = 0.0
            eta = 0.0
            k = 0
            if self.precon is not None:
                z = self.precon * y
            else:
                z = y

            u = self.op @ z ; nMatvec += 1
            v = u.copy()

        while not finished:

            k += 1
            sigma = np.dot(r0,v)
            alpha = rho/sigma

            # First pass
            w -= alpha * u
            d *= theta * theta * eta / alpha
            d += z
            theta = np.linalg.norm(w)/residNorm
            c = 1.0 / np.sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            m = 2.0 * k - 1.0
            if residNorm * np.sqrt(m+1) < threshold or nMatvec >= matvec_max:
                finished = True
                continue

            # Second pass
            m += 1
            y -= alpha * v

            if self.precon is not None:
                z = self.precon * y
            else:
                z = y

            u = self.op @ z ; nMatvec += 1
            w -= alpha * u
            d *= theta * theta * eta / alpha
            d += z
            theta = np.linalg.norm(w)/residNorm
            c = 1.0 / np.sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            if residNorm * np.sqrt(m+1) < threshold or nMatvec >= matvec_max:
                finished = True
                continue

            # Final updates
            rho_next = np.dot(r0,w)
            beta = rho_next/rho
            rho = rho_next

            # Update y
            y *= beta
            y += w

            # Partial update of v with current u
            v *= beta
            v += u
            v *= beta

            # Update u
            if self.precon is not None:
                z = self.precon * y
            else:
                z = y

            u = self.op @ z ; nMatvec += 1

            # Complete update of v
            v += u

            # Display current info if requested
            self.logger.info('%5d  %8.2e' % (nMatvec, residNorm))


        self.converged = residNorm * np.sqrt(m+1) < threshold
        self.nMatvec = nMatvec
        self.bestSolution = self.x = x
        self.residNorm = residNorm


class CGS( KrylovMethod ):
    """
    A pure Python implementation of the conjugate gradient squared (CGS)
    algorithm. CGS may be used to solve unsymmetric systems of linear equations,
    i.e., systems of the form

        A x = b

    where the operator A may be unsymmetric.

    CGS requires 2 operator-vector products with A, 3 dot products and 7 daxpys
    per iteration. It does not require products with the adjoint of A.

    If a preconditioner is supplied, CGS needs to solve two preconditioning
    systems per iteration. The original description appears in [Sonn89]_, which
    our implementation roughly follows.


    Reference:

    .. [Sonn89] P. Sonneveld, *CGS, A Fast Lanczos-Type Solver for Nonsymmetric
                Linear Systems*, SIAM Journal on Scientific and Statistical
                Computing **10** (1), pp. 36--52, 1989.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Conjugate Gradient Squared'
        self.acronym = 'CGS'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CGS method.
        The vector `rhs` should be a Numpy array.

        :keywords:
            :guess:      Initial guess (Numpy array, default: 0)
            :matvec_max: Max. number of matrix-vector produts (2n)
        """
        n = rhs.shape[0]
        nMatvec = 0

        # Initial guess is zero unless one is supplied
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n)).astype(result_type)
        matvec_max = kwargs.get('matvec_max', 2*n)

        r0 = rhs  # Fixed vector throughout
        if guess_supplied:
            r0 = rhs - self.op @ x

        rho = np.dot(r0,r0)
        residNorm = np.abs(np.sqrt(rho))
        self.residNorm0 = residNorm
        threshold = max( self.abstol, self.reltol * self.residNorm0 )
        self.logger.info('Initial residual = %8.2e\n' % self.residNorm0)
        self.logger.info('Threshold = %8.2e\n' % threshold)

        finished = (residNorm <= threshold or nMatvec >= matvec_max)

        if not finished:
            r = r0.copy()   # Initial residual vector
            u = r0
            p = r0.copy()

        while not finished:

            if self.precon is not None:
                y = self.precon * p
            else:
                y = p

            v = self.op @ y ; nMatvec += 1
            sigma = np.dot(r0,v)
            alpha = rho/sigma
            q = u - alpha * v

            if self.precon is not None:
                z = self.precon * (u+q)
            else:
                z = u+q

            # Update solution and residual
            x += alpha * z
            Az = self.op @ z ; nMatvec += 1
            r -= alpha * Az

            # Update residual norm and check convergence
            residNorm = np.linalg.norm(r)

            if residNorm <= threshold or nMatvec >= matvec_max:
                finished = True
                continue

            rho_next = np.dot(r0,r)
            beta = rho_next/rho
            rho = rho_next
            u = r + beta * q

            # Update p in-place
            p *= beta
            p += q
            p *= beta
            p += u

            # Display current info if requested
            self.logger.info('%5d  %8.2e\n' % (nMatvec, residNorm))


        self.converged = residNorm <= threshold
        self.nMatvec = nMatvec
        self.bestSolution = self.x = x
        self.residNorm = residNorm