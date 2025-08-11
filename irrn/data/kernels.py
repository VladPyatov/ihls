import abc
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import measurements, interpolation
import torch as th


class KernelSamplerBase:
    kernel_size: int

    def __init__(self, kernel_size: int) -> None:
        """
        Initializing parameters of kernels sampler
        """
        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size

    @abc.abstractmethod
    def sample_kernel(self) -> np.array:
        """
        Method, which randomly samples kernel.

        :return: kernel as NumPy array of shape [self.kernel_size, self.kernel_size]
        """
        pass

    def _visualize(self, num_kernels: int = 10):
        fig, ax = plt.subplots(1, num_kernels, figsize=(num_kernels*3, 3))
        for i in range(num_kernels):
            ax[i].imshow(self.sample_kernel())
            ax[i].axis('off')
        plt.show()

    def sample_kernels_batch(self, batch_size: int) -> th.Tensor:
        """
        This method samples batch of kernels

        :param batch_size: number of kernels to sample
        :return: batch of randomly sampled kernels
        """
        kernels = []
        for i in range(batch_size):
            kernels.append(self.sample_kernel()[None, :, :])
        kernels = np.stack(kernels, axis=0)
        kernels = th.from_numpy(kernels)
        return kernels


class GaussianKernelSampler(KernelSamplerBase):
    """
    Class, that samples random noisy bi-variate Gaussian kernels.
    Credit: https://github.com/assafshocher/BlindSR_dataset_generator/blob/master/BSD100RK.ipynb
    """
    scale_factor: int
    min_eigval: float
    max_eigval: float
    min_noise_level: float
    max_noise_level: float

    def __init__(self, kernel_size: int, scale_factor: int, min_eigval: float = 0.35, max_eigval: float = 5,
                 min_noise_level: float = 0, max_noise_level: float = 0.4) -> None:
        """
        Initializing parameters of kernels sampler.
        Use scale_factor=1 for blur kernels and scale_factor > 1 for downscale kernels.

        :param kernel_size: size of canvas to be used for sampling
        :param scale_factor: scale factor of super-resolution problem
        :param min_eigval: lower limit of eigenvalues of covariance matrix
        :param max_eigval: upper limit of eigenvalues of covariance matrix
        :param min_noise_level: lower limit of multiplicative noise amount to be added to kernel
        :param max_noise_level: upper limit of multiplicative noise amount to be added to kernel
        """
        super().__init__(kernel_size)
        self.scale_factor = scale_factor
        self.min_eigval = min_eigval
        self.max_eigval = max_eigval
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level

    def shift_kernel(self, kernel: np.array) -> np.array:
        """
        Function, that shifts SR kernel.
        There are two reasons for shifting the kernel:
        1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
           the degradation process included shifting so we always assume center of mass is center of the kernel.
        2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
           pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
           top left corner of the first pixel. that is why different shift size needed between od and even size.
        Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
        The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

        :param kernel: kernel of shape [h, w] to be shifted
        :return: shifted kernel of shape [h, w]
        """
        # First calculate the current center of mass for the kernel
        current_center_of_mass = np.array(measurements.center_of_mass(kernel))

        # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
        wanted_center_of_mass = self.kernel_size // 2 + 0.5 * (1 - self.scale_factor % 2)

        # Define the shift vector for the kernel shifting (x,y)
        shift_vec = wanted_center_of_mass - current_center_of_mass

        # Finally shift the kernel and return
        return interpolation.shift(kernel, shift_vec)

    def sample_kernel(self) -> np.array:
        """
        Function, that samples random noisy bi-variate Gaussian kernels.
        :return: Gaussian-like kernel as NumPy array of shape [self.kernel_size, self.kernel_size]
        """
        kernel_size = np.array([self.kernel_size, self.kernel_size])
        noise_level = self.min_noise_level + np.random.rand() * (self.max_noise_level - self.min_noise_level)
        # Set random eigenvalues (lambdas) and angle (theta) for COV matrix
        lambda_1 = self.min_eigval + np.random.rand() * (self.max_eigval - self.min_eigval)
        lambda_2 = self.min_eigval + np.random.rand() * (self.max_eigval - self.min_eigval)
        theta = np.random.rand() * np.pi
        noise = -noise_level + np.random.rand(*kernel_size) * noise_level * 2

        # Set COV matrix using Lambdas and Theta
        LAMBDA = np.diag([lambda_1, lambda_2])
        Q = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        SIGMA = Q @ LAMBDA @ Q.T
        INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

        # Set expectation position (shifting kernel for aligned image)
        MU = kernel_size // 2 + 0.5 * (self.scale_factor - kernel_size % 2)
        MU = MU[None, None, :, None]

        # Create meshgrid for Gaussian
        [X, Y] = np.meshgrid(range(kernel_size[0]), range(kernel_size[1]))
        Z = np.stack([X, Y], 2)[:, :, :, None]

        # Calculate Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ_t = ZZ.transpose(0, 1, 3, 2)
        raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

        # shift the kernel so it will be centered
        raw_kernel_centered = self.shift_kernel(raw_kernel)

        # Normalize the kernel and return
        kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
        return kernel.astype(np.float32)


class ShakeKernelSampler(KernelSamplerBase):
    """
    Class, that samples random motion blur kernels.
    References
        [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from
          Motion Blur" Image Processing, IEEE Transactions on. vol.21, no.8, pp. 3502 - 3517, Aug. 2012
          doi:10.1109/TIP.2012.2192126; Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
        [Boracchi and Foi 2011] Giacomo Boracchi and Alessandro Foi, "Uniform motion blur in Poissonian noise:
          blur/noise trade-off", Image Processing, IEEE Transactions on. vol. 20, no. 2, pp. 592-598, Feb. 2011
          doi: 10.1109/TIP.2010.2062196; Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
    """
    trajectory_size: int
    anxiety: float
    num_samples: int
    max_total_length: int
    time_fractions: List[float]

    def __init__(self, kernel_size: int, trajectory_size: int = 64, anxiety: float = None, num_samples: int = 2000,
                 max_total_length: int = 60, time_fractions: List[float] = None) -> None:
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012].
        Each trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration, is
        affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the previous
        particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming at inverting
        the particle velocity may arises, mimicking a sudden movement that occurs when the user presses the camera
        button or tries to compensate the camera shake. At each step, the velocity is normalized to guarantee that
        trajectories corresponding to equal exposures have the same length. Each perturbation (Gaussian, inertial, and
        impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi 2011] can be obtained by
        setting anxiety to 0 (when no impulsive changes occurs)

        :param kernel_size: size of canvas to be used for sampling
        :param trajectory_size: size (in pixels) of the square support of the Trajectory curve
        :param anxiety: amount of shake, which scales random vector added at each sample
        :param num_samples: number of samples where the Trajectory is sampled
        :param max_total_length: maximum trajectory length computed as sum of all distanced between consecutive points
        """
        super().__init__(kernel_size)
        if anxiety is None:
            self.anxiety = 0.1 * np.random.rand()
        else:
            self.anxiety = anxiety
        self.trajectory_size = trajectory_size
        self.num_samples = num_samples
        self.max_total_length = max_total_length
        if time_fractions is None:
            self.time_fractions = [1/100, 1/10, 1/2, 1]
        else:
            self.time_fractions = time_fractions
        self.x = None
        self.TotLength = None
        self.nAbruptShakes = None

    def sample_trajectory(self) -> np.array:
        """
        Method, which samples random trajectory by simulating Brownian motion.
        """
        abruptShakesCounter = 0
        TotLength = 0
        # term determining, at each sample, the strength of the component leading towards the previous position
        centripetal = 0.7 * np.random.rand()

        # term determining, at each sample, the random component of the new direction
        gaussianTerm = 10 * np.random.rand()

        # probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
        freqBigShakes = 0.2 * np.random.rand()

        # v is the initial velocity vector, initialized at random direction
        init_angle = np.pi * np.random.rand()

        # initial velocity vector having norm 1
        v0 = np.cos(init_angle) + 1j * np.sin(init_angle)

        # the speed of the initial velocity vector
        v = v0 * self.max_total_length / (self.num_samples - 1)

        if self.anxiety > 0:
            v = v0 * self.anxiety

        # initialize the trajectory vector
        x = np.zeros((self.num_samples, 1), dtype=np.complex64)

        for t in range(self.num_samples - 1):
            # determine if there is an abrupt (impulsive) shake
            if np.random.rand() < freqBigShakes * self.anxiety:
                # if yes, determine the next direction which is likely to be opposite to the previous one
                nextDirection = 2 * v * (np.exp(1j * (np.pi + (np.random.rand() - 0.5))))
                abruptShakesCounter = abruptShakesCounter + 1
            else:
                nextDirection = 0

            # determine the random component motion vector at the next step
            dv = nextDirection + self.anxiety * (
                    gaussianTerm * (np.random.randn() + 1j * np.random.randn()) - centripetal * x[t]) * (
                         self.max_total_length / (self.num_samples - 1))
            v = v + dv

            # velocity vector normalization
            v = (v / np.abs(v)) * self.max_total_length / (self.num_samples - 1)

            # update particle position
            x[t + 1] = x[t] + v

            # compute total length
            TotLength = TotLength + np.abs(x[t + 1] - x[t])

        # Center the Trajectory

        # Set the lowest position in zero
        x = x - 1j * np.min(np.imag(x)) - np.min(np.real(x))

        # Center the Trajectory
        x = x - 1j * np.remainder(np.imag(x[0]), 1) - np.remainder(np.real(x[0]), 1) + 1 + 1j
        x = x + 1j * np.ceil((self.trajectory_size - np.max(np.imag(x))) / 2) + np.ceil(
            (self.trajectory_size - np.max(np.real(x))) / 2)

        self.x = x
        self.TotLength = TotLength
        self.nAbruptShakes = abruptShakesCounter

    def sample_kernel(self) -> np.array:
        """
        PSFs are obtained by sampling the continuous trajectory on a regular pixel grid using linear interpolation
        at subpixel level

        :return: blur kernel as NumPy array of shape [self.kernel_size, self.kernel_size]
        """
        kernel = self._sample_with_possible_bad()
        while np.any(np.isnan(kernel)):
            kernel = self._sample_with_possible_bad()
        return kernel.astype(np.float32)

    def _sample_with_possible_bad(self) -> np.array:
        """
        PSFs are obtained by sampling the continuous trajectory on a regular pixel grid using linear interpolation
        at subpixel level

        :return: blur kernel as NumPy array of shape [self.kernel_size, self.kernel_size]
        """
        self.sample_trajectory()
        PSFsize = (self.kernel_size, self.kernel_size)

        if isinstance(self.time_fractions, float):
            self.time_fractions = [self.time_fractions]

        # PSFnumber = len(self.time_fractions)
        numt = len(self.x)
        x = self.x[:, 0]

        # center with respect to baricenter
        x = x - np.mean(x) + (PSFsize[1] + 1j * PSFsize[0] + 1 + 1j) / 2

        #    x = np.max(1, np.min(PSFsize[1], np.real(x))) + 1j*np.max(1, np.min(PSFsize[0], np.imag(x)))

        # generate PSFs
        PSFS = []
        PSF = np.zeros(PSFsize)

        def triangle_fun(d):
            return max(0, (1 - np.abs(d)))

        def triangle_fun_prod(d1, d2):
            return triangle_fun(d1) * triangle_fun(d2)

        # set the exposure time
        for jj in range(len(self.time_fractions)):
            if jj == 0:
                prevT = 0
            else:
                prevT = self.time_fractions[jj - 1]

            # sample the trajectory until time self.time_fractions
            for t in range(len(x)):
                if (self.time_fractions[jj] * numt >= t) and (prevT * numt < t - 1):
                    t_proportion = 1
                elif (self.time_fractions[jj] * numt >= t - 1) and (prevT * numt < t - 1):
                    t_proportion = self.time_fractions[jj] * numt - t + 1
                elif (self.time_fractions[jj] * numt >= t) and (prevT * numt < t):
                    t_proportion = t - prevT * numt
                elif (self.time_fractions[jj] * numt >= t - 1) and (prevT * numt < t):
                    t_proportion = (self.time_fractions[jj] - prevT) * numt
                else:
                    t_proportion = 0

                m2 = min(PSFsize[1] - 2, max(1, int(np.floor(np.real(x[t])))))
                M2 = m2 + 1
                m1 = min(PSFsize[0] - 2, max(1, int(np.floor(np.imag(x[t])))))
                M1 = m1 + 1

                # linear interp. (separable)
                PSF[m1, m2] = PSF[m1, m2] + t_proportion * triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - m1)
                PSF[m1, M2] = PSF[m1, M2] + t_proportion * triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - m1)
                PSF[M1, m2] = PSF[M1, m2] + t_proportion * triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - M1)
                PSF[M1, M2] = PSF[M1, M2] + t_proportion * triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - M1)

            PSFS.append(PSF / PSF.sum())
        return PSFS[-1]
