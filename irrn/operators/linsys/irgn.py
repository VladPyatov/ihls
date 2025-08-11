from typing import Tuple, Union, Any, List, Callable

import torch as th

from irrn.operators.base import LearnableLinearSystemOperatorBase, LearnableLinearOperatorBase
from irrn.operators.degradation.conv_decimate import ImageKernelJacobian, ImageJacobian, KernelJacobian
from irrn.operators.diag import LearnableDiagonalOperator, LearnableNumberOperator
from irrn.utils import MultiVector


def assert_is_tensor(element: Any) -> None:
    assert element.__class__ == th.Tensor, f'Expected element to be torch.Tensor, but given {element.__class__}'


def assert_is_image_kernel(element: Any) -> None:
    assert isinstance(element, MultiVector), f'Expected element to be MultiVector, but given {element.__class__}'


class QMSingleComponentSystemOperator(LearnableLinearSystemOperatorBase):
    """
    This class implements matrix of QML as a linear operator for problems, involving image estimation:
    R = J^T G^T G J + diag(\gamma \tau \Phi^T \Phi + \lambda)
    """
    def __init__(self, jacobian: Union[ImageJacobian, KernelJacobian], reg_fidelity: LearnableLinearOperatorBase,
                 reg_operators: LearnableLinearOperatorBase, reg_scales: LearnableDiagonalOperator,
                 weight_decay: LearnableNumberOperator, step_size: LearnableNumberOperator) -> None:
        """
        Initializing operators

        :param jacobian:      (J)         linear operator, representing Jacobian matrix
        :param reg_fidelity:  (G)         learned linear operators, representing data fidelity regularization
        :param reg_operators: (\Phi)      learned linear operators, representing regularization
        :param reg_scales:    (\tau)      learned diagonal matrix, representing image regularization weights
        :param weight_decay:  (\lambda)   learned number, representing image decay regularization weight
        :param step_size:     (\gamma)    learned number, representing step size for image update
        """
        super(QMSingleComponentSystemOperator, self).__init__()
        if isinstance(jacobian, ImageJacobian):
            self._permute_components = False
        elif isinstance(jacobian, KernelJacobian):
            self._permute_components = True
        else:
            raise AssertionError
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_operators, LearnableLinearOperatorBase)
        assert isinstance(reg_scales, LearnableDiagonalOperator)
        assert isinstance(weight_decay, LearnableNumberOperator)
        assert isinstance(step_size, LearnableNumberOperator)
        self.jacobian = jacobian
        self.reg_fidelity = reg_fidelity
        self.reg_operators = reg_operators
        self.reg_scales = reg_scales
        self.weight_decay = weight_decay
        self.step_size = step_size

    def _components_correct_order(self, image_component: th.Tensor, kernel_component: th.Tensor) -> Tuple[th.Tensor]:
        if self._permute_components:
            return kernel_component, image_component
        else:
            return image_component, kernel_component

    def apply(self, vector: th.Tensor, latent_image: th.Tensor, latent_kernel: th.Tensor) -> th.Tensor:
        """
        This method performs batched linear transformation of input vector.

        :param vector: batch with input vectors of shape [B, C, H_v, W_v] to be transformed by linear operator
        :param latent_image: batch with latent images of shape [B, C, H, W], needed to parametrize linear system
        :param latent_kernel: batch with latent kernels of shape [B, 1, h, w], needed to parametrize linear system
        :return: vector of shape [B, ...], transformed by linear operator
        """
        assert_is_tensor(vector)
        assert_is_tensor(latent_image)
        assert_is_tensor(latent_kernel)

        latent_param = self._components_correct_order(latent_image, latent_kernel)[1]
        ret = self.jacobian.transpose_apply(vector, latent_param, operator_between=self.reg_fidelity.transpose_apply)
        ret += self.step_size(self.reg_operators.transpose_apply(vector, operator_between=self.reg_scales), n=2)
        ret += self.weight_decay(vector)
        return ret

    def right_hand_side(self, degraded: th.Tensor, latent_image: th.Tensor, latent_kernel: th.Tensor) -> th.Tensor:
        """
        This method implements right hand side of linear system, given parametrization arguments

        :param degraded: batch with degraded images of shape [B, C, H_d, W_d] which we opt to restore
        :param latent_image: batch with latent images of shape [B, C, H, W], needed to parametrize linear system
        :param latent_kernel: batch with latent kernels of shape [B, 1, h, w], needed to parametrize linear system
        :return: right hand side of linear system
        """
        assert_is_tensor(degraded)
        assert_is_tensor(latent_image)
        assert_is_tensor(latent_kernel)

        latent_params = self._components_correct_order(latent_image, latent_kernel)
        b = self.jacobian.T(
            self.reg_fidelity.transpose_apply(degraded - self._degraded_self(*latent_params)), latent_params[1]) - \
            self.step_size(self.reg_operators.transpose_apply(latent_params[0], operator_between=self.reg_scales))
        return b

    def perform_step(self, inputs: Tuple[th.Tensor], component_step: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        """
        This method makes a step for image and kernel.

        :param inputs: latent images and kernels, one of which should be updated
        :param component_step: step direction to take for either image or kernel
        :return: latent images and kernels, one of which was updated
        """
        assert len(component_step) == 1
        if len(inputs) == 1:
            return (inputs[0] + self.step_size(component_step[0]), )
        assert len(inputs) == 2

        inputs_ordered = self._components_correct_order(*inputs)
        ret = (inputs_ordered[0] + self.step_size(component_step[0]), inputs_ordered[1])
        return self._components_correct_order(*ret)

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        """
        This method should return all tensors, which may require gradients computation
        :return: tuple with tensors, that may require gradients computation
        """
        return \
            (*self.reg_fidelity.tensors_for_grad, *self.reg_operators.tensors_for_grad,
             *self.reg_scales.tensors_for_grad, *self.weight_decay.tensors_for_grad, *self.step_size.tensors_for_grad)

    def to(self, *args, **kwargs) -> 'QMSingleComponentSystemOperator':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self.reg_fidelity = self.reg_fidelity.to(*args, **kwargs)
        self.reg_operators = self.reg_operators.to(*args, **kwargs)
        self.reg_scales = self.reg_scales.to(*args, **kwargs)
        self.weight_decay = self.weight_decay.to(*args, **kwargs)
        self.step_size = self.step_size.to(*args, **kwargs)
        return self


class QMImageKernelSystemOperator(LearnableLinearSystemOperatorBase):
    """
    This class implements matrix of QM as a linear operator for problems, involving image and kernel estimation:
    R = J^T G^T G J + blockdiag(\gamma_1 \tau \Phi^T \Phi + \lambda_1, \gamma_2 \theta \Psi^T \Psi + \lambda_2)
    """
    jacobian: ImageKernelJacobian

    def __init__(self, jacobian: ImageKernelJacobian, reg_fidelity: LearnableLinearOperatorBase,
                 reg_image: LearnableLinearOperatorBase, diag_image: LearnableDiagonalOperator,
                 reg_kernel: LearnableLinearOperatorBase, diag_kernel: LearnableDiagonalOperator,
                 weight_decay_image: LearnableNumberOperator, weight_decay_kernel: LearnableNumberOperator,
                 step_size_image: LearnableNumberOperator, step_size_kernel: LearnableNumberOperator) -> None:
        """
        Initializing operators

        :param jacobian:            (J)         linear operator, representing Jacobian matrix
        :param reg_fidelity:        (G)         learned linear operators, representing data fidelity regularization
        :param reg_image:           (\Phi)      learned linear operators, representing image regularization
        :param diag_image:          (\tau)      learned diagonal matrix, representing image regularization weights
        :param reg_kernel:          (\Psi)      learned linear operators, representing kernel regularization
        :param diag_kernel:         (\theta)    learned diagonal matrix, representing kernel regularization weights
        :param weight_decay_image:  (\lambda_1) learned number, representing image decay regularization weight
        :param weight_decay_kernel: (\lambda_2) learned number, representing image decay regularization weight
        :param step_size_image:     (\gamma_1)  learned number, representing step size for image update
        :param step_size_kernel:    (\gamma_2)  learned number, representing step size for kernel update
        """
        super(QMImageKernelSystemOperator, self).__init__()
        assert isinstance(jacobian, ImageKernelJacobian)
        assert isinstance(reg_fidelity, LearnableLinearOperatorBase)
        assert isinstance(reg_image, LearnableLinearOperatorBase)
        assert isinstance(diag_image, LearnableDiagonalOperator)
        assert isinstance(reg_kernel, LearnableLinearOperatorBase)
        assert isinstance(diag_kernel, LearnableDiagonalOperator)
        assert isinstance(weight_decay_image, LearnableNumberOperator)
        assert isinstance(weight_decay_kernel, LearnableNumberOperator)
        assert isinstance(step_size_image, LearnableNumberOperator)
        assert isinstance(step_size_kernel, LearnableNumberOperator)
        self.jacobian = jacobian
        self.system_images = \
            QMSingleComponentSystemOperator(self.jacobian.images_jacobian, reg_fidelity, reg_image, diag_image,
                                            weight_decay_image, step_size_image)
        self.system_kernels = \
            QMSingleComponentSystemOperator(self.jacobian.kernels_jacobian, reg_fidelity, reg_kernel, diag_kernel,
                                            weight_decay_kernel, step_size_kernel)

    def apply(self, vector: MultiVector, degraded: th.Tensor, latent_images: th.Tensor, latent_kernels: th.Tensor
              ) -> MultiVector:
        """
        This method performs batched linear transformation of input MultiVector vector.

        :param vector: input MultiVector vector of shape [B, ...] to be transformed by linear operator
        :param latent_images: batch of images of shape [B, C, H, W], which parametrize linear system at current step
        :param latent_kernels: batch of kernels of shape [B, 1, h, w], which parametrize linear system at current step
        :return: MultiVector vector of shape [B, ...], transformed by linear operator
        """
        assert_is_image_kernel(vector)
        assert_is_tensor(latent_images)
        assert_is_tensor(latent_kernels)

        ret = self.jacobian.transpose_apply(
            vector, latent_images, latent_kernels, operator_between=self.reg_fidelity.transpose_apply)
        ret.elements[0] += \
            self.step_size_image(
                self.reg_image.transpose_apply(vector.elements[0], operator_between=self.diag_image), n=2) + \
            self.weight_decay_image(vector.elements[0])
        ret.elements[1] += \
            self.step_size_kernel(
                self.reg_kernel.transpose_apply(vector.elements[1], operator_between=self.diag_kernel), n=2) + \
            self.weight_decay_kernel(vector.elements[1])
        return ret

    def right_hand_side(self, degraded: th.Tensor, latent_images: th.Tensor, latent_kernels: th.Tensor) -> MultiVector:
        """
        This method implements right hand side of linear system, given parametrization arguments

        :param degraded: batch of observations required for restoration
        :param latent_images: batch of images of shape [B, C, H, W], which parametrize linear system at current step
        :param latent_kernels: batch of kernels of shape [B, 1, h, w], which parametrize linear system at current step
        :return: right hand side of linear system
        """
        assert_is_tensor(degraded)
        assert_is_tensor(latent_images)
        assert_is_tensor(latent_kernels)

        b = self.reg_fidelity.transpose_apply(degraded - self._degraded_self(latent_images, latent_kernels))
        b = self.jacobian.T(b, latent_images, latent_kernels)
        b = b - MultiVector(
            (self.step_size_image(self.reg_image.transpose_apply(latent_images, operator_between=self.diag_image)),
            self.step_size_kernel(self.reg_kernel.transpose_apply(latent_kernels, operator_between=self.diag_kernel))))
        return b

    def prepare_for_step(self, step_idx: int, degraded: th.Tensor, latents: Tuple[th.Tensor, ...],
                         parts_scheduler: Union[List, Callable] = None
                         ) -> Union['QMImageKernelSystemOperator', 'QMSingleComponentSystemOperator']:
        """
        This method chooses system for the next optimization step. Since image and kernel vectors may be optimized
        either jointly or separately, this method selects and returns system object for optimization of exact part.

        :param step_idx: index of current step (step number)
        :param parts_scheduler: parameters, which are needed for system part selection
        :return: correct system for next optimization step
        """
        if isinstance(parts_scheduler, (list, tuple)):
            optim_part = parts_scheduler[step_idx]
        else:
            optim_part = parts_scheduler(step_idx)
        return self._choose_system_operator(optim_part)

    def _choose_system_operator(self, system_type: str
                               ) -> Union['QMImageKernelSystemOperator', 'QMSingleComponentSystemOperator']:
        assert isinstance(system_type, str)
        if system_type[0] == 'b':
            return self
        elif system_type[0] == 'i':
            return self.system_images
        elif system_type[0] == 'k':
            return self.system_kernels
        else:
            raise ValueError(f'Expected system type to be one of "both", "image", "kernel", '
                             f'but given {system_type}')

    def perform_step(self, inputs: Tuple[th.Tensor], deltas: Tuple[th.Tensor]) -> Tuple[th.Tensor]:
        """
        This method makes a step for image and kernel, which are stored in Jacobian.

        :param inputs: image and kernel under restoration to update
        :param deltas: step directions to take for image and kernel
        :return: updated image and kernel
        """
        assert len(inputs) == 2
        assert len(deltas) == 2
        return self.project_image(inputs[0] + self.step_size_image(deltas[0])), self.project_kernel(inputs[1] + self.step_size_kernel(deltas[1]))

    @staticmethod
    def project_image(x):
        return x #th.clamp(x, 0, 1)

    @staticmethod
    def project_kernel(x):
        #x = th.clamp(x, 0, 1)
        #x = x/x.sum()
        return x

    @property
    def tensors_for_grad(self) -> Tuple[th.Tensor]:
        """
        This method should return all tensors, which may require gradients computation
        :return: tuple with tensors, that may require gradients computation
        """
        return \
            (*self.reg_fidelity.tensors_for_grad, *self.reg_image.tensors_for_grad, *self.diag_image.tensors_for_grad,
             *self.reg_kernel.tensors_for_grad, *self.diag_kernel.tensors_for_grad,
             *self.weight_decay_image.tensors_for_grad, *self.weight_decay_kernel.tensors_for_grad,
             *self.step_size_image.tensors_for_grad, *self.step_size_kernel.tensors_for_grad)

    @property
    def reg_fidelity(self):
        return self.system_images.reg_fidelity

    @property
    def reg_image(self):
        return self.system_images.reg_operators

    @property
    def diag_image(self):
        return self.system_images.reg_scales

    @property
    def reg_kernel(self):
        return self.system_kernels.reg_operators

    @property
    def diag_kernel(self):
        return self.system_kernels.reg_scales

    @property
    def weight_decay_image(self):
        return self.system_images.weight_decay

    @property
    def weight_decay_kernel(self):
        return self.system_kernels.weight_decay

    @property
    def step_size_image(self):
        return self.system_images.step_size

    @property
    def step_size_kernel(self):
        return self.system_kernels.step_size

    def to(self, *args, **kwargs) -> 'LearnableLinearOperatorBase':
        """
        Moves and/or casts the parameters and buffers.

        :param args, kwargs: parameters, specifying how to cast parameters
        :return: self
        """
        self.system_images = self.system_images.to(*args, **kwargs)
        self.system_kernels = self.system_kernels.to(*args, **kwargs)
        return self
