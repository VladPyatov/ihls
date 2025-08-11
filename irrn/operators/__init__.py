from .grad import AutogradGradLinearOperator
from .base import LinearOperatorBase, LearnableLinearOperatorBase, JacobianOperatorBase, \
    LearnableLinearSystemOperatorBase, LinearDegradationOperatorBase, IdentityOperator
from .conv import LearnableConvolutionOperator, LearnableCNNOperator, \
    LearnableKPNConvOperator, LearnableKPNSAConvOperator
from .pad import Pad2DOperator
from .diag import LearnableDiagonalOperator, LearnableNumberOperator
from .degradation.conv_decimate import ImageKernelJacobian
from .fweight import LearnableFourierWeightOperator
from .matmul import LearnableMatMulOperator
from .linsys import IRLSSystemOperatorHandler, IRLSSystemOperator, QMImageKernelSystemOperator, \
    QMSingleComponentSystemOperator, WienerFilteringSystemOperator
from .patch_group import PatchGroupOperator, LearnableConvPatchGroupOperator, LearnableCNNPatchGroupOperator, \
    LearnableKPNConvPatchGroupOperator
