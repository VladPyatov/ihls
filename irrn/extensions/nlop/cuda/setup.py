from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='group_transpose_cuda',
    ext_modules=[
        CUDAExtension(name='nlop_cuda',
                      sources=['group_transpose.cpp',
                               'group_transpose_cuda_kernel.cu']),
        ],
    cmdclass={
        'build_ext': BuildExtension
    })
