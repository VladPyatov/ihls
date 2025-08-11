from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='group_transpose',
    ext_modules=[
        CppExtension(name='nlop_cpu',
                     sources=['group_transpose.cpp'],
                     extra_compile_args=['-fopenmp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
