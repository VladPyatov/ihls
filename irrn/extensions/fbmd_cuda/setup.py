#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Stamatis Lefkimmiatis
@email : stamatisl@gmail.com
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('./requirements.txt', 'rt') as f:
    requirements = f.readlines()

setup(
    name='fbmd_cuda',
    version='1.1.0',
    description='Fast Batched SVD and EIGH Matrix Decompositions based on CUBLAS',
    author='Stamatis Lefkimmiatis',
    author_email='stamatisl@gmail.com',
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(name='fbmd_cuda',
                      sources=['robust_fbmd_cuda.cpp', 'fbmd_cuda_kernel.cu'],
                      libraries=['cusolver', 'cublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
