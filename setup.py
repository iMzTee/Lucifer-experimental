"""Build script for fast_extract C++ extension.

Usage:
    python setup.py build_ext --inplace

Requires:
    - pybind11 (pip install pybind11)
    - C++ compiler (Visual Studio 2022 Build Tools on Windows)
"""

import sys
from setuptools import setup, Extension
import pybind11
import numpy as np

compile_args = []
if sys.platform == 'win32':
    compile_args = ['/O2', '/std:c++17']
else:
    compile_args = ['-O3', '-std=c++17']

ext_fast_extract = Extension(
    'fast_extract',
    sources=['fast_extract.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=compile_args,
)

ext_fast_step = Extension(
    'fast_step',
    sources=['fast_step.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=compile_args,
)

setup(
    name='lucifer_extensions',
    version='1.0',
    ext_modules=[ext_fast_extract, ext_fast_step],
)
