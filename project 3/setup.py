'''setup.py
Run to compile the Cython code in im2col_cython.pyx
Oliver W. Layton
Fall 2023
'''
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension('im2col_cython', ['im2col_cython.pyx'], include_dirs=[numpy.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
              ]

setup(ext_modules=cythonize(extensions))
