from setuptools import setup
from Cython.Build import cythonize
from distutils.core import Extension

setup(
    # ext_modules=cythonize(
    #                   "rltools/del_jupyter_temp.pyx"
    #         )
    ext_modules=[Extension("rltools.del_jupyter_temp", ["rltools/del_jupyter_temp.c"])]
)
