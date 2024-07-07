from setuptools import setup
from Cython.Build import cythonize
from distutils.core import Extension

setup(
    # ext_modules=cythonize(
    #                   "rltests/del_jupyter_temp.pyx"
    #         )
    ext_modules=[Extension("rltests.del_jupyter_temp", ["rltests/del_jupyter_temp.c"])]
)
