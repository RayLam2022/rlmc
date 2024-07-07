# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
setup(
    name="Example Cython",
    ext_modules=cythonize(["examples_cy.pyx"])
)
