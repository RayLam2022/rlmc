# 参考 https://github.com/python-poetry/poetry/blob/main/docs/building-extension-modules.md

import os
import shutil

from pathlib import Path

from setuptools import setup
from setuptools import Distribution
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


def build():
    # ext_modules=cythonize(
    #                   "rltools/del_jupyter_temp.pyx"
    #         )
    ext_modules=[Extension("rltools.del_jupyter_temp", ["rltools/del_jupyter_temp.c"])]
    #)
    distribution = Distribution({
        "name": "rltools",
        "ext_modules": ext_modules
    })
    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project  把生成的文件复制到指定目录
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension =output.relative_to(cmd.build_lib) # Path("src") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)

if __name__ == "__main__":
    build()
