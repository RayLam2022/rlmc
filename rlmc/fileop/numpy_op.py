"""
@File    :   numpy_op.py
@Time    :   2024/07/02 22:33:29
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import numpy as np

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["Numpy"]


class Numpy(AbstractFile):
    def __init__(self, file_path: str = "") -> None:
        self.file_path = file_path
        if file_path != "":
            self.data = self.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type != None:
            print(exc_type, exc_val, exc_tb)

    def read(self):
        data = np.load(self.file_path)
        return data

    def write(self, data, file_path: str, allow_pickle: bool = True):
        np.save(file_path, data, allow_pickle=allow_pickle)


if __name__ == "__main__":

    with Numpy(r"C:\Users\RayLam\Desktop\git_private\n.npy") as yl:
        print(yl.data)

    # yamlobj.write(nest_dict, "rlmc/configs/cfg1.yaml")
