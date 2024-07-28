'''
@File    :   mat_op.py
@Time    :   2024/07/16 17:53:32
@Author  :   RayLam
@Contact :   1027196450@qq.com
'''


import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import scipy.io as scio

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["Mat"]


class Mat(AbstractFile):
    def __init__(
        self, file_path: str = ""
    ) -> None:
        self.file_path = file_path
        if file_path != "":
            self.data = self.read(file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type != None:
            print(exc_type, exc_val, exc_tb)

    def read(self, file_path: str):
        data = scio.loadmat(file_path)
        print('mat_keys:', data.keys())
        return data

    def write(self, data, file_path: str) -> None:
        scio.savemat(file_path, data)
