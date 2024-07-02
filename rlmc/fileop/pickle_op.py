"""
@File    :   pickle_op.py
@Time    :   2024/07/02 23:10:29
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import pickle

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["Pickle"]


class Pickle(AbstractFile):
    def __init__(self, file_path: str = "", mode: str = "rb") -> None:
        self.file_path = file_path
        self.mode = mode

        if file_path != "":
            self.data = self.read()

    def __enter__(self):
        self.file = open(self.file_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type != None:
            print(exc_type, exc_val, exc_tb)

    def read(self):
        with open(self.file_path, self.mode) as f:
            data = pickle.load(f)
        return data

    def write(self, data, file_path: str, mode: str = "wb"):
        with open(file_path, mode) as f:
            pickle.dump(data, f)
