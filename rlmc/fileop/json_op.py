'''
@File    :   json_op.py
@Time    :   2024/07/02 22:13:11
@Author  :   RayLam
@Contact :   1027196450@qq.com
'''

import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import json

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["Json"]


class Json(AbstractFile):
    def __init__(
        self, file_path: str = "", mode: str = "r", encoding: str = "utf-8"
    ) -> None:
        self.file_path = file_path
        self.encoding = encoding
        self.mode = mode
        if file_path != "":
            self.data = self.read()

    def __enter__(self):
        self.file = open(self.file_path, self.mode, encoding=self.encoding)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type != None:
            print(exc_type, exc_val, exc_tb)

    def read(self):
        with open(self.file_path, self.mode, encoding=self.encoding) as f:
            data = json.load(f)
        return data

    def write(self, data, file_path: str, mode: str = "w", encoding: str = "utf-8", indent: int = 2):
        with open(file_path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)



    