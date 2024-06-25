"""
@File    :   yaml_op.py
@Time    :   2024/06/21 20:46:02
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if '.' not in sys.path: sys.path.append(".")
import os

import yaml

from rlmc.fileop.abstract_file import AbstractFile

__all__ = ["Yaml"]


class Yaml(AbstractFile):
    def __init__(self, file_path: str) -> None: 
        self.file_path = file_path
        self.data=self.read()

    def read(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def write(self, data, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)


if __name__ == "__main__":
    yamlobj = Yaml("rlmc/configs/script_paths.yaml")
    yaml_content = yamlobj.data
    print(yaml_content)
    nest_dict = {
        "a": 1,
        "b": {"c": 2, "d": 3, "e": {"f": 4}},
        "g": {"h": 5},
        "i": 6,
        "j": {"k": {7: {"m": 8}}},
        "n": [1, {"o": 1, "p": [1, 2, 3], "q": {"r": {"s": 100}}}, 3, [1, 2, 3], 5],
    }
    # yamlobj.write(nest_dict, "rlmc/configs/cfg1.yaml")
