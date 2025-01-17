"""
@File    :   __init__.py
@Time    :   2024/06/21 22:38:06
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import Any, Callable, Generic, TypeVar

from rlmc.fileop.yaml_op import Yaml
from rlmc.fileop.json_op import Json
from rlmc.fileop.numpy_op import Numpy
from rlmc.fileop.pickle_op import Pickle
from rlmc.fileop.mat_op import Mat

__all__ = ["file_processor"]

T=TypeVar('T')

files_op_dict = {
    "yaml": Yaml,
    "yml": Yaml,
    "json": Json,
    "npy": Numpy,
    "pkl": Pickle,
    "pickle": Pickle,
    "mat": Mat,
}


def file_processor(file_path: str = "", file_ext: str = "") -> Any:
    if file_ext == "":
        file_type = file_path.split(".")[-1]
    else:
        file_type = file_ext
    return files_op_dict.get(file_type)(file_path)


if __name__ == "__main__":
    yamlobj = file_processor("rlmc/configs/script_paths.yaml")
    yaml_content = yamlobj.data
    # yaml_content = yamlobj.read("rlmc/configs/script_paths.yaml")
    print(yaml_content)
