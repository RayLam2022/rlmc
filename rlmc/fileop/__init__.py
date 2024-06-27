"""
@File    :   __init__.py
@Time    :   2024/06/21 22:38:06
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if '.' not in sys.path: sys.path.append(".")

from rlmc.fileop.yaml_op import Yaml


__all__ = ["file_processor"]

files_op_dict = {"yaml": Yaml, "yml": Yaml}


def file_processor(file_path):
    file_type = file_path.split(".")[-1]
    return files_op_dict.get(file_type)(file_path)


if __name__ == "__main__":
    yamlobj = file_processor("rlmc/configs/script_paths.yaml")
    yaml_content = yamlobj.read()
    print(yaml_content)
