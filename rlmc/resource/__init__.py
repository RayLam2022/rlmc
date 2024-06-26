import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import os.path as osp

__all__ = ["condarc", "pip", "cuda"]


def get_resource(path, returntype="content"):
    with open(path, "r", encoding="utf-8") as f:
        if returntype == "list":
            return f.readlines()
        elif returntype == "content":
            return f.read()
        else:
            return None


src_dir = osp.dirname(__file__)
condarc = get_resource(osp.join(src_dir, ".condarc"))
pip = get_resource(osp.join(src_dir, "pip.ini"))
cuda = get_resource(osp.join(src_dir, "cudarc.txt"))
