import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import Literal, Optional, Union

import os
import os.path as osp

__all__ = ["condarc", "pip", "cuda"]


def get_resource(
    path: str, returntype: Literal["content", "list"] = "content"
) -> Optional[str]:
    with open(path, "r", encoding="utf-8") as f:
        if returntype == "list":
            return f.readlines()
        elif returntype == "content":
            return f.read()
        else:
            return None


src_dir = osp.dirname(__file__)
condarc = get_resource(osp.join(src_dir, "condarc.txt"))
pip = get_resource(osp.join(src_dir, "pip.ini"))
cuda = get_resource(osp.join(src_dir, "cudarc.txt"))
