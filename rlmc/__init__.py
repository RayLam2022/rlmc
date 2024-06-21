"""
@File    :   __init__.py
@Time    :   2024/06/19 13:38:17
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

__version__ = "0.0.2"

import sys

sys.path.append(".")

from rlmc.utils.logger import Logger
from rlmc.utils.mc import MetaList
from rlmc.utils.register import Register
from rlmc.utils.sudict import SuDict
from rlmc.utils.multiprocess import MultiProcess
from rlmc.utils.multithread import MultiThread
from rlmc.utils.asynctask import AsyncTasks

# from rlmc.configs.cfg import Configs


reg = Register()
reg.register(MetaList)
reg.register(Logger)
reg.register(SuDict)
reg.register(MultiProcess)
reg.register(MultiThread)

__all__ = [
    "__version__",
    "Logger",
    "MetaList",
    "Register",
    "SuDict",
    "MultiProcess",
    "MultiThread",
    "AsyncTasks",
    "reg",
]
