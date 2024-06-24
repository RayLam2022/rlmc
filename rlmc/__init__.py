"""
@File    :   __init__.py
@Time    :   2024/06/19 13:38:17
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

__version__ = "0.0.7"

import sys

sys.path.append(".")

from rlmc.utils.logger import Logger
from rlmc.utils.mc import MetaList
from rlmc.utils.register import Register
from rlmc.utils.sudict import SuDict
from rlmc.utils.multiprocess import MultiProcess
from rlmc.utils.multithread import MultiThread
from rlmc.utils.asynctask import AsyncTasks, AsyncProducerConsumer, AsyncProducerConsumerTriple
from rlmc.utils.coroutine import Abstract_ManMachineChat

from rlmc.fileop import file_processor

# from rlmc.configs.cfg import Configs


reg = Register()
reg.register(MetaList)
reg.register(Logger)
reg.register(SuDict)
reg.register(MultiProcess)
reg.register(MultiThread)
reg.register(AsyncTasks)
reg.register(AsyncProducerConsumer)
reg.register(AsyncProducerConsumerTriple)
reg.register(Abstract_ManMachineChat)
reg.register(file_processor)

__all__ = [
    "__version__",
    "Logger",
    "MetaList",
    "Register",
    "SuDict",
    "MultiProcess",
    "MultiThread",
    "AsyncTasks",
    "AsyncProducerConsumer",
    "AsyncProducerConsumerTriple",
    "Abstract_ManMachineChat",
    "file_processor",
    "reg",
]


if __name__ == "__main__":
    print(reg.keys())
    yamlobj = reg["file_processor"]("rlmc/configs/script_paths.yaml")
    yaml_content = yamlobj.read()
    print(yaml_content)
