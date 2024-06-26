"""
@File    :   __init__.py
@Time    :   2024/06/19 13:38:17
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

__version__ = "1.0.2"

import sys

if "." not in sys.path:
    sys.path.append(".")

from rlmc.utils.logger import Logger
from rlmc.utils.mc import MetaList
from rlmc.utils.register import Register
from rlmc.utils.sudict import SuDict
from rlmc.utils.multiprocess import MultiProcess
from rlmc.utils.multithread import MultiThread
from rlmc.utils.asynctask import (
    AsyncTasks,
    AsyncProducerConsumer,
    AsyncProducerConsumerTriple,
)
from rlmc.utils.coroutine import Abstract_ManMachineChat
from rlmc.utils.downloadscript import HfDownload, MsDownload, AutodlDownload
from rlmc.utils.systeminfos import general_info, gpu_info

from rlmc.fileop.utils import FindContent, DatasetSplit
from rlmc.fileop import file_processor

from rlmc.configs import (
    user_setting,
    model_download_urls,
    dataset_download_urls,
    common_urls,
)
from rlmc.resource import condarc, pip, cuda


from rlmc.model.predictor import (
    ClsPredictor,
    SemanticSegmentationPredictor,
    PosePredictor,
    ObjDetectPredictor,
    ObbPredictor,
    RegressorPredictor,
    SequencePredictor,
)

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
reg.register(HfDownload)
reg.register(MsDownload)
reg.register(AutodlDownload)
reg.register(FindContent)
reg.register(DatasetSplit)
reg.register(general_info)
reg.register(gpu_info)
reg.register(file_processor)

cfg = SuDict()
cfg["common_urls"] = common_urls
cfg["user_setting"] = user_setting
cfg["model_download_urls"] = model_download_urls
cfg["dataset_download_urls"] = dataset_download_urls
cfg["source"] = SuDict({"pip": pip, "condarc": condarc, "cuda": cuda})


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
    "HfDownload",
    "MsDownload",
    "AutodlDownload",
    "FindContent",
    "DatasetSplit",
    "file_processor",
    "general_info",
    "gpu_info",
    "reg",
    "cfg",
]


if __name__ == "__main__":
    print(reg.keys())
    yamlobj = reg["file_processor"]("rlmc/configs/script_paths.yaml")
    yaml_content = yamlobj.read()
    print(yaml_content)
