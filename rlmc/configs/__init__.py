import sys

if "." not in sys.path:
    sys.path.append(".")

import os
import os.path as osp
import json
from rlmc.fileop import file_processor

__all__ = [
    "user_setting",
    "model_download_urls",
    "dataset_download_urls",
    "common_urls",
]

config_root = osp.dirname(__file__)


user_setting = file_processor(osp.join(config_root, "user_setting.yaml")).read()
common_urls = file_processor(osp.join(config_root, "common_urls.yaml")).read()
model_download_urls = file_processor(
    osp.join(config_root, "models/model_download_urls.yaml")
).read()
dataset_download_urls = file_processor(
    osp.join(config_root, "datasets/dataset_download_urls.yaml")
).read()
# print(user_setting)
