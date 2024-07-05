"""
@File    :   downloadscript.py
@Time    :   2024/06/18 22:37:02
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import Any, NoReturn, Union, Callable, Literal, Iterable, List, Dict
import os
import os.path as osp
import shutil

from rlmc.configs import user_setting, model_download_urls, dataset_download_urls
from rlmc.utils.logger import Logger

os.environ["XDG_CACHE_HOME"] = user_setting["XDG_CACHE_HOME"]
os.environ["MODELSCOPE_CACHE"] = f"{user_setting['XDG_CACHE_HOME']}/modelscope"
os.environ["HF_ENDPOINT"] = user_setting["HF_ENDPOINT"]

from huggingface_hub import snapshot_download, login


__all__ = ["HfDownload", "MsDownload", "AutodlDownload"]


logger = Logger(__name__, level=Logger.DEBUG)


class HfDownload:
    """how to use
    if __name__=='__main__':
        repo_id = "meta-llama/Llama-2-7b-hf"
        repo_type = "model"
        local_dir=f"{user_setting['XDG_CACHE_HOME']}/{repo_id.split('/')[-1]}"
        hf_download=HfDownload(repo_id,local_dir,repo_type=repo_type, is_login=False)
        hf_download.run()
    """

    def __init__(
        self,
        repo_id: str,
        local_dir: str,
        local_dir_use_symlinks: bool = False,
        is_login: bool = False,
        hf_token: str = "hf_JiIBOPMbYRjssOTAsdWo",
        repo_type: Literal["model", "dataset"] = "model",
        ignore_patterns: List[str] = ["*.h5", "*safetensors", "*msgpack"],
        force_download: bool = False,
        resume_download: bool = True,  # 断点续传
        etag_timeout: int = 1200,  # 超时阈值
    ) -> None:
        if is_login:
            login(token=hf_token)
        os.makedirs(local_dir, exist_ok=True)
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.local_dir_use_symlinks = local_dir_use_symlinks
        self.repo_type = repo_type
        self.ignore_patterns = ignore_patterns
        self.force_download = force_download
        self.resume_download = resume_download
        self.etag_timeout = etag_timeout

    def run(self) -> None:
        logger.info(f"************ Start downloading {self.repo_id} ************")
        snapshot_download(
            repo_id=self.repo_id,
            cache_dir=self.local_dir,
            # filename=filename,
            local_dir=self.local_dir,
            local_dir_use_symlinks=self.local_dir_use_symlinks,
            ignore_patterns=self.ignore_patterns,
            force_download=self.force_download,
            resume_download=self.resume_download,
            etag_timeout=self.etag_timeout,
        )

        logger.info(f"************ {self.repo_id} Download finish ************")
        logger.info(f"Models saved in {self.local_dir}")


class MsDownload:
    """how to use
    if __name__=='__main__':
        repo_id='skyline2006/llama-7b'
        ms_download=MsDownload(repo_id)
        ms_download.run()
    """

    def __init__(self, repo_id) -> None:
        self.repo_id = repo_id

    def run(self) -> None:
        from modelscope import snapshot_download as ms_snapshot_download

        logger.info(f"************ Start downloading {self.repo_id} ************")
        model_dir = ms_snapshot_download(self.repo_id)
        logger.info(f"************ {self.repo_id} Download finish ************")
        logger.info(f"Models saved in {self.local_dir}")


class AutodlDownload:
    def __init__(self, repo_id) -> None:
        self.repo_id = repo_id

    def run(self) -> None:
        import codewithgpu as cg

        logger.info(f"************ Start downloading {self.repo_id} ************")
        cg.model.download(self.repo_id)
        logger.info(f"************ {self.repo_id} Download finish ************")
        # logger.info(f'Models saved in {self.local_dir}')


if __name__ == "__main__":
    # repo_id = "meta-llama/Llama-2-7b-hf"
    # repo_type = "model"
    # local_dir=f"{user_setting['XDG_CACHE_HOME']}/{repo_id.split('/')[-1]}"
    # hf_download=HfDownload(repo_id,local_dir,repo_type=repo_type, is_login=False)
    # hf_download.run()

    repo_id = "skyline2006/llama-7b"
    ms_download = MsDownload(repo_id)
    ms_download.run()
