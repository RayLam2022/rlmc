"""
@File    :   systeminfos.py
@Time    :   2024/06/26 14:17:46
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

from typing import List
import os
import os.path as osp
import time
import platform
from importlib import import_module


try:
    pynvml = import_module("pynvml")
except:
    os.system(f"{sys.executable} -m pip install nvidia-ml-py3")
    print("Installing nvidia-ml-py3,if failed,pip install nvidia-ml-py")
    pynvml = import_module("pynvml")


__all__ = ["general_info", "gpu_info"]


def general_info() -> None:
    print("#" * 25 + " System Info " + "#" * 25)
    print(f"platform:{platform.uname()}")
    print(f"cwd:{os.getcwd()}")
    print(f"(.):{osp.abspath('.')}")
    print(f"user:{osp.expanduser('~')}")

    try:
        s = "".join(os.popen("python -V").readlines()).strip()
    except:
        s = "No python"
    finally:
        print(f"{s}")

    s = "".join(os.popen(f"{sys.executable} -V").readlines())
    # print(f"{s}")
    print(f"ppath:{sys.executable}")
    print("#" * 25 + " CUDA Info " + "#" * 25)
    try:
        s = "".join(os.popen("nvcc -V").readlines()).strip()
    except:
        s = "No cuda"
    finally:
        print(f"{s}")


def get_gpu_device() -> List:
    deviceCount = pynvml.nvmlDeviceGetCount()
    gpu_list = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", pynvml.nvmlDeviceGetName(handle))
        gpu_list.append(i)
    return gpu_list


def get_gpu_info(gpu_id: int) -> str:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    M = 1024**2
    gpu_info = "id:{}  total:{:.2f}M free:{:.2f}M  used:{:.2f}M free_rate:{}%".format(
        gpu_id, info.total / M, info.free / M, info.used / M, get_free_rate(gpu_id)
    )
    return gpu_info


def release() -> None:
    # 最后要关闭管理工具
    pynvml.nvmlShutdown()


def get_free_rate(gpu_id: int) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_rate = int((info.free / info.total) * 100)
    return free_rate


def gpu_info(keeptime: float = 0.001, interval_per_verbose: float = 0.2) -> None:
    """
    :param keeptime: minutes
    :return:
    """
    print("#" * 25 + " GPU Info " + "#" * 25)
    start = time.time()

    while True:
        pynvml.nvmlInit()
        gpu_devices = get_gpu_device()
        for gpuid in gpu_devices:
            print(get_gpu_info(gpuid))
        release()
        time.sleep(interval_per_verbose)
        if time.time() - start > 60 * keeptime:
            break


if __name__ == "__main__":
    general_info()
    gpu_info()
