"""
@File    :   multipro.py
@Time    :   2024/06/20 16:38:18
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

sys.path.append(".")

from typing import Any, Union, Callable, Iterable, List, Dict
import multiprocessing as mp
import time

from tqdm import tqdm
from rlmc.utils.logger import Logger


__all__ = ["MultiProcess"]

logger = Logger(__name__, level=Logger.DEBUG)


class MultiProcess:
    """
    多进程在jupyter会报错或无返回数据,要生成py再运行,另外win多进程要有if __name__=='__main__':
    """

    def __init__(self, func: Callable, worker: int) -> None:
        self.func = func
        self.worker = worker

    def manager(self) -> None:
        mgr = mp.Manager()
        self.mgrd = mgr.dict()

    def unpack_args(self, args: Union[tuple, dict]) -> Callable:
        if isinstance(args, tuple):
            return self.func(*args)
        elif isinstance(args, dict):
            return self.func(**args)
        else:
            raise TypeError("args must be tuple or dict")

    def run(self, tasklist: Iterable[Any], chunksize: int) -> List:
        pool = mp.Pool(processes=self.worker)
        try:
            res = list(tqdm(pool.imap(self.unpack_args, tasklist, chunksize=chunksize)))
            pool.close()  # 执行join()前必须执行close(),表示不能继续添加新的进程了
            pool.join()
        except:
            pool.terminate()
            res = []
        finally:
            return res


def test_unit(a, b):
    return a + b, b


def test_unit_no_return(a, b):
    print(a + b, b)


if __name__ == "__main__":

    a = [i for i in range(100)]
    b = [i + 1 for i in range(100)]
    task_list = list(zip(a, b))

    task_list = []
    for i in range(1000):
        task_list.append({"a": i, "b": i + 1})
    # print(task_list)

    mup = MultiProcess(test_unit, 4)
    res = mup.run(task_list, 20)
    print(res)
