"""
@File    :   multithread.py
@Time    :   2024/06/21 00:19:26
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if '.' not in sys.path: sys.path.append(".")

from typing import Any, Union, Callable, Iterable, List, Dict, Tuple
import concurrent.futures as cf

import time

from tqdm import tqdm
from rlmc.utils.logger import Logger


__all__ = ["MultiThread"]

logger = Logger(__name__, level=Logger.DEBUG)


class MultiThread:

    def __init__(self, func: Callable, worker: int) -> None:
        self.func = func
        self.worker = worker

    def unpack_args(self, idx_args: Tuple[int, Union[Dict, Tuple]]) -> Tuple[int, Any]:
        args = idx_args[1]
        idx = idx_args[0]
        if isinstance(args, tuple):
            return idx, self.func(*args)
        elif isinstance(args, dict):
            return idx, self.func(**args)
        else:
            raise TypeError("args must be tuple or dict")

    def run(self, tasklist: Iterable[Any]) -> List:
        collector = []
        with cf.ThreadPoolExecutor(max_workers=self.worker) as executor:
            future_to_result = {
                executor.submit(self.unpack_args, (idx, i)): (idx, i)
                for idx, i in enumerate(tasklist)
            }

        for future in cf.as_completed(future_to_result):
            result = future.result()
            print(f"Task {future_to_result[future]} returned {result}")
            collector.append(result)
        return collector


def test_unit(a, b):
    return a + b, b


def test_unit_no_return(a, b):
    print(a + b, b)


if __name__ == "__main__":

    a = [i for i in range(100)]
    b = [i + 1 for i in range(100)]
    task_list = list(zip(a, b))

    task_list = []
    for i in range(10):
        task_list.append({"a": i, "b": i + 1})
    # print(task_list)

    mut = MultiThread(test_unit, 4)
    res = mut.run(task_list)
    print(res)
