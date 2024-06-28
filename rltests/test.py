import sys

if '.' not in sys.path: sys.path.append(".")

from typing import Dict

import rlmc
from rlmc import Tainer


def test_unit(a, b):
    return a + b, b


def test_unit_no_return(a, b):
    print(a + b, b)


def main(): ...


if __name__ == "__main__":

    a = [i for i in range(100)]
    b = [i + 1 for i in range(100)]
    task_list = list(zip(a, b))

    task_list = []
    for i in range(10):
        task_list.append({"a": i, "b": i + 1})
    # print(task_list)

    mut = rlmc.MultiThread(test_unit, 4)
    res = mut.run(task_list)
    print(res)
