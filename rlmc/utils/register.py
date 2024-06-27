"""
@File    :   register.py
@Time    :   2024/06/18 22:37:02
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if '.' not in sys.path: sys.path.append(".")

from typing import Any, NoReturn, Union, Callable, Iterable, List, Dict

from rlmc.utils.logger import Logger

__all__ = ["Register"]


logger = Logger(__name__, level=Logger.WARNING)


class Register(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dict: Dict[str, Callable] = dict()

    def register(self, obj: Any) -> Callable:
        def add(key, val):
            if not callable(val):
                raise Exception(f"注册对象必须可调用，接收:{val}为非法对象")
            if key in self._dict:
                logger.warning(f"曾注册 {val.__name__}，现执行覆盖操作")
            self[key] = val
            return val

        if callable(obj):
            return add(obj.__name__, obj)
        else:
            return lambda x: add(obj, x)

    def __call__(self, obj: Any) -> Callable:
        return self.register(obj)

    def __setitem__(self, key: str, val: Callable) -> None:
        self._dict[key] = val

    def __getitem__(self, key: str) -> Callable:
        return self._dict[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def __str__(self) -> str:
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def meta(self, mt_str: str) -> Callable:
        mt = self._dict[mt_str]
        if not isinstance(mt, type):
            raise TypeError("需要装饰元类对象，现有非法对象传入")

        def wrapper(cls):
            cls_ = mt(cls.__name__, cls.__bases__, {})
            return cls_

        return wrapper


if __name__ == "__main__":
    rg = Register()

    @rg("乘法")
    def mult(a: int, b: int):
        return a * b

    @rg
    def minus(a: int, b: int):
        return a - b

    print(rg)
    res = rg["乘法"](7, 2)
    print(rg.values())
    print(res)
