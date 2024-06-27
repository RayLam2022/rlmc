import re
import copy
from typing import Any, Union, Optional, Generator, Iterable
from typing import Sequence, Tuple, Dict, List, Set
import pprint

__all__ = ["SuDict"]


class SuDict(dict):

    def __init__(self, *args, **kwargs) -> None:
        # 不能在实例中增加其他属性
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                self[arg[0]] = self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    self[key] = self._hook(val)

        for key, val in kwargs.items():
            self[key] = self._hook(val)

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setitem__(self, name, value):
        super().__setitem__(name, value)

    def __getitem__(self, name):
        # print('getitem',self.__class_getitem__(name))
        return super().__getitem__(name)

    def __missing__(self, name):
        return ""  # f'missing key:{name}'

    def __delattr__(self, name):
        del self[name]

    def _flatten(
        self,
        data: Dict,
        results: Optional[str] = None,
        temp_sep: str = "_",
        digit_symbol: str = "#",
    ) -> Generator:
        if results is None:
            results = ""
        for key, value in data.items():
            if isinstance(key, int):
                k = f"{digit_symbol}{key}"
            else:
                k = f"{key}"

            if isinstance(value, dict) and value:
                for item in self._flatten(value, f"{results}{temp_sep}{k}"):
                    yield item
            else:
                yield f"{results}{temp_sep}{k}", value

    def create_flat_kv(self, temp_sep: str = "_") -> List[Tuple]:
        data = copy.deepcopy(self)
        flatdic = {k: v for k, v in self._flatten(data, temp_sep=temp_sep)}
        return sorted(flatdic.items(), key=lambda d: len(d[0].split(temp_sep)))

    @classmethod
    def _hook(cls, item: Sequence):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else item
                    for item in value
                )
            else:
                base[key] = value
        return base

    def update(self, *args, **kwargs) -> None:
        other = dict()
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if (
                (k not in self)
                or (not isinstance(self[k], dict))
                or (not isinstance(v, dict))
            ):
                self[k] = v
            else:
                self[k].update(v)

    def keystr2keylist(self, key_str, sep=".", digit_symbol="#"):
        lis = key_str.split(sep)
        lis = [int(l[1:]) if l[0] == digit_symbol else l for l in lis]
        return lis

    def search_all_keys(
        self,
        key: Union[str, List],
        flat_kv: List[Tuple],
        is_head: bool = False,
        temp_sep: str = "_",
        digit_symbol: str = "#",
    ) -> List[Tuple]:

        assert isinstance(
            key, (str, list)
        ), "查找的key要是列表形式如[key1,key2...]，或者以分隔符.分隔的节点序列如key1.key2.key3...字符串形式"

        if isinstance(key, list):
            key = [f"{digit_symbol}{l}" if isinstance(l, int) else l for l in key]
            key = temp_sep.join(key)
        else:
            key.replace(".", temp_sep)

        collection = []
        for k in flat_kv:
            s = k[0][1:]

            if key in s:
                if is_head:
                    pattern = f"^{key}{temp_sep}.*|^{key}$"
                else:
                    pattern = f".*{temp_sep}{key}{temp_sep}.*|.*{temp_sep}{key}$"
                # print('pattern:',pattern,'s:',s)
                result = re.match(pattern, s)
                if not isinstance(result, type(None)):
                    collection.append((s.replace(temp_sep, "."), k[1]))

        return collection


if __name__ == "__main__":
    nest_dict = {
        "a": 1,
        "b": {"c": 2, "d": 3, "e": {"f": 4}},
        "g": {"h": 5},
        "i": 6,
        "j": {"k": {7: {"m": 8}}},
        "n": [1, {"o": 1, "p": [1, 2, 3], "q": {"r": {"s": 100}}}, 3, [1, 2, 3], 5],
    }

    d = SuDict(nest_dict)

    d.update(SuDict({6: {"x": 9}}))
    d.update({"s": "c"})
    pprint.pprint(d)
    print(d.j.k[7])
    print(d.j.k)
    print("-----")
    flat_kv = d.create_flat_kv()
    pprint.pprint(flat_kv)
    print(d.search_all_keys([6, "x"], flat_kv, is_head=True))

    # d.s.e=9
    # print(d.s.e)
