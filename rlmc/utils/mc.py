"""
@File    :   mc.py
@Time    :   2024/06/19 01:50:44
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import Any, List, Dict

__all__ = ["MetaList"]


# class MetaList(type):

#     def __new__(meta, name, bases, attrs):
#         _cls = super().__new__(meta, name, bases, attrs)
#         setattr(_cls, "__lshift__", lambda self, other: self.extend(other))
#         for method in dir(meta):
#             try:
#                 met = getattr(meta, method)
#                 if "__" not in method and callable(met):
#                     setattr(_cls, method, met)
#             except:
#                 ...
#                 # print('except',method)
#         return _cls


#     def head_anno(self, sep: str = "nan", is_eq_sep: bool = False) -> List:
#         if is_eq_sep:
#             index = [idx for idx, i in enumerate(self) if i == sep]
#         else:
#             index = [idx for idx, i in enumerate(self) if i != sep]
#         res = [[index[idx], index[idx + 1]] for idx, i in enumerate(index[:-1])]
#         if res[-1][1] != len(self) + 1:
#             res.append([res[-1][1], len(self) + 1])
#         # for i in res:
#         #     print(self[i[0]:i[1]])
#         return res

#     def tail_anno(self, sep: str = "nan", is_eq_sep: bool = False) -> List:
#         if is_eq_sep:
#             ind = [idx for idx, i in enumerate(self) if i == sep]
#         else:
#             ind = [idx for idx, i in enumerate(self) if i != sep]

#         index = [-1]
#         index.extend(ind)

#         res = [[index[idx] + 1, index[idx + 1] + 1] for idx, i in enumerate(index[:-1])]
#         # for i in res:
#         #     print(self[i[0]:i[1]])
#         return res

#     def appendleft(self, elem: Any) -> None:
#         self.insert(0, elem)


class Ext(list):

    def head_anno(self, sep: str = "nan", is_eq_sep: bool = False) -> List:
        if is_eq_sep:
            index = [idx for idx, i in enumerate(self) if i == sep]
        else:
            index = [idx for idx, i in enumerate(self) if i != sep]
        res = [[index[idx], index[idx + 1]] for idx, i in enumerate(index[:-1])]
        if res[-1][1] != len(self) + 1:
            res.append([res[-1][1], len(self) + 1])
        # for i in res:
        #     print(self[i[0]:i[1]])
        return res

    def tail_anno(self, sep: str = "nan", is_eq_sep: bool = False) -> List:
        if is_eq_sep:
            ind = [idx for idx, i in enumerate(self) if i == sep]
        else:
            ind = [idx for idx, i in enumerate(self) if i != sep]

        index = [-1]
        index.extend(ind)

        res = [[index[idx] + 1, index[idx + 1] + 1] for idx, i in enumerate(index[:-1])]
        # for i in res:
        #     print(self[i[0]:i[1]])
        return res

    def appendleft(self, elem: Any) -> None:
        self.insert(0, elem)


class MetaList(type):

    def __new__(meta, name, bases, attrs):

        _cls = super().__new__(meta, name, bases, attrs)
        setattr(_cls, "__lshift__", lambda self, other: self.extend(other))
        for method in Ext.__dict__:

            try:
                et = getattr(Ext, method)
                if "__" not in method and callable(et):
                    setattr(_cls, method, et)
            except:
                ...
                # print('except',method)
        return _cls


def metalist(cls):
    return MetaList(cls.__name__, cls.__bases__, {})


if __name__ == "__main__":

    @metalist
    class B(list): ...

    b = B()

    c = [10, "nan", "nan", 80, 70, 60, "nan", 0, 20, "nan", "nan"]
    c = c[::-1]
    print(c)
    b << c
    b.appendleft(666)
    print(b)
    # index=b.head_anno()
    index = b.tail_anno()

    for i in index:
        print(b[i[0] : i[1]])
    d = B([9, 0, "c", "0"])
    type(d)
    print(type(MetaList))
