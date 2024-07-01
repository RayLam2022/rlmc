"""
@File    :   opt.py
@Time    :   2024/06/30 00:50:09
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from torch import optim

__all__ = ["optimizer"]


optimizer = {
    "RMSprop": optim.RMSprop,
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}
