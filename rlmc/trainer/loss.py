'''
@File    :   loss.py
@Time    :   2024/06/28 11:33:05
@Author  :   RayLam
@Contact :   1027196450@qq.com
'''


import torch.nn as nn

__all__=['loss_method']


loss_method={'BCELoss':nn.BCELoss()}