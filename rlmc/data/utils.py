from typing import List

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize

__all__ = [
    "invert_transform",
    "onehot2mask",
    "mask2onehot",
    "letterbox_image",
    "metrics_all",
]


def invert_transform(torch_normalize_tensor, mean: List[float], std: List[float]):
    """
    反归一化同时调整成 h,w,c
    """
    mean_ = torch.tensor(mean)
    std_ = torch.tensor(std)
    MEAN = [-mean / std for mean, std in zip(mean_, std_)]
    STD = [1 / std for std in std_]
    denormalize = Normalize(mean=MEAN, std=STD)
    tensor = denormalize(torch_normalize_tensor) * 255
    tensor = tensor.permute(1, 2, 0).long()
    return tensor


def onehot2mask(torch_tensor, axis=0):
    mask = torch.argmax(torch_tensor, axis=axis)
    return mask


def mask2onehot(mask, num_classes):
    """
    (H,W) to (K,H,W)
    """
    mask_ = [mask == i for i in range(num_classes)]
    mask_ = np.array(mask_)
    return mask_


def letterbox_image(image, resize_width, resize_height):
    """
    图片统一尺寸，补灰条操作
    :param: image
    :return: new_image
    """
    ih = image.shape[0]
    iw = image.shape[1]

    w = resize_width
    h = resize_height

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh))
    new_image = np.full(
        (h, w, 3), 128, dtype=image.dtype
    )  # channels为3，仅jpg和三通道图

    new_image[
        (h - nh) // 2 : (h - nh) // 2 + nh, (w - nw) // 2 : (w - nw) // 2 + nw, :
    ] = image
    # print(new_image.shape)
    return new_image
