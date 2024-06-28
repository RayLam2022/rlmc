import cv2
import numpy as np


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
