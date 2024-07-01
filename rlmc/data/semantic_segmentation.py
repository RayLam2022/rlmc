"""
@File    :   semantic_segmentation.py
@Time    :   2024/06/30 00:25:55
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")

import random
import os
import os.path as osp
from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)

from rlmc.data.utils import mask2onehot

__all__ = ["SemanticSegmentationDataset"]



class SemanticSegmentationDataset(Dataset):
    def __init__(self, configs, data_dir):
        self.args = configs
        self.data_dir = data_dir
        self.imgs = glob(osp.join(data_dir, "images", "*.jpg"))
        self.input_size = configs.dataset.input_size
        self.transform_img = transforms.Compose(
            [
                ToTensor(), #转为torch.Tensor,变0~1
                Resize(
                    (self.input_size.height, self.input_size.width),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                Normalize(
                    self.args.dataset.transformNormalize.mean,
                    self.args.dataset.transformNormalize.std,
                ),
            ]
        )


    def __len__(
        self,
    ):
        return len(self.imgs)

    def __getitem__(self, item):
        file_name = osp.splitext(osp.basename(self.imgs[item]))[0]
        img = Image.open(self.imgs[item])
        img = self.transform_img(img)
        mask = Image.open(osp.join(self.data_dir, "masks", file_name + ".png"))
        mask=np.array(mask)
        mask[np.where(mask==255)]=0 # 剔除voc的255边

        mask=cv2.resize(mask,(self.input_size.width, self.input_size.height),interpolation =cv2.INTER_NEAREST)
        
        mask=torch.from_numpy(mask).long()
        mask=torch.unsqueeze(mask, dim=0)
        # mask_=mask2onehot(mask,self.args.dataset.num_classes)
        
        mask_=torch.zeros(self.args.dataset.num_classes,self.input_size.height,self.input_size.width)
        mask_=mask_.scatter(0, mask,1)  # scatter(dim , label ,1 )  # one-hot
        
        mask_=mask_.float()
        return img, mask_
