"""
@File    :   semantic_segmentation.py
@Time    :   2024/06/30 00:24:37
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import Union, Dict, List

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import segmentation_models_pytorch as smp
from PIL import Image


__all__ = ["SemanticSegmentationModel"]


seg_models = {
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
    "Linknet": smp.Linknet,
    "MAnet": smp.MAnet,
    "PAN": smp.PAN,
    "PSPNet": smp.PSPNet,
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
}


class SemanticSegmentationModel(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        # model_ref: https://github.com/qubvel/segmentation_models.pytorch
        self.args = configs
        # self.model_names = timm.list_models("deep",pretrained=self.args.model.pretrained)
        # print(self.model_names)
        # self.backbone=timm.create_model(self.args.model.name, pretrained=self.args.model.pretrained)
        self.model = seg_models[self.args.model.modeltype](
            encoder_name=self.args.model.backbone,
            encoder_weights=self.args.model.encoder_weights,
            in_channels=self.args.model.in_channels,
            classes=self.args.model.num_classes,
            activation=self.args.model.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out
