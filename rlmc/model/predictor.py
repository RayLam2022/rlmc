"""
@File    :   predictor.py
@Time    :   2024/06/30 00:24:58
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

from typing import Union, Dict, List

import torch
import torch.nn as nn


__all__ = [
    "ClsPredictor",
    "SemanticSegmentationPredictor",
    "PosePredictor",
    "ObjDetectPredictor",
    "RegressorPredictor",
    "SequencePredictor",
]


class ClsPredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs


class SemanticSegmentationPredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.args.model.hidden_dim),
            nn.Linear(self.args.model.hidden_dim, self.args.model.num_classes),
        )
        self.act_fn = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.mlp_head(x))


class PosePredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs


class ObjDetectPredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs


class ObbPredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs


class RegressorPredictor(nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs


class SequencePredictor(torch.nn.Module):
    def __init__(self, configs: Dict) -> None:
        super().__init__()
        self.args = configs
