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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SemanticSegmentationPredictor(nn.Module):
    def __init__(self, dim, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        return self.mlp_head(x)


class PosePredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ObjDetectPredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RegressorPredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SequencePredictor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
