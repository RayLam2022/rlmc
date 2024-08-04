"""
@File    :   __init__.py
@Time    :   2024/06/30 00:25:19
@Author  :   RayLam
@Contact :   1027196450@qq.com
"""

import sys

if "." not in sys.path:
    sys.path.append(".")


__all__ = ["predictors", "models"]

from rlmc.model.cv.semantic_segmentation import SemanticSegmentationModel

from rlmc.model.predictor import (
    ClsPredictor,
    SemanticSegmentationPredictor,
    PosePredictor,
    ObjDetectPredictor,
    RegressorPredictor,
    SequencePredictor,
)

models = {"seg": SemanticSegmentationModel}

predictors = {
    "cls": ClsPredictor,
    "seg": SemanticSegmentationPredictor,
    "pose": PosePredictor,
    "det": ObjDetectPredictor,
    "reg": RegressorPredictor,
    "seq": SequencePredictor,
}
