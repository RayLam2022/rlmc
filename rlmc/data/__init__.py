import sys

if "." not in sys.path:
    sys.path.append(".")

__all__ = [
    "datasets"
]

from rlmc.data.semantic_segmentation import SemanticSegmentationDataset

datasets = {
    "seg": SemanticSegmentationDataset,
}