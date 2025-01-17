"""
https://blog.csdn.net/sinat_29047129/article/details/103642140
https://www.cnblogs.com/Trevo/p/11795503.html
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""

import numpy as np

__all__ = ["metrics"]


class SegmentationMetric:
    def __init__(self, numClass: int) -> None:
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self) -> float:
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / np.maximum(
            self.confusionMatrix.sum(axis=1), 1
        )
        return classAcc

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    # MIoU
    def meanIntersectionOverUnion(self) -> float:
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(
            np.sum(self.confusionMatrix, axis=1)
            + np.sum(self.confusionMatrix, axis=0)
            - np.diag(self.confusionMatrix),
            1,
        )
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict: np.ndarray, imgLabel: np.ndarray):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self) -> float:
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1)
            + np.sum(self.confusionMatrix, axis=0)
            - np.diag(self.confusionMatrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict: np.ndarray, imgLabel: np.ndarray) -> None:
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self) -> None:
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


metrics = {"SegmetationMetric": SegmentationMetric}
