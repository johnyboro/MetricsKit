from metricskit.history import History
from metricskit.core import Conversion, Metric, MetricBatch, Metrics, MultiMetric
from metricskit.metrics import (
    AUROC,
    MAE,
    MSE,
    RMSE,
    R2,
    Accuracy,
    F1Score,
    FunctionMetric,
    MeanLoss,
    Precision,
    PrecisionRecallFScore,
    Recall,
)
from metricskit.segmentation import BinaryMaskAccuracy, DiceScore, MeanIoU

__all__ = [
    "AUROC",
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "Accuracy",
    "BinaryMaskAccuracy",
    "Conversion",
    "DiceScore",
    "F1Score",
    "FunctionMetric",
    "History",
    "MeanLoss",
    "MeanIoU",
    "Metric",
    "MetricBatch",
    "Metrics",
    "MultiMetric",
    "Precision",
    "PrecisionRecallFScore",
    "Recall",
]


def main() -> None:
    print("Hello from metricskit!")
