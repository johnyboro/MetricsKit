from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import torch
from metricskit.core import MetricBatch
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


@dataclass
class MeanLoss:
    requires: ClassVar[tuple[str, ...]] = (
        "loss",
        "batch_size or targets/preds/logits/probs/outputs",
    )
    name: str = "loss"

    _loss_sum: float = 0.0
    _num_samples: int = 0

    def reset(self) -> None:
        self._loss_sum = 0.0
        self._num_samples = 0

    def update(self, batch: MetricBatch) -> None:
        loss = batch.require_loss()
        loss_value = (
            float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
        )
        batch_size = batch.resolved_batch_size()
        self._loss_sum += loss_value * batch_size
        self._num_samples += batch_size

    def compute(self) -> float:
        if self._num_samples == 0:
            return float("nan")
        return self._loss_sum / self._num_samples


@dataclass
class Accuracy:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds or logits/probs")
    name: str = "accuracy"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        return float(accuracy_score(self._targets, self._preds))


@dataclass
class PrecisionRecallFScore:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds or logits/probs")
    average: str = "macro"
    zero_division: int = 0
    precision_name: str = "precision"
    recall_name: str = "recall"
    fscore_name: str = "f1_macro"

    name: str = field(init=False)
    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = self.fscore_name

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        return self.compute_all()[self.fscore_name]

    def compute_all(self) -> dict[str, float]:
        if not self._targets or not self._preds:
            val = float("nan")
            return {
                self.precision_name: val,
                self.recall_name: val,
                self.fscore_name: val,
            }

        precision, recall, fscore, _ = precision_recall_fscore_support(
            self._targets,
            self._preds,
            average=self.average,
            zero_division=self.zero_division,
        )
        return {
            self.precision_name: float(precision),
            self.recall_name: float(recall),
            self.fscore_name: float(fscore),
        }


@dataclass
class Precision:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds or logits/probs")
    average: str = "macro"
    zero_division: int = 0
    name: str = "precision"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        precision, _, _, _ = precision_recall_fscore_support(
            self._targets,
            self._preds,
            average=self.average,
            zero_division=self.zero_division,
        )
        return float(precision)


@dataclass
class Recall:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds or logits/probs")
    average: str = "macro"
    zero_division: int = 0
    name: str = "recall"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        _, recall, _, _ = precision_recall_fscore_support(
            self._targets,
            self._preds,
            average=self.average,
            zero_division=self.zero_division,
        )
        return float(recall)


@dataclass
class F1Score:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds or logits/probs")
    average: str = "macro"
    zero_division: int = 0
    name: str = "f1_macro"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        _, _, fscore, _ = precision_recall_fscore_support(
            self._targets,
            self._preds,
            average=self.average,
            zero_division=self.zero_division,
        )
        return float(fscore)


@dataclass
class AUROC:
    requires: ClassVar[tuple[str, ...]] = ("targets", "probs or logits")
    name: str = "auc"

    _targets: list[Any] = field(default_factory=list)
    _probs: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._probs.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._probs.extend(batch.probs_list())

    def compute(self) -> float:
        if not self._targets or not self._probs:
            return float("nan")
        try:
            return float(roc_auc_score(self._targets, self._probs))
        except ValueError:
            return float("nan")


@dataclass
class MAE:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds")
    name: str = "mae"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        return float(mean_absolute_error(self._targets, self._preds))


@dataclass
class MSE:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds")
    name: str = "mse"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        return float(mean_squared_error(self._targets, self._preds))


@dataclass
class RMSE(MSE):
    name: str = "rmse"

    def compute(self) -> float:
        mse = super().compute()
        return mse**0.5


@dataclass
class R2:
    requires: ClassVar[tuple[str, ...]] = ("targets", "preds")
    name: str = "r2"

    _targets: list[Any] = field(default_factory=list)
    _preds: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self._targets.clear()
        self._preds.clear()

    def update(self, batch: MetricBatch) -> None:
        self._targets.extend(batch.targets_list())
        self._preds.extend(batch.preds_list())

    def compute(self) -> float:
        if not self._targets or not self._preds:
            return float("nan")
        return float(r2_score(self._targets, self._preds))


@dataclass
class FunctionMetric:
    """Adapter for simple custom metrics computed per batch then averaged."""

    name: str
    fn: Callable[..., float]
    inputs: tuple[str, ...]

    _values: list[float] = field(default_factory=list)
    _weights: list[int] = field(default_factory=list)

    @property
    def requires(self) -> tuple[str, ...]:
        return self.inputs

    def reset(self) -> None:
        self._values.clear()
        self._weights.clear()

    def update(self, batch: MetricBatch) -> None:
        args = [batch.get_input(name) for name in self.inputs]
        self._values.append(float(self.fn(*args)))
        self._weights.append(batch.resolved_batch_size())

    def compute(self) -> float:
        if not self._values:
            return float("nan")
        total_weight = sum(self._weights)
        if total_weight == 0:
            return float("nan")
        return (
            sum(value * weight for value, weight in zip(self._values, self._weights))
            / total_weight
        )
