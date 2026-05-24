from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from metricskit.core import MetricBatch


def _flatten_matching_masks(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
        pred_masks = pred_masks[:, 0]
    if target_masks.ndim == 4 and target_masks.shape[1] == 1:
        target_masks = target_masks[:, 0]

    if pred_masks.shape != target_masks.shape:
        raise ValueError(
            "Inputs 'pred_masks' and 'target_masks' must have the same shape after "
            f"removing singleton channel dimensions, got {tuple(pred_masks.shape)} "
            f"and {tuple(target_masks.shape)}."
        )

    if pred_masks.ndim < 2:
        raise ValueError(
            "Inputs 'pred_masks' and 'target_masks' must include a batch dimension "
            "and at least one mask dimension."
        )

    return pred_masks.reshape(pred_masks.shape[0], -1), target_masks.reshape(
        target_masks.shape[0], -1
    )


@dataclass
class DiceScore:
    """Mean Dice score for binary segmentation masks."""

    requires: ClassVar[tuple[str, ...]] = ("pred_masks", "target_masks")

    name: str = "dice"
    threshold: float = 0.5
    from_logits: bool = True
    eps: float = 1e-7

    _sum: float = 0.0
    _count: int = 0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, batch: MetricBatch) -> None:
        pred_masks = batch.binary(
            "pred_masks",
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        target_masks = batch.binary("target_masks", threshold=0.5)
        pred_masks, target_masks = _flatten_matching_masks(pred_masks, target_masks)

        intersection = (pred_masks & target_masks).sum(dim=1)
        denominator = pred_masks.sum(dim=1) + target_masks.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)

        self._sum += float(dice.sum().item())
        self._count += int(dice.numel())

    def compute(self) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum / self._count


@dataclass
class MeanIoU:
    """Mean intersection-over-union for binary segmentation masks."""

    requires: ClassVar[tuple[str, ...]] = ("pred_masks", "target_masks")

    name: str = "iou"
    threshold: float = 0.5
    from_logits: bool = True
    eps: float = 1e-7

    _sum: float = 0.0
    _count: int = 0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, batch: MetricBatch) -> None:
        pred_masks = batch.binary(
            "pred_masks",
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        target_masks = batch.binary("target_masks", threshold=0.5)
        pred_masks, target_masks = _flatten_matching_masks(pred_masks, target_masks)

        intersection = (pred_masks & target_masks).sum(dim=1)
        union = (pred_masks | target_masks).sum(dim=1)
        iou = (intersection + self.eps) / (union + self.eps)

        self._sum += float(iou.sum().item())
        self._count += int(iou.numel())

    def compute(self) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum / self._count


@dataclass
class BinaryMaskAccuracy:
    """Pixel accuracy for binary segmentation masks."""

    requires: ClassVar[tuple[str, ...]] = ("pred_masks", "target_masks")

    name: str = "mask_accuracy"
    threshold: float = 0.5
    from_logits: bool = True

    _correct: int = 0
    _total: int = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, batch: MetricBatch) -> None:
        pred_masks = batch.binary(
            "pred_masks",
            from_logits=self.from_logits,
            threshold=self.threshold,
        )
        target_masks = batch.binary("target_masks", threshold=0.5)

        if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks[:, 0]
        if target_masks.ndim == 4 and target_masks.shape[1] == 1:
            target_masks = target_masks[:, 0]

        if pred_masks.shape != target_masks.shape:
            raise ValueError(
                "Inputs 'pred_masks' and 'target_masks' must have the same shape after "
                f"removing singleton channel dimensions, got {tuple(pred_masks.shape)} "
                f"and {tuple(target_masks.shape)}."
            )

        correct = pred_masks == target_masks
        self._correct += int(correct.sum().item())
        self._total += int(correct.numel())

    def compute(self) -> float:
        if self._total == 0:
            return float("nan")
        return self._correct / self._total
