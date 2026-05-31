from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, TypeAlias

import torch


Conversion: TypeAlias = Literal["auto", "binary", "multiclass", "multilabel", "none"]
MetricNames: TypeAlias = Iterable[str] | Literal["all"] | None


class Metric(Protocol):
    """Protocol implemented by stateful metrics consumed by Metrics."""

    name: str
    requires: tuple[str, ...]

    def reset(self) -> None: ...

    def update(self, batch: MetricBatch) -> None: ...

    def compute(self) -> float: ...


class MultiMetric(Metric, Protocol):
    """Protocol for metrics that export multiple scalar values."""

    def compute_all(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class MetricBatch:
    """Normalized view over one model batch for metric implementations."""

    conversion: Conversion = "auto"
    loss: torch.Tensor | float | None = None
    logits: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    preds: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    outputs: Any = None
    batch_size: int | None = None
    threshold: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Any:
        """Return a named input or raise a clear missing-input error."""
        return self.get_input(name)

    def tensor(self, name: str) -> torch.Tensor:
        """Return a named input and validate it is a torch.Tensor."""
        value = self.require(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Metric input '{name}' must be a torch.Tensor, got {type(value).__name__}."
            )
        return value

    def detach_cpu(self, name: str) -> torch.Tensor:
        """Return a named tensor detached and moved to CPU."""
        return self._to_cpu(self.tensor(name))

    def to_list(self, name: str) -> list[Any]:
        """Return a named tensor as a Python list."""
        return self.detach_cpu(name).tolist()

    def sigmoid(self, name: str) -> torch.Tensor:
        """Apply sigmoid to a named tensor."""
        return torch.sigmoid(self.tensor(name))

    def softmax(self, name: str, dim: int = -1) -> torch.Tensor:
        """Apply softmax to a named tensor."""
        return torch.softmax(self.tensor(name), dim=dim)

    def threshold_tensor(self, name: str, threshold: float = 0.5) -> torch.Tensor:
        """Threshold a named tensor into a boolean tensor."""
        return self.tensor(name) >= threshold

    def binary(
        self,
        name: str,
        *,
        from_logits: bool = False,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Convert a named tensor to a boolean tensor via optional sigmoid + threshold."""
        values = self.sigmoid(name) if from_logits else self.tensor(name)
        return values >= threshold

    def require_loss(self) -> torch.Tensor | float:
        value = self.require("loss")
        if not isinstance(value, torch.Tensor | float | int):
            raise TypeError(
                f"Metric input 'loss' must be a torch.Tensor or number, got {type(value).__name__}."
            )
        return value

    def require_targets(self) -> torch.Tensor:
        return self.tensor("targets")

    def require_preds(self) -> torch.Tensor:
        if self.preds is not None:
            return self.preds

        if self.conversion == "none":
            raise ValueError("Metric requires explicit 'preds'.")

        if self.logits is not None:
            return self._preds_from_scores(self.logits, logits=True)

        if self.probs is not None:
            return self._preds_from_scores(self.probs, logits=False)

        raise ValueError("Metric requires 'preds', or convertible 'logits'/'probs'.")

    def require_probs(self) -> torch.Tensor:
        if self.probs is not None:
            return self.probs

        if self.logits is None:
            raise ValueError("Metric requires 'probs' or 'logits'.")

        conversion = self._resolve_conversion(self.logits)

        if conversion == "binary":
            if self.logits.ndim == 1 or self.logits.shape[-1] == 1:
                return torch.sigmoid(self.logits).reshape(-1)
            return torch.softmax(self.logits, dim=-1)[:, 1]

        if conversion == "multiclass":
            return torch.softmax(self.logits, dim=-1)

        if conversion == "multilabel":
            return torch.sigmoid(self.logits)

        raise ValueError("Metric requires explicit 'probs'.")

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        if self.targets is not None:
            return int(self.targets.shape[0])
        if self.preds is not None:
            return int(self.preds.shape[0])
        if self.logits is not None:
            return int(self.logits.shape[0])
        if self.probs is not None:
            return int(self.probs.shape[0])
        if isinstance(self.outputs, torch.Tensor):
            return int(self.outputs.shape[0])
        raise ValueError("Unable to infer batch size.")

    def targets_cpu(self) -> torch.Tensor:
        return self.detach_cpu("targets")

    def preds_cpu(self) -> torch.Tensor:
        return self._to_cpu(self.require_preds())

    def probs_cpu(self) -> torch.Tensor:
        return self._to_cpu(self.require_probs())

    def targets_list(self) -> list[Any]:
        return self.to_list("targets")

    def preds_list(self) -> list[Any]:
        return self.preds_cpu().tolist()

    def probs_list(self) -> list[Any]:
        return self.probs_cpu().tolist()

    def get_input(self, name: str) -> Any:
        if name == "loss":
            if self.loss is None:
                raise self._missing_input_error(name)
            return self.loss
        if name == "targets":
            if self.targets is None:
                raise self._missing_input_error(name)
            return self.targets
        if name == "preds":
            return self.require_preds()
        if name == "probs":
            return self.require_probs()
        if name == "logits":
            if self.logits is None:
                raise self._missing_input_error(name)
            return self.logits
        if name == "outputs":
            if self.outputs is None:
                raise self._missing_input_error(name)
            return self.outputs
        if name == "batch_size":
            return self.resolved_batch_size()
        if name in self.extras:
            return self.extras[name]
        raise self._missing_input_error(name)

    def _preds_from_scores(self, scores: torch.Tensor, *, logits: bool) -> torch.Tensor:
        conversion = self._resolve_conversion(scores)

        if conversion == "binary":
            if scores.ndim == 1 or scores.shape[-1] == 1:
                probs = torch.sigmoid(scores) if logits else scores
                return (probs.reshape(-1) >= self.threshold).long()
            return torch.argmax(scores, dim=-1)

        if conversion == "multiclass":
            return torch.argmax(scores, dim=-1)

        if conversion == "multilabel":
            probs = torch.sigmoid(scores) if logits else scores
            return (probs >= self.threshold).long()

        raise ValueError("Metric requires explicit 'preds'.")

    def _resolve_conversion(self, scores: torch.Tensor) -> Conversion:
        if self.conversion != "auto":
            return self.conversion

        if scores.ndim == 1 or scores.shape[-1] == 1:
            return "binary"

        if self.targets is None:
            raise ValueError(
                "Cannot infer conversion without 'targets'. Set conversion explicitly."
            )

        if tuple(self.targets.shape) == tuple(scores.shape):
            return "multilabel"

        if self.targets.ndim == 1:
            if scores.shape[-1] == 2:
                return "binary"
            return "multiclass"

        raise ValueError("Cannot infer conversion. Set conversion explicitly.")

    @staticmethod
    def _to_cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu()

    def _missing_input_error(self, name: str) -> ValueError:
        available = self.available_inputs
        available_text = ", ".join(available) if available else "none"
        return ValueError(
            f"Metric input '{name}' was not provided. Available inputs: {available_text}."
        )

    @property
    def available_inputs(self) -> tuple[str, ...]:
        names: list[str] = []
        for name in ("loss", "logits", "probs", "preds", "targets", "outputs"):
            if getattr(self, name) is not None:
                names.append(name)
        if self.batch_size is not None:
            names.append("batch_size")
        names.extend(self.extras.keys())
        return tuple(names)


@dataclass
class Metrics:
    metrics: Iterable[Metric] | dict[str, Metric]
    conversion: Conversion = "auto"
    threshold: float = 0.5

    _metrics: dict[str, Metric] = field(init=False, repr=False)
    _cache: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.metrics, dict):
            self._metrics = dict(self.metrics)
        else:
            self._metrics = {metric.name: metric for metric in self.metrics}

        if not self._metrics:
            raise ValueError("Metrics requires at least one metric.")

    def reset(self) -> None:
        """Reset all metric state and cached computed values."""
        for metric in self._metrics.values():
            metric.reset()
        self._invalidate_cache()

    def update(
        self,
        *,
        loss: torch.Tensor | float | None = None,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        preds: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        outputs: Any = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Consume one batch and forward a normalized MetricBatch to each metric."""
        batch = MetricBatch(
            conversion=self.conversion,
            loss=loss,
            logits=logits,
            probs=probs,
            preds=preds,
            targets=targets,
            outputs=outputs,
            batch_size=batch_size,
            threshold=self.threshold,
            extras=kwargs,
        )

        for metric in self._metrics.values():
            try:
                metric.update(batch)
            except (TypeError, ValueError) as exc:
                requires = getattr(metric, "requires", ())
                required_text = ", ".join(requires) if requires else "not declared"
                raise type(exc)(
                    f"Failed to update metric '{metric.name}'. Required inputs: {required_text}. {exc}"
                ) from exc

        self._invalidate_cache()

    def get(self, name: str) -> float:
        """Get a single metric by name."""
        self._ensure_cache()
        if name not in self._cache:
            raise ValueError(
                f"Metric of name: {name} not supported, supported metrics: {self.supported_metrics}"
            )
        return self._cache[name]

    def get_all(self, names: MetricNames = None) -> dict[str, float]:
        """Compute selected metrics, or all configured metrics if names is None."""
        self._ensure_cache()
        if names is None or names == "all":
            selected = self.supported_metrics
        else:
            selected = tuple(names)
        return {name: self.get(name) for name in selected}

    @property
    def supported_metrics(self) -> tuple[str, ...]:
        """List of supported metric names."""
        self._ensure_cache()
        return tuple(self._cache.keys())

    def to_dict(
        self,
        names: MetricNames = None,
        prefix: str | None = None,
    ) -> dict[str, float]:
        """
        Logger-friendly export.
        Example:
            to_dict(prefix="val") -> {"val/loss": ..., "val/f1_macro": ...}
        """
        values = self.get_all(names)
        if prefix is None:
            return values

        return {f"{prefix}/{name}": metric for name, metric in values.items()}

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    def _ensure_cache(self) -> None:
        if self._cache:
            return

        for metric in self._metrics.values():
            for name, value in self._compute_metric_values(metric).items():
                if name in self._cache:
                    raise ValueError(f"Duplicate metric output name: {name}")
                self._cache[name] = value

    def _compute_metric_values(self, metric: Metric) -> dict[str, float]:
        compute_all = getattr(metric, "compute_all", None)
        if callable(compute_all):
            return {name: float(value) for name, value in compute_all().items()}
        return {metric.name: float(metric.compute())}
