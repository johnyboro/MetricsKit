# MetricsKit

A simple Python metrics and history tracker.

## History

`History` stores metric rows in an in-memory SQLite database. Each row has a
`phase`, optional `epoch`, optional `step`, metric values, and arbitrary
dimensions such as `fold`, `seed`, `run`, or `task`.

```python
from metricskit import History

history = History()

history.add(
    "val",
    {"loss": 0.42, "accuracy": 0.91},
    epoch=3,
    fold=1,
    seed=42,
)

latest = history.latest(phase="val", fold=1)
rows = history.rows(phase="val", fold=1)
accuracy_by_epoch = history.aggregate(
    "mean",
    phase="val",
    metric="accuracy",
    group_by=["epoch"],
)

log_values = history.to_log_dict(phase="val", fold=1, prefix_by=["fold", "phase"])
# {"fold_1/val/accuracy": 0.91, "fold_1/val/loss": 0.42}
```

Adding another row with the same `phase`, `epoch`, `step`, and dimensions
replaces the existing metric values.

```python
history.add("val", {"loss": 0.5}, epoch=3, fold=1)
history.add("val", {"loss": 0.4}, epoch=3, fold=1)

assert history.latest(phase="val", epoch=3, fold=1)["loss"] == 0.4
```

K-fold runs use dimensions directly:

```python
for fold in range(5):
    for epoch in range(10):
        history.add("train", {"loss": 0.8}, epoch=epoch, fold=fold)
        history.add("val", {"accuracy": 0.9}, epoch=epoch, fold=fold)

mean_accuracy_by_epoch = history.aggregate(
    "mean",
    phase="val",
    metric="accuracy",
    group_by=["epoch"],
)

std_accuracy_by_epoch = history.aggregate(
    "std",
    phase="val",
    metric="accuracy",
    group_by=["epoch"],
)
```

## Logger output

`Metrics.to_dict()` returns a flat metric dictionary and includes metrics that
export multiple values, such as `PrecisionRecallFScore`.

```python
from metricskit import Metrics, PrecisionRecallFScore

metrics = Metrics([PrecisionRecallFScore()])

# After update(...):
metrics.to_dict(prefix="val")
# {"val/precision": ..., "val/recall": ..., "val/f1_macro": ...}
```

`History.to_log_dict()` formats the latest matching row for loggers such as
Weights & Biases. Prefixes can include arbitrary dimensions.

```python
history.add("val", metrics, epoch=epoch, fold=fold)

wandb.log(
    history.to_log_dict(
        phase="val",
        fold=fold,
        prefix_by=["fold", "phase"],
    ),
    step=epoch,
)
# logs fold_1/val/precision, fold_1/val/recall, fold_1/val/f1_macro, ...
```
