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
```
