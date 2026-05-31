from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias, cast

from metricskit.core import Metrics

MetricValue: TypeAlias = float | int
MetricRow: TypeAlias = dict[str, MetricValue | str | int | float | bool | None]
DimensionValue: TypeAlias = str | int | float | bool | None
Aggregation: TypeAlias = Literal[
    "mean", "avg", "min", "max", "sum", "count", "last", "std"
]

_RESERVED_DIMENSIONS = {
    "phase",
    "epoch",
    "step",
    "metric",
    "metrics",
    "group_by",
    "aggregation",
}


class History:
    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection or sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._created_order = self._load_created_order()
        self._create_schema()

    def add(
        self,
        phase: str,
        metrics: Metrics | Mapping[str, MetricValue],
        *,
        epoch: int | None = None,
        step: int | None = None,
        **dimensions: DimensionValue,
    ) -> None:
        self._validate_dimensions(dimensions)
        metric_values = self._coerce_metrics(metrics)
        self._validate_metric_dimension_names(metric_values, dimensions)
        record_id = self._find_record(phase, epoch, step, dimensions)
        created_order = self._next_created_order()

        with self._connection:
            if record_id is None:
                cursor = self._connection.execute(
                    """
                    INSERT INTO records (phase, epoch, step, created_order)
                    VALUES (?, ?, ?, ?)
                    """,
                    (phase, epoch, step, created_order),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("Failed to insert history record.")
                record_id = cursor.lastrowid
            else:
                self._connection.execute(
                    "UPDATE records SET created_order = ? WHERE id = ?",
                    (created_order, record_id),
                )
                self._connection.execute(
                    "DELETE FROM metric_values WHERE record_id = ?",
                    (record_id,),
                )
                self._connection.execute(
                    "DELETE FROM dimensions WHERE record_id = ?",
                    (record_id,),
                )

            self._connection.executemany(
                """
                INSERT INTO metric_values (record_id, name, value)
                VALUES (?, ?, ?)
                """,
                [
                    (record_id, name, float(value))
                    for name, value in metric_values.items()
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO dimensions (record_id, key, value)
                VALUES (?, ?, ?)
                """,
                [
                    (record_id, key, self._encode_dimension(value))
                    for key, value in dimensions.items()
                ],
            )

    def rows(
        self,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metric: str | Iterable[str] | None = None,
        **dimensions: DimensionValue,
    ) -> list[MetricRow]:
        self._validate_dimensions(dimensions)
        metric_names = self._normalize_metric_filter(metric)
        record_ids = self._matching_record_ids(phase, epoch, step, dimensions)
        if not record_ids:
            return []

        placeholders = self._placeholders(record_ids)
        metric_sql = ""
        params: list[object] = list(record_ids)
        if metric_names is not None:
            metric_sql = f" AND name IN ({self._placeholders(metric_names)})"
            params.extend(metric_names)

        metric_rows = self._connection.execute(
            f"""
            SELECT record_id, name, value
            FROM metric_values
            WHERE record_id IN ({placeholders}){metric_sql}
            ORDER BY record_id, name
            """,
            params,
        ).fetchall()
        metrics_by_record: dict[int, dict[str, float]] = {}
        for row in metric_rows:
            metrics_by_record.setdefault(int(row["record_id"]), {})[row["name"]] = (
                float(row["value"])
            )

        return [
            self._record_to_row(record_id, metrics_by_record.get(record_id, {}))
            for record_id in record_ids
            if metrics_by_record.get(record_id)
        ]

    def latest(
        self,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metric: str | Iterable[str] | None = None,
        **dimensions: DimensionValue,
    ) -> MetricRow | None:
        rows = self.rows(
            phase=phase,
            epoch=epoch,
            step=step,
            metric=metric,
            **dimensions,
        )
        return rows[-1] if rows else None

    def to_log_dict(
        self,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metric: str | Iterable[str] | None = None,
        prefix_by: Iterable[str] | None = ("phase",),
        include_context: bool = False,
        **dimensions: DimensionValue,
    ) -> MetricRow:
        row = self.latest(
            phase=phase,
            epoch=epoch,
            step=step,
            metric=metric,
            **dimensions,
        )
        if row is None:
            return {}

        metric_names = self._metric_names_from_rows([row])
        prefix_parts = self._log_prefix_parts(row, list(prefix_by or []))
        result: MetricRow = {}

        if include_context:
            for name, value in row.items():
                if name not in metric_names:
                    result[name] = value

        for name in metric_names:
            if name not in row:
                continue
            key = "/".join([*prefix_parts, name]) if prefix_parts else name
            result[key] = row[name]

        return result

    def series(
        self,
        metric: str,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        **dimensions: DimensionValue,
    ) -> list[MetricRow]:
        rows = self.rows(
            phase=phase,
            epoch=epoch,
            step=step,
            metric=metric,
            **dimensions,
        )
        return [
            {
                "phase": row["phase"],
                "epoch": row["epoch"],
                "step": row["step"],
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"phase", "epoch", "step", metric}
                },
                "value": row[metric],
            }
            for row in rows
        ]

    def aggregate(
        self,
        aggregation: Aggregation,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metric: str | Iterable[str] | None = None,
        group_by: Iterable[str] | None = None,
        **dimensions: DimensionValue,
    ) -> float | int | list[MetricRow] | None:
        self._validate_aggregation(aggregation)
        groups = list(group_by or [])
        rows = self.rows(
            phase=phase,
            epoch=epoch,
            step=step,
            metric=metric,
            **dimensions,
        )
        if not rows:
            return [] if groups else None

        metric_names = self._metric_names_from_rows(rows)
        if not metric_names:
            return [] if groups else None

        grouped: dict[tuple[object, ...], dict[str, list[float]]] = {}
        for row in rows:
            group_key = tuple(
                "metric" if group == "metric" else row.get(group) for group in groups
            )
            for metric_name in metric_names:
                if metric_name not in row:
                    continue
                if "metric" in groups:
                    metric_group = tuple(
                        metric_name if group == "metric" else row.get(group)
                        for group in groups
                    )
                    grouped.setdefault(metric_group, {}).setdefault("value", []).append(
                        self._metric_float(row, metric_name)
                    )
                else:
                    grouped.setdefault(group_key, {}).setdefault(
                        metric_name, []
                    ).append(self._metric_float(row, metric_name))

        if not groups:
            values = [
                value
                for metrics in grouped.values()
                for values in metrics.values()
                for value in values
            ]
            return self._aggregate_values(aggregation, values)

        result: list[MetricRow] = []
        for group_key, metric_values in grouped.items():
            row: MetricRow = {}
            for group, value in zip(groups, group_key, strict=True):
                row[group] = self._coerce_row_value(value)
            row.update(
                {
                    name: self._aggregate_values(aggregation, values)
                    for name, values in metric_values.items()
                }
            )
            result.append(row)
        return result

    def aggregate_dict(
        self,
        aggregation: Aggregation | Iterable[Aggregation],
        *,
        phase: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metric: str | Iterable[str] | None = None,
        group_by: Iterable[str] | None = None,
        prefix_by: Iterable[str] | None = ("phase",),
        include_context: bool = False,
        **dimensions: DimensionValue,
    ) -> MetricRow:
        groups = list(group_by or [])
        prefixes = list(prefix_by or [])
        aggregations = self._normalize_aggregations(aggregation)
        context: MetricRow = {
            "phase": phase,
            "epoch": epoch,
            "step": step,
            **dimensions,
        }
        result: MetricRow = {}

        for aggregation_name in aggregations:
            aggregated = self.aggregate(
                aggregation_name,
                phase=phase,
                epoch=epoch,
                step=step,
                metric=metric,
                group_by=groups,
                **dimensions,
            )
            result.update(
                self._format_aggregate_result(
                    aggregated,
                    aggregation_name,
                    metric=metric,
                    groups=groups,
                    prefixes=prefixes,
                    context=context,
                    include_context=include_context,
                )
            )

        return result

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY,
                    phase TEXT NOT NULL,
                    epoch INTEGER,
                    step INTEGER,
                    created_order INTEGER NOT NULL
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS metric_values (
                    record_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS dimensions (
                    record_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE
                )
                """)
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_lookup ON records(phase, epoch, step)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metric_values(record_id, name)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_dimensions_lookup ON dimensions(key, value)"
            )

    def _coerce_metrics(
        self,
        metrics: Metrics | Mapping[str, MetricValue],
    ) -> dict[str, MetricValue]:
        values = metrics.to_dict() if isinstance(metrics, Metrics) else dict(metrics)
        if not values:
            raise ValueError("History requires at least one metric value.")
        return values

    def _find_record(
        self,
        phase: str,
        epoch: int | None,
        step: int | None,
        dimensions: Mapping[str, DimensionValue],
    ) -> int | None:
        records = self._connection.execute(
            """
            SELECT id
            FROM records
            WHERE phase = ?
              AND ((epoch IS NULL AND ? IS NULL) OR epoch = ?)
              AND ((step IS NULL AND ? IS NULL) OR step = ?)
            ORDER BY created_order
            """,
            (phase, epoch, epoch, step, step),
        ).fetchall()

        for record in records:
            record_id = int(record["id"])
            if self._record_dimensions(record_id) == dict(dimensions):
                return record_id
        return None

    def _matching_record_ids(
        self,
        phase: str | None,
        epoch: int | None,
        step: int | None,
        dimensions: Mapping[str, DimensionValue],
    ) -> list[int]:
        clauses: list[str] = []
        params: list[object] = []
        if phase is not None:
            clauses.append("phase = ?")
            params.append(phase)
        if epoch is not None:
            clauses.append("epoch = ?")
            params.append(epoch)
        if step is not None:
            clauses.append("step = ?")
            params.append(step)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        records = self._connection.execute(
            f"SELECT id FROM records {where} ORDER BY created_order",
            params,
        ).fetchall()

        record_ids: list[int] = []
        for record in records:
            record_id = int(record["id"])
            record_dimensions = self._record_dimensions(record_id)
            if all(
                record_dimensions.get(key) == value for key, value in dimensions.items()
            ):
                record_ids.append(record_id)
        return record_ids

    def _record_to_row(self, record_id: int, metrics: Mapping[str, float]) -> MetricRow:
        record = self._connection.execute(
            "SELECT phase, epoch, step FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
        row: MetricRow = {
            "phase": record["phase"],
            "epoch": record["epoch"],
            "step": record["step"],
        }
        row.update(self._record_dimensions(record_id))
        row.update(metrics)
        return row

    def _record_dimensions(self, record_id: int) -> dict[str, DimensionValue]:
        rows = self._connection.execute(
            "SELECT key, value FROM dimensions WHERE record_id = ? ORDER BY key",
            (record_id,),
        ).fetchall()
        return {row["key"]: self._decode_dimension(row["value"]) for row in rows}

    def _load_created_order(self) -> int:
        try:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(created_order), 0) AS created_order FROM records"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["created_order"] if row is not None else 0)

    def _next_created_order(self) -> int:
        self._created_order += 1
        return self._created_order

    def _normalize_metric_filter(
        self,
        metric: str | Iterable[str] | None,
    ) -> list[str] | None:
        if metric is None:
            return None
        if isinstance(metric, str):
            return [metric]
        names = list(metric)
        return names or [""]

    def _metric_names_from_rows(self, rows: Iterable[MetricRow]) -> list[str]:
        known_metrics = {
            row["name"]
            for row in self._connection.execute(
                "SELECT DISTINCT name FROM metric_values ORDER BY name"
            ).fetchall()
        }
        names: list[str] = []
        for row in rows:
            for name in row:
                if name not in known_metrics:
                    continue
                if name not in names:
                    names.append(name)
        return names

    def _aggregate_values(
        self,
        aggregation: Aggregation,
        values: list[float],
    ) -> float | int | None:
        if not values:
            return None
        if aggregation in {"mean", "avg"}:
            return sum(values) / len(values)
        if aggregation == "min":
            return min(values)
        if aggregation == "max":
            return max(values)
        if aggregation == "sum":
            return sum(values)
        if aggregation == "count":
            return len(values)
        if aggregation == "last":
            return values[-1]
        if aggregation == "std":
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return math.sqrt(variance)
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    def _format_aggregate_result(
        self,
        aggregated: float | int | list[MetricRow] | None,
        aggregation: Aggregation,
        *,
        metric: str | Iterable[str] | None,
        groups: list[str],
        prefixes: list[str],
        context: MetricRow,
        include_context: bool,
    ) -> MetricRow:
        if aggregated is None:
            return {}

        if isinstance(aggregated, int | float):
            metric_name = self._scalar_aggregate_metric_name(metric)
            parts = self._aggregate_prefix_parts(context, prefixes, groups=[])
            key = "/".join([*parts, f"{metric_name}_{aggregation}"])
            result: MetricRow = {key: aggregated}
            if include_context:
                result.update({name: value for name, value in context.items() if value is not None})
            return result

        result: MetricRow = {}
        for row in aggregated:
            if include_context:
                for name, value in {**context, **row}.items():
                    if name not in self._aggregate_metric_columns(row, groups):
                        result[name] = value

            parts = self._aggregate_prefix_parts(context | row, prefixes, groups=groups)
            for metric_name in self._aggregate_metric_columns(row, groups):
                value = row[metric_name]
                key_metric_name = str(row["metric"]) if metric_name == "value" and "metric" in row else metric_name
                key = "/".join([*parts, f"{key_metric_name}_{aggregation}"])
                result[key] = value
        return result

    def _aggregate_prefix_parts(
        self,
        row: MetricRow,
        prefixes: list[str],
        *,
        groups: list[str],
    ) -> list[str]:
        parts = self._log_prefix_parts(row, prefixes)
        for group in groups:
            if group == "metric" or group in prefixes:
                continue
            parts.extend(self._log_prefix_parts(row, [group]))
        return parts

    def _aggregate_metric_columns(self, row: MetricRow, groups: list[str]) -> list[str]:
        context_names = {"phase", "epoch", "step", *groups}
        return [name for name in row if name not in context_names]

    def _scalar_aggregate_metric_name(
        self,
        metric: str | Iterable[str] | None,
    ) -> str:
        if isinstance(metric, str):
            return metric
        return "value"

    def _normalize_aggregations(
        self,
        aggregation: Aggregation | Iterable[Aggregation],
    ) -> list[Aggregation]:
        aggregations = (
            [cast(Aggregation, aggregation)]
            if isinstance(aggregation, str)
            else list(aggregation)
        )
        if not aggregations:
            raise ValueError("At least one aggregation is required.")
        for aggregation_name in aggregations:
            self._validate_aggregation(aggregation_name)
        return aggregations

    def _metric_float(self, row: MetricRow, name: str) -> float:
        value = row[name]
        if not isinstance(value, int | float):
            raise TypeError(f"Metric '{name}' must be numeric, got {type(value).__name__}.")
        return float(value)

    def _log_prefix_parts(self, row: MetricRow, prefix_by: Iterable[str]) -> list[str]:
        parts: list[str] = []
        for name in prefix_by:
            if name not in row:
                raise ValueError(f"Cannot prefix log metrics by unknown key: {name}")
            value = row[name]
            if value is None:
                raise ValueError(f"Cannot prefix log metrics by null key: {name}")
            if name == "phase":
                parts.append(str(value))
            else:
                parts.append(f"{name}_{value}")
        return parts

    def _coerce_row_value(self, value: object) -> MetricValue | str | bool | None:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise TypeError(f"History row value must be scalar, got {type(value).__name__}.")

    def _validate_aggregation(self, aggregation: Aggregation) -> None:
        if aggregation not in {
            "mean",
            "avg",
            "min",
            "max",
            "sum",
            "count",
            "last",
            "std",
        }:
            raise ValueError(
                "Unsupported aggregation. Expected one of: mean, avg, min, max, sum, count, last, std."
            )

    def _validate_dimensions(self, dimensions: Mapping[str, DimensionValue]) -> None:
        reserved = _RESERVED_DIMENSIONS.intersection(dimensions)
        if reserved:
            reserved_names = ", ".join(sorted(reserved))
            raise ValueError(
                f"Dimension name(s) reserved by History: {reserved_names}."
            )

    def _validate_metric_dimension_names(
        self,
        metrics: Mapping[str, MetricValue],
        dimensions: Mapping[str, DimensionValue],
    ) -> None:
        collisions = set(metrics).intersection(dimensions)
        if collisions:
            collision_names = ", ".join(sorted(collisions))
            raise ValueError(
                f"Metric and dimension name(s) must be distinct: {collision_names}."
            )

    def _encode_dimension(self, value: DimensionValue) -> str:
        return json.dumps(value, sort_keys=True)

    def _decode_dimension(self, value: str) -> DimensionValue:
        return json.loads(value)

    def _placeholders(self, values: Iterable[object]) -> str:
        return ", ".join("?" for _ in values)
