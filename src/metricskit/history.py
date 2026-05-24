from metricskit.core import Metrics
from typing import Iterable, Mapping, TypeAlias, Literal

from dataclasses import dataclass, field

MetricValue: TypeAlias = float | int
MetricRow: TypeAlias = dict[str, MetricValue]
EpochSelector: TypeAlias = int | range | Iterable[int] | Literal["all"] | None


@dataclass
class History:
    rows: dict[str, list[MetricRow]] = field(default_factory=dict)

    def add_metrics(
        self, phase: str, metrics: Metrics, epoch: int | None = None
    ) -> None:
        self.add_row(phase, metrics.to_dict(), epoch=epoch)

    def add_row(
        self,
        phase: str,
        row: Mapping[str, MetricValue],
        epoch: int | None = None,
    ) -> None:
        row_dict = dict(row)

        if phase not in self.rows:
            self.rows[phase] = []

        phase_rows = self.rows[phase]

        if epoch is None:
            phase_rows.append(row_dict)
            return

        if epoch < 0:
            epoch += len(phase_rows)

        if epoch == len(phase_rows):
            phase_rows.append(row_dict)
            return

        if 0 <= epoch < len(phase_rows):
            phase_rows[epoch] = row_dict
            return

        raise IndexError(
            f"Epoch index {epoch} out of range for phase '{phase}' with "
            f"{len(phase_rows)} stored epoch(s)"
        )

    def get_metrics(
        self,
        epoch: EpochSelector = None,
        phases: Iterable[str] | None = None,
        prefix: bool = False,
    ) -> dict[str, MetricRow] | dict[str, list[MetricRow]]:
        if phases is None:
            selected_phases = list(self.rows.keys())
        else:
            selected_phases = list(phases)

        multi = not (epoch is None or isinstance(epoch, int))

        if multi:
            result_multi: dict[str, list[MetricRow]] = {}
            for phase in selected_phases:
                phase_rows = self.rows.get(phase, [])
                indices = self._resolve_epochs(phase, epoch)
                result_multi[phase] = [
                    self._format_row(phase, phase_rows[i], prefix=prefix)
                    for i in indices
                ]
            return result_multi

        result_single: dict[str, MetricRow] = {}
        for phase in selected_phases:
            phase_rows = self.rows.get(phase, [])
            idx = self._resolve_epochs(phase, epoch)[0]
            result_single[phase] = self._format_row(
                phase,
                phase_rows[idx],
                prefix=prefix,
            )
        return result_single

    def _format_row(self, phase: str, row: MetricRow, *, prefix: bool) -> MetricRow:
        if not prefix:
            return dict(row)

        return {f"{phase}/{name}": value for name, value in row.items()}

    def _resolve_epochs(self, phase: str, epochs: EpochSelector) -> list[int]:
        phase_rows = self.rows.get(phase, [])
        n_rows = len(phase_rows)
        if not phase_rows:
            raise ValueError(f"No rows found for phase '{phase}'")

        if epochs is None:
            requested = [-1]
        elif epochs == "all":
            requested = list(range(n_rows))
        elif isinstance(epochs, int):
            requested = [epochs]
        else:
            requested = list(epochs)

        resolved: list[int] = []

        for raw_idx in requested:
            idx = raw_idx + n_rows if raw_idx < 0 else raw_idx
            if idx < 0 or idx >= n_rows:
                raise IndexError(
                    f"Epoch index {raw_idx} out of range for phase '{phase}' "
                    f"with {n_rows} stored epoch(s)"
                )
            resolved.append(idx)

        return resolved
