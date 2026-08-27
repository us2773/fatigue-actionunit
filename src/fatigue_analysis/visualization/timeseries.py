"""時系列可視化。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fatigue_analysis.domain.models import OpenFaceSeries

SERIES_STYLES = {
    "raw_validated": {"color": "#1f77b4", "linestyle": "-"},
    "trend": {"color": "#d62728", "linestyle": "-"},
    "residual": {"color": "#2ca02c", "linestyle": "--"},
}


def plot_timeseries(
    series_by_kind: Mapping[str, OpenFaceSeries],
    *,
    signal_ids: tuple[str, ...],
    output_png: Path,
    plot_data_csv: Path,
    run_id: str,
) -> None:
    """1サンプル複数信号の時系列PNGと元CSVを保存する。"""

    if not series_by_kind:
        raise ValueError("series_by_kind は空にできません。")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_data_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_id = next(iter(series_by_kind.values())).sample_id
    figure, axes = plt.subplots(
        len(signal_ids),
        1,
        figsize=(8, max(2.5, 2.2 * len(signal_ids))),
        squeeze=False,
        sharex=True,
    )
    rows: list[dict[str, object]] = []
    for axis, signal_id in zip(axes[:, 0], signal_ids, strict=True):
        for series_kind, series in series_by_kind.items():
            if signal_id not in series.signals:
                continue
            style = SERIES_STYLES.get(series_kind, {"linestyle": "-"})
            axis.plot(
                series.timestamps_s,
                series.signals[signal_id],
                label=series_kind,
                linewidth=1.4,
                **style,
            )
            rows.extend(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "signal_id": signal_id,
                    "series_kind": series_kind,
                    "timestamp_s": float(timestamp),
                    "value": float(value),
                }
                for timestamp, value in zip(
                    series.timestamps_s,
                    series.signals[signal_id],
                    strict=True,
                )
            )
        axis.set_ylabel(f"{signal_id}\nintensity")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("timestamp_s")
    figure.suptitle(f"{sample_id} / {run_id}")
    figure.tight_layout()
    figure.savefig(output_png, dpi=150)
    plt.close(figure)

    with plot_data_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = (
            "run_id",
            "sample_id",
            "signal_id",
            "series_kind",
            "timestamp_s",
            "value",
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
