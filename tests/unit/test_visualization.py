from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

from fatigue_analysis.domain.models import OpenFaceSeries
from fatigue_analysis.visualization.distributions import (
    DistributionGroup,
    plot_feature_distribution,
)
from fatigue_analysis.visualization.timeseries import plot_timeseries


def test_plot_timeseries_writes_png_and_plot_data(tmp_path: Path) -> None:
    """時系列PNGと描画元CSVを保存する。"""

    raw_series = _series("raw_validated", np.array([0.1, 0.2, 0.3]))
    trend_series = _series("trend", np.array([0.1, 0.2, 0.3]))
    output_png = tmp_path / "timeseries.png"
    plot_data_csv = tmp_path / "timeseries.csv"

    plot_timeseries(
        {"raw_validated": raw_series, "trend": trend_series},
        signal_ids=("AU01_r",),
        output_png=output_png,
        plot_data_csv=plot_data_csv,
        run_id="run-test",
    )

    rows = list(csv.DictReader(plot_data_csv.open("r", encoding="utf-8-sig")))
    assert output_png.exists()
    assert output_png.stat().st_size > 0
    assert len(rows) == 6
    assert rows[0]["signal_id"] == "AU01_r"


def test_plot_feature_distribution_writes_png_and_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """箱ひげ図+全点PNGと群別統計CSVを保存する。"""

    rows = [
        {"Name": "s1", "is_baseface": "1", "class": "", "AU01_r__trend_stats__mean": "0.1"},
        {"Name": "s2", "is_baseface": "0", "class": "1", "AU01_r__trend_stats__mean": "0.3"},
        {"Name": "s3", "is_baseface": "0", "class": "3", "AU01_r__trend_stats__mean": "0.7"},
    ]
    output_png = tmp_path / "dist.png"
    stats_csv = tmp_path / "dist_stats.csv"
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plot_feature_distribution(
        rows,
        feature_column="AU01_r__trend_stats__mean",
        groups=(
            DistributionGroup(label="baseline", where={"is_baseface": ("1",)}),
            DistributionGroup(label="non_base", where={"is_baseface": ("0",)}),
        ),
        output_png=output_png,
        stats_csv=stats_csv,
    )

    stats_rows = list(csv.DictReader(stats_csv.open("r", encoding="utf-8-sig")))
    assert output_png.exists()
    assert output_png.stat().st_size > 0
    assert stats_rows[0]["group"] == "baseline"
    assert stats_rows[0]["count"] == "1"
    assert stats_rows[1]["count"] == "2"
    axis = plt.gcf().axes[0]
    visible_counts = _visible_scatter_counts_by_group(axis, group_count=2)
    stats_counts = [int(row["count"]) for row in stats_rows]
    assert visible_counts == stats_counts
    assert _minimum_scatter_zorder(plt.gcf().axes[0]) > _maximum_box_patch_zorder(
        plt.gcf().axes[0]
    )
    original_close(plt.gcf())


def _series(series_kind: str, values: np.ndarray) -> OpenFaceSeries:
    return OpenFaceSeries(
        sample_id="sample-001",
        timestamps_s=np.arange(len(values), dtype=float),
        success=np.ones(len(values), dtype=float),
        confidence=np.ones(len(values), dtype=float),
        signals={"AU01_r": values},
        frame_source_rows=np.arange(2, len(values) + 2, dtype=np.int64),
        series_kind=series_kind,
        provenance={},
    )


def _visible_scatter_counts_by_group(axis: plt.Axes, *, group_count: int) -> list[int]:
    counts = [0 for _ in range(group_count)]
    for collection in axis.collections:
        if not isinstance(collection, PathCollection):
            continue
        for x_value, _ in collection.get_offsets():
            group_index = int(round(float(x_value))) - 1
            if 0 <= group_index < group_count:
                counts[group_index] += 1
    return counts


def _minimum_scatter_zorder(axis: plt.Axes) -> float:
    return min(
        collection.get_zorder()
        for collection in axis.collections
        if isinstance(collection, PathCollection)
    )


def _maximum_box_patch_zorder(axis: plt.Axes) -> float:
    return max(patch.get_zorder() for patch in axis.patches)
