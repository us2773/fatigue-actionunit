from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fatigue_analysis.adapters.report_csv import read_report_csv
from fatigue_analysis.domain.errors import NodeExecutionError, OutputConflictError
from fatigue_analysis.domain.models import OpenFaceSeries
from fatigue_analysis.nodes.features.peaks import compute_raw_peaks
from fatigue_analysis.nodes.features.statistics import compute_trend_statistics
from fatigue_analysis.nodes.features.stft_power import compute_raw_stft_mean_power
from fatigue_analysis.nodes.outputs.feature_table import (
    feature_column_name,
    feature_records_to_wide_rows,
)
from fatigue_analysis.nodes.preprocessing.lowess import compute_lowess_series


def _series(values: np.ndarray, *, timestamps: np.ndarray | None = None) -> OpenFaceSeries:
    if timestamps is None:
        timestamps = np.arange(len(values), dtype=float) / 30.0
    return OpenFaceSeries(
        sample_id="sample-001",
        timestamps_s=timestamps.astype(float),
        success=np.ones(len(values), dtype=float),
        confidence=np.ones(len(values), dtype=float),
        signals={"AU01_r": values.astype(float)},
        frame_source_rows=np.arange(2, len(values) + 2, dtype=np.int64),
        series_kind="raw_validated",
        provenance={},
    )


def test_lowess_constant_series_returns_constant_trend() -> None:
    """既知の一定系列でLOWESS trendとresidualを検証する。"""

    raw_series = _series(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))

    trend, residual = compute_lowess_series(raw_series, frac=1.0, it=0, delta=0.0)

    assert trend.series_kind == "trend"
    assert residual.series_kind == "residual"
    assert np.allclose(trend.signals["AU01_r"], [2, 2, 2, 2, 2])
    assert np.allclose(residual.signals["AU01_r"], [0, 0, 0, 0, 0])
    assert np.allclose(
        raw_series.signals["AU01_r"],
        trend.signals["AU01_r"] + residual.signals["AU01_r"],
    )


def test_trend_statistics_mean_and_population_variance() -> None:
    """trend平均とddof=0の母分散を算術定義で検証する。"""

    trend_series = _series(np.array([1.0, 2.0, 3.0]))
    trend_series = OpenFaceSeries(
        sample_id=trend_series.sample_id,
        timestamps_s=trend_series.timestamps_s,
        success=trend_series.success,
        confidence=trend_series.confidence,
        signals=trend_series.signals,
        frame_source_rows=trend_series.frame_source_rows,
        series_kind="trend",
        provenance={},
    )

    records = compute_trend_statistics(
        trend_series,
        feature_instance="trend_stats",
        metrics=("mean", "variance"),
        ddof=0,
    )

    values = {record.feature_id: record.value for record in records}
    assert values["mean"] == 2.0
    assert values["variance"] == pytest.approx(2.0 / 3.0)


def test_raw_peaks_count_and_rate_use_raw_duration() -> None:
    """raw系列からピーク数と有効時間で割ったrateを算出する。"""

    raw_series = _series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        timestamps=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    )

    records = compute_raw_peaks(
        raw_series,
        feature_instance="raw_peaks",
        height=0.5,
        prominence=0.5,
        minimum_distance_seconds=0.05,
        sampling_rate_hz=10,
    )

    values = {record.feature_id: record.value for record in records}
    assert values["count"] == 2.0
    assert values["rate_hz"] == pytest.approx(5.0)


def test_stft_default_settings_emit_ten_non_dc_bins() -> None:
    """既定設定でDCを除く1.5〜15Hzの10binを出力する。"""

    timestamps = np.arange(60, dtype=float) / 30.0
    values = np.sin(2 * math.pi * 3.0 * timestamps)
    records = compute_raw_stft_mean_power(
        _series(values, timestamps=timestamps),
        feature_instance="raw_stft",
    )

    assert len(records) == 10
    assert records[0].feature_id == "mean_power_1p5_hz"
    assert records[-1].feature_id == "mean_power_15_hz"
    max_record = max(records, key=lambda record: record.value or 0.0)
    assert max_record.feature_id == "mean_power_3_hz"


def test_stft_rejects_non_integer_frequency_step() -> None:
    """npersegが整数にならない設定を拒否する。"""

    with pytest.raises(NodeExecutionError, match="整数"):
        compute_raw_stft_mean_power(
            _series(np.ones(30)),
            feature_instance="raw_stft",
            sampling_rate_hz=30,
            frequency_step_hz=2.2,
        )


def test_feature_column_name_and_wide_rows(tmp_path: Path) -> None:
    """特徴量列名とreport列を保ったwide行を生成する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)
    records = compute_raw_peaks(
        _series(np.array([0.0, 1.0, 0.0]), timestamps=np.array([0.0, 0.5, 1.0])),
        feature_instance="raw_peaks",
        height=0.5,
        prominence=0.5,
        minimum_distance_seconds=0.1,
        sampling_rate_hz=2,
    )

    assert feature_column_name(records[0]) == "AU01_r__raw_peaks__count"
    rows = feature_records_to_wide_rows(report, records)

    assert rows[0]["Name"] == "sample-001"
    assert rows[0]["AU01_r__raw_peaks__count"] == 1.0
    assert rows[0]["AU01_r__raw_peaks__rate_hz"] == 1.0


def test_duplicate_feature_column_is_rejected(tmp_path: Path) -> None:
    """同一サンプル同一列名の重複を拒否する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)
    records = compute_raw_peaks(
        _series(np.array([0.0, 1.0, 0.0]), timestamps=np.array([0.0, 0.5, 1.0])),
        feature_instance="raw_peaks",
        height=0.5,
        prominence=0.5,
        minimum_distance_seconds=0.1,
        sampling_rate_hz=2,
    )

    with pytest.raises(OutputConflictError, match="重複"):
        feature_records_to_wide_rows(report, (records[0], records[0]))
