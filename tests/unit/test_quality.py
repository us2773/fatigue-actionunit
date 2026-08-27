from __future__ import annotations

import numpy as np

from fatigue_analysis.domain.models import OpenFaceSeries
from fatigue_analysis.nodes.preprocessing.quality import apply_quality_pipeline


def _series(
    *,
    timestamps: list[float],
    success: list[float],
    confidence: list[float],
    au01: list[float],
    extra_signals: dict[str, list[float]] | None = None,
) -> OpenFaceSeries:
    signals = {"AU01_r": np.array(au01, dtype=float)}
    if extra_signals:
        signals.update(
            {
                signal_id: np.array(values, dtype=float)
                for signal_id, values in extra_signals.items()
            }
        )
    return OpenFaceSeries(
        sample_id="sample",
        timestamps_s=np.array(timestamps, dtype=float),
        success=np.array(success, dtype=float),
        confidence=np.array(confidence, dtype=float),
        signals=signals,
        frame_source_rows=np.arange(2, len(timestamps) + 2, dtype=np.int64),
        series_kind="raw_loaded",
        provenance={},
    )


def test_trailing_invalid_success_frames_are_removed() -> None:
    """末尾の連続success不正フレームだけを除去する。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1, 0.2, 0.3],
            success=[1, 1, 0, 0],
            confidence=[0.9, 0.9, 0.9, 0.9],
            au01=[0.1, 0.2, 0.3, 0.4],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert result.validated_series is not None
    assert result.status.trailing_removed_count == 2
    assert np.allclose(result.validated_series.timestamps_s, [0.0, 0.1])


def test_trailing_nan_signal_frames_are_removed() -> None:
    """末尾の連続NaNフレームを除去する。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1, 0.2],
            success=[1, 1, 1],
            confidence=[0.9, 0.9, 0.9],
            au01=[0.1, 0.2, np.nan],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert result.validated_series is not None
    assert result.status.trailing_removed_count == 1
    assert np.allclose(result.validated_series.signals["AU01_r"], [0.1, 0.2])


def test_initial_trim_uses_timestamp_duration_and_keeps_boundary() -> None:
    """先頭除外はtimestamp期間で計算し、境界timestampを採用側へ含める。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            success=[1, 1, 1, 1, 1],
            confidence=[0.9, 0.9, 0.9, 0.9, 0.9],
            au01=[0.0, 0.1, 0.2, 0.3, 0.4],
        ),
        initial_trim_ratio=0.25,
        confidence_threshold=0.7,
    )

    assert result.validated_series is not None
    assert result.status.initial_trimmed_count == 1
    assert np.allclose(result.validated_series.timestamps_s, [1.0, 2.0, 3.0, 4.0])


def test_invalid_frame_inside_adopted_range_excludes_sample() -> None:
    """採用区間途中の無効フレームはサンプル全体を除外する。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1, 0.2],
            success=[1, 0, 1],
            confidence=[0.9, 0.9, 0.9],
            au01=[0.1, 0.2, 0.3],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert result.validated_series is None
    assert result.status.reason_code == "invalid_frame_in_adopted_range"


def test_confidence_threshold_boundary() -> None:
    """confidence 0.7は採用し、0.6999は除外する。"""

    accepted = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1],
            success=[1, 1],
            confidence=[0.7, 0.7],
            au01=[0.1, 0.2],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )
    rejected = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1],
            success=[1, 1],
            confidence=[0.7, 0.6999],
            au01=[0.1, 0.2],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert accepted.validated_series is not None
    assert rejected.validated_series is None
    assert rejected.status.reason_code == "confidence_below_threshold"


def test_unselected_signal_nan_does_not_exclude_sample() -> None:
    """未選択AUの欠損だけでは除外しない。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.1],
            success=[1, 1],
            confidence=[0.9, 0.9],
            au01=[0.1, 0.2],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert result.validated_series is not None


def test_non_monotonic_timestamp_excludes_sample() -> None:
    """timestampが単調増加でなければ除外する。"""

    result = apply_quality_pipeline(
        _series(
            timestamps=[0.0, 0.2, 0.1],
            success=[1, 1, 1],
            confidence=[0.9, 0.9, 0.9],
            au01=[0.1, 0.2, 0.3],
        ),
        initial_trim_ratio=0.0,
        confidence_threshold=0.7,
    )

    assert result.validated_series is None
    assert result.status.reason_code == "non_monotonic_timestamp"
