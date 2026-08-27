"""OpenFace時系列の品質処理。"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from fatigue_analysis.domain.models import (
    ExclusionRecord,
    OpenFaceSeries,
    QualityResult,
    SampleStatus,
)

STATUS_OK: Final[str] = "ok"
STATUS_EXCLUDED: Final[str] = "excluded"


def apply_quality_pipeline(
    series: OpenFaceSeries,
    *,
    initial_trim_ratio: float,
    confidence_threshold: float,
) -> QualityResult:
    """末尾無効除去、先頭trim、採用区間gateを順に適用する。"""

    original_frame_count = len(series.timestamps_s)
    if original_frame_count == 0:
        return _excluded_result(
            series,
            trailing_removed_count=0,
            initial_trimmed_count=0,
            adopted_indices=np.array([], dtype=np.int64),
            stage="quality",
            reason_code="empty_series",
            message="時系列が空です。",
        )

    invalid_mask = _invalid_frame_mask(series)
    trailing_removed_count = _count_trailing_true(invalid_mask)
    end_index = original_frame_count - trailing_removed_count
    if end_index <= 0:
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=0,
            adopted_indices=np.array([], dtype=np.int64),
            stage="quality",
            reason_code="all_frames_invalid",
            message="全フレームが無効です。",
        )

    candidate_indices = np.arange(end_index, dtype=np.int64)
    candidate_timestamps = series.timestamps_s[candidate_indices]
    if np.any(np.diff(candidate_timestamps) <= 0):
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=0,
            adopted_indices=candidate_indices,
            stage="quality",
            reason_code="non_monotonic_timestamp",
            message="timestampが単調増加ではありません。",
        )

    start_timestamp = float(candidate_timestamps[0])
    end_timestamp = float(candidate_timestamps[-1])
    duration_s = end_timestamp - start_timestamp
    if duration_s <= 0:
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=0,
            adopted_indices=candidate_indices,
            stage="quality",
            reason_code="non_positive_duration",
            message="採用候補区間の時間長が0以下です。",
        )

    cutoff_timestamp = start_timestamp + duration_s * initial_trim_ratio
    adopted_mask = candidate_timestamps >= cutoff_timestamp
    adopted_indices = candidate_indices[adopted_mask]
    initial_trimmed_count = int(len(candidate_indices) - len(adopted_indices))
    if len(adopted_indices) == 0:
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=initial_trimmed_count,
            adopted_indices=adopted_indices,
            stage="trim",
            reason_code="empty_after_initial_trim",
            message="先頭区間除外後にフレームが残りません。",
        )

    adopted_invalid_indices = adopted_indices[invalid_mask[adopted_indices]]
    if len(adopted_invalid_indices) > 0:
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=initial_trimmed_count,
            adopted_indices=adopted_indices,
            stage="quality",
            reason_code="invalid_frame_in_adopted_range",
            message="採用区間に無効または欠損フレームがあります。",
            offending_indices=adopted_invalid_indices,
        )

    low_confidence_indices = adopted_indices[
        series.confidence[adopted_indices] < confidence_threshold
    ]
    if len(low_confidence_indices) > 0:
        return _excluded_result(
            series,
            trailing_removed_count=trailing_removed_count,
            initial_trimmed_count=initial_trimmed_count,
            adopted_indices=adopted_indices,
            stage="quality",
            reason_code="confidence_below_threshold",
            message="採用区間にconfidenceしきい値未満のフレームがあります。",
            offending_indices=low_confidence_indices,
        )

    validated = series.select_rows(
        adopted_indices,
        series_kind="raw_validated",
        provenance={
            "node": "quality_pipeline",
            "initial_trim_ratio": initial_trim_ratio,
            "confidence_threshold": confidence_threshold,
            "trailing_removed_count": trailing_removed_count,
            "initial_trimmed_count": initial_trimmed_count,
        },
    )
    status = SampleStatus(
        sample_id=series.sample_id,
        original_frame_count=original_frame_count,
        trailing_removed_count=trailing_removed_count,
        initial_trimmed_count=initial_trimmed_count,
        adopted_frame_count=len(adopted_indices),
        adopted_start_s=float(validated.timestamps_s[0]),
        adopted_end_s=float(validated.timestamps_s[-1]),
        status=STATUS_OK,
        reason_code=None,
        message="採用しました。",
    )
    return QualityResult(validated_series=validated, status=status, exclusions=())


def _invalid_frame_mask(series: OpenFaceSeries) -> NDArray[np.bool_]:
    signal_invalid = np.zeros(len(series.timestamps_s), dtype=bool)
    for values in series.signals.values():
        signal_invalid |= ~np.isfinite(values)
    return (
        (series.success != 1)
        | ~np.isfinite(series.success)
        | ~np.isfinite(series.confidence)
        | ~np.isfinite(series.timestamps_s)
        | signal_invalid
    )


def _count_trailing_true(mask: NDArray[np.bool_]) -> int:
    count = 0
    for value in mask[::-1]:
        if not bool(value):
            break
        count += 1
    return count


def _excluded_result(
    series: OpenFaceSeries,
    *,
    trailing_removed_count: int,
    initial_trimmed_count: int,
    adopted_indices: NDArray[np.int64],
    stage: str,
    reason_code: str,
    message: str,
    offending_indices: NDArray[np.int64] | None = None,
) -> QualityResult:
    offending = offending_indices if offending_indices is not None else adopted_indices
    frame_source_rows = tuple(int(row) for row in series.frame_source_rows[offending])
    status = SampleStatus(
        sample_id=series.sample_id,
        original_frame_count=len(series.timestamps_s),
        trailing_removed_count=trailing_removed_count,
        initial_trimmed_count=initial_trimmed_count,
        adopted_frame_count=len(adopted_indices),
        adopted_start_s=(
            float(series.timestamps_s[adopted_indices[0]])
            if len(adopted_indices) > 0
            else None
        ),
        adopted_end_s=(
            float(series.timestamps_s[adopted_indices[-1]])
            if len(adopted_indices) > 0
            else None
        ),
        status=STATUS_EXCLUDED,
        reason_code=reason_code,
        message=message,
    )
    exclusion = ExclusionRecord(
        sample_id=series.sample_id,
        stage=stage,
        reason_code=reason_code,
        frame_source_rows=frame_source_rows,
        message=message,
    )
    return QualityResult(
        validated_series=None,
        status=status,
        exclusions=(exclusion,),
    )
