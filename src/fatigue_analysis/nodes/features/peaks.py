"""raw系列のピーク特徴量。"""

from __future__ import annotations

from scipy.signal import find_peaks

from fatigue_analysis.domain.errors import NodeExecutionError
from fatigue_analysis.domain.models import FeatureRecord, OpenFaceSeries


def compute_raw_peaks(
    raw_series: OpenFaceSeries,
    *,
    feature_instance: str,
    height: float = 0.1,
    prominence: float = 0.1,
    minimum_distance_seconds: float = 0.1667,
    sampling_rate_hz: float = 30.0,
) -> tuple[FeatureRecord, ...]:
    """品質処理後raw系列からピーク数とピーク頻度を算出する。"""

    if sampling_rate_hz <= 0:
        raise NodeExecutionError("sampling_rate_hz は0より大きい値が必要です。")
    if minimum_distance_seconds < 0:
        raise NodeExecutionError("minimum_distance_seconds は0以上が必要です。")
    duration_s = float(raw_series.timestamps_s[-1] - raw_series.timestamps_s[0])
    if duration_s <= 0:
        raise NodeExecutionError("peak_rate_hz の計算には正の時間長が必要です。")

    distance_frames = max(1, round(minimum_distance_seconds * sampling_rate_hz))
    records: list[FeatureRecord] = []
    for signal_id, values in raw_series.signals.items():
        peaks, _ = find_peaks(
            values,
            height=height,
            prominence=prominence,
            distance=distance_frames,
        )
        peak_count = int(len(peaks))
        records.append(
            FeatureRecord(
                sample_id=raw_series.sample_id,
                signal_id=signal_id,
                source_series=raw_series.series_kind,
                feature_id="count",
                feature_instance=feature_instance,
                value=float(peak_count),
                unit="count",
                status="ok",
            )
        )
        records.append(
            FeatureRecord(
                sample_id=raw_series.sample_id,
                signal_id=signal_id,
                source_series=raw_series.series_kind,
                feature_id="rate_hz",
                feature_instance=feature_instance,
                value=peak_count / duration_s,
                unit="hz",
                status="ok",
            )
        )
    return tuple(records)
