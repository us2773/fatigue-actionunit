"""trend系列の基本統計特徴量。"""

from __future__ import annotations

import numpy as np

from fatigue_analysis.domain.errors import NodeExecutionError
from fatigue_analysis.domain.models import FeatureRecord, OpenFaceSeries


def compute_trend_statistics(
    trend_series: OpenFaceSeries,
    *,
    feature_instance: str,
    metrics: tuple[str, ...] = ("mean", "variance"),
    ddof: int = 0,
) -> tuple[FeatureRecord, ...]:
    """LOWESS trend系列から平均と分散を算出する。"""

    if ddof < 0:
        raise NodeExecutionError("ddof は0以上が必要です。")
    unknown_metrics = sorted(set(metrics) - {"mean", "variance"})
    if unknown_metrics:
        raise NodeExecutionError("未知の統計量です: " + ", ".join(unknown_metrics))

    records: list[FeatureRecord] = []
    for signal_id, values in trend_series.signals.items():
        if "mean" in metrics:
            records.append(
                FeatureRecord(
                    sample_id=trend_series.sample_id,
                    signal_id=signal_id,
                    source_series=trend_series.series_kind,
                    feature_id="mean",
                    feature_instance=feature_instance,
                    value=float(np.mean(values)),
                    unit="intensity",
                    status="ok",
                )
            )
        if "variance" in metrics:
            if len(values) <= ddof:
                records.append(
                    FeatureRecord(
                        sample_id=trend_series.sample_id,
                        signal_id=signal_id,
                        source_series=trend_series.series_kind,
                        feature_id="variance",
                        feature_instance=feature_instance,
                        value=None,
                        unit="intensity_squared",
                        status="missing",
                        reason_code="insufficient_length",
                    )
                )
            else:
                records.append(
                    FeatureRecord(
                        sample_id=trend_series.sample_id,
                        signal_id=signal_id,
                        source_series=trend_series.series_kind,
                        feature_id="variance",
                        feature_instance=feature_instance,
                        value=float(np.var(values, ddof=ddof)),
                        unit="intensity_squared",
                        status="ok",
                    )
                )
    return tuple(records)
