"""LOWESSによるtrend/residual生成。"""

from __future__ import annotations

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from fatigue_analysis.domain.models import OpenFaceSeries


def compute_lowess_series(
    series: OpenFaceSeries,
    *,
    frac: float = 0.1,
    it: int = 3,
    delta: float = 0.0,
) -> tuple[OpenFaceSeries, OpenFaceSeries]:
    """AU強度raw系列からLOWESS trendとresidualを生成する。"""

    trend_signals: dict[str, np.ndarray] = {}
    residual_signals: dict[str, np.ndarray] = {}
    for signal_id, values in series.signals.items():
        trend_values = lowess(
            endog=values,
            exog=series.timestamps_s,
            frac=frac,
            it=it,
            delta=delta,
            is_sorted=True,
            missing="raise",
            return_sorted=False,
        )
        trend_signals[signal_id] = np.asarray(trend_values, dtype=float)
        residual_signals[signal_id] = values - trend_signals[signal_id]

    common_provenance = {
        "node": "lowess",
        "input_series": series.series_kind,
        "frac": frac,
        "it": it,
        "delta": delta,
    }
    trend_series = OpenFaceSeries(
        sample_id=series.sample_id,
        timestamps_s=series.timestamps_s,
        success=series.success,
        confidence=series.confidence,
        signals=trend_signals,
        frame_source_rows=series.frame_source_rows,
        series_kind="trend",
        provenance=common_provenance,
    )
    residual_series = OpenFaceSeries(
        sample_id=series.sample_id,
        timestamps_s=series.timestamps_s,
        success=series.success,
        confidence=series.confidence,
        signals=residual_signals,
        frame_source_rows=series.frame_source_rows,
        series_kind="residual",
        provenance=common_provenance,
    )
    return trend_series, residual_series
