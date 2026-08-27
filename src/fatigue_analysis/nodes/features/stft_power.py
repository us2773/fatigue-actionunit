"""raw系列のSTFT平均パワー特徴量。"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import stft

from fatigue_analysis.domain.errors import NodeExecutionError
from fatigue_analysis.domain.models import FeatureRecord, OpenFaceSeries


def compute_raw_stft_mean_power(
    raw_series: OpenFaceSeries,
    *,
    feature_instance: str,
    sampling_rate_hz: float = 30.0,
    frequency_step_hz: float = 1.5,
    max_frequency_hz: float = 15.0,
    overlap_ratio: float = 0.5,
    window: str = "hann",
    detrend: bool = False,
    boundary: str | None = "zeros",
    padded: bool = True,
    exclude_dc: bool = True,
) -> tuple[FeatureRecord, ...]:
    """品質処理後raw系列から周波数binごとの平均パワーを算出する。"""

    nperseg = _resolve_nperseg(sampling_rate_hz, frequency_step_hz)
    noverlap = round(nperseg * overlap_ratio)
    if not 0 <= noverlap < nperseg:
        raise NodeExecutionError("overlap_ratio から不正なnoverlapが得られました。")

    records: list[FeatureRecord] = []
    for signal_id, values in raw_series.signals.items():
        frequencies, _, coefficients = stft(
            values,
            fs=sampling_rate_hz,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=("constant" if detrend else False),
            boundary=boundary,
            padded=padded,
        )
        mean_power = np.mean(np.abs(coefficients) ** 2, axis=1)
        for frequency_hz, power_value in zip(frequencies, mean_power, strict=True):
            if exclude_dc and math.isclose(float(frequency_hz), 0.0, abs_tol=1e-12):
                continue
            if frequency_hz <= max_frequency_hz + 1e-12:
                records.append(
                    FeatureRecord(
                        sample_id=raw_series.sample_id,
                        signal_id=signal_id,
                        source_series=raw_series.series_kind,
                        feature_id=f"mean_power_{_format_frequency(frequency_hz)}_hz",
                        feature_instance=feature_instance,
                        value=float(power_value),
                        unit="power",
                        status="ok",
                    )
                )
    return tuple(records)


def _resolve_nperseg(sampling_rate_hz: float, frequency_step_hz: float) -> int:
    if sampling_rate_hz <= 0:
        raise NodeExecutionError("sampling_rate_hz は0より大きい値が必要です。")
    if frequency_step_hz <= 0:
        raise NodeExecutionError("frequency_step_hz は0より大きい値が必要です。")
    raw_nperseg = sampling_rate_hz / frequency_step_hz
    rounded = round(raw_nperseg)
    if not math.isclose(raw_nperseg, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise NodeExecutionError(
            "sampling_rate_hz / frequency_step_hz は整数になる必要があります。"
        )
    return int(rounded)


def _format_frequency(frequency_hz: float) -> str:
    formatted = f"{frequency_hz:.10g}"
    return formatted.replace(".", "p")
