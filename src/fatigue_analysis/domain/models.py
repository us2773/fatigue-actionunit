"""研究データを表す不変ドメインモデル。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleMetadata:
    """`report.csv` の1行を表す不変メタデータ。"""

    sample_id: str
    participant_id: str
    check_date: date
    class_code: int | None
    fatigue_level: int | None
    is_baseface: bool
    raw_columns: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        sample_id: str,
        participant_id: str,
        check_date: date,
        class_code: int | None,
        fatigue_level: int | None,
        is_baseface: bool,
        raw_columns: Mapping[str, str],
    ) -> "SampleMetadata":
        """元列値を外部から変更できない形で保持して生成する。"""

        return cls(
            sample_id=sample_id,
            participant_id=participant_id,
            check_date=check_date,
            class_code=class_code,
            fatigue_level=fatigue_level,
            is_baseface=is_baseface,
            raw_columns=MappingProxyType(dict(raw_columns)),
        )


@dataclass(frozen=True)
class ReportTable:
    """検証済み `report.csv` 全体。"""

    columns: tuple[str, ...]
    samples: tuple[SampleMetadata, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """report上のサンプルIDを元順序で返す。"""

        return tuple(sample.sample_id for sample in self.samples)


@dataclass(frozen=True)
class OpenFaceSeries:
    """OpenFace CSVから得た1サンプル分の時系列。"""

    sample_id: str
    timestamps_s: NDArray[np.float64]
    success: NDArray[np.float64]
    confidence: NDArray[np.float64]
    signals: Mapping[str, NDArray[np.float64]]
    frame_source_rows: NDArray[np.int64]
    series_kind: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """配列長を検証し、外部変更されにくい形へコピーする。"""

        length = len(self.timestamps_s)
        arrays = {
            "success": self.success,
            "confidence": self.confidence,
            "frame_source_rows": self.frame_source_rows,
        }
        for name, array in arrays.items():
            if len(array) != length:
                raise ValueError(f"{name} の長さがtimestampと一致しません。")
        for signal_id, values in self.signals.items():
            if len(values) != length:
                raise ValueError(f"{signal_id} の長さがtimestampと一致しません。")

        object.__setattr__(self, "timestamps_s", np.array(self.timestamps_s, dtype=float))
        object.__setattr__(self, "success", np.array(self.success, dtype=float))
        object.__setattr__(self, "confidence", np.array(self.confidence, dtype=float))
        object.__setattr__(
            self,
            "signals",
            MappingProxyType(
                {
                    signal_id: np.array(values, dtype=float)
                    for signal_id, values in self.signals.items()
                }
            ),
        )
        object.__setattr__(
            self, "frame_source_rows", np.array(self.frame_source_rows, dtype=np.int64)
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def select_rows(
        self,
        row_indices: NDArray[np.int64],
        *,
        series_kind: str,
        provenance: Mapping[str, Any],
    ) -> "OpenFaceSeries":
        """指定行だけを残した新しい時系列を返す。"""

        return OpenFaceSeries(
            sample_id=self.sample_id,
            timestamps_s=self.timestamps_s[row_indices],
            success=self.success[row_indices],
            confidence=self.confidence[row_indices],
            signals={
                signal_id: values[row_indices]
                for signal_id, values in self.signals.items()
            },
            frame_source_rows=self.frame_source_rows[row_indices],
            series_kind=series_kind,
            provenance=provenance,
        )


@dataclass(frozen=True)
class SampleStatus:
    """品質処理後のサンプル状態。"""

    sample_id: str
    original_frame_count: int
    trailing_removed_count: int
    initial_trimmed_count: int
    adopted_frame_count: int
    adopted_start_s: float | None
    adopted_end_s: float | None
    status: str
    reason_code: str | None
    message: str


@dataclass(frozen=True)
class ExclusionRecord:
    """除外理由CSVへ出力する1件分の理由。"""

    sample_id: str
    stage: str
    reason_code: str
    frame_source_rows: tuple[int, ...]
    message: str


@dataclass(frozen=True)
class QualityResult:
    """品質処理の結果。"""

    validated_series: OpenFaceSeries | None
    status: SampleStatus
    exclusions: tuple[ExclusionRecord, ...]
