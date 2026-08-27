"""研究データを表す不変ドメインモデル。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping


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
