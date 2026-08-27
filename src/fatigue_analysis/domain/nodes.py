"""node登録のための最小契約。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fatigue_analysis.config.models import FeatureConfig
from fatigue_analysis.domain.models import FeatureRecord, OpenFaceSeries


@dataclass(frozen=True)
class FeatureNodeSpec:
    """特徴量nodeの公開仕様。"""

    node_id: str
    version: str
    source_series: str
    supported_signal_type: str
    metrics: tuple[str, ...]


FeatureNodeRunner = Callable[
    [OpenFaceSeries, FeatureConfig],
    tuple[FeatureRecord, ...],
]


@dataclass(frozen=True)
class FeatureNode:
    """登録済み特徴量node。"""

    spec: FeatureNodeSpec
    run: FeatureNodeRunner
