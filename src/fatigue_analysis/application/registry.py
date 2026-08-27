"""特徴量node registry。"""

from __future__ import annotations

from fatigue_analysis.config.models import FeatureConfig
from fatigue_analysis.domain.errors import ConfigError
from fatigue_analysis.domain.models import FeatureRecord, OpenFaceSeries
from fatigue_analysis.domain.nodes import FeatureNode, FeatureNodeSpec
from fatigue_analysis.nodes.features.peaks import compute_raw_peaks
from fatigue_analysis.nodes.features.statistics import compute_trend_statistics
from fatigue_analysis.nodes.features.stft_power import compute_raw_stft_mean_power


class FeatureRegistry:
    """node_idから特徴量nodeを決定的に取得するregistry。"""

    def __init__(self, nodes: tuple[FeatureNode, ...]) -> None:
        self._nodes: dict[str, FeatureNode] = {}
        for node in sorted(nodes, key=lambda item: item.spec.node_id):
            if node.spec.node_id in self._nodes:
                raise ConfigError(f"node_idが重複しています: {node.spec.node_id}")
            self._nodes[node.spec.node_id] = node

    def get(self, node_id: str) -> FeatureNode:
        """node_idに対応するnodeを返す。"""

        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise ConfigError(f"未登録nodeです: {node_id}") from exc

    def list_specs(self) -> tuple[FeatureNodeSpec, ...]:
        """登録済みnode仕様をnode_id順で返す。"""

        return tuple(node.spec for node in self._nodes.values())


def default_feature_registry() -> FeatureRegistry:
    """初期特徴量nodeを登録したregistryを返す。"""

    return FeatureRegistry(
        (
            FeatureNode(
                spec=FeatureNodeSpec(
                    node_id="trend_statistics",
                    version="1",
                    source_series="trend",
                    supported_signal_type="au_intensity",
                    metrics=("mean", "variance"),
                ),
                run=_run_trend_statistics,
            ),
            FeatureNode(
                spec=FeatureNodeSpec(
                    node_id="raw_peaks",
                    version="1",
                    source_series="raw_validated",
                    supported_signal_type="au_intensity",
                    metrics=("count", "rate_hz"),
                ),
                run=_run_raw_peaks,
            ),
            FeatureNode(
                spec=FeatureNodeSpec(
                    node_id="raw_stft_mean_power",
                    version="1",
                    source_series="raw_validated",
                    supported_signal_type="au_intensity",
                    metrics=("mean_power_by_frequency",),
                ),
                run=_run_raw_stft_mean_power,
            ),
        )
    )


def _run_trend_statistics(
    series: OpenFaceSeries, feature: FeatureConfig
) -> tuple[FeatureRecord, ...]:
    metrics = tuple(str(value) for value in feature.params.get("metrics", ("mean", "variance")))
    ddof = int(feature.params.get("ddof", 0))
    return compute_trend_statistics(
        series,
        feature_instance=feature.instance_id,
        metrics=metrics,
        ddof=ddof,
    )


def _run_raw_peaks(
    series: OpenFaceSeries, feature: FeatureConfig
) -> tuple[FeatureRecord, ...]:
    return compute_raw_peaks(
        series,
        feature_instance=feature.instance_id,
        height=float(feature.params.get("height", 0.1)),
        prominence=float(feature.params.get("prominence", 0.1)),
        minimum_distance_seconds=float(
            feature.params.get("minimum_distance_seconds", 0.1667)
        ),
        sampling_rate_hz=float(feature.params.get("sampling_rate_hz", 30.0)),
    )


def _run_raw_stft_mean_power(
    series: OpenFaceSeries, feature: FeatureConfig
) -> tuple[FeatureRecord, ...]:
    return compute_raw_stft_mean_power(
        series,
        feature_instance=feature.instance_id,
        sampling_rate_hz=float(feature.params.get("sampling_rate_hz", 30.0)),
        frequency_step_hz=float(feature.params.get("frequency_step_hz", 1.5)),
        max_frequency_hz=float(feature.params.get("max_frequency_hz", 15.0)),
        overlap_ratio=float(feature.params.get("overlap_ratio", 0.5)),
        window=str(feature.params.get("window", "hann")),
        detrend=bool(feature.params.get("detrend", False)),
        boundary=feature.params.get("boundary", "zeros"),
        padded=bool(feature.params.get("padded", True)),
        exclude_dc=bool(feature.params.get("exclude_dc", True)),
    )
