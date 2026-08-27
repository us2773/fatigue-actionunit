"""既存成果物またはOpenFace CSVから可視化成果物を生成するservice。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence

from fatigue_analysis.adapters.openface_csv import read_openface_columns, read_openface_csv
from fatigue_analysis.config.models import AnalysisConfig
from fatigue_analysis.domain.errors import ConfigError
from fatigue_analysis.domain.signals import au_number_to_signal_id
from fatigue_analysis.nodes.preprocessing.lowess import compute_lowess_series
from fatigue_analysis.nodes.preprocessing.quality import apply_quality_pipeline
from fatigue_analysis.visualization.distributions import (
    DistributionGroup,
    plot_feature_distribution,
)
from fatigue_analysis.visualization.timeseries import plot_timeseries


@dataclass(frozen=True)
class VisualizationArtifact:
    """生成した可視化成果物のmanifest用情報。"""

    key: str
    path: Path


def plot_distributions_from_feature_csv(
    config: AnalysisConfig,
    *,
    feature_csv: Path,
    output_root: Path,
) -> tuple[VisualizationArtifact, ...]:
    """既存特徴量CSVを読み、設定された特徴量分布図を生成する。"""

    if not config.visualizations.distributions:
        return ()
    if not feature_csv.exists():
        raise ConfigError(f"特徴量CSVが見つかりません: {feature_csv}")

    with feature_csv.open("r", encoding=config.outputs.csv_encoding, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ConfigError(f"特徴量CSVにヘッダーがありません: {feature_csv}")
        rows = tuple(dict(row) for row in reader)
        feature_columns = tuple(
            column for column in reader.fieldnames if "__" in column
        )

    csv_rows = _string_rows(rows)
    artifacts: list[VisualizationArtifact] = []
    available_columns = _available_columns(csv_rows, feature_columns)
    for config_index, raw_distribution in enumerate(
        config.visualizations.distributions,
        start=1,
    ):
        distribution = _parse_distribution_config(
            raw_distribution,
            config_index=config_index,
            available_columns=available_columns,
        )
        matched_features = _match_feature_columns(
            distribution.feature_patterns,
            feature_columns,
            config_index=config_index,
        )
        _validate_exclusive_groups(
            csv_rows,
            distribution.groups,
            config_index=config_index,
        )
        for feature_column in matched_features:
            artifact_stem = f"dist_{config_index:02d}_{feature_column}"
            output_png = output_root / "figures" / "distributions" / f"{artifact_stem}.png"
            stats_csv = output_root / "plot_data" / "distributions" / f"{artifact_stem}_stats.csv"
            plot_feature_distribution(
                csv_rows,
                feature_column=feature_column,
                groups=distribution.groups,
                output_png=output_png,
                stats_csv=stats_csv,
                color_by=distribution.color_by,
            )
            artifacts.extend(
                (
                    VisualizationArtifact(
                        key=f"distribution_{config_index:02d}_{feature_column}_png",
                        path=output_png,
                    ),
                    VisualizationArtifact(
                        key=f"distribution_{config_index:02d}_{feature_column}_stats_csv",
                        path=stats_csv,
                    ),
                )
            )
    return tuple(artifacts)


def plot_timeseries_from_openface_csv(
    config: AnalysisConfig,
    *,
    sample_id: str,
    au_numbers: tuple[int, ...],
    series_kinds: tuple[str, ...],
    output_root: Path,
    plot_id: str,
) -> tuple[VisualizationArtifact, ...]:
    """1サンプルのOpenFace CSVから時系列図を生成する。"""

    if not sample_id:
        raise ConfigError("--sample-id は空にできません。")
    if not au_numbers:
        raise ConfigError("--au は1件以上必要です。")
    allowed_series = {"raw_validated", "trend", "residual"}
    unknown_series = sorted(set(series_kinds) - allowed_series)
    if unknown_series:
        raise ConfigError("未対応のseriesです: " + ", ".join(unknown_series))

    openface_csv = config.paths.openface_csv_dir / f"{sample_id}.csv"
    available_columns = read_openface_columns(openface_csv)
    signal_ids = tuple(au_number_to_signal_id(au) for au in au_numbers)
    missing_signals = [
        signal_id for signal_id in signal_ids if signal_id not in available_columns
    ]
    if missing_signals:
        raise ConfigError("OpenFace CSVに対象AU列がありません: " + ", ".join(missing_signals))

    raw_loaded = read_openface_csv(
        openface_csv,
        sample_id=sample_id,
        signal_ids=signal_ids,
    )
    quality_result = apply_quality_pipeline(
        raw_loaded,
        initial_trim_ratio=config.preprocessing.initial_trim_ratio,
        confidence_threshold=config.preprocessing.confidence_threshold,
    )
    if quality_result.validated_series is None:
        raise ConfigError(
            f"時系列可視化に使える採用区間がありません: {quality_result.status.message}"
        )

    raw_validated = quality_result.validated_series
    series_by_kind = {"raw_validated": raw_validated}
    if "trend" in series_kinds or "residual" in series_kinds:
        if not config.preprocessing.lowess.enabled:
            raise ConfigError("trend/residualの時系列可視化にはLOWESS前処理が必要です。")
        trend, residual = compute_lowess_series(
            raw_validated,
            frac=config.preprocessing.lowess.frac,
            it=config.preprocessing.lowess.it,
            delta=config.preprocessing.lowess.delta,
        )
        series_by_kind["trend"] = trend
        series_by_kind["residual"] = residual

    selected_series = {
        series_kind: series_by_kind[series_kind] for series_kind in series_kinds
    }
    signal_part = "_".join(signal_ids)
    artifact_stem = f"timeseries_{sample_id}_{signal_part}"
    output_png = output_root / "figures" / "timeseries" / f"{artifact_stem}.png"
    plot_data_csv = output_root / "plot_data" / "timeseries" / f"{artifact_stem}.csv"
    plot_timeseries(
        selected_series,
        signal_ids=signal_ids,
        output_png=output_png,
        plot_data_csv=plot_data_csv,
        run_id=plot_id,
    )
    return (
        VisualizationArtifact(key="timeseries_png", path=output_png),
        VisualizationArtifact(key="timeseries_plot_data_csv", path=plot_data_csv),
    )


@dataclass(frozen=True)
class _DistributionConfig:
    feature_patterns: tuple[str, ...]
    groups: tuple[DistributionGroup, ...]
    color_by: str | None


def _parse_distribution_config(
    raw_distribution: Mapping[str, Any],
    *,
    config_index: int,
    available_columns: set[str],
) -> _DistributionConfig:
    _ensure_mapping(raw_distribution, f"visualizations.distributions[{config_index}]")
    allowed_keys = {"feature_patterns", "groups", "color_by", "facet_by"}
    unknown_keys = sorted(set(raw_distribution) - allowed_keys)
    if unknown_keys:
        raise ConfigError(
            f"visualizations.distributions[{config_index}] に未知キーがあります: "
            + ", ".join(unknown_keys)
        )

    facet_by = raw_distribution.get("facet_by")
    if facet_by is not None:
        raise ConfigError("visualizations.distributions[].facet_by は未対応です。")

    feature_patterns = _string_tuple(
        raw_distribution.get("feature_patterns"),
        f"visualizations.distributions[{config_index}].feature_patterns",
    )
    raw_groups = raw_distribution.get("groups")
    _ensure_sequence(raw_groups, f"visualizations.distributions[{config_index}].groups")
    groups = tuple(
        _parse_group(raw_group, config_index=config_index, group_index=group_index)
        for group_index, raw_group in enumerate(raw_groups, start=1)
    )
    if len(groups) < 2:
        raise ConfigError("visualizations.distributions[].groups は2件以上が必要です。")

    color_by = raw_distribution.get("color_by")
    if color_by is not None:
        if not isinstance(color_by, str) or color_by == "":
            raise ConfigError("visualizations.distributions[].color_by は文字列が必要です。")
        if color_by not in available_columns:
            raise ConfigError(f"color_by列がfeatures_wide.csvにありません: {color_by}")

    for group in groups:
        for column in group.where:
            if column not in available_columns:
                raise ConfigError(f"group条件列がfeatures_wide.csvにありません: {column}")

    return _DistributionConfig(
        feature_patterns=feature_patterns,
        groups=groups,
        color_by=color_by,
    )


def _parse_group(
    raw_group: Mapping[str, Any],
    *,
    config_index: int,
    group_index: int,
) -> DistributionGroup:
    section = f"visualizations.distributions[{config_index}].groups[{group_index}]"
    _ensure_mapping(raw_group, section)
    unknown_keys = sorted(set(raw_group) - {"label", "where"})
    if unknown_keys:
        raise ConfigError(f"{section} に未知キーがあります: " + ", ".join(unknown_keys))
    label = raw_group.get("label")
    if not isinstance(label, str) or label == "":
        raise ConfigError(f"{section}.label は空でない文字列が必要です。")
    where = raw_group.get("where")
    _ensure_mapping(where, f"{section}.where")
    return DistributionGroup(
        label=label,
        where={
            str(column): _string_tuple(values, f"{section}.where.{column}")
            for column, values in where.items()
        },
    )


def _match_feature_columns(
    patterns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    *,
    config_index: int,
) -> tuple[str, ...]:
    matched: list[str] = []
    for pattern in patterns:
        for feature_column in feature_columns:
            if fnmatchcase(feature_column, pattern) and feature_column not in matched:
                matched.append(feature_column)
    if not matched:
        raise ConfigError(
            f"visualizations.distributions[{config_index}].feature_patterns "
            "に一致する特徴量列がありません。"
        )
    return tuple(matched)


def _validate_exclusive_groups(
    rows: Sequence[Mapping[str, str]],
    groups: tuple[DistributionGroup, ...],
    *,
    config_index: int,
) -> None:
    for row in rows:
        matched_group_labels = [
            group.label for group in groups if _matches_where(row, group.where)
        ]
        if len(matched_group_labels) > 1:
            sample_id = row.get("Name", "")
            raise ConfigError(
                f"visualizations.distributions[{config_index}] のgroup条件が重複しています: "
                f"Name={sample_id}, groups={', '.join(matched_group_labels)}"
            )


def _matches_where(row: Mapping[str, str], where: Mapping[str, tuple[str, ...]]) -> bool:
    return all(str(row.get(column, "")) in values for column, values in where.items())


def _string_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            str(column): "" if value is None else str(value)
            for column, value in row.items()
        }
        for row in rows
    )


def _available_columns(
    rows: Sequence[Mapping[str, str]],
    feature_columns: tuple[str, ...],
) -> set[str]:
    columns: set[str] = set(feature_columns)
    for row in rows:
        columns.update(row)
    return columns


def _string_tuple(value: Any, section: str) -> tuple[str, ...]:
    _ensure_sequence(value, section)
    result = tuple(str(item) for item in value)
    if not result:
        raise ConfigError(f"{section} は空にできません。")
    return result


def _ensure_mapping(value: Any, section: str) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} はmappingである必要があります。")


def _ensure_sequence(value: Any, section: str) -> None:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ConfigError(f"{section} はlistである必要があります。")
