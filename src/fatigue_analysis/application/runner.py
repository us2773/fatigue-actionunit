"""最小特徴量パイプラインrunner。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fatigue_analysis.adapters.filesystem import (
    create_run_directories,
    sha256_file,
    write_csv_rows,
    write_json,
)
from fatigue_analysis.adapters.openface_csv import read_openface_columns, read_openface_csv
from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.application.registry import FeatureRegistry, default_feature_registry
from fatigue_analysis.application.run_manifest import build_manifest, current_utc
from fatigue_analysis.config.models import AnalysisConfig, FeatureConfig
from fatigue_analysis.domain.errors import ConfigError
from fatigue_analysis.domain.models import (
    ExclusionRecord,
    FeatureRecord,
    OpenFaceSeries,
    SampleStatus,
)
from fatigue_analysis.domain.signals import resolve_au_intensity_signal_ids
from fatigue_analysis.nodes.outputs.feature_table import (
    feature_column_name,
    feature_output_columns,
    feature_records_to_wide_rows,
)
from fatigue_analysis.nodes.preprocessing.lowess import compute_lowess_series
from fatigue_analysis.nodes.preprocessing.quality import apply_quality_pipeline


@dataclass(frozen=True)
class RunResult:
    """runnerの実行結果。"""

    run_id: str
    status: str
    run_dir: Path
    feature_csv: Path
    manifest_json: Path
    sample_count: int
    excluded_count: int


def run_features(
    config: AnalysisConfig,
    *,
    run_id: str,
    repo_root: Path,
    registry: FeatureRegistry | None = None,
) -> RunResult:
    """設定に従い、OpenFace CSVから特徴量wide CSVとmanifestを生成する。"""

    active_registry = registry or default_feature_registry()
    started_at = current_utc()
    run_dir = create_run_directories(
        config.paths.output_root,
        run_id,
        overwrite=config.outputs.overwrite,
    )

    report = read_report_csv(
        config.paths.report_csv,
        expected_baselines_per_person=config.report.expected_baselines_per_person,
    )
    validate_report_paths(
        report,
        openface_csv_dir=config.paths.openface_csv_dir,
        require_openface_csv=True,
    )

    feature_records: list[FeatureRecord] = []
    sample_statuses: list[SampleStatus] = []
    exclusions: list[ExclusionRecord] = []
    for sample in report.samples:
        openface_csv_path = config.paths.openface_csv_dir / f"{sample.sample_id}.csv"
        available_columns = read_openface_columns(openface_csv_path)
        required_signal_ids = _resolve_required_signal_ids(config, available_columns)
        raw_loaded = read_openface_csv(
            openface_csv_path,
            sample_id=sample.sample_id,
            signal_ids=required_signal_ids,
        )
        quality_result = apply_quality_pipeline(
            raw_loaded,
            initial_trim_ratio=config.preprocessing.initial_trim_ratio,
            confidence_threshold=config.preprocessing.confidence_threshold,
        )
        sample_statuses.append(quality_result.status)
        exclusions.extend(quality_result.exclusions)
        if quality_result.validated_series is None:
            continue

        raw_validated = quality_result.validated_series
        trend_series: OpenFaceSeries | None = None
        if _requires_trend(config.features, active_registry):
            if not config.preprocessing.lowess.enabled:
                raise ConfigError("trend_statisticsにはLOWESS前処理が必要です。")
            trend_series, _ = compute_lowess_series(
                raw_validated,
                frac=config.preprocessing.lowess.frac,
                it=config.preprocessing.lowess.it,
                delta=config.preprocessing.lowess.delta,
            )

        for feature in config.features:
            node = active_registry.get(feature.node_id)
            source_series = (
                trend_series
                if node.spec.source_series == "trend"
                else raw_validated
            )
            if source_series is None:
                raise ConfigError(f"{feature.node_id} の入力系列を生成できません。")
            feature_signal_ids = resolve_au_intensity_signal_ids(
                feature.signals.au_numbers,
                available_columns=source_series.signals.keys(),
            )
            feature_records.extend(
                node.run(_with_signals(source_series, feature_signal_ids), feature)
            )

    if not any(status.status == "ok" for status in sample_statuses):
        raise ConfigError("正常処理できたサンプルが0件です。")

    wide_rows = feature_records_to_wide_rows(report, tuple(feature_records))
    feature_columns = _feature_columns_in_order(tuple(feature_records))
    feature_csv = run_dir / "features" / "features_wide.csv"
    write_csv_rows(
        feature_csv,
        wide_rows,
        fieldnames=feature_output_columns(report.columns, feature_columns),
        encoding=config.outputs.csv_encoding,
    )

    sample_status_csv = run_dir / "validation" / "sample_status.csv"
    write_csv_rows(
        sample_status_csv,
        (status.__dict__ for status in sample_statuses),
        fieldnames=(
            "sample_id",
            "original_frame_count",
            "trailing_removed_count",
            "initial_trimmed_count",
            "adopted_frame_count",
            "adopted_start_s",
            "adopted_end_s",
            "status",
            "reason_code",
            "message",
        ),
        encoding=config.outputs.csv_encoding,
    )

    exclusions_csv = run_dir / "validation" / "exclusions.csv"
    write_csv_rows(
        exclusions_csv,
        (exclusion.__dict__ for exclusion in exclusions),
        fieldnames=("sample_id", "stage", "reason_code", "frame_source_rows", "message"),
        encoding=config.outputs.csv_encoding,
    )

    status = "succeeded_with_warnings" if exclusions else "succeeded"
    artifacts = {
        "features_wide_csv": _artifact_value(run_dir, feature_csv),
        "sample_status_csv": _artifact_value(run_dir, sample_status_csv),
        "exclusions_csv": _artifact_value(run_dir, exclusions_csv),
        "features_wide_sha256": sha256_file(feature_csv),
    }
    manifest = build_manifest(
        run_id=run_id,
        status=status,
        config=config,
        started_at=started_at,
        finished_at=current_utc(),
        sample_statuses=tuple(sample_statuses),
        feature_records=tuple(feature_records),
        artifacts=artifacts,
        repo_root=repo_root,
    )
    manifest_json = run_dir / "manifest.json"
    write_json(manifest_json, manifest)

    return RunResult(
        run_id=run_id,
        status=status,
        run_dir=run_dir,
        feature_csv=feature_csv,
        manifest_json=manifest_json,
        sample_count=len(report.samples),
        excluded_count=len(exclusions),
    )


def _resolve_required_signal_ids(
    config: AnalysisConfig, available_columns: tuple[str, ...]
) -> tuple[str, ...]:
    ordered: list[str] = []
    selections = [config.signals.au_intensity.au_numbers]
    selections.extend(feature.signals.au_numbers for feature in config.features)
    for au_numbers in selections:
        for signal_id in resolve_au_intensity_signal_ids(
            au_numbers,
            available_columns=available_columns,
        ):
            if signal_id not in ordered:
                ordered.append(signal_id)
    return tuple(ordered)


def _requires_trend(
    features: tuple[FeatureConfig, ...],
    registry: FeatureRegistry,
) -> bool:
    return any(registry.get(feature.node_id).spec.source_series == "trend" for feature in features)


def _with_signals(series: OpenFaceSeries, signal_ids: tuple[str, ...]) -> OpenFaceSeries:
    return OpenFaceSeries(
        sample_id=series.sample_id,
        timestamps_s=series.timestamps_s,
        success=series.success,
        confidence=series.confidence,
        signals={signal_id: series.signals[signal_id] for signal_id in signal_ids},
        frame_source_rows=series.frame_source_rows,
        series_kind=series.series_kind,
        provenance=series.provenance,
    )


def _feature_columns_in_order(records: tuple[FeatureRecord, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    for record in records:
        column = feature_column_name(record)
        if column not in columns:
            columns.append(column)
    return tuple(columns)


def _artifact_value(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.as_posix()
