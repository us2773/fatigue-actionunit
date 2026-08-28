"""実行前に入力と生成予定を要約するplan生成。"""

from __future__ import annotations

from dataclasses import dataclass

from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.config.models import AnalysisConfig


@dataclass(frozen=True)
class ExecutionPlan:
    """CLIで表示する実行計画の要約。"""

    sample_count: int
    baseline_count: int
    movie_count: int
    openface_csv_existing_count: int
    openface_csv_missing_count: int
    openface_csv_missing_sample_ids: tuple[str, ...]
    feature_instances: tuple[str, ...]
    timeseries_count: int
    distribution_count: int
    output_root: str

    @property
    def feature_count(self) -> int:
        """実行予定の特徴量設定数を返す。"""

        return len(self.feature_instances)


def build_execution_plan(config: AnalysisConfig) -> ExecutionPlan:
    """設定と入力契約から非破壊の実行計画を作る。"""

    report = read_report_csv(
        config.paths.report_csv,
        expected_baselines_per_person=config.report.expected_baselines_per_person,
    )
    validate_report_paths(
        report,
        openface_csv_dir=config.paths.openface_csv_dir,
        require_openface_csv=False,
    )

    movie_count = sum(
        1
        for sample in report.samples
        if (config.paths.movie_dir / f"{sample.sample_id}.mp4").exists()
    )
    missing_openface = tuple(
        sample.sample_id
        for sample in report.samples
        if not (config.paths.openface_csv_dir / f"{sample.sample_id}.csv").exists()
    )
    existing_openface_count = len(report.samples) - len(missing_openface)
    feature_instances = tuple(feature.instance_id for feature in config.features)

    return ExecutionPlan(
        sample_count=len(report.samples),
        baseline_count=sum(1 for sample in report.samples if sample.is_baseface),
        movie_count=movie_count,
        openface_csv_existing_count=existing_openface_count,
        openface_csv_missing_count=len(missing_openface),
        openface_csv_missing_sample_ids=missing_openface,
        feature_instances=feature_instances,
        timeseries_count=len(config.visualizations.timeseries),
        distribution_count=len(config.visualizations.distributions),
        output_root=config.paths.output_root.as_posix(),
    )
