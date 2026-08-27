from __future__ import annotations

import csv
from pathlib import Path

from fatigue_analysis.application.planner import build_execution_plan
from fatigue_analysis.config.loader import load_config


def test_build_execution_plan_summarizes_inputs_without_openface_requirement(
    tmp_path: Path,
) -> None:
    """OpenFace CSV不足を実行前planとして要約する。"""

    config_path = _write_project(tmp_path)
    config = load_config(config_path)

    plan = build_execution_plan(config)

    assert plan.sample_count == 2
    assert plan.baseline_count == 1
    assert plan.movie_count == 2
    assert plan.openface_csv_existing_count == 1
    assert plan.openface_csv_missing_count == 1
    assert plan.openface_csv_missing_sample_ids == ("sample-002",)
    assert plan.feature_instances == ("trend_stats",)


def _write_project(tmp_path: Path) -> Path:
    movie_dir = tmp_path / "data" / "01_raw" / "movie"
    openface_dir = tmp_path / "data" / "01_raw" / "openface_csv"
    report_dir = tmp_path / "data" / "02_report"
    output_root = tmp_path / "data" / "outputs"
    movie_dir.mkdir(parents=True)
    openface_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    output_root.mkdir(parents=True)
    (movie_dir / "sample-001.mp4").write_text("dummy", encoding="utf-8")
    (movie_dir / "sample-002.mp4").write_text("dummy", encoding="utf-8")
    (report_dir / "report.csv").write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n"
        "sample-002,1,2026/8/24,,,1\n",
        encoding="utf-8",
    )
    with (openface_dir / "sample-001.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "success", "confidence", "AU01_r"])
        writer.writerow(["0.0", "1", "0.9", "0.1"])

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
schema_version: 1
paths:
  report_csv: {(report_dir / "report.csv").as_posix()}
  movie_dir: {movie_dir.as_posix()}
  openface_csv_dir: {openface_dir.as_posix()}
  output_root: {output_root.as_posix()}
openface:
  powershell_script: scripts/run_openface.ps1
  local_environment_config: conf/openface.local.ps1
  skip_existing: true
report:
  expected_baselines_per_person: 1
  extra_column_types: {{}}
signals:
  au_intensity:
    au_numbers: all
preprocessing:
  initial_trim_ratio: 0.0
  confidence_threshold: 0.70
  lowess:
    enabled: true
    frac: 0.5
    it: 0
    delta: 0.0
features:
  - instance_id: trend_stats
    node_id: trend_statistics
    signals:
      au_numbers: [1]
    params:
      metrics: [mean, variance]
      ddof: 0
analysis:
  filters: []
  derived_columns: {{}}
visualizations:
  timeseries: []
  distributions: []
outputs:
  save_intermediate_nodes: []
  csv_encoding: utf-8-sig
  float_precision: 10
  overwrite: false
""".lstrip(),
        encoding="utf-8",
    )
    return config_path
