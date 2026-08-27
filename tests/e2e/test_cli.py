from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_cli_validate_and_features(tmp_path: Path) -> None:
    """CLIでvalidateからfeatures生成までを最小E2Eとして確認する。"""

    config_path = _write_synthetic_project(tmp_path)

    validate = subprocess.run(
        [sys.executable, "-m", "fatigue_analysis", "validate", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert validate.returncode == 0
    assert "Inputs are valid" in validate.stdout

    plan = subprocess.run(
        [sys.executable, "-m", "fatigue_analysis", "plan", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert plan.returncode == 0
    assert "Plan summary" in plan.stdout
    assert "openface_csv=2 existing, 0 missing" in plan.stdout

    nodes = subprocess.run(
        [sys.executable, "-m", "fatigue_analysis", "list", "nodes"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert nodes.returncode == 0
    assert "trend_statistics: version=1" in nodes.stdout
    assert "source_series=trend" in nodes.stdout

    features = subprocess.run(
        [
            sys.executable,
            "-m",
            "fatigue_analysis",
            "features",
            "--config",
            str(config_path),
            "--run-id",
            "e2e-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert features.returncode == 0
    assert "Run complete" in features.stdout

    feature_csv = tmp_path / "data" / "outputs" / "e2e-run" / "features" / "features_wide.csv"
    manifest_json = tmp_path / "data" / "outputs" / "e2e-run" / "manifest.json"
    rows = list(csv.DictReader(feature_csv.open("r", encoding="utf-8-sig")))

    assert manifest_json.exists()
    assert len(rows) == 2
    assert rows[0]["class_label"] == "だるい"
    assert rows[0]["fatigue_level_label"] == "元気"
    assert "AU01_r__trend_stats__mean" in rows[0]


def _write_synthetic_project(tmp_path: Path) -> Path:
    movie_dir = tmp_path / "data" / "01_raw" / "movie"
    openface_dir = tmp_path / "data" / "01_raw" / "openface_csv"
    report_dir = tmp_path / "data" / "02_report"
    output_root = tmp_path / "data" / "outputs"
    movie_dir.mkdir(parents=True)
    openface_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    output_root.mkdir(parents=True)

    for sample_id in ("sample-001", "sample-002"):
        (movie_dir / f"{sample_id}.mp4").write_text("dummy", encoding="utf-8")
    (report_dir / "report.csv").write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n"
        "sample-002,1,2026/8/24,,,1\n",
        encoding="utf-8",
    )
    _write_openface_csv(openface_dir / "sample-001.csv", phase=0.0)
    _write_openface_csv(openface_dir / "sample-002.csv", phase=0.2)

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
  - instance_id: raw_peaks
    node_id: raw_peaks
    signals:
      au_numbers: [1]
    params:
      height: 0.5
      prominence: 0.2
      minimum_distance_seconds: 0.1
      sampling_rate_hz: 30
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


def _write_openface_csv(path: Path, *, phase: float) -> None:
    timestamps = np.arange(60, dtype=float) / 30.0
    values = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * timestamps + phase)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "success", "confidence", "AU01_r"])
        for timestamp, value in zip(timestamps, values, strict=True):
            writer.writerow([f"{timestamp:.10f}", "1", "0.9", f"{value:.10f}"])
