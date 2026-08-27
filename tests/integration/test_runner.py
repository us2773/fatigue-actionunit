from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fatigue_analysis.application.runner import run_features
from fatigue_analysis.config.loader import load_config


def test_run_features_writes_wide_csv_and_manifest(tmp_path: Path) -> None:
    """合成reportとOpenFace CSVからwide CSVとmanifestを生成する。"""

    config_path = _write_synthetic_project(tmp_path)
    config = load_config(config_path)

    result = run_features(config, run_id="it-run-001", repo_root=Path.cwd())

    assert result.status == "succeeded"
    assert result.sample_count == 2
    assert result.excluded_count == 0
    assert result.feature_csv.exists()
    assert result.manifest_json.exists()
    distribution_png = (
        result.run_dir
        / "figures"
        / "distributions"
        / "dist_01_AU01_r__trend_stats__mean.png"
    )
    distribution_stats = (
        result.run_dir
        / "plot_data"
        / "distributions"
        / "dist_01_AU01_r__trend_stats__mean_stats.csv"
    )
    assert distribution_png.exists()
    assert distribution_stats.exists()

    rows = list(csv.DictReader(result.feature_csv.open("r", encoding="utf-8-sig")))
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["Name"] == "sample-001"
    assert rows[0]["class_label"] == "だるい"
    assert rows[0]["fatigue_level_label"] == "元気"
    assert rows[1]["class_label"] == ""
    assert rows[1]["fatigue_level_label"] == ""
    assert "AU01_r__trend_stats__mean" in rows[0]
    assert "AU01_r__raw_peaks__count" in rows[0]
    assert "AU01_r__raw_stft__mean_power_3_hz" in rows[0]
    assert (
        manifest["artifacts"]["distribution_01_AU01_r__trend_stats__mean_png"]
        == "figures/distributions/dist_01_AU01_r__trend_stats__mean.png"
    )


def test_run_features_reproducible_feature_csv(tmp_path: Path) -> None:
    """同一入力・設定の2runで特徴量CSV内容が一致する。"""

    config_path = _write_synthetic_project(tmp_path)
    config = load_config(config_path)

    first = run_features(config, run_id="it-run-a", repo_root=Path.cwd())
    second = run_features(config, run_id="it-run-b", repo_root=Path.cwd())

    assert first.feature_csv.read_bytes() == second.feature_csv.read_bytes()


def _write_synthetic_project(tmp_path: Path) -> Path:
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
    _write_openface_csv(openface_dir / "sample-001.csv", phase=0.0)
    _write_openface_csv(openface_dir / "sample-002.csv", phase=0.2)

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
schema_version: 1
paths:
  report_csv: {(_posix(report_dir / "report.csv"))}
  movie_dir: {_posix(movie_dir)}
  openface_csv_dir: {_posix(openface_dir)}
  output_root: {_posix(output_root)}
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
  - instance_id: raw_stft
    node_id: raw_stft_mean_power
    signals:
      au_numbers: [1]
    params:
      sampling_rate_hz: 30
      frequency_step_hz: 1.5
      max_frequency_hz: 15
      overlap_ratio: 0.5
      window: hann
      detrend: false
      boundary: zeros
      padded: true
      exclude_dc: true
analysis:
  filters: []
  derived_columns: {{}}
visualizations:
  timeseries: []
  distributions:
    - feature_patterns: ["AU01_r__trend_stats__mean"]
      groups:
        - label: baseline
          where: {{is_baseface: [1]}}
        - label: non_base
          where: {{is_baseface: [0]}}
      color_by: person
      facet_by: null
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


def _posix(path: Path) -> str:
    return path.as_posix()
