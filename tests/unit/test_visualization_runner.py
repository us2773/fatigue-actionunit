from __future__ import annotations

from pathlib import Path

import pytest

from fatigue_analysis.application.visualization_runner import (
    plot_distributions_from_feature_csv,
)
from fatigue_analysis.config.loader import load_config
from fatigue_analysis.domain.errors import ConfigError


def test_distribution_runner_rejects_overlapping_groups(tmp_path: Path) -> None:
    """同じ行が複数groupへ入る設定を拒否する。"""

    config = load_config(_write_config(tmp_path))

    with pytest.raises(ConfigError, match="重複"):
        feature_csv = tmp_path / "features_wide.csv"
        feature_csv.write_text(
            "Name,is_baseface,class,person,AU01_r__trend_stats__mean\n"
            "sample-001,0,1,1,0.4\n",
            encoding="utf-8-sig",
        )
        plot_distributions_from_feature_csv(
            config,
            feature_csv=feature_csv,
            output_root=tmp_path / "plot",
        )


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
schema_version: 1
paths:
  report_csv: {(tmp_path / "report.csv").as_posix()}
  movie_dir: {(tmp_path / "movie").as_posix()}
  openface_csv_dir: {(tmp_path / "openface").as_posix()}
  output_root: {(tmp_path / "outputs").as_posix()}
openface:
  powershell_script: scripts/run_openface.ps1
  local_environment_config: conf/openface.local.ps1
  skip_existing: true
report:
  expected_baselines_per_person: 0
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
      metrics: [mean]
      ddof: 0
analysis:
  filters: []
  derived_columns: {{}}
visualizations:
  timeseries: []
  distributions:
    - feature_patterns: ["AU01_r__trend_stats__mean"]
      groups:
        - label: first
          where: {{is_baseface: [0]}}
        - label: second
          where: {{class: [1]}}
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
