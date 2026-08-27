from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fatigue_analysis.adapters import openface_runner
from fatigue_analysis.adapters.openface_runner import run_openface_conversions
from fatigue_analysis.config.loader import load_config
from fatigue_analysis.domain.errors import ExternalToolError


def test_openface_runner_skips_existing_csv(tmp_path: Path) -> None:
    """既存CSVがあれば既定で再生成しない。"""

    config = load_config(_write_project(tmp_path))
    output_csv = tmp_path / "data" / "01_raw" / "openface_csv" / "sample-001.csv"
    output_csv.write_text("timestamp,success,confidence,AU01_r\n", encoding="utf-8")

    results = run_openface_conversions(config, run_id="openface-test")

    assert results[0].status == "skipped"


def test_openface_runner_invokes_powershell_and_checks_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PowerShellを呼び出し、生成CSVの存在を確認する。"""

    config = load_config(_write_project(tmp_path))

    def fake_run(
        args: tuple[str, ...],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors
        output_index = args.index("-OutputCsv") + 1
        Path(args[output_index]).write_text(
            "timestamp,success,confidence,AU01_r\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="created")

    monkeypatch.setattr(openface_runner.subprocess, "run", fake_run)

    results = run_openface_conversions(config, run_id="openface-test")

    assert results[0].status == "created"


def test_openface_runner_raises_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """外部コマンド失敗を握りつぶさず例外にする。"""

    config = load_config(_write_project(tmp_path))

    def fake_run(
        args: tuple[str, ...],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors
        return subprocess.CompletedProcess(args=args, returncode=9, stderr="failed")

    monkeypatch.setattr(openface_runner.subprocess, "run", fake_run)

    with pytest.raises(ExternalToolError, match="sample-001"):
        run_openface_conversions(config, run_id="openface-test")


def _write_project(tmp_path: Path) -> Path:
    movie_dir = tmp_path / "data" / "01_raw" / "movie"
    openface_dir = tmp_path / "data" / "01_raw" / "openface_csv"
    report_dir = tmp_path / "data" / "02_report"
    output_root = tmp_path / "data" / "outputs"
    script_path = tmp_path / "scripts" / "run_openface.ps1"
    local_config_path = tmp_path / "conf" / "openface.local.ps1"
    for directory in (movie_dir, openface_dir, report_dir, output_root, script_path.parent, local_config_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    (movie_dir / "sample-001.mp4").write_text("dummy", encoding="utf-8")
    (report_dir / "report.csv").write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    script_path.write_text("param()\n", encoding="utf-8")
    local_config_path.write_text('$ContainerName = "dummy"\n', encoding="utf-8")

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
  powershell_script: {script_path.as_posix()}
  local_environment_config: {local_config_path.as_posix()}
  skip_existing: true
report:
  expected_baselines_per_person: 0
  extra_column_types: {{}}
signals:
  au_intensity:
    au_numbers: [1]
preprocessing:
  initial_trim_ratio: 0.0
  confidence_threshold: 0.70
  lowess:
    enabled: true
    frac: 0.5
    it: 0
    delta: 0.0
features: []
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
