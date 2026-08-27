"""PowerShell経由でOpenFace変換を実行するadapter。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.config.models import AnalysisConfig
from fatigue_analysis.domain.errors import ExternalToolError, InputContractError


@dataclass(frozen=True)
class OpenFaceConversionResult:
    """1動画のOpenFace変換結果。"""

    sample_id: str
    status: str
    movie_path: Path
    output_csv_path: Path
    message: str


def run_openface_conversions(
    config: AnalysisConfig,
    *,
    run_id: str,
    force: bool = False,
    powershell_executable: str = "powershell",
) -> tuple[OpenFaceConversionResult, ...]:
    """report対象動画をOpenFace CSVへ変換する。"""

    report = read_report_csv(
        config.paths.report_csv,
        expected_baselines_per_person=config.report.expected_baselines_per_person,
    )
    validate_report_paths(report, movie_dir=config.paths.movie_dir)

    script_path = config.openface.powershell_script
    local_config_path = config.openface.local_environment_config
    if not script_path.exists():
        raise InputContractError(f"OpenFace PowerShell scriptが見つかりません: {script_path}")
    if not local_config_path.exists():
        raise InputContractError(f"OpenFaceローカル設定が見つかりません: {local_config_path}")

    results: list[OpenFaceConversionResult] = []
    for sample in report.samples:
        movie_path = config.paths.movie_dir / f"{sample.sample_id}.mp4"
        output_csv_path = config.paths.openface_csv_dir / f"{sample.sample_id}.csv"
        if output_csv_path.exists() and config.openface.skip_existing and not force:
            results.append(
                OpenFaceConversionResult(
                    sample_id=sample.sample_id,
                    status="skipped",
                    movie_path=movie_path,
                    output_csv_path=output_csv_path,
                    message="CSV already exists.",
                )
            )
            continue

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            powershell_executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-InputVideo",
            str(movie_path),
            "-OutputCsv",
            str(output_csv_path),
            "-LocalEnvironmentConfig",
            str(local_config_path),
            "-RunId",
            run_id,
        ]
        if force:
            command.append("-Force")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            results.append(
                OpenFaceConversionResult(
                    sample_id=sample.sample_id,
                    status="failed",
                    movie_path=movie_path,
                    output_csv_path=output_csv_path,
                    message=completed.stderr.strip() or completed.stdout.strip(),
                )
            )
            continue
        if not output_csv_path.exists():
            results.append(
                OpenFaceConversionResult(
                    sample_id=sample.sample_id,
                    status="failed",
                    movie_path=movie_path,
                    output_csv_path=output_csv_path,
                    message="PowerShell completed but CSV was not created.",
                )
            )
            continue
        results.append(
            OpenFaceConversionResult(
                sample_id=sample.sample_id,
                status="created",
                movie_path=movie_path,
                output_csv_path=output_csv_path,
                message=completed.stdout.strip(),
            )
        )

    failed = [result for result in results if result.status == "failed"]
    if failed:
        failed_ids = ", ".join(result.sample_id for result in failed)
        raise ExternalToolError(f"OpenFace変換に失敗しました: {failed_ids}")
    return tuple(results)
