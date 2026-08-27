"""run manifestの組み立て。"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fatigue_analysis.config.models import AnalysisConfig
from fatigue_analysis.domain.models import FeatureRecord, SampleStatus


def build_manifest(
    *,
    run_id: str,
    status: str,
    config: AnalysisConfig,
    started_at: datetime,
    finished_at: datetime,
    sample_statuses: tuple[SampleStatus, ...],
    feature_records: tuple[FeatureRecord, ...],
    artifacts: Mapping[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    """runの再現情報をJSON化できるdictとして組み立てる。"""

    return {
        "run_id": run_id,
        "status": status,
        "started_at": _isoformat(started_at),
        "finished_at": _isoformat(finished_at),
        "python": sys.version,
        "git": _git_state(repo_root),
        "dependencies": _dependency_versions(
            ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "PyYAML")
        ),
        "resolved_config": config.to_plain_dict(),
        "samples": [status_item.__dict__ for status_item in sample_statuses],
        "feature_columns": sorted(
            {
                f"{record.signal_id}__{record.feature_instance}__{record.feature_id}"
                for record in feature_records
            }
        ),
        "artifacts": dict(artifacts),
    }


def current_utc() -> datetime:
    """timezone付き現在UTC時刻を返す。"""

    return datetime.now(tz=timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _dependency_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    dirty = _run_git(repo_root, "status", "--short")
    return {
        "commit": commit,
        "dirty": bool(dirty),
    }


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()
