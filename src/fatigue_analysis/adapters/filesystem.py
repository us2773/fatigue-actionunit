"""成果物ファイルの安全な保存。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from fatigue_analysis.domain.errors import OutputConflictError


def create_run_directories(output_root: Path, run_id: str, *, overwrite: bool) -> Path:
    """runディレクトリを非破壊で作成する。"""

    run_dir = output_root / run_id
    if run_dir.exists() and not overwrite:
        raise OutputConflictError(f"既存runを上書きしません: {run_dir}")
    for child in (
        run_dir / "features",
        run_dir / "validation",
        run_dir / "figures" / "timeseries",
        run_dir / "figures" / "distributions",
        run_dir / "plot_data",
    ):
        child.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: tuple[str, ...],
    encoding: str,
) -> None:
    """dict行を指定列順でCSV保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """JSON成果物をUTF-8で保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """ファイル内容のSHA-256を返す。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
