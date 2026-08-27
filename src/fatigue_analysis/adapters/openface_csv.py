"""OpenFace CSVの読込adapter。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from fatigue_analysis.domain.errors import InputContractError
from fatigue_analysis.domain.models import OpenFaceSeries

CONTROL_COLUMNS = ("timestamp", "success", "confidence")


def read_openface_csv(
    openface_csv_path: Path,
    *,
    sample_id: str,
    signal_ids: Iterable[str],
) -> OpenFaceSeries:
    """OpenFace CSVを列名正規化済み時系列として読み込む。"""

    if not openface_csv_path.exists():
        raise InputContractError(f"OpenFace CSVが見つかりません: {openface_csv_path}")

    with openface_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise InputContractError("OpenFace CSVが空です。") from exc
        normalized_header = tuple(column.strip() for column in raw_header)
        _validate_header(normalized_header, tuple(signal_ids))

        rows = list(reader)

    column_indices = {column: index for index, column in enumerate(normalized_header)}
    source_rows = np.array(range(2, len(rows) + 2), dtype=np.int64)
    timestamps = _parse_float_column(rows, column_indices["timestamp"], "timestamp")
    if not np.all(np.isfinite(timestamps)):
        raise InputContractError("timestamp は有限値である必要があります。")

    selected_signal_ids = tuple(signal_ids)
    return OpenFaceSeries(
        sample_id=sample_id,
        timestamps_s=timestamps,
        success=_parse_float_column(rows, column_indices["success"], "success"),
        confidence=_parse_float_column(
            rows, column_indices["confidence"], "confidence"
        ),
        signals={
            signal_id: _parse_float_column(rows, column_indices[signal_id], signal_id)
            for signal_id in selected_signal_ids
        },
        frame_source_rows=source_rows,
        series_kind="raw_loaded",
        provenance={
            "adapter": "openface_csv",
            "path": openface_csv_path.as_posix(),
            "signals": selected_signal_ids,
        },
    )


def read_openface_columns(openface_csv_path: Path) -> tuple[str, ...]:
    """OpenFace CSVヘッダーを正規化済み列名として読む。"""

    if not openface_csv_path.exists():
        raise InputContractError(f"OpenFace CSVが見つかりません: {openface_csv_path}")
    with openface_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            return normalize_openface_columns(next(reader))
        except StopIteration as exc:
            raise InputContractError("OpenFace CSVが空です。") from exc


def normalize_openface_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """OpenFace CSV列名の前後空白を除去する。"""

    return tuple(column.strip() for column in columns)


def _validate_header(columns: tuple[str, ...], signal_ids: tuple[str, ...]) -> None:
    duplicated = sorted(
        column for column in set(columns) if columns.count(column) > 1
    )
    if duplicated:
        raise InputContractError(
            "OpenFace CSV列名の正規化後に重複があります: " + ", ".join(duplicated)
        )

    required_columns = (*CONTROL_COLUMNS, *signal_ids)
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise InputContractError(
            "OpenFace CSVの必須列が不足しています: " + ", ".join(missing_columns)
        )


def _parse_float_column(
    rows: list[list[str]],
    column_index: int,
    column_name: str,
) -> np.ndarray:
    values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        raw_value = row[column_index].strip() if column_index < len(row) else ""
        if raw_value == "":
            values.append(math.nan)
            continue
        try:
            values.append(float(raw_value))
        except ValueError as exc:
            raise InputContractError(
                f"{row_number}行目 {column_name} は数値が必要です。"
            ) from exc
    return np.array(values, dtype=float)
