"""FeatureRecordを研究者向けwide列へ変換する処理。"""

from __future__ import annotations

import re

from fatigue_analysis.domain.errors import OutputConflictError
from fatigue_analysis.domain.models import FeatureRecord, ReportTable

SAFE_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def feature_column_name(record: FeatureRecord) -> str:
    """特徴量recordからwide形式CSV列名を生成する。"""

    column_name = (
        f"{record.signal_id}__{record.feature_instance}__{record.feature_id}"
    )
    if SAFE_COLUMN_PATTERN.fullmatch(column_name) is None:
        raise OutputConflictError(f"特徴量列名に不正文字があります: {column_name}")
    return column_name


def feature_records_to_wide_rows(
    report: ReportTable,
    records: tuple[FeatureRecord, ...],
) -> tuple[dict[str, object], ...]:
    """report列を先頭に保ったwide形式行へ変換する。"""

    values_by_sample: dict[str, dict[str, float | None]] = {
        sample.sample_id: {} for sample in report.samples
    }
    seen_columns: set[tuple[str, str]] = set()
    for record in records:
        if record.sample_id not in values_by_sample:
            continue
        column_name = feature_column_name(record)
        key = (record.sample_id, column_name)
        if key in seen_columns:
            raise OutputConflictError(
                f"同一サンプルで特徴量列が重複しています: {record.sample_id}, {column_name}"
            )
        seen_columns.add(key)
        values_by_sample[record.sample_id][column_name] = record.value

    rows: list[dict[str, object]] = []
    for sample in report.samples:
        row: dict[str, object] = {
            column: sample.raw_columns.get(column, "") for column in report.columns
        }
        row.update(values_by_sample[sample.sample_id])
        rows.append(row)
    return tuple(rows)
