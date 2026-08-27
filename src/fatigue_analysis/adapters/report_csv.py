"""`report.csv` の読込と契約検証。"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

from fatigue_analysis.domain.errors import InputContractError
from fatigue_analysis.domain.models import ReportTable, SampleMetadata

REQUIRED_REPORT_COLUMNS: tuple[str, ...] = (
    "Name",
    "person",
    "check_date",
    "class",
    "fatigue_level",
    "is_baseface",
)


def read_report_csv(
    report_csv_path: Path,
    *,
    expected_baselines_per_person: int = 2,
) -> ReportTable:
    """研究者が作成した正本 `report.csv` を検証して読み込む。"""

    if not report_csv_path.exists():
        raise InputContractError(f"report.csv が見つかりません: {report_csv_path}")

    with report_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise InputContractError("report.csv にヘッダーがありません。")
        columns = tuple(reader.fieldnames)
        _validate_required_columns(columns)
        samples = tuple(
            _parse_report_row(row, row_number=index + 2)
            for index, row in enumerate(reader)
        )

    _validate_sample_ids(samples)
    _validate_baseline_counts(
        samples,
        expected_baselines_per_person=expected_baselines_per_person,
    )
    return ReportTable(columns=columns, samples=samples)


def validate_report_paths(
    report: ReportTable,
    *,
    movie_dir: Path,
    openface_csv_dir: Path | None = None,
    require_openface_csv: bool = False,
) -> None:
    """`Name` から動画とOpenFace CSVの存在を確認する。"""

    missing_movies = [
        sample.sample_id
        for sample in report.samples
        if not (movie_dir / f"{sample.sample_id}.mp4").exists()
    ]
    if missing_movies:
        raise InputContractError(
            "report.csv に対応する動画がありません: " + ", ".join(missing_movies)
        )

    if require_openface_csv:
        if openface_csv_dir is None:
            raise InputContractError("OpenFace CSVディレクトリが指定されていません。")
        missing_csvs = [
            sample.sample_id
            for sample in report.samples
            if not (openface_csv_dir / f"{sample.sample_id}.csv").exists()
        ]
        if missing_csvs:
            raise InputContractError(
                "report.csv に対応するOpenFace CSVがありません: "
                + ", ".join(missing_csvs)
            )


def _validate_required_columns(columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in REQUIRED_REPORT_COLUMNS if column not in columns]
    if missing_columns:
        raise InputContractError(
            "report.csv の必須列が不足しています: " + ", ".join(missing_columns)
        )


def _parse_report_row(row: Mapping[str, str], *, row_number: int) -> SampleMetadata:
    sample_id = _required_text(row, "Name", row_number=row_number)
    if "/" in sample_id or "\\" in sample_id:
        raise InputContractError(f"{row_number}行目 Name にパス区切り文字があります。")

    participant_id = _required_text(row, "person", row_number=row_number)
    check_date = _parse_date(
        _required_text(row, "check_date", row_number=row_number),
        row_number=row_number,
    )
    class_code = _parse_optional_int(row.get("class", ""), "class", row_number=row_number)
    fatigue_level = _parse_optional_int(
        row.get("fatigue_level", ""),
        "fatigue_level",
        row_number=row_number,
    )
    is_baseface = _parse_is_baseface(
        _required_text(row, "is_baseface", row_number=row_number),
        row_number=row_number,
    )

    if is_baseface:
        if class_code is not None or fatigue_level is not None:
            raise InputContractError(
                f"{row_number}行目は基準表情のためclass/fatigue_levelは空欄が必要です。"
            )
    else:
        if class_code not in {1, 2, 3, 4}:
            raise InputContractError(f"{row_number}行目 class は1〜4が必要です。")
        if fatigue_level not in {1, 2, 3}:
            raise InputContractError(f"{row_number}行目 fatigue_level は1〜3が必要です。")

    return SampleMetadata.create(
        sample_id=sample_id,
        participant_id=participant_id,
        check_date=check_date,
        class_code=class_code,
        fatigue_level=fatigue_level,
        is_baseface=is_baseface,
        raw_columns={key: value if value is not None else "" for key, value in row.items()},
    )


def _validate_sample_ids(samples: Iterable[SampleMetadata]) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicate_ids:
        raise InputContractError("Name が重複しています: " + ", ".join(duplicate_ids))


def _validate_baseline_counts(
    samples: Iterable[SampleMetadata],
    *,
    expected_baselines_per_person: int,
) -> None:
    if expected_baselines_per_person < 0:
        raise InputContractError("expected_baselines_per_person は0以上が必要です。")
    if expected_baselines_per_person == 0:
        return

    counts: Counter[str] = Counter()
    participants: set[str] = set()
    for sample in samples:
        participants.add(sample.participant_id)
        if sample.is_baseface:
            counts[sample.participant_id] += 1

    invalid_counts = {
        participant: counts[participant]
        for participant in sorted(participants)
        if counts[participant] != expected_baselines_per_person
    }
    if invalid_counts:
        details = ", ".join(
            f"{participant}:{count}" for participant, count in invalid_counts.items()
        )
        raise InputContractError(
            "参加者ごとの基準表情件数が期待値と一致しません。"
            f" expected={expected_baselines_per_person}, actual={details}"
        )


def _required_text(row: Mapping[str, str], column: str, *, row_number: int) -> str:
    value = row.get(column, "")
    if value is None or value.strip() == "":
        raise InputContractError(f"{row_number}行目 {column} は空欄にできません。")
    return value.strip()


def _parse_optional_int(value: str | None, column: str, *, row_number: int) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise InputContractError(f"{row_number}行目 {column} は整数が必要です。") from exc


def _parse_is_baseface(value: str, *, row_number: int) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise InputContractError(f"{row_number}行目 is_baseface は0または1が必要です。")


def _parse_date(value: str, *, row_number: int) -> date:
    for date_format in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    raise InputContractError(
        f"{row_number}行目 check_date は YYYY/M/D、YYYY/MM/DD、YYYY-MM-DD が必要です。"
    )
