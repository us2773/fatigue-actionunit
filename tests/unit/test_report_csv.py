from __future__ import annotations

from pathlib import Path

import pytest

from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.domain.errors import InputContractError


def test_report_csv_loads_valid_rows_and_preserves_extra_columns(tmp_path: Path) -> None:
    """有効なreportを読み込み、追加列と元列順を保持する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "\ufeffName,person,check_date,class,fatigue_level,is_baseface,note\n"
        "sample-001,001,2026/8/24,2,2,0,keep\n"
        "sample-002,001,2026-08-25,,,1,base\n",
        encoding="utf-8",
    )

    report = read_report_csv(report_path, expected_baselines_per_person=1)

    assert report.columns == (
        "Name",
        "person",
        "check_date",
        "class",
        "fatigue_level",
        "is_baseface",
        "note",
    )
    assert report.samples[0].participant_id == "001"
    assert report.samples[0].class_code == 2
    assert report.samples[1].is_baseface is True
    assert report.samples[1].raw_columns["note"] == "base"


def test_duplicate_name_is_rejected(tmp_path: Path) -> None:
    """Name重複を拒否する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,2,2,0\n"
        "sample-001,1,2026/8/25,,,1\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="重複"):
        read_report_csv(report_path, expected_baselines_per_person=1)


def test_baseface_requires_blank_class_and_fatigue(tmp_path: Path) -> None:
    """基準表情行のclass/fatigue_levelは空欄だけを受け付ける。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,2,1\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="基準表情"):
        read_report_csv(report_path, expected_baselines_per_person=1)


def test_non_baseface_requires_class_and_fatigue(tmp_path: Path) -> None:
    """非基準表情行のclass/fatigue_level空欄を拒否する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,,,0\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="class"):
        read_report_csv(report_path, expected_baselines_per_person=0)


@pytest.mark.parametrize(
    ("class_code", "fatigue_level"),
    [("0", "2"), ("5", "2"), ("1", "0"), ("1", "4")],
)
def test_class_and_fatigue_ranges_are_checked(
    tmp_path: Path, class_code: str, fatigue_level: str
) -> None:
    """class 1〜4、fatigue_level 1〜3以外を拒否する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        f"sample-001,1,2026/8/24,{class_code},{fatigue_level},0\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError):
        read_report_csv(report_path, expected_baselines_per_person=0)


def test_name_with_path_separator_is_rejected(tmp_path: Path) -> None:
    """Nameにパス区切り文字を許可しない。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "nested/sample,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="パス区切り"):
        read_report_csv(report_path, expected_baselines_per_person=0)


def test_baseline_count_per_person_is_checked(tmp_path: Path) -> None:
    """参加者ごとの基準表情件数を設定値で検証する。"""

    report_path = tmp_path / "report.csv"
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,,,1\n"
        "sample-002,1,2026/8/25,1,1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="基準表情件数"):
        read_report_csv(report_path, expected_baselines_per_person=2)


def test_validate_report_paths_checks_report_samples_only(tmp_path: Path) -> None:
    """reportに載った動画だけを入力対象として確認する。"""

    report_path = tmp_path / "report.csv"
    movie_dir = tmp_path / "movie"
    movie_dir.mkdir()
    (movie_dir / "sample-001.mp4").write_text("dummy", encoding="utf-8")
    (movie_dir / "unused.mp4").write_text("dummy", encoding="utf-8")
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)

    validate_report_paths(report, movie_dir=movie_dir, require_movie=True)


def test_validate_report_paths_rejects_missing_movie(tmp_path: Path) -> None:
    """reportに載った動画が欠けていれば拒否する。"""

    report_path = tmp_path / "report.csv"
    movie_dir = tmp_path / "movie"
    movie_dir.mkdir()
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)

    with pytest.raises(InputContractError, match="動画"):
        validate_report_paths(report, movie_dir=movie_dir, require_movie=True)


def test_validate_report_paths_accepts_openface_csv_without_movie(
    tmp_path: Path,
) -> None:
    """OpenFace CSV利用時は動画がなくても入力を受理する。"""

    report_path = tmp_path / "report.csv"
    openface_dir = tmp_path / "openface_csv"
    openface_dir.mkdir()
    (openface_dir / "sample-001.csv").write_text(
        "timestamp,success,confidence,AU01_r\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)

    validate_report_paths(
        report,
        openface_csv_dir=openface_dir,
        require_openface_csv=True,
    )


def test_validate_report_paths_rejects_missing_openface_csv(
    tmp_path: Path,
) -> None:
    """OpenFace CSV必須モードではCSV欠損を拒否する。"""

    report_path = tmp_path / "report.csv"
    openface_dir = tmp_path / "openface_csv"
    openface_dir.mkdir()
    report_path.write_text(
        "Name,person,check_date,class,fatigue_level,is_baseface\n"
        "sample-001,1,2026/8/24,1,1,0\n",
        encoding="utf-8",
    )
    report = read_report_csv(report_path, expected_baselines_per_person=0)

    with pytest.raises(InputContractError, match="OpenFace CSV"):
        validate_report_paths(
            report,
            openface_csv_dir=openface_dir,
            require_openface_csv=True,
        )
