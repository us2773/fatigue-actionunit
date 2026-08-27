from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fatigue_analysis.adapters.openface_csv import (
    normalize_openface_columns,
    read_openface_csv,
)
from fatigue_analysis.domain.errors import InputContractError


def test_normalize_openface_columns_strips_spaces() -> None:
    """OpenFace列名の前後空白を除去する。"""

    assert normalize_openface_columns([" timestamp", " AU01_r "]) == (
        "timestamp",
        "AU01_r",
    )


def test_read_openface_csv_loads_selected_signals(tmp_path: Path) -> None:
    """選択した信号だけを数値配列として読み込む。"""

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        " timestamp, success, confidence, AU01_r, AU02_r\n"
        "0.0,1,0.9,0.1,9.9\n"
        "0.1,1,0.8,0.2,8.8\n",
        encoding="utf-8",
    )

    series = read_openface_csv(csv_path, sample_id="sample", signal_ids=("AU01_r",))

    assert series.sample_id == "sample"
    assert np.allclose(series.timestamps_s, [0.0, 0.1])
    assert tuple(series.signals) == ("AU01_r",)
    assert np.allclose(series.signals["AU01_r"], [0.1, 0.2])
    assert np.array_equal(series.frame_source_rows, [2, 3])


def test_duplicate_normalized_columns_are_rejected(tmp_path: Path) -> None:
    """列名trim後の重複を拒否する。"""

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp, timestamp ,success,confidence,AU01_r\n"
        "0.0,0.0,1,0.9,0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="重複"):
        read_openface_csv(csv_path, sample_id="sample", signal_ids=("AU01_r",))


def test_missing_required_signal_is_rejected(tmp_path: Path) -> None:
    """選択信号列の不足を拒否する。"""

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp,success,confidence,AU01_r\n"
        "0.0,1,0.9,0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="AU45_r"):
        read_openface_csv(csv_path, sample_id="sample", signal_ids=("AU45_r",))


def test_non_numeric_value_is_rejected(tmp_path: Path) -> None:
    """数値化できない値を拒否する。"""

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "timestamp,success,confidence,AU01_r\n"
        "0.0,1,0.9,not-number\n",
        encoding="utf-8",
    )

    with pytest.raises(InputContractError, match="数値"):
        read_openface_csv(csv_path, sample_id="sample", signal_ids=("AU01_r",))
