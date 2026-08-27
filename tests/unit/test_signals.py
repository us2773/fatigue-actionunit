from __future__ import annotations

import pytest

from fatigue_analysis.domain.errors import ConfigError, InputContractError
from fatigue_analysis.domain.signals import (
    OPENFACE_AU_INTENSITY_SIGNAL_IDS,
    au_number_to_signal_id,
    resolve_au_intensity_signal_ids,
)


def test_au_number_to_signal_id_zero_pads_openface_name() -> None:
    """AU番号を内部SignalIdへ変換する。"""

    assert au_number_to_signal_id(1) == "AU01_r"
    assert au_number_to_signal_id(45) == "AU45_r"


def test_unknown_au_number_is_rejected() -> None:
    """OpenFace AU強度にないAU番号を拒否する。"""

    with pytest.raises(ConfigError, match="未知"):
        au_number_to_signal_id(3)


def test_resolve_all_returns_known_order() -> None:
    """all指定はOpenFace AU強度の決定的順序を返す。"""

    assert resolve_au_intensity_signal_ids("all") == OPENFACE_AU_INTENSITY_SIGNAL_IDS


def test_resolve_selected_au_numbers() -> None:
    """指定AU番号だけをSignalIdにする。"""

    signal_ids = resolve_au_intensity_signal_ids((1, 4, 45))

    assert signal_ids == ("AU01_r", "AU04_r", "AU45_r")


def test_resolve_all_uses_available_columns() -> None:
    """OpenFace CSV列が分かっている場合、allは存在するAU強度列だけにする。"""

    signal_ids = resolve_au_intensity_signal_ids(
        "all",
        available_columns={"timestamp", "AU04_r", "AU01_r"},
    )

    assert signal_ids == ("AU01_r", "AU04_r")


def test_missing_selected_signal_column_is_rejected() -> None:
    """選択AUがCSVに存在しなければ入力契約エラーにする。"""

    with pytest.raises(InputContractError, match="AU45_r"):
        resolve_au_intensity_signal_ids(
            (1, 45),
            available_columns={"timestamp", "AU01_r"},
        )
