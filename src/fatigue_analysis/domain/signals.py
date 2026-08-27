"""AU番号と内部SignalIdの変換。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from fatigue_analysis.domain.errors import ConfigError, InputContractError

OPENFACE_AU_INTENSITY_NUMBERS: tuple[int, ...] = (
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    12,
    14,
    15,
    17,
    20,
    23,
    25,
    26,
    45,
)
OPENFACE_AU_INTENSITY_SIGNAL_IDS: tuple[str, ...] = tuple(
    f"AU{au_number:02d}_r" for au_number in OPENFACE_AU_INTENSITY_NUMBERS
)


def au_number_to_signal_id(au_number: int) -> str:
    """人間向けAU番号をOpenFace AU強度SignalIdへ変換する。"""

    if au_number not in OPENFACE_AU_INTENSITY_NUMBERS:
        raise ConfigError(f"未知のAU番号です: {au_number}")
    return f"AU{au_number:02d}_r"


def resolve_au_intensity_signal_ids(
    au_numbers: str | Sequence[int],
    *,
    available_columns: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """AU番号指定を内部SignalIdへ解決する。"""

    normalized_columns = set(available_columns or ())
    if au_numbers == "all":
        signal_ids = OPENFACE_AU_INTENSITY_SIGNAL_IDS
        if normalized_columns:
            signal_ids = tuple(
                signal_id for signal_id in signal_ids if signal_id in normalized_columns
            )
            if not signal_ids:
                raise InputContractError("利用可能なAU強度列が見つかりません。")
        return signal_ids

    signal_ids = tuple(au_number_to_signal_id(value) for value in au_numbers)
    if len(set(signal_ids)) != len(signal_ids):
        raise ConfigError("AU番号指定に重複があります。")
    if normalized_columns:
        missing_signal_ids = sorted(set(signal_ids) - normalized_columns)
        if missing_signal_ids:
            raise InputContractError(
                "OpenFace CSVに必要なAU強度列がありません: "
                + ", ".join(missing_signal_ids)
            )
    return signal_ids
