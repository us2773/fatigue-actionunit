"""研究者向け表示に使う分類コードとラベルの定義。"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

CLASS_LABEL_COLUMN = "class_label"
FATIGUE_LEVEL_LABEL_COLUMN = "fatigue_level_label"

CLASS_LABELS: Mapping[int, str] = MappingProxyType(
    {
        1: "だるい",
        2: "混合・だるい寄り",
        3: "混合・こわばり寄り",
        4: "こわばり",
    }
)
FATIGUE_LEVEL_LABELS: Mapping[int, str] = MappingProxyType(
    {
        1: "元気",
        2: "やや疲労感",
        3: "はっきり疲労感",
    }
)
REPORT_LABEL_COLUMNS: tuple[str, str] = (
    CLASS_LABEL_COLUMN,
    FATIGUE_LEVEL_LABEL_COLUMN,
)


def class_label(class_code: int | None) -> str:
    """分類コードを出力用ラベルへ変換する。"""

    if class_code is None:
        return ""
    return CLASS_LABELS[class_code]


def fatigue_level_label(fatigue_level: int | None) -> str:
    """疲労印象度コードを出力用ラベルへ変換する。"""

    if fatigue_level is None:
        return ""
    return FATIGUE_LEVEL_LABELS[fatigue_level]
