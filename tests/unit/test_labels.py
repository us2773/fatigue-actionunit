from __future__ import annotations

from fatigue_analysis.domain.labels import class_label, fatigue_level_label


def test_report_code_labels_are_human_readable() -> None:
    """report.csvの数値コードを研究者向けラベルへ変換する。"""

    assert class_label(1) == "だるい"
    assert class_label(2) == "混合・だるい寄り"
    assert class_label(3) == "混合・こわばり寄り"
    assert class_label(4) == "こわばり"
    assert fatigue_level_label(1) == "元気"
    assert fatigue_level_label(2) == "やや疲労感"
    assert fatigue_level_label(3) == "はっきり疲労感"


def test_empty_report_codes_emit_empty_labels() -> None:
    """基準表情などの空欄コードは空欄ラベルとして出力する。"""

    assert class_label(None) == ""
    assert fatigue_level_label(None) == ""
