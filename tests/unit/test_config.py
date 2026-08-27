from __future__ import annotations

from pathlib import Path

import pytest

from fatigue_analysis.config.loader import load_config
from fatigue_analysis.domain.errors import ConfigError


def test_example_config_loads() -> None:
    """追跡可能な雛形YAMLを正本設定として読み込める。"""

    config = load_config(Path("conf/analysis.example.yaml"))

    assert config.schema_version == 1
    assert config.paths.report_csv.as_posix() == "data/02_report/report.csv"
    assert len(config.features) == 3
    assert config.features[0].instance_id == "trend_stats"


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    """未対応schema_versionは実行前に拒否する。"""

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        Path("conf/analysis.example.yaml")
        .read_text(encoding="utf-8")
        .replace("schema_version: 1", "schema_version: 999"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未対応"):
        load_config(config_path)


def test_duplicate_feature_instance_id_is_rejected(tmp_path: Path) -> None:
    """feature instance_idの重複はwide列衝突の前に拒否する。"""

    config_path = tmp_path / "analysis.yaml"
    config_text = Path("conf/analysis.example.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace("instance_id: raw_peaks", "instance_id: trend_stats")
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ConfigError, match="重複"):
        load_config(config_path)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    """未知キーを黙って無視しない。"""

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        Path("conf/analysis.example.yaml").read_text(encoding="utf-8")
        + "\nunexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未知キー"):
        load_config(config_path)
