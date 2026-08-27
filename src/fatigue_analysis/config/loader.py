"""YAML設定ファイルの入出力。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from fatigue_analysis.config.models import AnalysisConfig
from fatigue_analysis.domain.errors import ConfigError, OutputConflictError

EXAMPLE_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "analysis.example.yaml"
)


def load_config(config_path: Path) -> AnalysisConfig:
    """単一YAMLを読み込み、型付き設定として検証する。"""

    if not config_path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {config_path}")
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAMLを読み込めません: {config_path}") from exc
    if raw_config is None:
        raise ConfigError(f"設定ファイルが空です: {config_path}")
    return AnalysisConfig.from_mapping(raw_config)


def init_config(config_path: Path, *, overwrite: bool = False) -> None:
    """追跡可能な雛形からGit追跡外の正本YAMLを作る。"""

    if config_path.exists() and not overwrite:
        raise OutputConflictError(
            f"既存設定を上書きしません。必要なら明示的にoverwriteしてください: {config_path}"
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE_CONFIG_PATH, config_path)


def config_to_yaml_text(config: AnalysisConfig) -> str:
    """検証済み設定を表示用YAML文字列へ変換する。"""

    plain_config: dict[str, Any] = config.to_plain_dict()
    return yaml.safe_dump(
        plain_config,
        allow_unicode=True,
        sort_keys=False,
    )
