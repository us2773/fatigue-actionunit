"""YAML設定ファイルの入出力。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping, MutableSequence

import yaml

from fatigue_analysis.config.models import AnalysisConfig
from fatigue_analysis.domain.errors import ConfigError, OutputConflictError

EXAMPLE_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "analysis.example.yaml"
)


@dataclass(frozen=True)
class ConfigSetResult:
    """設定更新結果。"""

    config_path: Path
    backup_path: Path
    key: str
    old_value: Any
    new_value: Any


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


def set_config_value(config_path: Path, *, key: str, value_text: str) -> ConfigSetResult:
    """dot pathで設定値を更新し、更新後のYAML全体を検証する。"""

    raw_config = _load_raw_config(config_path)
    new_value = _parse_yaml_value(value_text)
    old_value = _set_by_dot_path(raw_config, key, new_value)
    AnalysisConfig.from_mapping(raw_config)

    backup_path = _backup_path(config_path)
    shutil.copyfile(config_path, backup_path)
    config_path.write_text(
        yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ConfigSetResult(
        config_path=config_path,
        backup_path=backup_path,
        key=key,
        old_value=old_value,
        new_value=new_value,
    )


def _load_raw_config(config_path: Path) -> MutableMapping[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {config_path}")
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAMLを読み込めません: {config_path}") from exc
    if not isinstance(raw_config, MutableMapping):
        raise ConfigError(f"設定ファイルはmappingである必要があります: {config_path}")
    return raw_config


def _parse_yaml_value(value_text: str) -> Any:
    try:
        return yaml.safe_load(value_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--value をYAML値として解釈できません: {value_text}") from exc


def _set_by_dot_path(raw_config: MutableMapping[str, Any], key: str, value: Any) -> Any:
    if key.strip() == "":
        raise ConfigError("--key は空にできません。")
    parts = tuple(part for part in key.split(".") if part != "")
    if not parts:
        raise ConfigError("--key は空にできません。")

    current: Any = raw_config
    for part in parts[:-1]:
        current = _descend(current, part, key)
    last = parts[-1]
    if isinstance(current, MutableMapping):
        if last not in current:
            raise ConfigError(f"設定keyが見つかりません: {key}")
        old_value = current[last]
        current[last] = value
        return old_value
    if isinstance(current, MutableSequence):
        index = _parse_index(last, key, length=len(current))
        old_value = current[index]
        current[index] = value
        return old_value
    raise ConfigError(f"設定keyの親要素が更新不能です: {key}")


def _descend(current: Any, part: str, key: str) -> Any:
    if isinstance(current, MutableMapping):
        if part not in current:
            raise ConfigError(f"設定keyが見つかりません: {key}")
        return current[part]
    if isinstance(current, MutableSequence):
        return current[_parse_index(part, key, length=len(current))]
    raise ConfigError(f"設定keyの途中要素が参照不能です: {key}")


def _parse_index(part: str, key: str, *, length: int | None = None) -> int:
    try:
        index = int(part)
    except ValueError as exc:
        raise ConfigError(f"list要素は整数indexで指定してください: {key}") from exc
    if index < 0:
        raise ConfigError(f"list要素indexは0以上が必要です: {key}")
    if length is not None and index >= length:
        raise ConfigError(f"list要素indexが範囲外です: {key}")
    return index


def _backup_path(config_path: Path) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config_path.with_name(f"{config_path.name}.{timestamp}.bak")
