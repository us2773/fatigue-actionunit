"""単一YAMLを表す型付き設定モデル。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from fatigue_analysis.domain.errors import ConfigError

SUPPORTED_SCHEMA_VERSION = 1
KNOWN_FEATURE_NODE_IDS = frozenset(
    {"trend_statistics", "raw_peaks", "raw_stft_mean_power"}
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "paths",
        "openface",
        "report",
        "signals",
        "preprocessing",
        "features",
        "analysis",
        "visualizations",
        "outputs",
    }
)


@dataclass(frozen=True)
class PathsConfig:
    """入出力パス設定。"""

    report_csv: Path
    movie_dir: Path
    openface_csv_dir: Path
    output_root: Path


@dataclass(frozen=True)
class OpenFaceConfig:
    """既存OpenFace実行環境を呼び出すための設定。"""

    powershell_script: Path
    local_environment_config: Path
    skip_existing: bool


@dataclass(frozen=True)
class ReportConfig:
    """`report.csv` の契約検証に使う設定。"""

    expected_baselines_per_person: int
    extra_column_types: Mapping[str, str]


@dataclass(frozen=True)
class AuSignalSelection:
    """人間向けAU番号指定。"""

    au_numbers: str | tuple[int, ...]


@dataclass(frozen=True)
class SignalsConfig:
    """分析対象シグナルの指定。"""

    au_intensity: AuSignalSelection


@dataclass(frozen=True)
class LowessConfig:
    """LOWESS前処理の設定。"""

    enabled: bool
    frac: float
    it: int
    delta: float


@dataclass(frozen=True)
class PreprocessingConfig:
    """品質処理と前処理の設定。"""

    initial_trim_ratio: float
    confidence_threshold: float
    lowess: LowessConfig


@dataclass(frozen=True)
class FeatureConfig:
    """1つの特徴量node実行設定。"""

    instance_id: str
    node_id: str
    signals: AuSignalSelection
    params: Mapping[str, Any]


@dataclass(frozen=True)
class AnalysisRulesConfig:
    """分布可視化で使う宣言的な抽出・派生設定。"""

    filters: tuple[Mapping[str, Any], ...]
    derived_columns: Mapping[str, Any]


@dataclass(frozen=True)
class VisualizationsConfig:
    """出力する可視化設定。"""

    timeseries: tuple[Mapping[str, Any], ...]
    distributions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class OutputsConfig:
    """成果物保存の設定。"""

    save_intermediate_nodes: tuple[str, ...]
    csv_encoding: str
    float_precision: int
    overwrite: bool


@dataclass(frozen=True)
class AnalysisConfig:
    """研究実行条件を表す解決済み設定。"""

    schema_version: int
    paths: PathsConfig
    openface: OpenFaceConfig
    report: ReportConfig
    signals: SignalsConfig
    preprocessing: PreprocessingConfig
    features: tuple[FeatureConfig, ...]
    analysis: AnalysisRulesConfig
    visualizations: VisualizationsConfig
    outputs: OutputsConfig

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "AnalysisConfig":
        """未検証mappingから型付き設定を生成する。"""

        _ensure_mapping(raw_config, "root")
        _reject_unknown_keys(raw_config, TOP_LEVEL_KEYS, "root")

        schema_version = _required_int(raw_config, "schema_version", "root")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version={schema_version} は未対応です。"
                f"対応版は {SUPPORTED_SCHEMA_VERSION} です。"
            )

        config = cls(
            schema_version=schema_version,
            paths=_parse_paths(_required_mapping(raw_config, "paths", "root")),
            openface=_parse_openface(
                _required_mapping(raw_config, "openface", "root")
            ),
            report=_parse_report(_required_mapping(raw_config, "report", "root")),
            signals=_parse_signals(_required_mapping(raw_config, "signals", "root")),
            preprocessing=_parse_preprocessing(
                _required_mapping(raw_config, "preprocessing", "root")
            ),
            features=_parse_features(_required_sequence(raw_config, "features", "root")),
            analysis=_parse_analysis(raw_config.get("analysis", {})),
            visualizations=_parse_visualizations(raw_config.get("visualizations", {})),
            outputs=_parse_outputs(_required_mapping(raw_config, "outputs", "root")),
        )
        _validate_feature_instance_ids(config.features)
        return config

    def to_plain_dict(self) -> dict[str, Any]:
        """表示とmanifest保存に使えるプリミティブ値へ変換する。"""

        return _plain_value(asdict(self))


def _parse_paths(raw_paths: Mapping[str, Any]) -> PathsConfig:
    _reject_unknown_keys(
        raw_paths,
        {"report_csv", "movie_dir", "openface_csv_dir", "output_root"},
        "paths",
    )
    return PathsConfig(
        report_csv=Path(_required_str(raw_paths, "report_csv", "paths")),
        movie_dir=Path(_required_str(raw_paths, "movie_dir", "paths")),
        openface_csv_dir=Path(_required_str(raw_paths, "openface_csv_dir", "paths")),
        output_root=Path(_required_str(raw_paths, "output_root", "paths")),
    )


def _parse_openface(raw_openface: Mapping[str, Any]) -> OpenFaceConfig:
    _reject_unknown_keys(
        raw_openface,
        {"powershell_script", "local_environment_config", "skip_existing"},
        "openface",
    )
    return OpenFaceConfig(
        powershell_script=Path(
            _required_str(raw_openface, "powershell_script", "openface")
        ),
        local_environment_config=Path(
            _required_str(raw_openface, "local_environment_config", "openface")
        ),
        skip_existing=_required_bool(raw_openface, "skip_existing", "openface"),
    )


def _parse_report(raw_report: Mapping[str, Any]) -> ReportConfig:
    _reject_unknown_keys(
        raw_report,
        {"expected_baselines_per_person", "extra_column_types"},
        "report",
    )
    expected = _required_int(
        raw_report, "expected_baselines_per_person", "report"
    )
    if expected < 0:
        raise ConfigError("report.expected_baselines_per_person は0以上が必要です。")
    extra_column_types = raw_report.get("extra_column_types", {})
    _ensure_mapping(extra_column_types, "report.extra_column_types")
    return ReportConfig(
        expected_baselines_per_person=expected,
        extra_column_types=dict(extra_column_types),
    )


def _parse_signals(raw_signals: Mapping[str, Any]) -> SignalsConfig:
    _reject_unknown_keys(raw_signals, {"au_intensity"}, "signals")
    return SignalsConfig(
        au_intensity=_parse_au_signal_selection(
            _required_mapping(raw_signals, "au_intensity", "signals"),
            "signals.au_intensity",
        )
    )


def _parse_preprocessing(raw_preprocessing: Mapping[str, Any]) -> PreprocessingConfig:
    _reject_unknown_keys(
        raw_preprocessing,
        {"initial_trim_ratio", "confidence_threshold", "lowess"},
        "preprocessing",
    )
    initial_trim_ratio = _required_number(
        raw_preprocessing, "initial_trim_ratio", "preprocessing"
    )
    confidence_threshold = _required_number(
        raw_preprocessing, "confidence_threshold", "preprocessing"
    )
    if not 0 <= initial_trim_ratio < 1:
        raise ConfigError("preprocessing.initial_trim_ratio は 0以上1未満が必要です。")
    if not 0 <= confidence_threshold <= 1:
        raise ConfigError("preprocessing.confidence_threshold は 0以上1以下が必要です。")
    return PreprocessingConfig(
        initial_trim_ratio=float(initial_trim_ratio),
        confidence_threshold=float(confidence_threshold),
        lowess=_parse_lowess(
            _required_mapping(raw_preprocessing, "lowess", "preprocessing")
        ),
    )


def _parse_lowess(raw_lowess: Mapping[str, Any]) -> LowessConfig:
    _reject_unknown_keys(raw_lowess, {"enabled", "frac", "it", "delta"}, "lowess")
    enabled = _required_bool(raw_lowess, "enabled", "lowess")
    frac = _required_number(raw_lowess, "frac", "lowess")
    iterations = _required_int(raw_lowess, "it", "lowess")
    delta = _required_number(raw_lowess, "delta", "lowess")
    if not 0 < frac <= 1:
        raise ConfigError("preprocessing.lowess.frac は 0より大きく1以下が必要です。")
    if iterations < 0:
        raise ConfigError("preprocessing.lowess.it は0以上が必要です。")
    if delta < 0:
        raise ConfigError("preprocessing.lowess.delta は0以上が必要です。")
    return LowessConfig(enabled=enabled, frac=float(frac), it=iterations, delta=float(delta))


def _parse_features(raw_features: tuple[Any, ...]) -> tuple[FeatureConfig, ...]:
    features: list[FeatureConfig] = []
    for index, raw_feature in enumerate(raw_features):
        section = f"features[{index}]"
        _ensure_mapping(raw_feature, section)
        _reject_unknown_keys(
            raw_feature, {"instance_id", "node_id", "signals", "params"}, section
        )
        instance_id = _required_str(raw_feature, "instance_id", section)
        node_id = _required_str(raw_feature, "node_id", section)
        if node_id not in KNOWN_FEATURE_NODE_IDS:
            known = ", ".join(sorted(KNOWN_FEATURE_NODE_IDS))
            raise ConfigError(f"{section}.node_id={node_id!r} は未登録です: {known}")
        features.append(
            FeatureConfig(
                instance_id=instance_id,
                node_id=node_id,
                signals=_parse_au_signal_selection(
                    _required_mapping(raw_feature, "signals", section),
                    f"{section}.signals",
                ),
                params=dict(_required_mapping(raw_feature, "params", section)),
            )
        )
    return tuple(features)


def _parse_analysis(raw_analysis: Any) -> AnalysisRulesConfig:
    _ensure_mapping(raw_analysis, "analysis")
    _reject_unknown_keys(raw_analysis, {"filters", "derived_columns"}, "analysis")
    filters = raw_analysis.get("filters", [])
    derived_columns = raw_analysis.get("derived_columns", {})
    _ensure_sequence(filters, "analysis.filters")
    _ensure_mapping(derived_columns, "analysis.derived_columns")
    return AnalysisRulesConfig(filters=tuple(filters), derived_columns=dict(derived_columns))


def _parse_visualizations(raw_visualizations: Any) -> VisualizationsConfig:
    _ensure_mapping(raw_visualizations, "visualizations")
    _reject_unknown_keys(
        raw_visualizations, {"timeseries", "distributions"}, "visualizations"
    )
    timeseries = raw_visualizations.get("timeseries", [])
    distributions = raw_visualizations.get("distributions", [])
    _ensure_sequence(timeseries, "visualizations.timeseries")
    _ensure_sequence(distributions, "visualizations.distributions")
    return VisualizationsConfig(
        timeseries=tuple(timeseries),
        distributions=tuple(distributions),
    )


def _parse_outputs(raw_outputs: Mapping[str, Any]) -> OutputsConfig:
    _reject_unknown_keys(
        raw_outputs,
        {"save_intermediate_nodes", "csv_encoding", "float_precision", "overwrite"},
        "outputs",
    )
    save_intermediate_nodes = _required_sequence(
        raw_outputs, "save_intermediate_nodes", "outputs"
    )
    csv_encoding = _required_str(raw_outputs, "csv_encoding", "outputs")
    float_precision = _required_int(raw_outputs, "float_precision", "outputs")
    overwrite = _required_bool(raw_outputs, "overwrite", "outputs")
    if float_precision < 0:
        raise ConfigError("outputs.float_precision は0以上が必要です。")
    return OutputsConfig(
        save_intermediate_nodes=tuple(str(value) for value in save_intermediate_nodes),
        csv_encoding=csv_encoding,
        float_precision=float_precision,
        overwrite=overwrite,
    )


def _parse_au_signal_selection(
    raw_selection: Mapping[str, Any], section: str
) -> AuSignalSelection:
    _reject_unknown_keys(raw_selection, {"au_numbers"}, section)
    raw_au_numbers = raw_selection.get("au_numbers")
    if raw_au_numbers == "all":
        return AuSignalSelection(au_numbers="all")
    _ensure_sequence(raw_au_numbers, f"{section}.au_numbers")
    au_numbers = tuple(_coerce_int(value, f"{section}.au_numbers") for value in raw_au_numbers)
    if not au_numbers:
        raise ConfigError(f"{section}.au_numbers は空にできません。")
    if len(set(au_numbers)) != len(au_numbers):
        raise ConfigError(f"{section}.au_numbers に重複があります。")
    if any(value <= 0 for value in au_numbers):
        raise ConfigError(f"{section}.au_numbers は正のAU番号だけを受け付けます。")
    return AuSignalSelection(au_numbers=au_numbers)


def _validate_feature_instance_ids(features: tuple[FeatureConfig, ...]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for feature in features:
        if feature.instance_id in seen:
            duplicates.add(feature.instance_id)
        seen.add(feature.instance_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ConfigError(f"features.instance_id が重複しています: {duplicate_list}")


def _required_mapping(
    raw_mapping: Mapping[str, Any], key: str, section: str
) -> Mapping[str, Any]:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    value = raw_mapping[key]
    _ensure_mapping(value, f"{section}.{key}")
    return value


def _required_sequence(
    raw_mapping: Mapping[str, Any], key: str, section: str
) -> tuple[Any, ...]:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    value = raw_mapping[key]
    _ensure_sequence(value, f"{section}.{key}")
    return tuple(value)


def _required_str(raw_mapping: Mapping[str, Any], key: str, section: str) -> str:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    value = raw_mapping[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} は空でない文字列が必要です。")
    return value


def _required_bool(raw_mapping: Mapping[str, Any], key: str, section: str) -> bool:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    value = raw_mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} は真偽値が必要です。")
    return value


def _required_int(raw_mapping: Mapping[str, Any], key: str, section: str) -> int:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    return _coerce_int(raw_mapping[key], f"{section}.{key}")


def _required_number(raw_mapping: Mapping[str, Any], key: str, section: str) -> float:
    if key not in raw_mapping:
        raise ConfigError(f"{section}.{key} は必須です。")
    value = raw_mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{section}.{key} は数値が必要です。")
    return float(value)


def _coerce_int(value: Any, section: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section} は整数が必要です。")
    return int(value)


def _ensure_mapping(value: Any, section: str) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} はmappingである必要があります。")


def _ensure_sequence(value: Any, section: str) -> None:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ConfigError(f"{section} はlistである必要があります。")


def _reject_unknown_keys(
    raw_mapping: Mapping[str, Any], allowed_keys: frozenset[str] | set[str], section: str
) -> None:
    unknown_keys = sorted(set(raw_mapping) - set(allowed_keys))
    if unknown_keys:
        raise ConfigError(f"{section} に未知キーがあります: {', '.join(unknown_keys)}")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_value(child) for child in value]
    return value
