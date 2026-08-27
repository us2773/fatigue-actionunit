"""非対話CLIの入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.config.loader import config_to_yaml_text, init_config, load_config
from fatigue_analysis.domain.errors import (
    ConfigError,
    FatigueAnalysisError,
    InputContractError,
)
from fatigue_analysis.domain.signals import OPENFACE_AU_INTENSITY_SIGNAL_IDS

DEFAULT_CONFIG_PATH = Path("conf/analysis.yaml")


def main(argv: list[str] | None = None) -> int:
    """CLI引数を解釈し、終了コードを返す。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except FatigueAnalysisError as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fatigue-analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="設定YAMLを操作する")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    config_init = config_subparsers.add_parser("init", help="設定雛形を作成する")
    config_init.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_init.add_argument("--overwrite", action="store_true")
    config_init.set_defaults(handler=_handle_config_init)

    config_validate = config_subparsers.add_parser("validate", help="設定を検証する")
    config_validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_validate.set_defaults(handler=_handle_config_validate)

    config_show = config_subparsers.add_parser("show", help="解決済み設定を表示する")
    config_show.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_show.set_defaults(handler=_handle_config_show)

    list_parser = subparsers.add_parser("list", help="利用可能な項目を表示する")
    list_subparsers = list_parser.add_subparsers(dest="list_target", required=True)

    list_signals = list_subparsers.add_parser("signals", help="利用可能な信号を表示する")
    list_signals.set_defaults(handler=_handle_list_signals)

    list_nodes = list_subparsers.add_parser("nodes", help="登録予定nodeを表示する")
    list_nodes.set_defaults(handler=_handle_list_nodes)

    validate_parser = subparsers.add_parser("validate", help="入力ファイルを検証する")
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    validate_parser.add_argument(
        "--require-openface-csv",
        action="store_true",
        help="OpenFace CSVの存在も検証する",
    )
    validate_parser.set_defaults(handler=_handle_validate_inputs)

    return parser


def _handle_config_init(args: argparse.Namespace) -> int:
    init_config(args.config, overwrite=bool(args.overwrite))
    print(f"Created config template: {args.config}")
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        "Config is valid: "
        f"schema_version={config.schema_version}, "
        f"features={len(config.features)}"
    )
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(config_to_yaml_text(config), end="")
    return 0


def _handle_list_signals(args: argparse.Namespace) -> int:
    del args
    print("au_intensity:")
    for signal_id in OPENFACE_AU_INTENSITY_SIGNAL_IDS:
        print(f"  {signal_id}")
    return 0


def _handle_list_nodes(args: argparse.Namespace) -> int:
    del args
    print("trend_statistics")
    print("raw_peaks")
    print("raw_stft_mean_power")
    return 0


def _handle_validate_inputs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = read_report_csv(
        config.paths.report_csv,
        expected_baselines_per_person=config.report.expected_baselines_per_person,
    )
    try:
        validate_report_paths(
            report,
            movie_dir=config.paths.movie_dir,
            openface_csv_dir=config.paths.openface_csv_dir,
            require_openface_csv=bool(args.require_openface_csv),
        )
    except InputContractError:
        raise
    print(f"Inputs are valid: samples={len(report.samples)}")
    return 0
