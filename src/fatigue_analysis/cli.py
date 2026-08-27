"""非対話CLIの入口。"""

from __future__ import annotations

import argparse
import uuid
import sys
from pathlib import Path

from fatigue_analysis.adapters.report_csv import read_report_csv, validate_report_paths
from fatigue_analysis.adapters.openface_runner import run_openface_conversions
from fatigue_analysis.application.planner import ExecutionPlan, build_execution_plan
from fatigue_analysis.application.registry import default_feature_registry
from fatigue_analysis.application.runner import run_features
from fatigue_analysis.application.visualization_runner import (
    plot_distributions_from_feature_csv,
    plot_timeseries_from_openface_csv,
)
from fatigue_analysis.config.loader import (
    config_to_yaml_text,
    init_config,
    load_config,
    set_config_value,
)
from fatigue_analysis.domain.errors import (
    ConfigError,
    ExternalToolError,
    FatigueAnalysisError,
    InputContractError,
    OutputConflictError,
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
    except InputContractError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except OutputConflictError as exc:
        print(f"Output conflict: {exc}", file=sys.stderr)
        return 3
    except ExternalToolError as exc:
        print(f"External tool error: {exc}", file=sys.stderr)
        return 4
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

    config_set = config_subparsers.add_parser("set", help="設定値を検証付きで更新する")
    config_set.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_set.add_argument("--key", required=True)
    config_set.add_argument("--value", required=True)
    config_set.set_defaults(handler=_handle_config_set)

    list_parser = subparsers.add_parser("list", help="利用可能な項目を表示する")
    list_subparsers = list_parser.add_subparsers(dest="list_target", required=True)

    list_signals = list_subparsers.add_parser("signals", help="利用可能な信号を表示する")
    list_signals.set_defaults(handler=_handle_list_signals)

    list_nodes = list_subparsers.add_parser("nodes", help="登録済みnode仕様を表示する")
    list_nodes.set_defaults(handler=_handle_list_nodes)

    list_features = list_subparsers.add_parser("features", help="登録済み特徴量nodeを表示する")
    list_features.set_defaults(handler=_handle_list_nodes)

    validate_parser = subparsers.add_parser("validate", help="入力ファイルを検証する")
    validate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    validate_parser.add_argument(
        "--require-openface-csv",
        action="store_true",
        help="OpenFace CSVの存在も検証する",
    )
    validate_parser.set_defaults(handler=_handle_validate_inputs)

    plan_parser = subparsers.add_parser("plan", help="実行前の対象と生成予定を表示する")
    plan_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    plan_parser.set_defaults(handler=_handle_plan)

    features_parser = subparsers.add_parser("features", help="特徴量CSVを生成する")
    features_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    features_parser.add_argument("--run-id", default=None)
    features_parser.set_defaults(handler=_handle_features)

    run_parser = subparsers.add_parser("run", help="パイプラインを一括実行する")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument("--run-id", default=None)
    run_parser.set_defaults(handler=_handle_features)

    openface_parser = subparsers.add_parser("openface", help="動画からOpenFace CSVを生成する")
    openface_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    openface_parser.add_argument("--run-id", default=None)
    openface_parser.add_argument("--force", action="store_true")
    openface_parser.set_defaults(handler=_handle_openface)

    plot_parser = subparsers.add_parser("plot", help="既存成果物から可視化を生成する")
    plot_subparsers = plot_parser.add_subparsers(dest="plot_target", required=True)

    plot_distributions = plot_subparsers.add_parser(
        "distributions",
        help="既存features_wide.csvから分布図を生成する",
    )
    plot_distributions.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    plot_distributions.add_argument("--run-id", required=True)
    plot_distributions.add_argument("--plot-id", default=None)
    plot_distributions.set_defaults(handler=_handle_plot_distributions)

    plot_timeseries = plot_subparsers.add_parser(
        "timeseries",
        help="OpenFace CSVから時系列図を生成する",
    )
    plot_timeseries.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    plot_timeseries.add_argument("--sample-id", required=True)
    plot_timeseries.add_argument("--au", type=int, nargs="+", required=True)
    plot_timeseries.add_argument(
        "--series",
        nargs="+",
        default=("raw_validated", "trend"),
        choices=("raw_validated", "trend", "residual"),
    )
    plot_timeseries.add_argument("--plot-id", default=None)
    plot_timeseries.set_defaults(handler=_handle_plot_timeseries)

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


def _handle_config_set(args: argparse.Namespace) -> int:
    result = set_config_value(args.config, key=args.key, value_text=args.value)
    print(
        "Config updated: "
        f"key={result.key}, "
        f"old={result.old_value!r}, "
        f"new={result.new_value!r}, "
        f"backup={result.backup_path}"
    )
    return 0


def _handle_list_signals(args: argparse.Namespace) -> int:
    del args
    print("au_intensity:")
    for signal_id in OPENFACE_AU_INTENSITY_SIGNAL_IDS:
        print(f"  {signal_id}")
    return 0


def _handle_list_nodes(args: argparse.Namespace) -> int:
    del args
    registry = default_feature_registry()
    for spec in registry.list_specs():
        print(
            f"{spec.node_id}: "
            f"version={spec.version}, "
            f"source_series={spec.source_series}, "
            f"signal_type={spec.supported_signal_type}, "
            f"metrics={','.join(spec.metrics)}"
        )
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


def _handle_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plan = build_execution_plan(config)
    print(_format_plan(plan))
    return 0


def _handle_features(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = args.run_id or f"run-{uuid.uuid4().hex[:12]}"
    result = run_features(config, run_id=run_id, repo_root=Path.cwd())
    print(
        "Run complete: "
        f"run_id={result.run_id}, "
        f"status={result.status}, "
        f"samples={result.sample_count}, "
        f"excluded={result.excluded_count}, "
        f"output={result.run_dir}"
    )
    return 0


def _format_plan(plan: ExecutionPlan) -> str:
    lines = [
        "Plan summary:",
        f"  samples={plan.sample_count}",
        f"  baselines={plan.baseline_count}",
        f"  movies={plan.movie_count}",
        (
            "  openface_csv="
            f"{plan.openface_csv_existing_count} existing, "
            f"{plan.openface_csv_missing_count} missing"
        ),
        f"  features={plan.feature_count} ({', '.join(plan.feature_instances)})",
        (
            "  visualizations="
            f"timeseries:{plan.timeseries_count}, "
            f"distributions:{plan.distribution_count}"
        ),
        f"  output_root={plan.output_root}",
    ]
    if plan.openface_csv_missing_sample_ids:
        lines.append(
            "  missing_openface_csv_samples="
            + ", ".join(plan.openface_csv_missing_sample_ids[:10])
        )
        if len(plan.openface_csv_missing_sample_ids) > 10:
            remaining = len(plan.openface_csv_missing_sample_ids) - 10
            lines.append(f"  missing_openface_csv_samples_more={remaining}")
    return "\n".join(lines)


def _handle_openface(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = args.run_id or f"openface-{uuid.uuid4().hex[:12]}"
    results = run_openface_conversions(
        config,
        run_id=run_id,
        force=bool(args.force),
    )
    created = sum(1 for result in results if result.status == "created")
    skipped = sum(1 for result in results if result.status == "skipped")
    print(
        "OpenFace complete: "
        f"run_id={run_id}, created={created}, skipped={skipped}"
    )
    return 0


def _handle_plot_distributions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plot_id = args.plot_id or f"plot-{uuid.uuid4().hex[:12]}"
    run_dir = config.paths.output_root / args.run_id
    feature_csv = run_dir / "features" / "features_wide.csv"
    output_root = run_dir / "plots" / plot_id
    artifacts = plot_distributions_from_feature_csv(
        config,
        feature_csv=feature_csv,
        output_root=output_root,
    )
    print(
        "Plot complete: "
        f"target=distributions, "
        f"run_id={args.run_id}, "
        f"plot_id={plot_id}, "
        f"artifacts={len(artifacts)}, "
        f"output={output_root}"
    )
    return 0


def _handle_plot_timeseries(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plot_id = args.plot_id or f"plot-{uuid.uuid4().hex[:12]}"
    output_root = config.paths.output_root / "plots" / plot_id
    artifacts = plot_timeseries_from_openface_csv(
        config,
        sample_id=args.sample_id,
        au_numbers=tuple(args.au),
        series_kinds=tuple(args.series),
        output_root=output_root,
        plot_id=plot_id,
    )
    print(
        "Plot complete: "
        f"target=timeseries, "
        f"sample_id={args.sample_id}, "
        f"plot_id={plot_id}, "
        f"artifacts={len(artifacts)}, "
        f"output={output_root}"
    )
    return 0
