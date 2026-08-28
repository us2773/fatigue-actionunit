"""特徴量分布可視化。"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DistributionGroup:
    """分布図の1群定義。"""

    label: str
    where: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ScatterPoint:
    """分布図へ重ねる1点の描画情報。"""

    x: float
    y: float
    label: str | None


def plot_feature_distribution(
    rows: Sequence[Mapping[str, str]],
    *,
    feature_column: str,
    groups: tuple[DistributionGroup, ...],
    output_png: Path,
    stats_csv: Path,
    color_by: str | None = None,
) -> None:
    """箱ひげ図に全サンプル点を重ね、群別統計CSVを保存する。"""

    if len(groups) < 2:
        raise ValueError("分布可視化には2群以上が必要です。")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    stats_csv.parent.mkdir(parents=True, exist_ok=True)

    grouped_values: list[np.ndarray] = []
    scatter_points: list[ScatterPoint] = []
    stats_rows: list[dict[str, object]] = []
    figure, axis = plt.subplots(figsize=(8, 4.8))

    for group_index, group in enumerate(groups, start=1):
        matched_rows = [row for row in rows if _matches_where(row, group.where)]
        values = np.array(
            [float(row[feature_column]) for row in matched_rows if row.get(feature_column, "") != ""],
            dtype=float,
        )
        grouped_values.append(values)
        stats_rows.append(_stats_row(group.label, values, len(matched_rows)))

        for row in matched_rows:
            raw_value = row.get(feature_column, "")
            if raw_value == "":
                continue
            sample_id = row.get("Name", "")
            jitter = _stable_jitter(sample_id)
            color_value = row.get(color_by, "") if color_by else ""
            scatter_points.append(
                ScatterPoint(
                    x=group_index + jitter,
                    y=float(raw_value),
                    label=f"{color_by}={color_value}" if color_by else None,
                )
            )

    if all(len(values) == 0 for values in grouped_values):
        raise ValueError("描画可能な特徴量値がありません。")

    axis.boxplot(
        grouped_values,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d8e6f3", "edgecolor": "#333333"},
        medianprops={"color": "#b22222", "linewidth": 1.5},
    )
    for point in scatter_points:
        axis.scatter(
            point.x,
            point.y,
            s=28,
            alpha=0.8,
            label=point.label,
            edgecolors="#333333",
            linewidths=0.4,
            zorder=3,
        )
    axis.set_xticks(range(1, len(groups) + 1))
    axis.set_xticklabels([group.label for group in groups])
    axis.set_ylabel(feature_column)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_png, dpi=150)
    plt.close(figure)

    with stats_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = ("group", "matched_count", "count", "mean", "median", "q1", "q3")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)


def _matches_where(row: Mapping[str, str], where: Mapping[str, tuple[str, ...]]) -> bool:
    return all(str(row.get(column, "")) in values for column, values in where.items())


def _stable_jitter(sample_id: str) -> float:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    raw_value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return (raw_value - 0.5) * 0.18


def _stats_row(
    group_label: str,
    values: np.ndarray,
    matched_count: int,
) -> dict[str, object]:
    if len(values) == 0:
        return {
            "group": group_label,
            "matched_count": matched_count,
            "count": 0,
            "mean": "",
            "median": "",
            "q1": "",
            "q3": "",
        }
    return {
        "group": group_label,
        "matched_count": matched_count,
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }
