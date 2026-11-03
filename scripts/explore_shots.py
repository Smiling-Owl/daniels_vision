"""Create training-partition-only exploratory tables and figures after holdout locking."""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

try:
    from scripts import acquire_data as acquisition
    from scripts import reserve_holdout as holdout
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import reserve_holdout as holdout

NAVY = "#19324F"
GREEN = "#4C956C"
ORANGE = "#E98945"
GREY = "#AAB4BE"
LIGHT_GREY = "#D8DEE4"
SOURCE_CREDIT = "Training partition only • StatsBomb Open Data • La Liga 2015/16"
FIGURE_NAMES = [
    "phase05_01_class_balance.png", "phase05_02_training_shot_map.png",
    "phase05_03_geometry_distributions.png", "phase05_04_goal_rate_by_geometry.png",
    "phase05_05_goal_rate_by_category.png", "phase05_06_goal_rate_by_context_flags.png",
]
TABLE_NAMES = [
    "phase05_class_balance.csv", "phase05_categorical_goal_rates.csv",
    "phase05_context_flag_goal_rates.csv", "phase05_geometry_goal_rates.csv",
]


class EDAError(RuntimeError):
    """Source or partition failed validation before EDA output generation."""


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson interval for a binomial proportion; returns None for n=0."""
    if isinstance(successes, bool) or isinstance(total, bool) or not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("successes and total must be integers.")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total.")
    if total == 0:
        return None, None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) / total) + z2 / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == total else min(1.0, center + half)
    return min(low, p), max(high, p)


def _rate_row(dimension: str, category: str, frame: pd.DataFrame, rare_threshold: int = 30) -> dict[str, Any]:
    n = len(frame)
    goals = int(frame["is_goal"].sum())
    low, high = wilson_interval(goals, n)
    return {
        "dimension": dimension, "category": category, "shot_count": n, "goal_count": goals,
        "goal_rate": goals / n if n else None, "wilson_95_low": low, "wilson_95_high": high,
        "small_group_n_lt_30": n < rare_threshold,
    }


def categorical_rates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        for value in sorted(frame[column].astype(str).unique()):
            rows.append(_rate_row(column, value, frame.loc[frame[column].astype(str) == value]))
    return pd.DataFrame(rows, columns=[
        "dimension", "category", "shot_count", "goal_count", "goal_rate",
        "wilson_95_low", "wilson_95_high", "small_group_n_lt_30",
    ])


def geometry_rates(frame: pd.DataFrame, column: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        raise EDAError(f"{column} contains a non-numeric geometry value.")
    # The CSV uses a stable 12-significant-digit format. Allow only tiny serialization
    # roundoff at the final edge (notably pi), and clamp those values into the edge bin.
    tolerance = 1e-12
    if (numeric > edges[-1] + tolerance).any() or (numeric < edges[0] - tolerance).any():
        raise EDAError(f"{column} values fall outside the documented geometry bins.")
    numeric = numeric.clip(lower=edges[0], upper=edges[-1])
    values = pd.cut(numeric, bins=edges, labels=labels, include_lowest=True, right=True)
    if values.isna().any():
        raise EDAError(f"{column} values fall outside the documented geometry bins.")
    rows = []
    for label in labels:
        mask = values == label
        rows.append(_rate_row(column, str(label), frame.loc[mask]))
    return pd.DataFrame(rows)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe records to strict JSON values, mapping empty-bin NaNs to null."""
    records = []
    for source in frame.to_dict(orient="records"):
        row = {}
        for key, value in source.items():
            if pd.isna(value):
                row[key] = None
            elif hasattr(value, "item"):
                row[key] = value.item()
            else:
                row[key] = value
        records.append(row)
    return records


def context_rates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if not frame[column].isin([0, 1]).all():
            raise EDAError(f"Context indicator {column} must contain only 0 and 1.")
        for value, label in ((0, "NOT FLAGGED"), (1, "FLAGGED")):
            rows.append(_rate_row(column, label, frame.loc[frame[column] == value]))
    return pd.DataFrame(rows)


def select_training_rows(shots: pd.DataFrame, train_match_ids: set[int]) -> pd.DataFrame:
    """Return rows from locked training matches only; no labels/features guide selection."""
    return shots.loc[shots["match_id"].isin(train_match_ids)].copy()


def summarize_training(training: pd.DataFrame) -> dict[str, Any]:
    """Aggregate only the dataframe supplied by the caller, which must be training rows."""
    shot_count = len(training)
    goal_count = int(training["is_goal"].sum())
    class_table = pd.DataFrame([
        {"class": "Goal", "shot_count": goal_count, "percentage": goal_count / shot_count},
        {"class": "Non-goal", "shot_count": shot_count - goal_count, "percentage": (shot_count - goal_count) / shot_count},
    ])
    return {
        "shot_count": shot_count,
        "match_count": int(training["match_id"].nunique()),
        "goal_count": goal_count,
        "non_goal_count": shot_count - goal_count,
        "overall_goal_rate": goal_count / shot_count,
        "no_goal_majority_baseline_accuracy": (shot_count - goal_count) / shot_count,
        "penalty_shot_count": int(training["is_penalty"].sum()),
        "penalty_goal_count": int(training.loc[training["is_penalty"] == 1, "is_goal"].sum()),
        "class_table": class_table,
    }


def _figure_bytes(make_figure) -> bytes:
    figure = make_figure()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=220, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(figure)
    return buffer.getvalue()


def _finish_figure(fig, title: str) -> None:
    fig.suptitle(title, color=NAVY, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, SOURCE_CREDIT, ha="center", va="bottom", fontsize=8, color=NAVY)


def _rate_plot(ax, table: pd.DataFrame, title: str, color: str) -> None:
    nonempty = table.loc[table["shot_count"] > 0].copy()
    x = list(range(len(nonempty)))
    rates = nonempty["goal_rate"].astype(float).tolist()
    lows = nonempty["wilson_95_low"].astype(float).tolist()
    highs = nonempty["wilson_95_high"].astype(float).tolist()
    errors = [[rate - low for rate, low in zip(rates, lows)], [high - rate for rate, high in zip(rates, highs)]]
    ax.errorbar(x, rates, yerr=errors, fmt="o", color=color, ecolor=GREY, capsize=3, markersize=6)
    ax.set_xticks(x, [f"{row.category}{' *' if row.small_group_n_lt_30 else ''}\nn={row.shot_count}" for row in nonempty.itertuples()], rotation=25, ha="right")
    ax.set_ylim(0, max(0.25, min(1.0, max(highs, default=0.25) * 1.18)))
    ax.set_ylabel("Observed goal rate")
    ax.set_title(title, color=NAVY, fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)


def _make_figures(training: pd.DataFrame, summary: dict[str, Any], cat_rates: pd.DataFrame,
                  context_table: pd.DataFrame, geometry_table: pd.DataFrame) -> dict[str, bytes]:
    figures: dict[str, bytes] = {}

    def class_balance():
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="white")
        labels = ["Non-goal", "Goal"]
        counts = [summary["non_goal_count"], summary["goal_count"]]
        colors = [NAVY, ORANGE]
        bars = ax.bar(labels, counts, color=colors, width=0.62)
        ax.set_ylabel("Shots in training partition")
        ax.set_title("Class balance", color=NAVY, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
        for bar, count in zip(bars, counts):
            pct = 100 * count / summary["shot_count"]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{count:,}\n{pct:.1f}%", ha="center", va="bottom", color=NAVY)
        ax.text(0.98, 0.94, f"Descriptive no-goal baseline accuracy: {summary['no_goal_majority_baseline_accuracy']:.1%}\n(not a fitted model)", transform=ax.transAxes, ha="right", va="top", color=NAVY, fontsize=9)
        ax.set_ylim(0, max(counts) * 1.2)
        _finish_figure(fig, "Shot outcomes in the training partition")
        fig.tight_layout(rect=[0.02, 0.055, 0.98, 0.91])
        return fig
    figures[FIGURE_NAMES[0]] = _figure_bytes(class_balance)

    def shot_map():
        fig, ax = plt.subplots(figsize=(8.5, 8), facecolor="white")
        ax.set_facecolor("white")
        line = "#71917C"
        ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, edgecolor=line, linewidth=1.5))
        ax.plot([60, 60], [0, 80], color=line, linewidth=1.0, alpha=0.8)
        ax.plot([120, 120], [0, 80], color=line, linewidth=1.5)
        ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, edgecolor=line, linewidth=1.4))
        ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, edgecolor=line, linewidth=1.2))
        ax.add_patch(Rectangle((120, 36), 2, 8, fill=False, edgecolor=line, linewidth=1.2, clip_on=False))
        ax.plot([120], [40], marker="o", markersize=2.5, color=line)
        non_goal = training.loc[training["is_goal"] == 0]
        goal = training.loc[training["is_goal"] == 1]
        ax.scatter(non_goal["x"], non_goal["y"], s=9, color=GREY, alpha=0.28, linewidths=0, label=f"Non-goal (n={len(non_goal):,})")
        ax.scatter(goal["x"], goal["y"], s=19, color=ORANGE, alpha=0.83, edgecolors="white", linewidths=0.25, label=f"Goal (n={len(goal):,})")
        ax.set_xlim(40, 120)
        ax.set_ylim(80, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("StatsBomb x coordinate (native units; attack toward x = 120)")
        ax.set_ylabel("StatsBomb y coordinate (native units)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(False)
        _finish_figure(fig, "Training shot locations")
        fig.tight_layout(rect=[0.04, 0.055, 0.97, 0.91])
        return fig
    figures[FIGURE_NAMES[1]] = _figure_bytes(shot_map)

    def geometry_distributions():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
        axes[0].hist(training["distance_sb_units"], bins=30, color=NAVY, alpha=0.9)
        axes[0].set_xlabel("Distance to goal centre (StatsBomb units; not metres)")
        axes[0].set_ylabel("Training shots")
        axes[0].set_title("Distance", color=NAVY, fontweight="bold")
        axes[1].hist(training["angle_sb_radians"], bins=30, range=(0, math.pi), color=GREEN, alpha=0.9)
        axes[1].set_xlabel("Angle subtended by goalposts (radians)")
        axes[1].set_ylabel("Training shots")
        axes[1].set_title("Goal angle", color=NAVY, fontweight="bold")
        for ax in axes: ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
        _finish_figure(fig, "Shot geometry distributions")
        fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.90])
        return fig
    figures[FIGURE_NAMES[2]] = _figure_bytes(geometry_distributions)

    def geometry_goal_rates():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
        _rate_plot(axes[0], geometry_table.loc[geometry_table["dimension"] == "distance_sb_units"], "Distance bands", NAVY)
        _rate_plot(axes[1], geometry_table.loc[geometry_table["dimension"] == "angle_sb_radians"], "Angle bands", GREEN)
        fig.text(0.5, 0.065, "Error bars: 95% Wilson intervals; independent-shot approximation, with within-match clustering caveat. * marks n < 30.", ha="center", fontsize=8, color=NAVY)
        _finish_figure(fig, "Observed training goal rate by geometry")
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.90])
        return fig
    figures[FIGURE_NAMES[3]] = _figure_bytes(geometry_goal_rates)

    def categorical_goal_rates():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
        _rate_plot(axes[0], cat_rates.loc[cat_rates["dimension"] == "body_part"], "Body part", NAVY)
        _rate_plot(axes[1], cat_rates.loc[cat_rates["dimension"] == "shot_type"], "Shot type", GREEN)
        displayed = cat_rates.loc[cat_rates["dimension"].isin(["body_part", "shot_type"])]
        small_count = int(displayed["small_group_n_lt_30"].sum())
        fig.text(0.5, 0.065, f"Error bars: 95% Wilson intervals (independent-shot approximation; within-match clustering caveat). * marks n < 30; {small_count} displayed groups are small.", ha="center", fontsize=8, color=NAVY)
        _finish_figure(fig, "Observed training goal rate by shot category")
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.90])
        return fig
    figures[FIGURE_NAMES[4]] = _figure_bytes(categorical_goal_rates)

    def context_goal_rates():
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")
        for ax, feature, title, color in zip(
            axes, ["under_pressure", "first_time", "open_goal"],
            ["Under pressure", "First-time shot", "Open goal"], [NAVY, GREEN, ORANGE],
        ):
            _rate_plot(ax, context_table.loc[context_table["dimension"] == feature], title, color)
        fig.text(0.5, 0.065, "Error bars: 95% Wilson intervals (independent-shot approximation; within-match clustering caveat). NOT FLAGGED is not verified absence.", ha="center", fontsize=8, color=NAVY)
        _finish_figure(fig, "Observed training goal rate by provider context flags")
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.90])
        return fig
    figures[FIGURE_NAMES[5]] = _figure_bytes(context_goal_rates)
    return figures


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    # Reservation verifies the input digest using only match_id; labels are read only after
    # the match-level holdout has been materialized and its lock checked.
    split_manifest = holdout.reserve(root)
    shot_path = root / holdout.SHOT_RELATIVE
    shot_report = json.loads((root / holdout.REPORT_RELATIVE).read_text(encoding="utf-8"))
    shot_hash = acquisition.digest(shot_path.read_bytes())
    if shot_hash != split_manifest["shot_csv_sha256"]:
        raise EDAError("Shot CSV no longer matches the protected split source.")
    membership_path = root / holdout.MEMBERSHIP_RELATIVE
    membership_bytes = membership_path.read_bytes()
    membership_hash = acquisition.digest(membership_bytes)
    if membership_hash != split_manifest["membership_csv_sha256"]:
        raise EDAError("Split membership no longer matches its lock.")
    membership = pd.read_csv(membership_path, dtype={"match_id": "int64"})
    train_ids = set(membership.loc[membership["split"] == "train", "match_id"].astype(int))
    if not train_ids:
        raise EDAError("Locked split contains no training matches.")

    try:
        all_shots = pd.read_csv(shot_path, dtype={
            "event_id": "string", "match_date": "string", "home_team_id": "string", "away_team_id": "string",
            "team_id": "string", "opponent_id": "string", "player_id": "string",
        })
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise EDAError(f"Cannot parse shot CSV: {exc}") from exc
    required = [
        "match_id", "event_id", "is_goal", "statsbomb_xg", "x", "y", "distance_sb_units", "angle_sb_radians",
        "body_part", "shot_type", "technique", "play_pattern", "under_pressure", "first_time", "open_goal", "is_penalty",
    ]
    model_contract = json.loads((root / "docs/model_columns.json").read_text(encoding="utf-8"))
    allowed_features = [entry["column"] for entry in model_contract.get("features", [])]
    if allowed_features != ["distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique", "play_pattern", "under_pressure", "first_time", "open_goal"]:
        raise EDAError("Model feature contract differs from the approved Phase 4 allowlist.")
    missing_columns = [column for column in required if column not in all_shots.columns]
    if missing_columns:
        raise EDAError(f"Shot table is missing required columns: {missing_columns}.")
    # Filter immediately after parsing; no test outcome, xG, or feature relationships are calculated.
    training = select_training_rows(all_shots, train_ids)
    del all_shots
    if len(training) != split_manifest["train_shot_count"]:
        raise EDAError("Training shot count differs from the locked split.")
    if int(training["match_id"].nunique()) != split_manifest["train_match_count"]:
        raise EDAError("Training match count differs from the locked split.")
    if training.duplicated(["match_id", "event_id"]).any() or training["event_id"].duplicated().any():
        raise EDAError("Training shots contain duplicate event identifiers.")
    if training[required].isna().any().any():
        missing = training[required].columns[training[required].isna().any()].tolist()
        raise EDAError(f"Training rows have missing required values: {missing}.")
    if not training["is_goal"].isin([0, 1]).all():
        raise EDAError("Training goal labels must be 0/1.")
    for column in ["under_pressure", "first_time", "open_goal", "is_penalty"]:
        if not training[column].isin([0, 1]).all():
            raise EDAError(f"Training indicator {column} must be 0/1.")
    for column in ["x", "y", "distance_sb_units", "angle_sb_radians", "statsbomb_xg"]:
        if not pd.to_numeric(training[column], errors="coerce").map(math.isfinite).all():
            raise EDAError(f"Training numeric column {column} contains non-finite values.")
    if (training["statsbomb_xg"].lt(0) | training["statsbomb_xg"].gt(1)).any():
        raise EDAError("Training provider xG is outside [0,1].")
    if (training["angle_sb_radians"].lt(0) | training["angle_sb_radians"].gt(math.pi + 1e-12)).any():
        raise EDAError("Training shot angle is outside [0, pi].")
    if (training["distance_sb_units"] < 0).any():
        raise EDAError("Training distance must be non-negative.")

    summary = summarize_training(training)
    class_table = summary.pop("class_table")
    categorical_table = categorical_rates(training, ["body_part", "shot_type", "technique", "play_pattern"])
    context_table = context_rates(training, ["under_pressure", "first_time", "open_goal"])
    distance_edges = [0, 6, 12, 18, 24, 36, 80]
    distance_labels = ["0–6", "6–12", "12–18", "18–24", "24–36", "36–80"]
    angle_edges = [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi]
    angle_labels = ["0–π/6", "π/6–π/3", "π/3–π/2", "π/2–2π/3", "2π/3–5π/6", "5π/6–π"]
    distance_table = geometry_rates(training, "distance_sb_units", distance_edges, distance_labels)
    angle_table = geometry_rates(training, "angle_sb_radians", angle_edges, angle_labels)
    geometry_table = pd.concat([distance_table, angle_table], ignore_index=True)
    tables = {
        TABLE_NAMES[0]: class_table,
        TABLE_NAMES[1]: categorical_table,
        TABLE_NAMES[2]: context_table,
        TABLE_NAMES[3]: geometry_table,
    }
    figure_data = _make_figures(training, summary, categorical_table, context_table, geometry_table)
    table_data = {name: _csv_bytes(frame) for name, frame in tables.items()}

    output_manifest = {}
    for name, payload in {**figure_data, **table_data}.items():
        output_manifest[name] = {"bytes": len(payload), "sha256": acquisition.digest(payload)}
    source = shot_report.get("source", {})
    report = {
        "status": "passed", "scope": "training partition only",
        "source_commit": source.get("commit"), "shot_csv_sha256": shot_hash,
        "split_manifest_sha256": acquisition.digest((root / holdout.SPLIT_MANIFEST_RELATIVE).read_bytes()),
        "membership_csv_sha256": membership_hash,
        "seed": split_manifest["seed"], "algorithm": split_manifest["algorithm"],
        "training_match_ids": sorted(train_ids), "training_match_count": summary["match_count"],
        "training_shot_count": summary["shot_count"], "training_goal_count": summary["goal_count"],
        "training_non_goal_count": summary["non_goal_count"],
        "training_overall_goal_rate": summary["overall_goal_rate"],
        "training_no_goal_majority_baseline_accuracy": summary["no_goal_majority_baseline_accuracy"],
        "training_penalty_shot_count": summary["penalty_shot_count"],
        "training_penalty_goal_count": summary["penalty_goal_count"],
        "wilson_interval_note": "95% Wilson intervals use an independent-shot binomial approximation; within-match clustering can make them too narrow.",
        "findings": {
            "geometry_bins": json_records(geometry_table),
            "categories": json_records(categorical_table),
            "context_flags": json_records(context_table),
        },
        "outputs": output_manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    findings = render_findings(report)
    # Publish only after every validation, table, figure, and report has been built in memory.
    for name, payload in figure_data.items():
        acquisition.atomic_write(root / "reports/figures" / name, payload)
    for name, payload in table_data.items():
        acquisition.atomic_write(root / "reports/tables" / name, payload)
    acquisition.atomic_write(root / "docs/phase_05_eda_report.json", report_payload)
    acquisition.atomic_write(root / "docs/phase_05_eda_findings.md", findings.encode("utf-8"))
    return report


def render_findings(report: dict[str, Any]) -> str:
    findings = report["findings"]
    geometry = findings["geometry_bins"]
    distance = [row for row in geometry if row["dimension"] == "distance_sb_units" and row["shot_count"]]
    angle = [row for row in geometry if row["dimension"] == "angle_sb_radians" and row["shot_count"]]
    cats = findings["categories"]
    bodyparts = [row for row in cats if row["dimension"] == "body_part" and row["shot_count"] >= 30]
    shottypes = [row for row in cats if row["dimension"] == "shot_type" and row["shot_count"] >= 30]
    context = findings["context_flags"]
    # Tables retain the declared physical bin order; do not sort interval labels as text.
    low_distance = distance[0]
    high_distance = distance[-1]
    small_count = sum(row["small_group_n_lt_30"] for row in cats)
    return f"""# Phase 5 exploratory findings

Status: descriptive EDA complete. Every statistic, table, and figure below uses only the protected training partition. The held-out match labels, xG values, and feature relationships were not examined.

## Training-only summary

The training partition contains **{report['training_shot_count']:,} shots across {report['training_match_count']} matches**, with {report['training_goal_count']:,} goal-labelled shots and {report['training_non_goal_count']:,} non-goals. The overall goal rate is {report['training_overall_goal_rate']:.1%}. A rule that always predicts “non-goal” would attain {report['training_no_goal_majority_baseline_accuracy']:.1%} accuracy on these training rows. This is a descriptive class-imbalance baseline, not a trained model or a test result.

There are {report['training_penalty_shot_count']} in-match penalties, of which {report['training_penalty_goal_count']} are goal-labelled in the training partition. Penalties are a distinct shot type and should be reviewed separately in later probability evaluation.

## Geometry and shot context

In the training partition, the {low_distance['category']} distance band has a goal rate of {low_distance['goal_rate']:.1%} (n={low_distance['shot_count']:,}); the {high_distance['category']} band has {high_distance['goal_rate']:.1%} (n={high_distance['shot_count']:,}). Distance uses native StatsBomb coordinate units, not metres. Angle bins and category-specific counts and intervals are in the accompanying tables.

Observed differences across geometry, body part, shot type, technique, play pattern, and provider context flags are associations in this sample; they do not establish causal effects. The categorical table flags groups with fewer than 30 training shots ({small_count} in this sample). A rare-category rate can be unstable even when shown with a confidence interval.

Context values marked NOT FLAGGED mean StatsBomb did not record the positive flag. They do not establish that pressure, a first-time attempt, or an open goal was objectively absent.

## Limits and next step

The 95% Wilson intervals use a binomial approximation that treats shots as independent. Shots from the same match can be related, so these intervals may be too narrow. The no-goal baseline is only a reference for class imbalance; it is not a fitted classifier. No preprocessing, model fitting, cross-validation, or test evaluation was done in this phase.

Phase 6 should reuse the locked match-level membership in `docs/split_manifest.json`. Keep the test partition sealed until model design and training-only evaluation decisions are complete. See `reports/figures/phase05_*.png`, `reports/tables/phase05_*.csv`, and `docs/phase_05_eda_report.json` for the training-only visuals, numeric aggregates, and source/output fingerprints.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = run(args.root)
    except (EDAError, holdout.SplitError, acquisition.AcquisitionError, OSError, ValueError, KeyError) as exc:
        print(f"Exploratory analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "training_match_count", "training_shot_count", "training_goal_count",
        "training_no_goal_majority_baseline_accuracy", "training_penalty_shot_count", "training_penalty_goal_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
