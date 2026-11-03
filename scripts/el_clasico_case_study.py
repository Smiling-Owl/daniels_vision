"""Revisit two predeclared El Clásico matches using saved development OOF predictions."""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts import acquire_data as acquisition
    from scripts import build_shot_dataset as phase4
    from scripts import calibrate_evaluate as phase8
    from scripts import prepare_modeling as preparation
    from scripts import simulate_matches as phase9
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import build_shot_dataset as phase4
    import calibrate_evaluate as phase8
    import prepare_modeling as preparation
    import simulate_matches as phase9

PROTOCOL_RELATIVE = "docs/el_clasico_protocol.json"
REPORT_RELATIVE = "docs/el_clasico_report.json"
TABLE_DIRECTORY = "reports/tables"
FIGURE_DIRECTORY = "reports/figures"
OOF_VARIANT = "logistic_regression_sigmoid"
TRIALS = 100_000
CHUNK_SIZE = 10_000
SEED = 42
CASE_MATCHES = {
    266424: {"date": "2015-11-21", "home": "Real Madrid", "away": "Barcelona", "official_home_goals": 0, "official_away_goals": 4},
    267533: {"date": "2016-04-02", "home": "Barcelona", "away": "Real Madrid", "official_home_goals": 1, "official_away_goals": 2},
}
TABLE_NAMES = {
    "match_summary": "el_clasico_match_summary.csv",
    "scorelines": "el_clasico_scorelines.csv",
    "shots": "el_clasico_shots.csv",
}
FIGURE_NAMES = {
    "match_266424": "el_clasico_266424_outcomes.png",
    "match_267533": "el_clasico_267533_outcomes.png",
    "scorelines": "el_clasico_joint_scorelines.png",
}


class CaseStudyError(RuntimeError):
    """Frozen OOF, source, or case-study invariant failed."""


def select_oof_predictions(oof: pd.DataFrame, candidate: str, train_match_ids: set[int], test_match_ids: set[int]) -> pd.DataFrame:
    """Return only the preselected case rows from the hashed Phase 8 OOF table."""
    if candidate != OOF_VARIANT:
        raise CaseStudyError(f"El Clásico cases require the frozen OOF variant {OOF_VARIANT}; got {candidate!r}.")
    required = {"event_id", "match_id", "fold", "variant", "y_true", "p_goal"}
    if not required.issubset(oof.columns):
        raise CaseStudyError("Case-study predictions must include event, match, fold, variant, label, and probability columns.")
    selected = oof.loc[oof["variant"] == candidate].copy()
    if selected.empty or selected["event_id"].isna().any() or selected["event_id"].astype(str).duplicated().any():
        raise CaseStudyError("Selected OOF predictions must have unique non-missing event IDs.")
    if not set(CASE_MATCHES).issubset(train_match_ids) or set(CASE_MATCHES) & test_match_ids:
        raise CaseStudyError("Both predeclared case matches must be training matches and outside the test partition.")
    try:
        probabilities = selected["p_goal"].to_numpy(dtype=float)
        labels = selected["y_true"].to_numpy(dtype=float)
        folds = selected["fold"].to_numpy(dtype=float)
        match_ids = selected["match_id"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise CaseStudyError("OOF labels, folds, match IDs, and probabilities must be numeric where required.") from exc
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise CaseStudyError("Selected OOF probabilities must be finite and in [0,1].")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise CaseStudyError("Selected OOF labels must be finite binary integers.")
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all() or not np.isin(folds, np.arange(1, 11)).all():
        raise CaseStudyError("Selected OOF fold IDs must be integers from the locked ten-fold CV map.")
    if not np.isfinite(match_ids).all() or not np.equal(match_ids, np.floor(match_ids)).all():
        raise CaseStudyError("Selected OOF match IDs must be finite integers.")
    if set(selected["match_id"].astype(int)) - train_match_ids or set(selected["match_id"].astype(int)) & test_match_ids:
        raise CaseStudyError("Selected OOF rows include a non-training or held-out test match.")
    case_rows = selected.loc[selected["match_id"].astype(int).isin(CASE_MATCHES)].copy()
    if set(case_rows["match_id"].astype(int)) != set(CASE_MATCHES):
        raise CaseStudyError("Selected OOF predictions do not cover both predeclared case matches.")
    return case_rows


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseStudyError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaseStudyError(f"Expected a JSON object for {label}.")
    return value, payload


def _verify_file(root: Path, relative: str, entry: dict[str, Any], label: str) -> bytes:
    try:
        payload = (root / relative).read_bytes()
    except OSError as exc:
        raise CaseStudyError(f"Cannot read {label} {relative}: {exc}") from exc
    if len(payload) != entry.get("bytes") or acquisition.digest(payload) != entry.get("sha256"):
        raise CaseStudyError(f"{label} hash or byte count mismatch: {relative}.")
    return payload


def _protocol_data(root: Path) -> dict[str, Any]:
    phase8_inputs = phase8.verify_phase7(root)
    calibration_protocol, calibration_protocol_bytes = phase8.make_protocol(phase8_inputs)
    selection, selection_bytes, locked = phase8._verify_frozen_selection(
        root, phase8_inputs, calibration_protocol, calibration_protocol_bytes,
    )
    train_ids = set(phase8_inputs["split"]["train_match_ids"])
    test_ids = set(phase8_inputs["split"]["test_match_ids"])

    dev_report, dev_report_bytes = _read_json(root / phase8.DEV_REPORT_RELATIVE, "Phase 8 development report")
    if dev_report.get("status") != "passed" or dev_report.get("test_data_accessed") is not False:
        raise CaseStudyError("Phase 8 development report must certify training-only OOF predictions.")
    oof_relative = f"{phase8.TABLE_DIRECTORY}/{phase8.TABLE_NAMES['development_predictions']}"
    oof_entry = dev_report.get("outputs", {}).get(oof_relative)
    if not isinstance(oof_entry, dict):
        raise CaseStudyError("Phase 8 development report does not hash the OOF prediction table.")
    for relative, entry in dev_report.get("outputs", {}).items():
        _verify_file(root, relative, entry, "Phase 8 development output")
    oof_payload = _verify_file(root, oof_relative, oof_entry, "Phase 8 OOF predictions")
    oof = pd.read_csv(io.BytesIO(oof_payload), float_precision="round_trip")
    if not {"event_id", "match_id", "fold", "variant", "y_true", "p_goal"}.issubset(oof.columns):
        raise CaseStudyError("Saved Phase 8 OOF table is missing required columns.")
    case_oof = select_oof_predictions(oof, selection.get("global_candidate"), train_ids, test_ids)

    phase4_report, phase4_report_bytes = _read_json(root / "docs/shot_dataset_report.json", "Phase 4 source report")
    if phase4_report.get("status") != "passed":
        raise CaseStudyError("Phase 4 source report is not marked passed.")
    source = phase4_report.get("source", {})
    if source.get("commit") != phase4.SOURCE["commit"] or source.get("competition_id") != 11 or source.get("season_id") != 27:
        raise CaseStudyError("Phase 4 report does not identify the pinned La Liga 2015/16 snapshot.")
    phase4_files = {
        "shots": "data/processed/shots.csv",
        "reconciliation": "data/processed/goal_reconciliation.csv",
        "own_goals": "data/processed/own_goals.csv",
    }
    source_payloads = {}
    for key, relative in phase4_files.items():
        entry = phase4_report.get("outputs", {}).get(relative)
        if not isinstance(entry, dict):
            raise CaseStudyError(f"Phase 4 report is missing the source hash for {relative}.")
        source_payloads[key] = _verify_file(root, relative, entry, "Phase 4 source")

    cv_membership = preparation.load_cv_membership(root)
    fold_by_match = cv_membership.loc[cv_membership["match_id"].astype(int).isin(CASE_MATCHES)].groupby("match_id")["validation_fold"].nunique()
    if len(fold_by_match) != len(CASE_MATCHES) or not fold_by_match.eq(1).all():
        raise CaseStudyError("Every predeclared case match must appear in exactly one locked outer validation fold.")
    fold_join = case_oof.merge(
        cv_membership[["event_id", "match_id", "validation_fold"]].rename(columns={"validation_fold": "locked_fold"}),
        on="event_id", how="left", validate="one_to_one", suffixes=("", "_cv"),
    )
    if fold_join["locked_fold"].isna().any() or not fold_join["fold"].astype(int).eq(fold_join["locked_fold"].astype(int)).all():
        raise CaseStudyError("OOF prediction fold IDs differ from the locked Phase 6 CV membership.")
    if not fold_join["match_id"].astype(int).eq(fold_join["match_id_cv"].astype(int)).all():
        raise CaseStudyError("OOF prediction match IDs differ from the locked Phase 6 CV membership.")

    return {
        "phase8_inputs": phase8_inputs, "selection": selection, "selection_bytes": selection_bytes,
        "locked_selection": locked, "dev_report": dev_report, "dev_report_bytes": dev_report_bytes,
        "oof_relative": oof_relative, "oof_entry": oof_entry, "oof_payload": oof_payload,
        "case_oof": case_oof, "cv_membership": cv_membership,
        "phase4_report": phase4_report, "phase4_report_bytes": phase4_report_bytes,
        "phase4_files": phase4_files, "phase4_payloads": source_payloads,
    }


def join_case_shots(shots: pd.DataFrame, selected_oof: pd.DataFrame, reconciliation: pd.DataFrame, cv_membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join only the two predeclared training matches to frozen OOF rows by event ID."""
    try:
        joined, match_info = phase9.join_test_shots(shots, selected_oof, reconciliation, set(CASE_MATCHES))
    except (phase9.SimulationError, KeyError, ValueError) as exc:
        raise CaseStudyError(f"Cannot safely join El Clásico OOF rows and source match data: {exc}") from exc
    # Phase 9 deliberately returns only probability/label values from its
    # generic join. Reattach the frozen fold by event ID, never row position.
    prediction_folds = selected_oof[["event_id", "fold"]].copy()
    prediction_folds["event_id"] = prediction_folds["event_id"].astype(str)
    joined["event_id"] = joined["event_id"].astype(str)
    joined = joined.merge(prediction_folds, on="event_id", how="left", validate="one_to_one")
    if joined["fold"].isna().any():
        raise CaseStudyError("A case shot has no matching saved OOF fold by event ID.")
    fold_map = cv_membership[["event_id", "match_id", "validation_fold"]].copy()
    fold_map["event_id"] = fold_map["event_id"].astype(str)
    joined["event_id"] = joined["event_id"].astype(str)
    joined = joined.merge(fold_map, on="event_id", how="left", validate="one_to_one", suffixes=("", "_cv"))
    if joined["validation_fold"].isna().any() or not joined["match_id"].astype(int).eq(joined["match_id_cv"].astype(int)).all():
        raise CaseStudyError("A case shot does not match its locked Phase 6 event-to-match fold assignment.")
    if not joined["fold"].astype(int).eq(joined["validation_fold"].astype(int)).all():
        raise CaseStudyError("A case OOF prediction's fold differs from the locked Phase 6 assignment.")
    joined = joined.drop(columns=["match_id_cv"])
    folds = joined.groupby("match_id")["validation_fold"].nunique()
    if len(folds) != 2 or not folds.eq(1).all():
        raise CaseStudyError("Each case-study match must have one outer OOF fold across all of its shots.")
    for match_id, expected in CASE_MATCHES.items():
        row = match_info.loc[match_info["match_id"].astype(int) == match_id].iloc[0]
        actual = (str(row["match_date"])[:10], str(row["home_team_name"]), str(row["away_team_name"]), int(row["official_home_goals"]), int(row["official_away_goals"]))
        wanted = (expected["date"], expected["home"], expected["away"], expected["official_home_goals"], expected["official_away_goals"])
        if actual != wanted:
            raise CaseStudyError(f"Predeclared case metadata does not match source for match {match_id}: {actual!r}.")
    return joined, match_info


def _case_outputs(joined: pd.DataFrame, match_info: pd.DataFrame, own_goals: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, bytes]]:
    summary_records, scoreline_records = [], []
    figures: dict[str, bytes] = {}
    for match_id in sorted(CASE_MATCHES):
        metadata = match_info.loc[match_info["match_id"].astype(int) == match_id].iloc[0]
        rows = joined.loc[joined["match_id"].astype(int) == match_id].sort_values(["event_index", "event_id"], kind="stable")
        home = rows.loc[rows["is_home"].astype(int) == 1]
        away = rows.loc[rows["is_home"].astype(int) == 0]
        home_p, away_p = phase9._binary_probabilities(home["p_goal"]), phase9._binary_probabilities(away["p_goal"])
        home_pmf, away_pmf = phase9.exact_poisson_binomial(home_p), phase9.exact_poisson_binomial(away_p)
        joint, exact_outcomes = phase9.exact_joint_probabilities(home_pmf, away_pmf)
        home_scores = phase9.simulate_goals(home_p, TRIALS, match_id, 0, seed=SEED, chunk_size=CHUNK_SIZE)
        away_scores = phase9.simulate_goals(away_p, TRIALS, match_id, 1, seed=SEED, chunk_size=CHUNK_SIZE)
        checks = phase9._check_sampling(home_scores, away_scores, home_p, away_p, exact_outcomes)
        mc_home_wins = int(np.count_nonzero(home_scores > away_scores))
        mc_draws = int(np.count_nonzero(home_scores == away_scores))
        mc_away_wins = int(np.count_nonzero(home_scores < away_scores))
        score_counts = np.bincount(home_scores * len(away_pmf) + away_scores, minlength=joint.size).reshape(joint.shape)
        if int(score_counts.sum()) != TRIALS:
            raise CaseStudyError(f"Monte Carlo scoreline counts do not sum to {TRIALS} for match {match_id}.")
        for h in range(joint.shape[0]):
            for a in range(joint.shape[1]):
                count = int(score_counts[h, a])
                scoreline_records.append({
                    "match_id": match_id, "home_goals": h, "away_goals": a,
                    "exact_probability": float(joint[h, a]), "mc_count": count, "mc_probability": count / TRIALS,
                })
        own_match = own_goals.loc[own_goals["match_id"].astype(int) == match_id]
        home_points, away_points = phase9._expected_points(exact_outcomes)
        summary_records.append({
            "match_id": match_id, "match_date": str(metadata["match_date"]),
            "home_team_id": int(metadata["home_team_id"]), "home_team_name": str(metadata["home_team_name"]),
            "away_team_id": int(metadata["away_team_id"]), "away_team_name": str(metadata["away_team_name"]),
            "official_home_goals": int(metadata["official_home_goals"]), "official_away_goals": int(metadata["official_away_goals"]),
            "observed_shot_goals_home": int(metadata["shot_goals_home"]), "observed_shot_goals_away": int(metadata["shot_goals_away"]),
            "own_goals_credited_home": int(metadata["own_goals_credited_home"]), "own_goals_credited_away": int(metadata["own_goals_credited_away"]),
            "own_goal_against_events": int(len(own_match)), "outer_oof_fold": int(rows["validation_fold"].iloc[0]),
            "home_shot_count": int(len(home)), "away_shot_count": int(len(away)),
            "home_sum_goal_probability": float(home_p.sum()), "away_sum_goal_probability": float(away_p.sum()),
            "home_expected_shot_goals": float(home_p.sum()), "away_expected_shot_goals": float(away_p.sum()),
            "home_mc_mean_goals": checks["home_mean"], "away_mc_mean_goals": checks["away_mean"],
            "home_mean_mcse": checks["home_mean_mcse"], "away_mean_mcse": checks["away_mean_mcse"],
            "exact_home_win_probability": exact_outcomes["home_win"], "exact_draw_probability": exact_outcomes["draw"], "exact_away_win_probability": exact_outcomes["away_win"],
            "mc_home_win_count": mc_home_wins, "mc_draw_count": mc_draws, "mc_away_win_count": mc_away_wins,
            "mc_home_win_probability": checks["home_win_probability"], "mc_draw_probability": checks["draw_probability"], "mc_away_win_probability": checks["away_win_probability"],
            "home_expected_points_exact": home_points, "away_expected_points_exact": away_points,
            "trials": TRIALS,
        })
        figures.update(_outcome_figure(metadata, exact_outcomes, home_scores, away_scores))

    figures.update(_joint_figure(match_info, scoreline_records))
    shots_table = joined[["match_id", "match_date", "event_id", "event_index", "team_id", "team_name", "is_home", "is_goal", "p_goal", "fold", "validation_fold"]].copy()
    if not shots_table["fold"].astype(int).eq(shots_table["validation_fold"].astype(int)).all():
        raise CaseStudyError("Saved OOF prediction fold differs from the locked event fold.")
    tables = {
        TABLE_NAMES["match_summary"]: pd.DataFrame(summary_records),
        TABLE_NAMES["scorelines"]: pd.DataFrame(scoreline_records),
        TABLE_NAMES["shots"]: shots_table,
    }
    return tables, figures


def _outcome_figure(metadata: pd.Series, exact: dict[str, float], home_scores: np.ndarray, away_scores: np.ndarray) -> dict[str, bytes]:
    outcomes = ["home_win", "draw", "away_win"]
    labels = ["Home win", "Draw", "Away win"]
    mc = [float(np.mean(home_scores > away_scores)), float(np.mean(home_scores == away_scores)), float(np.mean(home_scores < away_scores))]
    exact_values = [exact[name] for name in outcomes]
    x = np.arange(3); width = 0.36
    fig, axis = plt.subplots(figsize=(7.2, 4.7), facecolor="white")
    axis.bar(x - width / 2, exact_values, width, label="Exact Poisson-binomial", color="#315A8A")
    axis.bar(x + width / 2, mc, width, label=f"Monte Carlo ({TRIALS:,})", color="#D88735")
    axis.set_xticks(x, labels=labels); axis.set_ylim(0, 1); axis.set_ylabel("Probability")
    axis.set_title(f"DEVELOPMENT OOF CASE STUDY\n{str(metadata['match_date'])[:10]} | {metadata['home_team_name']} vs {metadata['away_team_name']}", color="#23364D")
    axis.grid(axis="y", alpha=0.2); axis.legend()
    fig.text(0.5, 0.015, "StatsBomb La Liga 2015/16. Conditional on observed shots, not a pre-match forecast.\nOOF excludes this match from fitting, but this candidate was selected using development results.", ha="center", fontsize=7.5, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.9])
    buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"}); plt.close(fig)
    figure_key = f"match_{int(metadata['match_id'])}"
    if figure_key not in FIGURE_NAMES:
        raise CaseStudyError(f"No fixed outcome figure name is registered for match {figure_key}.")
    return {f"{FIGURE_DIRECTORY}/{FIGURE_NAMES[figure_key]}": buffer.getvalue()}


def _joint_figure(match_info: pd.DataFrame, scoreline_records: list[dict[str, Any]]) -> dict[str, bytes]:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), facecolor="white")
    matrices = []
    for axis, match_id in zip(axes, sorted(CASE_MATCHES), strict=True):
        metadata = match_info.loc[match_info["match_id"].astype(int) == match_id].iloc[0]
        rows = [row for row in scoreline_records if row["match_id"] == match_id]
        max_home = max(row["home_goals"] for row in rows)
        max_away = max(row["away_goals"] for row in rows)
        matrix = np.zeros((max_home + 1, max_away + 1), dtype=float)
        for row in rows:
            matrix[row["home_goals"], row["away_goals"]] = row["exact_probability"]
        matrices.append(matrix)
    max_mass = max(float(matrix.max()) for matrix in matrices)
    colorbar_image = None
    for axis, match_id, matrix in zip(axes, sorted(CASE_MATCHES), matrices, strict=True):
        colorbar_image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=max_mass)
        metadata = match_info.loc[match_info["match_id"].astype(int) == match_id].iloc[0]
        axis.set_title(f"{str(metadata['match_date'])[:10]}\n{metadata['home_team_name']} vs {metadata['away_team_name']}", color="#23364D")
        axis.set_xlabel("Away goals"); axis.set_ylabel("Home goals")
        axis.set_xticks(np.arange(matrix.shape[1])); axis.set_yticks(np.arange(matrix.shape[0]))
    fig.suptitle("DEVELOPMENT OOF CASE STUDY — complete conditional scoreline support", color="#23364D")
    fig.text(0.5, 0.012, "StatsBomb La Liga 2015/16. Conditional on observed shots, not a pre-match forecast.\nThese matches are development OOF cases; model selection used development results. Own goals are context only, not simulated.", ha="center", fontsize=8, color="#23364D")
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.17, top=0.82, wspace=0.35)
    colorbar_axis = fig.add_axes([0.9, 0.20, 0.015, 0.58])
    fig.colorbar(colorbar_image, cax=colorbar_axis, label="Exact joint probability")
    buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"}); plt.close(fig)
    return {f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['scorelines']}": buffer.getvalue()}


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaseStudyError(f"Cannot serialize case-study artifact: {exc}") from exc


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _lock_protocol(root: Path, protocol: dict[str, Any], payload: bytes) -> None:
    path = root / PROTOCOL_RELATIVE
    if path.exists() and path.read_bytes() != payload:
        raise CaseStudyError("Existing El Clásico case-study protocol differs; refusing to overwrite.")
    if not path.exists():
        acquisition.atomic_write(path, payload)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inputs = _protocol_data(root)
    shots = pd.read_csv(root / "data/processed/shots.csv", dtype={"event_id": str}, float_precision="round_trip")
    reconciliation = pd.read_csv(root / "data/processed/goal_reconciliation.csv", float_precision="round_trip")
    own_goals = pd.read_csv(root / "data/processed/own_goals.csv", float_precision="round_trip")
    joined, match_info = join_case_shots(shots, inputs["case_oof"], reconciliation, inputs["cv_membership"])
    own_goals = own_goals.loc[own_goals["match_id"].astype(int).isin(CASE_MATCHES)].copy()
    try:
        phase9.validate_own_goal_credits(own_goals, match_info)
    except phase9.SimulationError as exc:
        raise CaseStudyError(str(exc)) from exc

    protocol = {
        "status": "predeclared_el_clasico_development_oof_case_study",
        "scope": "two user-requested 2015/16 El Clásico matches from training development OOF only",
        "cases": {str(match_id): metadata for match_id, metadata in CASE_MATCHES.items()},
        "phase8_selected_variant": OOF_VARIANT,
        "selection_sha256": acquisition.digest(inputs["selection_bytes"]),
        "development_report_sha256": acquisition.digest(inputs["dev_report_bytes"]),
        "oof_predictions_sha256": inputs["oof_entry"]["sha256"],
        "cv_membership_sha256": acquisition.digest((root / preparation.CV_MEMBERSHIP_RELATIVE).read_bytes()),
        "cv_manifest_sha256": acquisition.digest((root / preparation.CV_MANIFEST_RELATIVE).read_bytes()),
        "phase4_report_sha256": acquisition.digest(inputs["phase4_report_bytes"]),
        "phase4_sources": {key: {"path": relative, "sha256": acquisition.digest(inputs["phase4_payloads"][key])} for key, relative in inputs["phase4_files"].items()},
        "trial_count_per_match": TRIALS, "chunk_size": CHUNK_SIZE, "seed": SEED,
        "rng_algorithm": "numpy.default_rng(SeedSequence([seed, match_id, side]))",
        "team_side_codes": {"home": 0, "away": 1},
        "probability_source": "phase08_development_predictions.csv OOF rows for the already-frozen global candidate; no resubstitution rows or final full-training model predictions.",
        "interpretation": "Development OOF case study: the outer validation match is excluded from that fold's fitting/calibration, but the variant was selected using overall development results. Not an independent test evaluation.",
        "own_goal_policy": "Own goals are retained as reconciliation context only; no own goals are sampled or added to shot-generated scorelines.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    protocol_payload = _json_bytes(protocol)
    _lock_protocol(root, protocol, protocol_payload)
    tables, figures = _case_outputs(joined, match_info, own_goals)
    outputs = {f"{TABLE_DIRECTORY}/{name}": _csv_bytes(frame) for name, frame in tables.items()}
    outputs.update(figures)
    output_entries = {relative: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for relative, payload in outputs.items()}
    summary = tables[TABLE_NAMES["match_summary"]]
    report = {
        "status": "passed", "scope": "DEVELOPMENT OOF CASE STUDY; conditional observed-shot simulations, not test evaluation",
        "candidate": OOF_VARIANT,
        "protocol_sha256": acquisition.digest(protocol_payload),
        "phase8_selection_sha256": protocol["selection_sha256"], "oof_predictions_sha256": protocol["oof_predictions_sha256"],
        "cases": summary.to_dict(orient="records"),
        "match_count": len(summary), "shot_count": len(joined), "trial_count_per_match": TRIALS,
        "checks": {
            "both_cases_are_training_matches": True,
            "both_cases_are_outside_test_partition": True,
            "one_outer_oof_fold_per_case_match": bool(summary["outer_oof_fold"].notna().all()),
            "exact_probabilities_sum_to_one": bool(np.allclose(summary[["exact_home_win_probability", "exact_draw_probability", "exact_away_win_probability"]].sum(axis=1), 1, atol=1e-12, rtol=0)),
            "mc_outcome_counts_sum_to_trials": bool(summary[["mc_home_win_count", "mc_draw_count", "mc_away_win_count"]].sum(axis=1).eq(TRIALS).all()),
            "own_goals_not_simulated": True,
        },
        "limitations": [
            protocol["interpretation"],
            "Conditional on recorded shots and fixed OOF probabilities; not a pre-match forecast or causal estimate of points deserved.",
            "Monte Carlo error is numerical sampling variation and excludes model/probability uncertainty.",
            "Shots may be dependent; the simulator holds the observed attempt set fixed and assumes independent Bernoulli outcomes.",
        ],
        "outputs": output_entries,
    }
    report_payload = _json_bytes(report)
    for relative, payload in outputs.items():
        acquisition.atomic_write(root / relative, payload)
    acquisition.atomic_write(root / REPORT_RELATIVE, report_payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.root)
    except (CaseStudyError, phase8.CalibrationError, phase8.holdout.SplitError, preparation.PreparationError, phase9.SimulationError, OSError, ValueError, KeyError) as exc:
        print(f"El Clásico case study failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "match_count": report["match_count"], "shot_count": report["shot_count"], "trials_per_match": report["trial_count_per_match"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
