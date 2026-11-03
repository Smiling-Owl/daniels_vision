"""Simulate conditional scorelines from frozen held-out shot probabilities."""
from __future__ import annotations

import argparse
import io
import json
import math
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
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import build_shot_dataset as phase4
    import calibrate_evaluate as phase8
    import prepare_modeling as preparation

PROTOCOL_RELATIVE = "docs/phase_09_protocol.json"
REPORT_RELATIVE = "docs/phase_09_simulation_report.json"
TABLE_DIRECTORY = "reports/tables"
FIGURE_DIRECTORY = "reports/figures"
TRIALS = 100_000
CHUNK_SIZE = 10_000
SEED = 42
EXAMPLE_CHECKPOINTS = (100, 1_000, 10_000, 100_000)
EXPECTED_CANDIDATE = "logistic_regression_sigmoid"
TABLE_NAMES = {
    "match_summary": "phase09_match_summary.csv",
    "scoreline_distribution": "phase09_scoreline_distribution.csv",
    "convergence": "phase09_convergence.csv",
    "example_shots": "phase09_example_shots.csv",
}
FIGURE_NAMES = {
    "example_scoreline": "phase09_example_scoreline_heatmap.png",
    "example_outcomes": "phase09_example_outcomes.png",
    "convergence": "phase09_example_convergence.png",
}


class SimulationError(RuntimeError):
    """A frozen input or simulation invariant failed."""


def read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulationError(f"Cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimulationError(f"Expected a JSON object for {description}.")
    return value, payload


def verify_hashed_file(root: Path, relative: str, entry: dict[str, Any], label: str) -> bytes:
    try:
        payload = (root / relative).read_bytes()
    except OSError as exc:
        raise SimulationError(f"Cannot read {label} {relative}: {exc}") from exc
    if len(payload) != entry.get("bytes") or acquisition.digest(payload) != entry.get("sha256"):
        raise SimulationError(f"{label} hash or byte count mismatch: {relative}.")
    return payload


def verify_phase_inputs(root: Path) -> dict[str, Any]:
    """Verify Phase 8 frozen test predictions and Phase 4 source tables before joining."""
    phase8_inputs = phase8.verify_phase7(root)
    calibration_protocol, calibration_protocol_bytes = phase8.make_protocol(phase8_inputs)
    selection, selection_bytes, locked = phase8._verify_frozen_selection(
        root, phase8_inputs, calibration_protocol, calibration_protocol_bytes,
    )
    test_report, test_report_bytes = read_json(root / phase8.REPORT_RELATIVE, "Phase 8 test report")
    if test_report.get("status") != "passed":
        raise SimulationError("Phase 8 test report is not marked passed.")
    if test_report.get("selection_sha256") != acquisition.digest(selection_bytes):
        raise SimulationError("Phase 8 test report does not match the frozen Phase 8 selection.")
    if test_report.get("protocol_sha256") != acquisition.digest(calibration_protocol_bytes):
        raise SimulationError("Phase 8 test report does not match its calibration protocol.")
    if test_report.get("source_shot_csv_sha256") != calibration_protocol.get("source_shot_csv_sha256"):
        raise SimulationError("Phase 8 test report refers to a different shot table.")
    if test_report.get("test_match_count") != len(phase8_inputs["split"]["test_match_ids"]):
        raise SimulationError("Phase 8 test report match count differs from the locked holdout.")
    if test_report.get("test_shot_count") != phase8_inputs["split"]["test_shot_count"]:
        raise SimulationError("Phase 8 test report shot count differs from the locked holdout.")
    for relative, entry in test_report.get("outputs", {}).items():
        verify_hashed_file(root, relative, entry, "Phase 8 test output")
    if selection.get("global_candidate") != EXPECTED_CANDIDATE:
        raise SimulationError(
            f"The frozen Phase 8 global candidate is {selection.get('global_candidate')!r}; "
            f"Phase 9 expects {EXPECTED_CANDIDATE!r}."
        )

    phase4_report, phase4_report_bytes = read_json(root / "docs/shot_dataset_report.json", "Phase 4 shot-dataset report")
    if phase4_report.get("status") != "passed":
        raise SimulationError("Phase 4 shot-dataset report is not marked passed.")
    source = phase4_report.get("source", {})
    if source.get("commit") != phase4.SOURCE["commit"] or source.get("competition_id") != 11 or source.get("season_id") != 27:
        raise SimulationError("Phase 4 dataset does not identify the pinned La Liga 2015/16 source snapshot.")
    phase4_files = {
        "shots": "data/processed/shots.csv",
        "reconciliation": "data/processed/goal_reconciliation.csv",
        "own_goals": "data/processed/own_goals.csv",
    }
    verified_phase4: dict[str, bytes] = {}
    for key, relative in phase4_files.items():
        entry = phase4_report.get("outputs", {}).get(relative)
        if not isinstance(entry, dict):
            raise SimulationError(f"Phase 4 report does not hash the required input {relative}.")
        verified_phase4[key] = verify_hashed_file(root, relative, entry, "Phase 4 source")

    prediction_relative = f"{phase8.TABLE_DIRECTORY}/{phase8.TABLE_NAMES['test_predictions']}"
    prediction_entry = test_report.get("outputs", {}).get(prediction_relative)
    if not isinstance(prediction_entry, dict):
        raise SimulationError("Phase 8 test report does not hash the test prediction table.")
    predictions_payload = verify_hashed_file(root, prediction_relative, prediction_entry, "Phase 8 test predictions")
    predictions = pd.read_csv(io.BytesIO(predictions_payload), float_precision="round_trip")
    if not {"variant", "event_id", "match_id", "y_true", "p_goal"}.issubset(predictions.columns):
        raise SimulationError("Phase 8 test prediction table is missing required columns.")
    candidate = predictions.loc[predictions["variant"] == EXPECTED_CANDIDATE].copy()
    if candidate.empty or candidate["event_id"].isna().any() or candidate["event_id"].astype(str).duplicated().any():
        raise SimulationError("Frozen-candidate test predictions must have unique, non-missing event IDs.")
    try:
        probabilities = candidate["p_goal"].to_numpy(dtype=float)
        labels = candidate["y_true"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimulationError("Frozen-candidate labels or probabilities are not numeric.") from exc
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise SimulationError("Frozen-candidate probabilities must be finite and inside [0,1].")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise SimulationError("Frozen-candidate labels must be finite binary integers.")
    if len(candidate) != phase8_inputs["split"]["test_shot_count"]:
        raise SimulationError("Frozen-candidate predictions do not cover the locked test shot count.")
    if set(candidate["match_id"].astype(int)) != set(phase8_inputs["split"]["test_match_ids"]):
        raise SimulationError("Frozen-candidate predictions do not cover exactly the locked test matches.")
    stored_metrics = [row for row in test_report.get("test_metrics", []) if row.get("variant") == EXPECTED_CANDIDATE]
    if len(stored_metrics) != 1:
        raise SimulationError("Phase 8 report must contain exactly one metric row for the frozen candidate.")
    recomputed = phase8.phase7.metrics_for(labels, probabilities, threshold=phase8.THRESHOLD)
    for name, value in recomputed.items():
        stored = stored_metrics[0].get(name)
        if isinstance(value, (int, np.integer)):
            if stored != int(value):
                raise SimulationError(f"Phase 8 candidate test metric {name} does not reproduce from saved probabilities.")
        elif not isinstance(stored, (int, float)) or not np.isclose(float(stored), float(value), rtol=0, atol=1e-12):
            raise SimulationError(f"Phase 8 candidate test metric {name} does not reproduce from saved probabilities.")
    return {
        "phase8_inputs": phase8_inputs, "selection": selection, "selection_bytes": selection_bytes,
        "phase8_model_manifest": locked["model_manifest"],
        "test_report": test_report, "test_report_bytes": test_report_bytes,
        "phase4_report": phase4_report, "phase4_report_bytes": phase4_report_bytes,
        "phase4_files": phase4_files, "phase4_payloads": verified_phase4,
        "prediction_relative": prediction_relative, "prediction_entry": prediction_entry,
        "predictions": candidate,
    }


def _binary_probabilities(values: Any) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimulationError("Bernoulli probabilities must be a numeric sequence.") from exc
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise SimulationError("Bernoulli probabilities must be one-dimensional finite values in [0,1].")
    return probabilities


def exact_poisson_binomial(probabilities: Any) -> np.ndarray:
    """Return the exact PMF of a sum of independent Bernoulli shot outcomes."""
    pmf = np.array([1.0], dtype=float)
    for probability in _binary_probabilities(probabilities):
        pmf = np.convolve(pmf, np.array([1.0 - probability, probability]))
    if (pmf < 0).any():
        raise SimulationError("Exact Poisson-binomial convolution produced invalid negative probability mass.")
    total = float(pmf.sum())
    if not math.isfinite(total) or not np.isclose(total, 1.0, rtol=0, atol=1e-12):
        raise SimulationError("Exact Poisson-binomial mass does not sum to one.")
    return pmf / total


def exact_joint_probabilities(home_pmf: Any, away_pmf: Any) -> tuple[np.ndarray, dict[str, float]]:
    home, away = np.asarray(home_pmf, dtype=float), np.asarray(away_pmf, dtype=float)
    if home.ndim != 1 or away.ndim != 1 or not len(home) or not len(away):
        raise SimulationError("Each team PMF must be a non-empty one-dimensional array.")
    if not np.isfinite(home).all() or not np.isfinite(away).all() or (home < 0).any() or (away < 0).any():
        raise SimulationError("Team PMFs must contain finite non-negative mass.")
    if not np.isclose(home.sum(), 1.0, atol=1e-12, rtol=0) or not np.isclose(away.sum(), 1.0, atol=1e-12, rtol=0):
        raise SimulationError("Team PMFs must each sum to one.")
    joint = np.outer(home, away)
    home_scores, away_scores = np.indices(joint.shape)
    outcomes = {
        "home_win": float(joint[home_scores > away_scores].sum()),
        "draw": float(joint[home_scores == away_scores].sum()),
        "away_win": float(joint[home_scores < away_scores].sum()),
    }
    if not np.isclose(sum(outcomes.values()), 1.0, atol=1e-12, rtol=0):
        raise SimulationError("Exact home/draw/away probabilities do not sum to one.")
    return joint, outcomes


def _rng_for(match_id: int, side: int, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(match_id), int(side)]))


def simulate_goals(probabilities: Any, trials: int, match_id: int, side: int, seed: int = SEED, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Draw Bernoulli shot outcomes in bounded-memory chunks using a stable RNG stream."""
    p = _binary_probabilities(probabilities)
    if trials <= 0 or chunk_size <= 0 or match_id <= 0 or side not in (0, 1):
        raise SimulationError("Simulation requires positive trial/chunk/match counts and side 0 (home) or 1 (away).")
    rng = _rng_for(match_id, side, seed)
    score = np.empty(trials, dtype=np.int32)
    offset = 0
    while offset < trials:
        count = min(chunk_size, trials - offset)
        if len(p):
            draws = rng.random((count, len(p))) < p
            score[offset:offset + count] = draws.sum(axis=1, dtype=np.int32)
        else:
            score[offset:offset + count] = 0
        offset += count
    return score


def _check_sampling(home_scores: np.ndarray, away_scores: np.ndarray, home_probabilities: np.ndarray, away_probabilities: np.ndarray, exact_outcomes: dict[str, float]) -> dict[str, Any]:
    trials = len(home_scores)
    if trials != len(away_scores) or trials <= 0:
        raise SimulationError("Home and away simulated score arrays must have equal non-zero trial counts.")
    checks = {"trials": trials}
    for side, scores, probabilities in (
        ("home", home_scores, home_probabilities), ("away", away_scores, away_probabilities),
    ):
        expected_mean = float(probabilities.sum())
        variance = float(np.sum(probabilities * (1 - probabilities)))
        observed_mean = float(scores.mean())
        mcse = math.sqrt(variance / trials)
        limit = 6 * mcse + 1e-12
        if abs(observed_mean - expected_mean) > limit:
            raise SimulationError(f"{side} Monte Carlo goal mean differs from its Bernoulli expectation beyond 6 MCSE.")
        checks[f"{side}_mean"] = observed_mean
        checks[f"{side}_expected_mean"] = expected_mean
        checks[f"{side}_mean_mcse"] = mcse
        checks[f"{side}_mean_tolerance"] = limit
        checks[f"{side}_mean_absolute_error"] = abs(observed_mean - expected_mean)
    home_win = home_scores > away_scores
    draw = home_scores == away_scores
    for label, observed, exact in (
        ("home_win", home_win, exact_outcomes["home_win"]),
        ("draw", draw, exact_outcomes["draw"]),
        ("away_win", away_scores > home_scores, exact_outcomes["away_win"]),
    ):
        estimate = float(np.mean(observed))
        tolerance = 6 * math.sqrt(exact * (1 - exact) / trials) + 1 / trials
        if abs(estimate - exact) > tolerance:
            raise SimulationError(f"Monte Carlo {label} probability differs from the exact value beyond the 6-MCSE tolerance.")
        checks[f"{label}_probability"] = estimate
        checks[f"{label}_exact_probability"] = exact
        checks[f"{label}_mc_tolerance"] = tolerance
        checks[f"{label}_absolute_error"] = abs(estimate - exact)
    return checks


def join_test_shots(shots: pd.DataFrame, predictions: pd.DataFrame, reconciliation: pd.DataFrame, test_match_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join frozen probabilities by event ID and match metadata by match ID."""
    if shots["event_id"].isna().any() or shots["event_id"].astype(str).duplicated().any():
        raise SimulationError("Phase 4 shot event IDs must be complete and globally unique.")
    if predictions["event_id"].isna().any() or predictions["event_id"].astype(str).duplicated().any():
        raise SimulationError("Frozen prediction event IDs must be complete and unique.")
    test_shots = shots.loc[shots["match_id"].astype(int).isin(test_match_ids)].copy()
    if set(test_shots["match_id"].astype(int)) != test_match_ids:
        raise SimulationError("Phase 4 shots do not cover exactly the locked test matches.")
    if set(test_shots["event_id"].astype(str)) != set(predictions["event_id"].astype(str)):
        raise SimulationError("Frozen probabilities and test shot events do not have exact one-to-one coverage.")
    pred = predictions.copy()
    pred["event_id"] = pred["event_id"].astype(str)
    pred = pred[["event_id", "match_id", "y_true", "p_goal"]]
    pred = pred.rename(columns={"match_id": "prediction_match_id", "y_true": "prediction_y_true"})
    joined = test_shots.copy()
    joined["event_id"] = joined["event_id"].astype(str)
    joined = joined.merge(pred, on="event_id", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise SimulationError("Some test shots have no frozen Phase 8 probability.")
    if not np.array_equal(joined["match_id"].astype(int).to_numpy(), joined["prediction_match_id"].astype(int).to_numpy()):
        raise SimulationError("Frozen prediction match IDs do not match Phase 4 events.")
    if not np.array_equal(joined["is_goal"].astype(int).to_numpy(), joined["prediction_y_true"].astype(int).to_numpy()):
        raise SimulationError("Frozen prediction labels do not match Phase 4 shot labels.")
    joined = joined.drop(columns=["prediction_match_id", "prediction_y_true", "_merge"])
    if reconciliation["match_id"].astype(int).duplicated().any():
        raise SimulationError("Phase 4 reconciliation must have one row per match.")
    match_info = reconciliation.loc[reconciliation["match_id"].astype(int).isin(test_match_ids)].copy()
    if len(match_info) != len(test_match_ids) or set(match_info["match_id"].astype(int)) != test_match_ids:
        raise SimulationError("Phase 4 reconciliation does not cover exactly the locked test matches.")
    if match_info[["home_team_id", "away_team_id"]].isna().any().any():
        raise SimulationError("Reconciliation has missing home/away team IDs.")
    joined = joined.merge(match_info, on="match_id", how="left", validate="many_to_one", suffixes=("", "_match"), indicator="_match_merge")
    if not joined["_match_merge"].eq("both").all():
        raise SimulationError("Test shots are missing reconciled match metadata.")
    for column in ("home_team_id", "away_team_id", "home_team_name", "away_team_name", "match_date"):
        match_column = f"{column}_match"
        if column not in joined or match_column not in joined:
            raise SimulationError(f"Shot rows and reconciliation are missing match metadata column {column}.")
        if not joined[column].astype(str).eq(joined[match_column].astype(str)).all():
            raise SimulationError(f"Shot-table {column} metadata differs from official match reconciliation.")
    # The reconciled match table is authoritative for home/away team identity.
    is_home = joined["team_id"].astype(int).eq(joined["home_team_id_match"].astype(int))
    is_away = joined["team_id"].astype(int).eq(joined["away_team_id_match"].astype(int))
    if not (is_home | is_away).all():
        raise SimulationError("A shot team ID is not the corresponding match's home or away team.")
    if not np.array_equal(joined["is_home"].astype(int).to_numpy(), is_home.astype(int).to_numpy()):
        raise SimulationError("Phase 4 home/away flags disagree with the reconciled team IDs.")
    for match_id, rows in joined.groupby("match_id", sort=False):
        match = match_info.loc[match_info["match_id"].astype(int) == int(match_id)].iloc[0]
        shot_home_goals = int(rows.loc[rows["team_id"].astype(int) == int(match["home_team_id"]), "is_goal"].astype(int).sum())
        shot_away_goals = int(rows.loc[rows["team_id"].astype(int) == int(match["away_team_id"]), "is_goal"].astype(int).sum())
        if (shot_home_goals, shot_away_goals) != (int(match["shot_goals_home"]), int(match["shot_goals_away"])):
            raise SimulationError(f"Per-team shot-goal counts differ from reconciliation for match {int(match_id)}.")
        if (int(match["official_home_goals"]) != shot_home_goals + int(match["own_goals_credited_home"])
                or int(match["official_away_goals"]) != shot_away_goals + int(match["own_goals_credited_away"])):
            raise SimulationError(f"Official goals do not reconcile to shot goals plus own-goal credits for match {int(match_id)}.")
    return joined.drop(columns=["_match_merge"]), match_info


def validate_own_goal_credits(own_goals: pd.DataFrame, match_info: pd.DataFrame) -> None:
    """Check own-goal beneficiary IDs and counts against official match reconciliation."""
    if own_goals["event_id"].isna().any() or own_goals["event_id"].astype(str).duplicated().any():
        raise SimulationError("Own-goal event IDs must be complete and unique.")
    match_ids = set(match_info["match_id"].astype(int))
    if not set(own_goals["match_id"].astype(int)).issubset(match_ids):
        raise SimulationError("Own-goal rows include matches outside the locked test match set.")
    for _, match in match_info.iterrows():
        match_id = int(match["match_id"])
        records = own_goals.loc[own_goals["match_id"].astype(int) == match_id]
        beneficiary_ids = records["beneficiary_team_id"].dropna().astype(int)
        if len(beneficiary_ids) != len(records) or not beneficiary_ids.isin([int(match["home_team_id"]), int(match["away_team_id"])]).all():
            raise SimulationError(f"Own-goal beneficiary IDs do not belong to match {match_id} home or away teams.")
        credited_home = int(beneficiary_ids.eq(int(match["home_team_id"])).sum())
        credited_away = int(beneficiary_ids.eq(int(match["away_team_id"])).sum())
        if (credited_home, credited_away) != (int(match["own_goals_credited_home"]), int(match["own_goals_credited_away"])):
            raise SimulationError(f"Own-goal beneficiary records do not corroborate reconciliation for match {match_id}.")


def _expected_points(outcomes: dict[str, float]) -> tuple[float, float]:
    home = 3 * outcomes["home_win"] + outcomes["draw"]
    away = 3 * outcomes["away_win"] + outcomes["draw"]
    if not np.isclose(home + away, 3 - outcomes["draw"], atol=1e-12, rtol=0):
        raise SimulationError("Expected points must sum to 3 minus the draw probability.")
    return float(home), float(away)


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SimulationError(f"Cannot serialize simulation provenance: {exc}") from exc


def _protocol_payload(inputs: dict[str, Any], example_match_id: int) -> tuple[dict[str, Any], bytes]:
    report = inputs["test_report"]
    protocol = {
        "status": "predeclared_conditional_heldout_simulation_protocol",
        "scope": "76 locked test matches; simulate only their observed shot events using the frozen Phase 8 global candidate",
        "source_snapshot_commit": phase4.SOURCE["commit"],
        "source_shots_sha256": inputs["test_report"]["source_shot_csv_sha256"],
        "phase4_report_sha256": acquisition.digest(inputs["phase4_report_bytes"]),
        "phase4_inputs": {key: {"path": relative, "sha256": acquisition.digest(inputs["phase4_payloads"][key])} for key, relative in inputs["phase4_files"].items()},
        "phase8_protocol_sha256": acquisition.digest(inputs["phase8_inputs"]["protocol_bytes"]),
        "phase8_selection_sha256": acquisition.digest(inputs["selection_bytes"]),
        "phase8_test_report_sha256": acquisition.digest(inputs["test_report_bytes"]),
        "phase8_test_predictions_sha256": report["outputs"][inputs["prediction_relative"]]["sha256"],
        "phase8_global_candidate": inputs["selection"]["global_candidate"],
        "test_match_count": inputs["phase8_inputs"]["split"]["test_match_count"],
        "test_shot_count": inputs["phase8_inputs"]["split"]["test_shot_count"],
        "trials_per_match": TRIALS, "chunk_size": CHUNK_SIZE, "seed": SEED,
        "rng_algorithm": "numpy.default_rng(SeedSequence([seed, match_id, side]))",
        "team_side_codes": {"home": 0, "away": 1},
        "example_match_rule": "Earliest held-out match by match_date then match_id, chosen without score or model-performance inspection.",
        "example_match_id": int(example_match_id),
        "example_convergence_prefixes": list(EXAMPLE_CHECKPOINTS),
        "shot_outcome_model": "Independent Bernoulli trials for the already-observed shots using frozen P(goal); no Poisson approximation.",
        "own_goal_policy": "Do not sample or add own goals; show official score and own-goal credits separately as reconciliation context.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "limits": [
            "Conditional on the observed shot events and frozen predicted probabilities; not a pre-match score forecast.",
            "Monte Carlo error describes finite simulation draws conditional on this model, not model or future-season uncertainty.",
            "Own goals are not generated by the shot model; shot-generated scorelines can differ from official scores.",
            "Only the 76 held-out matches are summarized; no league-table or full-season simulation is performed.",
        ],
    }
    payload = _json_bytes(protocol)
    return protocol, payload


def _lock_protocol(root: Path, value: dict[str, Any], payload: bytes) -> None:
    path = root / PROTOCOL_RELATIVE
    if path.exists():
        if path.read_bytes() != payload:
            raise SimulationError("Existing Phase 9 protocol differs; refusing to overwrite its frozen simulation settings.")
    else:
        acquisition.atomic_write(path, payload)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _build_outputs(joined: pd.DataFrame, matches: pd.DataFrame, own_goals: pd.DataFrame, example_match_id: int) -> tuple[dict[str, pd.DataFrame], dict[str, bytes], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    scoreline_records: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []
    example_shots = joined.loc[joined["match_id"].astype(int) == example_match_id].copy()
    if example_shots.empty:
        raise SimulationError("The preselected example match has no shot events.")
    example_shots = example_shots.sort_values(["event_index", "event_id"], kind="stable").reset_index(drop=True)
    figure_buffers: dict[str, bytes] = {}

    match_lookup = matches.set_index("match_id")
    if not match_lookup.index.is_unique:
        raise SimulationError("Simulation match metadata must contain exactly one row per match.")
    for match_id in sorted(int(value) for value in matches["match_id"]):
        metadata = match_lookup.loc[match_id]
        rows = joined.loc[joined["match_id"].astype(int) == match_id].sort_values(["event_index", "event_id"], kind="stable")
        home = rows.loc[rows["is_home"].astype(int) == 1]
        away = rows.loc[rows["is_home"].astype(int) == 0]
        home_probabilities = _binary_probabilities(home["p_goal"].to_numpy())
        away_probabilities = _binary_probabilities(away["p_goal"].to_numpy())
        home_pmf, away_pmf = exact_poisson_binomial(home_probabilities), exact_poisson_binomial(away_probabilities)
        joint, exact_outcomes = exact_joint_probabilities(home_pmf, away_pmf)
        home_scores = simulate_goals(home_probabilities, TRIALS, match_id, 0)
        away_scores = simulate_goals(away_probabilities, TRIALS, match_id, 1)
        checks = _check_sampling(home_scores, away_scores, home_probabilities, away_probabilities, exact_outcomes)
        home_counts = np.bincount(home_scores, minlength=len(home_pmf))
        away_counts = np.bincount(away_scores, minlength=len(away_pmf))
        score_counts = np.bincount(home_scores * len(away_pmf) + away_scores, minlength=joint.size).reshape(joint.shape)
        if int(score_counts.sum()) != TRIALS or int(home_counts.sum()) != TRIALS or int(away_counts.sum()) != TRIALS:
            raise SimulationError(f"Match {match_id} Monte Carlo counts do not sum to {TRIALS}.")
        for home_goals in range(joint.shape[0]):
            for away_goals in range(joint.shape[1]):
                count = int(score_counts[home_goals, away_goals])
                scoreline_records.append({
                    "match_id": match_id, "home_goals": home_goals, "away_goals": away_goals,
                    "exact_probability": float(joint[home_goals, away_goals]),
                    "mc_count": count, "mc_probability": count / TRIALS,
                })
        own_goal_rows = own_goals.loc[own_goals["match_id"].astype(int) == match_id]
        home_points, away_points = _expected_points(exact_outcomes)
        summary = {
            "match_id": match_id, "match_date": str(metadata["match_date"]),
            "home_team_id": int(metadata["home_team_id"]), "home_team_name": str(metadata["home_team_name"]),
            "away_team_id": int(metadata["away_team_id"]), "away_team_name": str(metadata["away_team_name"]),
            "official_home_goals": int(metadata["official_home_goals"]), "official_away_goals": int(metadata["official_away_goals"]),
            "observed_shot_goals_home": int(metadata["shot_goals_home"]), "observed_shot_goals_away": int(metadata["shot_goals_away"]),
            "own_goals_credited_home": int(metadata["own_goals_credited_home"]), "own_goals_credited_away": int(metadata["own_goals_credited_away"]),
            "own_goal_against_events": int(len(own_goal_rows)),
            "official_minus_observed_shot_goals_home": int(metadata["official_home_goals"] - metadata["shot_goals_home"]),
            "official_minus_observed_shot_goals_away": int(metadata["official_away_goals"] - metadata["shot_goals_away"]),
            "home_shot_count": int(len(home)), "away_shot_count": int(len(away)),
            "home_sum_goal_probability": float(home_probabilities.sum()), "away_sum_goal_probability": float(away_probabilities.sum()),
            "home_expected_shot_goals": float(home_probabilities.sum()), "away_expected_shot_goals": float(away_probabilities.sum()),
            "home_mc_mean_goals": checks["home_mean"], "away_mc_mean_goals": checks["away_mean"],
            "home_mean_mcse": checks["home_mean_mcse"], "away_mean_mcse": checks["away_mean_mcse"],
            "home_mean_mc_tolerance": checks["home_mean_tolerance"], "away_mean_mc_tolerance": checks["away_mean_tolerance"],
            "home_mean_absolute_error": checks["home_mean_absolute_error"], "away_mean_absolute_error": checks["away_mean_absolute_error"],
            "home_exact_goal_variance": float(np.sum(home_probabilities * (1 - home_probabilities))),
            "away_exact_goal_variance": float(np.sum(away_probabilities * (1 - away_probabilities))),
            "exact_home_win_probability": exact_outcomes["home_win"], "exact_draw_probability": exact_outcomes["draw"],
            "exact_away_win_probability": exact_outcomes["away_win"],
            "mc_home_win_count": int(np.sum(home_scores > away_scores)),
            "mc_draw_count": int(np.sum(home_scores == away_scores)),
            "mc_away_win_count": int(np.sum(home_scores < away_scores)),
            "mc_home_win_probability": checks["home_win_probability"], "mc_draw_probability": checks["draw_probability"],
            "mc_away_win_probability": checks["away_win_probability"],
            "home_win_probability_absolute_error": checks["home_win_absolute_error"],
            "draw_probability_absolute_error": checks["draw_absolute_error"],
            "away_win_probability_absolute_error": checks["away_win_absolute_error"],
            "home_win_probability_mc_tolerance": checks["home_win_mc_tolerance"],
            "draw_probability_mc_tolerance": checks["draw_mc_tolerance"],
            "away_win_probability_mc_tolerance": checks["away_win_mc_tolerance"],
            "home_expected_points_exact": home_points, "away_expected_points_exact": away_points,
            "trials": TRIALS,
            "home_mean_check_passed": abs(checks["home_mean"] - checks["home_expected_mean"]) <= checks["home_mean_tolerance"],
            "away_mean_check_passed": abs(checks["away_mean"] - checks["away_expected_mean"]) <= checks["away_mean_tolerance"],
            "mc_outcome_checks_passed": True,
        }
        if sum(summary[key] for key in ("mc_home_win_count", "mc_draw_count", "mc_away_win_count")) != TRIALS:
            raise SimulationError(f"Match {match_id} Monte Carlo outcome counts do not sum to {TRIALS}.")
        summaries.append(summary)

        if match_id == example_match_id:
            for checkpoint in EXAMPLE_CHECKPOINTS:
                home_win_count = int(np.sum(home_scores[:checkpoint] > away_scores[:checkpoint]))
                draw_count = int(np.sum(home_scores[:checkpoint] == away_scores[:checkpoint]))
                away_win_count = checkpoint - home_win_count - draw_count
                for name, count, exact in (
                    ("home_win", home_win_count, exact_outcomes["home_win"]),
                    ("draw", draw_count, exact_outcomes["draw"]),
                    ("away_win", away_win_count, exact_outcomes["away_win"]),
                ):
                    estimate = count / checkpoint
                    mcse = math.sqrt(exact * (1 - exact) / checkpoint)
                    convergence_records.append({
                        "match_id": match_id, "trials": checkpoint, "outcome": name,
                        "exact_probability": exact, "mc_count": count, "mc_probability": estimate,
                        "absolute_error": abs(estimate - exact), "mc_standard_error": mcse,
                    })
            figure_buffers.update(_example_figures(metadata, example_shots, home_pmf, away_pmf, joint, exact_outcomes, home_scores, away_scores, convergence_records))

    tables = {
        TABLE_NAMES["match_summary"]: pd.DataFrame(summaries),
        TABLE_NAMES["scoreline_distribution"]: pd.DataFrame(scoreline_records),
        TABLE_NAMES["convergence"]: pd.DataFrame(convergence_records),
        TABLE_NAMES["example_shots"]: example_shots[["match_id", "event_id", "event_index", "team_id", "team_name", "is_home", "is_goal", "p_goal"]].copy(),
    }
    return tables, figure_buffers, {"example_match_id": example_match_id, "match_count": len(summaries), "trial_count": TRIALS}


def _example_figures(metadata: pd.Series, shots: pd.DataFrame, home_pmf: np.ndarray, away_pmf: np.ndarray, joint: np.ndarray, outcomes: dict[str, float], home_scores: np.ndarray, away_scores: np.ndarray, convergence_records: list[dict[str, Any]]) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    fig, axis = plt.subplots(figsize=(7.3, 5.5), facecolor="white")
    image = axis.imshow(joint, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest")
    axis.set_xlabel(f"{metadata['away_team_name']} goals")
    axis.set_ylabel(f"{metadata['home_team_name']} goals")
    axis.set_xticks(np.arange(joint.shape[1]))
    axis.set_yticks(np.arange(joint.shape[0]))
    axis.set_title(f"{metadata['match_date']} | {metadata['home_team_name']} vs {metadata['away_team_name']}\nConditional scoreline probabilities", color="#23364D")
    fig.colorbar(image, ax=axis, label="Exact joint probability")
    official = f"Official score {int(metadata['official_home_goals'])}–{int(metadata['official_away_goals'])}; own goals credited H/A {int(metadata['own_goals_credited_home'])}/{int(metadata['own_goals_credited_away'])}"
    fig.text(0.5, 0.015, f"StatsBomb La Liga 2015/16 • {official}\nOwn goals shown for context, not simulated. Conditional on observed shots, not a pre-match forecast.", ha="center", fontsize=8, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.98])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"}); plt.close(fig)
    output[f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['example_scoreline']}"] = buf.getvalue()

    mc_wdl = [float(np.mean(home_scores > away_scores)), float(np.mean(home_scores == away_scores)), float(np.mean(home_scores < away_scores))]
    labels = ["Home win", "Draw", "Away win"]
    x = np.arange(3); width = 0.37
    fig, axis = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    exact = [outcomes["home_win"], outcomes["draw"], outcomes["away_win"]]
    axis.bar(x - width/2, exact, width, label="Exact Poisson-binomial", color="#315A8A")
    axis.bar(x + width/2, mc_wdl, width, label=f"Monte Carlo ({TRIALS:,})", color="#D88735")
    axis.set_xticks(x, labels=labels); axis.set_ylim(0, 1); axis.set_ylabel("Probability")
    axis.set_title(f"{metadata['match_date']} | {metadata['home_team_name']} vs {metadata['away_team_name']}\nExact and simulated outcomes conditional on observed shots", color="#23364D")
    axis.legend(); axis.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.015, "StatsBomb La Liga 2015/16. Conditional on observed shots, not a pre-match forecast.\nMC differences reflect sampling error for fixed model probabilities, not model uncertainty.", ha="center", fontsize=8, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"}); plt.close(fig)
    output[f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['example_outcomes']}"] = buf.getvalue()

    frame = pd.DataFrame(convergence_records)
    fig, axis = plt.subplots(figsize=(7.4, 4.7), facecolor="white")
    colors = {"home_win": "#315A8A", "draw": "#398564", "away_win": "#D88735"}
    for outcome in ("home_win", "draw", "away_win"):
        subset = frame.loc[frame["outcome"] == outcome]
        axis.plot(subset["trials"], subset["absolute_error"], marker="o", color=colors[outcome], label=f"{outcome.replace('_', ' ').title()} absolute error")
        axis.plot(subset["trials"], 6 * subset["mc_standard_error"] + 1 / subset["trials"], linestyle=":", color=colors[outcome], alpha=0.7)
    axis.set_xscale("log"); axis.set_xlabel("Trials (prefix of the same seeded run)"); axis.set_ylabel("Absolute probability error")
    axis.set_title(f"{metadata['match_date']} | {metadata['home_team_name']} vs {metadata['away_team_name']}\nMonte Carlo convergence to exact outcome probabilities", color="#23364D")
    axis.grid(alpha=0.2); axis.legend(fontsize=8)
    fig.text(0.5, 0.015, "StatsBomb La Liga 2015/16. Conditional on observed shots, not a pre-match forecast.\nDotted: 6 MC standard errors + 1/N; individual errors need not fall monotonically.", ha="center", fontsize=8, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"}); plt.close(fig)
    output[f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['convergence']}"] = buf.getvalue()
    return output


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inputs = verify_phase_inputs(root)
    test_ids = set(inputs["phase8_inputs"]["split"]["test_match_ids"])
    shots = pd.read_csv(root / "data/processed/shots.csv", dtype={"event_id": str}, float_precision="round_trip")
    reconciliation = pd.read_csv(root / "data/processed/goal_reconciliation.csv", float_precision="round_trip")
    own_goals = pd.read_csv(root / "data/processed/own_goals.csv", float_precision="round_trip")
    joined, match_info = join_test_shots(shots, inputs["predictions"], reconciliation, test_ids)
    own_goals = own_goals.loc[own_goals["match_id"].astype(int).isin(test_ids)].copy()
    validate_own_goal_credits(own_goals, match_info)

    # Pick the showcase from metadata order only; no scores or model results influence it.
    example_match_id = int(match_info.sort_values(["match_date", "match_id"], kind="stable").iloc[0]["match_id"])
    protocol, protocol_bytes = _protocol_payload(inputs, example_match_id)
    _lock_protocol(root, protocol, protocol_bytes)

    tables, figures, run_info = _build_outputs(joined, match_info, own_goals, example_match_id)
    outputs: dict[str, bytes] = {}
    for name, frame in tables.items():
        outputs[f"{TABLE_DIRECTORY}/{name}"] = _csv_bytes(frame)
    outputs.update(figures)
    output_entries = {relative: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for relative, payload in outputs.items()}
    summary = tables[TABLE_NAMES["match_summary"]]
    report = {
        "status": "passed", "scope": "conditional shot-goal score simulation for the frozen 76-match test partition",
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "phase8_selection_sha256": acquisition.digest(inputs["selection_bytes"]),
        "phase8_test_report_sha256": acquisition.digest(inputs["test_report_bytes"]),
        "phase8_test_predictions_sha256": inputs["test_report"]["outputs"][inputs["prediction_relative"]]["sha256"],
        "source_shots_sha256": acquisition.digest(inputs["phase4_payloads"]["shots"]),
        "source_reconciliation_sha256": acquisition.digest(inputs["phase4_payloads"]["reconciliation"]),
        "source_own_goals_sha256": acquisition.digest(inputs["phase4_payloads"]["own_goals"]),
        "candidate": inputs["selection"]["global_candidate"],
        "match_count": int(len(summary)), "shot_count": int(len(joined)), "trial_count_per_match": TRIALS,
        "example_match_id": example_match_id,
        "official_home_goals_test_total": int(summary["official_home_goals"].sum()),
        "official_away_goals_test_total": int(summary["official_away_goals"].sum()),
        "observed_shot_goals_home_test_total": int(summary["observed_shot_goals_home"].sum()),
        "observed_shot_goals_away_test_total": int(summary["observed_shot_goals_away"].sum()),
        "own_goals_credited_home_test_total": int(summary["own_goals_credited_home"].sum()),
        "own_goals_credited_away_test_total": int(summary["own_goals_credited_away"].sum()),
        "checks": {
            "all_match_trial_counts_sum": bool(summary["trials"].eq(TRIALS).all()),
            "all_simulated_outcome_counts_sum": bool((summary["mc_home_win_count"] + summary["mc_draw_count"] + summary["mc_away_win_count"]).eq(TRIALS).all()),
            "all_means_within_six_mcse": bool(summary["home_mean_check_passed"].all() and summary["away_mean_check_passed"].all()),
            "all_exact_outcome_probabilities_sum_to_one": bool(np.allclose(summary[["exact_home_win_probability", "exact_draw_probability", "exact_away_win_probability"]].sum(axis=1), 1, atol=1e-12, rtol=0)),
            "own_goals_not_simulated": True,
            "test_matches_only": set(summary["match_id"].astype(int)) == test_ids,
        },
        "limitations": protocol["limits"], "outputs": output_entries,
    }
    report_bytes = _json_bytes(report)
    for relative, payload in outputs.items():
        acquisition.atomic_write(root / relative, payload)
    acquisition.atomic_write(root / REPORT_RELATIVE, report_bytes)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.root)
    except (SimulationError, phase8.CalibrationError, phase8.holdout.SplitError, preparation.PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"Phase 9 simulation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "candidate", "match_count", "shot_count", "trial_count_per_match", "example_match_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
