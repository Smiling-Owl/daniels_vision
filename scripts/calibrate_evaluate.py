"""Develop nested sigmoid calibration on training folds, then evaluate frozen variants."""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

try:
    from scripts import acquire_data as acquisition
    from scripts import prepare_modeling as preparation
    from scripts import reserve_holdout as holdout
    from scripts import train_models as phase7
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import prepare_modeling as preparation
    import reserve_holdout as holdout
    import train_models as phase7

PROTOCOL_RELATIVE = "docs/phase_08_protocol.json"
SELECTION_RELATIVE = "docs/phase_08_selection.json"
DEV_REPORT_RELATIVE = "docs/phase_08_development_report.json"
REPORT_RELATIVE = "docs/phase_08_test_report.json"
MODEL_DIRECTORY = "models/phase08"
MODEL_MANIFEST_RELATIVE = f"{MODEL_DIRECTORY}/manifest.json"
TABLE_DIRECTORY = "reports/tables"
FIGURE_DIRECTORY = "reports/figures"
RAW_FAMILIES = ["knn_50", "gaussian_nb", "logistic_regression"]
BASELINE = "prior_baseline"
VARIANTS = [BASELINE, *(f"{name}_raw" for name in RAW_FAMILIES), *(f"{name}_sigmoid" for name in RAW_FAMILIES)]
OUTER_FOLDS = 10
INNER_FOLDS = 5
SEED = 42
THRESHOLD = 0.5
TABLE_NAMES = {
    "development_predictions": "phase08_development_predictions.csv",
    "development_metrics": "phase08_development_metrics.csv",
    "fold_metrics": "phase08_fold_metrics.csv",
    "development_reliability_bins": "phase08_development_reliability_bins.csv",
    "test_reliability_bins": "phase08_test_reliability_bins.csv",
    "test_predictions": "phase08_test_predictions.csv",
    "test_metrics": "phase08_test_metrics.csv",
    "training_metrics": "phase08_training_metrics.csv",
}
FIGURE_NAMES = {
    "development_reliability": "phase08_development_reliability.png",
    "test_reliability": "phase08_test_reliability.png",
    "test_confusion": "phase08_test_confusion_matrices.png",
}
METRIC_FIELDS = [*phase7.METRIC_NAMES, *phase7.COUNT_NAMES]


class CalibrationError(RuntimeError):
    """Calibration protocol, source artifact, or evaluation boundary is invalid."""


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrationError(f"Expected a JSON object in {path}.")
    return value, payload


def verify_entry(root: Path, relative: str, entry: dict[str, Any]) -> bytes:
    path = root / relative
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CalibrationError(f"Cannot read required Phase 7 artifact {relative}: {exc}") from exc
    if len(payload) != entry.get("bytes") or acquisition.digest(payload) != entry.get("sha256"):
        raise CalibrationError(f"Phase 7 artifact hash/size mismatch: {relative}.")
    return payload


def verify_phase7(root: Path) -> dict[str, Any]:
    """Verify every Phase 7 input, report, table, and saved pipeline before loading it."""
    root = root.resolve()
    split = holdout.reserve(root)
    protocol, protocol_bytes = read_json(root / phase7.PROTOCOL_RELATIVE)
    expected_phase7_protocol, expected_phase7_bytes = phase7.protocol_payload(root)
    if protocol_bytes != expected_phase7_bytes or protocol != expected_phase7_protocol:
        raise CalibrationError("Frozen Phase 7 model settings or library versions differ from the current factory.")
    model_manifest, model_manifest_bytes = read_json(root / phase7.MODEL_MANIFEST_RELATIVE)
    report, report_bytes = read_json(root / phase7.REPORT_RELATIVE)
    if report.get("status") != "passed" or model_manifest.get("status") != "passed":
        raise CalibrationError("Phase 7 model report or saved-model manifest is not marked passed.")
    if acquisition.digest(protocol_bytes) != report.get("protocol_sha256") or model_manifest.get("protocol_sha256") != report.get("protocol_sha256"):
        raise CalibrationError("Phase 7 protocol hash is inconsistent across its artifacts.")
    if acquisition.digest(model_manifest_bytes) != report.get("model_manifest_sha256"):
        raise CalibrationError("Phase 7 model manifest hash differs from its report.")
    if acquisition.digest((root / preparation.SHOT_RELATIVE).read_bytes()) != protocol.get("source_shot_csv_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different shot CSV.")
    if acquisition.digest((root / holdout.SPLIT_MANIFEST_RELATIVE).read_bytes()) != protocol.get("holdout_manifest_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different holdout lock.")
    if split.get("membership_csv_sha256") != protocol.get("holdout_membership_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different holdout membership.")
    if acquisition.digest((root / preparation.CV_MANIFEST_RELATIVE).read_bytes()) != protocol.get("cv_manifest_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different CV lock.")
    if acquisition.digest((root / preparation.CV_MEMBERSHIP_RELATIVE).read_bytes()) != protocol.get("cv_membership_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different CV map.")
    if acquisition.digest((root / preparation.CONTRACT_RELATIVE).read_bytes()) != protocol.get("model_contract_sha256"):
        raise CalibrationError("Phase 7 protocol is for a different feature contract.")
    if model_manifest.get("source_shot_csv_sha256") != protocol.get("source_shot_csv_sha256"):
        raise CalibrationError("Phase 7 model manifest source hash differs from its protocol.")
    for relative, entry in report.get("outputs", {}).items():
        verify_entry(root, relative, entry)
    if set(model_manifest.get("pipelines", {})) != set(report.get("pipeline_outputs", {})):
        raise CalibrationError("Phase 7 report and model manifest list different fitted pipelines.")
    for relative, entry in model_manifest.get("pipelines", {}).items():
        if report["pipeline_outputs"].get(relative) != entry:
            raise CalibrationError(f"Phase 7 pipeline report entry differs for {relative}.")
        verify_entry(root, relative, entry)
    return {
        "split": split, "protocol": protocol, "protocol_bytes": protocol_bytes,
        "model_manifest": model_manifest, "model_manifest_bytes": model_manifest_bytes,
        "report": report, "report_bytes": report_bytes,
    }


def make_protocol(phase7_inputs: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    p7 = phase7_inputs["protocol"]
    protocol = {
        "status": "predeclared_nested_sigmoid_calibration_protocol",
        "scope": "development on locked training partition; test is inaccessible until frozen selection",
        "source_shot_csv_sha256": p7["source_shot_csv_sha256"],
        "phase7_protocol_sha256": acquisition.digest(phase7_inputs["protocol_bytes"]),
        "phase7_report_sha256": acquisition.digest(phase7_inputs["report_bytes"]),
        "phase7_model_manifest_sha256": acquisition.digest(phase7_inputs["model_manifest_bytes"]),
        "phase7_oof_sha256": phase7_inputs["report"]["outputs"][f"{phase7.TABLE_DIRECTORY}/{phase7.TABLE_NAMES['oof_predictions']}"]["sha256"],
        "holdout_manifest_sha256": p7["holdout_manifest_sha256"],
        "cv_manifest_sha256": p7["cv_manifest_sha256"],
        "cv_membership_sha256": p7["cv_membership_sha256"],
        "model_contract_sha256": p7["model_contract_sha256"],
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "seed": SEED,
        "inner_splitter": "StratifiedGroupKFold(shuffle=True, random_state=42)",
        "calibration": {"method": "sigmoid", "ensemble": False, "n_jobs": 1},
        "raw_model_families": RAW_FAMILIES,
        "variants": VARIANTS,
        "threshold": THRESHOLD,
        "threshold_rule": "predict goal when P(goal) >= 0.5",
        "family_selection_rule": "For each KNN, GaussianNB, and logistic-regression family, choose sigmoid calibration only when its nested pooled training OOF log loss is strictly lower than the raw model; ties choose raw.",
        "global_candidate_rule": "Among the three family-selected variants, choose the lowest pooled nested training OOF log loss; exact ties break by variant name. This is a development candidate, not an unbiased final estimate.",
        "prohibited": ["test-informed selection", "threshold search", "hyperparameter tuning", "SMOTE", "class weighting", "test-based recalibration"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    payload = (json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return protocol, payload


def lock_json(root: Path, relative: str, value: dict[str, Any], payload: bytes) -> None:
    validate_locked_candidate(root, relative, value, payload)
    path = root / relative
    if not path.exists():
        acquisition.atomic_write(path, payload)


def validate_locked_candidate(root: Path, relative: str, value: dict[str, Any], payload: bytes) -> None:
    path = root / relative
    if path.exists():
        current = path.read_bytes()
        try:
            json.loads(current.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"Existing locked {relative} is invalid JSON: {exc}") from exc
        if current != payload or value != json.loads(current.decode("utf-8")):
            raise CalibrationError(f"Existing locked {relative} differs from this protocol; refusing replacement.")


def positive_probabilities(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_ if hasattr(estimator, "classes_") else estimator.named_steps["classifier"].classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1 or probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise CalibrationError("Estimator does not expose a unique positive-class probability for class 1.")
    result = probabilities[:, int(positive[0])]
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise CalibrationError("Predicted probabilities must be finite values in [0,1].")
    return result


def validate_group_splits(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], expected_folds: int) -> None:
    if len(splits) != expected_folds:
        raise CalibrationError(f"Expected {expected_folds} group folds, got {len(splits)}.")
    seen = np.zeros(len(frame), dtype=np.int8)
    seen_groups: list[int] = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        train, validation = frame.iloc[train_indices], frame.iloc[validation_indices]
        if train.empty or validation.empty or set(train[preparation.GROUP_COLUMN]) & set(validation[preparation.GROUP_COLUMN]):
            raise CalibrationError(f"Fold {fold} has empty data or a match overlap.")
        if train[preparation.TARGET_COLUMN].nunique() != 2 or validation[preparation.TARGET_COLUMN].nunique() != 2:
            raise CalibrationError(f"Fold {fold} train and validation must each contain both classes.")
        seen[validation_indices] += 1
        seen_groups.extend(int(value) for value in validation[preparation.GROUP_COLUMN].unique())
    if not np.all(seen == 1) or len(seen_groups) != len(set(seen_groups)) or set(seen_groups) != set(frame[preparation.GROUP_COLUMN]):
        raise CalibrationError("Validation folds must cover every row and match exactly once.")


def grouped_splits(frame: pd.DataFrame, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [(train, validation) for train, validation in splitter.split(
        frame[preparation.FEATURE_COLUMNS], frame[preparation.TARGET_COLUMN], frame[preparation.GROUP_COLUMN]
    )]
    validate_group_splits(frame, splits, folds)
    return splits


def make_calibrator(raw_model_name: str, inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=phase7.make_pipeline(raw_model_name), method="sigmoid", cv=inner_splits,
        ensemble=False, n_jobs=1,
    )


def fit_calibrator(raw_model_name: str, train: pd.DataFrame, inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> CalibratedClassifierCV:
    validate_group_splits(train, inner_splits, INNER_FOLDS)
    calibrator = make_calibrator(raw_model_name, inner_splits)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        calibrator.fit(train[preparation.FEATURE_COLUMNS], train[preparation.TARGET_COLUMN])
    return calibrator


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def reliability_bins(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    probabilities = predictions["p_goal"].to_numpy(dtype=float)
    labels = predictions["y_true"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise CalibrationError("Reliability probabilities must be finite and inside [0,1].")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise CalibrationError("Reliability labels must contain finite binary integers 0 and 1.")
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = predictions.loc[predictions["variant"] == variant]
        if rows.empty:
            raise CalibrationError(f"No {scope} predictions found for variant {variant}.")
        bin_index = np.minimum(np.floor(rows["p_goal"].to_numpy(dtype=float) * 10).astype(int), 9)
        for index in range(10):
            selected = rows.loc[bin_index == index]
            count = int(len(selected))
            records.append({
                "scope": scope, "variant": variant, "bin_index": index,
                "bin_lower": index / 10, "bin_upper": (index + 1) / 10,
                "shot_count": count,
                "goal_count": int(selected["y_true"].sum()) if count else None,
                "mean_probability": float(selected["p_goal"].mean()) if count else None,
                "observed_goal_rate": float(selected["y_true"].mean()) if count else None,
            })
    return pd.DataFrame(records)


def metric_table(predictions: pd.DataFrame, scope: str, selections: dict[str, Any] | None = None) -> pd.DataFrame:
    records = []
    selections = selections or {}
    for variant in VARIANTS:
        rows = predictions.loc[predictions["variant"] == variant]
        values = phase7.metrics_for(rows["y_true"], rows["p_goal"], threshold=THRESHOLD)
        record: dict[str, Any] = {
            "scope": scope, "variant": variant, "shot_count": int(len(rows)),
            "goal_count": int(rows["y_true"].sum()), "non_goal_count": int(len(rows) - rows["y_true"].sum()),
            "family_selected": variant in selections.get("family_choices", {}).values(),
            "global_candidate": variant == selections.get("global_candidate"),
        }
        record.update(values)
        records.append(record)
    return pd.DataFrame(records)


def _prediction_frame(variant: str, frame: pd.DataFrame, probability: np.ndarray, fold: int | None = None) -> pd.DataFrame:
    result = pd.DataFrame({
        "event_id": frame[preparation.ID_COLUMN].astype(str).to_numpy(),
        "match_id": frame[preparation.GROUP_COLUMN].astype(int).to_numpy(),
        "variant": variant, "y_true": frame[preparation.TARGET_COLUMN].astype(int).to_numpy(),
        "p_goal": probability,
    })
    if fold is not None:
        result["fold"] = int(fold)
    result["y_pred"] = (result["p_goal"] >= THRESHOLD).astype(int)
    return result


def _load_training(root: Path, split: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = preparation.load_training_rows(root, set(split["train_match_ids"]))
    membership = preparation.load_cv_membership(root)
    joined = phase7.join_cv_assignments(training, membership, set(split["test_match_ids"]))
    if len(joined) != split["train_shot_count"] or joined[preparation.GROUP_COLUMN].nunique() != split["train_match_count"]:
        raise CalibrationError("Training rows do not match the locked holdout counts.")
    return training, joined


def _load_phase7_predictions(root: Path, phase7_inputs: dict[str, Any], training: pd.DataFrame, joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_path = f"{phase7.TABLE_DIRECTORY}/{phase7.TABLE_NAMES['oof_predictions']}"
    fold_path = f"{phase7.TABLE_DIRECTORY}/{phase7.TABLE_NAMES['fold_metrics']}"
    train_path = f"{phase7.TABLE_DIRECTORY}/{phase7.TABLE_NAMES['training_predictions']}"
    oof = pd.read_csv(root / oof_path, float_precision="round_trip")
    fold_metrics = pd.read_csv(root / fold_path, float_precision="round_trip")
    train_predictions = pd.read_csv(root / train_path, float_precision="round_trip")
    expected_ids = set(training[preparation.ID_COLUMN].astype(str))
    for source, label in ((oof, "OOF"), (train_predictions, "training")):
        for model_name in ["prior_baseline", *RAW_FAMILIES]:
            rows = source.loc[source["model"] == model_name]
            if len(rows) != len(training) or rows["event_id"].astype(str).duplicated().any() or set(rows["event_id"].astype(str)) != expected_ids:
                raise CalibrationError(f"Phase 7 {label} predictions for {model_name} lack exact training-event coverage.")
            aligned = rows.set_index(rows["event_id"].astype(str)).reindex(training[preparation.ID_COLUMN].astype(str))
            if not np.array_equal(aligned["y_true"].to_numpy(dtype=int), training[preparation.TARGET_COLUMN].to_numpy(dtype=int)):
                raise CalibrationError(f"Phase 7 {label} predictions have labels that differ from locked training rows.")
    if set(fold_metrics["model"]) != {"prior_baseline", *RAW_FAMILIES} or fold_metrics["fold"].nunique() != OUTER_FOLDS:
        raise CalibrationError("Phase 7 fold metrics do not contain all four models and ten folds.")
    return oof, fold_metrics, train_predictions


def _raw_development_predictions(oof: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    frame = joined[[preparation.ID_COLUMN, preparation.GROUP_COLUMN, preparation.TARGET_COLUMN, "validation_fold"]].copy()
    training_ids = set(frame[preparation.ID_COLUMN].astype(str))
    predictions = []
    for model_name in ["prior_baseline", *RAW_FAMILIES]:
        selected = oof.loc[oof["model"] == model_name, ["event_id", "y_true", "p_goal"]].copy()
        selected["event_id"] = selected["event_id"].astype(str)
        if len(selected) != len(frame) or selected["event_id"].duplicated().any() or set(selected["event_id"]) != training_ids:
            raise CalibrationError(f"Raw Phase 7 OOF event coverage invalid for {model_name}.")
        joined_predictions = frame.merge(selected, left_on=preparation.ID_COLUMN, right_on="event_id", how="left", validate="one_to_one", sort=False)
        if not np.array_equal(joined_predictions["y_true"].to_numpy(dtype=int), joined_predictions[preparation.TARGET_COLUMN].to_numpy(dtype=int)):
            raise CalibrationError(f"Raw Phase 7 OOF labels mismatch for {model_name}.")
        variant = BASELINE if model_name == "prior_baseline" else f"{model_name}_raw"
        predictions.append(pd.DataFrame({
            "event_id": joined_predictions[preparation.ID_COLUMN].astype(str),
            "match_id": joined_predictions[preparation.GROUP_COLUMN].astype(int),
            "fold": joined_predictions["validation_fold"].astype(int), "variant": variant,
            "y_true": joined_predictions[preparation.TARGET_COLUMN].astype(int),
            "p_goal": joined_predictions["p_goal"].astype(float),
            "y_pred": (joined_predictions["p_goal"].astype(float) >= THRESHOLD).astype(int),
        }))
    return pd.concat(predictions, ignore_index=True)


def _raw_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_model in ["prior_baseline", *RAW_FAMILIES]:
        variant = BASELINE if source_model == "prior_baseline" else f"{source_model}_raw"
        for record in fold_metrics.loc[fold_metrics["model"] == source_model].to_dict(orient="records"):
            row = {
                "scope": "development", "variant": variant, "fold": int(record["fold"]),
                "train_shot_count": int(record["train_shot_count"]), "validation_shot_count": int(record["validation_shot_count"]),
                "train_match_count": int(record["train_match_count"]), "validation_match_count": int(record["validation_match_count"]),
                "train_goal_count": int(record["train_goal_count"]), "train_non_goal_count": int(record["train_non_goal_count"]),
                "validation_goal_count": int(record["validation_goal_count"]), "validation_non_goal_count": int(record["validation_non_goal_count"]),
                "threshold": THRESHOLD,
            }
            row.update({metric: record[metric] for metric in METRIC_FIELDS})
            rows.append(row)
    return pd.DataFrame(rows)


def _calibrated_development(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        outer_validation = joined.loc[joined["validation_fold"] == outer_fold].reset_index(drop=True)
        outer_training = joined.loc[joined["validation_fold"] != outer_fold].reset_index(drop=True)
        if outer_training.empty or outer_validation.empty or set(outer_training[preparation.GROUP_COLUMN]) & set(outer_validation[preparation.GROUP_COLUMN]):
            raise CalibrationError(f"Outer fold {outer_fold} has empty data or match overlap.")
        if outer_training[preparation.TARGET_COLUMN].nunique() != 2 or outer_validation[preparation.TARGET_COLUMN].nunique() != 2:
            raise CalibrationError(f"Outer fold {outer_fold} must retain both classes on train and validation sides.")
        inner_splits = grouped_splits(outer_training, INNER_FOLDS, SEED)
        for family in RAW_FAMILIES:
            calibrator = fit_calibrator(family, outer_training, inner_splits)
            with threadpool_limits(limits=1):
                probability = positive_probabilities(calibrator, outer_validation[preparation.FEATURE_COLUMNS])
            variant = f"{family}_sigmoid"
            prediction = _prediction_frame(variant, outer_validation, probability, outer_fold)
            values = phase7.metrics_for(prediction["y_true"], prediction["p_goal"], threshold=THRESHOLD)
            metrics.append({
                "scope": "development", "variant": variant, "fold": outer_fold,
                "train_shot_count": len(outer_training), "validation_shot_count": len(outer_validation),
                "train_match_count": int(outer_training[preparation.GROUP_COLUMN].nunique()),
                "validation_match_count": int(outer_validation[preparation.GROUP_COLUMN].nunique()),
                "train_goal_count": int(outer_training[preparation.TARGET_COLUMN].sum()),
                "train_non_goal_count": int(len(outer_training) - outer_training[preparation.TARGET_COLUMN].sum()),
                "validation_goal_count": int(outer_validation[preparation.TARGET_COLUMN].sum()),
                "validation_non_goal_count": int(len(outer_validation) - outer_validation[preparation.TARGET_COLUMN].sum()),
                **values,
            })
            predictions.append(prediction)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def _select_variants(development_predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = metric_table(development_predictions, "development")
    by_variant = metrics.set_index("variant")
    choices: dict[str, str] = {}
    family_scores = {}
    for family in RAW_FAMILIES:
        raw, calibrated = f"{family}_raw", f"{family}_sigmoid"
        raw_loss = float(by_variant.loc[raw, "log_loss"])
        calibrated_loss = float(by_variant.loc[calibrated, "log_loss"])
        chosen = calibrated if calibrated_loss < raw_loss else raw
        choices[family] = chosen
        family_scores[family] = {"raw_oof_log_loss": raw_loss, "sigmoid_oof_log_loss": calibrated_loss, "selected_variant": chosen}
    candidate = sorted(choices.values(), key=lambda variant: (float(by_variant.loc[variant, "log_loss"]), variant))[0]
    selection = {
        "status": "frozen_before_test_evaluation",
        "family_choices": choices, "family_development_scores": family_scores,
        "global_candidate": candidate,
        "global_candidate_oof_log_loss": float(by_variant.loc[candidate, "log_loss"]),
        "selection_rule": "Lower nested pooled training OOF log loss; ties choose raw per family, then lexical variant name for the global candidate.",
        "test_metrics_used_for_selection": False,
    }
    metrics["family_selected"] = metrics["variant"].isin(choices.values())
    metrics["global_candidate"] = metrics["variant"].eq(candidate)
    return selection, metrics


def _verify_frozen_selection(root: Path, phase7_inputs: dict[str, Any], protocol: dict[str, Any], protocol_bytes: bytes) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    current_protocol, current_protocol_bytes = read_json(root / PROTOCOL_RELATIVE)
    if current_protocol_bytes != protocol_bytes or current_protocol != protocol:
        raise CalibrationError("Phase 8 calibration protocol differs from its predeclared lock.")
    selection, selection_bytes = read_json(root / SELECTION_RELATIVE)
    dev_report, dev_report_bytes = read_json(root / DEV_REPORT_RELATIVE)
    if dev_report.get("status") != "passed" or dev_report.get("test_data_accessed") is not False:
        raise CalibrationError("Development report does not certify training-only preparation.")
    if acquisition.digest(selection_bytes) != dev_report.get("selection_sha256"):
        raise CalibrationError("Frozen selection hash differs from the training-only development report.")
    if selection.get("status") != "frozen_before_test_evaluation" or selection.get("created_before_test_evaluation") is not True or selection.get("test_data_accessed_at_freeze") is not False:
        raise CalibrationError("Selection is missing its pre-test freeze marker.")
    if selection.get("protocol_sha256") != acquisition.digest(protocol_bytes):
        raise CalibrationError("Selection refers to a different calibration protocol.")
    if selection.get("phase7_protocol_sha256") != acquisition.digest(phase7_inputs["protocol_bytes"]):
        raise CalibrationError("Selection refers to a different Phase 7 protocol.")
    if selection.get("phase7_model_manifest_sha256") != acquisition.digest(phase7_inputs["model_manifest_bytes"]):
        raise CalibrationError("Selection refers to different Phase 7 model artifacts.")
    if selection.get("model_manifest_sha256") != dev_report.get("model_manifest_sha256"):
        raise CalibrationError("Selection and development report disagree on Phase 8 model manifest.")
    model_manifest, model_manifest_bytes = read_json(root / MODEL_MANIFEST_RELATIVE)
    if acquisition.digest(model_manifest_bytes) != selection.get("model_manifest_sha256") or model_manifest.get("protocol_sha256") != acquisition.digest(protocol_bytes):
        raise CalibrationError("Final calibrated model manifest differs from frozen selection.")
    for relative, entry in model_manifest.get("pipelines", {}).items():
        verify_entry(root, relative, entry)
    if model_manifest.get("pipelines") != selection.get("final_calibrated_model_hashes"):
        raise CalibrationError("Frozen selection model hashes differ from the Phase 8 model manifest.")
    for relative, entry in dev_report.get("outputs", {}).items():
        verify_entry(root, relative, entry)

    development_relative = f"{TABLE_DIRECTORY}/{TABLE_NAMES['development_predictions']}"
    development_entry = dev_report.get("outputs", {}).get(development_relative)
    if not development_entry or verify_entry(root, development_relative, development_entry) is None:
        raise CalibrationError("Development prediction table is not locked by its report.")
    if development_entry.get("sha256") != selection.get("development_predictions_sha256"):
        raise CalibrationError("Frozen selection does not match the development prediction hash.")
    metrics_relative = f"{TABLE_DIRECTORY}/{TABLE_NAMES['development_metrics']}"
    metrics_entry = dev_report.get("outputs", {}).get(metrics_relative)
    if not metrics_entry:
        raise CalibrationError("Development metrics table is not locked by its report.")
    verify_entry(root, metrics_relative, metrics_entry)
    if metrics_entry.get("sha256") != selection.get("development_metrics_sha256"):
        raise CalibrationError("Frozen selection does not match the development metrics hash.")
    predictions = pd.read_csv(root / development_relative, float_precision="round_trip")
    recomputed, recomputed_metrics = _select_variants(predictions)
    for key in ("family_choices", "global_candidate"):
        if recomputed[key] != selection.get(key) or recomputed[key] != dev_report.get("selection", {}).get(key):
            raise CalibrationError(f"Frozen selection {key} does not reproduce from nested OOF predictions.")
    for family in RAW_FAMILIES:
        expected_scores = recomputed["family_development_scores"][family]
        actual_scores = selection.get("family_development_scores", {}).get(family, {})
        for score_name in ("raw_oof_log_loss", "sigmoid_oof_log_loss"):
            if not np.isclose(float(expected_scores[score_name]), float(actual_scores.get(score_name, np.nan)), rtol=0, atol=1e-14):
                raise CalibrationError("Frozen family choice scores do not reproduce from OOF predictions.")
    candidate_score = float(recomputed_metrics.set_index("variant").loc[selection["global_candidate"], "log_loss"])
    if not np.isclose(candidate_score, float(selection.get("global_candidate_oof_log_loss", np.nan)), rtol=0, atol=1e-14):
        raise CalibrationError("Frozen global candidate score does not reproduce from OOF predictions.")
    return selection, selection_bytes, {"development_report": dev_report, "development_predictions": predictions, "model_manifest": model_manifest}


def load_frozen_test_rows(root: Path, selection: dict[str, Any], selection_bytes: bytes, split: dict[str, Any]) -> pd.DataFrame:
    """Read test rows only after checking the immutable pre-test selection artifact exists."""
    path = root / SELECTION_RELATIVE
    if not path.is_file():
        raise CalibrationError("Frozen selection must exist before test rows are loaded.")
    current = path.read_bytes()
    if current != selection_bytes or json.loads(current.decode("utf-8")) != selection:
        raise CalibrationError("Selection changed before test data access.")
    test_ids = set(split["test_match_ids"])
    rows = preparation.load_training_rows(root, test_ids)
    if len(rows) != split["test_shot_count"] or set(rows[preparation.GROUP_COLUMN]) != test_ids:
        raise CalibrationError("Loaded test rows do not match the locked holdout counts and IDs.")
    return rows


def _reliability_figure(bins: pd.DataFrame, scope: str) -> bytes:
    colors = {"knn_50": "#315A8A", "gaussian_nb": "#398564", "logistic_regression": "#D88735"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.0), facecolor="white", gridspec_kw={"height_ratios": [3, 1]})
    for column, family in enumerate(RAW_FAMILIES):
        axis, count_axis = axes[0, column], axes[1, column]
        subset = bins.loc[bins["variant"] == BASELINE]
        subset = subset.loc[subset["shot_count"] > 0]
        if not subset.empty:
            axis.plot(subset["mean_probability"], subset["observed_goal_rate"], marker="o", linestyle="--", color="#777777", label="Dummy prior baseline")
        histogram_values = []
        histogram_labels = []
        histogram_colors = []
        for variant, style, label in ((f"{family}_raw", "o-", "Raw"), (f"{family}_sigmoid", "s-", "Sigmoid calibrated")):
            variant_bins = bins.loc[bins["variant"] == variant].sort_values("bin_index")
            points = variant_bins.loc[variant_bins["shot_count"] > 0]
            if not points.empty:
                axis.plot(points["mean_probability"], points["observed_goal_rate"], style, color=colors[family], alpha=0.65 if label == "Raw" else 1.0, label=label)
            histogram_values.append(variant_bins["shot_count"].to_numpy(dtype=int))
            histogram_labels.append(label)
            histogram_colors.append(colors[family] if label == "Raw" else "#A8CBB8")
        axis.plot([0, 1], [0, 1], color="#222222", linewidth=1, linestyle=":", label="Perfect calibration")
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        axis.set_xlabel("Mean predicted goal probability")
        axis.set_ylabel("Observed goal rate")
        axis.set_title(phase7.MODEL_LABELS[family], color="#23364D")
        axis.grid(alpha=0.2); axis.legend(fontsize=7, loc="upper left")
        x = np.arange(10)
        width = 0.38
        count_axis.bar(x - width / 2, histogram_values[0], width, label=histogram_labels[0], color=histogram_colors[0])
        count_axis.bar(x + width / 2, histogram_values[1], width, label=histogram_labels[1], color=histogram_colors[1])
        count_axis.set_xticks(x, labels=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)], rotation=45, ha="right", fontsize=7)
        count_axis.set_xlabel("Predicted probability bin")
        count_axis.set_ylabel("Shots")
        count_axis.grid(axis="y", alpha=0.2)
        count_axis.legend(fontsize=7, loc="upper right")
    scope_label = "Training-only development" if scope.startswith("training") else "Locked test after selection freeze"
    fig.suptitle(f"Reliability and bin support\n{scope_label}", color="#23364D", fontsize=12)
    fig.text(0.5, 0.012, "StatsBomb La Liga 2015/16. Ten fixed equal-width bins; empty reliability points omitted. Bin counts show support; shots within matches may depend; no confidence intervals.", ha="center", fontsize=8, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.91])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"})
    plt.close(fig)
    return buffer.getvalue()


def _test_confusion_figure(predictions: pd.DataFrame) -> bytes:
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5), facecolor="white")
    for axis, variant in zip(axes.flat, VARIANTS, strict=False):
        rows = predictions.loc[predictions["variant"] == variant]
        matrix = sklearn.metrics.confusion_matrix(rows["y_true"], rows["y_pred"], labels=[0, 1])
        axis.imshow(matrix, cmap=plt.get_cmap("Blues"))
        title = phase7.MODEL_LABELS.get(variant, phase7.MODEL_LABELS.get(variant.removesuffix("_raw").removesuffix("_sigmoid"), variant))
        axis.set_title(f"{title}\n{variant.rsplit('_', 1)[-1] if variant.endswith(('raw', 'sigmoid')) else 'baseline'}", fontsize=9, color="#23364D")
        axis.set_xticks([0, 1], labels=["0 non-goal", "1 goal"]); axis.set_yticks([0, 1], labels=["0 non-goal", "1 goal"])
        axis.set_xlabel("Predicted class"); axis.set_ylabel("Actual class")
        for (row, col), count in np.ndenumerate(matrix):
            axis.text(col, row, str(int(count)), ha="center", va="center", color="white" if count > matrix.max() * 0.55 else "#23364D", fontsize=11)
    axes.flat[-1].axis("off")
    fig.suptitle("Frozen model variants on the locked holdout (actual rows, predicted columns)", color="#23364D", fontsize=12)
    fig.text(0.5, 0.025, "Threshold: P(goal) >= 0.5. La Liga 2015/16 StatsBomb open data.", ha="center", fontsize=8, color="#23364D")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.92])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"})
    plt.close(fig)
    return buffer.getvalue()


def _load_saved_model(root: Path, relative: str, entry: dict[str, Any]) -> Any:
    payload = verify_entry(root, relative, entry)
    try:
        return joblib.load(io.BytesIO(payload))
    except Exception as exc:
        raise CalibrationError(f"Cannot load saved pipeline {relative}: {exc}") from exc


def run_evaluation(root: Path) -> dict[str, Any]:
    root = root.resolve()
    phase7_inputs = verify_phase7(root)
    protocol, protocol_bytes = make_protocol(phase7_inputs)
    selection, selection_bytes, locked = _verify_frozen_selection(root, phase7_inputs, protocol, protocol_bytes)
    # This is the first operation that reads holdout feature or target columns.
    test_rows = load_frozen_test_rows(root, selection, selection_bytes, phase7_inputs["split"])
    if test_rows[preparation.TARGET_COLUMN].nunique() != 2:
        raise CalibrationError("Locked test partition must contain both target classes.")

    phase7_manifest = phase7_inputs["model_manifest"]
    test_parts = []
    model_map = [("prior_baseline", BASELINE), *((family, f"{family}_raw") for family in RAW_FAMILIES)]
    for model_name, variant in model_map:
        relative = f"{phase7.MODEL_DIRECTORY}/{model_name}.joblib"
        estimator = _load_saved_model(root, relative, phase7_manifest["pipelines"][relative])
        with threadpool_limits(limits=1):
            probabilities = positive_probabilities(estimator, test_rows[preparation.FEATURE_COLUMNS])
        test_parts.append(_prediction_frame(variant, test_rows, probabilities))
    for family in RAW_FAMILIES:
        relative = f"{MODEL_DIRECTORY}/{family}_sigmoid.joblib"
        estimator = _load_saved_model(root, relative, locked["model_manifest"]["pipelines"][relative])
        with threadpool_limits(limits=1):
            probabilities = positive_probabilities(estimator, test_rows[preparation.FEATURE_COLUMNS])
        test_parts.append(_prediction_frame(f"{family}_sigmoid", test_rows, probabilities))
    test_predictions = pd.concat(test_parts, ignore_index=True)
    if set(test_predictions["variant"]) != set(VARIANTS) or any(
        len(test_predictions.loc[test_predictions["variant"] == variant]) != len(test_rows)
        or test_predictions.loc[test_predictions["variant"] == variant, "event_id"].duplicated().any()
        for variant in VARIANTS
    ):
        raise CalibrationError("Frozen test predictions do not cover every test event exactly once per variant.")
    test_predictions["family_selected"] = test_predictions["variant"].isin(selection["family_choices"].values())
    test_predictions["global_candidate"] = test_predictions["variant"].eq(selection["global_candidate"])
    test_metrics = metric_table(test_predictions, "test", selection)
    development_bins_path = root / TABLE_DIRECTORY / TABLE_NAMES["development_reliability_bins"]
    development_bins = pd.read_csv(development_bins_path, float_precision="round_trip")
    if set(development_bins["scope"]) != {"development"}:
        raise CalibrationError("Reliability table changed before frozen test evaluation.")
    test_bins = reliability_bins(test_predictions, "test")
    test_reliability_figure = _reliability_figure(test_bins, "locked test evaluation after selection freeze")
    confusion_figure = _test_confusion_figure(test_predictions)
    outputs = {
        f"{TABLE_DIRECTORY}/{TABLE_NAMES['test_predictions']}": _csv_bytes(test_predictions),
        f"{TABLE_DIRECTORY}/{TABLE_NAMES['test_metrics']}": _csv_bytes(test_metrics),
        f"{TABLE_DIRECTORY}/{TABLE_NAMES['test_reliability_bins']}": _csv_bytes(test_bins),
        f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['test_reliability']}": test_reliability_figure,
        f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['test_confusion']}": confusion_figure,
    }
    output_manifest = {relative: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for relative, payload in outputs.items()}
    report = {
        "status": "passed", "scope": "fixed held-out evaluation after selection freeze; reruns reproduce the frozen evaluation",
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "selection_sha256": acquisition.digest(selection_bytes),
        "phase7_protocol_sha256": acquisition.digest(phase7_inputs["protocol_bytes"]),
        "phase7_model_manifest_sha256": acquisition.digest(phase7_inputs["model_manifest_bytes"]),
        "source_shot_csv_sha256": protocol["source_shot_csv_sha256"],
        "holdout_membership_sha256": phase7_inputs["split"]["membership_csv_sha256"],
        "test_match_count": int(test_rows[preparation.GROUP_COLUMN].nunique()),
        "test_shot_count": int(len(test_rows)), "test_goal_count": int(test_rows[preparation.TARGET_COLUMN].sum()),
        "test_non_goal_count": int(len(test_rows) - test_rows[preparation.TARGET_COLUMN].sum()),
        "threshold": THRESHOLD,
        "selection": {
            "family_choices": selection["family_choices"], "global_candidate": selection["global_candidate"],
            "selection_basis": "nested training-only OOF log loss, frozen before this test evaluation",
            "test_metrics_changed_selection": False,
        },
        "test_metrics": test_metrics.to_dict(orient="records"),
        "limitations": [
            "One historical league season; results may not generalize to other seasons or competitions.",
            "Shots within matches can be related; match grouping reduces leakage but does not make shots independent.",
            "Sigmoid calibration is not a guarantee of calibration on future data.",
            "No Monte Carlo score simulation is included in this phase.",
        ],
        "outputs": output_manifest,
    }
    report_payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for relative, payload in outputs.items():
        acquisition.atomic_write(root / relative, payload)
    acquisition.atomic_write(root / REPORT_RELATIVE, report_payload)
    return report


def run(root: Path, stage: str = "all") -> dict[str, Any]:
    if stage == "develop":
        return run_development(root)
    if stage == "evaluate":
        return run_evaluation(root)
    if stage == "all":
        development = run_development(root)
        evaluation = run_evaluation(root)
        return {"development_status": development["status"], "evaluation": evaluation}
    raise CalibrationError(f"Unknown stage {stage!r}; choose develop, evaluate, or all.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("develop", "evaluate", "all"), default="all")
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.stage)
    except (CalibrationError, preparation.PreparationError, holdout.SplitError, phase7.ModelingError, OSError, ValueError, KeyError, Warning) as exc:
        print(f"Calibration/evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"stage": args.stage, "status": "passed", "report": str((args.root.resolve() / (DEV_REPORT_RELATIVE if args.stage == 'develop' else REPORT_RELATIVE)))}, indent=2))
    return 0


def _fit_final_calibrators(training: pd.DataFrame) -> dict[str, Any]:
    inner_splits = grouped_splits(training, INNER_FOLDS, SEED)
    fitted = {}
    for family in RAW_FAMILIES:
        calibrator = fit_calibrator(family, training, inner_splits)
        with threadpool_limits(limits=1):
            probability = positive_probabilities(calibrator, training[preparation.FEATURE_COLUMNS])
        buffer = io.BytesIO()
        joblib.dump(calibrator, buffer, compress=3, protocol=5)
        payload = buffer.getvalue()
        restored = joblib.load(io.BytesIO(payload))
        with threadpool_limits(limits=1):
            smoke = positive_probabilities(restored, training[preparation.FEATURE_COLUMNS])
        np.testing.assert_allclose(smoke, probability, rtol=0, atol=1e-14)
        fitted[family] = {"estimator": calibrator, "payload": payload, "training_probability": probability}
    return fitted


def run_development(root: Path) -> dict[str, Any]:
    root = root.resolve()
    phase7_inputs = verify_phase7(root)
    protocol, protocol_bytes = make_protocol(phase7_inputs)
    lock_json(root, PROTOCOL_RELATIVE, protocol, protocol_bytes)  # Freeze before any calibration fit.
    training, joined = _load_training(root, phase7_inputs["split"])
    phase7_oof, p7_fold_metrics, p7_training_predictions = _load_phase7_predictions(root, phase7_inputs, training, joined)

    raw_predictions = _raw_development_predictions(phase7_oof, joined)
    calibrated_predictions, calibrated_fold_metrics = _calibrated_development(joined)
    development_predictions = pd.concat([raw_predictions, calibrated_predictions], ignore_index=True)
    if set(development_predictions["variant"]) != set(VARIANTS) or any(
        len(development_predictions.loc[development_predictions["variant"] == variant]) != len(training)
        for variant in VARIANTS
    ):
        raise CalibrationError("Nested development OOF predictions do not cover every training row once per variant.")
    development_predictions["y_true"] = development_predictions["y_true"].astype(int)
    if development_predictions.duplicated(["variant", "event_id"]).any():
        raise CalibrationError("A training event has duplicate development predictions for a model variant.")

    selection, development_metrics = _select_variants(development_predictions)
    raw_folds = _raw_fold_metrics(p7_fold_metrics)
    fold_metrics = pd.concat([raw_folds, calibrated_fold_metrics], ignore_index=True)
    fold_metrics["family_selected"] = fold_metrics["variant"].isin(selection["family_choices"].values())
    fold_metrics["global_candidate"] = fold_metrics["variant"].eq(selection["global_candidate"])
    reliability = reliability_bins(development_predictions, "development")
    development_figure = _reliability_figure(reliability, "training-only nested OOF development")

    final_calibrators = _fit_final_calibrators(training)
    model_payloads = {f"{MODEL_DIRECTORY}/{family}_sigmoid.joblib": final_calibrators[family]["payload"] for family in RAW_FAMILIES}
    model_entries = {relative: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for relative, payload in model_payloads.items()}
    model_manifest = {
        "status": "passed", "scope": "final sigmoid pipelines fitted on all locked training matches only",
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "phase7_protocol_sha256": protocol["phase7_protocol_sha256"],
        "source_shot_csv_sha256": protocol["source_shot_csv_sha256"],
        "cv_membership_sha256": protocol["cv_membership_sha256"],
        "model_contract_sha256": protocol["model_contract_sha256"],
        "versions": protocol["versions"], "pipelines": model_entries,
    }
    model_manifest_payload = (json.dumps(model_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    manifest_path = root / MODEL_MANIFEST_RELATIVE
    if manifest_path.exists():
        if manifest_path.read_bytes() != model_manifest_payload:
            raise CalibrationError("Existing Phase 8 model manifest differs from this deterministic training run.")
        for relative, payload in model_payloads.items():
            if not (root / relative).exists() or acquisition.digest((root / relative).read_bytes()) != acquisition.digest(payload):
                raise CalibrationError(f"Saved Phase 8 model {relative} differs from its manifest.")
    else:
        for relative, payload in model_payloads.items():
            acquisition.atomic_write(root / relative, payload)
        acquisition.atomic_write(manifest_path, model_manifest_payload)

    # Final training-set resubstitution metrics are kept separate from nested OOF metrics.
    train_prediction_parts = []
    for model_name in ["prior_baseline", *RAW_FAMILIES]:
        rows = p7_training_predictions.loc[p7_training_predictions["model"] == model_name].copy()
        variant = BASELINE if model_name == "prior_baseline" else f"{model_name}_raw"
        rows["event_id"] = rows["event_id"].astype(str)
        labels = rows.set_index("event_id").reindex(training[preparation.ID_COLUMN].astype(str))["y_true"].to_numpy(dtype=int)
        if not np.array_equal(labels, training[preparation.TARGET_COLUMN].to_numpy(dtype=int)):
            raise CalibrationError(f"Phase 7 resubstitution labels mismatch for {model_name}.")
        probabilities = rows.set_index("event_id").reindex(training[preparation.ID_COLUMN].astype(str))["p_goal"].to_numpy(dtype=float)
        train_prediction_parts.append(_prediction_frame(variant, training, probabilities))
    for family in RAW_FAMILIES:
        train_prediction_parts.append(_prediction_frame(f"{family}_sigmoid", training, final_calibrators[family]["training_probability"]))
    training_predictions = pd.concat(train_prediction_parts, ignore_index=True)
    training_metrics = metric_table(training_predictions, "training_resubstitution", selection)
    training_metrics["interpretation"] = "Optimistic same-rows-used-to-fit diagnostic; not validation or test performance."
    development_metrics["selection_rule"] = "Nested pooled training OOF log loss; family ties choose raw."

    tables = {
        TABLE_NAMES["development_predictions"]: development_predictions,
        TABLE_NAMES["development_metrics"]: development_metrics,
        TABLE_NAMES["fold_metrics"]: fold_metrics,
        TABLE_NAMES["development_reliability_bins"]: reliability,
        TABLE_NAMES["training_metrics"]: training_metrics,
    }
    table_payloads = {f"{TABLE_DIRECTORY}/{name}": _csv_bytes(frame) for name, frame in tables.items()}
    files = {relative: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for relative, payload in table_payloads.items()}
    development_figure_relative = f"{FIGURE_DIRECTORY}/{FIGURE_NAMES['development_reliability']}"
    files[development_figure_relative] = {"bytes": len(development_figure), "sha256": acquisition.digest(development_figure)}
    dev_report = {
        "status": "passed", "scope": "development on locked training partition only",
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "phase7_oof_sha256": protocol["phase7_oof_sha256"],
        "training_shot_count": int(len(training)), "training_match_count": int(training[preparation.GROUP_COLUMN].nunique()),
        "training_goal_count": int(training[preparation.TARGET_COLUMN].sum()),
        "training_non_goal_count": int(len(training) - training[preparation.TARGET_COLUMN].sum()),
        "selection": selection,
        "development_metrics": development_metrics.to_dict(orient="records"),
        "outputs": files,
        "model_manifest_sha256": acquisition.digest(model_manifest_payload),
        "test_data_accessed": False,
    }
    selection.update({
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "source_shot_csv_sha256": protocol["source_shot_csv_sha256"],
        "phase7_protocol_sha256": protocol["phase7_protocol_sha256"],
        "phase7_model_manifest_sha256": protocol["phase7_model_manifest_sha256"],
        "phase7_oof_sha256": protocol["phase7_oof_sha256"],
        "cv_membership_sha256": protocol["cv_membership_sha256"],
        "model_manifest_sha256": acquisition.digest(model_manifest_payload),
        "final_calibrated_model_hashes": model_entries,
        "development_predictions_sha256": files[f"{TABLE_DIRECTORY}/{TABLE_NAMES['development_predictions']}"]["sha256"],
        "development_metrics_sha256": files[f"{TABLE_DIRECTORY}/{TABLE_NAMES['development_metrics']}"]["sha256"],
        "training_shot_count": len(training), "training_match_count": int(training[preparation.GROUP_COLUMN].nunique()),
        "created_before_test_evaluation": True,
        "test_data_accessed_at_freeze": False,
    })
    selection_payload = (json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    validate_locked_candidate(root, SELECTION_RELATIVE, selection, selection_payload)
    dev_report["selection_sha256"] = acquisition.digest(selection_payload)
    dev_report_payload = (json.dumps(dev_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for relative, payload in table_payloads.items():
        acquisition.atomic_write(root / relative, payload)
    acquisition.atomic_write(root / development_figure_relative, development_figure)
    acquisition.atomic_write(root / DEV_REPORT_RELATIVE, dev_report_payload)
    lock_json(root, SELECTION_RELATIVE, selection, selection_payload)
    return dev_report


if __name__ == "__main__":
    raise SystemExit(main())



