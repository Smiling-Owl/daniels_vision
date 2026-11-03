"""Prepare training-only grouped cross-validation folds and preprocessing diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scripts import acquire_data as acquisition
    from scripts import reserve_holdout as holdout
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import reserve_holdout as holdout

SHOT_RELATIVE = "data/processed/shots.csv"
CONTRACT_RELATIVE = "docs/model_columns.json"
CV_MANIFEST_RELATIVE = "docs/cv_manifest.json"
CV_MEMBERSHIP_RELATIVE = "data/processed/cv_membership.csv"
CV_SUMMARY_RELATIVE = "reports/tables/phase06_cv_summary.csv"
PHASE_REPORT_RELATIVE = "docs/phase_06_preparation_report.json"
SEED = 42
FOLDS = 10
ALGORITHM = "StratifiedGroupKFold(shuffle=True, random_state=42)"
FEATURE_COLUMNS = [
    "distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique",
    "play_pattern", "under_pressure", "first_time", "open_goal",
]
NUMERIC_COLUMNS = ["distance_sb_units", "angle_sb_radians"]
CATEGORICAL_COLUMNS = ["body_part", "shot_type", "technique", "play_pattern"]
BINARY_COLUMNS = ["under_pressure", "first_time", "open_goal"]
TARGET_COLUMN = "is_goal"
GROUP_COLUMN = "match_id"
ID_COLUMN = "event_id"
TRAIN_COLUMNS = [ID_COLUMN, GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


class PreparationError(RuntimeError):
    """Input data or a locked cross-validation plan is invalid."""


def make_preprocessor() -> ColumnTransformer:
    """Return a fresh, unfitted transform for the exact approved feature contract."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    features = contract.get("features")
    if not isinstance(features, list):
        raise PreparationError("Feature contract must contain a feature list.")
    columns = [item.get("column") for item in features if isinstance(item, dict)]
    expected_types = {
        **{column: "numeric" for column in NUMERIC_COLUMNS},
        **{column: "categorical" for column in CATEGORICAL_COLUMNS},
        **{column: "binary" for column in BINARY_COLUMNS},
    }
    actual_types = {item.get("column"): item.get("type") for item in features if isinstance(item, dict)}
    if columns != FEATURE_COLUMNS or actual_types != expected_types:
        raise PreparationError("Feature contract must match the approved two numeric, four categorical, and three binary inputs exactly.")
    if contract.get("target", {}).get("column") != TARGET_COLUMN or contract.get("group_column") != GROUP_COLUMN:
        raise PreparationError("Target or match grouping differs from the approved model contract.")
    return contract


def load_contract(root: Path) -> tuple[dict[str, Any], bytes]:
    path = root / CONTRACT_RELATIVE
    try:
        raw = path.read_bytes()
        contract = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cannot read model feature contract: {exc}") from exc
    if not isinstance(contract, dict):
        raise PreparationError("Model feature contract must be a JSON object.")
    return validate_contract(contract), raw


def load_training_rows(root: Path, train_match_ids: set[int]) -> pd.DataFrame:
    """Stream the CSV and retain only locked training matches and approved columns."""
    path = root / SHOT_RELATIVE
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not set(TRAIN_COLUMNS).issubset(reader.fieldnames):
                raise PreparationError("Shot table is missing columns required by the model contract.")
            for row in reader:
                try:
                    match_id = int(row[GROUP_COLUMN])
                except (TypeError, ValueError) as exc:
                    raise PreparationError("Shot table contains an invalid match identifier.") from exc
                if match_id not in train_match_ids:
                    continue
                rows.append({column: row[column] for column in TRAIN_COLUMNS})
    except OSError as exc:
        raise PreparationError(f"Cannot read shot CSV: {exc}") from exc
    if not rows:
        raise PreparationError("No rows from locked training matches were found.")
    frame = pd.DataFrame(rows)
    try:
        frame[GROUP_COLUMN] = pd.to_numeric(frame[GROUP_COLUMN], errors="raise").astype("int64")
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
        frame[BINARY_COLUMNS] = frame[BINARY_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise PreparationError(f"Training rows contain a malformed numeric, binary, or target value: {exc}") from exc
    if frame[TRAIN_COLUMNS].isna().any().any():
        missing = frame[TRAIN_COLUMNS].columns[frame[TRAIN_COLUMNS].isna().any()].tolist()
        raise PreparationError(f"Training rows have missing required values: {missing}.")
    for column in [TARGET_COLUMN, *BINARY_COLUMNS]:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise PreparationError(f"Training indicator/target {column} must contain finite integer values.")
        if not set(values).issubset({0.0, 1.0}):
            raise PreparationError(f"Training indicator/target {column} must contain only 0 and 1.")
        frame[column] = frame[column].astype("int64")
    if frame[ID_COLUMN].astype(str).str.strip().eq("").any() or frame[ID_COLUMN].duplicated().any():
        raise PreparationError("Training event identifiers must be non-empty and unique.")
    if any(frame[column].astype(str).str.strip().eq("").any() for column in CATEGORICAL_COLUMNS):
        raise PreparationError("Training categorical features cannot be blank.")
    if not np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
        raise PreparationError("Training numeric features must be finite.")
    if set(frame[GROUP_COLUMN]) != train_match_ids:
        raise PreparationError("Training rows do not match the locked training match IDs.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise PreparationError("Training partition must contain both target classes.")
    return frame.reset_index(drop=True)


def _membership_bytes(frame: pd.DataFrame, assignments: list[int]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([ID_COLUMN, GROUP_COLUMN, "validation_fold"])
    writer.writerows(
        (event_id, int(match_id), int(fold))
        for event_id, match_id, fold in zip(frame[ID_COLUMN], frame[GROUP_COLUMN], assignments, strict=True)
    )
    return output.getvalue().encode("utf-8")


def _summary_bytes(records: list[dict[str, Any]]) -> bytes:
    columns = [
        "fold", "train_shot_count", "validation_shot_count", "train_match_count",
        "validation_match_count", "train_goal_count", "train_non_goal_count",
        "validation_goal_count", "validation_non_goal_count",
        "train_goal_rate", "validation_goal_rate", "transformed_feature_count",
    ]
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({column: record[column] for column in columns})
    return output.getvalue().encode("utf-8")


def evaluate_fold(frame: pd.DataFrame, train_indices: np.ndarray, validation_indices: np.ndarray, fold: int) -> tuple[dict[str, Any], np.ndarray, ColumnTransformer]:
    train = frame.iloc[train_indices]
    validation = frame.iloc[validation_indices]
    if set(train[GROUP_COLUMN]) & set(validation[GROUP_COLUMN]):
        raise PreparationError(f"Fold {fold} has match overlap between training and validation.")
    if train[TARGET_COLUMN].nunique() != 2 or validation[TARGET_COLUMN].nunique() != 2:
        raise PreparationError(f"Fold {fold} training and validation partitions must both contain both classes.")
    X_train, X_validation = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    preprocessor = make_preprocessor()
    transformed_train = preprocessor.fit_transform(X_train, y_train)
    transformed_validation = preprocessor.transform(X_validation)
    if not np.isfinite(transformed_train).all() or not np.isfinite(transformed_validation).all():
        raise PreparationError(f"Fold {fold} preprocessing produced non-finite values.")
    if transformed_train.shape[1] != transformed_validation.shape[1]:
        raise PreparationError(f"Fold {fold} train/validation feature dimensions differ.")
    fitted_categories = preprocessor.named_transformers_["categorical"].categories_
    unknown_counts = {
        column: int((~validation[column].isin(categories)).sum())
        for column, categories in zip(CATEGORICAL_COLUMNS, fitted_categories, strict=True)
    }
    record = {
        "fold": fold,
        "train_shot_count": int(len(train)), "validation_shot_count": int(len(validation)),
        "train_match_count": int(train[GROUP_COLUMN].nunique()),
        "validation_match_count": int(validation[GROUP_COLUMN].nunique()),
        "train_goal_count": int(train[TARGET_COLUMN].sum()),
        "train_non_goal_count": int(len(train) - train[TARGET_COLUMN].sum()),
        "validation_goal_count": int(validation[TARGET_COLUMN].sum()),
        "validation_non_goal_count": int(len(validation) - validation[TARGET_COLUMN].sum()),
        "train_goal_rate": float(train[TARGET_COLUMN].mean()),
        "validation_goal_rate": float(validation[TARGET_COLUMN].mean()),
        "transformed_feature_count": int(transformed_train.shape[1]),
        "unknown_category_counts": unknown_counts,
    }
    return record, transformed_validation, preprocessor


def prepare(root: Path) -> dict[str, Any]:
    root = root.resolve()
    split_manifest = holdout.reserve(root)
    contract, contract_bytes = load_contract(root)
    shot_path = root / SHOT_RELATIVE
    shot_hash = acquisition.digest(shot_path.read_bytes())
    train_ids, test_ids = set(split_manifest["train_match_ids"]), set(split_manifest["test_match_ids"])
    if train_ids & test_ids or not train_ids or not test_ids:
        raise PreparationError("Locked training and holdout match IDs overlap or are empty.")
    frame = load_training_rows(root, train_ids)
    expected_train_shots = int(split_manifest["train_shot_count"])
    if len(frame) != expected_train_shots:
        raise PreparationError("Loaded training row count differs from the locked split.")

    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    assignments = np.full(len(frame), -1, dtype=np.int64)
    records: list[dict[str, Any]] = []
    validation_match_ids: list[int] = []
    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN], groups=frame[GROUP_COLUMN]), start=1
    ):
        if (assignments[validation_index] != -1).any():
            raise PreparationError("A training event was assigned to multiple validation folds.")
        assignments[validation_index] = fold_number
        validation_match_ids.extend(int(value) for value in frame.iloc[validation_index][GROUP_COLUMN].unique())
        record, _, _ = evaluate_fold(frame, train_index, validation_index, fold_number)
        if set(frame.iloc[train_index][GROUP_COLUMN]) & test_ids or set(frame.iloc[validation_index][GROUP_COLUMN]) & test_ids:
            raise PreparationError("A held-out test match entered training-only cross-validation.")
        records.append(record)
    if (assignments < 1).any() or set(assignments) != set(range(1, FOLDS + 1)):
        raise PreparationError("Cross-validation folds do not cover every training shot exactly once.")
    if len(validation_match_ids) != len(set(validation_match_ids)) or set(validation_match_ids) != train_ids:
        raise PreparationError("Each training match must appear in validation exactly once across the folds.")
    if sum(record["validation_shot_count"] for record in records) != len(frame):
        raise PreparationError("Validation folds do not account for every training shot exactly once.")
    if sum(record["validation_match_count"] for record in records) != len(train_ids):
        raise PreparationError("Validation folds do not account for every training match exactly once.")
    membership = _membership_bytes(frame, assignments.tolist())
    cv_manifest_path = root / CV_MANIFEST_RELATIVE
    membership_path = root / CV_MEMBERSHIP_RELATIVE
    summary_path = root / CV_SUMMARY_RELATIVE
    phase_report_path = root / PHASE_REPORT_RELATIVE
    split_manifest_hash = acquisition.digest((root / holdout.SPLIT_MANIFEST_RELATIVE).read_bytes())
    contract_hash = acquisition.digest(contract_bytes)
    membership_hash = acquisition.digest(membership)
    manifest = {
        "status": "locked", "algorithm": ALGORITHM, "n_splits": FOLDS, "seed": SEED,
        "shot_csv_sha256": shot_hash,
        "holdout_manifest_sha256": split_manifest_hash,
        "holdout_membership_csv_sha256": split_manifest["membership_csv_sha256"],
        "model_contract_sha256": contract_hash,
        "scikit_learn_version": sklearn.__version__,
        "training_match_count": int(frame[GROUP_COLUMN].nunique()),
        "training_shot_count": int(len(frame)),
        "cv_membership_sha256": membership_hash,
        "validation_folds": records,
    }
    summary_payload = _summary_bytes(records)
    report = {
        "status": "passed", "scope": "locked training partition only",
        "shot_csv_sha256": shot_hash, "holdout_manifest_sha256": split_manifest_hash,
        "holdout_membership_csv_sha256": split_manifest["membership_csv_sha256"],
        "model_contract_sha256": contract_hash, "cv_membership_sha256": membership_hash,
        "algorithm": ALGORITHM, "n_splits": FOLDS, "seed": SEED,
        "scikit_learn_version": sklearn.__version__,
        "training_match_count": int(frame[GROUP_COLUMN].nunique()), "training_shot_count": int(len(frame)),
        "target": TARGET_COLUMN, "group_column": GROUP_COLUMN,
        "features": FEATURE_COLUMNS,
        "feature_types": {"numeric": NUMERIC_COLUMNS, "categorical": CATEGORICAL_COLUMNS, "binary": BINARY_COLUMNS},
        "preprocessing": {
            "numeric": "StandardScaler fitted separately on each fold's training rows",
            "categorical": "OneHotEncoder(handle_unknown='ignore', sparse_output=False) fitted separately on each fold's training rows",
            "binary": "passed through as 0/1",
            "missing_values": "not imputed; missing or invalid training inputs fail validation",
            "models_fitted": False,
            "transformed_matrices_exported": False,
            "interpretation_note": "Dense one-hot features allow GaussianNB compatibility, but GaussianNB treats encoded indicators with a conditional Gaussian approximation. KNN distances are affected by scaling and the added one-hot dimensions.",
        },
        "validation_folds": records,
    }
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    report_payload = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

    # Existing folds are adopted only when every locked input and exact materialized output agrees.
    artifacts = [(membership_path, membership), (cv_manifest_path, manifest_payload)]
    exists = [path.exists() for path, _ in artifacts]
    if any(exists) and not all(exists):
        raise PreparationError("CV lock and membership must either both exist or both be absent; refusing partial adoption.")
    if all(exists):
        try:
            old_manifest_bytes = cv_manifest_path.read_bytes()
            old_manifest = json.loads(old_manifest_bytes.decode("utf-8"))
            old_membership = membership_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"Cannot verify existing CV lock: {exc}") from exc
        for key in (
            "shot_csv_sha256", "holdout_manifest_sha256", "holdout_membership_csv_sha256",
            "model_contract_sha256", "scikit_learn_version", "cv_membership_sha256", "algorithm", "seed", "n_splits",
        ):
            if old_manifest.get(key) != manifest.get(key):
                raise PreparationError(f"Existing CV lock differs for {key}; refusing to regenerate folds.")
        if old_manifest_bytes != manifest_payload:
            raise PreparationError("Existing CV lock contents differ from deterministic preparation; refusing to adopt changes.")
        if old_membership != membership or acquisition.digest(old_membership) != old_manifest.get("cv_membership_sha256"):
            raise PreparationError("Existing CV membership differs from the locked folds.")
    else:
        acquisition.atomic_write(membership_path, membership)
        acquisition.atomic_write(cv_manifest_path, manifest_payload)
    acquisition.atomic_write(summary_path, summary_payload)
    acquisition.atomic_write(phase_report_path, report_payload)
    return report


def load_cv_membership(root: Path) -> pd.DataFrame:
    """Load only the membership map whose bytes match its lock and current source contracts."""
    root = root.resolve()
    split = holdout.reserve(root)
    _, contract_bytes = load_contract(root)
    manifest_path, membership_path = root / CV_MANIFEST_RELATIVE, root / CV_MEMBERSHIP_RELATIVE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = membership_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cannot read CV lock and membership: {exc}") from exc
    checks = {
        "shot_csv_sha256": acquisition.digest((root / SHOT_RELATIVE).read_bytes()),
        "holdout_manifest_sha256": acquisition.digest((root / holdout.SPLIT_MANIFEST_RELATIVE).read_bytes()),
        "holdout_membership_csv_sha256": split["membership_csv_sha256"],
        "model_contract_sha256": acquisition.digest(contract_bytes),
        "scikit_learn_version": sklearn.__version__,
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise PreparationError(f"CV lock no longer matches current {key}.")
    if manifest.get("status") != "locked" or manifest.get("algorithm") != ALGORITHM or manifest.get("seed") != SEED or manifest.get("n_splits") != FOLDS:
        raise PreparationError("CV manifest status, algorithm, seed, or fold count is invalid.")
    if acquisition.digest(payload) != manifest.get("cv_membership_sha256"):
        raise PreparationError("CV membership hash differs from its lock.")
    frame = pd.read_csv(StringIO(payload.decode("utf-8")), dtype={ID_COLUMN: "string", GROUP_COLUMN: "int64", "validation_fold": "int64"})
    if list(frame.columns) != [ID_COLUMN, GROUP_COLUMN, "validation_fold"] or frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].str.strip().eq("").any() or frame[ID_COLUMN].duplicated().any():
        raise PreparationError("CV membership table has invalid columns or duplicate events.")
    if set(frame[GROUP_COLUMN]) != set(split["train_match_ids"]):
        raise PreparationError("CV membership contains matches outside the locked training partition.")
    if set(frame[GROUP_COLUMN]) & set(split["test_match_ids"]):
        raise PreparationError("CV membership includes a held-out test match.")
    if set(frame["validation_fold"]) != set(range(1, FOLDS + 1)):
        raise PreparationError("CV membership fold labels are incomplete or invalid.")
    if len(frame) != manifest.get("training_shot_count") or frame[GROUP_COLUMN].nunique() != manifest.get("training_match_count"):
        raise PreparationError("CV membership row or match count differs from the lock.")
    if frame.groupby(GROUP_COLUMN)["validation_fold"].nunique().ne(1).any():
        raise PreparationError("A match is assigned to multiple validation folds.")
    training = load_training_rows(root, set(split["train_match_ids"]))
    expected = dict(zip(training[ID_COLUMN], training[GROUP_COLUMN], strict=True))
    actual = dict(zip(frame[ID_COLUMN], frame[GROUP_COLUMN], strict=True))
    if actual != expected:
        raise PreparationError("CV membership event-to-match mapping differs from the locked training rows.")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = prepare(args.root)
    except (PreparationError, holdout.SplitError, OSError, ValueError, KeyError) as exc:
        print(f"Modeling preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "training_match_count", "training_shot_count", "n_splits", "cv_membership_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
