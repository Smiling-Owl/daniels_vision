"""Train fixed, untuned models with fold-local preprocessing on locked training data."""
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
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

try:
    from scripts import acquire_data as acquisition
    from scripts import prepare_modeling as preparation
    from scripts import reserve_holdout as holdout
except ImportError:  # pragma: no cover
    import acquire_data as acquisition
    import prepare_modeling as preparation
    import reserve_holdout as holdout

PROTOCOL_RELATIVE = "docs/phase_07_protocol.json"
REPORT_RELATIVE = "docs/phase_07_model_report.json"
MODEL_DIRECTORY = "models/phase07"
MODEL_MANIFEST_RELATIVE = f"{MODEL_DIRECTORY}/manifest.json"
TABLE_DIRECTORY = "reports/tables"
FIGURE_DIRECTORY = "reports/figures"
THRESHOLD = 0.5
EPSILON_DESCRIPTION = "Log loss uses natural logarithms; sklearn.metrics.log_loss clips probabilities to machine epsilon for numerical stability. No manual probability calibration is applied."
METRIC_NAMES = ["log_loss", "brier_score", "average_precision", "roc_auc", "accuracy", "precision", "recall", "f1"]
COUNT_NAMES = ["tn", "fp", "fn", "tp", "predicted_goal_count", "predicted_non_goal_count", "exact_p0_count", "exact_p1_count"]
MODEL_LABELS = {
    "prior_baseline": "Dummy prior baseline", "knn_50": "KNN (k=50)",
    "gaussian_nb": "Gaussian Naive Bayes", "logistic_regression": "Logistic regression",
}
MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "prior_baseline": {"estimator": "DummyClassifier", "strategy": "prior"},
    "knn_50": {"estimator": "KNeighborsClassifier", "n_neighbors": 50, "weights": "uniform", "algorithm": "brute", "n_jobs": 1},
    "gaussian_nb": {"estimator": "GaussianNB", "var_smoothing": 1e-9},
    "logistic_regression": {"estimator": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
}
TABLE_NAMES = {
    "fold_metrics": "phase07_fold_metrics.csv",
    "model_comparison": "phase07_model_comparison.csv",
    "oof_predictions": "phase07_oof_predictions.csv",
    "training_predictions": "phase07_training_predictions.csv",
}
FIGURE_NAMES = {
    "model_comparison": "phase07_model_comparison.png",
    "oof_confusion": "phase07_oof_confusion_matrices.png",
}


class ModelingError(RuntimeError):
    """Locked training inputs or model outputs failed validation."""


def assert_locked_payload(existing: bytes, expected: bytes, description: str) -> None:
    try:
        json.loads(existing.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelingError(f"Existing {description} is not valid UTF-8 JSON: {exc}") from exc
    if existing != expected:
        raise ModelingError(f"Existing {description} differs from the predeclared content; refusing replacement.")


def protocol_payload(root: Path) -> tuple[dict[str, Any], bytes]:
    split = holdout.reserve(root)
    preparation.load_cv_membership(root)  # Verify the locked event-to-fold map before protocol lock.
    cv_manifest_path = root / preparation.CV_MANIFEST_RELATIVE
    protocol = {
        "status": "predeclared_untuned_protocol",
        "scope": "locked training partition only; held-out test partition excluded",
        "source_shot_csv_sha256": acquisition.digest((root / preparation.SHOT_RELATIVE).read_bytes()),
        "holdout_manifest_sha256": acquisition.digest((root / holdout.SPLIT_MANIFEST_RELATIVE).read_bytes()),
        "holdout_membership_sha256": split["membership_csv_sha256"],
        "cv_manifest_sha256": acquisition.digest(cv_manifest_path.read_bytes()),
        "cv_membership_sha256": acquisition.digest((root / preparation.CV_MEMBERSHIP_RELATIVE).read_bytes()),
        "model_contract_sha256": acquisition.digest((root / preparation.CONTRACT_RELATIVE).read_bytes()),
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "joblib_version": joblib.__version__,
        "training_shot_count": int(split["train_shot_count"]),
        "training_match_count": int(split["train_match_count"]),
        "cross_validation": {"folds": preparation.FOLDS, "method": preparation.ALGORITHM},
        "feature_columns": preparation.FEATURE_COLUMNS,
        "target": preparation.TARGET_COLUMN,
        "group_column": preparation.GROUP_COLUMN,
        "models": MODEL_PARAMETERS,
        "probability_threshold": THRESHOLD,
        "threshold_rule": "predict goal when P(goal) >= 0.5",
        "metrics": METRIC_NAMES,
        "metrics_note": EPSILON_DESCRIPTION,
        "prohibited_during_this_phase": [
            "hyperparameter search", "SMOTE or other resampling", "class weighting", "probability calibration",
            "threshold optimization", "held-out test feature or outcome inspection",
        ],
    }
    payload = (json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return protocol, payload


def lock_protocol(root: Path) -> tuple[dict[str, Any], bytes]:
    protocol, payload = protocol_payload(root)
    path = root / PROTOCOL_RELATIVE
    if path.exists():
        existing = path.read_bytes()
        assert_locked_payload(existing, payload, "model protocol")
    else:
        acquisition.atomic_write(path, payload)
    return protocol, payload


def join_cv_assignments(training: pd.DataFrame, membership: pd.DataFrame, test_match_ids: set[int] | None = None) -> pd.DataFrame:
    required = {preparation.ID_COLUMN, preparation.GROUP_COLUMN, "validation_fold"}
    if not required.issubset(membership.columns):
        raise ModelingError("CV membership is missing event, match, or fold columns.")
    if training[preparation.ID_COLUMN].isna().any() or training[preparation.ID_COLUMN].duplicated().any():
        raise ModelingError("Training input event IDs must be complete and unique.")
    if membership[preparation.ID_COLUMN].isna().any() or membership[preparation.ID_COLUMN].duplicated().any():
        raise ModelingError("CV membership event IDs must be complete and unique.")
    expected_ids, membership_ids = set(training[preparation.ID_COLUMN]), set(membership[preparation.ID_COLUMN])
    if expected_ids != membership_ids:
        raise ModelingError("Training events and CV membership do not have exact one-to-one coverage.")
    train_groups = training.set_index(preparation.ID_COLUMN)[preparation.GROUP_COLUMN]
    fold_groups = membership.set_index(preparation.ID_COLUMN)[preparation.GROUP_COLUMN]
    if not train_groups.sort_index().equals(fold_groups.sort_index()):
        raise ModelingError("CV membership maps an event to a different match than the shot table.")
    if test_match_ids and set(membership[preparation.GROUP_COLUMN]) & test_match_ids:
        raise ModelingError("A held-out test match appears in CV membership.")
    joined = training.merge(
        membership[[preparation.ID_COLUMN, "validation_fold"]],
        on=preparation.ID_COLUMN, how="left", validate="one_to_one", sort=False,
    )
    if joined["validation_fold"].isna().any() or len(joined) != len(training):
        raise ModelingError("CV join did not preserve exact event coverage.")
    if joined.groupby(preparation.GROUP_COLUMN)["validation_fold"].nunique().ne(1).any():
        raise ModelingError("Shots from a match are assigned to multiple validation folds.")
    return joined


def make_classifier(name: str):
    if name == "prior_baseline":
        return DummyClassifier(strategy="prior")
    if name == "knn_50":
        return KNeighborsClassifier(n_neighbors=50, weights="uniform", algorithm="brute", n_jobs=1)
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=1e-9)
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    raise ModelingError(f"Unknown predeclared model: {name}")


def make_pipeline(name: str) -> Pipeline:
    return Pipeline([
        ("preprocessor", preparation.make_preprocessor()),
        ("classifier", make_classifier(name)),
    ])


def positive_probabilities(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.named_steps["classifier"].classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1 or probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ModelingError("Fitted classifier does not expose one probability column for positive class 1.")
    result = probabilities[:, int(positive[0])]
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ModelingError("Predicted positive-class probabilities must be finite values in [0,1].")
    return result


def metrics_for(y_true: Any, probability: Any, threshold: float = THRESHOLD) -> dict[str, Any]:
    try:
        y_raw = np.asarray(y_true, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelingError("Metric targets must be finite binary values.") from exc
    p = np.asarray(probability, dtype=float)
    if y_raw.ndim != 1 or p.ndim != 1 or len(y_raw) != len(p) or len(y_raw) == 0:
        raise ModelingError("Metric inputs must be non-empty, aligned one-dimensional arrays.")
    if not np.isfinite(y_raw).all() or not np.equal(y_raw, np.floor(y_raw)).all() or not set(np.unique(y_raw)).issubset({0.0, 1.0}):
        raise ModelingError("Metric targets must contain only finite binary integer values 0 and 1.")
    y = y_raw.astype(int)
    if set(np.unique(y)) != {0, 1}:
        raise ModelingError("Metric inputs must contain both binary classes 0 and 1.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ModelingError("Metric probabilities must be finite values in [0,1].")
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p, pos_label=1)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_goal_count": int((p >= threshold).sum()),
        "predicted_non_goal_count": int((p < threshold).sum()),
        "exact_p0_count": int((p == 0).sum()),
        "exact_p1_count": int((p == 1).sum()),
    }


def _fit_predict(model_name: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = make_pipeline(model_name)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[preparation.FEATURE_COLUMNS], train[preparation.TARGET_COLUMN])
        probabilities = positive_probabilities(model, validation[preparation.FEATURE_COLUMNS])
    return model, probabilities


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _make_figures(comparison: pd.DataFrame, oof_predictions: pd.DataFrame) -> dict[str, bytes]:
    figures: dict[str, bytes] = {}
    colors = {"prior_baseline": "#777777", "knn_50": "#315A8A", "gaussian_nb": "#398564", "logistic_regression": "#D88735"}

    def comparison_plot():
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")
        measures = [("oof_log_loss", "Log loss\n(lower is better)"), ("oof_brier_score", "Brier score\n(lower is better)"), ("oof_average_precision", "Average precision\n(higher is better)")]
        for axis, (column, title) in zip(axes, measures, strict=True):
            ordered = comparison.sort_values(column, ascending=column != "oof_average_precision")
            bars = axis.barh([MODEL_LABELS[name] for name in ordered["model"]], ordered[column], color=[colors[name] for name in ordered["model"]])
            axis.set_title(title, fontsize=10, color="#23364D")
            axis.grid(axis="x", alpha=0.2)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            largest = float(ordered[column].max())
            axis.set_xlim(0, largest * 1.18 if largest > 0 else 1)
            axis.invert_yaxis()
        fig.suptitle("Untuned model comparison using training-only out-of-fold predictions", color="#23364D", fontsize=13)
        fig.text(0.5, 0.025, "La Liga 2015/16 StatsBomb open data. Ten match-grouped folds; no test-set results shown.", ha="center", fontsize=8, color="#23364D")
        fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.91])
        return fig
    figures[FIGURE_NAMES["model_comparison"]] = _figure_bytes(comparison_plot)

    def confusion_plot():
        names = list(MODEL_PARAMETERS)
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), facecolor="white")
        palette = plt.get_cmap("Blues")
        for axis, name in zip(axes.flat, names, strict=True):
            rows = oof_predictions.loc[oof_predictions["model"] == name]
            matrix = confusion_matrix(rows["y_true"], rows["y_pred"], labels=[0, 1])
            axis.imshow(matrix, cmap=palette)
            axis.set_title(MODEL_LABELS[name], color="#23364D")
            axis.set_xticks([0, 1], labels=["0 non-goal", "1 goal"])
            axis.set_yticks([0, 1], labels=["0 non-goal", "1 goal"])
            axis.set_xlabel("Predicted class at threshold 0.5")
            axis.set_ylabel("Actual class")
            for (row, col), value in np.ndenumerate(matrix):
                axis.text(col, row, str(int(value)), ha="center", va="center", color="white" if value > matrix.max() * 0.55 else "#23364D", fontsize=12)
        fig.suptitle("Pooled out-of-fold confusion counts (training partition only)", color="#23364D", fontsize=13)
        fig.text(0.5, 0.025, "La Liga 2015/16 StatsBomb open data. Fixed threshold: P(goal) >= 0.5; no test-set results shown.", ha="center", fontsize=8, color="#23364D")
        fig.tight_layout(rect=[0.03, 0.07, 0.97, 0.93])
        return fig
    figures[FIGURE_NAMES["oof_confusion"]] = _figure_bytes(confusion_plot)
    return figures


def _figure_bytes(draw) -> bytes:
    fig = draw()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white", metadata={"Software": "Data Mining Final Project"})
    plt.close(fig)
    return buffer.getvalue()


def _metric_row(model_name: str, fold: int | str, train: pd.DataFrame, validation: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    result = {
        "model": model_name, "fold": fold,
        "train_shot_count": len(train), "validation_shot_count": len(validation),
        "train_match_count": int(train[preparation.GROUP_COLUMN].nunique()),
        "validation_match_count": int(validation[preparation.GROUP_COLUMN].nunique()),
        "train_goal_count": int(train[preparation.TARGET_COLUMN].sum()),
        "train_non_goal_count": int(len(train) - train[preparation.TARGET_COLUMN].sum()),
        "validation_goal_count": int(validation[preparation.TARGET_COLUMN].sum()),
        "validation_non_goal_count": int(len(validation) - validation[preparation.TARGET_COLUMN].sum()),
    }
    result.update(metrics)
    return result


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    split = holdout.reserve(root)
    phase6 = preparation.load_cv_membership(root)
    protocol, protocol_bytes = lock_protocol(root)  # Written before the first estimator fit.
    training = preparation.load_training_rows(root, set(split["train_match_ids"]))
    joined = join_cv_assignments(training, phase6, set(split["test_match_ids"]))
    if len(joined) != split["train_shot_count"] or joined[preparation.GROUP_COLUMN].nunique() != split["train_match_count"]:
        raise ModelingError("Training event count does not match the locked holdout manifest.")

    fold_metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    training_prediction_frames: list[pd.DataFrame] = []
    pipeline_payloads: dict[str, bytes] = {}
    oof_by_model: dict[str, pd.DataFrame] = {}
    for model_name in MODEL_PARAMETERS:
        model_oof_parts = []
        for fold in range(1, preparation.FOLDS + 1):
            validation = joined.loc[joined["validation_fold"] == fold].reset_index(drop=True)
            train = joined.loc[joined["validation_fold"] != fold].reset_index(drop=True)
            if train.empty or validation.empty or set(train[preparation.GROUP_COLUMN]) & set(validation[preparation.GROUP_COLUMN]):
                raise ModelingError(f"Fold {fold} has empty data or match overlap.")
            estimator, probability = _fit_predict(model_name, train, validation)
            del estimator
            metrics = metrics_for(validation[preparation.TARGET_COLUMN], probability)
            fold_metric_rows.append(_metric_row(model_name, fold, train, validation, metrics))
            model_oof_parts.append(pd.DataFrame({
                "event_id": validation[preparation.ID_COLUMN].astype(str),
                "match_id": validation[preparation.GROUP_COLUMN].astype(int),
                "fold": fold, "model": model_name,
                "y_true": validation[preparation.TARGET_COLUMN].astype(int),
                "p_goal": probability,
                "y_pred": (probability >= THRESHOLD).astype(int),
            }))
        oof = pd.concat(model_oof_parts, ignore_index=True)
        if len(oof) != len(joined) or oof["event_id"].nunique() != len(joined) or set(oof["event_id"]) != set(joined[preparation.ID_COLUMN]):
            raise ModelingError(f"Model {model_name} OOF predictions do not cover every training event exactly once.")
        oof_by_model[model_name] = oof
        pooled = metrics_for(oof["y_true"], oof["p_goal"])
        if pooled["tn"] + pooled["fp"] + pooled["fn"] + pooled["tp"] != len(joined):
            raise ModelingError(f"Model {model_name} pooled confusion counts do not sum to all training OOF rows.")
        if int(oof["y_true"].sum()) != int(joined[preparation.TARGET_COLUMN].sum()):
            raise ModelingError(f"Model {model_name} OOF labels do not preserve the locked training goal count.")
        model_fold_rows = [row for row in fold_metric_rows if row["model"] == model_name]
        comparison = {"model": model_name}
        for metric in METRIC_NAMES:
            comparison[f"oof_{metric}"] = pooled[metric]
            values = np.asarray([row[metric] for row in model_fold_rows], dtype=float)
            comparison[f"fold_mean_{metric}"] = float(values.mean())
            comparison[f"fold_std_{metric}"] = float(values.std(ddof=1))
        for count in COUNT_NAMES:
            comparison[f"oof_{count}"] = pooled[count]

        final_estimator, final_probability = _fit_predict(model_name, joined, joined)
        training_metrics = metrics_for(joined[preparation.TARGET_COLUMN], final_probability)
        for metric in METRIC_NAMES:
            comparison[f"training_resubstitution_{metric}"] = training_metrics[metric]
        for count in COUNT_NAMES:
            comparison[f"training_resubstitution_{count}"] = training_metrics[count]
        comparison["training_resubstitution_is_optimistic"] = True
        comparison_rows.append(comparison)
        training_prediction_frames.append(pd.DataFrame({
            "event_id": joined[preparation.ID_COLUMN].astype(str),
            "match_id": joined[preparation.GROUP_COLUMN].astype(int),
            "model": model_name,
            "y_true": joined[preparation.TARGET_COLUMN].astype(int),
            "p_goal": final_probability,
            "y_pred": (final_probability >= THRESHOLD).astype(int),
        }))
        buffer = io.BytesIO()
        joblib.dump(final_estimator, buffer, compress=3, protocol=5)
        serialized = buffer.getvalue()
        restored = joblib.load(io.BytesIO(serialized))
        with threadpool_limits(limits=1):
            smoke = positive_probabilities(restored, joined[preparation.FEATURE_COLUMNS])
            np.testing.assert_allclose(smoke, final_probability, rtol=0, atol=1e-14)
        pipeline_payloads[f"{MODEL_DIRECTORY}/{model_name}.joblib"] = serialized

    oof_frame = pd.concat(oof_by_model.values(), ignore_index=True)
    training_predictions = pd.concat(training_prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    comparison_table = pd.DataFrame(comparison_rows)
    tables = {
        TABLE_NAMES["fold_metrics"]: fold_metrics,
        TABLE_NAMES["model_comparison"]: comparison_table,
        TABLE_NAMES["oof_predictions"]: oof_frame,
        TABLE_NAMES["training_predictions"]: training_predictions,
    }
    table_payloads = {f"{TABLE_DIRECTORY}/{name}": _csv_bytes(frame) for name, frame in tables.items()}
    figure_payloads = _make_figures(comparison_table, oof_frame)
    all_generated = {**table_payloads, **{f"{FIGURE_DIRECTORY}/{name}": data for name, data in figure_payloads.items()}}
    output_hashes = {name: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for name, payload in all_generated.items()}
    model_hashes = {name: {"bytes": len(payload), "sha256": acquisition.digest(payload)} for name, payload in pipeline_payloads.items()}
    model_manifest = {
        "status": "passed", "scope": "models fitted on all locked training rows for future use; no test predictions",
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "source_shot_csv_sha256": protocol["source_shot_csv_sha256"],
        "holdout_manifest_sha256": protocol["holdout_manifest_sha256"],
        "cv_manifest_sha256": protocol["cv_manifest_sha256"],
        "cv_membership_sha256": protocol["cv_membership_sha256"],
        "model_contract_sha256": protocol["model_contract_sha256"],
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "joblib_version": joblib.__version__,
        "models": MODEL_PARAMETERS, "pipelines": model_hashes,
    }
    model_manifest_payload = (json.dumps(model_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    report = {
        "status": "passed", "scope": "locked training partition only; no test evaluation",
        "training_shot_count": int(len(joined)), "training_match_count": int(joined[preparation.GROUP_COLUMN].nunique()),
        "training_goal_count": int(joined[preparation.TARGET_COLUMN].sum()),
        "training_non_goal_count": int(len(joined) - joined[preparation.TARGET_COLUMN].sum()),
        "protocol_sha256": acquisition.digest(protocol_bytes),
        "cv_manifest_sha256": protocol["cv_manifest_sha256"],
        "cv_membership_sha256": protocol["cv_membership_sha256"],
        "model_manifest_sha256": acquisition.digest(model_manifest_payload),
        "threshold": THRESHOLD,
        "oof_metrics_scope": "pooled predictions with one validation prediction per training event",
        "fold_summary_scope": "unweighted mean and sample standard deviation across the ten folds; no confidence interval claimed",
        "training_resubstitution_scope": "same rows used to fit each full-training pipeline; optimistic descriptive reference, not validation",
        "metrics_note": EPSILON_DESCRIPTION,
        "models": comparison_rows,
        "outputs": output_hashes,
        "pipeline_outputs": model_hashes,
    }
    report_payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")

    manifest_path = root / MODEL_MANIFEST_RELATIVE
    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        assert_locked_payload(existing, model_manifest_payload, "saved-model manifest")
        for relative, payload in pipeline_payloads.items():
            path = root / relative
            if not path.exists() or acquisition.digest(path.read_bytes()) != acquisition.digest(payload):
                raise ModelingError(f"Saved pipeline {relative} is missing or differs from its manifest.")
    else:
        for relative, payload in pipeline_payloads.items():
            acquisition.atomic_write(root / relative, payload)
        acquisition.atomic_write(manifest_path, model_manifest_payload)
    for relative, payload in all_generated.items():
        acquisition.atomic_write(root / relative, payload)
    acquisition.atomic_write(root / REPORT_RELATIVE, report_payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.root)
    except (ModelingError, preparation.PreparationError, holdout.SplitError, OSError, ValueError, KeyError, Warning) as exc:
        print(f"Model training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "training_shot_count", "training_match_count", "training_goal_count", "training_non_goal_count", "protocol_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
