import unittest
from io import StringIO

import numpy as np
import pandas as pd

from scripts import prepare_modeling as preparation
from scripts import train_models as training


def model_rows() -> pd.DataFrame:
    rows = []
    labels = [0, 1, 0, 1, 1, 0]
    for i, label in enumerate(labels):
        rows.append({
            "event_id": f"event-{i}", "match_id": 10 + i // 2,
            "distance_sb_units": float(4 + i * 10), "angle_sb_radians": float(i) / 10,
            "body_part": "Foot" if i % 2 == 0 else "Head", "shot_type": "Open Play",
            "technique": "Normal", "play_pattern": "Regular Play",
            "under_pressure": i % 2, "first_time": int(i == 0), "open_goal": 0,
            "is_goal": label,
        })
    return pd.DataFrame(rows)


class ModelTrainingTests(unittest.TestCase):
    def test_cv_membership_joins_by_event_id_not_row_position(self):
        shots = model_rows()
        membership = pd.DataFrame({
            "event_id": ["event-3", "event-0", "event-5", "event-2", "event-1", "event-4"],
            "match_id": [11, 10, 12, 11, 10, 12],
            "validation_fold": [2, 1, 3, 2, 1, 3],
        })
        joined = training.join_cv_assignments(shots, membership)
        self.assertEqual(joined["event_id"].tolist(), shots["event_id"].tolist())
        self.assertEqual(joined["validation_fold"].tolist(), [1, 1, 2, 2, 3, 3])

    def test_cv_join_rejects_incomplete_map_and_test_match(self):
        shots = model_rows()
        membership = pd.DataFrame({
            "event_id": shots["event_id"], "match_id": shots["match_id"], "validation_fold": [1, 1, 2, 2, 3, 3],
        })
        with self.assertRaisesRegex(training.ModelingError, "exact one-to-one"):
            training.join_cv_assignments(shots, membership.iloc[:-1])
        with self.assertRaisesRegex(training.ModelingError, "test match"):
            training.join_cv_assignments(shots, membership, {12})

    def test_metrics_match_known_confusion_counts_and_probability_scores(self):
        metrics = training.metrics_for([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.4])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 1, 1))
        self.assertEqual(metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"], 4)
        self.assertAlmostEqual(metrics["brier_score"], np.mean((np.array([0, 0, 1, 1]) - np.array([0.1, 0.8, 0.6, 0.4])) ** 2))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_zero_precision_is_defined_as_zero(self):
        metrics = training.metrics_for([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 2)

    def test_fractional_or_nonfinite_targets_are_rejected_before_cast(self):
        for labels in ([0, 1.5, 1], [0, np.nan, 1]):
            with self.subTest(labels=labels), self.assertRaisesRegex(training.ModelingError, "binary"):
                training.metrics_for(labels, [0.1, 0.5, 0.9])

    def test_prediction_csv_round_trip_preserves_extreme_probabilities_and_logloss(self):
        probabilities = np.array([1e-14, 1.0 - 1e-14])
        predictions = pd.DataFrame({"p_goal": probabilities, "y_true": [1, 0]})
        encoded = training._csv_bytes(predictions)
        restored = pd.read_csv(StringIO(encoded.decode("utf-8")), float_precision="round_trip")
        np.testing.assert_array_equal(restored["p_goal"].to_numpy(), probabilities)
        before = training.metrics_for([1, 0], probabilities)["log_loss"]
        after = training.metrics_for(restored["y_true"], restored["p_goal"])["log_loss"]
        self.assertEqual(before, after)
        self.assertGreater(after, 10)

    def test_prior_baseline_uses_current_fold_training_prevalence(self):
        frame = model_rows()
        train = frame.iloc[[0, 1, 2]]
        validation = frame.iloc[[4, 5]]
        estimator, probabilities = training._fit_predict("prior_baseline", train, validation)
        expected_prevalence = train[preparation.TARGET_COLUMN].mean()
        self.assertTrue(np.all(probabilities == expected_prevalence))
        self.assertNotEqual(expected_prevalence, frame[preparation.TARGET_COLUMN].mean())

    def test_fold_pipeline_scaler_is_fit_on_training_rows_only(self):
        frame = model_rows()
        train = frame.iloc[[0, 1, 2, 3]]
        validation = frame.iloc[[4, 5]]
        estimator, _ = training._fit_predict("prior_baseline", train, validation)
        scaler = estimator.named_steps["preprocessor"].named_transformers_["numeric"]
        expected = train[preparation.NUMERIC_COLUMNS].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        self.assertFalse(np.allclose(scaler.mean_, frame[preparation.NUMERIC_COLUMNS].mean().to_numpy()))

    def test_protocol_and_model_manifest_reject_changed_content(self):
        expected = b'{"status": "predeclared"}\n'
        training.assert_locked_payload(expected, expected, "model protocol")
        with self.assertRaisesRegex(training.ModelingError, "differs"):
            training.assert_locked_payload(b'{"status": "changed"}\n', expected, "model protocol")
        with self.assertRaisesRegex(training.ModelingError, "not valid"):
            training.assert_locked_payload(b"corrupt", expected, "saved-model manifest")

    def test_only_approved_predictors_enter_model_matrix(self):
        self.assertEqual(training.preparation.FEATURE_COLUMNS, [
            "distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique",
            "play_pattern", "under_pressure", "first_time", "open_goal",
        ])
        for excluded in ("match_id", "event_id", "is_goal", "shot_outcome", "statsbomb_xg", "x", "y"):
            self.assertNotIn(excluded, training.preparation.FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
