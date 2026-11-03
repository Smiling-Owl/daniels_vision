import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import calibrate_evaluate as calibration
from scripts import prepare_modeling, train_models
from tests.test_phase6 import fixture_project


def prediction_rows():
    rows = []
    for variant in calibration.VARIANTS:
        for index, (label, probability) in enumerate([(0, 0.05), (1, 0.35), (0, 0.65), (1, 0.95)]):
            rows.append({
                "event_id": f"{variant}-{index}", "match_id": index,
                "variant": variant, "y_true": label, "p_goal": probability,
            })
    return pd.DataFrame(rows)


def grouped_rows() -> pd.DataFrame:
    rows = []
    for match_id in range(1, 11):
        for event_no, label in enumerate([0, 1, 0, 1]):
            rows.append({
                "event_id": f"match-{match_id}-shot-{event_no}", "match_id": match_id,
                "distance_sb_units": float(match_id * 10 + event_no),
                "angle_sb_radians": float(match_id / 20 + event_no / 100),
                "body_part": "Head" if event_no == 1 else "Foot",
                "shot_type": "Open Play", "technique": "Normal", "play_pattern": "Regular Play",
                "under_pressure": event_no % 2, "first_time": int(event_no == 0), "open_goal": 0,
                "is_goal": label,
            })
    return pd.DataFrame(rows)


class CalibrationPreparationTests(unittest.TestCase):
    def test_nested_group_folds_and_calibration_pipeline_keep_preprocessing_local(self):
        frame = grouped_rows()
        outer_splits = calibration.grouped_splits(frame, 5, 42)
        train_indices, validation_indices = outer_splits[0]
        outer_train = frame.iloc[train_indices].reset_index(drop=True)
        outer_validation = frame.iloc[validation_indices]
        self.assertFalse(set(outer_train.match_id) & set(outer_validation.match_id))
        self.assertEqual(set(outer_train.match_id) | set(outer_validation.match_id), set(frame.match_id))
        self.assertEqual(outer_train.is_goal.nunique(), 2)
        self.assertEqual(outer_validation.is_goal.nunique(), 2)

        inner_splits = calibration.grouped_splits(outer_train, 5, 43)
        coverage = np.zeros(len(outer_train), dtype=int)
        for inner_train_indices, inner_validation_indices in inner_splits:
            inner_train = outer_train.iloc[inner_train_indices]
            inner_validation = outer_train.iloc[inner_validation_indices]
            self.assertFalse(set(inner_train.match_id) & set(inner_validation.match_id))
            self.assertEqual(inner_train.is_goal.nunique(), 2)
            self.assertEqual(inner_validation.is_goal.nunique(), 2)
            coverage[inner_validation_indices] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(outer_train), dtype=int))

        calibrator = calibration.make_calibrator("logistic_regression", inner_splits)
        self.assertIsInstance(calibrator.estimator, train_models.Pipeline)
        self.assertFalse(hasattr(calibrator.estimator.named_steps["preprocessor"], "transformers_"))
        calibrator = calibration.fit_calibrator("logistic_regression", outer_train, inner_splits)
        fitted_base = calibrator.calibrated_classifiers_[0].estimator
        self.assertIsInstance(fitted_base, train_models.Pipeline)
        scaler = fitted_base.named_steps["preprocessor"].named_transformers_["numeric"]
        np.testing.assert_allclose(scaler.mean_, outer_train[prepare_modeling.NUMERIC_COLUMNS].mean().to_numpy())
        self.assertFalse(np.allclose(scaler.mean_, frame[prepare_modeling.NUMERIC_COLUMNS].mean().to_numpy()))

    def test_reliability_bins_keep_empty_bins_undefined_and_report_counts(self):
        predictions = prediction_rows()
        bins = calibration.reliability_bins(predictions, "development")
        row = bins.loc[(bins.variant == calibration.BASELINE) & (bins.bin_index == 0)].iloc[0]
        self.assertEqual(row.shot_count, 1)
        empty = bins.loc[(bins.variant == calibration.BASELINE) & (bins.bin_index == 1)].iloc[0]
        self.assertEqual(empty.shot_count, 0)
        self.assertTrue(pd.isna(empty.observed_goal_rate))
        self.assertEqual(int(bins.loc[bins.variant == calibration.BASELINE, "shot_count"].sum()), 4)

    def test_reliability_boundaries_and_invalid_values(self):
        predictions = prediction_rows()
        predictions.loc[predictions.index[0], "p_goal"] = 1.0
        bins = calibration.reliability_bins(predictions, "test")
        self.assertEqual(int(bins.loc[(bins.variant == calibration.BASELINE) & (bins.bin_index == 9), "shot_count"].iloc[0]), 2)
        predictions.loc[predictions.index[0], "p_goal"] = np.inf
        with self.assertRaisesRegex(calibration.CalibrationError, "probabilities"):
            calibration.reliability_bins(predictions, "test")
        predictions = prediction_rows()
        predictions["y_true"] = predictions["y_true"].astype(float)
        predictions.loc[predictions.index[0], "y_true"] = 0.5
        with self.assertRaisesRegex(calibration.CalibrationError, "labels"):
            calibration.reliability_bins(predictions, "test")

    def test_reliability_figure_contains_scope_and_count_context(self):
        bins = calibration.reliability_bins(prediction_rows(), "development")
        image = calibration._reliability_figure(bins, "training-only nested OOF development")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(image), 10_000)

    def test_locked_selection_refuses_modified_candidate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            path = root / "selection.json"
            path.write_text('{"candidate":"old"}\n', encoding="utf-8")
            changed = {"candidate": "new"}
            payload = (json.dumps(changed, indent=2) + "\n").encode()
            with self.assertRaisesRegex(calibration.CalibrationError, "differs"):
                calibration.validate_locked_candidate(root, "selection.json", changed, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"candidate": "old"})

    def test_test_rows_cannot_load_before_frozen_selection_is_present(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            selection = {"candidate": "frozen"}
            payload = (json.dumps(selection) + "\n").encode()
            split = {"test_match_ids": {7}, "test_shot_count": 1}
            with patch.object(calibration.preparation, "load_training_rows") as loader:
                with self.assertRaisesRegex(calibration.CalibrationError, "must exist"):
                    calibration.load_frozen_test_rows(root, selection, payload, split)
                loader.assert_not_called()
                selection_path = root / calibration.SELECTION_RELATIVE
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                selection_path.write_bytes(payload)
                loader.return_value = pd.DataFrame({"match_id": [7], "is_goal": [1]})
                loaded = calibration.load_frozen_test_rows(root, selection, payload, split)
                self.assertEqual(len(loaded), 1)
                loader.assert_called_once_with(root, {7})

    def test_synthetic_pipeline_freezes_training_choice_before_test_access(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            fixture_project(root)
            prepare_modeling.prepare(root)
            train_models.run(root)

            phase7 = calibration.verify_phase7(root)
            oof_relative = f"{train_models.TABLE_DIRECTORY}/{train_models.TABLE_NAMES['oof_predictions']}"
            oof_path = root / oof_relative
            original_oof = oof_path.read_bytes()
            oof_path.write_bytes(original_oof + b"\n")
            with self.assertRaisesRegex(calibration.CalibrationError, "hash/size mismatch"):
                calibration.verify_phase7(root)
            oof_path.write_bytes(original_oof)

            with patch.object(calibration, "load_frozen_test_rows", side_effect=AssertionError("development accessed test rows")):
                development = calibration.run(root, "develop")
            self.assertFalse(development["test_data_accessed"])
            selection_path = root / calibration.SELECTION_RELATIVE
            frozen_bytes = selection_path.read_bytes()
            phase7 = calibration.verify_phase7(root)
            protocol, protocol_bytes = calibration.make_protocol(phase7)

            selection_path.write_bytes(frozen_bytes + b" ")
            with self.assertRaisesRegex(calibration.CalibrationError, "selection hash"):
                calibration._verify_frozen_selection(root, phase7, protocol, protocol_bytes)
            selection_path.write_bytes(frozen_bytes)

            original_loader = calibration.load_frozen_test_rows
            def selection_checked_loader(project_root, selection, selection_bytes, split):
                self.assertEqual((project_root / calibration.SELECTION_RELATIVE).read_bytes(), selection_bytes)
                self.assertEqual(selection_bytes, frozen_bytes)
                return original_loader(project_root, selection, selection_bytes, split)

            with patch.object(calibration, "load_frozen_test_rows", side_effect=selection_checked_loader):
                evaluation = calibration.run(root, "evaluate")
            self.assertEqual(evaluation["status"], "passed")
            self.assertEqual(evaluation["test_shot_count"], 40)
            self.assertTrue((root / "reports/tables/phase08_development_reliability_bins.csv").is_file())
            self.assertTrue((root / "reports/tables/phase08_test_reliability_bins.csv").is_file())


if __name__ == "__main__":
    unittest.main()
