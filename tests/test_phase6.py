import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import acquire_data as acquisition
from scripts import prepare_modeling as prep


PROJECT = Path(__file__).resolve().parents[1]


def fixture_project(root: Path) -> None:
    for folder in ("data/processed", "docs", "reports/tables"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    contract_bytes = (PROJECT / prep.CONTRACT_RELATIVE).read_bytes()
    (root / prep.CONTRACT_RELATIVE).write_bytes(contract_bytes)
    rows = []
    for match_id in range(1, 21):
        for event_no in range(10):
            rows.append({
                "event_id": f"event-{match_id}-{event_no}", "match_id": match_id,
                "distance_sb_units": 5.0 + event_no + match_id / 10,
                "angle_sb_radians": 0.1 + event_no / 20,
                "body_part": "Foot", "shot_type": "Open Play", "technique": "Normal",
                "play_pattern": "Regular Play", "under_pressure": event_no % 2,
                "first_time": int(event_no == 0), "open_goal": 0,
                "is_goal": int(event_no in (0, 1)),
            })
    frame = pd.DataFrame(rows)
    shot_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (root / prep.SHOT_RELATIVE).write_bytes(shot_bytes)
    report = {
        "status": "passed", "eligible_shot_count": len(frame),
        "outputs": {prep.SHOT_RELATIVE: {"sha256": acquisition.digest(shot_bytes)}},
        "source": {"commit": "test-source"},
    }
    (root / prep.holdout.REPORT_RELATIVE).write_text(json.dumps(report), encoding="utf-8")


def small_fold_frame() -> pd.DataFrame:
    rows = []
    for match_id, categories in ((1, ["Foot", "Head"]), (2, ["Foot", "Foot"]), (3, ["Rare", "Head"])):
        for index, body_part in enumerate(categories):
            rows.append({
                "event_id": f"{match_id}-{index}", "match_id": match_id,
                "distance_sb_units": float(match_id * 10 + index),
                "angle_sb_radians": float(match_id + index),
                "body_part": body_part, "shot_type": "Open Play", "technique": "Normal",
                "play_pattern": "Regular Play", "under_pressure": index,
                "first_time": 0, "open_goal": 0, "is_goal": index,
            })
    return pd.DataFrame(rows)


class ModelingPreparationTests(unittest.TestCase):
    def test_preprocessor_is_fresh_dense_and_ignores_unseen_categories(self):
        frame = small_fold_frame()
        train_index = np.array([0, 1, 2, 3])
        validation_index = np.array([4, 5])
        record, transformed, preprocessor = prep.evaluate_fold(frame, train_index, validation_index, 1)
        self.assertEqual(record["unknown_category_counts"]["body_part"], 1)
        self.assertEqual(transformed.shape[0], 2)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertIsInstance(preprocessor, type(prep.make_preprocessor()))

    def test_scaler_is_fitted_on_fold_training_rows_only(self):
        frame = small_fold_frame()
        train_index = np.array([0, 1, 2, 3])
        validation_index = np.array([4, 5])
        _, _, preprocessor = prep.evaluate_fold(frame, train_index, validation_index, 1)
        expected = frame.iloc[train_index][prep.NUMERIC_COLUMNS].mean().to_numpy()
        np.testing.assert_allclose(preprocessor.named_transformers_["numeric"].mean_, expected)
        self.assertFalse(np.allclose(expected, frame[prep.NUMERIC_COLUMNS].mean().to_numpy()))

    def test_feature_contract_rejects_extra_or_reordered_input(self):
        contract = json.loads((PROJECT / prep.CONTRACT_RELATIVE).read_text(encoding="utf-8"))
        contract["features"].append({"column": "statsbomb_xg", "type": "numeric"})
        with self.assertRaisesRegex(prep.PreparationError, "exactly"):
            prep.validate_contract(contract)

    def test_fractional_or_nonfinite_target_and_flags_fail_before_integer_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            path = root / prep.SHOT_RELATIVE
            original = pd.read_csv(path)
            for column, bad_value in (("is_goal", 0.5), ("under_pressure", np.nan)):
                broken = original.copy()
                broken[column] = broken[column].astype("float64")
                broken.loc[0, column] = bad_value
                broken.to_csv(path, index=False, lineterminator="\n")
                with self.subTest(column=column), self.assertRaisesRegex(prep.PreparationError, "missing|required|finite integer"):
                    prep.load_training_rows(root, set(range(1, 21)))

    def test_model_rows_exclude_identifiers_reference_and_audit_fields(self):
        self.assertEqual(prep.FEATURE_COLUMNS, [
            "distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique",
            "play_pattern", "under_pressure", "first_time", "open_goal",
        ])
        self.assertNotIn("statsbomb_xg", prep.TRAIN_COLUMNS)
        self.assertNotIn("shot_outcome", prep.TRAIN_COLUMNS)
        self.assertNotIn("event_index", prep.TRAIN_COLUMNS)

    def test_preparation_covers_training_only_groups_and_both_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            report = prep.prepare(root)
            membership = prep.load_cv_membership(root)
            split = json.loads((root / "docs/split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["training_shot_count"], split["train_shot_count"])
            self.assertEqual(report["training_match_count"], split["train_match_count"])
            self.assertEqual(len(membership), split["train_shot_count"])
            self.assertFalse(set(membership["match_id"]) & set(split["test_match_ids"]))
            self.assertEqual(membership["event_id"].nunique(), len(membership))
            folds = report["validation_folds"]
            self.assertEqual(sum(row["validation_shot_count"] for row in folds), len(membership))
            self.assertEqual(sum(row["validation_match_count"] for row in folds), len(split["train_match_ids"]))
            for fold in folds:
                self.assertGreater(fold["train_goal_count"], 0)
                self.assertGreater(fold["train_non_goal_count"], 0)
                self.assertGreater(fold["validation_goal_count"], 0)
                self.assertGreater(fold["validation_non_goal_count"], 0)

    def test_altered_cv_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            prep.prepare(root)
            membership_path = root / prep.CV_MEMBERSHIP_RELATIVE
            membership_path.write_bytes(membership_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(prep.PreparationError, "hash"):
                prep.load_cv_membership(root)

    def test_changed_contract_is_rejected_by_existing_cv_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_project(root)
            prep.prepare(root)
            contract_path = root / prep.CONTRACT_RELATIVE
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["features"][0]["definition"] += " changed"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(prep.PreparationError, "contract_sha256"):
                prep.load_cv_membership(root)


if __name__ == "__main__":
    unittest.main()
