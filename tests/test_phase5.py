import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import acquire_data as acquisition
from scripts import explore_shots as eda
from scripts import reserve_holdout as holdout


class HoldoutReservationTests(unittest.TestCase):
    def test_fixed_cardinality_subset_has_exact_shots_and_full_partition(self):
        counts = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        test_ids = holdout.choose_match_subset(counts, target_shots=7, target_matches=2, seed=42)
        self.assertEqual(len(test_ids), 2)
        self.assertEqual(sum(counts[mid] for mid in test_ids), 7)
        train_ids = set(counts) - set(test_ids)
        self.assertFalse(set(test_ids) & train_ids)
        self.assertEqual(set(test_ids) | train_ids, set(counts))

    def test_seed_and_input_order_are_deterministic(self):
        counts = {mid: (mid % 5) + 1 for mid in range(1, 31)}
        expected = holdout.choose_match_subset(counts, 24, 8, seed=42)
        self.assertEqual(holdout.choose_match_subset(dict(reversed(list(counts.items()))), 24, 8, seed=42), expected)
        self.assertEqual(holdout.choose_match_subset(counts, 24, 8, seed=42), expected)

    def test_infeasible_subset_fails_without_fallback(self):
        with self.assertRaisesRegex(holdout.SplitError, "No exact split"):
            holdout.choose_match_subset({1: 2, 2: 4, 3: 6}, target_shots=5, target_matches=2)

    def test_reservation_reuses_locked_membership_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            first = holdout.reserve(root)
            membership_path = root / holdout.MEMBERSHIP_RELATIVE
            original_membership = membership_path.read_bytes()
            second = holdout.reserve(root)
            self.assertEqual(first["membership_csv_sha256"], second["membership_csv_sha256"])
            self.assertEqual(membership_path.read_bytes(), original_membership)
            membership_path.write_bytes(original_membership + b"\n")
            with self.assertRaisesRegex(holdout.SplitError, "membership CSV hash"):
                holdout.reserve(root)

    def test_reservation_refuses_changed_source_and_recovers_missing_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            locked = holdout.reserve(root)
            membership_path = root / holdout.MEMBERSHIP_RELATIVE
            membership_path.unlink()
            recovered = holdout.reserve(root)
            self.assertEqual(recovered["membership_csv_sha256"], locked["membership_csv_sha256"])
            shot_path = root / holdout.SHOT_RELATIVE
            shot_path.write_bytes(shot_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(holdout.SplitError, "Shot CSV hash"):
                holdout.reserve(root)

    @staticmethod
    def _write_source(root: Path) -> None:
        shot_path = root / holdout.SHOT_RELATIVE
        shot_path.parent.mkdir(parents=True, exist_ok=True)
        shot_path.write_text(
            "match_id,event_id,is_goal\n1,e1,1\n2,e2,0\n3,e3,0\n4,e4,0\n4,e5,1\n5,e6,0\n5,e7,0\n5,e8,1\n5,e9,0\n5,e10,0\n",
            encoding="utf-8",
        )
        report = {
            "status": "passed", "eligible_shot_count": 10,
            "outputs": {holdout.SHOT_RELATIVE: {"sha256": acquisition.digest(shot_path.read_bytes())}},
            "source": {"commit": acquisition.SOURCE["commit"]},
        }
        report_path = root / holdout.REPORT_RELATIVE
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")


class ExploratoryAnalysisTests(unittest.TestCase):
    def test_wilson_interval_handles_known_and_boundary_cases(self):
        low, high = eda.wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775401687666166)
        low, high = eda.wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.7224598312333834)
        self.assertAlmostEqual(high, 1.0)
        self.assertEqual(eda.wilson_interval(0, 0), (None, None))

    def test_geometry_bins_include_zero_and_final_edges(self):
        frame = pd.DataFrame({"distance_sb_units": [0.0, 1.0, 2.0], "is_goal": [0, 1, 1]})
        table = eda.geometry_rates(frame, "distance_sb_units", [0, 1, 2], ["0–1", "1–2"])
        self.assertEqual(table["shot_count"].tolist(), [2, 1])
        self.assertEqual(table["goal_count"].tolist(), [1, 1])
        angle_frame = pd.DataFrame({"angle_sb_radians": [0.0, 3.141592653589793], "is_goal": [0, 1]})
        angle = eda.geometry_rates(angle_frame, "angle_sb_radians", [0, 3.141592653589793], ["0–π"])
        self.assertEqual(angle.loc[0, "shot_count"], 2)


    def test_geometry_rounding_clamps_only_tiny_final_edge_error(self):
        pi = 3.141592653589793
        frame = pd.DataFrame({"angle_sb_radians": [pi + 2e-13], "is_goal": [1]})
        table = eda.geometry_rates(frame, "angle_sb_radians", [0, pi], ["0–π"])
        self.assertEqual(table.loc[0, "shot_count"], 1)
        outside = pd.DataFrame({"angle_sb_radians": [pi + 1e-6], "is_goal": [1]})
        with self.assertRaisesRegex(eda.EDAError, "outside"):
            eda.geometry_rates(outside, "angle_sb_radians", [0, pi], ["0–π"])

    def test_empty_geometry_bin_serializes_as_json_null(self):
        frame = pd.DataFrame({"distance_sb_units": [1.0], "is_goal": [0]})
        table = eda.geometry_rates(frame, "distance_sb_units", [0, 2, 4], ["0–2", "2–4"])
        records = eda.json_records(table)
        self.assertEqual(records[1]["shot_count"], 0)
        self.assertIsNone(records[1]["goal_rate"])
        json.dumps(records, allow_nan=False)

    def test_findings_follow_physical_bin_order_not_lexical_order(self):
        rows = [
            {"dimension": "distance_sb_units", "category": "0–6", "shot_count": 2, "goal_rate": 0.5},
            {"dimension": "distance_sb_units", "category": "6–12", "shot_count": 3, "goal_rate": 0.3},
            {"dimension": "distance_sb_units", "category": "36–80", "shot_count": 4, "goal_rate": 0.1},
            {"dimension": "angle_sb_radians", "category": "0–π/6", "shot_count": 1, "goal_rate": 1.0},
        ]
        report = {
            "findings": {"geometry_bins": rows, "categories": [], "context_flags": []},
            "training_shot_count": 10, "training_match_count": 1,
            "training_goal_count": 3, "training_non_goal_count": 7,
            "training_overall_goal_rate": 0.3, "training_no_goal_majority_baseline_accuracy": 0.7,
            "training_penalty_shot_count": 0, "training_penalty_goal_count": 0,
        }
        text = eda.render_findings(report)
        self.assertIn("the 0–6 distance band", text)
        self.assertIn("the 36–80 band", text)
        self.assertIn("groups with fewer than 30 training shots (0 in this sample)", text)

    def test_test_label_and_feature_changes_do_not_change_training_aggregates(self):
        full = pd.DataFrame({
            "match_id": [1, 1, 2, 3, 3],
            "is_goal": [0, 1, 0, 0, 1],
            "is_penalty": [0, 0, 0, 1, 0],
            "body_part": ["Foot", "Head", "Foot", "Head", "Foot"],
        })
        train_ids = {1, 2}
        first_train = eda.select_training_rows(full, train_ids)
        first_summary = eda.summarize_training(first_train)
        first_table = eda.categorical_rates(first_train, ["body_part"]).to_dict(orient="records")
        changed = full.copy()
        changed.loc[changed["match_id"] == 3, "is_goal"] = [1, 0]
        changed.loc[changed["match_id"] == 3, "body_part"] = ["Changed", "Changed"]
        second_train = eda.select_training_rows(changed, train_ids)
        second_summary = eda.summarize_training(second_train)
        second_table = eda.categorical_rates(second_train, ["body_part"]).to_dict(orient="records")
        self.assertEqual(first_summary["goal_count"], second_summary["goal_count"])
        self.assertEqual(first_summary["no_goal_majority_baseline_accuracy"], second_summary["no_goal_majority_baseline_accuracy"])
        self.assertEqual(first_table, second_table)


if __name__ == "__main__":
    unittest.main()

