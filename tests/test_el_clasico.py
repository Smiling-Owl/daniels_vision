import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import el_clasico_case_study as case
from scripts import simulate_matches as phase9


def case_fixture():
    shot_rows = [
        {"match_id": 266424, "event_id": "a1", "event_index": 1, "match_date": "2015-11-21", "home_team_id": 1, "home_team_name": "Real Madrid", "away_team_id": 2, "away_team_name": "Barcelona", "team_id": 1, "team_name": "Real Madrid", "is_home": 1, "is_goal": 0},
        *[{"match_id": 266424, "event_id": f"a{i}", "event_index": i, "match_date": "2015-11-21", "home_team_id": 1, "home_team_name": "Real Madrid", "away_team_id": 2, "away_team_name": "Barcelona", "team_id": 2, "team_name": "Barcelona", "is_home": 0, "is_goal": 1} for i in range(2, 6)],
        {"match_id": 267533, "event_id": "b1", "event_index": 1, "match_date": "2016-04-02", "home_team_id": 2, "home_team_name": "Barcelona", "away_team_id": 1, "away_team_name": "Real Madrid", "team_id": 2, "team_name": "Barcelona", "is_home": 1, "is_goal": 1},
        *[{"match_id": 267533, "event_id": f"b{i}", "event_index": i, "match_date": "2016-04-02", "home_team_id": 2, "home_team_name": "Barcelona", "away_team_id": 1, "away_team_name": "Real Madrid", "team_id": 1, "team_name": "Real Madrid", "is_home": 0, "is_goal": 1} for i in range(2, 4)],
    ]
    reconciliation = pd.DataFrame([
        {"match_id": 266424, "match_date": "2015-11-21", "home_team_id": 1, "home_team_name": "Real Madrid", "away_team_id": 2, "away_team_name": "Barcelona", "official_home_goals": 0, "official_away_goals": 4, "shot_goals_home": 0, "shot_goals_away": 4, "own_goals_credited_home": 0, "own_goals_credited_away": 0},
        {"match_id": 267533, "match_date": "2016-04-02", "home_team_id": 2, "home_team_name": "Barcelona", "away_team_id": 1, "away_team_name": "Real Madrid", "official_home_goals": 1, "official_away_goals": 2, "shot_goals_home": 1, "shot_goals_away": 2, "own_goals_credited_home": 0, "own_goals_credited_away": 0},
    ])
    predictions = pd.DataFrame([
        {"event_id": row["event_id"], "match_id": row["match_id"], "fold": 2 if row["match_id"] == 266424 else 6, "variant": case.OOF_VARIANT, "y_true": row["is_goal"], "p_goal": 0.2 + i * 0.1}
        for i, row in enumerate(shot_rows)
    ])
    membership = pd.DataFrame([
        {"event_id": row["event_id"], "match_id": row["match_id"], "validation_fold": 2 if row["match_id"] == 266424 else 6}
        for row in shot_rows
    ])
    return pd.DataFrame(shot_rows), predictions, reconciliation, membership


class ElClasicoCaseStudyTests(unittest.TestCase):
    def test_selection_requires_frozen_candidate_and_training_only_cases(self):
        _, predictions, _, _ = case_fixture()
        selected = case.select_oof_predictions(predictions, case.OOF_VARIANT, set(case.CASE_MATCHES), set())
        self.assertEqual(set(selected.match_id), set(case.CASE_MATCHES))
        with self.assertRaises(case.CaseStudyError):
            case.select_oof_predictions(predictions, "knn_sigmoid", set(case.CASE_MATCHES), set())
        with self.assertRaisesRegex(case.CaseStudyError, "training matches"):
            case.select_oof_predictions(predictions, case.OOF_VARIANT, {266424}, set())
        with self.assertRaisesRegex(case.CaseStudyError, "outside the test partition"):
            case.select_oof_predictions(predictions, case.OOF_VARIANT, set(case.CASE_MATCHES), {267533})

    def test_join_uses_event_and_match_fold_and_rejects_tampering(self):
        shots, predictions, reconciliation, membership = case_fixture()
        joined, info = case.join_case_shots(shots, predictions, reconciliation, membership)
        self.assertEqual(len(joined), 8)
        self.assertEqual(set(joined.validation_fold), {2, 6})
        self.assertEqual(set(info.match_id), set(case.CASE_MATCHES))
        bad = predictions.copy()
        bad.loc[0, "match_id"] = 267533
        with self.assertRaisesRegex(case.CaseStudyError, "safely join"):
            case.join_case_shots(shots, bad, reconciliation, membership)
        bad = predictions.copy()
        bad.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(case.CaseStudyError, "safely join"):
            case.join_case_shots(shots, bad, reconciliation, membership)
        bad = predictions.copy()
        bad.loc[0, "event_id"] = "missing-event"
        with self.assertRaisesRegex(case.CaseStudyError, "safely join"):
            case.join_case_shots(shots, bad, reconciliation, membership)
        bad_membership = membership.copy()
        bad_membership.loc[0, "validation_fold"] = 3
        with self.assertRaisesRegex(case.CaseStudyError, "locked Phase 6"):
            case.join_case_shots(shots, predictions, reconciliation, bad_membership)

    def test_exact_joint_and_mc_outputs_cover_all_support(self):
        shots, predictions, reconciliation, membership = case_fixture()
        joined, info = case.join_case_shots(shots, predictions, reconciliation, membership)
        goals = pd.DataFrame(columns=["match_id", "beneficiary_team_id", "event_id"])
        with mock.patch.object(case, "TRIALS", 2_000), mock.patch.object(case, "CHUNK_SIZE", 500):
            tables, figures = case._case_outputs(joined, info, goals)
        summary = tables[case.TABLE_NAMES["match_summary"]]
        scorelines = tables[case.TABLE_NAMES["scorelines"]]
        self.assertEqual(len(summary), 2)
        self.assertEqual(scorelines.groupby("match_id").mc_count.sum().to_dict(), {266424: 2_000, 267533: 2_000})
        self.assertTrue(summary[["mc_home_win_count", "mc_draw_count", "mc_away_win_count"]].sum(axis=1).eq(2_000).all())
        self.assertTrue(np.allclose(scorelines.groupby("match_id").exact_probability.sum(), 1.0))
        self.assertEqual(len(figures), 3)
        for match_id in case.CASE_MATCHES:
            exact = phase9.exact_poisson_binomial([0.2, 0.3])
            self.assertAlmostEqual(float(exact.sum()), 1.0)

    def test_protocol_lock_refuses_changed_bytes(self):
        root = Path("C:/project")
        payload = case._json_bytes({"seed": 42})
        with mock.patch.object(Path, "exists", return_value=False), mock.patch.object(case.acquisition, "atomic_write") as write:
            case._lock_protocol(root, {"seed": 42}, payload)
            write.assert_called_once_with(root / case.PROTOCOL_RELATIVE, payload)
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(Path, "read_bytes", return_value=payload):
            with self.assertRaisesRegex(case.CaseStudyError, "refusing to overwrite"):
                case._lock_protocol(root, {"seed": 43}, case._json_bytes({"seed": 43}))


if __name__ == "__main__":
    unittest.main()
