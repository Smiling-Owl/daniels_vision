import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from scripts import acquire_data
from scripts import simulate_matches as simulation


class MonteCarloSimulationTests(unittest.TestCase):
    def test_poisson_binomial_exact_small_case_and_independent_match_outcomes(self):
        pmf = simulation.exact_poisson_binomial([0.5, 0.5])
        np.testing.assert_allclose(pmf, [0.25, 0.5, 0.25])
        joint, outcomes = simulation.exact_joint_probabilities(pmf, pmf)
        self.assertEqual(joint.shape, (3, 3))
        self.assertAlmostEqual(outcomes["home_win"], 0.3125)
        self.assertAlmostEqual(outcomes["draw"], 0.375)
        self.assertAlmostEqual(outcomes["away_win"], 0.3125)
        self.assertAlmostEqual(sum(outcomes.values()), 1.0)
        home_points, away_points = simulation._expected_points(outcomes)
        self.assertAlmostEqual(home_points + away_points, 3 - outcomes["draw"])

    def test_poisson_binomial_empty_and_deterministic_shots(self):
        np.testing.assert_array_equal(simulation.exact_poisson_binomial([]), [1.0])
        np.testing.assert_array_equal(simulation.exact_poisson_binomial([0.0, 1.0, 1.0]), [0.0, 0.0, 1.0, 0.0])
        _, outcomes = simulation.exact_joint_probabilities([0.0, 1.0], [0.0, 1.0])
        self.assertEqual(outcomes, {"home_win": 0.0, "draw": 1.0, "away_win": 0.0})

    def test_invalid_probabilities_are_rejected(self):
        for probabilities in ([1.1], [-0.1], [np.nan], [np.inf], [[0.2, 0.3]]):
            with self.subTest(probabilities=probabilities), self.assertRaises(simulation.SimulationError):
                simulation.exact_poisson_binomial(probabilities)

    def test_seeded_simulation_is_chunk_invariant_and_keyed_by_match_side(self):
        probabilities = [0.0, 0.07, 0.35, 1.0]
        one_chunk = simulation.simulate_goals(probabilities, 10_007, 765, 0, chunk_size=10_007)
        many_chunks = simulation.simulate_goals(probabilities, 10_007, 765, 0, chunk_size=613)
        np.testing.assert_array_equal(one_chunk, many_chunks)
        rerun = simulation.simulate_goals(probabilities, 10_007, 765, 0, chunk_size=1_000)
        np.testing.assert_array_equal(one_chunk, rerun)
        different_side = simulation.simulate_goals(probabilities, 10_007, 765, 1, chunk_size=1_000)
        different_match = simulation.simulate_goals(probabilities, 10_007, 766, 0, chunk_size=1_000)
        self.assertFalse(np.array_equal(one_chunk, different_side))
        self.assertFalse(np.array_equal(one_chunk, different_match))

    def test_mc_checks_match_exact_mean_and_outcome_probabilities(self):
        home_p = np.array([0.05, 0.22, 0.4, 0.7])
        away_p = np.array([0.1, 0.25, 0.55])
        home_pmf = simulation.exact_poisson_binomial(home_p)
        away_pmf = simulation.exact_poisson_binomial(away_p)
        _, exact = simulation.exact_joint_probabilities(home_pmf, away_pmf)
        home = simulation.simulate_goals(home_p, 100_000, 21, 0)
        away = simulation.simulate_goals(away_p, 100_000, 21, 1)
        checks = simulation._check_sampling(home, away, home_p, away_p, exact)
        self.assertEqual(checks["trials"], 100_000)
        self.assertTrue(abs(checks["home_mean"] - home_p.sum()) <= checks["home_mean_tolerance"])
        self.assertTrue(abs(checks["away_mean"] - away_p.sum()) <= checks["away_mean_tolerance"])

    def test_shot_prediction_join_rejects_event_match_and_label_mismatch(self):
        shots, predictions, matches = sample_join_data()
        joined, _ = simulation.join_test_shots(shots, predictions, matches, {501})
        self.assertEqual(joined.sort_values("event_index")["p_goal"].tolist(), [0.2, 0.3])
        bad = predictions.copy()
        bad.loc[0, "match_id"] = 999
        with self.assertRaisesRegex(simulation.SimulationError, "match IDs"):
            simulation.join_test_shots(shots, bad, matches, {501})
        bad = predictions.copy()
        bad.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(simulation.SimulationError, "labels"):
            simulation.join_test_shots(shots, bad, matches, {501})
        bad = predictions.iloc[:-1]
        with self.assertRaisesRegex(simulation.SimulationError, "exact one-to-one"):
            simulation.join_test_shots(shots, bad, matches, {501})

    def test_match_team_ids_must_belong_to_home_or_away(self):
        shots, predictions, matches = sample_join_data()
        shots.loc[0, "team_id"] = 5
        with self.assertRaisesRegex(simulation.SimulationError, "not the corresponding"):
            simulation.join_test_shots(shots, predictions, matches, {501})

    def test_own_goal_credits_must_match_beneficiary_and_official_reconciliation(self):
        _, _, matches = sample_join_data()
        goals = pd.DataFrame([{"match_id": 501, "event_id": "og-1", "beneficiary_team_id": 10}])
        simulation.validate_own_goal_credits(goals, matches)
        goals.loc[0, "beneficiary_team_id"] = 20
        with self.assertRaisesRegex(simulation.SimulationError, "do not corroborate"):
            simulation.validate_own_goal_credits(goals, matches)
        goals.loc[0, "beneficiary_team_id"] = 99
        with self.assertRaisesRegex(simulation.SimulationError, "do not belong"):
            simulation.validate_own_goal_credits(goals, matches)

    def test_small_match_build_emits_full_joint_support_and_exact_monte_carlo_counts(self):
        shots, predictions, matches = sample_join_data()
        joined, match_info = simulation.join_test_shots(shots, predictions, matches, {501})
        own_goals = pd.DataFrame([{"match_id": 501, "beneficiary_team_id": 10}])
        with patch.object(simulation, "TRIALS", 1_000), patch.object(simulation, "CHUNK_SIZE", 137), patch.object(simulation, "EXAMPLE_CHECKPOINTS", (100, 1_000)):
            tables, figures, info = simulation._build_outputs(joined, match_info, own_goals, 501)
        summary = tables[simulation.TABLE_NAMES["match_summary"]]
        scorelines = tables[simulation.TABLE_NAMES["scoreline_distribution"]]
        self.assertEqual(info["match_count"], 1)
        self.assertEqual(info["trial_count"], 1_000)
        self.assertEqual(len(scorelines), 4)  # full 0..1 by 0..1 score support
        self.assertAlmostEqual(scorelines.exact_probability.sum(), 1.0)
        self.assertEqual(int(scorelines.mc_count.sum()), 1_000)
        self.assertEqual(int(summary.iloc[0].own_goals_credited_home), 1)
        self.assertEqual(int(summary.iloc[0].official_home_goals), 1)
        self.assertEqual(len(tables[simulation.TABLE_NAMES["convergence"]]), 6)
        self.assertEqual(len(figures), 3)

    def test_locked_protocol_refuses_changed_content_and_input_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            target = root / "locked.bin"
            original = b"pinned"
            target.write_bytes(original)
            entry = {"bytes": len(original), "sha256": acquire_data.digest(original)}
            self.assertEqual(simulation.verify_hashed_file(root, "locked.bin", entry, "fixture"), original)
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(simulation.SimulationError, "hash or byte count"):
                simulation.verify_hashed_file(root, "locked.bin", entry, "fixture")

            protocol_path = root / simulation.PROTOCOL_RELATIVE
            protocol_path.parent.mkdir(parents=True, exist_ok=True)
            protocol_path.write_bytes(b'{"seed":42}\n')
            with self.assertRaisesRegex(simulation.SimulationError, "refusing to overwrite"):
                simulation._lock_protocol(root, {"seed": 43}, b'{"seed":43}\n')
            self.assertEqual(protocol_path.read_bytes(), b'{"seed":42}\n')


def sample_join_data():
    shots = pd.DataFrame([
        {"match_id": 501, "event_id": "home-shot", "event_index": 1, "team_id": 10, "team_name": "Home", "is_home": 1, "is_goal": 0},
        {"match_id": 501, "event_id": "away-shot", "event_index": 2, "team_id": 20, "team_name": "Away", "is_home": 0, "is_goal": 1},
    ])
    shots["home_team_id"] = 10
    shots["home_team_name"] = "Home"
    shots["away_team_id"] = 20
    shots["away_team_name"] = "Away"
    shots["match_date"] = "2015-08-01"
    predictions = pd.DataFrame([
        {"event_id": "home-shot", "match_id": 501, "y_true": 0, "p_goal": 0.2},
        {"event_id": "away-shot", "match_id": 501, "y_true": 1, "p_goal": 0.3},
    ])
    matches = pd.DataFrame([{
        "match_id": 501, "match_date": "2015-08-01", "home_team_id": 10, "home_team_name": "Home",
        "away_team_id": 20, "away_team_name": "Away", "official_home_goals": 1,
        "official_away_goals": 1, "shot_goals_home": 0, "shot_goals_away": 1,
        "own_goals_credited_home": 1, "own_goals_credited_away": 0,
    }])
    return shots, predictions, matches


if __name__ == "__main__":
    unittest.main()
