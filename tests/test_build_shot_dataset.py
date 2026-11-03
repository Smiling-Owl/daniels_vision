import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_shot_dataset as builder
from scripts import acquire_data as acquisition


def match(match_id=1, home_score=1, away_score=0):
    return {
        "match_id": match_id, "match_date": "2016-01-01", "competition": {"competition_id": 11},
        "season": {"season_id": 27}, "home_team": {"home_team_id": 10, "home_team_name": "Home"},
        "away_team": {"away_team_id": 20, "away_team_name": "Away"},
        "home_score": home_score, "away_score": away_score,
    }


def shot(event_id="shot-1", outcome="Goal", location=None, xg=0.2):
    return {
        "id": event_id, "index": 10, "period": 1, "minute": 5, "second": 3, "possession": 2,
        "type": {"name": "Shot"}, "team": {"id": 10, "name": "Home"},
        "player": {"id": 99, "name": "Striker"}, "location": [108, 40] if location is None else location,
        "play_pattern": {"name": "Regular Play"},
        "shot": {"outcome": {"name": outcome}, "body_part": {"name": "Right Foot"},
                 "type": {"name": "Open Play"}, "technique": {"name": "Normal"},
                 "statsbomb_xg": xg},
    }


class ShotDatasetBuilderTests(unittest.TestCase):
    def test_geometry_is_symmetric_and_uses_native_120_by_80_units(self):
        distance, angle = builder.geometry(108, 40)
        self.assertAlmostEqual(distance, 12.0)
        self.assertAlmostEqual(angle, 2 * __import__("math").atan2(4, 12))
        self.assertEqual(builder.geometry(108, 30), builder.geometry(108, 50))

    def test_geometry_rejects_undefined_post_angle(self):
        with self.assertRaisesRegex(builder.DatasetBuildError, "undefined"):
            builder.geometry(120, 36)
        with self.assertRaisesRegex(builder.DatasetBuildError, "undefined"):
            builder.geometry(120, 44)

    def test_rejects_unrecognized_outcome_and_missing_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [shot(outcome="Unknown")], [match(home_score=0)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "unrecognized outcome"):
                    builder.build_records(project)
            missing_location = shot()
            missing_location.pop("location")
            self._prepare(project, [missing_location], [match(home_score=1)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "missing location"):
                    builder.build_records(project)

    def test_rejects_nonfinite_or_out_of_range_xg(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [shot(xg=float("nan"))], [match()])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(acquisition.AcquisitionError, "non-finite"):
                    builder.build_records(project)

    def test_own_goal_against_is_counted_once_and_for_is_audit_only(self):
        against = {
            "id": "against", "index": 20, "period": 2, "minute": 70, "second": 0,
            "type": {"name": "Own Goal Against"}, "team": {"id": 20, "name": "Away"},
            "player": {"id": 98, "name": "Defender"}, "related_events": ["for"],
        }
        own_for = {
            "id": "for", "index": 19, "period": 2, "minute": 70, "second": 0,
            "type": {"name": "Own Goal For"}, "team": {"id": 10, "name": "Home"},
            "player": {"id": 97, "name": "Forward"}, "related_events": ["against"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [shot(), own_for, against], [match(home_score=2)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                shots, recon, own_goals, _ = builder.build_records(project)
            self.assertEqual(len(shots), 1)
            self.assertEqual(len(own_goals), 1)
            self.assertEqual(own_goals[0]["own_goal_for_event_id"], "for")
            self.assertEqual(recon[0]["shot_goals_home"], 1)
            self.assertEqual(recon[0]["own_goals_conceded_home"], 0)
            self.assertEqual(recon[0]["own_goals_conceded_away"], 1)
            self.assertEqual(recon[0]["own_goals_credited_home"], 1)
            self.assertEqual(recon[0]["own_goals_credited_away"], 0)
            self.assertEqual(recon[0]["reconciled_home_goals"], 2)

    def test_own_goal_for_must_match_beneficiary_team(self):
        against = {
            "id": "against", "index": 20, "period": 2, "minute": 70, "second": 0,
            "type": {"name": "Own Goal Against"}, "team": {"id": 10, "name": "Home"},
            "player": {"id": 98, "name": "Defender"}, "related_events": ["for"],
        }
        misplaced_for = {
            "id": "for", "index": 19, "period": 2, "minute": 70, "second": 0,
            "type": {"name": "Own Goal For"}, "team": {"id": 10, "name": "Home"},
            "player": {"id": 97, "name": "Forward"}, "related_events": ["against"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [misplaced_for, against], [match(home_score=0, away_score=1)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "wrong beneficiary"):
                    builder.build_records(project)

    def test_reconciliation_mismatch_fails_instead_of_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [shot(outcome="Blocked")], [match(home_score=1)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "reconciliation failed"):
                    builder.build_records(project)
            self.assertFalse((project / "data/processed/shots.csv").exists())

    def test_model_feature_allowlist_excludes_leakage_columns(self):
        expected = [
            "distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique",
            "play_pattern", "under_pressure", "first_time", "open_goal",
        ]
        self.assertEqual(builder.FEATURE_COLUMNS, expected)
        self.assertTrue({"is_goal", "statsbomb_xg", "shot_outcome", "match_id", "event_id"}.isdisjoint(builder.FEATURE_COLUMNS))

    def test_rejects_duplicate_shot_identifiers(self):
        first = shot(event_id="duplicate", outcome="Blocked")
        second = shot(event_id="duplicate", outcome="Wayward")
        second["index"] = 11
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [first, second], [match(home_score=0)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, r"Duplicate event ID"):
                    builder.build_records(project)


    def test_rejects_duplicate_event_id_across_match_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [shot(event_id="reused", outcome="Blocked")], [
                match(match_id=1, home_score=0), match(match_id=2, home_score=0)
            ])
            raw = project / "data/raw/statsbomb" / acquisition.SOURCE["commit"]
            (raw / "data/events/2.json").write_text(
                json.dumps([shot(event_id="reused", outcome="Blocked")]), encoding="utf-8"
            )
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1, 2]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "Duplicate event ID in source snapshot"):
                    builder.build_records(project)


    def test_off_camera_flags_and_freeze_frame_absence_are_reported_audit_only(self):
        flagged = shot(event_id="cam-true", outcome="Blocked")
        flagged["off_camera"] = True
        flagged["shot"]["off_camera"] = False  # Conflicting nested value must be ignored.
        not_flagged = shot(event_id="cam-false", outcome="Blocked")
        not_flagged["off_camera"] = False
        not_flagged["shot"]["off_camera"] = True  # Conflicting nested value must be ignored.
        absent = shot(event_id="cam-absent", outcome="Blocked")
        for index, event in enumerate((flagged, not_flagged, absent), start=10):
            event["index"] = index
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [flagged, not_flagged, absent], [match(home_score=0)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                report = builder.build(project)
            self.assertEqual(report["off_camera_counts"], {"absent": 1, "false": 1, "true": 1})
            self.assertEqual(report["freeze_frame_missing_count"], 3)
            rows = builder.pd.read_csv(project / "data/processed/shots.csv")
            self.assertEqual(rows["off_camera"].tolist(), [1, 0, 0])
            self.assertNotIn("off_camera", builder.FEATURE_COLUMNS)

    def test_official_scores_must_be_nonnegative_integers_not_booleans(self):
        for invalid in (True, False, -1, 1.0):
            with self.subTest(score=invalid), self.assertRaisesRegex(builder.DatasetBuildError, "non-negative integer"):
                builder._nonnegative_int(invalid, "official score")
        self.assertEqual(builder._nonnegative_int(0, "official score"), 0)

    def test_explicit_null_boolean_flag_is_rejected(self):
        event = shot(outcome="Blocked")
        event["under_pressure"] = None
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self._prepare(project, [event], [match(home_score=0)])
            with patch.object(acquisition, "run"), patch.object(acquisition, "validate_matches", return_value=[1]):
                with self.assertRaisesRegex(builder.DatasetBuildError, "under_pressure must be true, false, or absent"):
                    builder.build_records(project)

    @staticmethod
    def _prepare(project: Path, events: list[dict], matches: list[dict]) -> None:
        raw = project / "data/raw/statsbomb" / acquisition.SOURCE["commit"]
        (raw / "data/events").mkdir(parents=True, exist_ok=True)
        (raw / "data/matches/11").mkdir(parents=True, exist_ok=True)
        (raw / "data/matches/11/27.json").write_text(json.dumps(matches), encoding="utf-8")
        (raw / "data/events/1.json").write_text(json.dumps(events), encoding="utf-8")
        manifest = acquisition.manifest_template()
        manifest["complete"] = True
        manifest["files"]["data/matches/11/27.json"] = {"sha256": "fixture"}
        acquisition.save_manifest(raw / "manifest.json", manifest)


if __name__ == "__main__":
    unittest.main()
