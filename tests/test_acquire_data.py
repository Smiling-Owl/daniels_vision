import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import acquire_data as acquisition


def match(match_id: int, home: str = "A", away: str = "B") -> dict:
    return {
        "match_id": match_id,
        "competition": {"competition_id": 11},
        "season": {"season_id": 27},
        "home_team": {"home_team_name": home},
        "away_team": {"away_team_name": away},
    }


def full_league() -> list[dict]:
    teams = [f"Team {number:02d}" for number in range(20)]
    one_leg = []
    for round_number in range(19):
        rotated = [teams[0], *teams[1 + round_number :], *teams[1 : 1 + round_number]]
        for index in range(10):
            one_leg.append((rotated[index], rotated[19 - index]))
    fixtures = one_leg + [(away, home) for home, away in one_leg]
    return [match(index + 1, home, away) for index, (home, away) in enumerate(fixtures)]


class AcquisitionValidationTests(unittest.TestCase):
    def test_match_validation_checks_schedule_ids_and_team_coverage(self) -> None:
        self.assertEqual(
            acquisition.validate_matches(
                [match(1, "A", "B"), match(2, "B", "A")],
                count=2,
                teams=2,
                per_team=2,
            ),
            [1, 2],
        )

    def test_match_validation_rejects_duplicate_id_and_wrong_competition(self) -> None:
        with self.assertRaisesRegex(acquisition.AcquisitionError, "duplicate"):
            acquisition.validate_matches(
                [match(1, "A", "B"), match(1, "B", "A")],
                count=2,
                teams=2,
                per_team=2,
            )
        wrong = match(1)
        wrong["competition"]["competition_id"] = 43
        with self.assertRaisesRegex(acquisition.AcquisitionError, "competition"):
            acquisition.validate_matches([wrong], count=1, teams=2, per_team=1)

    def test_event_validation_rejects_duplicate_ids_and_missing_types(self) -> None:
        event = {"id": "event-1", "type": {"name": "Pass"}}
        with self.assertRaisesRegex(acquisition.AcquisitionError, "duplicate event"):
            acquisition.validate_events([event, event], 99)
        with self.assertRaisesRegex(acquisition.AcquisitionError, "invalid type"):
            acquisition.validate_events([{"id": "event-2"}], 99)

    def test_json_validation_rejects_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(acquisition.AcquisitionError, "non-finite"):
            acquisition.parse_json(b'{"value": Infinity}', "fixture")
        with self.assertRaisesRegex(acquisition.AcquisitionError, "non-finite"):
            acquisition.parse_json(b'{"value": 1e999}', "fixture")

    def test_manifest_config_matches_source_snapshot(self) -> None:
        root = Path(__file__).resolve().parents[1]
        snapshot = json.loads((root / "docs/source_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["commit"], acquisition.SOURCE["commit"])
        self.assertEqual(snapshot["dataset"]["competition_id"], acquisition.SOURCE["competition_id"])
        self.assertEqual(snapshot["dataset"]["season_id"], acquisition.SOURCE["season_id"])
        self.assertEqual(snapshot["dataset"]["expected_match_count"], acquisition.SOURCE["expected_match_count"])

    def test_corrupt_cached_file_fails_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = "README.md"
            expected_bytes = b"trusted snapshot"
            (root / path).write_bytes(b"modified snapshot")
            manifest = acquisition.manifest_template()
            manifest["files"][path] = {
                "relative_path": path,
                "source_url": acquisition.url_for(path),
                "sha256": acquisition.digest(expected_bytes),
                "bytes": len(expected_bytes),
                "retrieved_at_utc": "2026-10-22T00:00:00Z",
            }
            with self.assertRaisesRegex(acquisition.AcquisitionError, "integrity"):
                acquisition.check_cache(root, path, manifest)

    def test_verify_only_accepts_complete_snapshot_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            raw = project / "data/raw/statsbomb" / acquisition.SOURCE["commit"]
            matches = full_league()
            files: dict[str, bytes] = {
                "data/competitions.json": json.dumps(
                    [{"competition_id": 11, "season_id": 27}]
                ).encode(),
                "data/matches/11/27.json": json.dumps(matches).encode(),
                "README.md": b"StatsBomb Open Data",
                "LICENSE.pdf": b"%PDF-1.7 fixture",
            }
            for row in matches:
                path = f"data/events/{row['match_id']}.json"
                files[path] = json.dumps(
                    [{"id": f"event-{row['match_id']}", "type": {"name": "Pass"}}]
                ).encode()
            manifest = acquisition.manifest_template()
            manifest["complete"] = True
            for path, payload in files.items():
                (raw / path).parent.mkdir(parents=True, exist_ok=True)
                (raw / path).write_bytes(payload)
                record = {
                    "relative_path": path,
                    "source_url": acquisition.url_for(path),
                    "sha256": acquisition.digest(payload),
                    "bytes": len(payload),
                    "retrieved_at_utc": "2026-10-22T00:00:00Z",
                }
                if path.startswith("data/events/"):
                    record["event_count"] = 1
                manifest["files"][path] = record
            acquisition.save_manifest(raw / "manifest.json", manifest)

            with patch.object(acquisition, "fetch", side_effect=AssertionError("network called")):
                result = acquisition.run(project, verify_only=True)
            self.assertEqual(result["files"], 384)
            self.assertEqual(result["events"], 380)

    def test_atomic_write_retries_transient_windows_access_denied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "manifest.json"
            target.write_text("old", encoding="utf-8")
            real_replace = acquisition.os.replace
            attempts = 0

            def replace_with_one_transient_lock(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    error = PermissionError("transient Windows lock")
                    error.winerror = 5
                    raise error
                real_replace(source, destination)

            with patch.object(acquisition.os, "replace", side_effect=replace_with_one_transient_lock), patch.object(acquisition.time, "sleep") as sleep:
                acquisition.atomic_write(target, b"new")

            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(0.1)
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(list(Path(temporary).glob(".*.tmp")), [])

    def test_atomic_write_preserves_target_after_persistent_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "manifest.json"
            target.write_text("old", encoding="utf-8")

            def locked_replace(source, destination):
                error = PermissionError("persistent Windows lock")
                error.winerror = 32
                raise error

            with patch.object(acquisition.os, "replace", side_effect=locked_replace), patch.object(acquisition.time, "sleep") as sleep:
                with self.assertRaisesRegex(PermissionError, "persistent Windows lock"):
                    acquisition.atomic_write(target, b"new")

            self.assertEqual(sleep.call_count, 4)
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(temporary).glob(".*.tmp")), [])

    def test_fetch_requests_gzip_and_returns_identical_raw_bytes(self) -> None:
        import gzip

        class Response:
            status = 200

            def __init__(self, payload: bytes, encoding: str | None):
                self.payload = payload
                self.headers = {"Content-Encoding": encoding} if encoding else {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self) -> bytes:
                return self.payload

        original = b'{"event": "raw-json", "number": 1}'
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return Response(gzip.compress(original), "gzip") if len(requests) == 1 else Response(original, None)

        with patch.object(acquisition.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertEqual(acquisition.fetch("https://example.test/events.json"), original)
            self.assertEqual(acquisition.fetch("https://example.test/events.json"), original)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].get_header("Accept-encoding"), "gzip")
        self.assertEqual(requests[1].get_header("Accept-encoding"), "gzip")

if __name__ == "__main__":
    unittest.main()


