"""Fetch and verify a pinned StatsBomb Open Data snapshot (stdlib only)."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import zlib
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE = {
    "repository": "https://github.com/hudl/open-data",
    "commit": "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb",
    "commit_date": "2026-09-07T16:27:15Z",
    "competition_id": 11, "season_id": 27, "season": "2015/2016",
    "expected_match_count": 380, "expected_team_count": 20, "matches_per_team": 38,
}
BASE_URL = f"https://raw.githubusercontent.com/hudl/open-data/{SOURCE['commit']}"
RETRIES, TIMEOUT, WORKERS, PROGRESS_EVERY = 3, 60, 6, 20


class AcquisitionError(RuntimeError):
    """Invalid source data or a failed integrity check."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json(data: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite value {value}")
    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite value {value}")
        return parsed
    try:
        return json.loads(data.decode("utf-8"), parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AcquisitionError(f"Invalid JSON in {label}: {exc}") from exc


def validate_competitions(value: Any) -> None:
    if not isinstance(value, list) or not any(
        isinstance(row, dict) and row.get("competition_id") == 11 and row.get("season_id") == 27
        for row in value
    ):
        raise AcquisitionError("Pinned index does not list La Liga 2015/16 (11/27).")


def validate_matches(value: Any, count: int = 380, teams: int = 20, per_team: int = 38) -> list[int]:
    if not isinstance(value, list) or len(value) != count:
        raise AcquisitionError(f"Expected {count} matches; found {len(value) if isinstance(value, list) else 'non-list'}.")
    ids: list[int] = []
    frequencies: dict[str, int] = {}
    for index, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise AcquisitionError(f"Match row {index} is not an object.")
        match_id = row.get("match_id")
        if isinstance(match_id, bool) or not isinstance(match_id, int) or match_id <= 0:
            raise AcquisitionError(f"Match row {index} has an invalid match_id.")
        ids.append(match_id)
        competition = row.get("competition")
        season = row.get("season")
        if not isinstance(competition, dict) or competition.get("competition_id") != 11:
            raise AcquisitionError(f"Match {match_id} is not in competition 11.")
        if not isinstance(season, dict) or season.get("season_id") != 27:
            raise AcquisitionError(f"Match {match_id} is not in season 27.")
        names: list[str] = []
        for side, field in (("home_team", "home_team_name"), ("away_team", "away_team_name")):
            team = row.get(side)
            name = team.get(field) if isinstance(team, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise AcquisitionError(f"Match {match_id} has invalid {side}.")
            names.append(name.strip())
        if names[0] == names[1]:
            raise AcquisitionError(f"Match {match_id} has the same team twice.")
        for name in names:
            frequencies[name] = frequencies.get(name, 0) + 1
    if len(set(ids)) != count:
        raise AcquisitionError("Match index contains duplicate match IDs.")
    if len(frequencies) != teams:
        raise AcquisitionError(f"Expected {teams} teams; found {len(frequencies)}.")
    invalid = {name: n for name, n in frequencies.items() if n != per_team}
    if invalid:
        raise AcquisitionError(f"Expected {per_team} matches per team: {invalid}.")
    return ids


def validate_events(value: Any, match_id: int) -> int:
    if not isinstance(value, list) or not value:
        raise AcquisitionError(f"Match {match_id} events must be a non-empty list.")
    seen: set[str] = set()
    for index, event in enumerate(value, start=1):
        if not isinstance(event, dict):
            raise AcquisitionError(f"Match {match_id} event {index} is not an object.")
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise AcquisitionError(f"Match {match_id} event {index} has invalid ID.")
        if event_id in seen:
            raise AcquisitionError(f"Match {match_id} has duplicate event ID {event_id}.")
        seen.add(event_id)
        kind = event.get("type")
        if not isinstance(kind, dict) or not isinstance(kind.get("name"), str) or not kind["name"].strip():
            raise AcquisitionError(f"Match {match_id} event {event_id} has invalid type.")
    return len(value)


def validate_bytes(path: str, data: bytes) -> int | None:
    if path == "data/competitions.json":
        validate_competitions(parse_json(data, path))
    elif path == "data/matches/11/27.json":
        validate_matches(parse_json(data, path))
    elif path.startswith("data/events/") and path.endswith(".json"):
        try:
            match_id = int(Path(path).stem)
        except ValueError as exc:
            raise AcquisitionError(f"Invalid event file path: {path}") from exc
        return validate_events(parse_json(data, path), match_id)
    elif path == "LICENSE.pdf":
        if not data.startswith(b"%PDF"):
            raise AcquisitionError("Upstream licence does not start with %PDF.")
    elif path == "README.md":
        if not data.strip():
            raise AcquisitionError("Upstream README is empty.")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AcquisitionError("Upstream README is not UTF-8.") from exc
    else:
        raise AcquisitionError(f"Unexpected repository path: {path}")
    return None


def expected_files(ids: list[int]) -> list[str]:
    return ["data/competitions.json", "data/matches/11/27.json", "README.md", "LICENSE.pdf"] + [
        f"data/events/{match_id}.json" for match_id in ids
    ]


def url_for(path: str) -> str:
    return f"{BASE_URL}/{path}"


def manifest_template() -> dict[str, Any]:
    return {"manifest_version": 1, **SOURCE, "complete": False, "files": {}}


def atomic_replace(temp: Path, path: Path) -> None:
    """Replace a file, retrying only transient Windows sharing/access locks."""
    delays = (0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temp, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return manifest_template()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcquisitionError("Manifest must be a JSON object.")
    expected = manifest_template()
    for key, expected_value in expected.items():
        if key in ("files", "complete"):
            continue
        if value.get(key) != expected_value:
            raise AcquisitionError(f"Manifest source metadata mismatch: {key}.")
    if not isinstance(value.get("complete"), bool):
        raise AcquisitionError("Manifest 'complete' must be true or false.")
    if not isinstance(value.get("files"), dict):
        raise AcquisitionError("Manifest files must be an object.")
    return value
def check_cache(root: Path, path: str, manifest: dict[str, Any]) -> tuple[bytes, int | None] | None:
    record = manifest["files"].get(path)
    if record is None:
        return None  # Never trust unmanifested bytes as a cache.
    if not isinstance(record, dict) or record.get("source_url") != url_for(path):
        raise AcquisitionError(f"Invalid manifest source for {path}.")
    local = root / path
    if not local.exists():
        return None
    data = local.read_bytes()
    if len(data) != record.get("bytes") or digest(data) != record.get("sha256"):
        raise AcquisitionError(f"Cached file failed manifest integrity check: {path}.")
    count = validate_bytes(path, data)
    if path.startswith("data/events/") and record.get("event_count") != count:
        raise AcquisitionError(f"Cached event count differs from manifest: {path}.")
    return data, count


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "beyond-scoreline/1.0", "Accept-Encoding": "gzip"},
    )
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status != 200:
                    raise urllib.error.URLError(f"HTTP {response.status}")
                payload = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        return gzip.decompress(payload)
                    except (OSError, EOFError, zlib.error) as exc:
                        raise AcquisitionError(f"Invalid gzip response from {url}: {exc}") from exc
                return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < RETRIES:
            time.sleep(min(2**attempt, 4))
    raise AcquisitionError(f"Download failed after {RETRIES} attempts: {url}: {last_error}")


def acquire_one(root: Path, path: str, manifest: dict[str, Any], manifest_path: Path, lock: threading.Lock) -> tuple[int, int | None]:
    with lock:
        cached = check_cache(root, path, manifest)
    if cached is not None:
        return len(cached[0]), cached[1]
    data = fetch(url_for(path))
    count = validate_bytes(path, data)
    record: dict[str, Any] = {
        "relative_path": path, "source_url": url_for(path), "sha256": digest(data),
        "bytes": len(data), "retrieved_at_utc": utc_now(),
    }
    if path.startswith("data/events/"):
        record["event_count"] = count
    atomic_write(root / path, data)
    with lock:
        manifest["files"][path] = record
        manifest["complete"] = False
        save_manifest(manifest_path, manifest)
    return len(data), count


def verify_local(root: Path, paths: list[str], manifest: dict[str, Any]) -> dict[str, Any]:
    missing = [path for path in paths if path not in manifest["files"]]
    if missing:
        raise AcquisitionError(f"Manifest incomplete; first missing file: {missing[0]}")
    byte_count = event_count = 0
    for path in paths:
        result = check_cache(root, path, manifest)
        if result is None:
            raise AcquisitionError(f"Verification failed; missing file: {path}")
        data, count = result
        byte_count += len(data)
        event_count += count or 0
    if manifest.get("complete") is not True:
        raise AcquisitionError("Manifest is not marked complete.")
    return {"files": len(paths), "bytes": byte_count, "events": event_count}


def run(root: Path, verify_only: bool = False) -> dict[str, Any]:
    raw = root / "data/raw/statsbomb" / SOURCE["commit"]
    manifest_path = raw / "manifest.json"
    manifest = load_manifest(manifest_path)
    if verify_only:
        # This branch is local-only and never calls fetch().
        validate_competitions(parse_json((raw / "data/competitions.json").read_bytes(), "competitions"))
        ids = validate_matches(parse_json((raw / "data/matches/11/27.json").read_bytes(), "matches"))
        summary = verify_local(raw, expected_files(ids), manifest)
        summary["manifest"] = str(manifest_path)
        return summary

    lock = threading.Lock()
    sizes: list[int] = []
    for path in ("data/competitions.json", "data/matches/11/27.json", "README.md", "LICENSE.pdf"):
        size, _ = acquire_one(raw, path, manifest, manifest_path, lock)
        sizes.append(size)
    ids = validate_matches(parse_json((raw / "data/matches/11/27.json").read_bytes(), "matches"))
    event_paths = [f"data/events/{match_id}.json" for match_id in ids]
    events = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(acquire_one, raw, path, manifest, manifest_path, lock) for path in event_paths]
        for i, future in enumerate(as_completed(futures), start=1):
            size, count = future.result()
            sizes.append(size)
            events += count or 0
            if i % PROGRESS_EVERY == 0 or i == len(futures):
                print(f"Validated event files: {i}/{len(futures)}", flush=True)
    paths = expected_files(ids)
    with lock:
        missing = [path for path in paths if path not in manifest["files"]]
        if missing:
            raise AcquisitionError(f"Incomplete acquisition; first missing file: {missing[0]}")
        manifest["complete"] = True
        manifest["completed_at_utc"] = utc_now()
        save_manifest(manifest_path, manifest)
    return {"matches": len(ids), "event_files": len(event_paths), "events": events,
            "bytes_downloaded_or_verified": sum(sizes), "files": len(paths), "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify local files without network access.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = run(args.root.resolve(), verify_only=args.verify_only)
    except (AcquisitionError, OSError) as exc:
        print(f"Acquisition failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




