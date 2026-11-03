"""Build a validated, auditable shot-level table from the pinned raw snapshot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Support both `python scripts/build_shot_dataset.py` and package imports.
    from scripts import acquire_data as acquisition
except ImportError:  # pragma: no cover - direct execution from scripts directory
    import acquire_data as acquisition

SOURCE = acquisition.SOURCE
OUTCOMES = {
    "Blocked", "Goal", "Off T", "Saved", "Wayward", "Post",
    "Saved Off Target", "Saved to Post",
}
OWN_GOAL_TYPES = {"Own Goal Against", "Own Goal For"}
SHOT_COLUMNS = [
    "match_id", "event_id", "event_index", "match_date",
    "home_team_id", "home_team_name", "away_team_id", "away_team_name",
    "team_id", "team_name", "opponent_id", "opponent_name",
    "player_id", "player_name", "period", "minute", "second", "possession_id",
    "is_home", "is_penalty", "x", "y", "distance_sb_units", "angle_sb_radians",
    "body_part", "shot_type", "technique", "play_pattern", "under_pressure",
    "first_time", "open_goal", "off_camera", "is_goal", "shot_outcome", "statsbomb_xg",
]
RECONCILIATION_COLUMNS = [
    "match_id", "match_date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "official_home_goals", "official_away_goals",
    "shot_goals_home", "shot_goals_away", "own_goals_conceded_home",
    "own_goals_conceded_away", "own_goals_credited_home", "own_goals_credited_away",
    "reconciled_home_goals", "reconciled_away_goals", "reconciled",
]
FEATURE_COLUMNS = [
    "distance_sb_units", "angle_sb_radians", "body_part", "shot_type", "technique",
    "play_pattern", "under_pressure", "first_time", "open_goal",
]
OWN_GOAL_COLUMNS = [
    "match_id", "event_id", "event_index", "match_date", "period", "minute", "second",
    "offending_team_id", "offending_team_name", "beneficiary_team_id", "beneficiary_team_name",
    "player_id", "player_name", "own_goal_for_event_id",
]


class DatasetBuildError(RuntimeError):
    """Source data cannot safely produce the requested shot table."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetBuildError(f"{label} must be an object.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetBuildError(f"{label} must be a non-negative integer.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetBuildError(f"{label} must be a positive integer.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetBuildError(f"{label} must be a non-empty string.")
    return value.strip()


def _number(value: Any, label: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DatasetBuildError(f"{label} must be a finite number.")
    result = float(value)
    if low is not None and result < low or high is not None and result > high:
        raise DatasetBuildError(f"{label} is outside the expected range.")
    return result


def _name_id(entity: Any, label: str) -> tuple[int, str]:
    row = _object(entity, label)
    return _positive_int(row.get("id"), f"{label}.id"), _text(row.get("name"), f"{label}.name")


def _flag(container: dict[str, Any], key: str, label: str) -> bool:
    if key not in container:
        return False
    value = container[key]
    if not isinstance(value, bool):
        raise DatasetBuildError(f"{label}.{key} must be true, false, or absent.")
    return value


def geometry(x: float, y: float) -> tuple[float, float]:
    """StatsBomb 120x80 coordinates; goal center (120,40), posts (120,36),(120,44)."""
    dx = 120.0 - x
    dy_low, dy_high = 36.0 - y, 44.0 - y
    distance = math.hypot(dx, 40.0 - y)
    cross = abs(dx * dy_high - dy_low * dx)
    dot = dx * dx + dy_low * dy_high
    if cross == 0.0 and dot == 0.0:
        raise DatasetBuildError("Shot location makes the goal angle undefined at a post.")
    angle = math.atan2(cross, dot)
    return distance, angle


def _related_event_id(event: dict[str, Any], wanted: str, events_by_id: dict[str, dict[str, Any]]) -> str | None:
    related = event.get("related_events", [])
    if related is None:
        related = []
    if not isinstance(related, list):
        raise DatasetBuildError(f"Event {event.get('id')} related_events must be a list.")
    matches = [
        candidate for event_id in related
        if isinstance(event_id, str)
        and (candidate := events_by_id.get(event_id)) is not None
        and candidate.get("type", {}).get("name") == wanted
    ]
    if len(matches) > 1:
        raise DatasetBuildError(f"Own-goal event {event.get('id')} has multiple related {wanted} records.")
    return matches[0]["id"] if matches else None


def _validate_match_index(matches: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    try:
        ids = acquisition.validate_matches(matches)
    except acquisition.AcquisitionError as exc:
        raise DatasetBuildError(f"Invalid match index: {exc}") from exc
    return {match_id: row for match_id, row in zip(ids, matches)}


def build_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read and validate all sources; return records and a report before writing outputs."""
    raw = root / "data/raw/statsbomb" / SOURCE["commit"]
    manifest_path = raw / "manifest.json"
    manifest = acquisition.load_manifest(manifest_path)
    try:
        acquisition.run(root, verify_only=True)  # No network; verifies all source hashes and structure.
    except (acquisition.AcquisitionError, OSError) as exc:
        raise DatasetBuildError(f"Raw snapshot verification failed: {exc}") from exc

    matches_raw = acquisition.parse_json((raw / "data/matches/11/27.json").read_bytes(), "matches")
    match_index = _validate_match_index(matches_raw)
    shot_rows: list[dict[str, Any]] = []
    own_goal_rows: list[dict[str, Any]] = []
    period_exclusions = Counter()
    outcome_counts = Counter()
    own_against_by_beneficiary: Counter[tuple[int, int]] = Counter()
    own_for_by_team: Counter[tuple[int, int]] = Counter()
    source_event_count = 0
    seen_global_event_ids: set[str] = set()
    raw_flag_absent_counts = Counter()
    off_camera_counts = Counter()
    freeze_frame_missing_count = 0

    for match_id in sorted(match_index):
        match = match_index[match_id]
        home, away = _object(match.get("home_team"), "home_team"), _object(match.get("away_team"), "away_team")
        home_id, home_name = _positive_int(home.get("home_team_id"), "home_team_id"), _text(home.get("home_team_name"), "home_team_name")
        away_id, away_name = _positive_int(away.get("away_team_id"), "away_team_id"), _text(away.get("away_team_name"), "away_team_name")
        if home_id == away_id:
            raise DatasetBuildError(f"Match {match_id} repeats a team ID.")
        match_date = _text(match.get("match_date"), f"match {match_id} date")
        try:
            datetime.strptime(match_date, "%Y-%m-%d")
        except ValueError as exc:
            raise DatasetBuildError(f"Match {match_id} has invalid match_date {match_date!r}.") from exc
        official_home = _nonnegative_int(match.get("home_score"), f"match {match_id} home_score")
        official_away = _nonnegative_int(match.get("away_score"), f"match {match_id} away_score")
        path = raw / "data/events" / f"{match_id}.json"
        events = acquisition.parse_json(path.read_bytes(), f"events for match {match_id}")
        source_event_count += len(events)
        events_by_id = {event["id"]: event for event in events}
        match_shots: list[dict[str, Any]] = []
        match_own: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                raise DatasetBuildError(f"Match {match_id} contains a non-object event.")
            kind = _object(event.get("type"), f"match {match_id} event type").get("name")
            event_id = _text(event.get("id"), f"match {match_id} event ID")
            if event_id in seen_global_event_ids:
                raise DatasetBuildError(f"Duplicate event ID in source snapshot: {event_id}.")
            seen_global_event_ids.add(event_id)
            if kind == "Shot":
                period = event.get("period")
                if isinstance(period, bool) or not isinstance(period, int):
                    raise DatasetBuildError(f"Shot {event_id} has invalid period.")
                if period not in (1, 2):
                    period_exclusions[period] += 1
                    continue
                shot = _object(event.get("shot"), f"shot {event_id}")
                if "off_camera" not in event:
                    off_camera_counts["absent"] += 1
                    off_camera = False
                else:
                    off_camera = event["off_camera"]
                    if not isinstance(off_camera, bool):
                        raise DatasetBuildError(f"Event {event_id}.off_camera must be true, false, or absent.")
                    off_camera_counts[str(off_camera).lower()] += 1
                if "freeze_frame" not in shot or shot["freeze_frame"] is None:
                    freeze_frame_missing_count += 1
                for flag_name, container in (("under_pressure", event), ("first_time", shot), ("open_goal", shot)):
                    if flag_name not in container:
                        raw_flag_absent_counts[flag_name] += 1
                location = event.get("location")
                if not isinstance(location, list) or len(location) != 2:
                    raise DatasetBuildError(f"Shot {event_id} is missing location coordinates.")
                x = _number(location[0], f"shot {event_id} x", 0, 120)
                y = _number(location[1], f"shot {event_id} y", 0, 80)
                distance, angle = geometry(x, y)
                outcome = _object(shot.get("outcome"), f"shot {event_id} outcome")
                outcome_name = _text(outcome.get("name"), f"shot {event_id} outcome.name")
                if outcome_name not in OUTCOMES:
                    raise DatasetBuildError(f"Shot {event_id} has unrecognized outcome {outcome_name!r}.")
                body_part = _text(_object(shot.get("body_part"), f"shot {event_id} body_part").get("name"), f"shot {event_id} body_part.name")
                shot_type = _text(_object(shot.get("type"), f"shot {event_id} shot type").get("name"), f"shot {event_id} shot type.name")
                technique = _text(_object(shot.get("technique"), f"shot {event_id} technique").get("name"), f"shot {event_id} technique.name")
                xg = _number(shot.get("statsbomb_xg"), f"shot {event_id} statsbomb_xg", 0, 1)
                team_id, team_name = _name_id(event.get("team"), f"shot {event_id} team")
                if team_id == home_id:
                    opponent_id, opponent_name, is_home = away_id, away_name, True
                elif team_id == away_id:
                    opponent_id, opponent_name, is_home = home_id, home_name, False
                else:
                    raise DatasetBuildError(f"Shot {event_id} team is not a match participant.")
                player_id, player_name = _name_id(event.get("player"), f"shot {event_id} player")
                index = _positive_int(event.get("index"), f"shot {event_id} index")
                minute = event.get("minute")
                second = event.get("second")
                if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
                    raise DatasetBuildError(f"Shot {event_id} has invalid minute.")
                if isinstance(second, bool) or not isinstance(second, int) or not 0 <= second <= 59:
                    raise DatasetBuildError(f"Shot {event_id} has invalid second.")
                possession = event.get("possession")
                possession_id = _positive_int(possession, f"shot {event_id} possession") if possession is not None else None
                play_pattern = _text(_object(event.get("play_pattern"), f"shot {event_id} play_pattern").get("name"), f"shot {event_id} play_pattern.name")
                match_shots.append({
                    "match_id": match_id, "event_id": event_id, "event_index": index, "match_date": match_date,
                    "home_team_id": home_id, "home_team_name": home_name, "away_team_id": away_id, "away_team_name": away_name,
                    "team_id": team_id, "team_name": team_name, "opponent_id": opponent_id, "opponent_name": opponent_name,
                    "player_id": player_id, "player_name": player_name, "period": period, "minute": minute,
                    "second": second, "possession_id": possession_id, "is_home": int(is_home),
                    "is_penalty": int(shot_type == "Penalty"), "x": x, "y": y, "distance_sb_units": distance,
                    "angle_sb_radians": angle, "body_part": body_part, "shot_type": shot_type,
                    "technique": technique, "play_pattern": play_pattern,
                    "under_pressure": int(_flag(event, "under_pressure", f"event {event_id}")),
                    "first_time": int(_flag(shot, "first_time", f"shot {event_id}")),
                    "open_goal": int(_flag(shot, "open_goal", f"shot {event_id}")),
                    "off_camera": int(off_camera),
                    "is_goal": int(outcome_name == "Goal"), "shot_outcome": outcome_name, "statsbomb_xg": xg,
                })
                outcome_counts[outcome_name] += 1
            elif kind in OWN_GOAL_TYPES:
                period = event.get("period")
                if isinstance(period, bool) or not isinstance(period, int):
                    raise DatasetBuildError(f"Own-goal event {event_id} has invalid period.")
                if period not in (1, 2):
                    continue
                team_id, team_name = _name_id(event.get("team"), f"own-goal {event_id} team")
                if team_id == home_id:
                    beneficiary_id, beneficiary_name = away_id, away_name
                elif team_id == away_id:
                    beneficiary_id, beneficiary_name = home_id, home_name
                else:
                    raise DatasetBuildError(f"Own-goal event {event_id} team is not a match participant.")
                player = event.get("player")
                if player is None:
                    player_id, player_name = None, None
                else:
                    player_id, player_name = _name_id(player, f"own-goal {event_id} player")
                if kind == "Own Goal Against":
                    index = _positive_int(event.get("index"), f"own-goal {event_id} index")
                    related_id = _related_event_id(event, "Own Goal For", events_by_id)
                    if related_id is not None:
                        paired_team = _object(events_by_id[related_id].get("team"), f"own-goal pair {related_id} team")
                        if paired_team.get("id") != beneficiary_id:
                            raise DatasetBuildError(f"Own Goal For event {related_id} credits the wrong beneficiary.")
                    match_own.append({
                        "match_id": match_id, "event_id": event_id, "event_index": index,
                        "match_date": match_date, "period": period, "minute": event.get("minute"),
                        "second": event.get("second"), "offending_team_id": team_id,
                        "offending_team_name": team_name, "beneficiary_team_id": beneficiary_id,
                        "beneficiary_team_name": beneficiary_name, "player_id": player_id,
                        "player_name": player_name, "own_goal_for_event_id": related_id,
                    })
                    own_against_by_beneficiary[(match_id, beneficiary_id)] += 1
                else:
                    own_for_by_team[(match_id, team_id)] += 1
        shot_rows.extend(match_shots)
        own_goal_rows.extend(match_own)

    if not shot_rows:
        raise DatasetBuildError("No eligible regulation-time shot records found.")
    shot_ids = [(row["match_id"], row["event_id"]) for row in shot_rows]
    if len(shot_ids) != len(set(shot_ids)):
        raise DatasetBuildError("Duplicate (match_id, event_id) shot identifiers found.")
    # Construct each match's score audit. Own Goal Against records are credited to the
    # opposing team; paired Own Goal For records are corroboration only.
    recon_rows: list[dict[str, Any]] = []
    for match_id in sorted(match_index):
        match = match_index[match_id]
        home, away = match["home_team"], match["away_team"]
        home_id, away_id = home["home_team_id"], away["away_team_id"]
        home_shot_goals = sum(r["match_id"] == match_id and r["team_id"] == home_id and r["is_goal"] for r in shot_rows)
        away_shot_goals = sum(r["match_id"] == match_id and r["team_id"] == away_id and r["is_goal"] for r in shot_rows)
        og_home = own_against_by_beneficiary[(match_id, home_id)]  # Goals credited to home.
        og_away = own_against_by_beneficiary[(match_id, away_id)]  # Goals credited to away.
        ogf_home = own_for_by_team[(match_id, home_id)]
        ogf_away = own_for_by_team[(match_id, away_id)]
        rec_home, rec_away = home_shot_goals + og_home, away_shot_goals + og_away
        official_home, official_away = match["home_score"], match["away_score"]
        recon_rows.append({
            "match_id": match_id, "match_date": match["match_date"],
            "home_team_id": home_id, "home_team_name": home["home_team_name"],
            "away_team_id": away_id, "away_team_name": away["away_team_name"],
            "official_home_goals": official_home, "official_away_goals": official_away,
            "shot_goals_home": home_shot_goals, "shot_goals_away": away_shot_goals,
            "own_goals_conceded_home": og_away, "own_goals_conceded_away": og_home,
            "own_goals_credited_home": og_home, "own_goals_credited_away": og_away,
            "reconciled_home_goals": rec_home, "reconciled_away_goals": rec_away,
            "reconciled": rec_home == official_home and rec_away == official_away,
        })
    if any(not row["reconciled"] for row in recon_rows):
        failed = [row["match_id"] for row in recon_rows if not row["reconciled"]]
        raise DatasetBuildError(f"Goal reconciliation failed for match IDs: {failed[:10]}.")
    if len(own_goal_rows) != sum(own_against_by_beneficiary.values()):
        raise DatasetBuildError("Own Goal Against row count does not match aggregation.")
    if sum(own_against_by_beneficiary.values()) != sum(own_for_by_team.values()):
        raise DatasetBuildError("Own Goal Against / Own Goal For counts do not match.")
    for match_id, row in match_index.items():
        for team_id in (row["home_team"]["home_team_id"], row["away_team"]["away_team_id"]):
            if own_against_by_beneficiary[(match_id, team_id)] != own_for_by_team[(match_id, team_id)]:
                raise DatasetBuildError(f"Own-goal corroboration mismatch for match {match_id}, team {team_id}.")

    shot_rows.sort(key=lambda r: (r["match_date"], r["match_id"], r["event_index"]))
    own_goal_rows.sort(key=lambda r: (r["match_date"], r["match_id"], r["event_index"]))
    return shot_rows, recon_rows, own_goal_rows, {
        "source_event_count": source_event_count,
        "duplicate_event_id_count": 0,
        "duplicate_event_id_check_passed": True,
        "flag_absent_counts": dict(sorted(raw_flag_absent_counts.items())),
        "off_camera_counts": dict(sorted(off_camera_counts.items())),
        "freeze_frame_missing_count": freeze_frame_missing_count,
        "period_exclusions": {str(k): v for k, v in sorted(period_exclusions.items())},
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "manifest": manifest,
        "raw_root": raw,
    }


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    frame = pd.DataFrame(rows, columns=columns)
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def build(root: Path) -> dict[str, Any]:
    shots, reconciliation, own_goals, details = build_records(root)
    shot_frame = pd.DataFrame(shots, columns=SHOT_COLUMNS)
    feature_columns = FEATURE_COLUMNS
    required_columns = feature_columns + [
        "match_id", "event_id", "event_index", "player_id", "player_name", "period",
        "is_goal", "shot_outcome", "statsbomb_xg", "x", "y", "team_id", "opponent_id",
    ]
    missing_required = [column for column in required_columns if shot_frame[column].isna().any()]
    if missing_required:
        raise DatasetBuildError(f"Required shot columns contain missing values: {missing_required}.")
    if any(column not in shot_frame for column in feature_columns):
        raise DatasetBuildError("Model feature allowlist is missing a column.")
    source_manifest = details["manifest"]
    match_record = source_manifest["files"]["data/matches/11/27.json"]
    output_data = {
        "data/processed/shots.csv": _csv_bytes(shots, SHOT_COLUMNS),
        "data/processed/goal_reconciliation.csv": _csv_bytes(reconciliation, RECONCILIATION_COLUMNS),
        "data/processed/own_goals.csv": _csv_bytes(own_goals, OWN_GOAL_COLUMNS),
    }
    report = {
        "status": "passed", "source": {
            "repository": SOURCE["repository"], "commit": SOURCE["commit"],
            "competition_id": SOURCE["competition_id"], "season_id": SOURCE["season_id"],
            "match_index_sha256": match_record["sha256"],
            "manifest_path": f"data/raw/statsbomb/{SOURCE['commit']}/manifest.json",
            "verified_source_files": len(source_manifest["files"]),
            "verified_source_event_files": sum(name.startswith("data/events/") for name in source_manifest["files"]),
        "source_file_sha256": {name: record["sha256"] for name, record in sorted(source_manifest["files"].items())},
        "manifest_sha256": acquisition.digest(details["raw_root"].joinpath("manifest.json").read_bytes()),
        },
        "source_event_count": details["source_event_count"],
        "eligible_shot_count": len(shots), "goal_count": sum(row["is_goal"] for row in shots),
        "non_goal_count": sum(not row["is_goal"] for row in shots),
        "class_balance": {str(key): value for key, value in sorted(Counter(int(row["is_goal"]) for row in shots).items())},
        "outcome_counts": details["outcome_counts"], "period_exclusions": details["period_exclusions"],
        "excluded_period_shot_count": sum(details["period_exclusions"].values()),
        "duplicate_event_id_count": details["duplicate_event_id_count"],
        "duplicate_event_id_check_passed": details["duplicate_event_id_check_passed"],
        "raw_flag_absent_counts": details["flag_absent_counts"],
        "off_camera_counts": details["off_camera_counts"],
        "freeze_frame_missing_count": details["freeze_frame_missing_count"],
        "penalty_shot_count": sum(row["is_penalty"] for row in shots),
        "penalty_goal_count": sum(row["is_penalty"] and row["is_goal"] for row in shots),
        "shot_columns": SHOT_COLUMNS, "model_feature_allowlist": feature_columns,
        "target_column": "is_goal", "group_column": "match_id",
        "excluded_from_model_inputs": [
            "match_id", "event_id", "event_index", "match_date", "home_team_id", "home_team_name",
            "away_team_id", "away_team_name", "team_id", "team_name", "opponent_id", "opponent_name",
            "player_id", "player_name", "period", "minute", "second", "possession_id", "is_home",
            "is_penalty", "x", "y", "off_camera", "is_goal", "shot_outcome", "statsbomb_xg",
        ],
        "null_counts": {column: int(value) for column, value in shot_frame.isna().sum().items()},
        "null_policy": {"possession_id": "optional source identifier; blanks are retained", "all other shot columns": "required and must be present"},
        "coordinate_ranges": {"x": [float(shot_frame["x"].min()), float(shot_frame["x"].max())], "y": [float(shot_frame["y"].min()), float(shot_frame["y"].max())]},
        "distance_sb_units_range": [float(shot_frame["distance_sb_units"].min()), float(shot_frame["distance_sb_units"].max())],
        "angle_sb_radians_range": [float(shot_frame["angle_sb_radians"].min()), float(shot_frame["angle_sb_radians"].max())],
        "categorical_values": {column: sorted(shot_frame[column].astype(str).unique().tolist()) for column in ["body_part", "shot_type", "technique", "play_pattern", "shot_outcome"]},
        "own_goal_against_records": len(own_goals), "own_goal_for_corroboration_records": sum(row["own_goals_credited_home"] + row["own_goals_credited_away"] for row in reconciliation),
        "all_matches_reconciled": all(row["reconciled"] for row in reconciliation),
        "outputs": {}, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for relative_path, payload in output_data.items():
        report["outputs"][relative_path] = {"bytes": len(payload), "sha256": acquisition.digest(payload)}
    report_payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # All parsing, validation, reconciliation, and serialization finish before publishing.
    for relative_path, payload in output_data.items():
        acquisition.atomic_write(root / relative_path, payload)
    acquisition.atomic_write(root / "docs/shot_dataset_report.json", report_payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = build(args.root.resolve())
    except (DatasetBuildError, acquisition.AcquisitionError, OSError, ValueError) as exc:
        print(f"Shot dataset build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("eligible_shot_count", "goal_count", "non_goal_count", "penalty_shot_count", "own_goal_against_records", "all_matches_reconciled")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
