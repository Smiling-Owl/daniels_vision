"""Reserve and lock a deterministic match-grouped 80/20 holdout without reading labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts import acquire_data as acquisition
except ImportError:  # pragma: no cover
    import acquire_data as acquisition

SEED = 42
TEST_FRACTION_NUMERATOR = 20
TEST_FRACTION_DENOMINATOR = 100
ALGORITHM = "seeded-match-count-subset-sum-bitset-v1"
MEMBERSHIP_RELATIVE = "data/processed/split_membership.csv"
SPLIT_MANIFEST_RELATIVE = "docs/split_manifest.json"
SHOT_RELATIVE = "data/processed/shots.csv"
REPORT_RELATIVE = "docs/shot_dataset_report.json"


class SplitError(RuntimeError):
    """The source table or locked split cannot be safely used."""


def choose_match_subset(
    match_counts: dict[int, int], target_shots: int, target_matches: int, seed: int = SEED,
) -> list[int]:
    """Choose a fixed-cardinality match set with exactly target_shots observations.

    Only integer match identifiers and shot counts enter this algorithm. Labels and
    features are deliberately outside its inputs.
    """
    if target_shots < 0 or target_matches < 0:
        raise SplitError("Target counts must be non-negative.")
    if len(match_counts) != len(set(match_counts)):
        raise SplitError("Match identifiers must be unique.")
    if any(isinstance(mid, bool) or not isinstance(mid, int) or mid <= 0 for mid in match_counts):
        raise SplitError("Match identifiers must be positive integers.")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in match_counts.values()):
        raise SplitError("Every match must have a positive integer shot count.")
    if target_matches > len(match_counts):
        raise SplitError("Target match count exceeds available matches.")

    ordered_ids = sorted(match_counts)
    random.Random(seed).shuffle(ordered_ids)
    mask = (1 << (target_shots + 1)) - 1
    reachable = [0] * (target_matches + 1)
    reachable[0] = 1  # Bit s is set when a subset of the processed groups totals s shots.
    history = [reachable.copy()]
    for match_id in ordered_ids:
        count = match_counts[match_id]
        for group_count in range(min(target_matches, len(history)), 0, -1):
            reachable[group_count] = (
                reachable[group_count] | (reachable[group_count - 1] << count)
            ) & mask
        history.append(reachable.copy())

    if not ((history[-1][target_matches] >> target_shots) & 1):
        raise SplitError(
            f"No exact split exists for {target_shots} shots in {target_matches} matches; "
            "the script will not silently change the target."
        )

    selected: list[int] = []
    remaining_shots, remaining_matches = target_shots, target_matches
    for index in range(len(ordered_ids), 0, -1):
        # Prefer skipping the current (earlier seeded-order) match whenever possible.
        if (history[index - 1][remaining_matches] >> remaining_shots) & 1:
            continue
        match_id = ordered_ids[index - 1]
        selected.append(match_id)
        remaining_shots -= match_counts[match_id]
        remaining_matches -= 1
    if remaining_shots != 0 or remaining_matches != 0:
        raise SplitError("Internal subset reconstruction failed.")
    return sorted(selected)


def membership_bytes(match_counts: dict[int, int], test_match_ids: set[int]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["match_id", "split", "shot_count"])
    for match_id in sorted(match_counts):
        writer.writerow([match_id, "test" if match_id in test_match_ids else "train", match_counts[match_id]])
    return output.getvalue().encode("utf-8")


def source_match_counts(root: Path) -> tuple[dict[int, int], str, dict[str, Any]]:
    report_path = root / REPORT_RELATIVE
    shot_path = root / SHOT_RELATIVE
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        shot_bytes = shot_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"Cannot read the validated shot table/report: {exc}") from exc
    if report.get("status") != "passed":
        raise SplitError("Shot dataset report is not marked passed.")
    expected = report.get("outputs", {}).get(SHOT_RELATIVE, {}).get("sha256")
    actual = acquisition.digest(shot_bytes)
    if not isinstance(expected, str) or actual != expected:
        raise SplitError("Shot CSV hash does not match the Phase 4 report.")
    try:
        # Restrict input to group identifiers. No target or feature values are read.
        matches = pd.read_csv(shot_path, usecols=["match_id"], dtype={"match_id": "int64"})
    except (ValueError, pd.errors.ParserError) as exc:
        raise SplitError(f"Cannot read match IDs from the shot table: {exc}") from exc
    if matches.empty or matches["match_id"].isna().any():
        raise SplitError("Shot table has no rows or missing match identifiers.")
    counts_series = matches.groupby("match_id", sort=True).size()
    counts = {int(match_id): int(count) for match_id, count in counts_series.items()}
    if int(counts_series.sum()) != report.get("eligible_shot_count"):
        raise SplitError("Shot table row count disagrees with Phase 4 report.")
    return counts, actual, report


def _validate_membership(payload: bytes, match_counts: dict[int, int], manifest: dict[str, Any]) -> None:
    try:
        rows = list(csv.DictReader(StringIO(payload.decode("utf-8"))))
        normalized = [
            {"match_id": int(row["match_id"]), "split": row["split"], "shot_count": int(row["shot_count"])}
            for row in rows
        ]
    except (UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"Split membership CSV is malformed: {exc}") from exc
    expected_ids = set(match_counts)
    actual_ids = [row["match_id"] for row in normalized]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise SplitError("Split membership must contain every match exactly once.")
    test_ids = sorted(row["match_id"] for row in normalized if row["split"] == "test")
    train_ids = sorted(row["match_id"] for row in normalized if row["split"] == "train")
    if len(test_ids) + len(train_ids) != len(normalized):
        raise SplitError("Split column must contain only 'train' or 'test'.")
    for row in normalized:
        if row["shot_count"] != match_counts[row["match_id"]]:
            raise SplitError(f"Shot count changed for match {row['match_id']}.")
    expected_test_ids = manifest.get("test_match_ids")
    expected_train_ids = manifest.get("train_match_ids")
    if test_ids != expected_test_ids or train_ids != expected_train_ids:
        raise SplitError("Split membership differs from its locked match-ID lists.")
    if len(test_ids) != manifest.get("test_match_count") or sum(match_counts[mid] for mid in test_ids) != manifest.get("test_shot_count"):
        raise SplitError("Test partition counts differ from the split lock.")
    if len(train_ids) != manifest.get("train_match_count") or sum(match_counts[mid] for mid in train_ids) != manifest.get("train_shot_count"):
        raise SplitError("Training partition counts differ from the split lock.")


def reserve(root: Path) -> dict[str, Any]:
    root = root.resolve()
    counts, shot_hash, source_report = source_match_counts(root)
    total_shots = sum(counts.values())
    test_shots = (total_shots * TEST_FRACTION_NUMERATOR + TEST_FRACTION_DENOMINATOR - 1) // TEST_FRACTION_DENOMINATOR
    test_matches = 76 if len(counts) == 380 else round(len(counts) * TEST_FRACTION_NUMERATOR / TEST_FRACTION_DENOMINATOR)
    membership_path = root / MEMBERSHIP_RELATIVE
    manifest_path = root / SPLIT_MANIFEST_RELATIVE
    membership_exists, manifest_exists = membership_path.exists(), manifest_path.exists()

    if membership_exists and not manifest_exists:
        raise SplitError("Split membership exists without its lock manifest; refusing to adopt it.")
    if manifest_exists:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitError(f"Cannot read split lock manifest: {exc}") from exc
        if manifest.get("shot_csv_sha256") != shot_hash:
            raise SplitError("Source shot CSV changed after the holdout was locked.")
        if manifest.get("seed") != SEED or manifest.get("algorithm") != ALGORITHM:
            raise SplitError("Split lock algorithm or seed does not match this script.")
        if manifest.get("test_shot_count") != test_shots or manifest.get("test_match_count") != test_matches:
            raise SplitError("Locked split target differs from the required holdout counts.")
        test_ids = set(manifest.get("test_match_ids", []))
        if membership_exists:
            payload = membership_path.read_bytes()
            if acquisition.digest(payload) != manifest.get("membership_csv_sha256"):
                raise SplitError("Split membership CSV hash does not match its lock.")
            _validate_membership(payload, counts, manifest)
        else:
            # Recover only the missing materialized membership when deterministic reconstruction
            # exactly reproduces the already locked IDs and membership fingerprint.
            reconstructed = set(choose_match_subset(counts, test_shots, test_matches, SEED))
            if reconstructed != test_ids:
                raise SplitError("Missing membership cannot be recreated from the locked split IDs.")
            payload = membership_bytes(counts, reconstructed)
            if acquisition.digest(payload) != manifest.get("membership_csv_sha256"):
                raise SplitError("Recreated membership does not match the lock fingerprint.")
            acquisition.atomic_write(membership_path, payload)
        return manifest

    test_ids = set(choose_match_subset(counts, test_shots, test_matches, SEED))
    payload = membership_bytes(counts, test_ids)
    train_ids = sorted(set(counts) - test_ids)
    manifest = {
        "status": "locked", "algorithm": ALGORITHM, "seed": SEED,
        "test_fraction": "20% rounded up to a whole shot", "shot_csv_sha256": shot_hash,
        "membership_csv_sha256": acquisition.digest(payload),
        "total_shot_count": total_shots, "total_match_count": len(counts),
        "test_shot_count": sum(counts[mid] for mid in test_ids), "test_match_count": len(test_ids),
        "train_shot_count": sum(counts[mid] for mid in train_ids), "train_match_count": len(train_ids),
        "test_match_ids": sorted(test_ids), "train_match_ids": train_ids,
        "source_commit": source_report.get("source", {}).get("commit"),
    }
    acquisition.atomic_write(membership_path, payload)
    acquisition.atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = reserve(args.root)
    except (SplitError, OSError, ValueError) as exc:
        print(f"Holdout reservation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "test_shot_count", "test_match_count", "train_shot_count", "train_match_count",
        "shot_csv_sha256", "membership_csv_sha256",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
