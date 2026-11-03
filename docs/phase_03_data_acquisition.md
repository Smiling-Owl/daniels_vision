# Phase 3 data acquisition

Status: acquisition and programmatic verification completed successfully. The learner checkpoint is to run the notebook in VS Code and save its outputs.

## Purpose

Keep a reproducible copy of the original StatsBomb event files for La Liga 2015/2016. This phase obtains and inventories source data. It does not clean shots, create model labels, split data, or train models.

## Source snapshot

- Provider: StatsBomb Open Data, hosted at https://github.com/hudl/open-data
- Repository revision: 4b73468fc5b0f1950f9f66fada70ad3a4f9327cb
- Source commit timestamp: 2026-09-07T16:27:15Z
- Competition ID: 11
- Season ID: 27
- Expected indexed matches: 380

A commit is a fixed repository version. The downloader uses this revision in every source URL rather than following a changing branch. The acquisition manifest records the actual UTC retrieval time separately from the source commit timestamp.

## Files and terminology

JSON is a text format containing lists and named fields. One match event file contains events such as passes, shots, and fouls. An event is not automatically a shot or a model observation. The match index tells us which event files belong to this season.

The downloader requests gzip compression when supported to reduce network transfer. It decompresses the response before hashing and saving, so local JSON retains the original uncompressed source bytes. Compression changes transport size, not the data content.

Original source bytes are retained under:

    data/raw/statsbomb/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/

This folder will contain competition metadata, the season match index, the match event files, the upstream README and licence, and an acquisition manifest. The downloader records each file's source URL, byte count, retrieval time, and SHA-256 hash.

A hash is a fingerprint of the file bytes. Recomputing it detects whether a local file differs from the downloaded copy recorded in the manifest. It is an integrity check, not a guarantee that every football annotation in the source is correct.

## Reproduce acquisition from the project folder

Use VS Code Terminal > New Terminal. Confirm that the prompt points to the project folder, then run:

```powershell
.\.venv\Scripts\python.exe scripts\acquire_data.py
```

On the first run this needs an internet connection and downloads missing files. Subsequent runs reuse files that match their recorded hashes. If a download is interrupted, rerun the same command; successful recorded files can be reused. Do not manually edit the raw files. An integrity error must be investigated rather than hidden by relabelling a changed file as valid.

## Verify without downloading

```powershell
.\.venv\Scripts\python.exe scripts\acquire_data.py --verify-only
```

This mode must not contact the network. It checks that required files exist, match the saved manifest, and pass the expected structural checks. An absent or damaged file produces a failure rather than an incomplete success report.

## Notebook walkthrough

Open notebooks/01_data_acquisition.ipynb. Select the same project .venv kernel used in Phase 1. Read the Markdown explanations and run cells in order. The notebook invokes the acquisition script rather than maintaining a second downloader implementation, then shows the inventory and a raw event example. Existing verified downloads should be reused.

Save the notebook with its outputs. Capture the completed summary and offline verification output in reports/screenshots for the project record. Do not describe the event count as the shot count.

## Attribution and publication

Keep the source licence and README with the raw files. Credit StatsBomb. The upstream README requests the StatsBomb logo when sharing research or analysis; review the saved licence before distributing data or deploying the portfolio. The raw dataset is excluded from Git by .gitignore; reproduce it with the downloader and pinned source configuration rather than committing all raw JSON files.

## Completion checkpoint

- The pinned competition/season is confirmed.
- The match index contains 380 unique matches and 20 teams with 38 appearances each.
- All expected event files download and parse.
- Event identifiers are unique within their match files.
- File fingerprints and source URLs are recorded.
- A full offline verification run passes.
- The explanatory notebook runs successfully.
- Acquisition results are recorded with actual counts, not estimates.

Phase 4 will extract shot events into a table, audit eligible outcomes and input fields, and reconcile shot goals with official scores. No model result should be claimed in Phase 3.


## Verified results

Verified on 2026-10-22T09:07:20+00:00: 380 matches, 20 teams, 38 appearances per team, 380 event files, 1,295,354 individual events, and 384 source files containing 1,122,501,352 original bytes. All 10 downloader tests passed. Every acquisition notebook code cell executed successfully from the notebooks directory, and the separate verify-only command passed in the network-restricted execution environment. Interactive notebook display remains for the user to confirm. No temporary download files remained after verification.

The portable acquisition report is docs/acquisition_report.json. A copy of the complete file manifest is docs/data_manifest.json, so provenance can be retained with project documentation while raw data remains excluded from Git. The raw manifest is authoritative for resume checks. Its completion timestamp can change on a later successful rerun; the docs copy records this verification checkpoint.
