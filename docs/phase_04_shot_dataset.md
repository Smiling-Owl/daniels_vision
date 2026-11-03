# Phase 4 constructing the shot dataset

Status: extraction and programmatic validation completed successfully on 22 October 2026. The learner checkpoint is to run the notebook in VS Code and save its outputs.

## Purpose

Convert the preserved match events into an auditable table with one row per regulation-time shot. This table supports later modelling and match analysis. This phase does not split the data, fit preprocessing, or train models.

## Initial source audit

The pinned season contains 9,168 shots: 1,014 goal-labelled shots and 8,154 other documented shot outcomes. All shots are in periods 1 and 2; 97 are in-match penalties. Location, player, outcome, body part, shot type, and the supplied StatsBomb xG are present for every shot. The audit found 29 Own Goal Against events and 29 paired Own Goal For events. Shot goals plus the opponent's Own Goal Against count reconcile the official score in all 380 matches. The reproducible extraction script confirmed these counts and score reconciliations.

## Observation and target

One table row is one shot event, identified by its match and event identifiers. A target value of 1 means the shot outcome is Goal. A value of 0 is assigned only to known non-goal outcomes: Blocked, Off T, Saved, Wayward, Post, Saved Off Target, and Saved to Post. An unknown or missing outcome must cause an audit failure instead of silently becoming a non-goal. Penalties during the match are retained and identified explicitly; shootouts are outside the scope.

## Coordinate and feature policy

StatsBomb's specification gives a 120 by 80 coordinate frame. The attacked goal centre is (120, 40), and the posts are (120, 36) and (120, 44). Keep the source orientation; do not flip away-team shots. The official locations diagram was checked in Appendix 2, PDF page 34 of the pinned specification.

The distance feature is the Euclidean distance to the goal centre in native StatsBomb coordinate units. It is not labelled metres. The angle feature is the opening between the vectors from the shot to each goalpost, in radians within this standardized coordinate frame. A position exactly on a post has an undefined vector angle and must be investigated rather than silently receiving an invented angle. Actual physical pitch dimensions have not been established.

Initial candidate predictors are distance_sb_units, angle_sb_radians, body_part, shot_type, technique, play_pattern, under_pressure, first_time, and open_goal. Categories remain readable text. No scaling, imputation, category encoding, or oversampling is fitted on the full dataset.

Context flags are recorded sparsely in the source. Use a documented presence-indicator policy: absent under_pressure, first_time, or open_goal means the provider did not flag the condition; store false for that indicator. It is not proof that the condition was objectively absent. Reject malformed explicit values rather than converting arbitrary strings or numbers to booleans. This policy does not apply to arbitrary missing numerical or categorical fields.

Do not use shot outcome, goal label, StatsBomb's existing xG, final score, shot end location, event duration, or post-shot flags as predictors. Match/player/team identifiers are for traceability, grouping, and display, outside the initial predictor list. Source xG is a comparison field rather than a target or a ground-truth probability. The feature contract in docs/model_columns.json records these roles.

## Own goals and score reconciliation

Own goals are not ordinary goal-labelled shot rows in this dataset. Each own goal is represented by a conceding-team Own Goal Against event and a benefiting-team Own Goal For event. Count the Against event once and credit the opponent; use the For event as a corroborating check. Adding both would double-count the goal.

For each team in each match:

    official goals = goal-labelled shots + opponent Own Goal Against events

Save a match-level reconciliation table and an own-goal event table. A discrepancy must be explained before marking the dataset validated. Future simulation results describe shot-generated scorelines unless own goals are explicitly handled separately.

## Running the extraction

From a terminal in E:\dev-work\daniels_vision:

```powershell
.\.venv\Scripts\python.exe scripts\build_shot_dataset.py
```

The script verifies input provenance and structure, extracts the documented columns, derives geometry, validates labels and coordinates, checks duplicates and scores, and saves outputs only after validation passes. Raw JSON remains unchanged. A rerun should reproduce the same CSV bytes for this source snapshot.

## Notebook checkpoint

Open E:\dev-work\daniels_vision\notebooks\02_shot_dataset.ipynb and select the project .venv kernel. Read the explanations and run the cells in order. Inspect the shot preview, class counts, missing-value summary, feature roles, penalty counts, and goal reconciliation. Save the notebook with outputs and take screenshots of the table preview, quality summary, and score reconciliation in reports/screenshots.

## Files

- data/processed/shots.csv: one row per eligible shot, including traceability fields, candidate inputs, target, and external reference xG.
- data/processed/goal_reconciliation.csv: official scores reconciled with shot goals and own goals.
- data/processed/own_goals.csv: own goals counted once using their Against event.
- docs/shot_dataset_report.json: validation counts, input/output fingerprints, and quality results.
- docs/model_columns.json: explicit predictor, target, grouping, and exclusion roles.
- docs/shot_data_dictionary.md: per-column definitions, units, origin, and missing-data policies.

The CSV contains several column roles. Never fit a model to all columns simply by dropping the target. Use the approved predictor list later.

## Sources

StatsBomb Open Data: https://github.com/hudl/open-data/tree/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb

Pinned event specification: https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf

The original data and specification belong to their source. Retain StatsBomb attribution and review the preserved licence before publication.

## Verified completion results

The shot table contains 9,168 rows and 35 columns, including the nine candidate predictors and is_goal as target. There are 1,014 goal shots (11.06%) and 8,154 non-goal shots. Of 97 penalties, 69 scored. No shots were excluded by the period rule; there are no duplicate event IDs or missing values in the resulting shot table. Absent context flags are represented by the documented NOT FLAGGED indicator policy, not by numerical imputation.

All 380 score reconciliations passed: 1,014 shot goals plus 29 own goals equal 1,043 official goals. Own Goal Against events are counted once; their paired Own Goal For events are corroboration only. Sixteen shots are explicitly flagged off-camera; the flag is retained for audit and excluded from the predictors. Eighty-one shots lack freeze frames; those frames are not required for this feature set and no shots were discarded for that reason.

All 23 project tests passed. Every Phase 4 notebook code cell executed successfully from the notebooks directory. Repeated builds produced identical CSV bytes for all three outputs, and every output matched its recorded SHA-256. See docs/phase_04_verification.json for the reproducibility checkpoint. The notebook interactive display and saved screenshots remain for the learner to confirm. No split, fitted preprocessing, model training, or simulation has been performed.
