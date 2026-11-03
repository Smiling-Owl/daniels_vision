# Phase 6: cross-validation and preprocessing

Status: preparation and automated verification complete; learner VS Code checkpoint pending. Phase 5 was confirmed by the learner before this phase began.

## What this phase prepares

Keep the Phase 5 allocation: 304 training matches with 7,334 shots and 76 held-out test matches with 1,834 shots. The test set does not participate in fold selection, category discovery, scaling, tuning, calibration, or feature decisions. This is held-out-match evaluation within one historical season, not an established future-season forecasting protocol.

Cross-validation temporarily divides only the training partition into ten parts. In each round, nine parts supply fitting rows and the remaining part supplies validation rows. Every training shot is validated once across ten rounds. All shots from a match stay together, avoiding the same match contributing shots on both sides of a round.

Use StratifiedGroupKFold with ten folds, shuffling and seed 42. Stratification attempts to distribute goal/non-goal outcomes while preserving whole matches; it does not promise identical fold sizes or goal rates. Verify both target classes are present on each side of every fold. The fold map is generated once, fingerprinted, and reused for every model. Do not choose folds according to model scores.

## Feature handling

The fixed Phase 4 contract supplies exactly nine raw predictors:

| Kind | Columns | Preparation |
| --- | --- | --- |
| Numeric | distance_sb_units, angle_sb_radians | StandardScaler fitted on the current fitting rows |
| Categorical | body_part, shot_type, technique, play_pattern | One-hot encoding, learned from the current fitting rows |
| Binary | under_pressure, first_time, open_goal | Pass through as 0/1 |

Standardization subtracts a fitting-sample mean and divides by its scale. It prevents distance and angle from being compared only on their different original numerical ranges. Categorical encoding makes separate indicator columns rather than implying that categories such as body parts have a numerical order. These choices define the initial representation; they do not establish optimal feature weights for KNN.

The encoder accepts an unseen validation category by producing zeros for that feature's indicator block. Such cases are counted in the preparation report. The representation may therefore contain different numbers of columns across folds, because each fold learns its own category vocabulary. It is consistent within each fold. No category vocabulary is learned from held-out test rows.

The dense output is compatible with the planned Gaussian Naive Bayes classifier. Gaussian likelihoods for binary and one-hot columns are an approximation, and their dependence also limits the Naive Bayes assumption. We will evaluate its actual probability quality rather than assuming its probabilities are calibrated.

The current verified training data are complete. This phase rejects missing or invalid predictor values rather than silently dropping rows or adding an imputation policy. Provider flags retain their documented meaning: zero means not flagged, not a confirmed real-world absence.

## The leakage boundary

For each fold, create a fresh preprocessor, fit it on fitting rows, then transform those rows and validation rows. The validation rows must never influence scaling statistics or encoder categories. In Phase 7 the transformer and classifier must be placed together in a scikit-learn Pipeline so cross-validation refits both at the correct boundary.

Do not create one globally scaled or encoded training matrix and feed it into cross-validation. This phase deliberately exports neither such a matrix nor a fitted global transformer. It checks each fold's transformed arrays for compatible shape and finite values without fitting a classifier. Identifiers, is_goal, shot_outcome, StatsBomb xG, and other excluded fields are never predictors.

Tuning and calibration, if introduced later, must respect training boundaries as well. A cross-validation score used to choose settings is a development estimate; the reserved test set provides the later final evaluation after decisions are frozen. Initial classification reporting uses the previously documented 0.5 threshold; probability metrics remain central for xG and simulation.

## Learner walkthrough

1. Open E:\dev-work\daniels_vision in VS Code.
2. Open notebooks/04_modeling_preparation.ipynb and select the project's .venv kernel.
3. Read each explanation and run code cells from top to bottom with Shift + Enter.
4. Confirm the unchanged holdout allocation and the ten-row cross-validation summary.
5. Check that all ten rounds have nonzero goal and non-goal counts, disjoint fitting/validation matches, and finite transformed features.
6. Read the encoding and scaling explanation before continuing. A wider transformed matrix does not mean new raw predictors were added.
7. Save the notebook with Ctrl + S. Save screenshots of the allocation and fold summary under reports/screenshots.

Command-line entry point from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_modeling.py
```

## Outputs and completion

The notebook, preparation report, locked cross-validation membership, fold summary table, and automated verification record document this phase. Actual counts and completed checks are recorded below. No classifier is trained, no test prediction is made, and no Monte Carlo simulation is run here.

## Verified actual results

- Ten validation folds cover exactly 7,334 training shots and 304 training matches, each once.
- Each validation fold has 29-31 matches and 726-743 shots; its goal count is 78-85.
- Every fitting and validation subset contains both target classes, with no match overlap and no held-out test matches.
- All ten fold-specific transforms produce 28 columns: two standardized numeric columns, 23 category indicators, and three binary flags. Nine raw predictors remain the input contract.
- No unseen validation categories occurred in these ten folds. Synthetic tests separately verify unseen-category handling.
- All transformed fitting and validation values are finite. No rows were dropped or imputed.
- Fold 1 fits preprocessing on 6,605 shots and validates on 729. Its distance mean is 19.0016059615 and scale is 8.64627544879 in native coordinate units. The notebook shows the calculation on three rows.
- All 42 project tests passed, including eight Phase 6 tests. Every notebook code cell executed from a notebooks/ working directory in the project environment.
- Repeated execution reproduced identical CV membership, manifest, summary table, and preparation report. The original shot CSV, model contract, holdout membership, and holdout manifest were unchanged.

See docs/phase_06_verification.json for recorded fingerprints. The locked CV membership SHA-256 is adbf280606b59a47b83288afaba60c3432fa111752f424f90c84682fb2adfab5. Reuse it for all Phase 7 model comparisons, joining by event_id rather than assuming that a rearranged dataframe preserves its row positions.

Before fitting a final model on all training rows, select settings from training-only validation. Fit the final model's preprocessing on the full training partition only, then transform the reserved test rows with that fitted preprocessing for the eventual final evaluation. Never refit it on the test partition.
