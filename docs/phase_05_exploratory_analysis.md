# Phase 5 exploratory analysis

Status: analysis and automated verification complete; learner VS Code notebook checkpoint pending. The user confirmed the Phase 4 checkpoint before this phase.

## Why a holdout is reserved now

The original roadmap placed splitting in Phase 6. Detailed exploration of how features relate to goals should use training data rather than the future test set. Therefore this phase reserves the holdout before producing feature-outcome charts. Phase 6 will reuse this same membership and document the split, create training-only cross-validation folds, and fit preprocessing. It must not create a second holdout or search for a better-performing split.

Phase 4 already performed basic whole-snapshot label and integrity audits. The new boundary protects subsequent feature exploration and modelling decisions; it does not claim that the public dataset's labels have never been seen.

## Allocation policy

Use the verified shot CSV and match identifiers. Shuffle sorted match identifiers reproducibly with seed 42, then choose a whole-match subset satisfying the predefined size constraints. The allocation depends on match IDs and shot counts, not on goal rates, xG values, or model scores.

The verified feasibility target is 76 test matches containing 1,834 shots, with 304 training matches containing 7,334 shots. The test-row count is ceil(0.20 * 9,168), approximately 20.0044 percent. Eighty percent cannot be an exact integer split of 9,168 rows. A fixed-cardinality subset-sum allocation preserves match groups while meeting both rounded row and match targets.

This is a seeded, size-constrained match allocation; it is not described as a uniform draw from all possible subsets or as outcome-stratified sampling. Store and hash the membership so reruns and all later models use the same split. Source or membership changes must be investigated instead of silently creating a new split.

The membership table contains match IDs, allocation labels, and shot counts; it does not contain test goal counts. Keep test-outcome summaries out of the EDA report.

## Questions answered by the training data

1. How common are goal and non-goal outcomes in the training sample?
2. Where were shots and goal-labelled shots recorded?
3. How are distance and angle distributed?
4. How does observed conversion vary across predefined geometry bins?
5. How do body-part and shot-type groups differ in shot volume and conversion?
6. How do conversion rates differ between provider-flagged and not-flagged contexts?

The tables also cover technique and play pattern. These are descriptive associations within one historical training sample, not evidence that changing one factor will cause a goal or that a model will perform well.

## Reading conversion rates and uncertainty

A conversion rate is goal-labelled shots divided by shots in a group. Always read its denominator. A category with one shot cannot support the same confidence as a category with thousands of shots. Highlight groups with fewer than 30 shots as small-sample descriptions; this threshold is a display convention rather than a guarantee of reliability above 30.

The rate charts use 95 percent Wilson intervals under an independent-shot approximation. Shots within a match can be dependent, so these are pointwise descriptive intervals rather than match-cluster-adjusted uncertainty estimates or causal intervals. They are not prediction intervals for future scores. More rigorous clustered uncertainty can be added in a later extension.

Flagged and not-flagged context comparisons follow the Phase 4 source policy. A missing provider flag encoded as zero means not flagged, not a confirmed absence of pressure or another condition. Penalties are a distinct shot type and should not be interpreted as comparable to arbitrary open-play shots without accounting for context.

## Figures and tables

Six figures are saved under reports/figures with the phase05_ prefix. Each figure must identify the training-only scope, native coordinate units where relevant, and StatsBomb as source. Figures cover target balance, the shot map, geometry distributions, conversion by geometry, conversion by body part/shot type, and context flags. Tables under reports/tables retain counts and rates so narrative claims can be traced to numbers.

Provider xG is not a model input, a training target, or a true probability label. This phase does not fit an xG model, calibrate probabilities, tune classification thresholds, or run Monte Carlo simulations.

## Running this phase

From a terminal in the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\reserve_holdout.py
.\.venv\Scripts\python.exe scripts\explore_shots.py
```

Run the first command successfully before the second. Existing locked membership should be verified and reused. Do not delete a split just to obtain different results.

Open E:\dev-work\daniels_vision\notebooks\03_exploratory_analysis.ipynb in VS Code, select the project .venv kernel, read the explanations, and run cells in order. The notebook should show the locked allocation sizes, training-only summaries and figures, and the generated findings. Save it with outputs and collect screenshots in reports/screenshots.

Read the generated findings in docs/phase_05_eda_findings.md. Treat their statements as descriptions of this training sample. Model generalization and Monte Carlo probabilities remain untested at this stage.

## Completion checks

- The input CSV matches its Phase 4 fingerprint.
- Training and test matches are disjoint and cover the expected data.
- The fixed split satisfies the documented row and match counts.
- Existing split integrity is checked on rerun.
- Detailed EDA uses training rows only.
- Tables and charts agree with their underlying counts.
- The six exported images have been visually inspected.
- The notebook executes successfully and results are recorded.
- Raw data, the shot table, and the model predictor contract remain unchanged.

## Verified results

Training contains 7,334 shots across 304 matches: 802 goals and 6,532 non-goals. The training goal rate is 10.9 percent; an always-non-goal rule has 89.1 percent descriptive accuracy. The 0-6 native-unit distance band converted 100 of 214 shots (46.7 percent); the 36-80 band converted 1 of 162 (0.6 percent). These are sample associations, not model scores.

All 34 project tests passed. Every notebook code cell executed in the project environment. All six figures were visually inspected. Repeated execution produced identical hashes for the six PNGs and four CSV tables. The shot dataset, model-column contract, split membership, and split manifest were unchanged. See docs/phase_05_verification.json for the verification record and docs/phase_05_eda_findings.md for interpretation.
