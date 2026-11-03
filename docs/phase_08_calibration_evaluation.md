# Phase 8: probability calibration and final held-out evaluation

Status: nested development, frozen selection, final test evaluation, and automated verification complete; learner VS Code checkpoint pending. The learner confirmed Phase 7 before this phase began.

## What this phase completes

This phase investigates probability quality, compares a fixed sigmoid-calibration option with each original classifier, freezes choices using training-only evidence, and then evaluates the fixed candidates on the reserved test matches. It supplies the final accuracy, goal precision, goal recall, and test confusion matrices required by the assignment. It does not run a Monte Carlo simulation.

Calibration asks whether a set of shots receiving a stated probability scores at approximately that frequency. A classifier's ability to rank shots and its ability to assign well-matched probabilities are different properties. Neither high accuracy nor a probability in the interval [0,1] establishes calibration.

## The protocol declared before this phase's fits

Keep the Phase 7 base settings and the nine-feature contract unchanged. Evaluate seven fixed variants: the prior baseline, the three original classifiers, and a sigmoid-calibrated version of each classifier. Sigmoid calibration learns a simple smooth mapping from a model's score to a probability. Scikit-learn uses decision_function scores when available, otherwise predict_proba scores; this is not always a sigmoid applied directly to the raw probability column.

Use only sigmoid calibration in this initial experiment. Do not search additional calibration methods, base-model parameters, or binary thresholds. The goal classification rule remains P(goal) >= 0.5, applied explicitly to predict_proba outputs. It can differ from a classifier's default predict method on ties.

### Training-only nested fitting

For each of the ten locked outer folds:

1. Hold that fold's validation matches aside.
2. Take the remaining training matches and create five stratified, match-grouped inner folds with shuffle=True and random_state=42.
3. Give these precomputed inner indices to CalibratedClassifierCV with method='sigmoid', ensemble=False, and n_jobs=1. Its estimator is the whole preprocessing-plus-classifier Pipeline, not a preprocessed global matrix.
4. Obtain inner out-of-fold scores to fit the calibrator, refit the base pipeline on all outer fitting rows, and predict the outer validation rows.
5. Verify that inner fitting/calibration matches and outer validation matches are disjoint at their appropriate boundaries and that all fitting/validation subsets have both target classes.

Repeat this for the three calibrated candidates. Reuse the verified Phase 7 original-model OOF predictions for the raw comparison. This preserves one outer validation prediction per training shot for each variant without training a calibrator on the same validation outcomes used to assess it.

The ten-fold comparison remains development evidence because it is used to choose variants. It is not an unbiased estimate of a selection procedure, and the inner folds do not turn model selection into an untouched final test. The separately reserved matches provide final evaluation after the choices are frozen.

### Freeze choices before opening test rows

Within each classifier family, select the raw or sigmoid variant with lower pooled outer-OOF log loss; an exact tie keeps the raw version. Among the three selected classifier-family variants, choose the lowest-log-loss candidate for the later simulation, with a deterministic variant-name tie break. The baseline remains a reference rather than a simulation candidate. These selection rules are fixed before results, not chosen after seeing the test scores.

Fit final calibrated versions using five match-grouped inner folds within all 7,334 training shots and no test observations. The original fitted pipelines remain the verified Phase 7 artifacts. Save model fingerprints and a frozen selection record before reading any test predictors or outcomes. Evaluation must verify this record and must not replace a frozen choice with whichever model happens to score best on the test set.

### Final evaluation

After freezing the training-only choices and final model artifacts, load the 1,834 shots in the 76 reserved matches. Evaluate all seven predefined variants transparently. Mark which variants were selected within each family and which candidate was chosen for simulation. Keep the choice fixed even if another variant has a better test score. The final results measure held-out-match performance within this historical season, not established future-season or pre-match forecasting performance.

## What to read in the outputs

- Development metrics: pooled outer-OOF scores for the seven candidates, used for selection.
- Per-fold results: development variation across folds, not confidence intervals.
- Training metrics: final fitted models evaluated on their own training rows; an optimistic reference.
- Test metrics: probabilities and classifications on the fixed held-out shots, after choices were frozen.
- Reliability tables and charts: comparison of average predicted probability and observed goal fraction in fixed probability bins.
- Test confusion matrices: actual outcomes as rows and predicted outcomes as columns, with non-goal=0 and goal=1.

Log loss and Brier score are the probability metrics; lower is better. Goal precision, recall, F1, and accuracy use the fixed 0.5 threshold. Undefined precision when no goals are predicted is displayed as zero under the stated reporting convention. Saved probabilities retain full float precision. Log loss follows scikit-learn's internal numerical clipping at extreme values rather than altering the saved predictions.

Reliability plots use ten fixed equal-width probability bins. A point shows a bin's average predicted probability and observed scoring fraction. A point near the diagonal suggests agreement within that bin, not a proof of perfect calibration. Read the bin counts: high-probability bins can be sparse. Empty bins have undefined rates, not a zero observed scoring rate. No uncertainty intervals are claimed; shots within matches can be dependent, and this limits interpretation of small differences.

Calibration is not guaranteed to improve every metric or every sample. Prefer descriptions of measured differences over statements that a model is now reliable in all settings. Test reliability diagrams are final diagnostic descriptions, not permission to revise the model and keep calling the same test set untouched.

## Learner walkthrough

1. Open E:\dev-work\daniels_vision in VS Code.
2. Open notebooks/06_calibration_evaluation.ipynb and select the project's .venv kernel.
3. Read the protocol and run cells in order with Shift + Enter. Nested fits take longer than the earlier preparation checks; let the running cell finish.
4. Inspect development scores and the frozen training-only choices before the test-evaluation cell.
5. Run the evaluation cell and inspect final test metrics, the selected-variant indicators, and the test confusion matrices.
6. Inspect the development and test reliability plots separately. Use their bin-count table to judge how much data supports each point.
7. Save with Ctrl + S. Retain screenshots of the final test metric table and each required classifier's test confusion matrix under reports/screenshots. Do not substitute Phase 7 OOF matrices for these test matrices.

Existing source data, features, holdout, folds, and Phase 7 outputs remain protected. Reruns reproduce the frozen experiment; they do not create a fresh unseen test set. The selection record and model manifest must be verified rather than silently replaced when inputs change.

## Results and verification

Verified actual results and completed checks are recorded below. The next phase will use the frozen selected model's probabilities for a retrospective simulation conditional on observed shots. Simulation variation alone will not quantify model uncertainty, nor will it convert these shot-based estimates into pre-match forecasts.

## Verified training-only choices

| Classifier | Raw OOF log loss | Sigmoid OOF log loss | Frozen choice |
| --- | ---: | ---: | --- |
| knn_50 | 0.367442551 | 0.290694138 | KNN sigmoid |
| gaussian_nb | 1.297262211 | 0.323533058 | Gaussian NB sigmoid |
| logistic_regression | 0.274668493 | 0.274659442 | Logistic regression sigmoid |

All three family choices use sigmoid calibration under the predeclared strict-lower-log-loss rule. The frozen simulation candidate is sigmoid-calibrated logistic regression. Its OOF improvement over raw logistic regression is only about 0.000009 log-loss units. This tiny development difference is not evidence of a practically important or statistically established improvement; the selected variant follows the fixed rule.

## Verified final test results

The holdout contains **1,834 shots in 76 matches**, comprising **212 goals and 1,622 non-goals**. These counts were examined only after the training-only selection and final model artifacts had been frozen. The full table below includes the baseline, every raw classifier, and every calibrated classifier. All classification metrics use the fixed 0.5 threshold.

| Variant | Accuracy | Goal precision | Goal recall | Log loss | Brier score | Family selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior baseline | 88.44% | 0.00% | 0.00% | 0.358251 | 0.102271 | No |
| KNN raw | 89.86% | 79.55% | 16.51% | 0.372775 | 0.082939 | No |
| Gaussian NB raw | 88.00% | 46.30% | 23.58% | 1.217609 | 0.116486 | No |
| Logistic regression raw | 89.91% | 72.88% | 20.28% | 0.276830 | 0.080491 | No |
| KNN sigmoid | 89.86% | 74.07% | 18.87% | 0.290625 | 0.082859 | Yes |
| Gaussian NB sigmoid | 88.44% | 0.00% | 0.00% | 0.331772 | 0.094192 | Yes |
| Logistic regression sigmoid | 89.91% | 74.55% | 19.34% | 0.276997 | 0.080548 | Yes |

Raw logistic regression has slightly lower test log loss (0.276830) and Brier score (0.080491) than its calibrated version (0.276997 and 0.080548). The frozen candidate remains calibrated logistic regression: switching based on this test comparison would make the test set influence model selection. Treat the two versions as very similar in this experiment, not as evidence that calibration must improve every held-out sample.

For the frozen calibrated logistic-regression candidate, the test confusion counts are TN=1,608, FP=14, FN=171, TP=41. It identifies 41 of 212 goals (19.34 percent recall), while 41 of its 55 predicted goals are actual goals (74.55 percent precision). Accuracy is 89.91 percent. These numbers describe binary decisions at 0.5, rather than the full quality of its shot probabilities.

Sigmoid-calibrated Gaussian Naive Bayes has better test probability scores than its raw version but predicts no goals at 0.5. Its precision is reported as zero under the specified convention and its recall is zero. This is not a pipeline failure: the final calibrated probabilities never reach the decision threshold on these test rows. It illustrates why probability quality and threshold-based classification must be reported separately.

The baseline's probability is the training goal fraction, not the test goal fraction. Its test accuracy is 88.44 percent because it predicts no goals at 0.5. A baseline can appear close to the reliability diagonal at the overall prevalence while having no ability to distinguish high-quality shots from low-quality ones. Read calibration together with discrimination metrics and sample support.

## Verified artifacts and checks

- Seven candidates have exactly one development prediction for each training event: 51,338 prediction rows. The fold-metric table has 70 rows.
- Seven fixed candidates have exactly one test prediction for each held-out event: 12,838 prediction rows.
- Development and test reliability tables are separate. Each contains 70 rows: ten bins for each of seven variants. Empty-bin goal counts and rates are undefined (shown as blank or NaN), not evidence of zero scoring frequency.
- Three final calibrated models are saved under models/phase08; verified raw models remain under models/phase07. The selection record identifies the chosen candidate and artifact fingerprints.
- All 59 project tests passed, including synthetic end-to-end tests of the freeze boundary, corrupted input rejection, grouped folds, and fold-local calibration preprocessing.
- Every notebook code cell executed from the notebooks/ working directory in the project environment. All three figures were visually inspected.
- Rerunning development and evaluation reproduced identical prediction/metric CSVs, PNGs, serialized calibrated models, protocol, model manifest, frozen selection, and reports.
- Saved probabilities reproduce reported log loss, Brier score, AP, ROC AUC, and accuracy. Reliability-bin counts, goal totals, empty-bin handling, and weighted mean probabilities were checked independently.
- All seven test variants cover exactly the 76 reserved matches, with no training matches. The selection fingerprint was unchanged by evaluation. Source shots, feature contract, holdout, CV folds, and all protected Phase 7 artifacts were unchanged.

See docs/phase_08_verification.json for the execution record. The test set is now evaluated and its outcomes are visible. Subsequent notebook runs reproduce the frozen experiment; they do not restore an unseen test set. Changes motivated by these test results must be identified as later exploratory development rather than silently replacing the reported final experiment.

## Evidence to retain for the assignment

Save screenshots under reports/screenshots, using clear names such as phase08_test_probability_metrics.png, phase08_test_classification_metrics.png, phase08_test_confusion_knn_raw.png, phase08_test_confusion_nb_raw.png, and phase08_test_reliability.png. For calibrated-model sections, retain their corresponding confusion matrices and use the matching Phase 8 training and nested-CV results. Do not mix a raw model's training/CV results with a calibrated model's test results under one unspecified model name.

The next phase will implement and validate the Monte Carlo simulator using the frozen candidate, conditional on observed shots. It must retain the documented limits about own goals, dependencies between shots, historical scope, and model uncertainty.
