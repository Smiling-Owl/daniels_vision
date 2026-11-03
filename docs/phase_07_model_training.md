# Phase 7: train and compare initial classifiers

Status: training, cross-validation, and automated verification complete; learner VS Code checkpoint pending. The learner confirmed Phase 6 before this phase began.

## Purpose and boundaries

This phase trains the required KNN and Gaussian Naive Bayes classifiers, adds logistic regression as a portfolio benchmark, and compares them against a constant-probability baseline. It uses the locked 7,334-shot, 304-match training partition and the ten Phase 6 folds. The 1,834 shots in 76 test matches stay reserved. No test metrics or test confusion matrices are produced here.

The assignment's final testing results and cropped test-metric screenshots are still required. They will be produced after Phase 8 settles calibration and the final model specifications, before interpreting the final held-out results. The confusion matrices in this phase are cross-validation results and must not be presented as the assignment's testing screenshots.

## Fixed initial settings

These settings are declared before this phase's first fit. They are starting specifications, not optimized choices.

| Model | Fixed settings | Interpretation |
| --- | --- | --- |
| Prior baseline | DummyClassifier, strategy=prior | Predict each fitting subset's observed goal fraction for every validation shot |
| KNN | 50 neighbors, uniform weights, Euclidean distance | Goal probability is the fraction of goals among the 50 nearest fitting shots |
| Gaussian Naive Bayes | var_smoothing=1e-9 | Combine class-conditional feature likelihoods under the model's simplifying assumptions |
| Logistic regression | C=1, lbfgs, max_iter=2000 | Estimate goal probability from a regularized linear combination of the prepared features |

KNN uses a brute-force neighbor search and one worker. Fits and predictions use one numerical-library thread to improve repeatability on this environment. No class reweighting, synthetic oversampling, threshold search, parameter search, or probability calibration is performed in this initial comparison.

All classifiers receive the same nine raw predictors. Each fit creates a fresh Pipeline containing the Phase 6 preprocessing and the classifier. Each fold therefore learns its numeric means/scales and category vocabulary only from that fold's fitting rows. Event IDs join predictions to the locked fold map; identifiers and the target are not predictors. StatsBomb xG is excluded from model inputs and this comparison.

## Three different result types

1. Fold results: each of ten rounds fits on nine fold groups and evaluates on the remaining group, with matches kept together.
2. Pooled out-of-fold (OOF) results: combine the one validation prediction made for each of the 7,334 training shots. These are the main development comparison in this phase.
3. Training or resubstitution results: fit a model on all training rows and evaluate it on those same rows. These document fitting performance and can be optimistic. KNN can even count a fitting observation among its own neighbors.

Pooled OOF metrics are not necessarily equal to the arithmetic mean of ten fold metrics. Fold means and sample standard deviations are reported separately. Standard deviations describe variation across these folds; they are not confidence intervals. Once these results influence a decision, they are development evidence rather than an untouched final evaluation.

## Reading the metrics

The positive class is goal (1). Initial hard predictions use probability >= 0.5 applied explicitly to predict_proba outputs. Use this rule to reproduce the tables; a classifier's default predict method can handle exact ties differently. This threshold is fixed, not chosen from validation or test results.

- Accuracy: fraction of all shots classified correctly. Read alongside the approximately 89 percent non-goal frequency.
- Precision: among predicted goals, the fraction that were actual goals.
- Recall: among actual goals, the fraction identified as goals.
- F1: harmonic mean of precision and recall.
- Log loss: penalizes incorrect probabilities, especially confident errors; lower is better.
- Brier score: mean squared error of goal probabilities; lower is better.
- Average precision: summarizes precision-recall ranking; higher is better. It is the reported AP definition, not an interchangeable numerical integration of a PR curve.
- ROC AUC: ranking discrimination across thresholds; higher is better, without establishing calibration.

Precision is reported as zero when no goals are predicted, following an explicit zero-division convention. Always inspect predicted-positive counts and confusion matrices with it. A non-goal-heavy sample can reward an unhelpful classifier with high accuracy.

Probabilities of exactly zero or one can occur, especially for KNN and Gaussian Naive Bayes. They are recorded rather than silently altered in saved predictions. Scikit-learn's log-loss computation internally clips extreme probabilities to machine-precision limits. Reported log loss follows that implementation; finite loss does not prove probabilities are well behaved.

Probability calibration is a separate question: among shots assigned a particular probability, does the observed goal frequency agree? Phase 8 will examine that question using training-only boundaries. Do not use the initial probabilities for a final simulator merely because they lie between zero and one.

## Reading confusion matrices

Rows are actual outcomes and columns are predicted outcomes. With non-goal first and goal second:

| | Predicted non-goal | Predicted goal |
| --- | --- | --- |
| Actual non-goal | True negatives (TN) | False positives (FP) |
| Actual goal | False negatives (FN) | True positives (TP) |

Each OOF confusion matrix must sum to 7,334 shots. Actual goals must total 802 and non-goals 6,532. The displayed matrices apply the fixed 0.5 threshold; probability scores use the underlying probabilities.

## Learner walkthrough

1. Open E:\dev-work\daniels_vision in VS Code.
2. Open notebooks/05_model_training.ipynb and select the project's .venv kernel.
3. Read the model and validation explanations. Run each cell in order with Shift + Enter.
4. Allow the training cell to finish before running later cells. It performs repeated fits; a busy cell is not necessarily an error.
5. Inspect the pooled OOF comparison, per-fold results, training results, and both figures. Check the scope labels before interpreting scores.
6. Compare each classifier with the baseline, focusing on log loss and Brier score for the probability task. Record precision and recall as required classification measures.
7. Save the notebook with Ctrl + S. Save screenshots of training results, cross-validation results, and OOF confusion matrices under reports/screenshots with names clearly identifying their scope.

Command-line entry point from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

Saved pipelines include preprocessing and are fitted on the full training partition. They are initial uncalibrated models, not independently validated probability generators. Later calibration must refit within the appropriate training boundaries, using the fixed protocol and fold membership. The saved model manifest records their inputs, versions, settings, and fingerprints.

## Results and verification

The verified first-run results are recorded below. Keep the source shot table, feature contract, holdout, and CV membership unchanged. Model performance, convergence, prediction coverage, metric arithmetic, image layout, serialization round trips, and reproducibility must be checked before marking this phase complete.

## Verified initial development results

These are pooled OOF results for the 7,334 training shots, not final test scores.

| Model | Accuracy | Goal precision | Goal recall | Log loss | Brier score | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior baseline | 89.06% | 0.00% | 0.00% | 0.345174 | 0.097398 | 0.107440 |
| KNN (k=50) | 89.91% | 72.46% | 12.47% | 0.367443 | 0.082295 | 0.365256 |
| Gaussian Naive Bayes | 88.00% | 41.06% | 22.32% | 1.297262 | 0.116012 | 0.322360 |
| Logistic regression | 90.09% | 65.82% | 19.45% | 0.274668 | 0.079605 | 0.417418 |

Logistic regression has the lowest observed development log loss and Brier score among these fixed candidates. This does not establish statistical superiority, final test performance, or adequate calibration. KNN improves Brier score over the baseline but has worse log loss. Its OOF predictions assign probability zero to 806 shots, including 19 goals; confident errors receive a large log-loss penalty. Gaussian Naive Bayes assigns probability one to 183 OOF shots, including 60 non-goals. These are numerical model predictions, not statements of physical certainty. Keep the poor results rather than changing the fixed specifications retrospectively without a documented new development experiment.

At the 0.5 threshold, logistic regression detects 156 of 802 actual goals and misses 646. It also predicts 81 non-goals as goals. Its 90.09 percent accuracy therefore must be interpreted alongside its 19.45 percent goal recall. Scoring probability and choosing a binary threshold are different tasks; a useful xG estimate can remain below 0.5 for many shots that happen to score.

All four classifiers have exactly one OOF prediction per training event: 29,336 OOF prediction rows in total. The per-fold table contains 40 rows. Four pipelines were then fitted on all training rows, their optimistic resubstitution results recorded separately, and their serialized predictions checked after reloading.

All 52 project tests passed. Every notebook code cell executed successfully from the notebooks/ working directory. The two PNGs were visually reviewed, including the revised panel titles. Repeat execution reproduced identical prediction/metric CSVs, PNGs, serialized pipelines, model manifest, protocol, and report. Saved probabilities reproduce the reported log loss, Brier score, AP, ROC AUC, and accuracy. The original shot CSV, feature contract, holdout allocation, and CV allocation were unchanged. See docs/phase_07_verification.json for fingerprints and checks.

### Evidence to retain

Save notebook outputs and screenshots of the training resubstitution table, the pooled OOF comparison, ten-fold results, and OOF confusion matrices under reports/screenshots. Use filenames such as phase07_training_results.png, phase07_cv_results.png, phase07_fold_results.png, and phase07_oof_confusion.png. The final assignment still needs separate test-result screenshots after the evaluation protocol is frozen in the next phase.
