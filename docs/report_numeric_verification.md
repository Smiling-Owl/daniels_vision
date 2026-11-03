# Report numerical verification

Checked on 24 October 2026 against the project's pinned StatsBomb Open Data revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`.

## Result

No numerical mismatches were found in the audited report. Dataset totals were recounted from the 380 raw event JSON files. Classifier scores were independently recalculated from saved probabilities; the frozen final models also reproduced their training metrics and test probabilities without refitting. All printed classification tables matched the underlying values at their displayed rounding. Exact score distributions and the seeded Monte Carlo results were independently reproduced for all 76 test matches and both El Clasico examples.

## Dataset and split counts

| Scope | Matches | Shots | Goal shots | Non-goal shots |
|---|---:|---:|---:|---:|
| Full season | 380 | 9,168 | 1,014 | 8,154 |
| Training | 304 | 7,334 | 802 | 6,532 |
| Held-out test | 76 | 1,834 | 212 | 1,622 |

The 1,295,354 source events include passes and other event types, not just shots. All 9,168 source shot IDs are retained exactly once in the processed table. The 35 processed columns include nine model inputs; they are not 35 independent predictors. Each outer-fold transformation produces 28 encoded columns.

The official season score total is 1,043 goals: 1,014 goal-labelled shots plus 29 own-goal credits. Paired Own Goal For records corroborate those 29 credits and are not counted again. All 380 official match scores reconcile.

The split is exactly 80/20 by match count. The test shots represent 20.0044% of all shots, which rounds to approximately 20.00%. Ten-fold validation contains 29–31 matches, 726–743 shots, and 78–85 goals per validation fold. The five dataset examples, all twelve training geometry bins, and the stated geometry rates were independently checked.

## Selected model test results

These values use the same 1,834 test shots, Goal as the positive class, and a fixed probability threshold of 0.5. The report's selected variants use sigmoid calibration.

| Model | Accuracy | Goal precision | Goal recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 89.86% | 74.07% | 18.87% | 1,608 | 14 | 172 | 40 |
| Gaussian Naive Bayes | 88.44% | 0.00% | 0.00% | 1,622 | 0 | 212 | 0 |
| Logistic regression | 89.91% | 74.55% | 19.34% | 1,608 | 14 | 171 | 41 |

Gaussian Naive Bayes makes zero positive predictions at this threshold. Its precision is reported as 0 using `zero_division=0`; this is a reporting convention for the otherwise undefined ratio. Its saved probabilities are not all zero.

Checks covered all seven raw/calibrated/baseline variants, pooled development predictions, 70 model-by-fold rows, training resubstitution scores, reliability bins, all 72 classified examples, confusion matrices, the three-scope comparison, and the calibration comparison. The models' frozen file hashes and saved settings also matched.

## Simulation checks

An independent Poisson-binomial recurrence reproduced every exact scoreline probability and the corresponding home-win/draw/away-win probabilities. Independent full-array Bernoulli draws using the saved seed and match/side streams reproduced every scoreline count from 100,000 trials per fixture, across 78 fixtures. The convergence table's prefix counts, probabilities, absolute errors, and sampling standard errors also matched.

For the 76 test matches, official goals total 134 home and 84 away. Recorded shot goals total 131 home and 81 away, with three own-goal credits for each side. The simulation samples shot goals only. It remains conditional on the observed shots and fixed predicted probabilities; it is not a pre-match forecast or a measure of model uncertainty.

The November El Clasico exact conditional home/draw/away probabilities round to 9.2%, 14.2%, and 76.6%. The April probabilities round to 20.5%, 25.9%, and 53.6%. Both cases use saved development OOF predictions and are not additional independent test matches.

## Report clarifications

The revised Figure 1 caption states the training sample explicitly: 7,334 shots, including 802 goals and 6,532 non-goals. Table 1's caption identifies distance in StatsBomb units and angle in radians. The acquisition description now says 380 matches and their event files; the source has one season match-index JSON, not 380 match-index files. The own-goal total and zero-positive precision convention are explicit.

The audit recalculated saved OOF results without rerunning the nested cross-validation fits. It establishes consistency with the archived source data and predictions; it does not establish that provider annotations are error-free. Genuine application screenshots remain pending, as before.
