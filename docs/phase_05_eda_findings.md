# Phase 5 exploratory findings

Status: descriptive EDA complete. Every statistic, table, and figure below uses only the protected training partition. The held-out match labels, xG values, and feature relationships were not examined.

## Training-only summary

The training partition contains **7,334 shots across 304 matches**, with 802 goal-labelled shots and 6,532 non-goals. The overall goal rate is 10.9%. A rule that always predicts “non-goal” would attain 89.1% accuracy on these training rows. This is a descriptive class-imbalance baseline, not a trained model or a test result.

There are 76 in-match penalties, of which 52 are goal-labelled in the training partition. Penalties are a distinct shot type and should be reviewed separately in later probability evaluation.

## Geometry and shot context

In the training partition, the 0–6 distance band has a goal rate of 46.7% (n=214); the 36–80 band has 0.6% (n=162). Distance uses native StatsBomb coordinate units, not metres. Angle bins and category-specific counts and intervals are in the accompanying tables.

Observed differences across geometry, body part, shot type, technique, play pattern, and provider context flags are associations in this sample; they do not establish causal effects. The categorical table flags groups with fewer than 30 training shots (1 in this sample). A rare-category rate can be unstable even when shown with a confidence interval.

Context values marked NOT FLAGGED mean StatsBomb did not record the positive flag. They do not establish that pressure, a first-time attempt, or an open goal was objectively absent.

## Limits and next step

The 95% Wilson intervals use a binomial approximation that treats shots as independent. Shots from the same match can be related, so these intervals may be too narrow. The no-goal baseline is only a reference for class imbalance; it is not a fitted classifier. No preprocessing, model fitting, cross-validation, or test evaluation was done in this phase.

Phase 6 should reuse the locked match-level membership in `docs/split_manifest.json`. Keep the test partition sealed until model design and training-only evaluation decisions are complete. See `reports/figures/phase05_*.png`, `reports/tables/phase05_*.csv`, and `docs/phase_05_eda_report.json` for the training-only visuals, numeric aggregates, and source/output fingerprints.
