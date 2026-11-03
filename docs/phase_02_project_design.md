# Phase 2 dataset scope and research design

Status: Phase 2 research plan recorded on 22 October 2026. The user selected La Liga 2015/2016 as the core dataset. This is a prospective plan, not a record of completed modelling. Data-dependent decisions and counts remain subject to the acquisition and validation checkpoints.

## Verified availability

Inspected on 22 October 2026 using StatsBomb Open Data competition and match indexes. These are counts of indexed matches, not yet verified counts of downloaded event files or usable shots.

| Candidate | Indexed matches | Coverage observation |
| --- | ---: | --- |
| La Liga 2015/2016 | 380 | 20 teams; each appears in 38 indexed matches. Recommended core dataset. |
| FIFA World Cup 2018 | 64 | 32 teams. |
| FIFA World Cup 2022 | 64 | 32 teams. |
| Champions League, all currently listed seasons | 18 | One selected match per listed season; not complete tournament seasons. |

Other indexed La Liga seasons are predominantly Barcelona-focused selections. Do not describe them as complete league seasons without checking coverage.

Detailed counts, competition IDs, season IDs, team-frequency checks, and match IDs are in competition_coverage_audit.json. Raw source indexes were read from the repository's mutable master branch. Pin a repository revision and record file hashes during acquisition for reproducibility. No event files have been downloaded during this audit.

## Research question

How well do KNN and Gaussian Naive Bayes estimate goal probability from shot characteristics, and how do their estimated probabilities affect simulated match scorelines and outcomes?

## Observation and target

One observation is one eligible shot event. The supervised target is 1 when the recorded shot outcome is Goal and 0 for other documented shot outcomes. Missing or unrecognized outcomes must be investigated, not automatically labelled 0. Predicted probability of class 1 is the model's xG estimate.

The core scope is the indexed La Liga 2015/2016 season: competition ID 11, season ID 27, 380 indexed matches. Use regulation time including stoppage time (periods 1 and 2). Include in-match penalties and identify them explicitly as a shot type; inspect their frequency and probability quality separately. Exclude extra time and penalty shootouts if encountered and flag such records for review because they are unexpected in a regular league season. Reconcile own goals separately because they may not appear as ordinary goal-labelled shots. Do not silently equate shot-goal totals with official match scores. The initial simulator describes shot-generated scorelines; report any difference from the official score due to own goals or omitted event types, rather than silently adding or dropping goals.

## Feature policy

Candidate inputs: shot location, derived distance and angle to goal, body part, shot type, and contextual attributes whose meaning and availability are verified in the chosen event files. Check coordinate units before describing distances in metres. Exclude outcome, shot end location, final score, and post-shot fields. Use match IDs for grouping and audit only. Exclude StatsBomb xG from model inputs; retain it as an external probability reference, not ground truth. True outcomes are the evaluation labels. Do not assume the external model used an independent training sample.

## Evaluation plan and data-dependent checkpoint

Use a held-out test portion targeting the assignment's 80/20 requirement while retaining whole matches. Record the unit of the split and actual match and shot counts. Exact row-ratio feasibility is not yet established. Perform 10-fold match-grouped cross-validation inside training only, checking both classes in each validation fold. Fit preprocessing, any tuning, and calibration inside the appropriate training boundaries. Keep the test data out of model selection and calibration. Use the same split for all classifiers. Describe this as held-out-match generalization unless a chronological future-match protocol is separately defined.

Required reporting: training results, 10-fold cross-validation, testing results, test accuracy, precision, recall, and test confusion matrices. Probability extensions: log loss, Brier score, reliability plots, and precision-recall analysis. Document the goal decision threshold and choose it using training validation only. Compare against a constant training-prevalence probability baseline. Use KNN and Gaussian Naive Bayes as the required models; add logistic regression as the portfolio benchmark. Use a documented 0.5 threshold for the initial required binary classification reports; any alternative threshold must be selected using training validation only. Probability evaluation remains primary for xG and simulation.

## Simulation meaning

Sample each observed eligible shot as a Bernoulli event using its estimated scoring probability, sum simulated goals by team, and repeat. Report scoreline distributions, win/draw/loss probabilities, and expected points where meaningful. This is retrospective analysis conditional on observed chances, not a pre-match forecast. Explicitly document independence, rebound/possession dependencies, and any omitted goal types. Random conversion variation does not by itself quantify model uncertainty. Check expected simulated goals against summed probabilities and check convergence and probability totals.

## Sources and attribution

- StatsBomb Open Data: https://github.com/hudl/open-data
- Competition index: https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json
- La Liga 2015/2016 index: https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/11/27.json
- World Cup 2018 index: https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/3.json
- World Cup 2022 index: https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/106.json

Review the repository licence and comply with its attribution and publication conditions before distributing outputs. Record StatsBomb as the data source. Only listed competitions and seasons are covered; conclusions must reflect the selected sample.

## Completion checks and next phase

Phase 2 establishes the dataset, task, target, feature boundary, model comparison, evaluation principles, and simulation meaning. Dataset size in shots, class balance, feature availability, and exact split counts are not yet known. Phase 3 will pin the source revision, download and inventory the chosen event files, retain provenance, and verify source availability. Do not claim that all 380 event files or a specific number of shots have been verified before this check.


## Phase 5 evaluation-plan amendment

The holdout was reserved before detailed feature-outcome EDA, moving this operation forward from Phase 6. With seed 42, an allocation based only on match IDs and shot counts reserves 76 whole matches and 1,834 shots, leaving 304 matches and 7,334 shots for training. This meets the rounded 20 percent row target and exact 20 percent match target without optimizing labels or model scores. It is a seeded size-constrained allocation, not a uniform subset draw. Membership and input fingerprints are locked in docs/split_manifest.json. Phase 6 must reuse this split and create the ten folds using training matches only. Phase 4 already included basic whole-dataset integrity and label audits; this boundary protects subsequent EDA and model selection.

## Phase 7 implementation record

Initial fixed specifications are KNN with 50 uniform-weight neighbors, Gaussian Naive Bayes with var_smoothing=1e-9, logistic regression with C=1 and lbfgs, and a fitting-prevalence baseline. The protocol was locked before fitting. Phase 7 reports ten-fold OOF and full-training resubstitution results separately. Final test metrics and their required screenshots are deferred until the probability/calibration decisions in Phase 8 are settled; the holdout must not be used for those decisions. The initial binary decision rule remains P(goal) >= 0.5.

## Phase 8 implementation record

With base settings and the 0.5 decision threshold unchanged, sigmoid calibration was evaluated using ten locked outer folds and five inner match-grouped folds, with the whole preprocessing/classifier pipeline refitted inside each inner fold. The strict training-only pooled-log-loss rule selected sigmoid variants for all three families and sigmoid logistic regression for simulation; its development improvement over raw logistic regression was tiny. Final calibrated model files and the selection were frozen before test rows were loaded. All seven fixed variants were evaluated on the same 76 reserved matches. Raw logistic regression had slightly better test probability scores than its calibrated version; the frozen choice was retained. The held-out outcomes are now evaluated, so subsequent result-informed changes are exploratory rather than an untouched repetition of the final experiment.
