# El Clasico: out-of-fold development case study

Status: both case studies and automated verification complete; learner notebook checkpoint pending. The user requested El Clasico examples after the Phase 9 held-out simulation.

## Why these matches are separate

Both La Liga 2015/2016 fixtures are in the locked training partition:

| Date | Fixture | Official result | Outer validation fold | Recorded shots |
| --- | --- | --- | ---: | ---: |
| 21 November 2015 | Real Madrid vs Barcelona | 0-4 | 2 | 31 |
| 2 April 2016 | Barcelona vs Real Madrid | 1-2 | 7 | 29 |

The original Atletico Madrid vs Las Palmas example was chosen as the earliest held-out fixture. It and the full 76-match final evaluation remain intact. These two additional matches were requested for their football interest and portfolio relevance, not selected using a favorable model score.

## Which probabilities are used

Use only the selected sigmoid-calibrated logistic-regression variant's saved outer out-of-fold predictions from Phase 8. Each shot in a fixture was predicted by a model whose base-model fitting and calibration excluded that fixture's whole outer validation fold. The November and April fixtures therefore use different fold-specific fitted models from the same fixed model family and calibration procedure.

Do not use final-full-training predictions or training resubstitution probabilities for these cases. That final model has already seen both fixtures. This case study performs no refitting, threshold search, or model selection.

Out-of-fold exclusion does not turn these fixtures into a new final test set. They are part of the development partition used to compare and select model variants. Label their outputs **development OOF case study**, not final held-out evaluation. The original 76 test matches remain the independent evaluation for the frozen experiment. The grouped folds are not chronological: other fitting matches can occur later in the season, so these cases are not chronological backtests.

## How the simulation works

For each fixture, reuse the Phase 9 exact Poisson-binomial and Monte Carlo routines. Draw each observed shot as an independent Bernoulli event at its saved OOF scoring probability, sum goals by team, and repeat 100,000 times. Use seed 42 with separate streams derived from match ID and side. Compute the exact full joint score distribution under the same assumptions and compare means and home/draw/away frequencies within the predefined Monte Carlo sampling tolerances.

Expected goals equal the sum of shot probabilities. Expected points equal three times win probability plus draw probability. Scores include ordinary modelled shot goals only; official scores and any credited own goals remain separate context. No own goal is added to the simulation.

These are retrospective calculations conditional on recorded shots, not pre-match forecasts, estimates of what a team deserved, or complete counterfactual replays. Earlier simulated goals do not remove later rebound attempts or change tactics. Both exact and Monte Carlo calculations share this independence limitation. More repetitions reduce numerical sampling noise, not uncertainty in the fitted probabilities.

## Learner walkthrough

1. Open E:\dev-work\daniels_vision in VS Code.
2. Open notebooks/08_el_clasico_case_study.ipynb.
3. Select the project's .venv kernel.
4. Read the scope explanation and run cells in order with Shift + Enter.
5. Confirm that both matches are labelled development cases and show their outer fold identifiers.
6. Inspect each match's shot counts, expected goals, exact and simulated outcome probabilities, and most likely scorelines.
7. Read the heatmap axes carefully: the home and away teams reverse between the two fixtures.
8. Save the notebook with Ctrl + S. Save screenshots under reports/screenshots with case-specific names, such as el_clasico_2015_outcomes.png and el_clasico_2016_outcomes.png.

The original notebooks/07_monte_carlo_simulation.ipynb and all Phase 7-9 numeric results remain unchanged. Use the new notebook for these showcase cases rather than replacing the held-out example.

## Verification and actual results

Actual case results and completed checks are recorded below. Preserve the original train/test split and frozen model selection. The purpose of this addition is an interpretable football case study using the existing development validation work.

## Verified case results

Each fixture was simulated 100,000 times using the saved selected-variant OOF probabilities. These are exact outcome probabilities under the same independent-shot assumptions used by the simulator.

### 2015-11-21: Real Madrid vs Barcelona

Official score: 0-4. Outer OOF fold: 2. No own goals were credited in this fixture.

| Team | Shots | Expected shot goals | Monte Carlo mean goals | Exact expected points |
| --- | ---: | ---: | ---: | ---: |
| Real Madrid | 13 | 1.098079 | 1.102420 | 0.418438 |
| Barcelona | 18 | 2.838537 | 2.840860 | 2.439071 |

| Outcome | Exact probability | Monte Carlo estimate |
| --- | ---: | ---: |
| Real Madrid win | 9.1982% | 9.2220% |
| Draw | 14.2491% | 14.3650% |
| Barcelona win | 76.5527% | 76.4130% |

### 2016-04-02: Barcelona vs Real Madrid

Official score: 1-2. Outer OOF fold: 7. No own goals were credited in this fixture.

| Team | Shots | Expected shot goals | Monte Carlo mean goals | Exact expected points |
| --- | ---: | ---: | ---: | ---: |
| Barcelona | 14 | 0.900562 | 0.902220 | 0.873845 |
| Real Madrid | 15 | 1.544232 | 1.544670 | 1.867096 |

| Outcome | Exact probability | Monte Carlo estimate |
| --- | ---: | ---: |
| Barcelona win | 20.4929% | 20.4490% |
| Draw | 25.9059% | 25.9510% |
| Real Madrid win | 53.6012% | 53.6000% |


The first fixture assigns Barcelona the larger expected-goal total and a 76.55 percent conditional win probability. The second assigns Real Madrid the larger expected-goal total and a 53.60 percent conditional win probability. These describe the modelled conversion of recorded chances; they are not pre-kickoff predictions or evidence of general predictive accuracy from two examples.

The most likely scorelines are 1-3 in the first fixture (10.48 percent) and 0-1 in the second (13.57 percent), in home-away order. A most likely scoreline can still have a low probability because probability is spread across many possible scores. Neither the most likely score nor expected goals needs to equal the official result.

## Saved evidence and checks

The new notebook is notebooks/08_el_clasico_case_study.ipynb. The three tables are reports/tables/el_clasico_match_summary.csv, el_clasico_scorelines.csv, and el_clasico_shots.csv. The source audit contains all 60 case shots with event identifiers, probabilities, and prediction/locked-fold assignments. Full scoreline support is retained for both cases.

Three figures are under reports/figures: el_clasico_266424_outcomes.png, el_clasico_267533_outcomes.png, and el_clasico_joint_scorelines.png. Every figure labels the development OOF scope. The heatmap uses one common probability scale across both panels, with home goals on the vertical axis and away goals on the horizontal axis.

All 73 project tests passed, and every notebook code cell executed successfully. The saved case probabilities exactly match the selected Phase 8 OOF event probabilities and the Phase 6 fold map. Independent checks verified full joint probability mass, Monte Carlo count totals, expected goals, outcome probabilities, sampling tolerances, and expected-point arithmetic. The maximum outcome Monte Carlo error across the two cases was about 0.001397, or 0.1397 percentage points. All three figures were visually reviewed after correcting the shared colourbar layout.

Reruns reproduced identical case CSVs, figures, protocol, and report. All 54 protected pre-existing artifacts were unchanged, including the held-out results, original Atletico example, model files, and split/fold membership. See docs/el_clasico_protocol.json, docs/el_clasico_report.json, and docs/el_clasico_verification.json for provenance and checks.

For the portfolio, these two named fixtures can be the showcase examples, while the full 76-match held-out evaluation supports the reported model performance. Keep their scope labels and report them alongside the historical-season, shot-dependence, and probability-uncertainty limitations.
