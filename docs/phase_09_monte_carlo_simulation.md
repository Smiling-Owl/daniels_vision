# Phase 9: Monte Carlo match simulation

Status: simulation, exact-reference checks, and automated verification complete; learner VS Code checkpoint pending. The learner confirmed Phase 8 before this phase began.

## What is being simulated

Use the frozen sigmoid-calibrated logistic-regression probabilities saved in Phase 8. This model was selected using training-only development scores before test evaluation. Do not replace it with a model that happens to have a better test score. This phase does not refit a model or select new parameters.

The scope is the 1,834 observed eligible shots in the 76 held-out matches. Each simulation repeats conversion outcomes for those same chances. It is a retrospective calculation conditional on the observed shots, not a pre-match forecast or a simulation of how the match would unfold if earlier events changed.

For each shot, draw a random number uniformly between zero and one. Count a goal when the draw is less than the shot's fixed model probability. Sum the simulated goals separately for the home and away teams. Repeat 100,000 times per match, then count home wins, draws, away wins, and scorelines.

A 0.2 probability shot may score in some repetitions and fail in others. Across many independent repetitions, its simulated conversion frequency tends toward 0.2. The probability remains a model estimate rather than a known physical truth. The simulation uses the probability itself, not a hard class prediction at 0.5. A shot assigned 0.2 can still score in a repetition even though its binary class prediction would be non-goal.

## Fixed protocol

- Source: the Phase 8 selected-variant test probabilities, verified against the final report and selection record.
- Scope: exactly the 76 locked test matches; no simulated full-season league table from an incomplete set of fixtures.
- Trial count: 100,000 per match.
- Random seed: 42, with separate deterministic random streams derived from match ID and team side.
- Memory: process 10,000 trials at a time rather than allocating one huge array for every match.
- Shot ordering: stable event-index order within a team.
- Example: the earliest held-out match by match date and then match ID, chosen before computing outcome probabilities and regardless of score or model performance.
- Goal types: primary simulated scores contain modelled shot goals only; own goals remain separate audit information.

Separate match streams make a match's results reproducible even if other matches are processed in a different order. Separate home/away streams also preserve results when changing the internal chunk size, provided trial and shot ordering remain fixed. The seed makes an experiment repeatable; it does not remove sampling error or establish model accuracy.

## Expected goals and expected points

A team's expected simulated shot goals equal the sum of its shot probabilities. Under the independent-shot assumption, its goal-count variance equals the sum of p times (1-p). These are checks of the simulator as well as interpretable summaries of recorded chance quality.

Exact expected points are 3 times the win probability plus the draw probability. Home and away expected points need not sum to three: their sum is 3 minus the draw probability, since a draw awards only two total league points. Expected points here are conditional on the recorded shots, not a causal claim about points a team deserved.

A team's actual goal total may differ from its expected goals because conversion is uncertain and the model is imperfect. Avoid concluding that a team should have won from these estimates alone.

## Exact calculation as a reference

A sum of independent Bernoulli shots with different scoring probabilities follows a Poisson-binomial distribution. Compute each team's full goal-count distribution by repeatedly convolving [1-p, p]. This is not a Poisson approximation based only on summed xG.

Take the outer product of the two team distributions to obtain the complete joint scoreline distribution under the same assumptions. Sum the appropriate cells for home wins, draws, and away wins. Retain the full mathematical support in the exported scoreline table, including cells with zero observed Monte Carlo counts.

Comparing Monte Carlo estimates with this exact reference tests the numerical implementation. Agreement does not validate the independent-shot assumption, the model's probabilities, or the realism of counterfactual match dynamics. Both calculations use the same fixed probabilities and assumptions.

## Sampling error and convergence

For a team's Monte Carlo mean goals, the expected sampling standard error is sqrt(goal-count variance / number of trials). For an outcome frequency with exact probability q, it is sqrt(q*(1-q) / number of trials). Checks allow six such standard errors, with a small numerical/count allowance where appropriate. These are implementation checks, not proof that a football probability is known that accurately.

The example match also uses prefixes of the same simulated trials at 100, 1,000, 10,000, and 100,000 repetitions. Compare each prefix with the exact reference. A larger sample generally reduces typical sampling error, but an individual error path need not improve monotonically. Do not require every convergence line to decrease at every checkpoint.

Monte Carlo sampling error describes numerical randomness conditional on the fitted probabilities. It does not include uncertainty in fitted parameters, calibration, missing context, event recording, or the football modelling assumptions. More trials reduce the first kind of error, not the others.

## Own goals and observed scores

The source dataset reconciles official scores as ordinary shot goals plus credited own goals. Own goals were not ordinary goal-labelled shots used by the classifier and must not silently be treated as modelled Bernoulli attempts.

The match summary therefore retains the official home/away score, observed shot-goal counts, and own-goal credits separately. Compare simulated shot-generated scores with the observed shot-goal component when checking consistency. Where own goals occur, the official score can differ from that component. This phase neither samples own goals nor adds the observed own goals deterministically to every simulated repetition.

## Assumptions and limits

Shots within one possession can depend on each other. A rebound attempt, for example, might not occur if the earlier attempt scored. Treating every observed shot as an independent Bernoulli event ignores that dependence and freezes the set of attempts. The calculation also does not generate tactics, substitutions, red cards, future shots, or reactions to a different scoreline.

Only one historical season and 76 held-out matches are covered. The simulator is a conditional analysis of observed chance conversion, not a validated pre-match prediction system or a complete league simulation.

## Learner walkthrough

1. Open E:\dev-work\daniels_vision in VS Code.
2. Open notebooks/07_monte_carlo_simulation.ipynb and select the project's .venv kernel.
3. Read the scope and assumptions, then run the cells in order using Shift + Enter.
4. Let the simulation cell finish. It performs 100,000 repetitions for each of 76 matches.
5. Inspect the match summary and check the selected example's teams, shot counts, probability sums, and observed-score audit fields.
6. Read the scoreline distribution and home/draw/away probabilities alongside their exact references.
7. Inspect the convergence figure. An error increase between two checkpoints is possible and does not automatically indicate a bug.
8. Save the notebook with Ctrl + S and retain screenshots under reports/screenshots.

Command-line entry point from the project folder:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_matches.py
```

## Results and verification

Verified actual example results and completed checks are recorded below. Source data, model selection, classifier artifacts, and Phase 8 results remain unchanged. This phase implements and checks the simulator; the following phase will assemble the report and portfolio presentation around its verified results.

## Verified actual results

The simulator processed all **76 held-out matches and 1,834 shots**, with **100,000 repetitions per match**: 7.6 million conditional match repetitions. It uses the frozen sigmoid-calibrated logistic-regression candidate, with no classifier refit or changed selection.

The metadata-selected example is **Atlético Madrid vs Las Palmas, 22 August 2015**, match ID 3825563. The official score was 1-0; the observed shot-goal component was also 1-0, with no credited own goals in this match.

| Team | Recorded shots | Sum of shot probabilities / exact expected goals | Monte Carlo mean goals | Exact expected points |
| --- | ---: | ---: | ---: | ---: |
| Atlético Madrid | 12 | 0.723263 | 0.725350 | 1.427314 |
| Las Palmas | 7 | 0.576450 | 0.577990 | 1.173382 |

| Outcome | Exact probability | Monte Carlo probability |
| --- | ---: | ---: |
| Atlético Madrid win | 34.2670% | 34.2840% |
| Draw | 39.9304% | 39.8720% |
| Las Palmas win | 25.8026% | 25.8440% |

These probabilities are conditional on the observed shots under the independent-shot model. They are not probabilities issued before kickoff. The actual home win is one possible result, rather than a reason to change the simulation or conclude that a different outcome should have occurred.

The most likely individual scoreline is 0-0 with exact probability 24.74 percent; 1-0 has 19.65 percent. The most likely scoreline is not certain and is distinct from the combined probability of all draws or all home wins. A scoreline with zero Monte Carlo observations can still have a positive exact probability if it is rare relative to the finite sample size.

Across all 76 matches, the largest absolute Monte Carlo-versus-exact home/draw/away probability difference was **0.003813**, approximately **0.3813 percentage points**. Every numerical outcome and mean check passed its predefined tolerance. This measures implementation sampling error, not model uncertainty or football forecast accuracy.

The 76-match official scores contain **218 goals**: **212 ordinary shot goals plus six own goals**. The six own goals occurred in six matches and are retained as reconciliation context. They were neither sampled nor added to simulated scores. Three were credited to home teams and three to away teams.

## Saved outputs

- reports/tables/phase09_match_summary.csv: all 76 matches, including shots, probability sums, exact and simulated outcomes, means, MC standard errors, expected points, and own-goal reconciliation.
- reports/tables/phase09_scoreline_distribution.csv: 12,533 full-support scoreline rows across all matches, with exact probabilities and Monte Carlo counts/probabilities. The example contributes 104 scoreline cells.
- reports/tables/phase09_convergence.csv: 12 rows for three outcomes at four trial-count checkpoints (100, 1,000, 10,000, and 100,000), using prefixes of the same seeded run.
- reports/tables/phase09_example_shots.csv: the example's 19 source shot events with team identifiers, observed labels, and frozen scoring probabilities.
- reports/figures/phase09_example_scoreline_heatmap.png: complete exact example scoreline distribution.
- reports/figures/phase09_example_outcomes.png: exact and Monte Carlo outcome comparison.
- reports/figures/phase09_example_convergence.png: sampling convergence against the exact reference.
- docs/phase_09_protocol.json and docs/phase_09_simulation_report.json: frozen settings, sources, output fingerprints, counts, and checks.

## Completed verification

All **69 project tests passed**, including ten simulator tests covering known exact cases, no-shot/deterministic-shot cases, invalid inputs, seeded repeatability, chunk invariance, event-ID joins, team metadata, own goals, and complete scoreline support. Every notebook code cell executed in the project environment. The final run emitted no runtime warnings, and all three figures were visually inspected.

Independent checks reconstructed all 76 full scoreline grids, verified that probability masses and trial counts sum correctly, checked the product of independent marginals, recomputed expected goals from the original frozen shot probabilities, checked simulated means/outcomes against sampling tolerances, and verified expected points and official-score reconciliation. The notebook rerun reproduced identical CSVs, PNGs, protocol, and report. All protected source, split, fold, Phase 7, and Phase 8 files were unchanged. See docs/phase_09_verification.json for fingerprints and execution details.

Save notebook outputs and screenshots under reports/screenshots, for example phase09_example_summary.png, phase09_scoreline_distribution.png, phase09_outcome_comparison.png, and phase09_convergence.png. The notebook shows a chronological ten-match sample and compact example tables; the CSV retains all 76 matches and every scoreline cell.

Phase 10 will assemble the final academic report and portfolio presentation from the verified data, classifiers, final test evaluation, and simulator results. The simulator's numerical agreement with the exact reference does not remove its football-model assumptions.
