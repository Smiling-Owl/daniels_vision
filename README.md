# Beyond the Scoreline: La Liga Shot Outcome Modeling

A reproducible Python data-mining project that estimates goal probability for recorded football shots, evaluates K-Nearest Neighbours (KNN) and Gaussian Naive Bayes, and adds logistic regression as a benchmark. The analysis uses StatsBomb Open Data for La Liga 2015/2016. Its probability estimates are an xG-style project output; StatsBomb's own xG field is retained only for reference and is not used as the target or a model input.

![StatsBomb logo](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png)


## Project at a glance

- 9,168 shots from 380 matches; the target is whether the recorded shot outcome is Goal.
- A locked match-grouped split keeps all shots from one fixture together: 7,334 shots from 304 training matches and 1,834 shots from 76 held-out matches.
- Ten-fold stratified group cross-validation and nested, training-only sigmoid calibration inform the fixed model variants before the held-out evaluation.
- The positive class is Goal. Test classification metrics use the prespecified probability threshold 0.5.

| Frozen sigmoid variant | Test accuracy | Goal precision | Goal recall | Test log loss |
| --- | ---: | ---: | ---: | ---: |
| KNN (k=50) | 89.86% | 74.07% | 18.87% | 0.2906 |
| Gaussian Naive Bayes | 88.44% | 0.00% | 0.00% | 0.3318 |
| Logistic regression | 89.91% | 74.55% | 19.34% | 0.2770 |

The all-non-goal reference baseline also reaches 88.44% accuracy and predicts no goals. The selected models have low goal recall at this fixed threshold, so their accuracy should not be read as strong goal-detection performance. See the [numerical audit](docs/report_numeric_verification.md) for the verified counts, confusion matrices, and scope notes.

## Analysis and artifacts

- [Template-matched final project report](reports/final_project_report_template_matched.docx) — includes methods, training/OOF/test scopes, guide-question responses, confusion matrices, calibration, conditional simulation, and the El Clasico case study. Genuine application screenshots and the student's cover details remain to be added.
- [Portfolio presentation](reports/portfolio_presentation.pptx)
- [Report evidence notebook](notebooks/09_report_evidence.ipynb) — reads saved artifacts; it does not fit models or reselect variants.
- [Model comparison](reports/figures/phase07_model_comparison.png)
- [Held-out confusion matrices](reports/figures/phase08_test_confusion_matrices.png)
- [Training shot map](reports/figures/phase05_02_training_shot_map.png)
- [Conditional scoreline example](reports/figures/phase09_example_scoreline_heatmap.png)
- [El Clasico OOF scorelines](reports/figures/el_clasico_joint_scorelines.png)
- [Phase walkthroughs and verification records](docs/)
- [Portfolio summary and resume bullets](portfolio/README.md)

## Monte Carlo and El Clasico scope

The simulator runs 100,000 trials for each of the 76 held-out matches and checks the results against exact independent-shot Poisson-binomial distributions. It samples Bernoulli outcomes for the shots that actually occurred, using frozen probabilities. It does not sample own goals, reconstruct official scores, represent model uncertainty, or forecast a match before kickoff.

The El Clasico examples use saved development out-of-fold probabilities from the selected logistic-regression variant. Each example's match was excluded from the fit and calibration that produced its OOF predictions. The model family was selected using development results, so these examples are illustrative development cases, not additional independent test evidence.

## Reproduce

Use Python 3.14 and run these commands in PowerShell from the cloned project directory. Before acquiring data, read the [StatsBomb terms and attribution notice](THIRD_PARTY_NOTICES.md) and follow the provider's registration instructions.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Fetch and verify the pinned source snapshot locally; raw and full processed data stay out of Git.
.\.venv\Scripts\python.exe scripts\acquire_data.py
.\.venv\Scripts\python.exe scripts\acquire_data.py --verify-only

# Rebuild the shot table, lock partitions, and run the training-only workflow.
.\.venv\Scripts\python.exe scripts\build_shot_dataset.py
.\.venv\Scripts\python.exe scripts\reserve_holdout.py
.\.venv\Scripts\python.exe scripts\explore_shots.py
.\.venv\Scripts\python.exe scripts\prepare_modeling.py
.\.venv\Scripts\python.exe scripts\train_models.py
.\.venv\Scripts\python.exe scripts\calibrate_evaluate.py --stage develop
.\.venv\Scripts\python.exe scripts\calibrate_evaluate.py --stage evaluate
.\.venv\Scripts\python.exe scripts\simulate_matches.py
.\.venv\Scripts\python.exe scripts\el_clasico_case_study.py

# Run the deterministic unit-test suite.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The `notebooks/` directory contains the step-by-step workflow from environment check through report evidence. Select the project's `.venv` kernel in VS Code. The scripts validate saved manifests and refuse incompatible existing state; read each phase guide before rerunning a stage.

## Data, attribution, and licensing

The raw StatsBomb event JSON, full processed shot table, per-shot prediction CSVs, and fitted `.joblib` models are not included in the repository. The `.gitignore` keeps those locally generated artifacts out of Git; the acquisition, modeling, and analysis scripts recreate them. Aggregate result tables, figures, notebooks, documentation, and the academic report/presentation are retained. The acquisition script fetches the pinned public source revision for local research use; review the provider terms and registration instructions first. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact source revision and publication requirements.

This repository does not add a software license. The course report still has genuine screenshot placeholders; exported charts are analytical figures, not application screenshots. The report is not ready for course submission until those captures and the requested personal cover details are supplied and page layout is reviewed.
