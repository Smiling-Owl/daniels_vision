# Portfolio: La Liga Shot Outcome Modeling

![StatsBomb logo](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png)

**Repository:** [Smiling-Owl/daniels_vision](https://github.com/Smiling-Owl/daniels_vision)  
**Topic:** xG-style shot goal-probability modeling and conditional match simulation  
**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter

This project builds a reproducible shot-level workflow from a pinned StatsBomb Open Data snapshot for La Liga 2015/2016. It compares KNN and Gaussian Naive Bayes, with logistic regression as an additional benchmark. Match-grouped splits, fold-local preprocessing, nested training-only calibration, saved model artifacts, and hash-checked manifests keep evaluation scopes auditable.

## Held-out test results

The positive class is Goal; classification uses a fixed 0.5 probability threshold. The table reports the frozen, family-selected sigmoid variants on the same 1,834 held-out shots.

| Model | Accuracy | Goal precision | Goal recall | Log loss |
| --- | ---: | ---: | ---: | ---: |
| KNN (k=50) | 89.86% | 74.07% | 18.87% | 0.2906 |
| Gaussian Naive Bayes | 88.44% | 0.00% | 0.00% | 0.3318 |
| Logistic regression | 89.91% | 74.55% | 19.34% | 0.2770 |

The all-non-goal baseline reaches 88.44% accuracy while finding no goals. The selected models' goal recall is low; these results are a historical single-season comparison, not a deployment claim. Full counts and qualifications are in the [report numerical audit](../docs/report_numeric_verification.md).

## Selected visuals and deliverables

- [Model comparison](../reports/figures/phase07_model_comparison.png)
- [Held-out test confusion matrices](../reports/figures/phase08_test_confusion_matrices.png)
- [Training shot map](../reports/figures/phase05_02_training_shot_map.png)
- [Monte Carlo scoreline example](../reports/figures/phase09_example_scoreline_heatmap.png)
- [El Clasico development OOF scorelines](../reports/figures/el_clasico_joint_scorelines.png)
- [Academic report](../reports/final_project_report_template_matched.docx)
- [Presentation](../reports/portfolio_presentation.pptx)

The simulation runs 100,000 independent-shot Bernoulli trials per held-out match and validates against exact Poisson-binomial distributions. It is conditional on shots already taken, excludes own goals, and does not include model uncertainty; it is not a pre-match forecast. The El Clasico examples use development out-of-fold predictions from the selected model family and are not independent test cases.

## Reproduction

Use Python 3.14 in PowerShell from the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\acquire_data.py
.\.venv\Scripts\python.exe scripts\build_shot_dataset.py
.\.venv\Scripts\python.exe scripts\reserve_holdout.py
.\.venv\Scripts\python.exe scripts\explore_shots.py
.\.venv\Scripts\python.exe scripts\prepare_modeling.py
.\.venv\Scripts\python.exe scripts\train_models.py
.\.venv\Scripts\python.exe scripts\calibrate_evaluate.py --stage develop
.\.venv\Scripts\python.exe scripts\calibrate_evaluate.py --stage evaluate
.\.venv\Scripts\python.exe scripts\simulate_matches.py
.\.venv\Scripts\python.exe scripts\el_clasico_case_study.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Raw data, the full processed shot table, per-shot prediction CSVs, and fitted `.joblib` models are omitted; the scripts recreate them locally. Aggregate results, figures, and the report are retained as analysis. Review [StatsBomb's source and attribution requirements](../THIRD_PARTY_NOTICES.md) before acquiring or publishing analysis. The report still needs genuine application screenshots and personal cover details before course submission.

## Resume bullets

- Built a reproducible Python pipeline for 9,168 shot events with pinned-source provenance, explicit goal labels, validated features, and a match-grouped 80/20 holdout.
- Compared KNN, Gaussian Naive Bayes, and sigmoid-calibrated logistic regression using ten-fold grouped out-of-fold evaluation and frozen held-out test results.
- Implemented exact Poisson-binomial score distributions and seeded 100,000-trial conditional simulations for 76 held-out matches.
