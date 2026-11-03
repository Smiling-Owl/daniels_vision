# Phase 10 submission checklist

The report and exported results are assembled. The project verification run passed all 73 tests. Genuine UI screenshots remain outstanding because `reports/screenshots` contained only `.gitkeep` at report creation. Do not describe exported charts as screenshots. Add real captures from VS Code/Jupyter, insert them in the report, then render the revised report again.

| Assignment requirement | Report location or evidence | Status |
|---|---|---|
| Dataset description, at least five sentences, dataset picture | Section 1.1, extracted shot-row excerpt, training shot map | Included; exported figures are not UI screenshots |
| Dataset content, instances and attributes | Section 1.2 | Included |
| Independent and dependent attributes | Sections 1.3 and 1.4 | Included |
| Preprocessing, at least 15 sentences and three screenshots | Section 2; three capture slots in final checklist | Prose and exported evidence included; genuine screenshots pending |
| Splitting, at least 15 sentences and three screenshots | Section 3; three capture slots | Prose and tables included; genuine screenshots pending |
| Model 1 training, ten-fold CV, testing | Section 4.2 | Results included; three genuine screenshots pending |
| Model 2 training, ten-fold CV, testing | Section 4.3 | Results included; three genuine screenshots pending |
| Model 3 training, ten-fold CV, testing | Section 4.4 | Results included; three genuine screenshots pending |
| Test performance metrics and confusion matrices per classifier | Section 5, test metric tables and confusion counts | Exported results included; six cropped screenshots pending |
| Four guide questions, at least five sentences each | Section 6 | Included |
| Monte Carlo simulation | Section 7 | Included; 100,000 trials per held-out match, exact reference and limitations |
| El Clasico portfolio case study | Section 8 | Included as development OOF, not an independent test |
| Read-only calibrated screenshot views | `notebooks/09_report_evidence.ipynb` | Added; captures still need to be made in the running VS Code/Jupyter interface |
| References and source credit | References section | Included; add StatsBomb logo if publishing |

## Required screenshot files

Save original captures under `reports/screenshots/` with filenames corresponding to the following slots. Each training, validation, and test screenshot must show its scope clearly. The evaluation screenshots must show held-out test metrics or test confusion matrices, not training or cross-validation outputs.

| Capture group | Where to capture in VS Code/Jupyter |
|---|---|
| Dataset excerpt | `notebooks/09_report_evidence.ipynb` → Dataset and preprocessing evidence |
| Label/feature construction and quality reconciliation | `notebooks/02_shot_dataset.ipynb` → build and validation cells |
| Split allocation, group proof, and ten-fold summary | `notebooks/09_report_evidence.ipynb` → Split and grouped-fold evidence |
| KNN training, CV, test, and test metrics/confusion | `notebooks/09_report_evidence.ipynb` → KNN, 50 neighbours section |
| GaussianNB training, CV, test, and test metrics/confusion | `notebooks/09_report_evidence.ipynb` → Gaussian Naive Bayes section |
| Logistic training, CV, test, and test metrics/confusion | `notebooks/09_report_evidence.ipynb` → Logistic regression section |

The evidence notebook displays family-selected sigmoid variants, matching the report. Test metric and confusion captures may be cropped from the genuine visible interface; do not edit values or recreate the UI.

1. `preprocessing_01_dataset.png`
2. `preprocessing_02_labels_features.png`
3. `preprocessing_03_quality_reconciliation.png`
4. `split_01_match_allocation.png`
5. `split_02_group_separation.png`
6. `split_03_ten_fold_cv.png`
7. `knn_01_training.png`
8. `knn_02_cross_validation.png`
9. `knn_03_test.png`
10. `gaussian_nb_01_training.png`
11. `gaussian_nb_02_cross_validation.png`
12. `gaussian_nb_03_test.png`
13. `logistic_01_training.png`
14. `logistic_02_cross_validation.png`
15. `logistic_03_test.png`
16. `evaluation_knn_test_metrics.png`
17. `evaluation_knn_test_confusion.png`
18. `evaluation_gaussian_nb_test_metrics.png`
19. `evaluation_gaussian_nb_test_confusion.png`
20. `evaluation_logistic_test_metrics.png`
21. `evaluation_logistic_test_confusion.png`

## Final verification after screenshots

- Confirm the report's cover still leaves author, course, and instructor blank.
- Insert genuine screenshots at their matching chapters and keep captions/scope labels readable.
- Re-render the DOCX and inspect every page after insertion.
- Verify StatsBomb source attribution and include the official logo when distributing the analysis.
- Keep `reports/screenshots/` out of any public release until images have been checked for personal or account information.
