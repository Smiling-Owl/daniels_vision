# Phase 10 Report and portfolio preparation

This phase turns the frozen analysis into an academic report and a presentation. The report remains a submission draft until the genuine screenshots required by the assignment are inserted. Student, course, section and instructor details remain blank as requested.

## 1 Open the report

Open `E:\dev-work\daniels_vision\reports\final_project_report.docx` in Word. Read the executive summary and the research question first. Check that you can explain the observation unit: one recorded shot, with a goal or non-goal label. Read each method and result before using the report as your submission. Revise the prose into wording you understand and follow your course's rules about assistance and attribution.

The model sections distinguish raw classifiers from the sigmoid versions selected using development results. These are separate variants. Keep the variant name and evaluation scope visible in every screenshot.

## 2 Open the evidence notebook

1. Open `E:\dev-work\daniels_vision` in VS Code.
2. Open `notebooks\09_report_evidence.ipynb`.
3. Select this project's `.venv` Python kernel, as in Phase 1.
4. Read the introductory Markdown cells.
5. Run the code cells from top to bottom with **Shift + Enter**.
6. Confirm that the displayed allocation contains 304 development matches and 76 test matches.
7. Confirm that the main report tables use the family-selected sigmoid variants. The raw-model comparison is separately labelled.
8. Save the notebook with **Ctrl + S**.

This notebook reads saved results. It does not retrain models, change the split, choose new settings, or run new simulations. A missing file or failed assertion means the evidence is not ready to capture; retain the error message and resolve it before continuing.

## 3 Capture genuine evidence

Open `docs\phase_10_submission_checklist.md` beside the notebook. Follow its capture list in order. For each item:

1. Scroll to the matching notebook section and make the title, model variant and result readable.
2. Press **Windows + Shift + S** and select the relevant screen area.
3. Open the captured image in Snipping Tool and use **Save as**.
4. Save the image in `E:\dev-work\daniels_vision\reports\screenshots` using the filename in the checklist.
5. Check the saved image before moving on. It must show the real output clearly, without changing any numbers.

Capture at least three preprocessing views and three splitting views. Capture training, ten-fold validation and final test evidence for each reported classifier. For evaluation, capture the **test** metrics and **test** confusion matrix, as the assignment specifies. Training resubstitution, development OOF and held-out test results answer different questions and must retain their scope labels.

## 4 Insert the screenshots

In Word, use **Insert > Pictures > This Device** at the matching report section. Choose the genuine capture, retain its proportions and add a caption explaining what it demonstrates. Keep the original image files in the screenshots folder. The exported charts already in the report are analysis figures; they do not replace a required application screenshot.

After inserting all captures, check every page in **Print Layout**. Ensure that tables fit the margins, captions stay with their pictures and text is readable. Review the screenshot checklist against the completed report before removing its draft status. Add personal cover details when you are ready.

## 5 Rehearse the presentation

Open `reports\portfolio_presentation.pptx`. Its speaker notes identify the sources and provide the qualifications behind the numbers. Practise explaining:

- Why an all-non-goal baseline reaches 88.4% test accuracy while finding no goals.
- Why the positive class is goal, and what precision and recall measure.
- Why a whole match stays on one side of each split or fold.
- Why the selected calibrated model stays fixed even though raw logistic regression slightly outperforms it on test log loss.
- Why El Clasico uses development OOF predictions and adds no independent test evidence.
- Why simulation conditions on observed shots, assumes independent outcomes and excludes own goals.
- Why 100,000 Monte Carlo trials reduce sampling error without resolving model uncertainty.

## 6 Prepare the portfolio copy

Read `portfolio\README.md`, including the reproducibility commands, limitations and suggested resume bullets. Use claims you can explain and reproduce. Keep the historical conditional simulation description accurate. Publishing the project is a separate step: confirm the dataset's attribution and logo requirements, review screenshots for personal details and check what files belong in the public repository before uploading.

## Phase checkpoint

The analysis and draft artifacts are prepared. This phase's learner checkpoint is complete when the report contains the genuine required captures and you can explain the core results without reading the slides word for word.
