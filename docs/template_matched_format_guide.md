# Template-matched report format guide

## What changed

The new editable report reconstructs the supplied 42-page academic sample's visible layout from its PDF. Because the source is a PDF rather than an editable Word template, this is a reconstruction of the visible design, not a copy of the original DOCX styles. The original reference PDF and extracted university seals remain unchanged.

The report follows the sample's cover form, two-page contents, seven assignment chapters, references, and appendices. It uses Letter pages, one-inch margins, Times New Roman 12-point justified body text with 1.5 line spacing and first-line indents, numbered subsection headings, black grid tables, gray label cells, centered italic captions, and right-aligned Arial page numbers. Chapter starts and contents page references were refreshed in Microsoft Word and checked in the exported PDF. The cover has five rows and four columns; requested identity and date fields are blank. Chapter 4 presents training, held-out test, and ten-fold OOF results separately under each model. Chapter 5 has one model-and-scope classified sample table per section.

The football analysis, dataset, results, and model artifacts remain unchanged. The sample's unrelated dataset and procedures were not copied. The generated report is `reports/final_project_report_template_matched.docx`; the earlier `reports/final_project_report.docx` remains available and unchanged. The layout reference was the supplied `Galvez_Manguilimotan_DataMiningFinalProject.docx.pdf` sample.

## Genuine screenshot evidence

Screenshot spaces in the report are labeled as pending because the project's `reports/screenshots` folder contains no genuine VS Code or Jupyter captures. The exported plots and native Word tables are real analytical evidence; they are not represented as screenshots of software. Capture the actual notebook/application window yourself and save the original image in `reports/screenshots` when ready. Cropping a genuine capture for legibility is fine; do not alter displayed values or fabricate application UI.

## Chapter 5 sample table captures

A deterministic combined data file accompanies the report: `reports/tables/report_template_classified_samples.csv`. It contains 72 rows: eight sorted examples for each of three models across training resubstitution, development OOF, and held-out test. It preserves the complete event ID, match ID, actual label, goal probability, and predicted class. The report shortens event IDs only in print tables for readability.

In a Jupyter notebook, load this CSV and display one model/scope evidence table as follows. The explicit project path avoids assuming which directory VS Code used to open the notebook. Change the two values at the top to capture another section.

```python
from pathlib import Path
import pandas as pd
from IPython.display import display

project = Path(r"E:\dev-work\daniels_vision")
variant = "knn_50_sigmoid"          # gaussian_nb_sigmoid / logistic_regression_sigmoid
scope = "training_resubstitution"  # development_oof / held_out_test
examples = pd.read_csv(
    project / "reports/tables/report_template_classified_samples.csv",
    dtype={"event_id": str},
)
view = examples.loc[
    examples["variant"].eq(variant) & examples["scope"].eq(scope),
    ["event_id", "match_id", "y_true", "p_goal", "y_pred"],
].sort_values("event_id", kind="stable")
display(view.reset_index(drop=True))
```

For a capture, show the selected code cell and its output in the real notebook window, then use the operating system's screenshot function and save the unaltered capture with a clear filename under the project's `reports/screenshots` folder.

## Limits

The required screenshot evidence is still pending actual user capture. The Word layout was checked by opening and exporting through native Microsoft Word because the bundled `render_docx.py` renderer could not find LibreOffice. The output is an editable DOCX reconstruction based on the supplied PDF, not a source-template DOCX clone.
