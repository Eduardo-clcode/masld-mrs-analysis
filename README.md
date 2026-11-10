# Ex-vivo hepatic FAME ¹H-NMR analysis code

This repository consolidates the analysis workflow for ex-vivo hepatic fatty-acid methyl ester (FAME) ¹H-NMR measurements in a murine MASLD study. It brings together analysis stages maintained across fragmented project materials into one entry point, helping avoid repeated runs and duplicated workflow steps.

The implementation is based on the analysis materials and study files available to the authors. It is code and run instructions only: no study data, tables, generated figures, manuscript, or analysis-result files are included. It operates on an MRS workbook containing seven already-quantified peak-area variables; it does not process raw NMR acquisition files. Some preprocessing and model-selection details were not fully documented in the available materials, and those limitations are recorded in the analysis outputs. 

## Code

- `analysis/run_all.py` — single entry point that runs the three steps below in sequence.
- `analysis/run_analysis.py` — selected PCA/Kernel PCA, Ward clustering, and SVC analyses.
- `analysis/run_model_screening.py` — model-family screening sweep.
- `analysis/regenerate_figures.py` — regenerates the analysis figures from locally available input files; figures are not distributed here.
- `analysis/requirements.txt` — package versions used for this analysis implementation.

The scripts remain separate modules so each stage can also be run and checked independently. The combined runner uses the default input locations documented below.

## Required input files (not included)

With the required permissions, place these files locally in the ignored `inputs/` directory:

1. MRS workbook: `inputs/mrs_peak_areas.xlsx`
2. Animal-level histology table: `inputs/histology_animal_level.csv` (at minimum, `animal_id` and `NAS` columns)

Do not upload or commit these files or any animal/sample-level data to this repository. The MRS workbook is expected to contain the `PCA` worksheet and the seven already-integrated peak-area features used by the scripts. The scripts do not process raw NMR acquisition files.

## Run

From the repository root, install dependencies and run the scripts:

```powershell
python -m pip install -r .\analysis\requirements.txt
$env:MPLBACKEND = 'Agg'
python .\analysis\run_all.py
```

Alternative local paths can be supplied through `MRS_WORKBOOK` and `HISTOLOGY_CSV` environment variables. To compare regenerated quantitative figures against reference images, optionally set `REFERENCE_FIGURES_DIR` to a local folder containing those images. These paths are not tied to a particular submission-package layout.

The combined runner executes the selected analysis, model screening, and figure generation in that order. Analysis tables are written locally to `analysis/results/`, and figures to `analysis/regenerated_figures/`. These output folders are ignored by Git; neither input data nor generated figures are distributed here.

The figure-generation step produces these quantitative figures (each as PDF and PNG):

- `02_histology_scores_boxplots`
- `03_diet_time_pca_ward_k3`; `04_diet_time_pca_ward_k4_exploratory`
- `05_nas_kpca_ward_k3`; `06_nas_kpca_ward_k4_exploratory`
- `07_nas_binary_svc_decision_regions`; `08_nas_binary_measurement_loocv_confusion`
- `fig5_experimental_svc_three_stage`; `fig6_experimental_loocv_three_stage`
- `S1_nas_three_class_svc_decision_regions`; `S2_nas_three_class_measurement_loocv_confusion`
- `S3_nas_three_class_leave_one_animal_out_confusion`; `S4_nas_binary_leave_one_animal_out_confusion`

It does not recreate the representative histology/gross-liver photo panel because those source image files are not included.

## Reuse

No open-source license is included. Public visibility does not by itself grant permission to reuse the code. Confirm authorship, institutional, ethics-committee, and journal policies before redistribution or reuse.

