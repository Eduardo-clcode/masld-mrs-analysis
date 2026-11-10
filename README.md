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

With the required permissions, provide these files locally at the following paths relative to the repository root:

1. MRS workbook: 
2. Animal-level histology table

## Run

From the repository root, install dependencies and run the scripts:

```powershell
python -m pip install -r .\analysis\requirements.txt
$env:MPLBACKEND = 'Agg'
python .\analysis\run_all.py
```

It does not recreate the representative histology/gross-liver photo panel because those source image files are not included.

