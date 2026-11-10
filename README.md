# Ex-vivo hepatic FAME ¹H-NMR analysis code

This public repository contains a reconstructed set of analysis scripts for ex-vivo hepatic fatty-acid methyl ester (FAME) ¹H-NMR measurements in a murine MASLD study. It contains code and run instructions only: no study data, tables, generated figures, manuscript, or analysis-result files are included.

The scripts are a transparent reconstruction from the files available to the authors; they are not the original laboratory scripts recovered from the acquisition computer. Some original preprocessing and model-selection details were not fully documented, so outputs should be interpreted with that limitation in mind.

## Code

- `analysis/run_all.py` — single entry point that runs the three steps below in sequence.
- `analysis/run_analysis.py` — selected PCA/Kernel PCA, Ward clustering, and SVC analyses.
- `analysis/run_model_screening.py` — model-family screening sweep.
- `analysis/regenerate_figures.py` — regenerates the analysis figures from locally available input files; figures are not distributed here.
- `analysis/requirements.txt` — package versions used for this analysis implementation.

The scripts remain separate modules so each stage can also be run and checked independently. The combined runner uses the default input locations documented below.

## Required input files (not included)

With the required permissions, provide these files locally at the following paths relative to the repository root:

1. MRS workbook: `Base de Datos V2/Base de Datos/CG Analise (Western y CDAA).xlsx`
2. Animal-level histology table: `AoH_Final_Submission_Package/supplementary_data/histology_animal_level_data.csv`

Do not upload or commit these files or any animal/sample-level data to this repository. The MRS workbook is expected to contain the `PCA` worksheet and the seven peak-area features used by the scripts.

## Run

From the repository root, install dependencies and run the scripts:

```powershell
python -m pip install -r .\analysis\requirements.txt
$env:MPLBACKEND = 'Agg'
python .\analysis\run_all.py
```

The combined runner executes the selected analysis, model screening, and figure regeneration in that order. Analysis tables are written locally to `analysis/results/`, and figures to `analysis/regenerated_figures/`. These output folders are ignored by Git; neither input data nor generated figures are distributed here.

The figure-generation step produces the quantitative PCA/Kernel PCA, clustering, SVC decision-region, and confusion-matrix figures. It does not recreate the representative histology/gross-liver photo panel because those source image files are not included.

## Reuse

No open-source license is included. Public visibility does not by itself grant permission to reuse the code. Confirm authorship, institutional, ethics-committee, and journal policies before redistribution or reuse.

