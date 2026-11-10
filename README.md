# Ex-vivo hepatic FAME ¹H-NMR analysis code

This public repository contains a reconstructed set of analysis scripts for ex-vivo hepatic fatty-acid methyl ester (FAME) ¹H-NMR measurements in a murine MASLD study. It contains code and run instructions only: no study data, tables, generated figures, manuscript, or analysis-result files are included.

The scripts are a transparent reconstruction from the files available to the authors; they are not the original laboratory scripts recovered from the acquisition computer. Some original preprocessing and model-selection details were not fully documented, so outputs should be interpreted with that limitation in mind.

## Code

- `AoH_Reconstructed_Analysis/run_reconstructed_analysis.py` — selected PCA/Kernel PCA, Ward clustering, and SVC analyses.
- `AoH_Reconstructed_Analysis/run_model_screening.py` — model-family screening sweep.
- `AoH_Reconstructed_Analysis/regenerate_figures.py` — regenerates the analysis figures from locally available input files; figures are not distributed here.
- `AoH_Reconstructed_Analysis/requirements.txt` — package versions used for the reconstruction.

## Required input files (not included)

With the required permissions, provide these files locally at the following paths relative to the repository root:

1. MRS workbook: `Base de Datos V2/Base de Datos/CG Analise (Western y CDAA).xlsx`
2. Animal-level histology table: `AoH_Final_Submission_Package/supplementary_data/histology_animal_level_data.csv`

Do not upload or commit these files or any animal/sample-level data to this repository. The MRS workbook is expected to contain the `PCA` worksheet and the seven peak-area features used by the scripts.

## Run

From the repository root, install dependencies and run the scripts:

```powershell
python -m pip install -r .\AoH_Reconstructed_Analysis\requirements.txt
python .\AoH_Reconstructed_Analysis\run_reconstructed_analysis.py
python .\AoH_Reconstructed_Analysis\run_model_screening.py
$env:MPLBACKEND = 'Agg'
python .\AoH_Reconstructed_Analysis\regenerate_figures.py
```

Analysis tables are written to the local `AoH_Reconstructed_Analysis/results/` folder, and regenerated figures to `AoH_Reconstructed_Analysis/regenerated_figures/`. Both locations are ignored by Git to reduce the risk of publishing outputs or data accidentally.

## Reuse

No open-source license is included. Public visibility does not by itself grant permission to reuse the code. Confirm authorship, institutional, ethics-committee, and journal policies before redistribution or reuse.

