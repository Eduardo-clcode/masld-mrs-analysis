from __future__ import annotations

import json
import sys
from pathlib import Path

RUNTIME_PACKAGES = Path(__file__).resolve().parents[1] / ".runtime_pkgs"
if RUNTIME_PACKAGES.exists():
    sys.path.insert(0, str(RUNTIME_PACKAGES))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from run_analysis import (
    FEATURES,
    PROJECT_ROOT,
    load_data,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def transformed_inputs(X: np.ndarray) -> dict[str, np.ndarray]:
    centered = X - X.mean(axis=0, keepdims=True)
    standard = StandardScaler().fit_transform(X)
    robust = RobustScaler().fit_transform(X)
    if np.any(X <= 0):
        raise ValueError(
            "CLR was specified in the manuscript, but the input includes zero/negative values. "
            "No pseudocount was documented; refusing to invent one."
        )
    log_x = np.log(X)
    clr = log_x - log_x.mean(axis=1, keepdims=True)
    return {
        "center_only": centered,
        "standard": standard,
        "robust": robust,
        "clr": clr,
    }


def make_embedding(family: str, X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    if family in {"center_only_pca", "standard_pca", "robust_pca", "clr_pca"}:
        return PCA(n_components=2).fit_transform(X)
    if family == "kernel_pca_rbf":
        return KernelPCA(n_components=2, kernel="rbf", gamma=gamma, eigen_solver="arpack").fit_transform(X)
    if family == "kernel_pca_poly":
        # The manuscript specifies degree=2 but not gamma or coef0; scikit-learn defaults are recorded.
        return KernelPCA(n_components=2, kernel="poly", degree=2, eigen_solver="arpack").fit_transform(X)
    if family == "kernel_pca_cosine":
        return KernelPCA(n_components=2, kernel="cosine", eigen_solver="arpack").fit_transform(X)
    raise ValueError(family)


def evaluate(
    X_cluster: np.ndarray,
    reference: np.ndarray,
    task: str,
    family: str,
    preprocessing: str,
    cluster_space: str,
    k: int,
    gamma: float | None,
) -> dict[str, object]:
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_cluster)
    sizes = {str(key): int(value) for key, value in zip(*np.unique(labels, return_counts=True))}
    return {
        "task": task,
        "model_family": family,
        "preprocessing": preprocessing,
        "cluster_space": cluster_space,
        "n_components": X_cluster.shape[1] if cluster_space == "processed_matrix" else 2,
        "k": k,
        "rbf_gamma": gamma,
        "ARI": float(adjusted_rand_score(reference, labels)),
        "NMI": float(normalized_mutual_info_score(reference, labels)),
        "silhouette": float(silhouette_score(X_cluster, labels)),
        "cluster_sizes": json.dumps(sizes, sort_keys=True),
    }


def run() -> None:
    mrs_path = PROJECT_ROOT / "Base de Datos V2" / "Base de Datos" / "CG Analise (Western y CDAA).xlsx"
    hist_path = PROJECT_ROOT / "AoH_Final_Submission_Package" / "supplementary_data" / "histology_animal_level_data.csv"
    df, _ = load_data(mrs_path, hist_path)
    X_raw = df[[f"MRS_{peak}" for peak in FEATURES]].to_numpy(dtype=float)
    inputs = transformed_inputs(X_raw)
    targets = {
        "experimental": df["experimental_class"].to_numpy(),
        "histology": df["histology_class"].to_numpy(),
    }

    candidates: list[dict[str, object]] = []
    # Linear PCA families: evaluate Ward in both the 2-D embedding and the
    # processed feature matrix because the manuscript reports both spaces.
    linear_specs = [
        ("center_only_pca", "center_only"),
        ("standard_pca", "standard"),
        ("robust_pca", "robust"),
        ("clr_pca", "clr"),
    ]
    for task, reference in targets.items():
        for family, preprocessing in linear_specs:
            base = inputs[preprocessing]
            emb = make_embedding(family, base)
            for k in (3, 4):
                for space, matrix in (("PC1_PC2", emb), ("processed_matrix", base)):
                    row = evaluate(matrix, reference, task, family, preprocessing, space, k, None)
                    candidates.append(row)

        # Kernel PCA was documented after input scaling; both embedding and
        # standardized feature-space clustering are retained as candidates.
        kernel_specs: list[tuple[str, float | None]] = [
            ("kernel_pca_rbf", 0.01),
            ("kernel_pca_rbf", 0.1),
            ("kernel_pca_poly", None),
            ("kernel_pca_cosine", None),
        ]
        standard = inputs["standard"]
        for family, gamma in kernel_specs:
            emb = make_embedding(family, standard, gamma=gamma)
            for k in (3, 4):
                for space, matrix in (("PC1_PC2", emb), ("processed_matrix", standard)):
                    row = evaluate(matrix, reference, task, family, "standard", space, k, gamma)
                    candidates.append(row)

    out = pd.DataFrame(candidates)
    # Descriptive ranks only. The paper reports family-level selections but
    # does not specify every within-family tie-breaking detail.
    rank_cols = []
    for metric in ("ARI", "NMI", "silhouette"):
        col = f"rank_{metric}"
        out[col] = out.groupby(["task", "k"])[metric].rank(method="min", ascending=False)
        rank_cols.append(col)
    out["mean_descriptive_rank"] = out[rank_cols].mean(axis=1)
    out["rank_rule_status"] = "descriptive only; exact manuscript aggregation/tie-break rule not fully specified"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.loc[out["k"] == 3].to_csv(OUTPUT_DIR / "model_screening_k3_all_candidates.csv", index=False)
    out.loc[out["k"] == 4].to_csv(OUTPUT_DIR / "model_screening_k4_exploratory.csv", index=False)

    expected = {
        "experimental": [
            ("center_only_pca", "PC1_PC2", 0.701, 0.679, 0.580),
            ("standard_pca", "PC1_PC2", 0.856, 0.792, 0.655),
            ("robust_pca", "PC1_PC2", 0.779, 0.690, 0.712),
            ("clr_pca", "PC1_PC2", 0.280, 0.493, 0.642),
            ("kernel_pca_rbf", "PC1_PC2", 0.816, 0.759, 0.669),
            ("kernel_pca_poly", "processed_matrix", 0.845, 0.739, 0.564),
            ("kernel_pca_cosine", "processed_matrix", 0.845, 0.739, 0.564),
        ],
        "histology": [
            ("center_only_pca", "PC1_PC2", 0.582, 0.549, 0.580),
            ("standard_pca", "processed_matrix", 0.456, 0.491, 0.564),
            ("robust_pca", "PC1_PC2", 0.531, 0.559, 0.712),
            ("clr_pca", "PC1_PC2", 0.175, 0.386, 0.642),
            ("kernel_pca_rbf", "PC1_PC2", 0.516, 0.542, 0.574),
            ("kernel_pca_poly", "processed_matrix", 0.456, 0.491, 0.564),
            ("kernel_pca_cosine", "PC1_PC2", 0.587, 0.618, 0.575),
        ],
    }
    comparisons = []
    for task, entries in expected.items():
        for family, space, exp_ari, exp_nmi, exp_sil in entries:
            candidates_match = out.loc[
                (out["task"] == task)
                & (out["k"] == 3)
                & (out["model_family"] == family)
                & (out["cluster_space"] == space)
            ]
            if family == "kernel_pca_rbf":
                # Gamma is explicitly reported per target in the manuscript tables.
                expected_gamma = 0.01 if task == "experimental" else 0.1
                candidates_match = candidates_match.loc[candidates_match["rbf_gamma"] == expected_gamma]
            if candidates_match.empty:
                comparisons.append({"task": task, "model_family": family, "cluster_space": space, "status": "NO CANDIDATE"})
                continue
            row = candidates_match.iloc[0]
            diffs = [abs(float(row[m]) - expected) for m, expected in (("ARI", exp_ari), ("NMI", exp_nmi), ("silhouette", exp_sil))]
            comparisons.append({
                "task": task, "model_family": family, "cluster_space": space,
                "gamma_used": row["rbf_gamma"],
                "expected_ARI": exp_ari, "observed_ARI": row["ARI"],
                "expected_NMI": exp_nmi, "observed_NMI": row["NMI"],
                "expected_silhouette": exp_sil, "observed_silhouette": row["silhouette"],
                "max_absolute_difference": max(diffs),
                "status": "MATCH (rounded to 3 decimals)" if max(diffs) < 0.0005 else "REVIEW: does not match",
            })
    pd.DataFrame(comparisons).to_csv(OUTPUT_DIR / "manuscript_screening_benchmark_comparison.csv", index=False)

    notes = {
        "purpose": "Consolidated candidate sweep; all candidates are exported rather than selecting an undocumented winner.",
        "data": {"measurements": len(df), "animals": int(df["animal_id"].nunique()), "features": [f"MRS_{v}" for v in FEATURES]},
        "documented_choices": [
            "Ward linkage, Euclidean distance, k=3 primary and k=4 exploratory.",
            "Centering, StandardScaler, RobustScaler, CLR; linear PCA and RBF/polynomial/cosine Kernel PCA.",
            "Two-component PCA/Kernel PCA embedding for component-space candidates.",
            "RBF gamma values from the manuscript tables: 0.01 for experimental screening and 0.1 for histology screening.",
        ],
        "not_fully_documented": [
            "Whether every kernel PCA family received StandardScaler before KernelPCA; implemented as StandardScaler because the selected cosine model is explicitly described that way.",
            "Exact meaning of 'processed matrix' in the manuscript tables; here it is operationalized as the 7-column preprocessed input matrix and both it and PC1-PC2 are exported.",
            "KernelPCA degree-2 gamma/coef0 are not given; scikit-learn defaults are used and recorded by the source code.",
            "Zero replacement for CLR is not given; no pseudocount is added. The run stops if any input peak is zero or negative.",
            "The family-level ranking and tie-breaking details are not fully stated; rank columns are descriptive because the available methods do not fully specify a family-level selection rule.",
        ],
    }
    (OUTPUT_DIR / "model_screening_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"\nFull candidate CSVs written under: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()

