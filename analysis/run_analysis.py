from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RUNTIME_PACKAGES = Path(__file__).resolve().parents[1] / ".runtime_pkgs"
if RUNTIME_PACKAGES.exists():
    sys.path.insert(0, str(RUNTIME_PACKAGES))

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURES = ["0.9", "1.2", "1.6", "2.0", "2.2", "2.8", "5.3"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_CLASSES = ["Control", "Steatosis-like", "MASH-like"]
HISTOLOGY_CLASSES = ["No-MASH", "MASH-compatible"]
EXPECTED = {
    "measurements": 77,
    "animals": 42,
    "experimental_counts": {"Control": 11, "Steatosis-like": 12, "MASH-like": 54},
    "histology_counts": {"No-MASH": 25, "Indeterminate": 8, "MASH-compatible": 44},
}


def normalize_id(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def parse_experimental_group(source_group: str) -> tuple[str, str | None]:
    group = source_group.strip().lower()
    if "control" in group or "controle" in group or "controles" in group:
        return "Control", None
    match = re.search(r"(western|cdaa)\s*(\d+)\s*w", group)
    if not match:
        raise ValueError(f"Unrecognized experimental group block: {source_group!r}")
    diet, weeks_text = match.groups()
    weeks = int(weeks_text)
    if diet == "western" and weeks == 4:
        return "Steatosis-like", f"Western_{weeks}w"
    return "MASH-like", f"{diet.upper()}_{weeks}w"


def read_mrs_workbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="PCA", header=None)
    records: list[dict[str, object]] = []
    source_group: str | None = None
    for _, row in raw.iloc[2:].iterrows():
        if pd.notna(row.iloc[0]):
            source_group = str(row.iloc[0]).strip()
        if source_group is None or pd.isna(row.iloc[1]):
            continue
        measurement_id = normalize_id(row.iloc[1])
        if measurement_id.lower().startswith("id"):
            continue
        values = pd.to_numeric(row.iloc[15:22], errors="coerce").to_numpy(dtype=float)
        stage, specific_group = parse_experimental_group(source_group)
        record: dict[str, object] = {
            "measurement_id": measurement_id,
            "animal_id": re.sub(r"[AB]$", "", measurement_id),
            "source_group": source_group,
            "specific_group": specific_group or "Control",
            "experimental_class": stage,
        }
        record.update({f"MRS_{peak}": value for peak, value in zip(FEATURES, values)})
        record["all_peaks_missing"] = bool(np.isnan(values).all())
        records.append(record)

    df = pd.DataFrame(records)
    # Preserve the excluded row in the audit, but do not impute unavailable peak areas.
    df["included"] = ~df[[f"MRS_{peak}" for peak in FEATURES]].isna().any(axis=1)
    return df


def load_data(mrs_path: Path, histology_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_mrs = read_mrs_workbook(mrs_path)
    histology = pd.read_csv(histology_path)
    histology["animal_id"] = histology["animal_id"].map(normalize_id)
    required_hist = {"animal_id", "NAS"}
    if not required_hist.issubset(histology.columns):
        raise ValueError(f"Histology file must contain columns {sorted(required_hist)}")
    if histology["animal_id"].duplicated().any():
        raise ValueError("Histology source has duplicate animal IDs; resolve before analysis")

    df = all_mrs.loc[all_mrs["included"]].copy()
    df = df.merge(histology, on="animal_id", how="left", validate="many_to_one")
    if df["NAS"].isna().any():
        missing = df.loc[df["NAS"].isna(), "animal_id"].unique().tolist()
        raise ValueError(f"MRS animals missing histology: {missing}")
    nas = pd.to_numeric(df["NAS"], errors="raise")
    df["histology_class"] = np.select(
        [nas < 3, nas.between(3, 4, inclusive="both"), nas > 4],
        ["No-MASH", "Indeterminate", "MASH-compatible"],
        default="MISSING",
    )
    if (df["histology_class"] == "MISSING").any():
        raise ValueError("One or more NAS values did not map to a prespecified category")
    return df, all_mrs


def svc_pipeline(representation: str):
    if representation == "linear_pca":
        reducer = PCA(n_components=2)
    elif representation == "cosine_kernel_pca":
        reducer = KernelPCA(n_components=2, kernel="cosine", eigen_solver="arpack")
    else:
        raise ValueError(representation)
    return make_pipeline(
        StandardScaler(),
        reducer,
        SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced"),
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def cross_validated_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    representation: str,
) -> pd.DataFrame:
    splitter = LeaveOneOut() if groups is None else LeaveOneGroupOut()
    folds = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    pred = np.empty(len(y), dtype=object)
    fold_id = np.empty(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        model = svc_pipeline(representation)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
        fold_id[test_idx] = fold
    return pd.DataFrame({"true_label": y, "predicted_label": pred, "fold": fold_id})


def evaluate_cv(
    df: pd.DataFrame,
    task_name: str,
    label_column: str,
    labels: list[str],
    representation: str,
    validation: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    subset = df.loc[df[label_column].isin(labels)].reset_index(drop=True)
    X = subset[[f"MRS_{peak}" for peak in FEATURES]]
    y = subset[label_column].to_numpy()
    groups = subset["animal_id"].to_numpy() if validation == "leave_one_animal_out" else None
    pred = cross_validated_predictions(X, y, groups, representation)
    pred.insert(0, "measurement_id", subset["measurement_id"])
    pred.insert(1, "animal_id", subset["animal_id"])
    pred["task"] = task_name
    pred["validation"] = validation
    conf = confusion_matrix(y, pred["predicted_label"], labels=labels)
    conf_df = pd.DataFrame(conf, index=labels, columns=labels).rename_axis("true_label")
    conf_df.insert(0, "task", task_name)
    conf_df.insert(1, "validation", validation)
    summary: dict[str, object] = {
        "task": task_name,
        "validation": validation,
        "representation": representation,
        "n_measurements": len(subset),
        "n_animals": int(subset["animal_id"].nunique()),
        **metric_dict(y, pred["predicted_label"].to_numpy(), labels),
        "confusion_matrix": conf.tolist(),
        "class_order": labels,
    }
    return summary, pred, conf_df


def cluster_summary(
    embedding: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    task: str,
    representation: str,
) -> dict[str, object]:
    return {
        "task": task,
        "representation": representation,
        "n_measurements": int(len(labels)),
        "n_clusters": int(len(np.unique(labels))),
        "ARI": float(adjusted_rand_score(reference, labels)),
        "NMI": float(normalized_mutual_info_score(reference, labels)),
        "silhouette": float(silhouette_score(embedding, labels)),
        "cluster_sizes": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, mrs_all = load_data(args.mrs_workbook, args.histology_csv)
    features = [f"MRS_{peak}" for peak in FEATURES]
    count_exp = df["experimental_class"].value_counts().reindex(EXPERIMENTAL_CLASSES, fill_value=0).to_dict()
    count_hist = df["histology_class"].value_counts().reindex(
        ["No-MASH", "Indeterminate", "MASH-compatible"], fill_value=0
    ).to_dict()
    checks = {
        "n_measurements": int(len(df)),
        "n_animals": int(df["animal_id"].nunique()),
        "experimental_counts": {k: int(v) for k, v in count_exp.items()},
        "histology_counts": {k: int(v) for k, v in count_hist.items()},
        "excluded_measurements": mrs_all.loc[~mrs_all["included"], ["measurement_id", "source_group"]].to_dict("records"),
    }
    if checks["n_measurements"] != EXPECTED["measurements"]:
        raise ValueError(f"Expected {EXPECTED['measurements']} measurements, found {checks['n_measurements']}")
    if checks["n_animals"] != EXPECTED["animals"]:
        raise ValueError(f"Expected {EXPECTED['animals']} animals, found {checks['n_animals']}")
    if checks["experimental_counts"] != EXPECTED["experimental_counts"]:
        raise ValueError(f"Experimental counts differ from manuscript: {checks['experimental_counts']}")
    if checks["histology_counts"] != EXPECTED["histology_counts"]:
        raise ValueError(f"NAS category counts differ from manuscript: {checks['histology_counts']}")

    df.to_csv(args.output_dir / "curated_measurement_data.csv", index=False)
    mrs_all.loc[~mrs_all["included"]].to_csv(args.output_dir / "excluded_measurements.csv", index=False)

    X = df[features].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit_transform(scaled)
    kpca = KernelPCA(n_components=2, kernel="cosine", eigen_solver="arpack").fit_transform(scaled)
    exp_targets = df["experimental_class"].to_numpy()
    nas_targets = df["histology_class"].to_numpy()
    ward_exp = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(pca)
    ward_nas = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(kpca)
    cluster_rows = [
        cluster_summary(pca, ward_exp, exp_targets, "experimental", "StandardScaler + linear PCA"),
        cluster_summary(kpca, ward_nas, nas_targets, "histology", "StandardScaler + cosine Kernel PCA"),
    ]
    (args.output_dir / "selected_clustering_metrics.json").write_text(
        json.dumps(cluster_rows, indent=2), encoding="utf-8"
    )

    exp_summary, exp_pred, exp_conf = evaluate_cv(
        df, "experimental_three_class", "experimental_class", EXPERIMENTAL_CLASSES,
        "linear_pca", "measurement_level_LOOCV",
    )
    binary_summary = []
    binary_predictions = []
    binary_confusions = []
    for validation in ("measurement_level_LOOCV", "leave_one_animal_out"):
        summary, pred, conf = evaluate_cv(
            df, "histology_binary", "histology_class", HISTOLOGY_CLASSES,
            "cosine_kernel_pca", validation,
        )
        binary_summary.append(summary)
        binary_predictions.append(pred)
        binary_confusions.append(conf)

    pd.concat([exp_pred, *binary_predictions], ignore_index=True).to_csv(
        args.output_dir / "cross_validation_predictions.csv", index=False
    )
    pd.concat([exp_conf, *binary_confusions]).to_csv(
        args.output_dir / "cross_validation_confusion_matrices.csv"
    )
    summaries = [exp_summary, *binary_summary]
    (args.output_dir / "cross_validation_metrics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    # Descriptive post-hoc mapping: fit the selected embedding on all 77 records,
    # train the SVC only on definite NAS endpoints, and label NAS 3-4 separately.
    full_embedding_model = make_pipeline(
        StandardScaler(), KernelPCA(n_components=2, kernel="cosine", eigen_solver="arpack")
    )
    all_embedding = full_embedding_model.fit_transform(df[features])
    definite = df["histology_class"].isin(HISTOLOGY_CLASSES).to_numpy()
    posthoc_model = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")
    posthoc_model.fit(all_embedding[definite], df.loc[definite, "histology_class"])
    indeterminate = ~definite
    posthoc = df.loc[indeterminate, ["measurement_id", "animal_id", "NAS", "histology_class"]].copy()
    posthoc["posthoc_prediction"] = posthoc_model.predict(all_embedding[indeterminate])
    posthoc.to_csv(args.output_dir / "indeterminate_posthoc_projection.csv", index=False)

    package_versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "NOT INSTALLED / version unavailable"
    report = {
        "status": "CONSOLIDATED_ANALYSIS; verify metrics and unspecified choices before publication",
        "input_files": {"mrs_workbook": str(args.mrs_workbook), "histology_csv": str(args.histology_csv)},
        "checks": checks,
        "selected_clustering": cluster_rows,
        "cross_validation": summaries,
        "indeterminate_posthoc_counts": posthoc["posthoc_prediction"].value_counts().to_dict(),
        "software_versions_this_run": package_versions,
        "limitations": [
            "The workflow consolidates analysis steps from fragmented project materials; it is not a complete archive of the laboratory's original scripts.",
            "The submitted manuscript does not fully document every model-screening parameter or tie-breaking rule.",
            "Software versions here describe this implementation run and should not be interpreted as the original laboratory environment.",
            "The full-data post-hoc embedding is descriptive; cross-validation metrics use preprocessing fitted within each fold.",
        ],
    }
    (args.output_dir / "analysis_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the consolidated analyses specified in the study manuscript.")
    parser.add_argument(
        "--mrs-workbook", type=Path,
        default=PROJECT_ROOT / "Base de Datos V2" / "Base de Datos" / "CG Analise (Western y CDAA).xlsx",
    )
    parser.add_argument(
        "--histology-csv", type=Path,
        default=PROJECT_ROOT / "AoH_Final_Submission_Package" / "supplementary_data" / "histology_animal_level_data.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

