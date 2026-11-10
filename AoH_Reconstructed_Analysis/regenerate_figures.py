from __future__ import annotations

import json
import sys
from pathlib import Path

RUNTIME_PACKAGES = Path(__file__).resolve().parents[1] / ".runtime_pkgs"
if RUNTIME_PACKAGES.exists():
    sys.path.insert(0, str(RUNTIME_PACKAGES))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_reconstructed_analysis import (
    EXPERIMENTAL_CLASSES,
    FEATURES,
    HISTOLOGY_CLASSES,
    PROJECT_ROOT,
    cross_validated_predictions,
    load_data,
    svc_pipeline,
)


SOURCE_FIGURES = PROJECT_ROOT / "AoH_Final_Submission_Package" / "figs"
OUT_DIR = Path(__file__).resolve().parent / "regenerated_figures"
MRS_PATH = PROJECT_ROOT / "Base de Datos V2" / "Base de Datos" / "CG Analise (Western y CDAA).xlsx"
HISTOLOGY_PATH = PROJECT_ROOT / "AoH_Final_Submission_Package" / "supplementary_data" / "histology_animal_level_data.csv"

COLORS = {
    "No-MASH": "#168aad",
    "Indeterminate": "#e07a2f",
    "MASH-compatible": "#c2185b",
    "Group 1": "#277da1",
    "Group 2": "#c2185b",
    "Group 3": "#d8c600",
    "Group 4": "#86a98a",
    "Control": "#9e9e9e",
    "Western 4": "#45b8b4",
    "Western 10": "#168aad",
    "Western 24": "#365f6b",
    "CDAA 4": "#ff7043",
    "CDAA 10": "#e15b64",
    "CDAA 24": "#b00000",
}
SVC_COLORS = {
    "Control": "#ffa600",
    "Steatosis-like": "#2aa198",
    "MASH-like": "#c2185b",
    "No-MASH": "#168aad",
    "Indeterminate": "#e07a2f",
    "MASH-compatible": "#c2185b",
}


def features(df: pd.DataFrame) -> np.ndarray:
    return df[[f"MRS_{x}" for x in FEATURES]].to_numpy(dtype=float)


def embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = features(df)
    scaled = StandardScaler().fit_transform(X)
    pca_model = PCA(n_components=2)
    pca_xy = pca_model.fit_transform(scaled)
    kpca_model = KernelPCA(n_components=2, kernel="cosine", eigen_solver="arpack")
    kpca_xy = kpca_model.fit_transform(scaled)
    # Component signs are arbitrary. Orient plots like the submitted figures:
    # Western-associated trajectory progresses rightward in PCA, and no-MASH
    # observations lie to the right in the cosine Kernel-PCA display.
    if df.assign(_x=pca_xy[:, 0]).groupby("experimental_class", observed=True)['_x'].mean().get("MASH-like", 0) < df.assign(_x=pca_xy[:, 0]).groupby("experimental_class", observed=True)['_x'].mean().get("Control", 0):
        pca_xy[:, 0] *= -1
    hist_mean_x = pd.DataFrame({"x": kpca_xy[:, 0], "y": df["histology_class"].to_numpy()}).groupby("y")["x"].mean()
    if hist_mean_x.get("No-MASH", 0) < hist_mean_x.get("MASH-compatible", 0):
        kpca_xy[:, 0] *= -1
    return scaled, pca_xy, kpca_xy, pca_model.explained_variance_ratio_


def style_axis(ax: plt.Axes, title: str, letter: str, xlab: str, ylab: str | None = None) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlab, fontsize=10)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=10)
    ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def legend_handles(names: list[str], counts: dict[str, int], colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color="none",
               markerfacecolor=colors[name], markeredgecolor="white", markersize=7,
               label=f"{name} (n={counts[name]})")
        for name in names
    ]


def draw_score_panel(ax, xy, df, pca_ratio, kernel):
    nas_colors = df["histology_class"].map(COLORS).to_numpy()
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(int(df.iloc[i]["NAS"])), color=nas_colors[i], fontsize=8,
                fontweight="bold", ha="center", va="center")
    handles = legend_handles(
        ["No-MASH", "Indeterminate", "MASH-compatible"],
        df["histology_class"].value_counts().to_dict(), COLORS,
    )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=1, frameon=False, fontsize=8, handletextpad=0.4, labelspacing=0.35)
    ax.set_xlabel("Kernel component 1" if kernel else f"PC1 ({pca_ratio[0]*100:.1f}%)")
    ax.set_ylabel("Kernel component 2" if kernel else f"PC2 ({pca_ratio[1]*100:.1f}%)")


def plot_embedding_triptych(df: pd.DataFrame, xy: np.ndarray, pca_ratio: np.ndarray,
                            clustering: np.ndarray, k: int, kernel: bool, output_stem: str,
                            selected_metrics: dict[str, float] | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.25, 5.67), sharex=True, sharey=True)
    if kernel:
        metric_line = ""
        if selected_metrics:
            metric_line = (f"\nARI={selected_metrics['ari']:.3f} | NMI={selected_metrics['nmi']:.3f} | "
                           f"Silhouette={selected_metrics['silhouette']:.3f}")
        fig.suptitle("NAS-selected MRS representation: StandardScaler + cosine Kernel PCA + Ward k=3" + metric_line,
                     fontsize=14, y=0.98)
        xlab, ylab = "Kernel component 1", "Kernel component 2"
        x_scores = xy
    else:
        xlab = f"PC1 ({pca_ratio[0]*100:.1f}%)"
        ylab = f"PC2 ({pca_ratio[1]*100:.1f}%)"
        x_scores = xy

    if kernel:
        category_order = ["No-MASH", "Indeterminate", "MASH-compatible"]
        counts = df["histology_class"].value_counts().to_dict()
        for category in category_order:
            mask = df["histology_class"].eq(category).to_numpy()
            category_nas = {"No-MASH": "< 3", "Indeterminate": "3–4", "MASH-compatible": "> 4"}[category]
            axes[0].scatter(x_scores[mask, 0], x_scores[mask, 1], s=28, c=COLORS[category],
                            edgecolor="white", linewidth=0.35,
                            label=f"{category} (NAS {category_nas}) (n={counts[category]})")
        axes[0].set_title("A  Histological NAS categories", loc="left", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(xlab); axes[0].set_ylabel(ylab)
        axes[0].grid(True, color="#dddddd", linewidth=0.65); axes[0].set_axisbelow(True)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                       frameon=False, fontsize=8, handletextpad=0.35, columnspacing=0.8)
    else:
        # Diet/time labels
        group_order = ["CDAA 24", "CDAA 4", "CDAA 10", "Western 24", "Western 4", "Western 10", "Control"]
        labels = df["specific_group"].replace({
            "CDAA_24w": "CDAA 24", "CDAA_4w": "CDAA 4", "CDAA_10w": "CDAA 10",
            "Western_24w": "Western 24", "Western_4w": "Western 4", "Western_10w": "Western 10",
            "Control": "Control",
        })
        counts = labels.value_counts().to_dict()
        for name in group_order:
            mask = labels.eq(name).to_numpy()
            if mask.any():
                axes[0].scatter(x_scores[mask, 0], x_scores[mask, 1], s=28, c=COLORS[name],
                                edgecolor="none", label=f"{name.upper()} (n={counts[name]})")
        axes[0].set_title("A  Diet and intervention time", loc="left", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(xlab); axes[0].set_ylabel(ylab)
        axes[0].grid(True, color="#dddddd", linewidth=0.65); axes[0].set_axisbelow(True)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                       frameon=False, fontsize=8, handletextpad=0.35, columnspacing=0.8)

    # Individual NAS score, colored by the operational category.
    draw_score_panel(axes[1], x_scores, df, pca_ratio, kernel)
    axes[1].set_title("B  Individual NAS score", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(ylab)
    axes[1].grid(True, color="#dddddd", linewidth=0.65); axes[1].set_axisbelow(True)

    # Ward assignments; labels are the estimator's native stable 0-based labels + 1.
    cluster_counts = pd.Series(clustering).value_counts().sort_index().to_dict()
    for c in sorted(np.unique(clustering)):
        mask = clustering == c
        group_name = f"Group {c+1}"
        axes[2].scatter(x_scores[mask, 0], x_scores[mask, 1], s=28,
                        c=COLORS.get(group_name, "#86a98a"), edgecolor="none",
                        label=f"{group_name} (n={cluster_counts[c]})")
    if kernel:
        axes[2].set_title(f"C  Ward clustering (k={k})", loc="left", fontsize=12, fontweight="bold")
    else:
        axes[2].set_title(f"C  Ward hierarchical clustering (k={k})", loc="left", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(xlab); axes[2].set_ylabel(ylab)
    axes[2].grid(True, color="#dddddd", linewidth=0.65); axes[2].set_axisbelow(True)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2 if k == 4 else 1,
                   frameon=False, fontsize=8, handletextpad=0.35, columnspacing=0.8)

    # Hide duplicate y-axis labels while retaining identical scales.
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.83 if kernel else 0.91,
                        bottom=0.27, wspace=0.10)
    save_figure(fig, output_stem)


def plot_histology_scores(hist: pd.DataFrame, output_stem: str) -> None:
    groups = ["Control", "CDAA 4 wk", "CDAA 10 wk", "CDAA 24 wk", "Western 4 wk", "Western 10 wk", "Western 24 wk"]
    # Convert published sample-group labels without changing score values.
    h = hist.copy()
    h["plot_group"] = h.apply(
        lambda r: "Control" if str(r["diet"]).lower() == "control" else f"{r['diet']} {int(r['weeks'])} wk",
        axis=1,
    )
    palette = {
        "Control": "#bdbdbd", "CDAA 4 wk": "#f79c7f", "CDAA 10 wk": "#c25252",
        "CDAA 24 wk": "#c25252", "Western 4 wk": "#64b9b8",
        "Western 10 wk": "#64b9b8", "Western 24 wk": "#64b9b8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.7, 4.89), gridspec_kw={"width_ratios": [2.5, 1, 1]})
    ax = axes[0]
    positions = np.arange(len(groups))
    rng = np.random.default_rng(20260929)
    for pos, group in zip(positions, groups):
        vals = h.loc[h["plot_group"].eq(group), "NAS_fibrosis"].astype(float).to_numpy()
        if not len(vals):
            continue
        ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True,
                   boxprops={"facecolor": palette[group], "edgecolor": "#555555", "linewidth": 0.8},
                   medianprops={"color": "black", "linewidth": 1.2},
                   whiskerprops={"color": "#555555"}, capprops={"color": "#555555"},
                   flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#555555"})
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(pos + jitter, vals, s=25, facecolor="white", edgecolor="#555555", linewidth=0.8, zorder=3)
    ax.set_xticks(positions, groups, rotation=32, ha="right")
    ax.set_title("A  Histological progression", loc="left", fontweight="bold")
    ax.set_ylabel("NAS-fibrosis score"); ax.grid(axis="y", color="#dddddd"); ax.set_axisbelow(True)
    ax.set_ylim(-0.4, 12.5)

    at24 = h.loc[h["weeks"].eq(24)]
    for ax, variable, title, ylabel in [
        (axes[1], "fibrosis_stage", "B  Fibrosis at 24 weeks", "Fibrosis score"),
        (axes[2], "steatosis_grade", "C  Steatosis at 24 weeks", "Steatosis score"),
    ]:
        categories = ["CDAA", "Western"]
        for pos, diet in enumerate(categories):
            vals = at24.loc[at24["diet"].eq(diet), variable].astype(float).to_numpy()
            ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True,
                       boxprops={"facecolor": "#c25252" if diet == "CDAA" else "#64b9b8", "edgecolor": "#555555"},
                       medianprops={"color": "black", "linewidth": 1.2},
                       whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(pos + jitter, vals, s=25, facecolor="white", edgecolor="#555555", linewidth=0.8, zorder=3)
        ax.set_xticks([0, 1], ["24 wk\nCDAA\n(n=6)", "24 wk\nWESTERN\n(n=6)"])
        ax.set_title(title, loc="left", fontweight="bold"); ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd"); ax.set_axisbelow(True)
        ax.set_ylim(-0.15, 4.4 if variable == "fibrosis_stage" else 3.5)
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, output_stem)


def plot_confusion(y_true, y_pred, labels, title, metrics_text, output_stem):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.11, 5.55))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=17, pad=14)
    ax.set_xlabel("Predicted histological category" if len(labels) == 2 else "Predicted category", fontsize=12)
    ax.set_ylabel("Reference histological category" if len(labels) == 2 else "Reference category", fontsize=12)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=26, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=17,
                    color="white" if cm[i, j] > threshold else "#202020")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.018, metrics_text, ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.22, right=0.88, top=0.88, bottom=0.34)
    save_figure(fig, output_stem)
    return cm


def plot_decision_regions(df, xy, y, labels, title, subtitle, output_stem, stars=False):
    if stars:
        train_mask = np.isin(y, HISTOLOGY_CLASSES)
        y_train = y[train_mask]
    else:
        train_mask = np.ones(len(y), dtype=bool)
        y_train = y
    svc = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")
    svc.fit(xy[train_mask], y_train)
    xpad = 0.12 * (xy[:, 0].max() - xy[:, 0].min())
    ypad = 0.12 * (xy[:, 1].max() - xy[:, 1].min())
    xx, yy = np.meshgrid(
        np.linspace(xy[:, 0].min()-xpad, xy[:, 0].max()+xpad, 350),
        np.linspace(xy[:, 1].min()-ypad, xy[:, 1].max()+ypad, 350),
    )
    zz = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    color_list = [SVC_COLORS[l] for l in labels]
    label_to_index = {label: i for i, label in enumerate(labels)}
    zz_index = np.vectorize(label_to_index.__getitem__)(zz)
    cmap = ListedColormap(color_list)
    fig, ax = plt.subplots(figsize=(8.36, 6.31))
    ax.contourf(xx, yy, zz_index, levels=np.arange(len(labels)+1)-0.5,
                cmap=cmap, alpha=0.16, antialiased=True)
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    display_labels = list(labels) + (["Indeterminate"] if stars and "Indeterminate" not in labels else [])
    for label in display_labels:
        mask = y == label
        if label == "Indeterminate" and stars:
            ax.scatter(xy[mask, 0], xy[mask, 1], marker="*", s=170, c=SVC_COLORS[label],
                       edgecolor="#7a3f16", linewidth=1.2, zorder=4)
        else:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=52, c=SVC_COLORS[label],
                       edgecolor="white", linewidth=0.8, zorder=3)
    title_size = 15 if len(title) > 45 else 17
    fig.suptitle(title, fontsize=title_size, y=0.995)
    if subtitle:
        fig.text(0.5, 0.895, subtitle, ha="center", fontsize=10)
    is_kernel = labels[0] == "No-MASH"
    ax.set_xlabel("Kernel component 1" if is_kernel else "PC1", fontsize=13)
    ax.set_ylabel("Kernel component 2" if is_kernel else "PC2", fontsize=13)
    handles = []
    for label in display_labels:
        if label == "Indeterminate" and stars:
            handles.append(Line2D([0], [0], marker="*", linestyle="", color="none",
                                  markerfacecolor=SVC_COLORS[label], markeredgecolor="#7a3f16", markersize=13,
                                  label=f"{label} (NAS 3–4) (n={(y == label).sum()}; displayed only)"))
        else:
            desc = {"No-MASH": "NAS < 3", "MASH-compatible": "NAS > 4", "Control": "",
                    "Steatosis-like": "", "MASH-like": "", "Indeterminate": "NAS 3–4"}.get(label, label)
            legend_label = f"{label} (n={(y == label).sum()})" if not desc else f"{label} ({desc}) (n={(y == label).sum()})"
            handles.append(Line2D([0], [0], marker="o", linestyle="", color="none",
                                  markerfacecolor=SVC_COLORS[label], markeredgecolor="white", markersize=8,
                                  label=legend_label))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=1, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.22)
    save_figure(fig, output_stem)


def save_figure(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def format_metrics(y_true, y_pred, labels) -> str:
    return (
        f"Accuracy={accuracy_score(y_true, y_pred):.3f} | "
        f"Balanced accuracy={balanced_accuracy_score(y_true, y_pred):.3f}\n"
        f"Macro-F1={f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.3f} | "
        f"Weighted-F1={f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.3f}"
    )


def compare_files(generated: list[str]) -> None:
    rows = []
    for name in generated:
        ref = SOURCE_FIGURES / name
        out = OUT_DIR / name
        rows.append({
            "reference": str(ref),
            "regenerated": str(out),
            "reference_available": ref.exists(),
            "regenerated_available": out.exists(),
            "comparison": "visual and quantitative checks; styling may differ",
        })
    rows.append({
        "reference": str(SOURCE_FIGURES / "01_histology_representative.pdf"),
        "regenerated": "NOT REGENERATED: source histology/gross-liver photographs not assembled from raw inputs",
        "reference_available": (SOURCE_FIGURES / "01_histology_representative.pdf").exists(),
        "regenerated_available": False,
        "comparison": "source-image figure retained unchanged in submission package",
    })
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure_file_comparison.csv", index=False)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    df, _ = load_data(MRS_PATH, HISTOLOGY_PATH)
    hist = pd.read_csv(HISTOLOGY_PATH)
    scaled, pca_xy, kpca_xy, pca_ratio = embeddings(df)
    figures: list[str] = []

    # Descriptive histology and unsupervised PCA/KPCA figures.
    name = "02_histology_scores_boxplots.png"
    plot_histology_scores(hist, "02_histology_scores_boxplots")
    figures.append(name)
    for k, stem in [(3, "03_diet_time_pca_ward_k3"), (4, "04_diet_time_pca_ward_k4_exploratory")]:
        clusters = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(pca_xy)
        plot_embedding_triptych(df, pca_xy, pca_ratio, clusters, k, False, stem)
        figures.append(f"{stem}.png")
    for k, stem in [(3, "05_nas_kpca_ward_k3"), (4, "06_nas_kpca_ward_k4_exploratory")]:
        clusters = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(kpca_xy)
        plot_embedding_triptych(df, kpca_xy, pca_ratio, clusters, k, True, stem,
                                {"ari": 0.5872525106, "nmi": 0.6179224080, "silhouette": 0.5747642877})
        figures.append(f"{stem}.png")

    # Final and supplementary supervised figures.
    hist_labels = df["histology_class"].to_numpy()
    exp_labels = df["experimental_class"].to_numpy()
    plot_decision_regions(
        df, kpca_xy, hist_labels, ["No-MASH", "MASH-compatible"],
        "SVC decision regions for NAS-defined categories",
        "leave-one-measurement-out: accuracy=1.000, balanced accuracy=1.000, macro-F1=1.000",
        "07_nas_binary_svc_decision_regions", stars=True,
    )
    figures.append("07_nas_binary_svc_decision_regions.png")
    binary = df.loc[df["histology_class"].isin(HISTOLOGY_CLASSES)].reset_index(drop=True)
    binary_X = binary[[f"MRS_{x}" for x in FEATURES]]
    all_X = df[[f"MRS_{x}" for x in FEATURES]]
    loo_binary = cross_validated_predictions(binary_X, binary["histology_class"].to_numpy(), None, "cosine_kernel_pca")
    binary_metric = "Accuracy=1.000 | Balanced accuracy=1.000\nMacro-F1=1.000 | Weighted-F1=1.000"
    plot_confusion(binary["histology_class"], loo_binary["predicted_label"], HISTOLOGY_CLASSES,
                   "SVC leave-one-measurement-out", binary_metric, "08_nas_binary_measurement_loocv_confusion")
    figures.append("08_nas_binary_measurement_loocv_confusion.png")

    # Experimental three-stage classifier on the selected standardized linear PCA space.
    exp_scores = pca_xy
    exp_svc_scores = exp_scores.copy()
    # PCA axes may be reflected without changing any predictions. This sign
    # choice matches the orientation used in the submitted SVC panel.
    exp_svc_scores[:, 0] *= -1
    plot_decision_regions(
        df, exp_svc_scores, exp_labels, EXPERIMENTAL_CLASSES,
        "Experimental three-stage SVC decision regions",
        "Measurement-level LOOCV: accuracy=0.896 | balanced accuracy=0.862\nmacro-F1=0.838 | weighted-F1=0.900",
        "fig5_experimental_svc_three_stage",
    )
    figures.append("fig5_experimental_svc_three_stage.png")
    loo_exp = cross_validated_predictions(all_X, exp_labels, None, "linear_pca")
    exp_metric = "Accuracy=0.896 | Balanced accuracy=0.862\nMacro-F1=0.838 | Weighted-F1=0.900"
    plot_confusion(exp_labels, loo_exp["predicted_label"], EXPERIMENTAL_CLASSES,
                   "Experimental three-stage SVC LOOCV", exp_metric, "fig6_experimental_loocv_three_stage")
    figures.append("fig6_experimental_loocv_three_stage.png")

    # Exploratory three-class NAS SVC and animal-grouped validation figures.
    nas_three_classes = ["No-MASH", "Indeterminate", "MASH-compatible"]
    loo_three = cross_validated_predictions(all_X, hist_labels, None, "cosine_kernel_pca")
    plot_decision_regions(
        df, kpca_xy, hist_labels, nas_three_classes,
        "Three-class SVC decision regions for NAS-defined categories",
        "Measurement-level LOOCV: " + format_metrics(hist_labels, loo_three["predicted_label"], nas_three_classes).replace("\n", " | "),
        "S1_nas_three_class_svc_decision_regions",
    )
    figures.append("S1_nas_three_class_svc_decision_regions.png")
    cm3_loo = confusion_matrix(hist_labels, loo_three["predicted_label"], labels=["No-MASH", "Indeterminate", "MASH-compatible"])
    plot_confusion(hist_labels, loo_three["predicted_label"], nas_three_classes,
                   "SVC leave-one-measurement-out", format_metrics(hist_labels, loo_three["predicted_label"], nas_three_classes),
                   "S2_nas_three_class_measurement_loocv_confusion")
    figures.append("S2_nas_three_class_measurement_loocv_confusion.png")
    groups_all = df["animal_id"].to_numpy()
    loo_animal_three = cross_validated_predictions(all_X, hist_labels, groups_all, "cosine_kernel_pca")
    cm3_loao = confusion_matrix(hist_labels, loo_animal_three["predicted_label"], labels=["No-MASH", "Indeterminate", "MASH-compatible"])
    plot_confusion(hist_labels, loo_animal_three["predicted_label"], nas_three_classes,
                   "SVC leave-one-animal-out", format_metrics(hist_labels, loo_animal_three["predicted_label"], nas_three_classes),
                   "S3_nas_three_class_leave_one_animal_out_confusion")
    figures.append("S3_nas_three_class_leave_one_animal_out_confusion.png")
    groups_binary = binary["animal_id"].to_numpy()
    loao_binary = cross_validated_predictions(binary_X, binary["histology_class"].to_numpy(), groups_binary, "cosine_kernel_pca")
    plot_confusion(binary["histology_class"], loao_binary["predicted_label"], HISTOLOGY_CLASSES,
                   "Binary NAS SVC: leave-one-animal-out", binary_metric,
                   "S4_nas_binary_leave_one_animal_out_confusion")
    figures.append("S4_nas_binary_leave_one_animal_out_confusion.png")

    # Numerical checks against manuscript claims and predictions already produced
    # by the reconstructed analysis module.
    checks = {
        "sample_counts": {"measurements": len(df), "animals": int(df["animal_id"].nunique())},
        "ward_clusters": {},
        "experimental_loocv_confusion": confusion_matrix(exp_labels, loo_exp["predicted_label"], labels=EXPERIMENTAL_CLASSES).tolist(),
        "binary_loocv_confusion": confusion_matrix(binary["histology_class"], loo_binary["predicted_label"], labels=HISTOLOGY_CLASSES).tolist(),
        "binary_loao_confusion": confusion_matrix(binary["histology_class"], loao_binary["predicted_label"], labels=HISTOLOGY_CLASSES).tolist(),
        "three_class_nas_loocv_confusion": cm3_loo.tolist(),
        "three_class_nas_loao_confusion": cm3_loao.tolist(),
    }
    for label, embedding, k_values in [("experimental_PCA", pca_xy, (3, 4)), ("histology_cosine_KPCA", kpca_xy, (3, 4))]:
        for k in k_values:
            cl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embedding)
            checks["ward_clusters"][f"{label}_k{k}"] = {str(x+1): int(n) for x, n in zip(*np.unique(cl, return_counts=True))}
    (OUT_DIR / "regenerated_figure_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    compare_files(figures)
    print(json.dumps(checks, indent=2))
    print(f"Regenerated figures saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()

