from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("dark:purple", 4)


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> float:
    """Pearson r with zero-variance guard to avoid divide-by-zero warnings."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.nanstd(x) < eps or np.nanstd(y) < eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def set_paper_style():
    """Consistent rcParams for all figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 25,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, path: Optional[str], dpi: int = 300) -> None:
    if path is None:
        return
    _ensure_parent(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_program_heatmaps(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    cmap=None,
    figsize: tuple = (10,5),
    show_gene_ticks: bool = True,
    save_path: Optional[str] = None,
    signed_flags: Optional[Sequence[bool]] = None,
):
    """Side-by-side gene-program heatmaps.

    Parameters
    ----------
    signed_flags : list[bool], optional
        Per-panel flag indicating whether the matrix contains signed loadings
        (e.g. PCA eigenvectors).  Signed panels use a diverging colormap
        centred at 0; non-signed panels use a sequential colormap from 0.
    """
    n_panels = len(matrices)
    if n_panels == 0:
        return

    if signed_flags is None:
        signed_flags = [False] * n_panels

    nrows = min(2, n_panels)
    ncols = -(-n_panels // nrows)

    nonneg_mats = [m for m, s in zip(matrices, signed_flags) if not s]
    signed_mats = [m for m, s in zip(matrices, signed_flags) if s]

    cmap_signed = "RdBu_r"
    # Non-negative panels: use the warm (red) half of the same RdBu_r map so
    # the zero-background is the same white as the signed panels.
    import matplotlib.colors as mcolors
    _full = plt.cm.get_cmap("RdBu_r")
    cmap_nonneg = cmap if cmap is not None else mcolors.LinearSegmentedColormap.from_list(
        "RdBu_r_pos", _full(np.linspace(0.5, 1.0, 256)),
    )

    vmin_nn = 0.0
    vmax_nn = max((float(np.nanmax(m)) for m in nonneg_mats), default=1e-12) or 1e-12
    if signed_mats:
        abs_max_s = max(float(np.nanmax(np.abs(m))) for m in signed_mats)
        vmin_s, vmax_s = -abs_max_s, abs_max_s
    else:
        vmin_s, vmax_s = -1, 1

    fig_width, fig_height = figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height * nrows),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()
    last_nn_hm = None
    last_s_hm = None
    nn_axes = []
    s_axes = []
    for ax, mat, title, is_signed in zip(axes, matrices, titles, signed_flags):
        if is_signed:
            hm = sns.heatmap(mat, ax=ax, cmap=cmap_signed, cbar=False,
                             vmin=vmin_s, vmax=vmax_s, center=0)
            last_s_hm = hm
            s_axes.append(ax)
        else:
            hm = sns.heatmap(mat, ax=ax, cmap=cmap_nonneg, cbar=False,
                             vmin=vmin_nn, vmax=vmax_nn)
            last_nn_hm = hm
            nn_axes.append(ax)
        ax.set_title(title)
        ax.set_xlabel("Gene")
        ax.set_ylabel("Program")
        n_programs = mat.shape[0] if mat.ndim >= 1 else 0
        if n_programs > 0:
            program_ticks = np.linspace(0, n_programs - 1, num=min(6, n_programs))
            program_ticks = np.unique(np.round(program_ticks).astype(int))
            ax.set_yticks(program_ticks + 0.5)
            ax.set_yticklabels(program_ticks + 1)
        n_genes = mat.shape[1] if mat.ndim >= 2 else 0
        if n_genes > 0 and show_gene_ticks:
            tick_positions = np.linspace(0, n_genes - 1, num=min(6, n_genes))
            tick_positions = np.unique(np.round(tick_positions).astype(int))
            ax.set_xticks(tick_positions + 0.5)
            ax.set_xticklabels(tick_positions + 1)
            ax.tick_params(axis="x", rotation=0)
        elif not show_gene_ticks:
            ax.set_xticks([])
            ax.set_xlabel("")
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    if not any(signed_flags):
        if last_nn_hm is not None:
            cbar = fig.colorbar(last_nn_hm.collections[0], ax=axes,
                                shrink=0.85, location="right", pad=0.02)
            cbar.set_label("Program weight", rotation=270, labelpad=14)
    else:
        # Single colorbar using the full signed range so all panels share one
        # scale bar with consistent ticks and no overlap.
        sm = plt.cm.ScalarMappable(
            cmap=cmap_signed,
            norm=plt.Normalize(vmin=vmin_s, vmax=vmax_s),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes[:n_panels]),
                            shrink=0.85, location="right", pad=0.02)
        cbar.set_label("Loading", rotation=270, labelpad=14)
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_alpha_curves(
    alpha_true: np.ndarray,
    alpha_estimates: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    program_indices: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    alpha_lower_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    alpha_upper_list: Optional[Sequence[Optional[np.ndarray]]] = None,
):
    """Overlay temporal trajectories for selected programs with distinct styles per method.

    ``alpha_lower_list`` / ``alpha_upper_list`` are optional per-method arrays of
    shape (p, T) containing lower / upper credible-interval bounds.  When both
    are provided for a given method a shaded band is drawn around that method's
    curve.
    """
    P, T = alpha_true.shape
    programs = list(program_indices) if program_indices is not None else list(range(min(P, 4)))
    n_plots = len(programs)
    ncols = 2 if n_plots > 1 else 1
    nrows = -(-n_plots // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    x = np.arange(T)
    n_methods = len(labels)

    # Pad CI lists so indexing is always safe
    lo_list = list(alpha_lower_list) if alpha_lower_list is not None else [None] * n_methods
    hi_list = list(alpha_upper_list) if alpha_upper_list is not None else [None] * n_methods
    lo_list += [None] * (n_methods - len(lo_list))
    hi_list += [None] * (n_methods - len(hi_list))

    selected_true = alpha_true[programs]
    series = [selected_true] + [est[programs] for est in alpha_estimates]
    # Include CI bounds in the y-range so bands are never clipped
    for lo, hi in zip(lo_list, hi_list):
        if lo is not None:
            series.append(lo[programs])
        if hi is not None:
            series.append(hi[programs])
    y_min = min(np.nanmin(vals) for vals in series)
    y_max = max(np.nanmax(vals) for vals in series)
    y_range = y_max - y_min
    padding = 0.05 * y_range if y_range > 0 else 0.05 * (abs(y_max) + 1.0)
    y_lower, y_upper = y_min - padding, y_max + padding
    if colors is None:
        palette = sns.color_palette("colorblind", max(n_methods, len(PALETTE)))
        colors = palette[:n_methods]
    if linestyles is None:
        base_styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
        linestyles = list(itertools.islice(itertools.cycle(base_styles), n_methods))
    for ax, prog in zip(axes_flat, programs):
        ax.plot(x, alpha_true[prog], label="True", color="black", linewidth=2)
        for est, label, color, ls, lo, hi in zip(
            alpha_estimates, labels, colors, linestyles, lo_list, hi_list
        ):
            if lo is not None and hi is not None:
                ax.fill_between(
                    x, lo[prog], hi[prog],
                    color=color, alpha=0.20, linewidth=0,
                )
            ax.plot(x, est[prog], label=label, linestyle=ls, color=color)
        ax.set_ylabel(r"$\alpha_{t}$")
        ax.set_title(f"Program {prog + 1}")
        ax.set_ylim(y_lower, y_upper)
    # Remove any unused axes (when n_plots is not a multiple of ncols)
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)
    axes_flat[min(n_plots, len(axes_flat)) - 1].set_xlabel("Time")
    legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        borderaxespad=0.2,
        frameon=False,
    )
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_spatial_slice(
    coords: np.ndarray,
    slices: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    cmap: str = "viridis",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
):
    """Scatter plots comparing spatial effects."""
    n = len(slices)
    if n == 0:
        return
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    fig_width, fig_height = figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    vmin = min(np.min(s) for s in slices)
    vmax = max(np.max(s) for s in slices)
    for ax, values, title in zip(axes, slices, titles):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=30, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].set_ylabel("y")
    fig.colorbar(sc, ax=axes[:n], shrink=0.85, location="right")
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


_RECOVERY_METRIC_TITLES = {
    "w_rmse": "Program Weight",
    "subspace_similarity": "Program Similarity",
    "hellinger_score": "Program Similarity",
    "jaccard_tau90": "Gene Detection",
    "H_corr": "Cell Embedding",
    "H_corr_spearman": "Cell Embedding",
    "tv_score": "Program Similarity",
    "f1_tau90": "Gene Detection",
    "gene_topk_hit_mean": "Gene Detection",
}
_RECOVERY_METRIC_YLABELS = {
    "w_rmse": "RMSE",
    "subspace_similarity": "Subspace Similarity",
    "hellinger_score": "Hellinger Score",
    "jaccard_tau90": "Jaccard Score",
    "H_corr": "Pearson Correlation",
    "H_corr_spearman": "Spearman Correlation",
    "tv_score": "Total Variation Score",
    "f1_tau90": "F1 Score",
    "gene_topk_hit_mean": "Top-k Hit Rate",
}

# Main figure (simu_gaussian, simu_gaussian_logscale, supp_simu_singlecell):
# Hellinger + Detected Rate + Pearson H.
_DEFAULT_METRIC_COLS = [
    "hellinger_score",
    "gene_topk_hit_mean",
    "H_corr",
]

# Supplementary 2×3 grid (row-major):
#   row1: Program Weight, TV, Subspace  |  row2: Jaccard, F1, Spearman H
_SUPP_METRIC_LAYOUT_ORDER = [
    "w_rmse",
    "tv_score",
    "subspace_similarity",
    "jaccard_tau90",
    "f1_tau90",
    "H_corr_spearman",
]
_SUPP_METRIC_COLS = list(_SUPP_METRIC_LAYOUT_ORDER)

DEFAULT_RECOVERY_MAIN_METRICS = tuple(_DEFAULT_METRIC_COLS)

_SUPP1_METRIC_COLS = ["tv_score", "f1_tau90", "H_corr_spearman"]
_SUPP2_METRIC_COLS = ["w_rmse", "subspace_similarity", "jaccard_tau90"]

DEFAULT_RECOVERY_SUPP1_METRICS = tuple(_SUPP1_METRIC_COLS)
DEFAULT_RECOVERY_SUPP2_METRICS = tuple(_SUPP2_METRIC_COLS)

# Kernel misspecification study: one row of four panels.
RECOVERY_MAIN_METRICS_KERNEL_MISSPEC = (
    "w_rmse",
    "hellinger_score",
    "jaccard_tau90",
    "H_corr",
)


def plot_recovery_metrics_boxplot(
    df: pd.DataFrame,
    save_path,
    *,
    panel_order: Sequence[str],
    metric_cols: Optional[Sequence[str]] = None,
) -> None:
    """Main recovery boxplots; use ``metric_cols`` or a preset from this module (see defaults above)."""
    if metric_cols is None:
        metric_cols = [c for c in _DEFAULT_METRIC_COLS if c in df.columns]
    else:
        metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return
    method_order = [m for m in panel_order if m in df["method"].unique()]
    palette = dict(zip(method_order, sns.color_palette("tab10", len(method_order))))

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    for ax, col in zip(axes, metric_cols):
        sub = df[["method", col]].dropna()
        sns.boxplot(
            data=sub, x="method", y=col, order=method_order,
            hue="method", hue_order=method_order, palette=palette,
            legend=False, ax=ax, fliersize=2, linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(
            _RECOVERY_METRIC_YLABELS.get(col, col), fontsize=16,
        )
        ax.set_title(
            _RECOVERY_METRIC_TITLES.get(col, col.replace("_", " ")),
            fontsize=17, fontweight="bold",
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=35, ha="right", fontsize=13,
        )
    fig.tight_layout(pad=1.5)
    save_fig(fig, str(save_path))
    plt.show()
    plt.close(fig)


def plot_recovery_metrics_supp_boxplot(
    df: pd.DataFrame,
    save_path,
    *,
    panel_order: Sequence[str],
    metric_cols: Optional[Sequence[str]] = None,
) -> None:
    """Supplementary recovery figure: six metrics in a 2×3 grid when all are present."""
    if metric_cols is None:
        metric_cols = [c for c in _SUPP_METRIC_COLS if c in df.columns]
    else:
        metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return
    method_order = [m for m in panel_order if m in df["method"].unique()]
    palette = dict(zip(method_order, sns.color_palette("tab10", len(method_order))))

    n_panels = len(metric_cols)
    layout_set = set(_SUPP_METRIC_LAYOUT_ORDER)
    if n_panels == 6 and set(metric_cols) == layout_set:
        metric_cols = [c for c in _SUPP_METRIC_LAYOUT_ORDER if c in df.columns]
        fig, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 5.0 * 2))
        axes = np.asarray(axes).ravel()
    elif n_panels == 1:
        fig, ax = plt.subplots(figsize=(4.2, 5))
        axes = np.asarray([ax])
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 5))
        axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, metric_cols):
        sub = df[["method", col]].dropna()
        sns.boxplot(
            data=sub, x="method", y=col, order=method_order,
            hue="method", hue_order=method_order, palette=palette,
            legend=False, ax=ax, fliersize=2, linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(
            _RECOVERY_METRIC_YLABELS.get(col, col), fontsize=16,
        )
        ax.set_title(
            _RECOVERY_METRIC_TITLES.get(col, col.replace("_", " ")),
            fontsize=17, fontweight="bold",
        )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=35, ha="right", fontsize=13,
        )
    fig.tight_layout(pad=1.5)
    save_fig(fig, str(save_path))
    plt.show()
    plt.close(fig)


def plot_method_runtime_boxplot(
    df: pd.DataFrame,
    save_path,
    *,
    panel_order: Sequence[str],
    time_col: str = "runtime",
    log_y: bool = True,
) -> None:
    """Boxplot of per-fit wall times (seconds) across replicates, one panel."""
    if time_col not in df.columns or df[time_col].notna().sum() == 0:
        return
    sub = df[["method", time_col]].dropna()
    if sub.empty:
        return
    method_order = [m for m in panel_order if m in sub["method"].unique()]
    if not method_order:
        return
    palette = dict(zip(method_order, sns.color_palette("tab10", len(method_order))))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(method_order) + 2.5), 7.0))
    sns.boxplot(
        data=sub,
        x="method",
        y=time_col,
        order=method_order,
        hue="method",
        hue_order=method_order,
        palette=palette,
        legend=False,
        ax=ax,
        fliersize=2,
        linewidth=0.8,
        width=0.6,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Time (s)", fontsize=30)
    ax.set_title("Method Runtime", fontsize=35, fontweight="bold")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=30)
    ax.tick_params(axis="y", labelsize=16)
    if log_y and float(sub[time_col].min()) > 0.0:
        ax.set_yscale("log")
    fig.tight_layout(pad=1.5)
    save_fig(fig, str(save_path))
    plt.show()
    plt.close(fig)


def plot_H_recovery_scatter(
    true_H: np.ndarray,
    H_fit: np.ndarray,
    *,
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of true vs fitted H (cell embeddings), one panel per program."""
    p = true_H.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
    axes_flat = np.atleast_1d(axes)
    for j, ax in enumerate(axes_flat):
        ax.scatter(true_H[:, j], H_fit[:, j], s=1, alpha=0.2)
        lo = min(float(true_H[:, j].min()), float(H_fit[:, j].min()))
        hi = max(float(true_H[:, j].max()), float(H_fit[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        corr = _safe_pearson_corr(H_fit[:, j], true_H[:, j])
        ax.set_title(f"Prog {j + 1}: corr={corr:.4f}", fontsize=9)
        ax.set_xlabel("True H", fontsize=8)
        ax.set_ylabel("Fitted H", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_b_recovery_scatter(
    true_B_flat: Sequence[np.ndarray],
    b_fit: np.ndarray,
    *,
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of true vs fitted spatial effects b, one panel per program."""
    p = len(true_B_flat)
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
    axes_flat = np.atleast_1d(axes)
    for j, ax in enumerate(axes_flat):
        tb = true_B_flat[j]
        fb = b_fit[:, j]
        ax.scatter(tb, fb, s=1, alpha=0.2, color="purple")
        lo = min(float(tb.min()), float(fb.min()))
        hi = max(float(tb.max()), float(fb.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        corr = _safe_pearson_corr(fb, tb)
        ax.set_title(f"Prog {j + 1}: b corr={corr:.4f}", fontsize=9)
        ax.set_xlabel("True b", fontsize=8)
        ax.set_ylabel("Estimated b", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_spatial_slice_all_programs(
    coords: np.ndarray,
    true_B_slice: Sequence[np.ndarray],
    b_fit_slice: np.ndarray,
    t_idx: int,
    *,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side True vs stGP spatial scatter for every program at one time slice.

    """
    p = len(true_B_slice)
    fig, axes = plt.subplots(p, 2, figsize=(10, 4 * p), constrained_layout=True)
    axes = np.atleast_2d(axes)
    all_true = np.concatenate([np.ravel(np.asarray(tb)) for tb in true_B_slice])
    all_fit = np.ravel(np.asarray(b_fit_slice))
    vmin = min(float(np.nanmin(all_true)), float(np.nanmin(all_fit)))
    vmax = max(float(np.nanmax(all_true)), float(np.nanmax(all_fit)))
    for j in range(p):
        tb = true_B_slice[j]
        fb = b_fit_slice[:, j]
        for ax, vals, title in zip(axes[j], [tb, fb], [f"True Prog {j + 1}", f"stGP Prog {j + 1}"]):
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=vals, cmap=cmap, s=15, vmin=vmin, vmax=vmax,
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(sc, ax=axes[j], shrink=0.8, location="right")
    fig.suptitle(f"Spatial slice  t={t_idx}", fontsize=10)
    save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_spatialpca_nz_comparison(
    all_reps: dict,
    n_reps: int,
    params: dict,
    *,
    load_dataset_fn,
    save_dir,
) -> None:
    """Compare SpatialPCA-nz vs PCA: loading similarity and cell-embedding similarity.

    Produces two boxplots in *save_dir*:
      - loading_similarity_boxplot.png   (per-rep Pearson correlation of |W| rows)
      - embedding_similarity_boxplot.png (per-rep Spearman correlation of H columns)
    """
    from scipy.stats import spearmanr
    from metrics_utils import align_programs, project_to_simplex_rows

    pca_reps = all_reps.get("PCA", {})
    nz_reps  = all_reps.get("SpatialPCA-nz", {})

    records = []
    for rep in range(n_reps):
        if rep not in pca_reps or rep not in nz_reps:
            continue
        res_pca = pca_reps[rep]
        res_nz  = nz_reps[rep]

        W_pca = project_to_simplex_rows(np.asarray(res_pca.W, dtype=float))
        W_nz  = project_to_simplex_rows(np.asarray(res_nz.W, dtype=float))

        align = align_programs(W_pca, W_nz, sparsify_topk=None, sparsify_frac=0.1, project=True)
        perm = align.perm
        W_nz_aligned = W_nz[perm]

        loading_corrs = []
        for j in range(W_pca.shape[0]):
            r = np.corrcoef(W_pca[j], W_nz_aligned[j])[0, 1]
            if np.isfinite(r):
                loading_corrs.append(r)
        loading_corr = float(np.mean(loading_corrs)) if loading_corrs else float("nan")

        H_pca = np.asarray(res_pca.H, dtype=float)
        H_nz  = np.asarray(res_nz.H, dtype=float)[:, perm]
        emb_corrs = []
        for j in range(H_pca.shape[1]):
            r, _ = spearmanr(H_pca[:, j], H_nz[:, j])
            if np.isfinite(r):
                emb_corrs.append(r)
        emb_corr = float(np.mean(emb_corrs)) if emb_corrs else float("nan")

        records.append({"rep": rep, "Loading Similarity": loading_corr,
                        "Embedding Similarity": emb_corr})

    if not records:
        return

    save_dir = Path(save_dir)
    df = pd.DataFrame(records)
    df.to_csv(save_dir / "spatialpca_nz_vs_pca.csv", index=False)

    for col, fname in [
        ("Loading Similarity", "loading_similarity_boxplot.png"),
        ("Embedding Similarity", "embedding_similarity_boxplot.png"),
    ]:
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.boxplot(y=df[col], ax=ax, color=sns.color_palette("tab10")[1],
                    fliersize=2, linewidth=0.8, width=0.4)
        ax.set_ylabel(col, fontsize=16)
        ax.set_title(f"SpatialPCA-nz vs PCA\n{col}", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        fig.tight_layout(pad=1.5)
        save_fig(fig, str(save_dir / fname))
        plt.show()
        plt.close(fig)
