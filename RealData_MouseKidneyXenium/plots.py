"""Self-contained plotting routines used across the MERFISH analysis pipeline.

The module is grouped by figure category:
    - kernel diagnostics (temporal / spatial)
    - per-program weighted scores by age
    - per-program variance partition
    - gene trajectories over age
    - spatial program maps (stGP and Popari, signed and unsigned)
    - alpha(t) curves
    - UMAP embeddings
    - spatial clustering tiles
    - W program heatmap
    - runtime comparison
    - active-gene dot plots
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

NM_W_SINGLE = 88 / 25.4
NM_W_HALF = 120 / 25.4
NM_W_FULL = 180 / 25.4
NM_H_MAX = 225 / 25.4


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def set_nature_style(*, font: str | None = None) -> None:
    """Apply the shared publication style for all figures."""
    font_stack = [f for f in [font, "Arial", "Helvetica", "DejaVu Sans"] if f]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
        }
    )

# Kernel diagnostic colour scheme.
_ILL_COND = 1e6
_C_WELL = "#4393C3"   # well-conditioned: medium blue
_C_ILL = "#D6604D"    # ill-conditioned: medium red


def _save(fig: plt.Figure, out: str | Path | None, *, dpi: int = 400) -> None:
    """Persist ``fig`` to ``out`` (creates parents); no-op when ``out`` is None."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")


def _cond_status(K: np.ndarray) -> tuple[float, str]:
    """Return ``(condition_number, 'well-conditioned' | 'ill-conditioned')``."""
    eigs = np.linalg.eigvalsh(K)
    pos = eigs[eigs > 0]
    if pos.size == 0:
        return np.inf, "ill-conditioned"
    cond = float(pos[-1] / pos[0])
    return cond, "ill-conditioned" if cond > _ILL_COND else "well-conditioned"


def _bg_per_mouse(adata_full, mouse_ids_target) -> dict:
    """Per-mouse spatial coordinates of every cell in ``adata_full`` (used as
    grey background context behind the target-cell-type overlay)."""
    if adata_full is None:
        return {}
    bg_sp_all = np.asarray(adata_full.obsm["spatial"])
    bg_mouse_ids = adata_full.obs["ident"].astype(str).to_numpy()
    out: dict = {}
    for mid in np.unique(mouse_ids_target):
        mask = bg_mouse_ids == mid
        if mask.any():
            out[mid] = bg_sp_all[mask]
    return out


# ════════════════════════════════════════════════════════════════════════════
# Kernel diagnostics
# ════════════════════════════════════════════════════════════════════════════

def plot_kernel_age(
    K_age: np.ndarray,
    ages: np.ndarray,
    uniq_groups: np.ndarray,
    celltype: str,
    *,
    fig_dir: Path | None = None,
    kernel_type: str = "rbf",
    gamma_age: float | None = None,
    rho: float = 0.75,
    jitter: float = 0.0,
) -> plt.Figure:
    """Heatmap of ``K_age``, annotated with condition-number status."""
    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
    cond, status = _cond_status(K_age)
    T = K_age.shape[0]
    is_ill = "ill" in status

    fig, ax = plt.subplots(
        figsize=(max(NM_W_SINGLE, min(NM_W_HALF, T * 0.42 + 1.5)),
                 max(2.9, min(4.2, T * 0.36 + 1.4))),
        constrained_layout=True,
    )
    im = ax.imshow(
        K_age,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.025)
    cbar.set_label(r"$K_{\mathrm{age}}$", labelpad=4)
    cbar.ax.tick_params(length=2.5, width=0.7, labelsize=8.5)
    cbar.outline.set_linewidth(0.7)

    tick_labels = [f"{a:g}" for a in ages]
    ax.set_xticks(range(T))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(T))
    ax.set_yticklabels(tick_labels)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax.set_xlabel("Injury time (days)")
    ax.set_ylabel("Injury time (days)")
    ax.set_title("Temporal kernel", pad=7,
                 color=_C_ILL if is_ill else "#272727")

    if fig_dir is not None:
        out = fig_dir / f"kernel_age_{celltype}.png"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        print(f"[kernel] K_age ({kernel_type})  {status}  (cond={cond:.2e})  ->  {out}")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# Per-program weighted expression score boxplots by age
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Spatial program maps (stGP)
# ════════════════════════════════════════════════════════════════════════════

def _plot_spatial_programs_impl(
    *,
    adata,
    scores: pd.DataFrame,
    adata_full=None,
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """Shared backbone for the stGP spatial-program tile plots.

    Each panel shows one mouse's tissue section. When ``adata_full`` is given,
    every cell of that mouse is drawn as a faint grey cloud for anatomical
    context; the target cell type is then overlaid and coloured by the
    program score.
    """
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    mouse_ids = obs["ident"].astype(str).to_numpy()

    if "X_stgp_spatial" in adata.obsm:
        # stGP: use the spatial residual b instead of the score H if available.
        b = np.asarray(adata.obsm["X_stgp_spatial"])
        spatial_scores = pd.DataFrame(b, index=scores.index, columns=scores.columns)
    else:
        spatial_scores = scores

    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([
        float(obs.loc[obs["ident"].astype(str) == m, "age"].iloc[0])
        for m in uniq_mice
    ])
    order = np.argsort(age_per_mouse)
    uniq_mice = uniq_mice[order]
    age_per_mouse = age_per_mouse[order]
    n_mice = len(uniq_mice)

    bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
    mouse_mask_cache: dict = {mid: mouse_ids == mid for mid in uniq_mice}

    figs: list[plt.Figure] = []
    for prog in scores.columns.tolist():
        prog_vals = spatial_scores[prog].to_numpy(dtype=float)

        # Shared colour range across all slices of this program.
        abs99 = float(np.nanpercentile(np.abs(prog_vals), 99))
        vmin, vmax = -abs99, abs99

        nrows = int(np.ceil(n_mice / ncols))
        panel_w, panel_h = 2.4, 2.4
        fig_w = ncols * panel_w + 0.8     # +0.8 reserved for the colorbar
        fig_h = nrows * panel_h + 0.5     # +0.5 reserved for the suptitle

        # NOTE: constrained_layout must be OFF here because the colorbar is
        # placed at a fixed figure-coord position via fig.add_axes(...). With
        # constrained_layout the axes grid would expand to fill the figure
        # and overlap that fixed-position colorbar.
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.04, "hspace": 0.18},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05,
                            wspace=0.04, hspace=0.18)
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_mice:]:
            ax.axis("off")

        sc_ref = None
        for i, (mid, age) in enumerate(zip(uniq_mice, age_per_mouse)):
            ax = axes_flat[i]

            if mid in bg_by_mouse:
                bx = bg_by_mouse[mid]
                ax.scatter(bx[:, 0], bx[:, 1], c="#D8D8D8", s=bg_dot_size,
                           linewidths=0, rasterized=True, zorder=1)

            fg_mask = mouse_mask_cache[mid]
            sc_ref = ax.scatter(
                sp[fg_mask, 0], sp[fg_mask, 1],
                c=prog_vals[fg_mask], cmap=cmap, vmin=vmin, vmax=vmax,
                s=fg_dot_size, linewidths=0, rasterized=True, zorder=2,
            )

            ax.set_aspect("equal")
            ax.set_title(f"{age:.1f} days", fontsize=9, pad=2)
            ax.axis("off")

        if sc_ref is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
            cbar = fig.colorbar(sc_ref, cax=cbar_ax)
            cbar.set_label(f"{prog} score")
            cbar.ax.tick_params(labelsize=9)

        figs.append(fig)

    return figs


def plot_stgp_spatial_programs(
    *,
    stgp_adata,
    scores: pd.DataFrame,
    adata_full=None,
    celltype: str = "",
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
    dpi: int = 150,
) -> list[plt.Figure]:
    """One figure per stGP program, tiling all tissue sections by age.

    Uses ``obsm['X_stgp_spatial']`` (the residual ``b``) when available; falls
    back to ``scores``. The colour scale is symmetric about zero
    (+/- 99-percentile) and shared across all tissue panels within each
    programme figure.
    """
    return _plot_spatial_programs_impl(
        adata=stgp_adata, scores=scores, adata_full=adata_full, ncols=ncols,
        bg_dot_size=bg_dot_size, fg_dot_size=fg_dot_size, cmap=cmap,
    )


def plot_day14_kidney_stgp_pair(
    *,
    adata_left,
    adata_right,
    left_program: str = "stGP2",
    right_program: str = "stGP3",
    left_slice: str = "Day14L",
    right_slice: str = "Day14R",
    left_title: str = "L-stGP2",
    right_title: str = "R-stGP3",
    fg_dot_size: float = 5.0,
    cmap: str = "RdBu_r",
    out: str | Path | None = None,
    dpi: int = 400,
    quantile: float = 95,
) -> plt.Figure:
    """Day14 left/right kidney spatial stGP comparison with a bottom colourbar.    """

    def _program_values(adata, program: str) -> np.ndarray:
        if "X_stgp_spatial" not in adata.obsm:
            raise KeyError("adata.obsm missing 'X_stgp_spatial'.")
        m = re.search(r"\d+$", str(program))
        if m is None:
            raise ValueError(f"Cannot infer program index from {program!r}.")
        idx = int(m.group()) - 1
        vals = np.asarray(adata.obsm["X_stgp_spatial"])
        if idx < 0 or idx >= vals.shape[1]:
            raise IndexError(f"{program!r} is outside X_stgp_spatial with {vals.shape[1]} programs.")
        return vals[:, idx].astype(float)

    def _slice_mask(adata, slice_id: str) -> np.ndarray:
        if "ident" not in adata.obs.columns:
            raise KeyError("adata.obs missing 'ident'.")
        mask = adata.obs["ident"].astype(str).to_numpy() == str(slice_id)
        if not mask.any():
            raise ValueError(f"Slice {slice_id!r} not found in adata.obs['ident'].")
        return mask

    left_vals = _program_values(adata_left, left_program)
    right_vals = _program_values(adata_right, right_program)
    left_mask = _slice_mask(adata_left, left_slice)
    right_mask = _slice_mask(adata_right, right_slice)

    vabs = float(np.nanpercentile(np.abs(np.r_[left_vals, right_vals]), quantile))
    if not np.isfinite(vabs) or vabs <= 0:
        vabs = 1.0

    fig_width = 5.0
    fig_height = 3.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=False)
    panels = [
        (axes[0], adata_left, left_mask, left_vals, left_title),
        (axes[1], adata_right, right_mask, right_vals, right_title),
    ]
    sc_ref = None
    for ax, adata, mask, vals, title in panels:
        xy = np.asarray(adata.obsm["spatial"])
        sc_ref = ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=vals[mask], cmap=cmap, vmin=-vabs, vmax=vabs,
            s=fg_dot_size, linewidths=0, rasterized=True,
        )
        ax.set_title(title, fontsize=11, pad=2.2)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.945, bottom=0.19, wspace=0.009)

    if sc_ref is not None:
        cbar_ax = fig.add_axes([0.16, 0.07, 0.68, 0.11])
        cbar = fig.colorbar(sc_ref, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("stGP spatial activity", labelpad=2.5)
        cbar.ax.tick_params(labelsize=8, pad=1.2)

    _save(fig, out, dpi=dpi)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# Spatial kernel correlation (combined heatmap + reference scatter)
# ════════════════════════════════════════════════════════════════════════════

def plot_spatial_kernel_corr_combined(
    *,
    adata,
    bandwidth: float,
    slice_idx: int = 0,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Two-panel diagnostic for spatial kernel structure within one slice."""
    from matplotlib.colors import LinearSegmentedColormap

    if "age" not in adata.obs.columns:
        raise KeyError("adata.obs missing 'age' column")
    ages = np.sort(adata.obs["age"].astype(float).unique())
    slice_idx = min(max(slice_idx, 0), len(ages) - 1)
    target_age = float(ages[slice_idx])

    mask = adata.obs["age"].astype(float).to_numpy() == target_age
    coords_s = np.asarray(adata.obsm["spatial"][mask], dtype=float)
    n_s = coords_s.shape[0]

    # Z-score coordinates so that `bandwidth` (calibrated on z-scored coords
    # inside 02_run_stgp.py) is on the correct scale for distance computation.
    mu_s = coords_s.mean(axis=0)
    std_s = coords_s.std(axis=0, ddof=1)
    std_s[std_s < 1e-12] = 1.0
    coords_s = (coords_s - mu_s) / std_s

    # Subsample if needed to keep the heatmap tractable.
    rng = np.random.default_rng(0)
    max_n = 400
    sub = (np.sort(rng.choice(n_s, max_n, replace=False))
           if n_s > max_n else np.arange(n_s))
    coords_sub = coords_s[sub]
    d2 = np.sum((coords_sub[:, None, :] - coords_sub[None, :, :]) ** 2, axis=2)
    K = np.exp(-d2 / bandwidth)

    kernel_cmap = LinearSegmentedColormap.from_list(
        "kernel_blue_soft",
        ["#F7FBFF", "#DDEBF7", "#A8CEE4", "#4F97C7", "#0F4D92"],
    )

    fig = plt.figure(figsize=(NM_W_FULL, 3.35), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[0.95, 1.15],
        left=0.055,
        right=0.965,
        bottom=0.14,
        top=0.88,
        wspace=0.18,
    )

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        K,
        cmap=kernel_cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    cbar1 = fig.colorbar(im, ax=ax1, fraction=0.044, pad=0.018)
    cbar1.set_label(r"$K_{\mathrm{spa}}$", labelpad=3)
    cbar1.ax.tick_params(length=2.5, width=0.7, labelsize=8.5)
    cbar1.outline.set_linewidth(0.7)
    ax1.set_title(f"Kernel matrix (n={len(sub)})", pad=6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_linewidth(0.7)

    centre = coords_s.mean(axis=0)
    ref_cell = int(np.argmin(np.linalg.norm(coords_s - centre, axis=1)))
    d2_full = np.sum((coords_s - coords_s[ref_cell]) ** 2, axis=1)
    k_vals = np.exp(-d2_full / bandwidth)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(
        coords_s[:, 0], coords_s[:, 1],
        c=k_vals, cmap=kernel_cmap, s=7.5, vmin=0, vmax=1,
        linewidths=0, rasterized=True,
    )
    ax2.scatter(
        coords_s[ref_cell, 0], coords_s[ref_cell, 1],
        marker="*", s=150, c="#FFD166", edgecolors="#272727",
        linewidths=0.55, zorder=10, label="Reference cell",
    )
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.018)
    cbar.set_label(r"$K(\mathrm{ref}, x)$", labelpad=3)
    cbar.ax.tick_params(length=2.5, width=0.7, labelsize=8.5)
    cbar.outline.set_linewidth(0.7)
    ax2.legend(loc="upper right", fontsize=8.5, handletextpad=0.35,
               borderpad=0.2, frameon=False)
    ax2.set_title(f"Reference-cell correlation (slice {slice_idx}, "
                   f"{target_age:g} days)", pad=6)

    ax1.text(-0.08, 1.05, "a", transform=ax1.transAxes,
             fontsize=12, fontweight="bold", va="bottom", ha="left")
    ax2.text(-0.04, 1.05, "b", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="bottom", ha="left")

    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# W matrix heatmap (programs x active genes)
# ════════════════════════════════════════════════════════════════════════════

def plot_W_program_heatmap(
    W: pd.DataFrame, *, title: str | None = None,
    out: str | Path | None = None, dpi: int = 400,
    orientation: Literal["vertical", "horizontal"] = "horizontal",
) -> plt.Figure:
    """Heatmap of stGP gene weights with vertical or horizontal orientation."""
    from matplotlib.colors import LinearSegmentedColormap

    W_filtered = W.loc[:, (W != 0).any(axis=0)]
    programs = W_filtered.index.astype(str).tolist()
    genes = W_filtered.columns.astype(str).tolist()
    if not programs or not genes:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No active genes in W", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, out, dpi=dpi)
        return fig

    if orientation == "vertical":
        values = W_filtered.values.T
        x_labels = programs
        y_labels = genes
        x_label = "Program"
        y_label = "Gene"
        fig_w = max(3.0, len(programs) * 0.7 + 1.5)
        fig_h = max(4.0, len(genes) / 3.0 + 1.0)
        x_rotation = 90
        x_fontsize = 8
        y_fontsize = 6
        cbar_fraction = 0.05
    elif orientation == "horizontal":
        values = W_filtered.values
        x_labels = genes
        y_labels = programs
        x_label = "Gene"
        y_label = "Program"
        fig_w = max(6.0, len(genes) * 0.18)
        fig_h = max(2.5, len(programs) * 0.45)
        x_rotation = 90
        x_fontsize = 6
        y_fontsize = 8
        cbar_fraction = 0.025
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanpercentile(np.abs(W_filtered.values), 99)) or 1.0

    pink_cmap = LinearSegmentedColormap.from_list("pinkish",
                                                    ["white", "#FFD1DC", "red"])
    im = ax.imshow(values, aspect="auto", cmap=pink_cmap, vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=x_rotation, fontsize=x_fontsize)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=y_fontsize)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(axis="x", labelsize=x_fontsize, pad=1)
    ax.tick_params(axis="y", labelsize=y_fontsize, pad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=0.025)
    cbar.set_label("Weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig

