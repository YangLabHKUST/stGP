"""Plotting style and figure helpers for Human Brain MERFISH analyses."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Shared publication style

@dataclass(frozen=True)
class VarPartColors:
    age: str = "#E64B35"
    region: str = "#4DBBD5"
    both: str = "#3C5488"
    residuals: str = "#BFBFBF"


# Method palette (mirrors the mouse pipeline; PCA/NMF kept for back-compat).
METHOD_COLORS = {
    "stGP": "#E64B35",
    "PCA": "#91D1C2",
    "NMF": "#F39B7F",
    "SpatialPCA": "#4DBBD5",
    "MEFISTO": "#8491B4",
    "STAMP": "#B09C85",
    "Popari": "#00A087",
}


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
            "figure.dpi": 300,
            "savefig.dpi": 400,
            "savefig.transparent": False,
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
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.5,
        }
    )


# Shared helpers

DEFAULT_AGE_UNIT = "years"


def _save(fig: plt.Figure, out: str | Path | None, *, dpi: int = 400) -> None:
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")


def _bg_per_sample(adata_full, sample_ids_target) -> dict:
    """Per-sample spatial coordinates of every cell in ``adata_full``."""
    if adata_full is None:
        return {}
    bg_sp_all = np.asarray(adata_full.obsm["spatial"])
    if "id_region" not in adata_full.obs.columns:
        return {}
    bg_sample_ids = adata_full.obs["id_region"].astype(str).to_numpy()
    out: dict = {}
    for sid in np.unique(sample_ids_target):
        mask = bg_sample_ids == sid
        if mask.any():
            out[sid] = bg_sp_all[mask]
    return out


# Spatial program maps (stGP)

def _plot_spatial_programs_impl(
    *,
    adata,
    scores: pd.DataFrame,
    adata_full=None,
    age_unit: str = DEFAULT_AGE_UNIT,
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """Shared backbone for stGP spatial-program tile plots."""
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    sample_ids = obs["id_region"].astype(str).to_numpy()

    if "X_stgp_spatial" in adata.obsm:
        b = np.asarray(adata.obsm["X_stgp_spatial"])
        spatial_scores = pd.DataFrame(b, index=scores.index, columns=scores.columns)
    else:
        spatial_scores = scores

    uniq_samples = np.unique(sample_ids)
    age_per_sample = np.array([
        float(obs.loc[obs["id_region"].astype(str) == s, "age"].iloc[0])
        for s in uniq_samples
    ])
    order = np.argsort(age_per_sample)
    uniq_samples = uniq_samples[order]
    age_per_sample = age_per_sample[order]
    n_samples = len(uniq_samples)

    bg_by_sample = _bg_per_sample(adata_full, sample_ids)
    sample_mask_cache: dict = {sid: sample_ids == sid for sid in uniq_samples}

    figs: list[plt.Figure] = []
    age_suffix = "yr" if age_unit == "years" else "mo"
    for prog in scores.columns.tolist():
        prog_vals = spatial_scores[prog].to_numpy(dtype=float)
        abs99 = float(np.nanpercentile(np.abs(prog_vals), 99))
        vmin, vmax = -abs99, abs99

        nrows = int(np.ceil(n_samples / ncols))
        panel_w, panel_h = 2.4, 2.4
        fig_w = ncols * panel_w + 0.8
        fig_h = nrows * panel_h + 0.5

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.04, "hspace": 0.18},
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.02,
            right=0.90,
            top=0.95,
            bottom=0.05,
            wspace=0.04,
            hspace=0.18,
        )
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_samples:]:
            ax.axis("off")

        sc_ref = None
        for i, (sid, age) in enumerate(zip(uniq_samples, age_per_sample)):
            ax = axes_flat[i]
            if sid in bg_by_sample:
                bx = bg_by_sample[sid]
                ax.scatter(
                    bx[:, 0],
                    bx[:, 1],
                    c="#D8D8D8",
                    s=bg_dot_size,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                )
            fg_mask = sample_mask_cache[sid]
            sc_ref = ax.scatter(
                sp[fg_mask, 0],
                sp[fg_mask, 1],
                c=prog_vals[fg_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=fg_dot_size,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{age:.1f} {age_suffix}", fontsize=12, pad=2)
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
    age_unit: str = DEFAULT_AGE_UNIT,
    ncols: int = 4,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
    dpi: int = 150,
) -> list[plt.Figure]:
    """One figure per stGP program, tiling all tissue sections by age."""
    return _plot_spatial_programs_impl(
        adata=stgp_adata,
        scores=scores,
        adata_full=adata_full,
        age_unit=age_unit,
        ncols=ncols,
        bg_dot_size=bg_dot_size,
        fg_dot_size=fg_dot_size,
        cmap=cmap,
    )


# Alpha(t) posterior aging trajectory

def plot_alpha_over_age(
    *,
    ages: np.ndarray,
    alpha: np.ndarray,
    alpha_lower: np.ndarray | None = None,
    alpha_upper: np.ndarray | None = None,
    age_unit: str = DEFAULT_AGE_UNIT,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Posterior mean aging trajectory with optional 95% CI band."""
    color = "#2C7FB8"

    ages = np.asarray(ages, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()

    has_ci = (alpha_lower is not None) and (alpha_upper is not None)
    if has_ci:
        lo = np.asarray(alpha_lower, dtype=float).ravel()
        hi = np.asarray(alpha_upper, dtype=float).ravel()
        mask = np.isfinite(ages) & np.isfinite(alpha) & np.isfinite(lo) & np.isfinite(hi)
        lo, hi = lo[mask], hi[mask]
    else:
        mask = np.isfinite(ages) & np.isfinite(alpha)

    ages, alpha = ages[mask], alpha[mask]
    order = np.argsort(ages)
    ages, alpha = ages[order], alpha[order]
    if has_ci:
        lo, hi = lo[order], hi[order]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    if has_ci:
        ax.fill_between(
            ages,
            lo,
            hi,
            color=color,
            alpha=0.18,
            linewidth=0,
            label="95% posterior CI",
        )
        ax.plot(ages, lo, color=color, lw=0.8, ls="--", alpha=0.55)
        ax.plot(ages, hi, color=color, lw=0.8, ls="--", alpha=0.55)

    ax.plot(ages, alpha, color=color, lw=1.6, zorder=2)
    ax.scatter(ages, alpha, color=color, s=30, zorder=3, alpha=0.9, label="Posterior mean")

    ax.set_xlabel(f"Age ({age_unit})")
    ax.set_ylabel("Age effect")
    ax.grid(False)
    if title:
        ax.set_title(title)
    if has_ci:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# Spatial kernel correlation diagnostic

def plot_spatial_kernel_corr_combined(
    adata,
    *,
    bandwidth: float,
    slice_idx: int | str | float = 0,
    age_unit: str = DEFAULT_AGE_UNIT,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Two-panel diagnostic: spatial kernel matrix heatmap + per-cell scatter.

    Selects a tissue slice via ``slice_idx`` on ``obs['id_region']``.
    """
    import seaborn as sns

    col = adata.obs.get("id_region")
    if col is None:
        raise KeyError("adata.obs missing 'id_region' column")

    _is_int = isinstance(slice_idx, (int, np.integer)) and not isinstance(slice_idx, bool)
    if _is_int:
        uniq = col.dropna().unique()
        try:
            uniq_sorted = np.sort(uniq.astype(float))
        except (ValueError, TypeError):
            uniq_sorted = np.sort(np.asarray(uniq, dtype=object))
        if not len(uniq_sorted):
            raise ValueError("No non-null values in obs['id_region']")
        target_val = uniq_sorted[min(max(int(slice_idx), 0), len(uniq_sorted) - 1)]
        try:
            mask = np.isclose(
                pd.to_numeric(col, errors="coerce").to_numpy(),
                float(target_val),
                rtol=0.0,
                atol=1e-9,
            )
        except (TypeError, ValueError):
            mask = (col.astype(str) == str(target_val)).to_numpy()
    else:
        target_val = slice_idx
        mask = (col == target_val).to_numpy()
        if not mask.any():
            raise ValueError(f"No rows with obs['id_region'] == {target_val!r}")

    try:
        disp_val = f"{float(target_val):.1f}"
    except (TypeError, ValueError):
        disp_val = str(target_val)

    age_note = ""
    if "age" in adata.obs.columns:
        au = pd.to_numeric(adata.obs.loc[mask, "age"], errors="coerce").dropna().unique()
        if len(au) == 1:
            age_note = f", age={float(au[0]):.1f} {age_unit}"
        elif len(au) > 1:
            age_note = f", age non-constant {age_unit}"

    coords = np.asarray(adata.obsm["spatial"][mask], dtype=float)
    n_s = len(coords)
    coords = (coords - coords.mean(0)) / np.maximum(coords.std(0, ddof=1), 1e-12)

    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n_s, 400, replace=False)) if n_s > 400 else np.arange(n_s)
    cs = coords[sub]
    k_matrix = np.exp(-np.sum((cs[:, None] - cs[None]) ** 2, axis=2) / bandwidth)

    ref = int(np.argmin(np.linalg.norm(coords - coords.mean(0), axis=1)))
    k_vals = np.exp(-np.sum((coords - coords[ref]) ** 2, axis=1) / bandwidth)

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(k_matrix, ax=ax1, cmap="Blues", cbar_kws={"shrink": 0.8})
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Spatial kernel matrix  (slice {slice_idx}, n={len(sub)})", pad=8)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(
        coords[:, 0],
        coords[:, 1],
        c=k_vals,
        cmap="magma",
        s=18,
        vmin=0,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    ax2.scatter(
        coords[ref, 0],
        coords[ref, 1],
        marker="*",
        s=320,
        c="cyan",
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
        label="ref cell",
    )
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02).set_label("kernel correlation")
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    ax2.set_title(
        f"Reference-cell correlation  (slice {slice_idx}, "
        f"id_region={disp_val}{age_note})",
        pad=8,
    )

    if title:
        fig.suptitle(title, y=1.04)

    _save(fig, out, dpi=dpi)
    return fig


# W matrix heatmap

def plot_W_program_heatmap(
    W: pd.DataFrame,
    *,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
    orientation: Literal["vertical", "horizontal"] = "vertical",
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

    pink_cmap = LinearSegmentedColormap.from_list("pinkish", ["white", "#FFD1DC", "red"])
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


# Enrichment dot plots

def clean_term(term: str, width: int | None = None) -> str:
    """Shorten enrichment terms and optionally wrap labels for plotting."""
    term = re.sub(r"\s*\([^)]*\)\s*$", "", str(term))
    term = term.replace("_", " ").strip()
    term = re.sub(r"\s+", " ", term)
    if width is not None:
        term = "\n".join(textwrap.wrap(term, width=width))
    return term


def truncate_colormap(name: str, lo: float = 0.15, hi: float = 0.82, n: int = 256):
    """Return a truncated Matplotlib colormap."""
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_trunc",
        base(np.linspace(lo, hi, n)),
    )


def _parse_overlap(value) -> float:
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den)
        except ValueError:
            return np.nan
    return pd.to_numeric(value, errors="coerce")


def plot_enrichment_dotplot(
    res_df: pd.DataFrame,
    ax,
    title: str,
    cmap,
    *,
    n_top: int = 6,
    padj_thresh: float = 0.1,
) -> None:
    """Dot plot for top enrichment terms on an existing axis."""
    if res_df is None or len(res_df) == 0:
        ax.text(0.5, 0.5, "No terms", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    df = res_df.copy()
    term_col = "Term" if "Term" in df.columns else df.columns[0]
    padj_col = next(
        (c for c in ["Adjusted P-value", "adjusted_p_value", "p.adjust", "padj"] if c in df),
        None,
    )
    overlap_col = next((c for c in ["Overlap", "GeneRatio", "overlap"] if c in df), None)
    if padj_col is None:
        raise KeyError("Could not find an adjusted p-value column.")
    df[padj_col] = pd.to_numeric(df[padj_col], errors="coerce")
    df = df.dropna(subset=[padj_col])
    df = df[df[padj_col] <= padj_thresh].nsmallest(n_top, padj_col)
    if df.empty:
        ax.text(0.5, 0.5, f"No terms at FDR <= {padj_thresh:g}", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    ratio = (
        df[overlap_col].map(_parse_overlap).to_numpy(dtype=float)
        if overlap_col is not None
        else np.arange(len(df), dtype=float) + 1
    )
    score = -np.log10(df[padj_col].clip(lower=np.finfo(float).tiny))
    y = np.arange(len(df))
    sc = ax.scatter(
        score,
        y,
        c=ratio,
        s=70 + 550 * np.nan_to_num(ratio, nan=0.0),
        cmap=cmap,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_term(t, width=34) for t in df[term_col]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Gene ratio")
