from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 400
NM_W_SINGLE = 88 / 25.4
NM_W_HALF = 120 / 25.4
NM_W_FULL = 180 / 25.4


STYLE_PRESETS = {
    "simulation": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": DPI,
    },
    "human": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
    },
    "mouse_brain": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 15,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.15,
        "axes.labelsize": 17,
        "axes.titlesize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    },
    "kidney": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": DPI,
    },
}


def setup_publication_style(preset: str) -> None:
    """Apply one of the exact rcParams blocks used by the figure notebooks."""
    plt.rcParams.update(STYLE_PRESETS[preset])


def resolve_repro_root(start: str | Path | None = None) -> Path:
    base = Path.cwd().resolve() if start is None else Path(start).resolve()
    return base if (base / "FigureReproducing").exists() else base.parent


def force_vector_pdf_artists(fig, *, include_collections: bool = False) -> None:
    for ax in fig.axes:
        artists = list(ax.images)
        if include_collections:
            artists.extend(ax.collections)
        for artist in artists:
            artist.set_rasterized(False)


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _display_in_notebook(path: Path) -> None:
    try:
        from IPython.display import Image, display
    except ImportError:
        return

    display(Image(filename=str(path)))



def save_pair(
    fig,
    stem: str,
    *,
    out_dir: str | Path,
    dpi: int = DPI,
    bbox_inches="tight",
    pad_inches=0.04,
    vector_pdf: bool = False,
    include_collections: bool = False,
    close: bool = True,
    display_inline: bool | None = None,
    verbose: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    kwargs = {"bbox_inches": bbox_inches}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(png, dpi=dpi, **kwargs)
    if vector_pdf:
        force_vector_pdf_artists(fig, include_collections=include_collections)
    fig.savefig(pdf, **kwargs)
    if display_inline is None:
        display_inline = _running_in_notebook()
    if display_inline:
        _display_in_notebook(png)
    if close:
        plt.close(fig)
    if verbose:
        print(f"Saved {png.name} and {pdf.name}")
    return png, pdf


def p_to_stars(pval, *, nan_label="NA", nonsig_label="ns") -> str:
    if not np.isfinite(pval):
        return nan_label
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return nonsig_label


def program_index(program, *, prefix: str = "stGP") -> int:
    return int(str(program).replace(prefix, "")) - 1


def spatial_program_values(
    adata,
    program,
    *,
    obsm_key: str = "X_stgp_spatial",
    allow_transpose: bool = False,
) -> np.ndarray:
    idx = program_index(program)
    arr = np.asarray(adata.obsm[obsm_key])
    if allow_transpose and arr.shape[0] != adata.n_obs:
        arr = arr.T
    if idx < 0 or idx >= arr.shape[1]:
        raise IndexError(f"{program} not found in {obsm_key} with {arr.shape[1]} columns")
    return arr[:, idx].astype(float)


def ordered_stgp_alpha(info: dict, idx: int):
    ages = np.asarray(info["ages"], dtype=float)
    alpha = np.asarray(info["alpha"], dtype=float)
    lo = np.asarray(info.get("alpha_lower", []), dtype=float)
    hi = np.asarray(info.get("alpha_upper", []), dtype=float)
    order = np.argsort(ages)
    has_ci = lo.shape == alpha.shape and hi.shape == alpha.shape
    return ages[order], alpha[idx, order], lo[idx, order] if has_ci else None, hi[idx, order] if has_ci else None, order


def draw_alpha_ci(
    ax,
    x,
    y,
    lo=None,
    hi=None,
    *,
    color: str = "#2C7FB8",
    ci_fill_alpha: float = 0.18,
    ci_line_lw: float = 1.4,
    ci_line_alpha: float = 0.65,
    line_lw: float = 3.0,
    scatter_s: float = 72,
    ci_label: str | None = "95% posterior CI",
    mean_label: str | None = "Posterior mean",
    zero_line_color: str = "#8A8A8A",
    zero_line_lw: float = 1.0,
    zorder: int = 2,
):
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=ci_fill_alpha, linewidth=0, label=ci_label)
        ax.plot(x, lo, color=color, lw=ci_line_lw, ls="--", alpha=ci_line_alpha)
        ax.plot(x, hi, color=color, lw=ci_line_lw, ls="--", alpha=ci_line_alpha)
    ax.plot(x, y, color=color, lw=line_lw, zorder=zorder)
    ax.scatter(x, y, color=color, s=scatter_s, zorder=zorder + 1, label=mean_label)
    ax.axhline(0, color=zero_line_color, lw=zero_line_lw, ls=":", zorder=1)


def ordered_gene_blocks(W: pd.DataFrame, *, top_n_per_program: int = 15):
    rows = []
    used = set()
    for program in W.index.astype(str):
        weights = W.loc[program].astype(float)
        genes = weights[weights > 0].sort_values(ascending=False).head(top_n_per_program)
        for gene_name, weight in genes.items():
            if gene_name in used:
                continue
            used.add(gene_name)
            rows.append({"program": program, "gene": str(gene_name), "anchor_weight": float(weight)})
    order = [row["gene"] for row in rows]
    return pd.DataFrame(rows), W.loc[:, order]
