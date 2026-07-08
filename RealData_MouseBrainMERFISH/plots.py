"""Consolidated plotting, loading, and figure-helper utilities for MouseBrain MERFISH notebooks.

This file replaces the former plotting/ helper package so tutorial notebooks can import all figure-related helpers from one module.
"""
from __future__ import annotations



# === plotting/style.py ===

from dataclasses import dataclass

import matplotlib as mpl
from IPython.display import display


@dataclass(frozen=True)
class VarPartColors:
    age: str = "#E64B35"
    region: str = "#4DBBD5"
    both: str = "#3C5488"
    residuals: str = "#BFBFBF"


METHOD_COLORS = {
    "stGP": "#E64B35",
    "SpatialPCA": "#4DBBD5",
    "MEFISTO": "#8491B4",
    "STAMP": "#B09C85",
    "Popari": "#00A087",
}


def set_nature_style(*, font: str | None = None) -> None:
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
            "axes.titlesize": 14,
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


# === plotting/io.py ===
"""Loaders for stGP and baseline-method results.

Each ``load_<method>`` reads ``adata_with_scores.h5ad`` from the method's
result directory, pulls out the per-cell score matrix, and (when present)
the per-gene weight matrix. They all return a uniform ``MethodResult``
record so downstream code can iterate methods generically.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class MethodResult:
    """Uniform result envelope yielded by every per-method loader."""
    method: str
    celltype: str
    result_dir: Path
    adata: ad.AnnData
    scores: pd.DataFrame
    gene_weights: pd.DataFrame | None = None


def read_h5ad_compat(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> ad.AnnData:
    """Read h5ad files even in older AnnData environments used by NicheScope."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    try:
        return ad.read_h5ad(path)
    except Exception as err:
        msg = str(err)
        if "null" not in msg.lower() and "IOSpec" not in msg and "No read method" not in msg:
            raise

    import h5py
    import shutil

    cache_dir = Path(cache_dir) if cache_dir is not None else path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixed = cache_dir / f"{path.stem}_niche_scope_compat.h5ad"
    if not fixed.exists() or fixed.stat().st_mtime < path.stat().st_mtime:
        shutil.copy2(path, fixed)

    with h5py.File(fixed, "a") as h5:
        null_nodes: list[str] = []

        def collect_null_nodes(name, obj):
            enc = obj.attrs.get("encoding-type")
            if isinstance(enc, bytes):
                enc = enc.decode()
            if enc == "null":
                null_nodes.append(name)

        h5.visititems(collect_null_nodes)
        for name in sorted(null_nodes, key=lambda p: p.count("/"), reverse=True):
            if name in h5:
                del h5[name]

    return ad.read_h5ad(fixed)


def _scores_from_obsm(
    adata: ad.AnnData, obsm_key: str, prefix: str,
) -> pd.DataFrame:
    """Pull a (cells x k) DataFrame out of ``adata.obsm[obsm_key]``.

    Raises ``KeyError`` when the key is missing.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"Expected obsm[{obsm_key!r}]")
    X = np.asarray(adata.obsm[obsm_key])
    k = int(X.shape[1])
    cols = [f"{prefix}{i+1}" for i in range(k)]
    return pd.DataFrame(X, index=adata.obs_names.astype(str), columns=cols), cols


def _maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def save_pair(fig, stem, out_dir, *, dpi=400, bbox_inches="tight", pad_inches=0.04):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(pdf, bbox_inches=bbox_inches, pad_inches=pad_inches)
    display(fig)
    plt.close(fig)
    return png, pdf


# ════════════════════════════════════════════════════════════════════════════
# Per-method loaders
# ════════════════════════════════════════════════════════════════════════════

def load_stgp(result_dir: str | Path, *, celltype: str) -> MethodResult:
    """Load stGP results. Replaces the default ``stGP{i}`` column names with
    the W.csv row labels (so callers see the same labels everywhere).
    """
    result_dir = Path(result_dir)
    adata = read_h5ad_compat(result_dir / "adata_with_scores.h5ad")
    scores, default_cols = _scores_from_obsm(adata, "X_stgp", "stGP")

    gene_weights = _maybe_read_csv(result_dir / "W.csv")
    if gene_weights is not None and gene_weights.shape[0] == len(default_cols):
        scores.columns = [str(c) for c in gene_weights.index.tolist()]

    return MethodResult("stGP", str(celltype), result_dir, adata, scores, gene_weights)


def load_spatialpca(result_dir: str | Path, *, celltype: str) -> MethodResult:
    """SpatialPCA stores loadings as (genes x SPCA), so transpose them."""
    result_dir = Path(result_dir)
    adata = read_h5ad_compat(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_spatialpca", "SPCA")

    gene_weights = None
    raw = _maybe_read_csv(result_dir / "W_loadings.csv")
    if raw is not None:
        gene_weights = raw.T.copy()
        gene_weights.index = [str(x) for x in gene_weights.index]
        gene_weights.columns = [str(x) for x in gene_weights.columns]

    return MethodResult("SpatialPCA", str(celltype), result_dir, adata, scores, gene_weights)


def load_mefisto(result_dir: str | Path, *, celltype: str) -> MethodResult:
    result_dir = Path(result_dir)
    adata = read_h5ad_compat(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_mefisto", "MEFISTO")
    gene_weights = _maybe_read_csv(result_dir / "weights.csv")
    return MethodResult("MEFISTO", str(celltype), result_dir, adata, scores, gene_weights)


def load_stamp(result_dir: str | Path, *, celltype: str) -> MethodResult:
    result_dir = Path(result_dir)
    adata = read_h5ad_compat(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_stamp", "STAMP")
    gene_weights = _maybe_read_csv(result_dir / "W_loadings.csv")
    return MethodResult("STAMP", str(celltype), result_dir, adata, scores, gene_weights)


# ════════════════════════════════════════════════════════════════════════════
# Popari needs special handling for older anndata versions
# ════════════════════════════════════════════════════════════════════════════

def _load_popari_h5py(h5ad_path: Path) -> ad.AnnData:
    """h5py fallback for Popari ``.h5ad`` files that contain null-encoded fields.

    anndata < 0.12 does not register a reader for
    ``IOSpec(encoding_type='null', encoding_version='0.1.0')``, which Popari
    uses to store Python ``None`` values (e.g. ``uns['log1p']['base']``,
    ``uns['popari_hyperparameters']['spatial_affinity_constraint']``). These
    fields carry no analytical content; we only need ``obsm['X']`` and
    ``uns['M']``, so we read those directly via h5py and construct a minimal
    AnnData, bypassing the problematic fields entirely.
    """
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(h5ad_path, "r") as f:
        # obs_names
        obs_idx_raw = f["obs"]["_index"][:]
        obs_names = pd.Index(
            [x.decode() if isinstance(x, bytes) else str(x) for x in obs_idx_raw]
        )

        # obs columns (plain datasets and HDF5 categoricals).
        obs_dict: dict = {}
        for col in f["obs"]:
            if col.startswith("_"):
                continue
            try:
                ds = f["obs"][col]
                if isinstance(ds, h5py.Dataset):
                    raw = ds[:]
                    if raw.dtype.kind in ("S", "O"):
                        raw = [x.decode() if isinstance(x, bytes) else str(x)
                               for x in raw]
                    obs_dict[col] = raw
                elif isinstance(ds, h5py.Group) and "codes" in ds and "categories" in ds:
                    cats = [x.decode() if isinstance(x, bytes) else str(x)
                            for x in ds["categories"][:]]
                    obs_dict[col] = pd.Categorical.from_codes(ds["codes"][:],
                                                                categories=cats)
            except Exception:
                pass
        obs_df = pd.DataFrame(obs_dict, index=obs_names)

        # var_names
        var_idx_raw = f["var"]["_index"][:]
        var_names = pd.Index(
            [x.decode() if isinstance(x, bytes) else str(x) for x in var_idx_raw]
        )

        # obsm['X'] -- the program embedding.
        obsm: dict = {}
        if "obsm" in f and "X" in f["obsm"]:
            obsm["X"] = np.asarray(f["obsm"]["X"])

        # uns['M'] -- per-replicate gene-weight matrices.
        uns: dict = {}
        if "uns" in f and "M" in f["uns"]:
            try:
                M_item = f["uns"]["M"]
                uns["M"] = (
                    {k: np.asarray(M_item[k]) for k in M_item}
                    if isinstance(M_item, h5py.Group)
                    else np.asarray(M_item)
                )
            except Exception:
                pass

    adata = ad.AnnData(
        X=csr_matrix((len(obs_names), len(var_names)), dtype=np.float32),
        obs=obs_df,
        var=pd.DataFrame(index=var_names),
    )
    for key, val in obsm.items():
        adata.obsm[key] = val
    adata.uns.update(uns)
    return adata


def _popari_gene_weights_from_uns_M(adata: ad.AnnData, k: int,
                                      cols: list[str]) -> pd.DataFrame | None:
    """Reconstruct a (programs x genes) DataFrame from Popari's ``uns['M']``.

    ``uns['M']`` may be either a plain (genes x k) or (k x genes) array, or a
    dict of per-replicate matrices that we average. Returns None when the
    layout is unrecognisable.
    """
    if "M" not in adata.uns:
        return None
    try:
        M_obj = adata.uns["M"]
        if isinstance(M_obj, dict) and len(M_obj) > 0:
            mats = [np.asarray(v) for v in M_obj.values()]
            if len({m.shape for m in mats}) != 1:
                return None
            M = np.mean(np.stack(mats, axis=0), axis=0)
            if M.ndim == 2 and M.shape[1] == k and M.shape[0] == adata.n_vars:
                return pd.DataFrame(M.T, index=cols,
                                     columns=adata.var_names.astype(str))
            return None

        M = np.asarray(M_obj)
        if M.ndim == 2 and M.shape == (k, adata.n_vars):
            return pd.DataFrame(M, index=cols, columns=adata.var_names.astype(str))
        if M.ndim == 2 and M.shape == (adata.n_vars, k):
            return pd.DataFrame(M.T, index=cols, columns=adata.var_names.astype(str))
        return None
    except Exception:
        return None


def load_popari(result_dir: str | Path, *, celltype: str) -> MethodResult:
    result_dir = Path(result_dir)
    h5ad_path = result_dir / "res_popari.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing file: {h5ad_path}")

    try:
        adata = read_h5ad_compat(h5ad_path, cache_dir=result_dir)
    except Exception as e:
        # anndata < 0.12 raises IORegistryError for null-encoded fields.
        if "null" in str(e).lower() or "IOSpec" in str(e) or "No read method" in str(e):
            adata = _load_popari_h5py(h5ad_path)
        else:
            raise

    scores, cols = _scores_from_obsm(adata, "X", "Popari")
    gene_weights = _popari_gene_weights_from_uns_M(adata, len(cols), cols)
    return MethodResult("Popari", str(celltype), result_dir, adata, scores, gene_weights)


# ════════════════════════════════════════════════════════════════════════════
# Public dispatcher
# ════════════════════════════════════════════════════════════════════════════

_LOADERS = {
    "stGP":       load_stgp,
    "SpatialPCA": load_spatialpca,
    "MEFISTO":    load_mefisto,
    "STAMP":      load_stamp,
    "Popari":     load_popari,
}

MethodName = Literal["stGP", "SpatialPCA", "MEFISTO", "STAMP", "Popari"]


def load_method(method: MethodName, result_dir: str | Path,
                 *, celltype: str) -> MethodResult:
    """Dispatch to the right ``load_*`` function based on ``method``."""
    if method not in _LOADERS:
        raise ValueError(f"Unsupported method: {method}")
    return _LOADERS[method](result_dir, celltype=celltype)


# === plotting/metrics.py ===
"""OLS variance partition: how much of a per-cell score is explained by age,
by region, and by both jointly?

Used by ``MouseBrain_microglia.ipynb`` (post-hoc partition of per-method scores).
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarPartUniqueResult:
    age: float       # variance unique to age
    region: float    # variance unique to region
    shared: float    # variance jointly explained by age + region
    residuals: float # 1 - r2_full
    r2: float        # r-squared of the full model


def _empty_result() -> VarPartUniqueResult:
    nan = float("nan")
    return VarPartUniqueResult(age=nan, region=nan, shared=nan,
                                 residuals=nan, r2=nan)


def variance_partition_age_region_unique(
    y: np.ndarray,
    *,
    age: np.ndarray,
    region: np.ndarray,
) -> VarPartUniqueResult:
    """Decompose Var(y) into age-only, region-only, shared, and residuals.

    Computed via three OLS fits:
        full        = y ~ age + C(region)
        age_only    = y ~ age
        region_only = y ~ C(region)

    age_unique    = max(r2_full - r2_region_only, 0)
    region_unique = max(r2_full - r2_age_only,    0)
    shared        = max(r2_full - age_unique - region_unique, 0)
    residuals     = max(1 - r2_full, 0)

    Each component is clamped to [0, 1].
    """
    import statsmodels.formula.api as smf

    df = pd.DataFrame({
        "y": np.asarray(y, dtype=float).ravel(),
        "age": pd.to_numeric(np.asarray(age).ravel(), errors="coerce"),
        "region": pd.Series(np.asarray(region).ravel()).astype("category"),
    }).dropna()

    if df.shape[0] < 5 or df["region"].nunique() < 2:
        return _empty_result()

    try:
        full = smf.ols("y ~ age + C(region)", data=df).fit()
        age_only = smf.ols("y ~ age", data=df).fit()
        region_only = smf.ols("y ~ C(region)", data=df).fit()
    except Exception:
        return _empty_result()

    r2_full = float(full.rsquared)
    r2_age = float(age_only.rsquared)
    r2_region = float(region_only.rsquared)

    age_unique = max(r2_full - r2_region, 0.0)
    region_unique = max(r2_full - r2_age, 0.0)
    shared = max(r2_full - age_unique - region_unique, 0.0)
    residuals = max(1.0 - r2_full, 0.0)

    def _clamp(x: float) -> float:
        return float(min(max(x, 0.0), 1.0))

    return VarPartUniqueResult(
        age=_clamp(age_unique), region=_clamp(region_unique),
        shared=_clamp(shared), residuals=_clamp(residuals), r2=_clamp(r2_full),
    )


def aggregate_by_mouse_region(
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    mouse_col: str = "mouse_id",
    age_col: str = "age",
    region_col: str = "region",
) -> pd.DataFrame:
    """Mean per-cell scores within each (mouse, region) cell.

    The variance-partition fits are then run on these aggregated points so
    every (mouse, region) bucket carries equal statistical weight.
    """
    needed = {mouse_col, age_col, region_col}
    missing = [c for c in needed if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")

    df = pd.concat([obs[[mouse_col, age_col, region_col]].copy(),
                     scores.copy()], axis=1)
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    for c in scores.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[mouse_col, age_col, region_col])

    grp = df.groupby([mouse_col, region_col], observed=True)
    out = grp.mean(numeric_only=True).reset_index()
    out[region_col] = out[region_col].astype("category")
    return out


# === plotting/program_metrics.py ===
"""Similarity metrics between two gene-weight matrices (W) of different methods.

Used by ``MouseBrain_microglia.ipynb`` to ask "how well do baseline programs recover
stGP programs?".
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProgramMatch:
    left_program: str
    right_program: str
    similarity: float


def standardize_gene_weights(
    gene_weights: pd.DataFrame, *, signed: bool,
    nonneg: bool = True, row_normalize: bool = True,
) -> pd.DataFrame:
    """Coerce a (programs x genes) matrix into a comparable form.

    Parameters
    ----------
    signed : True for methods whose weights can be negative (SpatialPCA,
        MEFISTO, STAMP). The absolute value is taken so downstream cosine /
        Jaccard comparisons treat positive- and negative-loading genes
        symmetrically.
    nonneg : True for methods with non-negative weights (Popari, stGP).
        Negative entries (rare; can appear from numerical noise) are clipped
        to zero. Ignored when ``signed=True``.
    row_normalize : if True, scale each row to sum to 1 (so dot-products
        become weighted averages).
    """
    W = gene_weights.copy()
    W.index = W.index.astype(str)
    W.columns = W.columns.astype(str)
    W = W.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if signed:
        W = W.abs()
    elif nonneg:
        W = W.clip(lower=0.0)

    if row_normalize:
        s = W.sum(axis=1).replace(0.0, np.nan)
        W = W.div(s, axis=0).fillna(0.0)
    return W


def _align_columns(Wa: pd.DataFrame, Wb: pd.DataFrame):
    """Restrict both matrices to their shared genes."""
    genes = sorted(set(Wa.columns.astype(str)) & set(Wb.columns.astype(str)))
    if not genes:
        raise ValueError("No shared genes between the two weight matrices.")
    return Wa[genes].copy(), Wb[genes].copy(), genes


def cosine_similarity_matrix(Wa: pd.DataFrame, Wb: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine similarity between rows of Wa and Wb.

    Restricted to the shared gene set; clipped to ``[0, 1]`` (negative values
    are unreachable here because ``standardize_gene_weights`` produces
    non-negative rows).
    """
    Wa2, Wb2, _ = _align_columns(Wa, Wb)
    A = Wa2.to_numpy(dtype=float)
    B = Wb2.to_numpy(dtype=float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(A @ B.T, 0.0, 1.0)
    return pd.DataFrame(sim, index=Wa.index.astype(str), columns=Wb.index.astype(str))


def jaccard_top_genes_matrix(Wa: pd.DataFrame, Wb: pd.DataFrame,
                              *, top_n: int = 50) -> pd.DataFrame:
    """Pairwise Jaccard similarity of the top-``top_n`` genes per program."""
    Wa2, Wb2, _ = _align_columns(Wa, Wb)
    top_n = max(1, int(top_n))

    sets_a = [set(Wa2.loc[r].sort_values(ascending=False).index[:top_n].tolist())
              for r in Wa2.index]
    sets_b = [set(Wb2.loc[r].sort_values(ascending=False).index[:top_n].tolist())
              for r in Wb2.index]

    out = np.zeros((len(sets_a), len(sets_b)), dtype=float)
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            u = sa | sb
            out[i, j] = (len(sa & sb) / len(u)) if u else 0.0
    return pd.DataFrame(out, index=Wa.index.astype(str), columns=Wb.index.astype(str))


def hungarian_match(sim: pd.DataFrame) -> list[ProgramMatch]:
    """One-to-one program assignment that maximises total similarity (linear sum
    assignment). Matches are returned sorted by similarity, descending.
    """
    from scipy.optimize import linear_sum_assignment

    S = sim.to_numpy(dtype=float)
    cost = 1.0 - S
    r, c = linear_sum_assignment(cost)

    matches = [
        ProgramMatch(str(sim.index[i]), str(sim.columns[j]), float(S[i, j]))
        for i, j in zip(r.tolist(), c.tolist())
    ]
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


# === plotting/program_plots.py ===
"""Heatmap and dot-plot helpers for program-similarity figures."""


from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_similarity_heatmap(
    sim: pd.DataFrame, *, title: str | None = None,
    annotate: bool = True, out: str | Path | None = None, dpi: int = 400,
) -> plt.Figure:
    """Heatmap of a (left_program x right_program) similarity matrix."""
    import seaborn as sns

    df = sim.copy()
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    fig_w = max(7.5, 0.9 * len(df.columns) + 4.0)
    fig_h = max(6.0, 0.7 * len(df.index) + 3.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    sns.heatmap(
        df, ax=ax, cmap="RdBu_r", vmin=0.0, vmax=1.0,
        square=False, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Similarity"},
        annot=annotate, fmt=".2f", annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.tick_params(axis="y", rotation=0)
    if title:
        ax.set_title(title, pad=18)

    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


def plot_recovery_dotplot(
    best_sim: pd.DataFrame, *, title: str | None = None,
    out: str | Path | None = None, dpi: int = 400,
) -> plt.Figure:
    """Dot plot showing each method's best-match similarity to every stGP program.

    The dot's colour AND size both encode the similarity score (in [0, 1]).
    """
    df = best_sim.copy().fillna(0.0).clip(lower=0.0, upper=1.0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    rows, cols = df.index.tolist(), df.columns.tolist()

    xs, ys, vals = [], [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            xs.append(j)
            ys.append(i)
            vals.append(float(df.loc[r, c]))

    vals_arr = np.asarray(vals, dtype=float)
    sizes = 60.0 + 790.0 * vals_arr

    fig_w = max(7.5, 1.6 * len(cols) + 3.0)
    fig_h = max(6.0, 0.7 * len(rows) + 3.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    sc = ax.scatter(xs, ys, c=vals_arr, s=sizes, cmap="viridis",
                    vmin=0.0, vmax=1.0, edgecolors="black", linewidths=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="Best-match similarity")
    if title:
        ax.set_title(title, pad=18)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", color="0.90", linewidth=1.0)

    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# === plotting/enrichment.py ===
"""MSigDB term cleanup and combined-score enrichment bar-chart panel."""


import re
import textwrap
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── term-name cleanup ────────────────────────────────────────────────────────

# Pathway sources (Hallmark / GO / Reactome / KEGG / WikiPathways)
_PATHWAY_PREFIXES = ("GOBP", "GOCC", "GOMF", "HALLMARK", "REACTOME", "KEGG", "WP")

# M8 cell-atlas sources (longest prefix matched first)
_M8_SOURCE_PREFIXES = (
    "TABULA_MURIS_SENIS_",
    "TABULA_MURIS_",
    "DESCARTES_ORGANOGENESIS_",
    "DESCARTES_FETAL_",
    "DESCARTES_MAIN_FETAL_",
    "DESCARTES_",
    "ZHANG_",
)

# Common abbreviations to restore (case-insensitive whole-word matching).
_ABBREV = (
    (r"\bdna\b",      "DNA"),
    (r"\brna\b",      "RNA"),
    (r"\bmrna\b",     "mRNA"),
    (r"\bt cell\b",   "T cell"),
    (r"\bb cell\b",   "B cell"),
    (r"\bnk cell\b",  "NK cell"),
    (r"\bmhc\b",      "MHC"),
    (r"\btnf\b",      "TNF"),
    (r"\bifn\b",      "IFN"),
    (r"\bil(?=[-\s\d])", "IL"),    # IL-1, IL 6, IL2 …
    (r"\bhif(?=[-\s\d])", "HIF"),  # HIF-1, HIF1 …
    (r"\bopc\b",      "OPC"),
    (r"\bcns\b",      "CNS"),
    (r"\bkras\b",     "KRAS"),
    (r"\begfr\b",     "EGFR"),
    (r"\btp53\b",     "TP53"),
    (r"\bnf-?kb\b",   "NF-κB"),
    # Roman-numeral classes / types (e.g. "class i" -> "class I",
    # "MHC class ib" -> "MHC class Ib", "type iia" -> "type IIa")
    (r"\b(class|type|stage)\s+(i{1,3}|iv|v|vi{0,3}|ix|x)([ab]?)\b",
     lambda m: f"{m.group(1)} {m.group(2).upper()}{m.group(3)}"),
)

# Tissue/adjective synonyms — collapse pairs like "Lung pulmonary" -> "Lung".
_TISSUE_SYN = (
    ("lung",     "pulmonary"),
    ("pancreas", "pancreatic"),
    ("kidney",   "renal"),
    ("heart",    "cardiac"),
    ("liver",    "hepatic"),
    ("brain",    "cerebral"),
    ("colon",    "colonic"),
    ("stomach",  "gastric"),
    ("muscle",   "muscular"),
    ("bone",     "osseous"),
)
_SYN_MAP: dict[str, str] = {b: a for a, b in _TISSUE_SYN}
_SYN_MAP.update({a: b for a, b in _TISSUE_SYN})


def _dedupe_words(s: str) -> str:
    """Drop adjacent duplicate / synonymous words ('kidney kidney …')."""
    out: list[str] = []
    for w in s.split():
        wl = w.lower()
        if out and (out[-1].lower() == wl or _SYN_MAP.get(out[-1].lower()) == wl):
            continue
        out.append(w)
    return " ".join(out)


def clean_term(term: str, width: int | None = None) -> str:
    """Pretty-print MSigDB-style term IDs (Hallmark / GO / Reactome / M8)."""
    s = str(term)

    # 1) strip pathway-source prefix (Hallmark / GO_BP / Reactome / …)
    s = re.sub(rf"^(?:{'|'.join(_PATHWAY_PREFIXES)})_", "", s)

    # 2) strip cell-atlas source prefix (Tabula Muris / Descartes / Zhang)
    for p in _M8_SOURCE_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break

    # 3) strip uninformative direction suffixes (keep _AGEING — important for analysis)
    s = re.sub(r"_(UP|DN|DOWN)$", "", s)

    # 4) underscores -> spaces, sentence case
    s = s.replace("_", " ").strip()
    s = _dedupe_words(s)
    s = s.lower().capitalize()

    # 5) restore common abbreviations
    for pat, repl in _ABBREV:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    if width is not None:
        s = "\n".join(textwrap.wrap(s, width=width))
    return s


# ── colour palettes ──────────────────────────────────────────────────────────

def _trunc_cmap(name: str, lo: float = 0.15, hi: float = 0.82, n: int = 256):
    """Truncate a stock cmap so the dark end stays soft (Nature-Methods feel)."""
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{name}", base(np.linspace(lo, hi, n))
    )


PANEL_CMAPS: dict[str, mcolors.LinearSegmentedColormap] = {
    "Hallmark":              _trunc_cmap("Blues",   0.22, 0.80),
    "GO Biological process": _trunc_cmap("Reds",    0.22, 0.78),
    "GO Molecular Function": _trunc_cmap("BuGn",    0.22, 0.82),
    "GO Cellular Component": _trunc_cmap("Purples", 0.25, 0.76),
    "Cell-type signatures":  _trunc_cmap("Oranges", 0.23, 0.79),
}


# ── single-panel renderer ────────────────────────────────────────────────────

def plot_enrichment_panel(
    res: pd.DataFrame,
    ax: plt.Axes,
    set_name: str,
    cmap: mcolors.Colormap,
    *,
    n_top: int = 6,
    score_col: str = "Combined Score",
    padj_col: str = "Adjusted P-value",
    term_col: str = "Term",
    padj_threshold: float = 0.1,
) -> None:
    """Single panel: bar width = Combined Score, bar colour = −log10(adj. p-value),
    term names rendered inside the bar to save space.

    Parameters
    ----------
    res : the tidy output of ``gseapy.enrich(...).res2d`` (already ``Term``-cleaned).
    ax : axes to draw into.
    set_name : gene-set label (used as the panel title).
    cmap : sequential colormap to encode -log10(padj).
    n_top : keep the top-``n_top`` terms by combined score.
    padj_threshold : drop terms with ``padj`` above this.
    """
    df = res.dropna(subset=[score_col, padj_col]).copy()
    df = df[(df[score_col] > 0) & (df[padj_col] > 0) & (df[padj_col] < padj_threshold)]

    ax.set_title(set_name, loc="center", fontsize=11.5, weight="bold", pad=4)

    if df.empty:
        ax.text(0.5, 0.5, "No significant terms",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="0.55")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return

    df = df.sort_values(score_col, ascending=False).head(n_top).iloc[::-1]
    terms = df[term_col].astype(str).values
    scores = df[score_col].to_numpy(dtype=float)
    nlogp = -np.log10(np.clip(df[padj_col].to_numpy(dtype=float), 1e-50, None))

    vmin = max(float(np.floor(nlogp.min())), 1.0)
    vmax = max(float(np.ceil(nlogp.max())), vmin + 0.5)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(nlogp))

    y = np.arange(len(terms))
    ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.5, height=0.78)

    xmax = float(scores.max())
    longest_term = max((len(t) for t in terms), default=0)
    # heuristic: ~1.2× max bar, plus extra room for long labels
    xlim_mult = 1.18 + 0.012 * max(0, longest_term - 28)
    ax.set_xlim(0, xmax * xlim_mult)
    ax.set_xlabel("Combined score", fontsize=10)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=2.5, width=0.6, labelsize=9, pad=2)

    # in-bar labels — dark text with a soft white halo
    pad = 0.012 * xmax
    halo = [pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)]
    for i, term in enumerate(terms):
        ax.text(pad, i, term, va="center", ha="left",
                color="#1a1a1a", fontsize=8.5, clip_on=False,
                path_effects=halo)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, aspect=12, pad=0.015)
    cbar.set_label(r"$-\log_{10}$(adj. $p$-value)", fontsize=8.5)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=8)
    cbar.outline.set_linewidth(0.5)


# ── per-program multi-panel figure ──────────────────────────────────────────

def run_enrichment_for_program(
    *,
    gene_list: list[str],
    background_genes: list[str],
    gene_sets: dict[str, str | Path],
    padj_threshold: float = 0.1,
    n_top: int = 6,
    title: str | None = None,
) -> tuple[plt.Figure, list[pd.DataFrame]]:
    """Run gseapy.enrich against every gene-set in ``gene_sets`` and return a
    single multi-panel figure + a list of per-set result DataFrames.

    The returned DataFrames have ``Term`` already cleaned and a ``gene_set``
    column appended so they can be concatenated across panels.
    """
    import gseapy as gp

    n_sets = len(gene_sets)
    fig, axes = plt.subplots(
        n_sets, 1,
        figsize=(5.4, 2.5 * n_sets),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    results: list[pd.DataFrame] = []
    for ax, (set_name, gmt_path) in zip(axes_flat, gene_sets.items()):
        try:
            enr = gp.enrich(
                gene_list=gene_list,
                gene_sets=str(gmt_path),
                background=background_genes,
                verbose=False,
            )
            res = enr.res2d.copy()
        except Exception as e:
            ax.text(0.5, 0.5, f"enrichment failed:\n{e}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#b23")
            ax.set_title(set_name, loc="center", fontsize=11.5,
                         weight="bold", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        res["gene_set"] = set_name
        res["Term"] = res["Term"].apply(clean_term)
        results.append(res)

        cmap = PANEL_CMAPS.get(set_name, _trunc_cmap("Greys", 0.25, 0.85))
        plot_enrichment_panel(res, ax, set_name, cmap=cmap,
                              n_top=n_top, padj_threshold=padj_threshold)

    return fig, results


__all__ = [
    "clean_term",
    "plot_enrichment_panel",
    "run_enrichment_for_program",
    "PANEL_CMAPS",
]


# === plotting/plots.py ===
"""Self-contained plotting routines used across the MERFISH analysis pipeline.

The module is grouped by figure category:
    - kernel diagnostics (temporal / spatial)
    - per-program weighted scores by age
    - per-program variance partition
    - gene trajectories over age
    - spatial program maps (stGP and Popari, signed and unsigned)
    - alpha(t) curves
    - spatial clustering tiles
    - W program heatmap
    - runtime comparison
    - active-gene dot plots
"""


import re
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, out: str | Path | None, *, dpi: int = 400) -> None:
    """Persist ``fig`` to ``out`` (creates parents); no-op when ``out`` is None."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")

def _smooth_curve(x: np.ndarray, y: np.ndarray, n_grid: int = 300):
    """Return ``(x_fine, y_smooth)`` via a smoothing spline; falls back to
    linear interpolation when too few points for a spline.
    """
    from scipy.interpolate import interp1d, make_smoothing_spline

    x_fine = np.linspace(x.min(), x.max(), n_grid)
    if len(x) < 4:
        return x_fine, interp1d(x, y, kind="linear", fill_value="extrapolate")(x_fine)
    try:
        return x_fine, make_smoothing_spline(x, y)(x_fine)
    except Exception:
        return x_fine, interp1d(x, y, kind="linear", fill_value="extrapolate")(x_fine)


def _mean_by_group(X: np.ndarray, group_ids: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Per-group mean expression, used by ``plot_gene_trajectories_over_age``."""
    out = np.zeros((uniq.shape[0], X.shape[1]), dtype=float)
    for i, g in enumerate(uniq):
        idx = np.where(group_ids == g)[0]
        out[i, :] = np.nanmean(X[idx], axis=0) if idx.size > 0 else np.nan
    return out


def _program_label(prog_name: str) -> str:
    """Pretty label for a program: e.g. ``stGP3 -> 'Program 3'``."""
    m = re.search(r"\d+$", str(prog_name))
    return f"Program {m.group()}" if m else str(prog_name)


def _bg_per_mouse(adata_full, mouse_ids_target) -> dict:
    """Per-mouse spatial coordinates of every cell in ``adata_full`` (used as
    grey background context behind the target-cell-type overlay)."""
    if adata_full is None:
        return {}
    bg_sp_all = np.asarray(adata_full.obsm["spatial"])
    bg_mouse_ids = adata_full.obs["mouse_id"].astype(str).to_numpy()
    out: dict = {}
    for mid in np.unique(mouse_ids_target):
        mask = bg_mouse_ids == mid
        if mask.any():
            out[mid] = bg_sp_all[mask]
    return out


def _ordered_mice_by_age(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mouse/slice IDs sorted by age."""
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()
    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([
        float(obs.loc[obs["mouse_id"].astype(str) == m, "age"].iloc[0])
        for m in uniq_mice
    ])
    order = np.argsort(age_per_mouse)
    return uniq_mice[order], age_per_mouse[order]


def _normalise_slice_xy(xy: np.ndarray, ref_xy: np.ndarray | None = None) -> np.ndarray:
    """Center and scale one slice so non-registered mice can be stacked."""
    ref = xy if ref_xy is None or len(ref_xy) == 0 else np.asarray(ref_xy, dtype=float)
    centre = np.nanmedian(ref, axis=0)
    centred = np.asarray(xy, dtype=float) - centre
    ref_centred = ref - centre
    radius = np.nanpercentile(np.linalg.norm(ref_centred, axis=1), 95)
    if not np.isfinite(radius) or radius < 1e-12:
        radius = float(np.nanmax(np.ptp(ref, axis=0)))
    if not np.isfinite(radius) or radius < 1e-12:
        radius = 1.0
    return centred / radius


def _select_mice_by_target_ages(
    obs: pd.DataFrame,
    target_ages: Iterable[float],
    *,
    age_col: str = "age",
    mouse_col: str = "mouse_id",
) -> list[tuple[str, float, float]]:
    """Return `(mouse_id, observed_age, target_age)` closest to each target age."""
    mouse_ids = obs[mouse_col].astype(str).to_numpy()
    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([
        float(obs.loc[obs[mouse_col].astype(str) == m, age_col].iloc[0])
        for m in uniq_mice
    ])
    order = np.argsort(age_per_mouse)
    mouse_ids = uniq_mice[order]
    age_per_mouse = age_per_mouse[order]
    out: list[tuple[str, float, float]] = []
    used: set[str] = set()
    for target in target_ages:
        ages = age_per_mouse.copy()
        order = np.argsort(np.abs(ages - float(target)))
        picked = None
        for idx in order:
            mid = str(mouse_ids[idx])
            if mid not in used:
                picked = (mid, float(age_per_mouse[idx]), float(target))
                break
        if picked is not None:
            used.add(picked[0])
            out.append(picked)
    return out


def _style_spacetime_axes(ax, *, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.68))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="z", which="major", labelsize=10, pad=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#BFBFBF")
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 0.75)
        axis._axinfo["grid"]["linewidth"] = 0.6
    for axis in (ax.xaxis, ax.yaxis):
        axis.line.set_color((1, 1, 1, 0))
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
    ax.zaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.0


def plot_spacetime_cluster_stack(
    *,
    adata,
    cluster_labels: np.ndarray,
    adata_full=None,
    method_name: str = "",
    celltype: str = "",
    bg_dot_size: float = 0.35,
    fg_dot_size: float = 4.0,
    z_gap: float = 0.22,
    max_bg_per_slice: int | None = 6000,
    max_fg_per_slice: int | None = None,
    palette: str = "tab20",
    elev: float = 0,
    azim: float = -58,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """3D stack of all slices coloured by discrete domains.

    Each mouse is centered and scaled independently before stacking. This does
    not imply registration; it is a visual summary of age-ordered spatial
    organisation across comparable coronal brain slices.
    """
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"], dtype=float)
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()
    cluster_labels = np.asarray(cluster_labels).astype(str)
    if cluster_labels.shape[0] != adata.n_obs:
        raise ValueError("cluster_labels length must match adata.n_obs")

    uniq_mice, age_per_mouse = _ordered_mice_by_age(obs)
    bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
    rng = np.random.default_rng(0)

    uniq_clusters = pd.Index(pd.unique(cluster_labels)).astype(str)
    uniq_clusters = np.array(sorted(uniq_clusters, key=lambda x: (len(x), x)))
    cmap = plt.get_cmap(palette, max(len(uniq_clusters), 3))
    cluster_colors = {
        c: mcolors.to_hex(cmap(i / max(len(uniq_clusters) - 1, 1)))
        for i, c in enumerate(uniq_clusters)
    }

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    z_positions = np.arange(len(uniq_mice), dtype=float) * z_gap

    for z, mid in zip(z_positions, uniq_mice):
        fg_mask = mouse_ids == mid
        if not fg_mask.any():
            continue
        ref_xy = bg_by_mouse.get(mid, sp[fg_mask])

        if mid in bg_by_mouse:
            bg_xy = _normalise_slice_xy(bg_by_mouse[mid], ref_xy)
            bg_sel = np.arange(bg_xy.shape[0])
            if max_bg_per_slice is not None and bg_xy.shape[0] > max_bg_per_slice:
                bg_sel = np.sort(
                    rng.choice(bg_xy.shape[0], size=max_bg_per_slice, replace=False)
                )
            ax.scatter(
                bg_xy[bg_sel, 0], bg_xy[bg_sel, 1], np.full(bg_sel.size, z),
                c="#D8D8D8", s=bg_dot_size, alpha=0.10,
                linewidths=0, depthshade=False, rasterized=True, zorder=1,
            )

        fg_xy = _normalise_slice_xy(sp[fg_mask], ref_xy)
        fg_labels = cluster_labels[fg_mask]
        fg_sel = np.arange(fg_xy.shape[0])
        if max_fg_per_slice is not None and fg_xy.shape[0] > max_fg_per_slice:
            fg_sel = np.sort(
                rng.choice(fg_xy.shape[0], size=max_fg_per_slice, replace=False)
            )
        fg_xy = fg_xy[fg_sel]
        fg_labels = fg_labels[fg_sel]
        for c in uniq_clusters:
            cmask = fg_labels == c
            if not cmask.any():
                continue
            ax.scatter(
                fg_xy[cmask, 0], fg_xy[cmask, 1], np.full(int(cmask.sum()), z),
                c=cluster_colors[c], s=fg_dot_size, alpha=0.86,
                linewidths=0, depthshade=False, rasterized=True, zorder=2,
            )

    tick_step = max(1, int(np.ceil(len(uniq_mice) / 6)))
    tick_idx = np.arange(0, len(uniq_mice), tick_step)
    if tick_idx[-1] != len(uniq_mice) - 1:
        tick_idx = np.r_[tick_idx, len(uniq_mice) - 1]
    ax.set_zticks(z_positions[tick_idx])
    ax.set_zticklabels([f"{age_per_mouse[i]:.1f} mo" for i in tick_idx])
    _style_spacetime_axes(ax, elev=elev, azim=azim)

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=cluster_colors[c],
                   markersize=6, label=f"Domain {c}")
        for c in uniq_clusters
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, title=f"{method_name} domains" if method_name else "Domains",
    )

    _save(fig, out, dpi=dpi)
    return fig


def plot_spacetime_embedding_stack(
    *,
    adata,
    values: np.ndarray,
    adata_full=None,
    value_label: str = "embedding",
    bg_dot_size: float = 0.35,
    fg_dot_size: float = 4.0,
    z_gap: float = 0.22,
    max_bg_per_slice: int | None = 6000,
    max_fg_per_slice: int | None = None,
    cmap: str = "RdBu_r",
    color_scale: ColorScale = "symmetric",
    elev: float = 0,
    azim: float = -58,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """3D stack of all slices coloured by one continuous embedding dimension."""
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"], dtype=float)
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()
    values = np.asarray(values, dtype=float).ravel()
    if values.shape[0] != adata.n_obs:
        raise ValueError("values length must match adata.n_obs")

    if color_scale == "symmetric":
        vmax = float(np.nanpercentile(np.abs(values), 99))
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(values, 1))
        vmax = float(np.nanpercentile(values, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    uniq_mice, age_per_mouse = _ordered_mice_by_age(obs)
    bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.13)
    ax = fig.add_subplot(111, projection="3d")
    z_positions = np.arange(len(uniq_mice), dtype=float) * z_gap
    sc_ref = None

    for z, mid in zip(z_positions, uniq_mice):
        fg_mask = mouse_ids == mid
        if not fg_mask.any():
            continue
        ref_xy = bg_by_mouse.get(mid, sp[fg_mask])

        if mid in bg_by_mouse:
            bg_xy = _normalise_slice_xy(bg_by_mouse[mid], ref_xy)
            bg_sel = np.arange(bg_xy.shape[0])
            if max_bg_per_slice is not None and bg_xy.shape[0] > max_bg_per_slice:
                bg_sel = np.sort(
                    rng.choice(bg_xy.shape[0], size=max_bg_per_slice, replace=False)
                )
            ax.scatter(
                bg_xy[bg_sel, 0], bg_xy[bg_sel, 1], np.full(bg_sel.size, z),
                c="#D8D8D8", s=bg_dot_size, alpha=0.10,
                linewidths=0, depthshade=False, rasterized=True, zorder=1,
            )

        fg_xy = _normalise_slice_xy(sp[fg_mask], ref_xy)
        fg_vals = values[fg_mask]
        fg_sel = np.arange(fg_xy.shape[0])
        if max_fg_per_slice is not None and fg_xy.shape[0] > max_fg_per_slice:
            fg_sel = np.sort(
                rng.choice(fg_xy.shape[0], size=max_fg_per_slice, replace=False)
            )
        sc_ref = ax.scatter(
            fg_xy[fg_sel, 0], fg_xy[fg_sel, 1], np.full(fg_sel.size, z),
            c=fg_vals[fg_sel], cmap=cmap, vmin=vmin, vmax=vmax,
            s=fg_dot_size, alpha=0.88, linewidths=0,
            depthshade=False, rasterized=True, zorder=2,
        )

    tick_step = max(1, int(np.ceil(len(uniq_mice) / 6)))
    tick_idx = np.arange(0, len(uniq_mice), tick_step)
    if tick_idx[-1] != len(uniq_mice) - 1:
        tick_idx = np.r_[tick_idx, len(uniq_mice) - 1]
    ax.set_zticks(z_positions[tick_idx])
    ax.set_zticklabels([f"{age_per_mouse[i]:.1f} mo" for i in tick_idx])
    _style_spacetime_axes(ax, elev=elev, azim=azim)

    if sc_ref is not None:
        cbar_ax = fig.add_axes([0.30, 0.105, 0.40, 0.020])
        cbar = fig.colorbar(sc_ref, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(value_label)
        cbar.ax.tick_params(labelsize=9)

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Per-program weighted expression score boxplots by age
# ════════════════════════════════════════════════════════════════════════════

def plot_program_weighted_scores_by_age(
    adata,
    W_df: pd.DataFrame,
    *,
    age_col: str = "age",
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
    ncols: int = 2,
) -> plt.Figure:
    """Per-program weighted expression score by age (one subplot per program).

    For each program p and each cell c::

        score(c, p) = sum_g  W[p, g] * ( X[c, g] - mu_g )

    where ``X`` is the *already log-normalised* expression stored in ``adata.X``
    (matching the matrix that stGP was fit on) and ``mu_g`` is the overall
    gene mean (matching stGP's centering step). Per-cell scores are aggregated
    to per-slice means; the slice-level scatter is plotted with a smoothing
    spline trend so the temporal signal is visible without being drowned in
    within-slice noise.

    Layout defaults to a 2-column grid so 4 programs render as 2x2.
    """
    # Use adata.X as-is (already log-normalised by the stGP pipeline) and
    # centre by overall gene means to match the matrix on which stGP fits W.
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)

    var_names = adata.var_names.astype(str).tolist()
    w_genes = set(W_df.columns.astype(str))
    shared = [g for g in var_names if g in w_genes]
    if not shared:
        raise ValueError("No overlapping genes between W and adata.var_names.")

    gene_idx = [var_names.index(g) for g in shared]
    W_arr = W_df[shared].to_numpy(dtype=float)              # (p, n_shared)
    scores = Xc[:, gene_idx] @ W_arr.T                      # (n_cells, p)

    ages = pd.to_numeric(adata.obs[age_col], errors="coerce").to_numpy(float)
    uniq_ages = np.sort(np.unique(ages[np.isfinite(ages)]))

    prog_names = W_df.index.astype(str).tolist()
    p = len(prog_names)
    ncols = max(1, min(ncols, p))
    nrows = int(np.ceil(p / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5.5 + 0.5, nrows * 4.0 + 0.8),
                              constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[p:]:
        ax.axis("off")

    box_color = "#A8C8E1"
    trend_color = "#C0392B"

    for j, prog in enumerate(prog_names):
        ax = axes_flat[j]
        data_by_age, slice_means = [], []
        for age in uniq_ages:
            mask = ages == age
            vals = scores[mask, j] if mask.sum() > 0 else np.array([])
            data_by_age.append(vals)
            slice_means.append(float(np.mean(vals)) if vals.size > 0 else np.nan)

        x_pos = uniq_ages.astype(float)
        # Width of each box scaled to age spacing for a cleaner look.
        box_width = float(np.median(np.diff(x_pos))) * 0.6 if len(x_pos) >= 2 else 0.5

        ax.boxplot(
            data_by_age,
            positions=x_pos,
            widths=box_width,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.2),
            boxprops=dict(facecolor=box_color, alpha=0.6, edgecolor="#3A6A8E"),
            flierprops=dict(marker=".", markersize=1.0, alpha=0.15,
                            markerfacecolor="#888888", markeredgewidth=0),
            whiskerprops=dict(lw=0.7, color="#3A6A8E"),
            capprops=dict(lw=0.7, color="#3A6A8E"),
            showfliers=False,
        )

        slice_means_arr = np.array(slice_means, dtype=float)
        valid = np.isfinite(slice_means_arr)
        if valid.sum() >= 2:
            xv, yv = x_pos[valid], slice_means_arr[valid]
            ax.scatter(xv, yv, color=trend_color, s=42, zorder=5,
                       edgecolor="white", linewidth=0.8, label="Slice mean")
            try:
                x_fine, y_smooth = _smooth_curve(xv, yv)
                ax.plot(x_fine, y_smooth, color=trend_color, lw=2.4, zorder=4,
                        label="Smoothed trend")
            except Exception:
                ax.plot(xv, yv, color=trend_color, lw=2.0, zorder=4)

        ax.set_xlim(uniq_ages.min() - box_width, uniq_ages.max() + box_width)
        # Continuous x-axis with a few well-spaced ticks rather than one tick
        # per slice (per-slice ticks overlap at the young end where samples
        # are clustered).
        x_min, x_max = float(uniq_ages.min()), float(uniq_ages.max())
        nticks = 7
        ax.set_xticks(np.linspace(x_min, x_max, nticks))
        ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(x_min, x_max, nticks)])
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Weighted score (centered)")
        ax.set_title(_program_label(prog))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="#888888", lw=0.6, ls=":", zorder=1)

        if j == 0:
            ax.legend(loc="best", fontsize=8, frameon=False)

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Per-program variance decomposition stacked bar
# ════════════════════════════════════════════════════════════════════════════

def plot_program_variance_partition(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Horizontal stacked bar chart: one bar per stGP program, split by
    variance components (proportions, each row sums to 1).

    Accepts either the model-based GP columns (``sigma2_age``, ``tau2_spa``,
    ``sigma2_e``) or the post-hoc OLS columns (``Age``, ``Region``, ``Both``,
    ``Residuals``).
    """
    colors = VarPartColors()

    # Prefer GP model columns when present; sigma2_e is optional (excluded
    # when the caller computes signal-only proportions).
    if "sigma2_age" in df.columns:
        _gp_cols = ["sigma2_age", "tau2_spa", "sigma2_e"]
        _gp_colors = [colors.age, colors.region, colors.residuals]
        _gp_labels = [r"$\sigma^2_\mathrm{age}$",
                       r"$\tau^2_\mathrm{spa}$",
                       r"$\sigma^2_e$"]
        components = [c for c in _gp_cols if c in df.columns]
        comp_colors = [col for c, col in zip(_gp_cols, _gp_colors) if c in df.columns]
        comp_labels = [lbl for c, lbl in zip(_gp_cols, _gp_labels) if c in df.columns]
    else:
        components = ["Age", "Region", "Both", "Residuals"]
        comp_colors = [colors.age, colors.region, colors.both, colors.residuals]
        comp_labels = components

    programs = df.index.tolist()
    n = len(programs)
    prog_labels = [_program_label(p) for p in programs]

    fig, ax = plt.subplots(figsize=(7.0, max(3.5, 0.90 * n + 1.8)),
                           constrained_layout=True)
    y = np.arange(n)
    h = 0.55

    left = np.zeros(n)
    for comp, color, label in zip(components, comp_colors, comp_labels):
        if comp not in df.columns:
            continue
        vals = df[comp].fillna(0).clip(lower=0).to_numpy(float) * 100.0
        ax.barh(y, vals, height=h, left=left, color=color, label=label, linewidth=0)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 6:
                ax.text(l + v / 2, y[i], f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        left = left + vals

    ax.set_yticks(y)
    ax.set_yticklabels(prog_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Variance explained (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.22),
              ncol=len(components), frameon=False)

    if title:
        ax.set_title(title, pad=8)

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Gene trajectories over age
# ════════════════════════════════════════════════════════════════════════════

def plot_gene_trajectories_over_age(
    *, adata, genes: Iterable[str],
    gene_weights: pd.Series | None = None,
    group_col: str = "mouse_id", age_col: str = "age",
    title: str | None = None,
    out: str | Path | None = None, dpi: int = 400,
) -> plt.Figure:
    """Per-gene z-scored expression trajectory across mice (sorted by age)."""
    genes = [g for g in genes if g in set(adata.var_names.astype(str))]
    if len(genes) == 0:
        raise ValueError("No requested genes found in adata.var_names.")

    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    group_ids = adata.obs[group_col].astype(str).to_numpy()
    uniq_groups = np.unique(group_ids)
    ages = (adata.obs[[group_col, age_col]].dropna()
            .assign(**{group_col: lambda d: d[group_col].astype(str)})
            .drop_duplicates(subset=[group_col]).set_index(group_col)[age_col])
    age_per_group = np.array([float(ages.get(g, np.nan)) for g in uniq_groups])

    order = np.argsort(age_per_group)
    uniq_groups = uniq_groups[order]
    age_per_group = age_per_group[order]

    Xg = _mean_by_group(X, group_ids, uniq_groups)
    mu = np.nanmean(Xg, axis=0, keepdims=True)
    sd = np.nanstd(Xg, axis=0, keepdims=True) + 1e-12
    Xz = (Xg - mu) / sd

    if gene_weights is not None:
        w = gene_weights.reindex(genes).fillna(0.0).to_numpy(dtype=float)
        w = np.maximum(w, 0.0)
        if np.allclose(w, 0):
            w = np.arange(len(genes), dtype=float)
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
    else:
        w_norm = np.linspace(0, 1, len(genes))

    cmap = plt.get_cmap("YlOrRd")
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
    for j in range(len(genes)):
        ax.plot(age_per_group, Xz[:, j],
                color=cmap(w_norm[j]), alpha=0.28, lw=1.0, zorder=1)
    ax.plot(age_per_group, np.nanmean(Xz, axis=1), color="#7F0000", lw=2.8, zorder=2)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Expression change (z-score)")
    if title:
        ax.set_title(title)
    ax.text(0.98, 0.08, f"n = {len(genes)}",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Spatial program maps (stGP and Popari)
# ════════════════════════════════════════════════════════════════════════════

ColorScale = Literal["symmetric", "percentile"]


def _plot_spatial_programs_impl(
    *,
    adata,
    scores: pd.DataFrame,
    use_spatial_obsm: bool,
    color_scale: ColorScale,
    adata_full=None,
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """Shared backbone for the stGP / Popari spatial-program tile plots.

    Each panel shows one mouse's tissue section. When ``adata_full`` is given,
    every cell of that mouse is drawn as a faint grey cloud for anatomical
    context; the target cell type is then overlaid and coloured by the
    program score.

    Parameters
    ----------
    use_spatial_obsm : True for the stGP variant (uses ``obsm['X_stgp_spatial']``
        when available); False for Popari (which has no spatial residual).
    color_scale : "symmetric" -- vmin/vmax = +/- 99-percentile of |scores|
                                  (signed scores, e.g. stGP, MEFISTO).
                  "percentile" -- vmin = 1st pct, vmax = 99th pct
                                  (non-negative topic-style scores, e.g. Popari).
    """
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()

    if use_spatial_obsm and "X_stgp_spatial" in adata.obsm:
        # stGP: use the spatial residual b instead of the score H if available.
        b = np.asarray(adata.obsm["X_stgp_spatial"])
        spatial_scores = pd.DataFrame(b, index=scores.index, columns=scores.columns)
    else:
        spatial_scores = scores

    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([
        float(obs.loc[obs["mouse_id"].astype(str) == m, "age"].iloc[0])
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
        if color_scale == "symmetric":
            abs99 = float(np.nanpercentile(np.abs(prog_vals), 99))
            vmin, vmax = -abs99, abs99
        else:
            vmin = float(np.nanpercentile(prog_vals, 1))
            vmax = float(np.nanpercentile(prog_vals, 99))

        nrows = int(np.ceil(n_mice / ncols))
        panel_w, panel_h = 2.4, 2.4
        fig_w = ncols * panel_w + 0.8     # +0.8 reserved for the colorbar
        fig_h = nrows * panel_h + 0.5

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
            ax.set_title(f"{age:.1f} mo", fontsize=9, pad=2)
            ax.axis("off")

        if sc_ref is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
            cbar = fig.colorbar(sc_ref, cax=cbar_ax)
            cbar.set_label(f"{prog} score")
            cbar.ax.tick_params(labelsize=9)

        figs.append(fig)

    return figs


def plot_spatial_programs_selected_slices(
    *,
    adata,
    scores: pd.DataFrame,
    use_spatial_obsm: bool,
    color_scale: ColorScale,
    target_ages: Iterable[float],
    adata_full=None,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 5.0,
    cmap: str = "RdBu_r",
    ncols: int = 2,
) -> list[plt.Figure]:
    """One compact figure per program for a selected set of age slices."""
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()

    if use_spatial_obsm and "X_stgp_spatial" in adata.obsm:
        b = np.asarray(adata.obsm["X_stgp_spatial"])
        spatial_scores = pd.DataFrame(b, index=scores.index, columns=scores.columns)
    else:
        spatial_scores = scores

    selected = _select_mice_by_target_ages(obs, target_ages)
    if not selected:
        return []
    bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
    mouse_mask_cache: dict = {mid: mouse_ids == mid for mid, _, _ in selected}

    figs: list[plt.Figure] = []
    for prog in scores.columns.tolist():
        prog_vals = spatial_scores[prog].to_numpy(dtype=float)
        if color_scale == "symmetric":
            abs99 = float(np.nanpercentile(np.abs(prog_vals), 99))
            vmin, vmax = -abs99, abs99
        else:
            vmin = float(np.nanpercentile(prog_vals, 1))
            vmax = float(np.nanpercentile(prog_vals, 99))

        n_panels = len(selected)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 2.55 + 0.55, nrows * 2.55 + 0.25),
            gridspec_kw={"wspace": 0.04, "hspace": 0.15},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.02, right=0.88, top=0.97, bottom=0.04,
                            wspace=0.04, hspace=0.15)
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_panels:]:
            ax.axis("off")

        sc_ref = None
        for i, (mid, age, _target_age) in enumerate(selected):
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
            ax.set_title(f"{age:.1f} mo", fontsize=10, pad=2)
            ax.axis("off")

        if sc_ref is not None:
            cbar_ax = fig.add_axes([0.905, 0.18, 0.018, 0.64])
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
        adata=stgp_adata, scores=scores,
        use_spatial_obsm=True, color_scale="symmetric",
        adata_full=adata_full, ncols=ncols,
        bg_dot_size=bg_dot_size, fg_dot_size=fg_dot_size, cmap=cmap,
    )


def plot_popari_spatial_programs(
    *,
    popari_adata,
    scores: pd.DataFrame,
    adata_full=None,
    celltype: str = "",
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
    dpi: int = 150,
) -> list[plt.Figure]:
    """One figure per Popari topic, tiling all tissue sections by age.

    Topic scores are non-negative, so the colour scale runs from the 1st to the
    99th percentile (no symmetric divergent range).
    """
    return _plot_spatial_programs_impl(
        adata=popari_adata, scores=scores,
        use_spatial_obsm=False, color_scale="percentile",
        adata_full=adata_full, ncols=ncols,
        bg_dot_size=bg_dot_size, fg_dot_size=fg_dot_size, cmap=cmap,
    )


# ════════════════════════════════════════════════════════════════════════════
# alpha(t)  --  posterior aging trajectory
# ════════════════════════════════════════════════════════════════════════════

def plot_alpha_over_age(
    *, ages: np.ndarray, alpha: np.ndarray,
    alpha_lower: np.ndarray | None = None,
    alpha_upper: np.ndarray | None = None,
    title: str | None = None,
    out: str | Path | None = None, dpi: int = 400,
) -> plt.Figure:
    """Posterior mean aging trajectory with an optional 95% credible-interval band.

    Parameters
    ----------
    ages : (T,) array of calendar ages (months).
    alpha : (T,) posterior mean of the aging effect at each time point.
    alpha_lower, alpha_upper : (T,) arrays giving the lower / upper bounds of
        the pointwise 95% posterior credible interval. When both are supplied
        a shaded band is drawn around the mean curve.
    """
    COLOR = "#2C7FB8"

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
        ax.fill_between(ages, lo, hi, color=COLOR, alpha=0.18, linewidth=0,
                         label="95% posterior CI")
        ax.plot(ages, lo, color=COLOR, lw=0.8, ls="--", alpha=0.55)
        ax.plot(ages, hi, color=COLOR, lw=0.8, ls="--", alpha=0.55)

    ax.plot(ages, alpha, color=COLOR, lw=1.6, zorder=2)
    ax.scatter(ages, alpha, color=COLOR, s=30, zorder=3, alpha=0.9,
                label="Posterior mean")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Age effect")
    ax.grid(False)
    if has_ci:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Spatial clustering (one panel per slice)
# ════════════════════════════════════════════════════════════════════════════

def _cluster_color_lookup(cluster_labels: np.ndarray, palette: str = "tab20") -> tuple:
    uniq_clusters = np.sort(np.unique(cluster_labels))
    n_clusters = len(uniq_clusters)
    cmap = plt.get_cmap(palette, max(n_clusters, 3))
    label_to_color = {lbl: cmap(i / max(n_clusters - 1, 1))
                      for i, lbl in enumerate(uniq_clusters)}
    label_to_idx = {lbl: i for i, lbl in enumerate(uniq_clusters)}
    return uniq_clusters, cmap, label_to_color, label_to_idx


def plot_spatial_cluster_single_slice(
    *,
    adata,
    cluster_labels: np.ndarray,
    mouse_id: str,
    adata_full=None,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 7.0,
    palette: str = "tab20",
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Single-slice cluster/domain map for one mouse ID."""
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()
    cluster_labels = np.asarray(cluster_labels)
    mask = mouse_ids == str(mouse_id)
    if not mask.any():
        raise ValueError(f"mouse_id={mouse_id!r} not found")

    bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
    uniq_clusters, cmap, label_to_color, label_to_idx = _cluster_color_lookup(
        cluster_labels, palette=palette,
    )
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    if str(mouse_id) in bg_by_mouse:
        bx = bg_by_mouse[str(mouse_id)]
        ax.scatter(bx[:, 0], bx[:, 1], c="#E0E0E0", s=bg_dot_size,
                   linewidths=0, rasterized=True, zorder=1)
    fg_color_idx = np.array([label_to_idx[l] for l in cluster_labels[mask]])
    ax.scatter(
        sp[mask, 0], sp[mask, 1],
        c=fg_color_idx, cmap=cmap, vmin=-0.5, vmax=len(uniq_clusters) - 0.5,
        s=fg_dot_size, linewidths=0, rasterized=True, zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_title("stGP", fontsize=11, pad=2)
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=label_to_color[lbl],
                   markersize=6, markeredgecolor="white", markeredgewidth=0.4,
                   label=str(lbl))
        for lbl in uniq_clusters
    ]
    ax.legend(handles=handles, title="Domain", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    _save(fig, out, dpi=dpi)
    return fig


def plot_benchmark_cluster_methods_single_slice(
    *,
    method_adatas: dict[str, object],
    method_cluster_labels: dict[str, np.ndarray],
    mouse_id: str,
    adata_full=None,
    method_order: Iterable[str] = ("SpatialPCA", "MEFISTO", "STAMP", "Popari"),
    ncols: int = 2,
    bg_dot_size: float = 0.25,
    fg_dot_size: float = 5.5,
    palette: str = "tab20",
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Same-slice 2x2 comparison of benchmark-method clustering results."""
    methods = [m for m in method_order if m in method_adatas and m in method_cluster_labels]
    if not methods:
        raise ValueError("No benchmark methods available for this slice comparison.")

    ref_adata = method_adatas[methods[0]]
    ref_obs = ref_adata.obs
    ref_mouse_ids = ref_obs["mouse_id"].astype(str).to_numpy()
    if not np.any(ref_mouse_ids == str(mouse_id)):
        raise ValueError(f"mouse_id={mouse_id!r} not found")
    age = float(ref_obs.loc[ref_mouse_ids == str(mouse_id), "age"].iloc[0])

    n_panels = len(methods)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.45 + 0.35, nrows * 2.55 + 0.15),
        gridspec_kw={"wspace": 0.04, "hspace": 0.18},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04,
                        wspace=0.04, hspace=0.18)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    for ax, method in zip(axes_flat, methods):
        adata = method_adatas[method]
        labels = np.asarray(method_cluster_labels[method])
        obs = adata.obs
        sp = np.asarray(adata.obsm["spatial"])
        mouse_ids = obs["mouse_id"].astype(str).to_numpy()
        mask = mouse_ids == str(mouse_id)
        if not mask.any():
            ax.axis("off")
            continue

        bg_by_mouse = _bg_per_mouse(adata_full, mouse_ids)
        if str(mouse_id) in bg_by_mouse:
            bx = bg_by_mouse[str(mouse_id)]
            ax.scatter(bx[:, 0], bx[:, 1], c="#E0E0E0", s=bg_dot_size,
                       linewidths=0, rasterized=True, zorder=1)

        uniq_clusters, cmap, _label_to_color, label_to_idx = _cluster_color_lookup(
            labels, palette=palette,
        )
        fg_color_idx = np.array([label_to_idx[l] for l in labels[mask]])
        ax.scatter(
            sp[mask, 0], sp[mask, 1],
            c=fg_color_idx, cmap=cmap, vmin=-0.5, vmax=len(uniq_clusters) - 0.5,
            s=fg_dot_size, linewidths=0, rasterized=True, zorder=2,
        )
        ax.set_aspect("equal")
        ax.set_title(method, fontsize=10, pad=2)
        ax.axis("off")

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Spatial kernel correlation (single reference-cell scatter)
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
    """Single-panel spatial kernel correlation to a reference centroid cell."""
    if "age" not in adata.obs.columns:
        raise KeyError("adata.obs missing 'age' column")
    ages = np.sort(adata.obs["age"].astype(float).unique())
    target_age = float(ages[slice_idx])

    mask = adata.obs["age"].astype(float).to_numpy() == target_age
    coords_s = np.asarray(adata.obsm["spatial"][mask], dtype=float)

    # Z-score coordinates so that `bandwidth` (calibrated on z-scored coords
    # inside MouseBrain_microglia.ipynb) is on the correct scale for distance computation.
    mu_s = coords_s.mean(axis=0)
    std_s = coords_s.std(axis=0, ddof=1)
    std_s[std_s < 1e-12] = 1.0
    coords_s = (coords_s - mu_s) / std_s

    centre = coords_s.mean(axis=0)
    ref_cell = int(np.argmin(np.linalg.norm(coords_s - centre, axis=1)))
    d2_full = np.sum((coords_s - coords_s[ref_cell]) ** 2, axis=1)
    k_vals = np.exp(-d2_full / bandwidth)

    fig, ax = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)
    sc = ax.scatter(
        coords_s[:, 0], coords_s[:, 1],
        c=k_vals, cmap="magma", s=18, vmin=0, vmax=1,
        linewidths=0, rasterized=True,
    )
    ax.scatter(
        coords_s[ref_cell, 0], coords_s[ref_cell, 1],
        marker="*", s=220, c="#22D7E6", edgecolors="black",
        linewidths=0.6, zorder=10, label="ref cell",
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("kernel correlation")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# W matrix heatmap (programs x active genes)
# ════════════════════════════════════════════════════════════════════════════

def plot_W_program_heatmap(
    W: pd.DataFrame,
    *,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Heatmap of stGP gene weights with programs on x and active genes on y.
    All-zero gene columns are dropped first.
    """
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

    fig_w = max(3.0, len(programs) * 0.7 + 1.5)
    fig_h = max(4.0, len(genes) / 3.0 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanpercentile(np.abs(W_filtered.values), 99)) or 1.0

    pink_cmap = LinearSegmentedColormap.from_list("pinkish",
                                                    ["white", "#FFD1DC", "red"])
    im = ax.imshow(W_filtered.values.T, aspect="auto", cmap=pink_cmap,
                    vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(programs)))
    ax.set_xticklabels(programs, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    ax.set_xlabel("Program", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("Weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Runtime comparison bar chart
# ════════════════════════════════════════════════════════════════════════════

def plot_runtime_comparison(
    timing_df: pd.DataFrame, *,
    title: str | None = None,
    out: str | Path | None = None, dpi: int = 400,
) -> plt.Figure:
    """One bar per method; height = ``runtime_sec``; colour from ``METHOD_COLORS``."""
    methods = timing_df["method"].tolist()
    times = timing_df["runtime_sec"].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(max(7, 0.9 * len(methods) + 2), 5))
    ax.bar(range(len(methods)), times, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, out, dpi=dpi)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Per-program active-gene dot plot
# ════════════════════════════════════════════════════════════════════════════

def plot_active_gene_dotplot(
    adata,
    *,
    genes: list[str],
    age_col: str = "age",
    n_top: int = 20,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Dot plot of top active genes for one program across ages.

    Each dot encodes:
        - colour: mean z-scored expression (per-gene z-score across ages)
        - size:   fraction of cells expressing (raw expression > 0) at that age

    Parameters
    ----------
    genes : gene names *ordered by weight descending* (top = highest weight).
        Only genes present in ``adata.var_names`` are kept; at most ``n_top``
        are shown.
    """
    var_set = set(adata.var_names.astype(str))
    genes = [g for g in genes if g in var_set][:n_top]
    if not genes:
        raise ValueError("None of the requested genes found in adata.var_names.")

    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    age_vals = pd.to_numeric(adata.obs[age_col], errors="coerce").to_numpy()
    uniq_ages = np.sort(pd.unique(age_vals[np.isfinite(age_vals)]))
    n_g, n_a = len(genes), len(uniq_ages)

    means = np.full((n_g, n_a), np.nan)
    pct_ex = np.zeros((n_g, n_a))
    for j, a in enumerate(uniq_ages):
        mask = age_vals == a
        if not mask.any():
            continue
        blk = X[mask]
        means[:, j] = np.mean(blk, axis=0)
        pct_ex[:, j] = np.mean(blk > 0, axis=0)

    # Per-gene z-score across ages so colour reflects relative temporal change.
    mu = np.nanmean(means, axis=1, keepdims=True)
    sd = np.nanstd(means, axis=1, keepdims=True) + 1e-12
    z = (means - mu) / sd
    # Replace any remaining NaN (e.g. genes with no expression) with 0 to avoid
    # passing all-NaN colour arrays to matplotlib scatter, which triggers an empty
    # array concatenation error inside PathCollection.draw().
    z = np.where(np.isfinite(z), z, 0.0)

    even_xs = np.linspace(0, n_a - 1, n_a)
    xx, yy, cc, ss = [], [], [], []
    for i in range(n_g):
        for j in range(n_a):
            xx.append(even_xs[j])
            yy.append(i)
            cc.append(z[i, j])
            ss.append(pct_ex[i, j])

    ss_arr = np.asarray(ss)
    smin, smax = ss_arr.min(), ss_arr.max()
    sizes = 25 + 350 * (ss_arr - smin) / (smax - smin + 1e-12)

    # Tall-and-narrow: more vertical room per gene row, less horizontal per age col.
    fig_h = max(5.0, 0.62 * n_g + 2.5)
    fig_w = max(5.0, 0.38 * n_a + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    _vext = max(abs(np.nanmin(z)), abs(np.nanmax(z)))
    vext = _vext if np.isfinite(_vext) and _vext > 0 else 1.0
    scp = ax.scatter(xx, yy, c=cc, s=sizes, cmap="coolwarm",
                      vmin=-vext, vmax=vext,
                      edgecolors="k", linewidths=0.25, zorder=2)

    ax.set_yticks(range(n_g))
    ax.set_yticklabels(genes, fontsize=11)
    ax.invert_yaxis()
    ax.set_xticks(even_xs)
    ax.set_xticklabels([f"{a:g}" for a in uniq_ages], fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Age (months)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = plt.colorbar(scp, ax=ax, fraction=0.04, pad=0.02, shrink=0.6)
    cb.set_label("Averaged expression", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # Size legend tucked just below the x-axis tick labels.
    _legend_pcts = np.linspace(smin, smax, 3) if smin < smax else np.full(3, smin)
    _legend_sizes = [
        25 + 350 * (float(p) - smin) / (smax - smin + 1e-12)
        for p in _legend_pcts
    ]
    for pct, s in zip(_legend_pcts, _legend_sizes):
        ax.scatter([], [], s=s, c="grey", edgecolors="k", linewidths=0.25,
                   label=f"{int(round(pct * 100))}%")
    ax.legend(title="% expr.", loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False,
              fontsize=10, title_fontsize=10)

    if title:
        ax.set_title(title, fontsize=20, pad=8)
    _save(fig, out, dpi=dpi)
    return fig
