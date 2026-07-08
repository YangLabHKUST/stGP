"""Shared utilities for the MouseBrain MERFISH tutorial notebooks.

Keep engineering-oriented helpers here: paths, naming, lightweight I/O, timing,
and reusable statistical routines. The notebooks intentionally keep the main
analysis flow visible.
"""


import json
import os
import pickle
import re
import time
import traceback
from pathlib import Path
from typing import Callable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, norm, spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from plots import MethodResult, load_method
from plots import (
    plot_active_gene_dotplot,
    plot_alpha_over_age,
    plot_gene_trajectories_over_age,
    plot_popari_spatial_programs,
    plot_program_weighted_scores_by_age,
    plot_program_variance_partition,
    plot_runtime_comparison,
    plot_spacetime_cluster_stack,
    plot_spacetime_embedding_stack,
    plot_spatial_kernel_corr_combined,
    plot_spatial_cluster_single_slice,
    plot_benchmark_cluster_methods_single_slice,
    plot_spatial_programs_selected_slices,
    plot_stgp_spatial_programs,
    plot_W_program_heatmap,
)
from plots import (
    cosine_similarity_matrix,
    hungarian_match,
    jaccard_top_genes_matrix,
    standardize_gene_weights,
)
from plots import plot_recovery_dotplot, plot_similarity_heatmap
from plots import run_enrichment_for_program


# ════════════════════════════════════════════════════════════════════════════
# Cell-type lists
# ════════════════════════════════════════════════════════════════════════════

# Run order used by the baseline drivers (03a / 03b / 03c / 03d). Smallest cell types
# first so the slower baselines (MEFISTO, STAMP) clear quickly on the cheap
# cell types before tackling the expensive ones.
BASELINE_RUN_ORDER: list[str] = [
    "T cell", "NSC", "Neuroblast", "Macrophage", "Ependymal", "VSMC", "Pericyte",
    "OPC", "Microglia", "Endothelial", "Astrocyte", "Neuron-MSN",
    "Oligodendrocyte", "Neuron-Excitatory",
]


# ════════════════════════════════════════════════════════════════════════════
# Default directory layout (relative to the repo root)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_RAW_DATA = Path(os.environ.get("STGP_MOUSE_RAW_H5AD", "data/raw/aging_coronal.h5ad"))
DATA_QC = Path("data/qc/aging_coronal_qc.h5ad")
DATA_PROCESSED = Path("data/processed")
RESULTS_STGP = Path("Results/stgp")
RESULTS_BASELINES = Path("Results/baselines")
RESULTS_PROXIMITY = Path("Results/proximity")
RESULTS_ENRICHMENT = Path("Results/enrichment")
FIGURES_ROOT = Path("Figures")
TIMING_LOG = Path("Results/benchmark_runtimes.jsonl")

# ════════════════════════════════════════════════════════════════════════════
# Naming helpers
# ════════════════════════════════════════════════════════════════════════════

def safe_name(celltype: str) -> str:
    """Filesystem-safe variant of a cell-type name (e.g. 'T cell' -> 'T_cell')."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in celltype)


def p_to_stars(pval, *, nan_label="n.s.", nonsig_label="n.s."):
    """Format a p-value as significance stars for compact figure labels."""
    if not np.isfinite(pval):
        return nan_label
    return "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else nonsig_label))


# ════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ════════════════════════════════════════════════════════════════════════════

def append_jsonl(log_path: Path, record: dict) -> None:
    """Append one JSON record (one line) to a `.jsonl` log file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_fig(fig: plt.Figure, path: str | Path, *, dpi: int = 400) -> None:
    """Save a Matplotlib figure to `path` and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def time_and_log_baseline(
    *,
    method: str,
    celltype: str,
    fn: Callable[[], None],
    out_dir: Path,
    timing_log: Path | None = None,
    extra_timing: dict | None = None,
) -> str:
    """Run ``fn()`` inline, capture wall time, and write benchmark logs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    status = "completed"
    interrupted: BaseException | None = None
    try:
        fn()
    except KeyboardInterrupt as e:
        status = "interrupted"
        interrupted = e
    except Exception as e:
        status = f"error ({type(e).__name__}: {e})"
        traceback.print_exc()

    elapsed = time.perf_counter() - t0
    timing = {"method": method, "celltype": celltype,
              "runtime_sec": round(elapsed, 2), "status": status}
    if extra_timing:
        timing.update(extra_timing)
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))

    if timing_log is not None:
        append_jsonl(
            timing_log,
            {"celltype": celltype, "method": method,
             "runtime_sec": round(elapsed, 2), "status": status,
             "out_dir": str(out_dir.resolve())},
        )

    print(f"[{method}] -> {status} ({elapsed:.1f}s)")
    if interrupted is not None:
        raise interrupted
    return status


# Methods whose gene weights are signed (can be negative). For others (Popari /
# STAMP) we treat the weights as non-negative loadings.
_SIGNED_METHODS = {"SpatialPCA", "MEFISTO"}
_SELECTED_SPATIAL_AGES = (6.6, 18.8, 24.6, 34.5)


# ════════════════════════════════════════════════════════════════════════════
# Variance partition helpers
# ════════════════════════════════════════════════════════════════════════════

def _compute_stgp_varpart_from_model(
    stgp_result: dict, prog_labels: list[str],
) -> pd.DataFrame:
    """Spatio-temporal variance partition derived from stGP model parameters.

    For each program k:
        sigma2_age  = theta[k, 0]   (temporal kernel amplitude)
        tau2_spa    = theta[k, 1]   (spatial kernel amplitude)

    Proportions are computed over the signal only (sigma2_age + tau2_spa); the
    shared residual noise sigma2_e is excluded so the two bars sum to 100 %.
    """
    theta = np.asarray(stgp_result["theta"], dtype=float)      # (K, 2)
    rows = []
    for k, prog in enumerate(prog_labels):
        sig_age = float(theta[k, 0])
        tau_spa = float(theta[k, 1])
        total = sig_age + tau_spa or 1.0
        rows.append(dict(component=prog,
                          sigma2_age=sig_age / total,
                          tau2_spa=tau_spa / total))
    return pd.DataFrame(rows).set_index("component")


# ════════════════════════════════════════════════════════════════════════════
# Misc helpers
# ════════════════════════════════════════════════════════════════════════════

def _infer_gene_weights(adata, scores: pd.DataFrame, group_col: str = "mouse_id") -> pd.DataFrame:
    """Per-mouse Pearson correlation between scores and expression (used as a
    fallback for STAMP, which doesn't expose gene loadings directly).
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = np.asarray(X, dtype=float)
    genes = adata.var_names.astype(str).tolist()

    scores = scores.reindex(adata.obs_names.astype(str))
    S = scores.to_numpy(dtype=float)

    groups = adata.obs[group_col].astype(str).to_numpy()
    uniq = pd.unique(groups).tolist()

    Xg = np.zeros((len(uniq), X.shape[1]))
    Sg = np.zeros((len(uniq), S.shape[1]))
    for i, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        Xg[i] = np.mean(X[idx], axis=0)
        Sg[i] = np.mean(S[idx], axis=0)

    Xc = Xg - Xg.mean(0, keepdims=True)
    Sc = Sg - Sg.mean(0, keepdims=True)
    cov = Sc.T @ Xc
    sd_s = np.sqrt(np.sum(Sc ** 2, axis=0))
    sd_x = np.sqrt(np.sum(Xc ** 2, axis=0))
    corr = cov / (sd_s[:, None] * sd_x[None, :] + 1e-12)
    return pd.DataFrame(corr, index=scores.columns.astype(str), columns=genes)


def _pick_top_age_component(var_df: pd.DataFrame) -> str | None:
    """Return the component with the largest Age (or sigma2_age) contribution."""
    col = ("sigma2_age" if "sigma2_age" in var_df.columns
           else "Age" if "Age" in var_df.columns
           else None)
    return None if col is None else str(var_df[col].astype(float).idxmax())


def _top_genes_from_weights(w_series: pd.Series, n: int = 80) -> tuple[list[str], pd.Series]:
    """Top-n positive-weighted genes (or top-n by absolute value if too few positive)."""
    w = pd.to_numeric(w_series, errors="coerce").fillna(0.0)
    w.index = w.index.astype(str)

    w_pos = w[w > 0]
    if w_pos.shape[0] >= n:
        w_rank = w_pos.sort_values(ascending=False)
        genes = w_rank.index[:n].tolist()
        return genes, w_rank.reindex(genes)

    w_abs = w.abs().sort_values(ascending=False)
    genes = w_abs.index[:n].tolist()
    return genes, w_abs.reindex(genes)


def _embedding_for_method(m: MethodResult) -> np.ndarray:
    """Return the embedding used for clustering and spatial summaries."""
    if m.method == "stGP" and "X_stgp_spatial" in m.adata.obsm:
        emb = np.asarray(m.adata.obsm["X_stgp_spatial"], dtype=float)
    elif m.method == "SpatialPCA" and "X_spatialpca" in m.adata.obsm:
        emb = np.asarray(m.adata.obsm["X_spatialpca"], dtype=float)
    else:
        emb = m.scores.to_numpy(dtype=float)
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)


def _spectral_knn_labels(
    X: np.ndarray, n_clusters: int, *, random_state: int = 42,
) -> np.ndarray:
    """KNN-graph spectral clustering used by the newer real-data analyses."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    if n <= n_clusters:
        return np.arange(n, dtype=int) % max(n_clusters, 1)

    Xs = StandardScaler().fit_transform(X)
    k_nn = min(max(2, int(np.round(np.sqrt(n)))), n - 1)
    nn = (
        NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
        .fit(Xs)
        .kneighbors(return_distance=False)[:, 1:]
    )
    rows = np.repeat(np.arange(nn.shape[0]), k_nn)
    cols = nn.ravel()
    knn_graph = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    knn_graph = knn_graph.maximum(knn_graph.T)

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    ).fit_predict(knn_graph)


def _cluster_centroids(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Embedding-space centroids indexed by local cluster id."""
    centroids = np.zeros((n_clusters, X.shape[1]), dtype=float)
    global_mean = np.nanmean(X, axis=0)
    for k in range(n_clusters):
        mask = labels == k
        centroids[k] = np.nanmean(X[mask], axis=0) if mask.any() else global_mean
    return centroids


def _canonicalise_first_slice(
    labels: np.ndarray, centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic first-slice labels, sorted by the first embedding axis."""
    order = np.argsort(centroids[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    aligned = np.array([remap[int(x)] for x in labels], dtype=int)
    return aligned, centroids[order]


def _align_to_previous_slice(
    labels: np.ndarray, centroids: np.ndarray, prev_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match current local clusters to previous-slice labels by centroid distance."""
    d2 = np.sum((centroids[:, None, :] - prev_centroids[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(d2)
    remap = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    aligned = np.array([remap[int(x)] for x in labels], dtype=int)
    aligned_centroids = np.zeros_like(prev_centroids)
    for r, c in remap.items():
        aligned_centroids[c] = centroids[r]
    return aligned, aligned_centroids


def _slice_aligned_spectral_labels(
    adata, emb: np.ndarray, *, n_clusters: int, random_state: int = 42,
) -> np.ndarray:
    """Spectral-cluster each mouse slice and align labels across age."""
    mouse_ids = adata.obs["mouse_id"].astype(str).to_numpy()
    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([
        float(adata.obs.loc[adata.obs["mouse_id"].astype(str) == m, "age"].iloc[0])
        for m in uniq_mice
    ])
    uniq_mice = uniq_mice[np.argsort(age_per_mouse)]

    labels_all = np.zeros(adata.n_obs, dtype=int)
    prev_centroids: np.ndarray | None = None
    for mid in uniq_mice:
        idx = np.flatnonzero(mouse_ids == mid)
        n_eff = min(max(2, n_clusters), max(1, idx.size - 1))
        if n_eff < 2:
            labels_all[idx] = 1
            continue
        local_labels = _spectral_knn_labels(
            emb[idx], n_eff, random_state=random_state,
        )
        centroids = _cluster_centroids(emb[idx], local_labels, n_eff)
        if prev_centroids is None or prev_centroids.shape[0] != n_eff:
            aligned, prev_centroids = _canonicalise_first_slice(local_labels, centroids)
        else:
            aligned, prev_centroids = _align_to_previous_slice(
                local_labels, centroids, prev_centroids,
            )
        labels_all[idx] = aligned + 1
    return labels_all


def _domain_labels_for_method(
    m: MethodResult, emb: np.ndarray, *, n_clusters: int,
) -> np.ndarray:
    """STAMP uses its topic output directly; other methods use KNN spectral labels."""
    if m.method == "STAMP":
        return np.argmax(emb, axis=1).astype(int) + 1
    return _slice_aligned_spectral_labels(
        m.adata, emb, n_clusters=n_clusters, random_state=42,
    )


def _pick_pseudotime_mouse(adata, emb: np.ndarray) -> str | None:
    """Choose one information-rich slice for Slingshot downstream validation."""
    mouse_ids = adata.obs["mouse_id"].astype(str).to_numpy()
    ages = pd.to_numeric(adata.obs["age"], errors="coerce").to_numpy(float)
    rows = []
    for mid in np.unique(mouse_ids):
        idx = np.flatnonzero(mouse_ids == mid)
        if idx.size < 200:
            continue
        rows.append(dict(
            mouse_id=mid,
            n_cells=int(idx.size),
            age=float(np.nanmedian(ages[idx])),
            emb_var=float(np.nanmean(np.nanvar(emb[idx], axis=0))),
        ))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["n_cells", "emb_var", "age"]:
        span = df[col].max() - df[col].min()
        df[f"{col}_score"] = 0.0 if span == 0 else (df[col] - df[col].min()) / span
    df["score"] = 0.55 * df["n_cells_score"] + 0.30 * df["emb_var_score"] + 0.15 * df["age_score"]
    return str(df.sort_values("score", ascending=False).iloc[0]["mouse_id"])


def _plot_pseudotime_spatial(adata_slice, *, pt_col: str, out: Path) -> None:
    xy = np.asarray(adata_slice.obsm["spatial"], dtype=float)
    pt = pd.to_numeric(adata_slice.obs[pt_col], errors="coerce").to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=pt, s=5, cmap="viridis",
        linewidths=0, rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Slingshot pseudotime")
    save_fig(fig, out, dpi=400)


def _plot_domain_spatial(adata_slice, *, out: Path) -> None:
    xy = np.asarray(adata_slice.obsm["spatial"], dtype=float)
    labels = adata_slice.obs["clusterlabel"].astype(str).to_numpy()
    cats = adata_slice.obs["clusterlabel"].cat.categories.astype(str).tolist()
    cmap = plt.get_cmap("tab20", max(len(cats), 3))

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    for i, label in enumerate(cats):
        mask = labels == str(label)
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=5, linewidths=0, rasterized=True,
            color=cmap(i / max(len(cats) - 1, 1)),
            label=f"Domain {label}",
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(
        title="Domain", markerscale=2.2,
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
    )
    save_fig(fig, out, dpi=400)


def _section4b_stgp_pseudotime(
    stgp: MethodResult, cluster_labels: np.ndarray, emb: np.ndarray,
    *, safe_ct: str, fig_dir: Path, num_epochs: int = 10,
) -> None:
    """Run Slingshot on one representative slice after stGP domain clustering."""
    print("  [fig] stGP domain pseudotime on a representative slice ...")
    try:
        from pyslingshot import Slingshot
    except Exception as e:
        print(f"    [skip pseudotime] pyslingshot unavailable: {e}")
        return

    if "mouse_id" not in stgp.adata.obs.columns:
        print("    [skip pseudotime] missing mouse_id")
        return
    mouse_id = _pick_pseudotime_mouse(stgp.adata, emb)
    if mouse_id is None:
        print("    [skip pseudotime] no slice with enough cells")
        return

    mouse_ids = stgp.adata.obs["mouse_id"].astype(str).to_numpy()
    idx = np.flatnonzero(mouse_ids == mouse_id)
    adata_slice = stgp.adata[idx].copy()
    domain = pd.Categorical(cluster_labels[idx].astype(str))
    if len(domain.categories) < 2:
        print(f"    [skip pseudotime] mouse {mouse_id}: <2 domains")
        return

    adata_slice.obs["clusterlabel"] = domain
    adata_slice.obsm["X_DRM"] = np.asarray(emb[idx], dtype=float)

    # Root the curve at the domain with the lowest mean along the strongest
    # varying stGP spatial component in this slice. This makes direction
    # deterministic without hard-coding a cell-type-specific domain ID.
    comp_idx = int(np.nanargmax(np.nanvar(adata_slice.obsm["X_DRM"], axis=0)))
    cats = adata_slice.obs["clusterlabel"].cat.categories.astype(str).tolist()
    domain_means = []
    labels = adata_slice.obs["clusterlabel"].astype(str).to_numpy()
    for cat in cats:
        vals = adata_slice.obsm["X_DRM"][labels == cat, comp_idx]
        domain_means.append(float(np.nanmean(vals)))
    start_cluster = cats[int(np.nanargmin(domain_means))]
    start_node = cats.index(start_cluster)

    sl = Slingshot(
        adata_slice,
        celltype_key="clusterlabel",
        obsm_key="X_DRM",
        start_node=start_node,
    )
    sl.fit(num_epochs=num_epochs)
    adata_slice.obs["slingPseudotime_1"] = sl.unified_pseudotime
    if sl.curves is not None and sl.cell_weights is not None:
        for l_idx, curve in enumerate(sl.curves):
            pt = curve.pseudotimes_interp.copy()
            weight = sl.cell_weights[:, l_idx].copy()
            pt[weight <= 0] = np.nan
            adata_slice.obs[f"slingPseudotime_{l_idx + 1}"] = pt
            adata_slice.obs[f"slingCurveWeight_{l_idx + 1}"] = weight

    out_dir = fig_dir / "pseudotime"
    out_dir.mkdir(parents=True, exist_ok=True)
    age = float(pd.to_numeric(adata_slice.obs["age"], errors="coerce").median())
    label_suffix = f"mouse_{mouse_id}_age_{age:.1f}mo"
    adata_slice.obs.to_csv(out_dir / f"{safe_ct}_{label_suffix}_slingshot_obs.csv")
    _plot_pseudotime_spatial(
        adata_slice,
        pt_col="slingPseudotime_1",
        out=out_dir / f"{safe_ct}_{label_suffix}_slingshot_pseudotime_spatial.png",
    )
    _plot_domain_spatial(
        adata_slice,
        out=out_dir / f"{safe_ct}_{label_suffix}_stGP_domains_spatial.png",
    )


def _load_methods(safe_ct: str, celltype: str, stgp_dir: Path | None = None) -> list[MethodResult]:
    """Load every method whose result directory exists for this cell type."""
    method_dirs = {
        "stGP":       stgp_dir if stgp_dir is not None else RESULTS_STGP / safe_ct,
        "SpatialPCA": RESULTS_BASELINES / "spatialpca" / safe_ct,
        "MEFISTO":    RESULTS_BASELINES / "mefisto" / safe_ct,
        "STAMP":      RESULTS_BASELINES / "stamp" / safe_ct,
        "Popari":     RESULTS_BASELINES / "popari" / safe_ct,
    }
    methods: list[MethodResult] = []
    for name, d in method_dirs.items():
        if not d.exists():
            print(f"  [skip] {name}: result dir not found")
            continue
        try:
            m = load_method(name, d, celltype=celltype)
            methods.append(m)
            print(f"  [loaded] {name}: {m.scores.shape}")
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return methods


def _load_stgp_pickle(safe_ct: str, stgp_dir: Path | None = None) -> dict | None:
    """Best-effort load of ``stgp_result.pkl`` (used for theta + bw_spa)."""
    pkl = (stgp_dir if stgp_dir is not None else RESULTS_STGP / safe_ct) / "stgp_result.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [warn] could not load stgp_result.pkl: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Per-section figure builders
# ════════════════════════════════════════════════════════════════════════════

def _section1_variance_decomposition(
    stgp: MethodResult, stgp_res: dict | None,
    *, celltype: str, safe_ct: str, fig_dir: Path,
) -> pd.DataFrame | None:
    """Variance-decomposition stacked bar (stGP only). Returns the var_df used."""
    print("  [fig] Variance decomposition (stGP) ...")
    stgp_vdf: pd.DataFrame | None = None

    prog_labels = stgp.scores.columns.astype(str).tolist()
    stgp_vdf = _compute_stgp_varpart_from_model(stgp_res, prog_labels)
    stgp_vdf.to_csv(fig_dir / f"{safe_ct}_stGP_varpart.csv")

    fig = plot_program_variance_partition(stgp_vdf, title=None)
    save_fig(fig, fig_dir / f"{safe_ct}_stGP_variance_partition.png")

    return stgp_vdf


def _section2_program_scores_by_age(
    stgp: MethodResult, stgp_vdf: pd.DataFrame | None,
    *, celltype: str, safe_ct: str, fig_dir: Path,
) -> None:
    """Per-program weighted score boxplots, alpha-over-age curves, and the
    age-trajectory + dot plots that are anchored on stGP gene weights."""
    if stgp.gene_weights is None:
        return

    print("  [fig] Program weighted scores by age ...")
    try:
        fig = plot_program_weighted_scores_by_age(
            stgp.adata, stgp.gene_weights,
            title=None,
        )
        save_fig(fig, fig_dir / f"{safe_ct}_stGP_program_scores_by_age.png")
    except Exception as e:
        print(f"    [skip program scores] {e}")

    # Top-age-component gene-trajectory plot.
    if stgp_vdf is not None:
        best = _pick_top_age_component(stgp_vdf)
        if best and best in set(stgp.gene_weights.index.astype(str)):
            genes, w = _top_genes_from_weights(stgp.gene_weights.loc[best])
            try:
                fig = plot_gene_trajectories_over_age(
                    adata=stgp.adata, genes=genes, gene_weights=w,
                    title=None)
                save_fig(fig, fig_dir / f"{safe_ct}_stGP_{best}_gene_trajectories.png")
            except Exception as e:
                print(f"    [skip gene traj] {e}")

    # alpha(t) per program, with optional posterior CI.
    st_uns = stgp.adata.uns.get("stgp", {})
    ages_arr = np.asarray(st_uns.get("ages", []), dtype=float)
    alpha_arr = np.asarray(st_uns.get("alpha", []), dtype=float)
    alpha_lower_arr = np.asarray(st_uns.get("alpha_lower", []), dtype=float)
    alpha_upper_arr = np.asarray(st_uns.get("alpha_upper", []), dtype=float)
    has_alpha = (alpha_arr.ndim == 2 and ages_arr.ndim == 1
                  and alpha_arr.shape[1] == ages_arr.shape[0])
    if has_alpha:
        has_ci = (alpha_lower_arr.shape == alpha_arr.shape
                  and alpha_upper_arr.shape == alpha_arr.shape)
        progs = stgp.gene_weights.index.astype(str).tolist()
        for j, prog in enumerate(progs):
            try:
                fig = plot_alpha_over_age(
                    ages=ages_arr, alpha=alpha_arr[j],
                    alpha_lower=alpha_lower_arr[j] if has_ci else None,
                    alpha_upper=alpha_upper_arr[j] if has_ci else None,
                    title=None)
                save_fig(fig, fig_dir / f"{safe_ct}_stGP_{prog}_alpha_over_age.png")
            except Exception as e:
                print(f"    [skip alpha {prog}] {e}")

    # One active-gene dot plot per program.
    print("  [fig] Active-gene dot plots ...")
    for prog in stgp.gene_weights.index.astype(str).tolist():
        try:
            w = pd.to_numeric(stgp.gene_weights.loc[prog],
                               errors="coerce").fillna(0.0)
            top_genes = w[w > 0].sort_values(ascending=False).index.astype(str).tolist()
            if not top_genes:
                top_genes = w.abs().sort_values(ascending=False).index.astype(str).tolist()
            m = re.search(r"\d+$", prog)
            prog_num = m.group() if m else prog
            fig = plot_active_gene_dotplot(
                stgp.adata, genes=top_genes, n_top=20,
                title=None,
            )
            save_fig(fig, fig_dir / f"{safe_ct}_stGP_{prog}_active_gene_dotplot.png")
        except Exception as e:
            print(f"    [skip dotplot {prog}] {e}")


def _section4_spatial_maps_and_clustering(
    methods: list[MethodResult], stgp: MethodResult | None,
    *, celltype: str, safe_ct: str, fig_dir: Path,
) -> None:
    """Spatial program maps plus KNN spectral domains and stGP stack plots."""
    print("  [fig] Spatial program maps + KNN spectral domains ...")
    adata_full = ad.read_h5ad(DATA_QC) if DATA_QC.exists() else None

    # ---- (a) Spatial program maps -----------------------------------------
    selected_spatial_dir = fig_dir / "spatial_selected_2x2"
    if stgp is not None:
        try:
            figs = plot_stgp_spatial_programs(
                stgp_adata=stgp.adata, scores=stgp.scores,
                adata_full=adata_full, celltype=celltype,
            )
            for prog, fig in zip(stgp.scores.columns, figs):
                save_fig(fig, fig_dir / f"{safe_ct}_stGP_{prog}_spatial.png")

            figs_2x2 = plot_spatial_programs_selected_slices(
                adata=stgp.adata, scores=stgp.scores,
                use_spatial_obsm=True, color_scale="symmetric",
                target_ages=_SELECTED_SPATIAL_AGES,
                adata_full=adata_full,
            )
            for prog, fig in zip(stgp.scores.columns, figs_2x2):
                save_fig(fig, selected_spatial_dir / f"{safe_ct}_stGP_{prog}_spatial_2x2.png")
        except Exception as e:
            print(f"    [skip stGP spatial maps] {e}")
            traceback.print_exc()

    benchmark_root = fig_dir / "benchmark"
    for m in methods:
        if m.method == "stGP":
            continue
        method_dir = benchmark_root / m.method.lower()
        try:
            scores_named = m.scores.copy()
            scores_named.columns = [f"{m.method}_prog{i+1}"
                                     for i in range(scores_named.shape[1])]
            # Topic-style scores (Popari / STAMP) are non-negative; everything
            # else uses the signed colour scheme.
            if m.method in ("Popari", "STAMP"):
                figs = plot_popari_spatial_programs(
                    popari_adata=m.adata, scores=scores_named,
                    adata_full=adata_full, celltype=celltype, ncols=5,
                )
                figs_2x2 = plot_spatial_programs_selected_slices(
                    adata=m.adata, scores=scores_named,
                    use_spatial_obsm=False, color_scale="percentile",
                    target_ages=_SELECTED_SPATIAL_AGES,
                    adata_full=adata_full,
                )
            else:
                figs = plot_stgp_spatial_programs(
                    stgp_adata=m.adata, scores=scores_named,
                    adata_full=adata_full, celltype=celltype, ncols=5,
                )
                figs_2x2 = plot_spatial_programs_selected_slices(
                    adata=m.adata, scores=scores_named,
                    use_spatial_obsm=False, color_scale="symmetric",
                    target_ages=_SELECTED_SPATIAL_AGES,
                    adata_full=adata_full,
                )
            for prog, fig in zip(scores_named.columns, figs):
                save_fig(fig, method_dir / f"{prog}_spatial.png")
            for prog, fig in zip(scores_named.columns, figs_2x2):
                save_fig(
                    fig,
                    selected_spatial_dir / m.method.lower() / f"{prog}_spatial_2x2.png",
                )
        except Exception as e:
            print(f"    [skip {m.method} spatial maps] {e}")

    # ---- (b) KNN spectral domains + age-ordered 3D stack ------------------
    # Default to ``len(unique regions)`` clusters so anatomical structure is
    # roughly recoverable. STAMP is not reclustered: its topic/category output
    # is used directly by argmax, following the HumanBrain benchmark.
    stack_dir = fig_dir / "spacetime_stack"
    clust_dir = fig_dir / "clustering"
    method_cache: dict[str, MethodResult] = {}
    cluster_cache: dict[str, np.ndarray] = {}

    for m in methods:
        if "mouse_id" not in m.adata.obs.columns:
            continue
        try:
            region_labels = (m.adata.obs["region"].astype(str).to_numpy()
                              if "region" in m.adata.obs.columns else None)
            n_clusters = max(2, len(np.unique(region_labels))
                              if region_labels is not None else 6)

            emb = _embedding_for_method(m)
            cluster_labels = _domain_labels_for_method(
                m, emb, n_clusters=n_clusters,
            )
            method_cache[m.method] = m
            cluster_cache[m.method] = cluster_labels

            if m.method == "stGP":
                fig = plot_spacetime_cluster_stack(
                    adata=m.adata, cluster_labels=cluster_labels,
                    adata_full=adata_full, method_name=m.method, celltype=celltype,
                    elev=30,
                )
                save_fig(fig, stack_dir / f"{safe_ct}_stGP_spectral_domains_stack_tilt30.png")

                prog_names = m.scores.columns.astype(str).tolist()
                for j, prog in enumerate(prog_names):
                    if j >= emb.shape[1]:
                        continue
                    fig = plot_spacetime_embedding_stack(
                        adata=m.adata, values=emb[:, j], adata_full=adata_full,
                        value_label=f"{prog} spatial embedding",
                        color_scale="symmetric",
                        elev=30,
                    )
                    save_fig(
                        fig,
                        stack_dir / f"{safe_ct}_stGP_{prog}_spatial_embedding_stack_tilt30.png",
                    )
                _section4b_stgp_pseudotime(
                    m, cluster_labels, emb, safe_ct=safe_ct, fig_dir=fig_dir,
                )
        except Exception as e:
            print(f"    [skip clust] {m.method}: {e}")

    if "stGP" in method_cache and "stGP" in cluster_cache:
        stgp_m = method_cache["stGP"]
        stgp_labels = cluster_cache["stGP"]
        mouse_ids = stgp_m.adata.obs["mouse_id"].astype(str).to_numpy()
        uniq_mice = np.unique(mouse_ids)
        age_per_mouse = np.array([
            float(stgp_m.adata.obs.loc[stgp_m.adata.obs["mouse_id"].astype(str) == mid, "age"].iloc[0])
            for mid in uniq_mice
        ])
        order = np.argsort(age_per_mouse)
        for mid, age in zip(uniq_mice[order], age_per_mouse[order]):
            try:
                fig = plot_spatial_cluster_single_slice(
                    adata=stgp_m.adata, cluster_labels=stgp_labels,
                    mouse_id=str(mid), adata_full=adata_full,
                )
                save_fig(
                    fig,
                    clust_dir / "stGP_single_slices" /
                    f"{safe_ct}_stGP_clustering_mouse_{mid}_age_{age:.1f}mo.png",
                )
            except Exception as e:
                print(f"    [skip stGP slice {mid}] {e}")

            benchmark_methods = {
                name: method_cache[name]
                for name in ["SpatialPCA", "MEFISTO", "STAMP", "Popari"]
                if name in method_cache and name in cluster_cache
            }
            benchmark_labels = {
                name: cluster_cache[name]
                for name in benchmark_methods
            }
            if benchmark_methods:
                try:
                    fig = plot_benchmark_cluster_methods_single_slice(
                        method_adatas={name: m.adata for name, m in benchmark_methods.items()},
                        method_cluster_labels=benchmark_labels,
                        mouse_id=str(mid),
                        adata_full=adata_full,
                    )
                    save_fig(
                        fig,
                        clust_dir / "benchmark_methods_by_slice" /
                        f"{safe_ct}_benchmark_clustering_mouse_{mid}_age_{age:.1f}mo.png",
                    )
                except Exception as e:
                    print(f"    [skip benchmark slice {mid}] {e}")

    del adata_full


def _section5_program_similarity(
    methods: list[MethodResult], stgp: MethodResult | None,
    *, celltype: str, safe_ct: str, fig_dir: Path,
) -> None:
    """Cosine + Jaccard similarity of stGP programs vs each baseline."""
    if stgp is None or stgp.gene_weights is None:
        return
    print("  [fig] Program similarity ...")
    W_stgp = standardize_gene_weights(stgp.gene_weights, signed=False)
    sim_dir = fig_dir / "program_similarity"
    sim_dir.mkdir(parents=True, exist_ok=True)
    best_sim = pd.DataFrame(index=W_stgp.index.astype(str))

    for m in methods:
        if m.method == "stGP" or m.gene_weights is None:
            continue
        try:
            Wb = standardize_gene_weights(m.gene_weights,
                                            signed=m.method in _SIGNED_METHODS)
            sim_cos = cosine_similarity_matrix(W_stgp, Wb)
            sim_cos.to_csv(sim_dir / f"{safe_ct}_stGP_vs_{m.method}_cosine.csv")

            for top_n in (10, 20, 50):
                sim_jac = jaccard_top_genes_matrix(W_stgp, Wb, top_n=top_n)
                sim_jac.to_csv(sim_dir / f"{safe_ct}_stGP_vs_{m.method}_jaccard_top{top_n}.csv")
                fig = plot_similarity_heatmap(
                    sim_jac,
                    title=None,
                    out=sim_dir / f"{safe_ct}_stGP_vs_{m.method}_jaccard_top{top_n}_heatmap.png",
                )
                plt.close(fig)

            matches = hungarian_match(sim_cos)
            match_df = pd.DataFrame([vars(x) for x in matches])
            match_df.to_csv(sim_dir / f"{safe_ct}_match_stGP_to_{m.method}.csv", index=False)

            # Reorder cosine columns by Hungarian-matched best baseline program.
            match_map = {x.left_program: x.right_program for x in matches}
            col_order = [match_map.get(r, sim_cos.columns[0]) for r in sim_cos.index.tolist()]
            col_order_dedup, used = [], set()
            for c in col_order:
                if c in used:
                    for alt in sim_cos.columns:
                        if alt not in used:
                            c = alt
                            break
                col_order_dedup.append(c)
                used.add(c)
            sim_ord = sim_cos.loc[sim_cos.index.tolist(), col_order_dedup]

            fig = plot_similarity_heatmap(
                sim_ord, title=None,
                out=sim_dir / f"{safe_ct}_stGP_vs_{m.method}_cosine_heatmap.png")
            plt.close(fig)

            best_sim[m.method] = sim_cos.max(axis=1)
        except Exception as e:
            print(f"    [skip sim] {m.method}: {e}")

    if best_sim.shape[1] > 0:
        best_sim.to_csv(sim_dir / f"{safe_ct}_stGP_recovery_best_similarity.csv")
        try:
            fig = plot_recovery_dotplot(
                best_sim, title=None,
                out=sim_dir / f"{safe_ct}_stGP_recovery_dotplot.png")
            plt.close(fig)
        except Exception as e:
            print(f"    [skip dotplot] {e}")


def _section6_kernel_diagnostic_and_W_heatmap(
    stgp: MethodResult, stgp_res: dict | None,
    *, celltype: str, safe_ct: str, fig_dir: Path,
) -> None:
    """Spatial kernel correlation + W heatmap (stGP only)."""
    print("  [fig] Spatial kernel correlation + W heatmap ...")
    try:
        bw_spa = None
        if stgp_res is not None:
            bw_spa = stgp_res.get("bw_spa") or stgp_res.get("gamma_spa")
        if bw_spa is None:
            # Fallback: median pairwise squared distance / 2 on the median-age slice.
            sp = np.asarray(stgp.adata.obsm["spatial"], dtype=float)
            ages = stgp.adata.obs["age"].astype(float).to_numpy()
            uniq = np.sort(np.unique(ages))
            ref_age = float(uniq[len(uniq) // 2])
            mask = ages == ref_age
            coords = sp[mask][: min(500, mask.sum())]
            from scipy.spatial.distance import pdist
            bw_spa = float(np.median(pdist(coords) ** 2)) / 2.0

        # Default to the middle slice instead of hard-coding one slice index.
        n_slices = int(stgp.adata.obs["age"].astype(float).nunique())
        slice_idx_ref = n_slices // 2

        fig = plot_spatial_kernel_corr_combined(
            adata=stgp.adata, bandwidth=float(bw_spa),
            slice_idx=slice_idx_ref,
            title=None,
        )
        save_fig(fig, fig_dir / "Spatial_corr_Kernel.png")
    except Exception as e:
        print(f"    [skip kernel corr] {e}")

    if stgp.gene_weights is not None:
        try:
            fig = plot_W_program_heatmap(
                stgp.gene_weights, title=None)
            save_fig(fig, fig_dir / f"{safe_ct}_W_program.png")
        except Exception as e:
            print(f"    [skip W heatmap] {e}")


def _section7_timing(
    *, celltype: str, safe_ct: str, fig_dir: Path, stgp_dir: Path | None = None,
) -> None:
    """Per-method runtime bar chart from each method's ``timing.json``."""
    print("  [fig] Timing ...")
    timing_rows = []

    stgp_timing = (stgp_dir if stgp_dir is not None else RESULTS_STGP / safe_ct) / "timing.json"
    if stgp_timing.exists():
        timing_rows.append(json.loads(stgp_timing.read_text()))

    for method_key in ["pca", "nmf", "spatialpca", "mefisto", "stamp", "popari"]:
        tp = RESULTS_BASELINES / method_key / safe_ct / "timing.json"
        if tp.exists():
            timing_rows.append(json.loads(tp.read_text()))

    if not timing_rows:
        return

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(fig_dir / f"{safe_ct}_timing.csv", index=False)
    completed = timing_df[timing_df["status"] == "completed"]
    if len(completed) == 0:
        return

    try:
        fig = plot_runtime_comparison(completed, title=None)
        save_fig(fig, fig_dir / f"{safe_ct}_runtime_comparison.png")
    except Exception as e:
        print(f"    [skip timing fig] {e}")


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════

# Spatial scales (microns)
R_NEAR = 30.0          # near radius for the matched test
R_FAR = 150.0          # far radius for the matched test
R_IN, R_OUT = 20.0, 50.0   # shell radii for proximity enrichment
DENS_R = 50.0          # radius for variance-decomposition densities
FAR_REF = 300.0        # reference distance for distance-decay contrast
DECAY_BINS = [(0, 25), (25, 50), (50, 100), (100, 200)]

# Statistical thresholds
MIN_BLK = 8                 # min cells per side per slice (age) block
N_PERM_DEFAULT = 100        # number of spatial permutations for the null
HIGH_PCT, LOW_PCT = 75, 25  # quantile cutoffs for high vs low b
MIN_ENRICH_GROUP = 30       # min high/low cells for all-slice enrichment
MIN_ENRICH_SLICE_GROUP = 8  # min high/low cells for per-slice enrichment
MAX_OVERLAY_POINTS = 2500   # deterministic cap for dense spatial overlays

# Effectors with q < SIG_Q_THRESHOLD in the matched test drive the downstream
# (distance-decay, age-stratification, near-far violins) analyses.
SIG_Q_THRESHOLD = 0.05


def age_bin_label(a: float) -> str:
    """Coarse age bin for stratifying matched effects."""
    if a < 12:
        return "young (<12 mo)"
    if a < 24:
        return "middle (12-24 mo)"
    return "old (>=24 mo)"


AGE_BINS = ["young (<12 mo)", "middle (12-24 mo)", "old (>=24 mo)"]

# 14 distinguishable hues (Tableau 10 + extras), each visually distinct from
# the greys reserved for "no data" / null distributions.
CT_COLORS = {
    "T cell":            "#d62728",   # red
    "Microglia":         "#2ca02c",   # green
    "Oligodendrocyte":   "#17becf",   # cyan
    "Astrocyte":         "#ff7f0e",   # orange
    "NSC":               "#1f77b4",   # blue
    "Neuroblast":        "#9467bd",   # purple
    "Endothelial":       "#8c564b",   # brown
    "OPC":               "#bcbd22",   # olive
    "Pericyte":          "#e377c2",   # pink
    "Macrophage":        "#7f7f7f",   # mid grey
    "Ependymal":         "#aec7e8",   # light blue
    "VSMC":              "#c49c94",   # tan
    "Neuron-MSN":        "#98df8a",   # light green
    "Neuron-Excitatory": "#c5b0d5",   # lavender
}
CT_MARKERS = {
    "Microglia": "o", "T cell": "*", "NSC": "s", "Neuroblast": "^",
}

# Fallback effector list used only when the matched test produces no
# significant effectors (rare). The spatial permutation null is always run on
# every effector type; for ultra-dense cell types (e.g. Oligodendrocyte ~10k
# cells/slice) the null collapses to NaN under the MIN_BLK constraint and is
# rendered as "-" downstream. Significant downstream analyses are driven by
# whichever effectors pass the matched-test FDR (q < SIG_Q_THRESHOLD).
DEFAULT_PERM_EFFECTORS = ["T cell", "Astrocyte", "Endothelial", "NSC", "Neuroblast"]


# ════════════════════════════════════════════════════════════════════════════
#  Setup helpers
# ════════════════════════════════════════════════════════════════════════════

def load_target_data(target_ct: str, stgp_root: str) -> sc.AnnData:
    """Load the stGP-scored AnnData (``adata_with_scores.h5ad``) for one cell type."""
    safe = safe_name(target_ct)
    path = Path(stgp_root) / safe / "adata_with_scores.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"stGP result not found: {path}")
    return sc.read_h5ad(str(path))


def extract_target_arrays(adata_target: sc.AnnData) -> dict:
    """Pull per-cell arrays from the stGP-scored target AnnData.

    Note: the gene-expression matrix ``X`` is intentionally not pulled. The
    proximity tests only use coordinates, age, region, and the stGP embeddings
    ``H`` and ``B``; densifying the count matrix would waste a lot of RAM for
    no analytical benefit.
    """
    arrs = dict(
        age=np.asarray(adata_target.obs["age"]),
        region=np.asarray(adata_target.obs["region"]),
        coord=np.asarray(adata_target.obsm["spatial"]),
        H=np.asarray(adata_target.obsm["X_stgp"]),
        B=np.asarray(adata_target.obsm["X_stgp_spatial"]),
        var_names=np.asarray(adata_target.var_names),
    )
    arrs["n_programs"] = arrs["B"].shape[1]
    arrs["program_labels"] = [f"stGP{j+1}" for j in range(arrs["n_programs"])]
    return arrs


def extract_global_arrays(adata_all: sc.AnnData) -> dict:
    """Pull global per-cell arrays from the QC AnnData (all cell types). Same
    rationale as ``extract_target_arrays``: skip the expression matrix.
    """
    return dict(
        age=np.asarray(adata_all.obs["age"]),
        region=np.asarray(adata_all.obs["region"]),
        ct=np.asarray(adata_all.obs["celltype"]),
        coord=np.asarray(adata_all.obsm["spatial"]),
        var_names=np.asarray(adata_all.var_names),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Module 1 -- Matched within-block proximity effect
# ════════════════════════════════════════════════════════════════════════════

def _block_weights(v_n: np.ndarray, v_f: np.ndarray) -> tuple[float, float]:
    """Return (delta_median, inverse_variance_weight) for one slice block."""
    eff = float(np.median(v_n) - np.median(v_f))
    var_n = float(np.var(v_n, ddof=1)) / len(v_n)
    var_f = float(np.var(v_f, ddof=1)) / len(v_f)
    w = 1.0 / (var_n + var_f + 1e-9)
    return eff, w


def _ivw_aggregate(block_eff: list[float], block_w: list[float]) -> tuple[float, float]:
    """Inverse-variance-weighted mean and standard error of slice-level effects."""
    w = np.asarray(block_w)
    e = np.asarray(block_eff)
    eff = float(np.sum(w * e) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    return eff, se


def matched_proximity_effect(
    eff_type: str, k: int, *,
    target, glob, regions,
    R_near: float = R_NEAR, R_far: float = R_FAR, min_blk: int = MIN_BLK,
    target_subset: np.ndarray | None = None,
):
    """Slice-stratified delta median b (near - far) for one effector x program.

    Blocks are *one per age (slice)* -- region is intentionally not used so that
    cells from anatomically distinct compartments (e.g. ventricular Ependymal
    vs cortical Microglia) can still contribute.
    """
    if target_subset is None:
        target_subset = np.ones(len(target["age"]), dtype=bool)

    block_eff, block_w = [], []
    n_near = n_far = 0
    n_blocks = 0

    for age in np.unique(target["age"]):
        eff_mask = (glob["age"] == age) & (glob["ct"] == eff_type)
        if eff_mask.sum() < 1:
            continue
        tree = cKDTree(glob["coord"][eff_mask])

        blk = target_subset & (target["age"] == age)
        if blk.sum() < min_blk * 2:
            continue

        d_near, _ = tree.query(target["coord"][blk], k=1)
        near = d_near <= R_near
        far = d_near > R_far
        if near.sum() < min_blk or far.sum() < min_blk:
            continue

        v_n = target["B"][blk, k][near]
        v_f = target["B"][blk, k][far]
        eff, w = _block_weights(v_n, v_f)
        block_eff.append(eff)
        block_w.append(w)

        n_near += int(near.sum())
        n_far += int(far.sum())
        n_blocks += 1

    if not block_eff:
        return dict(effect=np.nan, se=np.nan, z=np.nan, p=np.nan,
                    n_near=0, n_far=0, n_blocks=0)

    eff_g, se = _ivw_aggregate(block_eff, block_w)
    z = eff_g / se
    p = float(2 * norm.sf(abs(z)))
    return dict(effect=eff_g, se=se, z=z, p=p,
                n_near=n_near, n_far=n_far, n_blocks=n_blocks)


def compute_matched_effect_table(target, glob, regions, effectors) -> pd.DataFrame:
    """Run the matched test for every (effector, program) pair. Returns a tidy
    DataFrame with FDR-adjusted (Benjamini-Hochberg) q-values appended.
    """
    rows = []
    for eff in effectors:
        for k in range(target["n_programs"]):
            r = matched_proximity_effect(eff, k, target=target, glob=glob, regions=regions)
            rows.append(dict(effector=eff, program=target["program_labels"][k], k=k, **r))

    df = pd.DataFrame(rows)
    df["q_bh"] = np.nan
    mask = df["p"].notna()
    if mask.any():
        _, q, _, _ = multipletests(df.loc[mask, "p"], method="fdr_bh")
        df.loc[mask, "q_bh"] = q
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Module 2 -- Spatial permutation null
# ════════════════════════════════════════════════════════════════════════════

def _matched_effect_with_locs(
    eff_xy_per_age, k, target, regions,
    R_near=R_NEAR, R_far=R_FAR, min_blk=MIN_BLK,
) -> float:
    """Same as ``matched_proximity_effect`` but with effector positions provided
    externally (used inside the spatial permutation null). Blocks by age only.
    """
    block_eff, block_w = [], []
    for age in np.unique(target["age"]):
        eff_xy = eff_xy_per_age.get(age)
        if eff_xy is None or len(eff_xy) < 1:
            continue
        tree = cKDTree(eff_xy)

        blk = (target["age"] == age)
        if blk.sum() < min_blk * 2:
            continue

        d, _ = tree.query(target["coord"][blk], k=1)
        near = d <= R_near
        far = d > R_far
        if near.sum() < min_blk or far.sum() < min_blk:
            continue

        v_n = target["B"][blk, k][near]
        v_f = target["B"][blk, k][far]
        eff, w = _block_weights(v_n, v_f)
        block_eff.append(eff)
        block_w.append(w)

    if not block_eff:
        return np.nan
    eff_g, _ = _ivw_aggregate(block_eff, block_w)
    return eff_g


def compute_permutation_null(
    target, glob, regions, effectors, k, *,
    n_perm=N_PERM_DEFAULT, seed=0,
):
    """Spatial permutation null for one program & a list of effectors.

    The pool of candidate positions is built per slice (age) only; effector
    counts are also tallied per slice. This matches the slice-blocked matched
    test.
    """
    rng = np.random.default_rng(seed)

    pool = {age: glob["coord"][glob["age"] == age]
            for age in np.unique(target["age"])}
    eff_counts = {
        eff: {age: int(((glob["age"] == age) & (glob["ct"] == eff)).sum())
              for age in np.unique(target["age"])}
        for eff in effectors
    }

    real = {eff: matched_proximity_effect(eff, k, target=target, glob=glob,
                                            regions=regions)["effect"]
            for eff in effectors}

    # Each draw picks `n_eff_in_slice` random positions from the slice-wide
    # pool, preserving per-slice effector cell density.
    nulls = {eff: [] for eff in effectors}
    for _ in range(n_perm):
        for eff in effectors:
            eff_xy_per_age = {}
            for age in np.unique(target["age"]):
                pool_xy = pool[age]
                n = eff_counts[eff][age]
                if n > 0 and len(pool_xy) >= n:
                    sel = rng.choice(len(pool_xy), size=n, replace=False)
                    eff_xy_per_age[age] = pool_xy[sel]
            nulls[eff].append(_matched_effect_with_locs(eff_xy_per_age, k, target, regions))

    rows = []
    for eff in effectors:
        n = np.array(nulls[eff])
        n = n[~np.isnan(n)]
        if len(n) == 0:
            rows.append(dict(effector=eff, observed=real[eff],
                              null_mean=np.nan, null_sd=np.nan,
                              p_perm=np.nan, n_null=0))
            continue
        p_emp = float((np.abs(n) >= abs(real[eff])).mean())
        rows.append(dict(effector=eff, observed=float(real[eff]),
                          null_mean=float(n.mean()), null_sd=float(n.std()),
                          p_perm=p_emp, n_null=int(len(n))))

    df = pd.DataFrame(rows)
    return df, nulls, real


# ════════════════════════════════════════════════════════════════════════════
#  Module 3 -- Distance decay
# ════════════════════════════════════════════════════════════════════════════

def matched_effect_in_bin(
    eff_type, k, lo, hi, *,
    target, glob, regions, far_ref=FAR_REF, min_blk=MIN_BLK,
) -> tuple[float, float]:
    """Slice-blocked delta median b for an effector at distance band (lo, hi]
    vs reference (> far_ref). Blocks per age (slice).
    """
    block_eff, block_w = [], []
    for age in np.unique(target["age"]):
        m = (glob["age"] == age) & (glob["ct"] == eff_type)
        if m.sum() < 1:
            continue
        tree = cKDTree(glob["coord"][m])

        blk = (target["age"] == age)
        if blk.sum() < min_blk * 2:
            continue

        d, _ = tree.query(target["coord"][blk], k=1)
        in_bin = (d > lo) & (d <= hi)
        ref = d > far_ref
        if in_bin.sum() < min_blk or ref.sum() < min_blk:
            continue

        v_in = target["B"][blk, k][in_bin]
        v_ref = target["B"][blk, k][ref]
        eff, w = _block_weights(v_in, v_ref)
        block_eff.append(eff)
        block_w.append(w)

    if not block_eff:
        return np.nan, np.nan
    return _ivw_aggregate(block_eff, block_w)


def compute_distance_decay(target, glob, regions, effectors, k, far_ref=FAR_REF) -> pd.DataFrame:
    """Distance-decay curves for the supplied list of effectors."""
    rows = []
    for eff in effectors:
        for lo, hi in DECAY_BINS:
            v, se = matched_effect_in_bin(eff, k, lo, hi, target=target, glob=glob,
                                            regions=regions, far_ref=far_ref)
            rows.append(dict(effector=eff, lo=lo, hi=hi, mid=(lo + hi) / 2,
                              effect=v, se=se))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  Module 4 -- Age-stratified proximity effect
# ════════════════════════════════════════════════════════════════════════════

def matched_effect_per_age_bin(
    eff_type, k, age_bin, *,
    target, glob, regions, min_blk=MIN_BLK,
) -> tuple[float, float, int]:
    """Slice-blocked delta median b restricted to slices in the given age bin."""
    ages_in_bin = [a for a in np.unique(target["age"]) if age_bin_label(a) == age_bin]

    block_eff, block_w = [], []
    for age in ages_in_bin:
        m = (glob["age"] == age) & (glob["ct"] == eff_type)
        if m.sum() < 1:
            continue
        tree = cKDTree(glob["coord"][m])

        blk = (target["age"] == age)
        if blk.sum() < min_blk * 2:
            continue

        d, _ = tree.query(target["coord"][blk], k=1)
        near = d <= R_NEAR
        far = d > R_FAR
        if near.sum() < min_blk or far.sum() < min_blk:
            continue

        v_n = target["B"][blk, k][near]
        v_f = target["B"][blk, k][far]
        eff, w = _block_weights(v_n, v_f)
        block_eff.append(eff)
        block_w.append(w)

    if not block_eff:
        return np.nan, np.nan, 0
    eff_g, se = _ivw_aggregate(block_eff, block_w)
    return eff_g, se, len(block_eff)


def compute_age_stratification(target, glob, regions, effectors, k) -> pd.DataFrame:
    """Age-stratified near-vs-far effect for the supplied list of effectors."""
    rows = []
    for eff in effectors:
        for b in AGE_BINS:
            v, se, nb = matched_effect_per_age_bin(
                eff, k, b, target=target, glob=glob, regions=regions)
            rows.append(dict(effector=eff, age_bin=b, effect=v, se=se, n_blocks=nb))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  Module 5 -- Abundance check (target vs other cell types over age)
# ════════════════════════════════════════════════════════════════════════════

def compute_abundance_check(
    target_ct, glob, *,
    other_cts=("T cell", "NSC", "Neuroblast"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell-type counts per slice + Spearman trend of count vs age."""
    cts_to_count = list({target_ct} | set(other_cts))

    rows = []
    for age in np.unique(glob["age"]):
        for ct in cts_to_count:
            n = int(((glob["age"] == age) & (glob["ct"] == ct)).sum())
            rows.append(dict(age=float(age), celltype=ct, count=n))
    df = pd.DataFrame(rows)

    summary = []
    for ct in cts_to_count:
        s = df[df.celltype == ct].sort_values("age")
        if len(s) >= 3:
            rho, p = spearmanr(s.age, s["count"])
        else:
            rho, p = np.nan, np.nan
        summary.append(dict(celltype=ct, total=int(s["count"].sum()),
                              n_slices=int((s["count"] > 0).sum()),
                              spearman_rho=float(rho), spearman_p=float(p)))
    return df, pd.DataFrame(summary)


# ════════════════════════════════════════════════════════════════════════════
#  Module 6 -- Proximity enrichment (high vs low b in shell)
# ════════════════════════════════════════════════════════════════════════════

def compute_proximity_enrichment(
    target, glob, regions, k, *,
    enrich_types=("T cell", "Oligodendrocyte", "OPC", "Astrocyte",
                   "Endothelial", "Pericyte", "Macrophage", "NSC", "Neuroblast"),
    R_in=R_IN, R_out=R_OUT, high_pct=HIGH_PCT, low_pct=LOW_PCT,
):
    """Per-region log2 fold-change in shell counts between high- and low-b cells."""
    b = target["B"][:, k]
    hi_thr = float(np.percentile(b, high_pct))
    lo_thr = float(np.percentile(b, low_pct))
    hi_mask = b >= hi_thr
    lo_mask = b <= lo_thr

    counts = np.zeros((len(target["age"]), len(enrich_types)), dtype=np.int32)
    for age in np.unique(target["age"]):
        blk = target["age"] == age
        idx = np.where(blk)[0]
        xy = target["coord"][blk]
        if xy.shape[0] < 1:
            continue
        for ti, ct in enumerate(enrich_types):
            m = (glob["age"] == age) & (glob["ct"] == ct)
            if m.sum() < 1:
                continue
            tree = cKDTree(glob["coord"][m])
            out_idx = tree.query_ball_point(xy, r=R_out)
            in_idx = tree.query_ball_point(xy, r=R_in)
            for j, (oi, ii) in enumerate(zip(out_idx, in_idx)):
                counts[idx[j], ti] = len(oi) - len(ii)

    rows = []
    for region in ["ALL"] + list(regions):
        reg_mask = (np.ones(len(target["age"]), dtype=bool)
                    if region == "ALL" else (target["region"] == region))
        n_hi = int((hi_mask & reg_mask).sum())
        n_lo = int((lo_mask & reg_mask).sum())
        if n_hi < 30 or n_lo < 30:
            continue
        for ti, ct in enumerate(enrich_types):
            hi_n = counts[hi_mask & reg_mask, ti]
            lo_n = counts[lo_mask & reg_mask, ti]
            if len(hi_n) < 30 or len(lo_n) < 30:
                continue
            hi_m, lo_m = float(hi_n.mean()), float(lo_n.mean())
            log2fc = float(np.log2((hi_m + 0.05) / (lo_m + 0.05)))
            try:
                _, pv = mannwhitneyu(hi_n, lo_n, alternative="two-sided")
            except ValueError:
                pv = 1.0
            rows.append(dict(region=region, celltype=ct,
                              hi_mean=hi_m, lo_mean=lo_m,
                              log2fc=log2fc, p=float(pv),
                              n_hi=int(len(hi_n)), n_lo=int(len(lo_n))))

    df = pd.DataFrame(rows)
    df["q_bh"] = np.nan
    if df["p"].notna().any():
        _, q, _, _ = multipletests(df["p"].fillna(1.0), method="fdr_bh")
        df["q_bh"] = q
    return df, hi_mask, lo_mask, list(enrich_types)


# ════════════════════════════════════════════════════════════════════════════
#  Module 7 -- Variance decomposition (single OLS, all cell types as predictors)
# ════════════════════════════════════════════════════════════════════════════

def per_cell_log_density(
    target, glob, density_types: list[str], *, radius: float = DENS_R,
) -> np.ndarray:
    """log(1 + count of {ct} cells within radius microns) for every target cell."""
    n = len(target["age"])
    log_density = np.zeros((n, len(density_types)))
    for age in np.unique(target["age"]):
        blk = target["age"] == age
        idx = np.where(blk)[0]
        xy = target["coord"][blk]
        if xy.shape[0] < 1:
            continue
        for ti, ct in enumerate(density_types):
            m = (glob["age"] == age) & (glob["ct"] == ct)
            if m.sum() < 1:
                continue
            tree = cKDTree(glob["coord"][m])
            cnt = np.array([len(idx_) for idx_ in tree.query_ball_point(xy, r=radius)])
            log_density[idx, ti] = np.log1p(cnt)
    return log_density


def _per_cell_log_density(target, glob, density_types: list[str]) -> np.ndarray:
    """Backward-compatible wrapper for older downstream cells/scripts."""
    return per_cell_log_density(target, glob, density_types)


# ════════════════════════════════════════════════════════════════════════════
#  Module 8 -- Downstream enrichment of spatial residual b by other cell types
# ════════════════════════════════════════════════════════════════════════════

def _add_bh_fdr(df: pd.DataFrame, *, p_col="p", q_col="q_bh") -> pd.DataFrame:
    """Append Benjamini-Hochberg FDR values while preserving skipped rows."""
    df = df.copy()
    df[q_col] = np.nan
    if df.empty or p_col not in df.columns:
        return df
    mask = df[p_col].notna()
    if mask.any():
        _, q, _, _ = multipletests(df.loc[mask, p_col], method="fdr_bh")
        df.loc[mask, q_col] = q
    return df


def _add_group_bh_fdr(
    df: pd.DataFrame, *, group_col: str, p_col="p", q_col="q_bh_by_slice",
) -> pd.DataFrame:
    """Append BH-FDR within each slice/age block."""
    df = df.copy()
    df[q_col] = np.nan
    if df.empty or group_col not in df.columns or p_col not in df.columns:
        return df
    for _, idx in df.groupby(group_col).groups.items():
        idx = list(idx)
        mask = df.loc[idx, p_col].notna()
        if mask.any():
            use_idx = df.loc[idx].index[mask]
            _, q, _, _ = multipletests(df.loc[use_idx, p_col], method="fdr_bh")
            df.loc[use_idx, q_col] = q
    return df


def _shell_counts_by_effector(
    target, glob, effectors: list[str], *, R_in=R_IN, R_out=R_OUT,
) -> dict[str, np.ndarray]:
    """Count effector cells in the (R_in, R_out] shell around each target cell."""
    counts_by_eff = {
        eff: np.zeros(len(target["age"]), dtype=np.int32)
        for eff in effectors
    }
    for age in np.unique(target["age"]):
        blk = target["age"] == age
        idx = np.where(blk)[0]
        xy = target["coord"][blk]
        if xy.shape[0] < 1:
            continue
        for eff in effectors:
            eff_mask = (glob["age"] == age) & (glob["ct"] == eff)
            if eff_mask.sum() < 1:
                continue
            tree = cKDTree(glob["coord"][eff_mask])
            out_idx = tree.query_ball_point(xy, r=R_out)
            in_idx = tree.query_ball_point(xy, r=R_in)
            counts_by_eff[eff][idx] = np.fromiter(
                (len(oi) - len(ii) for oi, ii in zip(out_idx, in_idx)),
                dtype=np.int32,
                count=len(idx),
            )
    return counts_by_eff


def compute_downstream_all_slices_enrichment(
    target, glob, effectors: list[str], k: int, *,
    counts_by_eff: dict[str, np.ndarray] | None = None,
    R_in=R_IN, R_out=R_OUT, high_pct=HIGH_PCT, low_pct=LOW_PCT,
    min_group=MIN_ENRICH_GROUP,
) -> pd.DataFrame:
    """All-slice high-vs-low b shell enrichment for every non-target cell type."""
    columns = [
        "program", "k", "effector", "hi_threshold", "lo_threshold",
        "n_hi", "n_lo", "hi_mean", "lo_mean", "log2fc", "p", "q_bh",
        "R_in", "R_out", "high_pct", "low_pct", "min_group", "valid",
    ]
    if not effectors:
        return pd.DataFrame(columns=columns)
    if counts_by_eff is None:
        counts_by_eff = _shell_counts_by_effector(target, glob, effectors,
                                                  R_in=R_in, R_out=R_out)

    b = target["B"][:, k]
    hi_thr = float(np.percentile(b, high_pct))
    lo_thr = float(np.percentile(b, low_pct))
    hi_mask = b >= hi_thr
    lo_mask = b <= lo_thr
    rows = []

    for eff in effectors:
        counts = counts_by_eff[eff]
        hi_n = counts[hi_mask]
        lo_n = counts[lo_mask]
        valid = len(hi_n) >= min_group and len(lo_n) >= min_group
        hi_m = float(np.mean(hi_n)) if len(hi_n) else np.nan
        lo_m = float(np.mean(lo_n)) if len(lo_n) else np.nan
        if valid:
            log2fc = float(np.log2((hi_m + 0.05) / (lo_m + 0.05)))
            try:
                _, pv = mannwhitneyu(hi_n, lo_n, alternative="two-sided")
            except ValueError:
                pv = np.nan
        else:
            log2fc, pv = np.nan, np.nan
        rows.append(dict(
            program=target["program_labels"][k], k=int(k), effector=eff,
            hi_threshold=hi_thr, lo_threshold=lo_thr,
            n_hi=int(len(hi_n)), n_lo=int(len(lo_n)),
            hi_mean=hi_m, lo_mean=lo_m, log2fc=log2fc, p=float(pv) if not np.isnan(pv) else np.nan,
            R_in=float(R_in), R_out=float(R_out), high_pct=float(high_pct),
            low_pct=float(low_pct), min_group=int(min_group), valid=bool(valid),
        ))

    df = pd.DataFrame(rows, columns=[c for c in columns if c != "q_bh"])
    return _add_bh_fdr(df, q_col="q_bh")[columns]


def compute_downstream_per_slice_enrichment(
    target, glob, effectors: list[str], k: int, *,
    counts_by_eff: dict[str, np.ndarray] | None = None,
    R_in=R_IN, R_out=R_OUT, high_pct=HIGH_PCT, low_pct=LOW_PCT,
    min_group=MIN_ENRICH_SLICE_GROUP,
) -> pd.DataFrame:
    """Per-slice high-vs-low b shell enrichment with slice-local b quantiles."""
    columns = [
        "program", "k", "age", "effector", "hi_threshold", "lo_threshold",
        "n_hi", "n_lo", "hi_mean", "lo_mean", "log2fc", "p", "q_bh",
        "q_bh_by_slice", "R_in", "R_out", "high_pct", "low_pct",
        "min_group", "valid",
    ]
    if not effectors:
        return pd.DataFrame(columns=columns)
    if counts_by_eff is None:
        counts_by_eff = _shell_counts_by_effector(target, glob, effectors,
                                                  R_in=R_in, R_out=R_out)

    rows = []
    b = target["B"][:, k]
    for age in np.unique(target["age"]):
        blk = target["age"] == age
        b_slice = b[blk]
        if len(b_slice) < min_group * 2:
            hi_thr = lo_thr = np.nan
            hi_mask = lo_mask = np.zeros(len(b_slice), dtype=bool)
        else:
            hi_thr = float(np.percentile(b_slice, high_pct))
            lo_thr = float(np.percentile(b_slice, low_pct))
            hi_mask = b_slice >= hi_thr
            lo_mask = b_slice <= lo_thr
        for eff in effectors:
            counts = counts_by_eff[eff][blk]
            hi_n = counts[hi_mask]
            lo_n = counts[lo_mask]
            valid = len(hi_n) >= min_group and len(lo_n) >= min_group
            hi_m = float(np.mean(hi_n)) if len(hi_n) else np.nan
            lo_m = float(np.mean(lo_n)) if len(lo_n) else np.nan
            if valid:
                log2fc = float(np.log2((hi_m + 0.05) / (lo_m + 0.05)))
                try:
                    _, pv = mannwhitneyu(hi_n, lo_n, alternative="two-sided")
                except ValueError:
                    pv = np.nan
            else:
                log2fc, pv = np.nan, np.nan
            rows.append(dict(
                program=target["program_labels"][k], k=int(k), age=float(age),
                effector=eff, hi_threshold=hi_thr, lo_threshold=lo_thr,
                n_hi=int(len(hi_n)), n_lo=int(len(lo_n)), hi_mean=hi_m,
                lo_mean=lo_m, log2fc=log2fc, p=float(pv) if not np.isnan(pv) else np.nan,
                R_in=float(R_in), R_out=float(R_out), high_pct=float(high_pct),
                low_pct=float(low_pct), min_group=int(min_group), valid=bool(valid),
            ))

    df = pd.DataFrame(rows, columns=[c for c in columns if c not in {"q_bh", "q_bh_by_slice"}])
    df = _add_bh_fdr(df, q_col="q_bh")
    df = _add_group_bh_fdr(df, group_col="age", q_col="q_bh_by_slice")
    return df[columns]


def compute_downstream_per_slice_matched_effects(
    target, glob, effectors: list[str], k: int, *,
    R_near=R_NEAR, R_far=R_FAR, min_blk=MIN_BLK,
) -> pd.DataFrame:
    """Per-slice near-vs-far delta median b for every non-target cell type."""
    columns = [
        "program", "k", "age", "effector", "effect", "se", "z", "p", "q_bh",
        "q_bh_by_slice", "n_near", "n_far", "R_near", "R_far", "min_blk",
        "valid",
    ]
    if not effectors:
        return pd.DataFrame(columns=columns)

    rows = []
    b = target["B"][:, k]
    for age in np.unique(target["age"]):
        blk = target["age"] == age
        xy = target["coord"][blk]
        b_slice = b[blk]
        for eff in effectors:
            eff_mask = (glob["age"] == age) & (glob["ct"] == eff)
            effect = se = z = pv = np.nan
            n_near = n_far = 0
            valid = False
            if eff_mask.sum() >= 1 and xy.shape[0] >= min_blk * 2:
                tree = cKDTree(glob["coord"][eff_mask])
                d_near, _ = tree.query(xy, k=1)
                near = d_near <= R_near
                far = d_near > R_far
                n_near = int(near.sum())
                n_far = int(far.sum())
                valid = n_near >= min_blk and n_far >= min_blk
                if valid:
                    v_n = b_slice[near]
                    v_f = b_slice[far]
                    effect, se = _block_weights(v_n, v_f)
                    if np.isfinite(se) and se > 0:
                        z = float(effect / se)
                        pv = float(2 * norm.sf(abs(z)))
            rows.append(dict(
                program=target["program_labels"][k], k=int(k), age=float(age),
                effector=eff, effect=effect, se=se, z=z, p=pv,
                n_near=n_near, n_far=n_far, R_near=float(R_near),
                R_far=float(R_far), min_blk=int(min_blk), valid=bool(valid),
            ))

    df = pd.DataFrame(rows, columns=[c for c in columns if c not in {"q_bh", "q_bh_by_slice"}])
    df = _add_bh_fdr(df, q_col="q_bh")
    df = _add_group_bh_fdr(df, group_col="age", q_col="q_bh_by_slice")
    return df[columns]


def summarise_downstream(
    df_all_enrich: pd.DataFrame,
    df_slice_enrich: pd.DataFrame,
    df_slice_match: pd.DataFrame,
    df_match_prog: pd.DataFrame,
    target_ct: str,
    k_label: str,
) -> dict:
    """Compact JSON summary of the downstream relationship tests."""
    valid_all = df_all_enrich[df_all_enrich["valid"].astype(bool)].copy()
    sig_all = valid_all[valid_all["q_bh"].notna() & (valid_all["q_bh"] < 0.05)]
    sig_slice_enrich = df_slice_enrich[
        df_slice_enrich["q_bh_by_slice"].notna()
        & (df_slice_enrich["q_bh_by_slice"] < 0.05)
    ]
    sig_slice_match = df_slice_match[
        df_slice_match["q_bh_by_slice"].notna()
        & (df_slice_match["q_bh_by_slice"] < 0.05)
    ]
    sig_global_match = df_match_prog[
        df_match_prog["q_bh"].notna() & (df_match_prog["q_bh"] < 0.05)
    ]

    if valid_all.empty:
        top_enriched = top_depleted = ""
    else:
        top_enriched = str(valid_all.sort_values("log2fc", ascending=False).iloc[0]["effector"])
        top_depleted = str(valid_all.sort_values("log2fc", ascending=True).iloc[0]["effector"])

    return dict(
        target_celltype=target_ct,
        program=k_label,
        n_effectors_tested=int(df_all_enrich["effector"].nunique()),
        n_slices_tested=int(df_slice_enrich["age"].nunique()) if "age" in df_slice_enrich else 0,
        n_all_slice_enrichment_sig=int(len(sig_all)),
        top_all_slice_enriched_effector=top_enriched,
        top_all_slice_depleted_effector=top_depleted,
        n_per_slice_enrichment_sig=int(len(sig_slice_enrich)),
        n_per_slice_matched_sig=int(len(sig_slice_match)),
        n_global_matched_sig=int(len(sig_global_match)),
        global_matched_sig_effectors="|".join(sig_global_match["effector"].tolist()),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Module 9 -- Demo slice picker
# ════════════════════════════════════════════════════════════════════════════

def pick_demo_slice(target, k, glob, *, prefer_old_with_effector="T cell") -> float:
    """Choose a slice with strong b heterogeneity and (if possible) plenty of
    the preferred effector cell type."""
    ages = np.unique(target["age"])

    candidates = []
    for age in ages:
        m = target["age"] == age
        if m.sum() < 50:
            continue
        var_b = float(np.var(target["B"][m, k]))
        eff_n = int(((glob["age"] == age) & (glob["ct"] == prefer_old_with_effector)).sum())
        candidates.append((age, var_b, eff_n))

    if not candidates:
        return float(ages[-1])

    cand_df = pd.DataFrame(candidates, columns=["age", "var_b", "eff_n"])
    cand_df["score"] = (
        cand_df["var_b"] / cand_df["var_b"].max()
        + 0.6 * cand_df["eff_n"] / max(cand_df["eff_n"].max(), 1)
        + 0.3 * cand_df["age"] / cand_df["age"].max()
    )
    return float(cand_df.sort_values("score", ascending=False).iloc[0]["age"])


# ════════════════════════════════════════════════════════════════════════════
#  Plotting -- in-axes renderers
# ════════════════════════════════════════════════════════════════════════════

def _significance_stars(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def render_matched_heatmap(ax, df_match, target_ct, k_focus_label, *, title_prefix="A"):
    """Heatmap of delta-median-b for every (effector, program), with FDR stars."""
    mat_eff = df_match.pivot(index="effector", columns="program", values="effect")
    mat_q = df_match.pivot(index="effector", columns="program", values="q_bh")
    if k_focus_label in mat_eff.columns:
        order = mat_eff[k_focus_label].sort_values(ascending=False).index
        mat_eff = mat_eff.loc[order]
        mat_q = mat_q.loc[order]

    vmax = max(np.nanpercentile(np.abs(mat_eff.values), 97), 0.01)
    im = ax.imshow(mat_eff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(mat_eff.shape[0]):
        for j in range(mat_eff.shape[1]):
            v = mat_eff.values[i, j]
            q = mat_q.values[i, j]
            if np.isnan(v):
                continue
            stars = _significance_stars(q)
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}\n{stars}", ha="center", va="center",
                     fontsize=7, color=color)

    ax.set_xticks(range(mat_eff.shape[1]))
    ax.set_xticklabels(mat_eff.columns, fontsize=9)
    ax.set_yticks(range(mat_eff.shape[0]))
    ax.set_yticklabels(mat_eff.index, fontsize=9)
    ax.set_xlabel("stGP program")
    ax.set_title(
        f"{title_prefix}   Matched proximity effect on {target_ct} $b$  "
        f"(near $\\leq${R_NEAR:.0f} um - far >{R_FAR:.0f} um)",
        loc="left", fontweight="bold", fontsize=10, pad=8,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(r"$\Delta$ median $b$", fontsize=9)
    return im


def render_forest_with_perm_null(
    ax, df_match, df_perm, k_label, target_ct, *,
    reliable=None, title_prefix="A", n_perm_used=None,
):
    """Forest plot of matched effects with permutation p-values annotated.

    Bars are coloured purely by sign of the effect. The right-hand gutter
    annotates ``p_perm`` from the spatial permutation null for every effector;
    when permutation collapsed (or wasn't run) the gutter shows ``"-"`` instead.

    When ``title_prefix`` is empty no title is drawn (used for the standalone
    figure).
    """
    df_b = df_match[(df_match.program == k_label) & (df_match.n_blocks > 0)].copy()
    df_b = df_b.sort_values("effect", ascending=True)
    y_pos = np.arange(len(df_b))
    clr = ["#c0392b" if v > 0 else "#2980b9" for v in df_b.effect]

    perm_lookup = {
        r.effector: (r.null_mean, r.null_sd, r.p_perm,
                      int(r.n_null) if "n_null" in r.index else 0)
        for _, r in df_perm.iterrows()
    }

    err = 1.96 * df_b.se.fillna(0).to_numpy()
    bar_lo = df_b.effect.to_numpy() - err
    bar_hi = df_b.effect.to_numpy() + err
    span = max(abs(float(bar_lo.min())), abs(float(bar_hi.max())))
    gutter = span * 0.34
    xmax = span * 1.08 + gutter
    p_anno_x = span * 1.10

    ax.barh(y_pos, df_b.effect, xerr=err, color=clr,
             edgecolor="black", lw=0.5, capsize=2.5, alpha=0.95, height=0.6,
             zorder=2)

    # Permutation p-value gutter on the right.
    for i, ename in enumerate(df_b.effector):
        pp, n_used = np.nan, 0
        if ename in perm_lookup:
            _, _, pp, n_used = perm_lookup[ename]
        if np.isnan(pp) or n_used == 0:
            txt = "$p_{perm}$ = -"
            col = "#7f8c8d"
        else:
            B = n_used
            if pp < 1.0 / B:
                txt = f"$p_{{perm}}$ < {1.0 / B:.2f}"
            else:
                txt = f"$p_{{perm}}$ = {pp:.2f}"
            col = "black" if pp < 0.05 else "#7f8c8d"
        ax.text(p_anno_x, i, txt, va="center", ha="left",
                 fontsize=10, color=col)

    ax.axvline(0, ls=":", color="black", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_b.effector, fontsize=12)
    ax.set_xlabel(r"$\Delta$ median $b$  (near - far)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    if title_prefix:
        ax.set_title(
            f"{title_prefix}   Effector ranking on {target_ct} {k_label}  "
            "(red = enriched-near, blue = depleted-near)",
            loc="left", fontweight="bold", fontsize=11, pad=10,
        )
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_xlim(-xmax, xmax)


def render_variance_coefs(ax, df_coefs, meta, k_label, title_prefix=""):
    """All non-intercept OLS coefficients sorted ascending. ``title_prefix=""``
    suppresses the title (standalone figure mode).
    """
    df = df_coefs[df_coefs.name != "intercept"].copy()
    df = df.sort_values("coef", ascending=True)
    y_pos = np.arange(len(df))
    clr = ["#c0392b" if v > 0 else "#2980b9" for v in df.coef]
    ax.barh(y_pos, df.coef, xerr=1.96 * df.se, color=clr, edgecolor="black",
             lw=0.5, alpha=0.9, capsize=2)
    ax.axvline(0, ls=":", color="black", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.name, fontsize=13)
    ax.set_xlabel("Coefficient", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    if title_prefix:
        ax.set_title(
            f"{title_prefix}   Effect of stGP spatial embedding "
            f"({k_label}, n={meta['n_predictors']} predictors, R^2={meta['r2']:.3f})",
            loc="left", fontweight="bold", pad=8,
        )
    ax.grid(True, axis="x", alpha=0.3)


def render_spatial_example(
    ax_main, target, glob, k, demo_age,
    target_ct, k_label, *, effector="T cell", title_prefix="",
):
    """Spatial scatter of one demo slice -- target cells coloured by ``b``,
    with one effector cell type overlaid as reference markers.
    """
    mg_blk = target["age"] == demo_age
    xy = target["coord"][mg_blk]
    b_d = target["B"][mg_blk, k]
    eff_xy = glob["coord"][(glob["age"] == demo_age) & (glob["ct"] == effector)]

    if xy.shape[0] < 1:
        ax_main.text(0.5, 0.5, f"No {target_ct} cells at {demo_age} mo",
                       transform=ax_main.transAxes, ha="center", va="center")
        ax_main.set_axis_off()
        return

    vlim = float(np.percentile(np.abs(b_d), 95))
    sc_handle = ax_main.scatter(xy[:, 0], xy[:, 1], c=b_d, cmap="RdBu_r",
                                  vmin=-vlim, vmax=vlim, s=22, alpha=0.85,
                                  edgecolor="none")

    if len(eff_xy) > 0:
        marker = CT_MARKERS.get(effector, "o")
        if effector == "T cell":
            # High-contrast star marker for T cell.
            ax_main.scatter(eff_xy[:, 0], eff_xy[:, 1], marker="*",
                              c="black", s=140, edgecolor="white", lw=0.7,
                              label=f"{effector} ({len(eff_xy)})", zorder=5)
        else:
            col = CT_COLORS.get(effector, "#27ae60")
            ax_main.scatter(eff_xy[:, 0], eff_xy[:, 1], marker=marker,
                              facecolor=col, s=70, alpha=0.85,
                              edgecolor="white", lw=0.5,
                              label=f"{effector} ({len(eff_xy)})", zorder=5)

    ax_main.set_aspect("equal")
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for sp in ax_main.spines.values():
        sp.set_visible(False)

    title = f"{demo_age} mo: {target_ct} coloured by $b$ {k_label} + {effector}"
    if title_prefix:
        title = f"{title_prefix}   {title}"
    ax_main.set_title(title, loc="left", fontweight="bold", pad=8)
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.92)
    plt.colorbar(sc_handle, ax=ax_main, fraction=0.025, pad=0.01,
                  label=f"$b$ {k_label}")


def render_near_far_violins(
    axes, target, glob, k, demo_age,
    target_ct, k_label, effectors, title_prefix="",
):
    """Per-effector near-vs-far violin pairs on a single demo slice."""
    mg_blk = target["age"] == demo_age
    xy = target["coord"][mg_blk]
    b_d = target["B"][mg_blk, k]

    if xy.shape[0] < 1:
        axes[0].text(0.5, 0.5, f"No {target_ct} cells at {demo_age} mo",
                       transform=axes[0].transAxes, ha="center", va="center")
        for ax in axes:
            ax.set_axis_off()
        return

    panels = []
    yvals_all = []
    for eff in effectors:
        eff_xy = glob["coord"][(glob["age"] == demo_age) & (glob["ct"] == eff)]
        if len(eff_xy) < 5:
            panels.append((eff, None, None, None, None))
            continue

        tree = cKDTree(eff_xy)
        d, _ = tree.query(xy, k=1)
        near = d <= R_NEAR
        far = d > R_FAR
        if near.sum() < 5 or far.sum() < 5:
            panels.append((eff, None, None, None, None))
            continue

        v_n = b_d[near]
        v_f = b_d[far]
        try:
            _, pmw = mannwhitneyu(v_n, v_f, alternative="two-sided")
        except ValueError:
            pmw = np.nan
        diff = float(np.median(v_n) - np.median(v_f))
        panels.append((eff, v_n, v_f, diff, pmw))
        yvals_all.extend(v_n.tolist())
        yvals_all.extend(v_f.tolist())

    if yvals_all:
        ymin, ymax = float(np.min(yvals_all)), float(np.max(yvals_all))
        pad = (ymax - ymin) * 0.08 + 0.1
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = None

    for ax, (eff, v_n, v_f, diff, pmw) in zip(axes, panels):
        if v_n is None:
            ax.text(0.5, 0.5, f"{eff}\n(insufficient cells)",
                     transform=ax.transAxes, ha="center", va="center",
                     fontsize=9, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        data = [v_n, v_f]
        parts = ax.violinplot(data, positions=[0, 1],
                                showmedians=False, showextrema=False, widths=0.75)
        col_eff = CT_COLORS.get(eff, "#c0392b")
        for pc, col in zip(parts["bodies"], [col_eff, "#7f8c8d"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.55)
            pc.set_edgecolor("black")
        ax.boxplot(data, positions=[0, 1], widths=0.20, patch_artist=True,
                     boxprops=dict(facecolor="white", alpha=0.95),
                     medianprops=dict(color="black", lw=1.4),
                     showfliers=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"near\n<={R_NEAR:.0f} um\n(n={len(v_n)})",
                              f"far\n>{R_FAR:.0f} um\n(n={len(v_f)})"], fontsize=8)
        ax.set_title(f"{eff}\nDelta_med={diff:+.2f}, p={pmw:.1e}",
                       fontsize=9, fontweight="bold", pad=4)
        ax.axhline(0, ls=":", color="black", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.2)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(f"$b$ {k_label}")


def render_distance_decay(ax, df_decay, k_label, target_ct, title_prefix="C"):
    """Distance-decay errorbar plot, colour coded by global ``CT_COLORS``."""
    for eff in df_decay.effector.unique():
        sub = df_decay[df_decay.effector == eff].dropna()
        if sub.empty:
            continue
        ax.errorbar(sub.mid, sub.effect, yerr=1.96 * sub.se,
                     marker="o", capsize=3, lw=2, ms=7,
                     color=CT_COLORS.get(eff, "grey"), label=eff)
    ax.axhline(0, ls=":", color="black", alpha=0.5)
    ax.set_xlabel("Distance bin midpoint (um)")
    ax.set_ylabel(f"$\\Delta$ median $b$ {k_label}\n(in-bin - >{FAR_REF:.0f} um)")
    if title_prefix:
        ax.set_title(f"{title_prefix}   Distance decay  ({target_ct})",
                      loc="left", fontweight="bold", pad=8)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)


def render_abundance(target_ct, df_counts, df_summary, fig_dir):
    """Cell abundance check across age -- saved once per cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: per-slice cell counts vs age (log scale).
    axA = axes[0]
    for ct in df_counts.celltype.unique().tolist():
        sub = df_counts[df_counts.celltype == ct].sort_values("age")
        axA.plot(sub.age, sub["count"],
                  marker=CT_MARKERS.get(ct, "o"), lw=2.0, ms=7,
                  color=CT_COLORS.get(ct, "grey"), label=ct, alpha=0.92)
    axA.set_yscale("log")
    axA.set_xlabel("Age (months)")
    axA.set_ylabel("Cells per coronal slice  (log)")
    axA.set_title(f"A   Cell-type abundance per slice  (target = {target_ct})",
                   loc="left", fontweight="bold", pad=8)
    axA.legend(loc="best", ncol=2, fontsize=9)
    axA.grid(True, alpha=0.3, which="both")

    # Panel B: Spearman rho of count-vs-age per cell type.
    axB = axes[1]
    df_summary_show = df_summary.set_index("celltype")
    rho_vals = df_summary_show["spearman_rho"].fillna(0)
    bar_clr = ["#c0392b" if v > 0 else "#2980b9" for v in rho_vals]
    axB.barh(range(len(rho_vals)), rho_vals, color=bar_clr,
              edgecolor="black", lw=0.5, alpha=0.9)

    # Place value text inside the bar near its tip (white for contrast). Tiny
    # bars get the text just outside the bar in black instead.
    xmax = max(abs(rho_vals.min()), abs(rho_vals.max()), 0.05)
    inset = xmax * 0.04
    for i, (ct, r) in enumerate(rho_vals.items()):
        p = df_summary_show.loc[ct, "spearman_p"]
        s = "***" if p < 1e-6 else "**" if p < 1e-3 else "*" if p < 0.05 else ""
        label = f"{r:+.2f}{s}"
        if abs(r) < inset * 1.5:
            x = r + np.sign(r) * 0.01 if r != 0 else 0.01
            ha = "left" if r >= 0 else "right"
            color = "black"
            weight = "normal"
        else:
            x = r - np.sign(r) * inset
            ha = "right" if r > 0 else "left"
            color = "white"
            weight = "bold"
        axB.text(x, i, label, va="center", ha=ha, fontsize=8,
                  color=color, fontweight=weight)
    axB.axvline(0, ls=":", color="black", alpha=0.6)
    axB.set_yticks(range(len(rho_vals)))
    axB.set_yticklabels(rho_vals.index, fontsize=9)
    axB.set_xlim(-xmax * 1.15, xmax * 1.15)
    axB.set_xlabel("Spearman rho between cell count and age")
    axB.set_title("B   Abundance trend over age",
                   loc="left", fontweight="bold", pad=8)
    axB.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, Path(fig_dir) / "abundance_check.png", dpi=400)


# ════════════════════════════════════════════════════════════════════════════
#  Standalone figure wrappers
# ════════════════════════════════════════════════════════════════════════════

def make_standalone_matched_heatmap(df_match, target_ct, k_label, savepath):
    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    render_matched_heatmap(ax, df_match, target_ct, k_label, title_prefix="")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_standalone_distance_decay(df_decay, target_ct, k_label, savepath):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    render_distance_decay(ax, df_decay, k_label, target_ct, title_prefix="")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_standalone_age_stratification(df_age, target_ct, k_label, savepath):
    """Heatmap of near-vs-far proximity effects across coarse age bins."""
    if df_age.empty:
        return
    pivot = df_age.pivot(index="effector", columns="age_bin", values="effect")
    nblk = df_age.pivot(index="effector", columns="age_bin", values="n_blocks")
    cols = [c for c in AGE_BINS if c in pivot.columns]
    pivot = pivot[cols]
    nblk = nblk[cols]
    if pivot.empty:
        return

    row_score = pivot.abs().max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[row_score.index]
    nblk = nblk.loc[row_score.index]

    vmax = max(float(np.nanpercentile(np.abs(pivot.values), 97)), 0.05)
    fig_h = max(3.8, 0.40 * pivot.shape[0] + 1.4)
    fig_w = max(5.2, 1.35 * pivot.shape[1] + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            nb = nblk.values[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}\n{int(nb) if np.isfinite(nb) else 0} slices",
                    ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Age bin")
    ax.set_ylabel("Effector cell type")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(r"$\Delta$ median $b$  (near - far)")
    save_fig(fig, savepath, dpi=400)


def make_standalone_enrichment(df_enrich, target_ct, k_label, savepath):
    """Region-stratified shell enrichment heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    pivot_fc = df_enrich.pivot(index="celltype", columns="region", values="log2fc")
    pivot_q = df_enrich.pivot(index="celltype", columns="region", values="q_bh")
    cols_present = [c for c in ["ALL", "CC/ACO", "CTX", "STR", "VEN"]
                    if c in pivot_fc.columns]
    pivot_fc = pivot_fc[cols_present]
    pivot_q = pivot_q[cols_present]

    vmax = max(np.nanpercentile(np.abs(pivot_fc.values), 95), 0.05)
    im = ax.imshow(pivot_fc.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(pivot_fc.shape[0]):
        for j in range(pivot_fc.shape[1]):
            v = pivot_fc.values[i, j]
            q = pivot_q.values[i, j]
            if np.isnan(v):
                continue
            stars = _significance_stars(q)
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}\n{stars}", ha="center", va="center",
                     fontsize=8, color=color)
    ax.set_xticks(range(pivot_fc.shape[1]))
    ax.set_xticklabels(pivot_fc.columns)
    ax.set_yticks(range(pivot_fc.shape[0]))
    ax.set_yticklabels(pivot_fc.index)
    ax.set_title(f"Shell enrichment around high-{k_label} {target_ct}\n"
                  f"({R_IN:.0f}-{R_OUT:.0f} um shell)",
                  fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                   label="log$_2$( High / Low )")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_standalone_variance(df_coefs, meta, target_ct, k_label, savepath):
    """Single-panel coefficient plot for the full OLS model -- no title."""
    n_pred = max(meta["n_predictors"], 1)
    fig, ax = plt.subplots(figsize=(8.5, max(0.36 * n_pred + 1.4, 5.5)))
    render_variance_coefs(ax, df_coefs, meta, k_label, title_prefix="")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_standalone_spatial(target, glob, k, demo_age, target_ct, k_label,
                              savepath, *, effector="T cell"):
    """Single demo-slice scatter -- target coloured by b, with one effector overlay."""
    fig, ax_main = plt.subplots(figsize=(9, 7))
    render_spatial_example(ax_main, target, glob, k, demo_age,
                              target_ct, k_label, effector=effector,
                              title_prefix="")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_standalone_near_far_violins(target, glob, k, target_ct, k_label,
                                        effectors, savepath):
    """Two row-grids of near-vs-far violins: top = youngest available slice,
    bottom = oldest available slice. Each row shows one violin pair per
    significant effector cell type.
    """
    n = len(effectors)
    if n == 0:
        return
    ages = np.unique(target["age"])
    if len(ages) < 1:
        return

    young_age = float(ages.min())
    old_age = float(ages.max())
    same_slice = (young_age == old_age)

    ncol = min(n, 5)
    nrow_per_slice = int(np.ceil(n / ncol))
    n_slice_groups = 1 if same_slice else 2
    total_rows = nrow_per_slice * n_slice_groups
    fig, axes = plt.subplots(total_rows, ncol,
                                figsize=(2.6 * ncol, 3.4 * total_rows + 0.6),
                                sharey=False, squeeze=False)

    def _render_one_slice(start_row, slice_age, label):
        row_axes = axes[start_row : start_row + nrow_per_slice].ravel().tolist()
        used = row_axes[:n]
        for ax in row_axes[n:]:
            ax.set_axis_off()
        render_near_far_violins(used, target, glob, k, slice_age,
                                  target_ct, k_label, effectors)
        # Row-level label aligned to the leftmost axes.
        bbox = used[0].get_position()
        fig.text(0.005, (bbox.y0 + bbox.y1) / 2,
                  f"{label}\n{slice_age:.1f} mo",
                  fontsize=11, fontweight="bold",
                  ha="left", va="center", rotation=0,
                  color="#333333")

    _render_one_slice(0, young_age, "youngest")
    if not same_slice:
        _render_one_slice(nrow_per_slice, old_age, "oldest")

    fig.tight_layout(rect=(0.06, 0.0, 1.0, 0.98))
    save_fig(fig, savepath, dpi=400)


def make_standalone_effector_ranking(df_match, df_perm, target_ct, k_label, savepath):
    """Forest plot of matched proximity effects with permutation p-values."""
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    render_forest_with_perm_null(ax, df_match, df_perm, k_label, target_ct,
                                    title_prefix="")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def _finite_symmetric_vmax(values, *, percentile=97, fallback=1.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    vmax = float(np.percentile(np.abs(vals), percentile))
    return max(vmax, 0.05)


def _format_age_labels(ages) -> list[str]:
    return [f"{float(a):.1f}" for a in ages]


def make_downstream_all_slices_figure(
    df_all_enrich: pd.DataFrame,
    df_match_prog: pd.DataFrame,
    target_ct: str,
    k_label: str,
    savepath: Path,
) -> None:
    """Ranked all-slice enrichment + matched-effect figure for one program."""
    if df_all_enrich.empty:
        return
    match_cols = (
        df_match_prog[["effector", "effect", "se", "q_bh"]]
        .rename(columns={"effect": "matched_effect", "se": "matched_se",
                         "q_bh": "matched_q"})
    )
    df = df_all_enrich.merge(match_cols, on="effector", how="left")
    df["sort_key"] = df["log2fc"].fillna(0)
    df = df.sort_values("sort_key", ascending=True)
    y = np.arange(len(df))
    colors = [CT_COLORS.get(eff, "#7f8c8d") for eff in df["effector"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(4.8, 0.36 * len(df) + 1.2)),
                             sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax0, ax1 = axes

    ax0.barh(y, df["log2fc"].fillna(0), color=colors, edgecolor="black",
             lw=0.45, alpha=0.88)
    for yi, (_, row) in enumerate(df.iterrows()):
        q = row["q_bh"]
        stars = _significance_stars(q) if pd.notna(q) else ""
        if stars:
            ax0.text(row["log2fc"] + np.sign(row["log2fc"]) * 0.03, yi, stars,
                     va="center", ha="left" if row["log2fc"] >= 0 else "right",
                     fontsize=9, fontweight="bold")
    ax0.axvline(0, ls=":", color="black", alpha=0.6)
    ax0.set_yticks(y)
    ax0.set_yticklabels(df["effector"], fontsize=9)
    ax0.set_xlabel("log2 shell enrichment\nhigh-b / low-b")
    ax0.set_title("All slices", loc="left", fontweight="bold", fontsize=11)
    ax0.grid(True, axis="x", alpha=0.25)

    xerr = 1.96 * df["matched_se"].fillna(0).to_numpy()
    ax1.barh(y, df["matched_effect"].fillna(0), xerr=xerr,
             color=colors, edgecolor="black", lw=0.45, alpha=0.88, capsize=2.5)
    for yi, (_, row) in enumerate(df.iterrows()):
        q = row["matched_q"]
        stars = _significance_stars(q) if pd.notna(q) else ""
        if stars:
            x = row["matched_effect"] if pd.notna(row["matched_effect"]) else 0
            ax1.text(x + np.sign(x) * 0.03, yi, stars,
                     va="center", ha="left" if x >= 0 else "right",
                     fontsize=9, fontweight="bold")
    ax1.axvline(0, ls=":", color="black", alpha=0.6)
    ax1.set_xlabel("Delta median b\nnear - far")
    ax1.set_title("Matched proximity", loc="left", fontweight="bold", fontsize=11)
    ax1.grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"{target_ct} {k_label}: cell-type relationship to spatial residual b",
                 x=0.02, y=0.995, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, savepath, dpi=400)


def make_downstream_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str,
    q_col: str,
    title: str,
    cbar_label: str,
    savepath: Path,
) -> None:
    """Cell-type x slice heatmap for per-slice downstream statistics."""
    if df.empty or value_col not in df.columns:
        return
    pivot = df.pivot_table(index="effector", columns="age", values=value_col,
                           aggfunc="mean")
    if pivot.empty:
        return
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    q_pivot = df.pivot_table(index="effector", columns="age", values=q_col,
                             aggfunc="min").reindex(index=pivot.index,
                                                    columns=pivot.columns)
    vals = pivot.to_numpy(dtype=float)
    vmax = _finite_symmetric_vmax(vals)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.38 * pivot.shape[1] + 3.5),
                                    max(4.8, 0.34 * pivot.shape[0] + 1.3)))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(_format_age_labels(pivot.columns), rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Slice age (months)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            q = q_pivot.iloc[i, j]
            if pd.notna(q) and q < 0.05:
                color = "white" if abs(vals[i, j]) > vmax * 0.55 else "black"
                ax.text(j, i, _significance_stars(q), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_downstream_effector_trend(
    df_slice_enrich: pd.DataFrame,
    df_slice_match: pd.DataFrame,
    effector: str,
    target_ct: str,
    k_label: str,
    savepath: Path,
) -> None:
    """Per-effector per-slice high-b/low-b enrichment trend."""
    sub_e = df_slice_enrich[df_slice_enrich["effector"] == effector].sort_values("age")
    if sub_e.empty:
        return

    color = CT_COLORS.get(effector, "#7f8c8d")
    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    ax.plot(sub_e["age"], sub_e["log2fc"], color=color, lw=1.6, alpha=0.85)
    ax.scatter(sub_e["age"], sub_e["log2fc"], s=34, marker="o",
               facecolor=color, edgecolor="black", lw=0.5, alpha=0.92, zorder=3)
    ax.axhline(0, ls=":", color="black", alpha=0.6)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("log2(high-b / low-b)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_downstream_effector_spatial(
    target, glob, k, demo_age: float, target_ct: str, k_label: str,
    effector: str, savepath: Path, *, max_effector_points=MAX_OVERLAY_POINTS,
) -> None:
    """Representative spatial overlay for one effector and one program."""
    target_blk = target["age"] == demo_age
    xy = target["coord"][target_blk]
    b_d = target["B"][target_blk, k]
    eff_xy = glob["coord"][(glob["age"] == demo_age) & (glob["ct"] == effector)]
    n_eff_total = int(len(eff_xy))
    if n_eff_total > max_effector_points:
        keep = np.linspace(0, n_eff_total - 1, max_effector_points, dtype=int)
        eff_xy = eff_xy[keep]

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    if xy.shape[0] < 1:
        ax.text(0.5, 0.5, f"No {target_ct} cells at {demo_age:.1f} mo",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        save_fig(fig, savepath, dpi=400)
        return

    vlim = _finite_symmetric_vmax(b_d, percentile=95, fallback=1.0)
    sc_handle = ax.scatter(xy[:, 0], xy[:, 1], c=b_d, cmap="RdBu_r",
                           vmin=-vlim, vmax=vlim, s=14, alpha=0.82,
                           edgecolor="none")
    if len(eff_xy) > 0:
        ax.scatter(eff_xy[:, 0], eff_xy[:, 1], c="black", s=9, alpha=0.55,
                   edgecolor="none", label=effector, zorder=5)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if len(eff_xy) > 0:
        ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc_handle, ax=ax, fraction=0.030, pad=0.01,
                 label=f"{k_label} spatial activity b")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def make_downstream_program_summary_figure(
    df_summary: pd.DataFrame,
    target_ct: str,
    savepath: Path,
) -> None:
    """Program x effector all-slice enrichment summary heatmap."""
    if df_summary.empty:
        return
    pivot = df_summary.pivot_table(index="effector", columns="program",
                                   values="log2fc", aggfunc="mean")
    if pivot.empty:
        return
    q_pivot = df_summary.pivot_table(index="effector", columns="program",
                                     values="q_bh", aggfunc="min").reindex(
                                         index=pivot.index, columns=pivot.columns)
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    q_pivot = q_pivot.loc[order]
    vals = pivot.to_numpy(dtype=float)
    vmax = _finite_symmetric_vmax(vals)

    fig, ax = plt.subplots(figsize=(max(4.8, 0.85 * pivot.shape[1] + 2.8),
                                    max(4.8, 0.34 * pivot.shape[0] + 1.3)))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("stGP program")
    ax.set_title(f"{target_ct}: all-slice cell-type enrichment by program",
                 loc="left", fontweight="bold", fontsize=11, pad=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            q = q_pivot.iloc[i, j]
            if pd.notna(q) and q < 0.05:
                color = "white" if abs(vals[i, j]) > vmax * 0.55 else "black"
                ax.text(j, i, _significance_stars(q), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("log2 shell enrichment (high-b / low-b)")
    fig.tight_layout()
    save_fig(fig, savepath, dpi=400)


def run_program_downstream(
    target_ct: str,
    k: int,
    *,
    target,
    glob,
    effectors: list[str],
    df_match: pd.DataFrame,
    sub_csv: Path,
    sub_fig: Path,
) -> dict:
    """Compute and render downstream cell-type enrichment for one program."""
    k_label = target["program_labels"][k]
    out_csv = Path(sub_csv) / "downstream"
    out_fig = Path(sub_fig) / "downstream"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    counts_by_eff = _shell_counts_by_effector(target, glob, effectors)
    df_all = compute_downstream_all_slices_enrichment(
        target, glob, effectors, k, counts_by_eff=counts_by_eff)
    df_slice_enrich = compute_downstream_per_slice_enrichment(
        target, glob, effectors, k, counts_by_eff=counts_by_eff)
    df_slice_match = compute_downstream_per_slice_matched_effects(
        target, glob, effectors, k)

    df_all.to_csv(out_csv / "all_slices_enrichment.csv", index=False)
    df_slice_enrich.to_csv(out_csv / "per_slice_enrichment.csv", index=False)
    df_slice_match.to_csv(out_csv / "per_slice_matched_effects.csv", index=False)

    df_match_prog = df_match[df_match["program"] == k_label].copy()
    summary = summarise_downstream(
        df_all, df_slice_enrich, df_slice_match, df_match_prog, target_ct, k_label)
    with open(out_csv / "downstream_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_downstream_all_slices_figure(
        df_all, df_match_prog, target_ct, k_label,
        out_fig / "all_slices_celltype_enrichment.png")
    make_downstream_heatmap(
        df_slice_enrich,
        value_col="log2fc",
        q_col="q_bh_by_slice",
        title=f"{target_ct} {k_label}: per-slice high-b shell enrichment",
        cbar_label="log2 shell enrichment",
        savepath=out_fig / "per_slice_enrichment_heatmap.png",
    )
    make_downstream_heatmap(
        df_slice_match,
        value_col="effect",
        q_col="q_bh_by_slice",
        title=f"{target_ct} {k_label}: per-slice matched proximity effect",
        cbar_label="Delta median b (near - far)",
        savepath=out_fig / "per_slice_matched_effect_heatmap.png",
    )

    by_ct_dir = out_fig / "by_celltype"
    for eff in effectors:
        eff_dir = by_ct_dir / safe_name(eff)
        eff_dir.mkdir(parents=True, exist_ok=True)
        make_downstream_effector_trend(
            df_slice_enrich, df_slice_match, eff, target_ct, k_label,
            eff_dir / "slice_trend.png")
        demo_age = pick_demo_slice(target, k, glob, prefer_old_with_effector=eff)
        make_downstream_effector_spatial(
            target, glob, k, demo_age, target_ct, k_label, eff,
            eff_dir / "spatial_overlay.png")

    return summary


def compile_celltype_downstream_summary(
    target_ct: str,
    *,
    csv_dir: Path,
    fig_dir: Path,
    program_labels: list[str],
) -> pd.DataFrame:
    """Collect per-program all-slice downstream rows into a cell-type summary."""
    rows = []
    for k_label in program_labels:
        p = Path(csv_dir) / k_label / "downstream" / "all_slices_enrichment.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    df_summary = pd.concat(rows, ignore_index=True)
    df_summary.to_csv(Path(csv_dir) / "downstream_program_summary.csv", index=False)
    make_downstream_program_summary_figure(
        df_summary, target_ct,
        Path(fig_dir) / "downstream_program_summary.png")
    return df_summary


# ════════════════════════════════════════════════════════════════════════════
#  Per-(target x program) orchestrator
# ════════════════════════════════════════════════════════════════════════════

def _select_significant_effectors(df_match, k_label, q_threshold=SIG_Q_THRESHOLD,
                                    max_n=12) -> list[str]:
    """Effectors with q_bh < threshold for the given program, sorted by absolute
    effect size (largest first). At most ``max_n`` are returned.
    """
    sub = df_match[df_match.program == k_label].copy()
    sub = sub[sub["q_bh"].notna() & (sub["q_bh"] < q_threshold)]
    sub["abs_effect"] = sub["effect"].abs()
    sub = sub.sort_values("abs_effect", ascending=False).head(max_n)
    return sub["effector"].tolist()


def _summarise_program(df_match, df_perm, target_ct, k_label, demo_age, target,
                       sig_effectors, var_meta) -> dict:
    """Per-program JSON summary entry."""
    top_pos = df_match[df_match.program == k_label].sort_values(
        "effect", ascending=False).head(3)
    top_neg = df_match[df_match.program == k_label].sort_values(
        "effect", ascending=True).head(3)
    perm_pass = df_perm[(df_perm.p_perm < 0.05) & (~df_perm.observed.isna())]

    return dict(
        target_celltype=target_ct,
        program=k_label,
        demo_age=demo_age,
        n_target_cells=int(len(target["age"])),
        n_predictors=int(var_meta["n_predictors"]),
        R2_full=float(var_meta["r2"]),
        sig_effectors=sig_effectors,
        top_pro_aging_effector=top_pos.iloc[0]["effector"] if len(top_pos) else "",
        top_pro_aging_effect=float(top_pos.iloc[0]["effect"]) if len(top_pos) else np.nan,
        top_pro_rejuv_effector=top_neg.iloc[0]["effector"] if len(top_neg) else "",
        top_pro_rejuv_effect=float(top_neg.iloc[0]["effect"]) if len(top_neg) else np.nan,
        n_perm_pass=int(len(perm_pass)),
        perm_pass_effectors="|".join(perm_pass.effector.tolist()) if len(perm_pass) else "",
    )


DEFAULT_GENESETS_DIR = Path("data/genesets")

# Canonical gene-set collections shipped with the notebook.
DEFAULT_GENE_SETS: dict[str, str] = {
    "GO Biological process": "m5.go.bp.v2026.1.Mm.symbols.gmt",
    "GO Molecular Function": "m5.go.mf.v2026.1.Mm.symbols.gmt",
    "GO Cellular Component": "m5.go.cc.v2026.1.Mm.symbols.gmt",
    "Cell-type signatures":  "m8.all.v2026.1.Mm.symbols.gmt",
}

# Programs with fewer than this many positive-weight genes are skipped: an
# enrichment test on 1-3 genes is statistically meaningless.
MIN_POS_GENES = 5


# ════════════════════════════════════════════════════════════════════════════
# Core logic
# ════════════════════════════════════════════════════════════════════════════

def _resolve_gene_sets(geneset_names: list[str] | None,
                        gsets_dir: Path) -> dict[str, Path]:
    """Resolve user-provided gene-set names into a ``{name: gmt_path}`` mapping.

    Raises ``ValueError`` for unknown names and ``FileNotFoundError`` for
    missing ``.gmt`` files.
    """
    if geneset_names is None:
        selected = DEFAULT_GENE_SETS
    else:
        missing = [n for n in geneset_names if n not in DEFAULT_GENE_SETS]
        if missing:
            raise ValueError(
                f"Unknown gene set(s): {missing}. "
                f"Available: {list(DEFAULT_GENE_SETS.keys())}"
            )
        selected = {n: DEFAULT_GENE_SETS[n] for n in geneset_names}

    resolved: dict[str, Path] = {}
    missing_files: list[str] = []
    for name, fname in selected.items():
        path = gsets_dir / fname
        if not path.exists():
            missing_files.append(str(path))
            continue
        resolved[name] = path

    if missing_files:
        raise FileNotFoundError(
            "Missing gene-set files:\n  " + "\n  ".join(missing_files) +
            f"\n(expected under {gsets_dir})"
        )
    return resolved


def _load_W(stgp_root: Path, celltype: str) -> pd.DataFrame:
    """Return the stGP gene-weight matrix (programs x genes) for one cell type."""
    W_path = stgp_root / safe_name(celltype) / "W.csv"
    if not W_path.exists():
        raise FileNotFoundError(f"W.csv not found: {W_path}")
    W = pd.read_csv(W_path, index_col=0)
    W.index = W.index.astype(str)
    W.columns = W.columns.astype(str)
    return W


def _enrich_one_program(
    *,
    program: str,
    weights: pd.Series,
    background_genes: list[str],
    gene_sets: dict[str, Path],
    celltype: str,
    safe: str,
    fig_dir: Path,
    res_dir: Path,
    padj_threshold: float,
    n_top: int,
) -> tuple[dict, pd.DataFrame | None]:
    """Run gseapy.enrich for one program. Returns (per-program stats dict,
    combined per-program DataFrame or None when no gene set returned terms).
    """
    s = weights.astype(float)
    gene_list = s[s > 0].sort_values(ascending=False).index.astype(str).tolist()
    n_pos = len(gene_list)

    if n_pos < MIN_POS_GENES:
        print(f"  [skip program] {program}: only {n_pos} positive genes "
              f"(< MIN_POS_GENES = {MIN_POS_GENES})")
        return dict(program=program, n_pos_genes=n_pos,
                    status="skipped_too_few_genes"), None

    t_prog = time.perf_counter()
    print(f"  [program] {program}: {n_pos} positive genes ...")

    fig, results = run_enrichment_for_program(
        gene_list=gene_list,
        background_genes=background_genes,
        gene_sets=gene_sets,
        padj_threshold=padj_threshold,
        n_top=n_top,
        title=f"{celltype} - {program}",
    )

    fig_path = fig_dir / f"{safe}_{program}_enrichment.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close(fig)

    per_prog = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not per_prog.empty:
        per_prog.insert(0, "program", program)
        per_prog.insert(1, "n_genes_input", n_pos)
        per_prog.to_csv(res_dir / f"{safe}_{program}_enrichment.csv", index=False)

    stats = dict(
        program=program,
        n_pos_genes=n_pos,
        n_terms_reported=int(len(per_prog)),
        runtime_sec=round(time.perf_counter() - t_prog, 2),
        status="done",
        figure=str(fig_path),
    )
    return stats, (per_prog if not per_prog.empty else None)


def _plot_program_enrichment_dotplot(
    combined: pd.DataFrame,
    *,
    out: Path,
    padj_threshold: float,
    top_terms: int = 16,
) -> None:
    """Program x pathway dot plot from the combined per-celltype enrichment table."""
    need = {"program", "gene_set", "Term", "Combined Score", "Adjusted P-value"}
    if combined.empty or not need.issubset(combined.columns):
        return

    df = combined.dropna(subset=["Term", "Combined Score", "Adjusted P-value"]).copy()
    df = df[(df["Combined Score"] > 0) & (df["Adjusted P-value"] > 0)]
    df = df[df["Adjusted P-value"] < padj_threshold]
    if df.empty:
        return

    df["term_label"] = df["Term"].astype(str)
    df["nlog10_padj"] = -np.log10(np.clip(df["Adjusted P-value"].astype(float), 1e-50, None))
    df = df.sort_values("Combined Score", ascending=False)
    top = (
        df.groupby("program", as_index=False, sort=False)
          .head(4)
          .sort_values("Combined Score", ascending=False)
    )
    terms = top.drop_duplicates("term_label").head(top_terms)["term_label"].tolist()
    if not terms:
        return
    plot_df = df[df["term_label"].isin(terms)].copy()

    programs = sorted(plot_df["program"].astype(str).unique().tolist())
    term_order = (
        plot_df.groupby("term_label")["Combined Score"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    x_lookup = {p: i for i, p in enumerate(programs)}
    y_lookup = {t: i for i, t in enumerate(term_order)}
    x = plot_df["program"].astype(str).map(x_lookup).to_numpy()
    y = plot_df["term_label"].map(y_lookup).to_numpy()
    score = plot_df["Combined Score"].astype(float).to_numpy()
    nlogp = plot_df["nlog10_padj"].astype(float).to_numpy()
    sizes = 35 + 430 * (score - score.min()) / (score.max() - score.min() + 1e-12)

    fig_h = max(4.5, 0.42 * len(term_order) + 1.6)
    fig_w = max(5.6, 0.92 * len(programs) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sc = ax.scatter(
        x, y, c=nlogp, s=sizes, cmap="magma",
        edgecolors="black", linewidths=0.35, alpha=0.88,
    )
    ax.set_xticks(range(len(programs)))
    ax.set_xticklabels(programs, rotation=30, ha="right")
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order)
    ax.set_xlabel("stGP program")
    ax.set_ylabel("")
    ax.grid(True, axis="both", color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"$-\log_{10}$(adj. $p$-value)")

    for frac in [0.25, 0.55, 0.85]:
        val = score.min() + frac * (score.max() - score.min())
        size = 35 + 430 * (val - score.min()) / (score.max() - score.min() + 1e-12)
        ax.scatter([], [], s=size, c="#777777", edgecolors="black",
                   linewidths=0.35, label=f"{val:.0f}")
    ax.legend(
        title="Combined\nscore", loc="lower right",
        frameon=False, borderaxespad=0.3, labelspacing=1.2,
    )
    fig.savefig(out, bbox_inches="tight", dpi=400)
    plt.close(fig)


def compile_enrichment_master_summary(
    celltypes: list[str],
    *,
    results_root: Path,
    top_per_program: int = 5,
) -> None:
    """Cross-cell-type table of the top-``top_per_program`` terms per
    ``(celltype, program, gene_set)``.
    """
    rows: list[pd.DataFrame] = []
    for ct in celltypes:
        combined = results_root / safe_name(ct) / f"{safe_name(ct)}_combined_enrichment.csv"
        if not combined.exists():
            continue
        df = pd.read_csv(combined)
        if df.empty or "Combined Score" not in df.columns:
            continue
        df = df.sort_values("Combined Score", ascending=False)
        top = (
            df.groupby(["program", "gene_set"], as_index=False, sort=False)
              .head(top_per_program)
              .copy()
        )
        rows.append(top)

    if not rows:
        print("[master] No per-cell-type enrichment CSVs found; skipping master summary.")
        return

    master = pd.concat(rows, ignore_index=True)
    out = results_root / "summary_all_celltypes.csv"
    master.to_csv(out, index=False)
    print(f"\n[master] Wrote {len(master):,} rows -> {out}")
