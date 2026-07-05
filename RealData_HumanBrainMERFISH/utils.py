"""Shared constants and analysis helpers for Human Brain MERFISH analyses.

Human-specific pipeline conventions:
    - Slice / sample grouping uses ``id_region`` (donor_id + ``_`` + region_id,
      set in ``preprocess_qc.py``) instead of ``mouse_id``.
    - ``age`` is the per-cell temporal covariate for stGP and plots.
    - The dataset has no atlas region annotation, so ``celltype2`` is used as a
      region-like covariate for variance partition when available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors


# Cell-type lists

# Alphabetical order -- used by tables and summaries.
ALL_CELLTYPES: list[str] = [
    "ast",      # Astrocyte
    "endo",     # Endothelial
    "ext",      # Excitatory neuron
    "inb",      # Inhibitory neuron
    "micro",    # Microglia
    "oli",      # Oligodendrocyte
    "opc",      # OPC
]

# Pretty labels used in figures / reports.
CELLTYPE_LABELS: dict[str, str] = {
    "ast": "Astrocyte",
    "endo": "Endothelial",
    "ext": "Excitatory neuron",
    "inb": "Inhibitory neuron",
    "micro": "Microglia",
    "oli": "Oligodendrocyte",
    "opc": "OPC",
}

# Run order used by stGP drivers. Microglia first so the headline
# non-neuronal cell type appears quickly when running the full pipeline.
RUN_ORDER_CELLTYPES: list[str] = [
    "micro", "endo", "opc", "ast", "oli", "inb", "ext",
]

# Run order used by baseline drivers. Smallest cell types first so slower
# baselines clear quickly on the cheap cell types.
BASELINE_RUN_ORDER: list[str] = [
    "micro", "endo", "opc", "ast", "inb", "oli", "ext",
]


# Default directory layout (relative to the repo root)

DEFAULT_RAW_DATA_DIR = Path(
    os.environ.get("STGP_HUMAN_RAW_DIR", "data/raw/HumanBrainMERFISH")
)
DATA_QC = Path("data/qc/human_merfish_qc.h5ad")
DATA_PROCESSED = Path("data/processed")
RESULTS_STGP = Path("Results/stgp")
RESULTS_BASELINES = Path("Results/baselines")
RESULTS_PROXIMITY = Path("Results/proximity")
FIGURES_ROOT = Path("Figures")
TIMING_LOG = Path("Results/benchmark_runtimes.jsonl")


# Region covariate for variance partition (human MERFISH has no atlas regions).

# Finer subtype labels (e.g. cortical layer for excitatory neurons); used as a
# region-like covariate in OLS variance partition when present with >=2 levels.
REGION_COL = "celltype2"


# Naming helpers

def safe_name(celltype: str) -> str:
    """Filesystem-safe variant of a cell-type name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in celltype)


# Analysis helpers

def as_1d_array(x) -> np.ndarray:
    """Return ``x`` as a dense one-dimensional NumPy array."""
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def best_program_by_correlation(
    adata,
    obsm_key: str,
    expr,
    *,
    use_abs_for: tuple[str, ...] = ("X_mefisto", "X_spatialpca"),
) -> int:
    """Program index with the strongest Pearson correlation to ``expr``.

    Signed methods such as MEFISTO and SpatialPCA are compared by absolute
    correlation, matching the exploratory notebook behavior.
    """
    scores = np.asarray(adata.obsm[obsm_key])
    y = as_1d_array(expr)
    corrs = [
        np.corrcoef(scores[:, k], y)[0, 1]
        for k in range(scores.shape[1])
    ]
    corrs = np.asarray(corrs, dtype=float)
    if obsm_key in use_abs_for:
        return int(np.nanargmax(np.abs(corrs)))
    return int(np.nanargmax(corrs))


def spectral_knn_labels(
    X,
    n_clusters: int,
    *,
    random_state: int = 0,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster rows of ``X`` using a symmetric KNN graph and spectral clustering."""
    X = np.asarray(X)
    if X.shape[0] <= n_clusters:
        raise ValueError(f"n_obs={X.shape[0]} must be > n_clusters={n_clusters}")
    k_nn = min(max(2, int(np.round(np.sqrt(X.shape[0])))), X.shape[0] - 1)
    nn = (
        NearestNeighbors(n_neighbors=k_nn + 1, metric=metric)
        .fit(X)
        .kneighbors(return_distance=False)[:, 1:]
    )
    rows = np.repeat(np.arange(nn.shape[0]), k_nn)
    cols = nn.ravel()
    graph = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(X.shape[0], X.shape[0]))
    graph = graph.maximum(graph.T)
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    ).fit_predict(graph)
    return labels.astype(str)


def majority_merge(raw_pred, y_true) -> tuple[np.ndarray, dict[str, str]]:
    """Map each raw cluster to the majority ground-truth class."""
    mapping: dict[str, str] = {}
    y_true = pd.Series(y_true).astype(str).to_numpy()
    raw_pred = pd.Series(raw_pred).astype(str).to_numpy()
    for cluster_id in pd.unique(raw_pred):
        vals = y_true[raw_pred == cluster_id]
        mapping[str(cluster_id)] = (
            "unassigned" if len(vals) == 0 else str(pd.Series(vals).value_counts().idxmax())
        )
    merged = np.array([mapping[str(cluster_id)] for cluster_id in raw_pred], dtype=object)
    return merged, mapping


def hungarian_accuracy(y_true, y_pred) -> float:
    """Best one-to-one label matching accuracy."""
    y_true = pd.Series(y_true).astype(str).to_numpy()
    y_pred = pd.Series(y_pred).astype(str).to_numpy()
    table = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    row_ind, col_ind = linear_sum_assignment(-table.to_numpy())
    return float(table.to_numpy()[row_ind, col_ind].sum() / len(y_true))


def evaluate_cluster_labels(y_true, raw_pred) -> tuple[dict[str, float], np.ndarray, dict[str, str]]:
    """Evaluate raw labels and majority-merged labels against ground truth."""
    raw_pred = pd.Series(raw_pred).astype(str).to_numpy()
    y_true = pd.Series(y_true).astype(str).to_numpy()
    merged_pred, mapping = majority_merge(raw_pred, y_true)
    metrics = {
        "raw_ari": adjusted_rand_score(y_true, raw_pred),
        "raw_nmi": normalized_mutual_info_score(y_true, raw_pred),
        "raw_hungarian_acc": hungarian_accuracy(y_true, raw_pred),
        "merged_ari": adjusted_rand_score(y_true, merged_pred),
        "merged_nmi": normalized_mutual_info_score(y_true, merged_pred),
        "merged_acc": accuracy_score(y_true, merged_pred),
        "merged_balanced_acc": balanced_accuracy_score(y_true, merged_pred),
        "merged_macro_f1": f1_score(y_true, merged_pred, average="macro"),
        "n_raw_labels": len(pd.unique(raw_pred)),
        "n_merged_labels": len(pd.unique(merged_pred)),
    }
    return metrics, merged_pred, mapping
