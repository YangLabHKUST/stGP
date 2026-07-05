from __future__ import annotations
from dataclasses import dataclass, field
import os
import pickle
import sys
import tempfile
import time
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _default_torch_device() -> str:
    """Return 'cuda' when a CUDA device is available, otherwise 'cpu'."""
    try:
        if torch.cuda.is_available():
            return "cuda:3"
    except ImportError:
        pass
    return "cpu"

_TORCH_DEVICE = _default_torch_device()

from stgp.estimation import (
    fit_pfactor,
    recover_low_rank_signal,
)

from scipy.sparse import block_diag as sp_block_diag, coo_matrix



def is_popari_available() -> bool:
    try:
        from popari.model import Popari  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class MethodResult:
    """
    Container for a fitted method.
    ----------
    name:
        Short identifier (e.g. ``"stGP"``, ``"PCA"``).
    W:
        (p, G) loading / gene-program weights (rows correspond to programs).
    H:
        (N, p) cell-by-program embeddings.
    alpha:
        (p, T) temporal effects per program (stGP only). 
    alpha_std:
        (p, T) posterior standard deviation of alpha (stGP only).
    alpha_lower:
        (p, T) lower bound of the posterior credible interval for alpha (stGP only).
    alpha_upper:
        (p, T) upper bound of the posterior credible interval for alpha (stGP only).
    b:
        (N, p) spatial effect embeddings. 
    theta:
        Optional variance components per program, relevant for stGP.
    Y_hat:
        (N, G) reconstructed expression.
    metadata:
        Additional information (runtime, iteration counts, raw model objects,
        etc.).
    """

    name: str
    W: np.ndarray
    H: np.ndarray
    alpha: Optional[np.ndarray]
    b: np.ndarray
    Y_hat: np.ndarray
    theta: Optional[np.ndarray] = None
    alpha_std: Optional[np.ndarray] = None
    alpha_lower: Optional[np.ndarray] = None
    alpha_upper: Optional[np.ndarray] = None
    metadata: MutableMapping[str, object] = field(default_factory=dict)


def method_result_to_dict(res: MethodResult) -> Dict[str, object]:
    return {
        "name": res.name,
        "W": res.W,
        "H": res.H,
        "alpha": res.alpha,
        "alpha_std": res.alpha_std,
        "alpha_lower": res.alpha_lower,
        "alpha_upper": res.alpha_upper,
        "b": res.b,
        "Y_hat": res.Y_hat,
        "theta": res.theta,
        "metadata": dict(res.metadata) if res.metadata is not None else {},
    }


def method_result_from_dict(payload: Mapping[str, object]) -> MethodResult:
    return MethodResult(
        name=str(payload.get("name", "")),
        W=payload.get("W"),
        H=payload.get("H"),
        alpha=payload.get("alpha"),
        alpha_std=payload.get("alpha_std"),
        alpha_lower=payload.get("alpha_lower"),
        alpha_upper=payload.get("alpha_upper"),
        b=payload.get("b"),
        Y_hat=payload.get("Y_hat"),
        theta=payload.get("theta"),
        metadata=payload.get("metadata", {}) or {},
    )

def save_method_result(
    path: str,
    result: MethodResult,
    *,
    params: Optional[Mapping[str, object]] = None,
    seed: Optional[int] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """
    Serialize a MethodResult with optional metadata (params/seed).
    """
    payload = {"method": method_result_to_dict(result)}
    if params is not None:
        payload["params"] = dict(params)
    if seed is not None:
        payload["seed"] = int(seed)
    if extra is not None:
        payload["extra"] = dict(extra)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_method_result(path: str) -> Tuple[MethodResult, Dict[str, object]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "method" in payload:
        method = method_result_from_dict(payload["method"])
        return method, payload
    # Backward compatibility: allow direct MethodResult pickles or dicts.
    if isinstance(payload, MethodResult):
        return payload, {"method": method_result_to_dict(payload)}
    if isinstance(payload, dict):
        return method_result_from_dict(payload), {"method": payload}
    raise ValueError(f"Unrecognized payload type in {path}: {type(payload)}")

def flatten_spatial_programs(b_list: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """
    Convert the nested ``B_list`` (list over programs, each a list over samples)
    into an array of shape ``(p, N)`` for easier comparison.
    """
    flat = []
    for program_blocks in b_list:
        flat.append(np.concatenate(program_blocks))
    return np.asarray(flat)


def stack_Y_list(Y_list: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``np.vstack(Y_list)`` with defensive copying."""
    return np.ascontiguousarray(np.vstack(Y_list), dtype=float)


def _centered_gene_stats(Y_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """
    Summarize per-gene means for centered Gaussian views.
    """
    Y = stack_Y_list(Y_list)
    gene_means = np.asarray(Y.mean(axis=0), dtype=float)
    return {
        "gene_mean_abs_max": float(np.max(np.abs(gene_means))),
        "gene_mean_abs_mean": float(np.mean(np.abs(gene_means))),
    }


def _warn_if_not_centered(stats: Mapping[str, float], *, name: str, tol: float = 1e-2) -> None:
    if stats.get("gene_mean_abs_max", 0.0) > tol:
        print(f"[warn] {name} input not well centered (abs max mean={stats['gene_mean_abs_max']:.3e}).")


def _maybe_centered_stats(data: Mapping[str, object], *, name: str) -> Optional[Dict[str, float]]:
    if "Y_log_centered_list" not in data:
        return None
    stats = _centered_gene_stats(data["Y_list"])
    _warn_if_not_centered(stats, name=name)
    return stats


def _infer_gaussian_view(data: Mapping[str, object]) -> str:
    """Infer which Gaussian view is present in data."""
    if "Y_log_centered_list" in data:
        return "log_centered"
    return "gaussian"


def _minmax01(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values.")
    lo = np.min(x)
    hi = np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{name} has zero or invalid range.")
    return (x - lo) / (hi - lo)


def _scale_mefisto_covariates(
    data: Mapping[str, object],
    *,
    covariate_mode: str = "age_xy",
    spatial_scale: str = "global",
) -> Tuple[Sequence[np.ndarray], Sequence[str]]:
    """Build per-slice MEFISTO covariates in age/x/y order, scaled to [0, 1]."""
    Nlist = [int(n) for n in data["Nlist"]]
    T = len(Nlist)
    ages = np.asarray(data["ages"], dtype=float)
    if ages.shape[0] != T:
        raise ValueError(f"Expected {T} ages, found {ages.shape[0]}.")
    if np.unique(ages).size < 2:
        raise ValueError("Need at least two distinct ages for temporal MEFISTO.")

    age01 = _minmax01(ages, "age")
    covariates = [np.full((Nlist[t], 1), age01[t], dtype=float) for t in range(T)]
    covariate_names = ["age"]

    if covariate_mode == "age":
        return covariates, covariate_names
    if covariate_mode != "age_xy":
        raise ValueError("covariate_mode must be one of {'age', 'age_xy'}")
    if "coords_list" not in data:
        raise ValueError("age_xy MEFISTO requires data['coords_list'].")

    coords_list = [np.asarray(c, dtype=float) for c in data["coords_list"]]
    if len(coords_list) != T:
        raise ValueError(f"Expected {T} coordinate blocks, found {len(coords_list)}.")
    for t, coords in enumerate(coords_list):
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"coords_list[{t}] must have at least two columns.")
        if coords.shape[0] != Nlist[t]:
            raise ValueError(
                f"coords_list[{t}] has {coords.shape[0]} rows, expected {Nlist[t]}."
            )
        if not np.isfinite(coords[:, :2]).all():
            raise ValueError(f"coords_list[{t}] contains non-finite x/y values.")

    if spatial_scale == "global":
        xy = np.vstack([c[:, :2] for c in coords_list])
        xy01 = np.column_stack([_minmax01(xy[:, 0], "global x"), _minmax01(xy[:, 1], "global y")])
        cuts = cumulative_counts(Nlist)
        for t in range(T):
            covariates[t] = np.column_stack([covariates[t], xy01[cuts[t]:cuts[t + 1]]])
    elif spatial_scale == "within_slice":
        for t, coords in enumerate(coords_list):
            xy01 = np.column_stack([
                _minmax01(coords[:, 0], f"x in slice {t}"),
                _minmax01(coords[:, 1], f"y in slice {t}"),
            ])
            covariates[t] = np.column_stack([covariates[t], xy01])
    else:
        raise ValueError("spatial_scale must be one of {'global', 'within_slice'}")

    covariate_names += ["x", "y"]
    return covariates, covariate_names


def _knn_adjacency_matrix(coords: np.ndarray, n_neighbors: int = 8):
    """
    Build a symmetric kNN adjacency matrix from coordinates.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("n_neighbors must be positive")
    k = min(k, max(n - 1, 1))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    data = np.ones_like(cols, dtype=float)
    mat = coo_matrix((data, (rows, cols)), shape=(n, n))
    mat = mat.maximum(mat.T)
    return mat.tocsr()


def _block_diag_adjacency(coords_list: Sequence[np.ndarray], n_neighbors: int = 8):
    """
    Build a block-diagonal adjacency matrix from a list of coordinate arrays.
    """
    mats = []
    for coords in coords_list:
        mat = _knn_adjacency_matrix(coords, n_neighbors=n_neighbors)
        mats.append(mat)
    return sp_block_diag(mats, format="csr")


def cumulative_counts(Nlist: Sequence[int]) -> np.ndarray:
    """Return the cumulative cell counts (useful for slicing stacked arrays)."""
    return np.cumsum(np.concatenate(([0], np.asarray(Nlist, dtype=int))))


def alpha_b_from_cell_embeddings(
    H: np.ndarray,
    Nlist: Sequence[int],
    center_blocks: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given cell-level embeddings ``H`` (N x p), recover
    ``alpha`` (p x T) and ``b`` (N x p) by averaging over each individual.
    """
    H = np.asarray(H, dtype=float)
    N = H.shape[0]
    cuts = cumulative_counts(Nlist)
    T = len(Nlist)
    p = H.shape[1]
    alpha = np.zeros((p, T), dtype=float)
    for t in range(T):
        block = H[cuts[t]:cuts[t + 1]]
        alpha[:, t] = block.mean(axis=0)
    alpha_cells = np.repeat(alpha.T, repeats=Nlist, axis=0)
    b = H - alpha_cells
    if center_blocks:
        # ensure spatial residuals sum to zero within each block and adjust alpha
        for t in range(T):
            block = b[cuts[t]:cuts[t + 1]]
            offset = block.mean(axis=0)
            b[cuts[t]:cuts[t + 1]] = block - offset
            alpha[:, t] += offset
    return alpha, b


def demean_blocks(H: np.ndarray, Nlist: Sequence[int]) -> np.ndarray:
    """
    Block-demean embeddings within each individual/time block.

    This is used for baselines that do not have an explicit temporal component:
    we avoid constructing a pseudo-``alpha`` and instead keep only the centered
    per-cell variation as a proxy for spatial structure.
    """
    H = np.asarray(H, dtype=float)
    cuts = cumulative_counts(Nlist)
    out = H.copy()
    for t in range(len(Nlist)):
        block = slice(int(cuts[t]), int(cuts[t + 1]))
        if block.start >= block.stop:
            continue
        mu = out[block].mean(axis=0, keepdims=True)
        out[block] = out[block] - mu
    return out


def pack_method_result(
    name: str,
    W: np.ndarray,
    H: np.ndarray,
    Y: np.ndarray,
    Nlist: Sequence[int],
    *,
    infer_temporal: bool = True,
    theta: Optional[np.ndarray] = None,
    metadata: Optional[MutableMapping[str, object]] = None,
) -> MethodResult:
    """
    Build a :class:`MethodResult`.

    - For methods with an explicit temporal component (stGP-like), set
      ``infer_temporal=True`` to compute a block-mean ``alpha`` and residual ``b``.
    - For baselines without temporal structure (PCA/SpatialPCA), set
      ``infer_temporal=False``; we store ``alpha=None`` and ``b`` as block-demeaned
      embeddings.
    """
    if infer_temporal:
        alpha, b = alpha_b_from_cell_embeddings(H, Nlist)
    else:
        alpha = None
        b = demean_blocks(H, Nlist)
    Y_hat = H @ W
    return MethodResult(
        name=name,
        W=np.asarray(W, dtype=float),
        H=np.asarray(H, dtype=float),
        alpha=alpha,
        b=b,
        Y_hat=Y_hat,
        theta=None if theta is None else np.asarray(theta, dtype=float),
        metadata={} if metadata is None else metadata,
    )


def true_quantities_from_datagen(data: Mapping[str, object]) -> Dict[str, np.ndarray]:
    """
    Convenience extractor for ``DataGen`` outputs (true programs / embeddings).

    Handles both spatial (``data["K_spa"] is not None``) and no-spatial
    (``data["K_spa"] is None``) datasets produced by ``DataGen``/``DataGenCounts``
    with ``spatial=False``.
    """
    Y = stack_Y_list(data["Y_list"])
    signal = stack_Y_list(data["Signal_list"])
    W_true = np.asarray(data["W"])
    H_true = np.vstack(data["H_list"])
    alpha_true = np.asarray(data["Alpha"])
    b_true = flatten_spatial_programs(data["B_list"])
    out = {
        "Y": Y,
        "signal": signal,
        "W": W_true,
        "H": H_true,
        "alpha": alpha_true,
        "b": b_true,
        "sigma2_age": np.asarray(data["sigma2_age"]),
        "tau2_spa": np.asarray(data["tau2_spa"]),
        "sigma2_e": float(data["sigma2_e"]),
        "spatial": bool(data.get("spatial", data.get("K_spa") is not None)),
    }
    if "Y_count_list" in data:
        out["Y_count"] = stack_Y_list(data["Y_count_list"])
    if "Mu_count_list" in data:
        out["signal_count"] = stack_Y_list(data["Mu_count_list"])
    return out


# ----------------------------------------------------------------------
# Method-specific runners
# ----------------------------------------------------------------------


def run_stgp_pfactor(
    data: Mapping[str, object],
    *,
    p: int,
    k: Optional[int] = None,
    max_sweeps: int = 500,
    inner_rank1_iters: int = 500,
    random_state: int = 0,
    theta_update: str = "mm",
    mom_enforce_nonneg: bool = False,
    method_name: Optional[str] = None,
    **kwargs,
) -> MethodResult:
    """
    Multi-factor stGP wrapper using :func:`fit_pfactor`.

    When ``data["K_spa"]`` is None (no-spatial / scRNA-seq mode), ``Kspa_list``
    is passed as None to ``fit_pfactor``, activating the temporal-only code path.
    """
    K_age = data.get("K_age_fit", data["K_age"])
    K_spa = data.get("K_spa_fit", data.get("K_spa"))  # None for no-spatial datasets
    start = time.perf_counter()
    res = fit_pfactor(
        data["Y_list"],
        data["Nlist"],
        K_age,
        K_spa,
        p=p,
        k=k,
        max_sweeps=max_sweeps,
        inner_rank1_iters=inner_rank1_iters,
        random_state=random_state,
        theta_update=theta_update,
        mom_enforce_nonneg=mom_enforce_nonneg,
        **kwargs,
    )
    runtime = time.perf_counter() - start
    theta = np.asarray(res["theta"], dtype=float)
    input_view = _infer_gaussian_view(data)
    centered_stats = _maybe_centered_stats(data, name="stGP-rankp")
    metadata = {
        "runtime": runtime,
        "sigma2_e": res["sigma2e"],
        "n_sweeps": res["info"]["n_sweeps"],
        "converged": res["info"]["converged"],
        "n_merges": res["info"].get("n_merges", 0),
        "input_view": input_view,
        "centered_gene_stats": centered_stats,
    }
    if method_name is None:
        method_name = "stGP-rankp" if theta_update == "mm" else f"stGP-rankp-{theta_update}"
    return MethodResult(
        name=method_name,
        W=np.asarray(res["W"], dtype=float),
        H=np.asarray(res["H"], dtype=float),
        alpha=np.asarray(res["alpha"], dtype=float),
        alpha_std=np.asarray(res["alpha_std"], dtype=float),
        alpha_lower=np.asarray(res["alpha_lower"], dtype=float),
        alpha_upper=np.asarray(res["alpha_upper"], dtype=float),
        b=np.asarray(res["b"], dtype=float),
        Y_hat=np.asarray(res["H"], dtype=float) @ np.asarray(res["W"], dtype=float),
        theta=theta,
        metadata=metadata,
    )


def run_pca_baseline(
    data: Mapping[str, object],
    *,
    p: int,
) -> MethodResult:
    """
    PCA baseline (computed via singular value decomposition) using
    :func:`recover_low_rank_signal`.
    """
    Y = stack_Y_list(data["Y_list"])
    Y_hat, U, S, Vt = recover_low_rank_signal(Y, p)
    H = U[:, :p] * S[:p]
    W = Vt[:p, :]
    input_view = _infer_gaussian_view(data)
    centered_stats = _maybe_centered_stats(data, name="PCA")
    metadata = {
        "runtime": np.nan,
        "singular_values": S[:p],
        "input_view": input_view,
        "centered_gene_stats": centered_stats,
    }
    out = pack_method_result(
        name="PCA",
        W=W,
        H=H,
        Y=Y,
        Nlist=data["Nlist"],
        infer_temporal=False,
        metadata=metadata,
    )
    out.Y_hat = np.asarray(Y_hat, dtype=float)
    return out


def run_nmf_baseline(
    data: Mapping[str, object],
    *,
    p: int,
    random_state: int = 0,
) -> MethodResult:
    """
    Non-negative Matrix Factorization baseline (sklearn).

    NMF requires non-negative input.  We use the log1p-normalised view
    ``Y_log_norm_list`` (normalize_total -> log1p, always >= 0).  If that view
    is absent we fall back to shifting the raw data to be non-negative.
    """
    from sklearn.decomposition import NMF

    if "Y_log_norm_list" in data:
        Y = np.ascontiguousarray(np.vstack(data["Y_log_norm_list"]), dtype=float)
    else:
        Y_raw = np.vstack(data["Y_list"]).astype(float)
        col_min = Y_raw.min(axis=0, keepdims=True)
        Y = np.maximum(Y_raw - col_min, 0.0)

    t0 = time.perf_counter()
    model = NMF(
        n_components=p,
        random_state=random_state,
        max_iter=1000,
        init="nndsvda",
    )
    H_nmf = model.fit_transform(Y)   # (N, p)
    W_nmf = model.components_        # (p, G)
    runtime = time.perf_counter() - t0

    Y_hat = H_nmf @ W_nmf
    b = demean_blocks(H_nmf, data["Nlist"])
    return MethodResult(
        name="NMF",
        W=W_nmf,
        H=H_nmf,
        alpha=None,
        b=b,
        Y_hat=Y_hat,
        theta=None,
        metadata={"runtime": runtime, "input_view": "log_norm"},
    )


def run_spatialpca_baseline(
    data: Mapping[str, object],
    *,
    n_components: int,
    bandwidth: Optional[float] = None,
    bandwidth_common: Optional[float] = 0.1,
    kernel_type: str = "gaussian",
    fast: bool = True,
    eigenvec_num: Optional[int] = None,
    remove_batch_effect: bool = False,
) -> MethodResult:
    """
    SpatialPCA baseline — joint multi-sample fit.

    Fits a single SpatialPCA model on all samples simultaneously using a
    block-diagonal spatial kernel.  All slices share the same loadings W and
    the same tau, so component ordering is consistent across slices.  This is
    required for meaningful cross-slice comparisons (program recovery, global
    clustering, temporal analysis).

    When ``bandwidth`` is None, ``bandwidth_common`` is passed through to
    ``multi_sample_spatialpca`` (R default 0.1; set to None for Silverman).
    """
    from SpatialPCA import multi_sample_spatialpca

    start = time.perf_counter()

    expr_list = [Y_t.T for Y_t in data["Y_list"]]
    model, pcs_list, *_ = multi_sample_spatialpca(
        expr_list,
        data["coords_list"],
        n_components=n_components,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        bandwidth_common=bandwidth_common,
        fast=fast,
        eigenvec_num=eigenvec_num,
        remove_batch_effect=remove_batch_effect,
    )
    H_blocks = [pcs.T for pcs in pcs_list]
    H = np.vstack(H_blocks)
    W = model.W_.T
    metadata = {
        "runtime": time.perf_counter() - start,
        "tau": model.tau_,
        "sigma2": model.sigma2_,
    }

    input_view = _infer_gaussian_view(data)
    centered_stats = _maybe_centered_stats(data, name="SpatialPCA")
    metadata["input_view"] = input_view
    metadata["centered_gene_stats"] = centered_stats

    return pack_method_result(
        name="SpatialPCA",
        W=W,
        H=H,
        Y=stack_Y_list(data["Y_list"]),
        Nlist=data["Nlist"],
        infer_temporal=False,
        metadata=metadata,
    )


def run_spatialpca_nz_baseline(
    data: Mapping[str, object],
    *,
    n_components: int,
    bandwidth: Optional[float] = None,
    bandwidth_common: Optional[float] = 0.1,
    kernel_type: str = "gaussian",
    fast: bool = True,
    eigenvec_num: Optional[int] = None,
    remove_batch_effect: bool = False,
) -> MethodResult:
    """
    SpatialPCA baseline **without** gene-wise z-scoring.

    The original SpatialPCA z-scores each gene to mean=0, std=1, which
    destroys the variance structure that distinguishes program-active from
    inactive genes.  This variant only gene-centers (demean) without
    dividing by std, preserving relative gene variance while still
    removing gene-level offsets.
    """
    from SpatialPCA import multi_sample_spatialpca

    start = time.perf_counter()

    expr_list = [Y_t.T for Y_t in data["Y_list"]]
    model, pcs_list, *_ = multi_sample_spatialpca(
        expr_list,
        data["coords_list"],
        n_components=n_components,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        bandwidth_common=bandwidth_common,
        fast=fast,
        eigenvec_num=eigenvec_num,
        remove_batch_effect=remove_batch_effect,
        zscore_genes=False,
    )
    H_blocks = [pcs.T for pcs in pcs_list]
    H = np.vstack(H_blocks)
    W = model.W_.T
    metadata = {
        "runtime": time.perf_counter() - start,
        "tau": model.tau_,
        "sigma2": model.sigma2_,
    }

    input_view = _infer_gaussian_view(data)
    centered_stats = _maybe_centered_stats(data, name="SpatialPCA-nz")
    metadata["input_view"] = input_view
    metadata["centered_gene_stats"] = centered_stats
    metadata["zscore_genes"] = False

    return pack_method_result(
        name="SpatialPCA-nz",
        W=W,
        H=H,
        Y=stack_Y_list(data["Y_list"]),
        Nlist=data["Nlist"],
        infer_temporal=False,
        metadata=metadata,
    )


def run_stamp_baseline(
    data: Mapping[str, object],
    *,
    p: int,
    n_neighbors: int = 8,
    layer: str = "counts",
    seed: int = 0,
    max_epochs: int = 800,
    min_epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    device: str = _TORCH_DEVICE,
    pseudocount: float = 0.3,
    temperature = 0.15
):
    """
    Run STAMP (scTM) on raw counts with spatial adjacency.
    """
    try:
        import anndata as ad
        import pandas as pd
        import sctm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "STAMP baseline requires the `scTM` package (pip install sctm)."
        ) from exc

    import numpy as _np_compat  # patch np.Inf removed in NumPy 2.0
    if not hasattr(_np_compat, "Inf"):
        _np_compat.Inf = _np_compat.inf

    counts_list = data.get("Y_count_list", data["Y_list"])
    input_view = "counts" if "Y_count_list" in data else _infer_gaussian_view(data)
    coords_list = data["coords_list"]
    # STAMP/Pyro uses float32 throughout; cast all inputs upfront to avoid dtype mismatches.
    ages = np.asarray(data.get("ages", np.arange(len(counts_list))), dtype=np.float32)
    Nlist = data["Nlist"]

    X = stack_Y_list(counts_list).astype(np.float32)
    coords = np.vstack(coords_list).astype(np.float32)
    obs = pd.DataFrame(index=np.arange(X.shape[0]))
    obs["age"] = np.repeat(ages, Nlist)

    adata = ad.AnnData(X=X, obs=obs)
    adata.layers[layer] = X.copy()
    adata.obsm["spatial"] = coords
    adj = _block_diag_adjacency(coords_list, n_neighbors=n_neighbors)
    adata.obsp["spatial_connectivities"] = adj.astype(np.float32)

    if hasattr(sctm, "seed"):
        try:
            sctm.seed.seed_everything(seed)
        except Exception:
            pass

    model = sctm.stamp.STAMP(
        adata,
        n_topics=int(p),
        layer=layer,
        time_covariate_keys="age",
        gene_likelihood="nb",
        verbose=False,
    )
    start = time.perf_counter()
    model.train(
        max_epochs=int(max_epochs),
        min_epochs=int(min_epochs),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        device=str(device),
        early_stop=True,
        shuffle=True,
    )
    runtime = time.perf_counter() - start

    topic_prop = model.get_cell_by_topic()

    with torch.inference_mode():
        beta_raw = model.model.get_cholesky(return_softmax=False)
    W = beta_raw.mean(dim=2).cpu().numpy()
    W_scaled = W / temperature
    W_scaled = np.exp(W_scaled) / np.sum(np.exp(W_scaled), axis=1, keepdims=True)

    H = topic_prop.to_numpy()
    W = W_scaled

    Y_for_metrics = stack_Y_list(counts_list)
    metadata = {"runtime": runtime, "n_topics": int(p), "input_view": input_view}
    metadata["nonneg_min"] = float(np.min(Y_for_metrics)) if Y_for_metrics.size else 0.0
    return pack_method_result(
        name="STAMP",
        W=W,
        H=H,
        Y=Y_for_metrics,
        Nlist=Nlist,
        infer_temporal=False,
        metadata=metadata,
    )


def run_popari_baseline(
    data: Mapping[str, object],
    *,
    p: int,
    input_view: str = "log_norm",
    n_neighbors: int = 8,
    lambda_Sigma_x_inv: float = 1e-4,
    expression_floor: float = 1e-8,
    torch_device: str = "cpu",
    torch_dtype: str = "float32",
    init_iters: int = 5,
    train_iters: int = 200,
    seed: int = 0,
    initialization_method: str = "svd",
):
    """
    Run Popari on log-normalized counts (default) with per-slice adjacency.

    Uses ``PopariDataset`` + ``compute_spatial_neighbors`` + save/load via
    ``dataset_path`` — the exact pattern from the Popari documentation and the
    working real-data scripts.  The save/load roundtrip reliably reconstructs
    the awkward-array adjacency list.

    ``initialization_method`` defaults to ``"svd"`` because after the save/load
    roundtrip X is sparse: ``"kmeans"`` fails (``np.concatenate`` on sparse),
    ``"leiden"`` fails on small simulation data (too few cells for Louvain),
    but ``"svd"`` uses ``scipy.sparse.vstack`` + ``TruncatedSVD`` which handle
    sparse matrices correctly.
    """
    try:
        import anndata as ad
        import awkward as ak
        import torch
        from popari.components import PopariDataset
        from popari.io import save_anndata
        from popari.model import Popari
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Popari baseline requires the `popari` package."
        ) from exc

    input_view = str(input_view).lower()
    if input_view not in {"counts", "log_norm"}:
        raise ValueError("input_view must be one of {'counts', 'log_norm'}")

    expr_list = None
    actual_view = input_view
    if input_view == "log_norm":
        expr_list = data.get("Y_log_norm_list")
        if expr_list is None:
            expr_list = data.get("Y_count_list")
            if expr_list is not None:
                actual_view = "counts"
        if expr_list is None:
            expr_list = data.get("Y_list")
            actual_view = "gaussian"
    else:
        expr_list = data.get("Y_count_list")
        if expr_list is None:
            expr_list = data.get("Y_log_norm_list")
            if expr_list is not None:
                actual_view = "log_norm"
        if expr_list is None:
            expr_list = data.get("Y_list")
            actual_view = "gaussian"
    coords_list = data["coords_list"]
    Nlist = data["Nlist"]
    replicate_names = [str(i) for i in range(len(expr_list))]

    floor = float(expression_floor)
    popari_datasets: list = []
    for Y, coords, name in zip(expr_list, coords_list, replicate_names):
        Y = np.asarray(Y, dtype=float)
        if np.any(Y < -1e-12):
            raise ValueError("Popari input must be nonneg.")
        from scipy.sparse import issparse
        if issparse(Y):
            X_dense = np.asarray(Y.todense(), dtype=np.float32)
        else:
            X_dense = np.ascontiguousarray(Y, dtype=np.float32)
        if floor > 0:
            X_dense = np.maximum(X_dense, floor)

        adata = ad.AnnData(X=X_dense)
        adata.obsm["spatial"] = np.asarray(coords, dtype=float)
        pop_ds = PopariDataset(adata, name)
        pop_ds.compute_spatial_neighbors()
        popari_datasets.append(pop_ds)

    dataset_h5ad = os.path.join(
        tempfile.mkdtemp(prefix="popari_bench_"), "popari_data.h5ad",
    )
    save_anndata(dataset_h5ad, popari_datasets)

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    torch_context = {"device": torch_device, "dtype": dtype_map.get(torch_dtype, torch.float32)}

    import traceback as _tb
    try:
        model = Popari(
            K=int(p),
            dataset_path=dataset_h5ad,
            lambda_Sigma_x_inv=float(lambda_Sigma_x_inv),
            torch_context=torch_context,
            initial_context=torch_context,
            random_state=int(seed),
            verbose=0,
            initialization_method=str(initialization_method),
        )

        start = time.perf_counter()
        for _ in range(int(init_iters)):
            model.estimate_parameters(update_spatial_affinities=False)
            model.estimate_weights(use_neighbors=False)
        for _ in range(int(train_iters)):
            model.estimate_parameters()
            model.estimate_weights()
        runtime = time.perf_counter() - start
    except Exception:
        print("[Popari traceback]")
        _tb.print_exc()
        raise
    finally:
        try:
            os.remove(dataset_h5ad)
            os.rmdir(os.path.dirname(dataset_h5ad))
        except OSError:
            pass

    H_blocks = [
        model.embedding_optimizer.embedding_state[ds.name].detach().cpu().numpy()
        for ds in model.datasets
    ]
    H = np.vstack(H_blocks)
    M_list = [
        model.parameter_optimizer.metagene_state[ds.name].detach().cpu().numpy()
        for ds in model.datasets
    ]
    M = np.mean(M_list, axis=0)
    W = M.T

    Y_for_metrics = stack_Y_list(expr_list)
    min_value = float(np.min(Y_for_metrics)) if Y_for_metrics.size else 0.0
    neg_frac = float(np.mean(Y_for_metrics < -1e-12)) if Y_for_metrics.size else 0.0
    metadata = {
        "runtime": runtime,
        "init_iters": int(init_iters),
        "train_iters": int(train_iters),
        "lambda_Sigma_x_inv": float(lambda_Sigma_x_inv),
        "expression_floor": float(expression_floor),
        "input_view": actual_view,
        "nonneg_min": min_value,
        "nonneg_frac_neg": neg_frac,
        "initialization_method": str(initialization_method),
    }
    return pack_method_result(
        name="Popari",
        W=W,
        H=H,
        Y=Y_for_metrics,
        Nlist=Nlist,
        infer_temporal=False,
        metadata=metadata,
    )


def run_mefisto_baseline(
    data: Mapping[str, object],
    *,
    p: int,
    seed: int = 1,
    sparse_gp: bool = True,
    n_inducing: int = 1000,
    train_iters: int = 1000,
    start_opt: int = 10,
    opt_freq: int = 10,
    covariate_mode: str = "age_xy",
    spatial_scale: str = "global",
) -> MethodResult:
    try:
        from mofapy2.run.entry_point import entry_point
        import mofax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MEFISTO baseline requires `mofapy2` and `mofax` to be installed."
        ) from exc

    T = len(data["Nlist"])
    data_mat = [[None for _ in range(T)]]
    for t in range(T):
        data_mat[0][t] = np.asarray(data["Y_list"][t], dtype=float)

    ent = entry_point()
    ent.set_data_options(use_float32=True, center_groups=False)
    ent.set_data_matrix(data_mat, likelihoods=["gaussian"])
    ent.set_model_options(
        factors=int(p),
        spikeslab_weights=True,
        ard_weights=True,
    )
    ent.set_train_options(
        iter=int(train_iters),
        convergence_mode="fast",
        seed=int(seed),
        verbose=False,
        quiet=True,
    )

    covariates, covariate_names = _scale_mefisto_covariates(
        data,
        covariate_mode=covariate_mode,
        spatial_scale=spatial_scale,
    )
    ent.set_covariates(covariates, covariates_names=list(covariate_names))
    frac_inducing = float(min(max(n_inducing / max(1, sum(data["Nlist"])), 1e-4), 0.8))
    ent.set_smooth_options(
        scale_cov=False,
        sparseGP=sparse_gp,
        frac_inducing=frac_inducing,
        model_groups=False,
        start_opt=int(start_opt),
        opt_freq=int(opt_freq),
    )
    ent.build()

    start = time.perf_counter()
    ent.run()
    runtime = time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="mefisto_") as tmpdir:
        h5_path = os.path.join(tmpdir, "mefisto.hdf5")
        ent.save(h5_path)
        model = mofax.mofa_model(h5_path)
        factors_df = model.get_factors(df=True)
        weights_df = model.get_weights(df=True)
        model.close()

    # Reorder samples to match stacked Y order: group-major, then cell order.
    # MOFA names samples like "sample1_group0" or "sample1_group1" depending
    # on internal indexing; we parse and sort by (group, sample) and then
    # align to the provided Nlist ordering.
    import re

    def parse_sample(name):
        # Handle MultiIndex entries (tuples) and string-based sample names.
        if isinstance(name, tuple):
            g_idx = None
            s_idx = None
            numeric = []
            for item in name:
                item_str = str(item)
                m = re.search(r"group(\d+)", item_str)
                if m is not None:
                    g_idx = int(m.group(1))
                m = re.search(r"sample(\d+)", item_str)
                if m is not None:
                    s_idx = int(m.group(1))
                if item_str.isdigit():
                    numeric.append(int(item_str))
            if g_idx is None and numeric:
                g_idx = numeric[0]
            if s_idx is None and len(numeric) > 1:
                s_idx = numeric[1]
            if g_idx is None or s_idx is None:
                return None
            return g_idx, s_idx, name

        name_str = str(name)
        patterns = [
            (r"sample(\d+)_group(\d+)", (1, 2)),
            (r"group(\d+)_sample(\d+)", (1, 2)),
        ]
        for pattern, order in patterns:
            m = re.search(pattern, name_str)
            if m is not None:
                s_idx = int(m.group(order[0]))
                g_idx = int(m.group(order[1]))
                return g_idx, s_idx, name
        return None

    parsed = [parse_sample(n) for n in factors_df.index]
    parsed_ok = [p for p in parsed if p is not None]
    if len(parsed_ok) == len(parsed):
        min_group = min(p[0] for p in parsed_ok)
        sorted_names = [
            name
            for _, _, name in sorted(
                ((g - min_group, s, name) for g, s, name in parsed_ok),
                key=lambda x: (x[0], x[1]),
            )
        ]
        order_mode = "parsed"
    else:
        # Fallback: keep MOFA's native ordering when sample names are opaque.
        print("[warn] MEFISTO sample names not parsed; falling back to native order.")
        sorted_names = list(factors_df.index)
        order_mode = "native"

    H_all = factors_df.loc[sorted_names].to_numpy()

    # Now truncate/partition H_all according to Nlist (defensive in case of
    # slight mismatches).
    expected_N = sum(int(n) for n in data["Nlist"])
    if H_all.shape[0] < expected_N:
        raise RuntimeError(
            f"MEFISTO returned {H_all.shape[0]} samples but expected {expected_N}."
        )
    H = H_all[:expected_N, :]
    W = weights_df.to_numpy().T

    input_view = _infer_gaussian_view(data)
    centered_stats = _maybe_centered_stats(data, name="MEFISTO")
    metadata = {
        "runtime": runtime,
        "smooth_options": {
            "scale_cov": False,
            "sparseGP": bool(sparse_gp),
            "frac_inducing": frac_inducing,
            "model_groups": False,
            "start_opt": int(start_opt),
            "opt_freq": int(opt_freq),
        },
        "input_view": input_view,
        "centered_gene_stats": centered_stats,
        "sample_order": order_mode,
        "covariates": list(covariate_names),
        "covariate_mode": covariate_mode,
        "spatial_scale": spatial_scale,
        "center_groups": False,
        "train_iters": int(train_iters),
        "n_inducing": int(n_inducing),
    }
    return pack_method_result(
        name="MEFISTO",
        W=W,
        H=H,
        Y=stack_Y_list(data["Y_list"]),
        Nlist=data["Nlist"],
        infer_temporal=False,
        metadata=metadata,
    )
