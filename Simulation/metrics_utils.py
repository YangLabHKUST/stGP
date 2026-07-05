from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stgp.estimation import align_programs_and_mse, project_simplex
from benchmark_utils import alpha_b_from_cell_embeddings
from benchmark_utils import MethodResult

# Methods whose W can contain negative loadings (SVD / factor-model eigenvectors).
# All other methods (stGP, NMF, Popari, STAMP, …) are non-negative by construction.
_SIGNED_METHODS = frozenset({"PCA", "SpatialPCA", "MEFISTO"})


@dataclass
class AlignmentResult:
    perm: np.ndarray
    W_hat: np.ndarray
    cost: float

def project_to_simplex_rows(W: ArrayLike) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    W_proj = np.zeros_like(W)
    for j in range(W.shape[0]):
        W_proj[j] = project_simplex(np.abs(W[j]))
    return W_proj

def sparsify_and_project_abs(
    W: ArrayLike,
    *,
    topk: Optional[int] = None,
    frac: Optional[float] = 0.1,
    project: bool = True,
) -> np.ndarray:
    W = np.abs(np.asarray(W, dtype=float))
    G = W.shape[1]
    k = None
    if frac is not None:
        k = max(int(np.ceil(float(frac) * G)), 1)
    if topk is not None:
        k = topk
    if (k is not None) and (k < G):
        for j in range(W.shape[0]):
            row = W[j]
            keep_idx = np.argpartition(row, -k)[-k:]
            mask = np.ones_like(row, dtype=bool)
            mask[keep_idx] = False
            row[mask] = 0.0
            W[j] = row
    if project:
        W = project_to_simplex_rows(W)
    return W

def align_programs(
    W_true: ArrayLike,
    W_est: ArrayLike,
    *,
    sparsify_topk: Optional[int] = None,
    sparsify_frac: Optional[float] = 0.1,
    project: bool = True,
) -> AlignmentResult:
    """
    Align estimated programs to truth.

    Steps:
      1) Take absolute loadings to ignore sign flips (PCA/SpatialPCA).
      2) Optionally sparsify each program to its top-|weights| (dense methods
         like PCA/SpatialPCA) via ``sparsify_topk`` or ``sparsify_frac``.
      3) Optionally project rows to the simplex for scale comparability.
      4) Run Hungarian matching via ``align_programs_and_mse``.
    """
    W_true_proc = sparsify_and_project_abs(
        W_true, topk=sparsify_topk, frac=sparsify_frac, project=project
    )
    W_est_proc = sparsify_and_project_abs(
        W_est, topk=sparsify_topk, frac=sparsify_frac, project=project
    )
    out = align_programs_and_mse(W_true_proc, W_est_proc)
    return AlignmentResult(
        perm=out["perm"],
        W_hat=out["B_aligned"],
        cost=out["overall_mse"],
    )

def align_method_for_plot(
    true_W: ArrayLike,
    method: "MethodResult",
    *,
    Nlist=None,
):
    """
    Hungarian-match estimated programs to true programs for visualization.

    Returns (W_aligned, alpha, b, perm, W_raw_permuted).

    ``W_aligned`` is the simplex-projected |W| used for metric computation.
    ``W_raw_permuted`` is the original (possibly signed) W reordered by perm,
    useful for displaying raw loadings (e.g. PCA eigenvectors) in heatmaps.
    """
    align = align_programs(true_W, method.W, sparsify_topk=None, sparsify_frac=0.1, project=True)
    perm = align.perm
    W_aligned = align.W_hat / (np.sum(align.W_hat, axis=1, keepdims=True) + 1e-12)

    W_raw_permuted = np.asarray(method.W, dtype=float)[perm]

    if method.alpha is not None:
        alpha = np.asarray(method.alpha)[perm]
    elif Nlist is not None:
        alpha_raw, _ = alpha_b_from_cell_embeddings(np.asarray(method.H, dtype=float), Nlist)
        alpha = alpha_raw[perm]
    else:
        alpha = None
    b = None if method.b is None else np.asarray(method.b)[:, perm]
    return W_aligned, alpha, b, perm, W_raw_permuted

def mse(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
def rmse(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.sqrt(mse(a, b)))

def corr_mean(a: ArrayLike, b: ArrayLike) -> float:
    """
    Average correlation across rows (programs). Returns NaN if no finite value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vals = []
    for j in range(a.shape[0]):
        x = a[j]
        y = b[j]
        sx = x.std()
        sy = y.std()
        if sx < 1e-12 or sy < 1e-12:
            vals.append(np.nan)
        else:
            vals.append(np.corrcoef(x, y)[0, 1])
    vals = np.asarray(vals)
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def spearman_corr_mean(a: ArrayLike, b: ArrayLike) -> float:
    """
    Average Spearman rank correlation across rows (programs). Returns NaN if no finite value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vals = []
    for j in range(a.shape[0]):
        x = a[j]
        y = b[j]
        sx = x.std()
        sy = y.std()
        if sx < 1e-12 or sy < 1e-12:
            vals.append(np.nan)
        else:
            r, _ = spearmanr(x, y)
            vals.append(float(r) if np.isfinite(r) else np.nan)
    vals = np.asarray(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def subspace_similarity(W_true: ArrayLike, W_est: ArrayLike) -> float:
    """
    1 - normalized Frobenius norm of principal angles between the column spaces of ``W_true.T`` and ``W_est.T``.
    """
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    k = min(W_true.shape[0], W_est.shape[0], W_true.shape[1], W_est.shape[1])
    angles = subspace_angles(W_true.T, W_est.T)[:k]
    return float(1.0 - np.linalg.norm(angles) / np.sqrt(k))


def tv_score_per_program(
    W_true: ArrayLike,
    W_est: ArrayLike,
) -> np.ndarray:
    """
    Per-program Total Variation score = 1 - TV_distance, in [0, 1].

    TV_distance(p, q) = 0.5 * ||p - q||_1.
    Score 1 means perfect recovery; 0 means no shared probability mass.
    Assumes rows are probability distributions (non-negative, sum to 1).
    """
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    assert W_true.shape == W_est.shape
    return 1.0 - 0.5 * np.sum(np.abs(W_true - W_est), axis=1)

def hellinger_score_per_program(
    W_true: ArrayLike,
    W_est: ArrayLike,
) -> np.ndarray:
    """
    Per-program Hellinger score = 1 - d_H, in [0, 1].

    d_H²(p, q) = 0.5 * Σ_g (√p_g - √q_g)²,  d_H = √(d_H²).
    More robust than TV for sparse gene programs; unlike KL, never blows
    up when a gene weight is zero.
    """
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    assert W_true.shape == W_est.shape
    diff_sqrt = np.sqrt(np.clip(W_true, 0.0, None)) - np.sqrt(np.clip(W_est, 0.0, None))
    d_H2 = 0.5 * np.sum(diff_sqrt ** 2, axis=1)
    return 1.0 - np.sqrt(np.clip(d_H2, 0.0, 1.0))


def tau_mass_set(row: np.ndarray, tau: float) -> set:
    """
    Return the minimal index set A s.t. Σ_{g in A} row[g] >= tau,
    selecting genes in descending weight order.
    """
    order = np.argsort(row)[::-1]
    cumsum = 0.0
    result = []
    for idx in order:
        result.append(int(idx))
        cumsum += float(row[idx])
        if cumsum >= tau:
            break
    return set(result)

def gene_matching_metrics(
    W_true: ArrayLike,
    W_est: ArrayLike,
    tau: float = 0.9,
) -> Dict[str, float]:
    """
    Gene-support matching metrics based on tau-mass sets.

    For each program k, A_k(tau) is the minimal top-gene set covering
    fraction tau of total weight. F1 and Jaccard are computed between
    the true and estimated A_k sets and averaged across programs.
    Key names carry the tau percentage as a suffix (e.g. 'f1_tau90').
    """
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    assert W_true.shape == W_est.shape
    K = W_true.shape[0]
    eps = 1e-12
    f1s, jacs = [], []
    for k in range(K):
        true_set = tau_mass_set(W_true[k], tau)
        est_set = tau_mass_set(W_est[k], tau)
        tp = len(true_set & est_set)
        prec = tp / (len(est_set) + eps)
        rec = tp / (len(true_set) + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
        union = len(true_set | est_set)
        jacs.append(tp / union if union > 0 else np.nan)
    suffix = f"_tau{int(tau * 100)}"
    return {
        f"f1{suffix}": float(np.mean(f1s)),
        f"jaccard{suffix}": float(np.nanmean(jacs)),
    }


def gene_topk_hit_rate_mean(
    W_true: ArrayLike,
    W_est: ArrayLike,
    *,
    eps: float = 1e-10,
) -> float:
    """
    For each program j, let n_j be the number of genes with W_true[j, g] > eps
    (true support size).  Take the n_j genes with largest |W_est[j, g]| as the
    predicted set.  Per-program hit rate is |true ∩ pred| / n_j (recall of the
    true support in the top-n_j predictions).  Return the mean across programs;
    programs with n_j = 0 contribute NaN and are ignored by nanmean.
    """
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    assert W_true.shape == W_est.shape
    p, G = W_true.shape
    scores = np.abs(W_est)
    rates: list[float] = []
    for j in range(p):
        true_idx = np.where(W_true[j] > eps)[0]
        n_j = int(true_idx.size)
        if n_j == 0:
            rates.append(float("nan"))
            continue
        if n_j >= G:
            pred_idx = np.arange(G, dtype=int)
        else:
            pred_idx = np.argpartition(scores[j], -n_j)[-n_j:]
        pred_set = set(int(x) for x in pred_idx)
        true_set = set(int(x) for x in true_idx)
        hits = len(true_set & pred_set)
        rates.append(hits / n_j)
    arr = np.asarray(rates, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def variance_component_metrics(
    theta_true: ArrayLike,
    theta_hat: ArrayLike,
    sigma2e_true: Optional[float] = None,
    sigma2e_hat: Optional[float] = None,
) -> Dict[str, float]:
    """
    RMSE + log(RMSE) for variance components. ``theta`` is assumed to be
    shape (p, 2) corresponding to (sigma2_age, tau2_spa).
    """
    theta_true = np.asarray(theta_true, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    rmse_sigma = rmse(theta_true[:, 0], theta_hat[:, 0])
    rmse_tau = rmse(theta_true[:, 1], theta_hat[:, 1])
    # Use np.log1p-based guarded log to avoid -inf when rmse_tau = 0 (no-spatial case).
    log_rmse_sigma = float(np.log(rmse_sigma)) if rmse_sigma > 0 else float("-inf")
    log_rmse_tau = float(np.log(rmse_tau)) if rmse_tau > 0 else float("nan")
    out = {
        "sigma2_age_rmse": rmse_sigma,
        "sigma2_age_log_rmse": log_rmse_sigma,
        "tau2_spa_rmse": rmse_tau,
        "tau2_spa_log_rmse": log_rmse_tau,
    }
    if sigma2e_true is not None and sigma2e_hat is not None:
        abs_err = abs(float(sigma2e_hat) - float(sigma2e_true))
        out["sigma2_e_abs_err"] = abs_err
        out["sigma2_e_log_abs_err"] = np.log(abs_err)
    return out


def program_metrics(
    W_true: ArrayLike,
    W_est: ArrayLike,
    *,
    tau: float = 0.9,
) -> Dict[str, float]:
    """
    Combine program-level metrics: RMSE, log(RMSE), subspace similarity,
    distribution recovery (TV + Hellinger), and tau-mass support overlap.
    W rows must be probability distributions aligned by permutation beforehand.
    """
    rmse_val = rmse(W_true, W_est)
    metrics = {
        "w_rmse": rmse_val,
        "w_log_rmse": np.log(rmse_val),
        "subspace_similarity": subspace_similarity(W_true, W_est),
    }
    metrics["tv_score"] = float(np.mean(tv_score_per_program(W_true, W_est)))
    metrics["hellinger_score"] = float(np.mean(hellinger_score_per_program(W_true, W_est)))
    metrics.update(gene_matching_metrics(W_true, W_est, tau=tau))
    metrics["gene_topk_hit_mean"] = gene_topk_hit_rate_mean(W_true, W_est)
    return metrics


def summarize_method_performance(
    true_data: Dict[str, ArrayLike],
    method: "MethodResult",
    *,
    tau: float = 0.9,
    project_loadings: bool = True,
    align_sparsify_topk: Optional[int] = None,
    align_sparsify_frac: Optional[float] = 0.1,
    manual_perm: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Aggregate all relevant metrics for a fitted method.
    """
    W_true = true_data["W"]

    # Align programs (Hungarian) before any elementwise program comparisons.
    W_est = np.asarray(method.W, dtype=float)
    if project_loadings:
        W_est = project_to_simplex_rows(W_est)

    if manual_perm is not None:
        perm = np.asarray(manual_perm, dtype=int)
        if perm.shape[0] != W_true.shape[0]:
            raise ValueError("manual_perm length must match number of programs")
        align_cost = np.nan
    else:
        align = align_programs(
            W_true,
            W_est,
            sparsify_topk=align_sparsify_topk,
            sparsify_frac=align_sparsify_frac,
            project=True,
        )
        perm = align.perm
        align_cost = float(align.cost)
    W_est = W_est[perm]

    metrics = {}
    metrics.update(program_metrics(W_true, W_est, tau=tau))

    # H_corr / H_corr_spearman: per-program correlation of full cell embeddings (works for all methods).
    H_true = np.asarray(np.vstack(true_data["H_list"]), dtype=float)   # (N, p)
    H_hat_aligned = np.asarray(method.H, dtype=float)[:, perm]  # (N, p) column-permuted
    metrics["H_corr"] = corr_mean(H_true.T, H_hat_aligned.T)
    metrics["H_corr_spearman"] = spearman_corr_mean(H_true.T, H_hat_aligned.T)

    if method.theta is not None and "sigma2_age" in true_data and "tau2_spa" in true_data:
        theta_true = np.column_stack(
            (np.asarray(true_data["sigma2_age"]), np.asarray(true_data["tau2_spa"]))
        )
        theta_hat = np.asarray(method.theta, dtype=float)[perm]
        sigma2e_true = true_data.get("sigma2_e")
        sigma2e_hat = None
        if method.metadata is not None:
            sigma2e_hat = method.metadata.get("sigma2_e")
        metrics.update(
            variance_component_metrics(
                theta_true=theta_true,
                theta_hat=theta_hat,
                sigma2e_true=sigma2e_true,
                sigma2e_hat=sigma2e_hat,
            )
        )
    metrics["align_cost"] = align_cost
    return metrics




