from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Literal, Mapping, MutableMapping, Sequence

import anndata as ad
from IPython.display import display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
import pandas as pd


DPI = 400
NM_W_SINGLE = 88 / 25.4
NM_W_HALF = 120 / 25.4
NM_W_FULL = 180 / 25.4


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": DPI,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
}


METHOD_COLORS = {
    "stGP": "#E64B35",
    "PCA": "#91D1C2",
    "NMF": "#F39B7F",
    "SpatialPCA": "#4DBBD5",
    "MEFISTO": "#8491B4",
    "STAMP": "#B09C85",
    "Popari": "#00A087",
}


CLUSTER_LABEL_COLORS = {
    1: "#1f78b4",
    2: "#e31a1c",
    3: "#9edae5",
    4: "#f4a6c8",
    5: "#33a02c",
    6: "#ff7f00",
    7: "#6a3d9a",
    8: "#b15928",
}


def cluster_color(label, *, default: str = "#7f7f7f") -> str:
    try:
        key = int(label)
    except (TypeError, ValueError):
        key = str(label)
    return CLUSTER_LABEL_COLORS.get(key, default)


@dataclass(frozen=True)
class VarPartColors:
    age: str = "#E64B35"
    region: str = "#4DBBD5"
    both: str = "#3C5488"
    residuals: str = "#BFBFBF"


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
        for ax in fig.axes:
            artists = list(ax.images)
            if include_collections:
                artists.extend(ax.collections)
            for artist in artists:
                artist.set_rasterized(False)
    fig.savefig(pdf, **kwargs)
    display(fig)
    plt.close(fig)
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


def spatial_program_values(
    adata,
    program,
    *,
    obsm_key: str = "X_stgp_spatial",
    allow_transpose: bool = False,
) -> np.ndarray:
    idx = int(str(program).replace("stGP", "")) - 1
    arr = np.asarray(adata.obsm[obsm_key])
    if allow_transpose and arr.shape[0] != adata.n_obs:
        arr = arr.T
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

# ---------------------------------------------------------------------------
# Simulation figure helpers
# ---------------------------------------------------------------------------

@dataclass
class SimulationMethodResult:
    name: str
    W: np.ndarray
    H: np.ndarray
    alpha: np.ndarray | None
    b: np.ndarray | None
    Y_hat: np.ndarray | None
    theta: np.ndarray | None = None
    alpha_std: np.ndarray | None = None
    alpha_lower: np.ndarray | None = None
    alpha_upper: np.ndarray | None = None
    metadata: MutableMapping[str, object] = field(default_factory=dict)


def _simulation_method_from_dict(payload: Mapping[str, object]) -> SimulationMethodResult:
    return SimulationMethodResult(
        name=str(payload.get("name", "")),
        W=np.asarray(payload.get("W"), dtype=float),
        H=np.asarray(payload.get("H"), dtype=float),
        alpha=None if payload.get("alpha") is None else np.asarray(payload.get("alpha"), dtype=float),
        alpha_std=None if payload.get("alpha_std") is None else np.asarray(payload.get("alpha_std"), dtype=float),
        alpha_lower=None if payload.get("alpha_lower") is None else np.asarray(payload.get("alpha_lower"), dtype=float),
        alpha_upper=None if payload.get("alpha_upper") is None else np.asarray(payload.get("alpha_upper"), dtype=float),
        b=None if payload.get("b") is None else np.asarray(payload.get("b"), dtype=float),
        Y_hat=None if payload.get("Y_hat") is None else np.asarray(payload.get("Y_hat"), dtype=float),
        theta=None if payload.get("theta") is None else np.asarray(payload.get("theta"), dtype=float),
        metadata=payload.get("metadata", {}) or {},
    )


class _SimulationResultUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "benchmark_utils" and name == "MethodResult":
            return SimulationMethodResult
        return super().find_class(module, name)


def _simulation_method_to_dict(res) -> dict[str, object]:
    return {
        "name": getattr(res, "name", ""),
        "W": getattr(res, "W", None),
        "H": getattr(res, "H", None),
        "alpha": getattr(res, "alpha", None),
        "alpha_std": getattr(res, "alpha_std", None),
        "alpha_lower": getattr(res, "alpha_lower", None),
        "alpha_upper": getattr(res, "alpha_upper", None),
        "b": getattr(res, "b", None),
        "Y_hat": getattr(res, "Y_hat", None),
        "theta": getattr(res, "theta", None),
        "metadata": dict(getattr(res, "metadata", {}) or {}),
    }


def load_method_result(path: str | Path) -> tuple[SimulationMethodResult, dict[str, object]]:
    """Load a serialized simulation method result for plotting notebooks."""
    with open(path, "rb") as f:
        payload = _SimulationResultUnpickler(f).load()
    if isinstance(payload, dict) and "method" in payload:
        method_payload = payload["method"]
        if isinstance(method_payload, SimulationMethodResult):
            return method_payload, payload
        return _simulation_method_from_dict(method_payload), payload
    if isinstance(payload, SimulationMethodResult):
        return payload, {"method": _simulation_method_to_dict(payload)}
    if isinstance(payload, dict):
        return _simulation_method_from_dict(payload), {"method": payload}
    raise ValueError(f"Unrecognized payload type in {path}: {type(payload)}")


def true_quantities_from_datagen(data: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Extract true quantities from simulation DataGen payloads for figure panels."""
    out = {
        "Y": np.ascontiguousarray(np.vstack(data["Y_list"]), dtype=float),
        "signal": np.ascontiguousarray(np.vstack(data["Signal_list"]), dtype=float),
        "W": np.asarray(data["W"]),
        "H": np.vstack(data["H_list"]),
        "alpha": np.asarray(data["Alpha"]),
        "b": np.asarray([np.concatenate(program_blocks) for program_blocks in data["B_list"]]),
        "sigma2_age": np.asarray(data["sigma2_age"]),
        "tau2_spa": np.asarray(data["tau2_spa"]),
        "sigma2_e": float(data["sigma2_e"]),
        "spatial": bool(data.get("spatial", data.get("K_spa") is not None)),
    }
    if "Y_count_list" in data:
        out["Y_count"] = np.ascontiguousarray(np.vstack(data["Y_count_list"]), dtype=float)
    if "Mu_count_list" in data:
        out["signal_count"] = np.ascontiguousarray(np.vstack(data["Mu_count_list"]), dtype=float)
    return out


def alpha_b_from_cell_embeddings(
    H: np.ndarray,
    Nlist: Sequence[int],
    center_blocks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=float)
    cuts = np.cumsum(np.concatenate(([0], np.asarray(Nlist, dtype=int))))
    alpha = np.zeros((H.shape[1], len(Nlist)), dtype=float)
    for t in range(len(Nlist)):
        alpha[:, t] = H[cuts[t] : cuts[t + 1]].mean(axis=0)
    alpha_cells = np.repeat(alpha.T, repeats=Nlist, axis=0)
    b = H - alpha_cells
    if center_blocks:
        for t in range(len(Nlist)):
            block = slice(int(cuts[t]), int(cuts[t + 1]))
            offset = b[block].mean(axis=0)
            b[block] = b[block] - offset
            alpha[:, t] += offset
    return alpha, b


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of one vector onto the probability simplex."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1))[0]
    if rho_idx.size == 0:
        return np.full_like(v, 1.0 / v.size)
    rho = int(rho_idx[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def project_to_simplex_rows(W) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    return np.vstack([_project_simplex(np.abs(row)) for row in W])


def sparsify_and_project_abs(
    W,
    *,
    topk: int | None = None,
    frac: float | None = 0.1,
    project: bool = True,
) -> np.ndarray:
    W = np.abs(np.asarray(W, dtype=float)).copy()
    n_genes = W.shape[1]
    k = max(int(np.ceil(float(frac) * n_genes)), 1) if frac is not None else None
    if topk is not None:
        k = int(topk)
    if k is not None and k < n_genes:
        for j in range(W.shape[0]):
            keep = np.argpartition(W[j], -k)[-k:]
            mask = np.ones_like(W[j], dtype=bool)
            mask[keep] = False
            W[j, mask] = 0.0
    return project_to_simplex_rows(W) if project else W


@dataclass
class AlignmentResult:
    perm: np.ndarray
    W_hat: np.ndarray
    cost: float


def align_programs(
    W_true,
    W_est,
    *,
    sparsify_topk: int | None = None,
    sparsify_frac: float | None = 0.1,
    project: bool = True,
) -> AlignmentResult:
    from scipy.optimize import linear_sum_assignment

    W_true_proc = sparsify_and_project_abs(W_true, topk=sparsify_topk, frac=sparsify_frac, project=project)
    W_est_proc = sparsify_and_project_abs(W_est, topk=sparsify_topk, frac=sparsify_frac, project=project)
    cost = ((W_true_proc[:, None, :] - W_est_proc[None, :, :]) ** 2).mean(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    perm = col_ind[order]
    return AlignmentResult(perm=perm, W_hat=W_est_proc[perm], cost=float(cost[row_ind, col_ind].mean()))


def align_method_for_plot(true_W, method: SimulationMethodResult, *, Nlist=None):
    """Hungarian-match estimated programs to true programs for visualization."""
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


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def rmse(a, b) -> float:
    return float(np.sqrt(mse(a, b)))


def corr_mean(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vals = [np.corrcoef(a[j], b[j])[0, 1] for j in range(a.shape[0])]
    return float(np.nanmean(vals))


def spearman_corr_mean(a, b) -> float:
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vals = [spearmanr(a[j], b[j]).statistic for j in range(a.shape[0])]
    return float(np.nanmean(vals))


def subspace_similarity(W_true, W_est) -> float:
    from scipy.linalg import subspace_angles

    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    k = min(W_true.shape[0], W_est.shape[0], W_true.shape[1], W_est.shape[1])
    angles = subspace_angles(W_true.T, W_est.T)[:k]
    return float(1.0 - np.linalg.norm(angles) / np.sqrt(k))


def tv_score_per_program(W_true, W_est) -> np.ndarray:
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    return 1.0 - 0.5 * np.sum(np.abs(W_true - W_est), axis=1)


def hellinger_score_per_program(W_true, W_est) -> np.ndarray:
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    diff_sqrt = np.sqrt(np.clip(W_true, 0.0, None)) - np.sqrt(np.clip(W_est, 0.0, None))
    d_h2 = 0.5 * np.sum(diff_sqrt**2, axis=1)
    return 1.0 - np.sqrt(np.clip(d_h2, 0.0, 1.0))


def tau_mass_set(row: np.ndarray, tau: float) -> set[int]:
    order = np.argsort(row)[::-1]
    cumsum = 0.0
    result = []
    for idx in order:
        result.append(int(idx))
        cumsum += float(row[idx])
        if cumsum >= tau:
            break
    return set(result)


def gene_matching_metrics(W_true, W_est, tau: float = 0.9) -> dict[str, float]:
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    f1s, jacs = [], []
    for k in range(W_true.shape[0]):
        true_set = tau_mass_set(W_true[k], tau)
        est_set = tau_mass_set(W_est[k], tau)
        tp = len(true_set & est_set)
        prec = tp / (len(est_set) + 1e-12)
        rec = tp / (len(true_set) + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        union = len(true_set | est_set)
        jacs.append(tp / union if union > 0 else np.nan)
    suffix = f"_tau{int(tau * 100)}"
    return {f"f1{suffix}": float(np.mean(f1s)), f"jaccard{suffix}": float(np.nanmean(jacs))}


def gene_topk_hit_rate_mean(W_true, W_est, *, eps: float = 1e-10) -> float:
    W_true = np.asarray(W_true, dtype=float)
    W_est = np.asarray(W_est, dtype=float)
    _, n_genes = W_true.shape
    scores = np.abs(W_est)
    rates = []
    for j in range(W_true.shape[0]):
        true_idx = np.where(W_true[j] > eps)[0]
        n_j = int(true_idx.size)
        if n_j == 0:
            rates.append(float("nan"))
            continue
        pred_idx = np.arange(n_genes, dtype=int) if n_j >= n_genes else np.argpartition(scores[j], -n_j)[-n_j:]
        rates.append(len(set(pred_idx) & set(true_idx)) / n_j)
    arr = np.asarray(rates, dtype=float)
    return float("nan") if np.all(np.isnan(arr)) else float(np.nanmean(arr))


def variance_component_metrics(
    theta_true,
    theta_hat,
    sigma2e_true: float | None = None,
    sigma2e_hat: float | None = None,
) -> dict[str, float]:
    theta_true = np.asarray(theta_true, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    rmse_sigma = rmse(theta_true[:, 0], theta_hat[:, 0])
    rmse_tau = rmse(theta_true[:, 1], theta_hat[:, 1])
    out = {
        "sigma2_age_rmse": rmse_sigma,
        "sigma2_age_log_rmse": float(np.log(rmse_sigma)) if rmse_sigma > 0 else float("-inf"),
        "tau2_spa_rmse": rmse_tau,
        "tau2_spa_log_rmse": float(np.log(rmse_tau)) if rmse_tau > 0 else float("nan"),
    }
    if sigma2e_true is not None and sigma2e_hat is not None:
        abs_err = abs(float(sigma2e_hat) - float(sigma2e_true))
        out["sigma2_e_abs_err"] = abs_err
        out["sigma2_e_log_abs_err"] = np.log(abs_err)
    return out


def program_metrics(W_true, W_est, *, tau: float = 0.9) -> dict[str, float]:
    rmse_val = rmse(W_true, W_est)
    metrics = {
        "w_rmse": rmse_val,
        "w_log_rmse": np.log(rmse_val),
        "subspace_similarity": subspace_similarity(W_true, W_est),
        "tv_score": float(np.mean(tv_score_per_program(W_true, W_est))),
        "hellinger_score": float(np.mean(hellinger_score_per_program(W_true, W_est))),
        "gene_topk_hit_mean": gene_topk_hit_rate_mean(W_true, W_est),
    }
    metrics.update(gene_matching_metrics(W_true, W_est, tau=tau))
    return metrics


def summarize_method_performance(
    true_data: Mapping[str, object],
    method: SimulationMethodResult,
    *,
    tau: float = 0.9,
    project_loadings: bool = True,
    align_sparsify_topk: int | None = None,
    align_sparsify_frac: float | None = 0.1,
) -> dict[str, float]:
    W_true = true_data["W"]
    W_est = np.asarray(method.W, dtype=float)
    if project_loadings:
        W_est = project_to_simplex_rows(W_est)
    align = align_programs(
        W_true,
        W_est,
        sparsify_topk=align_sparsify_topk,
        sparsify_frac=align_sparsify_frac,
        project=True,
    )
    perm = align.perm
    W_est = W_est[perm]
    metrics = program_metrics(W_true, W_est, tau=tau)
    H_true = np.asarray(np.vstack(true_data["H_list"]), dtype=float)
    H_hat_aligned = np.asarray(method.H, dtype=float)[:, perm]
    metrics["H_corr"] = corr_mean(H_true.T, H_hat_aligned.T)
    metrics["H_corr_spearman"] = spearman_corr_mean(H_true.T, H_hat_aligned.T)
    if method.theta is not None and "sigma2_age" in true_data and "tau2_spa" in true_data:
        theta_true = np.column_stack((np.asarray(true_data["sigma2_age"]), np.asarray(true_data["tau2_spa"])))
        theta_hat = np.asarray(method.theta, dtype=float)[perm]
        sigma2e_hat = method.metadata.get("sigma2_e") if method.metadata is not None else None
        metrics.update(
            variance_component_metrics(
                theta_true=theta_true,
                theta_hat=theta_hat,
                sigma2e_true=true_data.get("sigma2_e"),
                sigma2e_hat=sigma2e_hat,
            )
        )
    metrics["align_cost"] = float(align.cost)
    return metrics


DEFAULT_RECOVERY_MAIN_METRICS = ("hellinger_score", "gene_topk_hit_mean", "H_corr")

# ---------------------------------------------------------------------------
# Human brain MERFISH helpers
# ---------------------------------------------------------------------------

METHODS = ["stGP", "STAMP", "MEFISTO", "Popari", "SpatialPCA"]
LAYER_SPECS = [("L2/3", "CUX2", 0), ("L4", "RORB", 2), ("L5/6", "HS3ST4", 1)]
LAYER_MARKERS = {layer: gene for layer, gene, _ in LAYER_SPECS}
REPRESENTATIVE_AGE_OCCURRENCES = [(28, 2), (42, 2), (82, 2), (87, 1)]
CELLTYPE2_LAYER_MAP = {"L2/3": "L2/3", "L4": "L4", "L5/6": "L5/6", "L5/6-CC": "L5/6"}
BASELINE_OBSM_KEYS = {
    "STAMP": "X_stamp",
    "MEFISTO": "X_mefisto",
    "Popari": "X",
    "SpatialPCA": "X_spatialpca",
}
LAYER_COLORS = {
    "L2/3": "#4DBBD5",
    "L4": "#F39B7F",
    "L5/6": "#00A087",
    "L5/6-CC": "#8491B4",
    "ext": "#BFBFBF",
    "#unassigned": "#D9D9D9",
}


def as_1d_array(x) -> np.ndarray:
    try:
        from scipy import sparse

        if sparse.issparse(x):
            x = x.toarray()
    except Exception:
        pass
    return np.asarray(x).reshape(-1)


def best_program_by_correlation(
    adata,
    obsm_key: str,
    expr,
    *,
    use_abs_for: tuple[str, ...] = ("X_mefisto", "X_spatialpca"),
) -> int:
    scores = np.asarray(adata.obsm[obsm_key])
    y = as_1d_array(expr)
    corrs = np.asarray([np.corrcoef(scores[:, k], y)[0, 1] for k in range(scores.shape[1])], dtype=float)
    if obsm_key in use_abs_for:
        return int(np.nanargmax(np.abs(corrs)))
    return int(np.nanargmax(corrs))


# ---------------------------------------------------------------------------
# Mouse brain MERFISH helpers
# ---------------------------------------------------------------------------

@dataclass
class MouseMethodResult:
    method: str
    celltype: str
    result_dir: Path
    adata: object
    scores: pd.DataFrame
    gene_weights: pd.DataFrame | None = None


def _scores_from_obsm(adata, obsm_key: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    X = np.asarray(adata.obsm[obsm_key])
    cols = [f"{prefix}{i + 1}" for i in range(int(X.shape[1]))]
    return pd.DataFrame(X, index=adata.obs_names.astype(str), columns=cols), cols


def load_stgp(result_dir: str | Path, *, celltype: str) -> MouseMethodResult:
    result_dir = Path(result_dir)
    adata = ad.read_h5ad(result_dir / "adata_with_scores.h5ad")
    scores, default_cols = _scores_from_obsm(adata, "X_stgp", "stGP")
    weight_path = result_dir / "W.csv"
    gene_weights = pd.read_csv(weight_path, index_col=0) if weight_path.exists() else None
    if gene_weights is not None and gene_weights.shape[0] == len(default_cols):
        scores.columns = [str(c) for c in gene_weights.index.tolist()]
    return MouseMethodResult("stGP", str(celltype), result_dir, adata, scores, gene_weights)


def load_spatialpca(result_dir: str | Path, *, celltype: str) -> MouseMethodResult:
    result_dir = Path(result_dir)
    adata = ad.read_h5ad(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_spatialpca", "SPCA")
    gene_weights = None
    loadings_path = result_dir / "W_loadings.csv"
    if loadings_path.exists():
        raw = pd.read_csv(loadings_path, index_col=0)
        gene_weights = raw.T.copy()
        gene_weights.index = [str(x) for x in gene_weights.index]
        gene_weights.columns = [str(x) for x in gene_weights.columns]
    return MouseMethodResult("SpatialPCA", str(celltype), result_dir, adata, scores, gene_weights)


def load_mefisto(result_dir: str | Path, *, celltype: str) -> MouseMethodResult:
    result_dir = Path(result_dir)
    adata = ad.read_h5ad(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_mefisto", "MEFISTO")
    weights_path = result_dir / "weights.csv"
    gene_weights = pd.read_csv(weights_path, index_col=0) if weights_path.exists() else None
    return MouseMethodResult("MEFISTO", str(celltype), result_dir, adata, scores, gene_weights)


def load_stamp(result_dir: str | Path, *, celltype: str) -> MouseMethodResult:
    result_dir = Path(result_dir)
    adata = ad.read_h5ad(result_dir / "adata_with_scores.h5ad")
    scores, _ = _scores_from_obsm(adata, "X_stamp", "STAMP")
    loadings_path = result_dir / "W_loadings.csv"
    gene_weights = pd.read_csv(loadings_path, index_col=0) if loadings_path.exists() else None
    return MouseMethodResult("STAMP", str(celltype), result_dir, adata, scores, gene_weights)


def _load_popari_h5py(h5ad_path: Path):
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(h5ad_path, "r") as f:
        obs_idx_raw = f["obs"]["_index"][:]
        obs_names = pd.Index([x.decode() if isinstance(x, bytes) else str(x) for x in obs_idx_raw])
        obs_dict = {}
        for col in f["obs"]:
            if col.startswith("_"):
                continue
            try:
                ds = f["obs"][col]
                if isinstance(ds, h5py.Dataset):
                    raw = ds[:]
                    if raw.dtype.kind in ("S", "O"):
                        raw = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
                    obs_dict[col] = raw
                elif isinstance(ds, h5py.Group) and "codes" in ds and "categories" in ds:
                    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in ds["categories"][:]]
                    obs_dict[col] = pd.Categorical.from_codes(ds["codes"][:], categories=cats)
            except Exception:
                pass
        var_idx_raw = f["var"]["_index"][:]
        var_names = pd.Index([x.decode() if isinstance(x, bytes) else str(x) for x in var_idx_raw])
        obsm = {"X": np.asarray(f["obsm"]["X"])} if "obsm" in f and "X" in f["obsm"] else {}
        uns = {}
        if "uns" in f and "M" in f["uns"]:
            M_item = f["uns"]["M"]
            uns["M"] = {k: np.asarray(M_item[k]) for k in M_item} if isinstance(M_item, h5py.Group) else np.asarray(M_item)
    adata = ad.AnnData(
        X=csr_matrix((len(obs_names), len(var_names)), dtype=np.float32),
        obs=pd.DataFrame(obs_dict, index=obs_names),
        var=pd.DataFrame(index=var_names),
    )
    for key, val in obsm.items():
        adata.obsm[key] = val
    adata.uns.update(uns)
    return adata


def _popari_gene_weights_from_uns_M(adata, k: int, cols: list[str]) -> pd.DataFrame | None:
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
                return pd.DataFrame(M.T, index=cols, columns=adata.var_names.astype(str))
            return None
        M = np.asarray(M_obj)
        if M.ndim == 2 and M.shape == (k, adata.n_vars):
            return pd.DataFrame(M, index=cols, columns=adata.var_names.astype(str))
        if M.ndim == 2 and M.shape == (adata.n_vars, k):
            return pd.DataFrame(M.T, index=cols, columns=adata.var_names.astype(str))
    except Exception:
        return None
    return None


def load_popari(result_dir: str | Path, *, celltype: str) -> MouseMethodResult:
    result_dir = Path(result_dir)
    h5ad_path = result_dir / "res_popari.h5ad"
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as exc:
        if "null" in str(exc).lower() or "IOSpec" in str(exc) or "No read method" in str(exc):
            adata = _load_popari_h5py(h5ad_path)
        else:
            raise
    scores, cols = _scores_from_obsm(adata, "X", "Popari")
    gene_weights = _popari_gene_weights_from_uns_M(adata, len(cols), cols)
    return MouseMethodResult("Popari", str(celltype), result_dir, adata, scores, gene_weights)


_MOUSE_LOADERS = {
    "stGP": load_stgp,
    "SpatialPCA": load_spatialpca,
    "MEFISTO": load_mefisto,
    "STAMP": load_stamp,
    "Popari": load_popari,
}


def load_method(
    method: Literal["stGP", "SpatialPCA", "MEFISTO", "STAMP", "Popari"],
    result_dir: str | Path,
    *,
    celltype: str,
) -> MouseMethodResult:
    if method not in _MOUSE_LOADERS:
        raise ValueError(f"Unsupported method: {method}")
    return _MOUSE_LOADERS[method](result_dir, celltype=celltype)


def _trunc_cmap(name: str, lo: float = 0.15, hi: float = 0.82, n: int = 256):
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{name}", base(np.linspace(lo, hi, n)))


PANEL_CMAPS = {
    "Hallmark": _trunc_cmap("Blues", 0.22, 0.80),
    "GO Biological process": _trunc_cmap("Reds", 0.22, 0.78),
    "GO Molecular Function": _trunc_cmap("BuGn", 0.22, 0.82),
    "GO Cellular Component": _trunc_cmap("Purples", 0.25, 0.76),
    "Cell-type signatures": _trunc_cmap("Oranges", 0.23, 0.79),
}


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
    df = res.dropna(subset=[score_col, padj_col]).copy()
    df = df[(df[score_col] > 0) & (df[padj_col] > 0) & (df[padj_col] < padj_threshold)]
    ax.set_title(set_name, loc="center", fontsize=11.5, weight="bold", pad=4)
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No significant terms",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="0.55",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    df = df.sort_values(score_col, ascending=False).head(n_top).iloc[::-1]
    terms = df[term_col].astype(str).values
    scores = df[score_col].to_numpy(dtype=float)
    nlogp = -np.log10(np.clip(df[padj_col].to_numpy(dtype=float), 1e-50, None))
    vmin = max(float(np.floor(nlogp.min())), 1.0)
    vmax = max(float(np.ceil(nlogp.max())), vmin + 0.5)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    y = np.arange(len(terms))
    ax.barh(y, scores, color=cmap(norm(nlogp)), edgecolor="white", linewidth=0.5, height=0.78)
    xmax = float(scores.max())
    longest_term = max((len(t) for t in terms), default=0)
    ax.set_xlim(0, xmax * (1.18 + 0.012 * max(0, longest_term - 28)))
    ax.set_xlabel("Combined score", fontsize=10)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=2.5, width=0.6, labelsize=9, pad=2)
    halo = [pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)]
    pad = 0.012 * xmax
    for i, term in enumerate(terms):
        ax.text(
            pad,
            i,
            term,
            va="center",
            ha="left",
            color="#1a1a1a",
            fontsize=8.5,
            clip_on=False,
            path_effects=halo,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, aspect=12, pad=0.015)
    cbar.set_label(r"$-\log_{10}$(adj. $p$-value)", fontsize=8.5)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=8)
    cbar.outline.set_linewidth(0.5)


def _bg_per_mouse(adata_full, mouse_ids_target) -> dict:
    if adata_full is None:
        return {}
    bg_sp_all = np.asarray(adata_full.obsm["spatial"])
    bg_mouse_ids = adata_full.obs["mouse_id"].astype(str).to_numpy()
    out = {}
    for mid in np.unique(mouse_ids_target):
        mask = bg_mouse_ids == mid
        if mask.any():
            out[mid] = bg_sp_all[mask]
    return out


def _ordered_mice_by_age(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mouse_ids = obs["mouse_id"].astype(str).to_numpy()
    uniq_mice = np.unique(mouse_ids)
    age_per_mouse = np.array([float(obs.loc[mouse_ids == mid, "age"].iloc[0]) for mid in uniq_mice])
    order = np.argsort(age_per_mouse)
    return uniq_mice[order], age_per_mouse[order]


def _normalise_slice_xy(xy: np.ndarray, ref_xy: np.ndarray | None = None) -> np.ndarray:
    ref = xy if ref_xy is None else np.asarray(ref_xy, dtype=float)
    centre = np.nanmedian(ref, axis=0)
    centred = np.asarray(xy, dtype=float) - centre
    ref_centred = ref - centre
    radius = np.nanpercentile(np.linalg.norm(ref_centred, axis=1), 95)
    return centred / radius


def _maybe_subsample(n: int, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))



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
    color_scale: Literal["symmetric", "percentile"] = "symmetric",
    elev: float = 0,
    azim: float = -58,
    out: str | Path | None = None,
    dpi: int = DPI,
) -> plt.Figure:
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
            bg_sel = _maybe_subsample(bg_xy.shape[0], max_bg_per_slice, rng)
            ax.scatter(
                bg_xy[bg_sel, 0],
                bg_xy[bg_sel, 1],
                np.full(bg_sel.size, z),
                c="#D8D8D8",
                s=bg_dot_size,
                alpha=0.10,
                linewidths=0,
                depthshade=False,
                rasterized=True,
                zorder=1,
            )
        fg_xy = _normalise_slice_xy(sp[fg_mask], ref_xy)
        fg_vals = values[fg_mask]
        fg_sel = _maybe_subsample(fg_xy.shape[0], max_fg_per_slice, rng)
        sc_ref = ax.scatter(
            fg_xy[fg_sel, 0],
            fg_xy[fg_sel, 1],
            np.full(fg_sel.size, z),
            c=fg_vals[fg_sel],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=fg_dot_size,
            alpha=0.88,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            zorder=2,
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
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig
