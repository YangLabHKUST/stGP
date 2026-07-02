"""
Kernel construction utilities for stGP.

Provides builders for the temporal kernel K_age and per-sample spatial kernels
K_spa, matching the manuscript definitions:
    K_age(t, t') = exp(-(A_t - A_t')^2 / gamma_age)
    K_spa(i, i') = exp(-||S_i - S_i'||^2 / gamma_spa)
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Automatic bandwidth selection
# ---------------------------------------------------------------------------
#
# Design goals
# ~~~~~~~~~~~~
# The bandwidth gamma in  K(d) = exp(-d^2 / gamma)  controls the effective
# rank of the kernel matrix and thus the number of spatial / temporal modes
# the GP prior can express.  We want gamma to be small enough that K has
# sufficient effective rank (so the variance components are identifiable),
# yet large enough that the kernel is not too spiky (preserving meaningful
# smoothness).
#
# Temporal bandwidth (bandwidth_select_temporal)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uses the median of ALL non-zero pairwise distances on z-scored ages.
# This estimates the global correlation length of the temporal GP.  The
# median pairwise distance is stable across T (≈ 1.0 on z-scored data),
# so gamma_age is approximately constant — reflecting a fixed correlation
# length regardless of sampling density.
#
# Spatial bandwidth (bandwidth_select_spatial)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uses the proportional k-nearest-neighbor (kNN) median distance, where
# k = ceil(frac * n_cells) for each slice.  This estimates the local
# density scale, which adapts to cell count: denser slices get smaller
# gamma, sparser slices get larger gamma.  The proportion-based k ensures
# fair comparison across slices with different cell counts.
# ---------------------------------------------------------------------------

def _knn_median_distance(
    coords: np.ndarray,
    k: int,
    *,
    max_subsample: int = 2000,
    seed: int = 0,
) -> float:
    """
    Median of k-th nearest neighbor distances.

    For each point, find the distance to its k-th nearest neighbor (excluding
    itself).  Return the median of these N distances.  Sub-samples for large N.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return 1.0
    k = min(k, n - 1)
    if n > max_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_subsample, replace=False)
        coords = coords[idx]
        n = max_subsample
        k = min(k, n - 1)
    d = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    knn_dists = np.partition(d, k - 1, axis=1)[:, k - 1]
    return float(np.median(knn_dists))


def bandwidth_select_spatial(
    coords_list: Sequence[np.ndarray],
    *,
    frac: float = 0.05,
    rho: float = 0.9,
    max_subsample: int = 2000,
    seed: int = 0,
) -> float:
    r"""
    Select spatial bandwidth via proportional k-nearest-neighbor median distance.

    For each slice with n_t cells, set k = ceil(frac * n_t) and compute the
    median k-NN distance.  Take the overall median across slices and set

        gamma_spa = d_kNN^2 / |log(rho)|

    so that  K(d_kNN) = rho.

    Parameters
    ----------
    coords_list : sequence of (n_t, 2) arrays
        Per-slice spatial coordinates (z-scored or raw — the function is
        scale-invariant as long as all slices use the same units).
    frac : float, default 0.05
        Proportion of cells to use as k.  k = ceil(frac * n_t) for each slice.
        Ensures fair bandwidth across slices with different cell counts.
    rho : float, default 0.9
        Target correlation at the kNN distance.  K(d_kNN) = rho.
    """
    per_slice: list[float] = []
    for coords in coords_list:
        n = coords.shape[0]
        k = max(int(np.ceil(frac * n)), 1)
        m = _knn_median_distance(
            coords, k=k, max_subsample=max_subsample, seed=seed,
        )
        per_slice.append(m)

    d_knn = float(np.median(per_slice))
    return d_knn ** 2 / abs(np.log(rho))


def bandwidth_select_temporal(
    ages: np.ndarray,
    *,
    rho: float = np.exp(-2),
) -> float:
    r"""
    Select temporal bandwidth via median pairwise distance on z-scored ages.

    Z-score ages, compute the median of all non-zero pairwise |Z_t - Z_t'|,
    and set

        gamma_age = d_med^2 / |log(rho)|

    so that  K(d_med) = rho.  The median pairwise distance on z-scored data
    is approximately constant (≈ 1.0) regardless of T, reflecting the global
    correlation length of the temporal GP.

    The default rho = e^{-2} ≈ 0.135 means that two individuals separated by
    the "typical" age difference have prior correlation ≈ 0.14, a conservative
    setting that allows the temporal GP to express moderately smooth age
    trajectories.

    Parameters
    ----------
    ages : (T,) array
        Chronological ages.  Z-scored internally before computing distances.
    rho : float, default exp(-2) ≈ 0.135
        Target correlation at median pairwise distance.  K(d_med) = rho.

    Returns
    -------
    gamma_age : float
        Bandwidth in z-scored units.  Must be used with z-scored ages in
        ``build_K_age`` (either pass raw ages with ``standardize=True``,
        or pre-standardize and pass ``standardize=False``).

    Notes
    -----
    The function z-scores ages internally (idempotent if already z-scored).
    The returned gamma is always in z-scored units.
    """
    ages = np.asarray(ages, dtype=np.float64).ravel()
    if ages.size < 2:
        return 1.0
    std = ages.std(ddof=1)
    if std < 1e-12:
        std = 1.0
    z = (ages - ages.mean()) / std
    diff = np.abs(z[:, None] - z[None, :])
    upper = diff[np.triu_indices_from(diff, k=1)]
    nonzero = upper[upper > 0]
    if nonzero.size == 0:
        return 1.0
    d_med = float(np.median(nonzero))
    return d_med ** 2 / abs(np.log(rho))


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def rbf_kernel_1d(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    1-D RBF (squared-exponential) kernel on raw values.

    K(x_i, x_j) = exp(-(x_i - x_j)^2 / gamma)

    This is the form used in the manuscript for K_age.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    diff = x[:, None] - x[None, :]
    return np.exp(-(diff ** 2) / gamma)


def ar1_kernel_1d(x: np.ndarray, rho: float) -> np.ndarray:
    """
    AR(1) kernel: K(i, j) = rho^|step(i) - step(j)|, where step is the 
    sequential index of unique values in x.

    Correlation depends on the number of steps between distinct time points
    in the ordered sequence.  For repeated age values, K(i, j) = 1.0 
    (perfect correlation).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Find unique values and map original indices to unique indices
    unique_x, inverse_indices = np.unique(x, return_inverse=True)
    
    # Build AR(1) kernel on unique values
    T_unique = len(unique_x)
    idx_unique = np.arange(T_unique)
    dist_unique = np.abs(idx_unique[:, None] - idx_unique[None, :])
    K_unique = rho ** dist_unique
    
    # Index into the unique kernel using the inverse mapping
    K = K_unique[inverse_indices[:, None], inverse_indices[None, :]]
    
    return K


def build_K_age(
    ages: np.ndarray,
    gamma: float = None,
    *,
    kernel: str = "rbf",
    rho: float = 0.75,
    standardize: bool = True,
    jitter: float = 0.0,
) -> np.ndarray:
    """
    Construct the temporal kernel matrix K_age.

    Parameters
    ----------
    ages : (T,) array
        Chronological ages of the T individuals.  For kernel="ar1" only the
        length T is used; actual age values are ignored.
    gamma : float
        Bandwidth for the RBF kernel (ignored when kernel="ar1").

        When ``gamma`` was obtained from :func:`bandwidth_select_temporal`
        (which always works in z-scored space), ``ages`` must also be on the
        z-scored scale when this function is called.  The default
        ``standardize=True`` handles this automatically for raw-age inputs.
        Pass ``standardize=False`` only when ``ages`` are already z-scored
        upstream *or* when ``gamma`` was manually tuned for the raw-age scale.
    kernel : {"rbf", "ar1"}
        Kernel type.
    rho : float
        AR(1) coefficient (only used when kernel="ar1").
    standardize : bool
        If True (default), z-score ages before computing the RBF kernel.
        Has no effect when kernel="ar1" (AR(1) uses index-based distances).
        Set to False when ages are already on the correct scale for ``gamma``
        (e.g. pre-standardized upstream, or using a manually-specified gamma
        calibrated for raw age units).
    jitter : float
        Diagonal jitter added for numerical stability.

    Returns
    -------
    K_age : (T, T) ndarray
    """
    ages = np.asarray(ages, dtype=np.float64).ravel()
    if standardize:
        std = ages.std(ddof=1)
        if std < 1e-12:
            std = 1.0
        ages = (ages - ages.mean()) / std

    if kernel == "rbf":
        K = rbf_kernel_1d(ages, gamma)
    elif kernel == "ar1":
        K = ar1_kernel_1d(ages, rho)
    else:
        raise ValueError(f"kernel must be 'rbf' or 'ar1', got {kernel!r}")

    K = 0.5 * (K + K.T)
    if jitter > 0:
        K += jitter * np.eye(K.shape[0])
    return K


def build_K_spa(
    coords: np.ndarray,
    gamma: float,
    *,
    standardize: bool = True,
    jitter: float = 0.0,
) -> np.ndarray:
    """
    Construct a single spatial kernel matrix.

    K_spa(i, i') = exp(-||S_i - S_i'||^2 / gamma_spa)

    Parameters
    ----------
    coords : (n, 2) ndarray
        Spatial coordinates for one individual/sample.
    gamma : float
        Spatial bandwidth.
    standardize : bool
        If True, z-score each coordinate dimension before computing distances.
    jitter : float
        Diagonal jitter.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if standardize:
        mu = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, ddof=1, keepdims=True)
        std[std < 1e-12] = 1.0
        coords = (coords - mu) / std

    d2 = cdist(coords, coords, metric="sqeuclidean")
    K = np.exp(-d2 / gamma)
    np.fill_diagonal(K, 1.0)
    if jitter > 0:
        K += jitter * np.eye(K.shape[0])
    return K


def build_K_spa_list(
    coords_list: list[np.ndarray],
    gamma: float,
    *,
    standardize: bool = True,
    jitter: float = 0.0,
) -> list[np.ndarray]:
    """
    Construct per-individual spatial kernel matrices from a list of coordinate
    arrays (one per individual).
    """
    return [
        build_K_spa(c, gamma, standardize=standardize, jitter=jitter)
        for c in coords_list
    ]


def build_K_spa_list_from_stacked(
    coords: np.ndarray,
    nlist: np.ndarray,
    gamma: float,
    *,
    standardize: bool = True,
    jitter: float = 0.0,
) -> list[np.ndarray]:
    """
    Construct per-individual spatial kernels from a single stacked coordinate
    array and a list of cell counts per individual.
    """
    coords = np.asarray(coords, dtype=np.float64)
    nlist = np.asarray(nlist, dtype=int)
    cuts = np.cumsum(np.concatenate(([0], nlist)))
    return [
        build_K_spa(
            coords[cuts[t]: cuts[t + 1], :],
            gamma,
            standardize=standardize,
            jitter=jitter,
        )
        for t in range(len(nlist))
    ]
