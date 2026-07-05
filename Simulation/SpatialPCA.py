from __future__ import annotations

from typing import Optional, Sequence
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import linalg
from scipy.optimize import minimize_scalar
from scipy.sparse import block_diag as sp_block_diag, csr_matrix, isspmatrix
from scipy.sparse.linalg import eigsh


def scale_matrix_columns_like_r(X):
    """
    Column-wise center and scale like R's scale() on a matrix (NA-free).

    For X with shape (n_row, n_col), each column is centered and divided by
    its sample standard deviation (ddof=1), matching R's default.
    """
    X = np.asarray(X, float)
    means = np.nanmean(X, axis=0, keepdims=True)
    centered = X - means
    stds = np.nanstd(centered, axis=0, ddof=1, keepdims=True)
    stds[~np.isfinite(stds)] = 1.0
    stds[stds < 1e-12] = 1.0
    return centered / stds


def gene_wise_zscore_like_r_t_scale_t(expr):
    """
    Match R's t(scale(t(expr))) for expr of shape (n_genes, n_spots).

    Each gene (row) is standardized across spots, as in Seurat scale.data /
    SpatialPCA_Multiple_Sample merged expression.
    """
    expr = np.asarray(expr, float)
    return scale_matrix_columns_like_r(expr.T).T

def bandwidth_select(expr, method="silverman"):
    """
    Select a bandwidth from a gene x location matrix by computing a
    1D bandwidth per gene and taking the median.

    Parameters
    ----------
    expr : array-like, shape (n_genes, n_locations)
    method : str, currently only "silverman" is supported.

    Returns
    -------
    beta : float
        Chosen bandwidth.
    """
    expr = np.asarray(expr, float)
    if expr.ndim != 2:
        raise ValueError("expr must be 2-D (n_genes, n_locations).")
    if method.lower() != "silverman":
        raise NotImplementedError("Only Silverman bandwidth is implemented.")

    finite_mask = np.isfinite(expr)
    counts = finite_mask.sum(axis=1)

    # Means ignoring NaNs
    sums = np.nansum(expr, axis=1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = expr - means[:, None]
    centered[~finite_mask] = 0.0
    sq = np.sum(centered * centered, axis=1)
    stds = np.zeros_like(sums)
    valid_std = counts > 1
    stds[valid_std] = np.sqrt(sq[valid_std] / (counts[valid_std] - 1))

    q75 = np.nanpercentile(expr, 75, axis=1)
    q25 = np.nanpercentile(expr, 25, axis=1)
    iqr = q75 - q25
    alt_scale = iqr / 1.34
    alt_scale[~np.isfinite(alt_scale)] = 0.0
    stds[~np.isfinite(stds)] = 0.0

    scale = np.where(
        (stds > 0) & (alt_scale > 0),
        np.minimum(stds, alt_scale),
        np.where(stds > 0, stds, np.where(alt_scale > 0, alt_scale, np.nan)),
    )

    counts_float = counts.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bws = 0.9 * scale * np.power(counts_float, -0.2)
    invalid = (counts < 2) | (scale <= 0) | (~np.isfinite(bws)) | (bws <= 0)
    bws[invalid] = np.nan
    finite_bws = bws[np.isfinite(bws) & (bws > 0)]

    if finite_bws.size == 0:
        raise ValueError("Failed to compute any finite bandwidths.")
    return float(np.median(finite_bws))


def build_kernel(coords, bandwidth, kernel_type="gaussian", scale_passes=1):
    """
    Build spatial kernel matrix K from spatial coordinates.

    Parameters
    ----------
    coords : array-like, shape (n_locations, n_dims)
        Spatial coordinates (s_i).
    bandwidth : float
        Bandwidth gamma in exp(-||s-s'||^2 / gamma).
    kernel_type : {"gaussian", "cauchy", "quadratic"}
    scale_passes : int, default 1
        Number of times to apply column-wise scaling like R's scale(location)
        before distances. The R multi-sample pipeline assigns
        object@location <- scale(raw) and SpatialPCA_buildKernel then does
        location_normalized <- scale(object@location); use ``scale_passes=2``
        to match that path. Single-sample uses 1.

    Returns
    -------
    K : ndarray, shape (n_locations, n_locations)
    """
    coords = np.asarray(coords, float)
    # Match SpatialPCA R code: kernel is built on scaled coordinates
    # (location_normalized = scale(location)).
    for _ in range(int(scale_passes)):
        means = coords.mean(axis=0, keepdims=True)
        stds = coords.std(axis=0, ddof=1, keepdims=True)
        stds[~np.isfinite(stds)] = 1.0
        stds[stds < 1e-12] = 1.0
        coords = (coords - means) / stds
    d2 = squareform(pdist(coords, metric="euclidean")) ** 2

    if kernel_type == "gaussian":
        K = np.exp(-d2 / bandwidth)
    elif kernel_type == "cauchy":
        K = 1.0 / (1.0 + d2 / float(bandwidth))
    elif kernel_type == "quadratic":
        K = 1.0 - d2 / (d2 + float(bandwidth))
    else:
        raise ValueError(f"Unknown kernel_type '{kernel_type}'.")
    return K


def F_funct_sameG(W, G):
    """
    Direct translation of the R function F_funct_sameG(X, G):

        return - sum_i w_i^T G w_i

    where columns of W are the loading vectors w_i.
    """
    W = np.asarray(W, float)
    G = np.asarray(G, float)
    WG = W.T @ G
    return -float(np.sum(WG * W.T))


class SpatialPCA:
    """
    Python implementation of single-sample SpatialPCA.

    Parameters
    ----------
    n_components : int
        Number of spatial PCs (latent factors) to estimate.
    kernel_type : {"gaussian", "cauchy", "quadratic"}
        Spatial kernel type.
    bandwidth : float or None
        Bandwidth for the spatial kernel. If None, selected from expression
        using Silverman's rule (across genes).
    bandwidth_method : str
        Method for bandwidth selection ("silverman" currently).
    fast : bool
        Whether to use low-rank approximation for kernel eigen-decomposition.
    eigenvec_num : int or None
        If fast is True and eigenvec_num is not None, use this many top
        eigenvectors of the kernel.
    """

    def __init__(self,
                 n_components=20,
                 kernel_type="gaussian",
                 bandwidth=None,
                 bandwidth_method="silverman",
                 fast=True,
                 eigenvec_num=None):
        self.n_components = int(n_components)
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.bandwidth_method = bandwidth_method
        self.fast = fast
        self.eigenvec_num = eigenvec_num

        # Fitted attributes
        self.tau_ = None
        self.W_ = None
        self.sigma2_ = None
        self.kernel_matrix_ = None
        self.kernel_eigvals_ = None  # delta
        self.kernel_eigvecs_ = None  # U
        self.spatial_pcs_ = None     # Z (d x n)

        # Precomputed design quantities
        self.M_ = None
        self.YM_ = None
        self.tr_YMY_ = None
        self.H_ = None

    # ------------------------------------------------------------------
    # Kernel eigen-decomposition, optionally low-rank as in R code
    # ------------------------------------------------------------------
    def _kernel_eigendecomposition(self, K):
        # K can be dense or sparse. For large problems we should avoid
        # materializing a dense kernel matrix.
        K_is_sparse = isspmatrix(K)
        if K_is_sparse:
            K_mat = K.tocsr()
        else:
            K_mat = np.asarray(K, float)
        n = K_mat.shape[0]

        if not self.fast:
            if K_is_sparse:
                # Full eigendecomposition of a large sparse kernel is not
                # feasible; match R package guidance by switching to fast mode.
                raise ValueError("Full eigendecomposition requested but kernel_matrix is sparse. Use fast=True.")
            evals, evecs = linalg.eigh(K_mat)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
        else:
            if self.eigenvec_num is not None:
                # Use user-specified number of eigenvectors
                k = min(self.eigenvec_num, n - 1)
                vals, vecs = eigsh(K_mat if K_is_sparse else csr_matrix(K_mat), k=k, which="LM")
                idx = np.argsort(vals)[::-1]
                evals = vals[idx]
                evecs = vecs[:, idx]
            else:
                if n > 5000:
                    # Large n: keep top 20 eigenvectors like the R code
                    k = min(20, n - 1)
                    vals, vecs = eigsh(K_mat if K_is_sparse else csr_matrix(K_mat), k=k, which="LM")
                    idx = np.argsort(vals)[::-1]
                    evals = vals[idx]
                    evecs = vecs[:, idx]
                else:
                    # Smaller n: full eigendecomposition, then keep
                    # enough eigenvalues to explain 90% of total.
                    # For n <= 5000 it is acceptable to densify.
                    K_dense = K_mat.toarray() if K_is_sparse else K_mat
                    evals_all, evecs_all = linalg.eigh(K_dense)
                    idx_all = np.argsort(evals_all)[::-1]
                    evals_all = evals_all[idx_all]
                    evecs_all = evecs_all[:, idx_all]
                    total = np.sum(evals_all)
                    if total <= 0:
                        # Fallback: keep all positive eigenvalues
                        pos_mask = evals_all > 1e-8
                        evals = evals_all[pos_mask]
                        evecs = evecs_all[:, pos_mask]
                    else:
                        cum = np.cumsum(evals_all) / float(total)
                        ind = np.searchsorted(cum, 0.9) + 1
                        evals = evals_all[:ind]
                        evecs = evecs_all[:, :ind]

        # Clip tiny or negative eigenvalues for numerical stability
        evals = np.maximum(evals, 1e-8)
        return evals, evecs

    # ------------------------------------------------------------------
    # Design matrices: H, M, etc (with optional covariates)
    # ------------------------------------------------------------------
    def _prepare_design_matrices(self, expr, covariates=None):
        """
        expr : (k, n) normalized & row-zscored
        covariates : (n, q-1) or None (without intercept)
        """
        expr = np.asarray(expr, float)
        k, n = expr.shape

        # H is n x q: intercept + optional covariates
        if covariates is None:
            H = np.ones((n, 1), dtype=float)
            q = 1
        else:
            C = np.asarray(covariates, float)
            if C.shape[0] != n:
                raise ValueError("covariates must have shape (n_spots, n_covariates).")
            H = np.concatenate([np.ones((n, 1), dtype=float), C], axis=1)
            q = H.shape[1]

        # Projection matrix M = I - H (H'H)^{-1} H'
        HH = H.T @ H
        HH_inv = linalg.inv(HH)
        M = np.eye(n) - H @ HH_inv @ H.T

        # Residualized expression YM = Y M
        YM = expr @ M          # (k, n)
        MYt = M @ expr.T       # (n, k)
        YMMYt = YM @ MYt       # (k, k)
        tr_YMY = float(np.trace(expr @ M @ expr.T))

        return {
            "expr": expr,
            "k": k,
            "n": n,
            "q": q,
            "H": H,
            "M": M,
            "YM": YM,
            "MYt": MYt,
            "YMMYt": YMMYt,
            "tr_YMY": tr_YMY
        }

    # ------------------------------------------------------------------
    # Safe logdet
    # ------------------------------------------------------------------
    @staticmethod
    def _slogdet_spd(A):
        try:
            chol, lower = linalg.cho_factor(A, lower=True, check_finite=False)
            return 2.0 * np.sum(np.log(np.diag(chol)))
        except linalg.LinAlgError:
            sign, logdet = np.linalg.slogdet(A)
            if sign <= 0:
                raise np.linalg.LinAlgError("Matrix not SPD or numerical issues in logdet.")
            return logdet

    @staticmethod
    def _solve_sym_pos(A, B):
        """
        Solve A X = B for symmetric positive (semi) definite A.

        Fallback chain:
          1. Cholesky (fast, exact for PD matrices)
          2. linalg.solve  (handles slight indefiniteness)
          3. lstsq with rcond=1e-10 (robust to near-singularity; suppresses
             LinAlgWarning that occurs when rcond is extremely small)
        """
        try:
            chol, lower = linalg.cho_factor(A, lower=True, check_finite=False)
            return linalg.cho_solve((chol, lower), B, check_finite=False)
        except linalg.LinAlgError:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", linalg.LinAlgWarning)
                    return linalg.solve(A, B, check_finite=False)
            except linalg.LinAlgError:
                # Last resort: least-squares with regularisation
                X, _, _, _ = np.linalg.lstsq(A, B, rcond=1e-10)
                return X

    @staticmethod
    def _solve_sym_pos_multiple(A, rhs_list):
        """
        Solve A X_i = B_i for multiple right-hand sides with a shared A.
        """
        try:
            chol, lower = linalg.cho_factor(A, lower=True, check_finite=False)
            return [linalg.cho_solve((chol, lower), rhs, check_finite=False) for rhs in rhs_list]
        except linalg.LinAlgError:
            return [SpatialPCA._solve_sym_pos(A, rhs) for rhs in rhs_list]

    def _construct_G_each(self, tau, params):
        """
        Build the G_each matrix given tau using cached design matrices.
        """
        delta = params["delta"]
        UtU = params["UtU"]
        Ut = params["Ut"]
        UtX = params["UtX"]
        YMU = params["YMU"]
        XtU = params["XtU"]
        YMX = params["YMX"]
        YMMYt = params["YMMYt"]
        MYt = params["MYt"]

        tau_system = UtU.copy()
        diag_idx = np.diag_indices_from(tau_system)
        tau_system[diag_idx] += tau * delta
        # Diagonal jitter to handle near-singular UtU caused by
        # degenerate/nearly-degenerate eigenvectors from eigsh on
        # block-diagonal kernels with similar slices.
        tau_system[diag_idx] += 1e-8 * np.maximum(tau_system[diag_idx], 1.0)
        sol_Ut, sol_UtX = self._solve_sym_pos_multiple(tau_system, [Ut, UtX])

        YMU_tauD_UtU_inv_Ut = YMU @ sol_Ut
        YMU_tauD_UtU_inv_UtX = YMU @ sol_UtX
        XtU_inv_UtX = XtU @ sol_UtX
        XtU_inv_UtX = (XtU_inv_UtX + XtU_inv_UtX.T) / 2.0

        left = YMX - YMU_tauD_UtU_inv_UtX
        right = left.T
        middle_solution = self._solve_sym_pos(-XtU_inv_UtX, right)

        G_each = YMMYt - YMU_tauD_UtU_inv_Ut @ MYt - left @ middle_solution
        return (G_each + G_each.T) / 2.0

    # ------------------------------------------------------------------
    # Objective for tau: negative log-likelihood in log(tau)
    # ------------------------------------------------------------------
    def _objective_logtau(self, log_tau, params):
        """
        Negative log-likelihood as a function of log(tau),
        closely following SpatialPCA_estimate_parameter in R.
        """
        tau = np.exp(log_tau)
        k = params["k"]
        n = params["n"]
        q = params["q"]
        PCnum = self.n_components

        delta = params["delta"]
        inv_delta = params["inv_delta"]
        UtU = params["UtU"]
        XtU = params["XtU"]
        UtX = params["UtX"]
        XtX = params["XtX"]
        tr_YMY = params["tr_YMY"]

        G_each = self._construct_G_each(tau, params)

        inv_system = UtU.copy()
        diag_idx = np.diag_indices_from(inv_system)
        inv_system[diag_idx] += (1.0 / tau) * inv_delta
        # Same diagonal jitter as in _construct_G_each for symmetry.
        inv_system[diag_idx] += 1e-8 * np.maximum(inv_system[diag_idx], 1.0)

        try:
            inv_chol = linalg.cho_factor(inv_system, lower=True, check_finite=False)
            sol_inv_system_UtX = linalg.cho_solve(inv_chol, UtX, check_finite=False)
            log_det_tauK_I_part = 2.0 * np.sum(np.log(np.diag(inv_chol[0])))
        except linalg.LinAlgError:
            sol_inv_system_UtX = SpatialPCA._solve_sym_pos(inv_system, UtX)
            log_det_tauK_I_part = self._slogdet_spd(inv_system)

        log_det_tauK_I = log_det_tauK_I_part + np.sum(np.log(tau * delta))

        Xt_invmiddle_X = XtX - XtU @ sol_inv_system_UtX
        Xt_invmiddle_X = (Xt_invmiddle_X + Xt_invmiddle_X.T) / 2.0
        log_det_Xt_inv_X = self._slogdet_spd(Xt_invmiddle_X)

        sum_det = (0.5 * log_det_tauK_I + 0.5 * log_det_Xt_inv_X) * PCnum

        # Leading eigenvectors of G_each (symmetric)
        k_genes = G_each.shape[0]

        if PCnum < k_genes:
            vals, vecs = eigsh(csr_matrix(G_each), k=PCnum, which="LM")
            W_est_here = vecs
        else:
            evals_full, evecs_full = linalg.eigh(G_each)
            idx = np.argsort(evals_full)[::-1][:PCnum]
            W_est_here = evecs_full[:, idx]

        F_val = F_funct_sameG(W_est_here, G_each)
        # tr_YMY + F_val = k*(n-q)*sigma^2 and must be positive.
        # Numerical errors from ill-conditioned solves can make it negative;
        # clamp to a small positive value so the log stays finite.
        log_arg = max(float(tr_YMY + F_val), 1e-30)
        ll = -sum_det - (k * (n - q)) / 2.0 * np.log(log_arg)
        return -ll  # we minimize this

    # ------------------------------------------------------------------
    # Fit single dataset
    # ------------------------------------------------------------------
    def fit(
        self,
        expr,
        coords=None,
        covariates=None,
        kernel_matrix=None,
        *,
        zscore_genes: bool = True,
    ):
        """
        Fit SpatialPCA on a single dataset.

        Parameters
        ----------
        expr : array-like, shape (n_genes, n_spots)
            Normalized expression matrix. By default we z-score each gene
            across spots (``zscore_genes=True``), matching SpatialPCA R
            ``SpatialPCA_buildKernel``. Set ``zscore_genes=False`` if expr is
            already scaled like ``t(scale(t(...)))`` (e.g. merged multi-sample
            matrix from the official pipeline).
        coords : array-like, shape (n_spots, n_dims), optional
            Spatial coordinates. Required if kernel_matrix is None.
        covariates : array-like, shape (n_spots, n_covariates), optional
            Per-spot covariates (without intercept). If None, only
            intercept is used (q = 1).
        kernel_matrix : array-like, shape (n_spots, n_spots), optional
            Precomputed spatial kernel matrix. If provided, coords and
            bandwidth are ignored for kernel construction.
        zscore_genes : bool, default True
            If True, apply gene-wise z-scoring (row-wise across spots).

        Returns
        -------
        self
        """
        expr = np.asarray(expr, float)
        if zscore_genes:
            # Match SpatialPCA R code: gene-wise z-scoring before all downstream steps.
            means = np.nanmean(expr, axis=1, keepdims=True)
            centered = expr - means
            stds = np.nanstd(centered, axis=1, ddof=1, keepdims=True)
            stds[~np.isfinite(stds)] = 1.0
            stds[stds < 1e-12] = 1.0
            expr = centered / stds
        # Build or use kernel
        if kernel_matrix is None:
            if coords is None:
                raise ValueError("Either coords or kernel_matrix must be provided.")
            coords = np.asarray(coords, float)
            if coords.shape[0] != expr.shape[1]:
                raise ValueError("coords must have shape (n_spots, n_dims).")
            # Scale each spatial dimension to mean 0, sd 1
            if self.bandwidth is None:
                self.bandwidth = bandwidth_select(expr, method=self.bandwidth_method)
            K = build_kernel(coords,
                             bandwidth=self.bandwidth,
                             kernel_type=self.kernel_type)
        else:
            if isspmatrix(kernel_matrix):
                K = kernel_matrix.tocsr()
            else:
                K = np.asarray(kernel_matrix, float)
            if K.shape[0] != expr.shape[1] or K.shape[1] != expr.shape[1]:
                raise ValueError("kernel_matrix must have shape (n_spots, n_spots).")

        self.kernel_matrix_ = K

        # Prepare design matrices and residualized Y
        params = self._prepare_design_matrices(expr, covariates=covariates)

        # Kernel eigen-decomposition
        delta, U = self._kernel_eigendecomposition(K)
        params["delta"] = delta
        params["U"] = U
        params["inv_delta"] = 1.0 / delta

        Ut = U.T
        params["Ut"] = Ut
        H = params["H"]
        YM = params["YM"]
        params["YMU"] = YM @ U
        params["XtU"] = H.T @ U
        params["UtX"] = Ut @ H
        params["YMX"] = YM @ H
        params["UtU"] = Ut @ U
        params["XtX"] = H.T @ H

        # Optimize log(tau) in [-10, 10]
        res = minimize_scalar(lambda x: self._objective_logtau(x, params),
                              bounds=(-10.0, 10.0),
                              method="bounded")
        log_tau_opt = res.x
        self.tau_ = float(np.exp(log_tau_opt))

        # After tau is fixed, recompute G_each and estimate W, sigma^2
        tau = self.tau_
        k_genes = params["k"]
        n_spots = params["n"]
        q = params["q"]

        M = params["M"]
        tr_YMY = params["tr_YMY"]

        G_each = self._construct_G_each(tau, params)

        # Leading eigenvectors (loadings W)
        if self.n_components < k_genes:
            vals, vecs = eigsh(csr_matrix(G_each), k=self.n_components, which="LM")
            W = vecs   # (k_genes, d)
        else:
            evals_full, evecs_full = linalg.eigh(G_each)
            idx = np.argsort(evals_full)[::-1][:self.n_components]
            W = evecs_full[:, idx]

        # Noise variance sigma^2_0
        F_val = F_funct_sameG(W, G_each)
        sigma2_0 = (tr_YMY + F_val) / (k_genes * (n_spots - q))

        self.W_ = W
        self.sigma2_ = float(sigma2_0)
        self.kernel_eigvals_ = delta
        self.kernel_eigvecs_ = U
        self.M_ = M
        self.YM_ = YM
        self.tr_YMY_ = tr_YMY
        self.H_ = H

        # Compute spatial PCs Z (d x n)
        self._compute_spatial_pcs()
        return self

    # ------------------------------------------------------------------
    # Compute Spatial PCs Z given fitted parameters
    # ------------------------------------------------------------------
    def _compute_spatial_pcs(self):
        if self.W_ is None or self.tau_ is None or self.kernel_matrix_ is None:
            raise RuntimeError("Model must be fitted before computing spatial PCs.")

        W    = self.W_                  # (k, d)
        tau  = self.tau_
        K    = self.kernel_matrix_      # (n, n) dense or sparse
        n = K.shape[0]

        # Match SpatialPCA_SpatialPCs.R behavior: when fast=True, optionally
        # recompute a larger low-rank eigendecomposition for the kernel to
        # improve spatial PC estimation accuracy.
        if self.fast:
            if self.eigenvec_num is not None:
                k_eigs = min(self.eigenvec_num, n - 1)
                vals, vecs = eigsh(K if isspmatrix(K) else csr_matrix(np.asarray(K, float)), k=k_eigs, which="LM")
                idx = np.argsort(vals)[::-1]
                delta = np.maximum(vals[idx], 1e-8)
                U = vecs[:, idx]
            elif n > 5000:
                k_eigs = min(int(np.ceil(0.1 * n)), n - 1)
                vals, vecs = eigsh(K if isspmatrix(K) else csr_matrix(np.asarray(K, float)), k=k_eigs, which="LM")
                idx = np.argsort(vals)[::-1]
                delta = np.maximum(vals[idx], 1e-8)
                U = vecs[:, idx]
            else:
                U = self.kernel_eigvecs_
                delta = self.kernel_eigvals_
        else:
            U = self.kernel_eigvecs_
            delta = self.kernel_eigvals_
        M    = self.M_
        YM   = self.YM_

        Wt    = W.T                     # (d, k)
        WtYM  = Wt @ YM                 # (d, n)
        WtYMK = WtYM @ K                # (d, n)
        WtYMU = WtYM @ U                # (d, r)

        Ut   = U.T                      # (r, n)
        UtM  = Ut @ M                   # (r, n)
        UtMK = UtM @ K                  # (r, n)
        UtMU = UtM @ U                  # (r, r)

        inv_delta = 1.0 / delta
        system = UtMU.copy()
        diag_idx = np.diag_indices_from(system)
        system[diag_idx] += (1.0 / tau) * inv_delta
        try:
            chol = linalg.cho_factor(system, lower=True, check_finite=False)
            correction = linalg.cho_solve(chol, UtMK, check_finite=False)
        except linalg.LinAlgError:
            correction = self._solve_sym_pos(system, UtMK)

        Z = tau * (WtYMK - WtYMU @ correction)  # (d, n)
        self.spatial_pcs_ = Z

    def transform(self):
        """
        Return spatial PCs Z (d x n) after fitting.
        """
        if self.spatial_pcs_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.spatial_pcs_

def multi_sample_spatialpca(
    expr_list: Sequence[np.ndarray],
    location_list: Sequence[np.ndarray],
    *,
    integrated_expr: Optional[np.ndarray] = None,
    n_components: int = 20,
    kernel_type: str = "gaussian",
    bandwidth: Optional[float] = None,
    bandwidth_common: Optional[float] = 0.1,
    bandwidth_method: str = "silverman",
    fast: bool = True,
    eigenvec_num: Optional[int] = None,
    remove_batch_effect: bool = False,
    scale_locations_like_r: bool = True,
    zscore_genes: bool = True,
):
    """
    Multi-sample SpatialPCA aligned with R ``SpatialPCA_Multiple_Sample``.

    The R pipeline (``SpatialPCA_multiple_sample.R``) builds a block-diagonal
    kernel from per-sample Gaussian kernels sharing ``bandwidth_common``,
    sets merged expression to ``t(scale(t(integrated_data)))`` (gene-wise
    z-score across **all** spots), and calls ``SpatialPCA_EstimateLoading``
    **without** sample-indicator covariates. Seurat integration is not
    reimplemented here: pass ``integrated_expr`` if you already have an
    integrated matrix (genes × spots); otherwise expression matrices are
    concatenated and the same row-wise scaling is applied (analogous to
    using integrated ``scale.data``).

    Parameters
    ----------
    expr_list : sequence of array-like
        Each entry is shape (n_genes, n_spots_s), same gene order across
        samples. Ignored for merging if ``integrated_expr`` is provided
        (but lengths must still match ``location_list`` for spot counts).
    location_list : sequence of array-like
        Each entry is shape (n_spots_s, n_dims) in raw coordinates.
    integrated_expr : array-like of shape (n_genes, sum_s n_spots_s), optional
        If given, used as the merged expression before gene-wise scaling,
        instead of ``np.concatenate(expr_list, axis=1)``. Use this when
        you have batch-corrected / integrated values (e.g. Seurat
        ``integrated@scale.data``).
    n_components : int
        Number of spatial PCs (``SpatialPCnum`` in R).
    kernel_type : {"gaussian", "cauchy", "quadratic"}
        Spatial kernel type.
    bandwidth : float or None
        If set, this value is used as the common bandwidth for every block
        (``bandwidth.set.by.user`` in R).
    bandwidth_common : float or None, default 0.1
        When ``bandwidth`` is None, this value is used as the common
        bandwidth, matching the R default ``bandwidth_common=0.1``. Set to
        None to fall back to Silverman selection on the merged, gene-scaled
        matrix instead.
    bandwidth_method : str
        Silverman bandwidth selection when both ``bandwidth`` and
        ``bandwidth_common`` are None.
    fast : bool
        Low-rank kernel eigendecomposition (``fast=TRUE`` in R).
    eigenvec_num : int or None
        Optional rank for kernel eigenvectors when ``fast`` is True.
    remove_batch_effect : bool, default False
        If True, adds sample-indicator covariates (one-hot, drop last).
        The official R function does **not** use this; default False matches
        ``SpatialPCA_EstimateLoading`` without batch covariates.
    scale_locations_like_r : bool, default True
        If True, each sample's locations are scaled twice before distances:
        R assigns ``location <- scale(raw)`` then ``SpatialPCA_buildKernel``
        applies ``scale(location)`` again. If False, only one scaling pass
        is applied inside ``build_kernel`` (legacy behavior).

    Returns
    -------
    model : SpatialPCA
        Fitted model on all spots jointly.
    spatial_pcs_list : list of ndarray
        Per-sample slices of ``Z``, each (n_components, n_spots_s).
    location_scaled_list : list of ndarray
        Per-sample ``scale(location)`` (first pass only), shape
        (n_spots_s, n_dims), matching ``spatialpca_list[[i]]@location`` in R.
    """
    if len(expr_list) != len(location_list):
        raise ValueError("expr_list and location_list must have the same length.")

    if remove_batch_effect:
        warnings.warn(
            "remove_batch_effect=True is not part of SpatialPCA_Multiple_Sample "
            "(R calls SpatialPCA_EstimateLoading without sample covariates).",
            UserWarning,
            stacklevel=2,
        )

    n_samples = len(expr_list)
    expr_arrays = []
    loc_arrays = []
    n_spots_per_sample = []

    for i in range(n_samples):
        Y = np.asarray(expr_list[i], float)
        coords = np.asarray(location_list[i], float)
        expr_arrays.append(Y)
        loc_arrays.append(coords)
        n_spots_per_sample.append(int(Y.shape[1]))

    n_genes = expr_arrays[0].shape[0]
    for i, Y in enumerate(expr_arrays):
        if Y.shape[0] != n_genes:
            raise ValueError("All expr_list entries must have the same n_genes.")
        if Y.shape[1] != loc_arrays[i].shape[0]:
            raise ValueError(
                f"expr_list[{i}] n_spots ({Y.shape[1]}) must match "
                f"location_list[{i}] rows ({loc_arrays[i].shape[0]})."
            )

    total_spots = int(sum(n_spots_per_sample))

    if integrated_expr is not None:
        expr_all = np.asarray(integrated_expr, float)
        if expr_all.shape != (n_genes, total_spots):
            raise ValueError(
                f"integrated_expr must have shape ({n_genes}, {total_spots}); "
                f"got {expr_all.shape}."
            )
    else:
        expr_all = np.concatenate(expr_arrays, axis=1)
        if expr_all.shape != (n_genes, total_spots):
            raise ValueError("Concatenated expr dimensions do not match spot counts.")

    # R: MultipleSample_merge@normalized_expr = t(scale(t(integrated_data[...])))
    if zscore_genes:
        expr_merged_scaled = gene_wise_zscore_like_r_t_scale_t(expr_all)
    else:
        expr_merged_scaled = expr_all - expr_all.mean(axis=1, keepdims=True)

    # Bandwidth: explicit > bandwidth_common (R default 0.1) > Silverman
    if bandwidth is not None:
        bw_used = float(bandwidth)
    elif bandwidth_common is not None:
        bw_used = float(bandwidth_common)
    else:
        bw_used = float(bandwidth_select(expr_merged_scaled, method=bandwidth_method))

    scale_passes = 2 if scale_locations_like_r else 1

    Ks = []
    location_scaled_list = []
    for coords in loc_arrays:
        coords = np.asarray(coords, float)
        # R object@location after first scale (returned to user)
        location_scaled_list.append(scale_matrix_columns_like_r(coords))
        Ks.append(
            build_kernel(
                coords,
                bandwidth=bw_used,
                kernel_type=kernel_type,
                scale_passes=scale_passes,
            )
        )

    K_block_sparse = sp_block_diag([csr_matrix(K) for K in Ks], format="csr")

    model = SpatialPCA(
        n_components=n_components,
        kernel_type=kernel_type,
        bandwidth=bw_used,
        bandwidth_method=bandwidth_method,
        fast=fast,
        eigenvec_num=eigenvec_num,
    )

    covariates = None
    if remove_batch_effect and n_samples > 1:
        sample_ids = np.concatenate(
            [np.full(n, i, dtype=int) for i, n in enumerate(n_spots_per_sample)]
        )
        covariates = np.zeros((total_spots, n_samples - 1), dtype=float)
        for i in range(n_samples - 1):
            covariates[:, i] = (sample_ids == i).astype(float)

    model.fit(
        expr_merged_scaled,
        kernel_matrix=K_block_sparse,
        covariates=covariates,
        zscore_genes=False,
    )

    Z_all = model.transform()
    spatial_pcs_list = []
    start = 0
    for n_spots in n_spots_per_sample:
        end = start + n_spots
        spatial_pcs_list.append(Z_all[:, start:end])
        start = end

    return model, spatial_pcs_list, location_scaled_list, expr_merged_scaled, Ks
