import warnings
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import svds
from scipy.stats import norm as _norm
import time

def project_simplex(v):
    v = np.asarray(v, dtype=float).ravel()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, v.size + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w / w.sum()

def project_simplex_topk(v, k):
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if (k is None) or (k >= n):
        return project_simplex(v)
    idx = np.argpartition(v, -k)[-k:]
    w = np.zeros_like(v)
    w[idx] = project_simplex(v[idx])
    return w

def UT_times_vec(v, Nlist): # Compute U^T v 
    v = np.asarray(v)
    cuts = np.cumsum((0, *Nlist))
    return np.add.reduceat(v, cuts[:-1])

def stack_or_use(Y, dtype=np.float64):
    Y = np.vstack(Y) if isinstance(Y, (list, tuple)) else Y
    return np.ascontiguousarray(Y, dtype=dtype)

def posterior_mean_alpha_b(
    Y,
    h_hat,
    w,
    sigma2_age,
    tau2_spa,
    sigma2_e,
    K_age,
    K_spa_list,
    Nlist,
    eigvals_list=None,
    eigvecs_list=None,
    K_age_inv=None,
    jitter=1e-12,
    compute_cov_alpha=False,
    mode="alpha_first",
):
    r"""
    Posterior of alpha and b given (w, theta) via the pseudo-observation model.

    The reduced model is  z = U alpha + b + eta,  eta ~ N(0, sigma_eta^2 I_N),
    with  z = Y w / ||w||^2  and  sigma_eta^2 = sigma_e^2 / ||w||^2.

    Priors:  alpha ~ N(0, sigma^2 K_age),  b ~ N(0, tau^2 K_spa).
    Let  D = tau^2 K_spa + sigma_eta^2 I_N  (block-diagonal, D_t = tau^2 K_spa^t + sigma_eta^2 I_{N_t}).

    Both posterior means follow from standard Gaussian conditioning on
    Sigma_z = sigma^2 U K_age U^T + D:
        E[alpha | z] = sigma^2 K_age U^T Sigma_z^{-1} z,
        E[b     | z] = tau^2  K_spa     Sigma_z^{-1} z.

    We compute v = Sigma_z^{-1} z via Woodbury on Sigma_z = D + sigma^2 U K_age U^T:
        Sigma_z^{-1} = D^{-1} - D^{-1} U (sigma^{-2} K_age^{-1} + U^T D^{-1} U)^{-1} U^T D^{-1}.

    Marginal posterior covariance of alpha:
        V_{alpha,alpha} = ( sigma^{-2} K_age^{-1} + U^T D^{-1} U )^{-1}.

    The ``mode`` argument selects how alpha_hat and b_hat are obtained from v:

    "alpha_first" (default)
        alpha_hat = sigma^2 K_age U^T v          (exact posterior mean)
        b_hat     = h_hat - U alpha_hat           (residual; requires h_hat)

    "b_first"
        b_hat     = tau^2 K_spa v                (exact posterior mean, block-wise)
        alpha_hat = block_mean_t(h_hat - b_hat)  (residual; requires h_hat)

    All three modes are mathematically equivalent when h_hat equals the exact
    posterior mean E[h | z] = Sigma_h Sigma_z^{-1} z.

    Parameters
    ----------
    Y : (N, G) ndarray
    h_hat : (N,) ndarray – posterior mean of h.
        Used as the "other side" of the subtraction in modes "alpha_first"
        (b = h_hat - U alpha) and "b_first" (alpha = block_mean(h_hat - b)).
    w : (G,) ndarray
    sigma2_age, tau2_spa, sigma2_e : float
    K_age : (T, T) ndarray
    K_spa_list : list of (N_t, N_t) ndarrays
    Nlist : array-like of length T
    eigvals_list, eigvecs_list : precomputed spatial eigendecompositions (optional)
    K_age_inv : (T, T) ndarray – precomputed K_age^{-1} (optional)
    jitter : float
    compute_cov_alpha : bool – if True, also compute V_{alpha,alpha}.
    mode : {"alpha_first", "b_first"}
        Decomposition strategy; see above.

    Returns
    -------
    alpha_hat : (T,) ndarray
    b_hat : (N,) ndarray
    V_alpha : (T, T) ndarray or None
    """
    w = np.asarray(w, dtype=float).ravel()
    h_hat = np.asarray(h_hat, dtype=float).ravel()
    w2 = float(np.dot(w, w))
    sigma2_e = float(sigma2_e)
    sigma2_age = float(sigma2_age)
    tau2_spa = float(tau2_spa)

    N = h_hat.size
    T = int(len(Nlist))

    # Degenerate case: sigma2_age ~ 0 => alpha prior is zero, no age component
    if sigma2_age < jitter:
        alpha_hat = np.zeros(T, dtype=float)
        b_hat = h_hat.copy()
        V_alpha = np.zeros((T, T), dtype=float) if compute_cov_alpha else None
        return alpha_hat, b_hat, V_alpha

    if K_spa_list is None:
        # No-spatial mode: D_t = sigma_eta^2 * I, so D_t^{-1} = I / sigma_eta^2.
        eigvals_list = None
        eigvecs_list = None
        tau2_spa = 0.0
    elif eigvals_list is None or eigvecs_list is None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(K_spa_list)
    if K_age_inv is None:
        K_age_inv = precompute_K_age_inv(K_age, jitter=jitter)

    # z = Y w / ||w||^2,  sigma_eta^2 = sigma_e^2 / ||w||^2
    Yw = np.dot(Y, w)
    z = Yw / w2
    sigma2_eta = sigma2_e / w2
    inv_sigma2_eta = 1.0 / sigma2_eta

    UTDinvU_diag = np.empty(T, dtype=float)
    Dinv1_list = []

    if K_spa_list is None:
        # D_t^{-1} = (1/sigma_eta^2) I => Dinv_z = z / sigma_eta^2
        Dinv_z = z * inv_sigma2_eta
        for t, n_t in enumerate(Nlist):
            n_t = int(n_t)
            Dinv1_list.append(np.ones(n_t) * inv_sigma2_eta)
            UTDinvU_diag[t] = float(n_t) * inv_sigma2_eta
    else:
        # D_t = tau^2 K_spa^t + sigma_eta^2 I_{N_t}
        # D_t^{-1} = (1/sigma_eta^2) A_t^{-1}  where A_t = I + (tau^2/sigma_eta^2) K_spa^t
        kappa_D = tau2_spa / sigma2_eta if sigma2_eta > 0 else 0.0
        D_facts = build_A_blocks(kappa_D, 1.0, eigvals_list, eigvecs_list, jitter=jitter)
        Dinv_z = Ainv_vec(z, D_facts) * inv_sigma2_eta
        for t, (V, VT, d, Vt1) in enumerate(D_facts):
            dVt1 = d * Vt1
            Dinv1_list.append(V @ dVt1 * inv_sigma2_eta)
            UTDinvU_diag[t] = inv_sigma2_eta * float(np.dot(d, Vt1 * Vt1))

    # U^T D^{-1} z  and  (U^T D^{-1} U)_{tt} = 1_t^T D_t^{-1} 1_t
    UT_Dinv_z = UT_times_vec(Dinv_z, Nlist)

    # Schur complement: S = sigma^{-2} K_age^{-1} + U^T D^{-1} U
    S = (K_age_inv / sigma2_age) + np.diag(UTDinvU_diag)
    S_jit = S + jitter * np.eye(T)
    S_fact = None
    try:
        S_fact = la.cho_factor(S_jit, lower=True, check_finite=False)
        q = la.cho_solve(S_fact, UT_Dinv_z, check_finite=False)
    except la.LinAlgError:
        _ev, _evec = np.linalg.eigh(S_jit)
        _thresh = max(float(np.max(_ev)) * np.finfo(float).eps * T, 1e-10)
        _ev_safe = np.where(_ev > _thresh, _ev, _thresh)
        q = _evec @ ((_evec.T @ UT_Dinv_z) / _ev_safe)
        warnings.warn(
            f"posterior_mean_alpha_b: Schur complement S ({T}x{T}) ill-conditioned "
            f"(min_ev={float(np.min(_ev)):.2e}); using pseudoinverse fallback",
            RuntimeWarning, stacklevel=2,
        )

    # Sigma_z^{-1} z = D^{-1} z - D^{-1} U S^{-1} U^T D^{-1} z
    Dinv_U_q = np.empty(N, dtype=float)
    off = 0
    for t, n_t in enumerate(Nlist):
        n_t = int(n_t)
        Dinv_U_q[off:off + n_t] = float(q[t]) * Dinv1_list[t]
        off += n_t
    v = Dinv_z - Dinv_U_q

    # ------------------------------------------------------------------ #
    # Decompose v = Sigma_z^{-1} z into alpha_hat and b_hat.            #
    # Three modes are supported; see docstring for details.             #
    # ------------------------------------------------------------------ #
    UTv = UT_times_vec(v, Nlist)

    if mode == "alpha_first":
        # alpha from GP posterior formula; b as residual of h_hat
        alpha_hat = sigma2_age * np.dot(K_age, UTv)
        b_hat = h_hat - np.repeat(alpha_hat, Nlist)

    elif mode == "b_first":
        # b from GP posterior formula (block-wise K_spa v);
        # alpha as block-mean residual of h_hat.
        # No-spatial: tau2_spa = 0, so b_hat = 0; alpha from block means of h_hat.
        b_hat = np.zeros(N, dtype=float)
        if K_spa_list is not None:
            off = 0
            for t, (n_t, lam_t, (Vt, VTt, _d, _Vt1)) in enumerate(
                    zip(Nlist, eigvals_list, D_facts)):
                n_t = int(n_t)
                vt = v[off:off + n_t]
                b_hat[off:off + n_t] = tau2_spa * (Vt @ (lam_t * (VTt @ vt)))
                off += n_t
        diff = h_hat - b_hat
        alpha_hat = UT_times_vec(diff, Nlist) / np.asarray(Nlist, dtype=float)

    else:
        raise ValueError(
            f"posterior_mean_alpha_b: unknown mode={mode!r}. "
            "Choose 'alpha_first' or 'b_first'."
        )

    V_alpha = None
    if compute_cov_alpha:
        # V_{alpha,alpha} = S^{-1}
        if S_fact is not None:
            V_alpha = la.cho_solve(S_fact, np.eye(T), check_finite=False)
        else:
            _ev, _evec = np.linalg.eigh(S_jit)
            _thresh = max(float(np.max(_ev)) * np.finfo(float).eps * T, 1e-10)
            _ev_safe = np.where(_ev > _thresh, _ev, _thresh)
            V_alpha = _evec @ (np.diag(1.0 / _ev_safe) @ _evec.T)

    return alpha_hat, b_hat, V_alpha


def posterior_interval_alpha(alpha_hat, V_alpha, q=0.05):
    """
    Pointwise posterior interval for the aging trajectory alpha.

    Constructs the interval  alpha_hat_t +/- z_{1-q/2} * sqrt(V_{alpha,alpha}_{tt})
    for each time point t.

    Parameters
    ----------
    alpha_hat : (T,) ndarray
        Posterior mean of alpha.
    V_alpha : (T, T) ndarray
        Marginal posterior covariance of alpha.
    q : float, default 0.05
        Significance level; the interval has coverage 1-q.

    Returns
    -------
    alpha_lower : (T,) ndarray
    alpha_upper : (T,) ndarray
    alpha_std : (T,) ndarray
        Pointwise posterior standard deviations sqrt(diag(V_alpha)).
    """
    alpha_std = np.sqrt(np.maximum(np.diag(V_alpha), 0.0))
    z_crit = _norm.ppf(1.0 - q / 2.0)
    alpha_lower = alpha_hat - z_crit * alpha_std
    alpha_upper = alpha_hat + z_crit * alpha_std
    return alpha_lower, alpha_upper, alpha_std

# -----------------------------
# Block inverse via precomputed eigenbasis
# -----------------------------
def build_A_blocks(alpha, tau2_spa, eigvals_list, eigvecs_list, jitter=1e-12):
    """
    Construct block inverse factors for A_t = I + alpha * tau^2 * K_spa^(t)
    using precomputed eigendecompositions.

    Returns a list of (V, VT, d, Vt1) tuples where:
        d   = 1 / (1 + alpha*tau^2*lam + jitter)   (diagonal in eigenbasis)
        Vt1 = V.T @ ones_t                          (precomputed for C^{-1}1)
    Applying A_t^{-1} v is then O(N_t^2) via V @ (d * (VT @ v)).
    """
    alpha_tau2 = float(alpha) * float(tau2_spa)
    eig_facts = []
    for lam, (V, VT, Vt1) in zip(eigvals_list, eigvecs_list):
        d = 1.0 / (1.0 + alpha_tau2 * lam + jitter)
        eig_facts.append((V, VT, d, Vt1))
    return eig_facts


def Ainv_vec(v, facts):
    """Apply A^{-1} (block diagonal) to a vector using eig-based facts."""
    out = np.empty_like(v)
    off = 0
    for (V, VT, d, _) in facts:
        n = V.shape[0]
        out[off:off + n] = V @ (d * (VT @ v[off:off + n]))
        off += n
    return out

def T_age_apply(v, K_age, Nlist):
    """Efficiently apply T_age v = U K_age U^T v without forming N x N matrices."""
    u = UT_times_vec(v, Nlist)       # length T
    return np.repeat(np.dot(K_age, u), Nlist)

def T_spa_apply(v, Kspa_list):
    """Apply the block-diagonal spatial operator T_spa."""
    out = np.zeros_like(v)
    off = 0
    for K in Kspa_list:
        n = K.shape[0]
        out[off:off + n] = np.dot(K, v[off:off + n])
        off += n
    return out


def precompute_spatial_eigdecomp(Kspa_list):
    """
    Precompute eigenvalues AND eigenvectors of each spatial kernel K_spa^(t).

    This is the preferred precomputation for the main hot path: having eigenvectors
    available lets build_A_blocks avoid repeated O(N_t^3) Cholesky factorizations
    and instead apply C_t^{-1} in O(N_t^2) via diagonal solves in the eigenbasis.

    Parameters
    ----------
    Kspa_list : list of (n_t x n_t) ndarray

    Returns
    -------
    eigvals_list : list of 1D ndarrays  (ascending order)
    eigvecs_list : list of 2D ndarrays  (columns are eigenvectors, i.e. la.eigh convention)
    """
    eigvals_list = []
    eigvecs_list = []
    for K in Kspa_list:
        lam, V = np.linalg.eigh(K)
        VT = np.ascontiguousarray(V.T)   # C-contiguous for fast VT @ x
        Vt1 = VT.sum(axis=1)             # = V.T @ ones_Nt, constant per dataset
        eigvals_list.append(lam)
        eigvecs_list.append((V, VT, Vt1))
    return eigvals_list, eigvecs_list


# -----------------------------
# E-step: posterior of h
# -----------------------------
def trace_posterior_cov_h(
    alpha,
    sigma2_age,
    tau2_spa,
    K_age_inv,
    eigvals_list,
    Nlist,
    jitter=1e-12,
    r_diag=None,
    q_diag=None,
    eigvecs_list=None,
):
    """
    Compute tr(Sigma_tilde_h) where the posterior covariance of h is

        Sigma_tilde_h = (Sigma_h^{-1} + alpha * I_N)^{-1},   alpha = ||w||^2 / sigma_e^2.

    Uses the Woodbury identity on A = C + U (alpha sigma^2 K_age) U^T where
    C = I + alpha tau^2 T_spa is block-diagonal.

    When r_diag and q_diag are provided (precomputed by the caller), no
    additional O(N_t^2) work is needed.  Otherwise they are computed from
    eigvecs_list.
    """
    T = int(len(Nlist))
    N = int(sum(Nlist))

    if eigvals_list is None:
        # No-spatial mode: A_t = I for all t, so C^{-1} = I.
        # tr(C^{-1}) = N, r_diag[t] = N_t, q_diag[t] = N_t.
        tr_Cinv = float(N)
        if r_diag is None:
            r_diag = np.asarray(Nlist, dtype=float)
        if q_diag is None:
            q_diag = np.asarray(Nlist, dtype=float)
    else:
        alpha_tau2 = alpha * tau2_spa
        tr_Cinv = 0.0
        for lam in eigvals_list:
            tr_Cinv += float(np.sum(1.0 / (1.0 + jitter + alpha_tau2 * lam)))

        if (r_diag is None) or (q_diag is None):
            if eigvecs_list is None:
                raise ValueError("Either (r_diag, q_diag) or eigvecs_list must be provided")
            r_diag = np.empty(T, dtype=float)
            q_diag = np.empty(T, dtype=float)
            for t, (lam, (V, VT, Vt1)) in enumerate(zip(eigvals_list, eigvecs_list)):
                d = 1.0 / (1.0 + alpha_tau2 * lam + jitter)
                dVt1 = d * Vt1
                r_diag[t] = float(np.dot(d, Vt1 * Vt1))
                q_diag[t] = float(np.dot(dVt1, dVt1))

    if sigma2_age < jitter:
        # Degenerate temporal prior: sigma2_age ~ 0 => no age low-rank term.
        # Then (I + alpha * Sigma_h)^{-1} = C^{-1}, so tr(A^{-1}) = tr(C^{-1}).
        tr_Sigma_post = (N - tr_Cinv) / alpha
        return max(float(tr_Sigma_post), 0.0)

    S = (K_age_inv / (alpha * sigma2_age)) + np.diag(r_diag)
    S_eye = np.eye(T)
    try:
        _fact = la.cho_factor(S + jitter * S_eye, lower=True, check_finite=False)
        S_inv = la.cho_solve(_fact, S_eye, check_finite=False)
    except la.LinAlgError:
        _ev, _evec = np.linalg.eigh(S)
        _thresh = max(float(np.max(_ev)) * np.finfo(float).eps * T, 1e-10)
        _ev_safe = np.where(_ev > _thresh, _ev, _thresh)
        S_inv = _evec @ (np.diag(1.0 / _ev_safe) @ _evec.T)
        warnings.warn(
            f"trace_posterior_cov_h: S ({T}x{T}) is ill-conditioned "
            f"(min_ev={float(np.min(_ev)):.2e}); using pseudoinverse fallback",
            RuntimeWarning, stacklevel=2,
        )
    S_inv_diag = np.diag(S_inv)

    tr_Ainv = tr_Cinv - float(np.sum(S_inv_diag * q_diag))
    tr_Sigma_post = (N - tr_Ainv) / alpha
    return max(float(tr_Sigma_post), 0.0)


def h_posterior(
    Y,
    w,
    sigma2_age,
    tau2_spa,
    sigma2_e,
    K_age,
    Kspa_list,
    Nlist,
    eigvals_list=None,
    K_age_inv=None,
    jitter=1e-12,
    eigvecs_list=None,
):
    """
    Exact Woodbury solve for the rank-1 posterior mean of h.

    Solves:
        (I + alpha * Sigma_h) h = Sigma_h (Y w / sigma_e^2),
    where alpha = ||w||^2 / sigma_e^2 and
        Sigma_h = sigma2_age * U K_age U^T + tau2_spa * T_spa.

    Returns
    -------
    h : (N,) ndarray
    info : dict
        Diagnostic metadata, includes {"method": "woodbury", "alpha": ..., "tr_Sigma_post": ...}.
    tr_Sigma_post : float
        Posterior trace tr(Sigma_tilde_h).
    """
    rhs = np.dot(Y, w) / sigma2_e
    alpha = float(np.dot(w, w) / sigma2_e)

    if Kspa_list is None:
        # No-spatial mode: tau2_spa = 0, A_t = I, C_t^{-1} = I.
        tau2_spa = 0.0
        eigvals_list = None
        eigvecs_list = None
    elif eigvals_list is None or eigvecs_list is None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)
    if K_age_inv is None:
        K_age_inv = precompute_K_age_inv(K_age, jitter=jitter)

    if sigma2_age < jitter:
        # Degenerate temporal prior: alpha is deterministically zero.
        # The posterior of h reduces to the spatial-only system.
        if Kspa_list is None:
            h = np.zeros_like(rhs, dtype=float)
        else:
            b = tau2_spa * T_spa_apply(rhs, Kspa_list)
            C_facts = build_A_blocks(
                alpha, tau2_spa, eigvals_list, eigvecs_list, jitter=jitter,
            )
            h = Ainv_vec(b, C_facts)
        tr_Sigma_post = trace_posterior_cov_h(
            alpha=alpha,
            sigma2_age=0.0,
            tau2_spa=tau2_spa,
            K_age_inv=K_age_inv,
            eigvals_list=eigvals_list,
            Nlist=Nlist,
            jitter=jitter,
            eigvecs_list=eigvecs_list,
        )
        info = dict(
            method="woodbury_degenerate_age",
            alpha=alpha,
            tr_Sigma_post=tr_Sigma_post,
        )
        return h, info, tr_Sigma_post

    T = int(len(Nlist))
    r_diag = np.empty(T, dtype=float)
    q_diag = np.empty(T, dtype=float)
    Cinv1_list = []

    if Kspa_list is None:
        # No spatial: b = sigma2_age * T_age * rhs, x = A^{-1} b = b (identity),
        # Cinv1_t = 1_{N_t}, r_diag[t] = q_diag[t] = N_t.
        b = sigma2_age * T_age_apply(rhs, K_age, Nlist)
        x = b.copy()
        for t, n_t in enumerate(Nlist):
            n_t = int(n_t)
            Cinv1_list.append(np.ones(n_t))
            r_diag[t] = float(n_t)
            q_diag[t] = float(n_t)
    else:
        b = sigma2_age * T_age_apply(rhs, K_age, Nlist) + tau2_spa * T_spa_apply(rhs, Kspa_list)
        C_facts = build_A_blocks(alpha, tau2_spa, eigvals_list, eigvecs_list, jitter=jitter)
        x = Ainv_vec(b, C_facts)
        for t, (V, VT, d, Vt1) in enumerate(C_facts):
            dVt1 = d * Vt1
            c1 = V @ dVt1
            Cinv1_list.append(c1)
            r_diag[t] = float(np.dot(d, Vt1 * Vt1))
            q_diag[t] = float(np.dot(dVt1, dVt1))

    r = UT_times_vec(x, Nlist) # r = U^T x = U^T (A^{-1} b)
    # S = (alpha * sigma2_age * K_age)^{-1} + U^T A^{-1} U
    S = (K_age_inv / (alpha * float(sigma2_age))) + np.diag(r_diag)
    S_jit = S + jitter * np.eye(T)
    try:
        _fact = la.cho_factor(S_jit, lower=True, check_finite=False)
        q = la.cho_solve(_fact, r, check_finite=False)
    except la.LinAlgError:
        _ev, _evec = np.linalg.eigh(S_jit)
        _thresh = max(float(np.max(_ev)) * np.finfo(float).eps * T, 1e-10)
        _ev_safe = np.where(_ev > _thresh, _ev, _thresh)
        q = _evec @ ((_evec.T @ r) / _ev_safe)
        warnings.warn(
            f"h_posterior: Schur complement S ({T}x{T}) is ill-conditioned "
            f"(min_ev={float(np.min(_ev)):.2e}); using pseudoinverse fallback",
            RuntimeWarning, stacklevel=2,
        )

    # A^{-1} U q via cached A^{-1} 1 vectors.
    y = np.empty_like(x)
    off = 0
    for t, n in enumerate(Nlist):
        n = int(n)
        y[off:off + n] = float(q[t]) * Cinv1_list[t]
        off += n
    h = x - y

    tr_Sigma_post = trace_posterior_cov_h(
        alpha=alpha,
        sigma2_age=sigma2_age,
        tau2_spa=tau2_spa,
        K_age_inv=K_age_inv,
        eigvals_list=eigvals_list,
        Nlist=Nlist,
        jitter=jitter,
        r_diag=r_diag,
        q_diag=q_diag,
    )
    info = dict(method="woodbury", alpha=alpha, tr_Sigma_post=tr_Sigma_post)
    return h, info, tr_Sigma_post


# -----------------------------
# Initialization of theta and (h, w)
# -----------------------------
def init_theta_and_hw(Y, Nlist):
    """
    Rank-1 least-squares initialization that returns
    (sigma2_age, tau2_spa, sigma2_e, h0, w0).
    """
    # 1. Obtain initial w0 via truncated SVD (leading right singular vector)
    _, _, VT = svds(Y, k=1)
    w0 = VT[0]
    w0 = project_simplex(np.abs(w0))

    # 2. Least-squares estimate of h given w0
    denom = float(np.dot(w0, w0)) + 1e-12
    h0 = np.dot(Y, w0) / denom

    # 3. Residual-based estimate of sigma_e^2
    resid = Y - np.outer(h0, w0)
    sigma2_e = float(np.sum(resid * resid) / resid.size)  # More numerically stable than mean(resid**2)

    # 4. Decompose h0 variance into between/within components for sigma_age^2 and tau_spa^2
    T = len(Nlist)
    alpha_hat = np.empty(T, dtype=np.float64)
    off = 0
    has_within_variation = False
    for t, n in enumerate(Nlist):
        h_block = h0[off:off + n]
        alpha_hat[t] = float(h_block.mean())
        if n > 1:
            has_within_variation = True
        off += n

    var_total = float(h0.var(ddof=1))
    var_between = float(alpha_hat.var(ddof=1))
    var_within = var_total - var_between if has_within_variation else 1e-12

    eps = 1e-12 * var_total
    var_between = max(var_between, eps)
    var_within  = max(var_within,  eps)

    # Rescale so the between/within components sum back to the total variance
    scale = var_total / (var_between + var_within)
    sigma2_age = scale * var_between
    tau2_spa   = scale * var_within

    return sigma2_age, tau2_spa, sigma2_e, h0, w0


def precompute_K_age_inv(K_age, jitter=1e-12):
    """
    Precompute K_age^{-1} once (with diagonal jitter) for repeated Woodbury updates.
    """
    K_age = np.asarray(K_age, dtype=float)
    Tdim = K_age.shape[0]
    K_age_chol = la.cho_factor(
        K_age + jitter * np.eye(Tdim),
        lower=True,
        check_finite=False,
    )
    return la.cho_solve(K_age_chol, np.eye(Tdim), check_finite=False)

# -----------------------------
# Woodbury implementation of Sigma1^{-1} (N x N) applied to vectors
# -----------------------------
def build_sigma1_woodbury_state(
    w,
    sigma2_age,
    tau2_spa,
    sigma2_e,
    K_age,
    Kspa_list,
    Nlist,
    K_age_inv=None,
    jitter=1e-12,
    eigvals_list=None,
    eigvecs_list=None,
):
    """
    Build Woodbury state for Sigma1 = ||w||^2 Sigma_h + sigma_e^2 I_N, where
        Sigma_h = sigma_age^2 T_age + tau_spa^2 T_spa.

    This state is used both for Sigma1^{-1} * vector and for MM denominators.

    Parameters
    ----------
    w : (G,) ndarray
    sigma2_age : float
    tau2_spa : float
    sigma2_e : float
    K_age : (T x T) ndarray
    Kspa_list : list of (n_t x n_t) ndarray
    Nlist : list of int
        Number of cells per individual.
    jitter : float
        Diagonal jitter for Cholesky.

    Returns
    -------
    state : dict
        {
          "C_facts"      : list of Cholesky factors for base C0 blocks (see below),
          "B_fact"       : Cholesky factor of B (T x T),
          "w2"           : ||w||^2,
          "sqrt_w2"      : sqrt(||w||^2),
          "sigma2_e"     : sigma_e^2 (used to scale C0^{-1} to C^{-1}),
          "alpha"        : ||w||^2 / sigma_e^2,
          "Nlist"        : Nlist,
          "N"            : total number of cells,
          "R_diag"       : (T,) diag of R = U^T C^{-1} U,
          "M_spa_diag"   : (T,) diag of M_spa = V^T T_spa V,  V = sqrt(w2) C^{-1} U,
          "E_diag"       : (T,) diag of E = V^T V,
        }
    """
    w = np.asarray(w, dtype=float).ravel()
    w2 = float(np.dot(w, w))
    sqrt_w2 = np.sqrt(w2)
    sigma2_e = float(sigma2_e)
    if (not np.isfinite(sigma2_e)) or (sigma2_e <= 0.0):
        raise ValueError("sigma2_e must be positive and finite")
    sigma2_age_floor = max(float(jitter), np.finfo(float).tiny)
    sigma2_age = float(sigma2_age)
    if (not np.isfinite(sigma2_age)) or (sigma2_age <= 0.0):
        sigma2_age = sigma2_age_floor
    else:
        sigma2_age = max(sigma2_age, sigma2_age_floor)
    alpha = w2 / sigma2_e

    N = int(sum(Nlist))
    T = int(len(Nlist))

    R_diag = np.empty(T, dtype=float)
    M_spa_diag = np.empty(T, dtype=float)
    E_diag = np.empty(T, dtype=float)

    C_facts = None  # populated only when Kspa_list is not None
    if Kspa_list is None:
        # No-spatial mode: C_t = sigma_e^2 * I, so C_t^{-1} = I / sigma_e^2.
        # R_diag[t] = N_t / sigma_e^2, M_spa_diag[t] = 0, E_diag[t] = N_t * w2 / sigma_e^4.
        for t, n_t in enumerate(Nlist):
            n_t = int(n_t)
            R_diag[t] = float(n_t) / sigma2_e
            M_spa_diag[t] = 0.0
            E_diag[t] = float(n_t) * w2 / sigma2_e ** 2
    else:
        if eigvals_list is None or eigvecs_list is None:
            eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)

        C_facts = build_A_blocks(alpha, tau2_spa, eigvals_list, eigvecs_list, jitter=jitter)

        for t, (V, VT, d, Vt1) in enumerate(C_facts):
            dVt1 = d * Vt1
            c1_base = V @ dVt1
            c1 = c1_base / sigma2_e
            R_diag[t] = float(np.dot(d, Vt1 * Vt1)) / sigma2_e
            v_t = sqrt_w2 * c1
            Kv = np.dot(Kspa_list[t], v_t)
            M_spa_diag[t] = float(np.dot(v_t, Kv))
            E_diag[t] = float(np.dot(dVt1, dVt1)) * (sqrt_w2 / sigma2_e) ** 2

    # 3) M^{-1} = (1 / sigma^2) K_age^{-1}  (precompute K_age^{-1} once if provided)
    if K_age_inv is None:
        K_age_inv = precompute_K_age_inv(K_age, jitter=jitter)
    M_inv = K_age_inv / float(sigma2_age)

    # 4) B = M^{-1} + U_tilde^T C^{-1} U_tilde = M^{-1} + ||w||^2 * diag(R_diag)
    B = M_inv + w2 * np.diag(R_diag)
    B_eye = np.eye(T)
    B_fact = None
    for _jit in (jitter, jitter * 1e4, jitter * 1e8):
        try:
            B_fact = la.cho_factor(B + _jit * B_eye, lower=True, check_finite=False)
            break
        except la.LinAlgError:
            continue
    if B_fact is None:
        _ev, _evec = np.linalg.eigh(B)
        _min_needed = float(np.max(_ev)) * np.sqrt(np.finfo(float).eps)
        B_fact = la.cho_factor(B + _min_needed * B_eye, lower=True, check_finite=False)
        warnings.warn(
            f"build_sigma1_woodbury_state: B ({T}x{T}) is ill-conditioned "
            f"(min_ev={float(np.min(_ev)):.2e}); added jitter={_min_needed:.2e}",
            RuntimeWarning, stacklevel=2,
        )

    state = dict(
        C_facts=C_facts,
        no_spatial=(Kspa_list is None),
        B_fact=B_fact,
        w2=w2,
        sqrt_w2=sqrt_w2,
        sigma2_e=sigma2_e,
        alpha=alpha,
        Nlist=list(Nlist),
        N=N,
        R_diag=R_diag,
        M_spa_diag=M_spa_diag,
        E_diag=E_diag,
    )

    return state


def sigma1_inv_apply_woodbury(v, state):
    """
    Apply Sigma1^{-1} to any vector v in R^N using the precomputed Woodbury state.
    """
    v = np.asarray(v, dtype=float).ravel()
    C_facts = state["C_facts"]
    B_fact = state["B_fact"]
    sqrt_w2 = state["sqrt_w2"]
    Nlist = state["Nlist"]
    sigma2_e = float(state["sigma2_e"])
    no_spatial = state.get("no_spatial", False)

    # v0 = C^{-1} v.
    # No-spatial: C_t = sigma_e^2 I => C^{-1} v = v / sigma_e^2.
    if no_spatial or C_facts is None:
        v0 = v / sigma2_e
    else:
        v0 = Ainv_vec(v, C_facts) / sigma2_e
    # u0 = U^T v0, u = sqrt(w2) * u0
    u0 = UT_times_vec(v0, Nlist)   # (T,)
    u  = sqrt_w2 * u0              # (T,)
    # t solves B t = u
    t = la.cho_solve(B_fact, u, check_finite=False)   # (T,)
    # z0 = U t, z_tilde = sqrt(w2) z0
    z0 = np.repeat(t, Nlist)      # (N,)
    z_tilde = sqrt_w2 * z0
    # w_vec = C^{-1} z_tilde
    if no_spatial or C_facts is None:
        w_vec = z_tilde / sigma2_e
    else:
        w_vec = Ainv_vec(z_tilde, C_facts) / sigma2_e
    # Sigma1^{-1} v = v0 - w_vec
    return v0 - w_vec


def compute_mm_denoms_structured(
    sigma1_state,
    K_age,
    eigvals_list,
    tau2_spa,
    sigma2_e,
    G,
):
    """
    Structured MM denominators for the rank-1 model, avoiding N x N Cholesky.

    We use the decomposition:
        Sigma1 = ||w||^2 Sigma_h + sigma_e^2 I_N
               = C + U_tilde M U_tilde^T,

    with
        C = ||w||^2 tau^2 T_spa + sigma_e^2 I_N  (block-diagonal),
        U_tilde = sqrt(||w||^2) U,  M = sigma_age^2 K_age.

    Inputs
    ------
    sigma1_state : dict
        Output of build_sigma1_woodbury_state(...), containing:
          - "B_fact"     : Cholesky factor of B = M^{-1} + ||w||^2 diag(R_diag)
          - "w2"         : ||w||^2
          - "N"          : total number of cells N
          - "R_diag"     : diag(U^T C^{-1} U)
          - "M_spa_diag" : diag(V^T T_spa V), V = sqrt(||w||^2) C^{-1} U
          - "E_diag"     : diag(V^T V)
    K_age : (T x T) ndarray
        Temporal kernel.
    eigvals_list : list of 1D arrays
        eigvals_list[t] = eigenvalues of K_spa^(t), precomputed once.
    tau2_spa : float
        Spatial variance component tau^2.
    sigma2_e : float
        Noise variance sigma_e^2.
    G : int
        Number of genes.

    Returns
    -------
    age_denom, spa_denom, err_denom : floats
        Denominators used in the MM updates for sigma_age^2, tau_spa^2, and sigma_e^2.
    """
    w2       = sigma1_state["w2"]
    B_fact   = sigma1_state["B_fact"]
    R_diag   = sigma1_state["R_diag"]        # (T,)
    M_spa_d  = sigma1_state["M_spa_diag"]    # (T,)
    E_d      = sigma1_state["E_diag"]        # (T,)
    N        = int(sigma1_state["N"])
    T        = int(R_diag.size)

    # B^{-1} and its diagonal (T is small; this is cheap)
    B_inv = la.cho_solve(B_fact, np.eye(T), check_finite=False)
    B_inv_diag = np.diag(B_inv)

    # -------------------------
    # 1) age_denom = ||w||^2 * tr(Sigma1^{-1} T_age)
    #    with T_age = U K_age U^T, and R = U^T C^{-1} U = diag(R_diag).
    # -------------------------
    tr_KR = float(np.sum(np.diag(K_age) * R_diag))
    rr = R_diag[:, None] * R_diag[None, :]
    tr_KRBR = float(np.einsum("ij,ij->", K_age * rr, B_inv))
    age_denom = w2 * (tr_KR - w2 * tr_KRBR)

    # -------------------------
    # 2) spa_denom = ||w||^2 * tr(Sigma1^{-1} T_spa)
    #    No-spatial: T_spa = 0, so spa_denom = 0.
    # -------------------------
    if eigvals_list is None:
        # No-spatial mode: C_t = sigma_e^2 I => tr(C^{-1}) = N / sigma_e^2
        tr_Cinv = float(N) / sigma2_e
        spa_denom = 0.0
    else:
        # Vectorized computation - precompute denominator components
        w2_tau2 = w2 * tau2_spa
        tr_Cinv_Tspa = 0.0
        tr_Cinv = 0.0
        for lam in eigvals_list:
            denom = w2_tau2 * lam + sigma2_e
            inv_denom = 1.0 / denom  # Compute once, use twice
            tr_Cinv_Tspa += np.sum(lam * inv_denom)
            tr_Cinv += np.sum(inv_denom)

        tr_Binv_Mspa = float(np.sum(B_inv_diag * M_spa_d))
        spa_denom = w2 * (tr_Cinv_Tspa - tr_Binv_Mspa)

    # -------------------------
    # 3) err_denom for sigma_e^2 MM under (Q^T ⊗ I_N) orthogonalization along w.
    #
    # tr(Sigma1^{-1}) = tr(C^{-1}) - tr(B^{-1} E),
    # where E = U_tilde^T C^{-1} C^{-1} U_tilde = V^T V.
    # -------------------------
    # tr(C^{-1}) from eigvals only (or from no-spatial formula above)

    tr_Binv_E = float(np.sum(B_inv_diag * E_d))
    tr_Sigma1_inv = tr_Cinv - tr_Binv_E
    err_denom = tr_Sigma1_inv + (G - 1) * N / sigma2_e  # Note: (G-1) accounts for rank-1 signal subspace

    return age_denom, spa_denom, err_denom

def mm_update_theta_rank1(
    Y,
    w,
    *,
    sigma2_age,
    tau2_spa,
    sigma2_e,
    K_age,
    Kspa_list,
    Nlist,
    eigvals_list,
    K_age_inv,
    G,
    fro2=None,
    variance_floor=1e-3,
    fix_sigma2_e=False,
    eigvecs_list=None,
):
    """
    One MM update for the rank-1 variance components with fixed w.
    Returns updated (sigma2_age, tau2_spa, sigma2_e) and diagnostics.

    When ``fix_sigma2_e`` is True and ``sigma2_e`` is provided, the noise variance
    is not updated by the MM step (matches :func:`mom_update_theta_rank1`), so
    only ``sigma2_age`` and ``tau2_spa`` move while ``sigma2_e`` stays fixed.
    """
    if fro2 is None:
        fro2 = float(np.sum(Y * Y, dtype=np.float64))

    Yw = np.dot(Y, w)
    w2 = float(np.dot(w, w))
    sqrt_w2 = np.sqrt(w2)
    y1 = Yw / sqrt_w2
    y1_norm2 = float(np.dot(Yw, Yw) / w2)
    y_perp_norm2 = max(fro2 - y1_norm2, 0.0)

    sigma1_state = build_sigma1_woodbury_state(
        w=w,
        sigma2_age=sigma2_age,
        tau2_spa=tau2_spa,
        sigma2_e=sigma2_e,
        K_age=K_age,
        Kspa_list=Kspa_list,
        Nlist=Nlist,
        K_age_inv=K_age_inv,
        eigvals_list=eigvals_list,
        eigvecs_list=eigvecs_list,
    )
    z1 = sigma1_inv_apply_woodbury(y1, sigma1_state)

    z1_age = T_age_apply(z1, K_age, Nlist)
    age_numer = w2 * float(np.dot(z1, z1_age))
    if Kspa_list is not None:
        z1_spa = T_spa_apply(z1, Kspa_list)
        spa_numer = w2 * float(np.dot(z1, z1_spa))
    else:
        spa_numer = 0.0
    sigma2_e_sq = float(sigma2_e) * float(sigma2_e)
    err_numer = float(np.dot(z1, z1)) + y_perp_norm2 / sigma2_e_sq

    age_denom, spa_denom, err_denom = compute_mm_denoms_structured(
        sigma1_state,
        K_age,
        eigvals_list,
        tau2_spa,
        sigma2_e,
        G,
    )
    age_ratio = age_numer / age_denom
    err_ratio = err_numer / err_denom
    sigma2_age_new = min(max(sigma2_age * np.sqrt(age_ratio), variance_floor), 1e6)
    if Kspa_list is not None:
        spa_ratio = spa_numer / spa_denom
        tau2_spa_new = min(max(tau2_spa * np.sqrt(spa_ratio), variance_floor), 1e6)
    else:
        spa_ratio = 0.0
        tau2_spa_new = 0.0
    sigma2_e_new = min(max(sigma2_e * np.sqrt(err_ratio), variance_floor), 1e6)
    if fix_sigma2_e and (sigma2_e is not None):
        sigma2_e_new = float(sigma2_e)

    diagnostics = dict(
        w2=w2,
        y1_norm2=y1_norm2,
        y_perp_norm2=y_perp_norm2,
        age_numer=age_numer,
        spa_numer=spa_numer,
        err_numer=err_numer,
        age_denom=age_denom,
        spa_denom=spa_denom,
        err_denom=err_denom,
    )
    return sigma2_age_new, tau2_spa_new, sigma2_e_new, diagnostics


# -----------------------------
# Rank-1 model fitting
# -----------------------------
def _mom_theta_step(
    Y, w_new, h, tr_Sigma_post, *,
    K_age, Kspa_list, Nlist, eigvals_list,
    mom_enforce_nonneg, theta_update_label,
    variance_floor, fix_sigma2_e, sigma2_e,
    N, G,
):
    """Shared MoM theta-update logic used by 'mom' and 'mom_clip'."""
    enforce = mom_enforce_nonneg or (theta_update_label == "mom_clip")
    s2a, t2s, s2e, diag = mom_update_theta_rank1(
        Y, w_new,
        K_age=K_age, Kspa_list=Kspa_list, Nlist=Nlist,
        eigvals_list=eigvals_list, enforce_nonneg=enforce,
        variance_floor=variance_floor,
        fix_sigma2_e=fix_sigma2_e, sigma2_e=sigma2_e,
    )
    if not fix_sigma2_e:
        resid = Y - np.outer(h, w_new)
        w2_new = float(np.dot(w_new, w_new))
        s2e = max(
            (float(np.sum(resid * resid)) + w2_new * tr_Sigma_post) / (N * G),
            1e-12,
        )
    return s2a, t2s, s2e, diag


def fit_rank1(
    Y_list,
    Nlist,
    K_age,
    Kspa_list,
    k=None,
    max_iter=200,
    tol=1e-6,
    eigvals_list=None,
    K_age_inv=None,
    random_state=0,
    verbose=False,
    init=None,
    fix_sigma2_e=False,
    variance_floor=1e-6,
    theta_update="mm",
    mom_enforce_nonneg=False,
    eigvecs_list=None,
):
    """
    Fit the rank-1 spatio-temporal model described in the manuscript.
    The routine alternates between:
      - E-step: posterior mean of h via an exact Woodbury solve.
      - M-step: simplex (optionally top-k) projection for w.
      - Variance update via MM or MoM.
    """
    Y = np.vstack(Y_list)
    N, G = Y.shape

    if init is not None:
        h = np.asarray(init["h"], dtype=np.float64).ravel().copy()
        w = np.asarray(init["w"], dtype=np.float64).ravel().copy()
        sigma2_age = max(float(init["sigma2_age"]), variance_floor)
        tau2_spa = max(float(init.get("tau2_spa", 0.0)), variance_floor)
        sigma2_e = max(float(init["sigma2_e"]), variance_floor)
    else:
        sigma2_age, tau2_spa, sigma2_e, h, w = init_theta_and_hw(Y, Nlist)
        sigma2_age = max(sigma2_age, variance_floor)
        tau2_spa = max(tau2_spa, variance_floor)
        sigma2_e = max(sigma2_e, variance_floor)

    if Kspa_list is None:
        tau2_spa = 0.0
        eigvals_list = None
        eigvecs_list = None
    elif eigvals_list is None or eigvecs_list is None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)
    if K_age_inv is None:
        K_age_inv = precompute_K_age_inv(K_age)

    fro2 = float(np.sum(Y * Y, dtype=np.float64))
    mm_diag = {}
    update = np.inf

    for it in range(1, max_iter + 1):
        t_iter = time.perf_counter()

        # --- E-step: exact Woodbury solve for posterior mean of h
        h, _, tr_Sigma_post = h_posterior(
            Y, w, sigma2_age, tau2_spa, sigma2_e,
            K_age, Kspa_list, Nlist,
            eigvals_list=eigvals_list,
            K_age_inv=K_age_inv,
            eigvecs_list=eigvecs_list,
        )

        # --- M-step: w (simplex / optional k-sparse simplex)
        w_uncon = np.dot(Y.T, h) / (np.dot(h, h) + tr_Sigma_post)
        
        if k is not None:
            w_new = project_simplex_topk(w_uncon, k)
        else:
            w_new = project_simplex(w_uncon)

        # --- Variance update (MM or MoM)
        if theta_update == "mm":
            sigma2_age_new, tau2_spa_new, sigma2_e_new, mm_diag = mm_update_theta_rank1(
                Y, w_new,
                sigma2_age=sigma2_age, tau2_spa=tau2_spa, sigma2_e=sigma2_e,
                K_age=K_age, Kspa_list=Kspa_list, Nlist=Nlist,
                eigvals_list=eigvals_list, K_age_inv=K_age_inv,
                G=G, fro2=fro2,
                variance_floor=variance_floor,
                fix_sigma2_e=fix_sigma2_e,
                eigvecs_list=eigvecs_list,
            )
        elif theta_update in {"mom", "mom_clip"}:
            sigma2_age_new, tau2_spa_new, sigma2_e_new, mm_diag = _mom_theta_step(
                Y, w_new, h, tr_Sigma_post,
                K_age=K_age, Kspa_list=Kspa_list, Nlist=Nlist,
                eigvals_list=eigvals_list,
                mom_enforce_nonneg=mom_enforce_nonneg,
                theta_update_label=theta_update,
                variance_floor=variance_floor,
                fix_sigma2_e=fix_sigma2_e, sigma2_e=sigma2_e,
                N=N, G=G,
            )
        else:
            raise ValueError(
                "theta_update must be one of {'mm','mom','mom_clip'}"
            )
        t3 = time.perf_counter()

        eps_rel = 1e-12
        w_rel = float(np.sqrt(np.sum((w_new - w) ** 2)) / max(np.sqrt(np.sum(w ** 2)), eps_rel))
        s_age_rel = abs(sigma2_age_new - sigma2_age) / max(abs(sigma2_age), eps_rel)
        s_spa_rel = abs(tau2_spa_new - tau2_spa) / max(abs(tau2_spa), eps_rel)
        s_err_rel = abs(sigma2_e_new - sigma2_e) / max(abs(sigma2_e), eps_rel)
        update = max(w_rel, s_age_rel, s_spa_rel, s_err_rel)

        if verbose:
            print(
                f"[it={it:03d}] "
                f"delta_w={w_rel:.3e}, "
                f"delta_theta={update:.3e}, "
                f"time={t3 - t_iter:.3e}s"
            )

        w = w_new
        sigma2_age, tau2_spa, sigma2_e = (
            sigma2_age_new, tau2_spa_new, sigma2_e_new,
        )

        if update < tol:
            break

    # Final E-step with converged parameters to get h consistent with final theta
    h, _, tr_Sigma_post = h_posterior(
        Y, w, sigma2_age, tau2_spa, sigma2_e,
        K_age, Kspa_list, Nlist,
        eigvals_list=eigvals_list, K_age_inv=K_age_inv,
        eigvecs_list=eigvecs_list,
    )

    alpha_hat, b_hat, V_alpha = posterior_mean_alpha_b(
        Y=Y, h_hat=h, w=w,
        sigma2_age=sigma2_age, tau2_spa=tau2_spa, sigma2_e=sigma2_e,
        K_age=K_age, K_spa_list=Kspa_list, Nlist=Nlist,
        eigvals_list=eigvals_list, eigvecs_list=eigvecs_list,
        K_age_inv=K_age_inv,
        compute_cov_alpha=True,
    )
    alpha_lower, alpha_upper, alpha_std = posterior_interval_alpha(
        alpha_hat, V_alpha, q=0.05,
    )
    info = dict(
        n_iter=it,
        converged=(update < tol),
        last_update=float(update),
        fixed_sigma2_e=bool(fix_sigma2_e),
        theta_update=str(theta_update),
    )
    return {
        "h": h,
        "w": w,
        "theta": dict(
            sigma2_age=sigma2_age,
            tau2_spa=tau2_spa,
            sigma2_e=sigma2_e,
        ),
        "alpha": alpha_hat,
        "alpha_std": alpha_std,
        "alpha_lower": alpha_lower,
        "alpha_upper": alpha_upper,
        "b": b_hat,
        "tr_Sigma_post": tr_Sigma_post,
        "info": info,
    }


def fit_greedy(
    Y_list,
    Nlist,
    K_age,
    Kspa_list,
    p,
    k,
    inner_rank1_iters=200,
    tol=1e-6,
    random_state=0,
    verbose=False,
    variance_floor=1e-6,
    theta_update="mm",
    mom_enforce_nonneg=False,
):
    Y = stack_or_use(Y_list, dtype=np.float64)
    N, G = Y.shape

    if Kspa_list is not None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)
    else:
        eigvals_list, eigvecs_list = None, None
    K_age_inv = precompute_K_age_inv(K_age)

    H = np.zeros((N, p), float)
    W = np.zeros((p, G), dtype=np.float64)
    theta_list = np.ones((p, 2), dtype=float)
    sigma2_e_local = np.ones(p, float)

    residual = Y.copy()

    for j in range(p):
        rank1 = fit_rank1(
            residual, Nlist, K_age, Kspa_list, k,
            max_iter=inner_rank1_iters, eigvals_list=eigvals_list, tol=tol,
            K_age_inv=K_age_inv,
            verbose=(verbose >= 2),
            variance_floor=variance_floor,
            theta_update=theta_update,
            mom_enforce_nonneg=mom_enforce_nonneg,
            eigvecs_list=eigvecs_list,
        )
        H[:, j] = rank1["h"]
        W[j] = rank1["w"]
        theta_j = rank1["theta"]
        theta_list[j] = (theta_j["sigma2_age"], theta_j["tau2_spa"])
        sigma2_e_local[j] = theta_j["sigma2_e"]
        residual -= np.outer(rank1["h"], rank1["w"])

    sigma2_e = float(np.mean(sigma2_e_local))
    return H, W, theta_list, sigma2_e


# -----------------------------
# Rank-2 SVD split for a coupled factor pair
# -----------------------------
def split_coupled_factors(
    Y,
    H, W, tr_post, theta_list,
    mi, mj,
    sigma2_e,
    Nlist, K_age, Kspa_list, k,
    inner_rank1_iters,
    eigvals_list, eigvecs_list, K_age_inv,
    variance_floor, theta_update, mom_enforce_nonneg,
    random_state,
    N, G,
):
    """
    Re-extract the coupled factor pair (mi, mj) using a rank-2 SVD split.

    Strategy
    --------
    1. Form the joint-signal matrix ``Signal = h_i w_i^T + h_j w_j^T`` and
       compute its thin SVD.  The leading singular direction is the best
       rank-1 approximation of the combined signal and serves as a more
       principled starting point for r1a than a cold SVD of the raw residual.
    2. Fit factor ``mi`` (r1a) starting from SVD direction 1 of Signal.
    3. Fit factor ``mj`` (r1b) with a fresh SVD-of-residual initialisation
       (``init=None``): after r1a is subtracted, the dominant direction of
       ``residual_j`` naturally points toward the secondary signal component.
       Initialising from Vt2[1] is deliberately avoided — when the factors are
       already coupled, Signal is nearly rank-1 and Vt2[1] lies in its null
       space, which is a poor starting point.

    Returns
    -------
    Updated slices of H, W, tr_post, theta_list, and the recomputed
    sigma2_e, all packed in a dict.
    """
    residual_ij = (Y - np.dot(H, W)
                   + np.outer(H[:, mi], W[mi])
                   + np.outer(H[:, mj], W[mj]))

    # --- rank-2 SVD of the current joint-signal matrix -----------------
    Signal_ij = np.outer(H[:, mi], W[mi]) + np.outer(H[:, mj], W[mj])
    # full_matrices=False gives at most min(N,G) components; we only need 2
    U2, S2, Vt2 = np.linalg.svd(Signal_ij, full_matrices=False)

    w_init_a = project_simplex_topk(np.abs(Vt2[0]), k) if k else project_simplex(np.abs(Vt2[0]))

    init_a = dict(
        h=U2[:, 0] * S2[0],
        w=w_init_a,
        sigma2_age=theta_list[mi, 0],
        tau2_spa=theta_list[mi, 1],
        sigma2_e=sigma2_e,
    )

    # --- fit r1a from SVD direction 1 ----------------------------------
    # Starting from the leading rank-1 direction of the joint signal
    # matrix is more principled than a cold SVD of residual_ij, which
    # would re-discover the same dominant direction.
    r1a = fit_rank1(
        residual_ij, Nlist, K_age, Kspa_list, k,
        max_iter=inner_rank1_iters, eigvals_list=eigvals_list,
        K_age_inv=K_age_inv, random_state=random_state,
        fix_sigma2_e=True, variance_floor=variance_floor,
        theta_update=theta_update, mom_enforce_nonneg=mom_enforce_nonneg,
        eigvecs_list=eigvecs_list,
        init=init_a,
    )
    H[:, mi] = r1a["h"]
    W[mi] = r1a["w"]
    tr_post[mi] = r1a["tr_Sigma_post"]
    theta_list[mi] = (r1a["theta"]["sigma2_age"], r1a["theta"]["tau2_spa"])

    # --- fit r1b from the residual after removing r1a ------------------
    # Do NOT initialize from Vt2[1]: when Signal_ij is nearly rank-1
    # (factors already coupled), Vt2[1] sits in the null space of
    # Signal_ij and is a poor starting point.  Instead let fit_rank1
    # compute the SVD of residual_j to get its own natural init, which
    # points along the secondary signal direction in the data.
    residual_j = residual_ij - np.outer(H[:, mi], W[mi])
    r1b = fit_rank1(
        residual_j, Nlist, K_age, Kspa_list, k,
        max_iter=inner_rank1_iters, eigvals_list=eigvals_list,
        K_age_inv=K_age_inv, random_state=random_state + 1,
        fix_sigma2_e=True, variance_floor=variance_floor,
        theta_update=theta_update, mom_enforce_nonneg=mom_enforce_nonneg,
        eigvecs_list=eigvecs_list,
    )
    H[:, mj] = r1b["h"]
    W[mj] = r1b["w"]
    tr_post[mj] = r1b["tr_Sigma_post"]
    theta_list[mj] = (r1b["theta"]["sigma2_age"], r1b["theta"]["tau2_spa"])

    # --- recompute residual and sigma2_e --------------------------------
    residual = Y - np.dot(H, W)
    residual_norm_sq = float(np.sum(residual * residual))
    trace_correction = float(np.sum(np.sum(W * W, axis=1) * tr_post))
    sigma2_e_new = max((residual_norm_sq + trace_correction) / (N * G), 1e-12)

    return dict(
        residual=residual,
        residual_norm_sq=residual_norm_sq,
        sigma2_e=sigma2_e_new,
    )


# -----------------------------
# Multi-program (p > 1) case: greedy extraction from residuals
# -----------------------------
def fit_pfactor(
    Y_list,
    Nlist,
    K_age,
    Kspa_list,
    p,
    k=None,
    max_sweeps=500,
    inner_rank1_iters=200,
    inner_rank1_tol=1e-3,
    tol=1e-3,
    variance_floor=1e-6,
    init=None,
    random_state=0,
    verbose=0,
    theta_update="mm",
    mom_enforce_nonneg=False,
    eigvecs_list=None,
    eigvals_list=None,
    merge_threshold=0.9,
    max_merges=5,
):
    """
    Multi-factor (p programs) version with backfitting and warm starts.

    Parameters
    ----------
    verbose : int or bool
        0 (or False) = silent, 1 (or True) = sweep-level, 2 = rank1-level detail.

    Returns
    -------
    dict with keys H, W, theta, sigma2e, alpha, b, b_list, info.
    """
    verbose = int(verbose)
    Y = stack_or_use(Y_list, dtype=np.float64)
    N, G = Y.shape
    T = len(Nlist)

    if Kspa_list is None:
        eigvals_list, eigvecs_list = None, None
    elif eigvecs_list is None or eigvals_list is None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)
    K_age_inv = precompute_K_age_inv(K_age)

    if init is None:
        fit_init = fit_greedy(
            Y_list, Nlist, K_age, Kspa_list, p, k,
            variance_floor=variance_floor,
            theta_update=theta_update,
            mom_enforce_nonneg=mom_enforce_nonneg,
            verbose=verbose,
        )
        H = fit_init[0].copy()
        W = fit_init[1].copy()
        theta_list = fit_init[2].copy()
        sigma2_e = fit_init[3]
    else:
        W = init['W'].copy()
        H = init['H'].copy()
        theta_list = np.zeros([p, 2])
        theta_list[:, 0] = init['sigma2age'].copy()
        theta_list[:, 1] = init['tau2spa'].copy()
        sigma2_e = init['sigma2e']

    residual = Y - np.dot(H, W)
    Y_norm_sq = float(np.sum(Y * Y))
    drift_reset = 10
    cuts = np.cumsum(np.concatenate(([0], np.asarray(Nlist, int))))

    W_prev = W.copy()
    theta_list_prev = theta_list.copy()
    n_merges_done = 0

    tr_post = np.zeros(p, dtype=float)

    for sweep in range(1, max_sweeps + 1):
        start = time.time()
        if (sweep % drift_reset) == 0:
            np.dot(H, W, out=residual)
            np.subtract(Y, residual, out=residual)
        for j in range(p):
            # Temporarily add back the j-th component to obtain its residual slice
            residual += np.outer(H[:, j], W[j])
            init_j = dict(
                h=H[:, j],
                w=W[j],
                sigma2_age=theta_list[j, 0],
                tau2_spa=theta_list[j, 1],
                sigma2_e=sigma2_e,
            )
            rank1 = fit_rank1(
                residual, Nlist, K_age, Kspa_list, k,
                max_iter=inner_rank1_iters, eigvals_list=eigvals_list,
                tol=inner_rank1_tol,
                K_age_inv=K_age_inv,
                random_state=random_state, verbose=(verbose >= 2),
                init=init_j,
                fix_sigma2_e=True,
                variance_floor=variance_floor,
                theta_update=theta_update,
                mom_enforce_nonneg=mom_enforce_nonneg,
                eigvecs_list=eigvecs_list,
            )
            H[:, j] = rank1["h"]
            W[j] = rank1["w"]
            tr_post[j] = rank1["tr_Sigma_post"]
            residual -= np.outer(H[:, j], W[j])
            theta_list[j] = (rank1["theta"]["sigma2_age"],
                             rank1["theta"]["tau2_spa"])
        residual_norm_sq = float(np.sum(residual * residual))
        trace_correction = float(np.sum(
            np.sum(W * W, axis=1) * tr_post
        ))
        sigma2_e = max((residual_norm_sq + trace_correction) / (N * G), 1e-12)

        # --- Merge/split: detect near-duplicate W factors and re-extract ---
        if (merge_threshold > 0 and p > 1 and n_merges_done < max_merges):
            nw_chk = np.linalg.norm(W, axis=1, keepdims=True)
            Wn_chk = W / np.maximum(nw_chk, 1e-12)
            cos_mat = Wn_chk @ Wn_chk.T
            np.fill_diagonal(cos_mat, 0.0)
            mi, mj = np.unravel_index(
                np.argmax(np.abs(cos_mat)), cos_mat.shape)
            if mi > mj:
                mi, mj = mj, mi
            if abs(cos_mat[mi, mj]) > merge_threshold:
                n_merges_done += 1
                cos_before = float(abs(cos_mat[mi, mj]))
                split_result = split_coupled_factors(
                    Y=Y,
                    H=H, W=W, tr_post=tr_post, theta_list=theta_list,
                    mi=mi, mj=mj,
                    sigma2_e=sigma2_e,
                    Nlist=Nlist, K_age=K_age, Kspa_list=Kspa_list, k=k,
                    inner_rank1_iters=inner_rank1_iters,
                    eigvals_list=eigvals_list, eigvecs_list=eigvecs_list,
                    K_age_inv=K_age_inv,
                    variance_floor=variance_floor,
                    theta_update=theta_update,
                    mom_enforce_nonneg=mom_enforce_nonneg,
                    random_state=random_state + n_merges_done * 97,
                    N=N, G=G,
                )
                residual = split_result["residual"]
                residual_norm_sq = split_result["residual_norm_sq"]
                sigma2_e = split_result["sigma2_e"]
                if verbose >= 1:
                    print(f"[sweep={sweep:03d}] split factors {mi}&{mj} "
                          f"(cos={cos_before:.3f})")

        end = time.time()

        eps_rel = 1e-12
        W_norm = max(float(np.sqrt(np.sum(W_prev ** 2))), eps_rel)
        theta_log = np.log1p(theta_list)
        theta_log_prev = np.log1p(theta_list_prev)
        theta_norm = max(float(np.sqrt(np.sum(theta_log_prev ** 2))), eps_rel)
        d_W_rel = float(np.sqrt(np.sum((W - W_prev) ** 2))) / W_norm
        d_theta_rel = float(np.sqrt(np.sum((theta_log - theta_log_prev) ** 2))) / theta_norm
        rel_change = max(d_W_rel, d_theta_rel)

        if rel_change < tol:
            break

        if verbose >= 1:
            print(
                f"[sweep={sweep:03d}] dW_rel={d_W_rel:.3e} dTheta_rel={d_theta_rel:.3e} time={end-start:.3e}"
            )

        W_prev = W.copy()
        theta_list_prev = theta_list.copy()

    info = dict(
        n_sweeps=sweep,
        converged=(rel_change < tol),
        last_change=rel_change,
        n_merges=n_merges_done,
        theta_update=str(theta_update),
    )

    # -----------------------------
    # Posterior mean decomposition per factor via exact GP posterior:
    #   For each factor j form the leave-one-out residual
    #     R_j = Y - sum_{k!=j} h_k w_k^T
    #   and call posterior_mean_alpha_b on (R_j, w_j, theta_j) to obtain the
    #   exact posterior mean alpha_j and b_j = h_j - U alpha_j, as well as
    #   V_alpha_j = S_j^{-1} for the 95% credible interval.
    # -----------------------------
    alpha_mat = np.empty((T, p), dtype=float)
    b_mat     = np.empty((N, p), dtype=float)
    alpha_std_mat   = np.empty((p, T), dtype=float)
    alpha_lower_mat = np.empty((p, T), dtype=float)
    alpha_upper_mat = np.empty((p, T), dtype=float)
    HW_full = np.dot(H, W)
    for j in range(p):
        R_j = Y - HW_full + np.outer(H[:, j], W[j])
        alpha_j, b_j, V_alpha_j = posterior_mean_alpha_b(
            Y=R_j, h_hat=H[:, j], w=W[j],
            sigma2_age=theta_list[j, 0],
            tau2_spa=theta_list[j, 1],
            sigma2_e=sigma2_e,
            K_age=K_age, K_spa_list=Kspa_list, Nlist=Nlist,
            eigvals_list=eigvals_list, eigvecs_list=eigvecs_list,
            K_age_inv=K_age_inv,
            compute_cov_alpha=True, mode='b_first'
        )
        alpha_mat[:, j] = alpha_j
        b_mat[:, j]     = b_j
        lo, hi, sd = posterior_interval_alpha(alpha_j, V_alpha_j, q=0.05)
        alpha_std_mat[j]   = sd
        alpha_lower_mat[j] = lo
        alpha_upper_mat[j] = hi
    b_list = [b_mat[cuts[t]:cuts[t + 1], :] for t in range(T)]

    # Order factors by explained energy to mitigate rotational ambiguity for p>1
    comp_energy = np.sum(H * H, axis=0) * np.sum(W * W, axis=1)
    order = np.argsort(comp_energy)[::-1]
    if not np.all(order == np.arange(p)):
        H = H[:, order]
        W = W[order]
        theta_list = theta_list[order]
        alpha_mat = alpha_mat[:, order]
        b_mat = b_mat[:, order]
        b_list = [b_mat[cuts[t]:cuts[t + 1], :] for t in range(T)]
        alpha_std_mat = alpha_std_mat[order]
        alpha_lower_mat = alpha_lower_mat[order]
        alpha_upper_mat = alpha_upper_mat[order]

    return {
        'H': H,
        'W': W,
        'theta': theta_list,
        'sigma2e': sigma2_e,
        'alpha': alpha_mat.T,    # (p, T)
        'alpha_std': alpha_std_mat,      # (p, T)
        'alpha_lower': alpha_lower_mat,  # (p, T)
        'alpha_upper': alpha_upper_mat,  # (p, T)
        'b': b_mat,            # (N, p)
        'b_list': b_list,      # list[(N_t, p)]
        'info': info
    }
    
def post_backfit_prune(result, Y, total_energy, prune_energy_frac):
    """
    After backfitting, identify and remove degenerate factors whose
    explained-energy fraction falls below `prune_energy_frac`.

    Returns (pruned_indices, keep_indices) where pruned_indices lists
    the factor positions (0-based) that should be dropped.
    """
    H = result["H"]
    W = result["W"]
    p = W.shape[0]

    signal_energy = float(np.sum((H @ W) ** 2))
    energy_ref = max(signal_energy, total_energy, 1e-12)

    per_factor_energy = np.sum(H * H, axis=0) * np.sum(W * W, axis=1)
    fracs = per_factor_energy / energy_ref

    keep = []
    pruned = []
    for j in range(p):
        if fracs[j] < prune_energy_frac:
            pruned.append(j)
        else:
            keep.append(j)
    return pruned, keep


def fit_pfactor_auto(
    Y_list,
    Nlist,
    K_age,
    Kspa_list,
    p_max,
    k=None,
    inner_rank1_iters=500,
    inner_rank1_tol=1e-3,
    rel_improve_total_tol=0.05,
    min_component_norm=1e-8,
    variance_floor=1e-6,
    random_state=0,
    verbose=1,
    backfit_max_sweeps=500,
    backfit_tol=1e-3,
    theta_update="mm",
    mom_enforce_nonneg=False,
    prune_energy_frac=1e-2,
    merge_threshold=0.9,
    max_merges=5,
):
    """
    Data-driven rank decision for stGP via greedy-add then backfit-and-prune.

    Parameters
    ----------
    verbose : int or bool
        0 (or False) = silent, 1 (or True) = sweep-level, 2 = rank1-level detail.
    """

    verbose = int(verbose)
    p_max = int(p_max)
    if p_max <= 0:
        raise ValueError("p_max must be a positive integer")
    Y = stack_or_use(Y_list, dtype=np.float64)
    total_energy = float(np.sum(Y * Y))
    if total_energy <= 0.0 or (not np.isfinite(total_energy)):
        raise ValueError("Input data energy must be positive and finite")

    if Kspa_list is not None:
        eigvals_list, eigvecs_list = precompute_spatial_eigdecomp(Kspa_list)
    else:
        eigvals_list, eigvecs_list = None, None
    K_age_inv = precompute_K_age_inv(K_age)

    residual = Y.copy()
    residual_norm_sq = float(np.sum(residual * residual))

    H_list = []
    W_list = []
    theta_pairs = []
    sigma2_e_list = []
    selection_trace = []

    for j in range(p_max):
        rank1 = fit_rank1(
            residual,
            Nlist,
            K_age,
            Kspa_list,
            k=k,
            max_iter=inner_rank1_iters,
            tol=inner_rank1_tol,
            eigvals_list=eigvals_list,
            K_age_inv=K_age_inv,
            random_state=random_state + j,
            verbose=(verbose >= 2),
            variance_floor=variance_floor,
            theta_update=theta_update,
            mom_enforce_nonneg=mom_enforce_nonneg,
            eigvecs_list=eigvecs_list,
        )
        h_j = rank1["h"]
        w_j = rank1["w"]
        theta_j = rank1["theta"]
        sigma2_e_j = theta_j["sigma2_e"]

        component_energy = float(np.dot(h_j, h_j) * np.dot(w_j, w_j))
        component_norm = float(np.sqrt(max(component_energy, 0.0)))
        component = np.outer(h_j, w_j)

        residual_after = residual - component
        residual_after_norm_sq = float(np.sum(residual_after * residual_after))
        rel_improve_total = (residual_norm_sq - residual_after_norm_sq) / max(residual_norm_sq, 1e-12)

        keep = component_norm >= min_component_norm
        stop_reason = None
        if keep and (j > 0):
            if rel_improve_total < rel_improve_total_tol:
                keep = False
                stop_reason = "rel_improve_total"
        if (not keep) and (j == 0) and component_norm >= min_component_norm:
            keep = True
            stop_reason = None

        selection_trace.append(
            dict(
                factor_index=j + 1,
                rel_improve_total=rel_improve_total,
                component_norm=component_norm,
                residual_norm_before=residual_norm_sq,
                residual_norm_after=residual_after_norm_sq,
                keep=bool(keep),
                stop_reason=stop_reason,
            )
        )

        if not keep:
            break

        H_list.append(h_j)
        W_list.append(w_j)
        theta_pairs.append((theta_j["sigma2_age"], theta_j["tau2_spa"]))
        sigma2_e_list.append(sigma2_e_j)
        residual = residual_after
        residual_norm_sq = residual_after_norm_sq

    p_hat = int(len(H_list))
    if p_hat == 0:
        raise RuntimeError(
            "Automatic rank selection failed to find a usable factor; consider relaxing thresholds."
        )

    H_init = np.column_stack(H_list)
    W_init = np.vstack(W_list)
    theta_array = np.asarray(theta_pairs, dtype=float)
    sigma2_e_init = float(np.mean(sigma2_e_list))

    auto_info = dict(
        p_selected_greedy=p_hat,
        p_max=p_max,
        rel_improve_total_tol=rel_improve_total_tol,
        min_component_norm=min_component_norm,
        prune_energy_frac=prune_energy_frac,
        residual_fraction=float(residual_norm_sq / max(total_energy, 1e-12)),
        selection_trace=selection_trace,
        prune_history=[],
    )

    init = dict(
        H=H_init,
        W=W_init,
        sigma2age=theta_array[:, 0],
        tau2spa=theta_array[:, 1],
        sigma2e=sigma2_e_init,
    )

    _pfactor_kw = dict(
        Y_list=Y_list, Nlist=Nlist, K_age=K_age, Kspa_list=Kspa_list,
        k=k, max_sweeps=backfit_max_sweeps,
        inner_rank1_iters=inner_rank1_iters, inner_rank1_tol=inner_rank1_tol,
        tol=backfit_tol, variance_floor=variance_floor,
        random_state=random_state, verbose=verbose,
        theta_update=theta_update, mom_enforce_nonneg=mom_enforce_nonneg,
        eigvals_list=eigvals_list, eigvecs_list=eigvecs_list,
        merge_threshold=merge_threshold, max_merges=max_merges,
    )

    result = fit_pfactor(p=p_hat, init=init, **_pfactor_kw)

    if prune_energy_frac > 0:
        while p_hat > 1:
            pruned, keep_idx = post_backfit_prune(
                result, Y, total_energy, prune_energy_frac
            )
            if not pruned:
                break

            p_hat_new = len(keep_idx)
            auto_info["prune_history"].append(dict(
                p_before=p_hat,
                p_after=p_hat_new,
                pruned_indices=pruned,
            ))
            if verbose >= 1:
                print(
                    f"[auto_rank prune] p={p_hat} -> {p_hat_new} "
                    f"(dropped factors {pruned})"
                )

            keep_idx = np.array(keep_idx)
            H_kept = result["H"][:, keep_idx].copy()
            W_kept = result["W"][keep_idx].copy()
            theta_kept = result["theta"][keep_idx].copy()
            sigma2e_kept = result["sigma2e"]
            p_hat = p_hat_new

            init = dict(
                H=H_kept,
                W=W_kept,
                sigma2age=theta_kept[:, 0],
                tau2spa=theta_kept[:, 1],
                sigma2e=sigma2e_kept,
            )
            result = fit_pfactor(p=p_hat, init=init, **_pfactor_kw)

    auto_info["p_selected"] = p_hat
    result_info = dict(result.get("info", {}))
    result_info["auto_rank"] = auto_info
    result["info"] = result_info
    return result
    
def recover_low_rank_signal(Y, K):
    """
    PCA-style low-rank reconstruction via SVD.

    Notes
    -----
    - Classical PCA is defined on *centered* data. Here we column-center ``Y``,
      run an SVD on the centered matrix, and return the rank-``K`` reconstruction
      with the mean added back.
    - ``U, S, Vt`` correspond to the SVD of the centered matrix ``Y - mean(Y)``.
    """
    Y = np.asarray(Y, dtype=float)
    if K <= 0:
        mu = Y.mean(axis=0, keepdims=True)
        return np.broadcast_to(mu, Y.shape).copy(), np.zeros((Y.shape[0], 0)), np.zeros((0,)), np.zeros((0, Y.shape[1]))

    mu = Y.mean(axis=0, keepdims=True)
    Yc = Y - mu
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = int(min(K, S.shape[0]))
    Yc_hat = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return Yc_hat + mu, U, S, Vt


from scipy.optimize import linear_sum_assignment

def align_programs_and_mse(W, W_hat):
    diff = W[:, None, :] - W_hat[None, :, :]           
    C = (diff**2).mean(axis=2)                   
    row_ind, col_ind = linear_sum_assignment(C)
    W_hat_perm = W_hat[col_ind].copy()

    overall_mse = ((W - W_hat_perm)**2).sum()

    return {
        "perm": col_ind,            
        "B_aligned": W_hat_perm,         
        "overall_mse": overall_mse
    }