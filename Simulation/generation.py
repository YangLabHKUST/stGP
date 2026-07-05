import sys
import os
import hashlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stgp.preprocessing import log1p_norm_list, log1p_norm_centered_list

def _make_seed(*parts):
    h = hashlib.md5(str(parts).encode(), usedforsecurity=False).digest()
    return int.from_bytes(h[:4], "little") % (2**31)


def rbf_1d(x, gamma):
    x = np.asarray(x)[:, None]
    return np.exp(- (x - x.T) ** 2 / gamma)

def robust_mvn(mean, cov, rng, jitter=1e-10):
    """Sample from N(mean, cov) via Cholesky, using an explicit Generator."""
    if np.sum(np.diag(cov)) == 0:
        return mean
    p = cov.shape[0]
    L = np.linalg.cholesky(cov + jitter * np.eye(p))
    z = rng.standard_normal(size=p)
    return mean + L @ z

def draw_dirichlet_pruned(G, p, k, min_threshold, rng, *, overlap_frac=0.0):
    """
    Generate W (p x G): for each program pick k active genes, sample from a
    Dirichlet, prune entries below the threshold, renormalize, and return
    both W and the list of active indices.

    Parameters
    ----------
    overlap_frac : float, default 0.0
        Fraction of each program's k genes that are shared with (sampled
        from) already-chosen genes of previous programs.  0.0 = fully
        disjoint (original behaviour); 0.5 = half of the genes overlap
        with earlier programs.  Clamped so that at least 1 gene is unique.
    """
    W = np.zeros((p, G))
    active_idx_list = []
    all_used = set()
    for j in range(p):
        n_overlap = 0
        if overlap_frac > 0 and len(all_used) > 0:
            n_overlap = max(1, min(int(round(k * overlap_frac)), k - 1, len(all_used)))
        n_unique = k - n_overlap

        pool_used = np.array(sorted(all_used), dtype=int)
        pool_unused = np.array(sorted(set(range(G)) - all_used), dtype=int)

        shared = rng.choice(pool_used, size=n_overlap, replace=False) if n_overlap > 0 else np.array([], dtype=int)
        unique = rng.choice(pool_unused, size=min(n_unique, len(pool_unused)), replace=False)
        idx = np.concatenate([shared, unique]).astype(int)

        while True:
            w = rng.dirichlet(np.ones(len(idx)))
            mask = (w >= min_threshold)
            if not mask.any():
                continue
            w = w * mask
            w = w / w.sum()
            break
        W[j, idx] = w
        active_idx_list.append(idx)
        all_used.update(idx.tolist())
    return W, active_idx_list

def DataGen(
    T, G, p, k,
    avg_cell, std_cell, min_cell,
    gamma_age, gamma_spa,
    sigma2_age_list, tau2_spa_list,
    sigma2_e,
    r=0,
    age_cov_type="rbf",
    rho=0.75,
    *,
    age_jitter_sd=None,
    age_jitter_frac=0.0,
    spatial=True,
    overlap_frac=0.0,
):
    ss = np.random.SeedSequence(_make_seed("DataGen", T, r))
    rng = np.random.default_rng(ss)

    Nlist = np.maximum(rng.normal(avg_cell, std_cell, T).astype(int), min_cell)
    N = int(np.sum(Nlist))

    age_grid = np.linspace(3.0, 30.0, T)
    spacing = float(age_grid[1] - age_grid[0]) if T > 1 else 1.0
    if age_jitter_sd is None:
        jitter_sd = float(age_jitter_frac) * spacing
    else:
        jitter_sd = float(age_jitter_sd)
    ages = age_grid + rng.normal(0.0, jitter_sd, size=T)
    Z = (ages - ages.mean()) / ages.std(ddof=1)

    if age_cov_type == 'rbf':
        K_age = rbf_1d(Z, gamma_age)
    elif age_cov_type == 'ar1':
        idx = np.arange(T)
        age_dist = np.abs(idx[:, None] - idx[None, :])
        K_age = rho ** age_dist
    else:
        raise ValueError("age_cov_type must be one of {'rbf','ar1'}")
    K_age += 1e-6 * np.eye(T)

    if spatial:
        coords_list = []
        Kspa_list = []
        for t in range(T):
            Nt = int(Nlist[t])
            rng_coord = np.random.default_rng(_make_seed("coord", T, r, t))
            xy = rng_coord.uniform(0.0, 100.0, size=(Nt, 2))
            xy_scale = (xy - xy.mean(axis=0, keepdims=True)) / xy.std(axis=0, ddof=1, keepdims=True)
            coords_list.append(xy_scale)
            sqn = np.sum(xy_scale**2, axis=1, keepdims=True)
            D2 = np.maximum(sqn + sqn.T - 2 * xy_scale @ xy_scale.T, 0.0)
            Kspa = np.exp(- D2 / gamma_spa)
            Kspa += 1e-6 * np.eye(Nt)
            Kspa_list.append(Kspa)
    else:
        # No-spatial (scRNA-seq) mode: no coordinates, no spatial kernel.
        coords_list = None
        Kspa_list = None

    Alpha = np.zeros((p, T), dtype=float)
    B_list = [[None for _ in range(T)] for __ in range(p)]
    for j in range(p):
        rng_alpha = np.random.default_rng(_make_seed("alpha", T, r, j))
        Alpha[j] = robust_mvn(np.zeros(T), sigma2_age_list[j] * K_age, rng_alpha)
        for t in range(T):
            Nt = int(Nlist[t])
            if spatial:
                Kt = Kspa_list[t]
                rng_b = np.random.default_rng(_make_seed("spatial", T, r, j, t))
                b_draw = robust_mvn(np.zeros(Nt), tau2_spa_list[j] * Kt, rng_b)
            else:
                # No spatial component: b_t = 0 exactly.
                b_draw = np.zeros(Nt)
            B_list[j][t] = b_draw

    H_list = []
    for t in range(T):
        Nt = int(Nlist[t])
        H_t_cols = []
        for j in range(p):
            # H^t_ij = alpha_{tj} + b^t_{ij}.
            # No-spatial: b = 0, so H^t = 1_{N_t} alpha_t^T (all cells share the age effect).
            H_t_cols.append(Alpha[j, t] * np.ones(Nt) + B_list[j][t])
        H_t = np.column_stack(H_t_cols)   # N_t x p
        H_list.append(H_t)

    rng_w = np.random.default_rng(_make_seed("W", T, r))
    W, active_idx_list = draw_dirichlet_pruned(G, p, k, 1 / (2 * k), rng_w, overlap_frac=overlap_frac)

    rng_noise = np.random.default_rng(_make_seed("noise", T, r))
    Y_list = []
    Signal_list = []
    Noise_list = []
    for t in range(T):
        Nt = int(Nlist[t])
        Signal_t = H_list[t] @ W
        E_t = rng_noise.normal(0.0, np.sqrt(sigma2_e), size=(Nt, G))
        Y_t = Signal_t + E_t
        Y_list.append(Y_t)
        Signal_list.append(Signal_t)
        Noise_list.append(E_t)

    return {
        "G": G,
        "T": T,
        "p": p,
        "k": k,
        "Nlist": Nlist,
        "ages": ages,
        "Z_age": Z,
        "K_age": K_age,
        "coords_list": coords_list,   # None when spatial=False
        "K_spa": Kspa_list,           # None when spatial=False
        "Alpha": Alpha,
        "B_list": B_list,
        "H_list": H_list,
        "W": W,
        "active_idx_list": active_idx_list,
        "sigma2_age": np.array(sigma2_age_list),
        "tau2_spa": np.zeros(p) if not spatial else np.array(tau2_spa_list),
        "sigma2_e": sigma2_e,
        "spatial": bool(spatial),
        "Signal_list": Signal_list,
        "Noise_list": Noise_list,
        "Y_list": Y_list
    }


def DataGenCounts(
    T,
    G,
    p,
    k,
    avg_cell,
    std_cell,
    min_cell,
    gamma_age,
    sigma2_age_list,
    sigma2_e=0.1,
    *,
    gamma_spa=None,
    tau2_spa_list=None,
    count_dist="nb",
    dispersion=10.0,
    target_sum=250.0,
    target_library_mean=500.0,
    cell_offset_mean=0.25,
    cell_offset_std=0.40,
    signal_var_fraction_target=None,
    r=0,
    age_cov_type="rbf",
    rho=0.75,
    age_jitter_frac=0.0,
    spatial=True,
    overlap_frac=0.0,
):
    """
    Count-data generator for MERFISH-like simulations.

    Notes
    -----
    - The cell-level offset ``b_i`` already plays the role of library-size /
      capture-efficiency variation on log scale.
    - If ``cell_offset_mean`` is None, the generator calibrates its mean so that
      expected per-cell total counts are near ``target_library_mean``.
    """
    # --- spatial parameter validation ---
    if not spatial:
        if gamma_spa is None:
            gamma_spa = 1.0
        if tau2_spa_list is None:
            tau2_spa_list = [0.0] * p
    else:
        if gamma_spa is None:
            raise ValueError("gamma_spa must be provided when spatial=True.")
        if tau2_spa_list is None:
            raise ValueError("tau2_spa_list must be provided when spatial=True.")

    base = DataGen(
        T, G, p, k,
        avg_cell, std_cell, min_cell,
        gamma_age, gamma_spa,
        sigma2_age_list, tau2_spa_list,
        sigma2_e,
        r=r,
        age_cov_type=age_cov_type,
        rho=rho,
        age_jitter_frac=age_jitter_frac,
        spatial=spatial,
        overlap_frac=overlap_frac,
    )

    # --- gene baseline (three-tier prior) ---
    rng = np.random.default_rng(_make_seed("counts_merfish", T, r))
    W = np.asarray(base["W"], dtype=float)
    _, G_w = W.shape
    H_list = base["H_list"]
    active_idx_list = base.get("active_idx_list")

    if active_idx_list is not None:
        marker_idx = np.unique(
            np.concatenate([np.asarray(idx, dtype=int).ravel() for idx in active_idx_list])
        )
    else:
        marker_idx = np.flatnonzero(np.any(W > 0.0, axis=0))
    marker_idx = marker_idx[(marker_idx >= 0) & (marker_idx < G_w)]

    alpha_g = np.empty(G_w, dtype=float)
    gene_tier = np.full(G_w, "mostly_off", dtype="<U16")
    marker_mask = np.zeros(G_w, dtype=bool)
    marker_mask[marker_idx] = True
    non_marker_idx = np.flatnonzero(~marker_mask)
    n_house = int(np.clip(round(0.30 * G_w), 0, non_marker_idx.size))
    house_idx = rng.choice(non_marker_idx, size=n_house, replace=False) if n_house > 0 else np.array([], dtype=int)
    house_mask = np.zeros(G_w, dtype=bool)
    house_mask[house_idx] = True
    off_mask = ~(marker_mask | house_mask)

    if np.any(marker_mask):
        alpha_g[marker_mask] = rng.uniform(-1.4, 0.2, size=int(marker_mask.sum()))
        gene_tier[marker_mask] = "marker"
    if np.any(house_mask):
        alpha_g[house_mask] = rng.uniform(1.2, 2.2, size=int(house_mask.sum()))
        gene_tier[house_mask] = "housekeeping"
    if np.any(off_mask):
        alpha_g[off_mask] = rng.uniform(-3.6, -1.8, size=int(off_mask.sum()))

    # --- cell offsets and optional signal-share rescaling ---
    signal_list = [np.asarray(H_t, dtype=float) @ W for H_t in H_list]
    signal_mat = np.vstack(signal_list) if signal_list else np.zeros((0, G_w), dtype=float)
    nlist = [int(np.asarray(H_t).shape[0]) for H_t in H_list]
    n_cells = int(signal_mat.shape[0])

    b_raw = rng.normal(0.0, float(cell_offset_std), size=n_cells) if n_cells > 0 else np.zeros(0)
    if b_raw.size:
        b_raw -= float(np.mean(b_raw))

    alpha_mean = float(np.mean(alpha_g)) if alpha_g.size else 0.0
    alpha_dev = alpha_g - alpha_mean
    b_dev = b_raw.copy()

    def _signal_share(sig, a_dev, b_vec):
        if sig.size == 0:
            return 1.0
        eta = sig + a_dev[None, :] + b_vec[:, None]
        v_eta = float(np.var(eta))
        return float(np.var(sig)) / v_eta if v_eta > 1e-12 else 1.0

    if signal_var_fraction_target is not None and signal_mat.size:
        target_svf = float(signal_var_fraction_target)
        if not (0.0 < target_svf < 1.0):
            raise ValueError("signal_var_fraction_target must be in (0, 1).")
        v_sig = float(np.var(signal_mat))
        residual = max(v_sig * (1.0 - target_svf) / max(target_svf, 1e-12), 0.0)
        # 85 % of residual variance goes to alpha_g, 15 % to b_i
        for dev, var_target in [(alpha_dev, 0.85 * residual), (b_dev, 0.15 * residual)]:
            v_raw = float(np.var(dev))
            if v_raw > 1e-16:
                dev[:] = dev * np.sqrt(var_target / v_raw)
            else:
                dev[:] = 0.0
        # fine-tune: binary search to hit the target exactly
        if _signal_share(signal_mat, alpha_dev, b_dev) < target_svf:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _signal_share(signal_mat, mid * alpha_dev, mid * b_dev) >= target_svf:
                    lo = mid
                else:
                    hi = mid
            alpha_dev[:] = lo * alpha_dev
            b_dev[:] = lo * b_dev

    alpha_g = alpha_mean + alpha_dev
    b_var = float(np.var(b_dev)) if b_dev.size else 0.0
    if cell_offset_mean is None:
        eta0 = np.clip(alpha_g[None, :] + signal_mat, -20.0, 10.0)
        mean_sum = float(np.mean(np.exp(eta0).sum(axis=1))) if eta0.size else 1.0
        target_lib = max(float(target_library_mean), 1.0)
        mu_b = np.log(target_lib / max(mean_sum, 1e-12)) - 0.5 * b_var
    else:
        mu_b = float(cell_offset_mean)
    b_all = b_dev + mu_b

    # --- draw counts ---
    Y_count_list, Mu_list = [], []
    expected_lib_sizes_list, observed_lib_sizes_list, cell_offset_list = [], [], []
    off = 0
    for n_t, signal_t in zip(nlist, signal_list):
        b_t = np.clip(b_all[off:off + n_t], -4.0, 4.0)
        off += n_t
        log_mu = np.clip(alpha_g[None, :] + b_t[:, None] + signal_t, -20.0, 10.0)
        mu = np.exp(log_mu)
        if count_dist == "poisson":
            Y_count = rng.poisson(mu)
        elif count_dist == "nb":
            phi = float(dispersion)
            if phi <= 0:
                raise ValueError("dispersion must be positive for NB.")
            Y_count = rng.poisson(rng.gamma(shape=phi, scale=mu / phi))
        else:
            raise ValueError("count_dist must be one of {'poisson', 'nb'}")
        Y_count_list.append(Y_count.astype(int))
        Mu_list.append(mu)
        expected_lib_sizes_list.append(mu.sum(axis=1))
        observed_lib_sizes_list.append(Y_count.sum(axis=1))
        cell_offset_list.append(b_t)

    signal_share = _signal_share(signal_mat, alpha_g - alpha_g.mean(), b_all - b_all.mean())

    # --- normalised views ---
    Y_log_norm_list = log1p_norm_list(Y_count_list, target_sum=target_sum)
    Y_log_centered_list, gene_means = log1p_norm_centered_list(Y_count_list, target_sum=target_sum)

    out = dict(base)
    out.update(
        Y_count_list=Y_count_list,
        Mu_count_list=Mu_list,
        library_size_list=expected_lib_sizes_list,
        observed_library_size_list=observed_lib_sizes_list,
        Y_log_norm_list=Y_log_norm_list,
        Y_log_centered_list=Y_log_centered_list,
        Y_log_list=Y_log_centered_list,
        log1p_gene_means=gene_means,
        count_dist=count_dist,
        dispersion=float(dispersion),
        target_sum=float(target_sum),
        target_library_mean=float(target_library_mean),
        alpha_g=alpha_g,
        gene_tier=gene_tier,
        marker_gene_idx=marker_idx,
        housekeeping_gene_idx=house_idx,
        mostly_off_gene_idx=np.flatnonzero(off_mask),
        cell_offset_mean=float(mu_b),
        cell_offset_std=float(np.std(b_all, ddof=1)) if b_all.size > 1 else 0.0,
        cell_offset_list=cell_offset_list,
        signal_count_list=signal_list,
        signal_var_fraction=float(signal_share),
        signal_var_fraction_target=signal_var_fraction_target,
        Y_list=Y_log_centered_list,
    )
    return out

