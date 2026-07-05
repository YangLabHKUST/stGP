"""
Figure: count-regime multi-program recovery (Simu-1 flagship).

Workflow (recommended: fix simulated data once, then fit/plot):
  --stage data  : generate DataGenCounts per replicate and save to disk
                  (results/simu1/datasets/rep_<k>.pkl). Same random draws for
                  all later fits.
  --stage fit   : load saved data per rep, run methods, cache MethodResults.
  --stage plot  : load saved data + cached fits; metrics and figures.
  --stage all   : data, then fit, then plot.

Cached datasets: results/simu1/datasets/rep_<k>.pkl (dict + params snapshot).

Methods are run without parallel computing (some algorithms do not allow it).
Execute non-Popari methods in the stGP conda environment, and Popari
separately in the Popari environment.

Aggregate outputs in Figures/simu1_geneprogram_count/:
  recovery_metrics_multirep.csv
  recovery_metrics_boxplot.png
  recovery_metrics_supp_boxplot.png
  method_runtime_multirep.csv   — wall time (s) per method × replicate
  method_runtime_boxplot.png

Per-replicate figures (same filenames as simu_gaussian_logscale.py), one folder per rep:
  singlereps/rep_<k>/programs_heatmap.png
  singlereps/rep_<k>/alpha_curves.png
  singlereps/rep_<k>/spatial_slice_prog0.png
  singlereps/rep_<k>/spatial_slice_all_programs.png
  singlereps/rep_<k>/H_recovery_scatter.png
  singlereps/rep_<k>/b_recovery_scatter.png
"""
import argparse
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

from generation import DataGenCounts
from benchmark_utils import (
    is_popari_available,
    load_method_result,
    run_mefisto_baseline,
    run_nmf_baseline,
    run_pca_baseline,
    run_popari_baseline,
    run_spatialpca_baseline,
    run_spatialpca_nz_baseline,
    run_stamp_baseline,
    run_stgp_pfactor,
    save_method_result,
    true_quantities_from_datagen,
)
from metrics_utils import summarize_method_performance, align_method_for_plot
from plot_utils import (
    DEFAULT_RECOVERY_MAIN_METRICS,
    DEFAULT_RECOVERY_SUPP1_METRICS,
    DEFAULT_RECOVERY_SUPP2_METRICS,
    plot_alpha_curves,
    plot_b_recovery_scatter,
    plot_H_recovery_scatter,
    plot_program_heatmaps,
    plot_method_runtime_boxplot,
    plot_recovery_metrics_boxplot,
    plot_recovery_metrics_supp_boxplot,
    plot_spatialpca_nz_comparison,
    plot_spatial_slice,
    plot_spatial_slice_all_programs,
    set_paper_style,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "simu1"
DATASET_DIR = RESULTS_DIR / "datasets"
FIG_DIR = BASE_DIR / "Figures" / "simu1_geneprogram_count"
SINGLE_REP_ROOT = FIG_DIR / "singlereps"

# Time-slice index for spatial-slice plots (matches simu_gaussian_logscale.py).
T_SLICE = 10

DEFAULT_PARAMS = dict(
    T=20,
    G=100,
    p=4,
    k=10,
    avg_cell=150,
    std_cell=15,
    min_cell=100,
    gamma_age=0.5,
    gamma_spa=2.0,
    sigma2_age_list=[10.0, 5.0, 0.0, 8.0],
    tau2_spa_list  =[6.0, 10.0, 8.0, 0.0],
    sigma2_e=0.1,
    age_cov_type="rbf",
    rho=0.5,
    age_jitter_frac=0.15,
    count_dist="poisson",
    target_library_mean=500.0,
    cell_offset_mean=None,
    cell_offset_std=0.20,
    target_sum=500.0,
    overlap_frac=0.25,
)

METHOD_RUNNERS = {
    "stGP": lambda data, params, seed: run_stgp_pfactor(
        data, p=params["p"], k=params["k"], random_state=seed
    ),
    "PCA": lambda data, params, seed: run_pca_baseline(data, p=params["p"]),
    "NMF": lambda data, params, seed: run_nmf_baseline(
        data, p=params["p"], random_state=seed
    ),
    "SpatialPCA": lambda data, params, seed: run_spatialpca_baseline(
        data, n_components=params["p"], bandwidth=params["gamma_spa"],
        fast=True,
    ),
    "SpatialPCA-nz": lambda data, params, seed: run_spatialpca_nz_baseline(
        data, n_components=params["p"], bandwidth=params["gamma_spa"],
        fast=True,
    ),
    "Popari": lambda data, params, seed: run_popari_baseline(
        data,
        p=params["p"],
        n_neighbors=8,
        train_iters=120,
        seed=seed,
        expression_floor=1e-8,
        lambda_Sigma_x_inv=1e-3,
        torch_device = 'cuda:2'
    ),
    "STAMP": lambda data, params, seed: run_stamp_baseline(
        data, p=params["p"], n_neighbors=8, max_epochs=500, min_epochs=120, seed=seed
    ),
    "MEFISTO": lambda data, params, seed: run_mefisto_baseline(
        data, p=params["p"], seed=seed
    ),
}

METHOD_ALIASES = {
    "stgp": "stGP",
    "pca": "PCA",
    "nmf": "NMF",
    "spatialpca": "SpatialPCA",
    "spatialpca-nz": "SpatialPCA-nz",
    "popari": "Popari",
    "stamp": "STAMP",
    "mefisto": "MEFISTO",
}

PANEL_ORDER = ["stGP", "PCA", "NMF", "SpatialPCA", "SpatialPCA-nz", "Popari", "STAMP", "MEFISTO"]
PANEL_ORDER_MAIN = [m for m in PANEL_ORDER if m != "SpatialPCA-nz"]


def _parse_methods(methods_arg: str):
    key = methods_arg.lower().strip()
    if key in {"all", "auto"}:
        methods = list(METHOD_RUNNERS.keys())
        if key == "auto" and not is_popari_available():
            methods = [m for m in methods if m != "Popari"]
            print(
                "[info] Popari package not found in current environment; "
                "running methods without Popari."
            )
        return methods
    if key in {"all_no_popari", "no_popari", "nonpopari"}:
        return [m for m in METHOD_RUNNERS.keys() if m != "Popari"]
    selected = []
    for token in methods_arg.split(","):
        tok = token.strip()
        if not tok:
            continue
        resolved = METHOD_ALIASES.get(tok.lower(), tok)
        if resolved in METHOD_RUNNERS:
            selected.append(resolved)
        else:
            print(f"[warn] Unknown method '{tok}', skipping.")
    return list(dict.fromkeys(selected))


def _result_path(method: str, rep: int) -> Path:
    return RESULTS_DIR / f"{method}_rep{rep}.pkl"


def _dataset_path(rep: int) -> Path:
    return DATASET_DIR / f"rep_{rep}.pkl"


def save_dataset_cache(rep: int, data: dict, params: dict) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"data": data, "params": dict(params)}
    with _dataset_path(rep).open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_cache(
    rep: int,
    params: dict,
    *,
    ignore_params_check: bool = False,
) -> dict:
    """
    Load cached simulated data for one replicate.

    Raises FileNotFoundError if no cache. Raises ValueError if stored params
    differ from *params* unless ignore_params_check is True.
    """
    path = _dataset_path(rep)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset cache missing: {path}. Run with --stage data (or --stage all) first."
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    cached_params = payload.get("params")
    if cached_params != params and not ignore_params_check:
        raise ValueError(
            f"Dataset rep={rep} was generated with different DEFAULT_PARAMS than the "
            f"current run. Regenerate with: --stage data --overwrite --reps ...\n"
            f"Or pass --ignore-data-params-check (not recommended)."
        )
    return payload["data"]


def _load_all_reps(methods: list[str], n_reps: int) -> dict:
    results: dict = {}
    for method in methods:
        reps = {}
        for rep in range(n_reps):
            path = _result_path(method, rep)
            if not path.exists():
                continue
            res, _ = load_method_result(str(path))
            reps[rep] = res
        if reps:
            results[method] = reps
    return results


def _plot_single_rep(
    _rep: int,
    data,
    rep_results: dict,
    params: dict,
    fig_dir: Path,
) -> None:
    """Generate all per-replicate diagnostic plots into *fig_dir* (count data)."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "stGP" not in rep_results:
        return

    Nlist = data["Nlist"]
    true0 = true_quantities_from_datagen(data)
    W_true = true0["W"]

    aligned = {}
    for name, res in rep_results.items():
        W_al, alpha_al, b_al, perm, W_raw = align_method_for_plot(W_true, res, Nlist=Nlist)
        aligned[name] = dict(W=W_al, alpha=alpha_al, b=b_al, perm=perm, W_raw=W_raw)

    res_stgp = rep_results["stGP"]
    perm_stgp = aligned["stGP"]["perm"]
    alpha_stgp_aligned = aligned["stGP"]["alpha"]
    b_stgp_aligned = aligned["stGP"]["b"]

    lo_raw = getattr(res_stgp, "alpha_lower", None)
    hi_raw = getattr(res_stgp, "alpha_upper", None)
    if lo_raw is None and res_stgp.metadata is not None:
        lo_raw = res_stgp.metadata.get("alpha_lower")
    if hi_raw is None and res_stgp.metadata is not None:
        hi_raw = res_stgp.metadata.get("alpha_upper")
    alpha_lower_stgp = (
        np.asarray(lo_raw, dtype=float)[perm_stgp] if lo_raw is not None else None
    )
    alpha_upper_stgp = (
        np.asarray(hi_raw, dtype=float)[perm_stgp] if hi_raw is not None else None
    )

    _SIGNED_METHODS = {"PCA", "SpatialPCA", "MEFISTO"}

    W_true_plot = W_true / (np.sum(np.abs(W_true), axis=1, keepdims=True) + 1e-12)
    mats = [W_true_plot]
    labels = ["True"]
    signed_flags = [False]
    for name in PANEL_ORDER_MAIN:
        if name not in aligned:
            continue
        if name in _SIGNED_METHODS:
            mats.append(aligned[name]["W_raw"])
        else:
            mats.append(aligned[name]["W"])
        labels.append(name)
        signed_flags.append(name in _SIGNED_METHODS)

    plot_program_heatmaps(
        mats, labels, figsize=(28, 5),
        show_gene_ticks=False,
        save_path=str(fig_dir / "programs_heatmap.png"),
        signed_flags=signed_flags,
    )

    t_idx = T_SLICE
    coords = data["coords_list"][t_idx]
    start = int(np.sum(Nlist[:t_idx]))
    end = int(np.sum(Nlist[: t_idx + 1]))
    b_true = true0["b"][0, start:end]
    b_stgp_slice = b_stgp_aligned[start:end, 0]
    plot_spatial_slice(
        coords, [b_true, b_stgp_slice], ["True", "stGP"],
        save_path=str(fig_dir / "spatial_slice_prog0.png"),
    )

    alpha_true = true0["alpha"]
    alpha_true_centered = alpha_true - alpha_true.mean(axis=1, keepdims=True)
    plot_alpha_curves(
        alpha_true=alpha_true_centered,
        alpha_estimates=[alpha_stgp_aligned],
        labels=["stGP"],
        save_path=str(fig_dir / "alpha_curves.png"),
        alpha_lower_list=[alpha_lower_stgp],
        alpha_upper_list=[alpha_upper_stgp],
    )

    T = int(params["T"])
    true_B_slice = [np.asarray(data["B_list"][j][t_idx]) for j in range(params["p"])]
    b_fit_slice = b_stgp_aligned[start:end, :]
    plot_spatial_slice_all_programs(
        coords=coords,
        true_B_slice=true_B_slice,
        b_fit_slice=b_fit_slice,
        t_idx=t_idx,
        save_path=str(fig_dir / "spatial_slice_all_programs.png"),
    )

    true_H = np.vstack(data["H_list"])
    H_fit = res_stgp.H[:, perm_stgp]
    plot_H_recovery_scatter(
        true_H=true_H,
        H_fit=H_fit,
        save_path=str(fig_dir / "H_recovery_scatter.png"),
    )

    true_B_flat = [
        np.concatenate([data["B_list"][j][t] for t in range(T)])
        for j in range(len(data["B_list"]))
    ]
    plot_b_recovery_scatter(
        true_B_flat=true_B_flat,
        b_fit=b_stgp_aligned,
        save_path=str(fig_dir / "b_recovery_scatter.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="Simu-1: multi-program recovery (count).")
    parser.add_argument(
        "--stage",
        choices=["data", "fit", "plot", "all"],
        default="all",
        help="data: only save simulated datasets; fit: load data and fit methods; "
        "plot: metrics/figures; all: data then fit then plot.",
    )
    parser.add_argument(
        "--methods",
        default="auto",
        help="Comma-separated list, 'all', 'auto', or 'all_no_popari'.",
    )
    parser.add_argument("--reps", type=int, default=10, help="Number of replicates.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached datasets (data stage) and/or method results (fit stage).",
    )
    parser.add_argument(
        "--ignore-data-params-check",
        action="store_true",
        help="Load dataset caches even when DEFAULT_PARAMS differ from when they were saved.",
    )
    args = parser.parse_args()

    set_paper_style()
    params = dict(DEFAULT_PARAMS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_REP_ROOT.mkdir(parents=True, exist_ok=True)

    n_reps = args.reps
    selected_methods = _parse_methods(args.methods)
    target_methods = list(METHOD_RUNNERS.keys())
    ignore_pc = args.ignore_data_params_check

    # ------------------------------------------------------------------
    # Data stage: simulate once per rep and persist (shared by all methods)
    # ------------------------------------------------------------------
    if args.stage in {"data", "all"}:
        for rep in range(n_reps):
            out_ds = _dataset_path(rep)
            if out_ds.exists() and not args.overwrite:
                print(
                    f"[info] Dataset exists for rep={rep} ({out_ds}); "
                    "use --overwrite to regenerate.",
                    flush=True,
                )
                continue
            print(f"[data] generating rep={rep} ...", flush=True)
            data_rep = DataGenCounts(**params, r=rep)
            save_dataset_cache(rep, data_rep, params)
            print(f"[data] saved rep={rep}.", flush=True)

    if args.stage == "data":
        return

    # ------------------------------------------------------------------
    # Fit stage: load cached data, one result file per (method, replicate)
    # ------------------------------------------------------------------
    if args.stage in {"fit", "all"}:
        for rep in range(n_reps):
            try:
                data_rep = load_dataset_cache(rep, params, ignore_params_check=ignore_pc)
            except FileNotFoundError as exc:
                print(f"[error] {exc}", flush=True)
                raise SystemExit(1) from exc
            except ValueError as exc:
                print(f"[error] {exc}", flush=True)
                raise SystemExit(1) from exc
            for method in selected_methods:
                runner = METHOD_RUNNERS.get(method)
                if runner is None:
                    print(f"[warn] Unknown method '{method}', skipping.")
                    continue
                out_path = _result_path(method, rep)
                if out_path.exists() and not args.overwrite:
                    print(f"[info] Result exists for {method} rep={rep}; use --overwrite to recompute.")
                    continue
                try:
                    print(f"[fit] {method} rep={rep} ...", flush=True)
                    t0 = time.perf_counter()
                    res = runner(data_rep, params, rep)
                    wall_s = time.perf_counter() - t0
                except Exception as exc:
                    print(f"{method} rep={rep} skipped: {type(exc).__name__}: {exc}")
                    continue
                meta = dict(res.metadata) if res.metadata else {}
                meta["runtime"] = float(wall_s)
                res.metadata = meta
                save_method_result(str(out_path), res, params=params, seed=rep)
                print(
                    f"[fit] {method} rep={rep} done in {wall_s:.2f}s.",
                    flush=True,
                )

    if args.stage == "fit":
        return

    # ------------------------------------------------------------------
    # Load all cached replicate results
    # ------------------------------------------------------------------
    all_reps = _load_all_reps(target_methods, n_reps)
    if not all_reps:
        raise RuntimeError("No method results found. Run with --stage fit first.")

    # ------------------------------------------------------------------
    # Multi-replicate evaluation metrics
    # ------------------------------------------------------------------
    records = []
    for rep in range(n_reps):
        methods_this_rep = {m: reps[rep] for m, reps in all_reps.items() if rep in reps}
        if not methods_this_rep:
            continue
        try:
            data_rep = load_dataset_cache(rep, params, ignore_params_check=ignore_pc)
        except FileNotFoundError as exc:
            print(f"[warn] Metrics rep={rep}: {exc}", flush=True)
            continue
        except ValueError as exc:
            print(f"[warn] Metrics rep={rep}: {exc}", flush=True)
            continue
        for name, res in methods_this_rep.items():
            true_data = dict(true_quantities_from_datagen(data_rep))
            true_data["H_list"] = data_rep["H_list"]
            try:
                metrics = summarize_method_performance(
                    true_data,
                    res,
                    tau=0.9,
                    align_sparsify_topk=params["k"],
                    align_sparsify_frac=None,
                )
                metrics["method"] = name
                metrics["rep"] = rep
                records.append(metrics)
            except Exception as exc:
                print(f"[warn] Metrics for {name} rep={rep} failed: {exc}")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(FIG_DIR / "recovery_metrics_multirep.csv", index=False)
        plot_recovery_metrics_boxplot(
            df,
            FIG_DIR / "recovery_metrics_boxplot.png",
            panel_order=PANEL_ORDER_MAIN,
            metric_cols=DEFAULT_RECOVERY_MAIN_METRICS,
        )
        plot_recovery_metrics_supp_boxplot(
            df,
            FIG_DIR / "recovery_metrics_supp1_boxplot.png",
            panel_order=PANEL_ORDER_MAIN,
            metric_cols=DEFAULT_RECOVERY_SUPP1_METRICS,
        )
        plot_recovery_metrics_supp_boxplot(
            df,
            FIG_DIR / "recovery_metrics_supp2_boxplot.png",
            panel_order=PANEL_ORDER_MAIN,
            metric_cols=DEFAULT_RECOVERY_SUPP2_METRICS,
        )

    # ------------------------------------------------------------------
    # Runtime: wall time per method × rep (from saved MethodResult metadata)
    # ------------------------------------------------------------------
    runtime_rows = []
    for method, reps in all_reps.items():
        for rep, res in sorted(reps.items()):
            meta = res.metadata or {}
            rt = meta.get("runtime")
            if rt is None:
                continue
            try:
                rt_f = float(rt)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(rt_f):
                continue
            runtime_rows.append({"method": method, "rep": rep, "runtime": rt_f})

    if runtime_rows:
        df_rt = pd.DataFrame(runtime_rows)
        df_rt.to_csv(FIG_DIR / "method_runtime_multirep.csv", index=False)
        plot_method_runtime_boxplot(
            df_rt,
            FIG_DIR / "method_runtime_boxplot.png",
            panel_order=PANEL_ORDER_MAIN,
        )

    # ------------------------------------------------------------------
    # SpatialPCA-nz vs PCA comparison (separate folder)
    # ------------------------------------------------------------------
    NZ_DIR = FIG_DIR / "SpatialPCA-nz"
    NZ_DIR.mkdir(parents=True, exist_ok=True)
    nz_methods = {"PCA", "SpatialPCA-nz"}
    if nz_methods.issubset(all_reps.keys()):
        plot_spatialpca_nz_comparison(
            all_reps, n_reps, params,
            load_dataset_fn=lambda r: load_dataset_cache(r, params, ignore_params_check=ignore_pc),
            save_dir=NZ_DIR,
        )

    # ------------------------------------------------------------------
    # Per-replicate plots (one sub-folder per rep; matches simu_gaussian_logscale.py)
    # ------------------------------------------------------------------
    rep_indices = set()
    for reps in all_reps.values():
        rep_indices.update(reps.keys())

    print(
        f"Generating per-replicate figures for {len(rep_indices)} rep(s) under "
        f"{SINGLE_REP_ROOT} ...",
        flush=True,
    )
    for rep_idx in sorted(rep_indices):
        methods_this_rep = {
            m: reps[rep_idx] for m, reps in all_reps.items() if rep_idx in reps
        }
        if not methods_this_rep:
            continue
        if "stGP" not in methods_this_rep:
            print(
                f"[warn] stGP missing for rep={rep_idx}; skipping per-rep figures.",
                flush=True,
            )
            continue
        try:
            data_rep = load_dataset_cache(rep_idx, params, ignore_params_check=ignore_pc)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] Per-rep plots rep={rep_idx}: {exc}", flush=True)
            continue
        rep_fig_dir = SINGLE_REP_ROOT / f"rep_{rep_idx}"
        try:
            _plot_single_rep(rep_idx, data_rep, methods_this_rep, params, rep_fig_dir)
        except Exception as exc:
            print(
                f"[warn] Per-rep plots for rep={rep_idx} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
