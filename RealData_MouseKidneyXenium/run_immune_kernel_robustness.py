#!/usr/bin/env python
"""Kernel robustness checks for immune stGP programs in left/right kidneys.

This script mirrors the main analysis in ``immune_L.ipynb`` and
``immune_R.ipynb``:

* immune cells only
* injury_time_days > 0
* one side at a time (L/R)
* scaled expression
* fixed p=3 stGP fit with k=15 and random_state=0

It then sweeps temporal AR(1) rho and spatial bandwidth-selection rho in two
one-at-a-time robustness panels:

* temporal rho in 0.50..0.85, with spatial rho fixed at 0.50
* spatial rho in 0.30..0.70, with temporal rho fixed at 0.70

Each unique setting is written to:

    Results/stgp/Robustness/temporal_rho_0p70_spatial_rho_0p50/
        Immune_L/
        Immune_R/

Run from the stGP conda environment, for example:

    conda activate stGP
    cd /home/byual/stGP-0529/RealData_MouseKidneyXenium
    python run_immune_kernel_robustness.py --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from plots import plot_W_program_heatmap, set_nature_style  # noqa: E402
from stgp.estimation import fit_pfactor  # noqa: E402
from stgp.kernels import (  # noqa: E402
    bandwidth_select_spatial,
    bandwidth_select_temporal,
    build_K_age,
    build_K_spa_list_from_stacked,
)
from stgp.preprocessing import standardize_coords_list  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)
set_nature_style()


TEMPORAL_RHOS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SPATIAL_RHOS = (0.20, 0.30, 0.40, 0.50, 0.60)
BASELINE_TEMPORAL_RHO = 0.70
BASELINE_SPATIAL_RHO = 0.50


@dataclass(frozen=True)
class KernelSetting:
    temporal_rho: float
    spatial_rho: float
    panel: str

    @property
    def setting_id(self) -> str:
        return (
            f"temporal_rho_{format_rho(self.temporal_rho)}_"
            f"spatial_rho_{format_rho(self.spatial_rho)}"
        )


@dataclass
class PreparedSide:
    side: str
    label: str
    adata_immune: sc.AnnData
    y_list: list[np.ndarray]
    nlist: np.ndarray
    ages: np.ndarray
    slices: np.ndarray
    idx_sorted: list[np.ndarray]
    coords_list: list[np.ndarray]


def format_rho(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def build_settings(which: str) -> list[KernelSetting]:
    settings: list[KernelSetting] = []
    if which in {"all", "temporal"}:
        settings.extend(
            KernelSetting(rho, BASELINE_SPATIAL_RHO, "temporal")
            for rho in TEMPORAL_RHOS
        )
    if which in {"all", "spatial"}:
        settings.extend(
            KernelSetting(BASELINE_TEMPORAL_RHO, rho, "spatial")
            for rho in SPATIAL_RHOS
        )

    # Remove duplicate baseline while preserving order.
    unique: dict[tuple[float, float], KernelSetting] = {}
    ordered: list[KernelSetting] = []
    for setting in settings:
        key = (setting.temporal_rho, setting.spatial_rho)
        if key not in unique:
            unique[key] = setting
            ordered.append(setting)
    return ordered


def prepare_side(data_proc: Path, side: str) -> PreparedSide:
    label = f"Immune_{side}"
    adata_immune = sc.read_h5ad(data_proc / "Immune.h5ad")
    adata_immune.obs["side"] = adata_immune.obs["ident"].astype(str).str[-1]
    adata_immune = adata_immune[adata_immune.obs["injury_time_days"] > 0].copy()
    adata_immune = adata_immune[adata_immune.obs["side"] == side].copy()
    adata_immune.obs["age"] = adata_immune.obs["injury_time_days"].copy()

    age_arr = adata_immune.obs["injury_time_days"]
    groups = adata_immune.obs["ident"].astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_per_group = [np.sort(np.where(inv == t)[0]) for t in range(len(uniq))]

    adata_prep = adata_immune.copy()
    sc.pp.scale(adata_prep)
    y_list = [adata_prep.X[ix] for ix in idx_per_group]
    nlist = np.array([len(ix) for ix in idx_per_group])
    ages = np.array([age_arr.iloc[ix[0]] for ix in idx_per_group])

    sort_ord = np.argsort(ages)
    ages = ages[sort_ord]
    slices = uniq[sort_ord]
    nlist = nlist[sort_ord]
    y_list = [y_list[i] for i in sort_ord]
    idx_sorted = [idx_per_group[i] for i in sort_ord]

    coords_list = standardize_coords_list(
        [adata_immune.obsm["spatial"][ix] for ix in idx_per_group]
    )
    coords_list = [coords_list[i] for i in sort_ord]

    return PreparedSide(
        side=side,
        label=label,
        adata_immune=adata_immune,
        y_list=y_list,
        nlist=nlist,
        ages=ages,
        slices=slices,
        idx_sorted=idx_sorted,
        coords_list=coords_list,
    )


def metadata_matches(path: Path, setting: KernelSetting, side: str) -> bool:
    if not path.exists():
        return False
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        meta.get("side") == side
        and np.isclose(float(meta.get("temporal_rho")), setting.temporal_rho)
        and np.isclose(float(meta.get("spatial_rho")), setting.spatial_rho)
        and meta.get("fit_method") == "fit_pfactor"
        and int(meta.get("p")) == 3
        and int(meta.get("k")) == 15
    )


def fit_or_load_result(
    prepared: PreparedSide,
    setting: KernelSetting,
    side_dir: Path,
    *,
    resume: bool,
    verbose: int,
) -> dict:
    result_path = side_dir / "stgp_result.pkl"
    metadata_path = side_dir / "metadata.json"
    if resume and result_path.exists() and metadata_matches(metadata_path, setting, prepared.side):
        print(f"[resume] Loading {result_path}")
        with result_path.open("rb") as f:
            return pickle.load(f)

    print(
        f"[fit] {prepared.label} | temporal rho={setting.temporal_rho:.2f} | "
        f"spatial rho={setting.spatial_rho:.2f}"
    )
    t0 = time.perf_counter()

    gamma_spa = bandwidth_select_spatial(
        prepared.coords_list, frac=0.01, rho=setting.spatial_rho
    )
    # Kept for metadata parity with the notebooks. For AR(1), build_K_age uses
    # temporal_rho and ignores gamma_age.
    gamma_age = bandwidth_select_temporal(prepared.ages, rho=np.exp(-1.5))
    k_age = build_K_age(
        prepared.ages,
        gamma_age,
        kernel="ar1",
        rho=setting.temporal_rho,
        standardize=True,
    )
    k_spa_list = build_K_spa_list_from_stacked(
        np.vstack(prepared.coords_list),
        prepared.nlist,
        gamma_spa,
        standardize=False,
        jitter=1e-6,
    )

    res = fit_pfactor(
        Y_list=prepared.y_list,
        Nlist=prepared.nlist,
        K_age=k_age,
        Kspa_list=k_spa_list,
        p=3,
        k=15,
        inner_rank1_tol=1e-4,
        random_state=0,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    res["gamma_age"] = gamma_age
    res["gamma_spa"] = gamma_spa
    res["temporal_rho"] = setting.temporal_rho
    res["spatial_rho"] = setting.spatial_rho

    with result_path.open("wb") as f:
        pickle.dump(res, f)

    metadata = {
        "side": prepared.side,
        "label": prepared.label,
        "setting_id": setting.setting_id,
        "panel": setting.panel,
        "temporal_kernel": "ar1",
        "temporal_rho": setting.temporal_rho,
        "spatial_rho": setting.spatial_rho,
        "spatial_frac": 0.01,
        "gamma_age": float(gamma_age),
        "gamma_spa": float(gamma_spa),
        "fit_method": "fit_pfactor",
        "p": 3,
        "k": 15,
        "random_state": 0,
        "runtime_seconds": elapsed,
        "n_cells": int(prepared.adata_immune.n_obs),
        "n_genes": int(prepared.adata_immune.n_vars),
        "slices": prepared.slices.tolist(),
        "ages": prepared.ages.astype(float).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[fit] Saved {result_path} ({elapsed / 60:.1f} min)")

    del k_spa_list, k_age
    gc.collect()
    return res


def attach_scores(prepared: PreparedSide, res: dict) -> sc.AnnData:
    adata = prepared.adata_immune.copy()
    all_idx = np.concatenate(prepared.idx_sorted)

    h_arr = np.empty_like(res["H"])
    b_arr = np.empty_like(res["b"])
    h_arr[all_idx] = res["H"]
    b_arr[all_idx] = res["b"]

    adata.obsm["X_stgp"] = h_arr.astype(np.float32)
    adata.obsm["X_stgp_spatial"] = b_arr.astype(np.float32)
    adata.uns["stgp"] = {
        "groups": prepared.slices.tolist(),
        "ages": prepared.ages.astype(float).tolist(),
        "gamma_age": float(res["gamma_age"]),
        "gamma_spa": float(res["gamma_spa"]),
        "temporal_rho": float(res["temporal_rho"]),
        "spatial_rho": float(res["spatial_rho"]),
        "p_selected": int(res["W"].shape[0]),
        "alpha": np.asarray(res["alpha"]).tolist(),
        "alpha_lower": np.asarray(res["alpha_lower"]).tolist(),
        "alpha_upper": np.asarray(res["alpha_upper"]).tolist(),
        "theta": np.asarray(res["theta"]).tolist(),
        "sigma2e": float(res.get("sigma2e", np.nan)),
    }
    return adata


def save_core_tables(
    prepared: PreparedSide,
    adata: sc.AnnData,
    res: dict,
    side_dir: Path,
    *,
    save_adata: bool,
) -> pd.DataFrame:
    p_sel = int(res["W"].shape[0])
    w_df = pd.DataFrame(
        res["W"],
        index=[f"stGP{j + 1}" for j in range(p_sel)],
        columns=adata.var_names.astype(str),
    )
    w_df.to_csv(side_dir / "W.csv")

    scores = pd.DataFrame(
        adata.obsm["X_stgp_spatial"],
        index=adata.obs_names,
        columns=[f"stGP{j + 1}" for j in range(p_sel)],
    )
    scores.to_csv(side_dir / "spatial_b_scores.csv.gz")

    if save_adata:
        adata.write_h5ad(str(side_dir / "adata_with_scores.h5ad"), compression="gzip")

    top_rows = []
    for prog, row in w_df.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(25)
        for rank, (gene, weight) in enumerate(top.items(), start=1):
            top_rows.append(
                {"program": prog, "rank": rank, "gene": gene, "weight": float(weight)}
            )
    pd.DataFrame(top_rows).to_csv(side_dir / "top_genes_per_program.csv", index=False)
    return w_df


def plot_alpha_panel(stgp_info: dict, out: Path, title: str) -> None:
    ages = np.asarray(stgp_info["ages"], dtype=float)
    alpha = np.asarray(stgp_info["alpha"], dtype=float)
    alpha_lower = np.asarray(stgp_info["alpha_lower"], dtype=float)
    alpha_upper = np.asarray(stgp_info["alpha_upper"], dtype=float)
    p_sel = alpha.shape[0]
    order = np.argsort(ages)
    t_idx = np.arange(len(order))

    fig, axes = plt.subplots(
        1, p_sel, figsize=(3.2 * p_sel, 3.1), sharex=True, constrained_layout=True
    )
    color = "#2C7FB8"
    for j, ax in enumerate(np.atleast_1d(axes)):
        a = alpha[j][order]
        lo = alpha_lower[j][order]
        hi = alpha_upper[j][order]
        ax.fill_between(t_idx, lo, hi, alpha=0.18, color=color)
        ax.plot(t_idx, lo, lw=0.8, ls="--", color=color, alpha=0.55)
        ax.plot(t_idx, hi, lw=0.8, ls="--", color=color, alpha=0.55)
        ax.plot(t_idx, a, lw=1.8, color=color)
        ax.scatter(t_idx, a, s=28, color=color, zorder=3, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="0.75", lw=0.7, ls=":")
        ax.set_title(f"stGP{j + 1}", fontsize=11)
        ax.set_xticks(t_idx)
        ax.set_xticklabels([f"{x:g}" for x in ages[order]], rotation=35, ha="right")
        ax.set_xlabel("Injury time (days)")
        if j == 0:
            ax.set_ylabel("alpha")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_theta(stgp_info: dict, out: Path, title: str) -> None:
    theta = np.asarray(stgp_info["theta"], dtype=float)
    p_sel = theta.shape[0]
    prog_names = [f"stGP{j + 1}" for j in range(p_sel)]
    colors = plt.cm.tab10.colors[:p_sel]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
    for j, (color, name) in enumerate(zip(colors, prog_names)):
        axes[0].bar(j, theta[j, 0], color=color, edgecolor="white", linewidth=0.6)
        axes[1].bar(j, theta[j, 1], color=color, edgecolor="white", linewidth=0.6)
    for ax in axes:
        ax.set_xticks(range(p_sel))
        ax.set_xticklabels(prog_names, rotation=30, ha="right")
        ax.set_xlabel("Program")
    axes[0].set_ylabel("Temporal amplitude")
    axes[0].set_title("Temporal amplitude")
    axes[1].set_ylabel("Spatial noise fraction")
    axes[1].set_title("Spatial noise fraction")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_b_panel(adata: sc.AnnData, out: Path, title: str) -> None:
    stgp_info = adata.uns["stgp"]
    slices = np.asarray(stgp_info["groups"]).astype(str)
    ages = np.asarray(stgp_info["ages"], dtype=float)
    order = np.argsort(ages)
    slices = slices[order]
    ages = ages[order]

    b = np.asarray(adata.obsm["X_stgp_spatial"], dtype=float)
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    p_sel = b.shape[1]
    n_slices = len(slices)
    fig, axes = plt.subplots(
        p_sel,
        n_slices,
        figsize=(2.8 * n_slices, 2.7 * p_sel),
        squeeze=False,
        constrained_layout=True,
    )

    for j in range(p_sel):
        vmax = float(np.nanpercentile(np.abs(b[:, j]), 99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        for c, (slice_name, age) in enumerate(zip(slices, ages)):
            ax = axes[j, c]
            mask = adata.obs["ident"].astype(str).to_numpy() == slice_name
            sc_ref = ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=b[mask, j],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                s=5,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{slice_name}\n{age:g} days", fontsize=9)
            if c == 0:
                ax.text(
                    -0.02,
                    0.5,
                    f"stGP{j + 1}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                )
            if c == n_slices - 1:
                cbar = fig.colorbar(sc_ref, ax=ax, fraction=0.045, pad=0.01)
                cbar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def clean_term(term: str) -> str:
    s = str(term)
    s = re.sub(r"^(GOBP|GOCC|GOMF|HALLMARK|REACTOME|KEGG|WP)_", "", s)
    s = re.sub(r"_(UP|DN|DOWN)$", "", s)
    s = s.replace("_", " ").strip().lower().capitalize()
    for pattern, repl in (
        (r"\bdna\b", "DNA"),
        (r"\brna\b", "RNA"),
        (r"\bmrna\b", "mRNA"),
        (r"\bt cell\b", "T cell"),
        (r"\bb cell\b", "B cell"),
        (r"\bnk cell\b", "NK cell"),
        (r"\bmhc\b", "MHC"),
        (r"\btnf\b", "TNF"),
        (r"\bifn\b", "IFN"),
    ):
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def plot_go_panel(results_by_set: dict[str, pd.DataFrame], out: Path, title: str) -> None:
    n_panels = len(results_by_set)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(6.2, 2.35 * n_panels), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")

    for ax, (set_name, res) in zip(axes, results_by_set.items()):
        ax.set_title(set_name, loc="left", fontsize=10, weight="bold")
        if res.empty or "Adjusted P-value" not in res or "Combined Score" not in res:
            ax.text(0.5, 0.5, "No enrichment result", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue

        df = res.copy()
        df = df.dropna(subset=["Adjusted P-value", "Combined Score"])
        df = df[(df["Adjusted P-value"] > 0) & (df["Combined Score"] > 0)]
        df = df.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False])
        df = df.head(8).iloc[::-1]
        if df.empty:
            ax.text(0.5, 0.5, "No significant terms", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue

        scores = df["Combined Score"].astype(float).to_numpy()
        padj = df["Adjusted P-value"].astype(float).clip(lower=1e-300)
        neglog = -np.log10(padj).to_numpy(dtype=float)
        finite = np.isfinite(scores) & np.isfinite(neglog)
        scores = scores[finite]
        neglog = neglog[finite]
        df = df.loc[finite].copy()
        if df.empty:
            ax.text(0.5, 0.5, "No finite enrichment scores", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue

        vmin = float(np.nanmin(neglog))
        vmax = float(np.nanmax(neglog))
        if vmax <= vmin:
            pad = max(abs(vmin) * 0.05, 1e-6)
            vmin -= pad
            vmax += pad
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(neglog))
        labels = [clean_term(x)[:75] for x in df["Term"]]
        y = np.arange(len(df))
        ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Combined score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def run_go_analysis(w_df: pd.DataFrame, side_dir: Path) -> None:
    try:
        import gseapy as gp
    except ImportError as exc:
        (side_dir / "GO_IMPORT_ERROR.txt").write_text(
            "gseapy is not available in this Python environment.\n"
            f"{exc}\n"
        )
        return

    gene_sets = {
        "GO Biological Process": SCRIPT_DIR / "data/genesets/m5.go.bp.v2026.1.Mm.symbols.gmt",
        "GO Molecular Function": SCRIPT_DIR / "data/genesets/m5.go.mf.v2026.1.Mm.symbols.gmt",
        "GO Cellular Component": SCRIPT_DIR / "data/genesets/m5.go.cc.v2026.1.Mm.symbols.gmt",
    }
    missing = [str(path) for path in gene_sets.values() if not path.exists()]
    if missing:
        (side_dir / "GO_MISSING_GMT.txt").write_text("\n".join(missing) + "\n")
        return

    background_genes = list(w_df.columns)
    all_results: list[pd.DataFrame] = []
    for program in w_df.index:
        weights = w_df.loc[program].astype(float)
        gene_list = weights[weights > 0].sort_values(ascending=False).index.tolist()
        results_by_set: dict[str, pd.DataFrame] = {}
        for set_name, gmt_path in gene_sets.items():
            try:
                enr = gp.enrich(
                    gene_list=gene_list,
                    gene_sets=str(gmt_path),
                    background=background_genes,
                    verbose=False,
                )
                res = enr.res2d.copy()
            except Exception as exc:  # gseapy can raise on no-overlap programs.
                res = pd.DataFrame(
                    {
                        "Term": [f"ERROR: {type(exc).__name__}"],
                        "Adjusted P-value": [np.nan],
                        "Combined Score": [np.nan],
                    }
                )
            res["program"] = program
            res["gene_set"] = set_name
            res["n_input_genes"] = len(gene_list)
            res["input_genes"] = ",".join(gene_list)
            if "Term" in res:
                res["Term_clean"] = res["Term"].map(clean_term)
            results_by_set[set_name] = res
            all_results.append(res)

        plot_go_panel(
            results_by_set,
            side_dir / f"GO_{program}.png",
            title=f"GO enrichment: {program}",
        )

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            side_dir / "GO_enrichment_all_programs.csv", index=False
        )


def save_figures_and_go(prepared: PreparedSide, adata: sc.AnnData, w_df: pd.DataFrame, side_dir: Path) -> None:
    stgp_info = adata.uns["stgp"]
    title_suffix = (
        f"{prepared.label}; temporal rho={stgp_info['temporal_rho']:.2f}, "
        f"spatial rho={stgp_info['spatial_rho']:.2f}"
    )

    fig = plot_W_program_heatmap(
        w_df,
        title=f"Gene loadings (W) - {title_suffix}",
        out=side_dir / "W_heatmap.png",
        dpi=400,
    )
    plt.close(fig)

    plot_alpha_panel(
        stgp_info,
        side_dir / "alpha_trajectories.png",
        title=f"Age trajectories - {title_suffix}",
    )
    plot_theta(
        stgp_info,
        side_dir / "theta_barplot.png",
        title=f"Variance components - {title_suffix}",
    )
    plot_spatial_b_panel(
        adata,
        side_dir / "spatial_b_all_programs.png",
        title=f"Spatial b fields - {title_suffix}",
    )
    run_go_analysis(w_df, side_dir)


def cosine_similarity_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    av = av / np.maximum(np.linalg.norm(av, axis=1, keepdims=True), 1e-12)
    bv = bv / np.maximum(np.linalg.norm(bv, axis=1, keepdims=True), 1e-12)
    return av @ bv.T


def program_similarity(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr, spearmanr

    common = a.columns.intersection(b.columns)
    a = a.loc[:, common]
    b = b.loc[:, common]
    cosine = cosine_similarity_matrix(a, b)
    row_ind, col_ind = linear_sum_assignment(-cosine)

    rows = []
    for i, j in zip(row_ind, col_ind):
        x = a.iloc[i].to_numpy(dtype=float)
        y = b.iloc[j].to_numpy(dtype=float)
        pearson = pearsonr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
        spearman = spearmanr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
        rows.append(
            {
                "label_a": label_a,
                "program_a": a.index[i],
                "label_b": label_b,
                "program_b": b.index[j],
                "cosine": float(cosine[i, j]),
                "pearson": float(pearson),
                "spearman": float(spearman),
            }
        )
    return pd.DataFrame(rows)


def maybe_save_left_right_similarity(setting_dir: Path) -> None:
    left_w = setting_dir / "Immune_L" / "W.csv"
    right_w = setting_dir / "Immune_R" / "W.csv"
    if not left_w.exists() or not right_w.exists():
        return
    w_l = pd.read_csv(left_w, index_col=0)
    w_r = pd.read_csv(right_w, index_col=0)
    sim = program_similarity(w_l, w_r, "Immune_L", "Immune_R")
    sim.to_csv(setting_dir / "left_right_program_similarity.csv", index=False)


def save_global_robustness_summary(settings: Iterable[KernelSetting], out_dir: Path) -> None:
    baseline = KernelSetting(BASELINE_TEMPORAL_RHO, BASELINE_SPATIAL_RHO, "baseline")
    rows = []
    for side_label in ("Immune_L", "Immune_R"):
        baseline_w_path = out_dir / baseline.setting_id / side_label / "W.csv"
        if not baseline_w_path.exists():
            continue
        baseline_w = pd.read_csv(baseline_w_path, index_col=0)
        for setting in settings:
            w_path = out_dir / setting.setting_id / side_label / "W.csv"
            if not w_path.exists():
                continue
            w = pd.read_csv(w_path, index_col=0)
            sim = program_similarity(
                baseline_w,
                w,
                f"{side_label}_baseline",
                f"{side_label}_{setting.setting_id}",
            )
            sim["side"] = side_label
            sim["setting_id"] = setting.setting_id
            sim["temporal_rho"] = setting.temporal_rho
            sim["spatial_rho"] = setting.spatial_rho
            rows.append(sim)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            out_dir / "baseline_program_similarity_all_settings.csv", index=False
        )


def validate_inputs(base_dir: Path) -> None:
    required = [
        base_dir / "data/processed/Immune.h5ad",
        base_dir / "data/genesets/m5.go.bp.v2026.1.Mm.symbols.gmt",
        base_dir / "data/genesets/m5.go.mf.v2026.1.Mm.symbols.gmt",
        base_dir / "data/genesets/m5.go.cc.v2026.1.Mm.symbols.gmt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run immune stGP kernel robustness fits for left and right kidneys."
    )
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "Results/stgp/Robustness")
    parser.add_argument("--settings", choices=("all", "temporal", "spatial"), default="all")
    parser.add_argument("--sides", choices=("both", "L", "R"), default="both")
    parser.add_argument("--resume", action="store_true", help="Reuse matching stgp_result.pkl files.")
    parser.add_argument("--save-adata", action="store_true", help="Also save scored h5ad files.")
    parser.add_argument("--max-settings", type=int, default=None, help="Run only the first N settings.")
    parser.add_argument("--verbose-fit", type=int, default=1, help="fit_pfactor verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve()
    data_proc = base_dir / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_inputs(base_dir)
    settings = build_settings(args.settings)
    if args.max_settings is not None:
        settings = settings[: args.max_settings]

    sides = ("L", "R") if args.sides == "both" else (args.sides,)
    planned_fits = len(settings) * len(sides)
    print("Planned kernel settings:")
    for setting in settings:
        print(
            f"  {setting.setting_id} | panel={setting.panel} | "
            f"temporal rho={setting.temporal_rho:.2f} | spatial rho={setting.spatial_rho:.2f}"
        )
    print(f"Planned side-specific fits: {planned_fits}")

    prepared_by_side = {side: prepare_side(data_proc, side) for side in sides}
    (out_dir / "settings_manifest.json").write_text(
        json.dumps([setting.__dict__ | {"setting_id": setting.setting_id} for setting in settings], indent=2)
    )

    for setting in settings:
        setting_dir = out_dir / setting.setting_id
        setting_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {setting.setting_id} ===")
        for side in sides:
            prepared = prepared_by_side[side]
            side_dir = setting_dir / prepared.label
            side_dir.mkdir(parents=True, exist_ok=True)

            res = fit_or_load_result(
                prepared,
                setting,
                side_dir,
                resume=args.resume,
                verbose=args.verbose_fit,
            )
            adata = attach_scores(prepared, res)
            w_df = save_core_tables(
                prepared,
                adata,
                res,
                side_dir,
                save_adata=args.save_adata,
            )
            save_figures_and_go(prepared, adata, w_df, side_dir)
            del adata, res, w_df
            gc.collect()

        maybe_save_left_right_similarity(setting_dir)

    save_global_robustness_summary(settings, out_dir)
    print(f"\nDone. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
