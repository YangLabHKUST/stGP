"""Consolidated benchmark figures for the HumanBrain ext notebook."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from plots import METHOD_COLORS
from utils import best_program_by_correlation, evaluate_cluster_labels, spectral_knn_labels


METHODS = ["stGP", "STAMP", "MEFISTO", "Popari", "SpatialPCA"]
LAYER_SPECS = [("L2/3", "CUX2", 0), ("L4", "RORB", 2), ("L5/6", "HS3ST4", 1)]
LAYER_MARKERS = {layer: gene for layer, gene, _ in LAYER_SPECS}
LAYER_SAFE = {"L2/3": "L2-3", "L4": "L4", "L5/6": "L5-6"}
REPRESENTATIVE_AGE_OCCURRENCES = [(28, 2), (42, 2), (82, 2), (87, 1)]
CELLTYPE2_LAYER_MAP = {"L2/3": "L2/3", "L4": "L4", "L5/6": "L5/6", "L5/6-CC": "L5/6"}
PANEL_TITLE_FONTSIZE = 18

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

RAW_CLUSTER_COLORS = [
    "#4DBBD5",
    "#F39B7F",
    "#00A087",
    "#8491B4",
    "#E64B35",
    "#3C5488",
    "#91D1C2",
    "#B09C85",
]


def _dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def _safe_name(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def _display_title(value: str) -> str:
    text = str(value)
    if text == "stGP_b":
        return "stGP"
    if text.startswith("stGP_k") and text.removeprefix("stGP_k").isdigit():
        return f"k={text.removeprefix('stGP_k')}"
    return text


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def _paired_effect_summary(ref, other) -> dict[str, float | int]:
    ref = np.asarray(ref, dtype=float)
    other = np.asarray(other, dtype=float)
    valid = np.isfinite(ref) & np.isfinite(other)
    diff = ref[valid] - other[valid]
    return {
        "n_pairs": int(valid.sum()),
        "ref_mean": float(np.nanmean(ref[valid])) if valid.any() else np.nan,
        "other_mean": float(np.nanmean(other[valid])) if valid.any() else np.nan,
        "mean_diff": float(np.nanmean(diff)) if diff.size else np.nan,
        "ref_median": float(np.nanmedian(ref[valid])) if valid.any() else np.nan,
        "other_median": float(np.nanmedian(other[valid])) if valid.any() else np.nan,
        "median_diff": float(np.nanmedian(diff)) if diff.size else np.nan,
        "n_ref_gt": int((diff > 0).sum()),
        "n_ref_lt": int((diff < 0).sum()),
        "n_ref_eq": int((diff == 0).sum()),
    }


def _slice_order(adata, slices=None) -> list[str]:
    ids = adata.obs["id_region"].astype(str).to_numpy()
    if slices is None:
        slices = pd.unique(ids)
    slices = [str(s) for s in slices if str(s) in set(ids)]
    return sorted(slices, key=lambda s: float(adata.obs.loc[ids == s, "age"].iloc[0]))


def _slice_ages(adata, slices: list[str]) -> dict[str, float]:
    ids = adata.obs["id_region"].astype(str).to_numpy()
    return {sid: float(adata.obs.loc[ids == sid, "age"].iloc[0]) for sid in slices}


def _representative_slices_by_age(adata, slices: list[str]) -> list[str]:
    """Pick 28/42/82 second sections and the 87-year section when available."""
    age_by_slice = _slice_ages(adata, slices)
    chosen: list[str] = []
    for target_age, occurrence in REPRESENTATIVE_AGE_OCCURRENCES:
        matches = [sid for sid in slices if int(round(age_by_slice[sid])) == target_age]
        if not matches:
            matches = sorted(slices, key=lambda sid: (abs(age_by_slice[sid] - target_age), slices.index(sid)))[:1]
        pick = matches[min(occurrence - 1, len(matches) - 1)]
        if pick not in chosen:
            chosen.append(pick)
    return chosen


def _default_source_benchmark_dir(benchmark_dir: Path) -> Path:
    if (
        benchmark_dir.name == "benchmark"
        and benchmark_dir.parent.parent.name == "Figure"
    ):
        human_dir = benchmark_dir.parent.parent.parent
        celltype = benchmark_dir.parent.name
        return human_dir / "Results" / "benchmark" / celltype
    return benchmark_dir.with_name(f"{benchmark_dir.name}_source")


def _prepare_dirs(benchmark_dir: Path, source_benchmark_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": benchmark_dir,
        "source_root": source_benchmark_dir,
        "summary": benchmark_dir / "summary",
        "source_summary": source_benchmark_dir / "summary",
        "source": source_benchmark_dir / "summary" / "source_data",
        "clustering": benchmark_dir / "clustering",
        "source_clustering": source_benchmark_dir / "clustering",
    }
    for layer, safe in LAYER_SAFE.items():
        base = benchmark_dir / safe
        dirs[f"{layer}:base"] = base
        dirs[f"{layer}:embedding_full"] = base / "spatial_embedding" / "full"
        dirs[f"{layer}:embedding_rep"] = base / "spatial_embedding" / "representative"
        dirs[f"{layer}:correlation"] = base / "correlation"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _cleanup_stale_outputs(benchmark_dir: Path, source_benchmark_dir: Path, dirs: Mapping[str, Path]) -> None:
    stale_files = [
        dirs["summary"] / "stgp_advantage_heatmap.png",
        dirs["summary"] / "stgp_advantage_heatmap.csv",
        benchmark_dir / "clustering" / "marker_region" / "spatial" / "cluster_overview_all_slices.png",
        benchmark_dir / "clustering" / "celltype2" / "spatial" / "cluster_overview_labeled_slices.png",
    ]
    for layer, gene, _ in LAYER_SPECS:
        stale_files.append(dirs[f"{layer}:embedding_rep"] / f"{gene}.png")
    for stale in stale_files:
        stale.unlink(missing_ok=True)
    for stale in benchmark_dir.rglob("*welch_t_tests.csv"):
        stale.unlink(missing_ok=True)
    for stale in source_benchmark_dir.rglob("*welch_t_tests.csv"):
        stale.unlink(missing_ok=True)

    stale_dirs = [
        benchmark_dir / "clustering" / "marker_region" / "spatial",
        benchmark_dir / "clustering" / "celltype2" / "spatial",
    ]
    for layer, safe in LAYER_SAFE.items():
        stale_dirs.append(benchmark_dir / safe / "marker_expression")
    for stale_dir in stale_dirs:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)


def _continuous_limits(vals: np.ndarray, *, symmetric: bool = False, fixed_vmax: float | None = None) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if fixed_vmax is not None:
        return 0.0, fixed_vmax
    if symmetric or np.nanmin(vals) < 0:
        vmax = float(np.nanpercentile(np.abs(vals), 99))
        return -vmax, vmax
    return float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))


def _add_age_label(ax, sid: str, age_by_slice: Mapping[str, float]) -> None:
    ax.text(
        0.5,
        1.01,
        f"{age_by_slice[sid]:.0f} yr",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
    )


def _despine_metric_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles(labels: list[str], palette: Mapping[str, str]) -> list[Patch]:
    return [Patch(facecolor=palette.get(label, "#888888"), edgecolor="none", label=label) for label in labels]


def _raw_cluster_labels(raw_labels) -> pd.Series:
    vals = pd.Series(raw_labels).astype("string")

    def fmt(value):
        if pd.isna(value):
            return pd.NA
        text = str(value)
        try:
            return f"C{int(float(text)) + 1}"
        except ValueError:
            return f"C{text}"

    return vals.map(fmt)


def _raw_cluster_palette(labels) -> dict[str, str]:
    levels = sorted(pd.Series(labels).dropna().astype(str).unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    return {level: RAW_CLUSTER_COLORS[i % len(RAW_CLUSTER_COLORS)] for i, level in enumerate(levels)}


def _add_categorical_legend(ax, labels: np.ndarray, palette: Mapping[str, str], *, fontsize: float = 6.2) -> None:
    levels = [lab for lab in sorted(pd.unique(pd.Series(labels).dropna().astype(str)), key=str) if lab != "<NA>"]
    ax.axis("off")
    if not levels:
        return
    ax.legend(
        handles=_legend_handles(levels, palette),
        loc="center left",
        frameon=False,
        fontsize=fontsize,
        handlelength=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def _plot_continuous_tiles(
    adata,
    values: np.ndarray,
    slices: list[str],
    out: Path,
    *,
    ncols: int | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
    title: str | None = None,
    dpi: int = 400,
) -> Path:
    xy = np.asarray(adata.obsm["spatial"])
    ids = adata.obs["id_region"].astype(str).to_numpy()
    age_by_slice = _slice_ages(adata, slices)
    ncols = len(slices) if ncols is None else ncols
    nrows = int(np.ceil(len(slices) / ncols))
    if vmin is None or vmax is None:
        vmin, vmax = _continuous_limits(values, symmetric=np.nanmin(values) < 0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols + 0.35, 1.65 * nrows + (0.28 if title else 0.0)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sid in zip(axes.ravel(), slices):
        mask = ids == sid
        sc = ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=np.asarray(values)[mask],
            s=2.2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        _add_age_label(ax, sid, age_by_slice)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01, label=colorbar_label)
    if title:
        fig.suptitle(title, fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", y=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_spatial_grid(
    adata,
    rows: list[dict],
    slices: list[str],
    out: Path,
    *,
    dpi: int = 400,
) -> Path:
    xy = np.asarray(adata.obsm["spatial"])
    ids = adata.obs["id_region"].astype(str).to_numpy()
    age_by_slice = _slice_ages(adata, slices)
    n_rows, n_cols = len(rows), len(slices)
    panel_w, panel_h = 1.25, 1.35
    label_w, cbar_w = 0.95, 0.18
    fig_w = label_w + n_cols * panel_w + cbar_w + 0.25
    fig_h = n_rows * panel_h + 0.25
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [cbar_w / panel_w],
        left=label_w / fig_w,
        right=1.0 - 0.08 / fig_w,
        top=1.0 - 0.12 / fig_h,
        bottom=0.04,
        wspace=0.03,
        hspace=0.08,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_axes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]

    for row_i, row in enumerate(rows):
        values = row["values"]
        kind = row.get("kind", "continuous")
        if kind == "categorical":
            labels = pd.Series(values).astype(str).to_numpy()
            palette = row["palette"]
            for col_j, sid in enumerate(slices):
                ax = axes[row_i][col_j]
                mask = ids == sid
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    c=[palette.get(v, "#888888") for v in labels[mask]],
                    s=1.8,
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_aspect("equal")
                ax.axis("off")
                if row_i == 0:
                    _add_age_label(ax, sid, age_by_slice)
            _add_categorical_legend(cbar_axes[row_i], labels, palette)
        else:
            vals = np.asarray(values, dtype=float)
            cmap = row.get("cmap", "RdBu_r")
            vmin, vmax = row.get("vmin"), row.get("vmax")
            if vmin is None or vmax is None:
                vmin, vmax = _continuous_limits(vals, symmetric=np.nanmin(vals) < 0, fixed_vmax=row.get("fixed_vmax"))
            for col_j, sid in enumerate(slices):
                ax = axes[row_i][col_j]
                mask = ids == sid
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    c=vals[mask],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=1.8,
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_aspect("equal")
                ax.axis("off")
                if row_i == 0:
                    _add_age_label(ax, sid, age_by_slice)
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            cb = fig.colorbar(sm, cax=cbar_axes[row_i])
            cb.ax.tick_params(labelsize=6)
            cb.set_label(row.get("colorbar_label", "score"), fontsize=6.5, labelpad=2)

        first_pos = axes[row_i][0].get_position()
        fig.text(
            label_w / fig_w - 0.01,
            (first_pos.y0 + first_pos.y1) / 2,
            row["label"],
            ha="right",
            va="center",
            fontsize=8,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_representative_rows(
    adata,
    rows: list[dict],
    slices: list[str],
    out_dir: Path,
    *,
    dpi: int = 400,
) -> list[Path]:
    out_paths: list[Path] = []
    for row in rows:
        label = _safe_name(row["label"])
        title = _display_title(str(row["label"]))
        out = out_dir / f"{label}.png"
        if row.get("kind") == "categorical":
            out_paths.append(_plot_categorical_tiles(adata, row["values"], slices, out, palette=row["palette"], ncols=2, title=title, dpi=dpi))
        else:
            vals = np.asarray(row["values"], dtype=float)
            vmin, vmax = row.get("vmin"), row.get("vmax")
            if vmin is None or vmax is None:
                vmin, vmax = _continuous_limits(vals, symmetric=np.nanmin(vals) < 0, fixed_vmax=row.get("fixed_vmax"))
            out_paths.append(
                _plot_continuous_tiles(
                    adata,
                    vals,
                    slices,
                    out,
                    ncols=2,
                    cmap=row.get("cmap", "RdBu_r"),
                    vmin=vmin,
                    vmax=vmax,
                    colorbar_label=row.get("colorbar_label", ""),
                    title=title,
                    dpi=dpi,
                )
            )
    return out_paths


def _plot_categorical_tiles(
    adata,
    values,
    slices: list[str],
    out: Path,
    *,
    palette: Mapping[str, str],
    ncols: int | None = None,
    show_legend: bool = True,
    title: str | None = None,
    dpi: int = 400,
) -> Path:
    xy = np.asarray(adata.obsm["spatial"])
    ids = adata.obs["id_region"].astype(str).to_numpy()
    labels = pd.Series(values).astype(str).to_numpy()
    age_by_slice = _slice_ages(adata, slices)
    ncols = len(slices) if ncols is None else ncols
    nrows = int(np.ceil(len(slices) / ncols))
    fig_w = 1.6 * ncols + (0.9 if show_legend else 0.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 1.65 * nrows + (0.28 if title else 0.0)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sid in zip(axes.ravel(), slices):
        mask = ids == sid
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=[palette.get(v, "#888888") for v in labels[mask]],
            s=2.2,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        _add_age_label(ax, sid, age_by_slice)
    if show_legend:
        fig.legend(
            handles=_legend_handles([lab for lab in sorted(pd.unique(labels), key=str) if lab != "<NA>"], palette),
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            frameon=False,
            fontsize=7,
            handlelength=1.0,
            handletextpad=0.4,
        )
    if title:
        fig.suptitle(title, fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", y=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _boxplot_metric(
    metrics_df: pd.DataFrame,
    metric: str,
    y_label: str,
    out: Path,
    test_out: Path,
    *,
    methods: list[str],
    dpi: int = 400,
) -> pd.DataFrame:
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
    data = [metrics_df.loc[metrics_df["method"] == method, metric].dropna().to_numpy() for method in methods]
    colors = [METHOD_COLORS.get(method, "#999999") for method in methods]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        flierprops=dict(marker="o", markersize=3.2, alpha=0.55, markeredgewidth=0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    tests = []
    ref = metrics_df.loc[metrics_df["method"] == "stGP", ["id_region", metric]].rename(columns={metric: "ref"})
    raw_p = []
    comparisons = []
    for method in methods:
        if method == "stGP":
            continue
        other = metrics_df.loc[metrics_df["method"] == method, ["id_region", metric]].rename(columns={metric: "other"})
        paired = ref.merge(other, on="id_region").dropna()
        diff = paired["ref"].to_numpy(dtype=float) - paired["other"].to_numpy(dtype=float)
        if len(diff) < 2:
            p = np.nan
        elif np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(paired["ref"], paired["other"], alternative="greater").pvalue)
        raw_p.append(p)
        comparisons.append((method, p, _paired_effect_summary(paired["ref"], paired["other"])))
    raw_p = np.asarray(raw_p, dtype=float)
    adj = np.full_like(raw_p, np.nan)
    valid = np.isfinite(raw_p)
    if valid.any():
        _, adj[valid], _, _ = multipletests(raw_p[valid], method="holm")

    vals = np.concatenate([d for d in data if len(d) > 0])
    y_min, y_max = float(np.nanmin(vals)), float(np.nanmax(vals))
    y_range = y_max - y_min if y_max > y_min else 1.0
    y0 = y_max + 0.08 * y_range
    dy = 0.035 * y_range
    step = 0.13 * y_range
    for i, ((method, p, effect), p_adj) in enumerate(zip(comparisons, adj)):
        x1, x2 = 1, methods.index(method) + 1
        y = y0 + i * step
        ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=0.9, color="black", clip_on=False)
        stars = _stars(p_adj)
        ax.text((x1 + x2) / 2, y + dy, stars, ha="center", va="bottom", fontsize=8)
        tests.append(
            {
                "metric": metric,
                "ref_method": "stGP",
                "other_method": method,
                "test": "one-sided paired Wilcoxon signed-rank test",
                "alternative": "greater",
                "raw_p": p,
                "holm_adj_p": p_adj,
                "stars": stars,
                **effect,
            }
        )

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_ylim(top=y0 + len(comparisons) * step + 0.08 * y_range)
    _despine_metric_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    test_df = pd.DataFrame(tests)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_out, index=False)
    return test_df


def _method_labels_for_slice(method: str, adata, idx: np.ndarray, n_clusters: int, specs: Mapping[str, Mapping[str, str]]) -> np.ndarray:
    X = np.asarray(adata.obsm[specs[method]["obsm_key"]][idx])
    if specs[method].get("kind") == "argmax":
        return np.argmax(X, axis=1).astype(str)
    return spectral_knn_labels(X, n_clusters=n_clusters)


def _evaluate_ground_truth(
    *,
    adatas: Mapping[str, object],
    gt_labels: pd.Series,
    slices: list[str],
    out_dir: Path,
    specs: Mapping[str, Mapping[str, str]],
    methods: list[str],
    n_clusters: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series]]:
    obs_names = gt_labels.index
    ref_adata = adatas["stGP"]
    slice_ids = ref_adata.obs.loc[obs_names, "id_region"].astype(str).to_numpy()
    ages = pd.to_numeric(ref_adata.obs.loc[obs_names, "age"], errors="coerce").to_numpy(float)
    records, mapping_records, pred_tables, method_preds, method_merged_preds = [], [], [], {}, {}

    for method in methods:
        ad = adatas[method]
        method_pred = pd.Series(index=obs_names, dtype=object)
        method_merged = pd.Series(index=obs_names, dtype=object)
        for sid in slices:
            idx = np.flatnonzero(slice_ids == sid)
            if len(idx) <= n_clusters:
                continue
            raw_pred = _method_labels_for_slice(method, ad, idx, n_clusters, specs)
            method_pred.iloc[idx] = raw_pred
            y_true = gt_labels.iloc[idx].astype(str).to_numpy()
            metrics, merged_pred, mapping = evaluate_cluster_labels(y_true, raw_pred)
            metrics["raw_acc"] = metrics.pop("raw_hungarian_acc")
            method_merged.iloc[idx] = merged_pred
            records.append(
                {
                    "method": method,
                    "id_region": sid,
                    "age": float(ref_adata.obs.loc[obs_names[idx], "age"].iloc[0]),
                    "n_eval_cells": int(len(idx)),
                    **metrics,
                }
            )
            for raw_label, merged_label in mapping.items():
                mapping_records.append(
                    {
                        "method": method,
                        "id_region": sid,
                        "raw_label": raw_label,
                        "merged_label": merged_label,
                    }
                )
        method_preds[method] = method_pred
        method_merged_preds[method] = method_merged
        pred_tables.append(
            pd.DataFrame(
                {
                    "cell": obs_names,
                    "method": method,
                    "id_region": slice_ids,
                    "age": ages,
                    "raw_pred": method_pred.to_numpy(),
                    "merged_pred": method_merged.to_numpy(),
                    "ground_truth": gt_labels.to_numpy(),
                }
            )
        )

    metrics_df = pd.DataFrame(records)
    mapping_df = pd.DataFrame(mapping_records)
    pred_df = pd.concat(pred_tables, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "slice_metrics.csv", index=False)
    mapping_df.to_csv(out_dir / "cluster_to_layer_mapping.csv", index=False)
    pred_df.to_csv(out_dir / "cell_predictions.csv", index=False)
    (
        metrics_df.groupby("method", as_index=False)
        .agg(
            raw_ari_mean=("raw_ari", "mean"),
            raw_nmi_mean=("raw_nmi", "mean"),
            raw_acc_mean=("raw_acc", "mean"),
            merged_acc_mean=("merged_acc", "mean"),
            merged_macro_f1_mean=("merged_macro_f1", "mean"),
        )
        .to_csv(out_dir / "summary.csv", index=False)
    )
    return metrics_df, mapping_df, pred_df, method_preds, method_merged_preds


def _predict_method_clusters(
    *,
    adatas: Mapping[str, object],
    obs_names,
    slices: list[str],
    specs: Mapping[str, Mapping[str, str]],
    methods: list[str],
    n_clusters: int = 3,
) -> dict[str, pd.Series]:
    ref_adata = adatas["stGP"]
    slice_ids = ref_adata.obs.loc[obs_names, "id_region"].astype(str).to_numpy()
    method_preds: dict[str, pd.Series] = {}
    for method in methods:
        ad = adatas[method]
        method_pred = pd.Series(index=obs_names, dtype=object)
        for sid in slices:
            idx = np.flatnonzero(slice_ids == sid)
            if len(idx) <= n_clusters:
                continue
            method_pred.iloc[idx] = _method_labels_for_slice(method, ad, idx, n_clusters, specs)
        method_preds[method] = method_pred
    return method_preds


def _plot_cluster_overview(
    *,
    adata,
    gt_labels: pd.Series,
    method_preds: Mapping[str, pd.Series],
    slices: list[str],
    out: Path,
    methods: list[str],
    gt_label: str,
    gt_palette: Mapping[str, str],
    dpi: int = 400,
) -> Path:
    rows = [{"label": gt_label, "values": gt_labels, "kind": "categorical", "palette": gt_palette}]
    rows.extend({"label": method, "values": method_preds[method], "kind": "categorical", "palette": gt_palette} for method in methods)
    return _plot_spatial_grid(adata, rows, slices, out, dpi=dpi)


def _align_predictions_to_truth(
    method_preds: Mapping[str, pd.Series],
    gt_labels: pd.Series,
    methods: list[str],
    slice_ids,
    target_labels: list[str] | None = None,
    fallback_labels: pd.Series | None = None,
) -> dict[str, pd.Series]:
    aligned: dict[str, pd.Series] = {}
    target_labels = list(LAYER_MARKERS.keys()) if target_labels is None else list(target_labels)
    gt = gt_labels.astype(str)
    fallback = fallback_labels.astype(str) if fallback_labels is not None else None
    slice_ids = pd.Series(np.asarray(slice_ids).astype(str), index=gt_labels.index)
    for method in methods:
        pred = method_preds[method].astype("string")
        aligned_pred = pd.Series("#unassigned", index=pred.index, dtype=object)
        for sid in sorted(slice_ids.dropna().unique(), key=str):
            sid_mask = slice_ids == sid
            pred_sid = pred[sid_mask]
            gt_sid = gt[sid_mask]
            fallback_sid = fallback[sid_mask] if fallback is not None else None
            raw_labels = sorted(pred_sid.dropna().astype(str).unique(), key=str)
            table = np.zeros((len(raw_labels), len(target_labels)), dtype=float)
            for i, raw_label in enumerate(raw_labels):
                raw_mask = pred_sid.astype(str) == raw_label
                truth = gt_sid[raw_mask & gt_sid.isin(target_labels)]
                if truth.empty and fallback_sid is not None:
                    truth = fallback_sid[raw_mask & fallback_sid.isin(target_labels)]
                counts = truth.value_counts()
                for j, target in enumerate(target_labels):
                    table[i, j] = float(counts.get(target, 0))

            mapping: dict[str, str] = {raw_label: "#unassigned" for raw_label in raw_labels}
            if table.size and table.sum() > 0:
                row_ind, col_ind = linear_sum_assignment(-table)
                for i, j in zip(row_ind, col_ind):
                    if table[i, j] > 0:
                        mapping[raw_labels[i]] = target_labels[j]
            for i, raw_label in enumerate(raw_labels):
                if mapping[raw_label] != "#unassigned":
                    continue
                row = table[i]
                if row.sum() > 0:
                    mapping[raw_label] = target_labels[int(np.argmax(row))]
            aligned_pred.loc[sid_mask] = pred_sid.astype(str).map(mapping).fillna("#unassigned")
        aligned[method] = aligned_pred
    return aligned


def _plot_cluster_representatives(
    *,
    adata,
    gt_labels: pd.Series,
    aligned_method_preds: Mapping[str, pd.Series],
    slices: list[str],
    out_dir: Path,
    methods: list[str],
    gt_label: str,
    palette: Mapping[str, str],
    dpi: int = 400,
) -> list[Path]:
    if not slices:
        return []
    out_paths = [
        _plot_categorical_tiles(
            adata,
            gt_labels,
            slices,
            out_dir / f"{_safe_name(gt_label)}.png",
            palette=palette,
            ncols=2,
            show_legend=True,
            title=gt_label,
            dpi=dpi,
        )
    ]
    for method in methods:
        out_paths.append(
            _plot_categorical_tiles(
                adata,
                aligned_method_preds[method],
                slices,
                out_dir / f"{_safe_name(method)}.png",
                palette=palette,
                ncols=2,
                show_legend=True,
                title=_display_title(method),
                dpi=dpi,
            )
        )
    return out_paths


def _plot_layer_confusion_grid(
    pred_df: pd.DataFrame,
    out: Path,
    *,
    methods: list[str],
    target_labels: list[str] | None = None,
    dpi: int = 400,
) -> Path:
    target_labels = list(LAYER_MARKERS.keys()) if target_labels is None else list(target_labels)
    fig, axes = plt.subplots(1, len(methods), figsize=(2.05 * len(methods), 2.25), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        sub = pred_df[(pred_df["method"] == method)].dropna(subset=["ground_truth", "merged_pred"])
        table = pd.crosstab(sub["ground_truth"], sub["merged_pred"]).reindex(index=target_labels, columns=target_labels, fill_value=0)
        mat = table.to_numpy(dtype=float)
        denom = mat.sum(axis=1, keepdims=True)
        frac = np.divide(mat, denom, out=np.zeros_like(mat), where=denom > 0)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(np.arange(len(target_labels)))
        ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(target_labels)))
        ax.set_yticklabels(target_labels if ax is axes[0] else [], fontsize=7)
        ax.set_xlabel(method, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    cb.set_label("row fraction", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_stgp_layer_proportion_by_slice(
    pred_df: pd.DataFrame,
    out: Path,
    *,
    target_labels: list[str] | None = None,
    dpi: int = 400,
) -> Path:
    target_labels = list(LAYER_MARKERS.keys()) if target_labels is None else list(target_labels)
    sub = pred_df[(pred_df["method"] == "stGP")].dropna(subset=["merged_pred"]).copy()
    if sub.empty:
        return out
    counts = pd.crosstab(sub["id_region"], sub["merged_pred"]).reindex(columns=target_labels, fill_value=0)
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    age_by_slice = sub.groupby("id_region")["age"].first()
    order = sorted(props.index, key=lambda sid: (float(age_by_slice.get(sid, np.inf)), str(sid)))
    props = props.loc[order]
    fig, ax = plt.subplots(1, 1, figsize=(max(4.8, 0.34 * len(props) + 1.8), 2.7))
    bottom = np.zeros(len(props), dtype=float)
    x = np.arange(len(props))
    for label in target_labels:
        vals = props[label].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=LAYER_COLORS.get(label, "#888888"), width=0.82, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(props.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _despine_metric_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _run_supplemental_cluster_k(
    *,
    adatas: Mapping[str, object],
    adata,
    marker_region: pd.Series,
    celltype2_display: pd.Series,
    celltype2_gt_full: pd.Series,
    celltype2,
    specs: Mapping[str, Mapping[str, str]],
    methods: list[str],
    slices_sorted: list[str],
    rep_slices: list[str],
    out_root: Path,
    source_root: Path,
    dpi: int = 400,
) -> list[Path]:
    out_paths: list[Path] = []
    slice_ids_full = adata.obs["id_region"].astype(str).to_numpy()

    for k in [4, 5, 6]:
        k_dir = out_root / f"k{k}"
        source_k_dir = source_root / f"k{k}"
        stgp_raw = _predict_method_clusters(
            adatas=adatas,
            obs_names=adata.obs_names,
            slices=slices_sorted,
            specs=specs,
            methods=["stGP"],
            n_clusters=k,
        )["stGP"]
        stgp_clusters = _raw_cluster_labels(stgp_raw)
        stgp_palette = _raw_cluster_palette(stgp_clusters)

        source_df = pd.DataFrame(
            {
                "cell": adata.obs_names,
                "id_region": slice_ids_full,
                "age": pd.to_numeric(adata.obs["age"], errors="coerce").to_numpy(float),
                "stGP_raw_cluster": stgp_raw.to_numpy(),
                "stGP_cluster_label": stgp_clusters.to_numpy(),
                "marker_region": marker_region.to_numpy(),
                "celltype2_or_ext": celltype2_display.to_numpy(),
            }
        )
        source_k_dir.mkdir(parents=True, exist_ok=True)
        source_df.to_csv(source_k_dir / f"stGP_raw_clusters_k{k}.csv", index=False)

        marker_dir = k_dir / "marker_region"
        out_paths.append(
            _plot_spatial_grid(
                adata,
                [
                    {"label": "marker_region", "values": marker_region, "kind": "categorical", "palette": LAYER_COLORS},
                    {"label": f"stGP_k{k}", "values": stgp_clusters, "kind": "categorical", "palette": stgp_palette},
                ],
                slices=slices_sorted,
                out=marker_dir / f"cluster_overview_all_slices_k{k}.png",
                dpi=dpi,
            )
        )
        out_paths.extend(
            _plot_cluster_representatives(
                adata=adata,
                gt_labels=marker_region,
                aligned_method_preds={f"stGP_k{k}": stgp_clusters},
                slices=rep_slices,
                out_dir=marker_dir / "representative",
                methods=[f"stGP_k{k}"],
                gt_label=f"marker_region_k{k}",
                palette={**LAYER_COLORS, **stgp_palette},
                dpi=dpi,
            )
        )

        cell_dir = k_dir / "celltype2"
        out_paths.append(
            _plot_spatial_grid(
                adata,
                [
                    {"label": "celltype2", "values": celltype2_display, "kind": "categorical", "palette": LAYER_COLORS},
                    {"label": f"stGP_k{k}", "values": stgp_clusters, "kind": "categorical", "palette": stgp_palette},
                ],
                slices=slices_sorted,
                out=cell_dir / f"cluster_overview_all_slices_k{k}.png",
                dpi=dpi,
            )
        )
        out_paths.extend(
            _plot_cluster_representatives(
                adata=adata,
                gt_labels=celltype2_display,
                aligned_method_preds={f"stGP_k{k}": stgp_clusters},
                slices=rep_slices,
                out_dir=cell_dir / "representative",
                methods=[f"stGP_k{k}"],
                gt_label=f"celltype2_k{k}",
                palette={**LAYER_COLORS, **stgp_palette},
                dpi=dpi,
            )
        )

    return out_paths


def _plot_marker_correlation(
    layer_results: Mapping[str, Mapping[str, list[float]]],
    out_dirs: Mapping[str, Path],
    methods: list[str],
    *,
    dpi: int = 400,
) -> list[Path]:
    out_paths = []
    for layer, gene, _ in LAYER_SPECS:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
        data = [np.asarray(layer_results[layer][method], dtype=float) for method in methods]
        colors = [METHOD_COLORS.get(method, "#999999") for method in methods]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.3),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
            flierprops=dict(marker="o", markersize=3.2, alpha=0.55, markeredgewidth=0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for flier, color in zip(bp["fliers"], colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Pearson correlation")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        _add_paired_brackets(ax, layer_results[layer], methods, fontsize=8)
        _despine_metric_axes(ax)
        out = out_dirs[f"{layer}:correlation"] / "embedding_vs_marker_correlation.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.3), constrained_layout=True)
    for ax, (layer, _, _) in zip(axes, LAYER_SPECS):
        data = [np.asarray(layer_results[layer][method], dtype=float) for method in methods]
        colors = [METHOD_COLORS.get(method, "#999999") for method in methods]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=2.8, alpha=0.5, markeredgewidth=0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for flier, color in zip(bp["fliers"], colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("Pearson correlation" if ax is axes[0] else "")
        ax.set_title(f"{layer} ({LAYER_MARKERS[layer]})", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=4)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        _add_paired_brackets(ax, layer_results[layer], methods, fontsize=7)
        _despine_metric_axes(ax)
    out = out_dirs["summary"] / "embedding_vs_marker_correlation_all_layers.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(out)
    return out_paths


def _paired_wilcoxon_tests(values_by_method: Mapping[str, list[float]], methods: list[str]) -> list[dict]:
    ref = np.asarray(values_by_method["stGP"], dtype=float)
    raw_p, records = [], []
    for method in methods:
        if method == "stGP":
            continue
        other = np.asarray(values_by_method[method], dtype=float)
        valid = np.isfinite(ref) & np.isfinite(other)
        diff = ref[valid] - other[valid]
        if valid.sum() < 2:
            p = np.nan
        elif np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(ref[valid], other[valid], alternative="greater").pvalue)
        raw_p.append(p)
        records.append(
            {
                "other_method": method,
                "test": "one-sided paired Wilcoxon signed-rank test",
                "alternative": "greater",
                "raw_p": p,
                **_paired_effect_summary(ref[valid], other[valid]),
            }
        )
    raw_p = np.asarray(raw_p, dtype=float)
    adj = np.full_like(raw_p, np.nan)
    valid_p = np.isfinite(raw_p)
    if valid_p.any():
        _, adj[valid_p], _, _ = multipletests(raw_p[valid_p], method="holm")
    for record, p_adj in zip(records, adj):
        record["holm_adj_p"] = p_adj
        record["stars"] = _stars(p_adj)
    return records


def _add_paired_brackets(ax, values_by_method: Mapping[str, list[float]], methods: list[str], *, fontsize: int) -> None:
    tests = _paired_wilcoxon_tests(values_by_method, methods)
    data = [np.asarray(values_by_method[method], dtype=float) for method in methods]
    vals = np.concatenate([d[np.isfinite(d)] for d in data if np.isfinite(d).any()])
    if vals.size == 0:
        return
    y_min, y_max = float(np.nanmin(vals)), float(np.nanmax(vals))
    y_range = y_max - y_min if y_max > y_min else 1.0
    y0 = y_max + 0.08 * y_range
    dy = 0.035 * y_range
    step = 0.13 * y_range
    for i, test in enumerate(tests):
        x1, x2 = 1, methods.index(test["other_method"]) + 1
        y = y0 + i * step
        ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=0.9, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, y + dy, test["stars"], ha="center", va="bottom", fontsize=fontsize)
    ax.set_ylim(top=y0 + len(tests) * step + 0.08 * y_range)


def _plot_agewise_metrics(metrics_by_gt: Mapping[str, pd.DataFrame], out_dir: Path, *, methods: list[str], dpi: int = 400) -> list[Path]:
    metric_map = {"raw_ari": "ARI", "raw_nmi": "NMI", "raw_acc": "ACC"}
    out_paths = []
    for gt_name, df in metrics_by_gt.items():
        for metric, label in metric_map.items():
            fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.1))
            for method in methods:
                sub = (
                    df[df["method"] == method]
                    .groupby("age", as_index=False)[metric]
                    .mean()
                    .sort_values("age")
                )
                ax.plot(
                    sub["age"],
                    sub[metric],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.4 if method == "stGP" else 0.9,
                    color=METHOD_COLORS.get(method, "#999999"),
                    alpha=1.0 if method == "stGP" else 0.75,
                    label=method,
                )
            ax.set_xlabel("Age (yr)")
            ax.set_ylabel(label)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            _despine_metric_axes(ax)
            out = out_dir / f"per_slice_metric_by_age_{gt_name}_{label}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out)
    return out_paths


def _plot_pairwise_deltas(metrics_by_gt: Mapping[str, pd.DataFrame], out_dir: Path, *, methods: list[str], dpi: int = 400) -> list[Path]:
    metric_map = {"raw_ari": "ARI", "raw_nmi": "NMI", "raw_acc": "ACC"}
    out_paths = []
    baselines = [method for method in methods if method != "stGP"]
    for gt_name, df in metrics_by_gt.items():
        for metric, label in metric_map.items():
            ref = df[df["method"] == "stGP"][["id_region", metric]].rename(columns={metric: "stGP"})
            delta_data = []
            for baseline in baselines:
                other = df[df["method"] == baseline][["id_region", metric]].rename(columns={metric: baseline})
                paired = ref.merge(other, on="id_region").dropna()
                delta_data.append((paired["stGP"] - paired[baseline]).to_numpy(dtype=float))
            fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.2))
            bp = ax.boxplot(
                delta_data,
                patch_artist=True,
                widths=0.55,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=3.0, alpha=0.55, markeredgewidth=0),
            )
            for patch, baseline in zip(bp["boxes"], baselines):
                patch.set_facecolor(METHOD_COLORS.get(baseline, "#999999"))
                patch.set_alpha(0.7)
            for flier, baseline in zip(bp["fliers"], baselines):
                flier.set_markerfacecolor(METHOD_COLORS.get(baseline, "#999999"))
                flier.set_markeredgecolor(METHOD_COLORS.get(baseline, "#999999"))
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_xticks(range(1, len(baselines) + 1))
            ax.set_xticklabels(baselines, rotation=30, ha="right")
            ax.set_ylabel(f"stGP - baseline {label}")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
            _despine_metric_axes(ax)
            out = out_dir / f"stgp_pairwise_delta_{gt_name}_{label}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out)
    return out_paths


def _plot_method_rank(
    metrics_by_gt: Mapping[str, pd.DataFrame],
    out: Path,
    *,
    methods: list[str],
    source_out: Path | None = None,
    dpi: int = 400,
) -> Path:
    metric_map = {"raw_ari": "ARI", "raw_nmi": "NMI", "raw_acc": "ACC"}
    records = []
    for gt_name, df in metrics_by_gt.items():
        means = df.groupby("method")[list(metric_map)].mean()
        for metric, label in metric_map.items():
            ranks = means[metric].rank(ascending=False, method="average")
            for method, rank in ranks.items():
                records.append({"benchmark": f"{gt_name}:{label}", "method": method, "rank": float(rank)})
    rank_df = pd.DataFrame(records)
    rank_csv = source_out if source_out is not None else out.with_suffix(".csv")
    rank_csv.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(rank_csv, index=False)
    pivot = rank_df.pivot_table(index="benchmark", columns="method", values="rank").reindex(columns=methods)
    fig, ax = plt.subplots(1, 1, figsize=(4.6, max(2.4, 0.36 * len(pivot) + 1.0)))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r", vmin=1, vmax=len(methods), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Mean rank")
    _despine_metric_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def run_ext_benchmarking(
    *,
    adata,
    adata_prep,
    baseline_adatas: Mapping[str, object],
    fig_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    slices=None,
    methods: list[str] | None = None,
    dpi: int = 400,
) -> dict[str, object]:
    """Generate HumanBrain ext benchmark figures and source data.

    Figure files are written under ``benchmark_dir``. CSV source tables are
    written under ``source_dir`` so final figure notebooks can read exclusively
    from ``Results/``.
    """
    methods = METHODS if methods is None else list(methods)
    benchmark_dir = Path(benchmark_dir) if benchmark_dir is not None else Path(fig_dir) / "benchmark"
    source_dir = Path(source_dir) if source_dir is not None else _default_source_benchmark_dir(benchmark_dir)
    dirs = _prepare_dirs(benchmark_dir, source_dir)
    _cleanup_stale_outputs(benchmark_dir, source_dir, dirs)
    slices_sorted = _slice_order(adata, slices)
    rep_slices = _representative_slices_by_age(adata, slices_sorted)

    adatas = {
        "stGP": adata,
        "STAMP": baseline_adatas["STAMP"],
        "MEFISTO": baseline_adatas["MEFISTO"],
        "Popari": baseline_adatas["Popari"],
        "SpatialPCA": baseline_adatas["SpatialPCA"],
    }
    specs = {
        "stGP": {"obsm_key": "X_stgp_spatial"},
        "STAMP": {"obsm_key": "X_stamp", "kind": "argmax"},
        "MEFISTO": {"obsm_key": "X_mefisto"},
        "Popari": {"obsm_key": "X"},
        "SpatialPCA": {"obsm_key": "X_spatialpca"},
    }

    # Marker expression and marker-derived layer ground truth use log-normalized values.
    marker_X = adata_prep[:, list(LAYER_MARKERS.values())].X
    marker_expr = marker_X.toarray() if sp.issparse(marker_X) else np.asarray(marker_X)
    marker_z = (marker_expr - marker_expr.mean(axis=0)) / (marker_expr.std(axis=0) + 1e-8)
    marker_region = pd.Series(
        np.array(list(LAYER_MARKERS.keys()))[np.argmax(marker_z, axis=1)],
        index=adata.obs_names,
        name="marker_region",
    )

    # Marker display uses the unlogged expression matrix.
    marker_display_X = adata[:, list(LAYER_MARKERS.values())].X
    marker_display_expr = marker_display_X.toarray() if sp.issparse(marker_display_X) else np.asarray(marker_display_X)
    marker_display_by_gene = {
        gene: marker_display_expr[:, i]
        for i, gene in enumerate(LAYER_MARKERS.values())
    }

    layer_results, bl_k_per_layer, corr_records = {}, {}, []
    for layer, gene, stgp_k in LAYER_SPECS:
        gene_idx = np.where(adata_prep.var_names == gene)[0][0]
        expr_all = adata_prep.X[:, gene_idx]
        bl_k = {
            method: best_program_by_correlation(baseline_adatas[method], BASELINE_OBSM_KEYS[method], expr_all)
            for method in BASELINE_OBSM_KEYS
        }
        bl_k_per_layer[layer] = bl_k
        layer_results[layer] = {method: [] for method in methods}
        for sid in slices_sorted:
            mask = adata_prep.obs["id_region"].astype(str).to_numpy() == sid
            c_gene = _dense_1d(adata_prep.X[mask, gene_idx])
            for method in methods:
                if method == "stGP":
                    vals = np.asarray(adata.obsm["X_stgp_spatial"])[mask, stgp_k]
                    corr = np.corrcoef(vals, c_gene)[0, 1]
                else:
                    key = BASELINE_OBSM_KEYS[method]
                    vals = np.asarray(baseline_adatas[method].obsm[key])[mask, bl_k[method]]
                    corr = np.corrcoef(vals, c_gene)[0, 1]
                    if key in {"X_mefisto", "X_spatialpca"}:
                        corr = abs(corr)
                layer_results[layer][method].append(corr)
                corr_records.append({"layer": layer, "marker_gene": gene, "method": method, "id_region": sid, "correlation": corr})

    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv(dirs["source"] / "marker_embedding_correlations.csv", index=False)
    corr_df.groupby(["layer", "marker_gene", "method"], as_index=False)["correlation"].agg(["mean", "median", "std"]).to_csv(
        dirs["source"] / "marker_embedding_correlation_summary.csv"
    )
    corr_tests = []
    for layer, gene, _ in LAYER_SPECS:
        for test in _paired_wilcoxon_tests(layer_results[layer], methods):
            corr_tests.append({"layer": layer, "marker_gene": gene, "ref_method": "stGP", **test})
    pd.DataFrame(corr_tests).to_csv(dirs["source"] / "marker_embedding_correlation_wilcoxon_tests.csv", index=False)
    corr_paths = _plot_marker_correlation(layer_results, dirs, methods, dpi=dpi)

    spatial_paths = []
    celltype2 = adata.obs["celltype2"].astype("string")
    celltype2_gt_full = celltype2.map(CELLTYPE2_LAYER_MAP)
    celltype2_display = pd.Series(
        celltype2_gt_full.fillna("ext").astype(object).to_numpy(),
        index=adata.obs_names,
        name="celltype2_layer_or_ext",
    )
    celltype2_values = celltype2_display.to_numpy()
    for layer, gene, stgp_k in LAYER_SPECS:
        gene_idx = np.where(adata_prep.var_names == gene)[0][0]
        rows = [
            {
                "label": "stGP_b",
                "values": np.asarray(adata.obsm["X_stgp_spatial"])[:, stgp_k],
                "kind": "continuous",
                "cmap": "RdBu_r",
                "colorbar_label": "score",
            }
        ]
        for method in ["Popari", "STAMP", "MEFISTO", "SpatialPCA"]:
            key = BASELINE_OBSM_KEYS[method]
            rows.append(
                {
                    "label": method,
                    "values": np.asarray(baseline_adatas[method].obsm[key])[:, bl_k_per_layer[layer][method]],
                    "kind": "continuous",
                    "cmap": "RdBu_r" if key in {"X_mefisto", "X_spatialpca"} else "YlOrBr",
                    "colorbar_label": "score",
                }
            )
        rows.append(
            {
                "label": gene,
                "values": marker_display_by_gene[gene],
                "kind": "continuous",
                "cmap": "YlOrRd",
                "colorbar_label": "expression",
            }
        )
        rows.append({"label": "celltype2", "values": celltype2_values, "kind": "categorical", "palette": LAYER_COLORS})
        spatial_paths.append(
            _plot_spatial_grid(
                adata,
                rows,
                slices_sorted,
                dirs[f"{layer}:embedding_full"] / "embedding_marker_celltype2_all_slices.png",
                dpi=dpi,
            )
        )
        rep_rows = [row for row in rows if row["label"] != gene]
        spatial_paths.extend(_plot_representative_rows(adata, rep_rows, rep_slices, dirs[f"{layer}:embedding_rep"], dpi=dpi))
        spatial_paths.append(
            _plot_continuous_tiles(
                adata,
                marker_display_by_gene[gene],
                rep_slices,
                dirs[f"{layer}:embedding_rep"] / f"{gene}_expression.png",
                ncols=2,
                cmap="viridis",
                colorbar_label="expression",
                title=gene,
                dpi=dpi,
            )
        )

    metrics_by_gt: dict[str, pd.DataFrame] = {}
    pred_by_gt: dict[str, pd.DataFrame] = {}
    tests = []
    cluster_paths = []
    marker_fig_dir = dirs["clustering"] / "marker_region"
    marker_source_dir = dirs["source_clustering"] / "marker_region"
    slice_ids_full = adata.obs["id_region"].astype(str).to_numpy()
    marker_metrics, marker_mapping, marker_preds, marker_method_preds, marker_merged_preds = _evaluate_ground_truth(
        adatas=adatas,
        gt_labels=marker_region,
        slices=slices_sorted,
        out_dir=marker_source_dir,
        specs=specs,
        methods=methods,
    )
    metrics_by_gt["marker_region"] = marker_metrics
    pred_by_gt["marker_region"] = marker_preds
    marker_aligned_preds = _align_predictions_to_truth(marker_method_preds, marker_region, methods, slice_ids_full)
    cluster_paths.append(
        _plot_cluster_overview(
            adata=adata,
            gt_labels=marker_region,
            method_preds=marker_aligned_preds,
            slices=slices_sorted,
            out=marker_fig_dir / "cluster_overview_all_slices.png",
            methods=methods,
            gt_label="marker_region",
            gt_palette=LAYER_COLORS,
            dpi=dpi,
        )
    )
    cluster_paths.extend(
        _plot_cluster_representatives(
            adata=adata,
            gt_labels=marker_region,
            aligned_method_preds=marker_aligned_preds,
            slices=rep_slices,
            out_dir=marker_fig_dir / "representative",
            methods=methods,
            gt_label="marker_region",
            palette=LAYER_COLORS,
            dpi=dpi,
        )
    )
    cluster_paths.extend(
        [
            _plot_layer_confusion_grid(marker_preds, marker_fig_dir / "layer_confusion_by_method.png", methods=methods, dpi=dpi),
            _plot_stgp_layer_proportion_by_slice(marker_preds, marker_fig_dir / "stGP_layer_proportion_by_slice.png", dpi=dpi),
        ]
    )

    cell_mask = celltype2.isin(CELLTYPE2_LAYER_MAP.keys()).to_numpy()
    cell_fig_dir = dirs["clustering"] / "celltype2"
    cell_source_dir = dirs["source_clustering"] / "celltype2"
    cell_display_method_preds = _predict_method_clusters(
        adatas=adatas,
        obs_names=adata.obs_names,
        slices=slices_sorted,
        specs=specs,
        methods=methods,
        n_clusters=3,
    )
    cell_aligned_preds = _align_predictions_to_truth(
        cell_display_method_preds,
        celltype2_display,
        methods,
        slice_ids_full,
        fallback_labels=marker_region,
    )
    cluster_paths.append(
        _plot_cluster_overview(
            adata=adata,
            gt_labels=celltype2_display,
            method_preds=cell_aligned_preds,
            slices=slices_sorted,
            out=cell_fig_dir / "cluster_overview_all_slices.png",
            methods=methods,
            gt_label="celltype2",
            gt_palette=LAYER_COLORS,
            dpi=dpi,
        )
    )
    cluster_paths.extend(
        _plot_cluster_representatives(
            adata=adata,
            gt_labels=celltype2_display,
            aligned_method_preds=cell_aligned_preds,
            slices=rep_slices,
            out_dir=cell_fig_dir / "representative",
            methods=methods,
            gt_label="celltype2",
            palette=LAYER_COLORS,
            dpi=dpi,
        )
    )
    if np.any(cell_mask):
        adatas_mask = {method: adatas[method][cell_mask].copy() for method in methods}
        gt_celltype2 = pd.Series(
            celltype2_gt_full[cell_mask].astype(object).to_numpy(),
            index=adatas_mask["stGP"].obs_names,
            name="celltype2_layer",
        )
        slice_ids_mask = adatas_mask["stGP"].obs["id_region"].astype(str).to_numpy()
        cell_slices = [sid for sid in slices_sorted if sid in set(slice_ids_mask)]
        cell_metrics, cell_mapping, cell_preds, cell_method_preds, cell_merged_preds = _evaluate_ground_truth(
            adatas=adatas_mask,
            gt_labels=gt_celltype2,
            slices=cell_slices,
            out_dir=cell_source_dir,
            specs=specs,
            methods=methods,
        )
        metrics_by_gt["celltype2"] = cell_metrics
        pred_by_gt["celltype2"] = cell_preds
        cluster_paths.extend(
            [
                _plot_layer_confusion_grid(cell_preds, cell_fig_dir / "layer_confusion_by_method.png", methods=methods, dpi=dpi),
                _plot_stgp_layer_proportion_by_slice(cell_preds, cell_fig_dir / "stGP_layer_proportion_by_slice.png", dpi=dpi),
            ]
        )

    cluster_paths.extend(
        _run_supplemental_cluster_k(
            adatas=adatas,
            adata=adata,
            marker_region=marker_region,
            celltype2_display=celltype2_display,
            celltype2_gt_full=celltype2_gt_full,
            celltype2=celltype2,
            specs=specs,
            methods=methods,
            slices_sorted=slices_sorted,
            rep_slices=rep_slices,
            out_root=dirs["clustering"] / "supp_cluster_k",
            source_root=dirs["source_clustering"] / "supp_cluster_k",
            dpi=dpi,
        )
    )

    metric_map = {
        "raw_ari": "ARI",
        "raw_nmi": "NMI",
        "raw_acc": "ACC",
        "merged_acc": "Merged_ACC",
        "merged_macro_f1": "Merged_F1",
    }
    metric_paths = []
    for gt_name, metrics_df in metrics_by_gt.items():
        metrics_df.to_csv(dirs["source"] / f"{gt_name}_cluster_metrics.csv", index=False)
        out_dir = dirs["clustering"] / gt_name / "metrics"
        source_metric_dir = dirs["source_clustering"] / gt_name / "metrics"
        for metric, label in metric_map.items():
            metric_paths.append(out_dir / f"{label}.png")
            test_df = _boxplot_metric(
                metrics_df,
                metric,
                label,
                metric_paths[-1],
                source_metric_dir / f"{label}_wilcoxon_tests.csv",
                methods=methods,
                dpi=dpi,
            )
            test_df.insert(0, "ground_truth", gt_name)
            tests.append(test_df)

    if tests:
        pd.concat(tests, ignore_index=True).to_csv(dirs["source"] / "cluster_metric_wilcoxon_tests.csv", index=False)
    summary_paths = [
        _plot_method_rank(
            metrics_by_gt,
            dirs["summary"] / "method_rank_by_metric.png",
            methods=methods,
            source_out=dirs["source_summary"] / "method_rank_by_metric.csv",
            dpi=dpi,
        ),
    ]
    summary_paths.extend(_plot_agewise_metrics(metrics_by_gt, dirs["summary"], methods=methods, dpi=dpi))
    summary_paths.extend(_plot_pairwise_deltas(metrics_by_gt, dirs["summary"], methods=methods, dpi=dpi))

    return {
        "benchmark_dir": benchmark_dir,
        "source_dir": source_dir,
        "correlation_figures": corr_paths,
        "spatial_figures": spatial_paths,
        "cluster_figures": cluster_paths,
        "metric_figures": metric_paths,
        "summary_figures": summary_paths,
        "correlation_source": corr_df,
        "cluster_metrics": metrics_by_gt,
    }
