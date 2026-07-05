"""Step 01: QC + per-celltype extraction for the Human Brain MERFISH aging dataset
(Jeffries et al., Nature 2025).

QC passes (in order):
  1. load both cohorts (elderly, infant) and harmonise the obs columns;
  2. drop flagged samples (batch effects) and unused ages;
  3. restrict to the cell types studied in the paper;
  4. cell-level QC on counts, gene counts, bbox area, anisotropy;
  5. neuronal-contamination filter (drop non-neuronal cells whose neuronal-
     marker fraction is > ``--max-neuro-frac``);
  6. spatial proximity filter (drop non-neuronal cells closer than
     ``--min-dist-um`` micrometres to a neuron in the same sample).

Then writes one ``data/processed/<safe_name>.h5ad`` file per cell type.

Note: the QC details (anisotropy, neuronal contamination, proximity-to-neuron)
are tailored to the human MERFISH dataset and are deliberately different from
the mouse-pipeline QC.

Example::

    python 01_preprocess_qc.py \\
        --data-dir "${STGP_HUMAN_RAW_DIR:-data/raw/HumanBrainMERFISH}" \\
        --output data/qc/human_merfish_qc.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

from utils import (
    ALL_CELLTYPES,
    DATA_PROCESSED,
    DEFAULT_RAW_DATA_DIR,
    safe_name,
)


# ════════════════════════════════════════════════════════════════════════════
# Sample / age exclusions and column harmonisation
# ════════════════════════════════════════════════════════════════════════════

EXCLUDE_SAMPLES = [
    {"donor_id": "5823", "region_id": "rep1"},   # batch effect
]

# Donor 1278 (age 0.4) is too young to fit alongside the rest of the cohort.
EXCLUDE_AGES = [0.4]

COMMON_OBS_COLS = [
    "volume", "center_x", "center_y",
    "min_x", "min_y", "max_x", "max_y",
    "anisotropy", "perimeter_area_ratio", "solidity",
    "celltype", "celltype2",
    "donor_id", "region_id", "age", "sex",
]

# Marker genes used to score neuronal contamination of non-neuronal cells.
NEURONAL_MARKERS = ["SLC17A7", "RORB", "SATB2", "NEUROD6", "GAD1"]
NEURON_CELLTYPES = {"ext", "inb"}


# ════════════════════════════════════════════════════════════════════════════
# Per-cohort loader
# ════════════════════════════════════════════════════════════════════════════

def load_cohort(h5ad_dir: Path, cohort_name: str) -> sc.AnnData:
    """Load and concatenate every ``*.h5ad`` file in a directory.

    Only cells with a non-NaN ``celltype`` (i.e. that passed the upstream
    Seurat annotation) are kept. obs columns are restricted to
    ``COMMON_OBS_COLS`` so cohort-specific extras (e.g. infant-only stain
    intensities) don't bloat the merged AnnData with NaN columns.
    """
    files = sorted(h5ad_dir.glob("*.h5ad"))
    if not files:
        if not h5ad_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {h5ad_dir}\n"
                f"        Pass the dataset root via --data-dir. The default in "
                f"utils.DEFAULT_RAW_DATA_DIR is\n"
                f"        data/raw/HumanBrainMERFISH, or set STGP_HUMAN_RAW_DIR."
            )
        raise FileNotFoundError(
            f"No .h5ad files found in {h5ad_dir}\n"
            f"        Expected per-sample h5ad files under "
            f"<data-dir>/MERFISH_human_aging/<cohort>/MERFISH_h5ad/."
        )

    adatas = []
    for f in files:
        a = sc.read_h5ad(str(f))
        keep_cols = [c for c in COMMON_OBS_COLS if c in a.obs.columns]
        a.obs = a.obs[keep_cols].copy()

        n_before = a.n_obs
        ct = a.obs["celltype"].astype(str)
        valid_mask = ct.notna() & (ct != "nan") & (ct != "NaN") & (ct != "")
        a = a[valid_mask].copy()

        donor = a.obs["donor_id"].iloc[0] if "donor_id" in a.obs else "?"
        age = a.obs["age"].iloc[0] if "age" in a.obs else "?"
        print(f"  loaded {f.name}: {a.n_obs} cells "
              f"(dropped {n_before - a.n_obs} with NaN celltype), "
              f"donor={donor}, age={age}")
        adatas.append(a)

    adata = sc.concat(adatas, join="inner", merge="first")
    adata.obs["cohort"] = cohort_name
    return adata


def _exclude_samples(adata: sc.AnnData) -> sc.AnnData:
    """Drop cells matching any entry in ``EXCLUDE_SAMPLES``."""
    n_before = adata.n_obs
    mask = pd.Series(True, index=adata.obs.index)
    for exc in EXCLUDE_SAMPLES:
        hit = pd.Series(True, index=adata.obs.index)
        for col, val in exc.items():
            hit = hit & (adata.obs[col].astype(str) == str(val))
        n_hit = int(hit.sum())
        if n_hit > 0:
            print(f"[qc] Excluding {n_hit} cells: {exc}")
        mask = mask & ~hit
    adata = adata[mask].copy()
    print(f"[qc] After sample exclusion: {adata.n_obs} / {n_before} cells")
    return adata


def _exclude_ages(adata: sc.AnnData) -> sc.AnnData:
    """Drop cells whose age is in ``EXCLUDE_AGES``."""
    n_before = adata.n_obs
    ages = pd.to_numeric(adata.obs["age"], errors="coerce")
    mask = ~ages.isin(EXCLUDE_AGES)
    dropped_ages = sorted(ages[~mask].unique().tolist())
    adata = adata[mask].copy()
    if dropped_ages:
        print(f"[qc] Excluded ages {dropped_ages}: "
              f"{adata.n_obs} / {n_before} cells remain")
    return adata


# ════════════════════════════════════════════════════════════════════════════
# Spatial / contamination filters
# ════════════════════════════════════════════════════════════════════════════

def filter_high_neuronal_contamination(
    adata: sc.AnnData, max_neuro_frac: float = 0.15,
) -> sc.AnnData:
    """Filter non-neuronal cells whose neuronal-marker fraction is too high.

    Also stores the per-cell ``neuronal_contamination_frac`` in ``adata.obs``
    so it remains available in the output h5ad for downstream inspection.
    """
    avail = [g for g in NEURONAL_MARKERS if g in adata.var_names]
    if not avail:
        print("[qc] No neuronal marker genes found; skipping contamination filter")
        adata.obs["neuronal_contamination_frac"] = 0.0
        return adata

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    neuro_idx = [adata.var_names.get_loc(g) for g in avail]
    neuro_counts = X[:, neuro_idx].sum(axis=1)
    total_counts = X.sum(axis=1)
    neuro_frac = np.where(total_counts > 0, neuro_counts / total_counts, 0.0)
    adata.obs["neuronal_contamination_frac"] = neuro_frac

    is_neuron = adata.obs["celltype"].isin(NEURON_CELLTYPES)
    high_contam = (~is_neuron) & (neuro_frac > max_neuro_frac)
    n_flagged = int(high_contam.sum())
    print(f"[qc] Neuronal contamination filter (neuro_frac > {max_neuro_frac}): "
          f"removed {n_flagged} non-neuronal cells  [markers used: {avail}]")
    return adata[~high_contam].copy()


def filter_cells_near_neurons(
    adata: sc.AnnData, min_dist_um: float = 15.0,
) -> sc.AnnData:
    """Remove non-neuronal cells whose nearest neuron is < ``min_dist_um`` away.

    Operates per ``id_region`` so distances are computed within the same
    tissue section. Neurons themselves are never removed.
    """
    is_neuron = np.isin(adata.obs["celltype"].values, list(NEURON_CELLTYPES))
    coords = np.asarray(adata.obsm["spatial"])

    keep = np.ones(adata.n_obs, dtype=bool)
    for sid in adata.obs["id_region"].unique():
        smask = (adata.obs["id_region"] == sid).values
        neuron_mask = smask & is_neuron
        glial_mask = smask & ~is_neuron
        if not neuron_mask.any() or not glial_mask.any():
            continue
        tree = cKDTree(coords[neuron_mask])
        dists, _ = tree.query(coords[glial_mask], k=1)
        glial_idx = np.where(glial_mask)[0]
        keep[glial_idx[dists < min_dist_um]] = False

    n_removed = int((~keep).sum())
    print(f"[qc] Spatial proximity filter ({min_dist_um} um): "
          f"removed {n_removed} non-neuronal cells near neurons")
    return adata[keep].copy()


# ════════════════════════════════════════════════════════════════════════════
# Main QC pipeline
# ════════════════════════════════════════════════════════════════════════════

def preprocess_qc(
    data_dir: Path,
    output_h5ad: Path,
    *,
    restricted_celltypes: list[str] = ALL_CELLTYPES,
    min_counts: int = 20,
    min_genes: int = 10,
    max_quantile: float = 0.999,
    bbox_low_q: float = 0.001,
    bbox_high_q: float = 0.999,
    aniso_high_q: float = 0.995,
    min_dist_um: float = 15.0,
    max_neuro_frac: float = 0.15,
) -> sc.AnnData:
    """Full QC pipeline. Writes ``output_h5ad`` and returns the AnnData."""
    data_dir = Path(data_dir)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    merfish_dir = data_dir / "MERFISH_human_aging"

    print("[qc] Loading elderly cohort...")
    elderly = load_cohort(merfish_dir / "MERFISH_elderly_adult" / "MERFISH_h5ad",
                          "elderly")
    print("[qc] Loading infant cohort...")
    infant = load_cohort(merfish_dir / "MERFISH_infant_adult" / "MERFISH_h5ad",
                         "infant")

    elderly = _exclude_samples(elderly)

    print("[qc] Merging cohorts...")
    adata = sc.concat([elderly, infant], join="inner", merge="first")
    adata.var_names_make_unique()
    n_raw = adata.n_obs

    adata = _exclude_ages(adata)

    adata.obs["age"] = pd.to_numeric(
        adata.obs["age"], errors="coerce").astype(np.float64)

    # Slice-level grouping (human analogue of mouse_id): one id per tissue slice.
    adata.obs["id_region"] = (
        adata.obs["donor_id"].astype(str)
        + "_" + adata.obs["region_id"].astype(str)
    )

    coords = np.column_stack([
        pd.to_numeric(adata.obs["center_x"], errors="coerce").to_numpy(),
        pd.to_numeric(adata.obs["center_y"], errors="coerce").to_numpy(),
    ])
    adata.obsm["spatial"] = coords.astype(np.float64)

    if all(c in adata.obs.columns for c in ["min_x", "min_y", "max_x", "max_y"]):
        adata.obs["bbox_area"] = (
            (pd.to_numeric(adata.obs["max_x"], errors="coerce")
             - pd.to_numeric(adata.obs["min_x"], errors="coerce"))
            * (pd.to_numeric(adata.obs["max_y"], errors="coerce")
               - pd.to_numeric(adata.obs["min_y"], errors="coerce"))
        )

    # ---- Restrict to studied cell types --------------------------------
    ct = adata.obs["celltype"].astype(str)
    unknown_labels = ct[~ct.isin(restricted_celltypes)]
    if len(unknown_labels) > 0:
        discarded = unknown_labels.value_counts()
        print(f"[qc] Discarding {len(unknown_labels)} cells with non-target celltypes:")
        for label, count in discarded.items():
            print(f"      {label}: {count}")
    adata = adata[ct.isin(restricted_celltypes)].copy()
    print(f"[qc] After cell-type restriction: {adata.n_obs} / {n_raw} cells")

    # ---- Cell-level QC: counts / genes / bbox / anisotropy -------------
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    max_counts = float(adata.obs["total_counts"].quantile(max_quantile))
    max_genes_val = float(adata.obs["n_genes_by_counts"].quantile(max_quantile))
    qc_mask = (
        (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["total_counts"] <= max_counts)
        & (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes_val)
    )
    if "bbox_area" in adata.obs.columns:
        bbox = adata.obs["bbox_area"].to_numpy()
        bbox_low = float(np.nanquantile(bbox, bbox_low_q))
        bbox_high = float(np.nanquantile(bbox, bbox_high_q))
        bbox_filter = (~np.isfinite(bbox)) | ((bbox >= bbox_low) & (bbox <= bbox_high))
        qc_mask = qc_mask & bbox_filter
        print(f"[qc] bbox_area filter [{bbox_low:.1f}, {bbox_high:.1f}]: "
              f"keeps {bbox_filter.sum()} cells")
    if "anisotropy" in adata.obs.columns:
        aniso = pd.to_numeric(adata.obs["anisotropy"], errors="coerce").to_numpy()
        aniso_thresh = float(np.nanquantile(aniso, aniso_high_q))
        aniso_filter = np.isnan(aniso) | (aniso <= aniso_thresh)
        qc_mask = qc_mask & aniso_filter
        print(f"[qc] anisotropy filter (<= {aniso_thresh:.2f}, q{aniso_high_q}): "
              f"keeps {aniso_filter.sum()} cells")
    adata_qc = adata[qc_mask].copy()
    print(f"[qc] After counts/genes/bbox/anisotropy filter: {adata_qc.n_obs} cells")

    # ---- Human-specific filters (contamination + spatial proximity) ----
    adata_qc = filter_high_neuronal_contamination(adata_qc, max_neuro_frac=max_neuro_frac)
    print(f"[qc] After neuronal contamination filter: {adata_qc.n_obs} cells")
    adata_qc = filter_cells_near_neurons(adata_qc, min_dist_um=min_dist_um)
    print(f"[qc] After spatial proximity filter: {adata_qc.n_obs} cells")

    sc.pp.calculate_qc_metrics(adata_qc, percent_top=None, log1p=False, inplace=True)

    adata_qc.uns["preprocess_info"] = dict(
        data_dir=str(data_dir),
        n_cells=int(adata_qc.n_obs),
        n_genes=int(adata_qc.n_vars),
        celltypes=sorted(adata_qc.obs["celltype"].unique().tolist()),
        excluded_samples=[str(e) for e in EXCLUDE_SAMPLES],
        excluded_ages=EXCLUDE_AGES,
        min_dist_um=min_dist_um,
        max_neuro_frac=max_neuro_frac,
    )

    print(f"[qc] Writing: {output_h5ad} "
          f"(n_obs={adata_qc.n_obs}, n_vars={adata_qc.n_vars})")
    print(f"  celltypes: {sorted(adata_qc.obs['celltype'].unique().tolist())}")
    print(f"  ages: {sorted(adata_qc.obs['age'].dropna().unique().tolist())}")
    print(f"  donors: {sorted(adata_qc.obs['donor_id'].dropna().unique().tolist())}")
    print(f"  n_samples (id_region): {adata_qc.obs['id_region'].nunique()}")
    adata_qc.write_h5ad(str(output_h5ad), compression="gzip")
    return adata_qc


# ════════════════════════════════════════════════════════════════════════════
# Per-celltype extraction (mouse-pipeline-style)
# ════════════════════════════════════════════════════════════════════════════

def extract_celltype_files(
    adata: sc.AnnData, processed_dir: Path,
    *, celltypes: list[str] = ALL_CELLTYPES,
) -> None:
    """Write one ``processed_dir/<safe_name>.h5ad`` per cell type."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for ct in celltypes:
        out = processed_dir / f"{safe_name(ct)}.h5ad"
        if out.exists():
            print(f"[extract] {ct!r}: exists, skipping ({out})")
            continue
        sub = adata[adata.obs["celltype"].astype(str) == ct].copy()
        if sub.n_obs == 0:
            print(f"[extract] {ct!r}: no cells, skipping")
            continue
        sub.write_h5ad(str(out), compression="gzip")
        n_samples = sub.obs["id_region"].nunique() if "id_region" in sub.obs else -1
        n_donors = sub.obs["donor_id"].nunique() if "donor_id" in sub.obs else -1
        print(f"[extract] {ct}: {sub.n_obs} cells, {sub.n_vars} genes, "
              f"{n_donors} donors, {n_samples} samples -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR,
        help=f"Root of HumanBrainMERFISH+sc_Nature2025_Jeffries. "
             f"Default: {DEFAULT_RAW_DATA_DIR}",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/qc/human_merfish_qc.h5ad"),
        help="QC .h5ad output.",
    )
    p.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED,
                   help="Output dir for per-celltype .h5ad files.")
    p.add_argument("--min-counts", type=int, default=20)
    p.add_argument("--min-genes", type=int, default=10)
    p.add_argument(
        "--min-dist-um", type=float, default=15.0,
        help="Spatial proximity threshold (um): remove non-neurons closer than "
             "this to any neuron in the same sample.",
    )
    p.add_argument(
        "--max-neuro-frac", type=float, default=0.15,
        help="Max allowed neuronal-marker fraction in non-neuronal cells.",
    )
    args = p.parse_args()

    if args.output.exists():
        print(f"[qc] {args.output} exists - loading instead of re-running QC.")
        adata = sc.read_h5ad(str(args.output))
    else:
        adata = preprocess_qc(
            data_dir=args.data_dir,
            output_h5ad=args.output,
            min_counts=args.min_counts,
            min_genes=args.min_genes,
            min_dist_um=args.min_dist_um,
            max_neuro_frac=args.max_neuro_frac,
        )

    extract_celltype_files(adata, args.processed_dir)


if __name__ == "__main__":
    main()
