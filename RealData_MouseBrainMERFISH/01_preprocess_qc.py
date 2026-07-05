"""Step 01: QC + per-celltype extraction for the Mouse Brain MERFISH aging dataset.

Three QC passes (in order):
  1. restrict to the cell types studied in the paper;
  2. drop a curated list of marker genes that have high spillover/misallocation
     rates (Sun et al. 2025);
  3. cell-level QC on counts, gene counts and bounding-box area.

Then writes one ``data/processed/<safe_name>.h5ad`` file per cell type.

Example:
    python 01_preprocess_qc.py \\
        --input "${STGP_MOUSE_RAW_H5AD:-data/raw/aging_coronal.h5ad}" \\
        --output data/qc/aging_coronal_qc.h5ad
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

from utils import DATA_PROCESSED, DEFAULT_RAW_DATA, safe_name

ALL_CELLTYPES = [
    "Neuron-Excitatory", "Neuron-MSN", "Astrocyte", "Microglia",
    "Oligodendrocyte", "OPC", "Endothelial", "Pericyte", "VSMC",
    "Ependymal", "Neuroblast", "NSC", "Macrophage", "T cell",
]

# Markers with elevated spillover / misallocation rates (Sun et al. 2025).
EXCLUDE_MARKERS = [
    "Gfap", "Crym", "Drd2", "Nr4a2", "Ighm", "Slc17a7", "Aldoc",
    "Adora2a", "Cd4", "C1ql3", "Stmn2", "Pvalb", "Thbs4", "Gja1",
    "Atp1a2", "C4b", "Drd1", "Lamp5", "Slc1a2", "Sparc", "Map1lc3a",
    "Tox", "Penk", "Gad2", "Chat", "Apoe", "Aqp4", "Sulf2", "Sox9",
    "Clu", "Tubb3", "Slc32a1", "Aldh1l1", "Spock2", "Nfic", "Olig1",
    "Flt1", "Pbx3", "Pdgfra", "Adamts3", "Tac1", "Cdh2", "Slc1a3",
    "Agpat3", "Fgfr3", "Msmo1", "Ntm", "Efnb2", "Apod", "Cd47",
    "Gad1", "Cdk5r1", "Cfl1", "Jak1", "Sst", "Sox2", "Dpp6", "Stub1",
    "Igf2", "Elovl5", "Fads2", "Trim2", "Syt11", "C1qa", "Npy", "Htt",
    "Pcsk1n", "Akt1", "Csf1r", "Igf1r", "Sox11", "Slc17a6", "Mtor",
    "C1qb", "Sod2", "Btg2", "Gpm6b", "Vcam1", "Nr2e1", "Parp1",
]


def preprocess_qc(input_h5ad: Path, output_h5ad: Path) -> sc.AnnData:
    """Load raw AnnData, apply 3-stage QC, write `output_h5ad`, return the AnnData."""
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    print(f"[qc] reading: {input_h5ad}")
    adata = sc.read_h5ad(str(input_h5ad))
    adata.var_names_make_unique()
    adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce").astype(np.float64)
    adata.obs["bbox_area"] = (
        (adata.obs["max_x"] - adata.obs["min_x"])
        * (adata.obs["max_y"] - adata.obs["min_y"])
    )

    adata = adata[adata.obs["celltype"].astype(str).isin(ALL_CELLTYPES)].copy()

    var_names = adata.var_names.astype(str)
    excluded = sorted(set(EXCLUDE_MARKERS) & set(var_names))
    if excluded:
        adata = adata[:, ~var_names.isin(excluded)].copy()

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    max_counts = float(adata.obs["total_counts"].quantile(0.999))
    max_genes = float(adata.obs["n_genes_by_counts"].quantile(0.999))
    bbox = adata.obs["bbox_area"].to_numpy()
    bbox_lo = float(np.nanquantile(bbox, 0.001))
    bbox_hi = float(np.nanquantile(bbox, 0.999))
    bbox_ok = (~np.isfinite(bbox)) | ((bbox >= bbox_lo) & (bbox <= bbox_hi))
    keep = (
        (adata.obs["total_counts"] >= 40)
        & (adata.obs["total_counts"] <= max_counts)
        & (adata.obs["n_genes_by_counts"] >= 15)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & bbox_ok
    ).to_numpy()
    adata = adata[keep].copy()

    adata.uns["preprocess_info"] = dict(
        input_h5ad=str(input_h5ad),
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        excluded_markers=excluded,
    )
    print(f"[qc] writing: {output_h5ad} (n_obs={adata.n_obs}, n_vars={adata.n_vars})")
    adata.write_h5ad(str(output_h5ad), compression="gzip")
    return adata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=Path, default=DEFAULT_RAW_DATA,
                   help=f"Raw .h5ad input. Default: {DEFAULT_RAW_DATA}")
    p.add_argument("--output", type=Path, required=True, help="QC .h5ad output.")
    p.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED,
                   help="Output dir for per-celltype .h5ad files.")
    args = p.parse_args()

    if args.output.exists():
        print(f"[qc] {args.output} exists - loading instead of re-running QC.")
        adata = sc.read_h5ad(str(args.output))
    else:
        adata = preprocess_qc(args.input, args.output)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    for ct in ALL_CELLTYPES:
        out = args.processed_dir / f"{safe_name(ct)}.h5ad"
        if out.exists():
            print(f"[extract] {ct!r}: exists, skipping ({out})")
            continue
        sub = adata[adata.obs["celltype"].astype(str) == ct].copy()
        sub.write_h5ad(str(out), compression="gzip")
        n_mice = sub.obs["mouse_id"].nunique() if "mouse_id" in sub.obs else -1
        print(f"[extract] {ct}: {sub.n_obs} cells, {sub.n_vars} genes, "
              f"{n_mice} mice -> {out}")


if __name__ == "__main__":
    main()
