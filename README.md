# stGP: Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs

This repository contains the code for **stGP** (spatiotemporal Gene Programs).

## Introduction

stGP is a statistical framework for identifying interpretable cell-type-specific spatiotemporal gene programs (stGPs) from multi-sample spatiotemporal transcriptomic data measured across biological time by deciphering temporal trajectory and dynamic spatial patterns.

stGP's effectiveness relies on our innovations in the integration of Gaussian process priors and interpretable matrix factorization:

- stGP represents gene expression within each cell type as a small set of latent programs with non-negative gene loadings, making each program interpretable as a weighted gene set shared across samples. Variance components quantify the relative contributions of time and space to each program.
- stGP decomposes per-cell program activity into a sample-level temporal component that captures coordinated responses over biological time (e.g., age or stage), and a within-section spatial component that characterizes dynamic program deployment across tissue coordinates without requiring cross-section registration.
- For multi-program inference, stGP adopts a blockwise backfitting scheme that sequentially extracts rank-1 components from residuals, with automatic model selection to determine the number of programs.

<p align="center">
  <img src="FigureReproducing/Fig1_overview.png" width="85%" alt="Overview" />
</p>


## Installation

To install the released `stgp` package from PyPI into your current Python environment:

```shell
pip install stgp
```

To reproduce the notebook from the GitHub repository, you can also install the stGP environment from GitHub:

```bash
git clone https://github.com/YangLabHKUST/stGP.git
cd stGP
conda env create -f stGP.yml
conda activate stGP
pip install -e .
```

Normally the installation time will be about twenty minutes. We have tested our package on Linux (Ubuntu 22.04.5 LTS).

## Tutorials and Reproducibility

The tutorials for using stGP and codes for reproducing the analysis results presented in our paper are available on the tutorial website (<https://stgp-tutorial.readthedocs.io>).

- [Human aging DLPFC contating the benchmarking study and niche validation](https://stgp-tutorial.readthedocs.io/en/latest/tutorials/human_aging_dlpfc/index.html)
- [Mouse aging brain to discover the proximity effect](https://stgp-tutorial.readthedocs.io/en/latest/tutorials/mouse_aging_brain/index.html)
- [Mouse injured kidney with paired biological replicates](https://stgp-tutorial.readthedocs.io/en/latest/tutorials/mouse_injured_kidney/index.html)

You may also use the notebooks and codes in this repository to reproducing the figures in the manuscript.

The full benchmarking codes and results are released at a separate repository: <https://github.com/Jamesyu420/stgp-reproduce>.

## Data Availability

The aging mouse brain MERFISH dataset can be accessed from Zenodo at (<https://doi.org/10.5281/zenodo.13883177>). The aging human brain DLPFC MERFISH dataset can be obtained from (<https://publications.wenglab.org/SomaMut/>). The mouse kidney injury and repair Xenium dataset is available from (<https://doi.org/10.6084/m9.figshare.28761695.v1>).

## Reference

If you find `stGP` or any of the source code in this repository useful for your work, please cite:
> Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs with stGP.
> Baichen Yu, Ziyue Tan, Xiaomeng Wan, Hansheng Wang, and Can Yang.
> Preprint at Biorxiv, 2026.
> https://doi.org/10.64898/2026.07.03.736035

## Development

The software is developed and maintained by [Baichen Yu](mailto:mabyu@ust.hk).

## Contact

Please feel free to contact [Baichen Yu](mailto:mabyu@ust.hk) or [Prof. Can Yang](mailto:macyang@ust.hk) if any inquiries.