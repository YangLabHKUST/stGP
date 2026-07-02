# stGP: Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs

## Introduction

stGP is a statistical framework for identifying interpretable cell-type-specific spatiotemporal gene programs (stGPs) from multi-sample spatiotemporal transcriptomic data measured across biological time by deciphering temporal trajectory and dynamic spatial patterns.

stGP's effectiveness relies on our innovations in the integration of Gaussian process priors and interpretable matrix factorization:

- stGP represents gene expression within each cell type as a small set of latent programs with non-negative gene loadings, making each program interpretable as a weighted gene set shared across samples. Variance components quantify the relative contributions of time and space to each program.
- stGP decomposes per-cell program activity into a sample-level temporal component that captures coordinated responses over biological time (e.g., age or stage), and a within-section spatial component that characterizes dynamic program deployment across tissue coordinates without requiring cross-section registration.
- For multi-program inference, stGP adopts a blockwise backfitting scheme that sequentially extracts rank-1 components from residuals, with automatic model selection to determine the number of programs.


## Reference

If you find `stGP` useful for your work, please cite:
> Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs with stGP.
> Baichen Yu, Ziyue Tan, Xiaomeng Wan, Hansheng Wang, and Can Yang.
> Working paper, 2026.

## Development

The software is developed and maintained by [Baichen Yu](mailto:mabyu@ust.hk).

## Contact

Please feel free to contact [Baichen Yu](mailto:mabyu@ust.hk) or [Prof. Can Yang](mailto:macyang@ust.hk) if any inquiries.