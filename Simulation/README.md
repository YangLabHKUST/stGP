# stGP Simulation

This folder runs the main simulation benchmark for spatiotemporal gene program
recovery. The driver script is `simu_gaussian.py`: it simulates count data,
fits stGP and baseline methods, saves cached results, and regenerates the
summary figures.

## Main Workflow

`simu_gaussian.py` has four stages:

- `--stage data`: simulate each replicate once and save it under
  `results/simu1/datasets/`.
- `--stage fit`: load the saved replicates, fit the selected methods, and cache
  one result file per method and replicate.
- `--stage plot`: load the cached datasets/results and regenerate CSV summaries
  and figures. This does not refit methods.
- `--stage all`: run data, fit, and plot in one command.

Use the same `--reps` value for plotting that was used when fitting. The cached
results in this folder use 50 replicates.


Use the `stGP` environment for the non-Popari methods:

```bash
conda activate stGP
python3 simu_gaussian.py --stage data --reps 50
python3 simu_gaussian.py --stage fit --methods auto --reps 50
```

If Popari should be included, fit it separately in the Popari environment:

```bash
conda activate Popari
python3 simu_gaussian.py --stage fit --methods popari --reps 50
```

Then return to `stGP` and regenerate the combined plots:

```bash
conda activate stGP
MPLBACKEND=Agg python3 simu_gaussian.py --stage plot --methods all --reps 50
```

## Method Notes

- `--methods auto` skips Popari when the Popari package is not available.
- `--methods all` is useful after all method caches already exist.
- stGP, PCA, SpatialPCA, and MEFISTO use centered log-normalized data.
- NMF and Popari use log-normalized data.
- STAMP uses raw count data.
