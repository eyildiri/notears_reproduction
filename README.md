# NOTEARS reproduction project

This repository contains a Python reproduction framework for **DAGs with NO TEARS: Continuous Optimization for Structure Learning** (Zheng et al., NeurIPS 2018).

It is designed for an INFO-H512-style project: software implementation, experimental evaluation, baselines, metrics, plots, and reproducibility instructions.

## What is included

Core implementation:

- Linear NOTEARS from scratch.
- Smooth acyclicity constraint: `h(W) = trace(expm(W * W)) - d`.
- Augmented Lagrangian outer loop.
- L-BFGS-B inner optimization.
- L1 regularization through positive/negative variable splitting.
- Final hard-thresholding of small edge weights.

Experiments:

- Synthetic DAG generation:
  - Erdős-Rényi (`ER`)
  - Scale-free (`SF`)
- SEM data generation:
  - Gaussian noise
  - Exponential noise
  - Gumbel noise
- Experimental grid compatible with the paper:
  - `d in {10, 20, 50, 100}`
  - `n in {20, 1000}`
  - ER-2 and SF-4 style edge densities
  - multiple random seeds
- Metrics:
  - SHD
  - FDR
  - TPR
  - FPR
  - number of predicted edges
  - runtime
  - final acyclicity score
- Baselines:
  - simple correlation-threshold baseline
  - optional PC via `causal-learn`
  - optional GES via `causal-learn`
  - optional DirectLiNGAM via `lingam`
- Real-data hook:
  - Sachs experiment pipeline, if you provide `data/sachs.csv`
- Tiny exact-search sanity check:
  - brute-force topological-order benchmark for very small `d`

## Important limitation

This repository gives you a serious, executable reproduction framework. However, a literal full reproduction of every line of the paper requires external tools that are not reimplemented here:

- **FGS from Tetrad**: the paper uses the Fast Greedy Search implementation from Ramsey et al. This is Java/Tetrad-based. This repo provides an optional Python GES wrapper via `causal-learn` as a practical substitute.
- **GOBNILP**: the paper uses GOBNILP for exact global minimizer comparisons. This repo includes a small exhaustive-order sanity check, but not the full integer-programming solver.
- **Exact Sachs benchmark choice**: different libraries sometimes encode slightly different Sachs consensus graphs. The included default is transparent and can be replaced.

For the course project, this is usually enough: you can claim a **partial reproduction of the main synthetic experiments**, with an implementation and evaluation pipeline.

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Optional baselines:

```bash
pip install -r requirements-full.txt
```

## Quick test

```bash
python scripts/smoke_test.py
```

You should see outer augmented-Lagrangian iterations and final metrics.

## Run a small experiment

```bash
python scripts/run_synthetic.py --quick
python scripts/make_figures.py --metrics-csv results/synthetic/synthetic_metrics.csv
```

## Run a paper-style synthetic grid

This can take time, especially for `d=100`.

```bash
python scripts/run_synthetic.py \
  --d-values 10 20 50 100 \
  --n-values 20 1000 \
  --seeds 10 \
  --methods notears notears-l1 corr \
  --lambda1 0.1 \
  --threshold 0.3 \
  --save-matrices \
  --n-jobs 6

python scripts/make_figures.py --metrics-csv results/synthetic/synthetic_metrics.csv
```
(n-jobs is for parallelism (default = 4))
Optional with extra baselines:

```bash
python scripts/run_synthetic.py \
  --d-values 10 20 \
  --n-values 20 1000 \
  --seeds 3 \
  --methods notears notears-l1 corr pc ges lingam
```

## Run Sachs experiment

Place the dataset at:

```text
data/sachs.csv
```

Then run:

```bash
python scripts/run_sachs.py --data-path data/sachs.csv
```

## Tiny exact-search comparison

```bash
python scripts/run_tiny_exact.py --d 6 --n 100
```

This is only a sanity check and not a replacement for GOBNILP.

## Repository structure

```text
notears_reproduction/
  notears_repro/
    notears.py       # core NOTEARS implementation
    data.py          # synthetic DAG + SEM generators
    metrics.py       # SHD/FDR/TPR/FPR metrics
    baselines.py     # corr/PC/GES/LiNGAM wrappers
    experiments.py   # synthetic experiment runner
    plots.py         # figures and summary tables
    realdata.py      # Sachs pipeline
    exact.py         # tiny exact-order sanity check
    utils.py
  scripts/
    smoke_test.py
    run_synthetic.py
    make_figures.py
    run_sachs.py
    run_tiny_exact.py
  requirements.txt
  requirements-full.txt
  README.md
```

## Suggested project framing

A realistic research question:

> How well does NOTEARS recover the structure of synthetic causal DAGs under different sample sizes, graph densities, and noise distributions?

Recommended experiments for a 4-page report:

- `d = 10, 20, 50`
- `n = 20, 1000`
- graph types: `ER`, `SF`
- noise: at least Gaussian, optionally Exponential and Gumbel
- methods: NOTEARS, NOTEARS-L1, simple baseline, optionally PC/GES
- metrics: SHD, FDR, TPR, runtime

## Notes on matrix convention

Throughout this repo:

```text
W[i, j] != 0 means i -> j
```

The SEM is generated as:

```text
X = XW + Z
X = Z (I - W)^(-1)
```

