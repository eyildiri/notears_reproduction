from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from notears_repro.experiments import quick_synthetic, run_synthetic_experiment
from notears_repro.plots import save_summary_tables


def parse_args():
    p = argparse.ArgumentParser(description="Run synthetic NOTEARS reproduction experiments.")
    p.add_argument("--out-dir", default="results/synthetic", help="Output directory")
    p.add_argument("--quick", action="store_true", help="Run a tiny debug experiment")
    p.add_argument("--d-values", nargs="+", type=int, default=[10, 20, 50, 100])
    p.add_argument("--n-values", nargs="+", type=int, default=[20, 1000])
    p.add_argument("--seeds", type=int, default=10, help="Number of seeds, starting at 0")
    p.add_argument("--methods", nargs="+", default=["notears", "notears-l1", "corr"],
                   help="Methods: notears notears-l1 corr pc ges lingam")
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--save-matrices", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4, help="Number of parallel jobs to run. Default 1 (no parallelism).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.quick:
        df = quick_synthetic(out_dir, n_jobs=args.n_jobs)
    else:
        df = run_synthetic_experiment(
            out_dir=out_dir,
            d_values=args.d_values,
            n_values=args.n_values,
            seeds=tuple(range(args.seeds)),
            methods=args.methods,
            lambda1=args.lambda1,
            w_threshold=args.threshold,
            save_matrices=args.save_matrices,
            n_jobs=args.n_jobs,
        )
    save_summary_tables(df, out_dir)
    print(df.head())
    print(f"Saved metrics to {out_dir}")


if __name__ == "__main__":
    main()
