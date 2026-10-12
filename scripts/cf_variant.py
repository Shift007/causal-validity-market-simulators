#!/usr/bin/env python
"""Run a variant of the synthetic counterfactual evaluation in its own results branch.

Wraps ``counterfactual_synthetic.run`` with overrides of the evaluation protocol
and writes to ``<results_dir>_<tag>``, so the runs behind the paper tables are
never mixed with robustness checks.

Overrides:
  --ate_samples M          latent draws per window for the diffusion "fresh" mode
  --window_pool test|all   out-of-sample windows only, or train+test
  --n_windows N            number of windows drawn from the pool
  --latent_mode            diffusion: fresh (ATE) or invert (ITE)
  --mvae_latent            MacroVAE: posterior mean or prior draws
  --mvae_prior_samples M   prior draws per window for the MacroVAE

Usage (from the repository root):
    python scripts/cf_variant.py --tag oos \
        --config configs/synthetic_complex_filmonly/baseline.yaml \
        --model diffusion --window_pool test --n_windows 447
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.counterfactual_synthetic import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True, help="suffix of the results branch")
    ap.add_argument("--model", choices=["diffusion", "macrovae"], required=True)
    ap.add_argument("--ate_samples", type=int, default=None)
    ap.add_argument("--window_pool", choices=["test", "all"], default=None)
    ap.add_argument("--n_windows", type=int, default=None)
    ap.add_argument("--window_chunk", type=int, default=None)
    ap.add_argument("--latent_mode", choices=["fresh", "invert"], default="fresh")
    ap.add_argument("--mvae_latent", choices=["posterior", "prior"], default="posterior")
    ap.add_argument("--mvae_prior_samples", type=int, default=16)
    args = ap.parse_args()


    with open(ROOT / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cf = config["counterfactual_synthetic"]
    for key in ("ate_samples", "window_pool", "n_windows", "window_chunk"):
        if getattr(args, key) is not None:
            cf[key] = getattr(args, key)
    config["paths"]["results_dir"] = f'{config["paths"]["results_dir"]}_{args.tag}'

    run_dir = run(ROOT, config, model_filter=args.model, latent_mode=args.latent_mode,
                  mvae_latent=args.mvae_latent, mvae_prior_samples=args.mvae_prior_samples)
    print(f"[cf_variant] {args.tag} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
