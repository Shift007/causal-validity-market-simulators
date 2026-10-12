#!/usr/bin/env python
"""Generation and evaluation runs.

Usage (from the repository root):
    python scripts/generate.py wf_eval                  --config <yaml> --model diffusion|macrovae
    python scripts/generate.py counterfactual_synthetic --config <yaml> --model diffusion --latent_mode fresh|invert
    python scripts/generate.py counterfactual_synthetic --config <yaml> --model macrovae  --mvae_latent posterior|prior

wf_eval                   walk-forward out-of-sample generation, the input of the fidelity table
counterfactual_synthetic  interventional response on the synthetic environment against the exact truth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["wf_eval", "counterfactual_synthetic"])
    parser.add_argument("--config", type=str, required=True, help="experiment YAML under configs/")
    parser.add_argument("--model", type=str, default="both", choices=["both", "diffusion", "macrovae"])
    parser.add_argument(
        "--latent_mode", type=str, default="fresh", choices=["fresh", "invert"],
        help="counterfactual_synthetic: diffusion latent, fresh noise (ATE) or DDIM inversion (ITE).",
    )
    parser.add_argument(
        "--mvae_latent", type=str, default="posterior", choices=["posterior", "prior"],
        help="counterfactual_synthetic: MacroVAE latent, posterior mean or z ~ N(0, I).",
    )
    parser.add_argument(
        "--mvae_prior_samples", type=int, default=16,
        help="counterfactual_synthetic: prior draws per window when --mvae_latent prior.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    t0 = time.perf_counter()
    if args.task == "wf_eval":
        from src.generation.wf_eval import run
        run_dir = run(ROOT, config, model_filter=args.model)
    else:
        from src.generation.counterfactual_synthetic import run
        run_dir = run(ROOT, config, model_filter=args.model,
                      latent_mode=args.latent_mode, mvae_latent=args.mvae_latent,
                      mvae_prior_samples=args.mvae_prior_samples)
    print(f"\n  [OK] {args.task} -> {run_dir}  ({time.perf_counter() - t0:.0f}s)\n")


if __name__ == "__main__":
    main()
