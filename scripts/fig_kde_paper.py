#!/usr/bin/env python
"""Pooled return density on a log axis for the models of the fidelity table (CPU only).

Each model's out-of-sample returns are standardised per asset to unit variance,
so the comparison is about tail shape rather than scale, pooled over all
walk-forward windows and plotted against a standard Gaussian.

The four series are read from the same runs as the fidelity table:
    Real, MacroVAE   --real-run       (pooled/real.npy, pooled/mvae_synth.npy)
    Temporal         --temporal-run   (pooled/diff_synth.npy)
    Raw (no QT)      --raw-run        (pooled/diff_synth.npy)

Usage (from the repository root):
    python scripts/fig_kde_paper.py --real-run <run> --temporal-run <run> --raw-run <run> --out fig_kde_ablations.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kurtosis, norm

SUBSAMPLE = 200_000
GRID = np.linspace(-8.0, 8.0, 601)


def standardise_pool(a: np.ndarray) -> np.ndarray:
    """Per-asset z-score over every row, flattened. a: (..., C)."""
    flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
    sd = flat.std(axis=0, keepdims=True)
    return ((flat - flat.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0)).reshape(-1)


def kde_curve(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if z.size > SUBSAMPLE:
        z = rng.choice(z, SUBSAMPLE, replace=False)
    return gaussian_kde(z, bw_method=0.15)(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-run", type=Path, required=True)
    ap.add_argument("--temporal-run", type=Path, required=True)
    ap.add_argument("--raw-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    z = {
        "Real (test)": standardise_pool(np.load(args.real_run / "pooled" / "real.npy")),
        "Temporal 1.4M": standardise_pool(np.load(args.temporal_run / "pooled" / "diff_synth.npy")),
        "MacroVAE 9.0M": standardise_pool(np.load(args.real_run / "pooled" / "mvae_synth.npy")),
        "Temporal 1.4M (Raw)": standardise_pool(np.load(args.raw_run / "pooled" / "diff_synth.npy")),
    }
    stats = {k: {"n": int(v.size), "pooled_excess_kurtosis": float(kurtosis(v)),
                 "tail_mass_abs_z_ge_3": float((np.abs(v) >= 3).mean())} for k, v in z.items()}
    for k, s in stats.items():
        print(f"[kde] {k:<22} n={s['n']:>10,}  excess kurt={s['pooled_excess_kurtosis']:6.2f}"
              f"  P(|z|>=3)={100 * s['tail_mass_abs_z_ge_3']:.3f}%")

    style = {  # label -> (colour, width, linestyle, z-order)
        "Temporal 1.4M (Raw)": ("0.35", 1.6, (0, (1, 1)), 2),
        "MacroVAE 9.0M": ("#2e8b57", 1.8, "-", 3),
        "Temporal 1.4M": ("#c0392b", 2.0, "-", 4),
        "Real (test)": ("black", 2.2, "-", 5),
    }
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for s in (-1, 1):
        ax.axvspan(s * 3, s * 8, color="0.93", zorder=0)
    ax.plot(GRID, norm.pdf(GRID), color="0.72", lw=1.1, label="Gaussian $N(0,1)$", zorder=1)
    for label, (col, lw, ls, zo) in style.items():
        ax.plot(GRID, kde_curve(z[label], rng), color=col, lw=lw, ls=ls, label=label, zorder=zo)
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 1.0)
    ax.set_xlim(-8, 8)
    ax.text(0.0, 2.7e-5, "tail region  ($|z|\\geq 3$)", ha="center", fontsize=8, color="0.5")
    ax.set_xlabel("standardised log-return  (per-asset $z$-score)", fontsize=10)
    ax.set_ylabel("density (log scale)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), prop={"size": 7},
              frameon=False, handlelength=2.0, labelspacing=0.4)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")   # format from the suffix (.pdf or .png)
    plt.close(fig)
    args.out.with_suffix(".json").write_text(json.dumps({
        "real_run": str(args.real_run), "temporal_run": str(args.temporal_run),
        "raw_run": str(args.raw_run), "series": stats}, indent=2), encoding="utf-8")
    print(f"[kde] wrote {args.out}")


if __name__ == "__main__":
    main()
