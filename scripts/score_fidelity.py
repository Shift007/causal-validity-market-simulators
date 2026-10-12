#!/usr/bin/env python
"""Distributional fidelity table from saved walk-forward generations (CPU only).

Reads ``pooled/real.npy`` (N, T, C) and a generated array (N, S, T, C) from
``wf_eval`` runs and reports, for the real data and each model:

  Marginal   mean, std (bps), skewness, excess kurtosis of each window's pooled
             T x C returns, averaged over windows
  W1         Wasserstein-1 distance per asset (all windows pooled), averaged
             over assets, bps
  Temporal   lag-1 autocorrelation of r and of |r| (volatility clustering),
             leverage corr(r_t, |r_{t+1}|), per window and asset, averaged
  Cross      mean pairwise cross-asset correlation within a window, averaged

``--pooling sample`` (default) scores every generated sample as its own window,
like the real data. ``--pooling window`` pools the S samples of a window before
taking moments; it reproduces the table of the submitted paper but inflates the
higher moments of the generated rows relative to the real row.

Usage (from the repository root):
    python scripts/score_fidelity.py --real <run>/pooled/real.npy \
        --model "Temporal=<run>/pooled/diff_synth.npy" --model "MacroVAE=<run>/pooled/mvae_synth.npy" \
        --out results/paper_scores/fidelity_real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew, wasserstein_distance

BPS = 1e4


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two (M, L) arrays; NaN where a row is constant."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def window_stats(x: np.ndarray, moment_blocks: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-window statistics of windows ``x`` (M, T, C). Returns arrays over windows."""
    M, T, C = x.shape
    blocks = x.reshape(M, T * C) if moment_blocks is None else moment_blocks
    series = np.transpose(x, (0, 2, 1)).reshape(M * C, T)                 # one row per (window, asset)

    def per_window(v):
        return np.nanmean(v.reshape(M, C), axis=1)

    cc = np.stack([np.corrcoef(w, rowvar=False) for w in x])              # (M, C, C)
    off = ~np.eye(C, dtype=bool)
    return {
        "std_bps": BPS * blocks.std(axis=1),
        "skew": skew(blocks, axis=1),
        "kurt_excess": kurtosis(blocks, axis=1),
        "acf1_r": per_window(_corr_rows(series[:, :-1], series[:, 1:])),
        "acf1_abs_r": per_window(_corr_rows(np.abs(series[:, :-1]), np.abs(series[:, 1:]))),
        "leverage": per_window(_corr_rows(series[:, :-1], np.abs(series[:, 1:]))),
        "cross_corr": np.nanmean(cc[:, off].reshape(M, -1), axis=1),
    }


def score(real: np.ndarray, synth: np.ndarray | None, pooling: str) -> dict:
    """Fidelity row for the real data (``synth`` None) or for one model."""
    if synth is None:
        x, blocks = real, None
    else:
        N, S, T, C = synth.shape
        x = synth.reshape(N * S, T, C)
        blocks = synth.reshape(N, S * T * C) if pooling == "window" else None
    stats = window_stats(x, blocks)
    row = {"mean_bps": {"value": float(BPS * x.mean())}}
    for k, v in stats.items():
        v = v[np.isfinite(v)]
        row[k] = {"value": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
    if synth is not None:
        C = real.shape[2]
        row["w1_bps"] = {"value": float(BPS * np.mean([
            wasserstein_distance(real[:, :, c].ravel(), synth[..., c].ravel()) for c in range(C)]))}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--pooling", choices=["sample", "window"], default="sample")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    real = np.load(args.real).astype(np.float64)
    report = {"pooling": args.pooling, "real": str(args.real), "n_windows": int(real.shape[0]), "rows": {}}
    report["rows"]["Real"] = score(real, None, args.pooling)
    for spec in args.model:
        label, path = spec.split("=", 1)
        synth = np.load(path).astype(np.float64)
        report["rows"][label] = {"path": path, "n_samples": int(synth.shape[1]), **score(real, synth, args.pooling)}

    cols = ["mean_bps", "std_bps", "skew", "kurt_excess", "w1_bps", "acf1_r", "acf1_abs_r", "leverage", "cross_corr"]
    print(f"{'':<26}" + "".join(f"{c:>12}" for c in cols))
    for label, row in report["rows"].items():
        print(f"{label:<26}" + "".join(
            f"{row[c]['value']:>12.3f}" if c in row else f"{'--':>12}" for c in cols))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {args.out}  (pooling={args.pooling})")


if __name__ == "__main__":
    main()
