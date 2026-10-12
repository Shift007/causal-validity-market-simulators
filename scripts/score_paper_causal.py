#!/usr/bin/env python
"""Re-score the paper's causal tables from saved effect arrays, with window bootstrap.

CPU-only, no model is loaded. For each counterfactual run it reads
``theory_effects.npy`` and the model effects ``(N_windows, K_shocks, T, C)`` and
reports, with a bootstrap over windows:

  Table 3  Dir%      mean over shocks of the per-cell sign agreement
           PEHE      mean over shocks of sqrt(mean squared effect error), bps
           ATE err   mean over shocks of |mean(effect) - mean(theory)|, bps
  Table 4  A         asymmetry index, mean over shock pairs and assets of
                     (|S+| - |S-|) / (|S+| + |S-|), S = cumulative mean response
           A_pooled  same with sums pooled over assets and pairs before the ratio
           |S|       mean cumulative response magnitude, bps

Usage (from the repository root):
    python scripts/score_paper_causal.py --results results --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CF = "counterfactual_synthetic"
TABLE3 = [  # (label, branch, run, effects file)
    ("Temporal 1.4M / ATE", "synthetic_complex_filmonly", "run_20260724_150920", "diff_effects.npy"),
    ("Temporal 1.4M / ITE", "synthetic_complex_filmonly", "run_20260725_183406", "diff_effects.npy"),
    ("Snapshot 1.4M / ATE", "synthetic_complex_snapfilm", "run_20260725_180154", "diff_effects.npy"),
    ("Snapshot 1.4M / ITE", "synthetic_complex_snapfilm", "run_20260725_184608", "diff_effects.npy"),
    ("MacroVAE 5.7M / ITE", "synthetic_complex", "run_20260724_144849", "mvae_effects.npy"),
]
TABLE4 = [
    ("Theory", "synthetic_complex_nl_filmonly", "run_20260724_151522", "theory_effects.npy"),
    ("Temporal 1.4M", "synthetic_complex_nl_filmonly", "run_20260724_151522", "diff_effects.npy"),
    ("MacroVAE 5.7M", "synthetic_complex_nl", "run_20260724_145655", "mvae_effects.npy"),
]
BPS = 1e4


def table3_stats(eff, theory, idx):
    """Dir%, PEHE, ATE err for each bootstrap row of window indices ``idx`` (B, N)."""
    correct = (np.sign(eff) == np.sign(theory)).mean(axis=(2, 3))      # (N, K)
    sqerr = ((eff - theory) ** 2).mean(axis=(2, 3))                    # (N, K)
    gap = eff.mean(axis=(2, 3)) - theory.mean(axis=(2, 3))             # (N, K)
    dir_pct = 100.0 * correct[idx].mean(axis=1).mean(axis=1)           # (B,)
    pehe = BPS * np.sqrt(sqerr[idx].mean(axis=1)).mean(axis=1)
    ate = BPS * np.abs(gap[idx].mean(axis=1)).mean(axis=1)
    return {"dir_pct": dir_pct, "pehe_bps": pehe, "ate_err_bps": ate}


def asym_stats(eff, shocks, idx):
    """Asymmetry index variants for each bootstrap row of window indices."""
    cum = eff.sum(axis=2)                                              # (N, K, C)
    mean_cum = np.abs(cum[idx].mean(axis=1))                           # (B, K, C)
    shocks = np.asarray(shocks, float)
    pairs = [(int(np.where(np.isclose(shocks, d))[0][0]),
              int(np.where(np.isclose(shocks, -d))[0][0]))
             for d in sorted(set(np.abs(shocks)))]
    cp = np.stack([mean_cum[:, p] for p, _ in pairs], axis=1)          # (B, P, C)
    cn = np.stack([mean_cum[:, n] for _, n in pairs], axis=1)
    a = ((cp - cn) / (cp + cn + 1e-12)).mean(axis=(1, 2))
    a_pooled = (cp.sum(axis=(1, 2)) - cn.sum(axis=(1, 2))) / (cp.sum(axis=(1, 2)) + cn.sum(axis=(1, 2)))
    mag = BPS * mean_cum.mean(axis=(1, 2))
    return {"A": a, "A_pooled": a_pooled, "cum_response_bps": mag}


def summarise(stats):
    out = {}
    for key, v in stats.items():
        lo, hi = np.percentile(v[1:], [2.5, 97.5])
        out[key] = {"value": float(v[0]), "se": float(v[1:].std(ddof=1)),
                    "ci95": [float(lo), float(hi)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="directory holding <branch>/counterfactual_synthetic/<run>/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    report = {"n_boot": args.n_boot, "seed": args.seed, "results_root": str(args.results),
              "table3": {}, "table4": {}}

    def load(branch, run, name):
        d = args.results / branch / CF / run
        return np.load(d / name).astype(np.float64), d

    def boot_idx(n):
        idx = rng.integers(0, n, size=(args.n_boot + 1, n))
        idx[0] = np.arange(n)                      # row 0 = the full sample
        return idx

    print(f"{'Table 3':<22}{'Dir%':>18}{'PEHE bps':>18}{'ATE err bps':>18}")
    for label, branch, run, name in TABLE3:
        eff, d = load(branch, run, name)
        theory = np.load(d / "theory_effects.npy").astype(np.float64)
        s = summarise(table3_stats(eff, theory, boot_idx(eff.shape[0])))
        report["table3"][label] = {"run_dir": f"{branch}/{CF}/{run}", "effects": name,
                                   "n_windows": int(eff.shape[0]), **s}
        print(f"{label:<22}" + "".join(
            f"{s[k]['value']:>10.2f} ±{s[k]['se']:<6.2f}" for k in ("dir_pct", "pehe_bps", "ate_err_bps")))

    print(f"\n{'Table 4':<16}{'A (mean of ratios)':>30}{'A (pooled)':>30}{'|S| bps':>12}")
    for label, branch, run, name in TABLE4:
        eff, d = load(branch, run, name)
        shocks = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"]
        s = summarise(asym_stats(eff, shocks, boot_idx(eff.shape[0])))
        report["table4"][label] = {"run_dir": f"{branch}/{CF}/{run}", "effects": name,
                                   "n_windows": int(eff.shape[0]), **s}
        print(f"{label:<16}" + "".join(
            f"{s[k]['value']:>+9.3f} [{s[k]['ci95'][0]:+.3f}, {s[k]['ci95'][1]:+.3f}]" for k in ("A", "A_pooled"))
            + f"{s['cum_response_bps']['value']:>12.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
