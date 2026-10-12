#!/usr/bin/env python
"""Causal figures of the paper, drawn from explicitly named counterfactual runs (CPU only).

  trajectories   cumulative return of an equal-weighted index over ten consecutive
                 32-day blocks under a +/-1.5 ppt CPI shock: realised path, true
                 average response, and each model's response added to the path
  dose_response  sign-aligned mean cumulative response against the shock size on
                 the regime-switching environment

Every series is read from the run directory given on the command line, so the
figure and the tables come from the same arrays.

Usage (from code/):
    python scripts/fig_paper_causal.py trajectories --temporal <run> --snapshot <run> --macrovae <run> --out fig.png
    python scripts/fig_paper_causal.py dose_response --temporal <run> --snapshot <run> --macrovae <run> --out fig.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

C_REAL, C_TRUTH = "black", "0.45"
C_TEMP, C_SNAP, C_MVAE = "#c0392b", "#1f6fb2", "#2e8b57"
N_BLOCKS = 10


def load(run: Path, name: str) -> np.ndarray:
    return np.load(run / name).astype(np.float64)


def same_windows(runs: list[Path]) -> None:
    ref = np.load(runs[0] / "window_meta.npz", allow_pickle=True)["window_indices"]
    for r in runs[1:]:
        if not np.array_equal(ref, np.load(r / "window_meta.npz", allow_pickle=True)["window_indices"]):
            raise SystemExit(f"{r} was evaluated on different windows than {runs[0]}")


def pick_consecutive(t_bnd: np.ndarray, n: int, tlen: int) -> list[int]:
    """First n windows, in time order, whose target blocks do not overlap."""
    order = np.argsort(t_bnd)
    chosen, last = [int(order[0])], t_bnd[order[0]]
    for i in order[1:]:
        if t_bnd[i] >= last + tlen:
            chosen.append(int(i))
            last = t_bnd[i]
        if len(chosen) == n:
            break
    return chosen


def trajectories(args) -> dict:
    same_windows([args.temporal, args.snapshot, args.macrovae])
    real = load(args.temporal, "real_targets.npy")                      # (N, T, C)
    effects = {"truth": load(args.temporal, "theory_effects.npy"),      # (N, K, T, C)
               "temp": load(args.temporal, "diff_effects.npy"),
               "snap": load(args.snapshot, "diff_effects.npy"),
               "mvae": load(args.macrovae, "mvae_effects.npy")}
    shocks = json.loads((args.temporal / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"]
    t_bnd = np.load(args.temporal / "window_meta.npz", allow_pickle=True)["t_boundaries"]
    T = real.shape[1]
    blocks = pick_consecutive(np.asarray(t_bnd), N_BLOCKS, T)

    def index_path(eff, k):
        x = real[blocks] if eff is None else real[blocks] + eff[blocks, k]
        return 100.0 * np.cumsum(x.mean(axis=2).reshape(-1))            # equal-weighted index, %

    x = np.arange(len(blocks) * T)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)
    handles = labels = None
    for ax, k in zip(axes, (int(np.argmax(shocks)), int(np.argmin(shocks)))):
        ax.grid(alpha=0.3, zorder=0)
        for b in range(1, len(blocks)):
            ax.axvline(b * T, color="0.93", lw=0.7, zorder=0)
        ax.plot(x, index_path(None, k), color=C_REAL, lw=2.4, zorder=6, label="Realised path")
        ax.plot(x, index_path(effects["truth"], k), color=C_TRUTH, lw=2.2, ls=(0, (5, 2)), zorder=5,
                label="True average response")
        ax.plot(x, index_path(effects["temp"], k), color=C_TEMP, lw=1.9, zorder=4, label="Temporal 1.4M")
        ax.plot(x, index_path(effects["snap"], k), color=C_SNAP, lw=1.8, zorder=3, label="Snapshot 1.4M")
        ax.plot(x, index_path(effects["mvae"], k), color=C_MVAE, lw=1.7, zorder=2, label="MacroVAE 5.7M")
        ax.set_title(f"CPI shock of {100 * shocks[k]:+.1f} ppt", fontsize=12)
        ax.set_xlabel("trading day  (10 consecutive blocks)", fontsize=10)
        ax.set_xlim(0, len(x) * 1.01)
        ax.tick_params(labelsize=9)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("cumulative index return (%)", fontsize=11)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5,
               fontsize="small", frameon=False, handlelength=2.0, columnspacing=1.5)
    fig.tight_layout()
    return {"fig": fig, "blocks": blocks}


def dose_response(args) -> dict:
    from synthetic_dgp_complex import _DEFAULT_PARAMS_COMPLEX as P
    sign = np.sign(np.asarray(P["assets"]["B"], float)[:, 0])           # sign of each asset's CPI loading
    shocks = np.asarray(json.loads((args.temporal / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"])

    def dose(eff):
        return (eff.sum(axis=2) * sign[None, None, :]).mean(axis=(0, 2))

    series = [("Truth (closed form)", dose(load(args.temporal, "theory_effects.npy")), "k", "o-", 2.3),
              ("Temporal 1.4M", dose(load(args.temporal, "diff_effects.npy")), C_TEMP, "s-", 1.9),
              ("Snapshot 1.4M", dose(load(args.snapshot, "diff_effects.npy")), C_SNAP, "s-", 1.8),
              ("MacroVAE 5.7M", dose(load(args.macrovae, "mvae_effects.npy")), C_MVAE, "s-", 1.8)]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    for label, y, col, fmt, lw in series:
        ax.plot(100 * shocks, y, fmt, color=col, lw=lw, ms=4, label=label)
    ax.set_xlabel("CPI shock (ppt)", fontsize=10)
    ax.set_ylabel("sign-aligned cumulative return response", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return {"fig": fig, "dose": {label: y.tolist() for label, y, *_ in series}, "shocks": shocks.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["trajectories", "dose_response"])
    ap.add_argument("--temporal", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--macrovae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res = trajectories(args) if args.figure == "trajectories" else dose_response(args)
    fig = res.pop("fig")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    meta = {"figure": args.figure, "temporal": str(args.temporal), "snapshot": str(args.snapshot),
            "macrovae": str(args.macrovae), **res}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[fig_paper_causal] wrote {args.out}")


if __name__ == "__main__":
    main()
