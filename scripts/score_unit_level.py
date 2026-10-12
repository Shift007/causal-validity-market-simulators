#!/usr/bin/env python
"""Score saved counterfactual effects against the exact unit-level counterfactual (CPU only).

The synthetic environment is re-simulated from its seed, which gives the macro
path, the conditional variances and the standardised innovations of every day.
For each evaluation window and shock the unit-level counterfactual holds the
macro innovations u and the return innovations z fixed and re-simulates the
target block after moving the macro state at the window boundary:

    m'_{T+k}  = m_{T+k} + Phi^k (delta * w)                    k = 1..horizon
    h'_t      = omega + alpha * eps'_{t-1}^2 + gamma * dCPI'_t + delta_g * h'_{t-1}
    r'_t      = mu + load(m'_t) + z_t * sqrt(h'_t)

The unit-level effect is r' - r. It is compared with the average effect used in
the paper (``theory_effects.npy``) and with the model effects of each run.

Usage (from the repository root):
    python scripts/score_unit_level.py --config <yaml> --run <label>=<run_dir>:<effects.npy> ... --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.synthetic_dgp_complex import (  # noqa: E402
    ComplexSyntheticDGP, _merge, _regime_spec, _DEFAULT_PARAMS_COMPLEX)

BPS = 1e4


def unit_effects(dgp, p, t_bnd, shocks, w, horizon):
    """Exact unit-level effects, (N, K, horizon, C)."""
    mp, ap = p["macro"], p["assets"]
    Phi = np.asarray(mp["Phi"], float)
    B = np.asarray(ap["B"], float)
    omega, alpha = np.asarray(ap["omega"], float), np.asarray(ap["alpha"], float)
    gamma, delta = np.asarray(ap["gamma"], float), np.asarray(ap["delta"], float)
    cpi = int(p["cpi_idx"])
    r_on, r_idx, r_thr, B_lo, B_hi = _regime_spec(p)
    m, h, eps = dgp._m, dgp._h, dgp._eps
    z = eps / np.sqrt(h)

    def load(mm):
        if r_on:
            return np.where((mm[:, r_idx] > r_thr)[:, None], mm @ B_hi.T, mm @ B_lo.T)
        return mm @ B.T

    out = np.zeros((len(t_bnd), len(shocks), horizon, B.shape[0]))
    for wi, T in enumerate(t_bnd):
        days = np.arange(T + 1, T + 1 + horizon)
        base_load = load(m[days])
        for si, s in enumerate(shocks):
            d = s * w
            m_cf = np.empty((horizon, len(d)))
            prev_m, prev_h, prev_eps = m[T] + d, h[T], eps[T]
            for k, t in enumerate(days):
                d = Phi @ d
                m_cf[k] = m[t] + d
                h_cf = np.maximum(omega + alpha * prev_eps ** 2
                                  + gamma * (m_cf[k, cpi] - prev_m[cpi]) + delta * prev_h, 1e-12)
                eps_cf = z[t] * np.sqrt(h_cf)
                out[wi, si, k] = eps_cf - eps[t]
                prev_m, prev_h, prev_eps = m_cf[k], h_cf, eps_cf
            out[wi, si] += load(m_cf) - base_load
    return out


def score(eff, truth):
    dir_pct = 100.0 * (np.sign(eff) == np.sign(truth)).mean()
    pehe = BPS * np.sqrt(((eff - truth) ** 2).mean(axis=(0, 2, 3))).mean()
    return {"dir_pct": float(dir_pct), "pehe_bps": float(pehe)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="label=run_dir:effects_file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    syn = cfg["synthetic_dgp"]
    p = _merge(_DEFAULT_PARAMS_COMPLEX, syn.get("dgp_params"))
    n_obs = int(syn.get("n_obs", 10000))
    dgp = ComplexSyntheticDGP(params=syn.get("dgp_params"), seed=int(syn.get("seed", 42)))
    ret, _ = dgp.generate_series(N=n_obs)

    dates = pd.bdate_range(start=str(syn.get("start_date", "1980-01-03")), periods=n_obs, freq="B")
    train = np.asarray(dates <= pd.Timestamp(str(cfg["data_ingestion"]["train_split_date"])))
    cov = np.cov(dgp._m[train], rowvar=False)
    cpi = int(p["cpi_idx"])
    w = cov[:, cpi] / cov[cpi, cpi]

    report = {"config": str(args.config), "shock_weights": w.tolist(), "runs": {}}
    cache = {}
    for spec in args.run:
        label, rest = spec.split("=", 1)
        run_dir, name = rest.rsplit(":", 1)
        run_dir = Path(run_dir)
        eff = np.load(run_dir / name).astype(np.float64)
        theory = np.load(run_dir / "theory_effects.npy").astype(np.float64)
        t_bnd = np.load(run_dir / "window_meta.npz", allow_pickle=True)["t_boundaries"]
        shocks = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"]
        key = (t_bnd.tobytes(), tuple(shocks))
        if key not in cache:
            horizon = theory.shape[2]
            real = np.load(run_dir / "real_targets.npy").astype(np.float64)
            path = np.stack([ret[T + 1:T + 1 + horizon] for T in t_bnd])
            cache[key] = (unit_effects(dgp, p, t_bnd, shocks, w, horizon),
                          float(np.abs(path - real).max()))
        unit, target_gap = cache[key]
        resid_unit, resid_model = unit - theory, eff - theory
        corr = float(np.corrcoef(resid_unit.ravel(), resid_model.ravel())[0, 1])
        res = {
            "run_dir": str(run_dir), "effects": name, "n_windows": int(eff.shape[0]),
            "max_abs_gap_resimulated_vs_saved_targets": target_gap,
            "unit_minus_average_rms_bps": float(BPS * np.sqrt((resid_unit ** 2).mean())),
            "average_effect_rms_bps": float(BPS * np.sqrt((theory ** 2).mean())),
            "unit_mean_vs_average_max_abs_bps": float(BPS * np.abs(resid_unit.mean(axis=0)).max()),
            "vs_average": score(eff, theory),
            "vs_unit": score(eff, unit),
            "corr_model_residual_with_unit_residual": corr,
        }
        report["runs"][label] = res
        print(f"{label:<24} unit-avg RMS {res['unit_minus_average_rms_bps']:6.2f} bps"
              f" (avg effect RMS {res['average_effect_rms_bps']:.2f})"
              f" | vs average: Dir {res['vs_average']['dir_pct']:.1f} PEHE {res['vs_average']['pehe_bps']:.1f}"
              f" | vs unit: Dir {res['vs_unit']['dir_pct']:.1f} PEHE {res['vs_unit']['pehe_bps']:.1f}"
              f" | corr {corr:+.3f} | target gap {target_gap:.1e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
