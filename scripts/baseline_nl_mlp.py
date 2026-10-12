#!/usr/bin/env python
"""Non-linear regression reference for the regime-switching environment (CPU only).

A plain MLP regresses the cumulative return of the next ``target_len`` days on
the boundary macro state m_T, on the training period of the non-linear DGP. Its
interventional response is MLP(m_T + delta * w) - MLP(m_T), with the same
covariance-adjusted shock vector as the generators, evaluated on the windows of
a reference counterfactual run and scored with the same asymmetry index.

It is a point estimator of the mean cumulative response, not a simulator: it
calibrates how hard the curvature is to learn from the conditioning inputs.

Usage (from the repository root):
    python scripts/baseline_nl_mlp.py \
        --reference results/synthetic_complex_nl_filmonly/counterfactual_synthetic/run_20260724_151522
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_paper_causal import asym_stats, summarise  # noqa: E402
from synthetic_dgp_complex import ComplexSyntheticDGP  # noqa: E402

CONFIG = ROOT / "configs" / "synthetic_complex_nl" / "baseline.yaml"
OUT_DIR = ROOT / "results" / "synthetic_complex_nl_mlp_paperwindows"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True, help="counterfactual run providing windows and theory")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--mlp_seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    syn, cf = cfg["synthetic_dgp"], cfg["counterfactual_synthetic"]
    n_obs, T = int(syn["n_obs"]), int(cfg["preprocessing"]["target_length"])
    deltas = np.asarray(cf["shock_deltas"], float)
    shk = int(cf.get("shock_macro_index", 0))

    dgp = ComplexSyntheticDGP(params=syn.get("dgp_params"), seed=int(syn["seed"]))
    returns, macro = dgp.generate_series(N=n_obs)
    dates = pd.bdate_range(start=str(syn["start_date"]), periods=n_obs, freq="B")
    train = np.asarray(dates <= pd.Timestamp(str(cfg["data_ingestion"]["train_split_date"])))

    cov = np.cov(macro[train], rowvar=False)
    w = cov[:, shk] / cov[shk, shk] if cf.get("covariance_adjust", True) else np.eye(macro.shape[1])[shk]

    # (m_t, cumulative return over t+1 .. t+T) pairs from the training period only
    n = n_obs - T - 1
    cum = np.stack([returns[t + 1:t + 1 + T].sum(axis=0) for t in range(n)])
    xs, ys = StandardScaler().fit(macro[:n][train[:n]]), StandardScaler().fit(cum[train[:n]])
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=800,
                       early_stopping=True, n_iter_no_change=25, random_state=args.mlp_seed)
    mlp.fit(xs.transform(macro[:n][train[:n]]), ys.transform(cum[train[:n]]))

    def predict(m):
        return ys.inverse_transform(mlp.predict(xs.transform(m)))

    ref = ROOT / args.reference
    t_bnd = np.load(ref / "window_meta.npz", allow_pickle=True)["t_boundaries"].astype(int)
    theory = np.load(ref / "theory_effects.npy").astype(np.float64)              # (N, K, T, C)
    shocks = json.loads((ref / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"]
    assert np.allclose(shocks, deltas), "reference run and config disagree on the shocks"

    m_bnd = macro[t_bnd]
    base = predict(m_bnd)
    eff = np.stack([predict(m_bnd + d * w[None, :]) - base for d in deltas], axis=1)   # (N, K, C) cumulative

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(eff), size=(args.n_boot + 1, len(eff)))
    idx[0] = np.arange(len(eff))
    s_mlp = summarise(asym_stats(eff[:, :, None, :], shocks, idx))
    s_th = summarise(asym_stats(theory, shocks, idx))
    sign_ok = float((np.sign(eff) == np.sign(theory.sum(axis=2))).mean() * 100.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "mlp_cum_effects.npy", eff.astype(np.float32))
    out = {"reference_run": str(args.reference), "n_windows": int(len(eff)), "mlp_seed": args.mlp_seed,
           "mlp": {"hidden": [128, 64], "n_iter": int(mlp.n_iter_), "loss": float(mlp.loss_)},
           "w_shock": w.tolist(), "cum_direction_pct": sign_ok, "MLP": s_mlp, "Theory": s_th}
    (OUT_DIR / "baseline_nl_mlp.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name, s in (("Theory", s_th), ("MLP", s_mlp)):
        print(f"{name:<8} A {s['A']['value']:+.3f} [{s['A']['ci95'][0]:+.3f}, {s['A']['ci95'][1]:+.3f}]"
              f"   |S| {s['cum_response_bps']['value']:.1f} bps")
    print(f"MLP cumulative-response sign agreement: {sign_ok:.1f}%")
    print(f"written {OUT_DIR / 'baseline_nl_mlp.json'}")


if __name__ == "__main__":
    main()
