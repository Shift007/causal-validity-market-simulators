#!/usr/bin/env python
"""Asymmetry index of the exact (re-simulated) truth against the analytical one (CPU only).

On the regime-switching environment the analytical average effect evaluates the
regime on the expected macro path. This script re-simulates every evaluation
window under each shock with its innovations held fixed (``score_unit_level``),
averages the exact unit-level effects over windows, and reports the asymmetry
index and the response magnitude of that exact average next to the analytical
ones, together with the model run scored against both.

Usage (from code/):
    python scripts/score_nl_exact_truth.py --config <yaml> --run <label>=<run_dir>:<effects.npy> ... --out <json>
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
sys.path.insert(0, str(ROOT / "scripts"))
from score_paper_causal import asym_stats, table3_stats  # noqa: E402
from score_unit_level import unit_effects  # noqa: E402
from synthetic_dgp_complex import ComplexSyntheticDGP, _merge, _DEFAULT_PARAMS_COMPLEX  # noqa: E402

BPS = 1e4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="label=run_dir:effects_file")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    syn = cfg["synthetic_dgp"]
    p = _merge(_DEFAULT_PARAMS_COMPLEX, syn.get("dgp_params"))
    n_obs = int(syn.get("n_obs", 10000))
    dgp = ComplexSyntheticDGP(params=syn.get("dgp_params"), seed=int(syn.get("seed", 42)))
    dgp.generate_series(N=n_obs)
    dates = pd.bdate_range(start=str(syn.get("start_date", "1980-01-03")), periods=n_obs, freq="B")
    train = np.asarray(dates <= pd.Timestamp(str(cfg["data_ingestion"]["train_split_date"])))
    cov = np.cov(dgp._m[train], rowvar=False)
    cpi = int(p["cpi_idx"])
    w = cov[:, cpi] / cov[cpi, cpi]

    rng = np.random.default_rng(0)
    report, unit, key0 = {"config": str(args.config), "runs": {}}, None, None
    for spec in args.run:
        label, rest = spec.split("=", 1)
        run_dir, name = rest.rsplit(":", 1)
        run_dir = Path(run_dir)
        eff = np.load(run_dir / name).astype(np.float64)
        theory = np.load(run_dir / "theory_effects.npy").astype(np.float64)
        t_bnd = np.load(run_dir / "window_meta.npz", allow_pickle=True)["t_boundaries"]
        shocks = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["shock_deltas"]
        key = (t_bnd.tobytes(), tuple(shocks))
        if unit is None:
            unit, key0 = unit_effects(dgp, p, t_bnd, shocks, w, theory.shape[2]), key
            n = len(t_bnd)
            idx = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(args.n_boot, n))])
            truths = {}
            for tag, arr in (("analytical", theory), ("exact", unit)):
                a = asym_stats(arr, shocks, idx)
                truths[tag] = {"A": float(a["A"][0]),
                               "A_ci95": [float(x) for x in np.percentile(a["A"][1:], [2.5, 97.5])],
                               "cum_response_bps": float(a["cum_response_bps"][0])}
            # exact average effect broadcast to every window: the average truth without the analytical shortcut
            exact_mean = np.broadcast_to(unit.mean(axis=0, keepdims=True), unit.shape)
            truths["exact_mean_vs_analytical_mean_rms_bps"] = float(
                BPS * np.sqrt(((unit.mean(axis=0) - theory.mean(axis=0)) ** 2).mean()))
            truths["analytical_mean_rms_bps"] = float(BPS * np.sqrt((theory.mean(axis=0) ** 2).mean()))
            report["truths"] = truths
            print(json.dumps(truths, indent=1))
        assert key == key0, "runs must share windows and shocks"
        one = np.arange(len(t_bnd))[None]
        a = asym_stats(eff, shocks, idx)
        res = {"run_dir": str(run_dir), "A": float(a["A"][0]),
               "A_ci95": [float(x) for x in np.percentile(a["A"][1:], [2.5, 97.5])],
               "response_pct_of_analytical": float(100 * a["cum_response_bps"][0] / truths["analytical"]["cum_response_bps"]),
               "response_pct_of_exact": float(100 * a["cum_response_bps"][0] / truths["exact"]["cum_response_bps"]),
               "vs_analytical": {k: float(v[0]) for k, v in table3_stats(eff, theory, one).items()},
               "vs_exact_unit": {k: float(v[0]) for k, v in table3_stats(eff, unit, one).items()}}
        report["runs"][label] = res
        print(label, json.dumps(res, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
