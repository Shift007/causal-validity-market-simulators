#!/usr/bin/env python
"""SVAR as a LINEAR causal baseline on the synthetic testbeds, scored against the
EXACT closed-form counterfactual (``theory_effects.npy``).

Makes concrete the claim that linear methods cannot capture the
regime-switching non-linearity. A recursive SVAR (macro block-exogenous; returns
load on *contemporaneous* macro — Cholesky macro-first identification, which here
equals OLS of returns on macro because the return innovation is independent of the
macro innovation by construction) is estimated on the synthetic panel and its
counterfactual is propagated EXACTLY as the analytical theory, but with estimated
(Phi_hat, B_hat) instead of the true (Phi, B):

    Delta_svar[k] = B_hat @ (Phi_hat^{k+1} @ shock_vec)

On the LINEAR DGP the estimates are consistent (every
loading is OLS-recoverable) -> Delta_svar ~ truth (beta ~ 1, high Dir%).
On the REGIME DGP the single estimated B_hat is a frequency-weighted blend of the
two regimes -> Delta_svar is linear and sign-symmetric and MISSES the kink, while
the generative model reproduces it.

Scored with the SAME metrics as the models (``compute_causal_metrics`` + the
cumulative-path beta), so the SVAR row drops straight into the causal table.

Usage
-----
    python scripts/svar_synthetic_benchmark.py --config configs/synthetic_complex/baseline.yaml
    python scripts/svar_synthetic_benchmark.py --config configs/synthetic_complex_nl/baseline.yaml
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
from src.paths import resolve_paths                       # noqa: E402
from src.metrics.calibration import compute_causal_metrics  # noqa: E402


def _ols(X: np.ndarray, Y: np.ndarray):
    """OLS with intercept. X (n,k), Y (n,m) -> intercept (m,), coef (k,m)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    return beta[0], beta[1:]


def _latest_theory(results_dir: Path, cf_run: str | None) -> Path:
    """Locate a counterfactual_synthetic run with theory_effects.npy.

    The closed-form theory is model-independent, so the _ablation sibling of the
    experiment is an acceptable fallback (same config -> same windows/DGP)."""
    base = results_dir / "counterfactual_synthetic"
    if cf_run:
        cand = [base / cf_run]
    else:
        cand = sorted(base.glob("run_*")) if base.exists() else []
        if not cand:
            sib = Path(str(results_dir) + "_ablation") / "counterfactual_synthetic"
            cand = sorted(sib.glob("run_*")) if sib.exists() else []
    for r in reversed(cand):
        if (r / "theory_effects.npy").exists():
            return r
    raise FileNotFoundError(
        f"No theory_effects.npy under {base} (or its _ablation sibling). "
        "Run generate.py counterfactual_synthetic for this config first.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cf-run", default=None, help="specific CF run dir name (else latest)")
    args = ap.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = resolve_paths(ROOT, cfg)

    syn = cfg["synthetic_dgp"]
    n_obs = int(syn.get("n_obs", 7000))
    seed = int(syn.get("seed", 42))
    start_date = str(syn.get("start_date", "1995-01-02"))
    dgp_params = syn.get("dgp_params", None)
    cpi = 0  # CPI is macro index 0 in the complex DGP

    # ── Re-simulate the DGP (same seed/params) → [macro, returns] panel ───────
    from scripts.synthetic_dgp_complex import ComplexSyntheticDGP, _regime_spec
    dgp = ComplexSyntheticDGP(params=dgp_params, seed=seed)
    ret, m = dgp.generate_series(N=n_obs)            # (N, n_assets), (N, n_macro)
    n_assets, n_macro = ret.shape[1], m.shape[1]
    regime_on = bool(_regime_spec(dgp.p)[0])

    # ── Train mask: same information set as the models (dates <= split) ───────
    dates = pd.bdate_range(start=start_date, periods=n_obs, freq="B")
    tr = np.asarray(dates <= pd.Timestamp(str(cfg["data_ingestion"]["train_split_date"])))
    m_tr, ret_tr = m[tr], ret[tr]

    # ── Fit the recursive SVAR on the TRAIN portion ──────────────────────────
    #   macro VAR(1):  m_t = c_hat + Phi_hat m_{t-1}
    _, _coef = _ols(m_tr[:-1], m_tr[1:])
    Phi_hat = _coef.T                                # Phi_hat[i,j] = effect of m_j on m_i
    #   contemporaneous structural loading:  ret_t = a_hat + B_hat m_t
    #   (eps ⟂ u by construction -> OLS is the unbiased recursive-SVAR estimate)
    _, B_hat_T = _ols(m_tr, ret_tr)                  # (n_macro, n_assets)

    # ── Covariance-consistent shock weights w (match the CF run) ─────────────
    cov_adj = bool(cfg.get("counterfactual_synthetic", {}).get("covariance_adjust", False))
    if cov_adj:
        rc = np.cov(m_tr, rowvar=False)
        w = rc[:, cpi] / rc[cpi, cpi]
    else:
        w = np.zeros(n_macro); w[cpi] = 1.0

    shock_deltas = list(cfg["counterfactual_synthetic"]["shock_deltas"])
    shock_labels = list(cfg["counterfactual_synthetic"].get(
        "shock_labels", [f"{d:+.3f}" for d in shock_deltas]))
    K = len(shock_deltas)

    # ── Closed-form ground truth (same DGP, same windows) ────────────────────
    run = _latest_theory(paths.results_dir, args.cf_run)
    theory = np.load(run / "theory_effects.npy")     # (N, K, T, C)
    N, Kt, T, C = theory.shape
    assert Kt == K and C == n_assets, \
        f"shape mismatch: theory {theory.shape} vs K={K}, C={n_assets}"

    # ── SVAR counterfactual: Delta_svar[k] = B_hat @ (Phi_hat^{k+1} @ shock_vec)
    #   window-independent by linearity → broadcast over the N windows.
    svar_eff = np.zeros((N, K, T, C), np.float32)
    for si, d in enumerate(shock_deltas):
        dm = d * w
        for k in range(T):
            dm = Phi_hat @ dm                         # Phi^{k+1} shock
            svar_eff[:, si, k, :] = dm @ B_hat_T      # (n_assets,)

    # ── Metrics: identical definitions to counterfactual_synthetic ───────────
    def _path_beta(model_eff, theory_eff):
        betas, rhos = [], []
        for c in range(C):
            for k in range(K):
                vt = np.cumsum(theory_eff[:, k, :, c].mean(0))
                vm = np.cumsum(model_eff[:, k, :, c].mean(0))
                nt = float(np.linalg.norm(vt))
                if nt < 1e-12:
                    continue
                betas.append(float(vm @ vt) / (nt * nt))
                rhos.append(float(vm @ vt) / (float(np.linalg.norm(vm)) * nt + 1e-12))
        return (float(np.median(betas)) if betas else float("nan"),
                float(np.mean(rhos)) if rhos else float("nan"))

    dir_list, amp_list = [], []
    for si in range(K):
        cm = compute_causal_metrics(theory[:, si], svar_eff[:, si][None])
        dir_list.append(cm["directional_accuracy"])
        amp_list.append(cm["amplification_ratio"])
    beta_med, rho_mean = _path_beta(svar_eff, theory)

    # ── Sign-asymmetry index: |cumΔ(+δ)| vs |cumΔ(−δ)| (final cumulative) ─────
    def _asym(eff):
        out = []
        for si, d in enumerate(shock_deltas):
            if d <= 0:
                continue
            sj = next((j for j, dd in enumerate(shock_deltas) if abs(dd + d) < 1e-9), None)
            if sj is None:
                continue
            cp = np.abs(np.cumsum(eff[:, si].mean(0), axis=0)[-1])   # (C,)
            cn = np.abs(np.cumsum(eff[:, sj].mean(0), axis=0)[-1])   # (C,)
            out.append(float(np.mean((cp - cn) / (cp + cn + 1e-12))))
        return float(np.mean(out)) if out else float("nan")
    asym_theory, asym_svar = _asym(theory), _asym(svar_eff)

    # ── Report ───────────────────────────────────────────────────────────────
    exp = paths.results_dir.name
    print("=" * 72)
    print(f"SVAR linear-causal baseline vs closed-form truth — {exp}")
    print(f"  regime_on={regime_on}  cov_adjust={cov_adj}  "
          f"N={N} windows  K={K} shocks  T={T}  (VAR(1), train-fit)")
    print(f"  theory run: {run.name}")
    print("=" * 72)
    print(f"  Dir%  (mean over shocks): {100 * np.mean(dir_list):5.1f}%")
    print(f"  beta  (cum-path median):  {beta_med:+.3f}")
    print(f"  rho   (cum-path cosine):  {rho_mean:+.3f}")
    print(f"  amp   (mean ratio):       {np.nanmean(amp_list):+.3f}")
    print(f"  sign-asymmetry index  ->  theory: {asym_theory:+.3f}   SVAR: {asym_svar:+.3f}")
    print("  per-shock Dir%: "
          + "  ".join(f"{shock_labels[i]}:{100 * dir_list[i]:.0f}" for i in range(K)))

    out = {
        "experiment": exp, "regime_on": regime_on, "cov_adjust": cov_adj,
        "theory_run": run.name, "n_windows": int(N), "n_shocks": int(K),
        "var_lag": 1, "fit_sample": "train",
        "dir_mean": float(np.mean(dir_list)), "beta_path_median": beta_med,
        "rho_path_mean": rho_mean, "amp_mean": float(np.nanmean(amp_list)),
        "asym_theory": asym_theory, "asym_svar": asym_svar,
        "dir_per_shock": {shock_labels[i]: float(dir_list[i]) for i in range(K)},
    }
    odir = paths.results_dir / "svar_synthetic_benchmark"
    odir.mkdir(parents=True, exist_ok=True)
    np.save(odir / "svar_effects.npy", svar_eff)
    with open(odir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] saved {odir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
