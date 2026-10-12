#!/usr/bin/env python
"""
00b — COMPLEX synthetic DGP: multi-macro VAR(1) + matrix-loaded GARCH-X returns.

Version 2 of scripts/synthetic_dgp.py. Designed to be RICH and IDENTIFIABLE:
the single-macro original is degenerate (1 signal -> the flexible temporal
encoder overfits a wrong counterfactual). Here the conditioning is 4 distinct
macros, so the causal CPI effect is identifiable by controlling for the others.

DGP Specification  (6 assets + 4 macro variables):
  Macro VAR(1):  m_t = c + Phi @ m_{t-1} + u_t ,   u_t ~ N(0, Sigma_u)
                 m = [CPI, GROWTH, MONEY, RATES]
  Returns     :  R_{i,t} = mu_i + sum_j B_ij * m_{j,t} + eps_{i,t}
  Cond. Var   :  h_{i,t} = omega_i + alpha_i*eps_{i,t-1}^2 + gamma_i*dCPI_t + delta_i*h_{i,t-1}
  Innovations :  eps = z * sqrt(h),  z ~ standardised t(nu)

Design principles (a-priori, independent of the model):
  * Richness       : 4 distinct (non-collinear) macros -> rich conditioning.
  * Identifiability : observational = causal (B *is* the causal effect; no
                      hidden confounder); CPI effect isolable given the others.
  * Realism        : return std ~130 bps, excess kurtosis ~6-8, vol clustering;
                     persistent macros with moderate cross-correlation.
  * Verifiability  : exact analytical ground truth (linear VAR + linear loadings).

Optional NON-LINEAR variant (params["regime"], off by default): regime-switching
loadings B -> B_high on "high-inflation" days (CPI level above a threshold). The
macro VAR and the GARCH-X variance stay unchanged; only the macro->return MEAN
channel becomes non-linear, so the analytical counterfactual is still exact and
is re-evaluated per regime (kinked where a shock crosses the threshold).

Standalone usage (writes processed_data/<branch>/ ingest-level artifacts):
    python scripts/synthetic_dgp_complex.py --config configs/synthetic_complex/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_NAMES = ["SYNTH_CPI", "SYNTH_GROWTH", "SYNTH_MONEY", "SYNTH_RATES"]
ASSET_NAMES = ["SY_EQ1", "SY_EQ2", "SY_CM1", "SY_CM2", "SY_FI1", "SY_FI2"]
CPI_IDX = 0

_DEFAULT_PARAMS_COMPLEX: dict = {
    # ── Macro VAR(1): persistent, mildly coupled, distinct innovations ────────
    "macro": {
        "names": MACRO_NAMES,
        "c":     [0.0, 0.0, 0.0, 0.0],          # zero drift -> symmetric, mean 0
        # Phi[i,j] = effect of macro j_{t-1} on macro i_t
        "Phi": [
            [0.985, 0.000, 0.005, 0.000],       # CPI: persistent, +money
            [0.000, 0.970, 0.005, 0.000],       # GROWTH: persistent, +money
            [0.000, 0.000, 0.990, 0.000],       # MONEY: very persistent
            [0.020, 0.000, 0.000, 0.980],       # RATES: react to CPI_{t-1} (central bank)
        ],
        "sigma_u": [1.2e-3, 1.2e-3, 1.2e-3, 1.2e-3],   # innovation std per macro
        # innovation correlation (moderate, keeps macros DISTINCT/non-collinear)
        "corr_u": [
            [1.0, 0.0, 0.0, 0.2],               # CPI-RATES +0.2
            [0.0, 1.0, 0.3, 0.0],               # GROWTH-MONEY +0.3
            [0.0, 0.3, 1.0, 0.0],
            [0.2, 0.0, 0.0, 1.0],
        ],
        "init": [0.0, 0.0, 0.0, 0.0],
    },
    # ── Assets: economically-sensible loadings on the 4 macros ────────────────
    "assets": {
        "names": ASSET_NAMES,
        "mu":    [1e-4, 5e-5, 5e-5, 3e-5, 2e-5, 3e-5],
        # B[i,j] = causal loading of asset i on macro j  [CPI, GROWTH, MONEY, RATES]
        "B": [
            [-0.25, +0.40, +0.15, -0.30],       # SY_EQ1 cyclical equity
            [-0.10, +0.20, +0.10, -0.15],       # SY_EQ2 defensive equity
            [+0.35, +0.30, +0.10, -0.05],       # SY_CM1 energy commodity
            [+0.20, -0.10, +0.25, -0.20],       # SY_CM2 gold-like
            [-0.30, -0.20, +0.05, -0.40],       # SY_FI1 govt bond
            [-0.15, +0.15, +0.10, -0.25],       # SY_FI2 credit bond
        ],
        "omega": [6e-6, 4e-6, 7e-6, 5e-6, 3e-6, 4e-6],
        "alpha": [0.08, 0.06, 0.10, 0.07, 0.05, 0.06],
        "gamma": [2e-3, 1e-3, 3e-3, 2e-3, 8e-4, 1e-3],   # GARCH-X on dCPI
        "delta": [0.88, 0.90, 0.85, 0.88, 0.91, 0.90],
    },
    "nu": 5,   # t df -> excess kurtosis 6/(nu-4)=6 baseline, more with clustering
    # Return-innovation law: "student_t" (default, fat tails) or "gaussian". Only
    # the RETURN shocks z change; the macro VAR is Gaussian in both. Combined with
    # alpha=gamma=delta=0 (constant h=omega) this gives a homoskedastic-Gaussian
    # environment that strips the financial stylized facts while leaving the macro
    # process, the causal loadings B and the analytical counterfactual UNCHANGED.
    "innovations": "student_t",
    "cpi_idx": CPI_IDX,
    # ── Optional regime-switching loadings (NON-LINEAR variant; OFF by default) ─
    #   HIGH regime = days where macro `macro_idx` (CPI) sits above `threshold`
    #   (a persistent "high-inflation" regime). On those days the loading matrix
    #   switches B -> B_high (= B * per-macro `gain`, or an explicit `B_high`).
    #   Macro VAR + GARCH-X variance are untouched: the non-linearity is purely in
    #   the macro->return MEAN channel, so the analytical CF stays exact.
    "regime": {
        "enabled":   False,
        "macro_idx": CPI_IDX,
        "threshold": 0.0035,                 # ~ +0.5 sigma of the CPI process
        "gain":      [1.0, 1.0, 1.0, 1.0],   # per-macro-column gain in HIGH regime
        "B_high":    None,                   # optional explicit B_high (overrides gain)
    },
}


def _merge(defaults, override):
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    if override:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k].update(v)
            else:
                p[k] = v
    return p


def _sigma_from(sigma_u, corr_u):
    D = np.diag(np.asarray(sigma_u, float))
    return D @ np.asarray(corr_u, float) @ D


def _regime_spec(params):
    """Resolve the regime-switching loadings.

    Returns (enabled, macro_idx, threshold, B_low, B_high). When disabled,
    B_high is B_low, so the DGP reduces EXACTLY to the original linear one.
    HIGH regime = days where macro `macro_idx` exceeds `threshold`; on those days
    the loadings switch from B_low to B_high (= B*gain, or an explicit B_high).
    """
    B = np.asarray(params["assets"]["B"], float)
    rp = params.get("regime", {}) or {}
    enabled = bool(rp.get("enabled", False))
    idx = int(rp.get("macro_idx", params.get("cpi_idx", 0)))
    thr = float(rp.get("threshold", 0.0))
    if not enabled:
        return False, idx, thr, B, B
    if rp.get("B_high") is not None:
        B_high = np.asarray(rp["B_high"], float)
    else:
        gain = np.asarray(rp.get("gain", [1.0] * B.shape[1]), float)
        B_high = B * gain[None, :]
    return True, idx, thr, B, B_high


# ═════════════════════════════════════════════════════════════════════════════
class ComplexSyntheticDGP:
    """Multi-macro VAR(1) + matrix-loaded GARCH-X DGP with fat tails."""

    def __init__(self, params: dict | None = None, seed: int = 42):
        self.p = _merge(_DEFAULT_PARAMS_COMPLEX, params)
        self.rng = np.random.default_rng(seed)

    def generate_series(self, N: int = 12_000):
        mp, ap = self.p["macro"], self.p["assets"]
        c     = np.asarray(mp["c"], float)
        Phi   = np.asarray(mp["Phi"], float)
        Sig   = _sigma_from(mp["sigma_u"], mp["corr_u"])
        m0    = np.asarray(mp["init"], float)
        mu    = np.asarray(ap["mu"], float)
        B     = np.asarray(ap["B"], float)                  # (n_assets, n_macro)
        omega = np.asarray(ap["omega"], float)
        alpha = np.asarray(ap["alpha"], float)
        gamma = np.asarray(ap["gamma"], float)
        delta = np.asarray(ap["delta"], float)
        nu    = float(self.p["nu"]); cpi = int(self.p["cpi_idx"])
        innov = str(self.p.get("innovations", "student_t")).lower()
        n_macro, n_assets = Phi.shape[0], B.shape[0]

        N_total = N + 1_000                                  # burn-in
        if innov == "gaussian":
            z = self.rng.standard_normal(size=(N_total, n_assets))
        elif innov in ("student_t", "student-t", "t"):
            scale_t = np.sqrt(nu / (nu - 2.0))
            z = self.rng.standard_t(df=nu, size=(N_total, n_assets)) / scale_t
        else:
            raise ValueError(
                f"unknown innovations {innov!r} (expected 'student_t' or 'gaussian')")
        u   = self.rng.multivariate_normal(np.zeros(n_macro), Sig, size=N_total)

        m = np.empty((N_total, n_macro)); m[0] = m0
        for t in range(1, N_total):
            m[t] = c + Phi @ m[t - 1] + u[t]

        r_on, r_idx, r_thr, B_lo, B_hi = _regime_spec(self.p)
        if r_on:
            high = m[:, r_idx] > r_thr                        # (N_total,) regime mask
            macro_load = np.where(high[:, None],
                                  m @ B_hi.T, m @ B_lo.T)      # (N_total, n_assets)
        else:
            macro_load = m @ B.T                              # (N_total, n_assets)
        h   = np.empty((N_total, n_assets))
        eps = np.empty((N_total, n_assets))
        ret = np.empty((N_total, n_assets))
        h[0]   = omega / np.maximum(1.0 - alpha - delta, 1e-8)
        eps[0] = z[0] * np.sqrt(h[0])
        ret[0] = mu + macro_load[0] + eps[0]
        for t in range(1, N_total):
            dcpi   = m[t, cpi] - m[t - 1, cpi]
            h[t]   = np.maximum(omega + alpha * eps[t - 1] ** 2 + gamma * dcpi
                                + delta * h[t - 1], 1e-12)
            eps[t] = z[t] * np.sqrt(h[t])
            ret[t] = mu + macro_load[t] + eps[t]

        self._m, self._h, self._eps = m[1000:], h[1000:], eps[1000:]
        return ret[1000:].astype(np.float64), m[1000:].astype(np.float64)

    def get_state_at(self, t: int) -> dict:
        return {"m": self._m[t].copy(), "cpi": float(self._m[t, self.p["cpi_idx"]]),
                "h": self._h[t].copy(), "eps_sq": self._eps[t] ** 2}


# ═════════════════════════════════════════════════════════════════════════════
class TheoreticalCounterfactualComplex:
    """Exact analytical CPI-shock expectations for the multi-macro DGP.

    Shock CPI at T by delta; propagate through the VAR (E[m_{T+k}]=c+Phi E[m_{T+k-1}]);
    E[R_i] = mu_i + B_i . E[m]. Variance recursion identical to the GARCH-X form.
    """

    def __init__(self, params: dict | None = None):
        self.p = _merge(_DEFAULT_PARAMS_COMPLEX, params)

    def compute(self, m_T_baseline, shock_delta, h_T, eps_T_sq, horizon: int = 64,
                shock_idx: int | None = None, shock_vec=None):
        # shock_idx selects WHICH macro is intervened on (default CPI); shock_vec,
        # if given, is a full macro-vector move added to m_T (used for the Kubiak
        # covariance-adjusted scenario: shock one macro, the others follow Sigma).
        # The GARCH-X variance channel below stays keyed on dCPI -- the only
        # exogenous driver of h in the DGP -- so a non-CPI shock moves the mean
        # (via the loadings B and the VAR Phi) and perturbs the variance only
        # through any induced CPI-path change (e.g. MONEY->CPI via Phi / Sigma).
        mp, ap = self.p["macro"], self.p["assets"]
        c   = np.asarray(mp["c"], float);  Phi = np.asarray(mp["Phi"], float)
        mu  = np.asarray(ap["mu"], float); B = np.asarray(ap["B"], float)
        omega = np.asarray(ap["omega"], float); alpha = np.asarray(ap["alpha"], float)
        gamma = np.asarray(ap["gamma"], float); delta = np.asarray(ap["delta"], float)
        cpi = int(self.p["cpi_idx"])
        shk = cpi if shock_idx is None else int(shock_idx)
        m_T_base = np.asarray(m_T_baseline, float)
        if shock_vec is not None:
            m_T_sh = m_T_base + np.asarray(shock_vec, float)
        else:
            m_T_sh = m_T_base.copy(); m_T_sh[shk] += shock_delta

        n = len(mu)
        E_m_sh = np.empty((horizon, len(c))); E_m_b = np.empty((horizon, len(c)))
        prev_sh, prev_b = m_T_sh, m_T_base
        for k in range(horizon):
            E_m_sh[k] = c + Phi @ prev_sh; prev_sh = E_m_sh[k]
            E_m_b[k]  = c + Phi @ prev_b;  prev_b  = E_m_b[k]

        E_cpi_sh, E_cpi_b = E_m_sh[:, cpi], E_m_b[:, cpi]
        d_sh = np.empty(horizon); d_b = np.empty(horizon)
        d_sh[0] = E_cpi_sh[0] - m_T_sh[cpi]; d_b[0] = E_cpi_b[0] - m_T_base[cpi]
        d_sh[1:] = np.diff(E_cpi_sh);        d_b[1:] = np.diff(E_cpi_b)

        h_T = np.asarray(h_T, float); eps_T_sq = np.asarray(eps_T_sq, float)
        E_h_sh = np.empty((horizon, n)); E_h_b = np.empty((horizon, n))
        E_h_sh[0] = np.maximum(omega + alpha * eps_T_sq + gamma * d_sh[0] + delta * h_T, 1e-12)
        E_h_b[0]  = np.maximum(omega + alpha * eps_T_sq + gamma * d_b[0]  + delta * h_T, 1e-12)
        for k in range(1, horizon):
            E_h_sh[k] = np.maximum(omega + (alpha + delta) * E_h_sh[k-1] + gamma * d_sh[k], 1e-12)
            E_h_b[k]  = np.maximum(omega + (alpha + delta) * E_h_b[k-1]  + gamma * d_b[k],  1e-12)

        # Regime-dependent loadings (NON-LINEAR variant): on each horizon day the
        # loading switches to B_high where the EXPECTED regime macro is above
        # threshold, evaluated SEPARATELY on the shocked and baseline paths -- so a
        # shock that crosses the threshold makes delta_E_ret KINKED (the exact,
        # still-analytical non-linear counterfactual). Disabled -> linear as before.
        r_on, r_idx, r_thr, B_lo, B_hi = _regime_spec(self.p)
        if r_on:
            hi_sh = E_m_sh[:, r_idx] > r_thr     # (horizon,)
            hi_b  = E_m_b[:, r_idx]  > r_thr
            E_ret_sh = mu + np.where(hi_sh[:, None], E_m_sh @ B_hi.T, E_m_sh @ B_lo.T)
            E_ret_b  = mu + np.where(hi_b[:, None],  E_m_b  @ B_hi.T, E_m_b  @ B_lo.T)
        else:
            E_ret_sh = mu + E_m_sh @ B.T           # (horizon, n_assets)
            E_ret_b  = mu + E_m_b  @ B.T
        return {"E_ret": E_ret_sh, "E_ret_base": E_ret_b,
                "delta_E_ret": E_ret_sh - E_ret_b,
                "delta_E_vol": np.sqrt(E_h_sh) - np.sqrt(E_h_b),
                "E_cpi": E_cpi_sh, "E_cpi_base": E_cpi_b}

    def compute_from_path(self, dgp, shock_T, shock_delta, horizon: int = 64):
        s = dgp.get_state_at(shock_T)
        r = self.compute(s["m"], shock_delta, s["h"], s["eps_sq"], horizon)
        r["shock_T"] = shock_T; r["shock_delta"] = shock_delta
        return r


# ═════════════════════════════════════════════════════════════════════════════
def run(project_root: Path, config: dict) -> None:
    """Generate the complex synthetic data and write ingest-level artifacts."""
    syn = config.get("synthetic_dgp", {})
    params = syn.get("dgp_params", None)
    N = int(syn.get("n_obs", 12_000)); seed = int(syn.get("seed", 42))
    start = str(syn.get("start_date", "2000-01-03"))

    tickers = list(config["data_ingestion"]["tickers"])
    macro_vars = dict(config["data_ingestion"]["macro_vars"])
    p = _merge(_DEFAULT_PARAMS_COMPLEX, params)
    n_macro = len(p["macro"]["names"]); n_assets = len(p["assets"]["names"])
    if len(tickers) != n_assets:
        raise ValueError(f"complex DGP expects {n_assets} tickers, got {len(tickers)}")
    if len(macro_vars) != n_macro:
        raise ValueError(f"complex DGP expects {n_macro} macro vars, got {len(macro_vars)}")

    dgp = ComplexSyntheticDGP(params=params, seed=seed)
    returns, macro = dgp.generate_series(N=N)
    dates = pd.bdate_range(start=start, periods=N, freq="B")

    from src.paths import resolve_paths
    pdir = resolve_paths(project_root, config).processed_dir
    pdir.mkdir(parents=True, exist_ok=True)

    td = {tk: pd.DataFrame({"LogReturns": returns[:, i].astype(np.float32)}, index=dates)
          for i, tk in enumerate(tickers)}
    np.save(pdir / "ticker_data_RAW.npy", td)
    macro_df = pd.DataFrame(
        {name: macro[:, j].astype(np.float64) for j, name in enumerate(macro_vars)},
        index=dates)
    macro_df.to_parquet(pdir / "aligned_macro.parquet")
    json.dump({
        "source": "synthetic_dgp_complex_var_garchx", "n_obs": N,
        "start_date": str(dates[0].date()), "end_date": str(dates[-1].date()),
        "train_cutoff": config["data_ingestion"]["train_split_date"],
        "tickers": tickers, "macro_vars": macro_vars,
        "n_assets": n_assets, "n_macro": n_macro, "n_trading_days": N,
        "return_transform": "log_return_synthetic_complex", "dgp_nu": p["nu"],
        "innovations": p.get("innovations", "student_t"),
        "regime": {k: p["regime"].get(k) for k in ("enabled", "macro_idx", "threshold", "gain")},
    }, open(pdir / "data_config.json", "w"), indent=4)

    print(f"[complex] wrote {N} obs, {n_assets} assets, {n_macro} macros -> {pdir}")
    print(f"[complex] return std (bps): {np.round(returns.std(0) * 1e4, 1)}")
    print(f"[complex] macro std: {np.round(macro.std(0), 4)}")
    _r_on, _r_idx, _r_thr, _, _ = _regime_spec(p)
    if _r_on:
        _frac = float(np.mean(macro[:, _r_idx] > _r_thr))
        print(f"[complex] REGIME-SWITCHING on: macro[{_r_idx}] > {_r_thr} -> B*gain "
              f"(high-regime frac = {_frac:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import yaml
    cfgp = Path(a.config) if a.config else root / "config.yaml"
    run(root, yaml.safe_load(open(cfgp, encoding="utf-8")))
