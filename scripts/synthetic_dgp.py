#!/usr/bin/env python
"""
00 — Synthetic GARCH-X Data Generating Process.

Provides a mathematical ground truth for counterfactual evaluation of DDPM
and MacroVAE.  Emits the same processed_data/ artifacts as scripts/ingest.py
so that scripts/preprocess.py (and all downstream scripts) run unchanged.

DGP Specification  (3 assets + 1 macro variable):
  Macro (CPI)   :  CPI_t  = c0 + ρ·CPI_{t-1} + u_t ,  u_t  ~ N(0, σ_u²)
  Returns       :  R_{i,t} = μ_i + β_i·CPI_t + ε_{i,t}
  Cond. Variance:  h_{i,t} = ω_i + α_i·ε_{i,t-1}² + γ_i·ΔCPI_t + δ_i·h_{i,t-1}
  Innovations   :  ε_{i,t} = z_{i,t}·√h_{i,t} ,  z_{i,t} ~ Standardised t(ν)
                   (unit-variance: z = t_ν / √(ν/(ν-2)))

Usage (standalone):
    python scripts/synthetic_dgp.py [--config config.yaml]

When  synthetic_dgp.use_synthetic: true  in config.yaml, call this instead of
scripts/ingest.py  (the bypass is wired directly into scripts/ingest.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default DGP calibration  (daily-frequency, plausible financial magnitudes)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS: dict = {
    # ── Macro AR(1) ──────────────────────────────────────────────────────────
    "cpi": {
        "c0":      0.0,    # Zero drift → unconditional mean = 0 (symmetric distribution)
                           # Removes training-density asymmetry between pos/neg CPI shocks.
                           # μ_CPI = c0/(1-ρ) = 0 → P(CPI>0) = P(CPI<0) = 50%
        "rho":     0.99,   # Daily persistence  (≈ 0.77 at monthly horizon)
        "sigma_u": 2e-3,   # Daily innovation std
        "init":    0.0,    # Start at unconditional mean (consistent with c0=0)
    },
    # ── Assets  [equity-like, bond-like, commodity-like] ─────────────────────
    # Unconditional daily vol ≈ √(ω/(1−α−δ)):  ~1.0 %, ~0.6 %, ~1.2 %
    "assets": {
        "mu":    [1e-4,  2e-5,  5e-5],   # Tiny daily mean returns
        "beta":  [-0.30,  0.10, -0.20],  # CPI loading  (equity/commodity: neg)
        "omega": [4e-6,  2e-6,  6e-6],   # GARCH intercept
        "alpha": [0.08,  0.05,  0.10],   # ARCH coefficient  (α)
        "gamma": [2e-3,  8e-4,  3e-3],   # GARCH-X  (ΔCPI_t → variance)
        "delta": [0.88,  0.90,  0.85],   # GARCH persistence  (δ); α+δ < 1  ✓
    },
    # ── Fat-tail distribution ─────────────────────────────────────────────────
    "nu": 4,   # t-Student degrees of freedom  (excess kurtosis = 6/(ν-4) → ∞ for ν=4)
}


# ═════════════════════════════════════════════════════════════════════════════
# Task 1 — SyntheticDGP
# ═════════════════════════════════════════════════════════════════════════════

class SyntheticDGP:
    """Multivariate GARCH-X DGP with fat-tailed innovations.

    Parameters
    ----------
    params : dict or None
        Override any subset of ``_DEFAULT_PARAMS``.
        Sub-dicts are merged key-by-key; scalars replace entirely.
    seed : int
        NumPy random seed for reproducibility.
    """

    def __init__(self, params: dict | None = None, seed: int = 42):
        p = {k: (dict(v) if isinstance(v, dict) else v)
             for k, v in _DEFAULT_PARAMS.items()}
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(p.get(k), dict):
                    p[k].update(v)
                else:
                    p[k] = v
        self.p = p
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def generate_series(
        self, N: int = 20_000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate N time steps of the GARCH-X process.

        Returns
        -------
        returns : (N, 3)  float64 — log returns (after 1 000-step burn-in)
        macro   : (N, 1)  float64 — CPI path (stationary AR(1))
        """
        cpi_cfg = self.p["cpi"]
        a_cfg   = self.p["assets"]
        nu      = float(self.p["nu"])

        mu    = np.array(a_cfg["mu"],    dtype=np.float64)
        beta  = np.array(a_cfg["beta"],  dtype=np.float64)
        omega = np.array(a_cfg["omega"], dtype=np.float64)
        alpha = np.array(a_cfg["alpha"], dtype=np.float64)
        gamma = np.array(a_cfg["gamma"], dtype=np.float64)
        delta = np.array(a_cfg["delta"], dtype=np.float64)
        n_assets = len(mu)

        c0    = float(cpi_cfg["c0"])
        rho   = float(cpi_cfg["rho"])
        sig_u = float(cpi_cfg["sigma_u"])

        # 1 000-step burn-in to reach stationarity
        N_total  = N + 1_000
        scale_t  = np.sqrt(nu / (nu - 2.0))   # Standardise t so Var(z)=1
        z_all    = self.rng.standard_t(df=nu, size=(N_total, n_assets)) / scale_t
        u_cpi    = self.rng.normal(0.0, sig_u, size=N_total)

        # ── CPI path (AR(1)) ──────────────────────────────────────────────
        cpi = np.empty(N_total)
        cpi[0] = float(cpi_cfg["init"])
        for t in range(1, N_total):
            cpi[t] = c0 + rho * cpi[t - 1] + u_cpi[t]

        # ── Returns and conditional variance (GARCH-X) ────────────────────
        h   = np.empty((N_total, n_assets))
        eps = np.empty((N_total, n_assets))
        ret = np.empty((N_total, n_assets))

        # Initialise at unconditional variance (ignoring GARCH-X term)
        h_uncond = omega / np.maximum(1.0 - alpha - delta, 1e-8)
        h[0]   = h_uncond
        eps[0] = z_all[0] * np.sqrt(h[0])
        ret[0] = mu + beta * cpi[0] + eps[0]

        for t in range(1, N_total):
            dcpi   = cpi[t] - cpi[t - 1]
            h[t]   = (omega
                      + alpha * eps[t - 1] ** 2
                      + gamma * dcpi
                      + delta * h[t - 1])
            h[t]   = np.maximum(h[t], 1e-12)   # numerical floor
            eps[t] = z_all[t] * np.sqrt(h[t])
            ret[t] = mu + beta * cpi[t] + eps[t]

        # Discard burn-in, store h and eps for TheoreticalCounterfactual
        self._last_cpi = cpi[1_000:].copy()
        self._last_h   = h[1_000:].copy()
        self._last_eps = eps[1_000:].copy()

        returns_out = ret[1_000:].astype(np.float64)
        macro_out   = cpi[1_000:, np.newaxis].astype(np.float64)
        return returns_out, macro_out

    # ------------------------------------------------------------------
    # Convenience: expose last simulation state for TheoreticalCounterfactual
    # ------------------------------------------------------------------

    def get_state_at(self, t: int) -> dict:
        """Return {cpi, h, eps_sq} at index t of the last simulated path."""
        if not hasattr(self, "_last_h"):
            raise RuntimeError("Call generate_series() first.")
        return {
            "cpi":    float(self._last_cpi[t]),
            "h":      self._last_h[t].copy(),        # (3,) conditional variances
            "eps_sq": self._last_eps[t] ** 2,         # (3,) squared innovations
        }


# ═════════════════════════════════════════════════════════════════════════════
# Task 2 — TheoreticalCounterfactual
# ═════════════════════════════════════════════════════════════════════════════

class TheoreticalCounterfactual:
    """Exact analytical expectations of the GARCH-X DGP after a CPI shock.

    At time T the CPI level is perturbed by ``shock_delta``:
        CPI_T_shocked = CPI_T_baseline + shock_delta

    The class then computes, for each step k = 1, …, horizon:
        E[CPI_{T+k}]   — future macro path  (AR(1) mean reversion)
        E[h_{i,T+k}]   — conditional variance  (recursive GARCH-X)
        E[R_{i,T+k}]   — expected log return

    These serve as the **analytical ground truth** against which the outputs of
    DDPM and MacroVAE are benchmarked in the counterfactual experiment.

    Mathematical derivation
    -----------------------
    Let μ_CPI = c0 / (1 − ρ)  (unconditional CPI mean).

    CPI path:
        E[CPI_{T+k}] = μ_CPI + ρ^k · (CPI_T_shocked − μ_CPI)

    Expected ΔCPI:
        E[ΔCPI_{T+k}] = E[CPI_{T+k}] − E[CPI_{T+k−1}]   (k ≥ 2)
        E[ΔCPI_{T+1}] = E[CPI_{T+1}] − CPI_T_shocked

    Variance recursion (key identity: E[ε²_{i,t}] = E[h_{i,t}] for standardised z):
        k = 1:  E[h_{i,T+1}] = ω_i + α_i·ε²_{i,T} + γ_i·E[ΔCPI_{T+1}] + δ_i·h_{i,T}
        k ≥ 2:  E[h_{i,T+k}] = ω_i + (α_i+δ_i)·E[h_{i,T+k−1}] + γ_i·E[ΔCPI_{T+k}]

    Expected return:
        E[R_{i,T+k}] = μ_i + β_i · E[CPI_{T+k}]
    """

    def __init__(self, params: dict | None = None):
        p = {k: (dict(v) if isinstance(v, dict) else v)
             for k, v in _DEFAULT_PARAMS.items()}
        if params:
            for k, v in params.items():
                if isinstance(v, dict) and isinstance(p.get(k), dict):
                    p[k].update(v)
                else:
                    p[k] = v
        self.p = p

    def compute(
        self,
        cpi_T_baseline: float,
        shock_delta:    float,
        h_T:            np.ndarray,   # (n_assets,) conditional variance at T
        eps_T_sq:       np.ndarray,   # (n_assets,) squared innovation at T
        horizon:        int = 64,
    ) -> dict[str, np.ndarray]:
        """Compute analytical expectations for k = 1, …, horizon steps post-shock.

        Parameters
        ----------
        cpi_T_baseline : CPI level on the baseline path at time T.
        shock_delta    : Additive CPI shock  (e.g. +0.01 = +1 pp).
        h_T            : Conditional variances h_{i,T}  shape (n_assets,).
        eps_T_sq       : Realised ε²_{i,T}  shape (n_assets,);  same for both paths.
        horizon        : Number of future steps K.

        Returns
        -------
        dict with arrays (all float64):
          'E_cpi'        (K,)      — E[CPI_{T+k}]          shocked path
          'E_cpi_base'   (K,)      — E[CPI_{T+k}]          baseline path
          'E_h'          (K, 3)    — E[h_{i,T+k}]          shocked
          'E_h_base'     (K, 3)    — E[h_{i,T+k}]          baseline
          'E_ret'        (K, 3)    — E[R_{i,T+k}]          shocked
          'E_ret_base'   (K, 3)    — E[R_{i,T+k}]          baseline
          'delta_E_ret'  (K, 3)    — E[R]_shocked − E[R]_baseline  (ATE on return)
          'delta_E_vol'  (K, 3)    — √E[h]_shocked − √E[h]_baseline  (ATE on vol)
        """
        cpi_cfg = self.p["cpi"]
        a_cfg   = self.p["assets"]

        c0    = float(cpi_cfg["c0"])
        rho   = float(cpi_cfg["rho"])
        mu    = np.array(a_cfg["mu"],    dtype=np.float64)
        beta  = np.array(a_cfg["beta"],  dtype=np.float64)
        omega = np.array(a_cfg["omega"], dtype=np.float64)
        alpha = np.array(a_cfg["alpha"], dtype=np.float64)
        gamma = np.array(a_cfg["gamma"], dtype=np.float64)
        delta = np.array(a_cfg["delta"], dtype=np.float64)

        cpi_T_shocked = cpi_T_baseline + shock_delta
        mu_cpi        = c0 / (1.0 - rho)   # Unconditional CPI mean

        # ── CPI trajectories ─────────────────────────────────────────────────
        E_cpi      = np.empty(horizon)
        E_cpi_base = np.empty(horizon)
        for k in range(1, horizon + 1):
            E_cpi[k - 1]      = mu_cpi + rho**k * (cpi_T_shocked  - mu_cpi)
            E_cpi_base[k - 1] = mu_cpi + rho**k * (cpi_T_baseline - mu_cpi)

        # ── E[ΔCPI_{T+k}] for each path ──────────────────────────────────────
        E_dcpi      = np.empty(horizon)
        E_dcpi_base = np.empty(horizon)
        E_dcpi[0]      = E_cpi[0]      - cpi_T_shocked    # ΔCPI at T+1 (shocked path)
        E_dcpi_base[0] = E_cpi_base[0] - cpi_T_baseline   # ΔCPI at T+1 (baseline)
        for k in range(1, horizon):
            E_dcpi[k]      = E_cpi[k]      - E_cpi[k - 1]
            E_dcpi_base[k] = E_cpi_base[k] - E_cpi_base[k - 1]

        # ── Variance recursion ────────────────────────────────────────────────
        h_T      = np.asarray(h_T,     dtype=np.float64)
        eps_T_sq = np.asarray(eps_T_sq, dtype=np.float64)
        n        = len(mu)

        E_h      = np.empty((horizon, n))
        E_h_base = np.empty((horizon, n))

        # k = 1: uses realised ε²_T (same on both paths — shock affects CPI, not past ε)
        E_h[0]      = omega + alpha * eps_T_sq + gamma * E_dcpi[0]      + delta * h_T
        E_h_base[0] = omega + alpha * eps_T_sq + gamma * E_dcpi_base[0] + delta * h_T
        E_h[0]      = np.maximum(E_h[0],      1e-12)
        E_h_base[0] = np.maximum(E_h_base[0], 1e-12)

        # k ≥ 2: E[ε²] = E[h]  (Var(z) = 1 by standardisation)
        for k in range(1, horizon):
            E_h[k]      = (omega
                           + (alpha + delta) * E_h[k - 1]
                           + gamma * E_dcpi[k])
            E_h_base[k] = (omega
                           + (alpha + delta) * E_h_base[k - 1]
                           + gamma * E_dcpi_base[k])
            E_h[k]      = np.maximum(E_h[k],      1e-12)
            E_h_base[k] = np.maximum(E_h_base[k], 1e-12)

        # ── Expected returns ──────────────────────────────────────────────────
        E_ret      = mu + beta * E_cpi[:, np.newaxis]       # (K, 3)
        E_ret_base = mu + beta * E_cpi_base[:, np.newaxis]  # (K, 3)

        return {
            "E_cpi":        E_cpi,
            "E_cpi_base":   E_cpi_base,
            "E_h":          E_h,              # (K, 3)
            "E_h_base":     E_h_base,
            "E_ret":        E_ret,            # (K, 3)
            "E_ret_base":   E_ret_base,
            "delta_E_ret":  E_ret - E_ret_base,                      # ATE on mean
            "delta_E_vol":  np.sqrt(E_h) - np.sqrt(E_h_base),        # ATE on vol
        }

    # ------------------------------------------------------------------
    # Convenience wrapper for use with SyntheticDGP
    # ------------------------------------------------------------------

    def compute_from_path(
        self,
        dgp: SyntheticDGP,
        shock_T:     int,
        shock_delta: float,
        horizon:     int = 64,
    ) -> dict[str, np.ndarray]:
        """Extract state at shock_T from a simulated path and call compute().

        Parameters
        ----------
        dgp         : SyntheticDGP instance (generate_series must have been called).
        shock_T     : Time index in the simulated series where the shock occurs.
        shock_delta : Additive shock to CPI at time shock_T.
        horizon     : Steps K to compute.

        Returns
        -------
        Same dict as compute(), plus:
          'shock_T'   : int — shock time index
          'shock_delta': float
        """
        state = dgp.get_state_at(shock_T)
        result = self.compute(
            cpi_T_baseline = state["cpi"],
            shock_delta    = shock_delta,
            h_T            = state["h"],
            eps_T_sq       = state["eps_sq"],
            horizon        = horizon,
        )
        result["shock_T"]     = shock_T
        result["shock_delta"] = shock_delta
        return result


# ═════════════════════════════════════════════════════════════════════════════
# Task 3 — Pipeline integration helper (run())
# ═════════════════════════════════════════════════════════════════════════════

def run(project_root: Path, config: dict) -> None:
    """Generate synthetic data and write processed_data/ artifacts.

    Emits the **exact same files** as scripts/ingest.py so that
    scripts/preprocess.py (and all downstream scripts) run unchanged:
        processed_data/ticker_data_RAW.npy   — {ticker: pd.DataFrame(LogReturns)}
        processed_data/aligned_macro.parquet — pd.DataFrame, col = macro FRED-ID key
        processed_data/data_config.json

    Called automatically from scripts/ingest.py when
    ``synthetic_dgp.use_synthetic: true`` in config.yaml.
    """
    syn_cfg    = config.get("synthetic_dgp", {})
    dgp_params = syn_cfg.get("dgp_params", None)
    N_obs      = int(syn_cfg.get("n_obs", 20_000))
    seed       = int(syn_cfg.get("seed", 42))
    start_date = str(syn_cfg.get("start_date", "2000-01-03"))

    # ── Validate config consistency ───────────────────────────────────────────
    tickers = list(config["data_ingestion"]["tickers"])
    if len(tickers) != 3:
        raise ValueError(
            f"synthetic_dgp expects exactly 3 tickers in data_ingestion.tickers, "
            f"got {len(tickers)}: {tickers}"
        )
    macro_vars_cfg = dict(config["data_ingestion"]["macro_vars"])
    if len(macro_vars_cfg) != 1:
        raise ValueError(
            f"synthetic_dgp expects exactly 1 macro variable in data_ingestion.macro_vars, "
            f"got {len(macro_vars_cfg)}: {list(macro_vars_cfg)}"
        )
    macro_col = list(macro_vars_cfg.keys())[0]   # e.g. "SYNTH_CPI"

    # ── Simulate ─────────────────────────────────────────────────────────────
    dgp = SyntheticDGP(params=dgp_params, seed=seed)
    returns, macro = dgp.generate_series(N=N_obs)   # (N,3), (N,1)

    # ── Business-day DatetimeIndex ────────────────────────────────────────────
    date_idx = pd.bdate_range(start=start_date, periods=N_obs, freq="B")

    # ── ticker_data_RAW.npy  (same format as ingest.py) ──────────────────────
    ticker_aligned: dict = {}
    for i, ticker in enumerate(tickers):
        ticker_aligned[ticker] = pd.DataFrame(
            {"LogReturns": returns[:, i].astype(np.float32)},
            index=date_idx,
        )

    from src.paths import resolve_paths
    processed_dir = resolve_paths(project_root, config).processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "ticker_data_RAW.npy", ticker_aligned)

    # ── aligned_macro.parquet  (same format as ingest.py) ────────────────────
    macro_df = pd.DataFrame(
        {macro_col: macro[:, 0].astype(np.float64)},
        index=date_idx,
    )
    macro_df.to_parquet(processed_dir / "aligned_macro.parquet")

    # ── data_config.json ──────────────────────────────────────────────────────
    data_cfg = {
        "source":          "synthetic_dgp_garchx",
        "n_obs":           N_obs,
        "start_date":      str(date_idx[0].date()),
        "end_date":        str(date_idx[-1].date()),
        "train_cutoff":    config["data_ingestion"]["train_split_date"],
        "tickers":         tickers,
        "macro_vars":      macro_vars_cfg,
        "n_assets":        3,
        "n_macro":         1,
        "n_trading_days":  N_obs,
        "return_transform": "log_return_synthetic_garchx",
        "dgp_nu":          _DEFAULT_PARAMS["nu"] if dgp_params is None
                           else dgp_params.get("nu", _DEFAULT_PARAMS["nu"]),
    }
    with open(processed_dir / "data_config.json", "w") as fh:
        json.dump(data_cfg, fh, indent=4)

    print(f"[synthetic_dgp] Wrote {N_obs} obs, {len(tickers)} assets, 1 macro var")
    print(f"[synthetic_dgp] Date range: {date_idx[0].date()} → {date_idx[-1].date()}")
    print(f"[synthetic_dgp] Return std  per asset : {returns.std(axis=0)}")
    print(f"[synthetic_dgp] CPI range             : [{macro.min():.4f}, {macro.max():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Generate synthetic GARCH-X dataset")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.yaml (default: <project_root>/config.yaml)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import yaml  # noqa: PLC0415
    _config_path = Path(args.config) if args.config else _project_root / "config.yaml"
    with open(_config_path, "r", encoding="utf-8") as fh:
        _cfg = yaml.safe_load(fh)

    run(_project_root, _cfg)
