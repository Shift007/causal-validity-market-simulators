"""
Probabilistic calibration metrics for prediction interval evaluation.

Two families of metrics are provided:
1. **Interval calibration** (PICP / MPIW / Winkler) — assess whether a
   set of quantile paths constitutes a well-calibrated 80% prediction interval.
2. **Naive Random Walk baseline** — a simple Gaussian RW estimated from the
   asset-past window, providing a no-skill benchmark with identical conditioning.
3. **Causal metrics** (``compute_causal_metrics``) — evaluate model fidelity
   to an analytical ground-truth counterfactual effect (PEHE, Dir%, etc.).

Usage
-----
>>> from src.metrics.calibration import compute_interval_metrics, evaluate_naive_baseline
>>> met = compute_interval_metrics(y=real_price[1:], lower=p10[1:], upper=p90[1:])
>>> met["picp"], met["mpiw"], met["winkler"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Causal counterfactual metrics
# ---------------------------------------------------------------------------

def compute_causal_metrics(
    delta_true: np.ndarray,
    delta_pred_samples: np.ndarray,
    coverage_alpha: float = 0.10,
) -> dict[str, float]:
    """Five causal-inference metrics comparing predicted vs analytical effects.

    Parameters
    ----------
    delta_true : array_like, shape ``(...)``
        Ground-truth counterfactual effect from an analytical DGP
        (e.g. ``TheoreticalCounterfactual.compute()["delta_E_ret"]``).
    delta_pred_samples : array_like, shape ``(S, ...)``
        *S* generative samples of the model-predicted effect.
        With deterministic decoding (``n_samples=1``), ``S=1``;
        coverage_90 is returned as ``nan`` in that degenerate case.
    coverage_alpha : float
        Tail probability for the coverage interval.
        Default 0.10 → 90 % CI (5th–95th percentile across samples).

    Returns
    -------
    dict with keys:

    ``mae``
        Mean Absolute Error between the sample-mean prediction and the truth.
    ``directional_accuracy``
        Fraction of samples (averaged over all cells) that share the sign
        of ``delta_true``.  50 % = random, 100 % = perfect.
    ``amplification_ratio``
        ``mean(delta_pred_samples) / delta_true`` averaged over all cells
        (NaN-masked where ``|delta_true| < 1e-12``).
        1.0 = perfectly calibrated magnitude; > 1 = over-amplified;
        0–1 = under-amplified; negative = wrong direction.
    ``pehe``
        Precision in Estimation of Heterogeneous Effects — RMSE between
        the sample-mean prediction and the truth (lower is better).
    ``coverage_90``
        Fraction of cells where ``delta_true`` falls inside the 90 % CI
        of ``delta_pred_samples``.  ``nan`` when ``S == 1``.
    """
    dt: np.ndarray = np.asarray(delta_true,        dtype=np.float64)
    dp: np.ndarray = np.asarray(delta_pred_samples, dtype=np.float64)

    if dp.ndim == dt.ndim:
        dp = dp[np.newaxis]          # treat as S=1

    S = dp.shape[0]
    dp_mean: np.ndarray = dp.mean(axis=0)          # (...)

    # 1. MAE
    mae: float = float(np.abs(dp_mean - dt).mean())

    # 2. Directional accuracy — over all samples × all cells
    correct_sign: np.ndarray = np.sign(dp) == np.sign(dt)[np.newaxis]
    directional_accuracy: float = float(correct_sign.mean())

    # 3. Amplification ratio (signed, cell-wise, NaN where |true| is negligible)
    _eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_cell: np.ndarray = np.where(
            np.abs(dt) > _eps,
            dp_mean / dt,
            np.nan,
        )
    amplification_ratio: float = float(np.nanmean(ratio_cell))

    # 4. PEHE = RMSE of the sample-mean prediction vs truth
    pehe: float = float(np.sqrt(((dp_mean - dt) ** 2).mean()))

    # 5. Counterfactual coverage (90 %) — undefined for a single sample
    if S == 1:
        coverage_90: float = float("nan")
    else:
        p_lo: np.ndarray = np.percentile(dp, coverage_alpha * 100,        axis=0)
        p_hi: np.ndarray = np.percentile(dp, (1 - coverage_alpha) * 100,  axis=0)
        coverage_90 = float(((dt >= p_lo) & (dt <= p_hi)).mean())

    return {
        "mae":                  mae,
        "directional_accuracy": directional_accuracy,
        "amplification_ratio":  amplification_ratio,
        "pehe":                 pehe,
        "coverage_90":          coverage_90,
    }


# ---------------------------------------------------------------------------
# Core interval metric
# ---------------------------------------------------------------------------

def compute_interval_metrics(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> dict:
    """PICP, MPIW and Winkler score for a single 1D prediction interval.

    Parameters
    ----------
    y     : ``(T,)`` actual observed values
    lower : ``(T,)`` lower bound of the prediction interval
    upper : ``(T,)`` upper bound of the prediction interval
    alpha : mis-coverage level (default 0.2 → 80% PI)

    Returns
    -------
    dict with keys ``"picp"`` (float in [0,1]), ``"mpiw"`` (float ≥ 0),
    ``"winkler"`` (float ≥ 0).
    """
    T = len(y)
    width   = upper - lower
    inside  = (y >= lower) & (y <= upper)

    picp = float(inside.sum()) / T
    mpiw = float(width.mean())

    penalty = np.zeros(T)
    below   = y < lower
    above   = y > upper
    penalty[below] = (2.0 / alpha) * (lower[below] - y[below])
    penalty[above] = (2.0 / alpha) * (y[above]     - upper[above])

    winkler = float((width + penalty).mean())
    return {"picp": picp, "mpiw": mpiw, "winkler": winkler}


# ---------------------------------------------------------------------------
# Naive Random Walk baseline
# ---------------------------------------------------------------------------

def evaluate_naive_baseline(
    results: list[dict],
    tickers: list[str],
    n_assets: int,
    returns_to_price_fn,
    m_trajectories: int = 100,
    base_value: float = 100.0,
    alpha: float = 0.2,
    seed: int = 42,
    ticker_transforms: dict | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Gaussian Random Walk baseline for ALL assets.

    For each test window and each asset, estimates drift μ and σ from the
    historical ``past_raw`` context and simulates ``m_trajectories`` IID
    Gaussian paths.  Computes the same PICP / MPIW / Winkler metrics as the
    DDIM ensemble, providing a no-skill benchmark.

    Parameters
    ----------
    results : list of result dicts produced by the generation loop.
        Each dict must have keys ``"test_idx"``, ``"past_raw"`` ``(ctx, C)``,
        ``"real_raw"`` ``(W, C)``.
    tickers : list of ticker names (length C).
    n_assets : number of asset channels C.
    returns_to_price_fn : callable ``(returns, transform, base) → prices``.
    m_trajectories : number of RW paths per window per asset.
    base_value : price index origin (100.0 by convention).
    alpha : mis-coverage level for Winkler score (default 0.2 → 80% PI).
    seed : random seed for reproducibility.
    ticker_transforms : ``{ticker: transform_name}`` — same dict as used by the DDIM.

    Returns
    -------
    all_naive : list of ``{"per_asset": [dict, ...], "aggregate": dict}``
    detail_df : DataFrame with per-(window × asset) rows
    """
    rng = np.random.RandomState(seed)
    all_naive: list[dict] = []
    detail_rows: list[dict] = []

    for res in results:
        per_asset: list[dict] = []
        for c_idx in range(n_assets):
            past_raw_ch = res["past_raw"][:, c_idx]
            real_raw_ch = res["real_raw"][:, c_idx]
            W_len       = len(real_raw_ch)

            mu    = float(np.mean(past_raw_ch))
            sigma = float(np.std(past_raw_ch, ddof=1))

            rw_rets   = rng.normal(mu, max(sigma, 1e-8), size=(m_trajectories, W_len))
            tx        = (ticker_transforms or {}).get(tickers[c_idx], "log_return")
            rw_price  = returns_to_price_fn(rw_rets, tx, base=base_value)
            real_price = returns_to_price_fn(real_raw_ch, tx, base=base_value)
            junction  = base_value

            rw_future   = np.hstack([np.full((m_trajectories, 1), junction), rw_price])
            real_future = np.concatenate([[junction], real_price])

            p10 = np.percentile(rw_future, 10, axis=0)
            p90 = np.percentile(rw_future, 90, axis=0)

            met = compute_interval_metrics(
                y=real_future[1:], lower=p10[1:], upper=p90[1:], alpha=alpha,
            )
            per_asset.append(met)
            detail_rows.append({
                "Window":    res["test_idx"],
                "Ticker":    tickers[c_idx],
                "mu":        mu,
                "sigma":     sigma,
                "PICP (%)":  met["picp"] * 100,
                "MPIW":      met["mpiw"],
                "Winkler":   met["winkler"],
            })

        agg = {
            "picp":    float(np.mean([m["picp"]    for m in per_asset])),
            "mpiw":    float(np.mean([m["mpiw"]    for m in per_asset])),
            "winkler": float(np.mean([m["winkler"] for m in per_asset])),
        }
        all_naive.append({"per_asset": per_asset, "aggregate": agg})

    return all_naive, pd.DataFrame(detail_rows)
