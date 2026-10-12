"""
Stylized facts metrics for financial time series evaluation.

Aligned with Kubiak et al. (ICAIF '25) Tables 1 & 2:
  - Distributional: mean, std, skew, kurtosis, avg pairwise correlation
  - Heavy tails: power-law tail exponents (alpha_pos, alpha_neg) via Hill MLE
  - Temporal: linear unpredictability (ACF of returns), volatility clustering
    (ACF of |returns|), coarse-fine volatility, leverage effect
  - Wasserstein-1: overall, per volatility decile, top 5%/1%

All functions operate on data shaped (N_windows, target_len, C) where C = N_assets.
Each metric is computed per channel (= per ticker asset) and returned as a flat dict
keyed by ``{metric_name}_{ticker}`` for easy merging.

Usage
-----
>>> from src.metrics.stylized_facts import (
...     compute_marginal_metrics, compute_percentile_wasserstein,
...     compute_kurtosis_diff, compute_acf_error,
...     compute_cross_asset_correlation_error,
...     compute_distributional_moments, compute_tail_exponents,
...     compute_linear_unpredictability, compute_coarse_fine_volatility,
...     compute_leverage_effect, compute_wasserstein_by_volatility_decile,
... )
>>> metrics = compute_marginal_metrics(real, synth, tickers)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, kurtosis, skew
from statsmodels.tsa.stattools import acf


# ---------------------------------------------------------------------------
# Marginal distribution metrics
# ---------------------------------------------------------------------------

def compute_marginal_metrics(
    real: np.ndarray,
    synth: np.ndarray,
    ch_names: list[str],
) -> dict:
    """Wasserstein-1 and KS test per channel (= per asset).

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``
    ch_names    : list of C ticker names

    Returns
    -------
    dict with keys ``wasserstein_{ticker}``, ``ks_stat_{ticker}``, ``ks_pval_{ticker}``.
    """
    C = real.shape[-1]
    results: dict = {}
    for ch in range(C):
        r = real[:, :, ch].ravel()
        s = synth[:, :, ch].ravel()
        results[f"wasserstein_{ch_names[ch]}"] = wasserstein_distance(r, s)
        ks_stat, ks_pval = ks_2samp(r, s)
        results[f"ks_stat_{ch_names[ch]}"] = ks_stat
        results[f"ks_pval_{ch_names[ch]}"] = ks_pval
    return results


def compute_percentile_wasserstein(
    real: np.ndarray,
    synth: np.ndarray,
    ch_names: list[str],
    n_quantiles: int = 99,
) -> dict:
    """Wasserstein-1 computed on quantile vectors (1st–99th percentile).

    Captures shape differences rather than mass differences.

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``
    ch_names    : list of C ticker names
    n_quantiles : number of evenly-spaced probability levels

    Returns
    -------
    dict with keys ``wasserstein_pctl_{ticker}``.
    """
    C = real.shape[-1]
    probs = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    results: dict = {}
    for ch in range(C):
        r = real[:, :, ch].ravel()
        s = synth[:, :, ch].ravel()
        q_real  = np.quantile(r, probs)
        q_synth = np.quantile(s, probs)
        results[f"wasserstein_pctl_{ch_names[ch]}"] = wasserstein_distance(q_real, q_synth)
    return results


def compute_kurtosis_diff(
    real: np.ndarray,
    synth: np.ndarray,
    ch_names: list[str],
) -> dict:
    """Excess kurtosis difference (Synth − Real) per channel.

    Delta kurtosis measures fat-tail fidelity. Positive Δ = synthetic has heavier tails.

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``
    ch_names    : list of C ticker names

    Returns
    -------
    dict with keys ``kurtosis_real_{t}``, ``kurtosis_synth_{t}``,
    ``delta_kurtosis_{t}``, ``delta_kurtosis_pp_{t}`` (percent-points of real).
    """
    C = real.shape[-1]
    results: dict = {}
    for ch in range(C):
        r_kurt = kurtosis(real[:, :, ch].ravel(), fisher=True)
        s_kurt = kurtosis(synth[:, :, ch].ravel(), fisher=True)
        delta  = s_kurt - r_kurt
        delta_pp = (delta / abs(r_kurt)) * 100.0 if abs(r_kurt) > 1e-8 else np.nan
        results[f"kurtosis_real_{ch_names[ch]}"]     = r_kurt
        results[f"kurtosis_synth_{ch_names[ch]}"]    = s_kurt
        results[f"delta_kurtosis_{ch_names[ch]}"]    = delta
        results[f"delta_kurtosis_pp_{ch_names[ch]}"] = delta_pp
    return results


# ---------------------------------------------------------------------------
# Temporal / dependence metrics
# ---------------------------------------------------------------------------

def compute_acf_error(
    real: np.ndarray,
    synth: np.ndarray,
    ch_names: list[str],
    max_lag: int = 20,
) -> dict:
    """ACF L2 error on absolute-value returns — proxy for volatility clustering.

    The ACF of ``|r_t|`` measures ARCH-like persistence. Matching this indicates
    the model has learned volatility clustering dynamics.

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``
    ch_names    : list of C ticker names
    max_lag     : maximum ACF lag (capped at target_len // 2)

    Returns
    -------
    dict with keys ``acf_l2_{t}``, ``acf_real_{t}`` (array), ``acf_synth_{t}`` (array).
    """
    C = real.shape[-1]
    max_lag = min(max_lag, real.shape[1] // 2)
    results: dict = {}

    for ch in range(C):
        acf_real_list  = []
        acf_synth_list = []
        for i in range(real.shape[0]):
            r_abs = np.abs(real[i, :, ch])
            s_abs = np.abs(synth[i, :, ch])
            if r_abs.std() > 1e-8:
                acf_real_list.append(acf(r_abs, nlags=max_lag, fft=True))
            if s_abs.std() > 1e-8:
                acf_synth_list.append(acf(s_abs, nlags=max_lag, fft=True))

        if acf_real_list and acf_synth_list:
            mean_acf_real  = np.mean(acf_real_list,  axis=0)
            mean_acf_synth = np.mean(acf_synth_list, axis=0)
            l2_err = float(np.sqrt(np.sum((mean_acf_real - mean_acf_synth) ** 2)))
        else:
            mean_acf_real = mean_acf_synth = np.zeros(max_lag + 1)
            l2_err = np.nan

        results[f"acf_l2_{ch_names[ch]}"]    = l2_err
        results[f"acf_real_{ch_names[ch]}"]  = mean_acf_real
        results[f"acf_synth_{ch_names[ch]}"] = mean_acf_synth

    return results


def compute_cross_asset_correlation_error(
    real: np.ndarray,
    synth: np.ndarray,
) -> dict:
    """Cross-asset Pearson correlation matrix MAE.

    Measures whether the model preserves inter-asset dependency structure.

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``

    Returns
    -------
    dict with keys ``cross_asset_corr_mae`` (scalar),
    ``corr_real`` (C×C), ``corr_synth`` (C×C).
    """
    C = real.shape[-1]
    r_flat = real.reshape(-1, C)
    s_flat = synth.reshape(-1, C)

    corr_real  = np.corrcoef(r_flat.T)
    corr_synth = np.corrcoef(s_flat.T)
    mae = float(np.mean(np.abs(corr_real - corr_synth)))

    return {
        "cross_asset_corr_mae": mae,
        "corr_real":  corr_real,
        "corr_synth": corr_synth,
    }


# ---------------------------------------------------------------------------
# Convenience: merge per-channel dicts into per-ticker dict
# ---------------------------------------------------------------------------

def merge_per_asset_metrics(
    ch_names: list[str],
    marginal: dict,
    pctl_wass: dict,
    kurt_diff: dict,
    acf_mets: dict,
) -> dict[str, dict]:
    """Merge all per-channel flat dicts into ``{ticker: {metric: value}}``.

    Parameters
    ----------
    ch_names  : list of C ticker names
    marginal  : output of ``compute_marginal_metrics``
    pctl_wass : output of ``compute_percentile_wasserstein``
    kurt_diff : output of ``compute_kurtosis_diff``
    acf_mets  : output of ``compute_acf_error``

    Returns
    -------
    ``{ticker: {"wasserstein": ..., "ks_stat": ..., ...}}`` for each ticker.
    """
    result: dict[str, dict] = {}
    for ch, ticker in enumerate(ch_names):
        result[ticker] = {
            "wasserstein":       marginal.get(f"wasserstein_{ticker}"),
            "wasserstein_pctl":  pctl_wass.get(f"wasserstein_pctl_{ticker}"),
            "ks_stat":           marginal.get(f"ks_stat_{ticker}"),
            "ks_pval":           marginal.get(f"ks_pval_{ticker}"),
            "kurtosis_real":     kurt_diff.get(f"kurtosis_real_{ticker}"),
            "kurtosis_synth":    kurt_diff.get(f"kurtosis_synth_{ticker}"),
            "delta_kurtosis":    kurt_diff.get(f"delta_kurtosis_{ticker}"),
            "delta_kurtosis_pp": kurt_diff.get(f"delta_kurtosis_pp_{ticker}"),
            "acf_l2":            acf_mets.get(f"acf_l2_{ticker}"),
            "acf_real":          acf_mets.get(f"acf_real_{ticker}"),
            "acf_synth":         acf_mets.get(f"acf_synth_{ticker}"),
        }
    return result


# ---------------------------------------------------------------------------
# Paper-aligned metrics (Kubiak et al., ICAIF '25, Tables 1 & 2)
# ---------------------------------------------------------------------------

def compute_distributional_moments(
    data: np.ndarray,
    ch_names: list[str],
    scale_bps: float = 1e4,
) -> dict:
    """Mean, std, skewness, kurtosis, and avg pairwise correlation.

    Returns values in basis points for mean/std (×10^4), unitless for
    skew/kurt/corr.  Matches Table 2 columns 1-5.

    Parameters
    ----------
    data : ``(N_windows, target_len, C)``
    ch_names : list of C ticker names
    scale_bps : scaling factor for mean/std (default 1e4 = basis points)
    """
    N, T, C = data.shape
    flat = data.reshape(-1, C)  # (N*T, C)
    results: dict = {}

    means, stds, skews, kurts = [], [], [], []
    for ch in range(C):
        r = flat[:, ch]
        m = float(np.mean(r)) * scale_bps
        s = float(np.std(r)) * scale_bps
        sk = float(skew(r))
        kt = float(kurtosis(r, fisher=True))
        results[f"mean_{ch_names[ch]}"] = m
        results[f"std_{ch_names[ch]}"] = s
        results[f"skew_{ch_names[ch]}"] = sk
        results[f"kurt_{ch_names[ch]}"] = kt
        means.append(m); stds.append(s); skews.append(sk); kurts.append(kt)

    # Average pairwise correlation
    corr_mat = np.corrcoef(flat.T)  # (C, C)
    triu_idx = np.triu_indices(C, k=1)
    avg_corr = float(np.mean(corr_mat[triu_idx]))

    results["mean_avg"] = float(np.mean(means))
    results["std_avg"] = float(np.mean(stds))
    results["skew_avg"] = float(np.mean(skews))
    results["kurt_avg"] = float(np.mean(kurts))
    results["corr_avg"] = avg_corr
    return results


def _hill_estimator(tail_data: np.ndarray, k: int | None = None) -> float:
    """Hill MLE for the power-law tail exponent alpha.

    Parameters
    ----------
    tail_data : 1D array of strictly positive tail observations.
    k : number of upper order statistics to use.  Default: sqrt(n).

    Returns
    -------
    alpha : float — estimated tail exponent.
    """
    x = np.sort(tail_data)[::-1]  # descending
    n = len(x)
    if n < 10:
        return np.nan
    if k is None:
        k = max(10, int(np.sqrt(n)))
    k = min(k, n - 1)
    log_ratios = np.log(x[:k] / x[k])
    if np.any(log_ratios <= 0):
        # Degenerate tail — fallback
        return np.nan
    alpha = float(k / np.sum(log_ratios))
    return alpha


def compute_tail_exponents(
    data: np.ndarray,
    ch_names: list[str],
) -> dict:
    """Power-law tail exponents (alpha_pos, alpha_neg) via Hill MLE.

    Matches Table 2 columns alpha_pos and alpha_neg.

    Parameters
    ----------
    data : ``(N_windows, target_len, C)``
    ch_names : list of C ticker names
    """
    C = data.shape[-1]
    results: dict = {}
    alphas_pos, alphas_neg = [], []
    for ch in range(C):
        r = data[:, :, ch].ravel()
        pos = r[r > 0]
        neg = -r[r < 0]  # flip sign so all positive
        a_pos = _hill_estimator(pos)
        a_neg = _hill_estimator(neg)
        results[f"alpha_pos_{ch_names[ch]}"] = a_pos
        results[f"alpha_neg_{ch_names[ch]}"] = a_neg
        if not np.isnan(a_pos): alphas_pos.append(a_pos)
        if not np.isnan(a_neg): alphas_neg.append(a_neg)

    results["alpha_pos_avg"] = float(np.mean(alphas_pos)) if alphas_pos else np.nan
    results["alpha_neg_avg"] = float(np.mean(alphas_neg)) if alphas_neg else np.nan
    return results


def compute_linear_unpredictability(
    data: np.ndarray,
    ch_names: list[str],
    max_lag: int = 10,
) -> dict:
    """ACF of raw returns (not absolute), averaged over windows and lags 1-max_lag.

    Measures linear unpredictability: near-zero ACF of returns is a key
    stylized fact.  Matches Table 2 "Lin. Unpred." column.

    Parameters
    ----------
    data : ``(N_windows, target_len, C)``
    ch_names : list of C ticker names
    max_lag : maximum ACF lag (paper uses 10)
    """
    N, T, C = data.shape
    max_lag = min(max_lag, T // 2)
    results: dict = {}
    avg_values = []

    for ch in range(C):
        acf_list = []
        for i in range(N):
            r = data[i, :, ch]
            if r.std() > 1e-8:
                a = acf(r, nlags=max_lag, fft=True)
                acf_list.append(a[1:])  # exclude lag 0 (=1.0)
        if acf_list:
            mean_acf = np.mean(acf_list, axis=0)
            avg_val = float(np.mean(mean_acf))
        else:
            mean_acf = np.zeros(max_lag)
            avg_val = np.nan
        results[f"lin_unpred_{ch_names[ch]}"] = avg_val
        results[f"lin_unpred_acf_{ch_names[ch]}"] = mean_acf
        if not np.isnan(avg_val):
            avg_values.append(avg_val)

    results["lin_unpred_avg"] = float(np.mean(avg_values)) if avg_values else np.nan
    return results


def compute_coarse_fine_volatility(
    data: np.ndarray,
    ch_names: list[str],
    tau: int = 5,
    max_lag: int = 10,
) -> dict:
    """Coarse-fine volatility correlation at lags 1-max_lag with block size tau.

    Coarse vol = |sum of returns over tau days|.
    Fine vol = sum of |returns| over tau days.
    The correlation between lagged coarse and fine volatility captures
    gain/loss asymmetry.  Matches Table 2 "Coarse-Fine" column.

    Parameters
    ----------
    data : ``(N_windows, target_len, C)``
    ch_names : list of C ticker names
    tau : block size in days (paper uses 5)
    max_lag : maximum lag in blocks
    """
    N, T, C = data.shape
    n_blocks = T // tau
    if n_blocks < max_lag + 2:
        max_lag = max(1, n_blocks - 2)
    results: dict = {}
    avg_values = []

    for ch in range(C):
        corr_per_lag = np.zeros(max_lag)
        count = 0
        for i in range(N):
            r = data[i, :, ch]
            coarse = np.array([np.abs(r[b*tau:(b+1)*tau].sum())
                               for b in range(n_blocks)])
            fine = np.array([np.abs(r[b*tau:(b+1)*tau]).sum()
                             for b in range(n_blocks)])
            if coarse.std() < 1e-10 or fine.std() < 1e-10:
                continue
            for lag in range(1, max_lag + 1):
                if lag < n_blocks:
                    c_val = np.corrcoef(coarse[:-lag], fine[lag:])[0, 1]
                    if not np.isnan(c_val):
                        corr_per_lag[lag - 1] += c_val
            count += 1

        if count > 0:
            corr_per_lag /= count
        avg_val = float(np.mean(corr_per_lag))
        results[f"coarse_fine_{ch_names[ch]}"] = avg_val
        results[f"coarse_fine_lags_{ch_names[ch]}"] = corr_per_lag
        if count > 0:
            avg_values.append(avg_val)

    results["coarse_fine_avg"] = float(np.mean(avg_values)) if avg_values else np.nan
    return results


def compute_leverage_effect(
    data: np.ndarray,
    ch_names: list[str],
    max_lag: int = 10,
) -> dict:
    """Leverage effect: Cov(r_t, r_{t+k}^2) / Var(r_t) for lags 1-max_lag.

    Negative values indicate that negative returns predict higher future
    volatility (the "leverage effect").  Matches Table 2 "Leverage" column.

    Parameters
    ----------
    data : ``(N_windows, target_len, C)``
    ch_names : list of C ticker names
    max_lag : maximum lag (paper uses 10)
    """
    N, T, C = data.shape
    max_lag = min(max_lag, T // 2)
    results: dict = {}
    avg_values = []

    for ch in range(C):
        lev_per_lag = np.zeros(max_lag)
        count = 0
        for i in range(N):
            r = data[i, :, ch]
            var_r = np.var(r)
            if var_r < 1e-16:
                continue
            for lag in range(1, max_lag + 1):
                if lag < T:
                    r_t = r[:T - lag]
                    r_sq = r[lag:] ** 2
                    cov_val = np.mean((r_t - r_t.mean()) * (r_sq - r_sq.mean()))
                    lev_per_lag[lag - 1] += cov_val / var_r
            count += 1

        if count > 0:
            lev_per_lag /= count
        avg_val = float(np.mean(lev_per_lag))
        results[f"leverage_{ch_names[ch]}"] = avg_val
        results[f"leverage_lags_{ch_names[ch]}"] = lev_per_lag
        if count > 0:
            avg_values.append(avg_val)

    results["leverage_avg"] = float(np.mean(avg_values)) if avg_values else np.nan
    return results


def compute_wasserstein_by_volatility_decile(
    real: np.ndarray,
    synth: np.ndarray,
    spx_index: int,
    ch_names: list[str],
) -> dict:
    """W1 distance stratified by S&P 500 realized volatility deciles.

    Groups evaluation windows by the realized volatility of the S&P 500
    asset (``spx_index``), then computes Wasserstein-1 per decile and for
    the top 5% and top 1% most volatile windows.  Matches Table 1.

    Parameters
    ----------
    real, synth : ``(N_windows, target_len, C)``
    spx_index : int — column index of ES=F / S&P 500 in the C channels
    ch_names : list of C ticker names

    Returns
    -------
    dict with keys:
      ``w1_decile_{d}`` (d=1..10),
      ``w1_top5pct``, ``w1_top1pct``,
      ``w1_overall``,
      ``decile_counts``, ``decile_boundaries``.
    """
    N = real.shape[0]
    # Realized vol per window = std of SPX returns
    spx_vol = np.array([real[i, :, spx_index].std() for i in range(N)])

    # Decile thresholds from realized vol
    decile_edges = np.percentile(spx_vol, np.arange(0, 110, 10))
    decile_labels = np.digitize(spx_vol, decile_edges[1:-1], right=True)  # 0..9

    results: dict = {}
    results["decile_boundaries"] = decile_edges.tolist()
    results["decile_counts"] = [int(np.sum(decile_labels == d)) for d in range(10)]

    # Per-decile W1 (flattened across all assets within each decile)
    for d in range(10):
        mask = decile_labels == d
        if mask.sum() == 0:
            results[f"w1_decile_{d+1}"] = np.nan
            continue
        r_flat = real[mask].ravel()
        s_flat = synth[mask].ravel()
        results[f"w1_decile_{d+1}"] = float(wasserstein_distance(r_flat, s_flat))

    # Top 5% and top 1%
    for pct, label in [(95, "top5pct"), (99, "top1pct")]:
        threshold = np.percentile(spx_vol, pct)
        mask = spx_vol >= threshold
        if mask.sum() == 0:
            results[f"w1_{label}"] = np.nan
            continue
        results[f"w1_{label}"] = float(
            wasserstein_distance(real[mask].ravel(), synth[mask].ravel()))

    # Overall
    results["w1_overall"] = float(
        wasserstein_distance(real.ravel(), synth.ravel()))

    return results
