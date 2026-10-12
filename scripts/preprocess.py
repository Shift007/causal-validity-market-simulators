#!/usr/bin/env python
"""
02 — Preprocessing: build multi-asset sliding windows, fit QuantileTransformer,
compute wavelet level_scales, and save all dataset artifacts.

Replaces notebooks/02_preprocessing.ipynb.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config path/to/config.yaml
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis as sp_kurtosis_fn, skew as sp_skew_fn
from sklearn.preprocessing import QuantileTransformer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Preprocessing pipeline")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.yaml (default: <project_root>/config.yaml)")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def expanding_robust_tanh_normalize(df, c=3.0, min_periods=20, return_stats=False):
    """Expanding robust-tanh normalization — CAUSAL, bounded to (-1, 1).

    If ``return_stats=True`` also returns the per-row (expanding) median and MAD
    used in the transform. These are needed to invert tanh→raw in paper-faithful
    counterfactual shock application."""
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    medians = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    mads = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        s = df[col].astype(float)
        exp_median = s.expanding(min_periods=min_periods).median()
        exp_mad = (s - exp_median).abs().expanding(min_periods=min_periods).median()
        exp_mad = exp_mad.clip(lower=1e-8)
        z = (s - exp_median) / (1.4826 * exp_mad)
        result[col] = np.tanh(z / c)
        medians[col] = exp_median
        mads[col] = exp_mad
    if return_stats:
        return result, medians, mads
    return result


def safe_ffill_first(s, warn=True):
    """Forward-fill a Series, back-filling only the leading NaNs."""
    if s.isna().all():
        return s
    first_valid_idx = s.first_valid_index()
    if first_valid_idx is not None:
        first_pos = s.index.get_loc(first_valid_idx)
        s.iloc[:first_pos + 1] = s.loc[first_valid_idx]
    s = s.ffill()
    return s


def apply_macro_stationarity_transforms(macro_df, config):
    """Apply per-variable stationarity transforms to raw macro DataFrame."""
    macro_vars_cfg = config["data_ingestion"]["macro_vars"]
    fred_to_col = {fid: name for fid, name in macro_vars_cfg.items()}
    col_to_fred = {v: k for k, v in fred_to_col.items()}

    macro_cfg = config.get("preprocessing", {}).get("macro_scaling", {})
    stationarity_cfg = macro_cfg.get("stationarity_transforms", {})
    engineered_cfg = macro_cfg.get("engineered_features", {})

    result = macro_df.copy()

    # Engineered features from RAW levels (before transforms)
    for feat_name, feat_cfg in engineered_cfg.items():
        comp_ids = feat_cfg["components"]
        col_a = fred_to_col.get(comp_ids[0], comp_ids[0])
        col_b = fred_to_col.get(comp_ids[1], comp_ids[1])
        if col_a in result.columns and col_b in result.columns:
            result[feat_name] = result[col_a] - result[col_b]
            log.info("  Engineered: %s = %s − %s  (range: [%.4f, %.4f])",
                      feat_name, comp_ids[0], comp_ids[1],
                      result[feat_name].min(), result[feat_name].max())
        else:
            log.warning("  Cannot compute %s: missing %s or %s", feat_name, col_a, col_b)

    # Build transform lookup
    transform_lookup = {}
    for fred_id, transform_type in stationarity_cfg.items():
        transform_lookup[fred_id] = transform_type
    for feat_name, feat_cfg in engineered_cfg.items():
        transform_lookup[feat_name] = feat_cfg.get("transform", "none")

    # Apply per-column stationarity transforms
    transform_log = {}
    for col in result.columns:
        fred_id = col_to_fred.get(col, col)
        transform = transform_lookup.get(fred_id, "none")
        raw = result[col].copy()

        if transform == "log_diff":
            result[col] = np.log(raw.clip(lower=1e-10)).diff()
        elif transform == "yoy_log_diff":
            result[col] = np.log(raw.clip(lower=1e-10)) - np.log(raw.clip(lower=1e-10).shift(252))
        elif transform == "first_diff":
            result[col] = raw.diff()
        elif transform == "level":
            result[col] = raw
        elif transform == "log_level":
            result[col] = np.log(raw.clip(lower=1e-10))
        elif transform == "binary_map":
            result[col] = (raw * 2) - 1
            log.info("  Binary map: %s (%s) → mapped {0,1} → {-1,+1}", fred_id, col)
        elif transform == "already_growth":
            pass  # Data already in growth rate / YoY% from source — no transform needed
        # else 'none': leave as-is

        transform_log[col] = transform

    # Drop rows where ANY column is NaN from stationarity transforms
    n_before = len(result)
    result = result.dropna()
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        log.info("  Dropped %d leading NaN rows after stationarity transforms. %d rows remain.",
                  n_dropped, len(result))

    return result, transform_log


def expanding_zscore_normalize(df, min_periods=20):
    """Expanding z-score: (x - expanding_mean) / expanding_std. Causal."""
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        s = df[col].astype(float)
        exp_mean = s.expanding(min_periods=min_periods).mean()
        exp_std  = s.expanding(min_periods=min_periods).std()
        exp_std  = exp_std.clip(lower=1e-8)
        result[col] = (s - exp_mean) / exp_std
    return result


def save_diagnostic_report(report_path, sections):
    """Save a plain-text diagnostic report."""
    with open(report_path, "w", encoding="utf-8") as f:
        for title, lines in sections:
            f.write(f"{'=' * 80}\n{title}\n{'=' * 80}\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
    log.info("Diagnostic report saved → %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Project root & config ────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import yaml
    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.preprocessing.instance_dataset import compute_wavelet_level_scales

    # ── Paths & config extraction ────────────────────────────────────────
    from src.paths import resolve_paths
    paths = resolve_paths(project_root, config)
    # Unified processed_data root: ingest outputs (ticker_data_RAW.npy, aligned_macro.parquet)
    # and preprocess artifacts (raw_windows, QT, level_scales, …) share one folder.
    data_dir = paths.processed_dir
    processed_dir = paths.processed_dir
    results_dir = paths.results_dir / "diagnostics"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg = config["preprocessing"]
    target_len = prep_cfg["target_length"]
    context_len = prep_cfg.get("window_length", target_len)
    total_window_len = context_len + target_len
    macro_context_len = prep_cfg.get("macro_context_len", target_len)
    stride = prep_cfg["stride"]
    wavelet_type = prep_cfg["wavelet"]["type"]
    wavelet_level = prep_cfg["wavelet"]["level"]
    train_split_date = config["data_ingestion"]["train_split_date"]
    tickers = config["data_ingestion"]["tickers"]
    n_assets = len(tickers)

    inst_cfg = prep_cfg.get("instance_scaling", {})
    c_scale = inst_cfg.get("c", 3.0)
    mad_floor = inst_cfg.get("mad_floor", 1e-4)
    z_clip = inst_cfg.get("z_clip", 10.0)

    td_cfg = prep_cfg.get("target_distribution", {})
    strict_artifact_validation = bool(td_cfg.get("strict_artifact_validation", True))

    target_distribution_mode = td_cfg.get("mode", "optA")
    if target_distribution_mode not in ("optA", "none"):
        raise ValueError(
            f"target_distribution.mode={target_distribution_mode!r} not supported "
            "(allowed: 'optA' standard QT, 'none' ablation = skip QT). "
            "Set mode: optA or none in config."
        )
    # When mode=='none' the QT stage is bypassed entirely (ablation pipeline).
    selected_td_mode_cfg = td_cfg.get(target_distribution_mode, {}) if target_distribution_mode == "optA" else {}

    assert macro_context_len >= target_len

    macro_cfg = prep_cfg.get("macro_scaling", {})
    min_periods_macro = macro_cfg.get("min_periods", 20)
    tanh_c = macro_cfg.get("tanh_c", 3.0)

    log.info("=== MULTI-ASSET JOINT DISTRIBUTION (Log Returns) ===")
    log.info("Assets: %d, Context: %d, Target: %d, Total: %d",
             n_assets, context_len, target_len, total_window_len)
    log.info("Wavelet: %s level %d, Stride: %d", wavelet_type, wavelet_level, stride)
    log.info("Instance scaling: c=%.1f, mad_floor=%.0e, z_clip=%.1f",
             c_scale, mad_floor, z_clip)
    if target_distribution_mode == "optA":
        log.info("Target distribution: optA (standard QuantileTransformer + clip)")
    else:
        log.info("Target distribution: NONE (ablation — QT skipped, level_scales on raw z-residuals)")

    # Collect report sections
    report_sections = []

    # ── 1. Load ingest outputs ─────────────────────────────────────────
    log.info("Loading ingest outputs...")
    RAW_PATH = processed_dir / "ticker_data_RAW.npy"
    if not RAW_PATH.exists():
        legacy = processed_dir / "ticker_data.npy"
        if legacy.exists():
            log.warning("ticker_data_RAW.npy not found — falling back to ticker_data.npy")
            RAW_PATH = legacy
        else:
            raise FileNotFoundError("Neither ticker_data_RAW.npy nor ticker_data.npy found. Run ingest.py first.")

    ticker_raw = np.load(RAW_PATH, allow_pickle=True).item()
    macro_raw = pd.read_parquet(processed_dir / "aligned_macro.parquet")

    available_tickers = [t for t in tickers if t in ticker_raw]
    missing = [t for t in tickers if t not in ticker_raw]
    if missing:
        log.warning("Missing tickers in raw data (skipped): %s", missing)
    tickers = available_tickers
    n_assets = len(tickers)

    # Anti-leak sanity check
    sample_ticker = tickers[0]
    sample_raw = ticker_raw[sample_ticker]
    ret_vals = sample_raw["LogReturns"].values
    ret_std = np.nanstd(ret_vals)
    if ret_std < 1e-6:
        raise ValueError(f"ANTI-LEAK CHECK FAILED: LogReturns std={ret_std:.6f} for {sample_ticker}")
    log.info("Anti-leak check passed (LogReturns std=%.6f for %s)", ret_std, sample_ticker)

    # Extract log returns
    ticker_returns = {}
    for ticker in tickers:
        df = ticker_raw[ticker]
        lr = safe_ffill_first(df["LogReturns"].copy()).astype(np.float32)
        ticker_returns[ticker] = lr

    common_dates = None
    for ticker in tickers:
        idx = ticker_returns[ticker].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    log_returns_df = pd.DataFrame(
        {ticker: ticker_returns[ticker].loc[common_dates].values for ticker in tickers},
        index=common_dates,
    )
    log.info("Log returns matrix: %s", log_returns_df.shape)

    # Report: signal stats
    stat_lines = [f"{'Ticker':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"]
    stat_lines.append("-" * 55)
    for ticker in tickers:
        v = log_returns_df[ticker].values
        stat_lines.append(
            f"  {ticker:<10} {np.nanmean(v):>10.6f} {np.nanstd(v):>10.6f} "
            f"{np.nanmin(v):>10.6f} {np.nanmax(v):>10.6f}"
        )
    report_sections.append(("Signal Statistics (Raw Log Returns)", stat_lines))

    # ── 2a. Signal: no global normalization ──────────────────────────────
    for col in log_returns_df.columns:
        log_returns_df[col] = safe_ffill_first(log_returns_df[col].copy())
    signal_arr = log_returns_df.values.astype(np.float32)
    signal_dates = log_returns_df.index

    log.info("Signal array: %s, NaN count: %d", signal_arr.shape, np.isnan(signal_arr).sum())

    # ── 2b. Macro preprocessing: stationarity → expanding robust-tanh ───
    log.info("Applying macro stationarity transforms...")
    macro_transformed, macro_transform_log = apply_macro_stationarity_transforms(macro_raw, config)

    macro_vars_cfg = config["data_ingestion"]["macro_vars"]
    fred_to_col = {fid: name for fid, name in macro_vars_cfg.items()}
    col_to_fred = {v: k for k, v in fred_to_col.items()}

    # Report: stationarity transforms
    st_lines = [f"  {'FRED ID':<28s} {'Column':<38s} {'Transform':<12s} {'Valid Range':>22s}"]
    st_lines.append("  " + "─" * 102)
    for col in macro_transformed.columns:
        fred_id = col_to_fred.get(col, col)
        transform = macro_transform_log.get(col, "none")
        valid = macro_transformed[col].dropna()
        rng_str = f"[{valid.min():.6f}, {valid.max():.6f}]" if len(valid) > 0 else "[NaN]"
        st_lines.append(f"  {fred_id:<28s} {col[:37]:<38s} {transform:<12s} {rng_str:>22s}")
    report_sections.append(("Macro Stationarity Transforms", st_lines))

    # Stationarity diagnostic
    train_split_ts = pd.Timestamp(train_split_date)
    sd_lines = [f"  {'Column':<38s} {'std(train)':>11s} {'std(test)':>11s} {'ratio':>7s} {'Status':>8s}"]
    sd_lines.append("  " + "─" * 79)
    for col in macro_transformed.columns:
        tr = macro_transformed[col].loc[macro_transformed.index <= train_split_ts].dropna()
        te = macro_transformed[col].loc[macro_transformed.index > train_split_ts].dropna()
        if len(tr) > 0 and len(te) > 0:
            tr_std, te_std = tr.std(), te.std()
            ratio = te_std / (tr_std + 1e-10)
            status = "OK" if 0.5 < ratio < 2.0 else "DRIFT!"
            sd_lines.append(f"  {col[:37]:<38s} {tr_std:>11.6f} {te_std:>11.6f} {ratio:>7.2f} {status:>8s}")
    report_sections.append(("Stationarity Diagnostic — std(train) vs std(test)", sd_lines))

    # Expanding robust-tanh
    stationarity_cfg = macro_cfg.get("stationarity_transforms", {})
    binary_cols = set()
    for fred_id, tf in stationarity_cfg.items():
        if tf == "binary_map":
            col_name = fred_to_col.get(fred_id, fred_id)
            binary_cols.add(col_name)

    continuous_cols = [c for c in macro_transformed.columns if c not in binary_cols]
    binary_col_list = [c for c in macro_transformed.columns if c in binary_cols]

    log.info("Applying expanding robust-tanh (c=%.1f) to %d continuous macro vars...",
             tanh_c, len(continuous_cols))
    if binary_col_list:
        log.info("  Skipping robust-tanh for binary_map columns: %s", binary_col_list)

    if continuous_cols:
        macro_continuous_scaled, _diff_medians_continuous, _diff_mads_continuous = \
            expanding_robust_tanh_normalize(
                macro_transformed[continuous_cols], c=tanh_c,
                min_periods=min_periods_macro, return_stats=True,
            )
    else:
        macro_continuous_scaled = pd.DataFrame(index=macro_transformed.index)
        _diff_medians_continuous = pd.DataFrame(index=macro_transformed.index)
        _diff_mads_continuous = pd.DataFrame(index=macro_transformed.index)

    macro_scaled = macro_continuous_scaled.copy()
    for col in binary_col_list:
        macro_scaled[col] = macro_transformed[col]
    macro_scaled = macro_scaled[macro_transformed.columns]

    # Diffusion paper-faithful CF support: per-row expanding median + MAD aligned
    # with macro_scaled columns. Binary cols (not tanh-transformed) get placeholders
    # so de-/re-tanh round-trip is a no-op (we never shock binary vars).
    diff_macro_median_df = pd.DataFrame(index=macro_transformed.index,
                                        columns=macro_transformed.columns, dtype=float)
    diff_macro_mad_df = pd.DataFrame(index=macro_transformed.index,
                                     columns=macro_transformed.columns, dtype=float)
    for col in macro_transformed.columns:
        if col in _diff_medians_continuous.columns:
            diff_macro_median_df[col] = _diff_medians_continuous[col]
            diff_macro_mad_df[col] = _diff_mads_continuous[col]
        else:
            diff_macro_median_df[col] = 0.0
            diff_macro_mad_df[col] = 1.0
    # Back-fill leading NaNs (from expanding-window min_periods) the same way we
    # do for macro_scaled — without this the tanh→raw inverse blows up at NaNs.
    for col in diff_macro_median_df.columns:
        diff_macro_median_df[col] = safe_ffill_first(diff_macro_median_df[col].copy())
        diff_macro_mad_df[col] = safe_ffill_first(diff_macro_mad_df[col].copy())
    # Guard against any residual NaN / zero MAD (shouldn't happen, but be defensive).
    diff_macro_mad_df = diff_macro_mad_df.fillna(1.0).clip(lower=1e-8)
    diff_macro_median_df = diff_macro_median_df.fillna(0.0)

    for col in macro_scaled.columns:
        macro_scaled[col] = safe_ffill_first(macro_scaled[col].copy())

    log.info("Macro scaled range: [%.4f, %.4f]", macro_scaled.min().min(), macro_scaled.max().max())

    total_nans = macro_scaled.isna().sum().sum()
    if total_nans > 0:
        log.warning("%d NaNs remain in macro_scaled!", total_nans)
    else:
        log.info("No NaNs in macro_scaled — ready for windowing.")

    # ── 2c. MacroVAE macro: 6-month diffs + IN-SAMPLE (TRAIN-ONLY) z-score
    # Paper-aligned: Kubiak ICAIF 2025 §4.1 — "Indicators are standardized using
    # in-sample means and standard deviations." Replaces our previous
    # expanding_zscore_normalize which created spurious ±11σ outliers from
    # small expanding-std in early years (5 → 286 outliers on real vs synth)
    # and left trended dims (e.g. CN Exports) with persistent +1.79 bias.
    # NOTE: only the MVAE macro_zscore pathway is affected. The diffusion
    # macro_scaled (expanding_robust_tanh) is unchanged.
    log.info("Building MacroVAE z-scored macro (in-sample, train-only stats)...")
    SIXM_LAG = 126
    macro_6m_diff = macro_transformed.diff(SIXM_LAG).fillna(0)
    macro_6m_diff.columns = [f"{col}_6m_diff" for col in macro_6m_diff.columns]
    macro_ext = pd.concat([macro_transformed, macro_6m_diff], axis=1)
    D_ext = macro_ext.shape[1]
    D_base = macro_transformed.shape[1]  # base vars before momentum

    # Compute IN-SAMPLE (train) μ/σ on macro_ext rows where date ≤ train_split.
    # macro_ext.index is the date index; train_split_ts already defined above.
    _in_sample_mask = macro_ext.index <= train_split_ts
    mu_train = macro_ext.loc[_in_sample_mask].mean()
    sd_train = macro_ext.loc[_in_sample_mask].std().clip(lower=1e-8)
    macro_zscore = (macro_ext - mu_train) / sd_train
    for col in macro_zscore.columns:
        macro_zscore[col] = safe_ffill_first(macro_zscore[col].copy())

    # Per-window μ/σ for de-/re-standardization.
    # In paper-aligned form these are GLOBAL constants — we broadcast them to
    # (T, D_ext) so the downstream code (which indexes per-window) keeps working.
    mvae_exp_mean_df = pd.DataFrame(
        np.broadcast_to(mu_train.values, (len(macro_ext.index), D_ext)).copy(),
        index=macro_ext.index, columns=macro_ext.columns,
    )
    mvae_exp_std_df = pd.DataFrame(
        np.broadcast_to(sd_train.values, (len(macro_ext.index), D_ext)).copy(),
        index=macro_ext.index, columns=macro_ext.columns,
    )

    log.info("macro_zscore (IN-SAMPLE): %s, D_ext=%d (D_base=%d)",
             macro_zscore.shape, D_ext, D_base)
    log.info("  TRAIN-set μ range: [%+.4f, %+.4f]  σ range: [%.6f, %.6f]",
             float(mu_train.min()), float(mu_train.max()),
             float(sd_train.min()), float(sd_train.max()))
    log.info("  Test-portion z-stat: mean=%.3f std=%.3f (target ≈ 0, 1)",
             float(macro_zscore.loc[~_in_sample_mask].values.mean()),
             float(macro_zscore.loc[~_in_sample_mask].values.std()))

    # ── 3. Sliding windows ───────────────────────────────────────────────
    log.info("Creating sliding windows...")
    D = macro_scaled.shape[1]

    macro_dates = macro_scaled.index
    common_dates_all = signal_dates.intersection(macro_dates).sort_values()
    log.info("Common dates (signal ∩ macro): %d", len(common_dates_all))

    signal_aligned = log_returns_df.loc[common_dates_all].values.astype(np.float32)
    macro_aligned = macro_scaled.loc[common_dates_all].values.astype(np.float32)
    date_arr = common_dates_all.values
    macro_zscore_aligned = macro_zscore.loc[common_dates_all].values.astype(np.float32)
    mvae_exp_mean_aligned = mvae_exp_mean_df.loc[common_dates_all].values.astype(np.float32)
    mvae_exp_std_aligned = mvae_exp_std_df.loc[common_dates_all].values.astype(np.float32)
    diff_median_aligned = diff_macro_median_df.loc[common_dates_all].values.astype(np.float32)
    diff_mad_aligned = diff_macro_mad_df.loc[common_dates_all].values.astype(np.float32)

    T = len(signal_aligned)

    raw_windows_list = []
    macro_windows_list = []
    macro_zscore_windows_list = []
    mvae_exp_mean_list = []
    mvae_exp_std_list = []
    diff_median_windows_list = []
    diff_mad_windows_list = []
    window_dates_list = []

    for start in range(0, T - total_window_len + 1, stride):
        end = start + total_window_len
        # Pre-target macro window: it ends the day before the target sequence
        # starts. This follows the MacroVAE specification (Kubiak et al., §4.1:
        # the most recent values available before the first day of the
        # sequence) and gives the diffusion encoder a lookback with no overlap
        # with the target period.
        # macro_end (exclusive) = start + context_len = first day of target.
        # macro_start = macro_end - macro_context_len = e.g. 126 days back
        # (matches the MVAE 6-month momentum lag). With macro_context_len=126,
        # the first ~62 windows are skipped (insufficient pre-history).
        macro_end = end - target_len
        macro_start = macro_end - macro_context_len
        if macro_start < 0:
            continue

        raw_windows_list.append(signal_aligned[start:end])
        macro_windows_list.append(macro_aligned[macro_start:macro_end])
        macro_zscore_windows_list.append(macro_zscore_aligned[macro_start:macro_end])
        # macro_end - 1 = day before target start (paper-correct anchor for MVAE
        # snapshot and for the diffusion's last_val statistic).
        mvae_exp_mean_list.append(mvae_exp_mean_aligned[macro_end - 1])
        mvae_exp_std_list.append(mvae_exp_std_aligned[macro_end - 1])
        diff_median_windows_list.append(diff_median_aligned[macro_start:macro_end])
        diff_mad_windows_list.append(diff_mad_aligned[macro_start:macro_end])
        # window_dates kept at last target day for downstream period filtering
        # (window_end_ts selects the evaluation blocks).
        window_dates_list.append(date_arr[end - 1])

    raw_windows = np.array(raw_windows_list, dtype=np.float32)
    macro_windows = np.array(macro_windows_list, dtype=np.float32)
    macro_windows_zscore = np.array(macro_zscore_windows_list, dtype=np.float32)
    mvae_exp_mean = np.array(mvae_exp_mean_list, dtype=np.float32)
    mvae_exp_std = np.array(mvae_exp_std_list, dtype=np.float32)
    diff_macro_median_windows = np.array(diff_median_windows_list, dtype=np.float32)
    diff_macro_mad_windows = np.array(diff_mad_windows_list, dtype=np.float32)
    window_dates = np.array(window_dates_list)

    N_total = len(raw_windows)
    log.info("Total windows: %d", N_total)
    log.info("  raw_windows:          %s", raw_windows.shape)
    log.info("  macro_windows:        %s", macro_windows.shape)
    log.info("  macro_windows_zscore: %s", macro_windows_zscore.shape)

    # ── Train / Test split + Transition-Gap Removal ──────────────────────
    window_end_ts = pd.to_datetime(window_dates)
    target_start_approx = window_end_ts - pd.tseries.offsets.BDay(target_len - 1)

    is_train_end = window_end_ts <= train_split_ts
    target_after_split = target_start_approx > train_split_ts
    is_gap = (~is_train_end) & (~target_after_split)

    n_gap = is_gap.sum()
    log.info("Split date: %s", train_split_date)
    log.info("  Before gap removal: TRAIN=%d, TEST=%d, GAP=%d",
             is_train_end.sum(), target_after_split.sum(), n_gap)

    if n_gap > 0:
        keep_mask = ~is_gap
        raw_windows = raw_windows[keep_mask]
        macro_windows = macro_windows[keep_mask]
        macro_windows_zscore = macro_windows_zscore[keep_mask]
        mvae_exp_mean = mvae_exp_mean[keep_mask]
        mvae_exp_std = mvae_exp_std[keep_mask]
        diff_macro_median_windows = diff_macro_median_windows[keep_mask]
        diff_macro_mad_windows = diff_macro_mad_windows[keep_mask]
        window_dates = window_dates[keep_mask]
        window_end_ts = pd.to_datetime(window_dates)
        train_mask = np.array(window_end_ts <= train_split_ts)
        log.info("  After gap removal: TRAIN=%d, TEST=%d, removed %d gap windows",
                 train_mask.sum(), (~train_mask).sum(), n_gap)
    else:
        keep_mask = np.ones(len(raw_windows), dtype=bool)
        train_mask = np.array(is_train_end)

    N = len(raw_windows)
    log.info("Final dataset: %d windows (Train: %d, Test: %d)",
             N, train_mask.sum(), (~train_mask).sum())

    # ── 3b. Per-step MacroVAE z-score (Fix B: walk-forward in-sample stats) ──
    # Paper §4.1 says "indicators are standardized using in-sample means and
    # standard deviations". With quarterly retraining (paper), in-sample stats
    # naturally expand. With our annual retrain, each step S needs its own
    # mu_S/sd_S computed from data ≤ cutoff_S. This avoids OOD cond inputs
    # (which previously hit ±30σ in 2020/2021 → FiLM explosion).
    log.info("Computing per-step MacroVAE z-scored windows...")
    mvae_perstep_dir = data_dir / "mvae_perstep"
    mvae_perstep_dir.mkdir(parents=True, exist_ok=True)

    wf_cfg = config["training"]["walk_forward"]
    eval_end_ts = pd.Timestamp(wf_cfg["evaluation_end_date"])
    step_freq = str(wf_cfg.get("step_freq", "Q")).upper()
    if step_freq not in ("A", "Y", "YS"):
        log.warning("Per-step macro stats currently only support annual freq; "
                    "step_freq=%s will produce only 'initial' cutoff.", step_freq)

    cutoff_specs = [("initial", train_split_ts)]
    if step_freq in ("A", "Y", "YS"):
        for y in range(train_split_ts.year + 1, eval_end_ts.year + 1):
            cutoff = min(pd.Timestamp(f"{y}-12-31"), eval_end_ts)
            cutoff_specs.append((str(y), cutoff))

    log.info("  Cutoffs: %s", [f"{lbl}({ct.date()})" for lbl, ct in cutoff_specs])

    macro_ext_aligned_full = macro_ext.loc[common_dates_all]
    cutoff_summary = []
    for label, cutoff_ts in cutoff_specs:
        mask_in_sample = macro_ext.index <= cutoff_ts
        mu_c = macro_ext.loc[mask_in_sample].mean()
        sd_c = macro_ext.loc[mask_in_sample].std().clip(lower=1e-8)
        z_c = (macro_ext - mu_c) / sd_c
        for col in z_c.columns:
            z_c[col] = safe_ffill_first(z_c[col].copy())
        z_aligned = z_c.loc[common_dates_all].values.astype(np.float32)

        # Rebuild sliding windows using the SAME positions as the main loop.
        # KEEP this in sync with the main windowing (macro_end = end - target_len).
        z_win_list = []
        for start in range(0, T - total_window_len + 1, stride):
            end = start + total_window_len
            macro_end = end - target_len
            macro_start = macro_end - macro_context_len
            if macro_start < 0:
                continue
            z_win_list.append(z_aligned[macro_start:macro_end])
        z_win = np.array(z_win_list, dtype=np.float32)

        # Apply gap-removal to stay aligned with raw_windows / train_mask / window_dates.
        z_win = z_win[keep_mask]

        # Broadcast μ/σ to per-window arrays (constants under in-sample formulation).
        mean_per_win = np.broadcast_to(
            mu_c.values.astype(np.float32), (len(z_win), D_ext)
        ).copy()
        std_per_win = np.broadcast_to(
            sd_c.values.astype(np.float32), (len(z_win), D_ext)
        ).copy()

        np.save(mvae_perstep_dir / f"macro_zscore_{label}.npy", z_win)
        np.save(mvae_perstep_dir / f"exp_mean_{label}.npy", mean_per_win)
        np.save(mvae_perstep_dir / f"exp_std_{label}.npy", std_per_win)

        # Diagnostic: train-only and test-portion stats under this cutoff.
        z_train_vals = z_c.loc[mask_in_sample].values
        z_test_vals = z_c.loc[~mask_in_sample].values if (~mask_in_sample).any() else None
        summary = {
            "label": label,
            "cutoff": cutoff_ts.strftime("%Y-%m-%d"),
            "n_in_sample_rows": int(mask_in_sample.sum()),
            "z_train_mean": float(z_train_vals.mean()),
            "z_train_std": float(z_train_vals.std()),
            "z_test_mean": float(z_test_vals.mean()) if z_test_vals is not None else None,
            "z_test_std": float(z_test_vals.std()) if z_test_vals is not None else None,
            "z_test_max_abs": float(np.abs(z_test_vals).max()) if z_test_vals is not None else None,
        }
        cutoff_summary.append(summary)
        log.info("  [%-7s] cutoff=%s  train z=(%.3f,%.3f)  test z=(%s,%s)  test |max|=%s",
                 label, cutoff_ts.date(),
                 summary["z_train_mean"], summary["z_train_std"],
                 f"{summary['z_test_mean']:+.3f}" if summary["z_test_mean"] is not None else "n/a",
                 f"{summary['z_test_std']:.3f}" if summary["z_test_std"] is not None else "n/a",
                 f"{summary['z_test_max_abs']:.2f}" if summary["z_test_max_abs"] is not None else "n/a")

    with open(mvae_perstep_dir / "cutoffs.json", "w", encoding="utf-8") as f:
        json.dump(cutoff_summary, f, indent=2)
    log.info("Saved per-step artifacts: %s", mvae_perstep_dir)

    report_sections.append(("Window Summary", [
        f"  Total windows:     {N}",
        f"  raw_windows:       {raw_windows.shape}",
        f"  macro_windows:     {macro_windows.shape}",
        f"  Train: {train_mask.sum()} ({train_mask.mean()*100:.1f}%)",
        f"  Test:  {(~train_mask).sum()} ({(~train_mask).mean()*100:.1f}%)",
        f"  Gap removed:       {n_gap}",
    ]))

    # ── 4. Fit target distribution transformer + level_scales ────────────
    log.info("Fitting target distribution transformer (mode=%s)...", target_distribution_mode)

    train_raw = raw_windows[train_mask]
    N_train = len(train_raw)

    # Instance-scale train windows → collect z-score targets
    z_targets_list = []
    for i in range(N_train):
        window = train_raw[i]
        past = window[:context_len]
        target = window[context_len:]
        inst_median = np.median(past, axis=0)
        inst_mad = np.maximum(np.median(np.abs(past - inst_median), axis=0), mad_floor)
        denom = c_scale * 1.4826 * inst_mad
        z_target = np.clip((target - inst_median) / denom, -z_clip, z_clip)
        z_targets_list.append(z_target)

    z_targets_all = np.concatenate(z_targets_list, axis=0)
    log.info("Pooled z-score residuals: %s", z_targets_all.shape)

    # Report: pre-transform z-score statistics
    zt_lines = [f"   {'Asset':<12s} {'mean':>9s} {'std':>8s} {'skew':>8s} {'kurtosis':>10s}"]
    zt_lines.append("   " + "-" * 52)
    for c_idx in range(n_assets):
        v = z_targets_all[:, c_idx]
        zt_lines.append(
            f"   {tickers[c_idx]:<12s} {v.mean():>+9.5f} {v.std():>8.5f} "
            f"{sp_skew_fn(v):>+8.4f} {sp_kurtosis_fn(v, fisher=True):>+10.3f}"
        )
    report_sections.append(("Pre-Transform Z-Score Statistics", zt_lines))

    # Fit transformer (standard QuantileTransformer + percentile clip)
    qt_path = data_dir / "target_quantile_transformer.pkl"

    if target_distribution_mode == "optA":
        opta_cfg = td_cfg.get("optA", {})
        qt_obj = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(int(opta_cfg.get("n_quantiles", 2000)), N_train * target_len),
            random_state=42,
            subsample=min(int(opta_cfg.get("subsample", 200000)), N_train * target_len),
        )
        qt_obj.fit(z_targets_all)
        gauss_check = qt_obj.transform(z_targets_all)
        clip_percentiles = tuple(opta_cfg.get("clip_percentiles", [0.25, 99.75]))
        gauss_limits = np.percentile(gauss_check, clip_percentiles, axis=0)
        qt_artifact = {
            "quantile_transformer": qt_obj,
            "gauss_limits": gauss_limits,
            "asset_order": tickers,
            "percentiles": list(clip_percentiles),
            "method": "standard_qt_with_clipping",
            "mode": "optA",
        }
        qt_for_wavelet = qt_obj

        log.info("Fitted optA QT: %d samples x %d channels",
                 z_targets_all.shape[0], n_assets)

        # Report: post-transform statistics
        pt_lines = [f"   {'Asset':<12s} {'mean':>9s} {'std':>8s} {'skew':>8s} {'kurtosis':>10s}"]
        pt_lines.append("   " + "-" * 52)
        for c_idx in range(n_assets):
            v = gauss_check[:, c_idx]
            pt_lines.append(
                f"   {tickers[c_idx]:<12s} {v.mean():>+9.5f} {v.std():>8.5f} "
                f"{sp_skew_fn(v):>+8.4f} {sp_kurtosis_fn(v, fisher=True):>+10.3f}"
            )
        roundtrip_err = np.abs(z_targets_all - qt_obj.inverse_transform(gauss_check)).max()
        pt_lines.append(f"   Roundtrip max error: {roundtrip_err:.2e}")
        report_sections.append(("Post-Transform Statistics (should be ~N(0,1))", pt_lines))

        joblib.dump(qt_artifact, qt_path)
        log.info("Saved transformer artifact → %s", qt_path.name)
    else:
        # mode == "none" — skip QT entirely. level_scales operate on raw z-residuals.
        qt_artifact = None
        qt_for_wavelet = None
        log.info("QT skipped (mode=none): no transformer artifact saved")
        report_sections.append(("Post-Transform Statistics", [
            "   QT skipped — level_scales computed on RAW z-residuals (fat-tailed)."
        ]))

    # Compute level_scales (Gaussianised if QT present, raw z otherwise)
    if qt_artifact is not None:
        log.info("Computing level_scales on Gaussianised SWT coefficients...")
    else:
        log.info("Computing level_scales on RAW z-residual SWT coefficients (no QT)...")
    level_scales = compute_wavelet_level_scales(
        raw_windows=train_raw,
        context_len=context_len,
        target_len=target_len,
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level,
        c=c_scale,
        mad_floor=mad_floor,
        z_clip=z_clip,
        quantile_transformer=qt_artifact,
        n_samples=min(5000, len(train_raw)),
    )

    np.save(data_dir / "level_scales.npy", level_scales.astype(np.float32))
    log.info("level_scales %s saved", level_scales.shape)

    # Report: level_scales
    n_show = min(6, n_assets)
    ls_lines = [f"   {'Level':<8s} {' '.join(f'{tickers[i]:>10s}' for i in range(n_show))}"]
    ls_lines.append("   " + "-" * (10 + 11 * n_show))
    for level_idx in range(level_scales.shape[0]):
        row = " ".join(f"{level_scales[level_idx, i]:>10.4f}" for i in range(n_show))
        ls_lines.append(f"   {level_idx:<8d} {row}")
    report_sections.append(("Level Scales (Gaussianised SWT std per level×channel)", ls_lines))

    # ── 5. Save all artifacts ────────────────────────────────────────────
    log.info("Saving all artifacts...")

    np.save(data_dir / "raw_windows.npy", raw_windows.astype(np.float32))
    np.save(data_dir / "macro_windows.npy", macro_windows.astype(np.float32))
    np.save(data_dir / "diff_macro_median.npy", diff_macro_median_windows.astype(np.float32))
    np.save(data_dir / "diff_macro_mad.npy", diff_macro_mad_windows.astype(np.float32))
    np.save(data_dir / "train_mask.npy", train_mask)
    np.save(data_dir / "window_dates.npy", window_dates)

    # Global MacroVAE artifacts: use the LATEST cutoff (so any code that reads
    # `macro_windows_zscore.npy` / `mvae_macro_exp_mean.npy` directly — e.g.
    # generation, which always uses the latest checkpoint, — gets stats
    # consistent with that ckpt's training normalization.
    _latest_label = cutoff_specs[-1][0]
    np.save(data_dir / "macro_windows_zscore.npy",
            np.load(mvae_perstep_dir / f"macro_zscore_{_latest_label}.npy"))
    np.save(data_dir / "mvae_macro_exp_mean.npy",
            np.load(mvae_perstep_dir / f"exp_mean_{_latest_label}.npy"))
    np.save(data_dir / "mvae_macro_exp_std.npy",
            np.load(mvae_perstep_dir / f"exp_std_{_latest_label}.npy"))
    log.info("Legacy MVAE artifacts (macro_windows_zscore / exp_mean / exp_std) "
             "point to cutoff '%s'", _latest_label)

    if qt_artifact is None:
        qt_obj_meta = None
        qt_artifact_method = "none"
        _norm_method = "instance_scaling_swt_no_qt"
        _ls_basis = "raw_zresidual_swt_coefficients"
        _norm_sig = "instance_scaling_only"
    else:
        qt_obj_meta = qt_artifact["quantile_transformer"] if isinstance(qt_artifact, dict) else qt_artifact
        qt_artifact_method = qt_artifact.get("method", "legacy") if isinstance(qt_artifact, dict) else "legacy"
        _norm_method = "hybrid_instance_scaling_qt_swt"
        _ls_basis = "gaussianised_swt_coefficients"
        _norm_sig = "hybrid_instance_scaling_qt"

    norm_params = {
        "method": _norm_method,
        "context_len": context_len,
        "target_len": target_len,
        "total_window_len": total_window_len,
        "wavelet_type": wavelet_type,
        "wavelet_level": wavelet_level,
        "c": c_scale,
        "mad_floor": mad_floor,
        "z_clip": z_clip,
        "tickers": tickers,
        "n_assets": n_assets,
        "feature": "log_returns",
        "macro_context_len": macro_context_len,
        "train_split_date": str(train_split_date),
        "macro_stationarity_transforms": macro_transform_log,
        "macro_columns": macro_scaled.columns.tolist(),
        # macro_tanh_c: scaling constant used by expanding_robust_tanh on macro
        # inputs. Persisted so generation-time code (cf shock conversion) reads
        # the actual value used at preprocessing, not a fallback constant.
        "macro_tanh_c": float(tanh_c),
        "normalization_signal": _norm_sig,
        "normalization_macro": "stationarity_then_expanding_robust_tanh_c3_bounded",
        "level_scales_basis": _ls_basis,
        "leakage_prevention": "transition_gap_windows_removed",
        "target_distribution_mode": target_distribution_mode,
        "target_distribution_method": qt_artifact_method,
        "target_distribution_params": selected_td_mode_cfg,
        "mvae_macro_columns": macro_zscore.columns.tolist(),
        "mvae_d_ext": D_ext,
        "mvae_d_base": D_base,
    }
    np.save(data_dir / "normalization_params.npy", norm_params)

    log.info("  raw_windows:     %s", raw_windows.shape)
    log.info("  macro_windows:   %s", macro_windows.shape)
    log.info("  macro_win_zsc:   %s", macro_windows_zscore.shape)
    log.info("  mvae_exp_mean:   %s", mvae_exp_mean.shape)
    log.info("  mvae_exp_std:    %s", mvae_exp_std.shape)
    log.info("  level_scales:    %s (basis=%s)", level_scales.shape, _ls_basis)
    log.info("  train_mask:      Train=%d, Test=%d", train_mask.sum(), (~train_mask).sum())

    # ── 6. Verification: reload & sanity check ───────────────────────────
    log.info("Running verification sanity check...")

    ls_reload = np.load(data_dir / "level_scales.npy")

    if target_distribution_mode == "optA":
        qt_reload_data = joblib.load(data_dir / "target_quantile_transformer.pkl")

        def classify_transform_artifact(obj):
            if isinstance(obj, dict):
                if obj.get("method") == "evt_hybrid" or obj.get("mode") == "optB":
                    return "optB"
                if "gauss_limits" in obj:
                    return "optA"
            return "unknown"

        reloaded_mode = classify_transform_artifact(qt_reload_data)
        if strict_artifact_validation and reloaded_mode != target_distribution_mode:
            raise ValueError(
                f"Artifact mismatch: config={target_distribution_mode}, reloaded={reloaded_mode}"
            )
    else:
        # mode == "none": no QT artifact on disk, level_scales basis is raw z-residuals.
        qt_reload_data = None
        reloaded_mode = "none"

    ls_recomputed = compute_wavelet_level_scales(
        raw_windows=raw_windows[train_mask],
        context_len=context_len,
        target_len=target_len,
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level,
        c=c_scale,
        mad_floor=mad_floor,
        z_clip=z_clip,
        quantile_transformer=qt_reload_data,
        n_samples=min(5000, int(train_mask.sum())),
    )

    max_diff = np.abs(ls_reload - ls_recomputed).max()
    log.info("Recomputed level_scales vs saved: max diff = %.2e", max_diff)
    assert max_diff < 1e-5, f"Level scales mismatch! {max_diff}"
    if target_distribution_mode == "optA":
        log.info("level_scales verified: consistent with Gaussianised pipeline")
    else:
        log.info("level_scales verified: consistent with raw-z (no-QT) pipeline")

    # ── Save diagnostic report ───────────────────────────────────────────
    report_sections.append(("Verification", [
        f"  Reloaded mode: {reloaded_mode}",
        f"  level_scales max diff: {max_diff:.2e}",
        f"  Status: PASSED",
    ]))
    save_diagnostic_report(results_dir / "preprocess_report.txt", report_sections)

    log.info("All artifacts saved successfully. Done.")


if __name__ == "__main__":
    main()
