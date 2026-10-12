#!/usr/bin/env python
"""
01 — Data Ingestion: download raw financial & macroeconomic data, align to a
unified daily calendar, compute log returns, and save raw artifacts.

Replaces notebooks/01_data_ingestion.ipynb.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --config path/to/config.yaml
"""

import os
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Data ingestion pipeline")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.yaml (default: <project_root>/config.yaml)")
    return ap.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Project root ─────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import yaml
    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Synthetic DGP bypass ─────────────────────────────────────────────
    if config.get("synthetic_dgp", {}).get("use_synthetic", False):
        from scripts.synthetic_dgp import run as _synthetic_run
        log.info("use_synthetic=true — delegating to scripts/synthetic_dgp.py")
        _synthetic_run(project_root, config)
        log.info("Synthetic data written. Run scripts/preprocess.py next.")
        return

    ingestion_cfg = config["data_ingestion"]
    from src.paths import resolve_paths
    processed_dir = resolve_paths(project_root, config).processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    fred_api_key = os.environ.get("FRED_API_KEY") or ingestion_cfg.get("fred_api_key")
    if not fred_api_key:
        raise SystemExit("Set the FRED_API_KEY environment variable (free key from fred.stlouisfed.org).")
    tickers = ingestion_cfg["tickers"]
    fred_vars = ingestion_cfg["macro_vars"]
    start_date = ingestion_cfg["start_date"]
    end_date = ingestion_cfg["end_date"]
    train_cutoff = ingestion_cfg["train_split_date"]

    log.info("Asset universe : %d tickers", len(tickers))
    log.info("Macro universe : %d FRED series", len(fred_vars))
    log.info("Date range     : %s → %s", start_date, end_date)
    log.info("Train cutoff   : %s", train_cutoff)

    # Save initial data_config
    data_config = {
        "start_date": start_date,
        "end_date": end_date,
        "train_cutoff": train_cutoff,
        "tickers": tickers,
        "macro_vars": fred_vars,
    }
    with open(processed_dir / "data_config.json", "w") as f:
        json.dump(data_config, f, indent=4)

    # ── Download Macro Data from FRED ────────────────────────────────────
    log.info("Downloading macro data from FRED...")
    QUARTERLY_SERIES = {"GDPC1", "CLVMNACSCAB1GQEA19"}
    fred = Fred(api_key=fred_api_key)
    macro_frames = []
    macro_failures = []
    macro_info = {}

    for freq_id, name in fred_vars.items():
        try:
            series = fred.get_series(freq_id)
            if series is None or series.empty:
                macro_failures.append((freq_id, "empty series"))
                continue

            series = series.dropna()

            if freq_id in QUARTERLY_SERIES:
                series = series.resample("MS").interpolate(method="cubic")
                freq_label = "quarterly→monthly (cubic)"
            else:
                freq_label = "monthly"

            d0, d1 = str(series.index[0].date()), str(series.index[-1].date())
            macro_info[freq_id] = {
                "name": name, "freq": freq_label,
                "start": d0, "end": d1, "rows": len(series),
            }
            macro_frames.append(series.to_frame(name=freq_id))
            log.info("  OK  %-28s %5d rows  %s → %s  [%s]",
                      freq_id, len(series), d0, d1, freq_label)
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else e.__class__.__name__
            macro_failures.append((freq_id, reason))
            log.warning("  FAIL %-28s %s", freq_id, reason)

    if macro_failures:
        log.warning("FAILED %d/%d macro series: %s",
                     len(macro_failures), len(fred_vars), macro_failures)
    if not macro_frames:
        raise RuntimeError("No macro data downloaded.")

    macro_df = pd.concat(macro_frames, axis=1).sort_index().ffill().dropna()
    log.info("Macro panel: %d variables × %d months  (%s → %s)",
             macro_df.shape[1], len(macro_df),
             macro_df.index[0].date(), macro_df.index[-1].date())

    # ── Download Asset Price Data from yfinance ──────────────────────────
    log.info("Downloading asset price data from yfinance...")
    ticker_data_raw = {}
    ticker_failures = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
            if df.empty:
                ticker_failures.append((ticker, "empty dataframe"))
                continue
            if len(df) < 500:
                ticker_failures.append((ticker, f"too short: {len(df)} rows"))
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                ticker_failures.append((ticker, "missing Close column"))
                continue
            ticker_data_raw[ticker] = df
            log.info("  OK  %-12s %5d rows  %s → %s",
                      ticker, len(df), df.index[0].date(), df.index[-1].date())
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else e.__class__.__name__
            ticker_failures.append((ticker, reason))

    if ticker_failures:
        log.warning("FAILED %d/%d tickers: %s",
                     len(ticker_failures), len(tickers), ticker_failures)
    if not ticker_data_raw:
        raise RuntimeError("No asset data downloaded.")

    log.info("Downloaded %d/%d assets", len(ticker_data_raw), len(tickers))

    # ── Feature Engineering & Calendar Alignment ─────────────────────────
    log.info("Computing log returns and aligning calendars...")
    ticker_processed = {}
    valid_starts, valid_ends = [], []

    for ticker, df in ticker_data_raw.items():
        close_series = df["Close"].replace(0, np.nan).ffill().bfill()
        ret_raw = np.log(close_series / close_series.shift(1))
        common_idx = ret_raw.dropna().index

        if len(common_idx) < 100:
            log.warning("SKIP %s: only %d valid returns", ticker, len(common_idx))
            continue

        df_aligned = pd.DataFrame({"LogReturns": ret_raw.loc[common_idx]})
        ticker_processed[ticker] = df_aligned
        valid_starts.append(df_aligned.index.min())
        valid_ends.append(df_aligned.index.max())

    log.info("Computed log_return for %d/%d assets",
             len(ticker_processed), len(ticker_data_raw))

    if not valid_starts:
        raise RuntimeError("No tickers processed successfully.")

    global_start = max(max(valid_starts), macro_df.index.min())
    global_end = min(min(valid_ends), macro_df.index.max())
    if global_start >= global_end:
        raise ValueError("No common time interval found.")

    # UNION calendar
    all_trading_dates = set()
    for t, df in ticker_processed.items():
        dates_in_range = df.loc[global_start:global_end].index
        all_trading_dates.update(dates_in_range)

    master_index = pd.DatetimeIndex(sorted(all_trading_dates))
    log.info("Master Calendar: %d unified trading days (%s → %s)",
             len(master_index), global_start.date(), global_end.date())

    # Reindex all tickers to master calendar
    ticker_aligned = {}
    for t, df in ticker_processed.items():
        reindexed = df.reindex(master_index)
        n_missing = reindexed["LogReturns"].isna().sum()
        reindexed["LogReturns"] = reindexed["LogReturns"].fillna(0.0)
        ticker_aligned[t] = reindexed
        pct_filled = n_missing / len(reindexed) * 100
        log.info("  %-12s: %4d/%d days filled (%.1f%% non-trading)",
                  t, n_missing, len(reindexed), pct_filled)

    # Macro: forward-fill monthly/quarterly data to daily calendar
    macro_aligned = macro_df.reindex(master_index).ffill().bfill()

    # Verify alignment
    idx_lens = [len(ticker_aligned[t]) for t in ticker_aligned]
    assert len(set(idx_lens)) == 1, f"Index length mismatch: {idx_lens}"
    log.info("All %d tickers aligned to %d common trading days",
             len(ticker_aligned), idx_lens[0])
    log.info("Macro data: %d variables × %d days",
             macro_aligned.shape[1], len(macro_aligned))

    # ── Sanity checks ────────────────────────────────────────────────────
    for t, df in ticker_aligned.items():
        assert df["LogReturns"].isna().sum() == 0, f"NaN in {t}"
    assert macro_aligned.isna().sum().sum() == 0, "NaN in macro"
    log.info("No NaN in aligned data")

    # ── Save Results ─────────────────────────────────────────────────────
    np.save(processed_dir / "ticker_data_RAW.npy", ticker_aligned)
    macro_aligned.to_parquet(processed_dir / "aligned_macro.parquet")

    final_config = {
        "start_date": str(global_start.date()),
        "end_date": str(global_end.date()),
        "train_cutoff": train_cutoff,
        "tickers": list(ticker_aligned.keys()),
        "macro_vars": fred_vars,
        "macro_info": macro_info,
        "n_assets": len(ticker_aligned),
        "n_macro": macro_aligned.shape[1],
        "n_trading_days": len(master_index),
        "return_transform": "log_return",
    }
    with open(processed_dir / "data_config.json", "w") as f:
        json.dump(final_config, f, indent=4)

    log.info("Saved to %s:", processed_dir)
    log.info("  ticker_data_RAW.npy   — %d tickers × %d days",
             len(ticker_aligned), idx_lens[0])
    log.info("  aligned_macro.parquet — %d vars × %d days",
             macro_aligned.shape[1], len(macro_aligned))
    log.info("  data_config.json      — full metadata")
    log.info("Done.")


if __name__ == "__main__":
    main()
