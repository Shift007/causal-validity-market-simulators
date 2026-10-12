"""Identity tests for src/metrics/interventional.py.

On gen == true (identical arrays) every distance/error should collapse to
~0 (see module docstring for the documented O(1/n) caveat on the unbiased
MMD^2 estimator -- n is chosen large enough below to make that negligible).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.interventional import (
    energy_per_asset,
    interventional_report,
    interventional_vol_error,
    mmd_rbf,
    tail_effect_error,
    wasserstein1_per_asset,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_wasserstein1_identity(rng):
    x = rng.standard_t(df=5, size=(500, 4))
    w1 = wasserstein1_per_asset(x, x)
    assert w1.shape == (4,)
    assert np.allclose(w1, 0.0, atol=1e-10)


def test_energy_identity(rng):
    x = rng.standard_t(df=5, size=(500, 4))
    en = energy_per_asset(x, x)
    assert en.shape == (4,)
    assert np.allclose(en, 0.0, atol=1e-10)


def test_mmd_identity(rng):
    x = rng.standard_normal(size=(400, 3))
    mmd = mmd_rbf(x, x)
    # Unbiased MMD^2(X,X) is O(1/n), not exactly 0 -- see module docstring.
    assert abs(mmd) < 1e-2


def test_mmd_identity_1d(rng):
    x = rng.standard_normal(size=(400,))
    mmd = mmd_rbf(x, x)
    assert abs(mmd) < 1e-2


def test_mmd_distinguishes_different_distributions(rng):
    x = rng.standard_normal(size=(500, 1))
    y = rng.standard_normal(size=(500, 1)) * 3.0 + 2.0
    mmd_same = mmd_rbf(x, x)
    mmd_diff = mmd_rbf(x, y)
    assert mmd_diff > mmd_same
    assert mmd_diff > 0.05


def test_vol_error_identity():
    rng = np.random.default_rng(1)
    sig_m0 = np.abs(rng.standard_normal(size=(50, 6))) + 0.01
    sig_m1 = sig_m0 * 1.2
    out = interventional_vol_error(sig_m1, sig_m0, sig_m1, sig_m0)
    assert out["dsigma_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert out["dsigma_bias"] == pytest.approx(0.0, abs=1e-12)
    assert out["dsigma_corr"] == pytest.approx(1.0, abs=1e-8)


def test_vol_error_detects_mismatch():
    rng = np.random.default_rng(2)
    sig_m0 = np.abs(rng.standard_normal(size=(50, 6))) + 0.01
    sig_true_m1 = sig_m0 * 1.5
    sig_gen_m1 = sig_m0.copy()  # gen predicts NO vol change (e.g. OLS)
    out = interventional_vol_error(sig_gen_m1, sig_m0, sig_true_m1, sig_m0)
    assert out["dsigma_rmse"] > 0.0
    true_effect_rmse = float(np.sqrt(np.mean((sig_true_m1 - sig_m0) ** 2)))
    assert out["dsigma_rmse"] == pytest.approx(true_effect_rmse, rel=1e-8)


def test_tail_effect_identity(rng):
    x = rng.standard_t(df=5, size=(1000, 4))
    out = tail_effect_error(x, x, alpha=0.05)
    for k in ("q_alpha_error", "q_median_error", "q_1malpha_error", "es_error"):
        assert np.allclose(out[k], 0.0, atol=1e-10)


def test_tail_effect_identity_1d(rng):
    x = rng.standard_t(df=5, size=(1000,))
    out = tail_effect_error(x, x, alpha=0.05)
    for k in ("q_alpha_error", "q_median_error", "q_1malpha_error", "es_error"):
        assert abs(out[k]) < 1e-10


def test_interventional_report_identity(rng):
    x = rng.standard_t(df=5, size=(500, 3))
    sig0 = np.abs(rng.standard_normal(size=(50, 3))) + 0.01
    sig1 = sig0 * 1.3
    sig = {"m0": sig0, "m1": sig1}
    report = interventional_report(x, x, sig, sig)
    assert np.allclose(report["wasserstein1_per_asset"], 0.0, atol=1e-10)
    assert np.allclose(report["energy_per_asset"], 0.0, atol=1e-10)
    assert abs(report["mmd_overall"]) < 1e-2
    assert report["vol_error"]["dsigma_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert abs(report["tail_effect"]["es_error"][0]) < 1e-10
