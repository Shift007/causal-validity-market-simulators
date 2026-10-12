"""
Standard distributional / interventional metrics for causal counterfactual eval.

Complements ``src/metrics/calibration.py::compute_causal_metrics`` (which scores
the MEAN effect: PEHE, MAE, Direction%, coverage) with metrics that score the
full *distribution* p(r | do(m')), not just its mean. These require S > 1
un-averaged generative samples (or, for the ground truth, S_true Monte-Carlo
draws from the DGP).

No dependency beyond numpy / scipy / sklearn (POT is intentionally NOT used --
not installed in this environment; the unbiased-MMD estimator below plays the
same role as an exact OT solver at a fraction of the complexity for this use
case: a rough, cheap divergence between two 1D/low-D empirical distributions).

Metric roster (spec section 4.1 / section 3.2 of the review-response plan):
    wasserstein1_per_asset   -- W1, per asset, scipy.stats.wasserstein_distance
    energy_per_asset         -- energy distance, per asset, scipy.stats.energy_distance
    mmd_rbf                  -- unbiased MMD^2, RBF kernel, median-heuristic bandwidth
    interventional_vol_error -- RMSE/bias/corr of Delta-sigma (gen vs true)
    tail_effect_error        -- Expected Shortfall + quantile errors (gen vs true)
    interventional_report    -- aggregates all of the above into one dict

Conventions
-----------
* "gen" / "true" sample arrays are POOLED samples, shape ``(S, C)`` (S samples,
  C channels = assets). A 1D ``(S,)`` array is treated as a single asset and
  internally reshaped to ``(S, 1)``.
* S_gen and S_true need NOT be equal (e.g. 100 generative samples vs 1000
  Monte-Carlo DGP samples) -- every function below handles that.
* "sigma" arguments to ``interventional_vol_error`` are already-reduced
  per-cell dispersion statistics (e.g. std across S samples for each
  window/day/asset cell), NOT raw samples -- shape is arbitrary as long as
  gen/true arrays broadcast against each other elementwise.

Known statistical subtlety (documented so nobody "fixes" it into a bug): the
unbiased MMD^2 estimator is NOT bounded at exactly 0 when gen and true are the
literal same finite sample (it is unbiased *in expectation*, not per draw); for
n pooled points it deviates from 0 by O(1/n). The identity test in
``tests/test_interventional.py`` uses n large enough (>= 300) that this is
negligible (kept well under 1e-2).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce to shape (S, C); a 1D (S,) input becomes (S, 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _median_heuristic_bandwidth(x: np.ndarray) -> float:
    """Median pairwise Euclidean distance (excluding zero self-distances)."""
    n = x.shape[0]
    if n < 2:
        return 1.0
    # Pairwise distances via broadcasting (avoids scipy.spatial dependency).
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n, k=1)
    dvec = d[iu]
    dvec = dvec[dvec > 0]
    if dvec.size == 0:
        return 1.0
    return float(np.median(dvec))


def _expected_shortfall_and_quantiles(x: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    """Left-tail Expected Shortfall + (alpha, 0.5, 1-alpha) quantiles for a 1D sample.

    ES convention: mean of the observations at or below the alpha-quantile
    (the standard "average of the worst alpha-fraction" definition). Falls
    back to the quantile itself if (due to ties) no point satisfies `<=`.
    """
    x = np.asarray(x, dtype=np.float64)
    q_lo = float(np.quantile(x, alpha))
    q_mid = float(np.quantile(x, 0.5))
    q_hi = float(np.quantile(x, 1.0 - alpha))
    tail = x[x <= q_lo]
    es = float(tail.mean()) if tail.size > 0 else q_lo
    return q_lo, q_mid, q_hi, es


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def wasserstein1_per_asset(gen: np.ndarray, true: np.ndarray) -> np.ndarray:
    """1D Wasserstein-1 distance per asset.

    Parameters
    ----------
    gen, true : (S_gen|S_true, C) pooled samples per asset (S may differ).

    Returns
    -------
    (C,) array of W1 distances, one per asset/channel.
    """
    g = _as_2d(gen)
    t = _as_2d(true)
    assert g.shape[1] == t.shape[1], f"channel mismatch: gen C={g.shape[1]} true C={t.shape[1]}"
    C = g.shape[1]
    out = np.empty(C, dtype=np.float64)
    for c in range(C):
        out[c] = wasserstein_distance(g[:, c], t[:, c])
    return out


def energy_per_asset(gen: np.ndarray, true: np.ndarray) -> np.ndarray:
    """1D (statistical) energy distance per asset -- scipy.stats.energy_distance.

    Parameters
    ----------
    gen, true : (S_gen|S_true, C) pooled samples per asset.

    Returns
    -------
    (C,) array of energy distances, one per asset/channel.
    """
    g = _as_2d(gen)
    t = _as_2d(true)
    assert g.shape[1] == t.shape[1], f"channel mismatch: gen C={g.shape[1]} true C={t.shape[1]}"
    C = g.shape[1]
    out = np.empty(C, dtype=np.float64)
    for c in range(C):
        out[c] = energy_distance(g[:, c], t[:, c])
    return out


def mmd_rbf(gen: np.ndarray, true: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased MMD^2 with an RBF kernel.

    Parameters
    ----------
    gen, true : (S, ·) -- pooled samples; a 1D (S,) input is treated as a
        single feature/channel. Can be multivariate (S, C) for a *joint*
        (all-assets-at-once) MMD rather than per-asset.
    bandwidth : RBF sigma. If None, uses the median-heuristic (median
        pairwise Euclidean distance over the POOLED gen+true points).

    Returns
    -------
    float -- unbiased MMD^2 estimate (Gretton et al. 2012, eq. 3). Can be
    (slightly) negative due to estimator variance; this is expected and is
    NOT clipped to zero so downstream aggregation stays unbiased.
    """
    g = _as_2d(gen)
    t = _as_2d(true)
    assert g.shape[1] == t.shape[1], f"channel mismatch: gen C={g.shape[1]} true C={t.shape[1]}"
    m, n = g.shape[0], t.shape[0]
    if bandwidth is None:
        pooled = np.vstack([g, t])
        bandwidth = _median_heuristic_bandwidth(pooled)
    bandwidth = float(bandwidth) if bandwidth and bandwidth > 1e-12 else 1.0
    gamma = 1.0 / (2.0 * bandwidth ** 2)

    Kxx = rbf_kernel(g, g, gamma=gamma)
    Kyy = rbf_kernel(t, t, gamma=gamma)
    Kxy = rbf_kernel(g, t, gamma=gamma)

    term_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) if m > 1 else 0.0
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.mean()

    return float(term_xx + term_yy - 2.0 * term_xy)


def interventional_vol_error(sig_gen_m1: np.ndarray, sig_gen_m0: np.ndarray,
                              sig_true_m1: np.ndarray, sig_true_m0: np.ndarray) -> dict:
    """Error of the shock-induced volatility change (Delta-sigma), gen vs true.

    Delta-sigma = sigma(do(m1)) - sigma(do(m0)) -- the whole point of the
    ScaleHead / GARCH-X: does the model's OWN dispersion respond to the shock
    the way the true DGP's conditional volatility does? A model with a
    constant/homoskedastic residual (e.g. plain OLS) has
    ``sig_gen_m1 == sig_gen_m0`` identically, so Delta-sigma_gen == 0 by
    construction -- this function will then simply report the magnitude of
    the true effect as the (unavoidable) error, which is exactly the point
    the review wants illustrated for baselines.

    Parameters
    ----------
    sig_gen_m1, sig_gen_m0, sig_true_m1, sig_true_m0 : arrays of PER-CELL
        sigma (already reduced across samples, e.g. std over S draws),
        any shape, as long as all four broadcast against one another.

    Returns
    -------
    dict with keys ``dsigma_rmse``, ``dsigma_bias``, ``dsigma_corr``.
    """
    g1 = np.asarray(sig_gen_m1, dtype=np.float64)
    g0 = np.asarray(sig_gen_m0, dtype=np.float64)
    t1 = np.asarray(sig_true_m1, dtype=np.float64)
    t0 = np.asarray(sig_true_m0, dtype=np.float64)

    dsig_gen = (g1 - g0).ravel()
    dsig_true = (t1 - t0).ravel()
    diff = dsig_gen - dsig_true

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    if dsig_gen.std() < 1e-15 or dsig_true.std() < 1e-15:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(dsig_gen, dsig_true)[0, 1])

    return {"dsigma_rmse": rmse, "dsigma_bias": bias, "dsigma_corr": corr}


def tail_effect_error(samples_gen_m1: np.ndarray, samples_true_m1: np.ndarray,
                       alpha: float = 0.05) -> dict:
    """Expected Shortfall + quantile (alpha, 0.5, 1-alpha) errors, gen vs true.

    Parameters
    ----------
    samples_gen_m1, samples_true_m1 : (S, C) pooled samples of the SHOCKED
        (do(m1)) distribution; a 1D (S,) input is a single asset.
    alpha : tail probability (default 0.05 -> 5% ES / quantiles).

    Returns
    -------
    dict of scalars if C == 1, else dict of (C,) arrays, with keys
    ``q_alpha_{gen,true,error}``, ``q_median_{gen,true,error}``,
    ``q_1malpha_{gen,true,error}``, ``es_{gen,true,error}``
    (``*_error`` = gen - true, signed).
    """
    g = _as_2d(samples_gen_m1)
    t = _as_2d(samples_true_m1)
    assert g.shape[1] == t.shape[1], f"channel mismatch: gen C={g.shape[1]} true C={t.shape[1]}"
    C = g.shape[1]

    keys = ["q_alpha_gen", "q_median_gen", "q_1malpha_gen", "es_gen",
            "q_alpha_true", "q_median_true", "q_1malpha_true", "es_true",
            "q_alpha_error", "q_median_error", "q_1malpha_error", "es_error"]
    out: dict[str, list] = {k: [] for k in keys}

    for c in range(C):
        qlo_g, qmid_g, qhi_g, es_g = _expected_shortfall_and_quantiles(g[:, c], alpha)
        qlo_t, qmid_t, qhi_t, es_t = _expected_shortfall_and_quantiles(t[:, c], alpha)
        out["q_alpha_gen"].append(qlo_g)
        out["q_median_gen"].append(qmid_g)
        out["q_1malpha_gen"].append(qhi_g)
        out["es_gen"].append(es_g)
        out["q_alpha_true"].append(qlo_t)
        out["q_median_true"].append(qmid_t)
        out["q_1malpha_true"].append(qhi_t)
        out["es_true"].append(es_t)
        out["q_alpha_error"].append(qlo_g - qlo_t)
        out["q_median_error"].append(qmid_g - qmid_t)
        out["q_1malpha_error"].append(qhi_g - qhi_t)
        out["es_error"].append(es_g - es_t)

    if C == 1:
        return {k: v[0] for k, v in out.items()}
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def interventional_report(gen_samples: np.ndarray, true_samples: np.ndarray,
                           sig_gen: dict, sig_true: dict, alpha: float = 0.05) -> dict:
    """Aggregate all distributional metrics into one JSON-serializable dict.

    Parameters
    ----------
    gen_samples, true_samples : (S, C) pooled samples of the SHOCKED
        (do(m')) distribution -- gen from the generative model, true from
        a Monte-Carlo simulation of the DGP.
    sig_gen, sig_true : dict with keys ``"m0"`` (baseline sigma) and
        ``"m1"`` (shocked sigma), per-cell arrays (see
        ``interventional_vol_error``).
    alpha : tail probability for ``tail_effect_error`` (default 0.05).

    Returns
    -------
    dict with keys ``wasserstein1_per_asset``, ``energy_per_asset``,
    ``mmd_per_asset``, ``mmd_overall``, ``vol_error``, ``tail_effect``
    -- all numpy arrays converted to lists so ``json.dump`` works directly.
    """
    gen = _as_2d(gen_samples)
    true = _as_2d(true_samples)
    C = gen.shape[1]

    w1 = wasserstein1_per_asset(gen, true)
    en = energy_per_asset(gen, true)
    mmd_per_asset = np.array([mmd_rbf(gen[:, [c]], true[:, [c]]) for c in range(C)])
    mmd_overall = mmd_rbf(gen, true)
    vol_err = interventional_vol_error(sig_gen["m1"], sig_gen["m0"], sig_true["m1"], sig_true["m0"])
    tail = tail_effect_error(gen, true, alpha=alpha)

    return {
        "wasserstein1_per_asset": w1.tolist(),
        "energy_per_asset": en.tolist(),
        "mmd_per_asset": mmd_per_asset.tolist(),
        "mmd_overall": float(mmd_overall),
        "vol_error": vol_err,
        "tail_effect": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tail.items()},
    }
