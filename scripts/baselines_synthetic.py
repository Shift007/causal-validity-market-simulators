"""Linear reference baselines (OLS, ridge, VAR) on the synthetic environment. CPU only.

Fits, on the TRAIN period of a fresh (read-only-imported) DGP simulation:
  * OLS       -- regress contemporaneous returns r_t on contemporaneous macro
                 m_t (sklearn LinearRegression). Effect = B_hat_OLS @ shock_vec,
                 the SAME (T,C)-broadcast constant for every day/window (a
                 static model has no notion of horizon decay).
  * Ridge     -- same design matrix, RidgeCV over a small alpha grid.
  * VAR(p)    -- joint VAR on [macro, returns] (statsmodels), lag order p
                 chosen by AIC; the effect at horizon day k is
                 Psi_{k+1} @ shock_vec_full (the reduced-form MA/IRF
                 coefficient, k+1 because a LEVEL shock at the boundary T
                 first shows up in the horizon's day-0 = T+1 response --
                 exactly the same one-step convention
                 TheoreticalCounterfactualComplex.compute() uses via
                 `E_m_sh[0] = c + Phi @ m_T_sh`). shock_vec_full has the
                 macro shock in the macro slots and 0 in the return slots
                 (we are not shocking returns directly).

All three baselines are, by construction, STATE-INDEPENDENT (their
predicted effect does not vary across windows -- no h_T/eps_T/m_T
conditioning) and HOMOSKEDASTIC (constant residual variance -> Delta-sigma
predicted == 0 identically). Both properties are the point of this baseline:
it isolates what a "plain" linear/no-heterogeneity/no-GARCH model gets right
(the contemporaneous mean loading B, which the DGP docstring calls
"observational = causal" here, so OLS should recover the RIGHT SIGN/rough
MAGNITUDE) and gets structurally wrong (no window-level heterogeneity, no
volatility response).

Evaluation: 1000 boundary days sampled directly in DGP day-space (seed=42,
warmup=300 days excluded), independent of the processed_data window index
(this script only needs the DGP simulation + config, not the preprocessing
artifacts) -- PEHE / eps_ATE / Dir% (via compute_causal_metrics) and
the sigma-interventional error (via src.metrics.interventional).

Output: results/synthetic_complex_baselines/baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, RidgeCV
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.calibration import compute_causal_metrics  # noqa: E402
from scripts.synthetic_dgp_complex import (  # noqa: E402  (read-only import, not modified)
    ComplexSyntheticDGP,
    TheoreticalCounterfactualComplex,
    _sigma_from,
)
from src.metrics.interventional import interventional_vol_error  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "synthetic_complex" / "baseline.yaml"
OUT_DIR = ROOT / "results" / "synthetic_complex_baselines"
OUT_PATH = OUT_DIR / "baselines.json"

N_EVAL_WINDOWS = 1000
EVAL_SEED = 42
WARMUP_DAYS = 300
VAR_MAXLAGS = 15
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def score_effect_vs_theory(effect: np.ndarray, theory: np.ndarray, shock_labels: list[str]) -> dict:
    """effect, theory: (N,K,T,C). Per-shock + aggregate PEHE / eps_ATE / Dir% / MAE / Amp.

    The deterministic effect array is scored as a single sample, as in
    src/generation/counterfactual_synthetic.py.
    """
    K = theory.shape[1]
    per_shock = []
    for k in range(K):
        true_k = theory[:, k]                  # (N,T,C)
        pred_k = effect[:, k][np.newaxis]       # (1,N,T,C) -- S=1 deterministic sample
        cm = compute_causal_metrics(true_k, pred_k)
        eps = abs(float(pred_k.mean()) - float(true_k.mean()))
        per_shock.append({
            "shock_label": shock_labels[k] if k < len(shock_labels) else str(k),
            "pehe": cm["pehe"],
            "eps_ate": eps,
            "dir_pct": cm["directional_accuracy"] * 100.0,
            "mae": cm["mae"],
            "amp": cm["amplification_ratio"],
        })
    agg = {}
    for key in ("pehe", "eps_ate", "dir_pct", "mae", "amp"):
        vals = [p[key] for p in per_shock if not (isinstance(p[key], float) and np.isnan(p[key]))]
        agg[key] = float(np.mean(vals)) if vals else float("nan")
    agg["per_shock"] = per_shock
    return agg


def main() -> None:
    print("=" * 88)
    print("[baselines] provenance")
    print(f"  config       : {CONFIG_PATH}")
    print(f"  n_eval_windows: {N_EVAL_WINDOWS}  (eval_seed={EVAL_SEED}, warmup={WARMUP_DAYS}d)")
    print(f"  VAR maxlags  : {VAR_MAXLAGS} (AIC-selected)")
    print("=" * 88)

    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    syn_cfg = cfg["synthetic_dgp"]
    cf_cfg = cfg["counterfactual_synthetic"]
    n_obs = int(syn_cfg.get("n_obs", 7000))
    dgp_seed = int(syn_cfg.get("seed", 42))
    start_date = str(syn_cfg.get("start_date", "1995-01-02"))
    dgp_params = syn_cfg.get("dgp_params", None)
    train_split_date = str(cfg["data_ingestion"]["train_split_date"])
    shock_deltas = np.asarray(cf_cfg["shock_deltas"], dtype=np.float64)
    shock_labels = list(cf_cfg.get("shock_labels", [f"{d:+.4f}" for d in shock_deltas]))
    shk_idx = int(cf_cfg.get("shock_macro_index", 0))
    cov_adjust = bool(cf_cfg.get("covariance_adjust", True))
    target_len = int(cfg["preprocessing"]["target_length"])
    K = len(shock_deltas)

    print(f"[baselines] DGP: n_obs={n_obs} seed={dgp_seed} start={start_date} variant=complex")

    dgp = ComplexSyntheticDGP(params=dgp_params, seed=dgp_seed)
    returns, macro = dgp.generate_series(N=n_obs)
    date_idx = pd.bdate_range(start=start_date, periods=n_obs, freq="B")
    n_macro, n_assets = macro.shape[1], returns.shape[1]

    tc = TheoreticalCounterfactualComplex(params=dgp_params)

    train_mask = np.asarray(date_idx <= pd.Timestamp(train_split_date))
    print(f"[baselines] train days: {train_mask.sum()} / {n_obs}  (cutoff {train_split_date})")

    # ---- covariance-adjusted shock weight vector (identical recipe used project-wide) ----
    w_shock = np.zeros(n_macro, dtype=np.float64)
    if cov_adjust:
        raw_cov = np.cov(macro[train_mask], rowvar=False)
        w_shock = raw_cov[:, shk_idx] / raw_cov[shk_idx, shk_idx]
    else:
        w_shock[shk_idx] = 1.0
    print(f"[baselines] w_shock = {np.round(w_shock, 4)}")
    shock_vec_macro = shock_deltas[:, None] * w_shock[None, :]           # (K, n_macro)

    # =====================================================================
    # 1. OLS / Ridge -- contemporaneous regression on the TRAIN period
    # =====================================================================
    X_train = macro[train_mask]        # (T_train, n_macro)
    Y_train = returns[train_mask]      # (T_train, n_assets)
    print(f"[baselines] OLS/Ridge design: X={X_train.shape} Y={Y_train.shape}")

    ols = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
    B_ols = ols.coef_                  # (n_assets, n_macro)
    resid_ols = Y_train - ols.predict(X_train)
    sigma_resid_ols = resid_ols.std(axis=0)

    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True).fit(X_train, Y_train)
    B_ridge = ridge.coef_
    resid_ridge = Y_train - ridge.predict(X_train)
    sigma_resid_ridge = resid_ridge.std(axis=0)
    print(f"[baselines] OLS  B_hat shape={B_ols.shape}  Ridge alpha_={ridge.alpha_}")
    print(f"[baselines] residual sigma (OLS)  per asset (bps): {np.round(sigma_resid_ols * 1e4, 2)}")
    print(f"[baselines] residual sigma (Ridge) per asset (bps): {np.round(sigma_resid_ridge * 1e4, 2)}")

    ols_effect_per_shock = shock_vec_macro @ B_ols.T      # (K, n_assets)
    ridge_effect_per_shock = shock_vec_macro @ B_ridge.T  # (K, n_assets)

    # =====================================================================
    # 2. VAR(p) on [macro, returns] jointly -- AIC lag order, MA/IRF response
    # =====================================================================
    joint_train = np.hstack([X_train, Y_train])           # (T_train, n_macro+n_assets)
    var_model = VAR(joint_train)
    order_sel = var_model.select_order(maxlags=VAR_MAXLAGS)
    p_aic = max(int(order_sel.aic), 1)
    var_res = var_model.fit(p_aic)
    print(f"[baselines] VAR: AIC-selected lag order p={p_aic}  (search up to {VAR_MAXLAGS})")

    ma_coefs = var_res.ma_rep(maxn=target_len)            # (target_len+1, n_vars, n_vars); ma_coefs[0]=I
    shock_vec_full = np.zeros((K, n_macro + n_assets))
    shock_vec_full[:, :n_macro] = shock_vec_macro
    # day d (0-indexed target day) <-> Psi_{d+1} (one step past the shocked boundary)
    var_effect_path = np.stack([
        (ma_coefs[d + 1] @ shock_vec_full.T).T[:, n_macro:]     # (K, n_assets)
        for d in range(target_len)
    ], axis=1)                                             # (K, target_len, n_assets)

    # =====================================================================
    # 3. Evaluation windows: 1000 boundary days sampled directly in DGP day-space
    # =====================================================================
    valid_t = np.arange(WARMUP_DAYS, n_obs - target_len)
    rng_eval = np.random.default_rng(EVAL_SEED)
    n_eval = min(N_EVAL_WINDOWS, len(valid_t))
    t_bnd_eval = np.sort(rng_eval.choice(valid_t, size=n_eval, replace=False))
    print(f"[baselines] evaluation: {n_eval} boundary days sampled from "
          f"[{WARMUP_DAYS}, {n_obs - target_len}) (seed={EVAL_SEED})")

    theory_effects = np.zeros((n_eval, K, target_len, n_assets), np.float64)
    theory_dvol = np.zeros((n_eval, K, target_len, n_assets), np.float64)
    for wi, t_b in enumerate(t_bnd_eval):
        state = dgp.get_state_at(int(t_b))
        for si, sigma in enumerate(shock_deltas):
            tcres = tc.compute(state["m"], float(sigma), state["h"], state["eps_sq"],
                                horizon=target_len, shock_vec=float(sigma) * w_shock)
            theory_effects[wi, si] = tcres["delta_E_ret"]
            theory_dvol[wi, si] = tcres["delta_E_vol"]
    print(f"[baselines] theory_effects shape={theory_effects.shape}  "
          f"(N={n_eval}, K={K}, T={target_len}, C={n_assets})")
    assert theory_effects.shape == (n_eval, K, target_len, n_assets), "STOP -- unexpected theory shape"

    # Baseline "effects" broadcast identically across ALL windows (state-independent by construction).
    ols_effects = np.broadcast_to(ols_effect_per_shock[None, :, None, :],
                                   (n_eval, K, target_len, n_assets)).astype(np.float64)
    ridge_effects = np.broadcast_to(ridge_effect_per_shock[None, :, None, :],
                                     (n_eval, K, target_len, n_assets)).astype(np.float64)
    var_effects = np.broadcast_to(var_effect_path[None, :, :, :],
                                   (n_eval, K, target_len, n_assets)).astype(np.float64)

    # =====================================================================
    # 4. Score PEHE / eps_ATE / Dir% vs theory (same scorer as reeval_unitlevel.py)
    # =====================================================================
    results = {}
    for name, eff in (("ols", ols_effects), ("ridge", ridge_effects), ("var", var_effects)):
        score = score_effect_vs_theory(eff, theory_effects, shock_labels)
        results[name] = score
        print(f"[baselines] {name:<6} PEHE={score['pehe']:.5f}  eps_ATE={score['eps_ate']:.5f}  "
              f"Dir%={score['dir_pct']:.2f}%")

    # =====================================================================
    # 5. Sigma-interventional error: Delta-sigma_gen == 0 by construction for all
    #    three (homoskedastic linear models); Delta-sigma_true from the SAME
    #    analytical delta_E_vol already computed above (theory_dvol). Fed into
    #    interventional_vol_error as (m1=theory_dvol, m0=0) vs (m1=0, m0=0) so
    #    the function's OWN difference-of-levels contract is honoured exactly
    #    (see src/metrics/interventional.py) while only needing the already-
    #    available DIFFERENCE, not the individual sigma levels.
    # =====================================================================
    zeros_like_dvol = np.zeros_like(theory_dvol)
    vol_err = interventional_vol_error(
        sig_gen_m1=zeros_like_dvol, sig_gen_m0=zeros_like_dvol,
        sig_true_m1=theory_dvol, sig_true_m0=zeros_like_dvol,
    )
    print(f"[baselines] sigma-interventional error (identical for OLS/Ridge/VAR, all homoskedastic): "
          f"RMSE={vol_err['dsigma_rmse']:.6f}  bias={vol_err['dsigma_bias']:.6f}  "
          f"corr={vol_err['dsigma_corr']} (NaN expected: gen Delta-sigma is a constant 0 array)")
    for name in ("ols", "ridge", "var"):
        results[name]["sigma_interventional_error"] = vol_err

    output = {
        "baselines": results,
        "_provenance": {
            "config": str(CONFIG_PATH),
            "n_eval_windows": n_eval, "eval_seed": EVAL_SEED, "warmup_days": WARMUP_DAYS,
            "train_days": int(train_mask.sum()), "train_split_date": train_split_date,
            "shock_deltas": shock_deltas.tolist(), "shock_labels": shock_labels,
            "shock_macro_index": shk_idx, "covariance_adjust": cov_adjust,
            "w_shock": w_shock.tolist(),
            "ols_B_hat": B_ols.tolist(), "ridge_B_hat": B_ridge.tolist(), "ridge_alpha": float(ridge.alpha_),
            "ols_residual_sigma_per_asset": sigma_resid_ols.tolist(),
            "ridge_residual_sigma_per_asset": sigma_resid_ridge.tolist(),
            "var_lag_order_aic": p_aic, "var_maxlags_searched": VAR_MAXLAGS,
            "true_B_for_reference": np.asarray(dgp.p["assets"]["B"]).tolist(),
            "mlp_baseline": "SKIPPED (spec marks it optional; see module docstring)",
            "state_independence_note": (
                "OLS/Ridge/VAR effects are IDENTICAL across all n_eval_windows by "
                "construction (none of the three conditions on window-specific "
                "h_T/eps_T/m_T) -- only theory_effects varies by window. PEHE therefore "
                "measures 'how far is one global (or, for VAR, one global TIME-SHAPED) "
                "prediction from the window-varying truth', which is the intended sense "
                "of a naive/no-heterogeneity baseline."
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
    print(f"\n[baselines] wrote {OUT_PATH}")

    print("\n" + "=" * 88)
    print("TABLE (baselines): PEHE / eps_ATE / Dir% / sigma-RMSE  x  {OLS, Ridge, VAR}")
    print("=" * 88)
    print(f"{'model':<8} {'PEHE':>10} {'eps_ATE':>10} {'Dir%':>8} {'sigma_RMSE':>12}")
    for name in ("ols", "ridge", "var"):
        s = results[name]
        print(f"{name:<8} {s['pehe']:>10.5f} {s['eps_ate']:>10.5f} {s['dir_pct']:>7.2f}% "
              f"{s['sigma_interventional_error']['dsigma_rmse']:>12.6f}")


if __name__ == "__main__":
    main()
