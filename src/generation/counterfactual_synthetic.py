"""Synthetic counterfactual evaluation against analytical ground truth.

Sweeps N windows of the synthetic environment.  For each window and each
shock level it:
  1. Computes the exact analytical expected-return effect using
     ``TheoreticalCounterfactual`` (the DGP is known).
  2. Generates the diffusion response as a paired difference, shocked minus
     factual, under a shared latent: fresh x_T ~ N(0,I), M draws per window
     (``latent_mode="fresh"``, the prior-marginalised estimate), or the latent
     recovered by DDIM inversion of the observed window (``latent_mode="invert"``,
     the latent-preserving estimate).
  3. Generates the MacroVAE response the same way, decoding the posterior mean
     of the window (``mvae_latent="posterior"``) or draws z ~ N(0,I)
     (``mvae_latent="prior"``).
  4. Measures how closely each model reproduces the theoretical effect.

Shocks are absolute CPI level changes in decimal units (0.01 = 1 ppt),
bidirectional (negative/positive).

Output layout (<paths.results_dir>/counterfactual_synthetic/run_{ts}/):
    theory_effects.npy    (N, K, T, C)  analytical delta_E_ret per shock
    diff_effects.npy      (N, K, T, C)  diffusion ATE (M-sample mean)
    mvae_effects.npy      (N, K, T, C)  MacroVAE  (shocked − baseline)
    real_targets.npy      (N, T, C)     observed returns in the target window
    window_meta.npz       window indices, boundary t-indices, cpi_T per window
    metrics.json          aggregate MAE / Dir% / AmpRob per model
    manifest.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.diffusion.lab import (
    Diffusion,
    build_unet_model,
    inverse_full,
)
from src.generation.runner import (
    create_run_dir,
    get_device,
    load_config,
    load_mvae_model,
    load_preprocessing_artifacts,
    prune_old_runs,
    save_manifest,
    write_latest,
)
from src.metrics.calibration import compute_causal_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_diffusion(ckpt_path, n_assets, n_levels, macro_dim, macro_len,
                    context_len, device, macro_encoder_type="symmetric"):
    """Load a diffusion checkpoint with optional ablation encoder.

    If the checkpoint stores ``macro_encoder_type`` / ``macro_dim`` / ``macro_len``
    they OVERRIDE the function arguments (so callers can pass placeholders and
    rely on the saved metadata).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Saved metadata takes precedence — guarantees we build the right encoder.
    macro_encoder_type = ckpt.get("macro_encoder_type", macro_encoder_type)
    macro_dim = ckpt.get("macro_dim", macro_dim)
    macro_len = ckpt.get("macro_len", macro_len)
    base_ch = int(ckpt.get("base_ch", 32))
    model = build_unet_model(n_assets, n_levels, macro_dim, macro_len,
                             context_len, device,
                             macro_encoder_type=macro_encoder_type, base_ch=base_ch)
    model.load_state_dict(ckpt["ema_model"], strict=False)
    model.eval()
    return (Diffusion(model, timesteps=500, device=device),
            ckpt.get("norm_params", {}),
            macro_encoder_type)


def _wav_to_raw(wav_t, qt_artifact, wavelet_type, level_scales,
                inst_med, inst_mad, inst_c):
    med_t = torch.as_tensor(inst_med, dtype=torch.float32)
    mad_t = torch.as_tensor(inst_mad, dtype=torch.float32)
    return inverse_full(
        wav_t, level_scales, med_t, mad_t, qt_artifact,
        wavelet_type=wavelet_type, c=inst_c,
    )


def _pick_ckpt(ckpt_dir: Path, glob_annual: str, initial_name: str) -> Path | None:
    """Return latest annual checkpoint, falling back to initial."""
    annual = sorted(ckpt_dir.glob(glob_annual)) if ckpt_dir.exists() else []
    if annual:
        return annual[-1]
    init = ckpt_dir / initial_name
    return init if init.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(project_root: Path, config: dict | None = None,
        model_filter: str = "both",
        sgd_gamma: float | None = None,
        shock_macro_index: int | None = None,
        covariance_adjust: bool | None = None,
        latent_mode: str = "fresh",
        mvae_latent: str = "posterior",
        mvae_prior_samples: int = 16) -> Path:
    """Execute synthetic counterfactual evaluation.  Returns the run directory.

    Parameters
    ----------
    latent_mode : {"fresh", "invert"}
        Diffusion latent used for the paired baseline/shocked decode.
          "fresh"  (default, ATE) — x_T ~ N(0,I), averaged over M draws; the
                   macro effect is marginalized over the generative prior.
          "invert" (ITE, Pearl abduction) — x_T = ddim_invert(real_window,
                   factual_macro); baseline and shocked share the inverted
                   latent so the effect is anchored to the realized path u.
                   Forces M=1 (the inverted latent is deterministic). Everything
                   else (covariance-adjusted boundary shock, window pool) is
                   identical to the ATE path, giving an apples-to-apples ITE↔ATE
                   comparison. The MacroVAE is always paired encode/decode
                   (ITE-native), so its numbers are unaffected by this flag.
    mvae_latent : {"posterior", "prior"}
        MacroVAE latent used for the paired baseline/shocked decode.
          "posterior" (default): z = posterior mean of the observed window, the
                   latent-preserving resimulation of Kubiak et al. (Sec. 6.1).
          "prior": z ~ N(0, I), shared between baseline and shocked decode and
                   averaged over ``mvae_prior_samples`` draws per window. Same
                   estimand as the diffusion "fresh" mode.
    sgd_gamma : float or None
        Score Guidance Differential amplification factor for the ATE sampler:
            ε_final = ε_base + γ · (ε_shocked − ε_base)
        When None (default), the value is read from
        ``config["counterfactual_synthetic"]["sgd_gamma"]`` (current operating
        point: γ = 15). Pass explicitly to override the config for ablations.
        γ = 1 disables SGD (raw shocked-only prediction).
    """
    config = config or load_config(project_root)
    device = get_device()

    run_diffusion = model_filter in ("both", "diffusion")
    run_mvae      = model_filter in ("both", "macrovae")

    # ── Paths ──────────────────────────────────────────────────────────────
    from src.paths import resolve_paths
    _paths     = resolve_paths(project_root, config)
    data_dir   = _paths.processed_dir
    gen_cfg    = config.get("generation", {})
    synth_base = _paths.results_dir
    max_runs   = int(gen_cfg.get("max_runs_per_task", 3))

    # ── Per-task config ────────────────────────────────────────────────────
    cf_syn = config.get("counterfactual_synthetic", {})
    N_WINDOWS     = int(cf_syn.get("n_windows", 200))
    N_SAMPLES     = int(cf_syn.get("n_samples", 1))
    SEED          = int(cf_syn.get("seed", 0))
    SHOCK_SIGMAS  = list(cf_syn.get("shock_deltas",
                         cf_syn.get("shock_sigmas", [-0.010, -0.020, -0.030, -0.040])))
    SHOCK_LABELS  = list(cf_syn.get("shock_labels",
                                    [f"{s:+.3f}ppt" for s in SHOCK_SIGMAS]))
    DDIM_STEPS    = int(cf_syn.get("ddim_steps", 50))
    DDIM_ETA      = float(cf_syn.get("ddim_eta", 0.0))
    GUIDANCE      = float(cf_syn.get("guidance_scale", 1.0))
    N_ATE_SAMPLES = int(cf_syn.get("ate_samples", 16))    # M samples averaged per window
    if N_ATE_SAMPLES < 1:
        raise ValueError("counterfactual_synthetic.ate_samples must be >= 1")
    N_SHOCKS      = len(SHOCK_SIGMAS)
    SAVE_DIST     = bool(cf_syn.get("save_dist", False))   # §4.5: save per-cell sigma (dispersion over M) for interventional-distribution scoring

    # ── Latent mode: fresh-noise ATE (default) vs DDIM-inversion ITE ───────────
    mvae_latent = str(mvae_latent).lower()
    if mvae_latent not in ("posterior", "prior"):
        raise ValueError(f"mvae_latent must be 'posterior' or 'prior', got {mvae_latent!r}")
    MVAE_PRIOR_SAMPLES = int(mvae_prior_samples)
    mvae_post_var_mean = None
    latent_mode = str(latent_mode).lower()
    if latent_mode not in ("fresh", "invert"):
        raise ValueError(f"latent_mode must be 'fresh' or 'invert', got {latent_mode!r}")
    if latent_mode == "invert":
        N_ATE_SAMPLES = 1   # inverted latent is deterministic — no noise averaging
    print(f"[cf_synthetic] latent_mode = {latent_mode} (M={N_ATE_SAMPLES})", flush=True)

    # SGD gamma: explicit call arg wins, otherwise read from config (default 15.0)
    if sgd_gamma is None:
        sgd_gamma = float(cf_syn.get("sgd_gamma", 15.0))
    print(f"[cf_synthetic] sgd_gamma = {sgd_gamma}", flush=True)

    # Which macro variable is intervened on (0=CPI default, backward-compatible).
    # Selects the channel shocked in BOTH the analytical theory and the model
    # conditioning, so the whole pipeline shocks one consistent macro. Complex
    # testbed order: CPI=0, GROWTH=1, MONEY=2, RATES=3. Explicit arg wins.
    if shock_macro_index is None:
        shock_macro_index = int(cf_syn.get("shock_macro_index", 0))
    SHK = int(shock_macro_index)
    print(f"[cf_synthetic] shock_macro_index = {SHK}", flush=True)

    # Shocks are expressed as absolute CPI level deltas (decimal; 0.01 = 1 ppt).
    # No σ_u scaling needed — shock_deltas are passed directly to TheoreticalCounterfactual.

    # ── Load preprocessing artifacts ───────────────────────────────────────
    art = load_preprocessing_artifacts(data_dir)
    raw_windows       = art["raw_windows"]       # (N_total, ctx+tgt, C)
    macro_windows     = art["macro_windows"]     # (N_total, macro_len, D)
    mvae_macro_win    = art["mvae_macro_windows"]  # (N_total, macro_len, D_ext)
    mvae_exp_std      = art["mvae_exp_std"]      # (N_total, D_ext) or None
    train_mask        = art["train_mask"]
    window_dates      = art["window_dates"]
    n_assets          = art["n_assets"]
    context_len       = art["context_len"]
    target_len        = art["target_len"]
    wavelet_type      = art["wavelet_type"]
    wavelet_level     = art["wavelet_level"]
    n_levels          = art["n_levels"]
    tickers           = art["tickers"]
    D_mvae            = art["D_mvae"]
    d_base            = art["D_base"]   # number of base macro vars (excl. momentum)
    level_scales      = art["level_scales"]

    _raw_qt    = art["target_qt"]
    if _raw_qt is None:
        qt_artifact = None
    elif isinstance(_raw_qt, dict):
        qt_artifact = _raw_qt
    else:
        qt_artifact = {"quantile_transformer": _raw_qt, "gauss_limits": None}

    inst_cfg  = config["preprocessing"].get("instance_scaling", {})
    inst_c    = float(inst_cfg.get("c", 3.0))
    mad_floor = float(inst_cfg.get("mad_floor", 1e-4))
    z_clip    = float(inst_cfg.get("z_clip", 10.0))

    # Macro tanh scaling constant: macro_cond was bounded via
    # expanding_robust_tanh(x / (c × std_raw)).  To convert a raw shock to the
    # tanh-bounded space the encoder operates in:
    #   delta_tanh = shock_raw / (MACRO_TANH_C × exp_std_cpi)
    #
    # Resolution order:
    #   1. norm_params["macro_tanh_c"], written by preprocess.py
    #   2. preprocessing.macro_scaling.tanh_c in the config
    #   3. inst_c, the instance-scaling constant: last resort, a different
    #      quantity, used only when neither of the above is present
    _norm_params = art.get("norm_params", {})
    _macro_scaling_cfg = config.get("preprocessing", {}).get("macro_scaling", {})
    if "macro_tanh_c" in _norm_params:
        MACRO_TANH_C = float(_norm_params["macro_tanh_c"])
        _macro_tanh_c_src = "norm_params"
    elif "tanh_c" in _macro_scaling_cfg:
        MACRO_TANH_C = float(_macro_scaling_cfg["tanh_c"])
        _macro_tanh_c_src = "config.yaml"
    else:
        MACRO_TANH_C = float(inst_c)
        _macro_tanh_c_src = "fallback(inst_c)"
    print(f"[cf_synthetic] MACRO_TANH_C = {MACRO_TANH_C}  (source: {_macro_tanh_c_src})",
          flush=True)

    # ── Re-run DGP (same seed) to obtain GARCH states at window boundaries ─
    syn_cfg    = config.get("synthetic_dgp", {})
    n_obs      = int(syn_cfg.get("n_obs", 10000))
    dgp_seed   = int(syn_cfg.get("seed", 42))
    start_date = str(syn_cfg.get("start_date", "1980-01-03"))
    dgp_params = syn_cfg.get("dgp_params", None)   # incl. optional regime-switching (NL DGP)

    print(f"[cf_synthetic] Re-simulating DGP (seed={dgp_seed}, N={n_obs}) …", flush=True)
    _root = project_root
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    _variant = str(syn_cfg.get("variant", "")).lower()
    if _variant == "complex":
        from scripts.synthetic_dgp_complex import (
            ComplexSyntheticDGP as SyntheticDGP,
            TheoreticalCounterfactualComplex as TheoreticalCounterfactual,
        )
        _tc_base_key = "m"        # complex compute() takes the full macro VECTOR
    else:
        from scripts.synthetic_dgp import SyntheticDGP, TheoreticalCounterfactual
        _tc_base_key = "cpi"      # single-macro compute() takes the CPI scalar
    print(f"[cf_synthetic] DGP variant = {_variant or 'single'}", flush=True)

    dgp = SyntheticDGP(params=dgp_params, seed=dgp_seed)
    dgp.generate_series(N=n_obs)   # stores per-step macro/h/eps state
    tc  = TheoreticalCounterfactual(params=dgp_params)

    date_idx = pd.bdate_range(start=start_date, periods=n_obs, freq="B")

    # ── Sample windows ─────────────────────────────────────────────────────
    # window_pool selects the index universe the CF windows are drawn from:
    #   "test" (default) — out-of-sample windows only (~447 on the synthetic
    #           testbeds); preserves the historical behaviour.
    #   "all"           — every valid window (train+test, ~6812). The causal
    #           ground truth is analytical per-window and independent of the
    #           train/test split, and counterfactual inference is framed
    #           as IN-SAMPLE, so pooling over all windows is consistent and the
    #           only way to reach a 2000-window average on this testbed.
    WINDOW_POOL = str(cf_syn.get("window_pool", "test")).lower()
    if WINDOW_POOL == "all":
        pool_idx = np.arange(len(train_mask))
    else:
        pool_idx = np.where(~train_mask)[0]
    rng      = np.random.default_rng(SEED)
    n_sel    = min(N_WINDOWS, len(pool_idx))
    selected = np.sort(rng.choice(pool_idx, size=n_sel, replace=False))
    print(f"[cf_synthetic] Selected {n_sel}/{len(pool_idx)} windows "
          f"(pool={WINDOW_POOL})", flush=True)

    # Instance scaling stats / macro tensors are now computed vectorized in
    # PHASE A below (no per-window dataset.__getitem__ calls).
    wavelet_shape = (n_assets, n_levels, target_len)

    # ── Load Diffusion model (latest checkpoint) ───────────────────────────
    diffusion = None
    diff_ckpt_path = None

    diff_encoder_type = "symmetric"  # default; overridden by ckpt metadata if present
    if run_diffusion:
        diff_dir = _paths.diffusion_ckpt_dir
        diff_ckpt_path = _pick_ckpt(diff_dir, "ckpt_wf_20??.pt", "ckpt_wf_initial.pt")
        assert diff_ckpt_path is not None and diff_ckpt_path.exists(), \
            f"Diffusion checkpoint not found in {diff_dir}"
        # Placeholder args — _load_diffusion overrides from ckpt metadata when present.
        macro_dim     = macro_windows.shape[2]
        macro_seq_len = macro_windows.shape[1]
        diffusion, _, diff_encoder_type = _load_diffusion(
            diff_ckpt_path, n_assets, n_levels, macro_dim, macro_seq_len,
            context_len, device,
        )
        print(f"[cf_synthetic] Diffusion: {diff_ckpt_path.name}  "
              f"(encoder={diff_encoder_type})", flush=True)

    # ── Load MacroVAE model (latest checkpoint) ────────────────────────────
    mvae_model     = None
    mvae_ckpt_path = None

    if run_mvae:
        macrovae_root = project_root / "MacroVAE"
        if str(macrovae_root) not in sys.path:
            sys.path.insert(0, str(macrovae_root))
        mvae_dir = _paths.macrovae_ckpt_dir
        mvae_ckpt_path = _pick_ckpt(mvae_dir, "ckpt_mvae_20*.pt", "ckpt_mvae_initial.pt")
        assert mvae_ckpt_path is not None and mvae_ckpt_path.exists(), \
            f"MacroVAE checkpoint not found in {mvae_dir}"
        mvae_model, _, hp = load_mvae_model(
            mvae_ckpt_path, n_assets, target_len, D_mvae, device)
        print(f"[cf_synthetic] MacroVAE: {mvae_ckpt_path.name}", flush=True)

    # ── Output arrays ──────────────────────────────────────────────────────
    theory_effects = np.zeros((n_sel, N_SHOCKS, target_len, n_assets), np.float32)
    diff_effects   = np.zeros_like(theory_effects) if run_diffusion else None
    mvae_effects   = np.zeros_like(theory_effects) if run_mvae      else None
    real_targets   = np.zeros((n_sel, target_len, n_assets), np.float32)
    sig_base       = np.zeros((n_sel, N_SHOCKS, target_len, n_assets), np.float32) if (SAVE_DIST and run_diffusion) else None
    sig_shock      = np.zeros_like(sig_base) if (SAVE_DIST and run_diffusion) else None

    meta_widx  = selected.copy()
    meta_t_bnd = np.zeros(n_sel, dtype=np.int64)
    meta_cpi_T = np.zeros(n_sel, dtype=np.float64)

    # ── Across-windows chunking knob ──────────────────────────────────────
    # Each DDIM call processes (W_chunk · 2 · K · M) wavelet images.
    # Default 4 fits comfortably on 8 GB GPUs (e.g. K=8, M=16 → 1024/call).
    # Increase via config `counterfactual_synthetic.window_chunk` if VRAM allows.
    W_CHUNK = int(cf_syn.get("window_chunk", 4))
    W_CHUNK = max(1, min(W_CHUNK, n_sel))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE A — vectorized pre-loop setup (no GPU, no per-window Python loop)
    # ═══════════════════════════════════════════════════════════════════════
    t0 = time.perf_counter()
    _t_phase = {"setup": 0.0, "theory": 0.0, "ate_sample": 0.0,
                "ate_wav2raw": 0.0, "mvae": 0.0}
    _cuda = (device.type == "cuda") if hasattr(device, "type") else False

    def _sync():
        if _cuda:
            torch.cuda.synchronize()

    _ti = time.perf_counter()

    # Real targets & meta — direct numpy gather
    sel_raw = raw_windows[selected]                                  # (N, ctx+tgt, C)
    real_targets[:] = sel_raw[:, context_len:, :].astype(np.float32)

    # Instance scaling stats vectorized over windows (same logic as
    # InstanceScaledDataset._instance_scale, batched over axis 0).
    past_all = sel_raw[:, :context_len, :].astype(np.float32)        # (N, ctx, C)
    inst_med_all = np.median(past_all, axis=1)                       # (N, C)
    inst_mad_all = np.maximum(
        np.median(np.abs(past_all - inst_med_all[:, None, :]), axis=1),
        mad_floor,
    ).astype(np.float32)                                              # (N, C)

    # DGP boundary indices for all selected windows in one searchsorted call.
    w_dates_all = pd.to_datetime(window_dates[selected])
    t_last_all = date_idx.searchsorted(w_dates_all.values, side="left")
    t_bnd_all = np.clip(t_last_all - target_len, 0, n_obs - 1).astype(np.int64)
    meta_t_bnd[:] = t_bnd_all

    # ── Covariance-adjusted (Kubiak) vs pure-do() shock across macros ──────────
    # Kubiak (ICAIF'25, §6): shock macro SHK by δ and ADJUST the others by the
    # historical conditional-Gaussian update  Δ_rest = Σ_rest,SHK · Σ_SHK,SHK⁻¹ · δ.
    # Writing the full move as δ·w with w[j]=Σ[j,SHK]/Σ[SHK,SHK] (w[SHK]=1) unifies
    # both modes: pure do() ⇔ w = e_SHK (others held fixed; OOD if macros correlate).
    if covariance_adjust is None:
        covariance_adjust = bool(cf_syn.get("covariance_adjust", False))
    COV_ADJUST = bool(covariance_adjust)
    w_shock = np.zeros(d_base, dtype=np.float64)
    if COV_ADJUST:
        _tr_dgp = np.asarray(
            date_idx <= pd.Timestamp(str(config["data_ingestion"]["train_split_date"]))
        )
        raw_cov = np.cov(dgp._m[_tr_dgp, :d_base], rowvar=False)
        w_shock = raw_cov[:, SHK] / raw_cov[SHK, SHK]
    else:
        w_shock[SHK] = 1.0
    print(f"[cf_synthetic] covariance_adjust={COV_ADJUST}  w={np.round(w_shock, 3)}", flush=True)

    # Per-window per-channel std for z-scaling (snapshot block + per-macro momentum).
    if mvae_exp_std is not None and mvae_exp_std.ndim >= 2:
        es_snap_all = mvae_exp_std[selected, :d_base].astype(np.float32)            # (N, d_base)
        es_mom_all = (mvae_exp_std[selected, d_base:2 * d_base].astype(np.float32)
                      if D_mvae >= 2 * d_base else None)
    else:
        es_snap_all = np.ones((n_sel, d_base), dtype=np.float32)
        es_mom_all = None

    # MacroVAE z-scored snapshot+momentum per window
    mvae_cond_orig_all = mvae_macro_win[selected, -1, :].astype(np.float32)   # (N, D_ext)

    # Per-(shock, channel) raw move = σ·w, then z-scaled per window per channel.
    # (N, K, d_base) deltas for the snapshot/MVAE block, its momentum, and the
    # tanh-space symmetric block. With w = e_SHK these reduce to the single-channel
    # do() shock (only channel SHK moves), preserving the original behaviour.
    SHOCK_ARR = np.asarray(SHOCK_SIGMAS, dtype=np.float32)                    # (K,)
    draw = (SHOCK_ARR[:, None] * w_shock[None, :]).astype(np.float32)         # (K, d_base) raw
    dz_snap_all = draw[None] / np.maximum(es_snap_all[:, None, :], 1e-12)     # (N, K, d_base)
    dz_mom_all = (draw[None] / np.maximum(es_mom_all[:, None, :], 1e-12)
                  if es_mom_all is not None else None)
    dz_diff_all = draw[None] / np.maximum(es_snap_all[:, None, :] * MACRO_TANH_C, 1e-12)

    # Diffusion macro conditioning (depends on encoder type) — move to GPU once.
    if diff_encoder_type == "snapshot":
        macro_cond_all = torch.from_numpy(mvae_cond_orig_all).to(device)      # (N, D_ext)
        # Pre-build shocked snapshot tensors for ALL windows at once.
        macro_sh_all = macro_cond_all.unsqueeze(0).repeat(N_SHOCKS, 1, 1)     # (K, N, D_ext)
        dz_snap_t = torch.from_numpy(dz_snap_all).to(device).permute(1, 0, 2)  # (K, N, d_base)
        macro_sh_all[:, :, :d_base] = macro_sh_all[:, :, :d_base] + dz_snap_t
        if dz_mom_all is not None:
            dz_mom_t = torch.from_numpy(dz_mom_all).to(device).permute(1, 0, 2)  # (K, N, d_base)
            macro_sh_all[:, :, d_base:2 * d_base] = macro_sh_all[:, :, d_base:2 * d_base] + dz_mom_t
    else:
        # Symmetric encoder: (N, D, L) tanh-bounded time series — pull from raw store.
        # macro_windows: (N_total, macro_len, D) → transpose to (N, D, macro_len)
        macro_cond_all = torch.from_numpy(
            np.transpose(macro_windows[selected], (0, 2, 1)).astype(np.float32)
        ).to(device)                                                           # (N, D, L)
        # Shock geometry (macro_cond is tanh-bounded → scale in dz_diff_all).
        # "boundary" (default): the shift is added to the most recent value of
        # the macro window, the same intervention the closed-form
        # TheoreticalCounterfactual applies and the snapshot encoder reads.
        # "flat": the shift is added to every timestep, rewriting the whole
        # macro history. "recentN": the last N timesteps. The paper uses
        # "boundary" throughout.
        dz_diff_t = torch.from_numpy(dz_diff_all).to(device).permute(1, 0, 2)   # (K, N, d_base)
        macro_sh_all = macro_cond_all.unsqueeze(0).repeat(N_SHOCKS, 1, 1, 1)    # (K, N, D, L)
        _geom = str(cf_syn.get("shock_geometry", "boundary")).lower()
        if _geom == "flat":
            macro_sh_all[:, :, :d_base, :] = macro_sh_all[:, :, :d_base, :] + dz_diff_t[:, :, :, None]
        elif _geom == "boundary":
            macro_sh_all[:, :, :d_base, -1] = macro_sh_all[:, :, :d_base, -1] + dz_diff_t
        elif _geom.startswith("recent"):
            _rw = int(_geom.replace("recent", "") or 10)
            macro_sh_all[:, :, :d_base, -_rw:] = macro_sh_all[:, :, :d_base, -_rw:] + dz_diff_t[:, :, :, None]
        else:
            raise ValueError(f"unknown counterfactual_synthetic.shock_geometry {_geom!r}")
        print(f"[cf_synthetic] temporal shock_geometry = {_geom}", flush=True)

    # ── ITE: forward-encode the real targets to x0 (for DDIM inversion) ───────
    # x0 depends only on the returns (not on the macro), so we reuse the training
    # forward-encoder. Inversion under the FACTUAL macro happens per-chunk below.
    x0_invert = None
    if run_diffusion and latent_mode == "invert":
        from src.diffusion.lab import _precompute_x0_fast
        x0_invert, _, _ = _precompute_x0_fast(
            sel_raw, np.transpose(macro_windows[selected], (0, 2, 1)).astype(np.float32)
            if diff_encoder_type != "snapshot" else mvae_cond_orig_all,
            qt_artifact, wavelet_type, wavelet_level, level_scales,
            context_len, inst_c, mad_floor, z_clip,
        )   # (n_sel, C, n_levels, T) CPU tensor — macro arg unused for x0
        print(f"[cf_synthetic] ITE: forward-encoded {x0_invert.shape[0]} windows to x0",
              flush=True)

    _t_phase["setup"] += time.perf_counter() - _ti

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE B — theory (analytical, CPU, vectorized over shocks per window)
    # ═══════════════════════════════════════════════════════════════════════
    _ti = time.perf_counter()
    for wi in range(n_sel):
        t_bnd = int(t_bnd_all[wi])
        state = dgp.get_state_at(t_bnd)
        tc_base = state[_tc_base_key]      # CPI scalar (single) or macro vector (complex)
        h_T = state["h"]
        eps_T_sq = state["eps_sq"]
        meta_cpi_T[wi] = state["cpi"]
        for si, sigma in enumerate(SHOCK_SIGMAS):
            if _variant == "complex":
                tc_sh = tc.compute(tc_base, float(sigma), h_T, eps_T_sq,
                                   horizon=target_len,
                                   shock_vec=float(sigma) * w_shock)
            else:
                tc_sh = tc.compute(tc_base, float(sigma), h_T, eps_T_sq,
                                   horizon=target_len)
            theory_effects[wi, si] = tc_sh["delta_E_ret"].astype(np.float32)
    _t_phase["theory"] += time.perf_counter() - _ti

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE C — Diffusion ATE, chunked across windows
    # Layout per chunk (W windows, K shocks, M ATE samples):
    #   batch axis 0 = [base_w0_k0_m0..M-1, base_w0_k1, ..., base_w_{W-1}_k_{K-1},
    #                   shock_w0_k0_m0..M-1, ..., shock_w_{W-1}_k_{K-1}]
    #   total batch = 2 · W · K · M
    # Paired latents: shock_half x_T = base_half x_T (identical halves).
    # ═══════════════════════════════════════════════════════════════════════
    if run_diffusion:
        M = N_ATE_SAMPLES
        K = N_SHOCKS
        _exp_dims_per_W = [-1] * (macro_cond_all.dim() - 1)   # macro shape minus batch dim

        chunk_iter = range(0, n_sel, W_CHUNK)
        for chunk_start in tqdm(chunk_iter, desc="Diff (chunked)", unit="chunk"):
            chunk_end = min(chunk_start + W_CHUNK, n_sel)
            W = chunk_end - chunk_start

            # Macro slices for this chunk: (W, ...) base & (K, W, ...) shocked
            mac_W = macro_cond_all[chunk_start:chunk_end]              # (W, ...)
            mac_sh_W = macro_sh_all[:, chunk_start:chunk_end]          # (K, W, ...)

            # Expand to (W, K, M, ...) then flatten to (W·K·M, ...)
            mac_base_flat = (
                mac_W.unsqueeze(1).unsqueeze(1)
                     .expand(W, K, M, *_exp_dims_per_W)
                     .reshape(W * K * M, *mac_W.shape[1:])
                     .contiguous()
            )
            mac_sh_flat = (
                mac_sh_W.permute(1, 0, *range(2, mac_sh_W.dim()))
                        .unsqueeze(2)
                        .expand(W, K, M, *_exp_dims_per_W)
                        .reshape(W * K * M, *mac_W.shape[1:])
                        .contiguous()
            )
            mac_full = torch.cat([mac_base_flat, mac_sh_flat], dim=0)    # (2·W·K·M, ...)

            # Latent: fresh Gaussian (ATE) or DDIM-inverted real window (ITE).
            # Paired baseline ↔ shocked always share x_T.
            if latent_mode == "invert":
                # Invert each window's real x0 under its FACTUAL macro (M=1).
                x0_chunk = x0_invert[chunk_start:chunk_end].to(device)   # (W, C, n_levels, T)
                with torch.no_grad():
                    xT_win = diffusion.ddim_invert(x0_chunk, mac_W, steps=DDIM_STEPS)  # (W, ...)
                # Share the inverted latent across all K shocks (M=1).
                x_T_base = (
                    xT_win.unsqueeze(1)
                          .expand(W, K, *wavelet_shape)
                          .reshape(W * K, *wavelet_shape)
                          .contiguous()
                )
            else:
                # Per-shock fresh noise (M each).
                x_T_base = torch.randn(W * K * M, *wavelet_shape, device=device)
            x_T_full = torch.cat([x_T_base, x_T_base], dim=0)            # (2·W·K·M, ...)

            # Inversion stats: each window's stats repeated K·M times in each half.
            inst_med_chunk = inst_med_all[chunk_start:chunk_end]
            inst_mad_chunk = inst_mad_all[chunk_start:chunk_end]
            inst_med_half = np.repeat(inst_med_chunk, K * M, axis=0)     # (W·K·M, C)
            inst_mad_half = np.repeat(inst_mad_chunk, K * M, axis=0)
            inst_med_full = np.concatenate([inst_med_half, inst_med_half], axis=0)
            inst_mad_full = np.concatenate([inst_mad_half, inst_mad_half], axis=0)

            # SGD base cond: BASELINE macro for ALL 2·W·K·M samples.
            sgd_base = (
                mac_base_flat.repeat(2, *([1] * (mac_base_flat.dim() - 1))).contiguous()
                if sgd_gamma is not None and sgd_gamma != 1.0 else None
            )

            _sync(); _ti = time.perf_counter()
            with torch.no_grad():
                wav_chunk = diffusion.ddim_sample(
                    shape=(2 * W * K * M, *wavelet_shape),
                    macro_cond=mac_full,
                    steps=DDIM_STEPS, eta=DDIM_ETA,
                    guidance_scale=GUIDANCE, x_init=x_T_full,
                    sgd_base_cond=sgd_base,
                    sgd_gamma=(sgd_gamma if sgd_gamma is not None else 1.0),
                )
            _sync(); _t_phase["ate_sample"] += time.perf_counter() - _ti

            _ti = time.perf_counter()
            raw_chunk = _wav_to_raw(
                wav_chunk, qt_artifact, wavelet_type, level_scales,
                inst_med_full, inst_mad_full, inst_c,
            )                                                              # (2·W·K·M, T, C)
            _t_phase["ate_wav2raw"] += time.perf_counter() - _ti

            half = W * K * M
            raw_base = raw_chunk[:half].reshape(W, K, M, target_len, n_assets)
            raw_shock = raw_chunk[half:].reshape(W, K, M, target_len, n_assets)
            diff_effects[chunk_start:chunk_end] = (
                (raw_shock - raw_base).mean(axis=2).astype(np.float32)
            )
            if SAVE_DIST:
                # §4.5: per-cell dispersion across the M ATE samples. Delta-sigma =
                # sig_shock - sig_base is the shock's volatility effect (the ScaleHead's
                # job); mean-estimators (OLS/VAR) have Delta-sigma == 0 by construction.
                sig_base[chunk_start:chunk_end]  = raw_base.std(axis=2).astype(np.float32)
                sig_shock[chunk_start:chunk_end] = raw_shock.std(axis=2).astype(np.float32)

            # Free chunk-local GPU tensors before next chunk to keep VRAM low
            del mac_base_flat, mac_sh_flat, mac_full, x_T_base, x_T_full, wav_chunk
            if latent_mode == "invert":
                del x0_chunk, xT_win
            if sgd_base is not None:
                del sgd_base
            if _cuda:
                torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE D — MacroVAE, single batch over all windows
    # Encode all N windows once; decode K+1 conditions per window in one pass.
    # ═══════════════════════════════════════════════════════════════════════
    if run_mvae:
        _sync(); _ti = time.perf_counter()
        with torch.no_grad():
            x_block_all = torch.from_numpy(
                sel_raw[:, context_len:, :].astype(np.float32)
            ).to(device)                                                   # (N, W, C)

            cond_orig_t = torch.from_numpy(mvae_cond_orig_all).to(device)  # (N, D_ext)

            # Shocked conditions for all windows: (K, N, D_ext)
            cond_sh_all = cond_orig_t.unsqueeze(0).repeat(N_SHOCKS, 1, 1)
            dz_snap_mv = torch.from_numpy(dz_snap_all).to(device).permute(1, 0, 2)   # (K, N, d_base)
            cond_sh_all[:, :, :d_base] = cond_sh_all[:, :, :d_base] + dz_snap_mv
            if dz_mom_all is not None:
                dz_mom_mv = torch.from_numpy(dz_mom_all).to(device).permute(1, 0, 2)
                cond_sh_all[:, :, d_base:2 * d_base] = cond_sh_all[:, :, d_base:2 * d_base] + dz_mom_mv

            # Encode once (N latents)
            z_mu, z_logvar = mvae_model.encode(x_block_all)                # (N, latent)
            mvae_post_var_mean = float(z_logvar.exp().mean().item())

            # Build ((1+K)·N, D_ext) condition batch and matching ((1+K)·N, latent) z batch
            cond_stack = torch.cat([cond_orig_t.unsqueeze(0), cond_sh_all], dim=0)  # (1+K, N, D_ext)
            cond_flat = cond_stack.reshape((1 + N_SHOCKS) * n_sel, -1)

            def _mvae_decode(z):
                """Decode one (N, latent) batch under all 1+K conditions -> (1+K, N, T, C)."""
                z_flat = z.unsqueeze(0).expand(1 + N_SHOCKS, -1, -1) \
                          .reshape((1 + N_SHOCKS) * n_sel, -1).contiguous()
                norm_r, vol, mean_o = mvae_model.decode(z_flat, cond_flat)
                vol_sc = vol[:, 0, :].unsqueeze(1)
                out = ((norm_r + mean_o) * torch.exp(vol_sc)).cpu().numpy()
                return out.reshape(1 + N_SHOCKS, n_sel, target_len, n_assets)

            if mvae_latent == "prior":
                # Prior-marginalised: z ~ N(0, I) shared by baseline and shocked
                # decode, averaged over MVAE_PRIOR_SAMPLES draws per window.
                torch.manual_seed(SEED)
                mvae_out = np.zeros((1 + N_SHOCKS, n_sel, target_len, n_assets), np.float64)
                for _ in range(MVAE_PRIOR_SAMPLES):
                    mvae_out += _mvae_decode(torch.randn_like(z_mu))
                mvae_out /= MVAE_PRIOR_SAMPLES
            else:
                mvae_out = _mvae_decode(z_mu)
        # Effect = shocked − baseline. Want (n_sel, K, T, C).
        mvae_effects[:] = np.transpose(
            mvae_out[1:] - mvae_out[0:1], (1, 0, 2, 3)
        ).astype(np.float32)
        _sync(); _t_phase["mvae"] += time.perf_counter() - _ti

    elapsed = time.perf_counter() - t0
    print(f"[cf_synthetic] Generation: {n_sel} windows × {N_SHOCKS} shocks "
          f"× 2 models in {elapsed:.1f}s  (W_chunk={W_CHUNK})", flush=True)
    print(f"[cf_synthetic] phase totals (s): "
          f"setup={_t_phase['setup']:.2f} "
          f"theory={_t_phase['theory']:.2f} "
          f"ate_sample={_t_phase['ate_sample']:.2f} "
          f"ate_wav2raw={_t_phase['ate_wav2raw']:.2f} "
          f"mvae={_t_phase['mvae']:.2f}", flush=True)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    def _metrics(model_eff: np.ndarray, theory_eff: np.ndarray,
                 mask_run: bool) -> dict:
        """Compute causal metrics per shock (model_eff vs theory_eff).

        model_eff  : (N, K, T, C)
        theory_eff : (N, K, T, C)
        """
        if not mask_run:
            return {}

        per_shock: list[dict] = []
        for si in range(N_SHOCKS):
            # shape (1, N, T, C) — S=1 deterministic sample
            pred_s = model_eff[:, si][np.newaxis]   # (1, N, T, C)
            true   = theory_eff[:, si]              # (N, T, C)
            cm = compute_causal_metrics(true, pred_s)
            per_shock.append(cm)

        def _collect(key: str) -> list[float]:
            return [ps[key] for ps in per_shock]

        def _mean(key: str) -> float:
            vals = [v for v in _collect(key) if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        # ── NEW path metrics rho/beta/eps (ADDED alongside Dir/Amp) ───────────
        # Time-resolved comparison of the cumulative correction path vs the exact
        # TheoreticalCounterfactual (here the ground truth replaces the SVAR used
        # on real data). rho=direction, beta=gain, eps=normalized path error.
        def _path() -> dict:
            FLOOR = 1e-12
            rhos, betas, epss = [], [], []
            for c in range(model_eff.shape[3]):
                for k in range(model_eff.shape[1]):
                    vt = np.cumsum(theory_eff[:, k, :, c].mean(axis=0))
                    vm = np.cumsum(model_eff[:, k, :, c].mean(axis=0))
                    nt = float(np.linalg.norm(vt))
                    if nt < FLOOR:
                        continue
                    nm = float(np.linalg.norm(vm))
                    dot = float(vm @ vt)
                    rhos.append(dot / (nm * nt + 1e-12))
                    betas.append(dot / (nt * nt))
                    epss.append(float(np.linalg.norm(vm - vt) / nt))
            return {
                "rho_path_mean":    float(np.mean(rhos))    if rhos else float("nan"),
                "beta_path_median": float(np.median(betas)) if betas else float("nan"),
                "eps_path_median":  float(np.median(epss))  if epss else float("nan"),
            }

        _out = {
            # ── per-shock lists ──────────────────────────────────────
            "mae_per_shock":                  _collect("mae"),
            "pehe_per_shock":                 _collect("pehe"),
            "direction_accuracy_per_shock":   _collect("directional_accuracy"),
            "amplification_ratio_per_shock":  _collect("amplification_ratio"),
            "coverage_90_per_shock":          _collect("coverage_90"),
            # squared PEHE per shock
            "mse_per_shock": [ps["pehe"] ** 2 for ps in per_shock],
            # ── aggregate means ──────────────────────────────────────
            "mae_mean":                 _mean("mae"),
            "pehe_mean":                _mean("pehe"),
            "direction_accuracy_mean":  _mean("directional_accuracy"),
            "amplification_ratio_mean": _mean("amplification_ratio"),
            "coverage_90_mean":         _mean("coverage_90"),
            "mse_mean": float(np.mean([ps["pehe"] ** 2 for ps in per_shock])),
        }
        _out.update(_path())          # rho/beta/eps ADDED, not replacing
        return _out

    metrics = {
        "n_windows":    n_sel,
        "n_shocks":     N_SHOCKS,
        "shock_deltas": SHOCK_SIGMAS,
        "shock_labels": SHOCK_LABELS,
        "ate_samples":  N_ATE_SAMPLES,
        "shock_macro_index": SHK,
        "covariance_adjust": COV_ADJUST,
        "mvae_latent":  mvae_latent if run_mvae else None,
        "mvae_prior_samples": MVAE_PRIOR_SAMPLES if (run_mvae and mvae_latent == "prior") else None,
        "mvae_posterior_var_mean": mvae_post_var_mean,
        "diffusion":    _metrics(diff_effects, theory_effects, run_diffusion),
        "macrovae":     _metrics(mvae_effects, theory_effects, run_mvae),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    run_dir = create_run_dir(synth_base, "counterfactual_synthetic")
    print(f"[cf_synthetic] Output: {run_dir}", flush=True)

    np.save(run_dir / "theory_effects.npy", theory_effects)
    np.save(run_dir / "real_targets.npy",   real_targets)

    if run_diffusion:
        np.save(run_dir / "diff_effects.npy", diff_effects)
    if run_mvae:
        np.save(run_dir / "mvae_effects.npy", mvae_effects)
    if SAVE_DIST and sig_base is not None:
        np.save(run_dir / "sig_base.npy", sig_base)
        np.save(run_dir / "sig_shock.npy", sig_shock)
        print(f"[cf_synthetic] SAVE_DIST: saved sig_base/sig_shock shape={sig_base.shape}", flush=True)

    np.savez(run_dir / "window_meta.npz",
             window_indices=meta_widx,
             t_boundaries=meta_t_bnd,
             cpi_T=meta_cpi_T)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    shapes = {
        "theory_effects": list(theory_effects.shape),
        "real_targets":   list(real_targets.shape),
    }
    if diff_effects is not None:
        shapes["diff_effects"] = list(diff_effects.shape)
    if mvae_effects is not None:
        shapes["mvae_effects"] = list(mvae_effects.shape)

    save_manifest(
        run_dir,
        task="counterfactual_synthetic",
        config=config,
        checkpoint_paths={
            k: str(v) for k, v in [
                ("diffusion", diff_ckpt_path),
                ("macrovae",  mvae_ckpt_path),
            ] if v is not None
        },
        shapes=shapes,
        elapsed_seconds=elapsed,
        extra={
            "n_windows":      n_sel,
            "n_shocks":       N_SHOCKS,
            "ate_samples":    N_ATE_SAMPLES,
            "shock_sigmas":   SHOCK_SIGMAS,
            "shock_labels":   SHOCK_LABELS,
            "tickers":        tickers,
            "n_assets":       n_assets,
            "target_len":     target_len,
            "context_len":    context_len,
            "sgd_gamma":      sgd_gamma,
            "guidance_scale": GUIDANCE,
            "ddim_steps":     DDIM_STEPS,
            "ddim_eta":       DDIM_ETA,
            "diff_macro_encoder": diff_encoder_type,
            "latent_mode":    latent_mode,
        },
    )

    write_latest(synth_base, "counterfactual_synthetic", run_dir)
    pruned = prune_old_runs(synth_base, "counterfactual_synthetic", max_runs)
    if pruned:
        print(f"[cf_synthetic] Pruned {len(pruned)} old runs")

    # ── Human-readable per-run summary (config + ckpt + metrics) ───────────
    from src.run_summary import write_run_summary
    summary_path = write_run_summary(
        run_dir,
        task="counterfactual_synthetic",
        config=config,
        metrics=metrics,
        checkpoint_paths={k: str(v) for k, v in [
            ("diffusion", diff_ckpt_path),
            ("macrovae",  mvae_ckpt_path),
        ] if v is not None},
        elapsed_seconds=elapsed,
        extra={
            "n_windows":    n_sel,
            "n_shocks":     N_SHOCKS,
            "ate_samples":  N_ATE_SAMPLES,
            "shock_deltas": SHOCK_SIGMAS,
            "shock_labels": SHOCK_LABELS,
            "guidance_scale": GUIDANCE,
            "sgd_gamma":    sgd_gamma,
            "ddim_steps":   DDIM_STEPS,
            "tickers":      tickers,
        },
    )
    print(f"[cf_synthetic] Summary: {summary_path.name}")

    print(f"\n[cf_synthetic] Done -> {run_dir}")

    NEG_IDX = [i for i, d in enumerate(SHOCK_SIGMAS) if d < 0]
    POS_IDX = [i for i, d in enumerate(SHOCK_SIGMAS) if d > 0]

    def _dir_split(m_dict, lbl):
        if not m_dict:
            return
        dir_all = m_dict.get("direction_accuracy_mean", float("nan"))
        per = m_dict.get("direction_accuracy_per_shock", [])
        dir_neg = float(np.mean([per[i] for i in NEG_IDX])) if per else float("nan")
        dir_pos = float(np.mean([per[i] for i in POS_IDX])) if per else float("nan")
        amp     = m_dict.get("amplification_ratio_mean", float("nan"))
        print(f"  {lbl:<28} Dir={dir_all:.2%}  "
              f"(neg={dir_neg:.2%} / pos={dir_pos:.2%})  "
              f"Amp={amp:.3f}x")

    print("\n[cf_synthetic] -- ATE (fresh x_T ~ N(0,I)) -----------------------")
    _dir_split(metrics["diffusion"], f"Diffusion ATE (M={N_ATE_SAMPLES})")
    _dir_split(metrics["macrovae"],  "MacroVAE")

    return run_dir
