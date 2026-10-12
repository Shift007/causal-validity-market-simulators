"""Walk-forward OOS synthetic generation.

Generates diffusion + MacroVAE synthetic returns per quarter and saves them.
Visualization / metrics computation stays in the notebook.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.diffusion import generate_quarter, apply_qt_clip_override
from src.generation.runner import (
    build_eval_schedule,
    create_run_dir,
    generate_mvae,
    get_device,
    load_config,
    load_mvae_model,
    load_preprocessing_artifacts,
    prune_old_runs,
    read_latest,
    save_manifest,
    write_latest,
)


def run(project_root: Path, config: dict | None = None,
        model_filter: str = "both") -> Path:
    """Execute walk-forward OOS generation. Returns the run directory.

    Args:
        model_filter: 'both' (default), 'diffusion', or 'macrovae'.
            When a single model is selected, the other model's arrays are
            copied from the most recent previous run (if available).
    """
    config = config or load_config(project_root)
    device = get_device()

    run_diffusion = model_filter in ("both", "diffusion")
    run_mvae = model_filter in ("both", "macrovae")

    from src.paths import resolve_paths
    _paths = resolve_paths(project_root, config)
    data_dir = _paths.processed_dir
    gen_cfg = config.get("generation", {})
    synth_base = _paths.results_dir
    max_runs = int(gen_cfg.get("max_runs_per_task", 3))

    # Previous run for copying skipped model's arrays
    prev_run_dir = read_latest(synth_base, "wf_eval") if model_filter != "both" else None
    if prev_run_dir:
        print(f"Previous run (for copy): {prev_run_dir}")
    elif model_filter != "both":
        print("WARNING: no previous run found — skipped model will have no arrays")

    # ── Load artifacts ────────────────────────────────────────────────────
    art = load_preprocessing_artifacts(data_dir)
    raw_windows = art["raw_windows"]
    macro_windows = art["macro_windows"]
    mvae_macro_windows = art["mvae_macro_windows"]
    window_dates = art["window_dates"]
    target_qt = art["target_qt"]
    level_scales = art["level_scales"]
    n_assets = art["n_assets"]
    context_len = art["context_len"]
    target_len = art["target_len"]
    wavelet_type = art["wavelet_type"]
    wavelet_level = art["wavelet_level"]
    n_levels = art["n_levels"]
    tickers = art["tickers"]
    D_mvae = art["D_mvae"]

    window_end_ts = pd.to_datetime(window_dates)

    # ── Config params ─────────────────────────────────────────────────────
    anal_cfg = config.get("analysis", {})
    n_samples = int(anal_cfg.get("num_samples", 32))
    guidance_scale = float(anal_cfg.get("guidance_scale", 2.0))
    ddim_steps = int(anal_cfg.get("ddim_steps", 25))
    gen_batch_size = int(anal_cfg.get("batch_size", 32))
    max_eval_windows = int(anal_cfg.get("max_eval_windows", 0))
    # Sub-sampling stride for test windows (dataset sub-sampling only; the
    # checkpoints are unaffected).
    #   1   = all stride-1 windows (default, with overlap)
    #   0   = auto = target_len (non-overlapping)
    #   N>1 = custom
    eval_stride = int(anal_cfg.get("eval_stride", 1))
    if eval_stride == 0:
        eval_stride = target_len
    save_synth = bool(config.get("training", {}).get("walk_forward", {}).get("save_synth_arrays", True))

    # Diffusion inference-time controls (defaults match src.diffusion.lab CLI defaults).
    eta     = float(anal_cfg.get("eta", 0.0))
    qt_clip = float(anal_cfg.get("qt_clip", 3.2))

    wf_cfg = config["training"]["walk_forward"]
    ckpt_wf_dir = _paths.diffusion_ckpt_dir

    wavelet_shape = (n_assets, n_levels, target_len)
    global_cond_dim = macro_windows.shape[2]
    macro_seq_len = macro_windows.shape[1]

    # MacroVAE checkpoint dir
    mvae_ckpt_dir = _paths.macrovae_ckpt_dir

    # ── Build schedule ────────────────────────────────────────────────────
    schedule = build_eval_schedule(config)
    print(f"Walk-forward schedule: {len(schedule)} steps")

    # ── Create run directory ──────────────────────────────────────────────
    run_dir = create_run_dir(synth_base, "wf_eval")
    per_step_dir = run_dir / "per_step"
    per_step_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    t0_total = time.perf_counter()
    ckpt_paths_used = {}
    completed_steps = []

    # ── Accumulators for pooled arrays ────────────────────────────────────
    accum_real_diff = []
    accum_synth_diff = []
    accum_real_mvae = []
    accum_synth_mvae = []
    accum_test_idx = []   # global window indices, for downstream nudge / probes

    for eval_label, eval_start, eval_end, ckpt_name in schedule:
        print()
        print("=" * 70)
        print(f"Step {eval_label}  ({eval_start.date()} → {eval_end.date()})  ckpt: {ckpt_name}")

        # ── Check checkpoint + step artifacts ─────────────────────────────
        ckpt_path = ckpt_wf_dir / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: {ckpt_path.name} not found")
            continue

        # Global preprocessing artifacts (the new diffusion model trains and inverts
        # with a single QT + level_scales, no per-step pipeline needed).
        step_qt = art["target_qt"]
        step_ls = art["level_scales"]

        # ── Select eval windows ───────────────────────────────────────────
        eval_mask = (window_end_ts >= eval_start) & (window_end_ts <= eval_end)
        eval_indices = np.where(eval_mask)[0]
        n_total_wins = len(eval_indices)

        if n_total_wins == 0:
            print(f"  SKIP: no windows in range")
            continue

        # Apply stride first (sub-sampling), then optional cap.
        if eval_stride > 1:
            eval_indices = eval_indices[::eval_stride]

        if max_eval_windows > 0 and len(eval_indices) > max_eval_windows:
            step = len(eval_indices) / max_eval_windows
            sel = np.array([int(i * step) for i in range(max_eval_windows)])
            eval_indices = eval_indices[sel]

        print(f"  Windows: {len(eval_indices)} / {n_total_wins}")

        eval_raw = raw_windows[eval_indices]

        # ── Diffusion encoder dispatch ──────────────────────────────────────
        # Peek the checkpoint to detect macro encoder type (symmetric|snapshot).
        # For "snapshot" (ablation) we feed mvae_macro_windows[:, -1, :] instead
        # of the full time-series macro_windows.
        _peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _diff_encoder_type = _peek.get("macro_encoder_type", "symmetric")
        del _peek
        if _diff_encoder_type == "snapshot":
            eval_macro = mvae_macro_windows[eval_indices, -1, :]
        else:
            eval_macro = macro_windows[eval_indices]

        # Real targets — always needed regardless of which model
        real_diff = eval_raw[:, context_len:, :]  # (N_eval, target_len, C)

        # ── Diffusion generation (new src.diffusion.generate_quarter) ──────────
        synth_diff = None
        real_diff_from_diff = None
        if run_diffusion:
            print(f"  Diffusion: generating {len(eval_indices)} × {n_samples}  "
                  f"(encoder={_diff_encoder_type}) ...")
            t0_diff = time.perf_counter()
            ckpt_paths_used[f"diff_{eval_label}"] = str(ckpt_path)

            # Apply optional inference-time QT cap (mirrors the lab CLI default).
            step_qt_eff = apply_qt_clip_override(step_qt, qt_clip)

            real_diff_from_diff, synth_diff = generate_quarter(
                ckpt_path=ckpt_path,
                raw_q=eval_raw,
                macro_q=eval_macro,
                norm_params=art["norm_params"],
                qt_artifact=step_qt_eff,
                level_scales=step_ls,
                device=device,
                n_samples=n_samples,
                ddim_steps=ddim_steps,
                eta=eta,
                guidance_scale=guidance_scale,
                label=eval_label,
                verbose=True,
            )
            dt_diff = time.perf_counter() - t0_diff
            print(f"  Diffusion done in {dt_diff:.1f}s")
            torch.cuda.empty_cache()
        elif prev_run_dir:
            # Copy from previous run
            prev_diff = prev_run_dir / "per_step" / eval_label / "diff_synth.npy"
            if prev_diff.exists():
                synth_diff = np.load(prev_diff)
                print(f"  Diffusion: copied from previous run")
            else:
                print(f"  Diffusion: SKIP — no previous arrays for {eval_label}")

        # Use the diffusion path's real targets when available, otherwise fall back
        # to the canonical extraction (identical, both equal eval_raw[:, ctx:, :]).
        real_diff_canonical = real_diff_from_diff if real_diff_from_diff is not None else real_diff

        if synth_diff is not None:
            accum_real_diff.append(real_diff_canonical)
            accum_synth_diff.append(synth_diff)
            accum_test_idx.append(np.asarray(eval_indices))

        # ── MacroVAE generation ───────────────────────────────────────────
        synth_mvae = None

        if run_mvae:
            _year_label = ckpt_name.replace("ckpt_wf_", "")  # e.g. "2015" or "initial"
            _mvae_annual = mvae_ckpt_dir / f"ckpt_mvae_{_year_label}.pt"
            if _mvae_annual.exists():
                mvae_ckpt_path = _mvae_annual
            else:
                _mvae_cands = sorted(mvae_ckpt_dir.glob(f"ckpt_mvae_{_year_label}Q*.pt"))
                mvae_ckpt_path = _mvae_cands[-1] if _mvae_cands else mvae_ckpt_dir / "ckpt_mvae_initial.pt"

            if mvae_ckpt_path.exists():
                print(f"  MacroVAE: generating {len(eval_indices)} × {n_samples} ...")
                t0_mvae = time.perf_counter()

                macrovae_root = project_root / "MacroVAE"
                import sys
                if str(macrovae_root) not in sys.path:
                    sys.path.insert(0, str(macrovae_root))

                mvae_model, mvae_state, hp = load_mvae_model(
                    mvae_ckpt_path, n_assets, target_len, D_mvae, device
                )
                mvae_latent_dim = hp.get("latent_dim", 32)
                ckpt_paths_used[f"mvae_{eval_label}"] = str(mvae_ckpt_path)

                # Fix B: load the per-step z-scored macro matching this ckpt.
                # The ckpt label IS the cutoff (e.g., 'initial' for ckpt_initial
                # trained on data ≤ 2014-12-31; '2015' for ckpt_2015 trained
                # on data ≤ 2015-12-31). Each per-step file has cond normalized
                # with the in-sample stats up to that cutoff.
                _perstep_path = data_dir / "mvae_perstep" / f"macro_zscore_{_year_label}.npy"
                if _perstep_path.exists():
                    mvae_macro_step = np.load(_perstep_path)
                    cond_eval = mvae_macro_step[eval_indices, -1, :]
                    print(f"  Using per-step macro_zscore_{_year_label}.npy (in-sample stats ≤ {_year_label})")
                else:
                    # Fallback: use the global file (latest cutoff per preprocess).
                    print(f"  WARNING: per-step file not found ({_perstep_path.name}), "
                          f"falling back to legacy macro_windows_zscore.npy")
                    cond_eval = mvae_macro_windows[eval_indices, -1, :]
                synth_mvae = generate_mvae(
                    mvae_model, cond_eval, n_samples,
                    latent_dim=mvae_latent_dim,
                    target_len=target_len,
                    n_assets=n_assets,
                    device=device,
                )  # (N_eval, n_samples, tgt, C)
                dt_mvae = time.perf_counter() - t0_mvae
                print(f"  MacroVAE done in {dt_mvae:.1f}s")

                del mvae_model
                torch.cuda.empty_cache()
            else:
                print(f"  MacroVAE: SKIP — no checkpoint {mvae_ckpt_path.name}")
        elif prev_run_dir:
            # Copy from previous run
            prev_mvae = prev_run_dir / "per_step" / eval_label / "mvae_synth.npy"
            if prev_mvae.exists():
                synth_mvae = np.load(prev_mvae)
                print(f"  MacroVAE: copied from previous run")
            else:
                print(f"  MacroVAE: SKIP — no previous arrays for {eval_label}")

        if synth_mvae is not None:
            accum_real_mvae.append(real_diff_canonical)
            accum_synth_mvae.append(synth_mvae)

        # ── Save per-step arrays ─────────────────────────────────────
        step_out = per_step_dir / eval_label
        step_out.mkdir(parents=True, exist_ok=True)
        np.save(step_out / "real.npy", real_diff_canonical)
        if synth_diff is not None:
            np.save(step_out / "diff_synth.npy", synth_diff)
        if synth_mvae is not None:
            np.save(step_out / "mvae_synth.npy", synth_mvae)

        completed_steps.append(eval_label)
        print(f"  Saved: {step_out}")

    # ── Pooled OOS arrays ─────────────────────────────────────────────────
    pooled_dir = run_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    shapes = {}
    if accum_real_diff:
        pooled_real = np.concatenate(accum_real_diff, axis=0)
        pooled_synth_d = np.concatenate(accum_synth_diff, axis=0)
        pooled_test_idx = np.concatenate(accum_test_idx, axis=0)
        np.save(pooled_dir / "real.npy", pooled_real)
        np.save(pooled_dir / "diff_synth.npy", pooled_synth_d)
        np.save(pooled_dir / "test_idx.npy", pooled_test_idx)
        shapes["pooled_real"] = pooled_real.shape
        shapes["pooled_diff_synth"] = pooled_synth_d.shape
        shapes["pooled_test_idx"] = pooled_test_idx.shape
        print(f"\nPooled diffusion: real={pooled_real.shape}, synth={pooled_synth_d.shape}")

    if accum_synth_mvae:
        pooled_synth_m = np.concatenate(accum_synth_mvae, axis=0)
        np.save(pooled_dir / "mvae_synth.npy", pooled_synth_m)
        shapes["pooled_mvae_synth"] = pooled_synth_m.shape
        print(f"Pooled MacroVAE: synth={pooled_synth_m.shape}")

    # ── Manifest ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0_total
    save_manifest(
        run_dir,
        task="wf_eval",
        config=config,
        checkpoint_paths=ckpt_paths_used,
        shapes=shapes,
        elapsed_seconds=elapsed,
        extra={
            "n_samples_per_window": n_samples,
            "ddim_steps": ddim_steps,
            "guidance_scale": guidance_scale,
            "completed_steps": completed_steps,
            "n_assets": n_assets,
            "context_len": context_len,
            "target_len": target_len,
            "tickers": tickers,
            "model_filter": model_filter,
        },
    )

    write_latest(synth_base, "wf_eval", run_dir)
    pruned = prune_old_runs(synth_base, "wf_eval", max_runs)
    if pruned:
        print(f"Pruned {len(pruned)} old runs: {pruned}")

    print(f"\nDone: {len(completed_steps)} steps in {elapsed:.0f}s → {run_dir}")
    return run_dir


if __name__ == "__main__":
    import argparse
    from src.generation.runner import get_project_root

    parser = argparse.ArgumentParser(description="Walk-forward OOS generation")
    parser.add_argument("--model_filter", choices=["both", "diffusion", "macrovae"],
                        default="both")
    parser.add_argument("--eval_stride", type=int, default=None,
                        help="Override config analysis.eval_stride. "
                             "1=all (overlap), 0=auto=target_len (non-overlap), N>1=custom.")
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config(root)
    if args.eval_stride is not None:
        cfg.setdefault("analysis", {})["eval_stride"] = args.eval_stride
    run(root, config=cfg, model_filter=args.model_filter)
