#!/usr/bin/env python
"""
04 — MacroVAE Walk-Forward Training: train the MacroVAE baseline model
(Kubiak et al., 2024) using walk-forward expanding-window methodology.

Replaces notebooks/04_macrovae_training.ipynb.

Usage:
    python scripts/train_macrovae.py
    python scripts/train_macrovae.py --config path/to/config.yaml
    python scripts/train_macrovae.py --restart   # wipe & retrain from scratch
"""

import argparse
import logging
import shutil
import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="MacroVAE walk-forward training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.yaml (default: <project_root>/config.yaml)")
    ap.add_argument("--restart", action="store_true",
                    help="Wipe existing MacroVAE checkpoints and retrain from scratch")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (cuda / cpu)")
    ap.add_argument("--explode", type=int, nargs="?", const=10, default=0, metavar="N",
                    help="Diagnostic: generate samples every N epochs to track convergence")
    # ── Architecture ───────────────────────────────────────────────────────────
    ap.add_argument("--hidden_dim",     type=int,   default=128,
                    help="VAE encoder/decoder hidden size")
    ap.add_argument("--latent_dim",     type=int,   default=32,
                    help="Latent space dimensionality")
    ap.add_argument("--cond_embed_dim", type=int,   default=16,
                    help="Macro conditioning embedding size")
    # ── Training ───────────────────────────────────────────────────────────────
    ap.add_argument("--lr",            type=float, default=4e-4,
                    help="AdamW learning rate")
    ap.add_argument("--batch",         type=int,   default=512,
                    help="Mini-batch size")
    ap.add_argument("--kl_weight",     type=float, default=0.01,
                    help="Max KL annealing weight")
    ap.add_argument("--kl_anneal_eps", type=int,   default=2000,
                    help="KL warm-up steps (epochs × batches_per_epoch)")
    ap.add_argument("--epochs",        type=int,   default=1000,
                    help="Phase-0 training epochs (initial full training)")
    ap.add_argument("--wf_epochs",     type=int,   default=100,
                    help="Per-step fine-tuning epochs in walk-forward phase")
    ap.add_argument("--seed",          type=int,   default=None,
                    help="Seed for weight init and batch order (unseeded when omitted)")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Loss & Train helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _mvae_loss(x, normalized_recon, volatility, mean_out, mu, log_var, kl_weight):
    x_std = torch.std(x, dim=1, keepdim=True).clamp(min=1e-6)
    x_norm = x / x_std
    x_norm_mean = torch.mean(x_norm, dim=1, keepdim=True)
    x_norm_demeaned = x_norm - x_norm_mean

    returns_loss = F.mse_loss(normalized_recon, x_norm_demeaned)
    volatility_loss = F.mse_loss(volatility, torch.log(x_std))
    mean_loss = F.mse_loss(mean_out, x_norm_mean) * 5
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return returns_loss + volatility_loss + mean_loss + kl_weight * kl_loss


@torch.no_grad()
def _explode_diagnostic(model, x_sample, cond_sample, latent_dim, device, n_traj=64):
    """Probe convergence + posterior collapse.

    Generates samples from BOTH prior and posterior, plus inspects encoder output
    (mu, log_var) to detect posterior collapse. Returns a dict of indicators:
      mean_bps, std_bps : prior-sample synth statistics (esplosion check)
      vol_prior         : avg vol_scalar from prior-sampled z
      vol_post          : avg vol_scalar from posterior-mean z (in-distribution)
      logvar_mean       : posterior log-variance mean (CRITICAL collapse indicator)
      logvar_min        : posterior log-variance min over latent dims
      mu_norm           : avg ||mu(x)||_2 (vs sqrt(latent_dim) = N(0,1) baseline)
      kl_per_dim        : average KL per latent dimension
      dead_dims         : count of latent dims with KL < 0.01 (collapsed)
    """
    model.eval()
    x_t    = torch.from_numpy(x_sample.astype(np.float32)).to(device)
    cond_t = torch.from_numpy(cond_sample.astype(np.float32)).to(device)
    B = len(cond_t)

    # ── Encoder probe ────────────────────────────────────────────────────
    mu, log_var = model.encode(x_t)        # (B, Z)
    mu_norm     = float(mu.norm(dim=1).mean().cpu())
    logvar_mean = float(log_var.mean().cpu())
    logvar_min  = float(log_var.min().cpu())
    # KL per dim: 0.5 * (mu^2 + exp(log_var) - 1 - log_var)
    kl_per_dim_vec = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var)  # (B, Z)
    kl_per_dim     = float(kl_per_dim_vec.mean(dim=0).mean().cpu())
    dead_dims      = int((kl_per_dim_vec.mean(dim=0) < 0.01).sum().cpu())

    # ── Decoder from POSTERIOR mean (in-distribution check) ──────────────
    _, vol_post_out, _ = model.decode(mu, cond_t)
    vol_post = float(vol_post_out[:, 0, :].mean().cpu())

    # ── Decoder from PRIOR (what generate_mvae actually does) ────────────
    cond_exp = cond_t.repeat_interleave(n_traj, dim=0)
    z_prior  = torch.randn(B * n_traj, latent_dim, device=device)
    norm_ret, vol_out, mean_out = model.decode(z_prior, cond_exp)
    vol_scalar = vol_out[:, 0, :].unsqueeze(1)
    synth      = (norm_ret + mean_out) * torch.exp(vol_scalar)
    synth_np   = synth.cpu().numpy()
    mean_bps   = float(np.mean(synth_np)) * 1e4
    std_bps    = float(np.std(synth_np)) * 1e4
    vol_prior  = float(vol_scalar.mean().cpu())

    model.train()
    return {
        "mean_bps":    mean_bps,
        "std_bps":     std_bps,
        "vol_prior":   vol_prior,
        "vol_post":    vol_post,
        "logvar_mean": logvar_mean,
        "logvar_min":  logvar_min,
        "mu_norm":     mu_norm,
        "kl_per_dim":  kl_per_dim,
        "dead_dims":   dead_dims,
    }


def _train_mvae(model, x_train_np, cond_train_np, n_epochs, kl_offset, device, hparams,
                explode_every=0):
    """Train MacroVAE for n_epochs. Returns (model, final_loss, elapsed_sec, diagnostics)."""
    N  = len(x_train_np)
    bs = hparams["batch_size"]

    # ── Preload entire dataset to GPU once — eliminates per-batch CPU→GPU copies ──
    x_all    = torch.from_numpy(x_train_np.astype(np.float32)).to(device, non_blocking=True)
    cond_all = torch.from_numpy(cond_train_np.astype(np.float32)).to(device, non_blocking=True)

    # ── Precompute per-window normalization targets (GPU, once) ──────────────────
    with torch.no_grad():
        x_std_all       = x_all.std(dim=1, keepdim=True).clamp(min=1e-6)  # (N, 1, C)
        x_norm_all      = x_all / x_std_all
        x_norm_mean_all = x_norm_all.mean(dim=1, keepdim=True)             # (N, 1, C)
        x_demeaned_all  = x_norm_all - x_norm_mean_all                     # (N, T, C)
        log_x_std_all   = x_std_all.log()                                  # (N, 1, C)

    opt = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    model.train()

    # Pre-select a small (x, cond) sample for diagnostics
    _diag_n = min(32, len(cond_train_np))
    diag_x    = x_train_np[:_diag_n]
    diag_cond = cond_train_np[:_diag_n]

    t0 = _time.perf_counter()
    final_loss = float("inf")
    diagnostics = []  # list of dicts (per --explode tick)

    for epoch in range(n_epochs):
        eff_epoch = kl_offset + epoch
        kl_w = min(hparams["kl_max_weight"],
                   (eff_epoch + 1) / hparams["kl_anneal_eps"])
        epoch_loss, n_batches = 0.0, 0

        perm = torch.randperm(N, device=device)
        for start in range(0, N - bs + 1, bs):          # drop_last=True equivalent
            idx    = perm[start:start + bs]
            x_b    = x_all[idx]
            cond_b = cond_all[idx] + torch.rand_like(cond_all[idx])   # dithering augment

            # Precomputed targets — no recomputation inside forward
            tgt_demeaned = x_demeaned_all[idx]
            tgt_log_std  = log_x_std_all[idx]
            tgt_mean     = x_norm_mean_all[idx]

            opt.zero_grad(set_to_none=True)
            recon, vol, mean_out, mu, lv = model(x_b, cond_b)

            rec_loss  = F.mse_loss(recon, tgt_demeaned)
            vol_loss  = F.mse_loss(vol, tgt_log_std)
            mean_loss = F.mse_loss(mean_out, tgt_mean) * 5
            kl_loss   = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = rec_loss + vol_loss + mean_loss + kl_w * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)

        # --explode diagnostic: periodic posterior-collapse + sample probes
        if explode_every > 0 and (epoch + 1) % explode_every == 0:
            d = _explode_diagnostic(model, diag_x, diag_cond,
                                    hparams["latent_dim"], device)
            d["epoch"] = epoch + 1
            d["loss"]  = final_loss
            diagnostics.append(d)
            # Posterior-collapse FLAG: log_var << 0 (target ~ -2 to 0) and
            # vol_prior far from vol_post (decoder OOD on prior samples)
            _flag_collapse = "[COLLAPSE]" if d["logvar_mean"] < -3.0 else "[OK]    "
            log.info(
                "    [EXPLODE] ep %4d  loss=%.4f  %s  "
                "lv_mean=%+.2f  lv_min=%+.2f  mu_l2=%.2f  KL/dim=%.3f  "
                "dead=%d/%d  vol_prior=%+.2f  vol_post=%+.2f  "
                "synth: Mean=%+.1fbps Std=%.0fbps",
                epoch + 1, final_loss, _flag_collapse,
                d["logvar_mean"], d["logvar_min"], d["mu_norm"],
                d["kl_per_dim"], d["dead_dims"], hparams["latent_dim"],
                d["vol_prior"], d["vol_post"],
                d["mean_bps"], d["std_bps"],
            )
        # Regular progress logging — every 10 epochs (global default)
        elif (epoch + 1) % 10 == 0:
            log.info("    epoch %5d/%d  loss=%.4f  kl_w=%.5f  [%.0fs]",
                     epoch + 1, n_epochs, final_loss, kl_w, _time.perf_counter() - t0)

    # Print convergence + posterior-collapse summary if diagnostics were collected
    if diagnostics:
        log.info("    POSTERIOR-COLLAPSE WATCH (CRITICAL: lv_mean should drift toward 0,")
        log.info("    NOT stay << -3. dead dims should stay 0. vol_prior/vol_post should match.)")
        log.info("    %5s %8s %9s %8s %8s %7s %5s %9s %9s %10s %9s",
                 "epoch","loss","lv_mean","lv_min","mu_l2","KL/dim","dead",
                 "vol_pri","vol_post","Mean(bps)","Std(bps)")
        for d in diagnostics:
            log.info("    %5d %8.4f %+9.2f %+8.2f %8.2f %7.3f %2d/%2d %+9.2f %+9.2f %+10.1f %9.0f",
                     d["epoch"], d["loss"],
                     d["logvar_mean"], d["logvar_min"],
                     d["mu_norm"], d["kl_per_dim"],
                     d["dead_dims"], hparams["latent_dim"],
                     d["vol_prior"], d["vol_post"],
                     d["mean_bps"], d["std_bps"])
        log.info("    Target (paper-like, no collapse):")
        log.info("      lv_mean ≈ [-2, 0]    (NOT < -3)")
        log.info("      dead_dims = 0        (all latents active)")
        log.info("      vol_prior ≈ vol_post (within ±0.3)")
        log.info("      Mean ≈ +2 bps        Std ≈ 100-150 bps")

    model.eval()
    return model, final_loss, _time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_schedule(initial_split_date, evaluation_end_date, step_freq, window_end_ts):
    freq = step_freq.upper()
    freq_map = {"Q": "QS", "3M": "QS", "6M": "6MS", "H": "6MS", "A": "YS", "Y": "YS"}
    pd_freq = freq_map.get(freq, "QS")

    def _step_label(dt):
        if freq in ("Q", "3M"):
            return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
        elif freq in ("6M", "H"):
            return f"{dt.year}H{1 if dt.month <= 6 else 2}"
        return str(dt.year)

    def _step_end(dt):
        if freq in ("Q", "3M"):
            q = (dt.month - 1) // 3 + 1
            return pd.Timestamp(year=dt.year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
        elif freq in ("6M", "H"):
            q = 2 if dt.month <= 6 else 4
            return pd.Timestamp(year=dt.year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
        return pd.Timestamp(f"{dt.year}-12-31")

    period_starts = pd.date_range(start=initial_split_date, end=evaluation_end_date, freq=pd_freq)
    period_starts = period_starts[period_starts > initial_split_date]

    steps = [("initial", initial_split_date, None)]
    prev_label = "initial"
    for pstart in period_starts:
        label = _step_label(pstart)
        cutoff = min(_step_end(pstart), evaluation_end_date)
        steps.append((label, cutoff, prev_label))
        if cutoff >= evaluation_end_date:
            break
        prev_label = label

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── Project root & config ────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _mvae_path = str(project_root / "MacroVAE")
    if _mvae_path not in sys.path:
        sys.path.insert(0, _mvae_path)

    import yaml
    from src.macrovae_compat import MacroVAE

    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Hyperparameters — all from CLI args (see parse_args) ─────────────────
    hparams = {
        "hidden_dim":     args.hidden_dim,
        "latent_dim":     args.latent_dim,
        "cond_embed_dim": args.cond_embed_dim,
        "lr":             args.lr,
        "batch_size":     args.batch,
        "kl_max_weight":  args.kl_weight,
        "kl_anneal_eps":  args.kl_anneal_eps,
        "epochs_init":    args.epochs,
        "epochs_warm":    args.wf_epochs,
    }

    # ── Directories (paths from config) ──────────────────────────────────
    from src.paths import resolve_paths
    _paths = resolve_paths(project_root, config)
    data_dir = _paths.processed_dir
    ckpt_dir = _paths.macrovae_ckpt_dir

    if args.restart and ckpt_dir.exists():
        # Never wipe through a link (e.g. a junction to archived weights).
        if Path(project_root).resolve() not in ckpt_dir.resolve().parents:
            raise SystemExit(f"--restart refused: {ckpt_dir} resolves to "
                             f"{ckpt_dir.resolve()}, outside {Path(project_root).resolve()}")
        shutil.rmtree(ckpt_dir)
        log.info("RESTART: wiped %s", ckpt_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessing artifacts ────────────────────────────────────────
    log.info("Loading preprocessing artifacts...")
    raw_windows = np.load(data_dir / "raw_windows.npy")
    norm_params = np.load(data_dir / "normalization_params.npy", allow_pickle=True).item()

    # Per-step (Fix B) MacroVAE z-scored macro: each WF cutoff has its own
    # in-sample stats saved by preprocess.py under mvae_perstep/.
    _mvae_perstep_dir = data_dir / "mvae_perstep"
    assert _mvae_perstep_dir.exists(), (
        f"mvae_perstep/ not found in {data_dir}.\n"
        "Run scripts/preprocess.py to generate per-step in-sample z-score files."
    )
    # Pre-load a sample to know shape/D (use 'initial' which always exists).
    _sample = np.load(_mvae_perstep_dir / "macro_zscore_initial.npy")
    log.info("Loaded MacroVAE per-step z-scored macro (initial): %s", _sample.shape)

    context_len = int(norm_params["context_len"])
    target_len = int(norm_params["target_len"])
    tickers = list(norm_params.get("tickers", config["data_ingestion"]["tickers"]))
    C = raw_windows.shape[2]
    D = _sample.shape[2]
    tickers = tickers[:C]

    _raw_dates = np.load(data_dir / "window_dates.npy", allow_pickle=True)
    window_end_ts = pd.to_datetime(_raw_dates)

    wf_cfg = config["training"]["walk_forward"]
    initial_split_date = pd.Timestamp(wf_cfg["initial_split_date"])
    evaluation_end_date = pd.Timestamp(wf_cfg["evaluation_end_date"])
    step_freq = str(wf_cfg.get("step_freq", "Q"))

    log.info("raw_windows: %s  (N, %d+%d, C=%d)", raw_windows.shape, context_len, target_len, C)
    log.info("macro_windows (per-step initial): %s  (N, M=%d, D=%d)",
             _sample.shape, _sample.shape[1], D)
    log.info("Initial split: %s, Eval end: %s", initial_split_date.date(), evaluation_end_date.date())

    # KL annealing: Kubiak formula ramps 0→1 over kl_anneal_eps, clamped at kl_max.
    # With kl_anneal_eps=2000, kl_max=0.01 → max reached at epoch 20 (independent of epochs_init).
    _kl_max_epoch = int(hparams["kl_max_weight"] * hparams["kl_anneal_eps"])
    log.info("KL annealing: max weight %.4f reached at epoch %d (of %d init)",
             hparams["kl_max_weight"], _kl_max_epoch, hparams["epochs_init"])

    # Verify architecture
    assert hparams["hidden_dim"] % 8 == 0
    _tmp = MacroVAE(input_dim=C, sequence_length=target_len,
                    latent_dim=hparams["latent_dim"], hidden_dim=hparams["hidden_dim"],
                    condition_dim=D, condition_embedding_dim=hparams["cond_embed_dim"])
    n_params = sum(p.numel() for p in _tmp.parameters())
    del _tmp
    log.info("MacroVAE(%dh, %dz, embed=%d, D=%d) → %s params",
             hparams["hidden_dim"], hparams["latent_dim"], hparams["cond_embed_dim"],
             D, f"{n_params:,}")

    # ── Build training schedule ──────────────────────────────────────────
    training_steps = build_training_schedule(
        initial_split_date, evaluation_end_date, step_freq, window_end_ts)

    log.info("Training schedule (%d steps, freq=%s):", len(training_steps), step_freq)
    for lbl, cutoff, prev in training_steps:
        mask = window_end_ts <= cutoff
        log.info("  %-12s | cutoff=%s | prev=%-12s | n_train=%5d",
                 lbl, cutoff.date(), prev or "---", int(mask.sum()))

    # ═══════════════════════════════════════════════════════════════════════
    # Walk-Forward Training Loop
    # ═══════════════════════════════════════════════════════════════════════
    log.info("=" * 72)
    log.info("MacroVAE WALK-FORWARD TRAINING")
    log.info("=" * 72)

    _total_t0 = _time.perf_counter()
    _skipped = 0
    _trained = 0

    for step_idx, (step_label, cutoff, prev_label) in enumerate(training_steps):
        ckpt_path = ckpt_dir / f"ckpt_mvae_{step_label}.pt"

        if ckpt_path.exists():
            log.info("[%-12s] checkpoint exists — skipping", step_label)
            _skipped += 1
            continue

        # Build training data — load the per-step z-scored macro matching this cutoff.
        _step_zscore_path = _mvae_perstep_dir / f"macro_zscore_{step_label}.npy"
        if not _step_zscore_path.exists():
            log.error("Missing per-step file %s — re-run preprocess.py", _step_zscore_path.name)
            raise FileNotFoundError(_step_zscore_path)
        macro_windows_step = np.load(_step_zscore_path)

        train_mask = window_end_ts <= cutoff
        n_train = int(train_mask.sum())
        x_train = raw_windows[train_mask, context_len:, :]    # (N, target_len, C)
        cond_train = macro_windows_step[train_mask, -1, :]    # (N, D)
        log.info("  Using per-step macro_zscore_%s.npy (in-sample stats ≤ %s)",
                 step_label, cutoff.date())

        # Build or load model
        model = MacroVAE(
            input_dim=C, sequence_length=target_len,
            latent_dim=hparams["latent_dim"], hidden_dim=hparams["hidden_dim"],
            condition_dim=D, condition_embedding_dim=hparams["cond_embed_dim"],
        ).to(device)

        if prev_label is not None:
            prev_ckpt = ckpt_dir / f"ckpt_mvae_{prev_label}.pt"
            if not prev_ckpt.exists():
                log.warning("[%-12s] prev checkpoint %s not found — training from scratch",
                            step_label, prev_ckpt.name)
                n_epochs = hparams["epochs_init"]
                kl_off = 0
            else:
                prev_state = torch.load(prev_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(prev_state["model_state_dict"])
                n_epochs = hparams["epochs_warm"]
                kl_off = hparams["epochs_init"]
                log.info("[%-12s] warm-start from %s", step_label, prev_ckpt.name)
        else:
            n_epochs = hparams["epochs_init"]
            kl_off = 0
            log.info("[%-12s] Phase 0 — training from scratch", step_label)

        log.info("  n_train=%d, epochs=%d, cutoff=%s", n_train, n_epochs, cutoff.date())

        # Train
        model, final_loss, elapsed = _train_mvae(
            model, x_train, cond_train, n_epochs, kl_offset=kl_off,
            device=device, hparams=hparams, explode_every=args.explode)

        # Save checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "step_label": step_label,
            "epoch": n_epochs,
            "final_loss": final_loss,
            "n_train": n_train,
            "cutoff_date": str(cutoff.date()),
            "hyperparameters": {
                "hidden_dim": hparams["hidden_dim"],
                "latent_dim": hparams["latent_dim"],
                "condition_embedding_dim": hparams["cond_embed_dim"],
                "condition_dim": D,
                "input_dim": C,
                "sequence_length": target_len,
                "lr": hparams["lr"],
                "kl_max_weight": hparams["kl_max_weight"],
                "kl_anneal_eps": hparams["kl_anneal_eps"],
                "epochs_init": hparams["epochs_init"],
                "epochs_warm": hparams["epochs_warm"],
            },
        }, ckpt_path)

        _trained += 1
        log.info("  → saved %s  loss=%.4f  [%.0fs]", ckpt_path.name, final_loss, elapsed)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = _time.perf_counter() - _total_t0
    log.info("=" * 72)
    log.info("Done. Trained=%d, Skipped=%d, Total time=%.0fs (%.1f min)",
             _trained, _skipped, total_elapsed, total_elapsed / 60)
    log.info("Checkpoints in: %s", ckpt_dir)

    # Verify all checkpoints exist
    missing = [lbl for lbl, _, _ in training_steps
               if not (ckpt_dir / f"ckpt_mvae_{lbl}.pt").exists()]
    if missing:
        log.warning("Missing checkpoints: %s", missing)
    else:
        log.info("All %d checkpoints verified.", len(training_steps))


if __name__ == "__main__":
    main()
