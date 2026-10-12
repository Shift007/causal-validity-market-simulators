#!/usr/bin/env python
"""
lab.py
======
Diffusion model training + walk-forward orchestration. Hosts the model
(U-Net with a FiLM-injected macro encoder), the DDPM/DDIM
sampler with SGD support, and the walk-forward training driver. OOS
generation/evaluation lives in ``src/generation/*``; this module is the
training entry point invoked by ``scripts/train.py``.

Walk-forward protocol
---------------------
    Phase 0 : train from scratch on all data up to split_date → ckpt_wf_initial.pt
    Per step Q: warm-start from ckpt(Q-1) and fine-tune on data ≤ Q_end →
                ckpt_wf_{Q}.pt

Architecture
------------
    SymmetricMacroEncoder: linear sign-aware backbone + sigmoid volatility gate
        (guarantees antisymmetric response to ±shocks; macro-only conditioning).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Project root (this file lives at <root>/src/diffusion/lab.py) ───────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Live training-progress instrumentation ──────────────────────────────────
# Initialised by wf_train() at the start of a run, then ticked once per epoch in
# BOTH the Phase-0 loop (train()) and the per-step walk-forward loop. Each tick
# (a) prints a flushed one-line ETA and (b) atomically rewrites
# <ckpt_dir>/train_progress.json, so remaining time / progress is always fresh
# and readable at any instant (no waiting for the every-10-epochs prints). When
# _PROG is None (e.g. train() called standalone) the tick is a no-op.
_PROG = None  # dict | None


def _fmt_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def _progress_init(path, total_ep) -> None:
    global _PROG
    _PROG = {"path": Path(path), "t0": time.time(),
             "total_ep": int(total_ep), "done_ep": 0}


def _progress_add_done(n: int) -> None:
    """Account for skipped phases (resumed checkpoints) without per-epoch ticks."""
    if _PROG is not None:
        _PROG["done_ep"] += int(n)


def _progress_tick(phase: str, ep_in_phase: int, ep_in_phase_total: int, loss=None) -> None:
    g = _PROG
    if g is None:
        return
    import json
    import os as _os
    g["done_ep"] += 1
    done, total = g["done_ep"], g["total_ep"]
    el = time.time() - g["t0"]
    rate = el / max(1, done)
    eta = rate * max(0, total - done)
    pct = 100.0 * done / max(1, total)
    print(f"[progress] {phase}  ep {ep_in_phase}/{ep_in_phase_total}  "
          f"global {done}/{total} ({pct:4.1f}%)  elapsed {_fmt_hms(el)}  "
          f"ETA {_fmt_hms(eta)}  ({rate:.2f}s/ep)"
          + (f"  loss={loss:.4f}" if loss is not None else ""), flush=True)
    rec = {"phase": phase, "ep_in_phase": ep_in_phase,
           "ep_in_phase_total": ep_in_phase_total, "global_done": done,
           "global_total": total, "pct": round(pct, 2), "elapsed_s": round(el, 1),
           "eta_s": round(eta, 1), "elapsed_hms": _fmt_hms(el),
           "eta_hms": _fmt_hms(eta), "sec_per_ep": round(rate, 3),
           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if loss is not None:
        rec["loss"] = round(float(loss), 5)
    try:
        p = g["path"]
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        _os.replace(tmp, p)
    except Exception:
        pass


def _get_cfg_path() -> Path:
    """Return the active config path.

    If the env variable ``DIFFUSION_CONFIG_PATH`` is set (e.g. by
    ``scripts/train.py --config <config>``), that path is used.
    Otherwise falls back to ``<project_root>/config.yaml``.
    """
    import os as _os
    _env = _os.environ.get("DIFFUSION_CONFIG_PATH")
    return Path(_env) if _env else ROOT / "config.yaml"

# ── Project imports (preprocessing helpers only — no tight coupling) ─────────
from src.preprocessing.instance_dataset import (
    batch_quantile_transform,
    batch_inverse_quantile_transform,
)
# ── Fast vectorised SWT/iSWT (CPU bottleneck reducer) ───────────────────────
# pywt.swt / pywt.iswt accept ND arrays with `axis=-1` and run the transform
# on every leading "row" in a single C call.  Reshaping (B,C,W) → (B*C, W)
# turns the original O(B*C) python loop in src/preprocessing/instance_dataset.py
# into one pywt invocation:
#     forward : 1.23s → 0.06s   (~22× on B=4000, C=14, level=4)
#     inverse : 10.5s → 0.32s   (~33×)
# Bit-exact match with the serial pywt loop.  Threading was tried and made it
# WORSE (pywt holds the GIL through the python wrapping), so we just lean on
# the internal vectorisation.
import pywt as _pywt


def _fast_swt_forward(signal: "torch.Tensor", wavelet_type: str, wavelet_level: int) -> "torch.Tensor":
    """Batch SWT via pywt's axis-vectorised C path.  signal=(B,C,W) → wav=(B,C,n_levels,W)."""
    B, C, W = signal.shape
    n_levels = wavelet_level + 1
    sig_flat = signal.detach().cpu().numpy().reshape(B * C, W)
    coeffs = _pywt.swt(sig_flat, wavelet_type, level=wavelet_level,
                       trim_approx=False, axis=-1)
    # coeffs: list of (cA_k, cD_k) ordered deep→shallow, n_levels=wavelet_level entries
    cA = coeffs[0][0]                                       # (B*C, W) deepest approx
    details = [p[1] for p in coeffs]                        # n_levels-1 details
    wav = np.stack([cA] + details, axis=1).astype(np.float32, copy=False)   # (B*C, n_levels, W)
    return torch.from_numpy(wav.reshape(B, C, n_levels, W))


def _fast_iswt(wavelet: "torch.Tensor", wavelet_type: str) -> "torch.Tensor":
    """Batch inverse SWT via pywt's axis-vectorised C path."""
    B, C, n_levels, W = wavelet.shape
    wav_flat = wavelet.detach().cpu().numpy().reshape(B * C, n_levels, W)
    clist = [(wav_flat[:, 0], wav_flat[:, 1])]
    for k in range(2, n_levels):
        clist.append((None, wav_flat[:, k]))
    sig = _pywt.iswt(clist, wavelet_type, axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(sig.reshape(B, C, W))

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Diffusion walk-forward training")
    p.add_argument("--epochs", type=int, default=200,
                   help="Phase 0 training epochs (default 200).")
    p.add_argument("--wf_epochs", type=int, default=20,
                   help="Per-step fine-tuning epochs (default 20).")
    p.add_argument("--batch", type=int, default=512,
                   help="Mini-batch size (default 512). Pre-computed tensors live on "
                        "device, so memory cost is batch * (C, n_levels, W) ≈ a few MB.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_ch", type=int, default=32,
                   help="U-Net base channel width (default 32 ~1.4M params). "
                        "cond_dim/time_dim scale as 2*base_ch. 48 ~3.2M, 64 ~5.6M. "
                        "Saved in checkpoint metadata for the generation path.")
    p.add_argument("--wf_lr", type=float, default=None,
                   help="LR for walk-forward fine-tuning. Default=same as --lr.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default="diffusion_ckpt.pt",
                   help="Initial checkpoint filename. Combined with --ckpt_dir.")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/synthetic/diffusion",
                   help="Directory where checkpoints are written/read. "
                        "Default tracks paths.checkpoints_dir/diffusion from config.yaml.")
    # Walk-forward schedule bounds (overridable; default = read from config.yaml).
    p.add_argument("--eval_end_date", type=str, default=None,
                   help="End date for the OOS evaluation window (YYYY-MM-DD). "
                        "Default: read from config.yaml training.walk_forward.evaluation_end_date.")
    p.add_argument("--step_freq", type=str, default=None,
                   help="Walk-forward step frequency: Q (quarterly) or A (annual). "
                        "Default: read from config.yaml training.walk_forward.step_freq.")
    p.add_argument("--macro_encoder", type=str, default="symmetric",
                   choices=["symmetric", "snapshot"],
                   help="Macro encoder type. 'symmetric' (default) uses the time-series "
                        "encoder over (D_base, macro_len); 'snapshot' (ablation) uses a "
                        "linear projection over (D_ext,) = base snapshot + 6m momentum, "
                        "identical to MacroVAE input semantics.")
    p.add_argument("--cond_drop_prob", type=float, default=0.0,
                   help="Probability of replacing cond_emb with the learnable null "
                        "embedding during a training step (classifier-free guidance "
                        "training). 0.0 = pure conditional (default). 0.10-0.15 enables "
                        "proper CFG at sampling time.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay applied to non-macro_enc params (default 1e-4).")
    p.add_argument("--wd_macro", type=float, default=None,
                   help="Weight decay applied ONLY to macro_enc params. When set "
                        "and different from --weight_decay, the AdamW optimiser uses "
                        "two param groups for stratified regularisation. Macro encoder "
                        "is ~0.5%% of params but the sole macro→cond_emb path, hence "
                        "under-regularised by uniform WD. Recommended: 1e-2.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 2.  U-Net
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        s = self.scale(cond)[:, :, None, None]
        b = self.shift(cond)[:, :, None, None]
        return x * (s + 1) + b


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, cond_dim=None, groups=4, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.film = FiLM(dim_out, cond_dim) if cond_dim else None
        self.b1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out), nn.SiLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.b2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out), nn.SiLU(),
        )
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        h = self.b1(x)
        scale, shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (scale + 1) + shift
        if self.film and cond is not None:
            h = self.film(h, cond)
        h = self.b2(self.drop(h))
        return h + self.res(x)



# ── Macro encoders ───────────────────────────────────────────────────────────

class SymmetricMacroEncoder(nn.Module):
    """Linear sign-aware macro encoder with a volatility gate (the Temporal encoder).

    Sign-aware path (linear in the macro window, no activation):
      stats      = [last_val, recent_mean, window_mean, early_mean, mid_mean]   (5 x D)
      cond_aware = Linear(5D, cond_dim)

    Volatility gate (the only non-linearity):
      vstats = [window_std, vol_change]                                          (2 x D)
      gate   = sigmoid(Linear(2D, cond_dim))                                     in [0, 1]

    Output: cond_emb = cond_aware * gate

    Response to a shock
    -------------------
    cond_aware is linear, so opposite shifts of the input give exactly opposite
    shifts of cond_aware. The gate reads dispersion statistics. A shift added to
    every timestep leaves them unchanged, and the embedding then responds
    antisymmetrically. A shift added to the most recent value only (the
    "boundary" geometry used for the interventions of the paper) changes
    window_std and vol_change by an amount that depends on where that value sits
    relative to the rest of the window, so the gate gives the encoder a
    state-dependent component that is not antisymmetric.

    Parameters: 7 * macro_dim * cond_dim weights, plus biases and the null embedding.

    Input:  (B, D, macro_len), stationarised and tanh-bounded, oldest to newest
    Output: (B, cond_dim)
    """

    def __init__(self, macro_dim: int, macro_len: int, cond_dim: int,
                 recent_window: int = 10):
        super().__init__()
        self.macro_dim = macro_dim
        self.macro_len = macro_len
        self.cond_dim  = cond_dim
        self.recent_window = recent_window

        # Sign-aware: 5 linear stats per variable → linear projection
        self.proj_aware = nn.Linear(macro_dim * 5, cond_dim)

        # Sign-invariant volatility gate: 2 variance-based stats per variable
        self.proj_gate = nn.Linear(macro_dim * 2, cond_dim)

        # Learnable null/uncond embedding for classifier-free guidance.
        # zeros_like(macro) communicates "all macro at z-scored mean", a SPECIFIC
        # state, not "no information". A learnable parameter in cond_emb space
        # is the principled way: the model adjusts it during training to be the
        # marginal-score reference. Initialised to zeros.
        self.null_cond_emb = nn.Parameter(torch.zeros(cond_dim))

    def forward(self, x: torch.Tensor, use_null: bool = False) -> torch.Tensor:
        # x: (B, D, macro_len) — oldest first, newest last
        # use_null: if True, return the learnable null embedding for every sample
        if use_null:
            B = x.shape[0]
            return self.null_cond_emb.unsqueeze(0).expand(B, -1)
        rw = self.recent_window
        L  = x.shape[-1]

        # ── Sign-aware stats (linear in x) ───────────────────────────────────
        last_val    = x[:, :, -1]                          # (B, D)
        recent_mean = x[:, :, -rw:].mean(dim=-1)            # (B, D)
        window_mean = x.mean(dim=-1)                        # (B, D)
        early_mean  = x[:, :, :rw].mean(dim=-1)             # (B, D)
        mid_start   = max(0, (L - rw) // 2)
        mid_mean    = x[:, :, mid_start:mid_start + rw].mean(dim=-1)  # (B, D)

        aware_stats = torch.cat(
            [last_val, recent_mean, window_mean, early_mean, mid_mean],
            dim=-1,
        )                                                    # (B, 5*D)
        cond_aware = self.proj_aware(aware_stats)            # (B, cond_dim)

        # ── Volatility gate (dispersion statistics) ──────────────────────────
        window_std = x.std(dim=-1)                           # (B, D)
        recent_std = x[:, :, -rw:].std(dim=-1)
        early_std  = x[:, :, :rw].std(dim=-1)
        vol_change = recent_std - early_std                  # (B, D)

        vol_stats = torch.cat([window_std, vol_change], dim=-1)   # (B, 2*D)
        gate = torch.sigmoid(self.proj_gate(vol_stats))           # (B, cond_dim) in [0,1]

        return cond_aware * gate


class SnapshotOnlyMacroEncoder(nn.Module):
    """ABLATION encoder: single-value macro (NO time series).

    Companion to ``SymmetricMacroEncoder`` for the ablation experiment that
    isolates the contribution of TEMPORAL macro information to the diffusion's
    counterfactual response.  Whereas ``SymmetricMacroEncoder`` ingests a
    (D, macro_len) time series and extracts 5 temporal stats, this encoder
    ingests a single-value vector identical to what MacroVAE sees:

        c = [base_T,  momentum_6m_T]   ∈ R^{D_ext = 2·D_base}

    where ``base_T`` is the macro snapshot at day-before-target-start and
    ``momentum_6m_T = base_T − base_{T-126}`` is the 6-month log-diff.

    Architecture
    ------------
    Linear projection only (no activation, no volatility gate), matching the
    sign-aware philosophy of ``SymmetricMacroEncoder``:

        cond_emb = Linear(D_ext, cond_dim)(c)

    Why no activation
        Preserves the antisymmetry guarantee:
            cond_emb(c + δ) − cond_emb(c) = J · δ
        so cf shocks ±δ produce ±anti-parallel cond_emb shifts.

    Why no volatility gate
        The gate in ``SymmetricMacroEncoder`` uses temporal stats (``window_std``,
        ``recent_std``, ``early_std``) which are undefined on a single timestep.
        Forcing a fake gate would break the ablation: we want to isolate the
        absence of temporal information, not introduce a different mechanism.

    Comparability with ``SymmetricMacroEncoder``
    --------------------------------------------
    - Same output ``cond_dim`` → identical downstream FiLM
    - Same sign-symmetry guarantee
    - Smaller param count (≈ D_ext · cond_dim vs ≈ 5·D · cond_dim + gate),
      but this is the WHOLE POINT: less capacity to compensate for less info.

    Input:  (B, D_ext)             — base snapshot + momentum, z-scored
    Output: (B, cond_dim)
    """

    def __init__(self, macro_dim_ext: int, cond_dim: int):
        super().__init__()
        self.macro_dim_ext = macro_dim_ext
        self.cond_dim = cond_dim
        self.proj = nn.Linear(macro_dim_ext, cond_dim, bias=True)
        # Learnable null/uncond embedding (see SymmetricMacroEncoder for rationale).
        self.null_cond_emb = nn.Parameter(torch.zeros(cond_dim))

    def forward(self, x: torch.Tensor, use_null: bool = False) -> torch.Tensor:
        # x: (B, D_ext)  — pre-standardised by the MVAE in-sample z-score pipeline
        if use_null:
            B = x.shape[0]
            return self.null_cond_emb.unsqueeze(0).expand(B, -1)
        assert x.dim() == 2, (
            f"SnapshotOnlyMacroEncoder expects 2D input (B, D_ext); got {tuple(x.shape)}"
        )
        return self.proj(x)


class UNet(nn.Module):
    """
    Conditional U-Net for wavelet-domain diffusion. base_ch=32, 3 down/up levels.
    Input:  (B, C, H, W)  where H=n_levels=5, W=target_len=32, C=n_assets=14
    Output: (B, C, H, W)  predicted noise

    Conditioning: macro encoder only (SymmetricMacroEncoder, or
    SnapshotOnlyMacroEncoder in the snapshot ablation); no asset-past encoder.
    Matches MacroVAE baseline conditioning design.
    """

    def __init__(
        self,
        in_channels: int = 14,      # n_assets
        n_levels: int = 5,          # wavelet levels + 1
        base_ch: int = 32,
        time_dim: int = 64,
        macro_dim: int = 14,        # n_macro_vars (D for symmetric, D_ext for snapshot)
        macro_len: int = 126,       # macro_context_len; ignored for snapshot encoder
        ctx_len: int = 64,          # asset context length
        cond_dim: int = 64,         # unified conditioning output dim
        dropout: float = 0.1,
        macro_encoder_type: str = "symmetric",  # "symmetric" | "snapshot" (ablation)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.macro_encoder_type = macro_encoder_type

        # ── Time embedding ──────────────────────────────────────────────
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Macro encoder (symmetric default, snapshot for ablation) ────
        # symmetric: linear sign-aware + sign-invariant volatility gate over
        #     a (D, macro_len) time series. Guarantees antisymmetric cond_emb
        #     response to ±δ shocks.
        # snapshot: linear-only over a (D_ext,) snapshot+momentum vector.
        #     Identical input semantics to MacroVAE; used to isolate the
        #     contribution of temporal info via ablation.
        if macro_encoder_type == "symmetric":
            self.macro_enc = SymmetricMacroEncoder(macro_dim, macro_len, cond_dim)
        elif macro_encoder_type == "snapshot":
            # macro_dim here is interpreted as D_ext (= 2 * D_base)
            self.macro_enc = SnapshotOnlyMacroEncoder(macro_dim, cond_dim)
        else:
            raise ValueError(
                f"macro_encoder_type must be 'symmetric' or 'snapshot', "
                f"got {macro_encoder_type!r}"
            )
        total_cond_dim = cond_dim

        # ── U-Net backbone ───────────────────────────────────────────────
        dims = [base_ch, base_ch * 2, base_ch * 4]  # 32, 64, 128
        self.init_conv = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # Down path
        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlock(dims[i], dims[i], time_dim, total_cond_dim, dropout=dropout),
                ResBlock(dims[i], dims[i], time_dim, total_cond_dim, dropout=dropout),
                nn.Conv2d(dims[i], dims[i + 1], 3, stride=(1, 2), padding=1),
            ])
            for i in range(len(dims) - 1)
        ])

        # Bottleneck
        self.mid1 = ResBlock(dims[-1], dims[-1], time_dim, total_cond_dim, dropout=dropout)
        self.mid2 = ResBlock(dims[-1], dims[-1], time_dim, total_cond_dim, dropout=dropout)

        # Up path
        rdims = list(reversed(dims))
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(rdims[i], rdims[i + 1], 3,
                                   stride=(1, 2), padding=1, output_padding=(0, 1)),
                ResBlock(rdims[i + 1] * 2, rdims[i + 1], time_dim, total_cond_dim, dropout=dropout),
                ResBlock(rdims[i + 1], rdims[i + 1], time_dim, total_cond_dim, dropout=dropout),
            ])
            for i in range(len(rdims) - 1)
        ])

        self.final = nn.Sequential(
            ResBlock(dims[0], dims[0], time_dim, total_cond_dim, dropout=dropout),
            nn.Conv2d(dims[0], in_channels, 1),
        )

    def _encode_cond(self, macro_cond, use_null: bool = False):
        """Encode macro conditioning -> (B, cond_dim).

        use_null : if True, returns the learnable null/uncond embedding
        (one per encoder), used for CFG reference and condition dropout
        in training.
        """
        return self.macro_enc(macro_cond, use_null=use_null)

    def forward(self, x, t, macro_cond=None, cond=None):
        """
        x          : (B, C, H, W)  noisy wavelet image
        t          : (B,)           diffusion timestep
        macro_cond : (B, D, macro_len)   if macro_encoder_type='symmetric'
                     (B, D_ext)          if macro_encoder_type='snapshot' (ablation)
        cond       : pre-computed (B, cond_dim) — skips encoding if provided
        """
        t_emb = self.time_emb(t)             # (B, time_dim)

        if cond is None and macro_cond is not None:
            cond = self._encode_cond(macro_cond)

        h = self.init_conv(x)                # (B, base_ch, H, W)
        skips = []

        for res1, res2, down in self.downs:
            h = res1(h, t_emb, cond)
            h = res2(h, t_emb, cond)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb, cond)
        h = self.mid2(h, t_emb, cond)

        for up, res1, res2 in self.ups:
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = res1(h, t_emb, cond)
            h = res2(h, t_emb, cond)

        return self.final[1](self.final[0](h, t_emb, cond))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Diffusion (DDPM + DDIM)
# ═══════════════════════════════════════════════════════════════════════════

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.9999)


class Diffusion:
    def __init__(self, model, timesteps=500, device="cpu", cond_drop_prob=0.0):
        self.model = model
        self.T = timesteps
        self.device = device
        # Classifier-free guidance training: probability of replacing the
        # cond embedding with the learnable null embedding during a training
        # step. Default 0 = pure conditional training. Set to ~0.1-0.15 to
        # enable CFG-correct sampling (the null branch becomes meaningfully
        # learned).
        self.cond_drop_prob = float(cond_drop_prob)

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_cp = torch.cumprod(alphas, dim=0)
        alpha_cp_prev = F.pad(alpha_cp[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.sqrt_acp = alpha_cp.sqrt()
        self.sqrt_one_m_acp = (1 - alpha_cp).sqrt()
        self.alpha_cp = alpha_cp
        self.alpha_cp_prev = alpha_cp_prev

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        sm = self.sqrt_one_m_acp[t].view(-1, 1, 1, 1)
        return s * x0 + sm * noise, noise

    def compute_loss(self, x0, macro_cond):
        """DDPM noise-prediction loss, MSE(eps_pred, eps).

        The macro condition enters only through FiLM inside the network.
        Returns the loss and a dict with its float value.
        """
        cond_emb = self.model._encode_cond(macro_cond)
        B = x0.shape[0]

        # Condition dropout for classifier-free guidance training: with
        # probability cond_drop_prob a sample's embedding is replaced by the
        # learnable null embedding, so the unconditional score is learned too.
        if self.cond_drop_prob > 0.0:
            null_emb = self.model._encode_cond(macro_cond, use_null=True)
            drop_mask = (torch.rand(B, device=self.device) < self.cond_drop_prob)
            mask_b = drop_mask.view(B, 1).to(cond_emb.dtype)
            cond_emb = mask_b * null_emb + (1.0 - mask_b) * cond_emb

        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        loss = F.mse_loss(self.model(x_t, t, cond=cond_emb), noise)
        return loss, {"loss_eps": float(loss.detach().item())}

    @torch.no_grad()
    def ddim_sample(self, shape, macro_cond,
                    steps=100, eta=0.5, guidance_scale=1.0, x_init=None,
                    sgd_base_cond=None, sgd_gamma=1.0):
        """DDIM sampling with unified guidance.

        CFG (Classifier-Free Guidance) and SGD (Score Guidance Differential)
        are the SAME mathematical mechanism with a different reference:

            pred_noise = ε_ref + w · (ε_cond − ε_ref)

          - CFG : ε_ref = ε(null_cond),     w = guidance_scale,  used in wf
          - SGD : ε_ref = ε(baseline_cond), w = sgd_gamma,       used in cf

        They are MUTUALLY EXCLUSIVE — using both creates a hidden composed
        amplification (γ_effective = γ · g) that is rarely the intent.  The
        sampler now asserts that at most one is active.

        Parameters
        ----------
        x_init : (B, C, H, W) optional initial latent.
        sgd_base_cond : (B, D, macro_len) or None
            Baseline (un-shocked) macro conditioning for SGD reference.
        sgd_gamma : float
            SGD amplification factor γ.  Default 1.0 = disabled.
        """
        B, C = shape[0], shape[1]
        img = (x_init.clone().to(self.device)
               if x_init is not None
               else torch.randn(shape, device=self.device))

        # ── Single unified guidance setup ────────────────────────────────
        use_cfg = guidance_scale > 1.0
        use_sgd = (sgd_base_cond is not None) and (sgd_gamma != 1.0)
        assert not (use_cfg and use_sgd), (
            "CFG (guidance_scale > 1.0) and SGD (sgd_base_cond + γ≠1) are "
            "mutually exclusive — pick exactly one. Mixing them produces a "
            "hidden composed amplification γ_eff = γ·g."
        )

        cond = self.model._encode_cond(macro_cond)
        if use_cfg:
            # Learnable null embedding (NOT zeros macro — see encoder docstring).
            # The shape arg is ignored when use_null=True; we just need a tensor
            # to get the batch size right.
            ref_cond_emb = self.model._encode_cond(macro_cond, use_null=True)
            weight = guidance_scale
        elif use_sgd:
            ref_cond_emb = self.model._encode_cond(sgd_base_cond)
            weight = sgd_gamma
        else:
            ref_cond_emb = None
            weight = 1.0

        ts = torch.linspace(0, self.T - 1, steps + 1).long().flip(0)
        pairs = list(zip(ts[:-1].tolist(), ts[1:].tolist()))

        for t_idx, t_prev_idx in pairs:
            t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

            eps_cond = self.model(img, t_tensor, cond=cond)
            if ref_cond_emb is not None:
                eps_ref = self.model(img, t_tensor, cond=ref_cond_emb)
                pred_noise = eps_ref + weight * (eps_cond - eps_ref)
            else:
                pred_noise = eps_cond

            a_t = self.alpha_cp[t_idx]
            a_prev = self.alpha_cp[t_prev_idx] if t_prev_idx >= 0 else torch.tensor(1.0)
            a_prev = a_prev.to(self.device)

            pred_x0 = (img - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            # Progressive dynamic thresholding:
            #   tight clamp at noisy steps (a_t≈0) for stability,
            #   relaxed at clean steps (a_t≈1) so tail events are reachable.
            #   Range: 3.5 (first step) → 9.0 (last step)
            clamp_val = 3.5 + 5.5 * float(a_t.item())
            pred_x0 = pred_x0.clamp(-clamp_val, clamp_val)

            sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
            noise_term = sigma * torch.randn_like(img) if eta > 0 else 0.0
            img = a_prev.sqrt() * pred_x0 + dir_xt + noise_term

        return img

    @torch.no_grad()
    def ddim_invert(self, x0: torch.Tensor, macro_cond: torch.Tensor,
                    steps: int = 50) -> torch.Tensor:
        """Deterministic DDIM inversion x0 -> x_T.

        Use for latent-preserving counterfactual generation::

            x_T  = ddim_invert(x_real, macro_orig)
            x_cf = ddim_sample(..., macro_shocked, x_init=x_T)

        Returns
        -------
        x_T : (B, C, H, W) — inverted latent, ~ N(0, I).
        """
        cond = self.model._encode_cond(macro_cond)
        B = x0.shape[0]
        x_norm = x0.to(self.device)

        times = torch.linspace(0, self.T - 1, steps + 1).long()
        pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        x = x_norm
        for t, t_next in pairs:
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, cond=cond)
            a_t    = self.alpha_cp[t]
            a_next = self.alpha_cp[t_next]
            pred_x0 = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            x = a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * pred_noise
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 4.  EMA
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, beta=0.995, warmup=500):
        self.beta = beta
        self.warmup = warmup
        self.step = 0

    @torch.no_grad()
    def update(self, ema_model, model):
        if self.step < self.warmup:
            ema_model.load_state_dict(model.state_dict())
        else:
            ema_params   = [p.data for p in ema_model.parameters()]
            model_params = [p.data for p in model.parameters()]
            # Fused C++ kernel: ema = ema * beta + (1-beta) * model
            torch._foreach_mul_(ema_params, self.beta)
            torch._foreach_add_(ema_params, model_params, alpha=1 - self.beta)
        self.step += 1


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(data_dir: Path):
    """Load preprocessed artifacts from dataset/ folder."""
    print("[data] loading artifacts from", data_dir)
    raw_windows = np.load(data_dir / "raw_windows.npy")          # (N, ctx+tgt, C)
    macro_windows = np.load(data_dir / "macro_windows.npy")      # (N, macro_len, D)
    train_mask = np.load(data_dir / "train_mask.npy")            # (N,)
    norm_params = np.load(data_dir / "normalization_params.npy",
                          allow_pickle=True).item()

    # QT artifact — try both file names for forward/backward compat
    qt_artifact = None
    for qt_name in ("target_quantile_transformer.pkl", "quantile_transformers.joblib"):
        qt_path = data_dir / qt_name
        if qt_path.exists():
            qt_artifact = joblib.load(qt_path)
            # Support both old (sklearn obj) and new (dict) format
            if not isinstance(qt_artifact, dict):
                qt_artifact = {"quantile_transformer": qt_artifact, "gauss_limits": None}
            print(f"[data] loaded QT from {qt_name}")
            break
    if qt_artifact is None:
        print("[data] WARNING: no QT artifact found — skipping Gaussianisation")

    level_scales = np.load(data_dir / "level_scales.npy")        # (n_levels, C)

    # Window end dates — needed for walk-forward schedule
    wdates_path = data_dir / "window_dates.npy"
    window_dates = np.load(wdates_path, allow_pickle=True) if wdates_path.exists() else None

    print(f"[data] N={len(raw_windows)}  train={train_mask.sum()}  test={~train_mask.sum()}")
    print(f"[data] norm_params: {norm_params}")
    return raw_windows, macro_windows, train_mask, norm_params, qt_artifact, level_scales, window_dates


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Inverse transform (wavelet → raw returns)
# ═══════════════════════════════════════════════════════════════════════════

def inverse_full(
    wav_batch: torch.Tensor,      # (B, C, n_levels, W)
    level_scales: np.ndarray,      # (n_levels, C)
    inst_median: torch.Tensor,     # (B, C)
    inst_mad: torch.Tensor,        # (B, C)
    qt_artifact,
    wavelet_type: str = "haar",
    c: float = 3.0,
) -> np.ndarray:
    """Full inversion: wavelet -> raw returns (B, W, C)."""
    # 1. Undo level rescale
    ls = torch.from_numpy(level_scales)                             # (n_levels, C)
    ls = ls.permute(1, 0).unsqueeze(0).unsqueeze(-1)                # (1, C, n_levels, 1)
    wav_rescaled = wav_batch.cpu() * ls                             # (B, C, n_levels, W)

    # 2. Inverse SWT -> (B, C, W)  -- threaded (C-backed pywt releases GIL)
    gauss = _fast_iswt(wav_rescaled, wavelet_type)                  # (B, C, W)

    # 3. Inverse QT -> z-scores
    if qt_artifact is not None:
        z = batch_inverse_quantile_transform(gauss, qt_artifact)    # (B, C, W)
    else:
        z = gauss

    # 4. Undo z-score: raw = z * (c * 1.4826 * MAD) + median
    med = inst_median.cpu().unsqueeze(-1)                           # (B, C, 1)
    mad = inst_mad.cpu().unsqueeze(-1)                              # (B, C, 1)
    raw = z * (c * 1.4826 * mad) + med                             # (B, C, W)
    return raw.permute(0, 2, 1).numpy()                             # (B, W, C)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Training
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_x0_fast(raw_windows, macro_windows, qt_artifact,
                         wav_type, wav_level, level_scales,
                         ctx, c, mad_floor, z_clip,
                         chunk=512):
    """Vectorised precomputation — bypasses DataLoader, computes x0 in bulk numpy.

    Eliminates per-sample Python overhead from InstanceScaledDataset.__getitem__.

    Parameters
    ----------
    raw_windows   : (N, ctx+tgt, C) ndarray
    macro_windows : (N, macro_len, D) ndarray for symmetric encoder, OR
                    (N, D_ext)        ndarray for snapshot encoder (ablation)

    Returns
    -------
    x0_all    : (N, C, n_levels, W) float32 CPU tensor
    macro_all : (N, D, macro_len)   float32 CPU tensor  (symmetric mode), OR
                (N, D_ext)          float32 CPU tensor  (snapshot mode)
    past_all  : (N, C, ctx)         float32 CPU tensor
    """
    N = len(raw_windows)
    past   = raw_windows[:, :ctx, :].astype(np.float32)   # (N, ctx, C)
    target = raw_windows[:, ctx:, :].astype(np.float32)   # (N, tgt, C)

    med = np.median(past, axis=1)                          # (N, C)
    inst_mad = np.median(np.abs(past - med[:, np.newaxis, :]), axis=1)  # (N, C)
    inst_mad = np.maximum(inst_mad, mad_floor)

    denom = c * 1.4826 * inst_mad                          # (N, C)
    z_target = np.clip(
        (target - med[:, np.newaxis, :]) / denom[:, np.newaxis, :],
        -z_clip, z_clip,
    ).astype(np.float32)                                   # (N, tgt, C)
    z_past = np.clip(
        (past - med[:, np.newaxis, :]) / denom[:, np.newaxis, :],
        -z_clip, z_clip,
    ).astype(np.float32)                                   # (N, ctx, C)

    z_target_t = torch.from_numpy(z_target.transpose(0, 2, 1))                        # (N, C, tgt)
    z_past_t   = torch.from_numpy(z_past.transpose(0, 2, 1))                          # (N, C, ctx)
    # Macro tensor shape depends on encoder type:
    #   symmetric (3D input → 3D model input): (N, macro_len, D) → (N, D, macro_len)
    #   snapshot  (2D input → 2D model input): (N, D_ext) kept as-is
    if macro_windows.ndim == 3:
        macro_t = torch.from_numpy(macro_windows.astype(np.float32).transpose(0, 2, 1))
    elif macro_windows.ndim == 2:
        macro_t = torch.from_numpy(macro_windows.astype(np.float32))
    else:
        raise ValueError(
            f"macro_windows must be 2D (snapshot) or 3D (series); got shape {macro_windows.shape}"
        )

    ls = (torch.from_numpy(level_scales)
          .permute(1, 0).unsqueeze(0).unsqueeze(-1))       # (1, C, n_levels, 1)
    all_x0 = []
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            g = batch_quantile_transform(z_target_t[start:end], qt_artifact) if qt_artifact else z_target_t[start:end]
            wav = _fast_swt_forward(g, wav_type, wav_level)
            all_x0.append((wav / ls).clamp(-10, 10))

    return torch.cat(all_x0, 0), macro_t, z_past_t


def train(args, device, raw_windows, macro_windows, train_mask,
          norm_params, qt_artifact, level_scales):

    ctx = int(norm_params["context_len"])
    tgt = int(norm_params["target_len"])
    wav_type = str(norm_params["wavelet_type"])
    wav_level = int(norm_params["wavelet_level"])
    n_levels = wav_level + 1
    n_assets = raw_windows.shape[2]
    # macro_windows shape depends on encoder type:
    #   symmetric: (N, macro_len, D_base)
    #   snapshot:  (N, D_ext)
    macro_encoder_type = str(getattr(args, "macro_encoder", "symmetric"))
    if macro_windows.ndim == 3:
        macro_len = macro_windows.shape[1]
        macro_dim = macro_windows.shape[2]
    elif macro_windows.ndim == 2:
        # snapshot ablation: macro_len conceptually = 1 (single timestep)
        macro_len = 1
        macro_dim = macro_windows.shape[1]
    else:
        raise ValueError(f"macro_windows must be 2D or 3D, got {macro_windows.shape}")
    c_scale = float(norm_params.get("c", 3.0))
    mad_floor = float(norm_params.get("mad_floor", 1e-4))
    z_clip = float(norm_params.get("z_clip", 10.0))

    print(f"\n[train] assets={n_assets}  ctx={ctx}  tgt={tgt}  levels={n_levels}")
    print(f"[train] macro_encoder={macro_encoder_type}  macro_dim={macro_dim}  macro_len={macro_len}")
    print(f"[train] epochs={args.epochs}  batch={args.batch}")

    train_raw   = raw_windows[train_mask]
    train_macro = macro_windows[train_mask]

    base_ch = int(getattr(args, "base_ch", 32))
    model = build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                              macro_encoder_type=macro_encoder_type, base_ch=base_ch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] base_ch={base_ch}  params: {n_params:,}")

    ema_model = build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                                  macro_encoder_type=macro_encoder_type, base_ch=base_ch)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(beta=0.995, warmup=200)

    diffusion = Diffusion(
        model, timesteps=500, device=device,
        cond_drop_prob=float(getattr(args, "cond_drop_prob", 0.0)),
    )

    # Stratified weight decay: stronger L2 penalty on the macro encoder.
    # Rationale: the macro encoder is ~0.5% of the params (6.8k of 1.4M) but
    # is the SOLE path through which macro variables enter the model. With
    # uniform wd=1e-4, the macro encoder is effectively under-regularised and
    # can memorise condition-specific patterns. wd_macro=1e-2 restricts the
    # macro→cond_emb map, forcing it to capture coarser/more general signals.
    # With wd_macro unset, both groups use weight_decay.
    wd_other = float(getattr(args, "weight_decay", 1e-4))
    _wdm = getattr(args, "wd_macro", None)
    wd_macro = float(_wdm) if _wdm is not None else wd_other
    if abs(wd_macro - wd_other) < 1e-12:
        # Uniform: a single parameter group.
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=wd_other)
    else:
        macro_params = [p for n, p in model.named_parameters()
                        if n.startswith("macro_enc")]
        other_params = [p for n, p in model.named_parameters()
                        if not n.startswith("macro_enc")]
        n_macro = sum(p.numel() for p in macro_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"[train] stratified WD: macro_enc wd={wd_macro} ({n_macro:,} params), "
              f"other wd={wd_other} ({n_other:,} params)")
        optimizer = torch.optim.AdamW([
            {"params": other_params, "weight_decay": wd_other},
            {"params": macro_params, "weight_decay": wd_macro},
        ], lr=args.lr)
    # eta_min proportional to how "light" this run is: fewer epochs → higher floor → gentler annealing.
    # e.g. 100 epochs / ref=1000 → eta_min = lr×0.9; 1000 → eta_min=0 (full cosine).
    _ph0_ref = 1000
    _eta_min_ph0 = args.lr * max(0.0, 1.0 - args.epochs / _ph0_ref)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=_eta_min_ph0
    )

    ckpt_path = _ckpt_path_for("initial", args)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print("[train] pre-computing wavelet images ...", end=" ", flush=True)
    x0_all, macro_all, past_all = _precompute_x0_fast(
        train_raw, train_macro, qt_artifact, wav_type, wav_level, level_scales,
        ctx, c_scale, mad_floor, z_clip,
    )
    N_train   = len(train_raw)
    _n_batches = max(1, int(np.ceil(N_train / args.batch)))
    _rng_ph0  = np.random.default_rng(42)

    x0_all    = x0_all.to(device, non_blocking=True)
    macro_all = macro_all.to(device, non_blocking=True)
    past_all  = past_all.to(device, non_blocking=True)
    print(f"done  x0={tuple(x0_all.shape)}")


    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_eps = 0.0
        perm = _rng_ph0.permutation(N_train)
        for start in range(0, N_train, args.batch):
            end = min(start + args.batch, N_train)
            b_idx = perm[start:end]
            x0         = x0_all[b_idx]
            macro_cond = macro_all[b_idx]

            optimizer.zero_grad(set_to_none=True)
            total_loss, comps = diffusion.compute_loss(x0, macro_cond)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(ema_model, model)
            epoch_loss_eps   += comps["loss_eps"]

        scheduler.step()
        avg_eps   = epoch_loss_eps   / _n_batches

        _progress_tick("Phase0", epoch, args.epochs, loss=avg_eps)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  loss_eps={avg_eps:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_eps < best_loss:
            best_loss = avg_eps
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss,
                "norm_params": norm_params,
                "args": vars(args),
                "macro_encoder_type": macro_encoder_type,
                "macro_dim": macro_dim,
                "macro_len": macro_len,
                "base_ch": base_ch,
                "cond_drop_prob": float(getattr(args, "cond_drop_prob", 0.0)),
            }, ckpt_path)

    print(f"[train] done — best loss={best_loss:.5f}  ckpt={ckpt_path}")
    return ema_model, norm_params


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Walk-forward helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_wf_schedule(
    train_split_date: str,
    window_dates: np.ndarray,
    eval_end_date=None,
    step_freq: str = "Q",
):
    """Build rolling WF schedule (eval_label, q_start, q_end, ckpt_name).

    Parameters
    ----------
    train_split_date : str
        Last date of in-sample training data.
    window_dates : np.ndarray
        End dates of every preprocessing window (from window_dates.npy).
    eval_end_date : str or None
        Upper bound for the evaluation window. If None, uses the last date
        in window_dates (i.e. the full out-of-sample extent of the dataset).
    step_freq : str
        "Q" = quarterly steps (default), "A" or "Y" = annual steps.

    Each step label maps to the period AFTER the training cutoff; the ckpt_name
    column records which checkpoint to load (no future data leakage).
    """
    import pandas as pd
    split = pd.Timestamp(train_split_date)
    dates = pd.to_datetime(window_dates)

    # Bound the schedule: use eval_end_date if provided, else data extent.
    bound = pd.Timestamp(eval_end_date) if eval_end_date is not None else dates.max()

    # Period type: Q (quarterly) or A (annual)
    _freq = str(step_freq).upper()
    period_alias = "Y" if _freq in ("A", "Y") else "Q"

    first_period = (split + pd.DateOffset(days=1)).to_period(period_alias)
    last_period  = bound.to_period(period_alias)

    schedule = []
    p = first_period
    prev_ckpt = "initial"
    while p <= last_period:
        label   = str(p.year) if period_alias == "Y" else f"{p.year}Q{p.quarter}"
        q_start = p.start_time
        q_end   = p.end_time.normalize()
        schedule.append((label, q_start, q_end, prev_ckpt))
        prev_ckpt = label
        p = p + 1

    return schedule


def _ckpt_path_for(ckpt_name: str, args) -> Path:
    """Resolve checkpoint name to absolute path.

    Layout: ``<ROOT>/<args.ckpt_dir>/<stem>_<label>.pt``.
    All labels including 'initial' follow the same pattern so that
    wf_eval.py (which looks for 'ckpt_wf_initial.pt') finds the file.
    The directory is created on demand by the caller before saving.
    """
    base = ROOT / getattr(args, "ckpt_dir", ".")
    stem = Path(args.ckpt).stem
    return base / f"{stem}_{ckpt_name}.pt"


def build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                     macro_encoder_type: str = "symmetric", base_ch: int = 32):
    """Factory for UNet with the canonical project configuration.

    Parameters
    ----------
    macro_encoder_type : str, default "symmetric"
        - "symmetric" : standard time-series macro encoder; expects macro_dim=D_base
          and macro_cond shape (B, D_base, macro_len) at runtime.
        - "snapshot"  : ABLATION encoder; expects macro_dim=D_ext (=2·D_base) and
          macro_cond shape (B, D_ext) at runtime. macro_len is ignored.
    base_ch : int, default 32
        U-Net base channel width. The conditioning/time width scale with it
        (cond_dim = time_dim = 2·base_ch) so an upscale stays balanced rather
        than starving the FiLM/heads. base_ch=32 -> ~1.4M params (production),
        48 -> ~3.2M, 64 -> ~5.6M. Stored in the checkpoint metadata so the
        generation path reconstructs the matching size.
    """
    width = 2 * base_ch
    return UNet(
        in_channels=n_assets, n_levels=n_levels,
        base_ch=base_ch, time_dim=width,
        macro_dim=macro_dim, macro_len=macro_len,
        ctx_len=ctx, cond_dim=width, dropout=0.1,
        macro_encoder_type=macro_encoder_type,
    ).to(device)


def wf_train(args, device, raw_windows, macro_windows, window_dates,
             norm_params, qt_artifact, level_scales):
    """Walk-forward fine-tuning: quarterly expanding window, warm-start from previous ckpt.

    Phase 0 (initial checkpoint, diffusion_ckpt.pt) must exist before calling this.
    For each quarter Q → fine-tunes on data whose window_end ≤ Q_end → saves diffusion_ckpt_{label}.pt.
    The next quarter loads THIS checkpoint as warm-start.
    """
    import pandas as pd

    ctx       = int(norm_params["context_len"])
    tgt       = int(norm_params["target_len"])
    wav_type  = str(norm_params["wavelet_type"])
    wav_level = int(norm_params["wavelet_level"])
    n_levels  = wav_level + 1
    n_assets  = raw_windows.shape[2]
    macro_encoder_type = str(getattr(args, "macro_encoder", "symmetric"))
    if macro_windows.ndim == 3:
        macro_len = macro_windows.shape[1]
        macro_dim = macro_windows.shape[2]
    elif macro_windows.ndim == 2:
        macro_len = 1
        macro_dim = macro_windows.shape[1]
    else:
        raise ValueError(f"macro_windows must be 2D or 3D, got {macro_windows.shape}")
    c_scale   = float(norm_params.get("c", 3.0))
    mad_floor = float(norm_params.get("mad_floor", 1e-4))
    z_clip    = float(norm_params.get("z_clip", 10.0))
    split_date = str(norm_params["train_split_date"])

    dates    = pd.to_datetime(window_dates)
    _eval_end  = getattr(args, 'eval_end_date', None)
    _step_freq = getattr(args, 'step_freq', None) or "Q"
    schedule = build_wf_schedule(split_date, window_dates,
                                 eval_end_date=_eval_end, step_freq=_step_freq)

    _freq_label = "annual" if str(_step_freq).upper() in ("A", "Y") else "quarterly"
    print(f"\n[wf_train] {len(schedule)} {_freq_label} steps  wf_epochs={args.wf_epochs}  "
          f"eval_end={_eval_end or 'data_end'}")

    _total_ep = int(args.epochs) + len(schedule) * int(args.wf_epochs)
    _progress_init(_ckpt_path_for("initial", args).parent / "train_progress.json", _total_ep)
    print(f"[wf_train] live progress -> {_PROG['path']}  (total {_total_ep} epochs across "
          f"Phase0 + {len(schedule)} WF steps)", flush=True)


    # ── Phase 0: initial training ────────────────────────────────────────
    # Train from scratch on all windows up to split_date.
    # If the checkpoint already exists, skip (allows resuming interrupted runs).
    initial_ckpt = _ckpt_path_for("initial", args)
    if not initial_ckpt.exists():
        print(f"[wf_train] Phase 0 — training from scratch on "
              f"{(dates <= pd.Timestamp(split_date)).sum()} train windows")
        import pandas as _pd
        ph0_mask = np.array(dates <= _pd.Timestamp(split_date))
        train(args, device, raw_windows, macro_windows, ph0_mask,
              norm_params, qt_artifact, level_scales)
        print(f"[wf_train] Phase 0 complete -> {initial_ckpt.name}")
    else:
        print(f"[wf_train] Phase 0 — checkpoint exists ({initial_ckpt.name}) — skip")
        _progress_add_done(int(args.epochs))

    ls = torch.from_numpy(level_scales)
    ls = ls.permute(1, 0).unsqueeze(0).unsqueeze(-1)   # (1, C, n_levels, 1)

    prev_ckpt_path = initial_ckpt

    # Incremental precompute cache — avoids recomputing the full expanding dataset each quarter.
    # Only new windows (since last quarter) are processed; results are appended to the cache.
    _wf_cache_x0   = None  # (N_cached, C, nL, W)
    _wf_cache_mac  = None  # (N_cached, D, macro_len)
    _wf_cache_past = None  # (N_cached, C, ctx)
    _wf_n_cached   = 0     # windows already precomputed

    for label, q_start, q_end, _ in schedule:
        out_ckpt = _ckpt_path_for(label, args)
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if out_ckpt.exists():
            print(f"[wf_train] {label}: checkpoint exists — skip")
            _progress_add_done(int(args.wf_epochs))
            prev_ckpt_path = out_ckpt
            # Reset cache: skipped quarters didn't precompute tensors, so the next
            # active quarter must rebuild from scratch to avoid index-out-of-bounds.
            _wf_cache_x0 = _wf_cache_mac = _wf_cache_past = None
            _wf_n_cached = 0
            continue

        # Expanding training set: all windows whose end date ≤ q_end
        wf_mask = np.array(dates <= q_end)
        n_train = wf_mask.sum()
        print(f"\n[wf_train] {label}  ({q_start.date()} -> {q_end.date()})  "
              f"train_windows={n_train}  warm-start={prev_ckpt_path.name}")

        # Load previous checkpoint (warm-start)
        ckpt_data = torch.load(prev_ckpt_path, map_location=device, weights_only=False)
        base_ch = int(ckpt_data.get("base_ch", getattr(args, "base_ch", 32)))

        model = build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                                  macro_encoder_type=macro_encoder_type, base_ch=base_ch)
        model.load_state_dict(ckpt_data["ema_model"], strict=False)

        ema_model = build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                                      macro_encoder_type=macro_encoder_type, base_ch=base_ch)
        ema_model.load_state_dict(model.state_dict())
        ema = EMA(beta=0.995, warmup=100)

        diffusion = Diffusion(
        model, timesteps=500, device=device,
        cond_drop_prob=float(getattr(args, "cond_drop_prob", 0.0)),
    )
        # Each quarter: full cosine cycle from wf_lr → 0 (warm restart per quarter)
        # wf_lr defaults to args.lr — no artificial 0.1x penalty
        _wf_lr = args.wf_lr if args.wf_lr is not None else args.lr
        # Re-apply stratified WD setup for phase-1 optimizer.
        _wd_other = float(getattr(args, "weight_decay", 1e-4))
        _wdm1 = getattr(args, "wd_macro", None)
        _wd_macro = float(_wdm1) if _wdm1 is not None else _wd_other
        if abs(_wd_macro - _wd_other) < 1e-12:
            optimizer = torch.optim.AdamW(model.parameters(), lr=_wf_lr,
                                           weight_decay=_wd_other)
        else:
            _macro_p = [p for n, p in model.named_parameters() if n.startswith("macro_enc")]
            _other_p = [p for n, p in model.named_parameters() if not n.startswith("macro_enc")]
            optimizer = torch.optim.AdamW([
                {"params": _other_p, "weight_decay": _wd_other},
                {"params": _macro_p, "weight_decay": _wd_macro},
            ], lr=_wf_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.wf_epochs, 1), eta_min=0.0
        )

        N_wf = wf_mask.sum()
        # Incremental precompute: only process new windows, append to cache.
        _n_new = int(N_wf) - _wf_n_cached
        if _n_new > 0:
            _new_raw = raw_windows[wf_mask][_wf_n_cached:]        # (n_new, total_len, C)
            _new_mac = macro_windows[wf_mask][_wf_n_cached:]      # (n_new, macro_len, D)
            _nx0, _nmac, _npast = _precompute_x0_fast(
                _new_raw, _new_mac, qt_artifact, wav_type, wav_level, level_scales,
                ctx, c_scale, mad_floor, z_clip,
            )
            _wf_cache_x0   = torch.cat([_wf_cache_x0,   _nx0],   0) if _wf_cache_x0   is not None else _nx0
            _wf_cache_mac  = torch.cat([_wf_cache_mac,  _nmac],  0) if _wf_cache_mac  is not None else _nmac
            _wf_cache_past = torch.cat([_wf_cache_past, _npast], 0) if _wf_cache_past is not None else _npast
            _wf_n_cached = int(N_wf)
        _x0_wf   = _wf_cache_x0
        _mac_wf  = _wf_cache_mac
        _past_wf = _wf_cache_past
        _n_bat_wf = max(1, int(np.ceil(N_wf / args.batch)))
        _rng_wf   = np.random.default_rng(42)

        # Move to device once
        _x0_wf   = _x0_wf.to(device, non_blocking=True)
        _mac_wf  = _mac_wf.to(device, non_blocking=True)
        _past_wf = _past_wf.to(device, non_blocking=True)
        best_loss = float("inf")

        for epoch in range(1, args.wf_epochs + 1):
            model.train()
            epoch_loss_eps = 0.0
            perm = _rng_wf.permutation(N_wf)
            for start in range(0, N_wf, args.batch):
                end = min(start + args.batch, N_wf)
                b_idx = perm[start:end]
                x0         = _x0_wf[b_idx]
                macro_cond = _mac_wf[b_idx]

                optimizer.zero_grad(set_to_none=True)
                total_loss, comps = diffusion.compute_loss(x0, macro_cond)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(ema_model, model)
                epoch_loss_eps   += comps["loss_eps"]

            sched.step()
            avg_eps   = epoch_loss_eps   / _n_bat_wf
            _progress_tick(label, epoch, args.wf_epochs, loss=avg_eps)
            if avg_eps < best_loss:
                best_loss = avg_eps
            if epoch % 10 == 0 or epoch == args.wf_epochs:
                print(f"  [{label}] epoch {epoch}/{args.wf_epochs}  "
                      f"loss_eps={avg_eps:.5f}")

        torch.save({
            "epoch": args.wf_epochs,
            "label": label,
            "ema_model": ema_model.state_dict(),
            "norm_params": norm_params,
            "best_loss": best_loss,
            "macro_encoder_type": macro_encoder_type,
            "macro_dim": macro_dim,
            "macro_len": macro_len,
            "base_ch": base_ch,
            "cond_drop_prob": float(getattr(args, "cond_drop_prob", 0.0)),
        }, out_ckpt)
        print(f"  [{label}] -> {out_ckpt.name}  best_loss={best_loss:.5f}")
        prev_ckpt_path = out_ckpt

    print(f"\n[wf_train] done — {len(schedule)} quarterly checkpoints ready")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  Pipeline helper — single-quarter generation for external orchestrators
# ═══════════════════════════════════════════════════════════════════════════

def generate_quarter(
    ckpt_path: Path,
    raw_q: np.ndarray,
    macro_q: np.ndarray,
    norm_params: dict,
    qt_artifact: dict,
    level_scales: np.ndarray,
    device: torch.device,
    *,
    n_samples: int = 8,
    ddim_steps: int = 100,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    mad_inv_window: int = 0,
    label: str = "",
    verbose: bool = True,
):
    """Run diffusion generation for a single quarter's eval windows.

    Re-implements the per-quarter generation block from ``wf_evaluate`` as a
    reusable helper for external orchestrators (e.g. ``src/generation/wf_eval.py``).

    Parameters
    ----------
    ckpt_path : Path                          Per-quarter checkpoint to load.
    raw_q     : (N_q, ctx+tgt, C) ndarray     Raw windows (past+target) for this quarter.
    macro_q   : (N_q, macro_len, D) ndarray   Macro windows for this quarter.
    norm_params, qt_artifact, level_scales    Preprocessing artifacts.
    device                                    Torch device.
    n_samples (S)                             Synthetic samples per real window.
    ddim_steps, eta, guidance_scale           DDIM sampler controls.
    mad_inv_window                            Trailing-window MAD for inversion (0=full-ctx).
    label                                     Pretty label used in diagnostic prints.
    verbose                                   Toggle the per-quarter diagnostic prints.

    Returns
    -------
    real_arr  : (N_q, tgt, C) ndarray         Real targets (pass-through of raw_q[:, ctx:, :]).
    synth_arr : (N_q, S, tgt, C) ndarray      Synthetic samples reshaped per real window.
    """
    ctx       = int(norm_params["context_len"])
    tgt       = int(norm_params["target_len"])
    wav_type  = str(norm_params["wavelet_type"])
    wav_level = int(norm_params["wavelet_level"])
    n_levels  = wav_level + 1
    n_assets  = raw_q.shape[2]
    # macro_q shape depends on encoder type:
    #   symmetric: (N_q, macro_len, D_base)
    #   snapshot:  (N_q, D_ext)
    if macro_q.ndim == 3:
        macro_len = macro_q.shape[1]
        macro_dim = macro_q.shape[2]
    elif macro_q.ndim == 2:
        macro_len = 1
        macro_dim = macro_q.shape[1]
    else:
        raise ValueError(f"macro_q must be 2D or 3D, got {macro_q.shape}")
    c_scale   = float(norm_params.get("c", 3.0))
    mad_floor = float(norm_params.get("mad_floor", 1e-4))

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Saved metadata takes precedence
    macro_encoder_type = ckpt_data.get("macro_encoder_type", "symmetric")
    macro_dim = ckpt_data.get("macro_dim", macro_dim)
    macro_len = ckpt_data.get("macro_len", macro_len)
    base_ch = int(ckpt_data.get("base_ch", 32))
    ema_q = build_unet_model(n_assets, n_levels, macro_dim, macro_len, ctx, device,
                              macro_encoder_type=macro_encoder_type, base_ch=base_ch)
    ema_q.load_state_dict(ckpt_data["ema_model"], strict=False)
    ema_q.eval()
    diff_q = Diffusion(ema_q, timesteps=500, device=device)

    N_q = raw_q.shape[0]
    S   = int(n_samples)

    # ── Pre-compute instance stats (vectorised) ───────────────────────────
    past_q    = raw_q[:, :ctx, :].astype(np.float32)
    q_targets = raw_q[:, ctx:, :]                                  # (N_q, tgt, C)

    q_meds_np = np.median(past_q, axis=1)                          # (N_q, C)
    q_mads_np = np.maximum(
        np.median(np.abs(past_q - q_meds_np[:, None, :]), axis=1),
        mad_floor,
    ).astype(np.float32)                                            # (N_q, C)

    # Macro tensor for the model:
    #   symmetric: (N_q, D, macro_len) — transposed from (N_q, macro_len, D)
    #   snapshot:  (N_q, D_ext)        — already in the right shape
    if macro_q.ndim == 3:
        q_macros_np = np.transpose(macro_q, (0, 2, 1)).astype(np.float32)
    else:
        q_macros_np = macro_q.astype(np.float32)

    # ── Inversion scale: optional trailing-window MAD ────────────────────
    if mad_inv_window and mad_inv_window > 0 and mad_inv_window < ctx:
        short = past_q[:, -mad_inv_window:, :]                                  # (N_q, W, C)
        short_med = np.median(short, axis=1)                                    # (N_q, C)
        q_mads_inv = np.maximum(
            np.median(np.abs(short - short_med[:, None, :]), axis=1),
            mad_floor,
        ).astype(np.float32)
        if verbose:
            ratio = (q_mads_inv / q_mads_np).mean()
            print(f"  [{label}] mad_inv_window={mad_inv_window}: "
                  f"mean(mad_recent/mad_full)={ratio:.3f}")
    else:
        q_mads_inv = q_mads_np

    # ── Batched generation ────────────────────────────────────────────────
    total = N_q * S
    max_chunk = 1024 if device.type == "cuda" else 256

    mac_all = np.repeat(q_macros_np, S, axis=0)
    med_all = np.repeat(q_meds_np,   S, axis=0)
    mad_all = np.repeat(q_mads_inv,  S, axis=0)

    wav_chunks = []
    with torch.no_grad():
        for start in range(0, total, max_chunk):
            end       = min(start + max_chunk, total)
            mac_t     = torch.from_numpy(mac_all[start:end]).to(device, non_blocking=True)
            chunk_sh  = (end - start, n_assets, n_levels, tgt)
            wav_chunk = diff_q.ddim_sample(
                chunk_sh, mac_t,
                steps=ddim_steps, eta=eta, guidance_scale=guidance_scale,
            )
            wav_chunks.append(wav_chunk.cpu())

    wav_all = torch.cat(wav_chunks, dim=0)
    med_t = torch.tensor(med_all, dtype=torch.float32)
    mad_t = torch.tensor(mad_all, dtype=torch.float32)

    raw_synth_all = inverse_full(
        wav_all, level_scales, med_t, mad_t,
        qt_artifact, wav_type, c_scale,
    )                                                  # (N_q*S, tgt, C)

    # Free model memory before next quarter
    del ema_q, diff_q
    if device.type == "cuda":
        torch.cuda.empty_cache()

    real_arr  = q_targets                                                   # (N_q, tgt, C)
    synth_arr = raw_synth_all.reshape(N_q, S, tgt, n_assets)                # (N_q, S, tgt, C)
    return real_arr, synth_arr


def apply_qt_clip_override(qt_artifact: dict, qt_clip: float | None) -> dict:
    """Mutate qt_artifact's gauss_limits in-place when qt_clip > 0.

    Helper exposed for external pipelines (mirrors the ``[main]`` override block).
    """
    if qt_clip is not None and qt_clip > 0 and qt_artifact is not None:
        new_lim = float(qt_clip)
        n_feat = qt_artifact["quantile_transformer"].n_features_in_
        qt_artifact["gauss_limits"] = np.array(
            [[-new_lim] * n_feat, [+new_lim] * n_feat], dtype=np.float32,
        )
    return qt_artifact


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[main] device={device}")

    # Data — read paths.processed_dir from config.yaml so each experiment
    # can use a different folder without touching this file.
    _cfg_path = _get_cfg_path()
    data_dir = ROOT / "processed_data" / "synthetic"
    if _cfg_path.exists():
        import yaml as _yaml
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f)
        from src.paths import resolve_paths as _resolve_paths
        data_dir = _resolve_paths(ROOT, _cfg).processed_dir
        _wf_cfg = _cfg.get("training", {}).get("walk_forward", {})
        if not args.eval_end_date:
            args.eval_end_date = _wf_cfg.get("evaluation_end_date", None)
        if not args.step_freq:
            args.step_freq = _wf_cfg.get("step_freq", "Q")
    raw_windows, macro_windows, train_mask, norm_params, qt_artifact, level_scales, window_dates = \
        load_data(data_dir)

    if window_dates is None:
        raise FileNotFoundError(f"{data_dir}/window_dates.npy not found — run preprocess.py first")

    # ── Macro encoder dispatch ──────────────────────────────────────────
    # symmetric: use macro_windows (N, macro_len, D_base) as-is
    # snapshot (ablation): replace with macro_windows_zscore[:, -1, :] (N, D_ext),
    #   identical to what MacroVAE consumes — snapshot at day-before-target-start
    #   plus 6m momentum.
    if args.macro_encoder == "snapshot":
        _mvae_macro_path = data_dir / "macro_windows_zscore.npy"
        if not _mvae_macro_path.exists():
            raise FileNotFoundError(
                f"{_mvae_macro_path} required for --macro_encoder snapshot. "
                "Re-run scripts/preprocess.py to regenerate."
            )
        _mvae_windows = np.load(_mvae_macro_path)                 # (N, L, D_ext)
        macro_windows = _mvae_windows[:, -1, :].astype(np.float32)  # (N, D_ext)
        print(f"[main] macro_encoder=snapshot → macro_windows shape={macro_windows.shape}")
    else:
        print(f"[main] macro_encoder=symmetric → macro_windows shape={macro_windows.shape}")

    wf_train(
        args, device, raw_windows, macro_windows, window_dates,
        norm_params, qt_artifact, level_scales,
    )


if __name__ == "__main__":
    main()
