"""
Instance-Scaled Dataset for financial time-series diffusion.

Per-window instance scaling: stats computed exclusively from the context
(past) segment, so the target segment is never used to define its own scale.

Pipeline (Hybrid Normalization — Batch-Level QT + SWT):
    Dataset.__getitem__:
        raw_window[t-C : t+W] -> split past/target ->
        stats = median(past), MAD(past) ->
        z_past   = (past   - median) / (c * 1.4826 * MAD)
        z_target = (target - median) / (c * 1.4826 * MAD)

    Training loop (batch-level, vectorised):
        batch z_target -> batch_quantile_transform -> gauss_target
        -> fast SWT (in src/diffusion/lab.py) / level_scales -> model input

Helpers exported here:
    ``batch_quantile_transform``          — vectorised QT on (B, C, W) tensors
    ``batch_inverse_quantile_transform``  — vectorised inverse QT (with clip)
    ``compute_wavelet_level_scales``      — per-(level, channel) std of the
                                            Gaussianised wavelet coefficients

Inversion at inference uses ``src.diffusion.lab.inverse_full`` (pywt,
vectorised along the batch axis).
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    import joblib
except ImportError:  # graceful fallback
    joblib = None


# ---------------------------------------------------------------------------
# Core Dataset (stripped of QT + SWT — those happen at batch level)
# ---------------------------------------------------------------------------

class InstanceScaledDataset(Dataset):
    """On-the-fly instance scaling for multi-asset financial time series diffusion.

    For each sample index:
      1. Split the stored raw window into *context* (past) and *target* segments.
      2. Compute median / MAD on the context **per channel** (instance stats).
      3. Z-score both segments: ``z = (x - median) / (c * 1.4826 * MAD)``.

    Each channel represents one ticker's log returns (C = N_assets).
    Asset identity is implicit in the channel index.

    QT and SWT are applied at **batch level** in the training loop using the
    helper functions ``batch_quantile_transform`` and ``batch_swt_forward``.
    This eliminates per-sample sklearn overhead and enables ``num_workers > 0``.

    Parameters
    ----------
    raw_windows : ndarray ``(N, context_len + target_len, C)``
        Raw (un-normalised) sliding windows.  C = N_assets, each channel
        is one ticker's log returns.
    macro_windows : ndarray ``(N, macro_len, D)``
        Macro conditioning windows (already stationary + tanh-bounded).
    context_len : int
        Number of past timesteps used to compute instance stats.
    target_len : int
        Number of future timesteps (the generation target).
    c : float
        Robust z-score denominator factor (default 3.0).
    """

    def __init__(
        self,
        raw_windows: np.ndarray,
        macro_windows: np.ndarray,
        context_len: int = 128,
        target_len: int = 32,
        c: float = 3.0,
        mad_floor: float = 1e-4,
        z_clip: float = 10.0,
        mad_window: int | None = None,
    ):
        super().__init__()
        total_len = context_len + target_len
        assert raw_windows.shape[1] == total_len, (
            f"raw_windows length {raw_windows.shape[1]} "
            f"!= context_len({context_len}) + target_len({target_len})"
        )

        self.raw_windows = raw_windows.astype(np.float32)
        self.macro_windows = macro_windows.astype(np.float32)
        self.context_len = context_len
        self.target_len = target_len
        self.c = c
        self.mad_floor = mad_floor
        self.z_clip = z_clip
        self.mad_window = mad_window  # if set, MAD uses only last N days of context

    def __len__(self) -> int:
        return len(self.raw_windows)

    # ------------------------------------------------------------------
    # Instance scaling
    # ------------------------------------------------------------------

    def _instance_scale(self, past: np.ndarray, target: np.ndarray):
        """Compute per-channel median/MAD on *past* only, scale both.

        Parameters
        ----------
        past : ``(context_len, C)``
        target : ``(target_len, C)``

        Returns
        -------
        past_scaled, target_scaled : same shapes, z-scored
        inst_median : ``(C,)``   — per-channel median of past
        inst_mad    : ``(C,)``   — per-channel MAD of past (clamped ≥ mad_floor)
        """
        inst_median = np.median(past, axis=0)                          # (C,)
        # MAD window: use only the last N days of context if mad_window is set.
        # Captures recent volatility ramp-up (crisis onset) while keeping the
        # full-context median as a stable location estimator.
        mad_src = past[-self.mad_window:] if self.mad_window is not None else past
        inst_mad = np.median(np.abs(mad_src - inst_median), axis=0)    # (C,)
        inst_mad = np.maximum(inst_mad, self.mad_floor)

        denom = self.c * 1.4826 * inst_mad                            # (C,)
        past_scaled = np.clip((past - inst_median) / denom,
                              -self.z_clip, self.z_clip)               # (ctx, C)
        target_scaled = np.clip((target - inst_median) / denom,
                                -self.z_clip, self.z_clip)             # (tgt, C)

        return (
            past_scaled.astype(np.float32),
            target_scaled.astype(np.float32),
            inst_median.astype(np.float32),
            inst_mad.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # __getitem__  (z-score only — NO QT, NO SWT)
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        raw = self.raw_windows[idx]                                    # (total_len, C)
        past_raw = raw[: self.context_len]                             # (ctx, C)
        target_raw = raw[self.context_len :]                           # (tgt, C)

        # Instance scaling (stats from past ONLY)
        past_scaled, target_scaled, inst_median, inst_mad = \
            self._instance_scale(past_raw, target_raw)

        # Macro conditioning
        macro = self.macro_windows[idx]                                # (macro_len, D)

        return {
            # ---- Batch-level QT + SWT will be applied in training loop ----
            "z_target": torch.tensor(
                target_scaled, dtype=torch.float32
            ).T,                                                       # (C, W)
            "asset_past": torch.tensor(
                past_scaled, dtype=torch.float32
            ).T,                                                       # (C, ctx)
            "macro_cond": torch.tensor(
                macro, dtype=torch.float32
            ).T,                                                       # (D, macro_len)
            # ---- Inversion metadata ----
            "instance_median": torch.tensor(inst_median, dtype=torch.float32),  # (C,)
            "instance_mad": torch.tensor(inst_mad, dtype=torch.float32),        # (C,)
            # Raw target for debugging / direct comparison at inference
            "target_raw": torch.tensor(target_raw, dtype=torch.float32),        # (tgt, C)
        }


# ---------------------------------------------------------------------------
# Batch-level helpers (called in training / inference loop)
# ---------------------------------------------------------------------------

def batch_quantile_transform(z_targets: torch.Tensor, qt) -> torch.Tensor:
    """Vectorised Quantile Transform on a batch of z-scored targets.

    Call ONCE per batch instead of per-sample inside ``__getitem__``.

    Parameters
    ----------
    z_targets : ``(B, C, W)`` float tensor
    qt : dict
        Artifact dict: {'quantile_transformer': qt, 'gauss_limits': bounds}

    Returns
    -------
    gauss_targets : ``(B, C, W)`` float tensor (CPU)
    """
    qt_obj = qt['quantile_transformer']

    B, C, W = z_targets.shape
    # (B, C, W) → (B, W, C) → (B*W, C) for sklearn
    z_flat = z_targets.permute(0, 2, 1).reshape(-1, C).cpu().numpy()
    gauss_flat = qt_obj.transform(z_flat).astype(np.float32)           # (B*W, C)
    return (
        torch.from_numpy(gauss_flat)
        .reshape(B, W, C)
        .permute(0, 2, 1)                                              # (B, C, W)
    )


def batch_inverse_quantile_transform(gauss: torch.Tensor, qt_dict) -> torch.Tensor:
    """Vectorised inverse Quantile Transform with safety clipping.

    Parameters
    ----------
    gauss : ``(B, C, W)`` float tensor
    qt_dict : dict
        Artifact dict: {'quantile_transformer': qt, 'gauss_limits': bounds}

    Returns
    -------
    z_targets : ``(B, C, W)`` float tensor (CPU)
    """
    qt = qt_dict['quantile_transformer']
    gauss_limits = qt_dict.get('gauss_limits', None)
    
    B, C, W = gauss.shape
    gauss_numpy = gauss.permute(0, 2, 1).reshape(-1, C).cpu().numpy()   # (B*W, C)
    
    # Apply safety clipping BEFORE inverse transform
    if gauss_limits is not None:
        gauss_numpy = np.clip(gauss_numpy, 
                             gauss_limits[0], gauss_limits[1])  # Clip to historical extremes
    
    z_flat = qt.inverse_transform(gauss_numpy).astype(np.float32)       # (B*W, C)
    return (
        torch.from_numpy(z_flat)
        .reshape(B, W, C)
        .permute(0, 2, 1)                                              # (B, C, W)
    )


# ---------------------------------------------------------------------------
# Wavelet level-scale computation (run once on training data, save result)
# ---------------------------------------------------------------------------

def compute_wavelet_level_scales(
    raw_windows: np.ndarray,
    context_len: int,
    target_len: int,
    wavelet_type: str = "haar",
    wavelet_level: int = 4,
    c: float = 3.0,
    mad_floor: float = 1e-4,
    z_clip: float = 10.0,
    n_samples: int = 5000,
    seed: int = 42,
    quantile_transformer=None,
) -> np.ndarray:
    """Compute per-(level, channel) std on **Gaussianised** wavelet images.

    CRITICAL: Level scales must be computed on the SAME data distribution
    that the model sees at training time.  When a QuantileTransformer is
    used, the pipeline is::

        z_target → QT.transform → gauss_target → SWT → wavelet_coeffs

    so level_scales must be ``std(wavelet_coeffs)`` of the **Gaussianised**
    data, NOT the raw fat-tailed z-scores.  Passing ``quantile_transformer``
    ensures this.

    Parameters
    ----------
    raw_windows : ``(N, context_len + target_len, C)``
    quantile_transformer : dict or *None*
        Artifact dict ``{'quantile_transformer': qt, 'gauss_limits': bounds}``.
        When provided, z-scores are Gaussianised before SWT.

    Returns
    -------
    level_scales : ``(n_levels, C)``  — clamped ≥ 1e-6.
    """
    rng = np.random.RandomState(seed)
    N = len(raw_windows)
    idx = rng.choice(N, min(n_samples, N), replace=False)

    n_levels = wavelet_level + 1
    C = raw_windows.shape[2]
    W = target_len

    all_wav = []
    for i in idx:
        past = raw_windows[i, :context_len]
        target = raw_windows[i, context_len:]

        median = np.median(past, axis=0)
        mad = np.maximum(np.median(np.abs(past - median), axis=0), mad_floor)
        denom = c * 1.4826 * mad
        target_z = np.clip((target - median) / denom,
                           -z_clip, z_clip)            # (W, C) — clipped

        # Gaussianise BEFORE SWT (matches InstanceScaledDataset.__getitem__)
        if quantile_transformer is not None:
            qt = quantile_transformer['quantile_transformer']
            target_z = qt.transform(
                target_z  # (W, C) — sklearn-compatible 2D
            ).astype(np.float32)

        wav = np.zeros((n_levels, W, C), dtype=np.float32)
        for ch in range(C):
            coeffs = pywt.swt(
                target_z[:, ch], wavelet_type,
                level=wavelet_level, trim_approx=False,
            )
            cA = coeffs[0][0]
            details = [p[1] for p in coeffs]
            wav[:, :, ch] = np.stack([cA] + details)
        all_wav.append(wav)

    all_wav = np.stack(all_wav)                       # (n, Levels, W, C)
    scales = all_wav.std(axis=(0, 2))                 # (Levels, C)
    return np.maximum(scales, 1e-6).astype(np.float32)


