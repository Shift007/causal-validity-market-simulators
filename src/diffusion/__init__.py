"""
Diffusion model components for time series synthesis.

Public API:
- ``Diffusion``        : DDPM/DDIM forward + sampling logic (with SGD support).
- ``UNet``             : conditional U-Net backbone.
- ``EMA``              : exponential-moving-average wrapper for training.
- ``build_unet_model`` : factory using the canonical project configuration.
- ``train`` / ``wf_train`` : Phase 0 / walk-forward training entry points.
- ``generate_quarter`` : single-quarter generation helper used by src/generation.
- ``apply_qt_clip_override`` : inference-time QT cap helper.
- ``SymmetricMacroEncoder``  : sign-symmetric macro conditioning encoder.
- ``inverse_full`` : wavelet → raw-returns inverse transform.

Walk-forward OOS evaluation lives in ``src/generation/wf_eval.py``;
counterfactual generation in ``src/generation/counterfactual*.py``.
"""

from .lab import (
    Diffusion,
    UNet,
    EMA,
    build_unet_model,
    train,
    wf_train,
    generate_quarter,
    apply_qt_clip_override,
    SymmetricMacroEncoder,
    load_data,
    inverse_full,
)

__all__ = [
    "Diffusion",
    "UNet",
    "EMA",
    "build_unet_model",
    "train",
    "wf_train",
    "generate_quarter",
    "apply_qt_clip_override",
    "SymmetricMacroEncoder",
    "load_data",
    "inverse_full",
]
