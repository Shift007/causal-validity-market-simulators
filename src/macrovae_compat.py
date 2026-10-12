"""Thin adapter around the original MacroVAE of Kubiak et al. (ICAIF '25).

The baseline is the authors' own implementation, used unmodified from the
``MacroVAE`` git submodule. Two things are added here and nowhere else:

  * the encoder log-variance is clamped to [-20, 10], a numerical guard that is
    inactive on in-distribution inputs;
  * checkpoints written by an earlier copy of the class, in which the encoder
    layer ``expand_channels`` was named ``expand_conv``, load transparently.
"""
from __future__ import annotations

import sys
from pathlib import Path

_UPSTREAM = Path(__file__).resolve().parents[1] / "MacroVAE"
if not (_UPSTREAM / "macro_vae.py").exists():
    raise ImportError(
        "MacroVAE submodule not found. Run: git submodule update --init")
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from macro_vae import MacroVAE as _UpstreamMacroVAE  # noqa: E402

_RENAMED = {"encoder.expand_conv.": "encoder.expand_channels."}


class MacroVAE(_UpstreamMacroVAE):
    def encode(self, x):
        mu, log_var = super().encode(x)
        return mu, log_var.clamp(-20.0, 10.0)

    def load_state_dict(self, state_dict, *args, **kwargs):
        fixed = {}
        for key, value in state_dict.items():
            for old, new in _RENAMED.items():
                if key.startswith(old):
                    key = new + key[len(old):]
            fixed[key] = value
        return super().load_state_dict(fixed, *args, **kwargs)
