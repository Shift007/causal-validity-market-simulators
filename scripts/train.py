#!/usr/bin/env python
"""Diffusion model walk-forward training.

Thin wrapper around ``src.diffusion.lab``: it parses the command line
(``--config``, ``--device``, ``--restart`` and the training options) and calls
the training loop there. The architecture is fixed in
``src.diffusion.lab._UNET_CONFIG``.

Usage
-----
    python scripts/train.py                    # full walk-forward (Phase 0 + per-quarter)
    python scripts/train.py --device cuda      # force GPU
    python scripts/train.py --restart          # wipe checkpoints and retrain
    python scripts/train.py --epochs 200 \
                            --wf_epochs 30 \
                            --samples 8 \
                            --ddim_steps 100   # override training/eval hyperparams

Per-quarter checkpoints land in ``checkpoints/wf/ckpt_wf_<label>.pt`` to match
the layout consumed by ``src/generation/wf_eval.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Ensure the project root is importable when invoked as ``python scripts/train.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diffusion walk-forward training")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config YAML (default: <project_root>/config.yaml).")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: auto-detect CUDA -> CPU).")
    p.add_argument("--restart", action="store_true",
                   help="Delete the resolved checkpoint dir before training.")
    p.add_argument("--epochs", type=int, default=200,
                   help="Phase-0 epochs (default 200).")
    p.add_argument("--wf_epochs", type=int, default=30,
                   help="Per-step fine-tuning epochs (default 30).")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_ch", type=int, default=32,
                   help="U-Net base channel width (32~1.4M, 48~3.2M, 64~5.6M params). "
                        "cond_dim/time_dim scale as 2*base_ch. Saved in ckpt metadata.")
    p.add_argument("--wf_lr", type=float, default=None,
                   help="LR for walk-forward fine-tuning (default: same as --lr).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--macro_encoder", type=str, default=None,
                   choices=["symmetric", "snapshot"],
                   help="Macro encoder variant. 'symmetric' = time-series default; "
                        "'snapshot' = ablation (single-value snapshot+momentum, "
                        "MVAE-identical input). Default read from config.yaml "
                        "(diffusion.macro_encoder) or 'symmetric' if absent.")
    p.add_argument("--cond_drop_prob", type=float, default=None,
                   help="Probability of replacing cond_emb with the learnable null "
                        "embedding during training (classifier-free guidance). "
                        "0.0 = pure conditional (default). 0.10-0.15 enables CFG-correct "
                        "sampling.")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="AdamW weight_decay for non-macro_enc params (lab default 1e-4).")
    p.add_argument("--wd_macro", type=float, default=None,
                   help="AdamW weight_decay applied ONLY to macro_enc params (param "
                        "groups split). When different from --weight_decay, stratified "
                        "WD is enabled. Recommended for macro encoder regularisation: 1e-2.")
    return p.parse_args()


def main() -> None:
    cli = parse_args()

    # Resolve checkpoint_dir from paths: in config.yaml.
    import os as _os, yaml as _yaml
    _cfg_path = (ROOT / cli.config) if cli.config else ROOT / "config.yaml"
    if cli.config:
        _os.environ["DIFFUSION_CONFIG_PATH"] = str(_cfg_path)
    _ckpt_dir = "checkpoints/synthetic/diffusion"
    _wf_epochs_default = cli.wf_epochs   # CLI takes priority; config is fallback
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f)
        from src.paths import resolve_paths
        _ckpt_dir = str(resolve_paths(ROOT, _cfg).diffusion_ckpt_dir.relative_to(ROOT))
        _wf_cfg = _cfg.get("training", {}).get("walk_forward", {})
        # If the user did NOT pass --wf_epochs explicitly, read from config
        import sys as _sys
        if "--wf_epochs" not in _sys.argv:
            _wf_epochs_default = int(_wf_cfg.get("warmup_epochs", cli.wf_epochs))

    # The lab CLI parser is the source of truth — we extend its argv programmatically
    # so we benefit from every default and validation it already implements.
    argv: list[str] = [
        "--ckpt_dir", _ckpt_dir,
        "--ckpt", "ckpt_wf.pt",
        "--epochs", str(cli.epochs),
        "--wf_epochs", str(_wf_epochs_default),
        "--batch", str(cli.batch),
        "--lr", str(cli.lr),
        "--seed", str(cli.seed),
    ]
    if cli.device:
        argv += ["--device", cli.device]
    if cli.wf_lr is not None:
        argv += ["--wf_lr", str(cli.wf_lr)]
    argv += ["--base_ch", str(cli.base_ch)]
    if cli.cond_drop_prob is not None:
        argv += ["--cond_drop_prob", str(cli.cond_drop_prob)]
    if cli.weight_decay is not None:
        argv += ["--weight_decay", str(cli.weight_decay)]
    if cli.wd_macro is not None:
        argv += ["--wd_macro", str(cli.wd_macro)]

    # Macro encoder dispatch: CLI > config.yaml > "symmetric" fallback
    _enc = cli.macro_encoder
    if _enc is None and _cfg_path.exists():
        _enc = (_cfg.get("diffusion") or {}).get("macro_encoder", None)
    if _enc:
        argv += ["--macro_encoder", str(_enc)]

    # Optional restart: wipe the config-driven checkpoint dir.
    if cli.restart:
        ckpt_dir = ROOT / _ckpt_dir
        if ckpt_dir.exists():
            # Never wipe through a link: a checkpoint dir that resolves outside
            # the project (e.g. a junction to archived weights) is not ours to delete.
            if ROOT.resolve() not in ckpt_dir.resolve().parents:
                sys.exit(f"[train] --restart refused: {ckpt_dir} resolves to "
                         f"{ckpt_dir.resolve()}, outside {ROOT.resolve()}")
            print(f"[train] --restart: removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

    # Hand off to the lab CLI. ``parse_args()`` reads from ``sys.argv``; we replace it.
    sys.argv = ["src.diffusion.lab"] + argv
    from src.diffusion.lab import main as lab_main
    lab_main()


if __name__ == "__main__":
    main()
