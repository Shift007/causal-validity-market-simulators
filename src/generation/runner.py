"""Shared generation infrastructure: model loading, disk management, manifest."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml


# ─── Paths ────────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Resolve project root from CWD or known markers."""
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "config.yaml").exists():
        return cwd.parent
    if cwd.name == "scripts" and (cwd.parent / "config.yaml").exists():
        return cwd.parent
    raise FileNotFoundError(
        "Cannot find project root (config.yaml). "
        "Run from the project root, notebooks/, or scripts/ directory."
    )


def load_config(project_root: Path | None = None) -> dict:
    root = project_root or get_project_root()
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Run directory & latest tracking ──────────────────────────────────────────

def create_run_dir(base_dir: Path, task_name: str) -> Path:
    """Create timestamped run directory under base_dir/task_name/."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / task_name / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_latest(base_dir: Path, task_name: str, run_dir: Path) -> None:
    """Write latest.txt pointing to the current run."""
    latest_file = base_dir / task_name / "latest.txt"
    latest_file.write_text(run_dir.name)


def read_latest(base_dir: Path, task_name: str) -> Path | None:
    """Read latest run directory, or None if not found."""
    latest_file = base_dir / task_name / "latest.txt"
    if not latest_file.exists():
        return None
    run_name = latest_file.read_text().strip()
    run_dir = base_dir / task_name / run_name
    return run_dir if run_dir.exists() else None


def prune_old_runs(base_dir: Path, task_name: str, max_runs: int) -> list[str]:
    """Delete old runs exceeding max_runs. Returns list of deleted dir names.

    Disabled: run directories are the provenance of reported numbers, so they
    are never deleted automatically. ``max_runs_per_task`` in configs is inert.
    """
    return []
    task_dir = base_dir / task_name
    if not task_dir.exists():
        return []
    runs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    if len(runs) <= max_runs:
        return []
    to_delete = runs[: len(runs) - max_runs]
    deleted = []
    for d in to_delete:
        def _onerror(func, path, exc_info):
            import warnings
            warnings.warn(f"[prune_old_runs] skipped locked file: {path}")
        shutil.rmtree(d, onerror=_onerror)
        deleted.append(d.name)
    return deleted


# ─── Manifest ─────────────────────────────────────────────────────────────────

def config_hash(config: dict) -> str:
    """Deterministic hash of the config dict (for provenance)."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def save_manifest(
    run_dir: Path,
    *,
    task: str,
    config: dict,
    checkpoint_paths: dict[str, str],
    shapes: dict[str, tuple],
    extra: dict[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
) -> Path:
    """Save manifest.json with provenance information."""
    manifest = {
        "task": task,
        "timestamp": _dt.datetime.now().isoformat(),
        "config_hash": config_hash(config),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "checkpoint_paths": checkpoint_paths,
        "shapes": {k: list(v) for k, v in shapes.items()},
    }
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def load_manifest(run_dir: Path) -> dict:
    with open(run_dir / "manifest.json", "r") as f:
        return json.load(f)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model loaders ────────────────────────────────────────────────────────────

def load_mvae_model(ckpt_path: Path, n_assets: int, target_len: int,
                    D_mvae: int, device: torch.device):
    """Load MacroVAE from checkpoint. Returns (model, state_dict, hyperparams)."""
    mvae_path = str(Path(__file__).resolve().parent.parent.parent / "MacroVAE")
    if mvae_path not in sys.path:
        sys.path.insert(0, mvae_path)
    from src.macrovae_compat import MacroVAE

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = state.get("hyperparameters", {})

    model = MacroVAE(
        input_dim=hp.get("input_dim", n_assets),
        sequence_length=hp.get("sequence_length", target_len),
        latent_dim=hp.get("latent_dim", 32),
        hidden_dim=hp.get("hidden_dim", 128),
        condition_dim=hp.get("condition_dim", D_mvae),
        condition_embedding_dim=hp.get("condition_embedding_dim", 16),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state, hp


# ─── Preprocessing artifact loader ──────────────────────────────────────────

def load_preprocessing_artifacts(data_dir: Path):
    """Load preprocessing artifacts from dataset/. Returns dict with all arrays + params."""
    raw_windows = np.load(data_dir / "raw_windows.npy")
    macro_windows = np.load(data_dir / "macro_windows.npy")
    train_mask = np.load(data_dir / "train_mask.npy")
    norm_params = np.load(
        data_dir / "normalization_params.npy", allow_pickle=True
    ).item()
    window_dates = np.load(data_dir / "window_dates.npy", allow_pickle=True)

    # MacroVAE z-scored macro (required)
    mvae_macro_path = data_dir / "macro_windows_zscore.npy"
    assert mvae_macro_path.exists(), (
        f"macro_windows_zscore.npy not found in {data_dir}.\n"
        "Run scripts/preprocess.py to generate it."
    )
    mvae_macro_windows = np.load(mvae_macro_path)

    # MacroVAE expanding stats for counterfactual de-/re-standardization
    mvae_exp_mean_path = data_dir / "mvae_macro_exp_mean.npy"
    mvae_exp_std_path = data_dir / "mvae_macro_exp_std.npy"
    mvae_exp_mean = np.load(mvae_exp_mean_path) if mvae_exp_mean_path.exists() else None
    mvae_exp_std = np.load(mvae_exp_std_path) if mvae_exp_std_path.exists() else None

    n_assets = int(norm_params.get("n_assets", raw_windows.shape[2]))
    context_len = int(norm_params["context_len"])
    target_len = int(norm_params["target_len"])
    wavelet_type = str(norm_params["wavelet_type"])
    wavelet_level = int(norm_params["wavelet_level"])
    tickers = list(norm_params.get("tickers", []))[:n_assets]

    # QT + level_scales
    qt_path = data_dir / "target_quantile_transformer.pkl"
    target_qt = joblib.load(qt_path) if qt_path.exists() else None
    level_scales = np.load(data_dir / "level_scales.npy")

    return {
        "raw_windows": raw_windows,
        "macro_windows": macro_windows,
        "mvae_macro_windows": mvae_macro_windows,
        "mvae_exp_mean": mvae_exp_mean,
        "mvae_exp_std": mvae_exp_std,
        "train_mask": train_mask,
        "norm_params": norm_params,
        "window_dates": window_dates,
        "target_qt": target_qt,
        "level_scales": level_scales,
        "n_assets": n_assets,
        "context_len": context_len,
        "target_len": target_len,
        "wavelet_type": wavelet_type,
        "wavelet_level": wavelet_level,
        "n_levels": wavelet_level + 1,
        "tickers": tickers,
        "D_mvae": mvae_macro_windows.shape[2],
        "D_base": norm_params.get("mvae_d_base", mvae_macro_windows.shape[2] // 2),
    }


# ─── Walk-forward schedule ────────────────────────────────────────────────────

def build_eval_schedule(config: dict, window_dates_ts=None):
    """Build walk-forward evaluation schedule from config.

    Returns list of (label, eval_start, eval_end, ckpt_stem) tuples.
    """
    import pandas as pd

    wf_cfg = config["training"]["walk_forward"]
    initial_split = pd.Timestamp(wf_cfg["initial_split_date"])
    eval_end_date = pd.Timestamp(wf_cfg["evaluation_end_date"])
    step_freq = str(wf_cfg.get("step_freq", "Q")).upper()

    freq_map = {"Q": "QS", "3M": "QS", "6M": "6MS", "H": "6MS", "A": "YS", "Y": "YS"}
    pd_freq = freq_map.get(step_freq, "QS")

    period_starts = pd.date_range(start=initial_split, end=eval_end_date, freq=pd_freq)
    period_starts = period_starts[period_starts > initial_split]

    def _label(dt):
        if step_freq in ("Q", "3M"):
            return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
        elif step_freq in ("6M", "H"):
            return f"{dt.year}H{1 if dt.month <= 6 else 2}"
        return str(dt.year)

    def _period_end(dt):
        if step_freq in ("Q", "3M"):
            q = (dt.month - 1) // 3 + 1
            return pd.Timestamp(f"{dt.year}-{q * 3:02d}-01") + pd.offsets.MonthEnd(0)
        elif step_freq in ("6M", "H"):
            q = 2 if dt.month <= 6 else 4
            return pd.Timestamp(f"{dt.year}-{q * 3:02d}-01") + pd.offsets.MonthEnd(0)
        return pd.Timestamp(f"{dt.year}-12-31")

    schedule = []
    prev_label = "initial"
    for pstart in period_starts:
        label = _label(pstart)
        end = min(_period_end(pstart), eval_end_date)
        schedule.append((label, pstart, end, f"ckpt_wf_{prev_label}"))
        if end >= eval_end_date:
            break
        prev_label = label

    return schedule


# ─── MacroVAE decode helper ──────────────────────────────────────────────────

@torch.no_grad()
def generate_mvae(model, cond_np: np.ndarray, n_traj: int,
                  latent_dim: int, target_len: int, n_assets: int,
                  device: torch.device, batch_gen: int = 128) -> np.ndarray:
    """Generate MacroVAE synthetic returns.

    Parameters
    ----------
    cond_np : (B, D_mvae)
    n_traj : trajectories per conditioning row

    Returns
    -------
    synth : (B, n_traj, target_len, C) raw returns
    """
    B = len(cond_np)
    all_synth = []
    _vol_accum = []
    _mean_accum = []

    for start in range(0, B, batch_gen):
        end = min(start + batch_gen, B)
        cond_t = torch.from_numpy(cond_np[start:end].astype(np.float32)).to(device)
        b_size = end - start
        cond_exp = cond_t.repeat_interleave(n_traj, dim=0)
        z = torch.randn(b_size * n_traj, latent_dim, device=device)

        norm_ret, vol_out, mean_out = model.decode(z, cond_exp)
        vol_scalar = vol_out[:, 0, :].unsqueeze(1)
        synth = (norm_ret + mean_out) * torch.exp(vol_scalar)
        all_synth.append(
            synth.cpu().numpy().reshape(b_size, n_traj, target_len, n_assets)
        )
        _vol_accum.append(vol_out[:, 0, :].cpu().numpy())
        _mean_accum.append(mean_out[:, 0, :].cpu().numpy())

    # Diagnostic: print vol/mean head output summary (first call only)
    all_vol = np.concatenate(_vol_accum, axis=0)
    all_mean = np.concatenate(_mean_accum, axis=0)
    print(f"    vol_head: mean={all_vol.mean():.3f} std={all_vol.std():.3f} "
          f"→ exp(vol)={np.exp(all_vol).mean():.5f}")
    print(f"    mean_head: mean={all_mean.mean():.5f} std={all_mean.std():.5f}")

    return np.concatenate(all_synth, axis=0)


