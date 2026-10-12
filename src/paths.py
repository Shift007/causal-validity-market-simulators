"""Resolve the three I/O roots declared under `paths:` in config.yaml.

A config carries one experiment's layout:
    paths:
      checkpoints_dir: "checkpoints/synthetic"
      processed_dir:   "processed_data/synthetic"
      results_dir:     "results/synthetic"

Inside `checkpoints_dir` the canonical layout is
    diffusion/    ← U-Net + walk-forward ckpts
    macrovae/     ← baseline ckpts
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


_DEFAULTS = {
    "checkpoints_dir": "checkpoints/synthetic",
    "processed_dir":   "processed_data/synthetic",
    "results_dir":     "results/synthetic",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """Absolute paths resolved against ``project_root``."""
    project_root: Path
    checkpoints_dir: Path
    processed_dir:   Path
    results_dir:     Path

    @property
    def diffusion_ckpt_dir(self) -> Path:
        return self.checkpoints_dir / "diffusion"

    @property
    def macrovae_ckpt_dir(self) -> Path:
        return self.checkpoints_dir / "macrovae"

    def task_results_dir(self, task_name: str) -> Path:
        return self.results_dir / task_name

    def ensure_exists(self) -> None:
        for d in (self.checkpoints_dir, self.diffusion_ckpt_dir,
                  self.macrovae_ckpt_dir, self.processed_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


def resolve_paths(project_root: Path, config: Dict[str, Any]) -> ExperimentPaths:
    """Read `paths:` from config and return absolute paths.

    Falls back to synthetic defaults if the block is missing; the shipped
    configs always set it.
    """
    paths_cfg = (config or {}).get("paths", {}) or {}
    return ExperimentPaths(
        project_root    = project_root,
        checkpoints_dir = project_root / paths_cfg.get("checkpoints_dir", _DEFAULTS["checkpoints_dir"]),
        processed_dir   = project_root / paths_cfg.get("processed_dir",   _DEFAULTS["processed_dir"]),
        results_dir     = project_root / paths_cfg.get("results_dir",     _DEFAULTS["results_dir"]),
    )
