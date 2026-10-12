"""Human-readable per-run summary writer for evaluation scripts.

Each evaluation run already drops `manifest.json` + `metrics.json` (or
`metrics_report.json`) in its `run_dir`. This module adds a single
`summary_<timestamp>.txt` file that bundles config + metrics + the
ckpt paths actually used, so a run can be audited
without parsing JSON.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def _fmt_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.6e}"
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fmt_metrics_block(metrics: Mapping[str, Any], indent: int = 0) -> str:
    """Pretty-print arbitrarily nested metrics (dict/list/scalar)."""
    pad = "  " * indent
    lines: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_fmt_metrics_block(v, indent + 1))
        elif isinstance(v, (list, tuple)):
            if v and all(isinstance(x, (int, float)) for x in v):
                preview = ", ".join(_fmt_scalar(x) for x in v[:8])
                tail = "" if len(v) <= 8 else f", … ({len(v)} total)"
                lines.append(f"{pad}{k}: [{preview}{tail}]")
            else:
                lines.append(f"{pad}{k}: {v}")
        else:
            lines.append(f"{pad}{k}: {_fmt_scalar(v)}")
    return "\n".join(lines)


def write_run_summary(
    run_dir: Path,
    *,
    task: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    checkpoint_paths: Optional[Mapping[str, str]] = None,
    elapsed_seconds: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write `summary_<YYYYmmdd_HHMMSS>.txt` inside ``run_dir``.

    Returns the path of the written file. Re-writes are timestamped fresh
    every call (so a script can be re-run and still produce a new summary).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"summary_{ts}.txt"

    bar = "=" * 78
    chunks: list[str] = []
    chunks.append(bar)
    chunks.append(f"RUN     : {datetime.now().isoformat(timespec='seconds')}")
    chunks.append(f"SCRIPT  : {task}")
    chunks.append(f"RUN DIR : {run_dir}")
    if elapsed_seconds is not None:
        chunks.append(f"ELAPSED : {elapsed_seconds:.1f} s")
    chunks.append(bar)

    if checkpoint_paths:
        chunks.append("")
        chunks.append("=== CHECKPOINTS USED ===")
        for k, v in checkpoint_paths.items():
            chunks.append(f"  {k}: {v}")

    if extra:
        chunks.append("")
        chunks.append("=== RUN PARAMETERS ===")
        chunks.append(_fmt_metrics_block(extra, indent=1))

    chunks.append("")
    chunks.append("=== CONFIG (full snapshot) ===")
    chunks.append(yaml.safe_dump(dict(config), sort_keys=False,
                                 default_flow_style=False).rstrip())

    chunks.append("")
    chunks.append("=== METRICS ===")
    chunks.append(_fmt_metrics_block(metrics, indent=1))
    chunks.append("")
    chunks.append(bar)

    out_path.write_text("\n".join(chunks), encoding="utf-8")
    return out_path
