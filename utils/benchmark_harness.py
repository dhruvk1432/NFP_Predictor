"""
Benchmark harness utilities for baseline-vs-candidate keep-rule evaluation.

The keep rule is delegated to Data_ETA_Pipeline.feature_selection_engine.should_keep_change.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Data_ETA_Pipeline.feature_selection_engine import should_keep_change


def read_mae_from_summary_csv(path: Path, metric_col: str = "MAE") -> float:
    """Read MAE from a summary CSV (single-row or multi-row)."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in {csv_path}")
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {csv_path}")

    value = float(df[metric_col].iloc[-1])
    return value


def run_command_timed(cmd: str, workdir: Path | None = None) -> dict[str, Any]:
    """
    Run command and capture runtime/exit status/stdout/stderr.

    Uses shell-like tokenization via shlex.split without invoking a shell.
    """
    if not cmd.strip():
        raise ValueError("Command cannot be empty")

    start = time.perf_counter()
    proc = subprocess.run(
        shlex.split(cmd),
        cwd=str(workdir) if workdir else None,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_s = float(time.perf_counter() - start)
    return {
        "cmd": cmd,
        "workdir": str(workdir) if workdir else None,
        "returncode": int(proc.returncode),
        "runtime_s": runtime_s,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_keep_rule_report(
    baseline_mae: float,
    candidate_mae: float,
    baseline_runtime_s: float,
    candidate_runtime_s: float,
    min_mae_improvement_pct: float = 0.5,
    min_runtime_improvement_pct: float = 15.0,
    max_mae_loss_for_runtime_pct: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build machine-readable keep-rule decision report."""
    mae_improvement_pct = ((baseline_mae - candidate_mae) / abs(baseline_mae)) * 100.0
    runtime_improvement_pct = (
        (baseline_runtime_s - candidate_runtime_s) / abs(baseline_runtime_s)
    ) * 100.0

    keep = should_keep_change(
        baseline_mae=baseline_mae,
        candidate_mae=candidate_mae,
        baseline_runtime_s=baseline_runtime_s,
        candidate_runtime_s=candidate_runtime_s,
        min_mae_improvement_pct=min_mae_improvement_pct,
        min_runtime_improvement_pct=min_runtime_improvement_pct,
        max_mae_loss_for_runtime_pct=max_mae_loss_for_runtime_pct,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "keep_rule_report_v1",
        "thresholds": {
            "min_mae_improvement_pct": float(min_mae_improvement_pct),
            "min_runtime_improvement_pct": float(min_runtime_improvement_pct),
            "max_mae_loss_for_runtime_pct": float(max_mae_loss_for_runtime_pct),
        },
        "baseline": {
            "mae": float(baseline_mae),
            "runtime_s": float(baseline_runtime_s),
        },
        "candidate": {
            "mae": float(candidate_mae),
            "runtime_s": float(candidate_runtime_s),
        },
        "deltas_pct": {
            "mae_improvement_pct": float(mae_improvement_pct),
            "runtime_improvement_pct": float(runtime_improvement_pct),
        },
        "decision": {
            "keep": bool(keep),
            "label": "accept" if keep else "reject",
        },
        "metadata": metadata or {},
    }
    return report


def save_report(report: dict[str, Any], output_path: Path) -> Path:
    """Persist keep-rule report to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return out
