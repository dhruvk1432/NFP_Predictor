#!/usr/bin/env python3
"""Phase 0: noise-floor baseline for dynamic-reselection variance investigation.

Runs the same training cell N times (default 5) and reports the MAE sigma
across reps. This measures pure stochastic variance (RNG + system noise) and
sets the bar that any Phase-1..4 mitigation must clear to count as real
signal.

Per the plan (we-have-the-following-enchanted-balloon.md, Phase 0), the
default cell is the worst-affected tier: cap=60, cadence=12, trials=25.

Usage:
    python scripts/phase0_noise_floor.py                  # 5 reps, default cell
    python scripts/phase0_noise_floor.py --reps 3         # fewer reps
    python scripts/phase0_noise_floor.py --summary-only   # skip launching; just read existing reps

Behavior:
- Each rep writes to `_output_noise_floor/rep_<i>/`, isolated from main.
- Reps run sequentially (concurrency would defeat the purpose).
- Resume-safe: a rep whose summary CSV already exists is skipped on relaunch.
- After all reps complete, writes Notes/phase0_noise_floor.md with
  the mean / std / min / max / spread and the gate decision per the plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = REPO_ROOT / "_output_noise_floor"
NOTES_PATH = REPO_ROOT / "Notes" / "phase0_noise_floor.md"

# Default cell: matches the plan's "worst-affected tier" choice.
DEFAULT_CAP = 60
DEFAULT_CADENCE = 12
DEFAULT_TRIALS = 25

# Phase-0 gate: if sigma_MAE exceeds this, the diagnosis premise is broken.
SIGMA_GATE = 5.0


def rep_dir(i: int) -> Path:
    return OUTPUT_BASE / f"rep_{i:02d}"


def summary_csv_path(d: Path) -> Path:
    return d / "consensus_anchor" / "kalman_fusion" / "summary_statistics.csv"


def tuned_params_path(d: Path) -> Path:
    return d / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"


def extract_mae(d: Path) -> float | None:
    """Mirror of grid_search._extract_metrics for the MAE field only."""
    p = summary_csv_path(d)
    if not p.exists():
        return None
    try:
        row = pd.read_csv(p).iloc[0]
        return float(row["MAE"])
    except Exception as e:
        print(f"  WARN: failed to read {p}: {e}")
        return None


def extract_full_metrics(d: Path) -> dict:
    out: dict = {}
    p = summary_csv_path(d)
    if p.exists():
        try:
            row = pd.read_csv(p).iloc[0]
            out["MAE"] = float(row["MAE"])
            out["RMSE"] = float(row["RMSE"])
            out["DirAcc"] = float(row.get("Directional_Accuracy", float("nan")))
        except Exception as e:
            print(f"  WARN: failed to read {p}: {e}")
    j = tuned_params_path(d)
    if j.exists():
        try:
            tp = json.loads(j.read_text())
            out["trailing_window"] = tp.get("trailing_window")
            out["nsa_weight_scale"] = tp.get("nsa_weight_scale")
            out["HL"] = tp.get("half_life_years")
        except Exception as e:
            print(f"  WARN: failed to read {j}: {e}")
    return out


def run_one_rep(i: int, cap: int, cadence: int, trials: int) -> int:
    d = rep_dir(i)
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "run.log"

    if summary_csv_path(d).exists():
        mae = extract_mae(d)
        print(f"[rep {i:02d}] SKIP (already has summary; MAE={mae:.2f})")
        return 0

    env = os.environ.copy()
    env["DYNAMIC_FS_PASS2_MAX_FEATURES"] = str(cap)
    env["RESELECT_EVERY_N_MONTHS"] = str(cadence)
    env["N_OPTUNA_TRIALS"] = str(trials)
    env["OUTPUT_DIR"] = str(d)
    env["TEMP_DIR"] = str(d / "_temp")

    cmd = [sys.executable, "Train/train_lightgbm_nfp.py", "--train-all"]
    started = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[rep {i:02d}] START {started}  log={log_path}")

    t0 = time.time()
    try:
        with open(log_path, "w") as f:
            res = subprocess.run(
                cmd, cwd=str(REPO_ROOT), env=env,
                stdout=f, stderr=subprocess.STDOUT, check=False,
            )
        rc = int(res.returncode)
    except Exception as e:
        print(f"[rep {i:02d}] EXCEPTION: {e}")
        rc = -2
    elapsed_min = (time.time() - t0) / 60.0

    mae = extract_mae(d)
    mae_str = f"MAE={mae:.2f}" if mae is not None else "MAE=N/A"
    print(f"[rep {i:02d}] DONE  rc={rc}  elapsed={elapsed_min:.1f}min  {mae_str}")
    return rc


def summarize_and_write(cap: int, cadence: int, trials: int, n_reps: int) -> None:
    rows = []
    for i in range(1, n_reps + 1):
        d = rep_dir(i)
        m = extract_full_metrics(d)
        if m:
            m["rep"] = i
            rows.append(m)

    if not rows:
        print("No rep results found. Nothing to summarize.")
        return

    df = pd.DataFrame(rows).set_index("rep")
    mae = df["MAE"].dropna()
    n = len(mae)
    if n < 2:
        print(f"Only {n} rep(s) have MAE; need >=2 for sigma. Skipping notes write.")
        return

    mean = float(mae.mean())
    # Sample std (ddof=1), the meaningful "how spread out are my reps" stat
    sigma = float(mae.std(ddof=1))
    mn, mx = float(mae.min()), float(mae.max())
    spread = mx - mn

    # Gate (per plan): if sigma > 5.0, the premise (mitigations can recover
    # cadence gap) is broken because pure-stoch noise already exceeds the gap.
    gate_pass = sigma <= SIGMA_GATE

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0 — noise-floor baseline",
        "",
        f"_Run at {datetime.utcnow().isoformat(timespec='seconds')}Z_",
        "",
        "## Cell config",
        "",
        f"- `DYNAMIC_FS_PASS2_MAX_FEATURES` (cap): **{cap}**",
        f"- `RESELECT_EVERY_N_MONTHS` (cadence): **{cadence}**",
        f"- `N_OPTUNA_TRIALS` (trials): **{trials}**",
        f"- Reps completed: **{n}** of {n_reps}",
        "",
        "## Per-rep results",
        "",
        df.to_markdown(),
        "",
        "## Aggregate",
        "",
        f"- mean MAE: **{mean:.4f}**",
        f"- sigma (sample std, ddof=1): **{sigma:.4f}**",
        f"- min / max: {mn:.4f} / {mx:.4f}",
        f"- spread (max-min): {spread:.4f}",
        "",
        "## Gate (per plan)",
        "",
        f"- Threshold: sigma <= {SIGMA_GATE:.2f}",
        f"- Observed sigma: {sigma:.4f}",
        f"- **{'PASS' if gate_pass else 'FAIL'}** — "
        + (
            "phase 1 may proceed. Mitigations must reduce mean MAE by >= 2 sigma = "
            f"{2*sigma:.2f} to count as real signal."
            if gate_pass
            else "pure-stoch variance exceeds the cadence gap (cad60 87.71 vs cad48 92.57 = 4.86). "
            "Diagnosis premise is broken; do NOT proceed to phase 1 until investigated."
        ),
        "",
    ]
    NOTES_PATH.write_text("\n".join(lines))
    print(f"Wrote {NOTES_PATH}")
    print(f"  mean={mean:.4f}  sigma={sigma:.4f}  spread={spread:.4f}  gate={'PASS' if gate_pass else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 noise-floor baseline")
    ap.add_argument("--reps", type=int, default=5, help="Number of reps (default 5)")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--cadence", type=int, default=DEFAULT_CADENCE)
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument(
        "--summary-only", action="store_true",
        help="Skip launching reps; just read whatever is on disk and write notes."
    )
    args = ap.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for i in range(1, args.reps + 1):
            rc = run_one_rep(i, args.cap, args.cadence, args.trials)
            # Always continue to the next rep even on a non-zero rc — we want
            # to know the spread across whatever reps complete.
            if rc != 0:
                print(f"[rep {i:02d}] non-zero rc; continuing to next rep")

    summarize_and_write(args.cap, args.cadence, args.trials, args.reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
