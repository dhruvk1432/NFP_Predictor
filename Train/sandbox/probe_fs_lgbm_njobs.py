"""
Sandbox probe: is the feature-selection Boruta loop CPU-constrained, and what
``n_jobs`` value can we safely bump it to on this machine?

Why this script exists
----------------------
``Data_ETA_Pipeline.feature_selection_engine.LGB_PARAMS`` historically pinned
LightGBM ``n_jobs=1`` because a prior incident (LightGBM + ProcessPoolExecutor
+ ``n_jobs=-1``) deadlocked on macOS. The fix was correct but conservative:
on Linux/CI hosts (and possibly on this developer machine) Boruta is purely
CPU-bound and would benefit from multiple threads.

This probe sweeps small ``n_jobs`` candidates, runs a representative Boruta
workload in a *subprocess* (so deadlocks surface as timeouts rather than
hanging the probe), and records wall-clock + peak RSS. The output lets you
choose a safe production default for ``FS_LGBM_NJOBS`` empirically.

The probe is deliberately self-contained (no real master snapshots required)
so it can run quickly on any checkout.

Usage
-----
    python Train/sandbox/probe_fs_lgbm_njobs.py
        # synthetic 500x400 workload, sweeps n_jobs in [1, 2, 4]
        # writes _output/sandbox/fs_njobs_probe/report.csv

    python Train/sandbox/probe_fs_lgbm_njobs.py --candidates 1 2 4 8
        # custom sweep

    python Train/sandbox/probe_fs_lgbm_njobs.py --timeout 300 --runs 3
        # 5-minute watchdog, average over 3 repetitions per candidate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)


OUT_DIR = OUTPUT_DIR / "sandbox" / "fs_njobs_probe"
DEFAULT_CANDIDATES = [1, 2, 4]
DEFAULT_TIMEOUT_S = 180
DEFAULT_RUNS = 1


# ---------------------------------------------------------------------------
# Child-process worker
# ---------------------------------------------------------------------------

def _child_main() -> None:
    """Run a single Boruta benchmark and print a one-line JSON result.

    Invoked by the parent with ``--child`` and a JSON-encoded fixture spec
    on argv. Honors ``NFP_FS_LGBM_NJOBS`` (set by parent before spawn).
    """
    spec_path = Path(sys.argv[sys.argv.index("--spec") + 1])
    spec = json.loads(spec_path.read_text())
    seed = int(spec["seed"])
    n_samples = int(spec["n_samples"])
    n_features = int(spec["n_features"])
    n_runs = int(spec["n_runs"])

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f_{i:04d}" for i in range(n_features)],
    )
    # Embed a few real signals so Boruta has informative features to find
    coef = rng.standard_normal(min(10, n_features))
    y_arr = X.iloc[:, :len(coef)].values @ coef + rng.standard_normal(n_samples) * 0.5
    y = pd.Series(y_arr, name='y_mom')

    # Use a DatetimeIndex on X so we exercise the same code paths Boruta sees
    X.index = pd.date_range("2010-01-01", periods=n_samples, freq="MS")
    y.index = X.index

    import resource
    from Data_ETA_Pipeline.feature_selection_engine import (
        FS_LGBM_NJOBS,
        get_boruta_importance,
    )

    t0 = time.perf_counter()
    try:
        survivors = get_boruta_importance(X, y, n_runs=n_runs)
        status = "ok"
        n_survivors = len(survivors)
        error = ""
    except Exception as e:
        status = "error"
        n_survivors = 0
        error = repr(e)
    elapsed_s = time.perf_counter() - t0

    # ru_maxrss is KB on Linux, bytes on macOS — we report bytes-normalized MB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss_mb = rss / (1024 * 1024)
    else:
        peak_rss_mb = rss / 1024

    print(json.dumps({
        "status": status,
        "elapsed_s": elapsed_s,
        "peak_rss_mb": peak_rss_mb,
        "n_survivors": n_survivors,
        "resolved_fs_lgbm_njobs": FS_LGBM_NJOBS,
        "error": error,
    }))


# ---------------------------------------------------------------------------
# Parent driver
# ---------------------------------------------------------------------------

def _run_one_candidate(
    n_jobs: int,
    spec_path: Path,
    timeout_s: int,
) -> dict:
    env = dict(os.environ)
    env["NFP_FS_LGBM_NJOBS"] = str(n_jobs)
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--child", "--spec", str(spec_path),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            return {
                "n_jobs": n_jobs,
                "status": "child_nonzero_exit",
                "elapsed_s": wall,
                "peak_rss_mb": float("nan"),
                "n_survivors": 0,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            }
        # Last non-empty stdout line is our JSON
        out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
        if not out_lines:
            return {
                "n_jobs": n_jobs,
                "status": "no_output",
                "elapsed_s": wall,
                "peak_rss_mb": float("nan"),
                "n_survivors": 0,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            }
        result = json.loads(out_lines[-1])
        result["n_jobs"] = n_jobs
        return result
    except subprocess.TimeoutExpired:
        return {
            "n_jobs": n_jobs,
            "status": "timeout",
            "elapsed_s": float(timeout_s),
            "peak_rss_mb": float("nan"),
            "n_survivors": 0,
            "stderr_tail": "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--candidates", type=int, nargs="+", default=DEFAULT_CANDIDATES,
        help=f"n_jobs values to sweep (default: {DEFAULT_CANDIDATES})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_S,
        help=f"per-child timeout in seconds (default: {DEFAULT_TIMEOUT_S}). "
             "Deadlocks surface as timeouts.",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"repetitions per candidate, mean wall-clock reported (default: {DEFAULT_RUNS})",
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=400)
    parser.add_argument(
        "--boruta-runs", type=int, default=5,
        help="n_runs passed to get_boruta_importance (default: 5, sandbox-fast)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.child:
        _child_main()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_runs": args.boruta_runs,
    }
    spec_path = OUT_DIR / "fixture_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    logger.info(
        f"Probe sweep: candidates={args.candidates}, runs={args.runs}, "
        f"timeout={args.timeout}s, fixture={args.n_samples}x{args.n_features}, "
        f"boruta n_runs={args.boruta_runs}"
    )

    rows = []
    for nj in args.candidates:
        elapsed_samples = []
        last_result = None
        for r in range(args.runs):
            res = _run_one_candidate(nj, spec_path, args.timeout)
            last_result = res
            status = res["status"]
            elapsed = res.get("elapsed_s", float("nan"))
            logger.info(
                f"  n_jobs={nj} run={r+1}/{args.runs} status={status} "
                f"elapsed_s={elapsed:.2f} peak_rss_mb={res.get('peak_rss_mb', float('nan')):.1f}"
            )
            if status == "ok":
                elapsed_samples.append(elapsed)
            else:
                # Stop retrying on failure — likely persistent
                break
        if elapsed_samples:
            mean_elapsed = float(np.mean(elapsed_samples))
        else:
            mean_elapsed = float("nan")
        rows.append({
            "n_jobs": nj,
            "status": last_result["status"] if last_result else "unrun",
            "mean_elapsed_s": mean_elapsed,
            "successful_runs": len(elapsed_samples),
            "peak_rss_mb": last_result.get("peak_rss_mb", float("nan")) if last_result else float("nan"),
            "n_survivors": last_result.get("n_survivors", 0) if last_result else 0,
            "resolved_fs_lgbm_njobs": last_result.get("resolved_fs_lgbm_njobs", None) if last_result else None,
            "stderr_tail": last_result.get("stderr_tail", "") if last_result else "",
            "error": last_result.get("error", "") if last_result else "",
        })

    df = pd.DataFrame(rows)

    # Speedup column relative to n_jobs=1 (if observed and successful)
    base = df.loc[(df["n_jobs"] == 1) & (df["status"] == "ok"), "mean_elapsed_s"]
    if len(base) == 1 and not np.isnan(base.iloc[0]) and base.iloc[0] > 0:
        df["speedup_vs_n1"] = base.iloc[0] / df["mean_elapsed_s"]
    else:
        df["speedup_vs_n1"] = float("nan")

    out_csv = OUT_DIR / "report.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv}")
    print()
    print(df.to_string(index=False))
    print()
    print("Decision guide:")
    print("  - If speedup_vs_n1 >= 1.5 at n_jobs=2 AND status='ok' for all candidates,")
    print("    the safe production default is the highest candidate without a timeout.")
    print("  - If speedup is <1.2x even at higher n_jobs, this machine is not CPU-")
    print("    constrained on Boruta — leave n_jobs=1 in the default.")
    print("  - Any 'timeout' or 'child_nonzero_exit' on a candidate means do NOT")
    print("    raise the default to that value.")


if __name__ == "__main__":
    main()
