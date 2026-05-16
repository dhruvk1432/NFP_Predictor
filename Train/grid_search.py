"""
Grid search over (DYNAMIC_FS_PASS2_MAX_FEATURES, RESELECT_EVERY_N_MONTHS,
N_OPTUNA_TRIALS) for the Kalman fusion pipeline.

For each cell, launches a full ``--train-all`` subprocess with that cell's
config overridden via env vars. Captures the Kalman fusion MAE, RMSE,
AccelAcc, DirAcc, and tuned (HL, trailing_window, nsa_weight_scale).
Writes a cumulative results CSV after every cell so the grid can be
inspected mid-run.

Cells are ordered by ascending wall-clock cost (largest cadence first ⇒
fewest reselections ⇒ fastest), so cheap signal comes in early. The grid
supports **resume**: a cell whose row already exists in the results CSV
is skipped. Killing and restarting picks up where it left off.

Output layout:
  _output_grid/
    grid_results.csv                            # cumulative summary
    grid_master_YYYYMMDD_HHMMSS.log             # master log
    cell_NN_cap{C}_cad{D}_t{T}/
      <full _output tree for this cell>
      run.log                                   # subprocess log

Configuration: edit ``CAP_VALUES``, ``CADENCE_VALUES``, ``TRIALS_VALUES``
near the top of this file before launching. Defaults below are tuned for
~1.5-2 day overnight run on AWS m7i.4xlarge ≈ $35-45.

Usage (local or on AWS instance):
    python Train/grid_search.py
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# GRID SPEC — edit before launch.
# =============================================================================
# Step 3 baseline (proven good): cap=80, cadence=24, trials=25 → MAE=94.45.
# Grid extends in all three dimensions to find the optimum.
CAP_VALUES: list[int] = [60, 80, 100, 120]              # 4
CADENCE_VALUES: list[int] = [6, 12, 18, 24]              # 4
TRIALS_VALUES: list[int] = [25, 50]                       # 2
# Total cells: 4 × 4 × 2 = 32

# Order cells by descending cadence so cheap cells run first (fewer reselections
# ⇒ faster), giving us partial signal early in the run.
ORDER_KEY = "cadence_desc"  # one of {"cadence_desc", "cadence_asc", "linear"}

# Per-cell timeout (seconds). 6 hours is generous; cadence=6 with 60-month
# backtest realistically takes 3-4 hours.
PER_CELL_TIMEOUT_S: int = 6 * 3600

# Parallelism. Defaults sized for m7i.4xlarge (16 vCPU) — 3 cells × 4 cores
# = 12 cores, leaving 4 cores for the orchestrator + kernel + small slop.
# Each cell already saturates ~4 cores in practice (LightGBM + Python
# orchestration), so 3 in parallel ~= 3x wall-clock speedup.
# Override at launch:  GRID_PARALLEL=4 GRID_CORES_PER_CELL=4 python Train/grid_search.py
MAX_PARALLEL_CELLS: int = int(os.getenv("GRID_PARALLEL", "3"))
CORES_PER_CELL:     int = int(os.getenv("GRID_CORES_PER_CELL", "4"))
# =============================================================================

GRID_DIR = REPO_ROOT / "_output_grid"
RESULTS_CSV = GRID_DIR / "grid_results.csv"
# touch this file to stop the grid cleanly between cells (running cells will
# finish, no new cells will be submitted). Useful for "drain" shutdowns.
STOP_FILE = GRID_DIR / "STOP"

# Columns expected in grid_results.csv. Anything missing is filled with NaN.
RESULT_COLS = [
    "idx", "cap", "cadence", "trials",
    "MAE", "RMSE", "AccelAcc", "DirAcc",
    "HL", "trailing_window", "nsa_weight_scale",
    "elapsed_min", "exit_code", "cell_dir",
    "started_at", "finished_at",
]


# --- parallel infrastructure ----------------------------------------------
# CSV append + stdout prints are serialized so parallel cells don't interleave
# inside a single line / corrupt the header detection.
_results_lock = threading.Lock()
_print_lock   = threading.Lock()

def _safe_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)

# Slot queue gives each running cell a unique CPU range via taskset, so the
# kernel scheduler doesn't shuffle hot threads between cores.
_slots_q: "queue.Queue[int]" = queue.Queue()
for _s in range(MAX_PARALLEL_CELLS):
    _slots_q.put(_s)

def _cpu_range_for_slot(slot: int) -> str:
    start = slot * CORES_PER_CELL
    return f"{start}-{start + CORES_PER_CELL - 1}"


def _cell_id(idx: int, cap: int, cad: int, trials: int) -> str:
    return f"cell_{idx:02d}_cap{cap}_cad{cad}_t{trials}"


def _build_cell_list() -> list[dict]:
    cells = []
    for cap, cad, trials in itertools.product(CAP_VALUES, CADENCE_VALUES, TRIALS_VALUES):
        cells.append({"cap": int(cap), "cadence": int(cad), "trials": int(trials)})

    if ORDER_KEY == "cadence_desc":
        cells.sort(key=lambda c: (-c["cadence"], c["cap"], c["trials"]))
    elif ORDER_KEY == "cadence_asc":
        cells.sort(key=lambda c: (c["cadence"], c["cap"], c["trials"]))
    # else: leave as itertools.product order

    for i, c in enumerate(cells):
        c["idx"] = i
    return cells


def _load_done_indices() -> set[int]:
    if not RESULTS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return set()
    if "idx" not in df.columns:
        return set()
    return set(int(x) for x in df["idx"].dropna().tolist())


def _append_result(row: dict) -> None:
    # Ensure column order is stable. Serialized so parallel cells don't race.
    with _results_lock:
        out = {c: row.get(c) for c in RESULT_COLS}
        header = not RESULTS_CSV.exists()
        pd.DataFrame([out]).to_csv(RESULTS_CSV, mode="a", index=False, header=header)


def _extract_metrics(cell_dir: Path) -> dict:
    metrics: dict = {}
    csv_path = cell_dir / "consensus_anchor" / "kalman_fusion" / "summary_statistics.csv"
    json_path = cell_dir / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"

    if csv_path.exists():
        try:
            row = pd.read_csv(csv_path).iloc[0]
            metrics["MAE"] = float(row["MAE"])
            metrics["RMSE"] = float(row["RMSE"])
            metrics["AccelAcc"] = float(row["Acceleration_Accuracy"])
            metrics["DirAcc"] = float(row["Directional_Accuracy"])
        except Exception as e:
            print(f"  WARN: failed to read {csv_path}: {e}")

    if json_path.exists():
        try:
            tp = json.loads(json_path.read_text())
            metrics["HL"] = tp.get("half_life_years")
            metrics["trailing_window"] = tp.get("trailing_window")
            metrics["nsa_weight_scale"] = tp.get("nsa_weight_scale")
        except Exception as e:
            print(f"  WARN: failed to read {json_path}: {e}")

    return metrics


def _run_one_cell(cell: dict) -> dict:
    cid = _cell_id(cell["idx"], cell["cap"], cell["cadence"], cell["trials"])
    cell_dir = (GRID_DIR / cid).resolve()
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Acquire a CPU slot for this cell (blocks if all slots are busy).
    slot = _slots_q.get()
    cpu_range = _cpu_range_for_slot(slot)

    started_at = datetime.utcnow().isoformat(timespec="seconds")
    t0 = time.time()

    env = os.environ.copy()
    env["DYNAMIC_FS_PASS2_MAX_FEATURES"] = str(cell["cap"])
    env["RESELECT_EVERY_N_MONTHS"] = str(cell["cadence"])
    env["N_OPTUNA_TRIALS"] = str(cell["trials"])
    env["OUTPUT_DIR"] = str(cell_dir)
    env["TEMP_DIR"] = str(cell_dir / "_temp")
    # Cap thread pools per cell so several cells can coexist on one box.
    # LightGBM, joblib, OpenBLAS, MKL, numexpr, BLIS all honor at least one
    # of these. n_jobs=-1 in DEFAULT_LGBM_PARAMS falls back to
    # omp_get_max_threads(), which is bounded by OMP_NUM_THREADS.
    env["OMP_NUM_THREADS"]      = str(CORES_PER_CELL)
    env["OPENBLAS_NUM_THREADS"] = str(CORES_PER_CELL)
    env["MKL_NUM_THREADS"]      = str(CORES_PER_CELL)
    env["NUMEXPR_NUM_THREADS"]  = str(CORES_PER_CELL)
    env["BLIS_NUM_THREADS"]     = str(CORES_PER_CELL)

    log_file = cell_dir / "run.log"
    _safe_print(f"[grid {datetime.utcnow().strftime('%H:%M:%S')}] START {cid}  "
                f"slot={slot} cpus={cpu_range}")
    _safe_print(f"  [{cid}] cap={cell['cap']} cadence={cell['cadence']} "
                f"trials={cell['trials']}  log={log_file}")

    exit_code = None
    try:
        with open(log_file, "w") as f:
            # taskset pins this subprocess (and its threads, via affinity
            # inheritance) to the assigned core range, so OpenMP's auto-
            # detection lines up with our budget and we don't oversubscribe.
            cmd = ["taskset", "-c", cpu_range,
                   sys.executable, "Train/train_lightgbm_nfp.py", "--train-all"]
            res = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=PER_CELL_TIMEOUT_S,
                check=False,
            )
        exit_code = int(res.returncode)
    except subprocess.TimeoutExpired:
        _safe_print(f"  [{cid}] TIMEOUT after {PER_CELL_TIMEOUT_S}s")
        exit_code = -1
    except Exception as e:
        _safe_print(f"  [{cid}] EXCEPTION: {e}")
        exit_code = -2
    finally:
        _slots_q.put(slot)

    elapsed_min = (time.time() - t0) / 60.0
    finished_at = datetime.utcnow().isoformat(timespec="seconds")

    metrics = _extract_metrics(cell_dir)

    row = {
        "idx": cell["idx"],
        "cap": cell["cap"],
        "cadence": cell["cadence"],
        "trials": cell["trials"],
        "elapsed_min": round(elapsed_min, 1),
        "exit_code": exit_code,
        "cell_dir": str(cell_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        **metrics,
    }

    mae = metrics.get("MAE")
    mae_str = f"MAE={mae:.2f}" if mae is not None else "MAE=N/A"
    _safe_print(f"[grid {datetime.utcnow().strftime('%H:%M:%S')}] END   {cid}  "
                f"exit={exit_code}  elapsed={elapsed_min:.1f}min  {mae_str}")
    return row


def main() -> int:
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    cells = _build_cell_list()
    done_idx = _load_done_indices()

    print("=" * 72)
    print(f"Grid search: {len(cells)} cells")
    print(f"  cap     ∈ {CAP_VALUES}")
    print(f"  cadence ∈ {CADENCE_VALUES}")
    print(f"  trials  ∈ {TRIALS_VALUES}")
    print(f"  order:   {ORDER_KEY}")
    print(f"  per-cell timeout: {PER_CELL_TIMEOUT_S // 3600}h")
    print(f"  parallelism: up to {MAX_PARALLEL_CELLS} cells × {CORES_PER_CELL} "
          f"cores each (= {MAX_PARALLEL_CELLS * CORES_PER_CELL} cores total)")
    print(f"  resume: skipping {len(done_idx)} already-done cells")
    print(f"  output: {GRID_DIR}")
    print(f"  stop:   `touch {STOP_FILE}` halts new submissions; "
          f"in-flight cells finish.")
    print("=" * 72)

    # Print SKIP messages up-front so the parallel output below is clean.
    for cell in cells:
        if cell["idx"] in done_idx:
            cid = _cell_id(cell["idx"], cell["cap"], cell["cadence"], cell["trials"])
            print(f"SKIP cell {cell['idx']:>2}  {cid}  (already in {RESULTS_CSV.name})")

    remaining = [c for c in cells if c["idx"] not in done_idx]
    if not remaining:
        print("All cells already done.")
        return 0

    print(f"Submitting {len(remaining)} cells, up to {MAX_PARALLEL_CELLS} in parallel.")

    overall_t0 = time.time()
    futures: dict = {}
    n_submitted = 0
    n_aborted_before_submit = 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CELLS) as ex:
        for c in remaining:
            if STOP_FILE.exists():
                n_aborted_before_submit = len(remaining) - n_submitted
                print(f"STOP file present at {STOP_FILE}; halting "
                      f"submission. {n_aborted_before_submit} cells unsubmitted.")
                break
            futures[ex.submit(_run_one_cell, c)] = c
            n_submitted += 1

        # Drain completions. Results are appended in the main thread, so the
        # csv lock is only protecting against external readers (e.g. you
        # tail-ing grid_results.csv from another shell).
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                cid = _cell_id(c["idx"], c["cap"], c["cadence"], c["trials"])
                _safe_print(f"FAILED cell {cid}: {e}")
                continue
            _append_result(row)

    if STOP_FILE.exists():
        try:
            STOP_FILE.unlink()
            print(f"Removed {STOP_FILE} (drain complete).")
        except OSError:
            pass

    # Final ranking
    print()
    print("=" * 72)
    print("FINAL GRID RESULTS (sorted by MAE)")
    print("=" * 72)
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV).sort_values("MAE", na_position="last")
        cols = ["idx", "cap", "cadence", "trials", "MAE", "RMSE",
                "AccelAcc", "DirAcc", "HL", "nsa_weight_scale",
                "elapsed_min", "exit_code"]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))
    print(f"\nTotal wall time: {(time.time() - overall_t0)/3600:.1f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
