#!/usr/bin/env python3
"""
NFP Predictor - Full Pipeline Orchestrator
===========================================

Runs the complete NFP prediction pipeline end-to-end, or individual stages.

USAGE (CLI):
------------
    # Run full pipeline, reusing existing data if available:
    python run_full_project.py

    # Run full pipeline with fresh data (deletes and re-downloads everything):
    python run_full_project.py --fresh

    # Run only data collection + preparation (no training):
    python run_full_project.py --stage data

    # Run only the data loading stage:
    python run_full_project.py --stage load

    # Run only the data preparation stage:
    python run_full_project.py --stage prepare

    # Run only the training stage:
    python run_full_project.py --stage train

    # Skip specific steps (comma-separated):
    python run_full_project.py --skip noaa,prosper

    # Disable Optuna hyperparameter tuning (faster, uses static defaults):
    python run_full_project.py --stage train --no-tune

USAGE (importable):
-------------------
    from run_full_project import run_data_pipeline, run_training_pipeline

    # Run data collection + preparation
    success = run_data_pipeline(fresh=True)

    # Run training only
    success = run_training_pipeline(no_tune=True)

PIPELINE STAGES:
----------------
1. LOAD DATA: Fetch raw data from external sources (FRED, ADP, NOAA, etc.)
2. PREPARE DATA: Run feature selection engine and build master snapshots (quad-track)
3. TRAIN: Train all 4 NFP model variants (NSA/SA × first_release/revised),
          generate backtests, scorecards, and comparison output

Author: NFP Predictor Team
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR
from Data_ETA_Pipeline.perf_stats import (
    dump_perf_json,
    install_hooks,
    is_perf_enabled,
    profiled,
    register_atexit_dump,
    reset_perf_data,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

install_hooks()
register_atexit_dump("run_full_project", output_dir=TEMP_DIR / "perf")

DATA_DIRECTORIES = [
    DATA_PATH / "fred_data",
    DATA_PATH / "fred_data_prepared",
    DATA_PATH / "Exogenous_data",
    DATA_PATH / "NFP_target",
]

OUTPUT_DIRECTORIES = [
    OUTPUT_DIR / "models",
    OUTPUT_DIR / "backtest_results",
    OUTPUT_DIR / "backtest_historical",
]

# =============================================================================
# PIPELINE STEP DEFINITIONS
# =============================================================================

# Each step: (name, script_path, description, arguments)

LOAD_DATA_STEPS: List[Tuple[str, str, str, List[str]]] = [
    (
        "fred_employment",
        "Data_ETA_Pipeline/fred_employment_pipeline.py",
        "Download FRED employment data, build snapshots, and prepare for modeling",
        [],
    ),
    (
        "fred_exogenous",
        "Data_ETA_Pipeline/load_fred_exogenous.py",
        "Download exogenous economic indicators from FRED (VIX, claims, etc.)",
        [],
    ),
    (
        "adp",
        "Data_ETA_Pipeline/adp_pipeline.py",
        "Load ADP employment data and create NFP-aligned snapshots",
        [],
    ),
    (
        "noaa",
        "Data_ETA_Pipeline/noaa_pipeline.py",
        "Download NOAA storm data, create master file, and build weighted snapshots",
        [],
    ),
    (
        "prosper",
        "Data_ETA_Pipeline/load_prosper_data.py",
        "Fetch prediction market data from Prosper Trading",
        [],
    ),
    (
        "unifier",
        "Data_ETA_Pipeline/load_unifier_data.py",
        "Fetch ISM PMI, Consumer Confidence, JOLTS from Unifier API",
        [],
    ),
]

PREPARE_DATA_STEPS: List[Tuple[str, str, str, List[str]]] = [
    (
        "master_snapshots",
        "Data_ETA_Pipeline/create_master_snapshots.py",
        "Run feature selection engine and build master snapshots (quad-track: {nsa,sa} × {first_release,revised})",
        [],
    ),
]


def _build_train_steps(no_tune: bool = False) -> List[Tuple[str, str, str, List[str]]]:
    """Build training step definitions, optionally disabling Optuna tuning.

    Trains all 4 model variants (NSA/SA × first_release/revised) and generates
    a comparative scorecard via --train-all.
    """
    train_args = ["--train-all"]
    if no_tune:
        train_args.append("--no-tune")
    return [
        (
            "train_all_models",
            "Train/train_lightgbm_nfp.py",
            "Train all 4 model variants (NSA/SA × first_release/revised), backtest, and generate comparison",
            train_args,
        ),
    ]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    border = "=" * 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{border}")
    print(f" {text}")
    print(f"{border}{Colors.ENDC}\n")


def print_step(step_num: int, total: int, name: str, description: str) -> None:
    print(f"{Colors.CYAN}[{step_num}/{total}] {Colors.BOLD}{name}{Colors.ENDC}")
    print(f"    {description}")


def print_success(message: str) -> None:
    print(f"{Colors.GREEN}[OK] {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def delete_directories(directories: List[Path], verbose: bool = False) -> None:
    for directory in directories:
        if directory.exists():
            if verbose:
                print(f"  Deleting: {directory}")
            try:
                shutil.rmtree(directory)
                print_success(f"Deleted {directory.name}/")
            except Exception as e:
                print_error(f"Failed to delete {directory}: {e}")
        elif verbose:
            print(f"  Skipping (not found): {directory}")


@profiled("run_full_project.run_script")
def run_script(
    script_path: str,
    args: List[str] = None,
    timeout: Optional[int] = None
) -> Tuple[bool, float, str]:
    """Run a Python script as a subprocess, streaming output in real-time."""
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        return False, 0.0, f"Script not found: {full_path}"

    cmd = [sys.executable, str(full_path)]
    if args:
        cmd.extend(args)

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            print(f"    {line}", end="")
            output_lines.append(line)

        process.wait(timeout=timeout)
        output = "".join(output_lines)
        duration = time.time() - start_time
        return process.returncode == 0, duration, output

    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start_time
        return False, duration, f"Script timed out after {timeout}s"
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


@profiled("run_full_project.run_stage")
def run_stage(
    stage_name: str,
    steps: List[Tuple[str, str, str, List[str]]],
    skip_steps: List[str] = None,
    fresh: bool = False,
) -> Tuple[int, int, float]:
    """Run all steps in a pipeline stage. Returns (successful, failed, duration)."""
    skip_steps = skip_steps or []
    print_header(f"STAGE: {stage_name}")

    successful = 0
    failed = 0
    total_duration = 0.0

    active_steps = [s for s in steps if s[0] not in skip_steps]
    skipped = len(steps) - len(active_steps)

    if skipped > 0:
        print_warning(f"Skipping {skipped} step(s): {[s[0] for s in steps if s[0] in skip_steps]}")
        print()

    for i, (name, script, description, args) in enumerate(active_steps, 1):
        print_step(i, len(active_steps), name, description)

        step_args = list(args)
        if fresh and name == "fred_employment":
            step_args.append("--refresh")

        success, duration, output = run_script(script, step_args)
        total_duration += duration

        if success:
            print_success(f"Completed in {format_duration(duration)}")
            successful += 1
        else:
            print_error(f"Failed after {format_duration(duration)}")
            failed += 1

        print()

    if is_perf_enabled() and stage_name.strip().upper() in {"LOAD DATA", "PREPARE DATA"}:
        stage_key = stage_name.strip().lower().replace(" ", "_")
        dump_path = dump_perf_json(
            stage_name=f"stage_{stage_key}",
            output_dir=TEMP_DIR / "perf",
            extra={
                "successful_steps": successful,
                "failed_steps": failed,
                "duration_s": total_duration,
                "skip_steps": list(skip_steps),
            },
            reset=True,
        )
        if dump_path is not None:
            print(f"  Perf JSON: {dump_path}")

    return successful, failed, total_duration


def _print_summary(
    stage_durations: dict,
    total_successful: int,
    total_failed: int,
    total_duration: float,
) -> None:
    """Print pipeline execution summary."""
    print_header("PIPELINE SUMMARY")

    print(f"Total Steps: {total_successful + total_failed}")
    print(f"  {Colors.GREEN}Successful: {total_successful}{Colors.ENDC}")
    if total_failed > 0:
        print(f"  {Colors.FAIL}Failed: {total_failed}{Colors.ENDC}")
    else:
        print(f"  Failed: 0")

    print(f"\nDuration by Stage:")
    for stage_name, duration in stage_durations.items():
        print(f"  {stage_name}: {format_duration(duration)}")

    print(f"\n{Colors.BOLD}Total Duration: {format_duration(total_duration)}{Colors.ENDC}")

    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}PIPELINE COMPLETED SUCCESSFULLY{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}PIPELINE COMPLETED WITH {total_failed} FAILURE(S){Colors.ENDC}")

    print(f"\n{Colors.CYAN}Output Locations:{Colors.ENDC}")
    print(f"  Models:      {OUTPUT_DIR}/models/lightgbm_nfp/")
    print(f"  Predictions: {OUTPUT_DIR}/Predictions/")
    print(f"  NSA diag:    {OUTPUT_DIR}/NSA_prediction/")
    print(f"  SA diag:     {OUTPUT_DIR}/SA_prediction/")
    print(f"  Scorecard:   {OUTPUT_DIR}/models/lightgbm_nfp/")
    print(f"  Master Data: {DATA_PATH}/master_snapshots/")
    print(f"  Targets:     {DATA_PATH}/NFP_target/")
    print()


# =============================================================================
# PUBLIC API - importable functions for running individual stages
# =============================================================================

@profiled("run_full_project.run_data_pipeline")
def run_data_pipeline(
    fresh: bool = False,
    skip_steps: List[str] = None,
) -> bool:
    """
    Run data collection (load) and preparation stages only.

    Args:
        fresh: Delete existing data directories and re-download everything.
        skip_steps: Step names to skip (e.g. ['noaa', 'prosper']).

    Returns:
        True if all steps succeeded.
    """
    skip_steps = skip_steps or []
    if is_perf_enabled():
        reset_perf_data()
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    stage_durations = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{Colors.BOLD}{'#' * 70}")
    print(f"# NFP PREDICTOR - DATA PIPELINE")
    print(f"# Started: {timestamp}")
    print(f"# Mode: {'FRESH' if fresh else 'INCREMENTAL'}")
    print(f"{'#' * 70}{Colors.ENDC}")

    if fresh:
        print_header("CLEANUP: Deleting Existing Data")
        delete_directories(DATA_DIRECTORIES, verbose=True)
        print()

    # Load
    s, f, d = run_stage("LOAD DATA", LOAD_DATA_STEPS, skip_steps, fresh=fresh)
    total_successful += s
    total_failed += f
    stage_durations["Load Data"] = d

    if f > 0:
        print_warning("Some load steps failed. Continuing to prepare stage...")

    # Prepare
    s, f, d = run_stage("PREPARE DATA", PREPARE_DATA_STEPS, skip_steps, fresh=fresh)
    total_successful += s
    total_failed += f
    stage_durations["Prepare Data"] = d

    _print_summary(stage_durations, total_successful, total_failed, time.time() - start_time)
    return total_failed == 0


def run_training_pipeline(
    skip_steps: List[str] = None,
    no_tune: bool = False,
) -> bool:
    """
    Run training and output generation only (assumes data already exists).

    Args:
        skip_steps: Step names to skip.
        no_tune: If True, skip Optuna hyperparameter tuning (faster).

    Returns:
        True if all steps succeeded.
    """
    skip_steps = skip_steps or []
    start_time = time.time()
    stage_durations = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{Colors.BOLD}{'#' * 70}")
    print(f"# NFP PREDICTOR - TRAINING PIPELINE")
    print(f"# Started: {timestamp}")
    if no_tune:
        print(f"# Optuna tuning: DISABLED")
    print(f"{'#' * 70}{Colors.ENDC}")

    train_steps = _build_train_steps(no_tune=no_tune)
    s, f, d = run_stage("TRAIN MODELS", train_steps, skip_steps)
    stage_durations["Train Models"] = d

    _print_summary(stage_durations, s, f, time.time() - start_time)
    return f == 0


def run_full_pipeline(
    fresh: bool = False,
    stage: Optional[str] = None,
    skip_steps: List[str] = None,
    no_tune: bool = False,
) -> bool:
    """
    Execute the complete NFP prediction pipeline (or a single stage).

    Args:
        fresh: Delete existing data and re-download everything.
        stage: Run only this stage: 'load', 'prepare', 'data' (load+prepare), or 'train'.
        skip_steps: Step names to skip.
        no_tune: If True, skip Optuna hyperparameter tuning.

    Returns:
        True if all steps succeeded.
    """
    # Convenience: --stage data runs load + prepare
    if stage == "data":
        return run_data_pipeline(fresh=fresh, skip_steps=skip_steps)

    if stage == "train":
        return run_training_pipeline(skip_steps=skip_steps, no_tune=no_tune)

    skip_steps = skip_steps or []
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{Colors.BOLD}{'#' * 70}")
    print(f"# NFP PREDICTOR - FULL PIPELINE EXECUTION")
    print(f"# Started: {timestamp}")
    print(f"# Mode: {'FRESH (re-download all data)' if fresh else 'INCREMENTAL (reuse existing)'}")
    if stage:
        print(f"# Stage: {stage.upper()} only")
    if no_tune:
        print(f"# Optuna tuning: DISABLED")
    print(f"{'#' * 70}{Colors.ENDC}")

    if fresh:
        print_header("CLEANUP: Deleting Existing Data")
        print("Removing data directories...")
        delete_directories(DATA_DIRECTORIES, verbose=True)
        print("\nRemoving output directories...")
        delete_directories(OUTPUT_DIRECTORIES, verbose=True)
        print()

    total_successful = 0
    total_failed = 0
    stage_durations = {}

    # Stage 1: Load Data
    if stage is None or stage == "load":
        s, f, d = run_stage("LOAD DATA", LOAD_DATA_STEPS, skip_steps, fresh=fresh)
        total_successful += s
        total_failed += f
        stage_durations["Load Data"] = d
        if f > 0 and stage is None:
            print_warning("Some load steps failed. Continuing to prepare stage...")

    # Stage 2: Prepare Data
    if stage is None or stage == "prepare":
        s, f, d = run_stage("PREPARE DATA", PREPARE_DATA_STEPS, skip_steps, fresh=fresh)
        total_successful += s
        total_failed += f
        stage_durations["Prepare Data"] = d
        if f > 0 and stage is None:
            print_warning("Some prepare steps failed. Continuing to train stage...")

    # Stage 3: Train Models
    if stage is None:
        train_steps = _build_train_steps(no_tune=no_tune)
        s, f, d = run_stage("TRAIN MODELS", train_steps, skip_steps)
        total_successful += s
        total_failed += f
        stage_durations["Train Models"] = d

    total_duration = time.time() - start_time
    _print_summary(stage_durations, total_successful, total_failed, total_duration)
    return total_failed == 0


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def list_all_steps() -> None:
    """Print all available pipeline steps."""
    print("\n" + "=" * 70)
    print("NFP PREDICTOR PIPELINE STEPS")
    print("=" * 70)

    print("\n[LOAD DATA STAGE]")
    for name, script, desc, _ in LOAD_DATA_STEPS:
        print(f"  {name:20s} - {desc}")

    print("\n[PREPARE DATA STAGE]")
    for name, script, desc, _ in PREPARE_DATA_STEPS:
        print(f"  {name:20s} - {desc}")

    print("\n[TRAIN STAGE]")
    for name, script, desc, _ in _build_train_steps():
        print(f"  {name:20s} - {desc}")

    print("\nUse --skip <step_name> to skip specific steps.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NFP Predictor - Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_project.py                      # Full pipeline
  python run_full_project.py --fresh               # Fresh re-download + train
  python run_full_project.py --stage data          # Data collection + preparation only
  python run_full_project.py --stage train         # Training only
  python run_full_project.py --stage train --no-tune  # Training without Optuna
  python run_full_project.py --skip noaa,prosper   # Skip slow data sources
        """
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing data and re-download everything from scratch"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["load", "prepare", "data", "train"],
        help="Run only a specific stage: load, prepare, data (load+prepare), or train"
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of step names to skip (e.g., 'noaa,prosper')"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna hyperparameter tuning (faster, uses static LightGBM defaults)"
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available steps and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="(Kept for backwards compatibility)"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_steps:
        list_all_steps()
        return 0

    skip_steps = []
    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(",")]

    success = run_full_pipeline(
        fresh=args.fresh,
        stage=args.stage,
        skip_steps=skip_steps,
        no_tune=args.no_tune,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
