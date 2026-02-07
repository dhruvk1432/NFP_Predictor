#!/usr/bin/env python3
"""
NFP Predictor - Full Pipeline Orchestrator
===========================================

This script runs the complete NFP prediction pipeline end-to-end, from data loading
through model training, predictions, evaluation, and backtesting.

USAGE:
------
    # Run full pipeline, reusing existing data if available:
    python run_full_project.py

    # Run full pipeline with fresh data (deletes and re-downloads everything):
    python run_full_project.py --fresh

    # Run only the data loading stage:
    python run_full_project.py --stage load

    # Run only the data preparation stage:
    python run_full_project.py --stage prepare

    # Run only the training stage:
    python run_full_project.py --stage train

    # Skip specific steps (comma-separated):
    python run_full_project.py --skip noaa,prosper

    # Note: All script logs are now shown by default during execution
    # The --verbose flag is kept for backwards compatibility

PIPELINE STAGES:
----------------
1. LOAD DATA: Fetch raw data from external sources (FRED, ADP, NOAA, etc.)
2. PREPARE DATA: Transform and consolidate data into master snapshots
3. TRAIN: Train first release NFP models (nsa_first, sa_first), generate predictions and backtests

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories that will be deleted with --fresh flag
DATA_DIRECTORIES = [
    PROJECT_ROOT / "data" / "fred_data",
    PROJECT_ROOT / "data" / "fred_data_prepared",
    PROJECT_ROOT / "data" / "Exogenous_data",
    PROJECT_ROOT / "data" / "NFP_target",
]

# Output directories that will be deleted with --fresh flag
OUTPUT_DIRECTORIES = [
    PROJECT_ROOT / "_output" / "models",
    PROJECT_ROOT / "_output" / "backtest_results",
    PROJECT_ROOT / "_output" / "backtest_historical",
]

# =============================================================================
# PIPELINE STEP DEFINITIONS
# =============================================================================

# Each step is a tuple: (name, script_path, description, arguments)
# Scripts are run using subprocess with the project root as working directory

LOAD_DATA_STEPS: List[Tuple[str, str, str, List[str]]] = [
    (
        "fred_employment",
        "Data_ETA_Pipeline/fred_employment_pipeline.py",
        "Download FRED employment data, build snapshots, and prepare for modeling",
        [],  # Add ["--refresh"] if you want to force refresh
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
        "Consolidate all exogenous data into master snapshots (main step)",
        [],  # Default is sequential; add ["--workers", "4"] for parallel
    ),
]

TRAIN_STEPS: List[Tuple[str, str, str, List[str]]] = [
    (
        "train_all_models",
        "Train/train_lightgbm_nfp.py",
        "Train first release NFP models (nsa_first, sa_first)",
        ["--train-all"],
    ),
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a formatted section header."""
    border = "=" * 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{border}")
    print(f" {text}")
    print(f"{border}{Colors.ENDC}\n")


def print_step(step_num: int, total: int, name: str, description: str) -> None:
    """Print a formatted step indicator."""
    print(f"{Colors.CYAN}[{step_num}/{total}] {Colors.BOLD}{name}{Colors.ENDC}")
    print(f"    {description}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[OK] {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
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
    """
    Delete specified directories and their contents.

    Args:
        directories: List of directory paths to delete
        verbose: If True, print detailed progress
    """
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


def run_script(
    script_path: str,
    args: List[str] = None,
    verbose: bool = False,
    timeout: Optional[int] = None
) -> Tuple[bool, float, str]:
    """
    Run a Python script as a subprocess.

    Args:
        script_path: Path to the Python script (relative to project root)
        args: Additional command-line arguments to pass to the script
        verbose: Kept for backwards compatibility (logs always shown now)
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (success: bool, duration: float, output: str)
    """
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        return False, 0.0, f"Script not found: {full_path}"

    cmd = [sys.executable, str(full_path)]
    if args:
        cmd.extend(args)

    start_time = time.time()

    try:
        # Always stream output in real-time
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
        returncode = process.returncode

        duration = time.time() - start_time
        success = returncode == 0

        return success, duration, output

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, duration, f"Script timed out after {timeout}s"
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def run_stage(
    stage_name: str,
    steps: List[Tuple[str, str, str, List[str]]],
    skip_steps: List[str],
    verbose: bool = False,
    fresh: bool = False
) -> Tuple[int, int, float]:
    """
    Run all steps in a pipeline stage.

    Args:
        stage_name: Name of the stage (for display)
        steps: List of step definitions
        skip_steps: List of step names to skip
        verbose: Kept for backwards compatibility (logs always shown now)
        fresh: If True, this is a fresh run (may affect some step args)

    Returns:
        Tuple of (successful_count, failed_count, total_duration)
    """
    print_header(f"STAGE: {stage_name}")

    successful = 0
    failed = 0
    total_duration = 0.0

    # Filter out skipped steps
    active_steps = [s for s in steps if s[0] not in skip_steps]
    skipped = len(steps) - len(active_steps)

    if skipped > 0:
        print_warning(f"Skipping {skipped} step(s): {[s[0] for s in steps if s[0] in skip_steps]}")
        print()

    for i, (name, script, description, args) in enumerate(active_steps, 1):
        print_step(i, len(active_steps), name, description)

        # For fred_employment, add --refresh flag if doing fresh run
        step_args = list(args)
        if fresh and name == "fred_employment":
            step_args.append("--refresh")

        success, duration, output = run_script(script, step_args, verbose=verbose)
        total_duration += duration

        if success:
            print_success(f"Completed in {format_duration(duration)}")
            successful += 1
        else:
            print_error(f"Failed after {format_duration(duration)}")
            failed += 1

        print()

    return successful, failed, total_duration


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_full_pipeline(
    fresh: bool = False,
    stage: Optional[str] = None,
    skip_steps: List[str] = None,
    verbose: bool = False
) -> bool:
    """
    Execute the complete NFP prediction pipeline.

    Args:
        fresh: If True, delete existing data and re-download everything
        stage: If specified, run only this stage ('load', 'prepare', 'train')
        skip_steps: List of step names to skip
        verbose: Kept for backwards compatibility (logs always shown now)

    Returns:
        True if all steps succeeded, False otherwise
    """
    skip_steps = skip_steps or []

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print pipeline header
    print(f"\n{Colors.BOLD}{'#' * 70}")
    print(f"# NFP PREDICTOR - FULL PIPELINE EXECUTION")
    print(f"# Started: {timestamp}")
    print(f"# Mode: {'FRESH (re-download all data)' if fresh else 'INCREMENTAL (reuse existing)'}")
    if stage:
        print(f"# Stage: {stage.upper()} only")
    print(f"{'#' * 70}{Colors.ENDC}")

    # Handle fresh mode: delete existing data
    if fresh:
        print_header("CLEANUP: Deleting Existing Data")
        print("Removing data directories...")
        delete_directories(DATA_DIRECTORIES, verbose=verbose)
        print("\nRemoving output directories...")
        delete_directories(OUTPUT_DIRECTORIES, verbose=verbose)
        print()

    # Track overall results
    total_successful = 0
    total_failed = 0
    stage_durations = {}

    # Stage 1: Load Data
    if stage is None or stage == "load":
        successful, failed, duration = run_stage(
            "LOAD DATA",
            LOAD_DATA_STEPS,
            skip_steps,
            verbose=verbose,
            fresh=fresh
        )
        total_successful += successful
        total_failed += failed
        stage_durations["Load Data"] = duration

        if failed > 0 and stage is None:
            print_warning("Some load steps failed. Continuing to prepare stage...")

    # Stage 2: Prepare Data
    if stage is None or stage == "prepare":
        successful, failed, duration = run_stage(
            "PREPARE DATA",
            PREPARE_DATA_STEPS,
            skip_steps,
            verbose=verbose,
            fresh=fresh
        )
        total_successful += successful
        total_failed += failed
        stage_durations["Prepare Data"] = duration

        if failed > 0 and stage is None:
            print_warning("Some prepare steps failed. Continuing to train stage...")

    # Stage 3: Train Models
    if stage is None or stage == "train":
        successful, failed, duration = run_stage(
            "TRAIN MODELS",
            TRAIN_STEPS,
            skip_steps,
            verbose=verbose,
            fresh=fresh
        )
        total_successful += successful
        total_failed += failed
        stage_durations["Train Models"] = duration

    # Print summary
    total_duration = time.time() - start_time

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

    # Print output locations
    print(f"\n{Colors.CYAN}Output Locations:{Colors.ENDC}")
    print(f"  Models:      {PROJECT_ROOT}/_output/models/lightgbm_nfp/")
    print(f"  Backtests:   {PROJECT_ROOT}/_output/backtest_results/")
    print(f"  Master Data: {PROJECT_ROOT}/data/Exogenous_data/master_snapshots/")
    print(f"  Targets:     {PROJECT_ROOT}/data/NFP_target/")

    print()

    return total_failed == 0


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NFP Predictor - Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with existing data
  python run_full_project.py

  # Fresh run: delete all data and re-download
  python run_full_project.py --fresh

  # Run only the training stage
  python run_full_project.py --stage train

  # Skip slow data sources
  python run_full_project.py --skip noaa,prosper
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
        choices=["load", "prepare", "train"],
        help="Run only a specific stage of the pipeline"
    )

    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of step names to skip (e.g., 'noaa,prosper')"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="(Kept for backwards compatibility - logs are now always shown)"
    )

    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available steps and exit"
    )

    return parser.parse_args()


def list_all_steps() -> None:
    """Print a list of all pipeline steps."""
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
    for name, script, desc, _ in TRAIN_STEPS:
        print(f"  {name:20s} - {desc}")

    print("\nUse --skip <step_name> to skip specific steps.")
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle --list-steps
    if args.list_steps:
        list_all_steps()
        return 0

    # Parse skip list
    skip_steps = []
    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(",")]

    # Run the pipeline
    success = run_full_pipeline(
        fresh=args.fresh,
        stage=args.stage,
        skip_steps=skip_steps,
        verbose=args.verbose
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
