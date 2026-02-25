"""
A/B Test: Feature Selection Stage Configurations
=================================================
Tests whether the full 6-stage feature selection pipeline is overkill
compared to simpler tier configurations.

Runs ETL + backtest for each config on sa_revised (most important variant),
then compares MAE, RMSE, and coverage across configs.

Usage:
    python scripts/ab_feature_selection.py
    python scripts/ab_feature_selection.py --configs tier1 full
    python scripts/ab_feature_selection.py --skip-etl   # reuse existing snapshots
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger

# Use the same Python interpreter that's running this script
PYTHON = sys.executable

logger = setup_logger(__file__, TEMP_DIR)

# ── Experiment configurations ──
CONFIGS = {
    "tier1":          (0, 1, 4),          # Minimal: Variance + Dual Filter + Cluster
    "tier1_vintage":  (0, 1, 3, 4),      # + Vintage Stability
    "tier1_boruta":   (0, 1, 2, 4),      # + Boruta
    "full":           (0, 1, 2, 3, 4, 5, 6),  # Current pipeline (control)
}

MASTER_BASE = DATA_PATH / "master_snapshots"
SOURCE_CACHES_DIR = MASTER_BASE / "source_caches"
REGIME_CACHES_DIR = MASTER_BASE / "regime_caches"
AB_RESULTS_DIR = OUTPUT_DIR / "ab_results"

# Target config: sa_revised (most important variant per user)
TARGET_TYPE = "sa"
TARGET_SOURCE = "revised"
MAE_THRESHOLD = 0.02  # 2% relative MAE degradation tolerance

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _invalidate_caches():
    """Delete source and regime caches to force fresh feature selection runs."""
    for cache_dir in [SOURCE_CACHES_DIR, REGIME_CACHES_DIR]:
        if cache_dir.is_dir():
            # Only delete caches matching our target config
            for f in cache_dir.glob("*.json"):
                if f"_{TARGET_TYPE}_{TARGET_SOURCE}" in f.name:
                    f.unlink()
                    logger.info(f"Deleted cache: {f.name}")

    # Also delete branch-level cache
    branch_cache = MASTER_BASE / f"selected_features_{TARGET_TYPE}_{TARGET_SOURCE}.json"
    if branch_cache.exists():
        branch_cache.unlink()
        logger.info(f"Deleted branch cache: {branch_cache.name}")

    # Delete candidate pool cache
    pool_cache = MASTER_BASE / f"candidate_pool_{TARGET_TYPE}_{TARGET_SOURCE}.json"
    if pool_cache.exists():
        pool_cache.unlink()
        logger.info(f"Deleted pool cache: {pool_cache.name}")


def _run_etl(stages_str: str) -> float:
    """Run create_master_snapshots.py with NFP_FS_STAGES set. Returns wall time in seconds."""
    env = os.environ.copy()
    env["NFP_FS_STAGES"] = stages_str
    env["NFP_PERF"] = "1"
    env["NFP_PERF_SERIAL_FS"] = "1"  # Serial for deterministic timing

    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "Data_ETA_Pipeline" / "create_master_snapshots.py"),
        "--target-source-scope", TARGET_SOURCE,
    ]

    logger.info(f"Running ETL with NFP_FS_STAGES={stages_str}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    wall_time = time.time() - t0

    if result.returncode != 0:
        logger.error(f"ETL failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    else:
        logger.info(f"ETL completed in {wall_time:.1f}s")

    return wall_time


def _run_backtest() -> float:
    """Run backtest for sa_revised with --no-tune. Returns wall time in seconds."""
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "Train" / "train_lightgbm_nfp.py"),
        "--train",
        "--target", TARGET_TYPE,
        "--release", "first",
        "--revised",
        "--no-tune",
    ]

    logger.info(f"Running backtest for {TARGET_TYPE}_{TARGET_SOURCE}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    wall_time = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Backtest failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    else:
        logger.info(f"Backtest completed in {wall_time:.1f}s")

    return wall_time


def _collect_results(config_name: str, etl_time: float, backtest_time: float):
    """Copy result artifacts into the A/B results directory."""
    dest = AB_RESULTS_DIR / config_name
    dest.mkdir(parents=True, exist_ok=True)

    # Copy summary statistics
    model_id = f"{TARGET_TYPE}_first_revised"
    summary_src = OUTPUT_DIR / f"{TARGET_TYPE.upper()}_prediction" / "summary_statistics.csv"
    if summary_src.exists():
        shutil.copy2(summary_src, dest / "summary_statistics.csv")

    # Copy model comparison
    comparison_src = OUTPUT_DIR / "models" / "lightgbm_nfp" / "model_comparison.csv"
    if comparison_src.exists():
        shutil.copy2(comparison_src, dest / "model_comparison.csv")

    # Copy backtest results
    backtest_src = OUTPUT_DIR / f"{TARGET_TYPE.upper()}_prediction" / "backtest_results.csv"
    if backtest_src.exists():
        shutil.copy2(backtest_src, dest / "backtest_results.csv")

    # Copy stability report
    stability_src = OUTPUT_DIR / "models" / "lightgbm_nfp" / model_id / "shortpass_stability.json"
    if stability_src.exists():
        shutil.copy2(stability_src, dest / "shortpass_stability.json")

    # Save timing metadata
    meta = {
        "config_name": config_name,
        "stages": list(CONFIGS[config_name]),
        "etl_wall_seconds": round(etl_time, 1),
        "backtest_wall_seconds": round(backtest_time, 1),
        "total_wall_seconds": round(etl_time + backtest_time, 1),
        "target_type": TARGET_TYPE,
        "target_source": TARGET_SOURCE,
    }
    with open(dest / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Results saved to {dest}")


def _compare_results(configs_run: list[str]):
    """Load results from all configs and generate comparison summary."""
    import pandas as pd

    rows = []
    for config_name in configs_run:
        dest = AB_RESULTS_DIR / config_name

        # Load summary stats
        summary_path = dest / "summary_statistics.csv"
        if not summary_path.exists():
            logger.warning(f"No summary_statistics.csv for {config_name}")
            continue

        summary = pd.read_csv(summary_path)
        # summary_statistics.csv has metric names as rows typically
        # Try to extract key metrics
        metrics = {}
        if 'Metric' in summary.columns and 'Value' in summary.columns:
            for _, row in summary.iterrows():
                metrics[row['Metric']] = row['Value']
        elif len(summary.columns) == 2:
            for _, row in summary.iterrows():
                metrics[str(row.iloc[0])] = row.iloc[1]
        else:
            # Columns might be metric names directly
            for col in summary.columns:
                if summary[col].dtype in ('float64', 'int64'):
                    metrics[col] = summary[col].iloc[0]

        # Load timing
        meta_path = dest / "run_metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        # Load stability
        stability_path = dest / "shortpass_stability.json"
        stability = {}
        if stability_path.exists():
            with open(stability_path) as f:
                stability = json.load(f)

        row = {
            "Config": config_name,
            "Stages": str(CONFIGS.get(config_name, "?")),
            "MAE": _safe_float(metrics.get("MAE", metrics.get("mae", None))),
            "RMSE": _safe_float(metrics.get("RMSE", metrics.get("rmse", None))),
            "Coverage_80": _safe_float(metrics.get("80%_Coverage",
                                                    metrics.get("80% Coverage", None))),
            "ETL_sec": meta.get("etl_wall_seconds", "?"),
            "Backtest_sec": meta.get("backtest_wall_seconds", "?"),
            "Total_sec": meta.get("total_wall_seconds", "?"),
            "Jaccard_mean": stability.get("jaccard_mean", "?"),
        }
        rows.append(row)

    if not rows:
        logger.error("No results to compare.")
        return

    comparison = pd.DataFrame(rows)

    # Compute relative MAE vs full pipeline
    full_mae = comparison.loc[comparison['Config'] == 'full', 'MAE']
    if not full_mae.empty and pd.notna(full_mae.iloc[0]):
        full_mae_val = full_mae.iloc[0]
        comparison['MAE_vs_Full_%'] = comparison['MAE'].apply(
            lambda x: round((x - full_mae_val) / full_mae_val * 100, 2)
            if pd.notna(x) and full_mae_val > 0 else None
        )
        comparison['Verdict'] = comparison['MAE_vs_Full_%'].apply(
            lambda x: "PASS" if x is not None and x <= MAE_THRESHOLD * 100
            else ("FAIL" if x is not None else "?")
        )

    # Save and print
    comparison_path = AB_RESULTS_DIR / "comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to {comparison_path}")

    print("\n" + "=" * 80)
    print("A/B FEATURE SELECTION COMPARISON")
    print("=" * 80)
    print(f"Target: {TARGET_TYPE}_{TARGET_SOURCE}")
    print(f"MAE tolerance: {MAE_THRESHOLD * 100:.0f}%")
    print()
    print(comparison.to_string(index=False))
    print("=" * 80)

    return comparison


def _safe_float(val):
    """Safely convert to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="A/B test feature selection configurations")
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Which configs to test (default: all)",
    )
    parser.add_argument(
        "--skip-etl", action="store_true",
        help="Skip ETL step; use existing master snapshots",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Only compare existing results (no ETL or backtest)",
    )
    args = parser.parse_args()

    AB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        _compare_results(args.configs)
        return

    total_t0 = time.time()

    for config_name in args.configs:
        stages = CONFIGS[config_name]
        stages_str = ",".join(str(s) for s in stages)

        print(f"\n{'=' * 60}")
        print(f"  CONFIG: {config_name} — stages={stages}")
        print(f"{'=' * 60}")

        etl_time = 0.0
        if not args.skip_etl:
            _invalidate_caches()
            etl_time = _run_etl(stages_str)

        backtest_time = _run_backtest()
        _collect_results(config_name, etl_time, backtest_time)

    total_time = time.time() - total_t0
    logger.info(f"\nAll configs completed in {total_time / 60:.1f} min")

    _compare_results(args.configs)


if __name__ == "__main__":
    main()
