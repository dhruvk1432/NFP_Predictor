"""
Master Snapshot Generation Pipeline (Auto-Feature Selection)
===================================
This module constructs the final, unified "Master Snapshots" used by the Machine Learning
model. It acts as the grand aggregator, combining independently generated datasets
(FRED, Unifier, ADP, NOAA, Prosper) into a single wide-format parquet file per prediction month.

Critical Architecture:
- Point-in-time accuracy is strictly maintained.
- It operates in a dual-track mode: once for 'nsa' targets and once for 'sa' targets.
- Prior to concatenation, if a valid JSON cache is not found for the specific (sa/nsa) target,
  it spins up parallel workers to run a rigorous 7-stage LightGBM feature selection engine on ALL sources.
- It strictly filters the final master outputs to ONLY include the surviving features,
  preventing massive horizontal array bloat.
"""

import pandas as pd
import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE, TARGET_TYPE
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map
from Data_ETA_Pipeline.feature_selection_engine import (
    load_snapshot_wide, _classify_series, run_full_source_pipeline, MIN_VALID_OBS
)
from Train.data_loader import load_target_data

logger = setup_logger(__file__, TEMP_DIR)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

MASTER_BASE = DATA_PATH / "master_snapshots"

SOURCES = {
    'FRED_Employment': DATA_PATH / "FRED_Employment_Snapshots" / "decades",
    'FRED_Exogenous': DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades",
    'Unifier': DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades",
    'ADP': DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades",
    'NOAA': DATA_PATH / "Exogenous_data" / "exogenous_noaa_snapshots" / "decades",
    'Prosper': DATA_PATH / "Exogenous_data" / "prosper" / "decades",
}

# Ordered by typical execution time (longest to shortest) to optimize ProcessPool scheduling
SOURCE_EXEC_ORDER = ['FRED_Employment', 'FRED_Exogenous', 'Unifier', 'Prosper', 'NOAA', 'ADP']


# =============================================================================
# CACHE LOGIC
# =============================================================================

def _get_cache_path(target_cat: str) -> Path:
    MASTER_BASE.mkdir(parents=True, exist_ok=True)
    slug = (TARGET_TYPE or "default").replace("_", "-")
    return MASTER_BASE / f"selected_features_{target_cat}_{slug}.json"


def _check_cache(target_cat: str) -> list[str]:
    """Return cached features if generated within the last 30 days, else None."""
    cache_path = _get_cache_path(target_cat)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days

        if data.get("target_type") == TARGET_TYPE and data.get("target_cat") == target_cat and days_old < 30:
            logger.info(f"[{target_cat.upper()}] Using cached features (Age: {days_old} days). Target: {TARGET_TYPE}")
            return data.get("features", [])

        logger.info(f"[{target_cat.upper()}] Cache expired/invalid (Age: {days_old} days). Rebuilding...")
        return None

    except Exception as e:
        logger.warning(f"Failed to read feature cache: {e}")
        return None


def _save_cache(features: list[str], target_cat: str) -> None:
    cache_path = _get_cache_path(target_cat)
    data = {
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "target_type": TARGET_TYPE,
        "target_cat": target_cat,
        "features": sorted(list(set(features)))
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved {len(features)} selected features to cache: {cache_path}")


# =============================================================================
# FEATURE SELECTION PIPELINE (PARALLEL WORKERS)
# =============================================================================

def _snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    return base_dir / decade / year / f"{date_ts.strftime('%Y-%m')}.parquet"


def _process_source_features(source_name: str, source_dir: Path, target_cat: str) -> list[str]:
    """Worker function for ProcessPoolExecutor to run the 7-stage engine on one source."""
    # 1. Find and load latest snapshot
    logger.info(f"[{target_cat.upper()}] [{source_name}] Finding latest snapshot...")
    latest_files = sorted(source_dir.rglob('*.parquet'))
    if not latest_files:
        logger.warning(f"[{target_cat.upper()}] [{source_name}] No snapshot files found.")
        return []

    latest_path = latest_files[-1]
    snap_wide = load_snapshot_wide(latest_path)
    if snap_wide.empty:
        logger.warning(f"[{target_cat.upper()}] [{source_name}] Empty latest snapshot.")
        return []

    logger.info(f"[{target_cat.upper()}] [{source_name}] Loaded {latest_path.stem}: {snap_wide.shape}")

    # 2. Apply MIN_VALID_OBS filter + zero-variance filter (matches notebook pre-processing)
    valid_counts = snap_wide.count()
    short_features = valid_counts[valid_counts < MIN_VALID_OBS].index
    if len(short_features) > 0:
        snap_wide = snap_wide.drop(columns=short_features)
        logger.info(f"[{target_cat.upper()}] [{source_name}] Dropped {len(short_features)} "
                     f"short-history features (<{MIN_VALID_OBS} obs)")

    zero_var = snap_wide.std() == 0
    if zero_var.any():
        snap_wide = snap_wide.loc[:, ~zero_var]
        logger.info(f"[{target_cat.upper()}] [{source_name}] Dropped {zero_var.sum()} zero-variance features")

    if snap_wide.empty:
        logger.warning(f"[{target_cat.upper()}] [{source_name}] No features remain after filtering.")
        return []

    # 3. Build series groups using source-specific taxonomy
    series_groups = defaultdict(list)
    for col in snap_wide.columns:
        grp = _classify_series(col, source_name)
        series_groups[grp].append(col)

    logger.info(f"[{target_cat.upper()}] [{source_name}] {len(series_groups)} groups, "
                f"{snap_wide.shape[1]} total features")

    # 4. Load targets (MoM and Acc)
    logger.info(f"[{target_cat.upper()}] [{source_name}] Loading targets for {TARGET_TYPE}...")
    if "revised" in (TARGET_TYPE or "").lower():
        from Train.data_loader import build_revised_target
        target_df = build_revised_target(target_cat)
        target_df['ds'] = pd.to_datetime(target_df['ds'])
        target_df = target_df.sort_values('ds')
        target_df['y_acc'] = target_df['y_mom'].diff()
    else:
        target_df = load_target_data(target_type=target_cat, release_type='first', use_cache=False)
        target_df['y_acc'] = target_df['y_mom'].diff()

    if target_df.empty or 'y_mom' not in target_df.columns:
        logger.error(f"[{target_cat.upper()}] [{source_name}] Failed to load valid targets.")
        return []

    target_indexed = target_df.dropna(subset=['y_mom']).set_index('ds')
    y_mom = target_indexed['y_mom']
    y_acc = target_indexed['y_acc'].dropna()

    # 5. Run full pipeline (Stages 1-7: MoM + Acc union)
    logger.info(f"[{target_cat.upper()}] [{source_name}] Matrix shape {snap_wide.shape}. "
                f"Starting 7-stage pipeline...")
    try:
        survivors = run_full_source_pipeline(
            snap_wide, y_mom, y_acc, source_name, source_dir, series_groups
        )
        logger.info(f"[{target_cat.upper()}] [{source_name}] Engine returned {len(survivors)} features.")
        return survivors
    except Exception as e:
        logger.error(f"[{target_cat.upper()}] [{source_name}] Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _run_parallel_feature_selection(target_cat: str) -> list[str]:
    logger.info(f"[{target_cat.upper()}] Starting Parallel 7-Stage Feature Selection Across All Sources...")
    all_selected = []

    # max_workers=6 maps perfectly to our 6 sources
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as executor:
        futures = {
            executor.submit(_process_source_features, name, SOURCES[name], target_cat): name
            for name in SOURCE_EXEC_ORDER
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                feats = future.result()
                all_selected.extend(feats)
                logger.info(f"+++ [{target_cat.upper()}] {name} completed successfully. Added {len(feats)} features.")
            except Exception as e:
                logger.error(f"--- [{target_cat.upper()}] {name} spawned an exception: {e}")

    all_selected = list(set(all_selected))
    logger.info(f"[{target_cat.upper()}] Feature Selection Complete. Total surviving features: {len(all_selected)}")
    return all_selected


# =============================================================================
# MASTER GENERATION
# =============================================================================

def _load_all_sources(obs_month: pd.Timestamp, allowed_features: set) -> pd.DataFrame:
    """Load and merge all sources for a single month, dropping unselected columns."""
    results = []
    for name, sdir in SOURCES.items():
        path = _snapshot_path(sdir, obs_month)
        if not path.exists(): continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"[{name}] Failed to read {path}: {e}")
            continue
        if df.empty: continue

        # Format normalization
        if 'series_name' in df.columns and 'value' in df.columns:
            wide = df.pivot(index='date', columns='series_name', values='value').reset_index()
        else:
            wide = df

        keep_cols = [c for c in wide.columns if c in allowed_features or c in ['date', 'snapshot_date']]
        wide = wide[keep_cols]

        if not wide.drop(columns=['date', 'snapshot_date'], errors='ignore').empty:
            results.append(wide)

    if not results: return pd.DataFrame()

    master = results[0]
    for nxt in results[1:]:
        master = pd.merge(master, nxt, on='date', how='outer')

    return master


def create_master_snapshots(skip_existing: bool = False):
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    nfp_map = get_nfp_release_map(start_date=start_dt, end_date=end_dt)
    snapshot_pairs = sorted(nfp_map.items(), key=lambda x: x[0])

    if not snapshot_pairs:
        logger.info("No NFP release dates found.")
        return

    # Loop dual branches specifically for Not Seasonally Adjusted & Seasonally Adjusted
    for target_cat in ['nsa', 'sa']:
        logger.info(f"========== COMMENCING BRANCH: {target_cat.upper()} ==========")
        target_master_dir = MASTER_BASE / target_cat / "decades"

        # 1. Check Cache / Run Engine
        allowed_list = _check_cache(target_cat)
        if allowed_list is None:
            logger.info(f"[{target_cat.upper()}] Executing vast compute pool for automated selection...")
            allowed_list = _run_parallel_feature_selection(target_cat)
            if not allowed_list:
                logger.error(f"[{target_cat.upper()}] Feature selection catastrophic failure. No features returned.")
                continue
            _save_cache(allowed_list, target_cat)

        allowed_set = set(allowed_list)
        logger.info(f"[{target_cat.upper()}] Proceeding strictly with {len(allowed_set)} whitelisted features.")

        # 2. Iterate and Build
        branch_snapshot_pairs = snapshot_pairs
        if skip_existing:
            branch_snapshot_pairs = [
                (obs, snap) for obs, snap in snapshot_pairs
                if not _snapshot_path(target_master_dir, obs).exists()
            ]

        for i, (obs_month, snap_date) in enumerate(branch_snapshot_pairs):
            try:
                master = _load_all_sources(obs_month, allowed_set)
                if master.empty:
                    logger.debug(f"[{target_cat.upper()}] Skipped {obs_month.date()} (no valid data)")
                    continue

                master['date'] = pd.to_datetime(master['date'])
                master['snapshot_date'] = snap_date

                save_dir = _snapshot_path(target_master_dir, obs_month).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                master.to_parquet(save_dir / f"{obs_month.strftime('%Y-%m')}.parquet", index=False)

                if i % 12 == 0:
                    logger.info(f"[{target_cat.upper()}] Generated {obs_month.date()} (Cols: {master.shape[1]})")

            except Exception as e:
                logger.error(f"[{target_cat.upper()}] Error processing {obs_month.date()}: {e}")

        logger.info(f"========== BRANCH COMPLETE: {target_cat.upper()} ==========\n")

    logger.info("Master Snapshots generation completely finished for all branches.")

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    start_time = time.time()
    create_master_snapshots(skip_existing=args.skip_existing)
    logger.info(f"Total Master Time: {(time.time() - start_time)/60:.1f} minutes")
