"""
Master Snapshot Generation Pipeline (Auto-Feature Selection)
===================================
This module constructs the final, unified "Master Snapshots" used by the Machine Learning
model. It acts as the grand aggregator, combining independently generated datasets
(FRED, Unifier, ADP, NOAA, Prosper) into a single wide-format parquet file per prediction month.

Critical Architecture:
- Point-in-time accuracy is strictly maintained.
- It operates in a quad-track mode: {nsa, sa} x {first_release, revised}.
- Prior to concatenation, if a valid JSON cache is not found for the specific combo,
  it spins up parallel workers to run a rigorous 6-stage LightGBM feature selection engine on ALL sources.
- It strictly filters the final master outputs to ONLY include the surviving features,
  preventing massive horizontal array bloat.
- Per-source caching: each source's feature selection results are cached independently,
  so a crash in one source doesn't waste completed work from others.
- Source-level batch loading: for the master generation phase, each source's snapshots
  are loaded once and cached in memory to avoid redundant file reads across months.
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
import hashlib

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map
from Data_ETA_Pipeline.feature_selection_engine import (
    load_snapshot_wide, _classify_series, run_full_source_pipeline, MIN_VALID_OBS
)
from Train.data_loader import load_target_data

logger = setup_logger(__file__, TEMP_DIR)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

MASTER_BASE = DATA_PATH / "master_snapshots"

SOURCES = {
    'FRED_Employment_NSA': DATA_PATH / "fred_data_prepared_nsa" / "decades",
    'FRED_Employment_SA':  DATA_PATH / "fred_data_prepared_sa"  / "decades",
    'FRED_Exogenous': DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades",
    'Unifier': DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades",
    'ADP': DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades",
    'NOAA': DATA_PATH / "Exogenous_data" / "exogenous_noaa_snapshots" / "decades",
    'Prosper': DATA_PATH / "Exogenous_data" / "prosper" / "decades",
}

# Ordered by typical execution time (longest to shortest) to optimize ProcessPool scheduling
SOURCE_EXEC_ORDER = ['FRED_Employment_NSA', 'FRED_Employment_SA', 'FRED_Exogenous',
                     'Unifier', 'Prosper', 'NOAA', 'ADP']

# All 4 combinations of target category and target source
TARGET_COMBOS = [
    ('nsa', 'first_release'),
    ('nsa', 'revised'),
    ('sa', 'first_release'),
    ('sa', 'revised'),
]


# =============================================================================
# CACHE LOGIC
# =============================================================================

def _get_cache_path(target_cat: str, target_source: str) -> Path:
    MASTER_BASE.mkdir(parents=True, exist_ok=True)
    return MASTER_BASE / f"selected_features_{target_cat}_{target_source}.json"


def _check_cache(target_cat: str, target_source: str) -> list[str]:
    """Return cached features if generated within the last 30 days, else None."""
    cache_path = _get_cache_path(target_cat, target_source)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days

        if (data.get("target_source") == target_source
                and data.get("target_cat") == target_cat
                and days_old < 30):
            logger.info(f"[{target_cat.upper()}/{target_source}] Using cached features "
                        f"(Age: {days_old} days).")
            return data.get("features", [])

        logger.info(f"[{target_cat.upper()}/{target_source}] Cache expired/invalid "
                    f"(Age: {days_old} days). Rebuilding...")
        return None

    except Exception as e:
        logger.warning(f"Failed to read feature cache: {e}")
        return None


def _save_cache(features: list[str], target_cat: str, target_source: str) -> None:
    cache_path = _get_cache_path(target_cat, target_source)
    data = {
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "target_source": target_source,
        "target_cat": target_cat,
        "features": sorted(list(set(features)))
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved {len(features)} selected features to cache: {cache_path}")


# =============================================================================
# PER-SOURCE CACHE LOGIC
# =============================================================================

def _get_source_cache_path(source_name: str, target_cat: str, target_source: str) -> Path:
    """Cache path for a single source's feature selection results."""
    cache_dir = MASTER_BASE / "source_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"source_{source_name}_{target_cat}_{target_source}.json"


def _check_source_cache(source_name: str, target_cat: str, target_source: str) -> list[str] | None:
    """Return cached features for a specific source if valid (< 30 days), else None."""
    cache_path = _get_source_cache_path(source_name, target_cat, target_source)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days
        if days_old < 30:
            feats = data.get("features", [])
            logger.info(f"[{source_name}] Using cached source features "
                        f"({len(feats)} features, {days_old} days old)")
            return feats
        return None
    except Exception:
        return None


def _save_source_cache(features: list[str], source_name: str,
                       target_cat: str, target_source: str) -> None:
    """Save a single source's feature selection results."""
    cache_path = _get_source_cache_path(source_name, target_cat, target_source)
    data = {
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "source_name": source_name,
        "target_cat": target_cat,
        "target_source": target_source,
        "features": sorted(list(set(features)))
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"[{source_name}] Saved {len(features)} features to source cache")


# =============================================================================
# FEATURE SELECTION PIPELINE (PARALLEL WORKERS)
# =============================================================================

def _snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    return base_dir / decade / year / f"{date_ts.strftime('%Y-%m')}.parquet"


def _process_source_features(source_name: str, source_dir: Path,
                             target_cat: str, target_source: str) -> list[str]:
    """Worker function for ProcessPoolExecutor to run the 6-stage engine on one source."""
    label = f"{target_cat.upper()}/{target_source}"

    # 1. Find and load latest snapshot
    logger.info(f"[{label}] [{source_name}] Finding latest snapshot...")
    latest_files = sorted(source_dir.rglob('*.parquet'))
    if not latest_files:
        logger.warning(f"[{label}] [{source_name}] No snapshot files found.")
        return []

    latest_path = latest_files[-1]
    snap_wide = load_snapshot_wide(latest_path)
    if snap_wide.empty:
        logger.warning(f"[{label}] [{source_name}] Empty latest snapshot.")
        return []

    logger.info(f"[{label}] [{source_name}] Loaded {latest_path.stem}: {snap_wide.shape}")

    # 2. Apply MIN_VALID_OBS filter + zero-variance filter (matches notebook pre-processing)
    valid_counts = snap_wide.count()
    short_features = valid_counts[valid_counts < MIN_VALID_OBS].index
    if len(short_features) > 0:
        snap_wide = snap_wide.drop(columns=short_features)
        logger.info(f"[{label}] [{source_name}] Dropped {len(short_features)} "
                     f"short-history features (<{MIN_VALID_OBS} obs)")

    zero_var = snap_wide.std() == 0
    if zero_var.any():
        snap_wide = snap_wide.loc[:, ~zero_var]
        logger.info(f"[{label}] [{source_name}] Dropped {zero_var.sum()} zero-variance features")

    if snap_wide.empty:
        logger.warning(f"[{label}] [{source_name}] No features remain after filtering.")
        return []

    # 3. Build series groups using source-specific taxonomy
    series_groups = defaultdict(list)
    for col in snap_wide.columns:
        grp = _classify_series(col, source_name)
        series_groups[grp].append(col)

    logger.info(f"[{label}] [{source_name}] {len(series_groups)} groups, "
                f"{snap_wide.shape[1]} total features")

    # 4. Load targets (MoM only)
    logger.info(f"[{label}] [{source_name}] Loading targets...")
    if target_source == 'revised':
        from Train.data_loader import build_revised_target
        target_df = build_revised_target(target_cat)
        target_df['ds'] = pd.to_datetime(target_df['ds'])
        target_df = target_df.sort_values('ds')
    else:
        target_df = load_target_data(target_type=target_cat, release_type='first', use_cache=False)

    if target_df.empty or 'y_mom' not in target_df.columns:
        logger.error(f"[{label}] [{source_name}] Failed to load valid targets.")
        return []

    target_indexed = target_df.dropna(subset=['y_mom']).set_index('ds')
    y_mom = target_indexed['y_mom']

    # 5. Run full pipeline (Stages 1-6: MoM only)
    logger.info(f"[{label}] [{source_name}] Matrix shape {snap_wide.shape}. "
                f"Starting 6-stage pipeline...")
    try:
        survivors = run_full_source_pipeline(
            snap_wide, y_mom, source_name, source_dir, series_groups
        )
        logger.info(f"[{label}] [{source_name}] Engine returned {len(survivors)} features.")
        return survivors
    except Exception as e:
        logger.error(f"[{label}] [{source_name}] Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _run_parallel_feature_selection(target_cat: str, target_source: str) -> list[str]:
    label = f"{target_cat.upper()}/{target_source}"
    logger.info(f"[{label}] Starting Feature Selection Across All Sources...")
    all_selected = []

    # OOM Protection Strategy:
    # Massive datasets (FRED) chew up ~15-20GB of RAM *per worker*.
    # If we run them in parallel with others, macOS will instantly SIGKILL
    # the process for RAM exhaustion (Exit Code 137).
    # Therefore, we run the massive datasets sequentially first, then
    # parallelize the smaller ones.

    massive_sources = ['FRED_Employment_NSA', 'FRED_Employment_SA', 'FRED_Exogenous']
    small_sources = [s for s in SOURCE_EXEC_ORDER if s not in massive_sources]

    # Route FRED Employment: only process the branch matching target_cat
    # (nsa model only needs NSA features, sa model only needs SA features)
    skip_source = 'FRED_Employment_SA' if target_cat == 'nsa' else 'FRED_Employment_NSA'

    def _run_or_load_source(name):
        """Check per-source cache first; run pipeline only if cache miss."""
        cached = _check_source_cache(name, target_cat, target_source)
        if cached is not None:
            return cached
        feats = _process_source_features(name, SOURCES[name], target_cat, target_source)
        _save_source_cache(feats, name, target_cat, target_source)
        return feats

    # 1. Run massive sources sequentially (skip the irrelevant FRED branch)
    for name in massive_sources:
        if name == skip_source:
            logger.info(f"[{label}] Skipping '{name}' (not needed for {target_cat} branch)")
            continue
        if name in SOURCES:
            logger.info(f"[{label}] Executing massive dataset '{name}' sequentially to prevent OOM...")
            try:
                feats = _run_or_load_source(name)
                all_selected.extend(feats)
                logger.info(f"+++ [{label}] {name} completed successfully. Added {len(feats)} features.")
            except Exception as e:
                logger.error(f"--- [{label}] {name} failed: {e}")

    # 2. Run small sources in parallel (cache-aware)
    # Separate cached vs uncached sources to avoid spawning workers unnecessarily
    uncached_small = []
    for name in small_sources:
        cached = _check_source_cache(name, target_cat, target_source)
        if cached is not None:
            all_selected.extend(cached)
            logger.info(f"+++ [{label}] {name} loaded from cache. Added {len(cached)} features.")
        else:
            uncached_small.append(name)

    if uncached_small:
        logger.info(f"[{label}] Executing {len(uncached_small)} uncached smaller datasets in parallel...")
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            futures = {
                executor.submit(_process_source_features, name, SOURCES[name],
                                target_cat, target_source): name
                for name in uncached_small
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    feats = future.result()
                    _save_source_cache(feats, name, target_cat, target_source)
                    all_selected.extend(feats)
                    logger.info(f"+++ [{label}] {name} completed successfully. Added {len(feats)} features.")
                except Exception as e:
                    logger.error(f"--- [{label}] {name} spawned an exception: {e}")

    all_selected = list(set(all_selected))
    logger.info(f"[{label}] Feature Selection Complete. Total surviving features: {len(all_selected)}")
    return all_selected


# =============================================================================
# MASTER GENERATION
# =============================================================================

def _normalize_to_wide(df):
    """Convert a raw source DataFrame to wide format with 'date' as a column."""
    if 'series_name' in df.columns and 'value' in df.columns:
        wide = df.pivot(index='date', columns='series_name', values='value').reset_index()
    else:
        wide = df

    if 'date' not in wide.columns:
        if isinstance(wide.index, pd.DatetimeIndex) or wide.index.name == 'date':
            wide = wide.reset_index()
        elif hasattr(wide.index, 'name') and wide.index.name is not None:
            wide = wide.reset_index()

    return wide


def _batch_load_source(source_name: str, source_dir: Path,
                       snapshot_months: list[pd.Timestamp],
                       allowed_features: set) -> dict[str, pd.DataFrame]:
    """Pre-load all snapshots for one source, filtered to allowed features.

    Returns a dict mapping month-key (YYYY-MM) to a filtered wide DataFrame.
    This eliminates redundant file reads when iterating over months.
    """
    result = {}
    for obs_month in snapshot_months:
        path = _snapshot_path(source_dir, obs_month)
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"[{source_name}] Failed to read {path}: {e}")
            continue
        if df.empty:
            continue

        wide = _normalize_to_wide(df)
        keep_cols = [c for c in wide.columns
                     if c in allowed_features or c in ['date', 'snapshot_date']]
        wide = wide[keep_cols]

        if not wide.drop(columns=['date', 'snapshot_date'], errors='ignore').empty:
            result[obs_month.strftime('%Y-%m')] = wide

    return result


def _load_all_sources_from_cache(obs_month: pd.Timestamp,
                                 source_caches: dict[str, dict],
                                 ) -> pd.DataFrame:
    """Assemble a master row from pre-loaded source caches (O(1) lookup per source)."""
    month_key = obs_month.strftime('%Y-%m')
    results = []
    for source_name, month_dict in source_caches.items():
        wide = month_dict.get(month_key)
        if wide is not None:
            results.append(wide)

    if not results:
        return pd.DataFrame()

    master = results[0]
    for nxt in results[1:]:
        master = pd.merge(master, nxt, on='date', how='outer')
    return master


def _get_progress_path(target_cat: str, target_source: str) -> Path:
    """Path for the branch progress checkpoint file."""
    progress_dir = MASTER_BASE / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    return progress_dir / f"progress_{target_cat}_{target_source}.json"


def _load_progress(target_cat: str, target_source: str) -> set[str]:
    """Load the set of successfully completed month-keys for a branch."""
    path = _get_progress_path(target_cat, target_source)
    if not path.exists():
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return set(data.get("completed_months", []))
    except Exception:
        return set()


def _save_progress(target_cat: str, target_source: str,
                   completed_months: set[str]) -> None:
    """Persist the set of completed month-keys for crash-resume."""
    path = _get_progress_path(target_cat, target_source)
    data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_cat": target_cat,
        "target_source": target_source,
        "completed_months": sorted(completed_months)
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def _clear_progress(target_cat: str, target_source: str) -> None:
    """Remove the progress file when a branch completes fully."""
    path = _get_progress_path(target_cat, target_source)
    if path.exists():
        path.unlink()


def create_master_snapshots(skip_existing: bool = False):
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    nfp_map = get_nfp_release_map(start_date=start_dt, end_date=end_dt)
    snapshot_pairs = sorted(nfp_map.items(), key=lambda x: x[0])

    if not snapshot_pairs:
        logger.info("No NFP release dates found.")
        return

    # Loop all 4 combinations: {nsa, sa} x {first_release, revised}
    # Each branch runs its own independent feature selection
    for target_cat, target_source in TARGET_COMBOS:
        label = f"{target_cat.upper()}/{target_source}"
        logger.info(f"========== COMMENCING BRANCH: {label} ==========")
        target_master_dir = MASTER_BASE / target_cat / target_source / "decades"

        # 1. Check Cache / Run Engine
        allowed_list = _check_cache(target_cat, target_source)
        if allowed_list is None:
            logger.info(f"[{label}] Executing vast compute pool for automated selection...")
            allowed_list = _run_parallel_feature_selection(target_cat, target_source)
            if not allowed_list:
                logger.error(f"[{label}] Feature selection catastrophic failure. No features returned.")
                continue
            _save_cache(allowed_list, target_cat, target_source)

        allowed_set = set(allowed_list)
        logger.info(f"[{label}] Proceeding strictly with {len(allowed_set)} whitelisted features.")

        # 2. Determine which months to process (skip_existing + progress resume)
        completed_months = _load_progress(target_cat, target_source)
        if completed_months:
            logger.info(f"[{label}] Resuming from checkpoint: "
                        f"{len(completed_months)} months already completed")

        branch_snapshot_pairs = []
        for obs, snap in snapshot_pairs:
            month_key = obs.strftime('%Y-%m')
            # Skip if already in progress tracker
            if month_key in completed_months:
                continue
            # Skip if file exists and skip_existing is set
            if skip_existing and _snapshot_path(target_master_dir, obs).exists():
                completed_months.add(month_key)
                continue
            branch_snapshot_pairs.append((obs, snap))

        if not branch_snapshot_pairs:
            logger.info(f"[{label}] All months already completed. Skipping.")
            _clear_progress(target_cat, target_source)
            logger.info(f"========== BRANCH COMPLETE: {label} ==========\n")
            continue

        logger.info(f"[{label}] {len(branch_snapshot_pairs)} months to generate...")

        # 3. Batch-load all sources for this branch (one read per source per month)
        skip_source = None
        if target_cat == 'nsa':
            skip_source = 'FRED_Employment_SA'
        elif target_cat == 'sa':
            skip_source = 'FRED_Employment_NSA'

        all_months = [obs for obs, _ in branch_snapshot_pairs]
        source_caches = {}
        for name, sdir in SOURCES.items():
            if name == skip_source:
                continue
            logger.info(f"[{label}] Batch-loading {name}...")
            source_caches[name] = _batch_load_source(
                name, sdir, all_months, allowed_set
            )
            loaded_count = len(source_caches[name])
            logger.info(f"[{label}] {name}: {loaded_count}/{len(all_months)} months loaded")

        # 4. Iterate and Build (with progress checkpointing)
        total = len(branch_snapshot_pairs)
        for i, (obs_month, snap_date) in enumerate(branch_snapshot_pairs):
            try:
                master = _load_all_sources_from_cache(obs_month, source_caches)
                if master.empty:
                    logger.debug(f"[{label}] Skipped {obs_month.date()} (no valid data)")
                    completed_months.add(obs_month.strftime('%Y-%m'))
                    continue

                master['date'] = pd.to_datetime(master['date'])
                master['snapshot_date'] = snap_date

                save_dir = _snapshot_path(target_master_dir, obs_month).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                master.to_parquet(save_dir / f"{obs_month.strftime('%Y-%m')}.parquet", index=False)

                completed_months.add(obs_month.strftime('%Y-%m'))

                if i % 12 == 0:
                    logger.info(f"[{label}] Generated {obs_month.date()} "
                                f"({i + 1}/{total}, Cols: {master.shape[1]})")
                    # Checkpoint every 12 months
                    _save_progress(target_cat, target_source, completed_months)

            except Exception as e:
                logger.error(f"[{label}] Error processing {obs_month.date()}: {e}")
                # Save progress so we can resume past this point
                _save_progress(target_cat, target_source, completed_months)

        # Branch complete — clean up progress file
        _clear_progress(target_cat, target_source)
        # Free memory from source caches
        del source_caches
        logger.info(f"========== BRANCH COMPLETE: {label} ==========\n")

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
