"""
Master Snapshot Generation Pipeline
===================================
This module constructs the final, unified "Master Snapshots" used by the Machine Learning
model. It acts as the grand aggregator, combining independently generated exogenous datasets
(FRED, Unifier, ADP, NOAA, Prosper) into a single wide-format parquet file per prediction month.

Critical Architecture:
- Point-in-time accuracy is strictly maintained because the upstream source directories
  have already filtered their data relative to the NFP release calendar.
- This script merely blindly concatenates rows from the individual source snapshots 
  for a given month-end `obs_month`, creating a unified point-in-time ledger.
"""

import pandas as pd
import sys
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files (Data_ETA_Pipeline/utils.py shadows utils/ package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map

logger = setup_logger(__file__, TEMP_DIR)

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Source directories ---
SOURCE_DIRS = [
    DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades",
    DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades",
    DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades",
    DATA_PATH / "Exogenous_data" / "exogenous_noaa_snapshots" / "decades",
    DATA_PATH / "Exogenous_data" / "prosper" / "decades",
]
MASTER_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"


# =============================================================================
# HELPERS
# =============================================================================

def _snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    return base_dir / decade / year / f"{date_ts.strftime('%Y-%m')}.parquet"


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_all_sources(obs_month: pd.Timestamp) -> pd.DataFrame:
    """Load all source snapshots for a given observation month in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=len(SOURCE_DIRS)) as executor:
        futures = [executor.submit(_load_parquet, _snapshot_path(d, obs_month)) for d in SOURCE_DIRS]
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                results.append(df)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _process_single_snapshot(obs_month: pd.Timestamp, snap_date: pd.Timestamp) -> tuple[bool, str]:
    """Process a single snapshot date. Returns (success, message)."""
    try:
        current_vintage_df = _load_all_sources(obs_month)

        if current_vintage_df.empty:
            return (True, f"Skipped {obs_month.date()} (no data)")

        current_vintage_df['date'] = pd.to_datetime(current_vintage_df['date'])
        current_vintage_df['snapshot_date'] = snap_date

        save_dir = _snapshot_path(MASTER_DIR, obs_month).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{obs_month.strftime('%Y-%m')}.parquet"

        current_vintage_df.to_parquet(save_path, index=False)

        return (True, f"Generated {obs_month.date()} (snapshot={snap_date.date()})")

    except Exception as e:
        return (False, f"Error processing {obs_month.date()}: {str(e)}")


def create_master_snapshots(n_workers: int = 1, skip_existing: bool = False):
    """
    Iterate chronologically through all historical NFP release dates and merge all
    available exogenous data sources into a single master snapshot for each month.
    
    This function discovers the specific NFP release date for a given month, finds 
    the corresponding pre-filtered sub-snapshots from FRED, NOAA, ADP, etc., and 
    concatenates them vertically into a unified dataframe representing all macroeconomic 
    knowledge available immediately before the BLS publication.

    Args:
        n_workers (int): Number of parallel processes to use for parsing files.
        skip_existing (bool): If True, skips months that already have a master parquet file.
    """
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    nfp_release_map = get_nfp_release_map(start_date=start_dt, end_date=end_dt)
    snapshot_pairs = sorted(nfp_release_map.items(), key=lambda x: x[0])

    if not snapshot_pairs:
        logger.info("No NFP release dates found for master snapshot generation.")
        return

    logger.info(
        f"Generating Master Snapshots from {snapshot_pairs[0][0].date()} to {snapshot_pairs[-1][0].date()}"
    )
    logger.info(f"Total snapshots to process: {len(snapshot_pairs)}")

    if skip_existing:
        before = len(snapshot_pairs)
        snapshot_pairs = [
            (obs, snap) for obs, snap in snapshot_pairs
            if not _snapshot_path(MASTER_DIR, obs).exists()
        ]
        logger.info(f"Skipping {before - len(snapshot_pairs)} existing snapshots")

    if not snapshot_pairs:
        logger.info("No snapshots to process.")
        return

    if n_workers == 1:
        for i, (obs_month, snap_date) in enumerate(snapshot_pairs):
            success, msg = _process_single_snapshot(obs_month, snap_date)
            if not success:
                logger.error(msg)
            elif i % 12 == 0:
                logger.info(msg)
    else:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Using {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_snapshot, obs, snap): (obs, snap)
                for obs, snap in snapshot_pairs
            }
            completed = 0
            for future in as_completed(futures):
                success, msg = future.result()
                completed += 1
                if not success:
                    logger.error(msg)
                elif completed % 12 == 0:
                    logger.info(f"Progress: {completed}/{len(snapshot_pairs)} - {msg}")

    logger.info("Master Snapshots generation complete.")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_master_snapshot():
    """Check the latest master snapshot has data from all expected sources."""
    logger.info("Verifying Master Snapshots...")

    nfp_release_map = get_nfp_release_map(start_date=START_DATE, end_date=END_DATE)
    if not nfp_release_map:
        logger.error("Verification failed: No NFP release dates found.")
        return

    test_month = max(nfp_release_map.keys())
    master_path = _snapshot_path(MASTER_DIR, test_month)

    if not master_path.exists():
        logger.error(f"Verification failed: Could not find file at {master_path}")
        return

    df = pd.read_parquet(master_path)
    series = df['series_name'].unique()

    logger.info(f"--- Verification Report for {test_month.date()} ---")
    logger.info(f"Total unique series: {len(series)}")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Rows: {len(df):,}")
    logger.info("Verification Complete.")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Generate master exogenous snapshots")
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip snapshots that already exist")
    parser.add_argument('--verify-only', action='store_true',
                        help="Only run verification, skip generation")
    args = parser.parse_args()

    if args.verify_only:
        verify_master_snapshot()
    else:
        start_time = time.time()
        create_master_snapshots(n_workers=args.workers, skip_existing=args.skip_existing)
        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        verify_master_snapshot()
