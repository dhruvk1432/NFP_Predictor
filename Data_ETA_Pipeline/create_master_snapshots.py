import pandas as pd
import sys
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files (Data_ETA_Pipeline/utils.py shadows utils/ package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, START_DATE, END_DATE
from utils.transforms import add_symlog_copies, add_pct_change_copies, compute_all_features
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map

logger = setup_logger(__file__, TEMP_DIR)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Paths ---
FRED_EXOG_DIR = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"
UNIFIER_DIR = DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades"
ADP_SNAPSHOTS_DIR = DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades"
NOAA_WEIGHTED_DIR = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / "decades"
PROSPER_DIR = DATA_PATH / "Exogenous_data" / "prosper" / "decades"
MASTER_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"

# =============================================================================
# NOAA AGGREGATION CONFIGURATION
# =============================================================================

NOAA_HUMAN_IMPACT_COLS = [
    "deaths_direct_weighted_log",
    "deaths_indirect_weighted_log",
    "injuries_direct_weighted_log",
    "injuries_indirect_weighted_log"
]

NOAA_ECONOMIC_DAMAGE_COLS = [
    "total_property_damage_real_weighted_log",
    "total_crop_damage_real_weighted_log"
]

# Columns to drop after aggregation
NOAA_COLS_TO_DROP = NOAA_HUMAN_IMPACT_COLS + NOAA_ECONOMIC_DAMAGE_COLS + ["storm_count_weighted_log"]


# =============================================================================
# SNAPSHOT LOADING
# =============================================================================

def _load_snapshot_cached(path: Path) -> pd.DataFrame:
    """Load a parquet file if it exists, with minimal overhead."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_all_snapshots_parallel(obs_month: pd.Timestamp) -> Tuple[pd.DataFrame, set]:
    """
    Load all source snapshots for a given observation month in parallel.
    I/O bound operations benefit from threading.

    Returns:
        Tuple of (combined DataFrame, set of ADP series names)
    """
    dirs = [
        (FRED_EXOG_DIR, "fred_exog"),
        (UNIFIER_DIR, "unifier"),
        (ADP_SNAPSHOTS_DIR, "adp"),
        (NOAA_WEIGHTED_DIR, "noaa"),
        (PROSPER_DIR, "prosper"),
    ]

    def get_path(base_dir: Path) -> Path:
        decade = f"{obs_month.year // 10 * 10}s"
        year = str(obs_month.year)
        filename = f"{obs_month.strftime('%Y-%m')}.parquet"
        return base_dir / decade / year / filename

    results = []
    adp_series = set()

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {
            executor.submit(_load_snapshot_cached, get_path(base_dir)): name
            for base_dir, name in dirs
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            df = future.result()
            if not df.empty:
                if name == "adp":
                    adp_series = set(df['series_name'].unique())
                results.append(df)

    if results:
        return pd.concat(results, ignore_index=True), adp_series
    return pd.DataFrame(), set()


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def preprocess_noaa_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Create NOAA composite indices and drop original granular columns."""
    if df.empty: return df

    # Check if we have NOAA data in this snapshot
    human_cols = [c for c in NOAA_HUMAN_IMPACT_COLS if c in df['series_name'].unique()]
    econ_cols = [c for c in NOAA_ECONOMIC_DAMAGE_COLS if c in df['series_name'].unique()]

    if not human_cols and not econ_cols:
        return df

    result_parts = []

    # 1. Keep non-NOAA data
    drop_mask = df['series_name'].isin(NOAA_COLS_TO_DROP)
    rest_df = df[~drop_mask].copy()
    result_parts.append(rest_df)

    # 2. Create Indices (Pivot -> Sum -> Melt logic manual equivalent)
    noaa_df = df[df['series_name'].isin(human_cols + econ_cols)].copy()

    if not noaa_df.empty:
        # Pivot to wide format to align dates
        wide = noaa_df.pivot_table(index='date', columns='series_name', values='value', aggfunc='first')

        indices = []

        # Human Impact Index
        available_human = [c for c in human_cols if c in wide.columns]
        if available_human:
            human_impact = wide[available_human].sum(axis=1).reset_index(name='value')
            human_impact['series_name'] = 'NOAA_Human_Impact_Index'
            indices.append(human_impact)

        # Economic Damage Index
        available_econ = [c for c in econ_cols if c in wide.columns]
        if available_econ:
            econ_damage = wide[available_econ].sum(axis=1).reset_index(name='value')
            econ_damage['series_name'] = 'NOAA_Economic_Damage_Index'
            indices.append(econ_damage)

        if indices:
            indices_df = pd.concat(indices, ignore_index=True)
            if 'snapshot_date' in df.columns:
                indices_df['snapshot_date'] = df['snapshot_date'].iloc[0]
            result_parts.append(indices_df)

    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def get_snapshot_path(base_dir, date_ts):
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    filename = f"{date_ts.strftime('%Y-%m')}.parquet"
    return base_dir / decade / year / filename


def load_snapshot(base_dir, date_ts):
    path = get_snapshot_path(base_dir, date_ts)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _process_single_snapshot(obs_month: pd.Timestamp, snap_date: pd.Timestamp) -> Tuple[bool, str]:
    """
    Process a single snapshot date. Returns (success, message).
    """
    try:
        # 1. LOAD DATA (The "Vintage") - using parallel I/O
        current_vintage_df, adp_series = _load_all_snapshots_parallel(obs_month)

        if current_vintage_df.empty:
            return (True, f"Skipped {obs_month.date()} (no data)")

        current_vintage_df['date'] = pd.to_datetime(current_vintage_df['date'])

        # 2. NOAA INDEX CREATION
        current_vintage_df = preprocess_noaa_indices(current_vintage_df)

        # 3. FULL TRANSFORMATION SUITE - only for ADP data
        # FRED exog, Unifier, and Prosper already have all features from their source pipelines.
        # NOAA indices are kept as-is (no transformations).
        # Only ADP data needs the Branch-and-Expand treatment here.
        if adp_series:
            adp_mask = current_vintage_df['series_name'].isin(adp_series)
            adp_df = current_vintage_df[adp_mask].copy()
            non_adp_df = current_vintage_df[~adp_mask].copy()

            adp_df = add_symlog_copies(adp_df)
            adp_df = add_pct_change_copies(adp_df)
            adp_df = compute_all_features(adp_df)

            current_vintage_df = pd.concat([non_adp_df, adp_df], ignore_index=True)

        # 4. SAVE
        current_vintage_df['snapshot_date'] = snap_date

        decade_str = f"{obs_month.year // 10 * 10}s"
        year_str = str(obs_month.year)
        save_dir = MASTER_DIR / decade_str / year_str
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{obs_month.strftime('%Y-%m')}.parquet"

        current_vintage_df.to_parquet(save_path, index=False)

        return (True, f"Generated {obs_month.date()} (snapshot={snap_date.date()})")

    except Exception as e:
        return (False, f"Error processing {obs_month.date()}: {str(e)}")


def create_master_snapshots(
    n_workers: int = 1,
    skip_existing: bool = False
):
    """
    Generate master snapshots for all dates in range.
    Pipeline: Load -> NOAA indices -> Symlog copies -> Save

    Args:
        n_workers: Number of parallel workers (1 = sequential, >1 = parallel)
        skip_existing: If True, skip snapshots that already exist
    """
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    nfp_release_map = get_nfp_release_map(start_date=start_dt, end_date=end_dt)
    snapshot_pairs = sorted(nfp_release_map.items(), key=lambda x: x[0])

    if snapshot_pairs:
        logger.info(
            f"Generating Master Snapshots from {snapshot_pairs[0][0].date()} to {snapshot_pairs[-1][0].date()}"
        )
        logger.info(f"Total snapshots to process: {len(snapshot_pairs)}")
    else:
        logger.info("No NFP release dates found for master snapshot generation.")
        return

    # Filter out existing if requested
    if skip_existing:
        dates_to_process = []
        for obs_month, snap_date in snapshot_pairs:
            save_path = get_snapshot_path(MASTER_DIR, obs_month)
            if not save_path.exists():
                dates_to_process.append((obs_month, snap_date))
        logger.info(f"Skipping {len(snapshot_pairs) - len(dates_to_process)} existing snapshots")
        snapshot_pairs = dates_to_process

    if not len(snapshot_pairs):
        logger.info("No snapshots to process.")
        return

    # Sequential processing (usually fastest due to I/O being the bottleneck)
    if n_workers == 1:
        for i, (obs_month, snap_date) in enumerate(snapshot_pairs):
            success, msg = _process_single_snapshot(obs_month, snap_date)

            if not success:
                logger.error(msg)
            elif i % 12 == 0:
                logger.info(msg)

    # Parallel processing using ProcessPoolExecutor for CPU-bound work
    else:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Using {n_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_snapshot, obs_month, snap_date): (obs_month, snap_date)
                for obs_month, snap_date in snapshot_pairs
            }

            completed = 0
            for future in as_completed(futures):
                success, msg = future.result()
                completed += 1

                if not success:
                    logger.error(msg)
                elif completed % 12 == 0:
                    logger.info(f"Progress: {completed}/{len(snapshot_dates)} - {msg}")

    logger.info("Master Snapshots generation complete.")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_master_snapshot():
    logger.info("Verifying Master Snapshots...")

    nfp_release_map = get_nfp_release_map(start_date=START_DATE, end_date=END_DATE)
    if not nfp_release_map:
        logger.error("Verification failed: No NFP release dates found.")
        return

    # Check the LAST generated snapshot (by observation month)
    test_month = max(nfp_release_map.keys())
    decade_str = f"{test_month.year // 10 * 10}s"
    year_str = str(test_month.year)
    master_path = MASTER_DIR / decade_str / year_str / f"{test_month.strftime('%Y-%m')}.parquet"

    if not master_path.exists():
        logger.error(f"Verification failed: Could not find file at {master_path}")
        return

    df = pd.read_parquet(master_path)
    series = df['series_name'].unique()

    logger.info(f"--- Verification Report for {test_month.date()} ---")
    logger.info(f"Total unique series: {len(series)}")

    # 1. Check Lags (expected from Branch-and-Expand)
    lags = [s for s in series if '_lag_' in s]
    logger.info(f"Lag features: {len(lags)}")

    # 2. Check Rolling (expected from Branch-and-Expand)
    rolling = [s for s in series if 'rolling_mean' in s or 'rolling_std' in s]
    logger.info(f"Rolling features: {len(rolling)}")

    # 3. Check NOAA Indices
    if any('NOAA_Human_Impact_Index' in s for s in series):
        logger.info("NOAA Indices present.")
    else:
        logger.warning("No NOAA data found.")

    # 4. Check Symlog copies
    # Symlog is applied per-source: FRED exog/Unifier/Prosper handle their own,
    # ADP gets symlog in this pipeline, NOAA does not get symlog.
    symlog_series = [s for s in series if s.endswith('_symlog')]
    non_symlog_series = [s for s in series if not s.endswith('_symlog')]
    logger.info(f"Original series: {len(non_symlog_series)}")
    logger.info(f"Symlog series: {len(symlog_series)}")

    # 5. Check Prosper data
    if 'series_code' in df.columns:
        prosper_codes = df[df['series_code'].str.contains('_ans', na=False)]['series_name'].unique()
        if len(prosper_codes) > 0:
            logger.info(f"Prosper data present: {len(prosper_codes)} series.")
        else:
            logger.warning("No Prosper data found.")
    else:
        logger.warning("No series_code column - cannot check Prosper data.")

    logger.info("Verification Complete.")

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate master snapshots (NOAA indices + symlog copies)"
    )
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

        create_master_snapshots(
            n_workers=args.workers,
            skip_existing=args.skip_existing
        )

        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        verify_master_snapshot()
