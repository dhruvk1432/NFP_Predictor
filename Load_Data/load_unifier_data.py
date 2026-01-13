import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unifier import unifier

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE, UNIFIER_TOKEN, UNIFIER_USER
# OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
from Prepare_Data.nfp_relative_timing import get_nfp_release_map, calculate_median_offset_from_nfp, apply_nfp_relative_adjustment
# OPTIMIZATION: Use shared utility for snapshot path
from Load_Data.utils import get_snapshot_path

logger = setup_logger(__file__, TEMP_DIR)

UNIFIER_SERIES = {
    # Existing series
    "Challenger_Job_Cuts": "USCGJCUTP",
    "ISM_Manufacturing_Index": "USNAPMEM",
    "ISM_NonManufacturing_Index": "USNPNE..Q",
    "CB_Consumer_Confidence": "USCNFCONQ",
    
    # NEW: Hours/Wages (hours lead employment changes)
    "AWH_All_Private": "USWHIP..O",           # Avg Weekly Hours - critical leading signal
    "AWH_Manufacturing": "USHKIM..O",          # Manufacturing hours
    "AHE_Private": "USWAGES.D",                # Avg Hourly Earnings - wage tightness
    
    # NEW: Housing/Consumer (leading indicators)
    "Housing_Starts": "USHOUSE.O",             # Housing cycle
    "Retail_Sales": "USRETTOTB",               # Consumer demand (100% release dates)
    
    # NEW: Regional PMIs (early signals)
    "Empire_State_Mfg": "USFRNFMFQ",           # First regional PMI each month
    "UMich_Expectations": "USUMCONEH",         # Forward-looking sentiment
    
    # NEW: Industrial activity
    "Industrial_Production": "USIPTOT.G",      # Manufacturing activity
}


def calculate_series_lag(df, series_name):
    """
    Calculate the median lag for a series from observations with first_release_date.

    Uses all available data with first_release_date to calculate median lag.
    Returns the median lag in days from end of month to first_release_date.
    Can be negative (e.g., CB Consumer Confidence released mid-month).
    """
    # Parse dates
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'])
    df['first_release_date_parsed'] = pd.to_datetime(df['first_release_date'], errors='coerce')

    # Find records with valid release dates
    has_release_date = df['first_release_date_parsed'].notna()

    if has_release_date.any():
        valid_df = df[has_release_date].copy()

        # Calculate end of event month for each observation
        valid_df['event_month_end'] = valid_df['timestamp_parsed'].dt.to_period('M').dt.to_timestamp('M')

        # Calculate lag in days from end of event month to first release (can be negative)
        valid_df['lag_days'] = (
            valid_df['first_release_date_parsed'] - valid_df['event_month_end']
        ).dt.days

        median_lag = int(valid_df['lag_days'].median())
        logger.info(f"{series_name}: Calculated median lag = {median_lag} days from end of month "
                   f"(based on {len(valid_df)} observations with first_release_date)")
        return median_lag
    else:
        logger.warning(f"{series_name}: No first_release_date available, using default 0 days lag")
        return 0

def get_effective_release_and_value(row, snap_date, median_lag_days, nfp_offset_days=None):
    """
    Determine the release date and value to use for a snapshot (single row version).

    NOTE: This function is kept for backwards compatibility and edge cases.
    For batch processing, use get_effective_release_and_value_vectorized() instead.

    CRITICAL LOGIC:
    - When first_release_date is MISSING:
      * Unifier sets last_revision_date = timestamp (lookahead bias!)
      * NEVER use last_revision_date in this case
      * Backfill: end_of_month + median_lag_days
      * ENHANCEMENT: Apply NFP-relative adjustment if nfp_offset_days provided
      * Value: latest_revised_value (only value we have)

    - When first_release_date is PRESENT:
      * Use most recent info available by snap_date
      * If last_revision_date < snap_date: use latest_revised_value
      * Else: use first_release_value

    Returns: (release_date, value)
    """
    first_release_date = row.get('first_release_date')
    last_revision_date = row.get('last_revision_date')
    first_release_value = row.get('first_release_value')
    latest_revised_value = row.get('latest_revised_value')
    timestamp = row['timestamp']
    date = row['date']

    # Case 1: Missing first_release_date - BACKFILL (NEVER use last_revision_date)
    if pd.isna(first_release_date):
        # Base estimate using median lag
        event_month_end = pd.Timestamp(date).to_period('M').to_timestamp('M')
        base_release = event_month_end + pd.Timedelta(days=median_lag_days)

        # Apply NFP-relative adjustment if available
        # Uses apply_nfp_relative_adjustment imported at module level
        if nfp_offset_days is not None:
            try:
                event_month = pd.Timestamp(date).replace(day=1)
                effective_release = apply_nfp_relative_adjustment(
                    event_month=event_month,
                    base_release_date=base_release,
                    median_offset_days=nfp_offset_days,
                    use_adjustment=True
                )
            except Exception:
                # If NFP adjustment fails, use base estimate
                effective_release = base_release
        else:
            effective_release = base_release

        # Use latest_revised_value (only value we have)
        value = latest_revised_value

        return effective_release, value

    # Case 2: Has first_release_date - use most recent info < snap_date
    # Changed from <= to strict < to prevent same-day data leakage
    if pd.notna(last_revision_date) and last_revision_date < snap_date:
        # Revision is available by snapshot time
        return last_revision_date, latest_revised_value
    else:
        # No revision yet (or revision happened after snapshot)
        return first_release_date, first_release_value


def get_effective_release_and_value_vectorized(df, snap_date, median_lag_days, nfp_offset_days=None):
    """
    Vectorized version of get_effective_release_and_value for batch processing.

    This is ~100x faster than using iterrows() with the single-row version.

    Args:
        df: DataFrame with columns: first_release_date, last_revision_date,
            first_release_value, latest_revised_value, timestamp, date
        snap_date: Snapshot date (pd.Timestamp)
        median_lag_days: Median lag in days for backfilling
        nfp_offset_days: Optional NFP offset for adjustment (currently unused in vectorized version)

    Returns:
        DataFrame with columns: date, release_date, value (only rows with release_date < snap_date)
    """
    result = df.copy()

    # Ensure date columns are datetime
    result['date'] = pd.to_datetime(result['date'])
    result['first_release_date'] = pd.to_datetime(result['first_release_date'], errors='coerce')
    result['last_revision_date'] = pd.to_datetime(result['last_revision_date'], errors='coerce')

    # Calculate base backfill release date for all rows (month-end + median_lag)
    result['event_month_end'] = result['date'].dt.to_period('M').dt.to_timestamp('M')
    result['backfill_release'] = result['event_month_end'] + pd.Timedelta(days=median_lag_days)

    # Case 1: Missing first_release_date - use backfill
    missing_first_release = result['first_release_date'].isna()

    # Case 2a: Has first_release_date AND revision available before snap_date
    has_revision_before_snap = (
        result['first_release_date'].notna() &
        result['last_revision_date'].notna() &
        (result['last_revision_date'] < snap_date)
    )

    # Case 2b: Has first_release_date but no revision before snap_date
    use_first_release = (
        result['first_release_date'].notna() &
        ~has_revision_before_snap
    )

    # Assign release_date based on cases
    result['release_date'] = pd.NaT
    result.loc[missing_first_release, 'release_date'] = result.loc[missing_first_release, 'backfill_release']
    result.loc[has_revision_before_snap, 'release_date'] = result.loc[has_revision_before_snap, 'last_revision_date']
    result.loc[use_first_release, 'release_date'] = result.loc[use_first_release, 'first_release_date']

    # Assign value based on cases
    result['value'] = np.nan
    result.loc[missing_first_release, 'value'] = result.loc[missing_first_release, 'latest_revised_value']
    result.loc[has_revision_before_snap, 'value'] = result.loc[has_revision_before_snap, 'latest_revised_value']
    result.loc[use_first_release, 'value'] = result.loc[use_first_release, 'first_release_value']

    # Filter to only include rows released before snap_date (strict <)
    result = result[result['release_date'] < snap_date].copy()

    # Clean up temporary columns
    result = result.drop(columns=['event_month_end', 'backfill_release'], errors='ignore')

    return result[['date', 'release_date', 'value']]

def fetch_unifier_snapshots(start_date=START_DATE, end_date=END_DATE):
    """
    Fetch Unifier data and create monthly snapshots aligned with NFP release dates.

    Each snapshot (YYYY-MM.parquet) contains exogenous data that was available
    when NFP for month YYYY-MM was released.

    Key improvements:
    - Never uses last_revision_date when first_release_date is missing (lookahead bias)
    - Backfills missing release dates using median lag from real data
    - Uses most recent value available by snapshot time
    """
    # Setup credentials
    unifier.user = UNIFIER_USER
    unifier.token = UNIFIER_TOKEN
    os.environ['UNIFIER_USER'] = unifier.user
    os.environ['UNIFIER_TOKEN'] = unifier.token

    # OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)

    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_unifier_data"

    # Fetch all data for each series once
    all_series_data = {}
    series_median_lags = {}

    for name, code in UNIFIER_SERIES.items():
        try:
            logger.info(f"Fetching {name} ({code})")
            df = unifier.get_dataframe(name="lseg_us_economics_pit", key=code)

            if df.empty:
                logger.warning(f"No data returned for {name} ({code})")
                continue

            # V2: Schema validation - check required columns exist
            required_cols = ["timestamp", "latest_revised_value"]
            optional_cols = ["last_revision_date", "first_release_date", "first_release_value"]

            missing_required = [c for c in required_cols if c not in df.columns]
            if missing_required:
                logger.error(f"Schema validation failed for {name}: missing required columns {missing_required}")
                logger.error(f"  Available columns: {df.columns.tolist()}")
                continue

            missing_optional = [c for c in optional_cols if c not in df.columns]
            if missing_optional:
                logger.warning(f"{name}: Missing optional columns {missing_optional} - will use defaults")

            # Select available columns
            all_cols = required_cols + optional_cols
            available_cols = [c for c in all_cols if c in df.columns]
            df = df[available_cols].copy()

            df['series_name'] = name
            df['series_code'] = code

            # Parse dates
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()  # First of month

            df['first_release_date'] = pd.to_datetime(df['first_release_date'], errors='coerce')
            df['last_revision_date'] = pd.to_datetime(df['last_revision_date'], errors='coerce')

            # Calculate median lag for this series
            median_lag = calculate_series_lag(df, name)
            series_median_lags[name] = median_lag

            # Store for later use
            all_series_data[name] = df
            logger.info(f"Fetched {len(df)} rows for {name}")

        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_series_data:
        logger.warning("No data fetched from Unifier.")
        return

    # Calculate NFP-relative timing offsets for each series (for backfill consistency)
    # Uses imported calculate_median_offset_from_nfp from nfp_relative_timing
    logger.info("Calculating NFP-relative timing offsets for each series...")
    series_nfp_offsets = {}

    for name, df in all_series_data.items():
        try:
            median_offset, num_obs = calculate_median_offset_from_nfp(
                df[df['first_release_date'].notna()],  # Only use rows with actual release dates
                event_col='date',
                release_col='first_release_date'
            )
            series_nfp_offsets[name] = median_offset if num_obs > 0 else None

            if num_obs > 0:
                logger.info(f"  {name}: {median_offset:.1f} days from NFP (based on {num_obs} observations)")
            else:
                logger.info(f"  {name}: No NFP offset calculated (no historical data)")
        except Exception as e:
            logger.warning(f"Could not calculate NFP offset for {name}: {e}")
            series_nfp_offsets[name] = None

    # Now create monthly snapshots aligned with NFP release dates
    for obs_month, nfp_release_date in nfp_release_map.items():
        snap_date = pd.Timestamp(nfp_release_date)

        # OPTIMIZATION: Use shared utility for snapshot path
        save_path = get_snapshot_path(base_dir, obs_month)

        snap_data_list = []

        for name, df in all_series_data.items():
            median_lag = series_median_lags[name]
            nfp_offset = series_nfp_offsets.get(name, None)

            # Use vectorized function instead of iterrows() - ~100x faster
            series_df = get_effective_release_and_value_vectorized(
                df, snap_date, median_lag, nfp_offset_days=nfp_offset
            )

            if series_df.empty:
                continue

            # Additional vectorized check: data for month M should NOT appear in snapshots
            # BEFORE NFP for month M is released (point-in-time correctness)
            # Create a Series mapping obs_date -> nfp_release_date for dates in nfp_release_map
            series_df['nfp_for_month'] = series_df['date'].map(nfp_release_map)

            # Filter out rows where: snap_date <= nfp_for_month AND release_date > nfp_for_month
            # (These would cause lookahead bias)
            has_nfp = series_df['nfp_for_month'].notna()
            lookahead_bias = (
                has_nfp &
                (snap_date <= series_df['nfp_for_month']) &
                (series_df['release_date'] > series_df['nfp_for_month'])
            )
            series_df = series_df[~lookahead_bias].copy()

            # Drop helper column and add series metadata
            series_df = series_df.drop(columns=['nfp_for_month'], errors='ignore')
            series_df['series_name'] = name
            series_df['series_code'] = UNIFIER_SERIES[name]

            # Remove duplicates (keep last by date)
            series_df = series_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

            if not series_df.empty:
                snap_data_list.append(series_df)

        if snap_data_list:
            full_snap = pd.concat(snap_data_list, ignore_index=True)
            full_snap['snapshot_date'] = snap_date
            full_snap.to_parquet(save_path)

        if obs_month.month == 12:
            logger.info(f"Generated snapshots for {obs_month.year}")

if __name__ == "__main__":
    logger.info(f"Fetching Unifier data from {START_DATE} to {END_DATE}")
    fetch_unifier_snapshots(start_date=START_DATE, end_date=END_DATE)