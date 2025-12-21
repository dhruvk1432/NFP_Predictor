"""
Create NFP-Weighted NOAA Storm Damage Aggregates

Reads NOAA_master.parquet with format:
- Columns: date, series_name, value, release_date
- Series names: {metric}_{STATE} (e.g., deaths_direct_ALABAMA)

Creates NFP-weighted national aggregates using state employment shares.
Snapshots are aligned with NFP release dates for point-in-time consistency.
CRITICAL UPDATE: Applies np.log1p() to the final aggregated values to handle skew.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from fredapi import Fred
from datetime import datetime

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, FRED_API_KEY, START_DATE, END_DATE, setup_logger, TEMP_DIR

logger = setup_logger(__file__, TEMP_DIR)

# State name to FRED code mapping (from NOAA format to 2-letter codes)
STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC'
}

# Expected damage/injury metrics
EXPECTED_METRICS = [
    'total_damage_real',
    'total_property_damage_real',
    'total_crop_damage_real',
    'deaths_direct',
    'deaths_indirect',
    'injuries_direct',
    'injuries_indirect'
]

# NOAA data path
NOAA_MASTER_PATH = DATA_PATH / "Exogenous_data" / "NOAA_data" / "NOAA_master.parquet"
NOAA_WEIGHTED_DIR = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / "decades"


def download_state_employment_vintages(fred: Fred, end_date: str = END_DATE) -> pd.DataFrame:
    """
    Download ALL vintages for all state employment series once (as of end_date).
    Returns DataFrame with columns: state_code, date, value, realtime_start
    Similar pattern to fred_snapshots.py
    """
    as_of_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    # Update cache name to reflect NSA data
    cache_path = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / f"state_employment_vintages_nsa_{as_of_str}.parquet"
    
    if cache_path.exists():
        logger.info(f"Loading state employment vintages from cache: {cache_path}")
        return pd.read_parquet(cache_path)
        
    states = list(STATE_NAME_TO_CODE.values())
    # Use NAN suffix for Non-Seasonally Adjusted data (goes back further)
    fred_codes = {state: f"{state}NAN" for state in states}
    
    logger.info(f"Downloading ALL vintages for 51 state employment series (NSA) as of {as_of_str}")
    
    all_vintages = []
    
    for i, (state, code) in enumerate(fred_codes.items(), 1):
        try:
            # Get ALL vintages as of end_date
            vintage_df = fred.get_series_as_of_date(code, as_of_date=as_of_str)
            
            if vintage_df.empty:
                logger.warning(f"[{i}/51] No data for {state} ({code})")
                continue
            
            # Transform to our format
            vintage_df['state_code'] = state
            vintage_df['date'] = pd.to_datetime(vintage_df['date'])
            vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
            vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')
            
            all_vintages.append(vintage_df[['state_code', 'date', 'value', 'realtime_start']])
            
            logger.info(f"[{i}/51] Downloaded {state} ({code}): {len(vintage_df)} vintages")
            
            # Rate limiting: sleep every 10 series
            if i % 10 == 0 and i < len(fred_codes):
                logger.info(f"Rate limiting: sleeping 5 seconds...")
                import time
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"[{i}/51] Error fetching {state} employment: {e}")
            continue
    
    if not all_vintages:
        raise ValueError("No state employment data retrieved")
    
    combined = pd.concat(all_vintages, ignore_index=True)
    logger.info(f"Downloaded {len(combined)} total vintage records for {combined['state_code'].nunique()} states")
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    logger.info(f"Saved vintage cache to {cache_path}")
    
    return combined


def get_state_employment_weights(vintages_df: pd.DataFrame, snap_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate employment-share weights from vintages as of snap_date.
    Uses point-in-time filtering: only data released by snap_date.
    
    If snap_date is before the earliest available vintage (e.g. pre-2005),
    falls back to the earliest vintage as a best-effort proxy.
    """
    # Check if we have any vintage by snap_date
    earliest_vintage = vintages_df['realtime_start'].min()
    
    if snap_date < earliest_vintage:
        # Fallback: Use earliest vintage
        logger.warning(f"Snapshot {snap_date.date()} predates earliest vintage ({earliest_vintage.date()}). Using earliest vintage as proxy.")
        known_df = vintages_df[vintages_df['realtime_start'] == earliest_vintage].copy()
    else:
        # Normal case: Filter to vintages known BEFORE snap_date (strict <)
        # Changed from <= to strict < to prevent same-day data leakage
        known_df = vintages_df[vintages_df['realtime_start'] < snap_date].copy()
    
    if known_df.empty:
        raise ValueError(f"No state employment data available as of {snap_date}")
    
    # For each (state_code, date), keep the latest vintage known by snap_date
    known_df = known_df.sort_values(['state_code', 'date', 'realtime_start'])
    latest_vintage = known_df.drop_duplicates(['state_code', 'date'], keep='last')
    
    # Get employment for snap_date's month (or most recent prior)
    snap_month = snap_date.to_period('M')
    
    state_employment = {}
    lags = []
    
    for state in latest_vintage['state_code'].unique():
        state_data = latest_vintage[latest_vintage['state_code'] == state]
        
        # Try to get exact month
        month_data = state_data[state_data['date'].dt.to_period('M') == snap_month]
        
        if not month_data.empty:
            row = month_data.iloc[-1]
            employment = row['value']
            state_employment[state] = float(employment)
            lags.append((row['realtime_start'] - row['date']).days)
        else:
            # Fall back to most recent available before snap_date
            prior_data = state_data[state_data['date'] <= snap_date]
            if not prior_data.empty:
                row = prior_data.sort_values('date').iloc[-1]
                employment = row['value']
                state_employment[state] = float(employment)
                lags.append((row['realtime_start'] - row['date']).days)
    
    if not state_employment:
        raise ValueError(f"No state employment data retrieved for {snap_date}")
    
    # Log lag stats
    if lags:
        avg_lag = sum(lags) / len(lags)
        min_lag = min(lags)
        max_lag = max(lags)
        logger.info(f"Vintage Lag Stats for {snap_date.date()}: Avg={avg_lag:.1f} days, Min={min_lag}, Max={max_lag}")

    # Calculate weights
    total_employment = sum(state_employment.values())
    weights_dict = {state: emp / total_employment for state, emp in state_employment.items()}
    
    # Create DataFrame
    weights_df = pd.DataFrame({
        'state_code': list(weights_dict.keys()),
        'employment': [state_employment[s] for s in weights_dict.keys()],
        'weight': list(weights_dict.values())
    })
    
    # Verify weights sum to ~1.0
    weight_sum = weights_df['weight'].sum()
    if not np.isclose(weight_sum, 1.0):
        logger.warning(f"Weights sum to {weight_sum}, expected 1.0")
    
    logger.info(f"Snapshot {snap_date.date()}: Calculated weights for {len(weights_df)} states (sum={weight_sum:.6f})")
    
    return weights_df


def load_and_parse_noaa_master() -> pd.DataFrame:
    """
    Load NOAA master file and parse series names into metric and state.
    Filter to 51 valid states only.
    """
    if not NOAA_MASTER_PATH.exists():
        raise FileNotFoundError(f"NOAA master file not found: {NOAA_MASTER_PATH}")
    
    logger.info(f"Loading NOAA data from {NOAA_MASTER_PATH}")
    noaa_full = pd.read_parquet(NOAA_MASTER_PATH)
    
    # Parse series_name: {metric}_{STATE}
    # Example: deaths_direct_ALABAMA, total_property_damage_real_WYOMING
    noaa_full['state_name'] = noaa_full['series_name'].str.rsplit('_', n=1).str[-1]
    noaa_full['metric'] = noaa_full['series_name'].str.rsplit('_', n=1).str[0]
    
    # Filter to 51 valid states
    valid_states = list(STATE_NAME_TO_CODE.keys())
    noaa_states = noaa_full[noaa_full['state_name'].isin(valid_states)].copy()
    
    # Filter to 6 required metrics only
    # Exclude: total_damage_real (it's property+crop, would be collinear), nominal, derived, storm_count
    required_metrics = [
        'storm_count',                 # Storm count (independent)
        'total_property_damage_real',  # Property damage (independent)
        'total_crop_damage_real',      # Agricultural damage (independent)
        'deaths_direct',               # Direct deaths
        'deaths_indirect',             # Indirect deaths
        'injuries_direct',             # Direct injuries
        'injuries_indirect'            # Indirect injuries
    ]
    noaa_states = noaa_states[noaa_states['metric'].isin(required_metrics)].copy()
    
    logger.info(f"Filtered NOAA data: {len(noaa_full)} → {len(noaa_states)} rows (51 states, 6 metrics)")
    logger.info(f"Unique states: {noaa_states['state_name'].nunique()}")
    logger.info(f"Unique metrics: {noaa_states['metric'].nunique()}")
    logger.info(f"Metrics: {sorted(noaa_states['metric'].unique())}")
    
    return noaa_states


def create_weighted_national_aggregates(
    noaa_states: pd.DataFrame,
    weights: pd.DataFrame,
    snap_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Apply NFP employment weights to state damage data and aggregate to national level.
    Uses release_date for point-in-time filtering.
    CRITICAL: Applies log1p transformation to the final aggregated result.

    Args:
        noaa_states: State-level NOAA data with columns: date, metric, value, release_date, state_name
        weights: Employment weights with columns: state_code, weight
        snap_date: NFP release date to align snapshot with

    Returns:
        DataFrame with columns: date, series_name, value, release_date
    """
    # Filter to data known by snap_date
    noaa_states['date'] = pd.to_datetime(noaa_states['date'])
    noaa_states['release_date'] = pd.to_datetime(noaa_states['release_date'])

    # Point-in-time filter: only include data released BEFORE snap_date (strict <)
    # Changed from <= to strict < to prevent same-day data leakage
    noaa_filtered = noaa_states[noaa_states['release_date'] < snap_date].copy()

    logger.info(f"Point-in-time filter: {len(noaa_states)} → {len(noaa_filtered)} rows (released by {snap_date.date()})")

    # Add state codes
    noaa_filtered['state_code'] = noaa_filtered['state_name'].map(STATE_NAME_TO_CODE)

    # Merge with weights
    noaa_weighted = noaa_filtered.merge(weights[['state_code', 'weight']], on='state_code', how='inner')

    logger.info(f"Merged NOAA data with weights: {len(noaa_weighted)} rows")

    # Apply weights: weighted_value = value * employment_weight
    noaa_weighted['weighted_value'] = noaa_weighted['value'] * noaa_weighted['weight']

    # Aggregate to national level: sum across states for each (date, metric)
    weighted_national = noaa_weighted.groupby(['date', 'metric'])['weighted_value'].sum().reset_index()

    # Get release_date for each event date (should be consistent across states)
    release_dates = noaa_filtered.groupby('date')['release_date'].first().reset_index()
    weighted_national = weighted_national.merge(release_dates, on='date', how='left')

    # --- TRANSFORM LOGIC ---
    # Apply log1p to the aggregated SUM. This stabilizes the feature while
    # preserving the relative economic impact of different periods.
    weighted_national['weighted_value'] = np.log1p(weighted_national['weighted_value'])

    # Rename for clarity - append _log so downstream code knows this is transformed
    weighted_national['series_name'] = weighted_national['metric'] + '_weighted_log'

    weighted_national = weighted_national.rename(columns={'weighted_value': 'value'})
    weighted_national = weighted_national[['date', 'series_name', 'value', 'release_date']]

    logger.info(f"Created weighted national aggregates (LOG TRANSFORMED): {len(weighted_national)} rows")
    logger.info(f"Unique metrics: {weighted_national['series_name'].nunique()}")
    logger.info(f"Date range: {weighted_national['date'].min()} to {weighted_national['date'].max()}")

    return weighted_national


def load_nfp_release_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load NFP target file to extract actual NFP release dates.
    These dates determine when each snapshot should be created.

    Returns:
        DataFrame with columns: ds (event date), release_date (NFP release)
    """
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"

    if not nfp_path.exists():
        raise FileNotFoundError(f"NFP target file not found: {nfp_path}")

    logger.info(f"Loading NFP release dates from {nfp_path}")
    nfp_df = pd.read_parquet(nfp_path)

    # Extract only ds and release_date columns
    nfp_releases = nfp_df[['ds', 'release_date']].copy()
    nfp_releases['ds'] = pd.to_datetime(nfp_releases['ds'])
    nfp_releases['release_date'] = pd.to_datetime(nfp_releases['release_date'])

    # Filter to date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    nfp_releases = nfp_releases[
        (nfp_releases['ds'] >= start_dt) & (nfp_releases['ds'] <= end_dt)
    ]

    logger.info(f"Loaded {len(nfp_releases)} NFP release dates")
    logger.info(f"Date range: {nfp_releases['ds'].min().date()} to {nfp_releases['ds'].max().date()}")

    return nfp_releases


def create_noaa_weighted_snapshots(
    start_date: str = START_DATE,
    end_date: str = END_DATE
):
    """
    Create snapshots of NFP-weighted NOAA data aligned with NFP release dates.
    Each snapshot is point-in-time correct using release_date filtering.

    CRITICAL CHANGE: Snapshots are now aligned with NFP release dates instead of
    generic month-end dates. This ensures temporal consistency with the target variable.

    Downloads state employment vintages ONCE, then processes all snapshots.
    Similar pattern to fred_snapshots.py to avoid rate limiting.
    """
    # Initialize FRED API
    fred = Fred(api_key=FRED_API_KEY)

    # Load NFP release dates to determine snapshot schedule
    logger.info("=" * 80)
    logger.info("STEP 1: Loading NFP release dates")
    logger.info("=" * 80)
    nfp_releases = load_nfp_release_dates(start_date, end_date)

    # Download ALL state employment vintages ONCE (as of end_date)
    logger.info("=" * 80)
    logger.info("STEP 2: Downloading state employment vintages (ONCE)")
    logger.info("=" * 80)
    try:
        employment_vintages = download_state_employment_vintages(fred, end_date=end_date)
    except Exception as e:
        logger.error(f"Failed to download state employment vintages: {e}")
        raise

    # Load full NOAA data once
    logger.info("=" * 80)
    logger.info("STEP 3: Loading NOAA master data")
    logger.info("=" * 80)
    noaa_states = load_and_parse_noaa_master()

    # Determine earliest available employment date
    min_employment_date = employment_vintages['date'].min()
    logger.info(f"Earliest state employment data: {min_employment_date.date()}")

    logger.info("=" * 80)
    logger.info(f"STEP 4: Creating {len(nfp_releases)} NOAA weighted snapshots (aligned with NFP)")
    logger.info(f"From {nfp_releases['release_date'].min().date()} to {nfp_releases['release_date'].max().date()}")
    logger.info("=" * 80)

    for i, (idx, row) in enumerate(nfp_releases.iterrows(), 1):
        event_date = row['ds']
        snap_date = row['release_date']  # NFP release date

        year = event_date.year
        decade = f"{year // 10 * 10}s"
        month_str = event_date.strftime('%Y-%m')

        # Create directory structure
        save_dir = NOAA_WEIGHTED_DIR / decade / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{month_str}.parquet"

        # Skip if exists
        if save_path.exists():
            logger.info(f"[{i}/{len(nfp_releases)}] Snapshot exists for {month_str}, skipping")
            continue

        try:
            # Calculate employment weights from vintages (point-in-time as of snap_date)
            try:
                weights = get_state_employment_weights(employment_vintages, snap_date)
            except ValueError:
                # If no weights available (e.g. pre-1990), treat as missing
                # We simply don't create the weighted snapshot for this month
                # create_master_snapshots.py will handle missing files by skipping them
                if i % 12 == 0:  # Reduce log noise
                    logger.info(f"[{i}/{len(nfp_releases)}] No weights for {month_str} (pre-1990), skipping")
                continue

            # Create weighted aggregates (with point-in-time filtering using snap_date)
            weighted_national = create_weighted_national_aggregates(noaa_states, weights, snap_date)

            # Add snapshot_date column (NFP release date)
            weighted_national['snapshot_date'] = snap_date

            # Reorder columns to match requirements: date, series_name, value, snapshot_date, release_date
            weighted_national = weighted_national[['date', 'series_name', 'value', 'snapshot_date', 'release_date']]

            # Save
            weighted_national.to_parquet(save_path, index=False)

            logger.info(f"[{i}/{len(nfp_releases)}] Saved {month_str}: {len(weighted_national)} rows (snapshot={snap_date.date()})")

        except Exception as e:
            logger.error(f"Error creating snapshot for {month_str}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 80)
    logger.info("NOAA weighted snapshot generation complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("Starting NOAA weighted aggregation")
    create_noaa_weighted_snapshots()
    logger.info("Complete!")