"""
ADP Employment Pipeline
=======================
Loads ADP employment data from investing.com REST API and creates monthly
snapshots aligned with NFP release dates.

Data Source:
    investing.com REST API (ADP_actual only, back to 2001)

Output:
    - Intermediate: DATA_PATH/Exogenous_data/ADP_data/ADP_Employment_Change.parquet
    - Final:        DATA_PATH/Exogenous_data/ADP_snapshots/decades/{decade}s/{year}/{YYYY-MM}.parquet
"""

import pandas as pd
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, START_DATE, END_DATE, TEMP_DIR, setup_logger
from Data_ETA_Pipeline.perf_stats import (
    install_hooks,
    profiled,
    register_atexit_dump,
)
from utils.transforms import add_symlog_copies, add_pct_change_copies, compute_all_features

logger = setup_logger(__file__, TEMP_DIR)
install_hooks()
register_atexit_dump("adp_pipeline", output_dir=TEMP_DIR / "perf")

# =============================================================================
# PATHS
# =============================================================================

EXOG_ADP_DIR = DATA_PATH / "Exogenous_data" / "ADP_data"
EXOG_ADP_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_ADP_PARQUET = EXOG_ADP_DIR / "ADP_Employment_Change.parquet"

ADP_SNAPSHOTS_BASE = DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades"
NFP_TARGET_PATH = DATA_PATH / "NFP_target" / "total_nsa_first_release.parquet"

# investing.com API endpoint for ADP event occurrences (event_id=1)
ADP_API_URL = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/1/occurrences"


# =============================================================================
# SECTION 1: API FETCHING
# =============================================================================

@profiled("adp_pipeline.fetch_adp_from_api")
def fetch_adp_from_api() -> pd.DataFrame:
    """
    Fetch raw ADP national employment change data from the investing.com REST API.

    This function pulls historical 'event' occurrences, capturing what the reported 
    ADP employment value was at the exact moment of its release. This is critical for 
    point-in-time modeling to avoid using cleanly revised current-day data.

    Data details:
    - Only 'actual' reported values are retained (forecasts and 'previous' estimates are dropped).
    - Extracted values are intrinsically measured in thousands (K), precisely natively aligning 
      with the scaling of the NFP target.
    - Dates are parsed to trace both the period the employment covers (`date`) and the specific
      timestamp the public became aware of it (`release_date`).

    Returns:
        pd.DataFrame: Structured DataFrame containing point-in-time releases. Columns include 
                      ['date', 'series_name', 'value', 'release_date', 'series_type'].
                      Returns an empty DataFrame upon network or parsing failure.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Origin': 'https://www.investing.com',
        'Referer': 'https://www.investing.com/',
    }
    params = {'domain_id': 1, 'limit': 1000}

    try:
        logger.info(f"Fetching ADP data from API: {ADP_API_URL}")
        resp = requests.get(ADP_API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        items = data.get('occurrences', [])
        logger.info(f"API returned {len(items)} occurrences")

        if not items:
            logger.warning("API returned empty occurrences list")
            return pd.DataFrame()

        rows = []
        for item in items:
            occurrence_time = item.get('occurrence_time')
            ref_period = item.get('reference_period')  # e.g. "Jan", "Feb"
            actual = item.get('actual')

            if not occurrence_time or not ref_period:
                continue

            # Parse release date from occurrence_time (strip tz for consistency)
            release_dt = pd.to_datetime(occurrence_time).tz_localize(None)

            # Derive reference date (first of the reference month)
            ref_month_num = pd.to_datetime(ref_period, format='%b').month
            release_year = release_dt.year

            # Handle year boundary: Jan/Feb release for Nov/Dec data → previous year
            if release_dt.month <= 2 and ref_month_num >= 11:
                ref_year = release_year - 1
            else:
                ref_year = release_year

            reference_date = pd.Timestamp(year=ref_year, month=ref_month_num, day=1)

            # ADP_actual (API values already in thousands, matching NFP target units)
            if actual is not None:
                rows.append({
                    'date': reference_date,
                    'series_name': 'ADP_actual',
                    'value': float(actual),
                    'release_date': release_dt,
                    'series_type': 'adp'
                })

        if not rows:
            logger.warning("No valid rows parsed from API response")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values(['date', 'series_name']).reset_index(drop=True)

        logger.info(f"Parsed {len(df)} ADP data points from API")
        logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"Series: {df['series_name'].unique().tolist()}")
        return df

    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing API data: {e}")
        return pd.DataFrame()


# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

@profiled("adp_pipeline.load_adp_data")
def load_adp_data() -> None:
    """
    Fetch ADP data from investing.com API and save to parquet.
    Skips if data already exists.
    """
    logger.info("Starting ADP data load")

    if CLEAN_ADP_PARQUET.exists():
        existing_data = pd.read_parquet(CLEAN_ADP_PARQUET)
        print(f"\u2713 ADP data already exists: {len(existing_data)} rows", flush=True)
        print(f"  Date range: {existing_data['date'].min().date()} to {existing_data['date'].max().date()}", flush=True)
        print(f"  Series: {existing_data['series_name'].unique().tolist()}", flush=True)
        logger.info("ADP data already exists, skipping")
        return

    print("Fetching ADP data from investing.com API...", flush=True)
    api_data = fetch_adp_from_api()

    if api_data.empty:
        raise RuntimeError("ADP data unavailable: API fetch failed")

    api_data.to_parquet(CLEAN_ADP_PARQUET, index=False)

    print(f"\u2713 Saved {len(api_data)} ADP rows", flush=True)
    print(f"  Date range: {api_data['date'].min().date()} to {api_data['date'].max().date()}", flush=True)
    print(f"  Series: {api_data['series_name'].unique().tolist()}", flush=True)

    logger.info("\u2713 ADP data load complete")


# =============================================================================
# SECTION 3: ADP SNAPSHOT CREATION
# =============================================================================

def load_nfp_release_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load NFP target file to extract actual NFP release dates.
    These dates determine when each snapshot should be created.

    Returns:
        DataFrame with columns: ds (event date), release_date (NFP release)
    """
    if not NFP_TARGET_PATH.exists():
        raise FileNotFoundError(f"NFP target file not found: {NFP_TARGET_PATH}")

    logger.info(f"Loading NFP release dates from {NFP_TARGET_PATH}")
    nfp_df = pd.read_parquet(NFP_TARGET_PATH)

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


@profiled("adp_pipeline.create_adp_snapshots")
def create_adp_snapshots(start_date: str = START_DATE, end_date: str = END_DATE):
    """
    Generate point-in-time master snapshots of the ADP data, precisely synchronized 
    with the release schedule of the Non-Farm Payrolls (NFP) report.

    This is the core anti-leakage defense mechanism for the ADP pipeline:
    For every monthly NFP release, it generates an isolated snapshot file containing 
    ONLY the ADP data that was publicly known *immediately prior* to the NFP report.

    Operational Logic:
    1. Iterates chronologically through every historical NFP release date.
    2. Filters the raw ADP dataset ensuring `release_date < NFP_release_date` (strict inequality ensures no same-day leakage).
    3. Retains only the most recent (freshest) reported value for any given reference month.
    4. Appends standard mathematical transformations (symmetric log scaling, momentum percentages).
    5. Saves the isolated chronological snapshot securely into the `decades/` subfolder structure.

    Args:
         start_date (str): Lower boundary for NFP events to process.
         end_date (str): Upper boundary for NFP events to process.
    """
    if not CLEAN_ADP_PARQUET.exists():
        raise FileNotFoundError(f"ADP data not found: {CLEAN_ADP_PARQUET}")

    df = pd.read_parquet(CLEAN_ADP_PARQUET)
    df['date'] = pd.to_datetime(df['date'])
    df['release_date'] = pd.to_datetime(df['release_date'])

    logger.info(f"Loaded {len(df)} raw ADP rows")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Release range: {df['release_date'].min().date()} to {df['release_date'].max().date()}")
    logger.info(f"Series: {df['series_name'].unique().tolist()}")

    # Load NFP release dates
    nfp_releases = load_nfp_release_dates(start_date, end_date)

    logger.info(f"Creating {len(nfp_releases)} ADP snapshots aligned with NFP releases...")

    snapshots_created = 0
    for _, row in nfp_releases.iterrows():
        event_date = row['ds']
        snap_date = row['release_date']

        # Point-in-time filter: strict < to prevent same-day leakage
        snapshot_df = df[df['release_date'] < snap_date].copy()

        if len(snapshot_df) == 0:
            continue

        # For each (date, series_name), keep MOST RECENT release as of this snapshot
        snapshot_df = snapshot_df.sort_values('release_date', ascending=False)
        snapshot_df = snapshot_df.drop_duplicates(subset=['date', 'series_name'], keep='first')

        snapshot_df = snapshot_df[['date', 'series_name', 'value', 'release_date']].copy()
        snapshot_df['snapshot_date'] = snap_date

        # Reorder columns
        snapshot_df = snapshot_df[['date', 'series_name', 'value', 'snapshot_date', 'release_date']]

        # Lean mode: skip symlog (trees are monotone-invariant), reduced features.
        snapshot_df = add_pct_change_copies(snapshot_df)
        snapshot_df = compute_all_features(snapshot_df, lean=True)

        # Determine file path
        year = event_date.year
        decade = f"{(year // 10) * 10}s"
        month_str = event_date.strftime('%Y-%m')

        output_dir = ADP_SNAPSHOTS_BASE / decade / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{month_str}.parquet"
        snapshot_df.to_parquet(output_file, index=False)

        snapshots_created += 1
        if snapshots_created % 50 == 0:
            logger.info(f"  Created {snapshots_created} snapshots...")

    logger.info(f"Created {snapshots_created} ADP snapshots")
    logger.info(f"Location: {ADP_SNAPSHOTS_BASE}")

    if snapshots_created > 0:
        logger.info(f"Sample snapshot (latest):")
        sample = pd.read_parquet(output_file)
        logger.info(f"  File: {output_file.name}")
        logger.info(f"  Rows: {len(sample)}")
        logger.info(f"  Columns: {sample.columns.tolist()}")
        logger.info(f"  Series: {sample['series_name'].unique().tolist()}")
        logger.info(f"  Date range: {sample['date'].min().date()} to {sample['date'].max().date()}")


def validate_snapshots():
    """Validate that snapshots were created correctly."""
    logger.info("Validating ADP snapshots...")

    snapshot_files = list(ADP_SNAPSHOTS_BASE.rglob("*.parquet"))
    logger.info(f"Found {len(snapshot_files)} snapshot files")

    if len(snapshot_files) == 0:
        logger.error("No snapshots found!")
        return

    import random
    samples = random.sample(snapshot_files, min(3, len(snapshot_files)))

    for file in samples:
        df = pd.read_parquet(file)
        logger.info(f"{file.relative_to(ADP_SNAPSHOTS_BASE)}:")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Series: {df['series_name'].nunique()}")
        logger.info(f"  Duplicates: {df.duplicated(subset=['date', 'series_name']).sum()}")

        if 'release_date' not in df.columns:
            logger.error(f"  Missing release_date column!")
        else:
            logger.info(f"  release_date present")

        # Verify point-in-time correctness
        if 'snapshot_date' in df.columns and 'release_date' in df.columns:
            snap_date = df['snapshot_date'].iloc[0]
            max_release = df['release_date'].max()
            if max_release <= snap_date:
                logger.info(f"  Point-in-time: OK (all releases <= snapshot)")
            else:
                logger.error(f"  Point-in-time VIOLATION: {max_release} > {snap_date}")

        if df.duplicated(subset=['date', 'series_name']).sum() > 0:
            logger.warning(f"  Contains duplicates!")


# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

def main():
    """
    Run the complete ADP pipeline:
    1. Fetch ADP data from API
    2. Create NFP-aligned snapshots with feature transforms
    3. Validate snapshots
    """
    logger.info("=" * 70)
    logger.info("ADP EMPLOYMENT PIPELINE")
    logger.info("=" * 70)

    load_adp_data()
    create_adp_snapshots()
    validate_snapshots()

    logger.info("=" * 70)
    logger.info("ADP pipeline complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
