"""
ADP Employment Pipeline
=======================
Loads ADP employment data from historical CSV, converts to MoM changes,
and creates monthly snapshots aligned with NFP release dates.

Merges: Load_Data/load_ADP_Employment_change.py + Prepare_Data/create_adp_snapshots.py

Output:
    - Intermediate: DATA_PATH/Exogenous_data/ADP_data/ADP_Employment_Change.parquet
    - Final:        DATA_PATH/Exogenous_data/ADP_snapshots/decades/{decade}s/{year}/{YYYY-MM}.parquet

Requires:
    pip install pandas pyarrow
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, START_DATE, END_DATE, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# =============================================================================
# PATHS
# =============================================================================

EXOG_ADP_DIR = DATA_PATH / "Exogenous_data" / "ADP_data"
EXOG_ADP_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_ADP_PARQUET = EXOG_ADP_DIR / "ADP_Employment_Change.parquet"
HISTORICAL_CSV = DATA_PATH.parent / "us-private-employment.csv"

ADP_SNAPSHOTS_BASE = DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades"
NFP_TARGET_PATH = DATA_PATH / "NFP_target" / "total_nsa_first_release.parquet"


# =============================================================================
# SECTION 1: ADP DATA LOADING (from load_ADP_Employment_change.py)
# =============================================================================

def get_first_wednesday(date: pd.Timestamp) -> pd.Timestamp:
    """Get the first Wednesday of the month following the given date.

    ADP typically releases employment data on the first Wednesday after month-end.
    """
    next_month = date + pd.DateOffset(months=1)
    first_day = next_month.replace(day=1)
    days_until_wed = (2 - first_day.dayofweek + 7) % 7  # Wednesday = 2
    if days_until_wed == 0:
        days_until_wed = 7  # If 1st is Wednesday, use that day
    return first_day + pd.Timedelta(days=days_until_wed - (7 if days_until_wed == 7 else 0))


def load_adp_from_csv() -> pd.DataFrame:
    """
    Load ADP private employment levels from CSV and convert to MoM changes.

    The CSV contains employment LEVELS (e.g., 134,588,000). We calculate the
    month-over-month change to get the ADP employment change figure.

    Returns:
        DataFrame with columns: date, series_name, value, release_date, series_type
    """
    if not HISTORICAL_CSV.exists():
        logger.error(f"Historical CSV not found: {HISTORICAL_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(HISTORICAL_CSV)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    # Calculate MoM change
    df['change'] = df['Private Employment'].diff()

    # First row has no change (NaN), drop it
    df = df.dropna(subset=['change'])

    # Impute release date: first Wednesday of following month
    df['release_date'] = df['DateTime'].apply(get_first_wednesday)

    # Format to match existing ADP data structure
    result = pd.DataFrame({
        'date': df['DateTime'],
        'series_name': 'ADP_actual',
        'value': df['change'],  # Raw employment change (not divided by 1000)
        'release_date': df['release_date'],
        'series_type': 'adp'
    })

    return result


def load_adp_data() -> None:
    """
    Load ADP employment data from historical CSV file.
    Saves intermediate parquet for downstream snapshot creation.
    """
    logger.info("Starting ADP data load")

    # Check if data already exists
    if CLEAN_ADP_PARQUET.exists():
        existing_data = pd.read_parquet(CLEAN_ADP_PARQUET)
        print(f"\u2713 ADP data already exists: {len(existing_data)} rows", flush=True)
        print(f"  Date range: {existing_data['date'].min().date()} to {existing_data['date'].max().date()}", flush=True)
        logger.info(f"ADP data already exists, skipping")
        return

    csv_data = load_adp_from_csv()
    if csv_data.empty:
        logger.error("No CSV data available")
        raise RuntimeError("ADP data unavailable: CSV file not found or empty")

    csv_data.to_parquet(CLEAN_ADP_PARQUET, index=False)
    print(f"\u2713 Saved {len(csv_data)} rows from historical CSV", flush=True)
    print(f"  Date range: {csv_data['date'].min().date()} to {csv_data['date'].max().date()}", flush=True)

    logger.info("\u2713 ADP data load complete")


# =============================================================================
# SECTION 2: ADP SNAPSHOT CREATION (from create_adp_snapshots.py)
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


def create_adp_snapshots(start_date: str = START_DATE, end_date: str = END_DATE):
    """
    Create ADP snapshots aligned with NFP release dates.

    For each NFP release:
    - snapshot_date = NFP release date
    - Include all ADP data where release_date < NFP release date (strict < to prevent leakage)
    - Keep release_date in output for traceability

    Saves to: ADP_snapshots/decades/{decade}s/{year}/{YYYY-MM}.parquet
    """
    if not CLEAN_ADP_PARQUET.exists():
        raise FileNotFoundError(f"ADP data not found: {CLEAN_ADP_PARQUET}")

    df = pd.read_parquet(CLEAN_ADP_PARQUET)
    df['date'] = pd.to_datetime(df['date'])
    df['release_date'] = pd.to_datetime(df['release_date'])

    logger.info(f"Loaded {len(df)} raw ADP rows")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Release range: {df['release_date'].min().date()} to {df['release_date'].max().date()}")

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
    1. Load ADP data from CSV and compute MoM changes
    2. Create NFP-aligned snapshots
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
