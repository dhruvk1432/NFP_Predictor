"""
Create ADP Employment Data Snapshots

Converts raw ADP data into monthly snapshots aligned with NFP release dates.
Each snapshot is point-in-time correct: only includes ADP data released before
the NFP release date for that month.

Output format matches other exogenous snapshots:
- date: reference month for the ADP data
- series_name: ADP_actual, ADP_forecast (ADP_previous dropped due to multicollinearity)
- value: employment change value
- snapshot_date: NFP release date (when this snapshot was "known")
- release_date: when ADP was actually published
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, START_DATE, END_DATE, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# Paths
ADP_DATA_DIR = DATA_PATH / "Exogenous_data" / "ADP_data"
ADP_RAW = ADP_DATA_DIR / "ADP_Employment_Change.parquet"
ADP_SNAPSHOTS_BASE = DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades"
NFP_TARGET_PATH = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"


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


def create_adp_snapshots(start_date: str = START_DATE, end_date: str = END_DATE):
    """
    Create ADP snapshots aligned with NFP release dates.

    For each NFP release:
    - snapshot_date = NFP release date
    - Include all ADP data where release_date <= NFP release date
    - Keep release_date in output for traceability

    Saves to: ADP_snapshots/decades/YYY0s/YYYY/YYYY-MM.parquet
    """
    # Load raw ADP data (long format)
    if not ADP_RAW.exists():
        raise FileNotFoundError(f"ADP data not found: {ADP_RAW}")

    df = pd.read_parquet(ADP_RAW)
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
        event_date = row['ds']  # The month this NFP refers to (e.g., 2024-11-01)
        snap_date = row['release_date']  # When NFP was released (e.g., 2024-12-06)

        # Point-in-time filter: only include ADP data released BEFORE the NFP release
        # Changed from <= to strict < to prevent same-day data leakage
        # Data released ON the NFP release date cannot be used for prediction
        snapshot_df = df[df['release_date'] < snap_date].copy()

        if len(snapshot_df) == 0:
            continue

        # For each (date, series_name), keep MOST RECENT release as of this snapshot
        # This handles any ADP revisions
        snapshot_df = snapshot_df.sort_values('release_date', ascending=False)
        snapshot_df = snapshot_df.drop_duplicates(subset=['date', 'series_name'], keep='first')

        # Keep release_date for traceability (matches NOAA weighted format)
        snapshot_df = snapshot_df[['date', 'series_name', 'value', 'release_date']].copy()
        snapshot_df['snapshot_date'] = snap_date

        # Reorder columns to match other snapshot formats
        snapshot_df = snapshot_df[['date', 'series_name', 'value', 'snapshot_date', 'release_date']]

        # Determine file path based on event_date (the month being predicted)
        year = event_date.year
        decade = f"{(year // 10) * 10}s"
        year_str = str(year)
        month_str = event_date.strftime('%Y-%m')

        output_dir = ADP_SNAPSHOTS_BASE / decade / year_str
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{month_str}.parquet"
        snapshot_df.to_parquet(output_file, index=False)

        snapshots_created += 1
        if snapshots_created % 50 == 0:
            logger.info(f"  Created {snapshots_created} snapshots...")

    logger.info(f"Created {snapshots_created} ADP snapshots")
    logger.info(f"Location: {ADP_SNAPSHOTS_BASE}")

    # Show sample
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

    # Find all snapshot files
    snapshot_files = list(ADP_SNAPSHOTS_BASE.rglob("*.parquet"))
    logger.info(f"Found {len(snapshot_files)} snapshot files")

    if len(snapshot_files) == 0:
        logger.error("No snapshots found!")
        return

    # Check a few random snapshots
    import random
    samples = random.sample(snapshot_files, min(3, len(snapshot_files)))

    for file in samples:
        df = pd.read_parquet(file)
        logger.info(f"{file.relative_to(ADP_SNAPSHOTS_BASE)}:")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Series: {df['series_name'].nunique()}")
        logger.info(f"  Duplicates: {df.duplicated(subset=['date', 'series_name']).sum()}")

        # Verify release_date is present
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


def verify_target_month_mapping():
    """Verify the target month mapping is correct by checking a few examples."""
    logger.info("Verifying target month mapping...")

    # Load raw CSV to check mapping
    raw_csv = ADP_DATA_DIR / "ADP_Employment_Change_raw.csv"
    if not raw_csv.exists():
        logger.warning("Raw CSV not found, skipping mapping verification")
        return

    import re
    df_raw = pd.read_csv(raw_csv)

    # Check first few rows with explicit month indicators
    logger.info("Sample target month mappings:")
    for i, row in df_raw.head(10).iterrows():
        release_str = row['Release Date']
        # Extract month from parentheses if present
        month_match = re.search(r'\(([A-Za-z]+)\)', release_str)
        if month_match:
            ref_month = month_match.group(1)
            logger.info(f"  {release_str} -> Reference month: {ref_month}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ADP SNAPSHOT CREATION (NFP-ALIGNED)")
    logger.info("=" * 70)

    create_adp_snapshots()
    validate_snapshots()
    verify_target_month_mapping()

    logger.info("=" * 70)
    logger.info("ADP snapshot system complete!")
    logger.info("=" * 70)
