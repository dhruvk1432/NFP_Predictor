"""
Create NOAA Master File

Combines state-level NOAA data with US national data into a single master file.
Output format matches FRED master snapshots: long format with series_name column.
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
NOAA_DATA_DIR = DATA_PATH / "Exogenous_data" / "NOAA_data"
US_NOAA_FILE = NOAA_DATA_DIR / "US_NOAA_data.parquet"
NOAA_MASTER_FILE = NOAA_DATA_DIR / "NOAA_master.parquet"


def create_noaa_master():
    """
    Create NOAA master file combining state-level and US national data.

    Output format:
    - date: event month
    - series_name: metric_STATE (e.g., storm_count_ALABAMA, deaths_direct_US)
    - value: numeric value
    - release_date: estimated release date (month-end + 75 days per NOAA NCEI)
    """
    logger.info("Creating NOAA master file...")

    # 1. Load and process US national data
    logger.info("\n1. Processing US national data...")
    df_us = pd.read_parquet(US_NOAA_FILE)
    df_us = df_us.reset_index()  # event_month becomes column
    df_us = df_us.rename(columns={'event_month': 'date'})

    # Convert to long format with series names like "storm_count_US"
    # UPDATED: Use release_date instead of known_by_month_end
    value_cols = [col for col in df_us.columns if col not in ['date', 'inflation_factor', 'release_date']]

    us_long = df_us.melt(
        id_vars=['date', 'release_date'],
        value_vars=value_cols,
        var_name='metric',
        value_name='value'
    )
    us_long['series_name'] = us_long['metric'] + '_US'
    us_long = us_long[['date', 'series_name', 'value', 'release_date']]

    logger.info(f"  US data: {len(us_long)} rows, {us_long['series_name'].nunique()} series")

    # 2. Load and process state-level data
    logger.info("\n2. Processing state-level data...")
    state_files = list(NOAA_DATA_DIR.glob("*_NOAA_data.parquet"))
    state_files = [f for f in state_files if f.name != 'US_NOAA_data.parquet']

    logger.info(f"  Found {len(state_files)} state files")

    all_states = []
    for file in state_files:
        state = file.stem.replace('_NOAA_data', '')
        df_state = pd.read_parquet(file)
        df_state = df_state.reset_index()  # event_month becomes column
        df_state = df_state.rename(columns={'event_month': 'date'})

        # Convert to long format
        state_long = df_state.melt(
            id_vars=['date', 'release_date'],
            value_vars=value_cols,
            var_name='metric',
            value_name='value'
        )
        state_long['series_name'] = state_long['metric'] + '_' + state
        state_long = state_long[['date', 'series_name', 'value', 'release_date']]

        all_states.append(state_long)

    df_states = pd.concat(all_states, ignore_index=True)
    logger.info(f"  State data: {len(df_states)} rows, {df_states['series_name'].nunique()} series")

    # 3. Combine US and state data
    logger.info("\n3. Combining US and state data...")
    df_master = pd.concat([us_long, df_states], ignore_index=True)
    df_master = df_master.sort_values(['date', 'series_name']).reset_index(drop=True)

    logger.info(f"  Combined: {len(df_master)} rows")
    logger.info(f"  Series: {df_master['series_name'].nunique()}")
    logger.info(f"  Date range: {df_master['date'].min()} to {df_master['date'].max()}")

    # 4. Save master file
    df_master.to_parquet(NOAA_MASTER_FILE, index=False)
    logger.info(f"\n✓ Saved NOAA master: {NOAA_MASTER_FILE}")

    # 5. Show sample
    logger.info(f"\nSample US series:")
    us_series = sorted([s for s in df_master['series_name'].unique() if s.endswith('_US')])
    logger.info(f"  {us_series[:10]}")

    logger.info(f"\nSample state series:")
    state_series = sorted([s for s in df_master['series_name'].unique() if not s.endswith('_US')])
    logger.info(f"  {state_series[:10]}")

    return df_master


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("NOAA MASTER CREATION")
    logger.info("=" * 70)
    
    df = create_noaa_master()
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ NOAA master file complete!")
    logger.info("=" * 70)
