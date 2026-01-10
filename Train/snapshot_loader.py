"""
Snapshot Data Loader

Loads point-in-time snapshots of endogenous (employment) and exogenous (macro) data.

NOTE: This module re-exports functions from data_loader.py for backwards compatibility.
New code should import directly from Train.data_loader instead.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger

# Re-export from data_loader for backwards compatibility
from Train.data_loader import (
    get_fred_snapshot_path,
    get_master_snapshot_path,
    load_fred_snapshot,
    load_master_snapshot,
    load_target_data,
)

logger = setup_logger(__file__, TEMP_DIR)


def load_snapshot_data(snapshot_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """
    Load all data available as of snapshot_date.

    Args:
        snapshot_date: Month-end timestamp (e.g., pd.Timestamp('2020-01-31'))

    Returns:
        Dictionary with:
            'endogenous': DataFrame [date, series_name, value, series_code, snapshot_date]
            'exogenous': DataFrame [date, series_name, value, snapshot_date]
            'snapshot_date': pd.Timestamp

    Raises:
        FileNotFoundError: If snapshot files don't exist
    """
    # Load endogenous employment data using data_loader functions
    endo_df = load_fred_snapshot(snapshot_date)
    if endo_df is None:
        endo_path = get_fred_snapshot_path(snapshot_date)
        raise FileNotFoundError(f"Endogenous snapshot not found: {endo_path}")

    logger.info(f"Loaded endogenous snapshot: {len(endo_df)} rows, {endo_df['series_name'].nunique()} series")

    # Load exogenous master snapshot using data_loader functions
    exog_df = load_master_snapshot(snapshot_date)
    if exog_df is None:
        exog_path = get_master_snapshot_path(snapshot_date)
        raise FileNotFoundError(f"Exogenous snapshot not found: {exog_path}")

    logger.info(f"Loaded exogenous snapshot: {len(exog_df)} rows, {exog_df['series_name'].nunique()} series")

    return {
        'endogenous': endo_df,
        'exogenous': exog_df,
        'snapshot_date': snapshot_date
    }


if __name__ == "__main__":
    # Test snapshot loading
    test_date = pd.Timestamp('2020-01-31')
    logger.info(f"Testing snapshot loader for {test_date.date()}")

    data = load_snapshot_data(test_date)

    logger.info(f"\nEndogenous data shape: {data['endogenous'].shape}")
    logger.info(f"Exogenous data shape: {data['exogenous'].shape}")
    logger.info(f"Snapshot date: {data['snapshot_date']}")

    # Show sample
    logger.info(f"\nEndogenous sample:\n{data['endogenous'].head()}")
    logger.info(f"\nExogenous sample:\n{data['exogenous'].head()}")
