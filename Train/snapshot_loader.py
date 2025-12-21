"""
Snapshot Data Loader

Loads point-in-time snapshots of endogenous (employment) and exogenous (macro) data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)


def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to fred_data snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to snapshot parquet file
    """
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = snapshot_date.year
    month_str = snapshot_date.strftime('%Y-%m')
    
    path = DATA_PATH / "fred_data" / "decades" / decade / str(year) / f"{month_str}.parquet"
    return path


def get_master_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to exogenous master snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to snapshot parquet file
    """
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = snapshot_date.year
    month_str = snapshot_date.strftime('%Y-%m')
    
    path = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades" / decade / str(year) / f"{month_str}.parquet"
    return path


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
    # Load endogenous employment data
    endo_path = get_fred_snapshot_path(snapshot_date)
    if not endo_path.exists():
        raise FileNotFoundError(f"Endogenous snapshot not found: {endo_path}")
    
    endo_df = pd.read_parquet(endo_path)
    logger.info(f"Loaded endogenous snapshot: {len(endo_df)} rows, {endo_df['series_name'].nunique()} series")
    
    # Load exogenous master snapshot
    exog_path = get_master_snapshot_path(snapshot_date)
    if not exog_path.exists():
        raise FileNotFoundError(f"Exogenous snapshot not found: {exog_path}")
    
    exog_df = pd.read_parquet(exog_path)
    logger.info(f"Loaded exogenous snapshot: {len(exog_df)} rows, {exog_df['series_name'].nunique()} series")
    
    return {
        'endogenous': endo_df,
        'exogenous': exog_df,
        'snapshot_date': snapshot_date
    }


def load_target_data(release_type: str = 'first', target_type: str = 'sa') -> pd.DataFrame:
    """
    Load target data for validation/testing.
    
    Args:
        release_type: 'first' or 'last' release
        target_type: 'nsa' for NBEATSx targets, 'sa' for LightGBM/SA mapper targets
        
    Returns:
        DataFrame with target values
        
    Notes:
        - For NSA targets (NBEATSx): use target_type='nsa' → y_nsa_first_release.parquet
        - For SA targets (LightGBM): use target_type='sa' → y_sa_first_release.parquet
        - Data format: ds column with YYYY-MM-01 contains data for month YYYY-MM
    """
    # Determine file name based on target type
    if target_type == 'nsa':
        if release_type == 'first':
            filename = "y_nsa_first_release.parquet"
        else:
            filename = "y_nsa_last_release.parquet"
    elif target_type == 'sa':
        if release_type == 'first':
            filename = "y_sa_first_release.parquet"
        else:
            filename = "y_sa_last_release.parquet"
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'nsa' or 'sa'")
    
    path = DATA_PATH / "NFP_target" / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {target_type.upper()} {release_type} release targets from {filename}: {df.shape}")
    
    return df


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
