"""
Path Utilities

Centralized snapshot path generation for consistent file access across the project.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH


def get_decade_year_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    """
    Build path using decade/year/month structure.
    
    Args:
        base_dir: Base directory for snapshots
        date_ts: Timestamp for the snapshot
        
    Returns:
        Path in format: base_dir/decades/{decade}s/{year}/{YYYY-MM}.parquet
    """
    decade = f"{date_ts.year // 10 * 10}s"
    year = date_ts.year
    month_str = date_ts.strftime('%Y-%m')
    
    return base_dir / "decades" / decade / str(year) / f"{month_str}.parquet"


def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to FRED employment (endogenous) snapshot.
    
    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)
        
    Returns:
        Path to snapshot parquet file
        
    Example:
        >>> get_fred_snapshot_path(pd.Timestamp('2020-03-31'))
        Path('.../Data/fred_data/decades/2020s/2020/2020-03.parquet')
    """
    base_dir = DATA_PATH / "fred_data"
    return get_decade_year_path(base_dir, snapshot_date)


def get_prepared_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to prepared (transformed) FRED snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to prepared snapshot parquet file
    """
    base_dir = DATA_PATH / "fred_data_prepared"
    return get_decade_year_path(base_dir, snapshot_date)


def get_exogenous_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to FRED exogenous (VIX, SP500, Oil, etc.) snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to exogenous snapshot parquet file
    """
    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data"
    return get_decade_year_path(base_dir, snapshot_date)


def get_master_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to consolidated master snapshot.
    
    Master snapshots contain all data sources (FRED, Unifier, ADP, NOAA)
    merged into a single point-in-time snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to master snapshot parquet file
    """
    base_dir = DATA_PATH / "Exogenous_data" / "master_snapshots"
    return get_decade_year_path(base_dir, snapshot_date)


def get_unifier_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to Unifier data snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to Unifier snapshot parquet file
    """
    base_dir = DATA_PATH / "Exogenous_data" / "unifier_data"
    return get_decade_year_path(base_dir, snapshot_date)


def get_noaa_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to NOAA storm data snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to NOAA snapshot parquet file
    """
    base_dir = DATA_PATH / "Exogenous_data" / "noaa_weighted"
    return get_decade_year_path(base_dir, snapshot_date)


def get_adp_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Build path to ADP employment snapshot.
    
    Args:
        snapshot_date: Month-end timestamp
        
    Returns:
        Path to ADP snapshot parquet file
    """
    base_dir = DATA_PATH / "Exogenous_data" / "adp_data"
    return get_decade_year_path(base_dir, snapshot_date)


def get_target_path(target_type: str = 'nsa', release_type: str = 'first') -> Path:
    """
    Build path to target data file.
    
    Args:
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        
    Returns:
        Path to target parquet file
    """
    filename = f"y_{target_type}_{release_type}_release.parquet"
    return DATA_PATH / "NFP_target" / filename


if __name__ == "__main__":
    # Test path generation
    test_date = pd.Timestamp('2020-03-31')
    
    print(f"FRED snapshot: {get_fred_snapshot_path(test_date)}")
    print(f"Master snapshot: {get_master_snapshot_path(test_date)}")
    print(f"Exogenous snapshot: {get_exogenous_snapshot_path(test_date)}")
    print(f"Target path: {get_target_path('nsa', 'first')}")
