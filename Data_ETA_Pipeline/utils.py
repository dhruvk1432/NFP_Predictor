"""
Shared utilities for data loading scripts.

This module contains common functions used across multiple data loaders
to reduce code duplication and ensure consistency.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def get_snapshot_path(base_dir: Union[str, Path], obs_month: pd.Timestamp) -> Path:
    """
    Generate the standard snapshot file path for a given observation month.

    All data loaders use the same directory structure:
    base_dir/decades/{decade}s/{year}/{year}-{month}.parquet

    Args:
        base_dir: Base directory for snapshots (e.g., DATA_PATH / "Exogenous_data" / "fred_data")
        obs_month: Observation month as a Timestamp (e.g., pd.Timestamp('2024-01-01'))

    Returns:
        Path object pointing to the snapshot file location

    Example:
        >>> get_snapshot_path(Path('/data/fred'), pd.Timestamp('2024-06-01'))
        PosixPath('/data/fred/decades/2020s/2024/2024-06.parquet')
    """
    base_dir = Path(base_dir)
    obs_month = pd.Timestamp(obs_month)

    decade = f"{obs_month.year // 10 * 10}s"
    year = obs_month.strftime('%Y')
    month = obs_month.strftime('%Y-%m')

    save_dir = base_dir / "decades" / decade / year
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / f"{month}.parquet"


def flatten_multiindex_columns(df: pd.DataFrame, sep: str = '_') -> pd.DataFrame:
    """
    Flatten a DataFrame's MultiIndex columns into single-level column names.

    Commonly used after pandas aggregation operations that create MultiIndex columns.

    Args:
        df: DataFrame with potentially MultiIndex columns
        sep: Separator to use when joining column levels (default: '_')

    Returns:
        DataFrame with flattened column names

    Example:
        >>> df.columns  # Before: MultiIndex([('value', 'mean'), ('value', 'max')])
        >>> df = flatten_multiindex_columns(df)
        >>> df.columns  # After: Index(['value_mean', 'value_max'])
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join(col).strip() for col in df.columns.values]
    return df


def get_decade_folder(year: int) -> str:
    """
    Return decade folder name like '2020s' for a given year.

    Args:
        year: Year (e.g., 2024)

    Returns:
        Decade folder name (e.g., '2020s')

    Example:
        >>> get_decade_folder(2024)
        '2020s'
        >>> get_decade_folder(1999)
        '1990s'
    """
    decade_start = (year // 10) * 10
    return f"{decade_start}s"
