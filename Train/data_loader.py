"""
Data Loading Utilities for LightGBM NFP Model

Functions for loading snapshots, target data, and building training datasets.
Extracted from train_lightgbm_nfp.py for maintainability.

OPTIMIZATIONS:
- LRU cache for snapshot loading (avoids redundant I/O)
- Vectorized pivot_snapshot_to_wide using pandas native operations
- Pre-computed lag indices for batch feature generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger
from Train.config import (
    MASTER_SNAPSHOTS_DIR,
    FRED_SNAPSHOTS_DIR,
    FRED_PREPARED_DIR,
    TARGET_PATH_NSA,
    TARGET_PATH_SA,
    USE_PREPARED_FRED_DATA,
)

logger = setup_logger(__file__, TEMP_DIR)

# Module-level cache for loaded data
_snapshot_cache: Dict[str, pd.DataFrame] = {}
_target_cache: Dict[str, pd.DataFrame] = {}


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """Build path to FRED employment snapshot file."""
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')

    # Use prepared data if configured
    base_dir = FRED_PREPARED_DIR if USE_PREPARED_FRED_DATA else FRED_SNAPSHOTS_DIR
    return base_dir / decade / year / f"{month_str}.parquet"


def get_master_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """Build path to master snapshot file."""
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')
    return MASTER_SNAPSHOTS_DIR / decade / year / f"{month_str}.parquet"


# =============================================================================
# DATA LOADING (with caching)
# =============================================================================

def load_fred_snapshot(snapshot_date: pd.Timestamp, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load FRED employment snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: date, value, series_name, series_code, release_date
    """
    cache_key = f"fred_{snapshot_date.strftime('%Y-%m')}"

    if use_cache and cache_key in _snapshot_cache:
        return _snapshot_cache[cache_key].copy()

    path = get_fred_snapshot_path(snapshot_date)
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    if use_cache:
        _snapshot_cache[cache_key] = df

    return df.copy() if use_cache else df


def load_master_snapshot(snapshot_date: pd.Timestamp, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load master snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: date, series_name, value, release_date, series_code, snapshot_date
    """
    cache_key = f"master_{snapshot_date.strftime('%Y-%m')}"

    if use_cache and cache_key in _snapshot_cache:
        return _snapshot_cache[cache_key].copy()

    path = get_master_snapshot_path(snapshot_date)
    if not path.exists():
        logger.warning(f"Master snapshot not found: {path}")
        return None

    df = pd.read_parquet(path)

    if use_cache:
        _snapshot_cache[cache_key] = df

    return df.copy() if use_cache else df


def clear_snapshot_cache() -> None:
    """Clear the snapshot cache to free memory."""
    global _snapshot_cache
    _snapshot_cache.clear()
    logger.info("Snapshot cache cleared")


def load_target_data(target_type: str = 'nsa', use_cache: bool = True) -> pd.DataFrame:
    """
    Load and prepare target data (NSA or SA NFP levels) with derived features.

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: ds, y (level), y_mom (month-on-month change),
        and additional momentum/divergence features
    """
    cache_key = f"target_{target_type.lower()}"

    if use_cache and cache_key in _target_cache:
        return _target_cache[cache_key].copy()

    # Load appropriate target file
    if target_type.lower() == 'sa':
        target_path = TARGET_PATH_SA
        logger.info("Loading SA (seasonally adjusted) target data")
    else:
        target_path = TARGET_PATH_NSA
        logger.info("Loading NSA (non-seasonally adjusted) target data")

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    df = pd.read_parquet(target_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # Calculate MoM change (our prediction target)
    df['y_mom'] = df['y'].diff()

    # Rolling averages (for momentum/divergence features) - vectorized
    df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
    df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
    df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()

    # MoM rolling averages
    df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()

    # Divergence: Current minus rolling average (momentum indicator)
    df['divergence_3m'] = df['y'] - df['y_rolling_3m']
    df['divergence_6m'] = df['y'] - df['y_rolling_6m']

    # Acceleration: Diff of diffs (second derivative - is hiring accelerating?)
    df['acceleration'] = df['y_mom'].diff()

    # YoY change
    df['y_yoy'] = df['y'].diff(12)
    df['y_mom_yoy'] = df['y_mom'].diff(12)

    # Drop first row (no MoM for first observation) but keep future rows with NaN y
    df = df[df.index != 0].reset_index(drop=True)

    if use_cache:
        _target_cache[cache_key] = df
        logger.info(f"Loaded {target_type.upper()} target data: {len(df)} observations (cached)")
    else:
        logger.info(f"Loaded {target_type.upper()} target data: {len(df)} observations")

    return df.copy() if use_cache else df


def load_all_target_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both NSA and SA target data with all derived features.

    Returns:
        Tuple of (nsa_df, sa_df) with columns:
        - ds: date
        - y: level
        - y_mom: month-on-month change
        - Additional momentum/divergence features
    """
    nsa_df = load_target_data('nsa')
    sa_df = load_target_data('sa')
    return nsa_df, sa_df


def get_lagged_target_features(
    target_df: pd.DataFrame,
    target_month: pd.Timestamp,
    prefix: str = 'nfp'
) -> Dict[str, float]:
    """
    Get lagged target features for a specific prediction month.

    CRITICAL: Only use data from BEFORE target_month to avoid look-ahead bias.
    The target is released in the following month, so for month M prediction,
    we only know NFP values through month M-1.

    Args:
        target_df: DataFrame with target values
        target_month: Month we're predicting
        prefix: Prefix for feature names ('nfp_nsa' or 'nfp_sa')

    Returns:
        Dictionary of lagged features
    """
    features = {}

    # Only use data from before target_month
    available_df = target_df[target_df['ds'] < target_month].copy()

    if available_df.empty:
        return features

    available_df = available_df.sort_values('ds')

    # Lag 1 (most recent known)
    if len(available_df) >= 1:
        features[f'{prefix}_mom_lag1'] = available_df['y_mom'].iloc[-1]
        features[f'{prefix}_level_lag1'] = available_df['y'].iloc[-1]

    # Lag 2
    if len(available_df) >= 2:
        features[f'{prefix}_mom_lag2'] = available_df['y_mom'].iloc[-2]

    # Lag 3
    if len(available_df) >= 3:
        features[f'{prefix}_mom_lag3'] = available_df['y_mom'].iloc[-3]

    # Lag 6 (half year ago)
    if len(available_df) >= 6:
        features[f'{prefix}_mom_lag6'] = available_df['y_mom'].iloc[-6]

    # Lag 12 (year ago - same month last year)
    if len(available_df) >= 12:
        features[f'{prefix}_mom_lag12'] = available_df['y_mom'].iloc[-12]
        features[f'{prefix}_yoy'] = available_df['y_yoy'].iloc[-1] if 'y_yoy' in available_df.columns else np.nan

    # Rolling averages (from available data)
    if len(available_df) >= 3:
        features[f'{prefix}_rolling_3m'] = available_df['y_mom'].iloc[-3:].mean()

    if len(available_df) >= 6:
        features[f'{prefix}_rolling_6m'] = available_df['y_mom'].iloc[-6:].mean()

    if len(available_df) >= 12:
        features[f'{prefix}_rolling_12m'] = available_df['y_mom'].iloc[-12:].mean()

    # Volatility
    if len(available_df) >= 6:
        features[f'{prefix}_volatility_6m'] = available_df['y_mom'].iloc[-6:].std()

    # Momentum (difference between short and long term)
    if len(available_df) >= 12:
        short_term = available_df['y_mom'].iloc[-3:].mean()
        long_term = available_df['y_mom'].iloc[-12:].mean()
        features[f'{prefix}_momentum'] = short_term - long_term

    return features


def pivot_snapshot_to_wide(
    snapshot_df: pd.DataFrame,
    target_month: pd.Timestamp
) -> pd.DataFrame:
    """
    Convert long-format snapshot to wide format, taking the latest value for target month.

    OPTIMIZED: Uses vectorized pandas operations instead of per-series loops.
    Pivots data first, then computes features in batches.

    Args:
        snapshot_df: Long-format snapshot DataFrame
        target_month: Month we're predicting (format: YYYY-MM-01)

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    # Ensure date is datetime
    df = snapshot_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Filter to data available before/on target month (no look-ahead bias)
    df = df[df['date'] <= target_month]

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format: rows = dates, columns = series_name
    # Use 'first' aggregation in case of duplicates
    wide_df = df.pivot_table(
        index='date',
        columns='series_name',
        values='value',
        aggfunc='first'
    ).sort_index()

    if wide_df.empty:
        return pd.DataFrame()

    features = {}
    n_rows = len(wide_df)

    # Vectorized feature generation for all series at once
    for series_name in wide_df.columns:
        series = wide_df[series_name].dropna()
        n = len(series)

        if n == 0:
            continue

        # Latest value
        latest = series.iloc[-1]
        features[f'{series_name}_latest'] = latest

        # Short-term lags and changes
        if n >= 2:
            features[f'{series_name}_lag1'] = series.iloc[-2]
            features[f'{series_name}_mom_change'] = latest - series.iloc[-2]

        if n >= 3:
            features[f'{series_name}_lag2'] = series.iloc[-3]

        if n >= 4:
            features[f'{series_name}_lag3'] = series.iloc[-4]

        # Medium-term lag (6 months)
        if n >= 7:
            features[f'{series_name}_lag6'] = series.iloc[-7]
            features[f'{series_name}_6m_change'] = latest - series.iloc[-7]

        # Long-term lags
        if n >= 13:
            features[f'{series_name}_lag12'] = series.iloc[-13]
            features[f'{series_name}_yoy_change'] = latest - series.iloc[-13]

        if n >= 19:
            features[f'{series_name}_lag18'] = series.iloc[-19]
            features[f'{series_name}_18m_change'] = latest - series.iloc[-19]

        if n >= 25:
            features[f'{series_name}_lag24'] = series.iloc[-25]
            features[f'{series_name}_2yr_change'] = latest - series.iloc[-25]

        # Rolling means - use pre-computed slices
        if n >= 3:
            features[f'{series_name}_rolling_mean_3'] = series.iloc[-3:].mean()

        if n >= 6:
            features[f'{series_name}_rolling_mean_6'] = series.iloc[-6:].mean()

        if n >= 12:
            features[f'{series_name}_rolling_mean_12'] = series.iloc[-12:].mean()

        if n >= 24:
            features[f'{series_name}_rolling_mean_24'] = series.iloc[-24:].mean()

        # Volatility features
        if n >= 6:
            features[f'{series_name}_volatility_6m'] = series.iloc[-6:].std()

        if n >= 12:
            features[f'{series_name}_volatility_12m'] = series.iloc[-12:].std()

        # Trend features (12-month linear trend)
        if n >= 12:
            recent_values = series.iloc[-12:].values
            if not np.isnan(recent_values).any():
                try:
                    x = np.arange(12)
                    slope = np.polyfit(x, recent_values, 1)[0]
                    features[f'{series_name}_trend_12m'] = slope
                except (np.linalg.LinAlgError, ValueError):
                    pass

    if not features:
        return pd.DataFrame()

    return pd.DataFrame([features])


def pivot_snapshot_to_wide_batch(
    snapshot_df: pd.DataFrame,
    target_months: List[pd.Timestamp]
) -> pd.DataFrame:
    """
    Batch version of pivot_snapshot_to_wide for multiple target months.

    HIGHLY OPTIMIZED for building training datasets. Instead of loading
    and pivoting the same snapshot multiple times, this processes all
    target months in a single pass.

    Args:
        snapshot_df: Long-format snapshot DataFrame
        target_months: List of target months to generate features for

    Returns:
        DataFrame with one row per target month, features as columns
    """
    if snapshot_df is None or snapshot_df.empty or not target_months:
        return pd.DataFrame()

    # Pre-process snapshot once
    df = snapshot_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Pivot to wide format once
    wide_df = df.pivot_table(
        index='date',
        columns='series_name',
        values='value',
        aggfunc='first'
    ).sort_index()

    if wide_df.empty:
        return pd.DataFrame()

    all_features = []

    for target_month in target_months:
        # Filter to available data for this target month
        available = wide_df[wide_df.index <= target_month]

        if available.empty:
            all_features.append({'target_month': target_month})
            continue

        features = {'target_month': target_month}

        for series_name in available.columns:
            series = available[series_name].dropna()
            n = len(series)

            if n == 0:
                continue

            latest = series.iloc[-1]
            features[f'{series_name}_latest'] = latest

            # Lags and changes (simplified for batch processing)
            if n >= 2:
                features[f'{series_name}_lag1'] = series.iloc[-2]
                features[f'{series_name}_mom_change'] = latest - series.iloc[-2]

            if n >= 7:
                features[f'{series_name}_lag6'] = series.iloc[-7]

            if n >= 13:
                features[f'{series_name}_lag12'] = series.iloc[-13]
                features[f'{series_name}_yoy_change'] = latest - series.iloc[-13]

            # Rolling means
            if n >= 3:
                features[f'{series_name}_rolling_mean_3'] = series.iloc[-3:].mean()

            if n >= 12:
                features[f'{series_name}_rolling_mean_12'] = series.iloc[-12:].mean()

        all_features.append(features)

    return pd.DataFrame(all_features)
