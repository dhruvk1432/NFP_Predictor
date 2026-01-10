"""
Data Loading Utilities for LightGBM NFP Model

Functions for loading snapshots, target data, and building training datasets.
Extracted from train_lightgbm_nfp.py for maintainability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
# DATA LOADING
# =============================================================================

def load_fred_snapshot(snapshot_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Load FRED employment snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)

    Returns:
        DataFrame with columns: date, value, series_name, series_code, release_date
    """
    path = get_fred_snapshot_path(snapshot_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_master_snapshot(snapshot_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Load master snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)

    Returns:
        DataFrame with columns: date, series_name, value, release_date, series_code, snapshot_date
    """
    path = get_master_snapshot_path(snapshot_date)
    if not path.exists():
        logger.warning(f"Master snapshot not found: {path}")
        return None
    return pd.read_parquet(path)


def load_target_data(target_type: str = 'nsa') -> pd.DataFrame:
    """
    Load and prepare target data (NSA or SA NFP levels) with derived features.

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted

    Returns:
        DataFrame with columns: ds, y (level), y_mom (month-on-month change),
        and additional momentum/divergence features
    """
    # Load appropriate target file
    if target_type.lower() == 'nsa':
        target_path = TARGET_PATH_NSA
    else:
        target_path = TARGET_PATH_SA

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    df = pd.read_parquet(target_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # Calculate MoM change (our prediction target)
    df['y_mom'] = df['y'].diff()

    # Add momentum features
    df['y_mom_lag1'] = df['y_mom'].shift(1)
    df['y_mom_lag2'] = df['y_mom'].shift(2)
    df['y_mom_lag3'] = df['y_mom'].shift(3)

    # Rolling statistics
    df['y_mom_rolling_3m'] = df['y_mom'].shift(1).rolling(3).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].shift(1).rolling(6).mean()
    df['y_mom_rolling_12m'] = df['y_mom'].shift(1).rolling(12).mean()

    # Volatility
    df['y_mom_volatility_6m'] = df['y_mom'].shift(1).rolling(6).std()

    # Trend (difference between 3m and 12m rolling)
    df['y_mom_trend'] = df['y_mom_rolling_3m'] - df['y_mom_rolling_12m']

    # YoY change
    df['y_yoy'] = df['y'] - df['y'].shift(12)

    return df


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

    Uses only data available at the snapshot date to predict month M.

    Args:
        snapshot_df: Long-format snapshot DataFrame
        target_month: Month we're predicting (format: YYYY-MM-01)

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    # Filter to data available before or at target_month
    df = snapshot_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Get latest value for each series
    latest_values = {}

    for series_name in df['series_name'].unique():
        series_data = df[df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        # Get data up to and including target_month
        available_data = series_data[series_data['date'] <= target_month]

        if not available_data.empty:
            latest_values[f"{series_name}_latest"] = available_data['value'].iloc[-1]

            # Add lag features if enough history
            if len(available_data) >= 2:
                latest_values[f"{series_name}_lag1"] = available_data['value'].iloc[-2]

            if len(available_data) >= 7:
                latest_values[f"{series_name}_lag6"] = available_data['value'].iloc[-7]

            if len(available_data) >= 13:
                latest_values[f"{series_name}_lag12"] = available_data['value'].iloc[-13]

            # MoM change
            if len(available_data) >= 2:
                mom_change = available_data['value'].iloc[-1] - available_data['value'].iloc[-2]
                latest_values[f"{series_name}_mom_change"] = mom_change

    # Convert to single-row DataFrame
    result = pd.DataFrame([latest_values])
    result['ds'] = target_month

    return result
