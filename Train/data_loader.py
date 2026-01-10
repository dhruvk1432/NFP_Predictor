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

    # Rolling averages (for momentum/divergence features)
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

    logger.info(f"Loaded {target_type.upper()} target data: {len(df)} observations from {df['ds'].min()} to {df['ds'].max()}")
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
    Includes short-term (1-3), medium-term (6), and long-term (12, 18, 24) lags
    for capturing macroeconomic cycles and structural trends.

    Args:
        snapshot_df: Long-format snapshot DataFrame
        target_month: Month we're predicting (format: YYYY-MM-01)

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    features = {}

    # Get unique series in the snapshot
    series_names = snapshot_df['series_name'].unique()

    for series_name in series_names:
        series_data = snapshot_df[snapshot_df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        if series_data.empty:
            continue

        # Get the latest available value (most recent date <= target_month)
        # This ensures no look-ahead bias
        available_data = series_data[series_data['date'] <= target_month]

        if not available_data.empty:
            # Latest value
            latest_row = available_data.iloc[-1]
            features[f'{series_name}_latest'] = latest_row['value']

            # --- Short-term lags (1, 2, 3 months) ---
            # Capture recent momentum and business cycle dynamics
            if len(available_data) >= 2:
                features[f'{series_name}_lag1'] = available_data.iloc[-2]['value']
                features[f'{series_name}_mom_change'] = latest_row['value'] - available_data.iloc[-2]['value']

            if len(available_data) >= 3:
                features[f'{series_name}_lag2'] = available_data.iloc[-3]['value']

            if len(available_data) >= 4:
                features[f'{series_name}_lag3'] = available_data.iloc[-4]['value']

            # --- Medium-term lag (6 months) ---
            # Semi-annual patterns
            if len(available_data) >= 7:
                features[f'{series_name}_lag6'] = available_data.iloc[-7]['value']
                features[f'{series_name}_6m_change'] = latest_row['value'] - available_data.iloc[-7]['value']

            # --- Long-term lags (12, 18, 24 months) ---
            # Structural trends and secular patterns in macroeconomic data
            if len(available_data) >= 13:
                features[f'{series_name}_lag12'] = available_data.iloc[-13]['value']
                features[f'{series_name}_yoy_change'] = latest_row['value'] - available_data.iloc[-13]['value']

            if len(available_data) >= 19:
                features[f'{series_name}_lag18'] = available_data.iloc[-19]['value']
                features[f'{series_name}_18m_change'] = latest_row['value'] - available_data.iloc[-19]['value']

            if len(available_data) >= 25:
                features[f'{series_name}_lag24'] = available_data.iloc[-25]['value']
                features[f'{series_name}_2yr_change'] = latest_row['value'] - available_data.iloc[-25]['value']

            # --- Rolling means (smoothed signals) ---
            if len(available_data) >= 3:
                features[f'{series_name}_rolling_mean_3'] = available_data['value'].iloc[-3:].mean()

            if len(available_data) >= 6:
                features[f'{series_name}_rolling_mean_6'] = available_data['value'].iloc[-6:].mean()

            if len(available_data) >= 12:
                features[f'{series_name}_rolling_mean_12'] = available_data['value'].iloc[-12:].mean()

            if len(available_data) >= 24:
                features[f'{series_name}_rolling_mean_24'] = available_data['value'].iloc[-24:].mean()

            # --- Volatility features ---
            if len(available_data) >= 6:
                recent_values = available_data['value'].iloc[-6:]
                features[f'{series_name}_volatility_6m'] = recent_values.std()

            if len(available_data) >= 12:
                recent_values = available_data['value'].iloc[-12:]
                features[f'{series_name}_volatility_12m'] = recent_values.std()

            # --- Trend features ---
            if len(available_data) >= 12:
                # Linear trend slope over last 12 months
                recent_values = available_data['value'].iloc[-12:].values
                x = np.arange(12)
                if not np.isnan(recent_values).any():
                    try:
                        slope = np.polyfit(x, recent_values, 1)[0]
                        features[f'{series_name}_trend_12m'] = slope
                    except (np.linalg.LinAlgError, ValueError):
                        pass

    if not features:
        return pd.DataFrame()

    return pd.DataFrame([features])
