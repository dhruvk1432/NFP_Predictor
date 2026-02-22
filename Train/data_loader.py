"""
Data Loading Utilities for LightGBM NFP Model

Functions for loading snapshots, target data, and building training datasets.
Extracted from train_lightgbm_nfp.py for maintainability.

OPTIMIZATIONS:
- LRU cache for snapshot loading (avoids redundant I/O)
- Vectorized pivot_snapshot_to_wide using pandas native operations
- Pre-computed lag indices for batch feature generation

MULTI-TARGET SUPPORT:
- Supports 4 target configurations: (nsa/sa) x (first/last release)
- Cache keys include both target_type and release_type
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger
from Train.config import (
    MASTER_SNAPSHOTS_DIR,
    FRED_SNAPSHOTS_DIR,
    FRED_PREPARED_DIR,
    USE_PREPARED_FRED_DATA,
    get_target_path,
    get_model_id,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    VALID_TARGET_SOURCES,
    REVISED_TARGET_SERIES,
)

logger = setup_logger(__file__, TEMP_DIR)

# Module-level cache for loaded data
_snapshot_cache: Dict[str, pd.DataFrame] = {}
_target_cache: Dict[str, pd.DataFrame] = {}


# =============================================================================
# FEATURE NAME SANITIZATION (for LightGBM compatibility)
# =============================================================================

def sanitize_feature_name(name: str) -> str:
    """
    Sanitize feature name for LightGBM compatibility.

    LightGBM doesn't support special JSON characters in feature names.
    This function replaces problematic characters with safe alternatives.

    Args:
        name: Original feature name

    Returns:
        Sanitized feature name
    """
    # Replace pipes with underscores
    name = name.replace('|', '_')
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Replace hyphens with underscores (keep at start/end for negative numbers)
    name = re.sub(r'(?<!^)-(?!$)', '_', name)
    # Remove or replace brackets and braces
    name = name.replace('[', '_').replace(']', '_')
    name = name.replace('{', '_').replace('}', '_')
    # Remove quotes
    name = name.replace('"', '').replace("'", '')
    name = name.replace('\\', '_')
    # Replace commas (JSON special character)
    name = name.replace(',', '_')
    # Replace parentheses
    name = name.replace('(', '_').replace(')', '_')
    # Replace question marks
    name = name.replace('?', '_')
    # Replace plus signs
    name = name.replace('+', 'plus')
    # Replace other special characters that might cause issues
    name = name.replace('/', '_')
    name = name.replace(':', '_')
    name = name.replace(';', '_')
    name = name.replace('!', '_')
    name = name.replace('@', '_')
    name = name.replace('#', '_')
    name = name.replace('$', '_')
    name = name.replace('%', 'pct')
    name = name.replace('&', '_and_')
    name = name.replace('*', '_')
    name = name.replace('=', '_')
    name = name.replace('<', '_lt_')
    name = name.replace('>', '_gt_')
    # Replace periods (e.g., "U.S." -> "US")
    name = name.replace('.', '_')
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize all column names in a DataFrame for LightGBM compatibility.

    Args:
        df: DataFrame with potentially problematic column names

    Returns:
        DataFrame with sanitized column names
    """
    df = df.copy()
    df.columns = [sanitize_feature_name(str(c)) for c in df.columns]
    return df


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Constructs the exact active file path to a processed FRED employment snapshot 
    for the specified month. It correctly routes to decades-based subfolders.

    Depending on `USE_PREPARED_FRED_DATA` from config, this will either point to 
    the raw snapshot or the preprocessed transformed data.

    Args:
        snapshot_date (pd.Timestamp): The vintage/snapshot month-end date.

    Returns:
        Path: The fully resolved filesystem path.
    """
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')

    # Use prepared data if configured
    base_dir = FRED_PREPARED_DIR if USE_PREPARED_FRED_DATA else FRED_SNAPSHOTS_DIR
    return base_dir / decade / year / f"{month_str}.parquet"


def get_raw_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Constructs the file path to a completely raw FRED employment snapshot (base levels only), 
    bypassing the preprocessing check. This is used explicitly for calculating true 
    revised target levels across snapshots.

    Args:
        snapshot_date (pd.Timestamp): The vintage/snapshot month-end date.

    Returns:
        Path: The fully resolved filesystem path to the raw data.
    """
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')
    return FRED_SNAPSHOTS_DIR / decade / year / f"{month_str}.parquet"


def get_master_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Constructs the file path to the fully merged master exogenous snapshot. This file 
    combines all varied data sources (NOAA, Unifier, ADP, Prosper, etc.) available at 
    a specific point in time.

    Args:
        snapshot_date (pd.Timestamp): The vintage/snapshot month-end date.

    Returns:
        Path: The fully resolved filesystem path to the master snapshot.
    """
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
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    # Check for empty or corrupt file
    if path.stat().st_size == 0:
        logger.warning(f"Skipping empty FRED snapshot: {path}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to read FRED snapshot {path}: {e}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

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
        if use_cache:
            _snapshot_cache[cache_key] = None
        # logger.warning(f"Master snapshot not found: {path}") # Suppressed as expected for early history
        return None

    # Check for empty or corrupt file
    if path.stat().st_size == 0:
        logger.warning(f"Skipping empty Master snapshot: {path}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to read Master snapshot {path}: {e}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    if use_cache:
        _snapshot_cache[cache_key] = df

    return df.copy() if use_cache else df


def clear_snapshot_cache() -> None:
    """Clear the snapshot cache to free memory."""
    global _snapshot_cache
    _snapshot_cache.clear()
    logger.info("Snapshot cache cleared")


def load_target_data(
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load and prepare target data (NSA or SA NFP levels) with derived features.

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised release
        target_source: 'first_release' for original release, 'revised' for once-revised
            (from M+1 FRED snapshot). Revised targets use raw FRED snapshot levels.
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: ds, y (level), y_mom (month-on-month change),
        and additional momentum/divergence features
    """
    # Validate inputs
    target_type = target_type.lower()
    release_type = release_type.lower()

    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: {target_type}. Must be one of {VALID_TARGET_TYPES}")
    if release_type not in VALID_RELEASE_TYPES:
        raise ValueError(f"Invalid release_type: {release_type}. Must be one of {VALID_RELEASE_TYPES}")
    if target_source not in VALID_TARGET_SOURCES:
        raise ValueError(f"Invalid target_source: {target_source}. Must be one of {VALID_TARGET_SOURCES}")

    cache_key = f"target_{get_model_id(target_type, release_type, target_source)}"

    if use_cache and cache_key in _target_cache:
        return _target_cache[cache_key].copy()

    # Revised target: build from raw FRED snapshots instead of loading parquet
    if target_source == 'revised':
        logger.info(f"Building revised {target_type.upper()} target from raw FRED snapshots...")
        df = build_revised_target(target_type)
        model_id = get_model_id(target_type, release_type, target_source)
        if use_cache:
            _target_cache[cache_key] = df
            logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations (cached)")
        return df.copy() if use_cache else df

    # Load appropriate target file using dynamic path
    target_path = get_target_path(target_type, release_type)
    logger.info(f"Loading {target_type.upper()} {release_type} release target data")

    if not target_path.exists():
        if use_cache:
            _target_cache[cache_key] = None
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

    model_id = get_model_id(target_type, release_type)
    if use_cache:
        _target_cache[cache_key] = df
        logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations (cached)")
    else:
        logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations")

    return df.copy() if use_cache else df


def load_all_target_data(release_type: str = 'first',
                        target_source: str = 'first_release') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both NSA and SA target data with all derived features for a given release type.

    Args:
        release_type: 'first' for initial release, 'last' for final revised
        target_source: 'first_release' or 'revised'

    Returns:
        Tuple of (nsa_df, sa_df) with columns:
        - ds: date
        - y: level
        - y_mom: month-on-month change
        - Additional momentum/divergence features
    """
    nsa_df = load_target_data('nsa', release_type=release_type, target_source=target_source)
    sa_df = load_target_data('sa', release_type=release_type, target_source=target_source)
    return nsa_df, sa_df


def build_revised_target(target_type: str = 'nsa') -> pd.DataFrame:
    """
    Build revised target data from raw FRED snapshots.

    For each month M, loads the M+1 snapshot and computes:
        revised_mom[M] = level[M] - level[M-1]  (both from M+1 snapshot)

    This captures the "once-revised" MoM change, which includes BLS revisions
    to the M-1 level that occur between the M and M+1 NFP releases.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        DataFrame matching load_target_data() output shape:
        ds, y, y_mom, release_date, and derived momentum features.
    """
    series_name = REVISED_TARGET_SERIES[target_type]

    # Load first-release target to get the list of months and release_dates
    first_release_path = get_target_path(target_type, 'first')
    first_release_df = pd.read_parquet(first_release_path)
    first_release_df['ds'] = pd.to_datetime(first_release_df['ds'])
    first_release_df = first_release_df.sort_values('ds').reset_index(drop=True)

    records = []
    for _, row in first_release_df.iterrows():
        m = row['ds']
        release_date = row.get('release_date', pd.NaT)

        # Load M+1 raw snapshot (levels)
        snapshot_date = m + pd.DateOffset(months=1)
        snapshot_path = get_raw_fred_snapshot_path(snapshot_date)

        if not snapshot_path.exists():
            records.append({'ds': m, 'y': np.nan, 'y_mom': np.nan,
                            'release_date': release_date})
            continue

        snap = pd.read_parquet(snapshot_path)
        series_data = snap[snap['series_name'] == series_name]
        if series_data.empty:
            records.append({'ds': m, 'y': np.nan, 'y_mom': np.nan,
                            'release_date': release_date})
            continue

        levels = (series_data[['date', 'value']]
                  .drop_duplicates('date')
                  .set_index('date')['value']
                  .sort_index())

        m_minus1 = m - pd.DateOffset(months=1)
        level_m = levels.get(m)
        level_m1 = levels.get(m_minus1)

        if level_m is not None and level_m1 is not None:
            records.append({'ds': m, 'y': float(level_m),
                            'y_mom': float(level_m) - float(level_m1),
                            'release_date': release_date})
        elif level_m is not None:
            records.append({'ds': m, 'y': float(level_m), 'y_mom': np.nan,
                            'release_date': release_date})
        else:
            records.append({'ds': m, 'y': np.nan, 'y_mom': np.nan,
                            'release_date': release_date})

    df = pd.DataFrame(records)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # Derived features (same as load_target_data)
    df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
    df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
    df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()
    df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()
    df['divergence_3m'] = df['y'] - df['y_rolling_3m']
    df['divergence_6m'] = df['y'] - df['y_rolling_6m']
    df['acceleration'] = df['y_mom'].diff()
    df['y_yoy'] = df['y'].diff(12)
    df['y_mom_yoy'] = df['y_mom'].diff(12)

    # Drop first row (no MoM for first observation)
    df = df[df.index != 0].reset_index(drop=True)

    logger.info(f"Built revised {target_type.upper()} target: {len(df)} months, "
                f"{df['y_mom'].notna().sum()} with valid MoM")
    return df


def clear_target_cache() -> None:
    """Clear the target data cache to free memory."""
    global _target_cache
    _target_cache.clear()
    logger.info("Target cache cleared")


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


def batch_lagged_target_features(
    target_df: pd.DataFrame,
    prefix: str = 'nfp',
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Vectorized batch computation of lagged target features for ALL months at once.

    Replaces per-month get_lagged_target_features() calls with a single pass
    using pandas shift/rolling. Returns a lookup dict for O(1) access per month.

    Semantics match get_lagged_target_features exactly:
    - shift(1) = "most recent value before this month" (lag 1)
    - rolling(N, min_periods=N) on shifted data = rolling average of N prior months

    Args:
        target_df: DataFrame with ds, y, y_mom, y_yoy columns
        prefix: Feature name prefix ('nfp_nsa' or 'nfp_sa')

    Returns:
        Dict mapping target_month -> {feature_name: value} (NaN features omitted)
    """
    df = target_df.sort_values('ds').set_index('ds')
    mom = df['y_mom']
    level = df['y']
    mom_shifted = mom.shift(1)

    # Build all lag columns vectorized
    result = pd.DataFrame(index=df.index)
    result[f'{prefix}_mom_lag1'] = mom.shift(1)
    result[f'{prefix}_level_lag1'] = level.shift(1)
    result[f'{prefix}_mom_lag2'] = mom.shift(2)
    result[f'{prefix}_mom_lag3'] = mom.shift(3)
    result[f'{prefix}_mom_lag6'] = mom.shift(6)
    result[f'{prefix}_mom_lag12'] = mom.shift(12)

    if 'y_yoy' in df.columns:
        result[f'{prefix}_yoy'] = df['y_yoy'].shift(1)

    # Rolling averages on shifted data (so rolling_3m at month M = mean of M-1, M-2, M-3)
    result[f'{prefix}_rolling_3m'] = mom_shifted.rolling(3, min_periods=3).mean()
    result[f'{prefix}_rolling_6m'] = mom_shifted.rolling(6, min_periods=6).mean()
    result[f'{prefix}_rolling_12m'] = mom_shifted.rolling(12, min_periods=12).mean()
    result[f'{prefix}_volatility_6m'] = mom_shifted.rolling(6, min_periods=6).std()
    result[f'{prefix}_momentum'] = result[f'{prefix}_rolling_3m'] - result[f'{prefix}_rolling_12m']

    # Convert to dict-of-dicts, dropping NaN values per row
    lookup: Dict[pd.Timestamp, Dict[str, float]] = {}
    cols = result.columns.tolist()
    arr = result.values  # numpy array for fast row access
    for i, ts in enumerate(result.index):
        row_vals = arr[i]
        features = {}
        for j, col in enumerate(cols):
            v = row_vals[j]
            if not np.isnan(v):
                features[col] = float(v)
        lookup[ts] = features

    return lookup


def prefilter_snapshot(
    df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Pre-filter a loaded snapshot to only include series matching selected features.

    Call this ONCE right after loading a snapshot, before passing it to
    multiple pivot_snapshot_to_wide() calls. This avoids repeated filtering
    inside each pivot call, reducing DataFrame copies and groupby operations
    on the full snapshot.

    Args:
        df: Snapshot DataFrame (wide or long format)
        selected_features: List of sanitized feature names to keep

    Returns:
        Filtered DataFrame with only relevant series/columns
    """
    if df is None or df.empty or not selected_features:
        return df

    selected_set = set(selected_features)
    is_wide = 'series_name' not in df.columns

    if is_wide:
        # Wide format: filter columns (keep 'date' index/column + selected)
        cols_to_keep = []
        for col in df.columns:
            if col == 'date':
                cols_to_keep.append(col)
            elif sanitize_feature_name(str(col)) in selected_set:
                cols_to_keep.append(col)
        if not cols_to_keep:
            return pd.DataFrame()
        return df[cols_to_keep]
    else:
        # Long format: filter rows by series_name
        unique_raw = df['series_name'].unique()
        allowed_raw = [raw for raw in unique_raw if sanitize_feature_name(str(raw)) in selected_set]
        if not allowed_raw:
            return pd.DataFrame()
        return df[df['series_name'].isin(set(allowed_raw))]


def pivot_snapshot_to_wide(
    snapshot_df: pd.DataFrame,
    target_month: pd.Timestamp,
    cutoff_date: Optional[pd.Timestamp] = None,
    selected_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert snapshot to wide format, taking the latest value for target month.

    Supports both formats:
    - Wide-format (from compute_features_wide): DatetimeIndex + feature columns.
      Skips the expensive pivot and just takes the last valid row per column.
    - Long-format (legacy): columns [date, series_name, value, ...].
      Pivots first, then takes the last valid row per column.

    Args:
        snapshot_df: Snapshot DataFrame (wide or long format)
        target_month: Month we're predicting (format: YYYY-MM-01)
        cutoff_date: Strict cutoff for data availability (e.g., NFP release date)
        selected_features: List of sanitized feature names to keep. If provided,
            filters data early for performance.

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    # Detect format: wide-format has no 'series_name' column
    is_wide = 'series_name' not in snapshot_df.columns

    if is_wide:
        # Wide-format: index=date (or 'date' column), columns=feature names
        wide_df = snapshot_df.copy()
        
        # Ensure 'date' is available as column or index for filtering
        if 'date' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['date'])
            wide_df = wide_df.set_index('date')
        
        if not isinstance(wide_df.index, pd.DatetimeIndex):
             # Try to convert index if not datetime
             try:
                 wide_df.index = pd.to_datetime(wide_df.index)
             except:
                 pass
        
        wide_df = wide_df.sort_index()

        # Apply cutoff
        cutoff = pd.to_datetime(cutoff_date) if cutoff_date is not None else pd.to_datetime(target_month)
        wide_df = wide_df[wide_df.index < cutoff]

        if wide_df.empty:
            return pd.DataFrame()

        # EARLY FILTERING (Wide): Filter columns
        if selected_features is not None:
            # Match columns to selected features
            # Columns in wide format are series names (potentially raw)
            # We need to sanitize them to check against selected_features
            
            # Optimization: 
            # 1. Sanitize all columns (once) -> Map sanitized to raw
            # 2. Keep only raw columns where sanitized version is in selected_features
            
            selected_set = set(selected_features)
            cols_to_keep = []
            
            for col in wide_df.columns:
                sanitized = sanitize_feature_name(str(col))
                if sanitized in selected_set:
                    cols_to_keep.append(col)
            
            if not cols_to_keep:
                return pd.DataFrame()
                
            wide_df = wide_df[cols_to_keep]

        # Take last valid value per column (vectorized)
        if wide_df.empty:
             return pd.DataFrame()
             
        last_valid = wide_df.ffill().iloc[-1]
        last_valid = last_valid.dropna()

        if last_valid.empty:
            return pd.DataFrame()

        # Sanitize feature names for LightGBM compatibility
        features = {sanitize_feature_name(str(k)): v for k, v in last_valid.items()}
        return pd.DataFrame([features])

    # Valid "Long" format has 'series_name' and 'value'
    # Optimize: Filter -> GroupBy -> Last (Avoids creating sparse wide DF)
    df = snapshot_df.copy()
    
    # Ensure 'date' is a column
    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff = pd.to_datetime(cutoff_date) if cutoff_date is not None else pd.to_datetime(target_month)
        df = df[df['date'] < cutoff]
    else:
        # If no date column and no DatetimeIndex, we can't filter by time
        # This shouldn't happen for valid snapshots
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
        
    # EARLY FILTERING: Filter by selected features if provided
    if selected_features is not None:
        try:
            # Get unique raw series names present in this filtered slice
            unique_raw = df['series_name'].unique()
        except KeyError as e:
            logger.error(f"KeyError accessing 'series_name'. Columns: {df.columns.tolist()}")
            logger.error(f"Original is_wide detection: {is_wide}")
            # Fallback: maybe it's in the index now?
            if 'series_name' in df.index.names:
                df = df.reset_index()
                unique_raw = df['series_name'].unique()
            else:
                raise e
        
        # Build mapping: raw -> sanitized
        # Only keep those that are in selected_features
        allowed_raw = []
        selected_set = set(selected_features)
        
        for raw in unique_raw:
            sanitized = sanitize_feature_name(str(raw))
            if sanitized in selected_set:
                allowed_raw.append(raw)
        
        if not allowed_raw:
            return pd.DataFrame()
            
        # Filter DataFrame to only allowed raw series
        df = df[df['series_name'].isin(allowed_raw)]
        
        if df.empty:
            return pd.DataFrame()

    # Sort by date so "last" is truly the latest
    df = df.sort_values('date')
    
    # Take last value for each series
    latest_values = df.groupby('series_name')['value'].last()
    
    if latest_values.empty:
        return pd.DataFrame()
        
    # Convert to dict -> DataFrame (1 row)
    features = {sanitize_feature_name(str(k)): v for k, v in latest_values.to_dict().items()}
    
    # Final check: if selected_features provided, ensure we only return those
    # (The pre-filtering above handled the bulk, this is just cleanup)
    if selected_features is not None:
         # Dict comprehension above already sanitized keys.
         # Just filter the dict.
         features = {k: v for k, v in features.items() if k in selected_set}
         
    return pd.DataFrame([features])


def pivot_snapshot_to_wide_batch(
    snapshot_df: pd.DataFrame,
    target_months: List[pd.Timestamp],
    cutoff_dates: Optional[Dict[pd.Timestamp, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Batch version of pivot_snapshot_to_wide for multiple target months.

    Supports both wide-format (from compute_features_wide) and long-format
    (legacy) snapshots. Instead of loading and pivoting the same snapshot
    multiple times, this processes all target months in a single pass.

    Args:
        snapshot_df: Snapshot DataFrame (wide or long format)
        target_months: List of target months to generate features for
        cutoff_dates: Optional mapping of target_month -> cutoff_date (e.g., NFP release date)

    Returns:
        DataFrame with one row per target month, features as columns
    """
    if snapshot_df is None or snapshot_df.empty or not target_months:
        return pd.DataFrame()

    # Detect format and get wide_df
    is_wide = 'series_name' not in snapshot_df.columns

    if is_wide:
        wide_df = snapshot_df.copy()
        if 'date' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['date'])
            wide_df = wide_df.set_index('date')
        wide_df.index = pd.to_datetime(wide_df.index)
        wide_df = wide_df.sort_index()
    else:
        df = snapshot_df.copy()
        df['date'] = pd.to_datetime(df['date'])
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
        if cutoff_dates is not None and target_month in cutoff_dates:
            cutoff = pd.to_datetime(cutoff_dates[target_month])
        else:
            cutoff = pd.to_datetime(target_month)
        available = wide_df[wide_df.index < cutoff]

        if available.empty:
            all_features.append({'target_month': target_month})
            continue

        features = {'target_month': target_month}

        for raw_series_name in available.columns:
            series = available[raw_series_name].dropna()

            if len(series) == 0:
                continue

            series_name = sanitize_feature_name(str(raw_series_name))
            features[series_name] = series.iloc[-1]

        all_features.append(features)

    return pd.DataFrame(all_features)
