import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, START_DATE, END_DATE
from utils.transforms import apply_symlog, apply_log1p

logger = setup_logger(__file__, TEMP_DIR)

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available. RobustScaler will be skipped.")
    SKLEARN_AVAILABLE = False

# Performance optimization: Pre-compile regex patterns and create sets for O(1) lookup
_SYMLOG_PATTERNS_SET = None
_LOG1P_PATTERNS_SET = None

# --- Paths ---
FRED_EXOG_DIR = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"
UNIFIER_DIR = DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades"
ADP_SNAPSHOTS_DIR = DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades"
NOAA_WEIGHTED_DIR = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / "decades"
PROSPER_DIR = DATA_PATH / "Exogenous_data" / "prosper" / "decades"
MASTER_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"

# =============================================================================
# DATA CLEANING CONFIGURATION
# =============================================================================

# 1. Stationarity Fix: Convert Levels to % Change (Drop Original)
PCT_CHANGE_SERIES = [
    "CCSA_monthly_avg"
]

# 2. Crash Handling: SymLog for Negative Values + Outliers (Replace Original)
# Formula: sign(x) * log1p(abs(x))
# NOTE: Panic indicators (CCSA_MoM_Pct, SP500_*) removed to preserve raw magnitude for linear extrapolation
SYMLOG_TRANSFORM_SERIES = [
    "ADP_actual",
    "Credit_Spreads_monthly_chg",
    "Yield_Curve_monthly_chg",
    # "CCSA_MoM_Pct",  # REMOVED: Need linear extrapolation for extreme spikes
    # "SP500_monthly_return",  # REMOVED: Preserve crash magnitude
    # "SP500_30d_return",  # REMOVED: Preserve crash magnitude
    # "SP500_max_drawdown",  # REMOVED: Preserve crash magnitude
    # "SP500_worst_day",  # REMOVED: Preserve crash magnitude
    "Oil_Prices_30d_crash",
    "Oil_Prices_zscore_min",
    "Credit_Spreads_zscore_max",
    "Yield_Curve_zscore_max",
    "Credit_Spreads_acceleration",
    "Yield_Curve_acceleration"
]

# 3. Skew Handling: Log1p for Positive Values + High Skew (Replace Original)
# Formula: log(1 + x)
LOG1P_TRANSFORM_SERIES = [
    "JOLTS_Layoffs",
    "Challenger_Job_Cuts",
    "Oil_Prices_volatility",
    "Credit_Spreads_vol_of_changes",
    "Yield_Curve_vol_of_changes",
    "Credit_Spreads_avg",
    "VIX_mean",
    "VIX_max",
    "VIX_volatility",
    "VIX_30d_spike",
    "SP500_volatility",
    "Credit_Spreads_accel_volatility",
    "Yield_Curve_accel_volatility"
]

# 4. Dimension Reduction: NOAA Aggregation
NOAA_HUMAN_IMPACT_COLS = [
    "deaths_direct_weighted_log",
    "deaths_indirect_weighted_log",
    "injuries_direct_weighted_log",
    "injuries_indirect_weighted_log"
]

NOAA_ECONOMIC_DAMAGE_COLS = [
    "total_property_damage_real_weighted_log",
    "total_crop_damage_real_weighted_log"
]

# Columns to drop after aggregation
NOAA_COLS_TO_DROP = NOAA_HUMAN_IMPACT_COLS + NOAA_ECONOMIC_DAMAGE_COLS + ["storm_count_weighted_log"]


# =============================================================================
# PERFORMANCE OPTIMIZATION HELPERS
# =============================================================================

def _get_pattern_match_mask(series_names: pd.Series, patterns: List[str]) -> pd.Series:
    """
    Efficiently check if series names contain any of the given patterns.
    Uses vectorized string operations instead of looping.
    """
    if not patterns:
        return pd.Series([False] * len(series_names), index=series_names.index)

    # Build a single regex pattern for all matches
    escaped_patterns = [re.escape(p) for p in patterns]
    combined_pattern = '|'.join(escaped_patterns)
    return series_names.str.contains(combined_pattern, regex=True, na=False)


def _load_snapshot_cached(path: Path) -> pd.DataFrame:
    """Load a parquet file if it exists, with minimal overhead."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_all_snapshots_parallel(snap_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load all source snapshots for a given date in parallel using ThreadPoolExecutor.
    I/O bound operations benefit from threading.
    """
    dirs = [
        (FRED_EXOG_DIR, "fred_exog"),
        (UNIFIER_DIR, "unifier"),
        (ADP_SNAPSHOTS_DIR, "adp"),
        (NOAA_WEIGHTED_DIR, "noaa"),
        (PROSPER_DIR, "prosper"),
    ]

    def get_path(base_dir: Path) -> Path:
        decade = f"{snap_date.year // 10 * 10}s"
        year = str(snap_date.year)
        filename = f"{snap_date.strftime('%Y-%m')}.parquet"
        return base_dir / decade / year / filename

    results = []

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {
            executor.submit(_load_snapshot_cached, get_path(base_dir)): name
            for base_dir, name in dirs
        }

        for future in as_completed(future_to_name):
            df = future.result()
            if not df.empty:
                results.append(df)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

# NOTE: apply_symlog and apply_log1p are imported from utils.transforms


def preprocess_noaa_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Create NOAA composite indices and drop original granular columns."""
    if df.empty: return df

    # Check if we have NOAA data in this snapshot
    human_cols = [c for c in NOAA_HUMAN_IMPACT_COLS if c in df['series_name'].unique()]
    econ_cols = [c for c in NOAA_ECONOMIC_DAMAGE_COLS if c in df['series_name'].unique()]
    
    if not human_cols and not econ_cols:
        return df

    result_parts = []
    
    # 1. Keep non-NOAA data
    drop_mask = df['series_name'].isin(NOAA_COLS_TO_DROP)
    rest_df = df[~drop_mask].copy()
    result_parts.append(rest_df)

    # 2. Create Indices (Pivot -> Sum -> Melt logic manual equivalent)
    noaa_df = df[df['series_name'].isin(human_cols + econ_cols)].copy()
    
    if not noaa_df.empty:
        # Pivot to wide format to align dates
        wide = noaa_df.pivot_table(index='date', columns='series_name', values='value', aggfunc='first')
        
        indices = []
        
        # Human Impact Index
        available_human = [c for c in human_cols if c in wide.columns]
        if available_human:
            human_impact = wide[available_human].sum(axis=1).reset_index(name='value')
            human_impact['series_name'] = 'NOAA_Human_Impact_Index'
            indices.append(human_impact)

        # Economic Damage Index
        available_econ = [c for c in econ_cols if c in wide.columns]
        if available_econ:
            econ_damage = wide[available_econ].sum(axis=1).reset_index(name='value')
            econ_damage['series_name'] = 'NOAA_Economic_Damage_Index'
            indices.append(econ_damage)

        if indices:
            indices_df = pd.concat(indices, ignore_index=True)
            if 'snapshot_date' in df.columns:
                indices_df['snapshot_date'] = df['snapshot_date'].iloc[0]
            result_parts.append(indices_df)

    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    return df

def preprocess_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specified series to Percentage Change and drop original Levels."""
    if df.empty: return df
    
    df = df.copy()
    processed_dfs = []
    dropped_series = set()

    for pattern in PCT_CHANGE_SERIES:
        matches = [s for s in df['series_name'].unique() if pattern in s]
        
        for s_name in matches:
            subset = df[df['series_name'] == s_name].sort_values('date')
            
            # Calculate % Change * 100 for readability
            subset['value'] = subset['value'].pct_change() * 100
            
            # Rename and Clean
            new_name = s_name.replace('_monthly_avg', '') + '_MoM_Pct'
            subset['series_name'] = new_name
            subset = subset.dropna(subset=['value'])
            
            processed_dfs.append(subset)
            dropped_series.add(s_name)

    # Keep non-converted series
    remaining = df[~df['series_name'].isin(dropped_series)]
    processed_dfs.append(remaining)
    
    return pd.concat(processed_dfs, ignore_index=True)

def preprocess_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SymLog and Log1p transforms in-place.

    OPTIMIZED: Uses single vectorized regex match instead of looping through patterns.
    """
    if df.empty:
        return df

    df = df.copy()
    series_names = df['series_name']

    # 1. SymLog - single vectorized mask for all patterns
    symlog_mask = _get_pattern_match_mask(series_names, SYMLOG_TRANSFORM_SERIES)
    if symlog_mask.any():
        df.loc[symlog_mask, 'value'] = apply_symlog(df.loc[symlog_mask, 'value'])

    # 2. Log1p - single vectorized mask for all patterns
    log1p_mask = _get_pattern_match_mask(series_names, LOG1P_TRANSFORM_SERIES)
    if log1p_mask.any():
        vals = df.loc[log1p_mask, 'value']
        # Check for negatives in one operation
        has_negatives = (vals < 0).any()
        if has_negatives:
            # Find which patterns have negatives for logging
            negative_series = df.loc[log1p_mask & (df['value'] < 0), 'series_name'].unique()
            logger.warning(f"Negative values in Log1p targets: {negative_series[:3]}... Using SymLog fallback.")
            df.loc[log1p_mask, 'value'] = apply_symlog(vals)
        else:
            df.loc[log1p_mask, 'value'] = apply_log1p(vals)

    return df

def apply_robust_scaling_vintage(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fit RobustScaler on historical data only (before current month).
    Apply to all data to avoid look-ahead bias.

    OPTIMIZED: Uses vectorized groupby operations with transform() instead of
    iterating and concatenating. Computes median/IQR directly for better performance.

    Args:
        df: DataFrame with 'date', 'series_name', 'value' columns
        snapshot_date: Current snapshot date (e.g., 2020-01-31)

    Returns:
        DataFrame with scaled values (fitted on history, applied to all)
    """
    if df.empty or not SKLEARN_AVAILABLE:
        return df

    df = df.copy()

    # Cutoff: Exclude current month from fitting
    cutoff_date = snapshot_date - pd.DateOffset(months=1)
    cutoff_date = cutoff_date + pd.offsets.MonthEnd(0)

    # Create history mask once
    hist_mask = df['date'] <= cutoff_date

    # Calculate median and IQR for each series using only historical data
    # This is equivalent to RobustScaler but vectorized
    def calc_robust_params(group):
        """Calculate median and IQR from historical portion of group."""
        group_hist = group[hist_mask.loc[group.index]]
        if len(group_hist) < 2:
            return pd.Series({'median': np.nan, 'iqr': np.nan})
        vals = group_hist['value']
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        return pd.Series({
            'median': vals.median(),
            'iqr': q3 - q1 if (q3 - q1) > 1e-10 else 1.0  # Avoid division by zero
        })

    # Get scaling parameters for each series
    params = df.groupby('series_name').apply(calc_robust_params, include_groups=False)

    # Apply scaling: (value - median) / IQR
    # Use map for efficient lookup
    df['_median'] = df['series_name'].map(params['median'])
    df['_iqr'] = df['series_name'].map(params['iqr'])

    # Only scale where we have valid parameters
    valid_mask = df['_median'].notna() & df['_iqr'].notna()
    df.loc[valid_mask, 'value'] = (
        (df.loc[valid_mask, 'value'] - df.loc[valid_mask, '_median'])
        / df.loc[valid_mask, '_iqr']
    )

    # Clean up temporary columns
    df.drop(columns=['_median', '_iqr'], inplace=True)

    return df


# =============================================================================
# DISCONTINUED SERIES FILTERING
# =============================================================================

# Cache for discontinued series - computed once and reused across all snapshots
_DISCONTINUED_SERIES_CACHE: Optional[set] = None


def _identify_discontinued_series() -> set:
    """
    Identify series that have no data in the last year before END_DATE.
    These are considered discontinued and should be excluded from all snapshots.

    Returns:
        Set of series names that are discontinued
    """
    global _DISCONTINUED_SERIES_CACHE

    if _DISCONTINUED_SERIES_CACHE is not None:
        return _DISCONTINUED_SERIES_CACHE

    end_dt = pd.to_datetime(END_DATE)
    cutoff_date = end_dt - pd.DateOffset(years=1)

    # Load the most recent snapshot to check which series have recent data
    recent_snapshot = _load_all_snapshots_parallel(end_dt)

    if recent_snapshot.empty:
        # Try previous month if END_DATE snapshot doesn't exist
        end_dt = end_dt - pd.DateOffset(months=1)
        end_dt = end_dt + pd.offsets.MonthEnd(0)
        recent_snapshot = _load_all_snapshots_parallel(end_dt)

    if recent_snapshot.empty:
        logger.warning("Could not load recent snapshot to identify discontinued series")
        _DISCONTINUED_SERIES_CACHE = set()
        return _DISCONTINUED_SERIES_CACHE

    recent_snapshot['date'] = pd.to_datetime(recent_snapshot['date'])

    # Find series with data in the last year
    active_mask = recent_snapshot['date'] >= cutoff_date
    active_series = set(recent_snapshot.loc[active_mask, 'series_name'].unique())
    all_series = set(recent_snapshot['series_name'].unique())

    # Discontinued = all series minus active series
    discontinued = all_series - active_series

    if discontinued:
        logger.info(f"Identified {len(discontinued)} discontinued series (no data after {cutoff_date.date()})")
        # Log a sample of discontinued series
        sample = list(discontinued)[:5]
        logger.info(f"Sample discontinued series: {sample}")

    _DISCONTINUED_SERIES_CACHE = discontinued
    return _DISCONTINUED_SERIES_CACHE


def filter_discontinued_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove discontinued series from the dataframe.
    A series is discontinued if it has no data in the last year before END_DATE.

    Args:
        df: DataFrame with 'series_name' column

    Returns:
        DataFrame with discontinued series removed
    """
    if df.empty:
        return df

    discontinued = _identify_discontinued_series()

    if not discontinued:
        return df

    # Filter out discontinued series
    mask = ~df['series_name'].isin(discontinued)
    filtered_df = df[mask].copy()

    return filtered_df


# =============================================================================
# MoM CHANGE FUNCTIONS
# =============================================================================

def add_mom_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Month-over-Month difference for Prosper series (which are already in %).
    Creates new series with '_MoM_Diff' suffix.

    OPTIMIZED: Uses vectorized groupby transform instead of looping through series.

    Args:
        df: DataFrame with 'date', 'series_name', 'value' columns

    Returns:
        DataFrame with original series + new MoM difference series
    """
    if df.empty:
        return df

    # Identify prosper series by series_code pattern (contains '_ans')
    if 'series_code' not in df.columns:
        return df

    prosper_mask = df['series_code'].str.contains('_ans', na=False)
    if not prosper_mask.any():
        return df

    # Extract prosper data
    prosper_df = df[prosper_mask].copy()

    # Sort once and calculate diff using groupby transform (vectorized)
    prosper_df = prosper_df.sort_values(['series_name', 'date'])
    prosper_df['value'] = prosper_df.groupby('series_name')['value'].diff()

    # Update names
    prosper_df['series_name'] = prosper_df['series_name'] + '_MoM_Diff'
    prosper_df['series_code'] = prosper_df['series_code'] + '_MoM_Diff'

    # Remove NaN from diff
    prosper_df = prosper_df.dropna(subset=['value'])

    if prosper_df.empty:
        return df

    return pd.concat([df, prosper_df], ignore_index=True)


def add_mom_pct_change(df: pd.DataFrame, exclude_patterns: list = None) -> pd.DataFrame:
    """
    Add Month-over-Month percentage change for all non-prosper series.
    Creates new series with '_MoM_Pct' suffix.

    OPTIMIZED: Uses vectorized operations with groupby transform and
    pre-computed exclusion masks instead of looping through series.

    Args:
        df: DataFrame with 'date', 'series_name', 'value' columns
        exclude_patterns: List of patterns to exclude (already have MoM or shouldn't be transformed)

    Returns:
        DataFrame with original series + new MoM percentage change series
    """
    if df.empty:
        return df

    if exclude_patterns is None:
        exclude_patterns = [
            '_MoM_Pct',      # Already a MoM percentage
            '_MoM_Diff',    # Already a MoM difference
            '_pct_change',  # Already percentage change
            '_return',      # Already a return
            '_chg',         # Already a change
            '_diff',        # Already a difference
            # Binary/count indicators - should remain as raw flags, not percentage changes
            # (pct_change on 0/1 data produces mostly NaN/inf which gets dropped)
            'Oil_days_negative',
            'Oil_Prices_went_negative',
            'SP500_crash_month',
            'SP500_circuit_breaker',
            'SP500_days_circuit_breaker',
            'SP500_bear_market',
            'VIX_panic_regime',
            'VIX_high_regime',
            'IURSA_weeks_high',
            'ICSA_weeks_high',
            'CCSA_weeks_high',
        ]

    # Build exclusion mask using vectorized string operations
    series_names = df['series_name']

    # Exclude prosper series (they get MoM_Diff instead)
    if 'series_code' in df.columns:
        prosper_mask = df['series_code'].str.contains('_ans', na=False)
    else:
        prosper_mask = pd.Series([False] * len(df), index=df.index)

    # Exclude series matching any pattern (case-insensitive)
    pattern_exclude_mask = _get_pattern_match_mask(
        series_names.str.lower(),
        [p.lower() for p in exclude_patterns]
    )

    # Series to process: not prosper and not matching exclude patterns
    include_mask = ~prosper_mask & ~pattern_exclude_mask

    if not include_mask.any():
        return df

    # Extract data to process
    pct_df = df[include_mask].copy()

    # Sort and calculate pct_change in one vectorized operation
    pct_df = pct_df.sort_values(['series_name', 'date'])
    pct_df['value'] = pct_df.groupby('series_name')['value'].pct_change() * 100

    # Update names
    pct_df['series_name'] = pct_df['series_name'] + '_MoM_Pct'
    if 'series_code' in pct_df.columns:
        pct_df['series_code'] = pct_df['series_code'].astype(str) + '_MoM_Pct'

    # Clean up: remove NaN and inf values
    pct_df['value'] = pct_df['value'].replace([np.inf, -np.inf], np.nan)
    pct_df = pct_df.dropna(subset=['value'])

    if pct_df.empty:
        return df

    return pd.concat([df, pct_df], ignore_index=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def get_snapshot_path(base_dir, date_ts):
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    filename = f"{date_ts.strftime('%Y-%m')}.parquet"
    return base_dir / decade / year / filename


def load_snapshot(base_dir, date_ts):
    path = get_snapshot_path(base_dir, date_ts)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _process_single_snapshot(
    snap_date: pd.Timestamp,
    apply_preprocessing: bool
) -> Tuple[bool, str]:
    """
    Process a single snapshot date. Returns (success, message).
    Designed for use in parallel processing.
    """
    try:
        # 1. LOAD DATA (The "Vintage") - using parallel I/O
        current_vintage_df = _load_all_snapshots_parallel(snap_date)

        if current_vintage_df.empty:
            return (True, f"Skipped {snap_date.date()} (no data)")

        current_vintage_df['date'] = pd.to_datetime(current_vintage_df['date'])

        # 2. FILTER DISCONTINUED SERIES (always applied, not just during preprocessing)
        # Remove series that have no data in the last year before END_DATE
        current_vintage_df = filter_discontinued_series(current_vintage_df)

        if current_vintage_df.empty:
            return (True, f"Skipped {snap_date.date()} (all series discontinued)")

        # 3. PREPROCESSING PIPELINE
        if apply_preprocessing:
            # A. Structural Changes
            current_vintage_df = preprocess_noaa_indices(current_vintage_df)
            current_vintage_df = preprocess_pct_change(current_vintage_df)

            # B. Value Transforms (SymLog / Log1p)
            current_vintage_df = preprocess_transforms(current_vintage_df)

            # C. Add MoM Changes (before scaling)
            current_vintage_df = add_mom_difference(current_vintage_df)
            current_vintage_df = add_mom_pct_change(current_vintage_df)

            # D. Scaling (Fit on HISTORY only)
            current_vintage_df = apply_robust_scaling_vintage(current_vintage_df, snap_date)

        # 4. SAVE
        current_vintage_df['snapshot_date'] = snap_date

        decade_str = f"{snap_date.year // 10 * 10}s"
        year_str = str(snap_date.year)
        save_dir = MASTER_DIR / decade_str / year_str
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{snap_date.strftime('%Y-%m')}.parquet"

        current_vintage_df.to_parquet(save_path, index=False)

        return (True, f"Generated {snap_date.date()}")

    except Exception as e:
        return (False, f"Error processing {snap_date.date()}: {str(e)}")


def create_master_snapshots(
    apply_preprocessing: bool = True,
    n_workers: int = 1,
    skip_existing: bool = False
):
    """
    Generate cleaned master snapshots for all dates in range.

    Args:
        apply_preprocessing: Whether to apply cleaning/scaling transforms
        n_workers: Number of parallel workers (1 = sequential, >1 = parallel)
                   Note: Due to GIL, multiprocessing is used for CPU-bound work.
                   For most cases, n_workers=1 is fastest due to I/O bottleneck.
        skip_existing: If True, skip snapshots that already exist
    """
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    snapshot_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')

    logger.info(f"Generating Cleaned Master Snapshots from {start_dt.date()} to {end_dt.date()}")
    logger.info(f"Total snapshots to process: {len(snapshot_dates)}")

    # Filter out existing if requested
    if skip_existing:
        dates_to_process = []
        for snap_date in snapshot_dates:
            save_path = get_snapshot_path(MASTER_DIR, snap_date)
            if not save_path.exists():
                dates_to_process.append(snap_date)
        logger.info(f"Skipping {len(snapshot_dates) - len(dates_to_process)} existing snapshots")
        snapshot_dates = dates_to_process

    if not len(snapshot_dates):
        logger.info("No snapshots to process.")
        return

    # Sequential processing (usually fastest due to I/O being the bottleneck)
    if n_workers == 1:
        for i, snap_date in enumerate(snapshot_dates):
            success, msg = _process_single_snapshot(snap_date, apply_preprocessing)

            if not success:
                logger.error(msg)
            elif i % 12 == 0:
                logger.info(msg)

    # Parallel processing using ProcessPoolExecutor for CPU-bound work
    else:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Using {n_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_snapshot, snap_date, apply_preprocessing): snap_date
                for snap_date in snapshot_dates
            }

            completed = 0
            for future in as_completed(futures):
                success, msg = future.result()
                completed += 1

                if not success:
                    logger.error(msg)
                elif completed % 12 == 0:
                    logger.info(f"Progress: {completed}/{len(snapshot_dates)} - {msg}")

    logger.info("Master Snapshots generation complete.")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_master_snapshot():
    logger.info("Verifying Master Snapshots...")
    
    # Check the LAST generated snapshot
    test_date = pd.to_datetime(END_DATE)
    decade_str = f"{test_date.year // 10 * 10}s"
    year_str = str(test_date.year)
    master_path = MASTER_DIR / decade_str / year_str / f"{test_date.strftime('%Y-%m')}.parquet"

    if not master_path.exists():
        # Try previous month
        test_date = test_date - pd.DateOffset(months=1)
        decade_str = f"{test_date.year // 10 * 10}s"
        year_str = str(test_date.year)
        master_path = MASTER_DIR / decade_str / year_str / f"{test_date.strftime('%Y-%m')}.parquet"
        
    if not master_path.exists():
        logger.error(f"Verification failed: Could not find file at {master_path}")
        return

    df = pd.read_parquet(master_path)
    series = df['series_name'].unique()

    logger.info(f"--- Verification Report for {test_date.date()} ---")
    logger.info(f"Total unique series: {len(series)}")

    # 1. Check Lags (Should NOT exist)
    lags = [s for s in series if '_lag' in s]
    if not lags:
        logger.info("✓ No Lags found (Correct).")
    else:
        logger.error(f"✗ Lags found! Logic error: {lags[:3]}...")

    # 2. Check Rolling (Should NOT exist)
    rolling = [s for s in series if 'rolling_mean' in s]
    if not rolling:
        logger.info("✓ No Rolling Means found (Correct).")
    else:
        logger.error(f"✗ Rolling means found! Logic error.")

    # 3. Check NOAA Indices
    if any('NOAA_Human_Impact_Index' in s for s in series):
        logger.info("✓ NOAA Indices present.")
    else:
        logger.warning("? No NOAA data found.")

    # 4. Check Scaling (ADP)
    adp = [s for s in series if 'ADP_actual' in s]
    if adp:
        vals = df[df['series_name'] == adp[0]]['value']
        logger.info(f"ADP (Scaled): Min={vals.min():.2f}, Max={vals.max():.2f}")
        if vals.max() < 100:
             logger.info("✓ ADP appears scaled.")

    # 5. Check Prosper data
    if 'series_code' in df.columns:
        prosper_codes = df[df['series_code'].str.contains('_ans', na=False)]['series_name'].unique()
        if len(prosper_codes) > 0:
            logger.info(f"✓ Prosper data present: {len(prosper_codes)} series.")
        else:
            logger.warning("? No Prosper data found.")
    else:
        logger.warning("? No series_code column - cannot check Prosper data.")

    # 6. Check MoM series
    mom_pct = [s for s in series if '_MoM_Pct' in s]
    mom_diff = [s for s in series if '_MoM_Diff' in s]
    logger.info(f"✓ MoM Percentage series: {len(mom_pct)}")
    logger.info(f"✓ MoM Difference series: {len(mom_diff)}")

    logger.info("Verification Complete.")

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate master snapshots with optimized preprocessing"
    )
    parser.add_argument('--no-preprocessing', action='store_true',
                        help="Skip all preprocessing transforms")
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip snapshots that already exist")
    parser.add_argument('--verify-only', action='store_true',
                        help="Only run verification, skip generation")
    args = parser.parse_args()

    if args.verify_only:
        verify_master_snapshot()
    else:
        start_time = time.time()

        create_master_snapshots(
            apply_preprocessing=not args.no_preprocessing,
            n_workers=args.workers,
            skip_existing=args.skip_existing
        )

        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        verify_master_snapshot()