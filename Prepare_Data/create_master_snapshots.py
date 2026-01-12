import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path
import pickle
import warnings

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
    """Apply SymLog and Log1p transforms in-place."""
    if df.empty: return df
    df = df.copy()

    # 1. SymLog
    for pattern in SYMLOG_TRANSFORM_SERIES:
        mask = df['series_name'].str.contains(pattern, regex=False)
        if mask.any():
            df.loc[mask, 'value'] = apply_symlog(df.loc[mask, 'value'])

    # 2. Log1p (using imported apply_log1p from utils.transforms)
    for pattern in LOG1P_TRANSFORM_SERIES:
        mask = df['series_name'].str.contains(pattern, regex=False)
        if mask.any():
            vals = df.loc[mask, 'value']
            # Safety: Fallback to SymLog if negatives found
            if (vals < 0).any():
                logger.warning(f"Negative values in {pattern} (Log1p target). Using SymLog fallback.")
                df.loc[mask, 'value'] = apply_symlog(vals)
            else:
                df.loc[mask, 'value'] = apply_log1p(vals)
                
    return df

def apply_robust_scaling_vintage(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fit RobustScaler on historical data only (before current month).
    Apply to all data to avoid look-ahead bias.

    Args:
        df: DataFrame with 'date', 'series_name', 'value' columns
        snapshot_date: Current snapshot date (e.g., 2020-01-31)

    Returns:
        DataFrame with scaled values (fitted on history, applied to all)
    """
    if df.empty or not SKLEARN_AVAILABLE: return df

    df = df.copy()

    # Cutoff: Exclude current month from fitting
    # If snapshot is 2020-01-31, only fit on data through 2019-12-31
    cutoff_date = snapshot_date - pd.DateOffset(months=1)
    cutoff_date = cutoff_date + pd.offsets.MonthEnd(0)

    # Scale by series_name
    scaled_groups = []
    for _, group in df.groupby('series_name'):
        group = group.sort_values('date')

        if len(group) < 2:
            scaled_groups.append(group)
            continue

        # Fit scaler only on historical data (before current month)
        hist_mask = group['date'] <= cutoff_date
        hist_data = group.loc[hist_mask, 'value']

        if len(hist_data) < 2:
            # Not enough history - skip scaling for this series
            scaled_groups.append(group)
            continue

        # Fit on history only
        scaler = RobustScaler()
        scaler.fit(hist_data.values.reshape(-1, 1))

        # Transform all data (including current month)
        group['value'] = scaler.transform(group['value'].values.reshape(-1, 1)).flatten()
        scaled_groups.append(group)

    return pd.concat(scaled_groups, ignore_index=True)


# =============================================================================
# MoM CHANGE FUNCTIONS
# =============================================================================

def add_mom_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Month-over-Month difference for Prosper series (which are already in %).
    Creates new series with '_MoM_Diff' suffix.

    Args:
        df: DataFrame with 'date', 'series_name', 'value' columns

    Returns:
        DataFrame with original series + new MoM difference series
    """
    if df.empty:
        return df

    df = df.copy()
    new_series = []

    # Identify prosper series by series_code pattern (contains '_ans')
    if 'series_code' in df.columns:
        prosper_mask = df['series_code'].str.contains('_ans', na=False)
    else:
        prosper_mask = pd.Series([False] * len(df))

    prosper_series_names = df.loc[prosper_mask, 'series_name'].unique()

    for s_name in prosper_series_names:
        subset = df[df['series_name'] == s_name].sort_values('date').copy()

        if len(subset) < 2:
            continue

        # Calculate MoM difference
        subset['value'] = subset['value'].diff()
        subset['series_name'] = s_name + '_MoM_Diff'
        if 'series_code' in subset.columns:
            subset['series_code'] = subset['series_code'] + '_MoM_Diff'

        subset = subset.dropna(subset=['value'])
        new_series.append(subset)

    if new_series:
        return pd.concat([df] + new_series, ignore_index=True)
    return df


def add_mom_pct_change(df: pd.DataFrame, exclude_patterns: list = None) -> pd.DataFrame:
    """
    Add Month-over-Month percentage change for all non-prosper series.
    Creates new series with '_MoM_Pct' suffix.

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
        ]

    df = df.copy()
    new_series = []

    # Identify prosper series to exclude (they get MoM_Diff instead)
    if 'series_code' in df.columns:
        prosper_mask = df['series_code'].str.contains('_ans', na=False)
        prosper_series_names = set(df.loc[prosper_mask, 'series_name'].unique())
    else:
        prosper_series_names = set()

    all_series = df['series_name'].unique()

    for s_name in all_series:
        # Skip prosper series
        if s_name in prosper_series_names:
            continue

        # Skip series that already have MoM/change patterns
        if any(pattern.lower() in s_name.lower() for pattern in exclude_patterns):
            continue

        subset = df[df['series_name'] == s_name].sort_values('date').copy()

        if len(subset) < 2:
            continue

        # Calculate MoM percentage change (* 100 for readability)
        subset['value'] = subset['value'].pct_change() * 100
        subset['series_name'] = s_name + '_MoM_Pct'
        if 'series_code' in subset.columns:
            subset['series_code'] = subset['series_code'].astype(str) + '_MoM_Pct'

        subset = subset.dropna(subset=['value'])

        # Skip if all values are inf/nan (e.g., division by zero)
        if subset['value'].replace([np.inf, -np.inf], np.nan).isna().all():
            continue

        # Replace inf with nan
        subset['value'] = subset['value'].replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=['value'])

        if not subset.empty:
            new_series.append(subset)

    if new_series:
        return pd.concat([df] + new_series, ignore_index=True)
    return df


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

def create_master_snapshots(apply_preprocessing: bool = True):
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    snapshot_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')

    logger.info(f"Generating Cleaned Master Snapshots from {start_dt.date()} to {end_dt.date()}")
    
    for i, snap_date in enumerate(snapshot_dates):
        
        # 1. LOAD DATA (The "Vintage")
        # Represents all data known as of snap_date
        fred_exog = load_snapshot(FRED_EXOG_DIR, snap_date)
        unifier_base = load_snapshot(UNIFIER_DIR, snap_date)
        adp = load_snapshot(ADP_SNAPSHOTS_DIR, snap_date)
        noaa_weighted = load_snapshot(NOAA_WEIGHTED_DIR, snap_date)
        prosper = load_snapshot(PROSPER_DIR, snap_date)

        current_vintage_df = pd.concat([fred_exog, unifier_base, adp, noaa_weighted, prosper], ignore_index=True)
        
        if current_vintage_df.empty: 
            continue
            
        current_vintage_df['date'] = pd.to_datetime(current_vintage_df['date'])
        
        # 2. PREPROCESSING PIPELINE
        # Only cleaning and scaling. No lag generation.
        if apply_preprocessing:
            # A. Structural Changes
            current_vintage_df = preprocess_noaa_indices(current_vintage_df)
            current_vintage_df = preprocess_pct_change(current_vintage_df)

            # B. Value Transforms (SymLog / Log1p) - excludes prosper (no transforms)
            current_vintage_df = preprocess_transforms(current_vintage_df)

            # C. Add MoM Changes (before scaling)
            # - Prosper series: MoM difference (already in %)
            # - Other series: MoM percentage change
            current_vintage_df = add_mom_difference(current_vintage_df)
            current_vintage_df = add_mom_pct_change(current_vintage_df)

            # D. Scaling (Fit on HISTORY only, exclude current month to avoid leakage)
            current_vintage_df = apply_robust_scaling_vintage(current_vintage_df, snap_date)

        # 3. SAVE
        # Save the clean, scaled, flat file. Feature engineering happens downstream.
        current_vintage_df['snapshot_date'] = snap_date
        
        decade_str = f"{snap_date.year // 10 * 10}s"
        year_str = str(snap_date.year)
        save_dir = MASTER_DIR / decade_str / year_str
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{snap_date.strftime('%Y-%m')}.parquet"

        current_vintage_df.to_parquet(save_path, index=False)

        if i % 12 == 0:
            logger.info(f"Generated Snapshot for {snap_date.date()}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-preprocessing', action='store_true')
    args = parser.parse_args()

    create_master_snapshots(apply_preprocessing=not args.no_preprocessing)
    verify_master_snapshot()