"""
Prepare FRED Employment Snapshots for LightGBM NFP Prediction

This module preprocesses the endogenous (employment) data from FRED snapshots
to address statistical issues (skewness, kurtosis, scaling) while keeping the
data in the same format as the raw snapshots.

IMPORTANT: This module does NOT create lag features or rolling statistics.
Those are created downstream in Train/train_lightgbm_nfp.py to avoid redundancy.

Key transformations (matching exogenous preprocessing approach):
1. Convert levels to MoM changes (what we're predicting)
2. Apply SymLog transform to handle extreme values (COVID crash)
3. Apply RobustScaler (fitted on history only) for normalization

The output maintains the same schema as the input:
    [date, value, series_name, series_code, release_date, snapshot_date]

This allows the existing train_lightgbm_nfp.py to work with either raw or
preprocessed data without modification.
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, START_DATE, END_DATE

logger = setup_logger(__file__, TEMP_DIR)

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available. RobustScaler will be skipped.")
    SKLEARN_AVAILABLE = False

# =============================================================================
# PATHS
# =============================================================================

FRED_SNAPSHOTS_DIR = DATA_PATH / "fred_data" / "decades"
PREPARED_FRED_DIR = DATA_PATH / "fred_data_prepared" / "decades"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Series that should have SymLog applied (high kurtosis MoM changes)
# These series experience extreme values (like COVID crash/recovery)
SYMLOG_TRANSFORM_SERIES = [
    # All employment series have potential for extreme MoM swings
    # Apply SymLog universally to MoM changes
]

# Apply SymLog to ALL MoM changes since employment data universally
# shows high kurtosis due to recessions/recoveries
# CHANGED: Set to False to preserve COVID-19 crash magnitude
APPLY_SYMLOG_TO_ALL = False

# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def apply_symlog(x):
    """
    Apply symmetric log transform: sign(x) * log1p(abs(x))

    This transform:
    - Handles negative values (unlike log)
    - Compresses extreme values (reduces kurtosis)
    - Preserves sign and relative magnitude
    - Is invertible: sign(y) * (exp(abs(y)) - 1)

    For NFP MoM changes, reduces:
    - Skewness: -6.5 -> -1.1
    - Kurtosis: 81 -> -0.7
    """
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_symlog(y):
    """Inverse of symlog transform for prediction recovery."""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


def calculate_mom_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert employment levels to Month-over-Month changes.

    The raw FRED snapshots contain employment LEVELS (e.g., 150M jobs).
    Our prediction target is the MoM CHANGE (e.g., +150K jobs).

    This transformation:
    1. Makes the series stationary (levels are non-stationary)
    2. Matches what we're actually predicting

    Args:
        df: DataFrame with columns ['date', 'value', 'series_name', ...]
            where 'value' contains employment levels

    Returns:
        DataFrame with MoM changes as 'value', original level stored in 'value_level'
    """
    if df.empty:
        return df

    df = df.copy()
    result_parts = []

    for series_name, group in df.groupby('series_name'):
        group = group.sort_values('date').copy()

        # Store original level for reference
        group['value_level'] = group['value']

        # Calculate MoM change (this is what we predict)
        group['value'] = group['value'].diff()

        # Keep track of raw change before any transforms
        group['value_raw'] = group['value']

        # Drop first row (NaN from diff)
        group = group.dropna(subset=['value'])

        result_parts.append(group)

    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    return df


def preprocess_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SymLog transform to MoM changes.

    This handles extreme values like:
    - COVID crash: -20M jobs in April 2020
    - Recovery months: +4M jobs

    Without this transform:
    - Skewness: -6.5 (highly negatively skewed)
    - Kurtosis: 81 (extreme fat tails)

    With SymLog:
    - Skewness: -1.1 (much closer to normal)
    - Kurtosis: -0.7 (normal-like tails)
    """
    if df.empty:
        return df

    df = df.copy()

    if APPLY_SYMLOG_TO_ALL:
        # Apply SymLog to all MoM changes
        df['value'] = apply_symlog(df['value'])
    else:
        # Only apply to specified series
        for pattern in SYMLOG_TRANSFORM_SERIES:
            mask = df['series_name'].str.contains(pattern, regex=False)
            if mask.any():
                df.loc[mask, 'value'] = apply_symlog(df.loc[mask, 'value'])

    return df


def apply_robust_scaling_vintage(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fit RobustScaler on historical data only (before current month).
    Apply to all data to avoid look-ahead bias.

    This is critical for proper backtesting - we can only use statistics
    that would have been available at prediction time.

    RobustScaler uses median and IQR, making it resistant to outliers.
    This is important for employment data which has extreme COVID values.

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
    # If snapshot is 2020-01-31, only fit on data through 2019-12-31
    cutoff_date = snapshot_date - pd.DateOffset(months=1)
    cutoff_date = cutoff_date + pd.offsets.MonthEnd(0)

    scaled_groups = []

    for series_name, group in df.groupby('series_name'):
        group = group.sort_values('date').copy()

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

        # Skip if constant values (avoid division by zero)
        if hist_data.std() < 1e-10:
            scaled_groups.append(group)
            continue

        # Fit on history only
        scaler = RobustScaler()
        scaler.fit(hist_data.values.reshape(-1, 1))

        # Transform all data (including current month)
        group['value'] = scaler.transform(group['value'].values.reshape(-1, 1)).flatten()
        scaled_groups.append(group)

    if scaled_groups:
        return pd.concat(scaled_groups, ignore_index=True)
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def get_snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    """Get path to snapshot file for a given date."""
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    filename = f"{date_ts.strftime('%Y-%m')}.parquet"
    return base_dir / decade / year / filename


def load_snapshot(base_dir: Path, date_ts: pd.Timestamp) -> pd.DataFrame:
    """Load a snapshot file if it exists."""
    path = get_snapshot_path(base_dir, date_ts)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def prepare_fred_snapshots(
    apply_mom_conversion: bool = True,
    apply_transforms: bool = True,
    apply_scaling: bool = True
):
    """
    Main pipeline to prepare FRED employment snapshots.

    Processing steps:
    1. Load raw snapshot (employment levels)
    2. Convert to MoM changes (stationary target)
    3. Apply SymLog transform (handle outliers)
    4. Apply RobustScaler (normalize, history-only fit)

    NOTE: This does NOT create lag features or rolling statistics.
    Those are created in train_lightgbm_nfp.py to avoid redundancy.

    Args:
        apply_mom_conversion: Convert levels to MoM changes
        apply_transforms: Apply SymLog to MoM changes
        apply_scaling: Apply RobustScaler normalization
    """
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    snapshot_dates = pd.date_range(start=start_dt, end=end_dt, freq='ME')

    logger.info(f"Preparing FRED Snapshots from {start_dt.date()} to {end_dt.date()}")
    logger.info(f"Options: mom_conversion={apply_mom_conversion}, transforms={apply_transforms}, scaling={apply_scaling}")

    for i, snap_date in enumerate(snapshot_dates):
        # 1. Load raw snapshot (employment LEVELS)
        raw_df = load_snapshot(FRED_SNAPSHOTS_DIR, snap_date)

        if raw_df.empty:
            continue

        raw_df['date'] = pd.to_datetime(raw_df['date'])
        working_df = raw_df.copy()

        # 2. Convert levels to MoM changes
        if apply_mom_conversion:
            working_df = calculate_mom_change(working_df)

        if working_df.empty:
            continue

        # 3. Apply SymLog transform
        if apply_transforms and apply_mom_conversion:
            working_df = preprocess_transforms(working_df)

        # 4. Apply RobustScaler (fit on history only)
        if apply_scaling:
            working_df = apply_robust_scaling_vintage(working_df, snap_date)

        # Add snapshot date
        working_df['snapshot_date'] = snap_date

        # Save in same schema as input
        decade_str = f"{snap_date.year // 10 * 10}s"
        year_str = str(snap_date.year)
        save_dir = PREPARED_FRED_DIR / decade_str / year_str
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{snap_date.strftime('%Y-%m')}.parquet"

        working_df.to_parquet(save_path, index=False)

        if i % 12 == 0:
            logger.info(f"Prepared snapshot for {snap_date.date()}")

    logger.info("FRED Snapshots preparation complete.")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_prepared_snapshot():
    """Verify the preprocessing worked correctly."""
    logger.info("Verifying Prepared FRED Snapshots...")

    # Check recent snapshot
    test_date = pd.to_datetime(END_DATE)

    # Try to find a valid snapshot
    for _ in range(12):
        decade_str = f"{test_date.year // 10 * 10}s"
        year_str = str(test_date.year)
        prepared_path = PREPARED_FRED_DIR / decade_str / year_str / f"{test_date.strftime('%Y-%m')}.parquet"

        if prepared_path.exists():
            break
        test_date = test_date - pd.DateOffset(months=1)

    if not prepared_path.exists():
        logger.error(f"Verification failed: Could not find file at {prepared_path}")
        return

    df = pd.read_parquet(prepared_path)
    series = df['series_name'].unique()

    logger.info(f"--- Verification Report for {test_date.date()} ---")
    logger.info(f"Total unique series: {len(series)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Check that we DON'T have redundant lag/rolling features
    lag_series = [s for s in series if '_lag' in s]
    rolling_series = [s for s in series if '_rolling' in s]

    if not lag_series:
        logger.info("OK: No lag features (created downstream)")
    else:
        logger.warning(f"WARNING: Found lag features that may be redundant: {lag_series[:3]}...")

    if not rolling_series:
        logger.info("OK: No rolling features (created downstream)")
    else:
        logger.warning(f"WARNING: Found rolling features that may be redundant: {rolling_series[:3]}...")

    # Check transforms were applied
    if 'value_raw' in df.columns:
        # Compare raw vs transformed
        total_series = df[df['series_name'] == 'total']
        if not total_series.empty:
            raw_vals = total_series['value_raw'].dropna()
            trans_vals = total_series['value'].dropna()

            logger.info(f"\n'total' series statistics:")
            logger.info(f"  Raw MoM - Skew: {raw_vals.skew():.3f}, Kurt: {raw_vals.kurtosis():.3f}")
            logger.info(f"  Transformed - Skew: {trans_vals.skew():.3f}, Kurt: {trans_vals.kurtosis():.3f}")
            logger.info(f"  Range: [{trans_vals.min():.3f}, {trans_vals.max():.3f}]")

    # Check schema matches expected format
    expected_cols = ['date', 'value', 'series_name']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
    else:
        logger.info("OK: Schema matches expected format")

    logger.info("Verification Complete.")


def compare_raw_vs_prepared():
    """
    Compare statistics of raw vs prepared data to validate transformations.
    """
    logger.info("Comparing Raw vs Prepared FRED Snapshots...")

    test_date = pd.to_datetime(END_DATE)

    # Find valid snapshots
    for _ in range(12):
        decade_str = f"{test_date.year // 10 * 10}s"
        year_str = str(test_date.year)

        raw_path = FRED_SNAPSHOTS_DIR / decade_str / year_str / f"{test_date.strftime('%Y-%m')}.parquet"
        prepared_path = PREPARED_FRED_DIR / decade_str / year_str / f"{test_date.strftime('%Y-%m')}.parquet"

        if raw_path.exists() and prepared_path.exists():
            break
        test_date = test_date - pd.DateOffset(months=1)

    if not raw_path.exists():
        logger.error("Could not find raw snapshot")
        return

    if not prepared_path.exists():
        logger.error("Could not find prepared snapshot - run prepare_fred_snapshots() first")
        return

    raw_df = pd.read_parquet(raw_path)
    prepared_df = pd.read_parquet(prepared_path)

    logger.info(f"\n=== Comparison for {test_date.date()} ===")
    logger.info(f"Raw: {len(raw_df)} rows, Prepared: {len(prepared_df)} rows")

    # Compare 'total' series (main NFP)
    for series_name in ['total', 'total_nsa']:
        raw_series = raw_df[raw_df['series_name'] == series_name]['value'].dropna()
        prep_series = prepared_df[prepared_df['series_name'] == series_name]['value'].dropna()

        if raw_series.empty or prep_series.empty:
            continue

        # Raw is levels, prepared is transformed MoM changes
        raw_mom = raw_series.diff().dropna()

        logger.info(f"\n{series_name}:")
        logger.info(f"  Raw Levels - Mean: {raw_series.mean():.0f}K")
        logger.info(f"  Raw MoM    - Skew: {raw_mom.skew():.3f}, Kurt: {raw_mom.kurtosis():.3f}")
        logger.info(f"  Prepared   - Skew: {prep_series.skew():.3f}, Kurt: {prep_series.kurtosis():.3f}")

        skew_improvement = abs(raw_mom.skew()) - abs(prep_series.skew())
        kurt_improvement = abs(raw_mom.kurtosis()) - abs(prep_series.kurtosis())

        logger.info(f"  Improvement - Skew: {skew_improvement:+.3f}, Kurt: {kurt_improvement:+.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FRED Employment Snapshots")
    parser.add_argument('--no-mom', action='store_true', help="Skip MoM conversion (keep levels)")
    parser.add_argument('--no-transforms', action='store_true', help="Skip SymLog transform")
    parser.add_argument('--no-scaling', action='store_true', help="Skip RobustScaler")
    parser.add_argument('--verify-only', action='store_true', help="Only run verification")
    parser.add_argument('--compare', action='store_true', help="Compare raw vs prepared")

    args = parser.parse_args()

    if args.verify_only:
        verify_prepared_snapshot()
    elif args.compare:
        compare_raw_vs_prepared()
    else:
        prepare_fred_snapshots(
            apply_mom_conversion=not args.no_mom,
            apply_transforms=not args.no_transforms,
            apply_scaling=not args.no_scaling
        )
        verify_prepared_snapshot()
