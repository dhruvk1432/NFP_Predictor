"""
Transform Utilities

Shared data transformation functions for NFP Predictor.
Includes SymLog transform, scaling, and other preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional


def apply_symlog(x: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """
    Apply symmetric log transform: sign(x) * log1p(abs(x))
    
    This transform:
    - Handles negative values (unlike log)
    - Compresses extreme values (reduces kurtosis)
    - Preserves sign and relative magnitude
    - Is invertible via inverse_symlog()
    
    For NFP MoM changes, reduces:
    - Skewness: -6.5 -> -1.1
    - Kurtosis: 81 -> -0.7
    
    Args:
        x: Input value(s) - can be scalar, array, or Series
        
    Returns:
        Transformed value(s) with same type as input
        
    Example:
        >>> apply_symlog(-1000)
        -6.908...
        >>> apply_symlog(1000)
        6.908...
    """
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_symlog(y: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """
    Inverse of symlog transform for prediction recovery.
    
    Formula: sign(y) * (exp(abs(y)) - 1)
    
    Args:
        y: Transformed value(s)
        
    Returns:
        Original scale value(s)
        
    Example:
        >>> inverse_symlog(apply_symlog(-1000))
        -1000.0
    """
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


def apply_log1p(x: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """
    Apply log1p transform: log(1 + x)
    
    Only valid for x >= 0. For values that can be negative, use apply_symlog.
    
    Args:
        x: Non-negative input value(s)
        
    Returns:
        Transformed value(s)
    """
    return np.log1p(x)


def inverse_log1p(y: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """
    Inverse of log1p transform.
    
    Args:
        y: Transformed value(s)
        
    Returns:
        Original scale value(s)
    """
    return np.expm1(y)


def add_symlog_copies(
    df: pd.DataFrame,
    skip_series: frozenset = frozenset()
) -> pd.DataFrame:
    """
    Create symlog-transformed copies of all series and concatenate with originals.

    Applied BEFORE diff/zscore so that downstream differencing and z-score
    calculations operate on the symlog-transformed scale as well.

    Args:
        df: Long-format DataFrame with columns [series_name, value, ...].
            Optionally contains series_code.
        skip_series: Set of series_name values to exclude from symlog
            (e.g. binary regime indicators where symlog is meaningless).

    Returns:
        DataFrame with original rows + symlog copies (series_name suffixed
        with '_symlog').
    """
    if df.empty:
        return df

    # Only symlog non-skipped series
    if skip_series:
        mask = ~df['series_name'].isin(skip_series)
        base_df = df[mask]
    else:
        base_df = df

    if base_df.empty:
        return df

    symlog_df = base_df.copy()
    symlog_df['value'] = apply_symlog(symlog_df['value'])
    symlog_df['series_name'] = symlog_df['series_name'] + '_symlog'
    if 'series_code' in symlog_df.columns:
        symlog_df['series_code'] = symlog_df['series_code'].astype(str) + '_symlog'

    return pd.concat([df, symlog_df], ignore_index=True)


def add_pct_change_copies(
    df: pd.DataFrame,
    skip_series: frozenset = frozenset()
) -> pd.DataFrame:
    """
    Create pct-change-transformed copies of raw series and concatenate with originals.

    Only applies to raw series (not _symlog or already _pct_chg).
    Applied AFTER add_symlog_copies and BEFORE compute_all_features so that
    pct_change variants get z-scores, rolling stats, and lags.

    Args:
        df: Long-format DataFrame with columns [series_name, value, ...].
        skip_series: Set of series_name values to exclude
            (e.g. binary regime indicators where pct_change is meaningless).

    Returns:
        DataFrame with original rows + pct_change copies (series_name suffixed
        with '_pct_chg').
    """
    if df.empty:
        return df

    # Only create pct_change for raw series (not symlog, not already pct_chg)
    mask = (
        ~df['series_name'].isin(skip_series)
        & ~df['series_name'].str.endswith('_symlog')
        & ~df['series_name'].str.endswith('_pct_chg')
    )
    base_df = df[mask]

    if base_df.empty:
        return df

    pct_blocks = []
    for series_name, group in base_df.groupby('series_name'):
        group = group.sort_values('date')
        pct_group = group.copy()
        pct_group['value'] = group['value'].pct_change() * 100
        pct_group['series_name'] = series_name + '_pct_chg'
        if 'series_code' in pct_group.columns:
            pct_group['series_code'] = pct_group['series_code'].astype(str) + '_pct_chg'
        pct_blocks.append(pct_group)

    if not pct_blocks:
        return df

    pct_df = pd.concat(pct_blocks, ignore_index=True)
    # Replace inf with NaN (from division by zero in pct_change)
    pct_df['value'] = pct_df['value'].replace([np.inf, -np.inf], np.nan)

    return pd.concat([df, pct_df], ignore_index=True)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute rolling z-score: (x - rolling_mean) / rolling_std."""
    rmean = series.rolling(window=window, min_periods=min_periods).mean()
    rstd = series.rolling(window=window, min_periods=min_periods).std()
    return (series - rmean) / rstd


def _emit(result_list: list, meta: pd.DataFrame, name: str, values: pd.Series):
    """Create a long-format block and append to result_list."""
    block = meta.copy()
    block['series_name'] = name
    block['value'] = values.values
    result_list.append(block)


def compute_all_features(
    df: pd.DataFrame,
    skip_series: frozenset = frozenset()
) -> pd.DataFrame:
    """
    Compute the full suite of derived features for all series in a long-format DataFrame.

    Expected input: DataFrame that has already been through add_symlog_copies() and
    add_pct_change_copies(), so it contains raw, _symlog, and _pct_chg variants.

    Feature suite per series type:
    - Binary (skip_series): level only (1 feature)
    - Pct_Chg (_pct_chg suffix): level + z-scores + rolling + lags (9 features)
    - Raw/Symlog: level + diff + diff z-scores + level z-scores + rolling + lags
      + multi-period changes (15 features)

    Args:
        df: Long-format DataFrame with columns [date, series_name, value, ...].
        skip_series: Set of series_name values that are binary indicators (level only).

    Returns:
        DataFrame in long format with all derived features.
    """
    if df.empty:
        return df

    base_cols = [c for c in ['date', 'release_date', 'series_code', 'snapshot_date']
                 if c in df.columns]
    series_list = df['series_name'].unique()
    result_list = []

    for series in series_list:
        sdf = df[df['series_name'] == series].copy().sort_values('date')
        meta = sdf[base_cols].copy()
        vals = sdf['value']

        is_binary = series in skip_series
        is_pct_chg = series.endswith('_pct_chg')

        # Level (always emitted)
        _emit(result_list, meta, series, vals)

        if is_binary:
            continue

        # --- MoM diff (raw + symlog only) ---
        if not is_pct_chg:
            diff = vals.diff()
            _emit(result_list, meta, f"{series}_diff", diff)
            _emit(result_list, meta, f"{series}_diff_zscore_3m",
                  _rolling_zscore(diff, 3, 2))
            _emit(result_list, meta, f"{series}_diff_zscore_12m",
                  _rolling_zscore(diff, 12, 6))

        # --- Level z-scores (all non-binary) ---
        _emit(result_list, meta, f"{series}_zscore_3m",
              _rolling_zscore(vals, 3, 2))
        _emit(result_list, meta, f"{series}_zscore_12m",
              _rolling_zscore(vals, 12, 6))

        # --- Rolling stats (all non-binary) ---
        _emit(result_list, meta, f"{series}_rolling_mean_3m",
              vals.rolling(3, min_periods=2).mean())
        _emit(result_list, meta, f"{series}_rolling_std_6m",
              vals.rolling(6, min_periods=3).std())

        # --- Lags (all non-binary) ---
        for lag in [1, 3, 6, 12]:
            _emit(result_list, meta, f"{series}_lag_{lag}m", vals.shift(lag))

        # --- Multi-period changes (raw + symlog only) ---
        if not is_pct_chg:
            for period in [3, 6, 12]:
                _emit(result_list, meta, f"{series}_chg_{period}m",
                      vals.diff(period))

    result = pd.concat(result_list, ignore_index=True)
    result = result.dropna(subset=['value'])
    return result


def generate_symlog_feature_names(
    features: set,
    skip: frozenset = frozenset()
) -> set:
    """
    Generate symlog-equivalent feature names from a set of original feature names.

    Inserts '_symlog' before the diff/zscore suffix so that the naming reflects
    the transformation order: raw -> symlog -> diff -> zscore.

    Args:
        features: Original feature name set.
        skip: Feature names to exclude (e.g. binary regime indicators).

    Returns:
        Set of symlog-equivalent feature names.

    Example:
        >>> generate_symlog_feature_names({'AHE_Private_diff_zscore_3m'})
        {'AHE_Private_symlog_diff_zscore_3m'}
    """
    result = set()
    for f in features:
        if f in skip:
            continue
        for suffix in ('_diff_zscore_12m', '_diff_zscore_3m', '_diff',
                       '_pct_chg_zscore_12m', '_pct_chg_zscore_3m', '_pct_chg'):
            if f.endswith(suffix):
                base = f[:-len(suffix)]
                result.add(f"{base}_symlog{suffix}")
                break
        else:
            result.add(f"{f}_symlog")
    return result


def generate_pct_chg_feature_names(
    features: set,
    skip: frozenset = frozenset()
) -> set:
    """
    Generate pct_chg-equivalent feature names from diff-based feature names.

    For each feature with a _diff suffix, creates the corresponding _pct_chg variant.
    Level-only features (no _diff) are skipped since pct_chg is a derivative.

    Args:
        features: Original feature name set.
        skip: Feature names to exclude (e.g. binary regime indicators).

    Returns:
        Set of pct_chg-equivalent feature names.

    Example:
        >>> generate_pct_chg_feature_names({'AHE_Private_diff_zscore_3m'})
        {'AHE_Private_pct_chg_zscore_3m'}
    """
    result = set()
    for f in features:
        if f in skip:
            continue
        for diff_suffix, pct_suffix in (
            ('_diff_zscore_12m', '_pct_chg_zscore_12m'),
            ('_diff_zscore_3m', '_pct_chg_zscore_3m'),
            ('_diff', '_pct_chg'),
        ):
            if f.endswith(diff_suffix):
                base = f[:-len(diff_suffix)]
                result.add(f"{base}{pct_suffix}")
                break
    return result


def calculate_z_score(
    series: pd.Series, 
    window: int = 252, 
    min_periods: int = 60
) -> pd.Series:
    """
    Calculate rolling Z-score for a series.
    
    Args:
        series: Input time series
        window: Rolling window size (default: 252 trading days = 1 year)
        min_periods: Minimum observations required
        
    Returns:
        Z-score series (standard deviations from rolling mean)
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    return (series - rolling_mean) / rolling_std


def winsorize(
    series: pd.Series, 
    lower_percentile: float = 0.01, 
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    Winsorize a series by clipping extreme values.
    
    Args:
        series: Input series
        lower_percentile: Lower bound percentile (default: 1%)
        upper_percentile: Upper bound percentile (default: 99%)
        
    Returns:
        Winsorized series
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    
    return series.clip(lower=lower, upper=upper)


def calculate_mom_pct_change(
    df: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'value',
    series_col: str = 'series_name'
) -> pd.DataFrame:
    """
    Calculate month-over-month percentage change within each series.
    
    Args:
        df: DataFrame with date, value, and series columns
        date_col: Name of date column
        value_col: Name of value column
        series_col: Name of series identifier column
        
    Returns:
        DataFrame with added 'pct_change' column
    """
    df = df.sort_values([series_col, date_col])
    df['pct_change'] = df.groupby(series_col)[value_col].pct_change() * 100
    
    return df


if __name__ == "__main__":
    # Test transforms
    print("Testing transforms...")
    
    # Test symlog
    test_values = [-1000, -100, -10, 0, 10, 100, 1000]
    for val in test_values:
        transformed = apply_symlog(val)
        recovered = inverse_symlog(transformed)
        print(f"  {val:6} -> {transformed:8.4f} -> {recovered:8.1f}")
    
    # Verify invertibility
    arr = np.array(test_values, dtype=float)
    recovered = inverse_symlog(apply_symlog(arr))
    assert np.allclose(arr, recovered), "Symlog is not perfectly invertible!"
    print("\n✓ Symlog invertibility verified")
