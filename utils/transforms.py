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
    # IMPORTANT: This must run BEFORE add_pct_change_copies() to prevent
    # creating symlog copies of derived series (pct_chg, symlog_pct_chg).
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
    Create pct-change-transformed copies of raw AND symlog series.

    Creates two sets of pct_change copies:
    - From raw series X → X_pct_chg
    - From symlog series X_symlog → X_symlog_pct_chg

    Applied AFTER add_symlog_copies and BEFORE compute_all_features so that
    pct_change variants get z-scores, rolling stats, and lags.

    Args:
        df: Long-format DataFrame with columns [series_name, value, ...].
        skip_series: Set of series_name values to exclude
            (e.g. binary regime indicators where pct_change is meaningless).

    Returns:
        DataFrame with original rows + pct_change copies (suffixed
        with '_pct_chg' or '_symlog_pct_chg').
    """
    if df.empty:
        return df

    # Create pct_change for raw AND symlog series (not already pct_chg)
    # Note: _symlog_pct_chg ends with _pct_chg so the guard catches both
    mask = (
        ~df['series_name'].isin(skip_series)
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
        # Raw series X → X_pct_chg, Symlog series X_symlog → X_symlog_pct_chg
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
    
    z = (series - rmean) / rstd
    
    # Handle constant sequences (std=0) -> Z-score should be 0 (value exactly equals mean)
    # Only replace if mean is valid (i.e. enough data exists), preventing backfill of missing data
    is_constant = (rstd == 0) & rmean.notna()
    if is_constant.any():
        z = z.mask(is_constant, 0.0)
        
    return z


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

    if df.empty:
        return df

    base_cols = [c for c in ['date', 'release_date', 'series_code', 'snapshot_date']
                 if c in df.columns]
    series_list = sorted(df['series_name'].unique())  # Sort for determinism
    
    # Collect all base (un-lagged) features first
    base_features_list = []

    for series in series_list:
        sdf = df[df['series_name'] == series].copy().sort_values('date')
        meta = sdf[base_cols].copy()
        vals = sdf['value']

        is_binary = series in skip_series
        is_pct_chg = series.endswith('_pct_chg') or series.endswith('_symlog_pct_chg')

        # Level (always emitted)
        _emit(base_features_list, meta, series, vals)

        if is_binary:
            continue

        # --- MoM diff (raw + symlog only) ---
        if not is_pct_chg:
            diff = vals.diff()
            _emit(base_features_list, meta, f"{series}_diff", diff)
            _emit(base_features_list, meta, f"{series}_diff_zscore_3m",
                  _rolling_zscore(diff, 3, 2))
            _emit(base_features_list, meta, f"{series}_diff_zscore_12m",
                  _rolling_zscore(diff, 12, 6))

        # --- Level z-scores (all non-binary) ---
        _emit(base_features_list, meta, f"{series}_zscore_3m",
              _rolling_zscore(vals, 3, 2))
        _emit(base_features_list, meta, f"{series}_zscore_12m",
              _rolling_zscore(vals, 12, 6))

        # --- Rolling stats (all non-binary) ---
        _emit(base_features_list, meta, f"{series}_rolling_mean_3m",
              vals.rolling(3, min_periods=2).mean())
        _emit(base_features_list, meta, f"{series}_rolling_std_6m",
              vals.rolling(6, min_periods=3).std())

        # (Old lag loop removed here - lags are applied globally below)

        # --- Multi-period changes (raw + symlog only) ---
        if not is_pct_chg:
            for period in [3, 6, 12]:
                _emit(base_features_list, meta, f"{series}_chg_{period}m",
                      vals.diff(period))

    # --- Global Lag Application ---
    # Apply lags to EVERY feature generated above
    # Lags: 1, 3, 6, 12, 18 months
    lags = [1, 3, 6, 12, 18]
    final_output_list = []
    
    for feature_block in base_features_list:
        # Add the un-lagged feature itself
        final_output_list.append(feature_block)
        
        # Add lagged variants
        # Note: feature_block is sorted by date because it comes from sorted sdf
        # and operations preserve order. shift() works correctly on value column.
        for lag in lags:
            lag_block = feature_block.copy()
            lag_block['value'] = lag_block['value'].shift(lag)
            
            # Suffix name and code
            lag_suffix = f"_lag_{lag}m"
            lag_block['series_name'] = lag_block['series_name'] + lag_suffix
            
            if 'series_code' in lag_block.columns:
                 lag_block['series_code'] = lag_block['series_code'].astype(str) + lag_suffix
            
            final_output_list.append(lag_block)

    result = pd.concat(final_output_list, ignore_index=True)
    result = result.dropna(subset=['value'])
    return result


# =============================================================================
# WIDE-FORMAT VECTORIZED FEATURE COMPUTATION
# =============================================================================

def _rolling_zscore_wide(df: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Vectorized rolling z-score across all columns at once."""
    rmean = df.rolling(window=window, min_periods=min_periods).mean()
    rstd = df.rolling(window=window, min_periods=min_periods).std()
    z = (df - rmean) / rstd
    # Handle constant sequences (std=0) -> z-score = 0
    z = z.where(rstd != 0, 0.0)
    # Restore NaN where mean was NaN (not enough data)
    z = z.where(rmean.notna(), np.nan)
    return z


def compute_features_wide(
    long_df: pd.DataFrame,
    apply_mom: bool = True,
) -> pd.DataFrame:
    """
    Vectorized wide-format feature computation for FRED employment snapshots.

    Replaces the sequential long-format pipeline of:
        convert_levels_to_mom → add_symlog_copies → add_pct_change_copies → compute_all_features

    All operations are vectorized across columns (no Python loops over series).

    Args:
        long_df: Long-format DataFrame with columns [date, value, series_name, ...].
        apply_mom: If True, convert levels to MoM changes first.

    Returns:
        Wide-format DataFrame with DatetimeIndex (date) and feature columns.
        Each column is a feature name (e.g. 'total', 'total_symlog', 'total_diff_lag_3m').
    """
    if long_df.empty:
        return pd.DataFrame()

    df = long_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Convert categorical columns to string for pivot
    for col in ['series_name', 'series_code']:
        if col in df.columns and hasattr(df[col], 'cat'):
            df[col] = df[col].astype(str)

    # --- Pivot to wide: rows=date, columns=series_name ---
    wide = df.pivot_table(index='date', columns='series_name', values='value', aggfunc='first')
    wide = wide.sort_index()

    # --- MoM conversion (diff of levels) ---
    if apply_mom:
        wide = wide.diff()
        wide = wide.iloc[1:]  # drop first row (NaN from diff)

    # --- Build all feature columns in wide format ---
    feature_frames = []  # list of DataFrames, all same index

    base_cols = list(wide.columns)  # original ~160 series (MoM values)

    # 1. Symlog of base
    symlog_wide = np.sign(wide) * np.log1p(np.abs(wide))
    symlog_wide.columns = [f"{c}_symlog" for c in base_cols]
    symlog_cols = list(symlog_wide.columns)

    # 2. Pct change of base and symlog
    pct_base = wide.pct_change() * 100
    pct_base = pct_base.replace([np.inf, -np.inf], np.nan)
    pct_base.columns = [f"{c}_pct_chg" for c in base_cols]
    pct_base_cols = list(pct_base.columns)

    pct_symlog = symlog_wide.pct_change() * 100
    pct_symlog = pct_symlog.replace([np.inf, -np.inf], np.nan)
    pct_symlog.columns = [f"{c}_symlog_pct_chg" for c in base_cols]
    pct_symlog_cols = list(pct_symlog.columns)

    # Combine all base series into one wide DataFrame
    all_wide = pd.concat([wide, symlog_wide, pct_base, pct_symlog], axis=1)

    # Identify column categories for feature generation
    raw_symlog_cols = base_cols + symlog_cols  # get diff + multi-period
    pct_chg_cols = pct_base_cols + pct_symlog_cols  # no diff, no multi-period
    all_series_cols = raw_symlog_cols + pct_chg_cols

    # --- Generate features (all vectorized) ---
    # Level (the base values themselves)
    feature_frames.append(all_wide)

    # Diff (raw + symlog only)
    diff_df = all_wide[raw_symlog_cols].diff()
    diff_df.columns = [f"{c}_diff" for c in raw_symlog_cols]
    feature_frames.append(diff_df)

    # Diff z-scores (raw + symlog only)
    for window, min_p, suffix in [(3, 2, '3m'), (12, 6, '12m')]:
        dz = _rolling_zscore_wide(diff_df, window, min_p)
        dz.columns = [f"{c}_diff_zscore_{suffix}" for c in raw_symlog_cols]
        feature_frames.append(dz)

    # Level z-scores (all non-binary)
    for window, min_p, suffix in [(3, 2, '3m'), (12, 6, '12m')]:
        lz = _rolling_zscore_wide(all_wide[all_series_cols], window, min_p)
        lz.columns = [f"{c}_zscore_{suffix}" for c in all_series_cols]
        feature_frames.append(lz)

    # Rolling stats (all non-binary)
    rm3 = all_wide[all_series_cols].rolling(3, min_periods=2).mean()
    rm3.columns = [f"{c}_rolling_mean_3m" for c in all_series_cols]
    feature_frames.append(rm3)

    rs6 = all_wide[all_series_cols].rolling(6, min_periods=3).std()
    rs6.columns = [f"{c}_rolling_std_6m" for c in all_series_cols]
    feature_frames.append(rs6)

    # Multi-period changes (raw + symlog only)
    for period in [3, 6, 12]:
        mc = all_wide[raw_symlog_cols].diff(period)
        mc.columns = [f"{c}_chg_{period}m" for c in raw_symlog_cols]
        feature_frames.append(mc)

    # --- Combine all base features ---
    all_features = pd.concat(feature_frames, axis=1)

    # --- Apply lags to ALL feature columns ---
    lag_frames = [all_features]  # include un-lagged
    for lag in [1, 3, 6, 12, 18]:
        lagged = all_features.shift(lag)
        lagged.columns = [f"{c}_lag_{lag}m" for c in all_features.columns]
        lag_frames.append(lagged)

    final_wide = pd.concat(lag_frames, axis=1)
    final_wide.index.name = 'date'

    return final_wide


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

    For each feature with a _diff suffix, creates both:
    - _pct_chg variant (from raw series)
    - _symlog_pct_chg variant (from symlog series)

    Args:
        features: Original feature name set.
        skip: Feature names to exclude (e.g. binary regime indicators).

    Returns:
        Set of pct_chg and symlog_pct_chg feature names.

    Example:
        >>> generate_pct_chg_feature_names({'AHE_Private_diff_zscore_3m'})
        {'AHE_Private_pct_chg_zscore_3m', 'AHE_Private_symlog_pct_chg_zscore_3m'}
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
                result.add(f"{base}_symlog{pct_suffix}")
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


def winsorize_covid_period(
    data: pd.DataFrame | pd.Series,
    covid_start: str = '2020-03-01',
    covid_end: str = '2020-05-01',
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.DataFrame | pd.Series:
    """
    Winsorize values during the COVID period by clipping them to quantile
    boundaries computed from the non-COVID portion of the data.

    This keeps the same number of rows (preserving time-series continuity)
    while neutralizing extreme COVID-era outliers that would otherwise
    distort correlations and other statistical measures.

    Args:
        data: Wide-format DataFrame (DatetimeIndex, columns = series) or
              a Series with DatetimeIndex.
        covid_start: First month of the COVID shock (inclusive).
        covid_end:   Last month of the COVID shock (inclusive).
        lower_percentile: Lower quantile for clipping (default 1%).
        upper_percentile: Upper quantile for clipping (default 99%).

    Returns:
        Copy of *data* with COVID-period values clipped per-column.
    """
    data = data.copy()
    covid_mask = (data.index >= pd.Timestamp(covid_start)) & \
                 (data.index <= pd.Timestamp(covid_end))

    if not covid_mask.any():
        return data

    if isinstance(data, pd.DataFrame):
        non_covid = data.loc[~covid_mask]
        # Vectorized: compute quantiles for all columns at once (returns Series)
        lower_bounds = non_covid.quantile(lower_percentile)
        upper_bounds = non_covid.quantile(upper_percentile)
        # Clip all COVID rows against per-column bounds in one operation
        data.loc[covid_mask] = data.loc[covid_mask].clip(
            lower=lower_bounds, upper=upper_bounds, axis=1
        )
    else:  # pd.Series
        non_covid = data.loc[~covid_mask]
        lower = non_covid.quantile(lower_percentile)
        upper = non_covid.quantile(upper_percentile)
        data.loc[covid_mask] = data.loc[covid_mask].clip(
            lower=lower, upper=upper
        )

    return data


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
