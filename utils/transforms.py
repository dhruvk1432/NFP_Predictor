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
