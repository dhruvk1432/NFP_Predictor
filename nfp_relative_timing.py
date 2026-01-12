"""
NFP-Relative Timing Utility

Shared module for maintaining consistent timing of backfilled release dates
relative to NFP releases. Used by Unifier, NOAA, and FRED exogenous data loaders.

Key Concept:
- Many economic indicators have consistent release timing relative to NFP
- Example: ISM Manufacturing releases 1 day after month-end, typically before NFP
- Example: JOLTS releases ~6 weeks after month-end, typically after NFP
- When backfilling missing release dates, we should maintain these historical patterns

This module provides utilities to:
1. Load NFP release dates from target files
2. Calculate median offset from NFP for each series
3. Apply consistent offsets when backfilling missing dates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import timedelta
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# Cache for NFP releases to avoid repeated file reads
_NFP_RELEASES_CACHE: Optional[pd.DataFrame] = None


def load_nfp_releases() -> pd.DataFrame:
    """
    Load NFP release dates from target file.
    
    Returns:
        DataFrame with columns: ['ds', 'release_date']
        - ds: Event month (e.g., 2020-01-01 for January 2020 NFP)
        - release_date: When NFP for that month was released
    """
    global _NFP_RELEASES_CACHE
    
    if _NFP_RELEASES_CACHE is not None:
        return _NFP_RELEASES_CACHE
    
    nfp_file = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    
    if not nfp_file.exists():
        raise FileNotFoundError(f"NFP target file not found: {nfp_file}")
    
    df = pd.read_parquet(nfp_file)
    
    # Extract only needed columns
    if 'ds' in df.columns and 'release_date' in df.columns:
        nfp_releases = df[['ds', 'release_date']].copy()
    else:
        raise ValueError(f"NFP file missing required columns. Has: {df.columns.tolist()}")
    
    nfp_releases['ds'] = pd.to_datetime(nfp_releases['ds'])
    nfp_releases['release_date'] = pd.to_datetime(nfp_releases['release_date'])
    
    # Remove duplicates if any
    nfp_releases = nfp_releases.drop_duplicates(subset=['ds'])
    
    _NFP_RELEASES_CACHE = nfp_releases
    
    logger.info(f"Loaded {len(nfp_releases)} NFP release dates")
    logger.info(f"  Range: {nfp_releases['ds'].min().date()} to {nfp_releases['ds'].max().date()}")
    
    return nfp_releases


def get_nfp_release_for_month(event_month: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Get the NFP release date for a specific event month.

    Args:
        event_month: Month for which NFP data refers (e.g., 2020-01-01)

    Returns:
        NFP release date, or None if not found
    """
    nfp_releases = load_nfp_releases()

    # Normalize to month start
    event_month = pd.Timestamp(event_month).replace(day=1)

    match = nfp_releases[nfp_releases['ds'] == event_month]

    if len(match) == 0:
        return None

    return match['release_date'].iloc[0]


def get_nfp_release_map(start_date=None, end_date=None) -> Dict[pd.Timestamp, pd.Timestamp]:
    """
    Get dictionary mapping observation months to NFP release dates.

    This is the format commonly used by data loaders for snapshot alignment.
    Uses cached NFP releases to avoid redundant file reads.

    Args:
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)

    Returns:
        Dict mapping observation_month -> nfp_release_date
    """
    nfp_releases = load_nfp_releases()

    df = nfp_releases.copy()

    # Apply date filters if provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df['ds'] >= start_dt]

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df[df['ds'] <= end_dt]

    return dict(zip(df['ds'], df['release_date']))


def calculate_median_offset_from_nfp(
    series_data: pd.DataFrame,
    event_col: str = 'date',
    release_col: str = 'release_date'
) -> Tuple[float, int]:
    """
    Calculate the median offset (in days) of a series' release dates from NFP releases.
    
    Args:
        series_data: DataFrame with event dates and actual release dates
        event_col: Column name for event/observation month
        release_col: Column name for actual release date
    
    Returns:
        Tuple of (median_offset_days, num_observations)
        - median_offset_days: Median days between NFP release and series release
          (negative = series releases before NFP, positive = after NFP)
        - num_observations: Number of observations used for calculation
    
    Example:
        If ISM releases 1 day after month-end and NFP releases 5 days after month-end,
        ISM is typically 4 days BEFORE NFP, so offset = -4.0
    """
    nfp_releases = load_nfp_releases()
    
    # Ensure columns exist
    if event_col not in series_data.columns or release_col not in series_data.columns:
        logger.warning(f"Missing columns. Has: {series_data.columns.tolist()}")
        return 0.0, 0
    
    # Filter to rows with actual release dates (not backfilled)
    has_release = series_data[series_data[release_col].notna()].copy()
    
    if len(has_release) == 0:
        return 0.0, 0
    
    has_release[event_col] = pd.to_datetime(has_release[event_col])
    has_release[release_col] = pd.to_datetime(has_release[release_col])
    
    # Normalize event month to month start
    has_release['event_month'] = has_release[event_col].dt.to_period('M').dt.to_timestamp()
    
    # Merge with NFP releases
    merged = has_release.merge(
        nfp_releases,
        left_on='event_month',
        right_on='ds',
        how='inner',
        suffixes=('_series', '_nfp')
    )
    
    if len(merged) == 0:
        logger.warning("No matching NFP releases found for series data")
        return 0.0, 0
    
    # Calculate offset: series_release - nfp_release (in days)
    merged['offset_days'] = (
        merged[f'{release_col}_series'] - merged['release_date_nfp']
    ).dt.total_seconds() / 86400  # Convert to days
    
    # Remove outliers (>365 days suggests data error)
    valid_offsets = merged[
        (merged['offset_days'].abs() <= 365) &
        (merged['offset_days'].notna())
    ]
    
    if len(valid_offsets) == 0:
        return 0.0, 0
    
    median_offset = valid_offsets['offset_days'].median()
    num_obs = len(valid_offsets)
    
    return median_offset, num_obs


def apply_nfp_relative_adjustment(
    event_month: pd.Timestamp,
    base_release_date: pd.Timestamp,
    median_offset_days: float,
    use_adjustment: bool = True
) -> pd.Timestamp:
    """
    Adjust a backfilled release date to maintain consistent NFP-relative timing.
    
    Args:
        event_month: Month the data refers to
        base_release_date: Initial estimate (e.g., from median lag or fixed rule)
        median_offset_days: Median offset from NFP (from calculate_median_offset_from_nfp)
        use_adjustment: If False, return base_release_date unchanged
    
    Returns:
        Adjusted release date
    
    Logic:
        1. Find NFP release for this event month
        2. Calculate target = NFP_release + median_offset
        3. If target is reasonable, use it; otherwise keep base estimate
    
    Example:
        - Event: January 2020
        - Base estimate: Feb 10, 2020 (from median lag)
        - NFP for Jan 2020: Feb 7, 2020
        - Median offset: -4 days (series typically 4 days before NFP)
        - Adjusted: Feb 3, 2020 (= Feb 7 - 4 days)
    """
    if not use_adjustment:
        return base_release_date
    
    # Get NFP release for this event month
    nfp_release = get_nfp_release_for_month(event_month)
    
    if nfp_release is None:
        logger.debug(f"No NFP release found for {event_month.date()}, using base estimate")
        return base_release_date
    
    # Calculate target date
    adjusted_release = nfp_release + timedelta(days=median_offset_days)
    
    # Sanity checks:
    # 1. Adjusted date should be after event month
    # 2. Adjusted date should not be more than 1 year after event
    # 3. Adjusted date should not be more than 180 days different from base estimate
    
    event_month_start = event_month.replace(day=1)
    one_year_later = event_month_start + pd.DateOffset(years=1)
    
    checks_pass = (
        (adjusted_release > event_month_start) and
        (adjusted_release < one_year_later) and
        (abs((adjusted_release - base_release_date).days) < 180)
    )
    
    if checks_pass:
        return adjusted_release
    else:
        logger.debug(
            f"NFP-relative adjustment failed sanity checks for {event_month.date()}, "
            f"using base estimate"
        )
        return base_release_date


def get_series_timing_stats(
    series_data: pd.DataFrame,
    series_name: str,
    event_col: str = 'date',
    release_col: str = 'release_date'
) -> Dict:
    """
    Get comprehensive timing statistics for a series relative to NFP.
    
    Useful for understanding and debugging release timing patterns.
    
    Returns:
        Dictionary with timing statistics
    """
    median_offset, num_obs = calculate_median_offset_from_nfp(
        series_data, event_col, release_col
    )
    
    stats = {
        'series_name': series_name,
        'median_offset_days': median_offset,
        'num_observations': num_obs,
        'typically_before_nfp': median_offset < 0,
        'typically_after_nfp': median_offset > 0,
    }
    
    if num_obs > 0:
        # Additional statistics
        nfp_releases = load_nfp_releases()
        has_release = series_data[series_data[release_col].notna()].copy()
        has_release['event_month'] = pd.to_datetime(has_release[event_col]).dt.to_period('M').dt.to_timestamp()
        
        merged = has_release.merge(
            nfp_releases,
            left_on='event_month',
            right_on='ds',
            how='inner',
            suffixes=('_series', '_nfp')
        )
        
        if len(merged) > 0:
            offsets = (merged[f'{release_col}_series'] - merged['release_date_nfp']).dt.total_seconds() / 86400
            stats.update({
                'min_offset_days': offsets.min(),
                'max_offset_days': offsets.max(),
                'std_offset_days': offsets.std(),
                'pct_before_nfp': (offsets < 0).mean() * 100,
            })
    
    return stats


if __name__ == "__main__":
    """Test the NFP-relative timing utilities."""
    logger.info("Testing NFP-Relative Timing Utilities")
    logger.info("="*60)
    
    # Test 1: Load NFP releases
    logger.info("\n1. Loading NFP releases...")
    nfp_releases = load_nfp_releases()
    logger.info(f"   Loaded {len(nfp_releases)} NFP release dates")
    logger.info(f"   Sample:\n{nfp_releases.head()}")
    
    # Test 2: Get release for specific month
    logger.info("\n2. Testing get_nfp_release_for_month...")
    test_month = pd.Timestamp('2020-01-01')
    nfp_date = get_nfp_release_for_month(test_month)
    logger.info(f"   NFP for {test_month.date()}: {nfp_date.date() if nfp_date else 'Not found'}")
    
    # Test 3: Apply adjustment
    logger.info("\n3. Testing apply_nfp_relative_adjustment...")
    base_est = pd.Timestamp('2020-02-10')
    adjusted = apply_nfp_relative_adjustment(
        event_month=test_month,
        base_release_date=base_est,
        median_offset_days=-4.0  # 4 days before NFP
    )
    logger.info(f"   Base estimate: {base_est.date()}")
    logger.info(f"   Adjusted (-4 days from NFP): {adjusted.date()}")
    logger.info(f"   Difference: {(adjusted - base_est).days} days")
    
    logger.info("\n" + "="*60)
    logger.info("NFP-Relative Timing Utilities test complete!")
