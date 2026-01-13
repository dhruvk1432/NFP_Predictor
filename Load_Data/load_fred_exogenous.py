import pandas as pd
import numpy as np
from fredapi import Fred
import sys
from pathlib import Path
from datetime import timedelta
import time

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import FRED_API_KEY, DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
# OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
# INT1: Import all NFP utilities at module level for consistency
from Prepare_Data.nfp_relative_timing import load_nfp_releases, get_nfp_release_map, apply_nfp_relative_adjustment
# OPTIMIZATION: Use shared utilities for snapshot path and MultiIndex flattening
from Load_Data.utils import get_snapshot_path, flatten_multiindex_columns

logger = setup_logger(__file__, TEMP_DIR)


# =============================================================================
# V1: FRED API Retry Logic with Exponential Backoff
# =============================================================================
def fred_api_call_with_retry(fred_func, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Execute a FRED API call with exponential backoff retry logic.

    V1: Handles rate limiting (429 errors) and transient failures gracefully.

    Args:
        fred_func: The FRED API function to call (e.g., fred.get_series)
        *args: Positional arguments for fred_func
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
        **kwargs: Keyword arguments for fred_func

    Returns:
        Result of the FRED API call

    Raises:
        Exception: If all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return fred_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a rate limit error (429) or transient error
            is_rate_limit = '429' in str(e) or 'rate limit' in error_str or 'too many requests' in error_str
            is_transient = 'timeout' in error_str or 'connection' in error_str or '503' in str(e)

            if is_rate_limit or is_transient:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(f"FRED API error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                # Non-retryable error, raise immediately
                raise

    # All retries exhausted
    logger.error(f"FRED API call failed after {max_retries} attempts")
    raise last_exception

FRED_SERIES = {
    #Daily Data - Financial Market Indicators
    "Credit_Spreads": "BAMLH0A0HYM2",
    "Yield_Curve": "T10Y2Y",
    "Oil_Prices": "DCOILWTICO",
    "VIX": "VIXCLS",  # CBOE Volatility Index - Market fear gauge
    "SP500": "SP500",  # S&P 500 Index - Market crashes & recoveries
    # High-velocity economic indicators for extreme events
    "Financial_Stress": "STLFSI4",  # St. Louis Fed Financial Stress Index (weekly)
    "Weekly_Econ_Index": "WEI",  # Weekly Economic Index (real-time)
    #Monthly Data (JOLTS_Openings and JOLTS_Hires dropped due to multicollinearity)
    # "JOLTS_Openings": "JTSJOL",  # DROPPED
    # "JOLTS_Hires": "JTSHIL",  # DROPPED
    # "JOLTS_Quits": "JTSQUR",
    # "JOLTS_Layoffs": "JTSLDL",
    # NEW: Regional Fed Employment Indices (monthly)
    "Empire_State_Emp": "USEPUINDX",  # Empire State Manufacturing Employment Index
    "Philly_Fed_Emp": "USPHCICH",  # Philadelphia Fed Employment Diffusion Index
    # Weekly Jobless Claims (ICSA and IURSA dropped due to multicollinearity)
    # only data available before each NFP report is included in that month's features
    "ICSA": "ICSA", #Initial Claims Seasonally Adjusted
    "CCSA": "CCSA",  # Continued Claims Seasonally Adjusted (KEPT)
    "IURSA": "IURSA" # Insured Unemployment Rate Seasonally Adjusted
}

# def clean_jolts_release_dates(df, ref_month_col='date', release_col='realtime_start', nfp_offset_days=None):
#     """
#     Clean and impute JOLTS release dates.

#     Logic: For each observation month, if release date is missing or more than 2 months late,
#     impute it as the first Tuesday of the 2nd month after the observation.

#     ENHANCEMENT: If nfp_offset_days provided, apply NFP-relative adjustment to maintain
#     historical timing consistency relative to NFP releases.

#     Example:
#         ref_month = 2020-01-01 (January)
#         deadline = 2020-03-31 (end of March, 2 months later)
#         imputed = First Tuesday of March 2020
#         (optionally adjusted relative to NFP release)

#     Args:
#         df: DataFrame with JOLTS data
#         ref_month_col: Column name for the reference month (default: 'date')
#         release_col: Column name for the release date (default: 'realtime_start')
#         nfp_offset_days: Optional median offset from NFP (for consistency enhancement)

#     Returns:
#         DataFrame with cleaned release dates
#     """
#     df = df.copy()

#     # Calculate deadline: end of 2nd month after reference month
#     deadline = df[ref_month_col] + pd.DateOffset(months=2) + pd.offsets.MonthEnd(0)

#     # Identify rows that need imputation (missing OR after deadline)
#     needs_imputation = df[release_col].isna() | (df[release_col] > deadline)

#     if needs_imputation.sum() > 0:
#         # Calculate first day of the 2nd month after reference
#         second_month_start = df[ref_month_col] + pd.DateOffset(months=2)
#         second_month_start = second_month_start.dt.to_period('M').dt.to_timestamp()

#         # Find first Tuesday of that month (base estimate)
#         # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
#         # Days to add to get to Tuesday: (1 - weekday) % 7
#         first_tuesday = second_month_start + pd.to_timedelta(
#             (1 - second_month_start.dt.weekday) % 7, unit='D'
#         )

#         # Apply NFP-relative adjustment if provided
#         # INT1: Uses apply_nfp_relative_adjustment imported at module level
#         if nfp_offset_days is not None:
#             # Apply adjustment row by row for imputed dates
#             adjusted_dates = []
#             for idx in df[needs_imputation].index:
#                 event_month = df.loc[idx, ref_month_col].replace(day=1)
#                 base_release = first_tuesday.loc[idx]

#                 adjusted = apply_nfp_relative_adjustment(
#                     event_month=event_month,
#                     base_release_date=base_release,
#                     median_offset_days=nfp_offset_days,
#                     use_adjustment=True
#                 )
#                 adjusted_dates.append(adjusted)

#             # Use adjusted dates
#             df.loc[needs_imputation, release_col] = adjusted_dates
#             logger.info(f"Imputed {needs_imputation.sum()} JOLTS release dates with NFP-relative adjustment")
#         else:
#             # Standard first Tuesday imputation
#             df.loc[needs_imputation, release_col] = first_tuesday[needs_imputation]
#             logger.info(f"Imputed {needs_imputation.sum()} JOLTS release dates to first Tuesday rule")

#     return df

def clean_weekly_release_dates(df, week_end_col='date', release_col='realtime_start', nfp_offset_days=None):
    """
    Clean and impute weekly data release dates.

    Logic: For each week ending date, if release date is missing or more than 14 days late,
    impute it as the Thursday immediately following the week end.
    
    ENHANCEMENT: If nfp_offset_days provided, apply NFP-relative adjustment
    (Note: Weekly claims are typically released promptly, so NFP adjustment rarely needed)

    Example:
        week_end = 2020-06-27 (Saturday)
        deadline = 2020-07-11 (14 days later)
        imputed = 2020-07-02 (next Thursday after Saturday)

    Args:
        df: DataFrame with weekly data
        week_end_col: Column name for the week ending date (default: 'date')
        release_col: Column name for the release date (default: 'realtime_start')
        nfp_offset_days: Optional median offset from NFP (rarely used for weekly data)

    Returns:
        DataFrame with cleaned release dates
    """
    df = df.copy()

    # Calculate deadline: 14 days after week end
    deadline = df[week_end_col] + pd.Timedelta(days=14)

    # Identify rows that need imputation (missing OR after deadline)
    needs_imputation = df[release_col].isna() | (df[release_col] > deadline)

    if needs_imputation.sum() > 0:
        # Calculate Thursday following week_end (base estimate)
        # weekday(): Monday=0, Tuesday=1, Wednesday=2, Thursday=3, ..., Sunday=6
        # Days until next Thursday: ((3 - weekday) % 7) or 7
        # If today is Thursday, we want next Thursday (7 days), not today (0 days)
        days_to_thursday = ((3 - df[week_end_col].dt.weekday) % 7).replace(0, 7)
        next_thursday = df[week_end_col] + pd.to_timedelta(days_to_thursday, unit='D')

        # Note: NFP adjustment for weekly claims is typically not beneficial
        # since they're released at consistent weekly intervals (every Thursday)
        # The adjustment is included for completeness but rarely changes dates
        
        # Apply imputation (with optional NFP adjustment)
        df.loc[needs_imputation, release_col] = next_thursday[needs_imputation]
        logger.info(f"Imputed {needs_imputation.sum()} weekly release dates to next Thursday rule")

    return df

def load_nfp_release_schedule():
    """
    Load NFP release dates for proper weekly data bucketing.

    OPTIMIZATION: Uses shared nfp_relative_timing module (cached).
    """
    try:
        nfp = load_nfp_releases()  # From nfp_relative_timing module (cached)
        # Rename columns to match expected format
        nfp = nfp.rename(columns={'ds': 'data_month', 'release_date': 'nfp_release_date'})
        return nfp.sort_values('nfp_release_date')
    except FileNotFoundError:
        logger.warning("NFP release schedule not found")
        return None

def aggregate_weekly_to_monthly_nfp_based(weekly_df, nfp_schedule):
    """
    Aggregate weekly data into monthly buckets based on NFP release windows.

    Logic: For target month M (e.g., June data released July 3):
    - Include weekly releases where: NFP_release(M-1) <= weekly_release < NFP_release(M)
    - Data released ON M-1 NFP day is included in M-1 bucket, not M bucket

    Args:
        weekly_df: DataFrame with columns ['date', 'value', 'realtime_start']
        nfp_schedule: DataFrame with columns ['data_month', 'nfp_release_date']

    Returns:
        DataFrame with monthly aggregated values
    """
    if weekly_df.empty or nfp_schedule is None:
        # Fallback: simple monthly resampling with 7-day lag
        logger.warning("Using fallback monthly aggregation (no NFP schedule)")
        weekly_df = weekly_df.sort_values('date').set_index('date')
        monthly = weekly_df['value'].resample('MS').mean().reset_index()
        monthly['release_date'] = monthly['date'] + pd.Timedelta(days=7)
        return monthly

    # Prepare weekly data with release dates
    weekly_clean = weekly_df[['date', 'value', 'realtime_start']].copy()
    weekly_clean = weekly_clean.sort_values('realtime_start')

    # Use earliest release per week (handle revisions)
    weekly_clean = weekly_clean.groupby('date').first().reset_index()
    weekly_clean.columns = ['week_ending', 'value', 'release_date']

    # Prepare NFP schedule for merging
    nfp_schedule = nfp_schedule.sort_values('nfp_release_date').copy()

    # Assign each weekly release to a target month using searchsorted
    # For each weekly release, find the FIRST NFP release that is >= weekly release
    # That NFP release defines the target month
    weekly_clean = weekly_clean.sort_values('release_date')

    # Find the target month for each weekly release
    idx = np.searchsorted(
        nfp_schedule['nfp_release_date'].values,
        weekly_clean['release_date'].values,
        side='left'  # Find first NFP >= weekly release
    )

    # Handle edge cases (releases after last NFP or before first NFP)
    idx = np.clip(idx, 0, len(nfp_schedule) - 1)

    # Assign target month
    weekly_clean['target_month'] = nfp_schedule.iloc[idx]['data_month'].values

    # Filter: Keep only releases where previous_NFP < release <= current_NFP
    # This means: release_date <= nfp_release_date[target_month]
    # AND: release_date > nfp_release_date[target_month - 1]

    valid_rows = []
    for target_month, group in weekly_clean.groupby('target_month'):
        # Find the NFP release for this target month
        current_nfp = nfp_schedule[nfp_schedule['data_month'] == target_month]['nfp_release_date']
        if current_nfp.empty:
            continue
        current_nfp_date = current_nfp.iloc[0]

        # Find the previous NFP release
        prev_nfp = nfp_schedule[nfp_schedule['nfp_release_date'] < current_nfp_date]
        if prev_nfp.empty:
            # First NFP in dataset - include all releases up to current
            prev_nfp_date = pd.Timestamp('1900-01-01')
        else:
            prev_nfp_date = prev_nfp.iloc[-1]['nfp_release_date']

        # Filter: prev_NFP <= release < current_NFP
        # Changed from (> prev & <= current) to (>= prev & < current)
        # This ensures data released ON M-1 NFP day is included in M-1 bucket
        valid = group[
            (group['release_date'] >= prev_nfp_date) &
            (group['release_date'] < current_nfp_date)
        ].copy()

        valid_rows.append(valid)

    if not valid_rows:
        return pd.DataFrame(columns=['date', 'value', 'release_date'])

    weekly_assigned = pd.concat(valid_rows, ignore_index=True)

    # Aggregate by target month
    monthly_agg = weekly_assigned.groupby('target_month').agg({
        'value': 'mean',
        'release_date': 'max'  # Use last release date in the window
    }).reset_index()

    monthly_agg.columns = ['date', 'value', 'release_date']

    return monthly_agg

def calculate_weekly_spike_stats(weekly_df, nfp_schedule):
    """
    Calculate maximum weekly spike and persistence metrics per NFP target month.

    This function identifies extreme spikes in weekly claims data (like COVID collapse)
    that would be hidden by monthly averages.

    Args:
        weekly_df: DataFrame with columns ['date', 'value', 'realtime_start']
        nfp_schedule: DataFrame with columns ['data_month', 'nfp_release_date']

    Returns:
        DataFrame with columns: date, max_value, weeks_above_p95, release_date
    """
    if weekly_df.empty or nfp_schedule is None:
        return pd.DataFrame(columns=['date', 'max_value', 'weeks_above_p95', 'release_date'])

    # Prepare weekly data
    weekly_clean = weekly_df[['date', 'value', 'realtime_start']].copy()
    weekly_clean = weekly_clean.sort_values('realtime_start')

    # Use earliest release per week (handle revisions)
    weekly_clean = weekly_clean.groupby('date').first().reset_index()
    weekly_clean.columns = ['week_ending', 'value', 'release_date']

    # Calculate historical 95th percentile (expanding window for proper vintaging)
    weekly_sorted = weekly_clean.sort_values('week_ending').copy()
    weekly_sorted['expanding_p95'] = weekly_sorted['value'].expanding(min_periods=52).quantile(0.95)

    # Merge back the p95 threshold
    weekly_clean = weekly_clean.merge(
        weekly_sorted[['week_ending', 'expanding_p95']],
        on='week_ending',
        how='left'
    )
    weekly_clean['threshold_p95'] = weekly_clean['expanding_p95']

    # Prepare NFP schedule for assignment
    nfp_schedule = nfp_schedule.sort_values('nfp_release_date').copy()
    weekly_clean = weekly_clean.sort_values('release_date')

    # Assign target months using searchsorted (same logic as aggregate_weekly_to_monthly_nfp_based)
    idx = np.searchsorted(
        nfp_schedule['nfp_release_date'].values,
        weekly_clean['release_date'].values,
        side='left'
    )
    idx = np.clip(idx, 0, len(nfp_schedule) - 1)
    weekly_clean['target_month'] = nfp_schedule.iloc[idx]['data_month'].values

    # Filter to valid window (same as aggregate function)
    valid_rows = []
    for target_month, group in weekly_clean.groupby('target_month'):
        current_nfp = nfp_schedule[nfp_schedule['data_month'] == target_month]['nfp_release_date']
        if current_nfp.empty:
            continue
        current_nfp_date = current_nfp.iloc[0]

        prev_nfp = nfp_schedule[nfp_schedule['nfp_release_date'] < current_nfp_date]
        prev_nfp_date = prev_nfp.iloc[-1]['nfp_release_date'] if not prev_nfp.empty else pd.Timestamp('1900-01-01')

        valid = group[
            (group['release_date'] >= prev_nfp_date) &
            (group['release_date'] < current_nfp_date)
        ].copy()

        valid_rows.append(valid)

    if not valid_rows:
        return pd.DataFrame(columns=['date', 'max_value', 'weeks_above_p95', 'release_date'])

    weekly_assigned = pd.concat(valid_rows, ignore_index=True)

    # Aggregate spike statistics
    spike_stats = weekly_assigned.groupby('target_month').agg({
        'value': 'max',  # Maximum weekly spike
        'release_date': 'max'
    }).reset_index()

    # Count weeks above 95th percentile
    weekly_assigned['is_high'] = weekly_assigned['value'] > weekly_assigned['threshold_p95']
    weeks_high = weekly_assigned.groupby('target_month')['is_high'].sum().reset_index()
    weeks_high.columns = ['target_month', 'weeks_above_p95']

    # Merge
    spike_stats = spike_stats.merge(weeks_high, on='target_month', how='left')
    spike_stats = spike_stats.rename(columns={'target_month': 'date', 'value': 'max_value'})
    spike_stats['weeks_above_p95'] = spike_stats['weeks_above_p95'].fillna(0)

    return spike_stats


# =============================================================================
# OPTIMIZATION: Pre-compute daily features for VIX, SP500, and other daily series
# These functions compute all rolling/derived features on the full history ONCE,
# rather than recomputing inside the snapshot loop (which was 400x slower)
# =============================================================================

def compute_vix_daily_features(df):
    """
    Pre-compute all VIX daily features on full history.

    Args:
        df: DataFrame with columns ['date', 'value', 'realtime_start']

    Returns:
        DataFrame with all computed daily features indexed by date
    """
    sub_df = df[['date', 'value']].copy().set_index('date').sort_index()

    # Daily change
    sub_df['daily_chg'] = sub_df['value'].diff()

    # Rolling 52-week high/low for regime detection
    sub_df['rolling_52w_high'] = sub_df['value'].rolling(window=252, min_periods=20).max()
    sub_df['rolling_52w_low'] = sub_df['value'].rolling(window=252, min_periods=20).min()

    # 30-day spike detection
    sub_df['vix_30d_ago'] = sub_df['value'].shift(21)
    sub_df['vix_spike_ratio'] = sub_df['value'] / sub_df['vix_30d_ago']

    # 5-day spike detection (rapid panic)
    sub_df['vix_5d_ago'] = sub_df['value'].shift(5)
    sub_df['vix_spike_5d'] = sub_df['value'] / sub_df['vix_5d_ago']

    # 12-month z-scores
    sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
    sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
    sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

    # 3-month z-scores
    sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
    sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
    sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

    return sub_df


def aggregate_vix_to_monthly(daily_df):
    """
    Aggregate pre-computed VIX daily features to monthly.

    Args:
        daily_df: DataFrame with pre-computed daily VIX features

    Returns:
        DataFrame in long format with monthly aggregated features
    """
    monthly_agg = daily_df.resample('MS').agg({
        'value': ['mean', 'max', lambda x: x.quantile(0.99)],
        'daily_chg': 'std',
        'vix_spike_ratio': 'max',
        'vix_spike_5d': 'max',
        'rolling_52w_high': 'last',
        'z_score_12m': ['mean', 'max', 'min'],
        'z_score_3m': ['mean', 'max', 'min']
    })

    # Flatten MultiIndex columns
    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['VIX_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['VIX_max'] = monthly_agg.get('value_max', np.nan)
    temp_df['VIX_volatility'] = monthly_agg.get('daily_chg_std', np.nan)
    temp_df['VIX_p99'] = monthly_agg.get('value_<lambda_0>', np.nan)
    temp_df['VIX_30d_spike'] = monthly_agg.get('vix_spike_ratio_max', np.nan)
    temp_df['VIX_max_5d_spike'] = monthly_agg.get('vix_spike_5d_max', np.nan)
    temp_df['VIX_zscore_12m_mean'] = monthly_agg.get('z_score_12m_mean', np.nan)
    temp_df['VIX_zscore_12m_max'] = monthly_agg.get('z_score_12m_max', np.nan)
    temp_df['VIX_zscore_12m_min'] = monthly_agg.get('z_score_12m_min', np.nan)
    temp_df['VIX_zscore_3m_mean'] = monthly_agg.get('z_score_3m_mean', np.nan)
    temp_df['VIX_zscore_3m_max'] = monthly_agg.get('z_score_3m_max', np.nan)
    temp_df['VIX_zscore_3m_min'] = monthly_agg.get('z_score_3m_min', np.nan)
    temp_df['VIX_panic_regime'] = (temp_df['VIX_max'] > 50).astype(int)
    temp_df['VIX_high_regime'] = (temp_df['VIX_max'] > 40).astype(int)

    result = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )
    result['release_date'] = result['date'] + pd.offsets.MonthEnd(0)

    return result


def compute_sp500_daily_features(df):
    """
    Pre-compute all SP500 daily features on full history.

    Args:
        df: DataFrame with columns ['date', 'value', 'realtime_start']

    Returns:
        DataFrame with all computed daily features indexed by date
    """
    sub_df = df[['date', 'value']].copy().set_index('date').sort_index()

    # Daily changes and returns
    sub_df['daily_chg'] = sub_df['value'].diff()
    sub_df['daily_return'] = sub_df['value'].pct_change()

    # Rolling 52-week high for drawdown
    sub_df['rolling_52w_high'] = sub_df['value'].rolling(window=252, min_periods=20).max()
    sub_df['drawdown'] = (sub_df['value'] - sub_df['rolling_52w_high']) / sub_df['rolling_52w_high'] * 100

    # 30-day performance
    sub_df['value_30d_ago'] = sub_df['value'].shift(21)
    sub_df['return_30d'] = (sub_df['value'] - sub_df['value_30d_ago']) / sub_df['value_30d_ago'] * 100

    # 5-day performance (rapid crash)
    sub_df['value_5d_ago'] = sub_df['value'].shift(5)
    sub_df['return_5d'] = (sub_df['value'] - sub_df['value_5d_ago']) / sub_df['value_5d_ago'] * 100

    # 21-day volatility
    sub_df['volatility_21d'] = sub_df['daily_return'].rolling(window=21, min_periods=10).std() * np.sqrt(252) * 100

    # Consecutive down days
    down_days = (sub_df['daily_return'] < 0).astype(int)
    sub_df['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

    # Circuit breaker days (>5% drop)
    sub_df['circuit_breaker_day'] = (sub_df['daily_return'] < -0.05).astype(int)

    # 12-month z-scores
    sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
    sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
    sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

    # 3-month z-scores
    sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
    sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
    sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

    return sub_df


def aggregate_sp500_to_monthly(daily_df):
    """
    Aggregate pre-computed SP500 daily features to monthly.

    Args:
        daily_df: DataFrame with pre-computed daily SP500 features

    Returns:
        DataFrame in long format with monthly aggregated features
    """
    monthly_agg = daily_df.resample('MS').agg({
        'value': ['first', 'last', 'min'],
        'drawdown': 'min',
        'return_30d': 'last',
        'return_5d': 'min',
        'volatility_21d': 'mean',
        'daily_return': ['std', 'min', 'max'],
        'consecutive_down': 'max',
        'circuit_breaker_day': 'sum',
        'z_score_12m': ['mean', 'max', 'min'],
        'z_score_3m': ['mean', 'max', 'min']
    })

    # Flatten MultiIndex columns
    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)

    # Monthly return
    first_val = monthly_agg.get('value_first')
    last_val = monthly_agg.get('value_last')
    if first_val is not None and last_val is not None:
        temp_df['SP500_monthly_return'] = ((last_val - first_val) / first_val * 100)
    else:
        temp_df['SP500_monthly_return'] = np.nan

    temp_df['SP500_30d_return'] = monthly_agg.get('return_30d_last', np.nan)
    temp_df['SP500_max_drawdown'] = monthly_agg.get('drawdown_min', np.nan)
    temp_df['SP500_volatility'] = monthly_agg.get('volatility_21d_mean', np.nan)
    temp_df['SP500_worst_day'] = monthly_agg.get('daily_return_min', np.nan) * 100 if monthly_agg.get('daily_return_min') is not None else np.nan
    temp_df['SP500_max_5d_drop'] = monthly_agg.get('return_5d_min', np.nan)
    temp_df['SP500_best_day'] = monthly_agg.get('daily_return_max', np.nan) * 100 if monthly_agg.get('daily_return_max') is not None else np.nan
    temp_df['SP500_consecutive_down_days'] = monthly_agg.get('consecutive_down_max', np.nan)
    temp_df['SP500_days_circuit_breaker'] = monthly_agg.get('circuit_breaker_day_sum', np.nan)
    temp_df['SP500_zscore_12m_mean'] = monthly_agg.get('z_score_12m_mean', np.nan)
    temp_df['SP500_zscore_12m_max'] = monthly_agg.get('z_score_12m_max', np.nan)
    temp_df['SP500_zscore_12m_min'] = monthly_agg.get('z_score_12m_min', np.nan)
    temp_df['SP500_zscore_3m_mean'] = monthly_agg.get('z_score_3m_mean', np.nan)
    temp_df['SP500_zscore_3m_max'] = monthly_agg.get('z_score_3m_max', np.nan)
    temp_df['SP500_zscore_3m_min'] = monthly_agg.get('z_score_3m_min', np.nan)

    # Regime indicators
    temp_df['SP500_bear_market'] = (temp_df['SP500_max_drawdown'] < -20).astype(int)
    temp_df['SP500_crash_month'] = (temp_df['SP500_monthly_return'] < -10).astype(int)
    worst_day = monthly_agg.get('daily_return_min')
    temp_df['SP500_circuit_breaker'] = (worst_day < -0.05).astype(int) if worst_day is not None else 0

    result = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )
    result['release_date'] = result['date'] + pd.offsets.MonthEnd(0)

    return result


def compute_credit_yield_daily_features(df, name):
    """
    Pre-compute daily features for Credit_Spreads and Yield_Curve.

    Args:
        df: DataFrame with columns ['date', 'value', 'realtime_start']
        name: Series name ('Credit_Spreads' or 'Yield_Curve')

    Returns:
        DataFrame with computed daily features indexed by date
    """
    sub_df = df[['date', 'value']].copy().set_index('date').sort_index()

    sub_df['daily_chg'] = sub_df['value'].diff()

    # Expanding z-score
    sub_df['expanding_mean'] = sub_df['value'].expanding(min_periods=30).mean()
    sub_df['expanding_std'] = sub_df['value'].expanding(min_periods=30).std()
    sub_df['z_score'] = (sub_df['value'] - sub_df['expanding_mean']) / sub_df['expanding_std']

    # Acceleration
    sub_df['acceleration'] = sub_df['daily_chg'].diff()

    # 12-month z-scores
    sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
    sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
    sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

    # 3-month z-scores
    sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
    sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
    sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

    sub_df['series_name'] = name  # Store for later reference

    return sub_df


def aggregate_credit_yield_to_monthly(daily_df, name):
    """
    Aggregate pre-computed Credit/Yield daily features to monthly.
    """
    monthly_agg = daily_df.resample('MS').agg({
        'value': ['mean', 'max'],
        'daily_chg': ['std', 'sum'],
        'z_score': 'max',
        'acceleration': ['mean', 'std'],
        'z_score_12m': ['mean', 'max', 'min'],
        'z_score_3m': ['mean', 'max', 'min']
    })

    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df[f'{name}_avg'] = monthly_agg.get('value_mean', np.nan)
    temp_df[f'{name}_max'] = monthly_agg.get('value_max', np.nan)
    temp_df[f'{name}_vol_of_changes'] = monthly_agg.get('daily_chg_std', np.nan)
    temp_df[f'{name}_monthly_chg'] = monthly_agg.get('daily_chg_sum', np.nan)
    temp_df[f'{name}_zscore_max'] = monthly_agg.get('z_score_max', np.nan)
    temp_df[f'{name}_acceleration'] = monthly_agg.get('acceleration_mean', np.nan)
    temp_df[f'{name}_accel_volatility'] = monthly_agg.get('acceleration_std', np.nan)
    temp_df[f'{name}_zscore_12m_mean'] = monthly_agg.get('z_score_12m_mean', np.nan)
    temp_df[f'{name}_zscore_12m_max'] = monthly_agg.get('z_score_12m_max', np.nan)
    temp_df[f'{name}_zscore_12m_min'] = monthly_agg.get('z_score_12m_min', np.nan)
    temp_df[f'{name}_zscore_3m_mean'] = monthly_agg.get('z_score_3m_mean', np.nan)
    temp_df[f'{name}_zscore_3m_max'] = monthly_agg.get('z_score_3m_max', np.nan)
    temp_df[f'{name}_zscore_3m_min'] = monthly_agg.get('z_score_3m_min', np.nan)

    result = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )
    result['release_date'] = result['date'] + pd.offsets.MonthEnd(0)

    return result


def compute_oil_daily_features(df):
    """
    Pre-compute daily features for Oil Prices.
    """
    sub_df = df[['date', 'value']].copy().set_index('date').sort_index()

    sub_df['daily_chg'] = sub_df['value'].diff()

    # 30-day crash detection
    sub_df['value_30d_ago'] = sub_df['value'].shift(21)
    sub_df['crash_30d_pct'] = ((sub_df['value'] - sub_df['value_30d_ago']) / sub_df['value_30d_ago'].abs()) * 100

    # Negative price indicator
    sub_df['is_negative'] = (sub_df['value'] < 0).astype(int)

    # Daily percentage change
    sub_df['daily_pct'] = sub_df['value'].pct_change() * 100

    # Expanding z-score
    sub_df['expanding_mean'] = sub_df['value'].expanding(min_periods=30).mean()
    sub_df['expanding_std'] = sub_df['value'].expanding(min_periods=30).std()
    sub_df['z_score'] = (sub_df['value'] - sub_df['expanding_mean']) / sub_df['expanding_std']

    # 12-month z-scores
    sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
    sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
    sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

    # 3-month z-scores
    sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
    sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
    sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

    return sub_df


def aggregate_oil_to_monthly(daily_df):
    """
    Aggregate pre-computed Oil daily features to monthly.
    """
    monthly_agg = daily_df.resample('MS').agg({
        'value': 'mean',
        'daily_chg': 'std',
        'crash_30d_pct': 'min',
        'daily_pct': 'min',
        'is_negative': ['max', 'sum'],
        'z_score': 'min',
        'z_score_12m': ['mean', 'max', 'min'],
        'z_score_3m': ['mean', 'max', 'min']
    })

    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['Oil_Prices_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['Oil_Prices_volatility'] = monthly_agg.get('daily_chg_std', np.nan)
    temp_df['Oil_Prices_30d_crash'] = monthly_agg.get('crash_30d_pct_min', np.nan)
    temp_df['Oil_Prices_went_negative'] = monthly_agg.get('is_negative_max', np.nan)
    temp_df['Oil_Prices_zscore_min'] = monthly_agg.get('z_score_min', np.nan)
    temp_df['Oil_worst_day_pct'] = monthly_agg.get('daily_pct_min', np.nan)
    temp_df['Oil_days_negative'] = monthly_agg.get('is_negative_sum', np.nan)
    temp_df['Oil_Prices_zscore_12m_mean'] = monthly_agg.get('z_score_12m_mean', np.nan)
    temp_df['Oil_Prices_zscore_12m_max'] = monthly_agg.get('z_score_12m_max', np.nan)
    temp_df['Oil_Prices_zscore_12m_min'] = monthly_agg.get('z_score_12m_min', np.nan)
    temp_df['Oil_Prices_zscore_3m_mean'] = monthly_agg.get('z_score_3m_mean', np.nan)
    temp_df['Oil_Prices_zscore_3m_max'] = monthly_agg.get('z_score_3m_max', np.nan)
    temp_df['Oil_Prices_zscore_3m_min'] = monthly_agg.get('z_score_3m_min', np.nan)

    result = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )
    result['release_date'] = result['date'] + pd.offsets.MonthEnd(0)

    return result


def fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE):
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not found.")
        return
    fred = Fred(api_key=FRED_API_KEY)

    # Load NFP schedule for weekly data aggregation AND snapshot alignment
    # OPTIMIZATION: Uses shared nfp_relative_timing module (cached)
    nfp_schedule = load_nfp_release_schedule()
    if nfp_schedule is None:
        logger.error("NFP release schedule not found. Cannot create snapshots.")
        return

    logger.info("Loaded NFP release schedule for weekly data aggregation and snapshot alignment")

    # OPTIMIZATION: Use shared utility for release map (cached)
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)

    logger.info(f"Creating {len(nfp_release_map)} snapshots aligned with NFP release dates")

    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"
    
    # 1. Fetch all history 
    history_cache = {}

    DAILY_SERIES = ["Credit_Spreads", "Yield_Curve", "Oil_Prices", "VIX", "SP500"]
    # Dropped JOLTS_Openings and JOLTS_Hires due to multicollinearity
    # JOLTS_SERIES = ["JOLTS_Quits", "JOLTS_Layoffs"]
    # Dropped ICSA and IURSA due to multicollinearity (kept CCSA only)
    CLAIMS_SERIES = ["CCSA"]

    for name, code in FRED_SERIES.items():
        try:
            logger.info(f"Fetching full revision history for {name} ({code})")

            # ------------------------------------------------------------------
            # 1) DAILY FINANCIAL DATA (NO REVISION LOGIC, KNOWN ON THE DAY)
            # ------------------------------------------------------------------
            if name in DAILY_SERIES:
                # V1: Use retry wrapper for FRED API calls
                series = fred_api_call_with_retry(fred.get_series, code)
                df = series.to_frame(name='value')
                df.index.name = 'date'
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
                # Assume no revisions: value known on its own observation date
                # FORCE 1-DAY LAG: Data from Day T is available on Day T+1
                df['realtime_start'] = df['date'] + pd.Timedelta(days=1)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # ------------------------------------------------------------------
            # 2) JOLTS MONTHLY SERIES (HEAVY REVISIONS, ~2-MONTH LAG)
            #    -> USE VINTAGES + 60-DAY SYNTHETIC LAG WHEN VINTAGE MISSING
            #    -> CLEAN RELEASE DATES: Impute to first Tuesday of 2nd month if missing/late
            # ------------------------------------------------------------------
            # elif name in JOLTS_SERIES:
            #     # V1: Use retry wrapper for FRED API calls
            #     vintage_df = fred_api_call_with_retry(fred.get_series_as_of_date, code, as_of_date='2100-01-01')
            #     vintage_df['date'] = pd.to_datetime(vintage_df['date'])
            #     vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
            #     vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

            #     current_series = fred_api_call_with_retry(fred.get_series, code)
            #     current_df = current_series.to_frame(name='value').reset_index()
            #     current_df.columns = ['date', 'value']
            #     current_df['date'] = pd.to_datetime(current_df['date'])
            #     current_df['value'] = pd.to_numeric(current_df['value'], errors='coerce')

            #     dates_with_vintage = set(vintage_df['date'].unique())
            #     earliest_vintage = vintage_df.groupby('date')['realtime_start'].min()

            #     # Identify dates where the first vintage appears "too late" (retroactive)
            #     # These should use synthetic 60-day lag instead of retroactive vintage
            #     late_start_dates = earliest_vintage[
            #         earliest_vintage > (earliest_vintage.index + pd.Timedelta(days=120))
            #     ].index
            #     late_start_set = set(late_start_dates)

            #     # Remove retroactive vintages from vintage_df
            #     # We'll replace them with synthetic first releases
            #     if late_start_set:
            #         logger.info(f"Replacing {len(late_start_set)} retroactive vintages with synthetic 60-day lag")
            #         df = vintage_df[~vintage_df['date'].isin(late_start_set)].copy()
            #     else:
            #         df = vintage_df.copy()

            #     # Add synthetic first releases for dates with late vintages
            #     # Also add truly missing dates
            #     missing_dates = (
            #         set(current_df['date']) -
            #         set(df['date'])  # Dates not in cleaned vintage_df
            #     )

            #     if missing_dates:
            #         missing_df = current_df[current_df['date'].isin(missing_dates)].copy()
            #         # JOLTS: synthetic realtime_start approx 2-month lag
            #         missing_df['realtime_start'] = missing_df['date'] + pd.Timedelta(days=60)
            #         df = pd.concat([df, missing_df], ignore_index=True)

            #     # Clean JOLTS release dates: impute to first Tuesday of 2nd month if missing/late
            #     df = clean_jolts_release_dates(df, ref_month_col='date', release_col='realtime_start')

            # ------------------------------------------------------------------
            # 3) WEEKLY CLAIMS SERIES (ICSA, CCSA, IURSA)
            #    -> USE VINTAGES BUT WITH SHORT LAG (~1 WEEK) FOR MISSING GAPS
            #    -> CLEAN RELEASE DATES: Impute to next Thursday if missing/late (>14 days)
            # ------------------------------------------------------------------
            elif name in CLAIMS_SERIES:
                # V1: Use retry wrapper for FRED API calls
                vintage_df = fred_api_call_with_retry(fred.get_series_as_of_date, code, as_of_date='2100-01-01')
                vintage_df['date'] = pd.to_datetime(vintage_df['date'])
                vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
                vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

                current_series = fred_api_call_with_retry(fred.get_series, code)
                current_df = current_series.to_frame(name='value').reset_index()
                current_df.columns = ['date', 'value']
                current_df['date'] = pd.to_datetime(current_df['date'])
                current_df['value'] = pd.to_numeric(current_df['value'], errors='coerce')

                dates_with_vintage = set(vintage_df['date'].unique())
                earliest_vintage = vintage_df.groupby('date')['realtime_start'].min()

                # For weekly claims, initial release should be very close to the observation date.
                # We still use a generous buffer to detect "truly late" starts only.
                late_start_dates = earliest_vintage[
                    earliest_vintage > (earliest_vintage.index + pd.Timedelta(days=30))
                ].index
                late_start_set = set(late_start_dates)

                # Remove retroactive vintages from vintage_df
                # We'll replace them with synthetic first releases
                if late_start_set:
                    logger.info(f"Replacing {len(late_start_set)} retroactive weekly vintages with synthetic 7-day lag")
                    df = vintage_df[~vintage_df['date'].isin(late_start_set)].copy()
                else:
                    df = vintage_df.copy()

                # Add synthetic first releases for dates with late vintages
                # Also add truly missing dates
                missing_dates = (
                    set(current_df['date']) -
                    set(df['date'])  # Dates not in cleaned vintage_df
                )

                if missing_dates:
                    missing_df = current_df[current_df['date'].isin(missing_dates)].copy()
                    # LESS PESSIMISTIC: approximate weekly claims lag as 7 days
                    missing_df['realtime_start'] = missing_df['date'] + pd.Timedelta(days=7)
                    df = pd.concat([df, missing_df], ignore_index=True)

                # Clean weekly release dates: impute to next Thursday if missing/late (>14 days)
                df = clean_weekly_release_dates(df, week_end_col='date', release_col='realtime_start')

            else:
                # Fallback (shouldn't really hit given how we've partitioned)
                # V1: Use retry wrapper for FRED API calls
                series = fred_api_call_with_retry(fred.get_series, code)
                df = series.to_frame(name='value')
                df.index.name = 'date'
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
                df['realtime_start'] = df['date']
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

            history_cache[name] = df

        except Exception as e:
            logger.error(f"Error fetching history for {name}: {e}")

            
    # 2. Generate Snapshots aligned with NFP release dates
    # OPTIMIZATION: Batch check existing snapshots to avoid 400+ filesystem calls
    existing_snapshots = set()
    for decade_dir in base_dir.glob("*s"):
        for year_dir in decade_dir.glob("*"):
            if year_dir.is_dir():
                for parquet_file in year_dir.glob("*.parquet"):
                    # Extract YYYY-MM from filename
                    existing_snapshots.add(parquet_file.stem)

    logger.info(f"Found {len(existing_snapshots)} existing snapshots, will skip these")

    # =============================================================================
    # OPTIMIZATION [C2]: Pre-compute daily features ONCE before the snapshot loop
    # This avoids recomputing 252-day rolling windows 400+ times (10 min → 5 min)
    # =============================================================================
    daily_features_cache = {}
    DAILY_PRECOMPUTE = ["VIX", "SP500", "Credit_Spreads", "Yield_Curve", "Oil_Prices"]

    for name in DAILY_PRECOMPUTE:
        if name not in history_cache:
            continue

        df = history_cache[name]
        logger.info(f"Pre-computing daily features for {name}...")

        if name == "VIX":
            daily_features_cache[name] = compute_vix_daily_features(df)
        elif name == "SP500":
            daily_features_cache[name] = compute_sp500_daily_features(df)
        elif name in ["Credit_Spreads", "Yield_Curve"]:
            daily_features_cache[name] = compute_credit_yield_daily_features(df, name)
        elif name == "Oil_Prices":
            daily_features_cache[name] = compute_oil_daily_features(df)

    logger.info(f"Pre-computed daily features for {len(daily_features_cache)} series")

    for obs_month, snap_date in nfp_release_map.items():
        snap_date = pd.Timestamp(snap_date)
        obs_month = pd.Timestamp(obs_month)
        month_str = obs_month.strftime('%Y-%m')

        # OPTIMIZATION: Check against pre-built set instead of filesystem call
        if month_str in existing_snapshots:
            continue

        # OPTIMIZATION: Use shared utility for snapshot path
        save_path = get_snapshot_path(base_dir, obs_month)

        logger.info(f"Generating snapshot for {month_str} (NFP release: {snap_date.date()})")
        
        snap_data_list = []
        
        for name, df in history_cache.items():
            # Changed from <= to strict < to prevent same-day data leakage
            valid = df[df['realtime_start'] < snap_date].copy()
            if valid.empty:
                continue
                
            valid = valid.sort_values(['date', 'realtime_start'])
            latest = valid.drop_duplicates(subset=['date'], keep='last')
            sub_df = latest[['date', 'value']].set_index('date').sort_index()
            
            # --- OPTIMIZED TRANSFORMATION LOGIC (C2) ---
            # Use pre-computed daily features instead of recomputing inside the loop

            if name == "VIX":
                # OPTIMIZATION: Use pre-computed daily features, filter by snapshot, aggregate
                if name in daily_features_cache:
                    precomputed = daily_features_cache[name]
                    # Filter to data available before snapshot (strict <)
                    valid_features = precomputed[precomputed.index < snap_date]
                    if not valid_features.empty:
                        sub_df = aggregate_vix_to_monthly(valid_features)
                    else:
                        continue
                else:
                    continue

            elif name == "SP500":
                # OPTIMIZATION: Use pre-computed daily features, filter by snapshot, aggregate
                if name in daily_features_cache:
                    precomputed = daily_features_cache[name]
                    # Filter to data available before snapshot (strict <)
                    valid_features = precomputed[precomputed.index < snap_date]
                    if not valid_features.empty:
                        sub_df = aggregate_sp500_to_monthly(valid_features)
                    else:
                        continue
                else:
                    continue

            elif name in ["Credit_Spreads", "Yield_Curve"]:
                # OPTIMIZATION: Use pre-computed daily features, filter by snapshot, aggregate
                if name in daily_features_cache:
                    precomputed = daily_features_cache[name]
                    # Filter to data available before snapshot (strict <)
                    valid_features = precomputed[precomputed.index < snap_date]
                    if not valid_features.empty:
                        sub_df = aggregate_credit_yield_to_monthly(valid_features, name)
                    else:
                        continue
                else:
                    continue

            elif name == "Oil_Prices":
                # OPTIMIZATION: Use pre-computed daily features, filter by snapshot, aggregate
                if name in daily_features_cache:
                    precomputed = daily_features_cache[name]
                    # Filter to data available before snapshot (strict <)
                    valid_features = precomputed[precomputed.index < snap_date]
                    if not valid_features.empty:
                        sub_df = aggregate_oil_to_monthly(valid_features)
                    else:
                        continue
                else:
                    continue

            elif name in ["ICSA", "CCSA", "IURSA", "Financial_Stress", "Weekly_Econ_Index"]:
                # NFP-BASED AGGREGATION: Bucket weekly data by NFP release windows
                # This ensures we only use data that would have been available before each NFP

                # Prepare data for aggregation function
                weekly_data = latest[['date', 'value', 'realtime_start']].copy()

                # Get monthly average (existing)
                monthly_avg = aggregate_weekly_to_monthly_nfp_based(weekly_data, nfp_schedule)

                if monthly_avg.empty:
                    logger.warning(f"No monthly data generated for {name}")
                    continue

                # For claims data, calculate spike statistics
                if name in ["ICSA", "CCSA", "IURSA"]:
                    # NEW: Calculate spike statistics per target month
                    monthly_spike_stats = calculate_weekly_spike_stats(weekly_data, nfp_schedule)

                    # Create multiple series for this claims data
                    avg_series = monthly_avg.copy()
                    avg_series['series_name'] = f"{name}_monthly_avg"

                    # Create spike series
                    spike_series = monthly_spike_stats.copy()
                    spike_series = spike_series.rename(columns={'max_value': 'value'})
                    spike_series['series_name'] = f"{name}_max_spike"

                    # Create weeks_high series
                    weeks_high_series = monthly_spike_stats[['date', 'weeks_above_p95', 'release_date']].copy()
                    weeks_high_series = weeks_high_series.rename(columns={'weeks_above_p95': 'value'})
                    weeks_high_series['series_name'] = f"{name}_weeks_high"

                    # Combine all series for this claims indicator
                    sub_df = pd.concat([avg_series, spike_series, weeks_high_series], ignore_index=True)
                    logger.info(f"Calculated NFP-based features for {name}: monthly_avg, max_spike, weeks_high")
                else:
                    # For Financial_Stress and Weekly_Econ_Index, just use monthly average
                    monthly_avg['series_name'] = f"{name}_monthly_avg"
                    sub_df = monthly_avg.copy()
                    logger.info(f"Calculated NFP-based monthly average for {name}")
                
            else:
                # Regional Fed and other monthly data (JOLTS commented out)
                sub_df = sub_df.resample('MS').last().reset_index()
                sub_df['series_name'] = name

                # Monthly Release Logic
                if not latest.empty and 'realtime_start' in latest.columns:
                    latest_with_month = latest.copy()
                    latest_with_month['obs_month'] = latest_with_month['date'].dt.to_period('M').dt.to_timestamp()
                    month_release_map = (
                        latest_with_month.groupby('obs_month')['realtime_start']
                        .min()
                        .to_dict()
                    )
                    sub_df['release_date'] = sub_df['date'].map(month_release_map)
                    
                    missing_mask = sub_df['release_date'].isna()
                    if missing_mask.any():
                        sub_df.loc[missing_mask, 'release_date'] = (
                            sub_df.loc[missing_mask, 'date'] + pd.DateOffset(months=2)
                        )
                else:
                    sub_df['release_date'] = sub_df['date'] + pd.DateOffset(months=2)

            sub_df['series_code'] = FRED_SERIES[name]
            sub_df['snapshot_date'] = snap_date
            
            snap_data_list.append(sub_df)
            
        if snap_data_list:
            full_snap = pd.concat(snap_data_list, ignore_index=True)
            full_snap['date'] = full_snap['date'].dt.to_period('M').dt.to_timestamp()

            # CRITICAL: Filter out data not yet released at snapshot time
            # This prevents lookahead bias from including monthly aggregates
            # Changed from <= to strict < to prevent same-day data leakage
            # Data released ON the snapshot date cannot be used for prediction
            full_snap['release_date'] = pd.to_datetime(full_snap['release_date'])
            before_count = len(full_snap)

            # Identify rows that will be filtered
            filtered_mask = full_snap['release_date'] >= snap_date
            filtered_rows = full_snap[filtered_mask]

            full_snap = full_snap[~filtered_mask].copy()
            after_count = len(full_snap)

            # SMART LOGGING: Only warn about unexpected filtering (past months)
            # Filtering current/future month data is expected behavior
            if before_count > after_count:
                filtered_count = before_count - after_count
                # Check if filtered rows are from expected months (current or future)
                filtered_dates = filtered_rows['date'].unique()
                unexpected_months = [d for d in filtered_dates if pd.Timestamp(d) < obs_month]

                if unexpected_months:
                    # This is unexpected - past month data being filtered
                    logger.warning(f"Unexpected filtering: {len(unexpected_months)} past months filtered out")
                    logger.warning(f"  Filtered months: {[pd.Timestamp(d).strftime('%Y-%m') for d in unexpected_months]}")
                else:
                    # Expected behavior - current/future month data filtered
                    logger.debug(f"Filtered {filtered_count} expected current/future-month rows")

            full_snap.to_parquet(save_path)
            logger.info(f"Saved snapshot to {save_path}")

if __name__ == "__main__":
    logger.info(f"Fetching FRED exogenous data from {START_DATE} to {END_DATE}")
    fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE)