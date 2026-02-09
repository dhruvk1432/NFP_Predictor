import pandas as pd
import numpy as np
from fredapi import Fred
import sys
from pathlib import Path
from datetime import timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files (Data_ETA_Pipeline/utils.py shadows utils/ package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import FRED_API_KEY, DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
# OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
# INT1: Import all NFP utilities at module level for consistency
from Data_ETA_Pipeline.fred_employment_pipeline import load_nfp_releases, get_nfp_release_map, apply_nfp_relative_adjustment
# OPTIMIZATION: Use shared utilities for snapshot path and MultiIndex flattening
from Data_ETA_Pipeline.utils import get_snapshot_path, flatten_multiindex_columns
from utils.transforms import add_symlog_copies, add_pct_change_copies, compute_all_features

logger = setup_logger(__file__, TEMP_DIR)


# =============================================================================
# V2: SP500 Data Fetching from Yahoo Finance (FRED only has 2016+ data)
# =============================================================================
def fetch_sp500_from_yahoo(start_date, end_date):
    """
    Fetch S&P 500 historical data from Yahoo Finance.

    Returns DataFrame with columns ['date', 'value'] to match FRED format.
    """
    logger.info(f"Fetching SP500 data from Yahoo Finance: {start_date} to {end_date}")

    try:
        data = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)

        if data.empty:
            raise ValueError("Yahoo Finance returned empty dataset for SP500")

        # Handle MultiIndex columns (yfinance returns MultiIndex for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take Close price
            data.columns = data.columns.droplevel(1)

        # Extract Close prices
        if 'Close' in data.columns:
            close_series = data['Close']
        else:
            raise ValueError("'Close' column not found in Yahoo Finance data")

        # Create DataFrame with proper format
        df = pd.DataFrame({
            'date': close_series.index,
            'value': close_series.values
        })

        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove any NaN values
        df = df.dropna(subset=['value'])

        if df.empty:
            raise ValueError("All SP500 values were NaN after processing")

        logger.info(f"Successfully fetched {len(df)} days of SP500 data from Yahoo Finance")
        return df

    except Exception as e:
        logger.error(f"CRITICAL: Failed to fetch SP500 from Yahoo Finance: {e}")
        raise RuntimeError(f"Cannot proceed without SP500 data: {e}")

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

# Binary/regime indicator features - these are 0/1 flags that should NOT be differenced
BINARY_REGIME_FEATURES = frozenset({
    'VIX_panic_regime',
    'VIX_high_regime',
    'SP500_bear_market',
    'SP500_crash_month',
    'SP500_circuit_breaker',
})

FRED_SERIES = {
    "Credit_Spreads": "BAMLH0A0HYM2",
    "Yield_Curve": "T10Y2Y",
    "Oil_Prices": "DCOILWTICO",
    "VIX": "VIXCLS",  
    "SP500": "SP500", 
    "Financial_Stress": "STLFSI4",  
    "Weekly_Econ_Index": "WEI", 
    "CCNSA": "CCNSA",  
    "CCSA": "CCSA", 
}

# =============================================================================
# Thread-safe rate limiting for parallel FRED API requests
# =============================================================================
_request_lock = threading.Lock()
_last_request_time = [0.0]

def _rate_limited_fetch(fetch_func, *args, per_request_delay=0.8, max_retries=3, **kwargs):
    """
    Execute a FRED API fetch with thread-safe rate limiting and retry logic.

    Args:
        fetch_func: The function to call for fetching data
        *args: Positional arguments for fetch_func
        per_request_delay: Minimum delay between requests (seconds)
        max_retries: Maximum retry attempts for rate limit errors
        **kwargs: Keyword arguments for fetch_func

    Returns:
        Result of fetch_func
    """
    # Enforce rate limiting across all threads
    with _request_lock:
        elapsed = time.time() - _last_request_time[0]
        if elapsed < per_request_delay:
            time.sleep(per_request_delay - elapsed)
        _last_request_time[0] = time.time()

    # Execute with retry logic
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            is_rate_limit = '429' in str(e) or 'rate limit' in error_str or 'too many requests' in error_str
            is_transient = 'timeout' in error_str or 'connection' in error_str or '503' in str(e)

            if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                delay = 2.0 * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(f"FRED API error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                raise

    raise last_exception

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
        logger.error("NFP release schedule not found")
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

def aggregate_weekly_to_monthly_nfp_based_custom(weekly_df, nfp_schedule, agg_func='mean'):
    """
    Aggregate weekly data into monthly buckets based on NFP release windows with custom aggregation function.

    Logic: For target month M (e.g., June data released July 3):
    - Include weekly releases where: NFP_release(M-1) <= weekly_release < NFP_release(M)
    - Data released ON M-1 NFP day is included in M-1 bucket, not M bucket

    Args:
        weekly_df: DataFrame with columns ['date', 'value', 'realtime_start']
        nfp_schedule: DataFrame with columns ['data_month', 'nfp_release_date']
        agg_func: Aggregation function ('mean', 'max', 'min', etc.)

    Returns:
        DataFrame with monthly aggregated values
    """
    if weekly_df.empty or nfp_schedule is None:
        # Fallback: simple monthly resampling with 7-day lag
        weekly_df = weekly_df.sort_values('date').set_index('date')
        monthly = weekly_df['value'].resample('MS').agg(agg_func).reset_index()
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

    # Aggregate by target month with custom aggregation function
    monthly_agg = weekly_assigned.groupby('target_month').agg({
        'value': agg_func,
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

    # 30-day spike detection
    sub_df['vix_30d_ago'] = sub_df['value'].shift(21)
    sub_df['vix_spike_ratio'] = sub_df['value'] / sub_df['vix_30d_ago']

    # 5-day spike detection (rapid panic)
    sub_df['vix_5d_ago'] = sub_df['value'].shift(5)
    sub_df['vix_spike_5d'] = sub_df['value'] / sub_df['vix_5d_ago']

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
        'value': ['mean', 'max'],
        'daily_chg': 'std',
        'vix_spike_ratio': 'max',
        'vix_spike_5d': 'max',
    })

    # Flatten MultiIndex columns
    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['VIX_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['VIX_max'] = monthly_agg.get('value_max', np.nan)
    temp_df['VIX_volatility'] = monthly_agg.get('daily_chg_std', np.nan)
    temp_df['VIX_30d_spike'] = monthly_agg.get('vix_spike_ratio_max', np.nan)
    temp_df['VIX_max_5d_spike'] = monthly_agg.get('vix_spike_5d_max', np.nan)
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

    # Daily percentage change
    sub_df['daily_pct'] = sub_df['value'].pct_change() * 100

    # Expanding z-score
    sub_df['expanding_mean'] = sub_df['value'].expanding(min_periods=30).mean()
    sub_df['expanding_std'] = sub_df['value'].expanding(min_periods=30).std()
    sub_df['z_score'] = (sub_df['value'] - sub_df['expanding_mean']) / sub_df['expanding_std']

    return sub_df


def aggregate_oil_to_monthly(daily_df):
    """
    Aggregate pre-computed Oil daily features to monthly.
    """
    monthly_agg = daily_df.resample('MS').agg({
        'value': ['mean'],
        'daily_chg': ['std'],
        'crash_30d_pct': ['min'],
        'daily_pct': ['min'],
        'z_score': ['min'],
    })

    monthly_agg = flatten_multiindex_columns(monthly_agg)

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['Oil_Prices_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['Oil_Prices_volatility'] = monthly_agg.get('daily_chg_std', np.nan)
    temp_df['Oil_Prices_30d_crash'] = monthly_agg.get('crash_30d_pct_min', np.nan)
    temp_df['Oil_Prices_zscore_min'] = monthly_agg.get('z_score_min', np.nan)
    temp_df['Oil_worst_day_pct'] = monthly_agg.get('daily_pct_min', np.nan)

    result = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )
    result['release_date'] = result['date'] + pd.offsets.MonthEnd(0)

    return result


def _fetch_single_series(fred, name, code, start_date, end_date, daily_series, claims_series):
    """
    Fetch a single FRED series with appropriate handling based on series type.

    Args:
        fred: FRED API client
        name: Series name (e.g., 'VIX', 'CCSA')
        code: FRED series code
        start_date: Start date for data fetch
        end_date: End date for data fetch
        daily_series: List of daily series names
        claims_series: List of weekly claims series names

    Returns:
        tuple: (name, DataFrame) or (name, None) on error
    """
    try:

        # ------------------------------------------------------------------
        # 1) DAILY FINANCIAL DATA (NO REVISION LOGIC, KNOWN ON THE DAY)
        # ------------------------------------------------------------------
        if name in daily_series:
            # V2: Special handling for SP500 - use Yahoo Finance ONLY (no FRED fallback)
            if name == "SP500":
                # Yahoo Finance is the only source - will raise error if it fails
                # Convert datetime to string if needed
                start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                df = fetch_sp500_from_yahoo(
                    start_date=start_str,
                    end_date=end_str
                )
            else:
                # Use rate-limited fetch for FRED API calls (other daily series)
                series = _rate_limited_fetch(fred.get_series, code)
                df = series.to_frame(name='value')
                df.index.name = 'date'
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Assume no revisions: value known on its own observation date
            # FORCE 1-DAY LAG: Data from Day T is available on Day T+1
            df['realtime_start'] = df['date'] + pd.Timedelta(days=1)

        # ------------------------------------------------------------------
        # 2) WEEKLY CLAIMS SERIES (CCSA)
        #    -> USE VINTAGES BUT WITH SHORT LAG (~1 WEEK) FOR MISSING GAPS
        #    -> CLEAN RELEASE DATES: Impute to next Thursday if missing/late (>14 days)
        # ------------------------------------------------------------------
        elif name in claims_series:
            # Use rate-limited fetch for FRED API calls
            vintage_df = _rate_limited_fetch(fred.get_series_as_of_date, code, as_of_date='2100-01-01')
            vintage_df['date'] = pd.to_datetime(vintage_df['date'])
            vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
            vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

            current_series = _rate_limited_fetch(fred.get_series, code)
            current_df = current_series.to_frame(name='value').reset_index()
            current_df.columns = ['date', 'value']
            current_df['date'] = pd.to_datetime(current_df['date'])
            current_df['value'] = pd.to_numeric(current_df['value'], errors='coerce')

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
            # Fallback for other series (Financial_Stress, Weekly_Econ_Index, Empire_State_Emp)
            # Use rate-limited fetch for FRED API calls
            series = _rate_limited_fetch(fred.get_series, code)
            df = series.to_frame(name='value')
            df.index.name = 'date'
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
            df['realtime_start'] = df['date']
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        return (name, df)

    except Exception as e:
        logger.error(f"Error fetching history for {name}: {e}")
        return (name, None)


def fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE, max_workers=3):
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not found")
        return
    fred = Fred(api_key=FRED_API_KEY)

    # Load NFP schedule for weekly data aggregation AND snapshot alignment
    # OPTIMIZATION: Uses shared nfp_relative_timing module (cached)
    nfp_schedule = load_nfp_release_schedule()
    if nfp_schedule is None:
        logger.error("NFP release schedule not found")
        return

    # OPTIMIZATION: Use shared utility for release map (cached)
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)

    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data"

    # Skip-if-exists: Check if all snapshots already exist
    existing_count = 0
    for obs_month in nfp_release_map.keys():
        snap_path = get_snapshot_path(base_dir, pd.Timestamp(obs_month))
        if snap_path.exists():
            existing_count += 1

    if existing_count == len(nfp_release_map):
        print(f"✓ FRED exogenous data already exists: {existing_count} monthly snapshots", flush=True)
        logger.info(f"FRED exogenous snapshots already exist, skipping")
        return

    # 1. Fetch all history with PARALLELIZATION
    logger.info("Starting FRED exogenous data download")
    history_cache = {}

    DAILY_SERIES = ["Credit_Spreads", "Yield_Curve", "Oil_Prices", "VIX", "SP500"]
    # Dropped JOLTS_Openings and JOLTS_Hires due to multicollinearity
    # JOLTS_SERIES = ["JOLTS_Quits", "JOLTS_Layoffs"]
    CLAIMS_SERIES = ["CCNSA", "CCSA"]

    # PARALLELIZATION: Fetch all series concurrently with rate limiting
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_name = {
            executor.submit(
                _fetch_single_series,
                fred, name, code, start_date, end_date, DAILY_SERIES, CLAIMS_SERIES
            ): name
            for name, code in FRED_SERIES.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result_name, df = future.result()
                if df is not None:
                    history_cache[result_name] = df
                else:
                    logger.warning(f"No data for {name}")
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")


    # 2. Generate Snapshots aligned with NFP release dates
    # OPTIMIZATION: Batch check existing snapshots to avoid 400+ filesystem calls
    existing_snapshots = set()
    for decade_dir in base_dir.glob("*s"):
        for year_dir in decade_dir.glob("*"):
            if year_dir.is_dir():
                for parquet_file in year_dir.glob("*.parquet"):
                    # Extract YYYY-MM from filename
                    existing_snapshots.add(parquet_file.stem)

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

        if name == "VIX":
            daily_features_cache[name] = compute_vix_daily_features(df)
        elif name == "SP500":
            daily_features_cache[name] = compute_sp500_daily_features(df)
        elif name in ["Credit_Spreads", "Yield_Curve"]:
            daily_features_cache[name] = compute_credit_yield_daily_features(df, name)
        elif name == "Oil_Prices":
            daily_features_cache[name] = compute_oil_daily_features(df)

    for obs_month, snap_date in nfp_release_map.items():
        snap_date = pd.Timestamp(snap_date)
        obs_month = pd.Timestamp(obs_month)
        month_str = obs_month.strftime('%Y-%m')

        # OPTIMIZATION: Check against pre-built set instead of filesystem call
        if month_str in existing_snapshots:
            continue

        # OPTIMIZATION: Use shared utility for snapshot path
        save_path = get_snapshot_path(base_dir, obs_month)

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
                    # For daily data, realtime_start = date + 1 day, so data from date D is available on D+1
                    # Therefore, we need date < snap_date - 1 day to ensure realtime_start < snap_date
                    cutoff_date = snap_date - pd.Timedelta(days=1)
                    valid_features = precomputed[precomputed.index < cutoff_date]
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
                    # For daily data, realtime_start = date + 1 day, so data from date D is available on D+1
                    # Therefore, we need date < snap_date - 1 day to ensure realtime_start < snap_date
                    cutoff_date = snap_date - pd.Timedelta(days=1)
                    valid_features = precomputed[precomputed.index < cutoff_date]
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
                    # For daily data, realtime_start = date + 1 day, so data from date D is available on D+1
                    # Therefore, we need date < snap_date - 1 day to ensure realtime_start < snap_date
                    cutoff_date = snap_date - pd.Timedelta(days=1)
                    valid_features = precomputed[precomputed.index < cutoff_date]
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
                    # For daily data, realtime_start = date + 1 day, so data from date D is available on D+1
                    # Therefore, we need date < snap_date - 1 day to ensure realtime_start < snap_date
                    cutoff_date = snap_date - pd.Timedelta(days=1)
                    valid_features = precomputed[precomputed.index < cutoff_date]
                    if not valid_features.empty:
                        sub_df = aggregate_oil_to_monthly(valid_features)
                    else:
                        continue
                else:
                    continue

            elif name in ["CCNSA", "CCSA", "Financial_Stress", "Weekly_Econ_Index"]:
                # NFP-BASED AGGREGATION: Bucket weekly data by NFP release windows
                # This ensures we only use data that would have been available before each NFP

                # Prepare data for aggregation function
                weekly_data = latest[['date', 'value', 'realtime_start']].copy()

                # Get monthly average (existing)
                monthly_avg = aggregate_weekly_to_monthly_nfp_based(weekly_data, nfp_schedule)

                if monthly_avg.empty:
                    continue

                # For claims data, calculate spike statistics
                if name in ["CCSA", "CCNSA"]:
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
                    # Data quality guard: drop first 2 months for weeks_high series
                    weeks_high_series = weeks_high_series.sort_values('date').iloc[2:].copy()

                    # Combine all series for this claims indicator
                    sub_df = pd.concat([avg_series, spike_series, weeks_high_series], ignore_index=True)
                elif name == "Weekly_Econ_Index":
                    # For Weekly_Econ_Index, calculate monthly average, max, and min
                    # Get monthly max
                    monthly_max = aggregate_weekly_to_monthly_nfp_based_custom(weekly_data, nfp_schedule, agg_func='max')
                    # Get monthly min
                    monthly_min = aggregate_weekly_to_monthly_nfp_based_custom(weekly_data, nfp_schedule, agg_func='min')

                    # Create multiple series
                    avg_series = monthly_avg.copy()
                    avg_series['series_name'] = f"{name}_monthly_avg"

                    max_series = monthly_max.copy()
                    max_series['series_name'] = f"{name}_monthly_max"

                    min_series = monthly_min.copy()
                    min_series['series_name'] = f"{name}_monthly_min"

                    # Combine all series for Weekly_Econ_Index
                    sub_df = pd.concat([avg_series, max_series, min_series], ignore_index=True)
                else:
                    # For Financial_Stress, just use monthly average
                    monthly_avg['series_name'] = f"{name}_monthly_avg"
                    sub_df = monthly_avg.copy()
                
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

            # Branch-and-Expand: create 3 base variants, then compute all features
            full_snap = add_symlog_copies(full_snap, skip_series=BINARY_REGIME_FEATURES)
            full_snap = add_pct_change_copies(full_snap, skip_series=BINARY_REGIME_FEATURES)
            full_snap = compute_all_features(full_snap, skip_series=BINARY_REGIME_FEATURES)

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
                    logger.warning(f"Unexpected: {len(unexpected_months)} past months filtered")

            full_snap.to_parquet(save_path)

        if obs_month.month == 12:
            logger.info(f"Saved {obs_month.year} snapshots")

    logger.info("✓ FRED exogenous data download complete")

if __name__ == "__main__":
    fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE)
