import pandas as pd
import numpy as np
from fredapi import Fred
import sys
from pathlib import Path
from datetime import timedelta

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import FRED_API_KEY, DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE

logger = setup_logger(__file__, TEMP_DIR)

FRED_SERIES = {
    #Daily Data - Financial Market Indicators
    "Credit_Spreads": "BAMLH0A0HYM2",
    "Yield_Curve": "T10Y3M",
    "Oil_Prices": "DCOILWTICO",
    "VIX": "VIXCLS",  # CBOE Volatility Index - Market fear gauge
    "SP500": "SP500",  # S&P 500 Index - Market crashes & recoveries
    # High-velocity economic indicators for extreme events
    "Financial_Stress": "STLFSI4",  # St. Louis Fed Financial Stress Index (weekly)
    "Weekly_Econ_Index": "WEI",  # Weekly Economic Index (real-time)
    #Monthly Data (JOLTS_Openings and JOLTS_Hires dropped due to multicollinearity)
    "JOLTS_Openings": "JTSJOL",  # DROPPED
    "JOLTS_Hires": "JTSHIL",  # DROPPED
    "JOLTS_Quits": "JTSQUR",
    "JOLTS_Layoffs": "JTSLDL",
    # NEW: Regional Fed Employment Indices (monthly)
    "Empire_State_Emp": "USEPUINDX",  # Empire State Manufacturing Employment Index
    "Philly_Fed_Emp": "USPHCICH",  # Philadelphia Fed Employment Diffusion Index
    # Weekly Jobless Claims (ICSA and IURSA dropped due to multicollinearity)
    # only data available before each NFP report is included in that month's features
    "ICSA": "ICSA", #Initial Claims Seasonally Adjusted
    "CCSA": "CCSA",  # Continued Claims Seasonally Adjusted (KEPT)
    "IURSA": "IURSA" # Insured Unemployment Rate Seasonally Adjusted
}

def clean_jolts_release_dates(df, ref_month_col='date', release_col='realtime_start', nfp_offset_days=None):
    """
    Clean and impute JOLTS release dates.

    Logic: For each observation month, if release date is missing or more than 2 months late,
    impute it as the first Tuesday of the 2nd month after the observation.
    
    ENHANCEMENT: If nfp_offset_days provided, apply NFP-relative adjustment to maintain
    historical timing consistency relative to NFP releases.

    Example:
        ref_month = 2020-01-01 (January)
        deadline = 2020-03-31 (end of March, 2 months later)
        imputed = First Tuesday of March 2020
        (optionally adjusted relative to NFP release)

    Args:
        df: DataFrame with JOLTS data
        ref_month_col: Column name for the reference month (default: 'date')
        release_col: Column name for the release date (default: 'realtime_start')
        nfp_offset_days: Optional median offset from NFP (for consistency enhancement)

    Returns:
        DataFrame with cleaned release dates
    """
    df = df.copy()

    # Calculate deadline: end of 2nd month after reference month
    deadline = df[ref_month_col] + pd.DateOffset(months=2) + pd.offsets.MonthEnd(0)

    # Identify rows that need imputation (missing OR after deadline)
    needs_imputation = df[release_col].isna() | (df[release_col] > deadline)

    if needs_imputation.sum() > 0:
        # Calculate first day of the 2nd month after reference
        second_month_start = df[ref_month_col] + pd.DateOffset(months=2)
        second_month_start = second_month_start.dt.to_period('M').dt.to_timestamp()

        # Find first Tuesday of that month (base estimate)
        # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
        # Days to add to get to Tuesday: (1 - weekday) % 7
        first_tuesday = second_month_start + pd.to_timedelta(
            (1 - second_month_start.dt.weekday) % 7, unit='D'
        )
        
        # Apply NFP-relative adjustment if provided
        if nfp_offset_days is not None:
            try:
                from nfp_relative_timing import apply_nfp_relative_adjustment
                
                # Apply adjustment row by row for imputed dates
                adjusted_dates = []
                for idx in df[needs_imputation].index:
                    event_month = df.loc[idx, ref_month_col].replace(day=1)
                    base_release = first_tuesday.loc[idx]
                    
                    adjusted = apply_nfp_relative_adjustment(
                        event_month=event_month,
                        base_release_date=base_release,
                        median_offset_days=nfp_offset_days,
                        use_adjustment=True
                    )
                    adjusted_dates.append(adjusted)
                
                # Use adjusted dates
                df.loc[needs_imputation, release_col] = adjusted_dates
                logger.info(f"Imputed {needs_imputation.sum()} JOLTS release dates with NFP-relative adjustment")
            except Exception as e:
                # Fallback to first Tuesday if NFP adjustment fails
                df.loc[needs_imputation, release_col] = first_tuesday[needs_imputation]
                logger.warning(f"NFP adjustment failed, using first Tuesday: {e}")
                logger.info(f"Imputed {needs_imputation.sum()} JOLTS release dates to first Tuesday rule")
        else:
            # Standard first Tuesday imputation
            df.loc[needs_imputation, release_col] = first_tuesday[needs_imputation]
            logger.info(f"Imputed {needs_imputation.sum()} JOLTS release dates to first Tuesday rule")

    return df

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
    """Load NFP release dates for proper weekly data bucketing."""
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    if not nfp_path.exists():
        logger.warning(f"NFP release schedule not found at {nfp_path}")
        return None

    nfp = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp.columns = ['data_month', 'nfp_release_date']
    nfp['data_month'] = pd.to_datetime(nfp['data_month'])
    nfp['nfp_release_date'] = pd.to_datetime(nfp['nfp_release_date'])
    return nfp.sort_values('nfp_release_date')

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

def fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE):
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not found.")
        return
    fred = Fred(api_key=FRED_API_KEY)

    # Load NFP schedule for weekly data aggregation AND snapshot alignment
    nfp_schedule = load_nfp_release_schedule()
    if nfp_schedule is None:
        logger.error("NFP release schedule not found. Cannot create snapshots.")
        return

    logger.info("Loaded NFP release schedule for weekly data aggregation and snapshot alignment")

    # Define date range for snapshots using NFP release dates (not month-end)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter NFP schedule to requested date range
    nfp_df_filtered = nfp_schedule[
        (nfp_schedule['data_month'] >= start_dt) &
        (nfp_schedule['data_month'] <= end_dt)
    ].copy()

    # Create mapping: observation_month -> NFP_release_date
    nfp_release_map = dict(zip(nfp_df_filtered['data_month'], nfp_df_filtered['nfp_release_date']))

    logger.info(f"Creating {len(nfp_release_map)} snapshots aligned with NFP release dates")

    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"
    
    # 1. Fetch all history 
    history_cache = {}

    DAILY_SERIES = ["Credit_Spreads", "Yield_Curve", "Oil_Prices", "VIX", "SP500"]
    # Dropped JOLTS_Openings and JOLTS_Hires due to multicollinearity
    JOLTS_SERIES = ["JOLTS_Quits", "JOLTS_Layoffs"]
    # Dropped ICSA and IURSA due to multicollinearity (kept CCSA only)
    CLAIMS_SERIES = ["CCSA"]

    for name, code in FRED_SERIES.items():
        try:
            logger.info(f"Fetching full revision history for {name} ({code})")

            # ------------------------------------------------------------------
            # 1) DAILY FINANCIAL DATA (NO REVISION LOGIC, KNOWN ON THE DAY)
            # ------------------------------------------------------------------
            if name in DAILY_SERIES:
                series = fred.get_series(code)
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
            elif name in JOLTS_SERIES:
                vintage_df = fred.get_series_as_of_date(code, as_of_date='2100-01-01')
                vintage_df['date'] = pd.to_datetime(vintage_df['date'])
                vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
                vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

                current_series = fred.get_series(code)
                current_df = current_series.to_frame(name='value').reset_index()
                current_df.columns = ['date', 'value']
                current_df['date'] = pd.to_datetime(current_df['date'])
                current_df['value'] = pd.to_numeric(current_df['value'], errors='coerce')

                dates_with_vintage = set(vintage_df['date'].unique())
                earliest_vintage = vintage_df.groupby('date')['realtime_start'].min()

                # Identify dates where the first vintage appears "too late" (retroactive)
                # These should use synthetic 60-day lag instead of retroactive vintage
                late_start_dates = earliest_vintage[
                    earliest_vintage > (earliest_vintage.index + pd.Timedelta(days=120))
                ].index
                late_start_set = set(late_start_dates)

                # Remove retroactive vintages from vintage_df
                # We'll replace them with synthetic first releases
                if late_start_set:
                    logger.info(f"Replacing {len(late_start_set)} retroactive vintages with synthetic 60-day lag")
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
                    # JOLTS: synthetic realtime_start approx 2-month lag
                    missing_df['realtime_start'] = missing_df['date'] + pd.Timedelta(days=60)
                    df = pd.concat([df, missing_df], ignore_index=True)

                # Clean JOLTS release dates: impute to first Tuesday of 2nd month if missing/late
                df = clean_jolts_release_dates(df, ref_month_col='date', release_col='realtime_start')

            # ------------------------------------------------------------------
            # 3) WEEKLY CLAIMS SERIES (ICSA, CCSA, IURSA)
            #    -> USE VINTAGES BUT WITH SHORT LAG (~1 WEEK) FOR MISSING GAPS
            #    -> CLEAN RELEASE DATES: Impute to next Thursday if missing/late (>14 days)
            # ------------------------------------------------------------------
            elif name in CLAIMS_SERIES:
                vintage_df = fred.get_series_as_of_date(code, as_of_date='2100-01-01')
                vintage_df['date'] = pd.to_datetime(vintage_df['date'])
                vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
                vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

                current_series = fred.get_series(code)
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
                series = fred.get_series(code)
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
    for obs_month, snap_date in nfp_release_map.items():
        snap_date = pd.Timestamp(snap_date)
        obs_month = pd.Timestamp(obs_month)

        year_str = obs_month.strftime('%Y')
        month_str = obs_month.strftime('%Y-%m')
        decade_str = f"{obs_month.year // 10 * 10}s"

        save_dir = base_dir / decade_str / year_str
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{month_str}.parquet"

        if save_path.exists():
            logger.info(f"Snapshot already exists for {month_str}, skipping.")
            continue
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
            
            # --- START OF EDITED TRANSFORMATION LOGIC ---

            if name == "VIX":
                # VIX: Volatility Index - Market Fear Gauge
                # Key features for detecting crashes like COVID
                sub_df['daily_chg'] = sub_df['value'].diff()

                # Calculate rolling 52-week high/low for regime detection
                sub_df['rolling_52w_high'] = sub_df['value'].rolling(window=252, min_periods=20).max()
                sub_df['rolling_52w_low'] = sub_df['value'].rolling(window=252, min_periods=20).min()

                # 30-day spike detection (VIX >3x in 30 days = extreme event)
                sub_df['vix_30d_ago'] = sub_df['value'].shift(21)  # ~30 trading days
                sub_df['vix_spike_ratio'] = sub_df['value'] / sub_df['vix_30d_ago']

                # NEW: 5-day spike detection (rapid panic - COVID went 17→82 in 5 days)
                sub_df['vix_5d_ago'] = sub_df['value'].shift(5)
                sub_df['vix_spike_5d'] = sub_df['value'] / sub_df['vix_5d_ago']

                # NEW: Calculate daily z-scores (standard deviations from 12-month rolling mean)
                sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
                sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
                sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

                # NEW: Calculate 3-month acceleration z-scores (63 trading days)
                sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
                sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
                sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

                monthly_agg = sub_df.resample('MS').agg({
                    'value': ['mean', 'max', lambda x: x.quantile(0.99)],  # Average, Peak, 99th percentile
                    'daily_chg': 'std',  # Volatility of volatility
                    'vix_spike_ratio': 'max',  # Largest 30-day spike in month
                    'vix_spike_5d': 'max',  # NEW: Largest 5-day spike in month (rapid panic)
                    'rolling_52w_high': 'last',  # Reference point for regime
                    'z_score_12m': ['mean', 'max', 'min'],  # Regime detection via 12m z-scores
                    'z_score_3m': ['mean', 'max', 'min']  # NEW: Acceleration via 3m z-scores
                })

                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                # Handle potential column name variations after MultiIndex flattening
                mean_col = 'value_mean' if 'value_mean' in monthly_agg.columns else 'value'
                max_col = 'value_max' if 'value_max' in monthly_agg.columns else 'max'
                vol_col = 'daily_chg_std' if 'daily_chg_std' in monthly_agg.columns else 'std'
                spike_col = 'vix_spike_ratio_max' if 'vix_spike_ratio_max' in monthly_agg.columns else 'max'

                temp_df['VIX_mean'] = monthly_agg[mean_col] if mean_col in monthly_agg.columns else np.nan
                temp_df['VIX_max'] = monthly_agg[max_col] if max_col in monthly_agg.columns else np.nan
                temp_df['VIX_volatility'] = monthly_agg[vol_col] if vol_col in monthly_agg.columns else np.nan

                # NEW: 99th percentile (tail risk)
                p99_col = 'value_<lambda_0>' if 'value_<lambda_0>' in monthly_agg.columns else None
                temp_df['VIX_p99'] = monthly_agg[p99_col] if p99_col and p99_col in monthly_agg.columns else np.nan

                if spike_col in monthly_agg.columns:
                    temp_df['VIX_30d_spike'] = monthly_agg[spike_col]
                else:
                    temp_df['VIX_30d_spike'] = np.nan

                # NEW: 5-day spike ratio (rapid panic)
                spike_5d_col = 'vix_spike_5d_max' if 'vix_spike_5d_max' in monthly_agg.columns else None
                temp_df['VIX_max_5d_spike'] = monthly_agg[spike_5d_col] if spike_5d_col and spike_5d_col in monthly_agg.columns else np.nan

                # Z-score features (standard deviations from 12-month mean)
                zscore_mean_col = 'z_score_12m_mean' if 'z_score_12m_mean' in monthly_agg.columns else None
                zscore_max_col = 'z_score_12m_max' if 'z_score_12m_max' in monthly_agg.columns else None
                zscore_min_col = 'z_score_12m_min' if 'z_score_12m_min' in monthly_agg.columns else None

                temp_df['VIX_zscore_12m_mean'] = monthly_agg[zscore_mean_col] if zscore_mean_col and zscore_mean_col in monthly_agg.columns else np.nan
                temp_df['VIX_zscore_12m_max'] = monthly_agg[zscore_max_col] if zscore_max_col and zscore_max_col in monthly_agg.columns else np.nan
                temp_df['VIX_zscore_12m_min'] = monthly_agg[zscore_min_col] if zscore_min_col and zscore_min_col in monthly_agg.columns else np.nan

                # NEW: 3-month acceleration z-score features
                zscore_3m_mean_col = 'z_score_3m_mean' if 'z_score_3m_mean' in monthly_agg.columns else None
                zscore_3m_max_col = 'z_score_3m_max' if 'z_score_3m_max' in monthly_agg.columns else None
                zscore_3m_min_col = 'z_score_3m_min' if 'z_score_3m_min' in monthly_agg.columns else None

                temp_df['VIX_zscore_3m_mean'] = monthly_agg[zscore_3m_mean_col] if zscore_3m_mean_col and zscore_3m_mean_col in monthly_agg.columns else np.nan
                temp_df['VIX_zscore_3m_max'] = monthly_agg[zscore_3m_max_col] if zscore_3m_max_col and zscore_3m_max_col in monthly_agg.columns else np.nan
                temp_df['VIX_zscore_3m_min'] = monthly_agg[zscore_3m_min_col] if zscore_3m_min_col and zscore_3m_min_col in monthly_agg.columns else np.nan

                # Regime indicators (will be further processed downstream)
                # VIX >50 = Extreme Panic (COVID hit 82)
                temp_df['VIX_panic_regime'] = (temp_df['VIX_max'] > 50).astype(int)
                # VIX >40 = High Fear
                temp_df['VIX_high_regime'] = (temp_df['VIX_max'] > 40).astype(int)

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)

            elif name == "SP500":
                # S&P 500: Market Crash Detection
                # Key features for detecting COVID-like crashes and recoveries
                sub_df['daily_chg'] = sub_df['value'].diff()
                sub_df['daily_return'] = sub_df['value'].pct_change()

                # Rolling 52-week high for drawdown calculation
                sub_df['rolling_52w_high'] = sub_df['value'].rolling(window=252, min_periods=20).max()
                sub_df['drawdown'] = (sub_df['value'] - sub_df['rolling_52w_high']) / sub_df['rolling_52w_high'] * 100

                # 30-day performance
                sub_df['value_30d_ago'] = sub_df['value'].shift(21)
                sub_df['return_30d'] = (sub_df['value'] - sub_df['value_30d_ago']) / sub_df['value_30d_ago'] * 100

                # NEW: 5-day performance (rapid crash detection)
                sub_df['value_5d_ago'] = sub_df['value'].shift(5)
                sub_df['return_5d'] = (sub_df['value'] - sub_df['value_5d_ago']) / sub_df['value_5d_ago'] * 100

                # Daily volatility (21-day rolling)
                sub_df['volatility_21d'] = sub_df['daily_return'].rolling(window=21, min_periods=10).std() * np.sqrt(252) * 100

                # NEW: Consecutive down days counter
                down_days = (sub_df['daily_return'] < 0).astype(int)  
                sub_df['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

                # NEW: Circuit breaker days counter (>5% drop)
                sub_df['circuit_breaker_day'] = (sub_df['daily_return'] < -0.05).astype(int)

                # Calculate daily z-scores (standard deviations from 12-month rolling mean)
                sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
                sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
                sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

                # NEW: Calculate 3-month acceleration z-scores (63 trading days)
                sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
                sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
                sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

                monthly_agg = sub_df.resample('MS').agg({
                    'value': ['first', 'last', 'min'],  # Month start/end/low
                    'drawdown': 'min',  # Maximum drawdown in month
                    'return_30d': 'last',  # Month-end 30-day return
                    'return_5d': 'min',  # NEW: Worst 5-day drop (rapid crash)
                    'volatility_21d': 'mean',  # Average volatility
                    'daily_return': ['std', 'min', 'max'],  # Monthly vol, worst day, best day (NEW)
                    'consecutive_down': 'max',  # NEW: Longest losing streak
                    'circuit_breaker_day': 'sum',  # NEW: Count of >5% down days
                    'z_score_12m': ['mean', 'max', 'min'],  # Regime detection via 12m z-scores
                    'z_score_3m': ['mean', 'max', 'min']  # NEW: Acceleration via 3m z-scores
                })

                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                # Handle column name variations
                first_col = 'value_first' if 'value_first' in monthly_agg.columns else 'first'
                last_col = 'value_last' if 'value_last' in monthly_agg.columns else 'last'
                ret_col = 'return_30d_last' if 'return_30d_last' in monthly_agg.columns else 'last'
                dd_col = 'drawdown_min' if 'drawdown_min' in monthly_agg.columns else 'min'
                vol_col = 'volatility_21d_mean' if 'volatility_21d_mean' in monthly_agg.columns else 'mean'
                worst_col = 'daily_return_min' if 'daily_return_min' in monthly_agg.columns else 'min'

                # Monthly return
                if first_col in monthly_agg.columns and last_col in monthly_agg.columns:
                    temp_df['SP500_monthly_return'] = ((monthly_agg[last_col] - monthly_agg[first_col])
                                                        / monthly_agg[first_col] * 100)
                else:
                    temp_df['SP500_monthly_return'] = np.nan

                temp_df['SP500_30d_return'] = monthly_agg[ret_col] if ret_col in monthly_agg.columns else np.nan
                temp_df['SP500_max_drawdown'] = monthly_agg[dd_col] if dd_col in monthly_agg.columns else np.nan
                temp_df['SP500_volatility'] = monthly_agg[vol_col] if vol_col in monthly_agg.columns else np.nan
                temp_df['SP500_worst_day'] = (monthly_agg[worst_col] * 100) if worst_col in monthly_agg.columns else np.nan

                # NEW: 5-day drop (rapid crash detection)
                ret_5d_col = 'return_5d_min' if 'return_5d_min' in monthly_agg.columns else None
                temp_df['SP500_max_5d_drop'] = monthly_agg[ret_5d_col] if ret_5d_col and ret_5d_col in monthly_agg.columns else np.nan

                # NEW: Best day (dead-cat bounce indicator)
                best_col = 'daily_return_max' if 'daily_return_max' in monthly_agg.columns else None
                temp_df['SP500_best_day'] = (monthly_agg[best_col] * 100) if best_col and best_col in monthly_agg.columns else np.nan

                # NEW: Longest losing streak
                consec_col = 'consecutive_down_max' if 'consecutive_down_max' in monthly_agg.columns else None
                temp_df['SP500_consecutive_down_days'] = monthly_agg[consec_col] if consec_col and consec_col in monthly_agg.columns else np.nan

                # NEW: Circuit breaker frequency
                cb_count_col = 'circuit_breaker_day_sum' if 'circuit_breaker_day_sum' in monthly_agg.columns else None
                temp_df['SP500_days_circuit_breaker'] = monthly_agg[cb_count_col] if cb_count_col and cb_count_col in monthly_agg.columns else np.nan

                # Z-score features (standard deviations from 12-month mean)
                zscore_mean_col = 'z_score_12m_mean' if 'z_score_12m_mean' in monthly_agg.columns else None
                zscore_max_col = 'z_score_12m_max' if 'z_score_12m_max' in monthly_agg.columns else None
                zscore_min_col = 'z_score_12m_min' if 'z_score_12m_min' in monthly_agg.columns else None

                temp_df['SP500_zscore_12m_mean'] = monthly_agg[zscore_mean_col] if zscore_mean_col and zscore_mean_col in monthly_agg.columns else np.nan
                temp_df['SP500_zscore_12m_max'] = monthly_agg[zscore_max_col] if zscore_max_col and zscore_max_col in monthly_agg.columns else np.nan
                temp_df['SP500_zscore_12m_min'] = monthly_agg[zscore_min_col] if zscore_min_col and zscore_min_col in monthly_agg.columns else np.nan

                # NEW: 3-month acceleration z-score features
                zscore_3m_mean_col = 'z_score_3m_mean' if 'z_score_3m_mean' in monthly_agg.columns else None
                zscore_3m_max_col = 'z_score_3m_max' if 'z_score_3m_max' in monthly_agg.columns else None
                zscore_3m_min_col = 'z_score_3m_min' if 'z_score_3m_min' in monthly_agg.columns else None

                temp_df['SP500_zscore_3m_mean'] = monthly_agg[zscore_3m_mean_col] if zscore_3m_mean_col and zscore_3m_mean_col in monthly_agg.columns else np.nan
                temp_df['SP500_zscore_3m_max'] = monthly_agg[zscore_3m_max_col] if zscore_3m_max_col and zscore_3m_max_col in monthly_agg.columns else np.nan
                temp_df['SP500_zscore_3m_min'] = monthly_agg[zscore_3m_min_col] if zscore_3m_min_col and zscore_3m_min_col in monthly_agg.columns else np.nan

                # Crash indicators
                # Bear market: Down >20% from 52w high
                temp_df['SP500_bear_market'] = (temp_df['SP500_max_drawdown'] < -20).astype(int)
                # Crash month: Down >10% in the month
                temp_df['SP500_crash_month'] = (temp_df['SP500_monthly_return'] < -10).astype(int)
                # Circuit breaker: Any day down >5%
                if worst_col in monthly_agg.columns:
                    temp_df['SP500_circuit_breaker'] = (monthly_agg[worst_col] < -0.05).astype(int)
                else:
                    temp_df['SP500_circuit_breaker'] = 0

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)

            elif name in ["Credit_Spreads", "Yield_Curve"]:
                # ENHANCED: Add extreme event detection for crash scenarios
                # Use volatility of CHANGES (not levels) and Momentum (sum of changes)
                sub_df['daily_chg'] = sub_df['value'].diff()

                # Extreme event features: Historical percentile & acceleration
                sub_df['expanding_mean'] = sub_df['value'].expanding(min_periods=30).mean()
                sub_df['expanding_std'] = sub_df['value'].expanding(min_periods=30).std()
                sub_df['z_score'] = (sub_df['value'] - sub_df['expanding_mean']) / sub_df['expanding_std']

                # Acceleration (rate of change of change)
                sub_df['acceleration'] = sub_df['daily_chg'].diff()

                # NEW: Calculate daily z-scores (standard deviations from 12-month rolling mean)
                sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
                sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
                sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

                # NEW: Calculate 3-month acceleration z-scores (63 trading days)
                sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
                sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
                sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

                monthly_agg = sub_df.resample('MS').agg({
                    'value': ['mean', 'max'],   # State & Peak spread (NEW: max for stress)
                    'daily_chg': ['std', 'sum'],  # Volatility & Direction
                    'z_score': 'max',  # How extreme vs. history (expanding)
                    'acceleration': ['mean', 'std'],  # Speed of change & volatility
                    'z_score_12m': ['mean', 'max', 'min'],  # NEW: 12-month regime detection
                    'z_score_3m': ['mean', 'max', 'min']  # NEW: 3-month acceleration
                })

                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                value_col = 'value_mean' if 'value_mean' in monthly_agg.columns else 'value'
                vol_col = 'daily_chg_std' if 'daily_chg_std' in monthly_agg.columns else 'std'
                sum_col = 'daily_chg_sum' if 'daily_chg_sum' in monthly_agg.columns else 'sum'
                z_col = 'z_score_max' if 'z_score_max' in monthly_agg.columns else 'max'
                accel_mean_col = 'acceleration_mean' if 'acceleration_mean' in monthly_agg.columns else 'mean'
                accel_std_col = 'acceleration_std' if 'acceleration_std' in monthly_agg.columns else 'std'

                temp_df[f'{name}_avg'] = monthly_agg[value_col] if value_col in monthly_agg.columns else np.nan

                # NEW: Peak spread (max stress)
                max_col = 'value_max' if 'value_max' in monthly_agg.columns else None
                temp_df[f'{name}_max'] = monthly_agg[max_col] if max_col and max_col in monthly_agg.columns else np.nan

                temp_df[f'{name}_vol_of_changes'] = monthly_agg[vol_col] if vol_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_monthly_chg'] = monthly_agg[sum_col] if sum_col in monthly_agg.columns else np.nan

                # Extreme event features (expanding z-score)
                temp_df[f'{name}_zscore_max'] = monthly_agg[z_col] if z_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_acceleration'] = monthly_agg[accel_mean_col] if accel_mean_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_accel_volatility'] = monthly_agg[accel_std_col] if accel_std_col in monthly_agg.columns else np.nan

                # NEW: Z-score 12m features (standard deviations from 12-month mean)
                zscore_12m_mean_col = 'z_score_12m_mean' if 'z_score_12m_mean' in monthly_agg.columns else None
                zscore_12m_max_col = 'z_score_12m_max' if 'z_score_12m_max' in monthly_agg.columns else None
                zscore_12m_min_col = 'z_score_12m_min' if 'z_score_12m_min' in monthly_agg.columns else None

                temp_df[f'{name}_zscore_12m_mean'] = monthly_agg[zscore_12m_mean_col] if zscore_12m_mean_col and zscore_12m_mean_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_zscore_12m_max'] = monthly_agg[zscore_12m_max_col] if zscore_12m_max_col and zscore_12m_max_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_zscore_12m_min'] = monthly_agg[zscore_12m_min_col] if zscore_12m_min_col and zscore_12m_min_col in monthly_agg.columns else np.nan

                # NEW: Z-score 3m features (3-month acceleration)
                zscore_3m_mean_col = 'z_score_3m_mean' if 'z_score_3m_mean' in monthly_agg.columns else None
                zscore_3m_max_col = 'z_score_3m_max' if 'z_score_3m_max' in monthly_agg.columns else None
                zscore_3m_min_col = 'z_score_3m_min' if 'z_score_3m_min' in monthly_agg.columns else None

                temp_df[f'{name}_zscore_3m_mean'] = monthly_agg[zscore_3m_mean_col] if zscore_3m_mean_col and zscore_3m_mean_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_zscore_3m_max'] = monthly_agg[zscore_3m_max_col] if zscore_3m_max_col and zscore_3m_max_col in monthly_agg.columns else np.nan
                temp_df[f'{name}_zscore_3m_min'] = monthly_agg[zscore_3m_min_col] if zscore_3m_min_col and zscore_3m_min_col in monthly_agg.columns else np.nan

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)

            elif name == "Oil_Prices":
                # ENHANCED: Add extreme event detection for crashes (COVID: $60 → -$37)
                # Use Dollar Change (diff) for mathematical stability with negative prices
                sub_df['daily_chg'] = sub_df['value'].diff()

                # 30-day crash detection (>40% drop = extreme event)
                sub_df['value_30d_ago'] = sub_df['value'].shift(21)  # ~30 trading days
                sub_df['crash_30d_pct'] = ((sub_df['value'] - sub_df['value_30d_ago'])
                                           / sub_df['value_30d_ago'].abs()) * 100

                # Negative price indicator (unprecedented Apr 2020)
                sub_df['is_negative'] = (sub_df['value'] < 0).astype(int)

                # NEW: Daily percentage change (captures -301% COVID drop)
                sub_df['daily_pct'] = sub_df['value'].pct_change() * 100

                # Historical z-score (expanding - kept for compatibility)
                sub_df['expanding_mean'] = sub_df['value'].expanding(min_periods=30).mean()
                sub_df['expanding_std'] = sub_df['value'].expanding(min_periods=30).std()
                sub_df['z_score'] = (sub_df['value'] - sub_df['expanding_mean']) / sub_df['expanding_std']

                # Calculate daily z-scores (standard deviations from 12-month rolling mean)
                sub_df['rolling_12m_mean'] = sub_df['value'].rolling(window=252, min_periods=60).mean()
                sub_df['rolling_12m_std'] = sub_df['value'].rolling(window=252, min_periods=60).std()
                sub_df['z_score_12m'] = (sub_df['value'] - sub_df['rolling_12m_mean']) / sub_df['rolling_12m_std']

                # NEW: Calculate 3-month acceleration z-scores (63 trading days)
                sub_df['rolling_3m_mean'] = sub_df['value'].rolling(window=63, min_periods=20).mean()
                sub_df['rolling_3m_std'] = sub_df['value'].rolling(window=63, min_periods=20).std()
                sub_df['z_score_3m'] = (sub_df['value'] - sub_df['rolling_3m_mean']) / sub_df['rolling_3m_std']

                monthly_agg = sub_df.resample('MS').agg({
                    'value': 'mean',        # State
                    'daily_chg': 'std',     # Volatility of dollar moves
                    'crash_30d_pct': 'min',  # Worst 30-day crash in month
                    'daily_pct': 'min',      # NEW: Worst single-day % drop (-301% COVID)
                    'is_negative': ['max', 'sum'],    # Did it go negative? How many days?
                    'z_score': 'min',        # How extreme vs. history (min = biggest crash)
                    'z_score_12m': ['mean', 'max', 'min'],  # 12-month regime detection
                    'z_score_3m': ['mean', 'max', 'min']  # NEW: 3-month acceleration
                })

                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                value_col = 'value_mean' if 'value_mean' in monthly_agg.columns else 'value'
                chg_col = 'daily_chg_std' if 'daily_chg_std' in monthly_agg.columns else 'daily_chg'
                crash_col = 'crash_30d_pct_min' if 'crash_30d_pct_min' in monthly_agg.columns else ('crash_30d_pct' if 'crash_30d_pct' in monthly_agg.columns else None)
                neg_col = 'is_negative_max' if 'is_negative_max' in monthly_agg.columns else ('is_negative' if 'is_negative' in monthly_agg.columns else None)
                zscore_col = 'z_score_min' if 'z_score_min' in monthly_agg.columns else ('z_score' if 'z_score' in monthly_agg.columns else None)

                temp_df['Oil_Prices_mean'] = monthly_agg[value_col]
                temp_df['Oil_Prices_volatility'] = monthly_agg[chg_col]

                # Extreme event features
                if crash_col and crash_col in monthly_agg.columns:
                    temp_df['Oil_Prices_30d_crash'] = monthly_agg[crash_col]
                if neg_col and neg_col in monthly_agg.columns:
                    temp_df['Oil_Prices_went_negative'] = monthly_agg[neg_col]
                if zscore_col and zscore_col in monthly_agg.columns:
                    temp_df['Oil_Prices_zscore_min'] = monthly_agg[zscore_col]

                # NEW: Worst single-day % drop
                worst_day_col = 'daily_pct_min' if 'daily_pct_min' in monthly_agg.columns else None
                temp_df['Oil_worst_day_pct'] = monthly_agg[worst_day_col] if worst_day_col and worst_day_col in monthly_agg.columns else np.nan

                # NEW: Days with negative prices
                neg_days_col = 'is_negative_sum' if 'is_negative_sum' in monthly_agg.columns else None
                temp_df['Oil_days_negative'] = monthly_agg[neg_days_col] if neg_days_col and neg_days_col in monthly_agg.columns else np.nan

                # Z-score 12m features (standard deviations from 12-month mean)
                zscore_12m_mean_col = 'z_score_12m_mean' if 'z_score_12m_mean' in monthly_agg.columns else None
                zscore_12m_max_col = 'z_score_12m_max' if 'z_score_12m_max' in monthly_agg.columns else None
                zscore_12m_min_col = 'z_score_12m_min' if 'z_score_12m_min' in monthly_agg.columns else None

                temp_df['Oil_Prices_zscore_12m_mean'] = monthly_agg[zscore_12m_mean_col] if zscore_12m_mean_col and zscore_12m_mean_col in monthly_agg.columns else np.nan
                temp_df['Oil_Prices_zscore_12m_max'] = monthly_agg[zscore_12m_max_col] if zscore_12m_max_col and zscore_12m_max_col in monthly_agg.columns else np.nan
                temp_df['Oil_Prices_zscore_12m_min'] = monthly_agg[zscore_12m_min_col] if zscore_12m_min_col and zscore_12m_min_col in monthly_agg.columns else np.nan

                # NEW: Z-score 3m features (3-month acceleration)
                zscore_3m_mean_col = 'z_score_3m_mean' if 'z_score_3m_mean' in monthly_agg.columns else None
                zscore_3m_max_col = 'z_score_3m_max' if 'z_score_3m_max' in monthly_agg.columns else None
                zscore_3m_min_col = 'z_score_3m_min' if 'z_score_3m_min' in monthly_agg.columns else None

                temp_df['Oil_Prices_zscore_3m_mean'] = monthly_agg[zscore_3m_mean_col] if zscore_3m_mean_col and zscore_3m_mean_col in monthly_agg.columns else np.nan
                temp_df['Oil_Prices_zscore_3m_max'] = monthly_agg[zscore_3m_max_col] if zscore_3m_max_col and zscore_3m_max_col in monthly_agg.columns else np.nan
                temp_df['Oil_Prices_zscore_3m_min'] = monthly_agg[zscore_3m_min_col] if zscore_3m_min_col and zscore_3m_min_col in monthly_agg.columns else np.nan

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)
                
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
                # JOLTS, Regional Fed, and other monthly data
                sub_df = sub_df.resample('MS').last().reset_index()
                sub_df['series_name'] = name
                
                # JOLTS Release Logic
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
            full_snap = full_snap[full_snap['release_date'] < snap_date].copy()
            after_count = len(full_snap)

            if before_count > after_count:
                logger.info(f"Filtered out {before_count - after_count} rows with release_date > snapshot_date")

            full_snap.to_parquet(save_path)
            logger.info(f"Saved snapshot to {save_path}")

if __name__ == "__main__":
    logger.info(f"Fetching FRED exogenous data from {START_DATE} to {END_DATE}")
    fetch_fred_exogenous_snapshots(start_date=START_DATE, end_date=END_DATE)