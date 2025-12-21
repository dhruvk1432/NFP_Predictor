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
    #Daily Data
    "Credit_Spreads": "BAMLH0A0HYM2",
    "Yield_Curve": "T10Y3M",
    "Oil_Prices": "DCOILWTICO",
    #Monthly Data (JOLTS_Openings and JOLTS_Hires dropped due to multicollinearity)
    # "JOLTS_Openings": "JTSJOL",  # DROPPED
    # "JOLTS_Hires": "JTSHIL",  # DROPPED
    "JOLTS_Quits": "JTSQUR",
    "JOLTS_Layoffs": "JTSLDL",
    # Weekly Jobless Claims (ICSA and IURSA dropped due to multicollinearity)
    # NOTE: These are aggregated monthly based on NFP release boundaries to ensure
    # only data available before each NFP report is included in that month's features
    # "ICSA": "ICSA",  # DROPPED - Initial Claims Seasonally Adjusted
    "CCSA": "CCSA",  # Continued Claims Seasonally Adjusted (KEPT)
    # "IURSA": "IURSA"  # DROPPED - Insured Unemployment Rate Seasonally Adjusted
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

    DAILY_SERIES = ["Credit_Spreads", "Yield_Curve", "Oil_Prices"]
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
                df['realtime_start'] = df['date']
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
            
            if name in ["Credit_Spreads", "Yield_Curve"]:
                # IMPROVED STRATEGY:
                # Use volatility of CHANGES (not levels) and Momentum (sum of changes)
                # Removed "_last" to eliminate collinearity with "_avg"
                sub_df['daily_chg'] = sub_df['value'].diff()

                monthly_agg = sub_df.resample('MS').agg({
                    'value': 'mean',           # State (removed 'last' for collinearity)
                    'daily_chg': ['std', 'sum']  # Volatility & Direction
                })
                # Flatten column names if multi-level
                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                # Check column names after aggregation
                value_col = 'value_mean' if 'value_mean' in monthly_agg.columns else 'value'

                temp_df[f'{name}_avg'] = monthly_agg[value_col]
                # Standard deviation of daily changes = Realized Volatility
                temp_df[f'{name}_vol_of_changes'] = monthly_agg['daily_chg_std']
                # Sum of daily changes = Monthly Momentum (approx Last - First)
                temp_df[f'{name}_monthly_chg'] = monthly_agg['daily_chg_sum']

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                # Financial data is available at end of month
                # date = first day of month, release_date = last day of month
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)

            elif name == "Oil_Prices":
                # UPDATED: Use Dollar Change (diff) instead of Log Returns
                # This ensures mathematical stability even when prices are negative (Apr 2020)
                # Removed min, max, end_of_month due to high collinearity
                sub_df['daily_chg'] = sub_df['value'].diff()

                monthly_agg = sub_df.resample('MS').agg({
                    'value': 'mean',        # State (removed min/max/last for collinearity)
                    'daily_chg': 'std'      # Volatility of dollar moves (not percent)
                })
                # Flatten column names if multi-level
                if isinstance(monthly_agg.columns, pd.MultiIndex):
                    monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

                temp_df = pd.DataFrame(index=monthly_agg.index)
                # Check column names after aggregation
                value_col = 'value_mean' if 'value_mean' in monthly_agg.columns else 'value'
                chg_col = 'daily_chg_std' if 'daily_chg_std' in monthly_agg.columns else 'daily_chg'

                temp_df['Oil_Prices_mean'] = monthly_agg[value_col]
                temp_df['Oil_Prices_volatility'] = monthly_agg[chg_col]

                sub_df = temp_df.reset_index().melt(
                    id_vars=['date'],
                    var_name='series_name',
                    value_name='value'
                )
                # Oil price data is available at end of month
                # date = first day of month, release_date = last day of month
                sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)
                
            elif name in ["ICSA", "CCSA", "IURSA"]:
                # NFP-BASED AGGREGATION: Bucket weekly data by NFP release windows
                # This ensures we only use data that would have been available before each NFP

                # Prepare data for aggregation function
                weekly_data = latest[['date', 'value', 'realtime_start']].copy()

                # Use NFP-based aggregation
                monthly_avg = aggregate_weekly_to_monthly_nfp_based(weekly_data, nfp_schedule)

                if monthly_avg.empty:
                    logger.warning(f"No monthly data generated for {name}")
                    continue

                monthly_avg['series_name'] = f"{name}_monthly_avg"
                sub_df = monthly_avg.copy()
                logger.info(f"Calculated NFP-based monthly average for {name}")
                
            else:
                # JOLTS and other monthly data
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