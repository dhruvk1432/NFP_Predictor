import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unifier import unifier
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE, UNIFIER_TOKEN, UNIFIER_USER
# OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map
# OPTIMIZATION: Use shared utility for snapshot path
from Data_ETA_Pipeline.utils import get_snapshot_path

logger = setup_logger(__file__, TEMP_DIR)

# OPTIMIZATION [H3]: Rate limiter for API calls (max 10 req/sec to avoid throttling)
class RateLimiter:
    """Simple rate limiter to prevent API throttling."""
    def __init__(self, max_per_second: float = 10.0):
        self.min_interval = 1.0 / max_per_second
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if needed to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

# VIF-selected features (53 of 144): pruned at threshold=10.0
# From prosper_exog_selection.ipynb analysis
SELECTED_FEATURES = {
    # Consumer Mood Index (7 features)
    'Consumer Mood Index | Consumer Mood Index | 18-34_diff',
    'Consumer Mood Index | Consumer Mood Index | 18-34_diff_zscore_3m',
    'Consumer Mood Index | Consumer Mood Index | Females',
    'Consumer Mood Index | Consumer Mood Index | Females_diff',
    'Consumer Mood Index | Consumer Mood Index | Females_diff_zscore_3m',
    'Consumer Mood Index | Consumer Mood Index | Males',
    'Consumer Mood Index | Consumer Mood Index | Males_diff_zscore_3m',
    # Consumer Spending Forecast (4 features)
    'Prosper Consumer Spending Forecast | Consumer Spending Forecast | 18-34',
    'Prosper Consumer Spending Forecast | Consumer Spending Forecast | 18-34_diff_zscore_3m',
    'Prosper Consumer Spending Forecast | Consumer Spending Forecast | Females_diff_zscore_3m',
    'Prosper Consumer Spending Forecast | Consumer Spending Forecast | Males_diff_zscore_12m',
    # Layoffs - Fewer (4 features)
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Fewer | 18-34_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Fewer | Females_diff_zscore_12m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Fewer | Females_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Fewer | Males_diff_zscore_3m',
    # Layoffs - More (6 features)
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | 18-34_diff_zscore_12m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | 18-34_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | Females',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | Females_diff',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | Females_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | More | Males_diff_zscore_3m',
    # Layoffs - Same (7 features)
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | 18-34_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | Females',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | Females_diff_zscore_12m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | Females_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | Males_diff_zscore_12m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | Males_diff_zscore_3m',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present? | Same | US 18+_diff_zscore_3m',
    # Employment - Full-time (6 features)
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | 18-34_diff',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | 18-34_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | Females_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | Males',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | Males_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed full-time | US 18+_diff_zscore_3m',
    # Employment - Employed (merged) (5 features)
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed | 18-34_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed | Females_diff',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed | Females_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed | Males_diff_zscore_12m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am employed | Males_diff_zscore_3m',
    # Employment - Unemployed (7 features)
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | 18-34',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | 18-34_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | Females',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | Females_diff_zscore_12m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | Females_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | Males_diff',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | Males_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I am unemployed | US 18+_diff_zscore_3m',
    # Employment - Know people laid off (6 features)
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | 18-34',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | 18-34_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | Females_diff',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | Females_diff_zscore_3m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | Males_diff_zscore_12m',
    'Which of the following most accurately describes your employment environment? (Check all that apply) | I know people who have been laid off | Males_diff_zscore_3m',
}

top_nfp_predictors = [
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present?',
    'Which of the following most accurately describes your employment environment? (Check all that apply)',
    'Prosper Consumer Spending Forecast',
    'Consumer Mood Index',
]

top_nfp_groups = [
    #'HH Income $50K+', 
    #'HH Income $100K+', 
    'US 18+', 
    '18-34', 
    #'US Midwest Region', 
    #'US South Region', 
    #'US Northeast Region', 
    #'US West Region', 
    #'US Pacific Division', 
    #'US South Atlantic Division', 
    #'US Mid Atlantic Division', 
    #'US East North Central Division', 
    'Males', 
    'Females'
]

def compute_mom_differences_and_zscores(df):
    """
    Keep original level values AND add MoM differences with z-scores.

    For each series, produce 4 variants:
    - level (original series_name)
    - {series_name}_diff (MoM difference)
    - {series_name}_diff_zscore_12m (12-month rolling z-score on diff)
    - {series_name}_diff_zscore_3m (3-month rolling z-score on diff)
    """
    series_list = df['series_name'].unique()
    result_list = []
    base_cols = ['date', 'release_date', 'series_code', 'snapshot_date']

    for series in series_list:
        series_df = df[df['series_name'] == series].copy()
        series_df = series_df.sort_values('date')

        # Always keep the original level value
        level_df = series_df[base_cols + ['value']].copy()
        level_df['series_name'] = series
        result_list.append(level_df)

        # Compute MoM difference
        series_df['value_diff'] = series_df['value'].diff()

        # Compute z-scores on differences (12-month)
        rolling_mean_12m = series_df['value_diff'].rolling(window=12, min_periods=6).mean()
        rolling_std_12m = series_df['value_diff'].rolling(window=12, min_periods=6).std()
        series_df['zscore_12m'] = (series_df['value_diff'] - rolling_mean_12m) / rolling_std_12m

        # Compute z-scores on differences (3-month)
        rolling_mean_3m = series_df['value_diff'].rolling(window=3, min_periods=2).mean()
        rolling_std_3m = series_df['value_diff'].rolling(window=3, min_periods=2).std()
        series_df['zscore_3m'] = (series_df['value_diff'] - rolling_mean_3m) / rolling_std_3m

        diff_df = series_df[base_cols].copy()
        diff_df['series_name'] = f"{series}_diff"
        diff_df['value'] = series_df['value_diff'].values

        zscore_12m_df = series_df[base_cols].copy()
        zscore_12m_df['series_name'] = f"{series}_diff_zscore_12m"
        zscore_12m_df['value'] = series_df['zscore_12m'].values

        zscore_3m_df = series_df[base_cols].copy()
        zscore_3m_df['series_name'] = f"{series}_diff_zscore_3m"
        zscore_3m_df['value'] = series_df['zscore_3m'].values

        result_list.extend([diff_df, zscore_12m_df, zscore_3m_df])

    result = pd.concat(result_list, ignore_index=True)
    result = result.dropna(subset=['value'])
    return result


def symbols_for_target(df: pd.DataFrame, target, *, contains: bool = False, case: bool = True):
    s = df["question_text"].astype(str)
    mask = s.str.contains(str(target), case=case, na=False) if contains else s.eq(str(target))
    return df.loc[mask, "symbol"].dropna().unique().tolist()


def create_series_code(symbol: str, answer_id) -> str:
    """Create unique series code from symbol and answer_id."""
    return f"{symbol}_ans{answer_id}"

def create_series_name(question: str, answer: str, symbol: str) -> str:
    """Create series_name as: question | answer | group."""
    # Extract group from symbol (e.g., "18-34_136" -> "18-34")
    group = symbol.rsplit('_', 1)[0]
    return f"{question} | {answer} | {group}"


def fetch_single_key(key: str, rate_limiter: RateLimiter) -> list:
    """
    Fetch data for a single Prosper key.

    OPTIMIZATION [H3]: Helper function for parallel fetching.

    Args:
        key: Prosper key to fetch
        rate_limiter: Rate limiter to prevent API throttling

    Returns:
        List of DataFrames for each answer in the key, or empty list on error
    """
    rate_limiter.wait()  # Respect rate limit

    try:
        df1 = unifier.get_dataframe(name="prosper_v2", key=key)

        if df1.empty:
            return []

        results = []
        answers = df1["answer_text"].unique()

        for answer in answers:
            # Filter for this answer and non-null values
            mask = (df1["answer_text"] == answer) & (df1["value"].notna())
            prosper_df = df1[mask].sort_values(by='date').copy()

            if prosper_df.empty:
                continue

            # Extract identifiers
            answer_id = prosper_df['answer_id'].iloc[0]
            symbol = prosper_df['symbol'].iloc[0]
            question = prosper_df['question_text'].iloc[0]

            # Create output dataframe matching unifier format exactly
            out_df = pd.DataFrame({
                'date': pd.to_datetime(prosper_df['date']).dt.to_period('M').dt.to_timestamp(),
                'release_date': pd.to_datetime(prosper_df['date']),
                'value': prosper_df['value'].values,
                'series_name': create_series_name(question, answer, symbol),
                'series_code': create_series_code(symbol, answer_id)
            })

            results.append(out_df)

        return results

    except Exception as e:
        logger.error(f"Error fetching key {key}: {e}")
        return []


EMPLOYMENT_QUESTION = (
    "Which of the following most accurately describes your employment environment? "
    "(Check all that apply)"
)


def filter_unwanted_series(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unwanted employment-related series from prosper data.

    Filters out:
    - I am retired or disabled (last asked 3-2010)
    - I am concerned with being laid off
    - Someone in my family has been laid off
    - I am disabled
    - I am retired
    """
    unwanted_answers = [
        'I am retired or disabled (last asked 3-2010)',
        'I am concerned with being laid off',
        'Someone in my family has been laid off',
        'I am disabled',
        'I am retired',
    ]

    # Filter out rows containing any of these answers in series_name
    mask = pd.Series([True] * len(combined_df), index=combined_df.index)
    for answer in unwanted_answers:
        mask &= ~combined_df['series_name'].str.contains(answer, regex=False, na=False)

    filtered_df = combined_df[mask].copy()

    removed_count = len(combined_df) - len(filtered_df)
    if removed_count > 0:
        logger.info(f"Filtered out {removed_count} rows from unwanted series")

    return filtered_df


def merge_employment_series(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge full-time + part-time employment as continuation of the legacy
    'I am employed (last asked 9-2009)' series, then rename to 'I am employed'.

    For each demographic group:
    1. Rename legacy 'I am employed (last asked 9-2009)' → 'I am employed'
    2. For dates after the legacy data ends, append rows with
       value = full-time + part-time
    3. Drop the part-time series entirely (avoid collinearity with full-time)
    """
    groups = ['US 18+', '18-34', 'Males', 'Females']
    extension_rows = []

    df = combined_df.copy()

    for group in groups:
        legacy_name = f"{EMPLOYMENT_QUESTION} | I am employed (last asked 9-2009) | {group}"
        ft_name = f"{EMPLOYMENT_QUESTION} | I am employed full-time | {group}"
        pt_name = f"{EMPLOYMENT_QUESTION} | I am employed part-time | {group}"
        merged_name = f"{EMPLOYMENT_QUESTION} | I am employed | {group}"

        legacy_data = df[df['series_name'] == legacy_name]
        ft_data = df[df['series_name'] == ft_name]
        pt_data = df[df['series_name'] == pt_name]

        if legacy_data.empty:
            logger.warning(f"No legacy 'I am employed' data for group={group}, skipping")
            continue

        legacy_code = legacy_data['series_code'].iloc[0]
        legacy_last_date = legacy_data['date'].max()

        # Rename legacy rows to the merged name
        df.loc[df['series_name'] == legacy_name, 'series_name'] = merged_name

        # Build extension: sum of full-time + part-time for dates after legacy ends
        if not ft_data.empty and not pt_data.empty:
            ft_agg = (ft_data.groupby('date')
                      .agg(value=('value', 'last'), release_date=('release_date', 'last'))
                      .reset_index())
            pt_agg = (pt_data.groupby('date')
                      .agg(value=('value', 'last'), release_date=('release_date', 'last'))
                      .reset_index())

            merged = ft_agg.merge(pt_agg, on='date', suffixes=('_ft', '_pt'))
            merged['value'] = merged['value_ft'] + merged['value_pt']
            merged['release_date'] = merged[['release_date_ft', 'release_date_pt']].max(axis=1)

            ext = merged[merged['date'] > legacy_last_date]
            if not ext.empty:
                extension_rows.append(pd.DataFrame({
                    'date': ext['date'].values,
                    'release_date': ext['release_date'].values,
                    'value': ext['value'].values,
                    'series_name': merged_name,
                    'series_code': legacy_code,
                }))
                logger.info(
                    f"Merged employment for {group}: {len(legacy_data)} legacy rows + "
                    f"{len(ext)} extension rows"
                )
        else:
            logger.warning(f"Missing full-time or part-time data for group={group}")

    # Drop all part-time rows
    for group in groups:
        pt_name = f"{EMPLOYMENT_QUESTION} | I am employed part-time | {group}"
        df = df[df['series_name'] != pt_name]

    # Append extension rows
    if extension_rows:
        df = pd.concat([df] + extension_rows, ignore_index=True)
        df = df.sort_values('release_date').reset_index(drop=True)

    return df


def fetch_prosper_snapshots(start_date=START_DATE, end_date=END_DATE, max_workers: int = 4):
    """
    Fetch Prosper survey data and create monthly snapshots.

    Each snapshot (YYYY-MM.parquet) contains ALL prosper data that was available
    by that month - i.e., all data with release_date < snapshot_date.

    Output format matches load_unifier_data.py:
    - date: observation date (first of month for the survey period)
    - release_date: when the data was released
    - value: the survey value
    - series_name: shortened question text
    - series_code: unique identifier ({symbol}_ans{answer_id})
    - snapshot_date: the snapshot cutoff date
    """
    # Setup credentials
    unifier.user = UNIFIER_USER
    unifier.token = UNIFIER_TOKEN
    os.environ['UNIFIER_USER'] = unifier.user
    os.environ['UNIFIER_TOKEN'] = unifier.token

    # OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)

    base_dir = Path(DATA_PATH) / "Exogenous_data" / "prosper"

    # Skip-if-exists: Check if all snapshots already exist
    existing_count = 0
    for obs_month in nfp_release_map.keys():
        snap_path = get_snapshot_path(base_dir, pd.Timestamp(obs_month))
        if snap_path.exists():
            existing_count += 1

    if existing_count == len(nfp_release_map):
        print(f"✓ Prosper data already exists: {existing_count} monthly snapshots", flush=True)
        logger.info(f"Prosper snapshots already exist, skipping")
        return

    # Step 1: Download metadata to get the list of keys we need
    logger.info("Starting Prosper data download")
    metadata_df = unifier.get_dataframe(name="prosper_v2", back_to="2026-01-01")

    # Get filtered NFP codes from the metadata
    prosper_codes = []
    for q in top_nfp_predictors:
        lst = symbols_for_target(metadata_df, q, contains=False, case=True)
        prosper_codes.extend(lst)

    filtered_nfp_codes = list(set([
        code for code in prosper_codes
        if code.rsplit('_', 1)[0] in top_nfp_groups
    ]))

    # Step 2: Download keys in parallel and collect all time series data
    # OPTIMIZATION [H3]: Use ThreadPoolExecutor for parallel fetching (3x speedup)
    all_prosper_data = []
    rate_limiter = RateLimiter(max_per_second=10.0)  # Prevent API throttling

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_key = {
            executor.submit(fetch_single_key, key, rate_limiter): key
            for key in filtered_nfp_codes
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            completed += 1

            try:
                results = future.result()
                if results:
                    all_prosper_data.extend(results)

                # Log progress every 25 keys
                if completed % 25 == 0 or completed == len(filtered_nfp_codes):
                    logger.info(f"Fetched {completed}/{len(filtered_nfp_codes)} keys")

            except Exception as e:
                logger.error(f"Error processing key {key}: {e}")

    if not all_prosper_data:
        logger.error("No prosper data collected")
        return

    # Combine all data
    combined_df = pd.concat(all_prosper_data, ignore_index=True)
    combined_df = combined_df.sort_values('release_date').reset_index(drop=True)

    # Filter out unwanted series (retired, disabled, layoff concerns, etc.)
    combined_df = filter_unwanted_series(combined_df)

    # Merge employment series: legacy "I am employed (last asked 9-2009)" +
    # (full-time + part-time) extension → "I am employed", drop part-time
    combined_df = merge_employment_series(combined_df)

    # Now create monthly snapshots aligned with NFP release dates
    # Each snapshot contains ALL data with release_date < snapshot_date
    for obs_month, nfp_release_date in nfp_release_map.items():
        snap_date = pd.Timestamp(nfp_release_date)

        # OPTIMIZATION: Use shared utility for snapshot path
        save_path = get_snapshot_path(base_dir, obs_month)

        # Get all data released BEFORE the snapshot date (strict <)
        snap_data = combined_df[combined_df['release_date'] < snap_date].copy()

        if not snap_data.empty:
            # Remove duplicates (keep last by date and series_code)
            snap_data = snap_data.sort_values('release_date').drop_duplicates(
                subset=['date', 'series_code'], keep='last'
            )
            snap_data['snapshot_date'] = snap_date

            # Apply MoM differencing and z-score calculation
            snap_data = compute_mom_differences_and_zscores(snap_data)
            snap_data = snap_data[snap_data['series_name'].isin(SELECTED_FEATURES)]

            snap_data.to_parquet(save_path, index=False)

        if obs_month.month == 12:
            logger.info(f"Saved {obs_month.year} snapshots")

    logger.info("✓ Prosper data download complete")

if __name__ == "__main__":
    fetch_prosper_snapshots(start_date=START_DATE, end_date=END_DATE)
