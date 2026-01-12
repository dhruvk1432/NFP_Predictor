import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unifier import unifier

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE, UNIFIER_TOKEN, UNIFIER_USER
# OPTIMIZATION: Use shared NFP loading utility (cached, avoids redundant file reads)
from nfp_relative_timing import get_nfp_release_map
# OPTIMIZATION: Use shared utility for snapshot path
from Load_Data.utils import get_snapshot_path

logger = setup_logger(__file__, TEMP_DIR)

top_nfp_predictors = [
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present?',
    'Which of the following most accurately describes your employment environment? (Check all that apply)',
    'Do you feel that there is "too much month at the end of your paycheck"?',
    'How many times a month do you eat at a FULL SERVICE restaurant (i.e. waiter/waitress)?',
    'Prosper Consumer Spending Forecast',
    'Do you plan to make any of the following major (big dollar) purchases within the next 6 months? (Check all that apply)',
    'Consumer Mood Index',
    'Overall, do you plan to save more, the same or less than you did last year?'
]

top_nfp_groups = [
    'Business Owners', 
    'HH Income <$50K', 
    'HH Income $50K+', 
    'HH Income $75K+', 
    'HH Income $100K+', 
    'US 18+', 
    '18-34', 
    '25-34', 
    '35-44', 
    '45-54', 
    '35-54', 
    'Parents with Children <18 at Home', 
    'AMZN', 
    'Amazon Prime Members', 
    'AAPL', 
    'NKE', 
    'Visa Users', 
    'Master Card Users', 
    'American Express Users', 
    'WMT', 
    'TGT', 
    'Costco Members', 
    'US Midwest Region', 
    'US South Region', 
    'US Northeast Region', 
    'US West Region', 
    'US Pacific Division', 
    'US South Atlantic Division', 
    'US Mid Atlantic Division', 
    'US East North Central Division', 
    'Males', 
    'Females'
]

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

def fetch_prosper_snapshots(start_date=START_DATE, end_date=END_DATE):
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

    base_dir = Path(DATA_PATH) / "Exogenous_data" / "prosper" / "decades"

    # Step 1: Download metadata to get the list of keys we need
    logger.info("Downloading prosper metadata to identify keys...")
    metadata_df = unifier.get_dataframe(name="prosper_v2", back_to="2026-01-01")
    logger.info(f"Downloaded metadata with {len(metadata_df)} records")

    # Get filtered NFP codes from the metadata
    prosper_codes = []
    for q in top_nfp_predictors:
        lst = symbols_for_target(metadata_df, q, contains=False, case=True)
        prosper_codes.extend(lst)

    filtered_nfp_codes = list(set([
        code for code in prosper_codes
        if code.rsplit('_', 1)[0] in top_nfp_groups
    ]))
    logger.info(f"Found {len(filtered_nfp_codes)} unique prosper keys to download")

    # Step 2: Download each key once and collect all time series data
    all_prosper_data = []

    for i, key in enumerate(filtered_nfp_codes):
        try:
            logger.info(f"Fetching key {i+1}/{len(filtered_nfp_codes)}: {key}")
            df1 = unifier.get_dataframe(name="prosper_v2", key=key, user=UNIFIER_USER, token=UNIFIER_TOKEN)

            if df1.empty:
                logger.warning(f"No data returned for key: {key}")
                continue

            # Get unique answers for this key
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
                # Columns: date, release_date, value, series_name, series_code (+ snapshot_date added later)
                out_df = pd.DataFrame({
                    'date': pd.to_datetime(prosper_df['date']).dt.to_period('M').dt.to_timestamp(),  # First of month
                    'release_date': pd.to_datetime(prosper_df['date']),  # The date column IS the release date
                    'value': prosper_df['value'].values,
                    'series_name': create_series_name(question, answer, symbol),
                    'series_code': create_series_code(symbol, answer_id)
                })

                all_prosper_data.append(out_df)

        except Exception as e:
            logger.error(f"Error fetching key {key}: {e}")
            continue

    logger.info(f"Processed {len(filtered_nfp_codes)} keys into {len(all_prosper_data)} series")

    if not all_prosper_data:
        logger.warning("No prosper data collected")
        return

    # Combine all data
    combined_df = pd.concat(all_prosper_data, ignore_index=True)
    combined_df = combined_df.sort_values('release_date').reset_index(drop=True)
    logger.info(f"Total prosper records collected: {len(combined_df)}")

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
            snap_data.to_parquet(save_path, index=False)

        if obs_month.month == 12:
            logger.info(f"Generated prosper snapshots for {obs_month.year}")

    logger.info("Completed saving prosper snapshots")

if __name__ == "__main__":
    logger.info(f"Fetching Prosper data from {START_DATE} to {END_DATE}")
    fetch_prosper_snapshots(start_date=START_DATE, end_date=END_DATE)
