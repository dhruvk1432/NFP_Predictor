"""
ADP Non-Farm Employment Change - Key NFP Predictor (2 Days Early)

Loads ADP employment data from historical CSV and converts to long format
for pipeline integration. Output format matches FRED snapshots.

Note: Web scraping is disabled due to Cloudflare verification requirements.
Data is sourced from us-private-employment.csv which contains employment levels.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# --- Paths ---
EXOG_ADP_DIR = DATA_PATH / "Exogenous_data" / "ADP_data"
EXOG_ADP_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_ADP_PARQUET = EXOG_ADP_DIR / "ADP_Employment_Change.parquet"

# Historical ADP data from CSV file (employment levels, not changes)
HISTORICAL_CSV = DATA_PATH.parent / "us-private-employment.csv"


def get_first_wednesday(date: pd.Timestamp) -> pd.Timestamp:
    """Get the first Wednesday of the month following the given date.

    ADP typically releases employment data on the first Wednesday after month-end.
    """
    next_month = date + pd.DateOffset(months=1)
    first_day = next_month.replace(day=1)
    days_until_wed = (2 - first_day.dayofweek + 7) % 7  # Wednesday = 2
    if days_until_wed == 0:
        days_until_wed = 7  # If 1st is Wednesday, use that day
    return first_day + pd.Timedelta(days=days_until_wed - (7 if days_until_wed == 7 else 0))


def load_adp_from_csv() -> pd.DataFrame:
    """
    Load ADP private employment levels from CSV and convert to MoM changes.

    The CSV contains employment LEVELS (e.g., 134,588,000). We calculate the
    month-over-month change to get the ADP employment change figure.

    Returns:
        DataFrame with columns: date, series_name, value, release_date, series_type
    """
    if not HISTORICAL_CSV.exists():
        logger.error(f"Historical CSV not found: {HISTORICAL_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(HISTORICAL_CSV)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    # Calculate MoM change
    df['change'] = df['Private Employment'].diff()

    # First row has no change (NaN), drop it
    df = df.dropna(subset=['change'])

    # Impute release date: first Wednesday of following month
    df['release_date'] = df['DateTime'].apply(get_first_wednesday)

    # Format to match existing ADP data structure
    result = pd.DataFrame({
        'date': df['DateTime'],
        'series_name': 'ADP_actual',
        'value': df['change'],  # Raw employment change (not divided by 1000)
        'release_date': df['release_date'],
        'series_type': 'adp'
    })

    return result


def main() -> None:
    """
    Load ADP employment data from historical CSV file.

    Web scraping is disabled due to Cloudflare "are you human" verification.
    Uses us-private-employment.csv as the sole data source.
    """
    logger.info("Starting ADP data load")

    # Check if data already exists
    if CLEAN_ADP_PARQUET.exists():
        existing_data = pd.read_parquet(CLEAN_ADP_PARQUET)
        print(f"✓ ADP data already exists: {len(existing_data)} rows", flush=True)
        print(f"  Date range: {existing_data['date'].min().date()} to {existing_data['date'].max().date()}", flush=True)
        logger.info(f"ADP data already exists, skipping")
        return

    # Load data from CSV
    csv_data = load_adp_from_csv()
    if csv_data.empty:
        logger.error("No CSV data available")
        raise RuntimeError("ADP data unavailable: CSV file not found or empty")

    # Save CSV data as the output parquet
    csv_data.to_parquet(CLEAN_ADP_PARQUET, index=False)
    print(f"✓ Saved {len(csv_data)} rows from historical CSV", flush=True)
    print(f"  Date range: {csv_data['date'].min().date()} to {csv_data['date'].max().date()}", flush=True)

    logger.info("✓ ADP data load complete")


if __name__ == "__main__":
    main()
