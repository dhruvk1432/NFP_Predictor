"""
NOAA Storm Events Pipeline
===========================
Downloads NOAA storm events data, aggregates to state-monthly features with
inflation-adjusted damages, combines into a master file, and creates
NFP-weighted national aggregate snapshots.

Merges: Load_Data/load_noaa_data.py + Prepare_Data/create_noaa_master.py + Prepare_Data/create_noaa_weighted.py

Pipeline stages:
    1. load_noaa_data()                  - Download & aggregate storm events by state/month
    2. create_noaa_master()              - Combine state + US data into long-format master
    3. create_noaa_weighted_snapshots()  - Build NFP-weighted national snapshots

Output:
    - State files:    DATA_PATH/Exogenous_data/NOAA_data/{STATE}_NOAA_data.parquet
    - US aggregate:   DATA_PATH/Exogenous_data/NOAA_data/US_NOAA_data.parquet
    - Master file:    DATA_PATH/Exogenous_data/NOAA_data/NOAA_master.parquet
    - Final output:   DATA_PATH/Exogenous_data/noaa_weighted_snapshots/decades/...

Requires:
    pip install requests pandas beautifulsoup4 tqdm python-dateutil pyarrow fredapi numpy
"""

# =====================================================================
# IMPORTS
# =====================================================================
import os
import re
import gzip
import io
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from fredapi import Fred

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import FRED_API_KEY, DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
from Data_ETA_Pipeline.fred_employment_pipeline import apply_nfp_relative_adjustment

# =====================================================================
# LOGGER
# =====================================================================
logger = setup_logger(__file__, TEMP_DIR)

# =====================================================================
# CONSTANTS & USER PARAMETERS
# =====================================================================

# Date range
START_DATE = START_DATE
END_DATE   = END_DATE

# This is no longer used for masking, but kept in case you want it later
REAL_DAMAGE_THRESHOLD = 0

# How many months after the end of an event month do we assume
# the data are fully known (for 'known_by_month_end')?
LAG_MONTHS = 3

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

OUTPUT_FOLDER = DATA_PATH / "Exogenous_data" / "NOAA_data"

# FRED API key (can also be set via environment variable FRED_API_KEY)
FRED_API_KEY = FRED_API_KEY

# All 51 states (50 states + DC)
ALL_STATES = [
    'ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO',
    'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO',
    'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA',
    'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI',
    'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY',
    'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO',
    'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA',
    'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA',
    'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING', 'DISTRICT OF COLUMBIA'
]

# State name to FRED code mapping (from NOAA format to 2-letter codes)
STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC'
}

# Expected damage/injury metrics
EXPECTED_METRICS = [
    'total_damage_real',
    'total_property_damage_real',
    'total_crop_damage_real',
    'deaths_direct',
    'deaths_indirect',
    'injuries_direct',
    'injuries_indirect'
]

# NOAA master / weighted paths
NOAA_DATA_DIR = DATA_PATH / "Exogenous_data" / "NOAA_data"
US_NOAA_FILE = NOAA_DATA_DIR / "US_NOAA_data.parquet"
NOAA_MASTER_FILE = NOAA_DATA_DIR / "NOAA_master.parquet"
NOAA_MASTER_PATH = DATA_PATH / "Exogenous_data" / "NOAA_data" / "NOAA_master.parquet"
NOAA_WEIGHTED_DIR = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / "decades"


# #####################################################################
# SECTION 1: NOAA DATA DOWNLOAD & STATE-LEVEL AGGREGATION
#             (from Load_Data/load_noaa_data.py)
# #####################################################################

# ---------------------------------------------------------------------
# Helpers (download & event-level)
# ---------------------------------------------------------------------

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")

def get_directory_listing(base_url: str) -> List[str]:
    """
    Fetch the HTML index for the stormevents csvfiles directory and
    return a list of filenames found there.
    """
    resp = requests.get(base_url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    filenames = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href:
            filenames.append(href)
    return filenames

def map_year_to_details_filename(filenames: List[str]) -> Dict[int, str]:
    """
    Given a list of filenames from the NOAA directory, return a dict:
        year -> StormEvents_details-ftp_v1.0_d{year}_cYYYYMMDD.csv.gz
    """
    pattern = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz")
    year_to_file = {}
    for name in filenames:
        m = pattern.match(name)
        if m:
            year = int(m.group(1))
            # If multiple per year ever exist, keep the latest or first; here we just take one
            year_to_file[year] = name
    return year_to_file

def safe_get_col(df: pd.DataFrame, base_name: str) -> str:
    """
    Find a column in df matching base_name with common case variants.
    Raises KeyError if none found.
    """
    candidates = [
        base_name,
        base_name.upper(),
        base_name.lower(),
        base_name.title(),
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find column for {base_name} in {list(df.columns)}")

def add_begin_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct an event 'begin_datetime' column from BEGIN_YEARMONTH, BEGIN_DAY, BEGIN_TIME.
    Some rows may be invalid; they will get NaT.
    """
    col_y = safe_get_col(df, "BEGIN_YEARMONTH")
    col_d = safe_get_col(df, "BEGIN_DAY")
    col_t = safe_get_col(df, "BEGIN_TIME")

    # Zero-pad everything to safe lengths
    ystr = df[col_y].astype(str).str.zfill(6)   # YYYYMM
    dstr = df[col_d].astype(str).str.zfill(2)   # DD
    tstr = df[col_t].astype(str).str.zfill(4)   # hhmm

    # Construct full datetime strings: YYYY-MM-DD hh:mm
    datestr = (
        ystr.str.slice(0, 4) + "-" +
        ystr.str.slice(4, 6) + "-" +
        dstr + " " +
        tstr.str.slice(0, 2) + ":" +
        tstr.str.slice(2, 4)
    )

    df["begin_datetime"] = pd.to_datetime(datestr, errors="coerce")
    return df

def download_and_filter_year(
    year: int,
    filename: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Download a single year's StormEvents_details CSV.GZ, filter by begin_datetime,
    and return the filtered DataFrame.
    """
    url = BASE_URL + filename
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # Decompress in-memory
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        df = pd.read_csv(gz, low_memory=False)

    # Add datetime column and filter
    df = add_begin_datetime_column(df)
    mask = (df["begin_datetime"] >= start_dt) & (df["begin_datetime"] <= end_dt)
    df = df.loc[mask].copy()

    return df

# ---------------------------------------------------------------------
# Helpers (CPI / inflation)
# ---------------------------------------------------------------------

def load_cpi_series(start_date: str) -> pd.DataFrame:
    """
    Load CPIAUCSL series from FRED API starting at given date.

    Uses env var FRED_API_KEY or script-level FRED_API_KEY.
    Returns DataFrame with columns:
        date (Timestamp)
        cpi  (float index level)
        month (Timestamp, month start)
    """
    api_key = os.getenv("FRED_API_KEY") or FRED_API_KEY
    if not api_key:
        raise RuntimeError("No FRED API key found in environment or script variable.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError("No CPI observations returned from FRED API.")

    rows = []
    for o in obs:
        date_str = o["date"]  # 'YYYY-MM-DD'
        value_str = o["value"]
        try:
            val = float(value_str)
        except ValueError:
            continue
        rows.append({"date": pd.to_datetime(date_str), "cpi": val})

    cpi_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Month timestamp for matching
    cpi_df["month"] = cpi_df["date"].dt.to_period("M").dt.to_timestamp()
    return cpi_df

# ---------------------------------------------------------------------
# Helpers (aggregation)
# ---------------------------------------------------------------------

def parse_damage_value(val) -> float:
    """
    Parse NOAA damage strings like '25K', '1.5M', '0', '' into
    a float in dollars (nominal).
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0

    # If it's purely numeric, treat as plain dollars
    if s.replace(".", "", 1).isdigit():
        try:
            return float(s)
        except ValueError:
            return 0.0

    mag = s[-1].upper()
    num_str = s[:-1]
    try:
        base = float(num_str)
    except ValueError:
        return 0.0

    multiplier = {
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "H": 1e2,
    }.get(mag, 1.0)

    return base * multiplier

def month_end(dt: pd.Timestamp) -> pd.Timestamp:
    """
    Return last day of the month for a datetime-like object.
    """
    ts = pd.Timestamp(dt)
    return ts.to_period("M").to_timestamp("M")

def calculate_noaa_release_date(event_date: pd.Timestamp, lag_days: int = 75, nfp_offset_days: float = None) -> pd.Timestamp:
    """
    Calculate NOAA release date based on documented 75-day lag.

    Per NOAA NCEI documentation:
    "The National Centers for Environmental Information (NCEI) regularly receives
    Storm Data from the National Weather Service (NWS) approximately 75 days after
    the end of a data month. (Ex: The January data month is usually available on
    or around April 15th)"

    ENHANCEMENT: If nfp_offset_days provided, apply NFP-relative adjustment to
    maintain historical timing consistency relative to NFP releases.

    Args:
        event_date: Event month (first day of month)
        lag_days: Days after month-end when data is available (default 75)
        nfp_offset_days: Optional median offset from NFP (for consistency enhancement)

    Returns:
        Estimated release date (month-end + lag_days, optionally NFP-adjusted)

    Example:
        Event month: 2020-01-01 (January 2020)
        Month end: 2020-01-31
        Release: 2020-01-31 + 75 days = 2020-04-15
        (optionally adjusted relative to NFP release for that month)
    """
    # Get last day of event month
    month_end = event_date.to_period('M').to_timestamp('M')
    # Add lag days (base estimate)
    base_release = month_end + pd.Timedelta(days=lag_days)

    # Apply NFP-relative adjustment if provided
    # INT1: Uses apply_nfp_relative_adjustment imported at module level
    if nfp_offset_days is not None:
        event_month = event_date.replace(day=1)
        adjusted_release = apply_nfp_relative_adjustment(
            event_month=event_month,
            base_release_date=base_release,
            median_offset_days=nfp_offset_days,
            use_adjustment=True
        )
        return adjusted_release
    else:
        return base_release


def calculate_first_friday(dt: pd.Timestamp, months_ahead: int = 1) -> pd.Timestamp:
    """
    Calculate the First Friday of the Nth month after the given date.
    Used for state employment data which follows BLS release schedule.

    Args:
        dt: Reference date (event month)
        months_ahead: Number of months to add (default 1 for BLS)

    Returns:
        First Friday of the target month

    Example:
        Event in January 2020 -> First Friday of February 2020
        - February 1, 2020 is a Saturday (weekday=5)
        - Days to Friday: (4 - 5 + 7) % 7 = 6
        - First Friday: February 7, 2020
    """
    # Move to target month
    target_month = dt + relativedelta(months=months_ahead)
    # First day of that month
    first_of_month = target_month.replace(day=1)
    # Calculate days until Friday (weekday 4)
    # Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
    days_to_friday = (4 - first_of_month.weekday() + 7) % 7
    # Special case: if 1st is already a Friday, days_to_friday=0
    if days_to_friday == 0:
        first_friday = first_of_month
    else:
        first_friday = first_of_month + pd.Timedelta(days=days_to_friday)

    return first_friday

def aggregate_to_state_monthly(
    df: pd.DataFrame,
    lag_months: int = 3,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Aggregate event-level NOAA data to monthly state-level features,
    inflation-adjust damages to today's CPI, and then expand to a full
    STATE x month grid from start_dt to end_dt, filling zeros where no events.
    """
    if "begin_datetime" not in df.columns:
        df = add_begin_datetime_column(df)
    df = df[~df["begin_datetime"].isna()].copy()

    if "STATE" not in df.columns:
        raise ValueError("Expected a 'STATE' column in the NOAA details file.")

    # Parse damage columns (nominal USD)
    for col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].apply(parse_damage_value)

    # Ensure fatalities / injuries columns
    for col in [
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Event month: first day of month the event begins
    df["event_month"] = df["begin_datetime"].dt.to_period("M").dt.to_timestamp()

    count_col = "EVENT_ID" if "EVENT_ID" in df.columns else "STATE"

    grouped = (
        df.groupby(["STATE", "event_month"], as_index=False)
          .agg(
              storm_count=(count_col, "count"),
              total_property_damage=("DAMAGE_PROPERTY", "sum"),
              total_crop_damage=("DAMAGE_CROPS", "sum"),
              deaths_direct=("DEATHS_DIRECT", "sum"),
              deaths_indirect=("DEATHS_INDIRECT", "sum"),
              injuries_direct=("INJURIES_DIRECT", "sum"),
              injuries_indirect=("INJURIES_INDIRECT", "sum"),
          )
    )

    grouped["total_deaths"] = grouped["deaths_direct"] + grouped["deaths_indirect"]
    grouped["total_injuries"] = grouped["injuries_direct"] + grouped["injuries_indirect"]
    grouped["total_damage"] = grouped["total_property_damage"] + grouped["total_crop_damage"]

    # -----------------------------------------------------------------
    # Build full STATE x month grid between start_dt and end_dt
    # -----------------------------------------------------------------
    if start_dt is None:
        start_dt = df["begin_datetime"].min()
    if end_dt is None:
        end_dt = df["begin_datetime"].max()

    start_month = pd.Timestamp(start_dt).to_period("M").to_timestamp()
    end_month = pd.Timestamp(end_dt).to_period("M").to_timestamp()
    month_range = pd.period_range(start=start_month, end=end_month, freq="M").to_timestamp()

    # CRITICAL FIX: Use ALL 51 states (50 states + DC) regardless of whether they had events
    # This ensures complete zero-filling: states with no events will have all zeros
    full_index = pd.MultiIndex.from_product(
        [ALL_STATES, month_range],
        names=["STATE", "event_month"],
    )
    full = pd.DataFrame(index=full_index).reset_index()

    # Merge the aggregated data onto the full grid
    full = full.merge(
        grouped,
        on=["STATE", "event_month"],
        how="left",
    )

    # Fill numeric NaNs with zeros where no events happened
    num_cols = [
        "storm_count",
        "total_property_damage",
        "total_crop_damage",
        "total_damage",
        "deaths_direct",
        "deaths_indirect",
        "total_deaths",
        "injuries_direct",
        "injuries_indirect",
        "total_injuries",
    ]
    for col in num_cols:
        full[col] = full[col].fillna(0)

    # -----------------------------------------------------------------
    # Inflation adjustment to "today's" dollars using CPIAUCSL
    # -----------------------------------------------------------------
    # Load CPI from earliest month in our grid
    cpi_df = load_cpi_series(start_date=start_month.strftime("%Y-%m-%d"))

    # Restrict CPI to our month_range
    cpi_df = cpi_df[cpi_df["month"].between(start_month, end_month)]
    if cpi_df.empty:
        raise RuntimeError("CPI data is empty or does not cover the requested date range.")

    cpi_today = cpi_df["cpi"].max()  # latest CPI in sample (approx "today")
    cpi_df["inflation_factor"] = cpi_today / cpi_df["cpi"]
    # NOAA-specific: Use documented 75-day lag from month-end
    # Per NOAA NCEI: Data available ~75 days after month end (e.g., Jan data -> Apr 15)
    cpi_df["release_date"] = cpi_df["month"].apply(
        lambda m: calculate_noaa_release_date(m, lag_days=75)
    )

    # Merge CPI info by month into the full grid
    full = full.merge(
        cpi_df[["month", "cpi", "inflation_factor", "release_date"]],
        left_on="event_month",
        right_on="month",
        how="left",
    ).drop(columns=["month"])

    # In case of any gaps, fill cpi and inflation_factor forward/backward
    full["cpi"] = full["cpi"].ffill().bfill()
    full["inflation_factor"] = full["inflation_factor"].ffill().bfill()
    full["release_date"] = full["release_date"].ffill().bfill()

    # Real (today's dollars) damage
    full["total_property_damage_real"] = full["total_property_damage"] * full["inflation_factor"]
    full["total_crop_damage_real"] = full["total_crop_damage"] * full["inflation_factor"]
    full["total_damage_real"] = full["total_damage"] * full["inflation_factor"]

    # Final column ordering
    cols = [
        "STATE",
        "event_month",
        "storm_count",
        "total_property_damage",
        "total_crop_damage",
        "total_damage",
        "total_property_damage_real",
        "total_crop_damage_real",
        "total_damage_real",
        "deaths_direct",
        "deaths_indirect",
        "total_deaths",
        "injuries_direct",
        "injuries_indirect",
        "total_injuries",
        "inflation_factor",
        "release_date",
    ]

    full = full[cols].sort_values(["STATE", "event_month"]).reset_index(drop=True)
    return full

# ---------------------------------------------------------------------
# load_noaa_data() — main entry for Section 1
# ---------------------------------------------------------------------

def load_noaa_data():
    start_dt = parse_date(START_DATE)
    end_dt   = parse_date(END_DATE)

    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    # Skip-if-exists: Check if NOAA data already exists
    us_data_path = OUTPUT_FOLDER / "US_NOAA_data.parquet"
    if us_data_path.exists():
        existing_data = pd.read_parquet(us_data_path)
        print(f"✓ NOAA data already exists: {len(existing_data)} months", flush=True)
        print(f"  Date range: {existing_data.index.min().date()} to {existing_data.index.max().date()}", flush=True)
        logger.info("NOAA data already exists, skipping")
        return

    logger.info("Starting NOAA data download")
    filenames = get_directory_listing(BASE_URL)
    year_to_file = map_year_to_details_filename(filenames)

    min_year = start_dt.year
    max_year = end_dt.year

    dfs = []
    for year in tqdm(range(min_year, max_year + 1), desc="Downloading years"):
        if year not in year_to_file:
            logger.warning(f"No file found for year {year}, skipping")
            continue
        fname = year_to_file[year]
        df_year = download_and_filter_year(year, fname, start_dt, end_dt)
        if not df_year.empty:
            dfs.append(df_year)

    if not dfs:
        logger.error("No data found in the specified date range")
        return

    full_df = pd.concat(dfs, ignore_index=True)

    # Sort by begin_datetime for cleanliness
    full_df = full_df.sort_values("begin_datetime").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Aggregate to monthly state-level features (real USD) on full grid
    # -----------------------------------------------------------------
    state_monthly = aggregate_to_state_monthly(
        full_df,
        lag_months=LAG_MONTHS,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # -----------------------------------------------------------------
    # Write one Parquet file per state
    # -----------------------------------------------------------------
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for state, df_state in state_monthly.groupby("STATE"):
        df_state = df_state.drop(columns=["STATE"]).set_index("event_month").sort_index()
        fname = f"{state}_NOAA_data.parquet"
        out_path = os.path.join(OUTPUT_FOLDER, fname)
        df_state.to_parquet(out_path)

    # -----------------------------------------------------------------
    # Also write aggregate US-level file (full month grid)
    # -----------------------------------------------------------------
    us_agg = (
        state_monthly
        .groupby("event_month", as_index=False)
        .agg(
            storm_count=("storm_count", "sum"),
            total_property_damage=("total_property_damage", "sum"),
            total_crop_damage=("total_crop_damage", "sum"),
            total_damage=("total_damage", "sum"),
            total_property_damage_real=("total_property_damage_real", "sum"),
            total_crop_damage_real=("total_crop_damage_real", "sum"),
            total_damage_real=("total_damage_real", "sum"),
            deaths_direct=("deaths_direct", "sum"),
            deaths_indirect=("deaths_indirect", "sum"),
            total_deaths=("total_deaths", "sum"),
            injuries_direct=("injuries_direct", "sum"),
            injuries_indirect=("injuries_indirect", "sum"),
            total_injuries=("total_injuries", "sum"),
            inflation_factor=("inflation_factor", "first"),
            release_date=("release_date", "first"),
        )
    )

    us_agg = us_agg.set_index("event_month").sort_index()
    us_path = os.path.join(OUTPUT_FOLDER, "US_NOAA_data.parquet")
    us_agg.to_parquet(us_path)

    logger.info("✓ NOAA data download complete")


# #####################################################################
# SECTION 2: NOAA MASTER FILE CREATION
#             (from Prepare_Data/create_noaa_master.py)
# #####################################################################

def create_noaa_master():
    """
    Create NOAA master file combining state-level and US national data.

    Output format:
    - date: event month
    - series_name: metric_STATE (e.g., storm_count_ALABAMA, deaths_direct_US)
    - value: numeric value
    - release_date: estimated release date (month-end + 75 days per NOAA NCEI)
    """
    logger.info("Creating NOAA master file...")

    # 1. Load and process US national data
    logger.info("\n1. Processing US national data...")
    df_us = pd.read_parquet(US_NOAA_FILE)
    df_us = df_us.reset_index()  # event_month becomes column
    df_us = df_us.rename(columns={'event_month': 'date'})

    # Convert to long format with series names like "storm_count_US"
    # UPDATED: Use release_date instead of known_by_month_end
    value_cols = [col for col in df_us.columns if col not in ['date', 'inflation_factor', 'release_date']]

    us_long = df_us.melt(
        id_vars=['date', 'release_date'],
        value_vars=value_cols,
        var_name='metric',
        value_name='value'
    )
    us_long['series_name'] = us_long['metric'] + '_US'
    us_long = us_long[['date', 'series_name', 'value', 'release_date']]

    logger.info(f"  US data: {len(us_long)} rows, {us_long['series_name'].nunique()} series")

    # 2. Load and process state-level data
    logger.info("\n2. Processing state-level data...")
    state_files = list(NOAA_DATA_DIR.glob("*_NOAA_data.parquet"))
    state_files = [f for f in state_files if f.name != 'US_NOAA_data.parquet']

    logger.info(f"  Found {len(state_files)} state files")

    all_states = []
    for file in state_files:
        state = file.stem.replace('_NOAA_data', '')
        df_state = pd.read_parquet(file)
        df_state = df_state.reset_index()  # event_month becomes column
        df_state = df_state.rename(columns={'event_month': 'date'})

        # Convert to long format
        state_long = df_state.melt(
            id_vars=['date', 'release_date'],
            value_vars=value_cols,
            var_name='metric',
            value_name='value'
        )
        state_long['series_name'] = state_long['metric'] + '_' + state
        state_long = state_long[['date', 'series_name', 'value', 'release_date']]

        all_states.append(state_long)

    df_states = pd.concat(all_states, ignore_index=True)
    logger.info(f"  State data: {len(df_states)} rows, {df_states['series_name'].nunique()} series")

    # 3. Combine US and state data
    logger.info("\n3. Combining US and state data...")
    df_master = pd.concat([us_long, df_states], ignore_index=True)
    df_master = df_master.sort_values(['date', 'series_name']).reset_index(drop=True)

    logger.info(f"  Combined: {len(df_master)} rows")
    logger.info(f"  Series: {df_master['series_name'].nunique()}")
    logger.info(f"  Date range: {df_master['date'].min()} to {df_master['date'].max()}")

    # 4. Save master file
    df_master.to_parquet(NOAA_MASTER_FILE, index=False)
    logger.info(f"\n✓ Saved NOAA master: {NOAA_MASTER_FILE}")

    # 5. Show sample
    logger.info(f"\nSample US series:")
    us_series = sorted([s for s in df_master['series_name'].unique() if s.endswith('_US')])
    logger.info(f"  {us_series[:10]}")

    logger.info(f"\nSample state series:")
    state_series = sorted([s for s in df_master['series_name'].unique() if not s.endswith('_US')])
    logger.info(f"  {state_series[:10]}")

    return df_master


# #####################################################################
# SECTION 3: NFP-WEIGHTED NATIONAL AGGREGATES
#             (from Prepare_Data/create_noaa_weighted.py)
# #####################################################################

def download_state_employment_vintages(fred: Fred, end_date: str = END_DATE) -> pd.DataFrame:
    """
    Download ALL vintages for all state employment series once (as of end_date).
    Returns DataFrame with columns: state_code, date, value, realtime_start
    Similar pattern to fred_snapshots.py
    """
    as_of_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    # Update cache name to reflect NSA data
    cache_path = DATA_PATH / "Exogenous_data" / "noaa_weighted_snapshots" / f"state_employment_vintages_nsa_{as_of_str}.parquet"

    if cache_path.exists():
        logger.info(f"Loading state employment vintages from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    states = list(STATE_NAME_TO_CODE.values())
    # Use NAN suffix for Non-Seasonally Adjusted data (goes back further)
    fred_codes = {state: f"{state}NAN" for state in states}

    logger.info(f"Downloading ALL vintages for 51 state employment series (NSA) as of {as_of_str}")

    all_vintages = []

    for i, (state, code) in enumerate(fred_codes.items(), 1):
        try:
            # Get ALL vintages as of end_date
            vintage_df = fred.get_series_as_of_date(code, as_of_date=as_of_str)

            if vintage_df.empty:
                logger.warning(f"[{i}/51] No data for {state} ({code})")
                continue

            # Transform to our format
            vintage_df['state_code'] = state
            vintage_df['date'] = pd.to_datetime(vintage_df['date'])
            vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
            vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

            all_vintages.append(vintage_df[['state_code', 'date', 'value', 'realtime_start']])

            logger.info(f"[{i}/51] Downloaded {state} ({code}): {len(vintage_df)} vintages")

            # Rate limiting: sleep every 10 series
            if i % 10 == 0 and i < len(fred_codes):
                logger.info(f"Rate limiting: sleeping 5 seconds...")
                time.sleep(5)

        except Exception as e:
            logger.error(f"[{i}/51] Error fetching {state} employment: {e}")
            continue

    if not all_vintages:
        raise ValueError("No state employment data retrieved")

    combined = pd.concat(all_vintages, ignore_index=True)
    logger.info(f"Downloaded {len(combined)} total vintage records for {combined['state_code'].nunique()} states")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    logger.info(f"Saved vintage cache to {cache_path}")

    return combined


def get_state_employment_weights(vintages_df: pd.DataFrame, snap_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate employment-share weights from vintages as of snap_date.
    Uses point-in-time filtering: only data released by snap_date.

    If snap_date is before the earliest available vintage (e.g. pre-2005),
    falls back to the earliest vintage as a best-effort proxy.
    """
    # Check if we have any vintage by snap_date
    earliest_vintage = vintages_df['realtime_start'].min()

    if snap_date < earliest_vintage:
        # Fallback: Use earliest vintage
        logger.warning(f"Snapshot {snap_date.date()} predates earliest vintage ({earliest_vintage.date()}). Using earliest vintage as proxy.")
        known_df = vintages_df[vintages_df['realtime_start'] == earliest_vintage].copy()
    else:
        # Normal case: Filter to vintages known BEFORE snap_date (strict <)
        # Changed from <= to strict < to prevent same-day data leakage
        known_df = vintages_df[vintages_df['realtime_start'] < snap_date].copy()

    if known_df.empty:
        raise ValueError(f"No state employment data available as of {snap_date}")

    # For each (state_code, date), keep the latest vintage known by snap_date
    known_df = known_df.sort_values(['state_code', 'date', 'realtime_start'])
    latest_vintage = known_df.drop_duplicates(['state_code', 'date'], keep='last')

    # Get employment for snap_date's month (or most recent prior)
    snap_month = snap_date.to_period('M')

    state_employment = {}
    lags = []

    for state in latest_vintage['state_code'].unique():
        state_data = latest_vintage[latest_vintage['state_code'] == state]

        # Try to get exact month
        month_data = state_data[state_data['date'].dt.to_period('M') == snap_month]

        if not month_data.empty:
            row = month_data.iloc[-1]
            employment = row['value']
            state_employment[state] = float(employment)
            lags.append((row['realtime_start'] - row['date']).days)
        else:
            # Fall back to most recent available before snap_date
            prior_data = state_data[state_data['date'] <= snap_date]
            if not prior_data.empty:
                row = prior_data.sort_values('date').iloc[-1]
                employment = row['value']
                state_employment[state] = float(employment)
                lags.append((row['realtime_start'] - row['date']).days)

    if not state_employment:
        raise ValueError(f"No state employment data retrieved for {snap_date}")

    # Log lag stats
    if lags:
        avg_lag = sum(lags) / len(lags)
        min_lag = min(lags)
        max_lag = max(lags)
        logger.info(f"Vintage Lag Stats for {snap_date.date()}: Avg={avg_lag:.1f} days, Min={min_lag}, Max={max_lag}")

    # Calculate weights
    total_employment = sum(state_employment.values())
    weights_dict = {state: emp / total_employment for state, emp in state_employment.items()}

    # Create DataFrame
    weights_df = pd.DataFrame({
        'state_code': list(weights_dict.keys()),
        'employment': [state_employment[s] for s in weights_dict.keys()],
        'weight': list(weights_dict.values())
    })

    # Verify weights sum to ~1.0
    weight_sum = weights_df['weight'].sum()
    if not np.isclose(weight_sum, 1.0):
        logger.warning(f"Weights sum to {weight_sum}, expected 1.0")

    logger.info(f"Snapshot {snap_date.date()}: Calculated weights for {len(weights_df)} states (sum={weight_sum:.6f})")

    return weights_df


def load_and_parse_noaa_master() -> pd.DataFrame:
    """
    Load NOAA master file and parse series names into metric and state.
    Filter to 51 valid states only.
    """
    if not NOAA_MASTER_PATH.exists():
        raise FileNotFoundError(f"NOAA master file not found: {NOAA_MASTER_PATH}")

    logger.info(f"Loading NOAA data from {NOAA_MASTER_PATH}")
    noaa_full = pd.read_parquet(NOAA_MASTER_PATH)

    # Parse series_name: {metric}_{STATE}
    # Example: deaths_direct_ALABAMA, total_property_damage_real_WYOMING
    noaa_full['state_name'] = noaa_full['series_name'].str.rsplit('_', n=1).str[-1]
    noaa_full['metric'] = noaa_full['series_name'].str.rsplit('_', n=1).str[0]

    # Filter to 51 valid states
    valid_states = list(STATE_NAME_TO_CODE.keys())
    noaa_states = noaa_full[noaa_full['state_name'].isin(valid_states)].copy()

    # Filter to 6 required metrics only
    # Exclude: total_damage_real (it's property+crop, would be collinear), nominal, derived, storm_count
    required_metrics = [
        'storm_count',                 # Storm count (independent)
        'total_property_damage_real',  # Property damage (independent)
        'total_crop_damage_real',      # Agricultural damage (independent)
        'deaths_direct',               # Direct deaths
        'deaths_indirect',             # Indirect deaths
        'injuries_direct',             # Direct injuries
        'injuries_indirect'            # Indirect injuries
    ]
    noaa_states = noaa_states[noaa_states['metric'].isin(required_metrics)].copy()

    logger.info(f"Filtered NOAA data: {len(noaa_full)} → {len(noaa_states)} rows (51 states, 6 metrics)")
    logger.info(f"Unique states: {noaa_states['state_name'].nunique()}")
    logger.info(f"Unique metrics: {noaa_states['metric'].nunique()}")
    logger.info(f"Metrics: {sorted(noaa_states['metric'].unique())}")

    return noaa_states


def create_weighted_national_aggregates(
    noaa_states: pd.DataFrame,
    weights: pd.DataFrame,
    snap_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Apply NFP employment weights to state damage data and aggregate to national level.
    Uses release_date for point-in-time filtering.
    CRITICAL: Applies log1p transformation to the final aggregated result.

    Args:
        noaa_states: State-level NOAA data with columns: date, metric, value, release_date, state_name
        weights: Employment weights with columns: state_code, weight
        snap_date: NFP release date to align snapshot with

    Returns:
        DataFrame with columns: date, series_name, value, release_date
    """
    # Filter to data known by snap_date
    noaa_states['date'] = pd.to_datetime(noaa_states['date'])
    noaa_states['release_date'] = pd.to_datetime(noaa_states['release_date'])

    # Point-in-time filter: only include data released BEFORE snap_date (strict <)
    # Changed from <= to strict < to prevent same-day data leakage
    noaa_filtered = noaa_states[noaa_states['release_date'] < snap_date].copy()

    logger.info(f"Point-in-time filter: {len(noaa_states)} → {len(noaa_filtered)} rows (released by {snap_date.date()})")

    # Add state codes
    noaa_filtered['state_code'] = noaa_filtered['state_name'].map(STATE_NAME_TO_CODE)

    # Merge with weights
    noaa_weighted = noaa_filtered.merge(weights[['state_code', 'weight']], on='state_code', how='inner')

    logger.info(f"Merged NOAA data with weights: {len(noaa_weighted)} rows")

    # Apply weights: weighted_value = value * employment_weight
    noaa_weighted['weighted_value'] = noaa_weighted['value'] * noaa_weighted['weight']

    # Aggregate to national level: sum across states for each (date, metric)
    weighted_national = noaa_weighted.groupby(['date', 'metric'])['weighted_value'].sum().reset_index()

    # Get release_date for each event date (should be consistent across states)
    release_dates = noaa_filtered.groupby('date')['release_date'].first().reset_index()
    weighted_national = weighted_national.merge(release_dates, on='date', how='left')

    # --- TRANSFORM LOGIC ---
    # Apply log1p to the aggregated SUM. This stabilizes the feature while
    # preserving the relative economic impact of different periods.
    weighted_national['weighted_value'] = np.log1p(weighted_national['weighted_value'])

    # Rename for clarity - append _log so downstream code knows this is transformed
    weighted_national['series_name'] = weighted_national['metric'] + '_weighted_log'

    weighted_national = weighted_national.rename(columns={'weighted_value': 'value'})
    weighted_national = weighted_national[['date', 'series_name', 'value', 'release_date']]

    logger.info(f"Created weighted national aggregates (LOG TRANSFORMED): {len(weighted_national)} rows")
    logger.info(f"Unique metrics: {weighted_national['series_name'].nunique()}")
    logger.info(f"Date range: {weighted_national['date'].min()} to {weighted_national['date'].max()}")

    return weighted_national


def load_nfp_release_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load NFP target file to extract actual NFP release dates.
    These dates determine when each snapshot should be created.

    Returns:
        DataFrame with columns: ds (event date), release_date (NFP release)
    """
    nfp_path = DATA_PATH / "NFP_target" / "total_nsa_first_release.parquet"

    if not nfp_path.exists():
        raise FileNotFoundError(f"NFP target file not found: {nfp_path}")

    logger.info(f"Loading NFP release dates from {nfp_path}")
    nfp_df = pd.read_parquet(nfp_path)

    # Extract only ds and release_date columns
    nfp_releases = nfp_df[['ds', 'release_date']].copy()
    nfp_releases['ds'] = pd.to_datetime(nfp_releases['ds'])
    nfp_releases['release_date'] = pd.to_datetime(nfp_releases['release_date'])

    # Filter to date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    nfp_releases = nfp_releases[
        (nfp_releases['ds'] >= start_dt) & (nfp_releases['ds'] <= end_dt)
    ]

    logger.info(f"Loaded {len(nfp_releases)} NFP release dates")
    logger.info(f"Date range: {nfp_releases['ds'].min().date()} to {nfp_releases['ds'].max().date()}")

    return nfp_releases


def create_noaa_weighted_snapshots(
    start_date: str = START_DATE,
    end_date: str = END_DATE
):
    """
    Create snapshots of NFP-weighted NOAA data aligned with NFP release dates.
    Each snapshot is point-in-time correct using release_date filtering.

    CRITICAL CHANGE: Snapshots are now aligned with NFP release dates instead of
    generic month-end dates. This ensures temporal consistency with the target variable.

    Downloads state employment vintages ONCE, then processes all snapshots.
    Similar pattern to fred_snapshots.py to avoid rate limiting.
    """
    # Initialize FRED API
    fred = Fred(api_key=FRED_API_KEY)

    # Load NFP release dates to determine snapshot schedule
    logger.info("=" * 80)
    logger.info("STEP 1: Loading NFP release dates")
    logger.info("=" * 80)
    nfp_releases = load_nfp_release_dates(start_date, end_date)

    # Download ALL state employment vintages ONCE (as of end_date)
    logger.info("=" * 80)
    logger.info("STEP 2: Downloading state employment vintages (ONCE)")
    logger.info("=" * 80)
    try:
        employment_vintages = download_state_employment_vintages(fred, end_date=end_date)
    except Exception as e:
        logger.error(f"Failed to download state employment vintages: {e}")
        raise

    # Load full NOAA data once
    logger.info("=" * 80)
    logger.info("STEP 3: Loading NOAA master data")
    logger.info("=" * 80)
    noaa_states = load_and_parse_noaa_master()

    # Determine earliest available employment date
    min_employment_date = employment_vintages['date'].min()
    logger.info(f"Earliest state employment data: {min_employment_date.date()}")

    logger.info("=" * 80)
    logger.info(f"STEP 4: Creating {len(nfp_releases)} NOAA weighted snapshots (aligned with NFP)")
    logger.info(f"From {nfp_releases['release_date'].min().date()} to {nfp_releases['release_date'].max().date()}")
    logger.info("=" * 80)

    for i, (idx, row) in enumerate(nfp_releases.iterrows(), 1):
        event_date = row['ds']
        snap_date = row['release_date']  # NFP release date

        year = event_date.year
        decade = f"{year // 10 * 10}s"
        month_str = event_date.strftime('%Y-%m')

        # Create directory structure
        save_dir = NOAA_WEIGHTED_DIR / decade / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{month_str}.parquet"

        # Skip if exists
        if save_path.exists():
            logger.info(f"[{i}/{len(nfp_releases)}] Snapshot exists for {month_str}, skipping")
            continue

        try:
            # Calculate employment weights from vintages (point-in-time as of snap_date)
            try:
                weights = get_state_employment_weights(employment_vintages, snap_date)
            except ValueError:
                # If no weights available (e.g. pre-1990), treat as missing
                # We simply don't create the weighted snapshot for this month
                # create_master_snapshots.py will handle missing files by skipping them
                if i % 12 == 0:  # Reduce log noise
                    logger.info(f"[{i}/{len(nfp_releases)}] No weights for {month_str} (pre-1990), skipping")
                continue

            # Create weighted aggregates (with point-in-time filtering using snap_date)
            weighted_national = create_weighted_national_aggregates(noaa_states, weights, snap_date)

            # Add snapshot_date column (NFP release date)
            weighted_national['snapshot_date'] = snap_date

            # Reorder columns to match requirements: date, series_name, value, snapshot_date, release_date
            weighted_national = weighted_national[['date', 'series_name', 'value', 'snapshot_date', 'release_date']]

            # Save
            weighted_national.to_parquet(save_path, index=False)

            logger.info(f"[{i}/{len(nfp_releases)}] Saved {month_str}: {len(weighted_national)} rows (snapshot={snap_date.date()})")

        except Exception as e:
            logger.error(f"Error creating snapshot for {month_str}: {e}")
            traceback.print_exc()
            continue

    logger.info("=" * 80)
    logger.info("NOAA weighted snapshot generation complete")
    logger.info("=" * 80)


# #####################################################################
# SECTION 4: UNIFIED ENTRY POINT
# #####################################################################

def main():
    """
    Run the full NOAA pipeline:
        1. Download & aggregate storm events by state/month
        2. Combine state + US data into long-format master
        3. Build NFP-weighted national snapshots
    """
    logger.info("=" * 80)
    logger.info("NOAA STORM EVENTS PIPELINE — START")
    logger.info("=" * 80)

    # Stage 1: Download & aggregate
    logger.info("STAGE 1: Download & aggregate storm events by state/month")
    load_noaa_data()

    # Stage 2: Create master file
    logger.info("STAGE 2: Combine state + US data into long-format master")
    create_noaa_master()

    # Stage 3: NFP-weighted snapshots
    logger.info("STAGE 3: Build NFP-weighted national snapshots")
    create_noaa_weighted_snapshots()

    logger.info("=" * 80)
    logger.info("NOAA STORM EVENTS PIPELINE — COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
