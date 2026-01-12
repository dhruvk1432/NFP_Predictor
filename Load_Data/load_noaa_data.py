"""
Download NOAA Storm Events "details" data for a given date range and
aggregate to monthly state-level features, saving one Parquet file
per state with inflation-adjusted damages.

Requires:
    pip install requests pandas beautifulsoup4 tqdm python-dateutil pyarrow

Data sources:
    NOAA NCEI Storm Events Database bulk CSV:
        https://www.ncei.noaa.gov/stormevents/ftp.jsp
    CPI (CPIAUCSL) from FRED (St. Louis Fed) via API
        https://fred.stlouisfed.org/series/CPIAUCSL

Environment:
    export FRED_API_KEY=your_fred_api_key
"""

import os
import re
import gzip
import io
import sys
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import FRED_API_KEY, DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
# INT1: Import NFP utilities at module level for consistency
from nfp_relative_timing import apply_nfp_relative_adjustment

# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------
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
    print(f"  Downloading {year}: {url}")
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
        Event in January 2020 → First Friday of February 2020
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
    STATE × month grid from start_dt to end_dt, filling zeros where no events.
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
    # Build full STATE × month grid between start_dt and end_dt
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
    # Per NOAA NCEI: Data available ~75 days after month end (e.g., Jan data → Apr 15)
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
# Main script
# ---------------------------------------------------------------------

def load_noaa_data():
    start_dt = parse_date(START_DATE)
    end_dt   = parse_date(END_DATE)

    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    print("Fetching directory listing from NOAA...")
    filenames = get_directory_listing(BASE_URL)
    year_to_file = map_year_to_details_filename(filenames)

    min_year = start_dt.year
    max_year = end_dt.year

    dfs = []
    print(f"Downloading and filtering years {min_year}–{max_year}...")
    for year in tqdm(range(min_year, max_year + 1)):
        if year not in year_to_file:
            # Not all years may be present (e.g., partial coverage near the end)
            print(f"  WARNING: No details file found for year {year}, skipping.")
            continue
        fname = year_to_file[year]
        df_year = download_and_filter_year(year, fname, start_dt, end_dt)
        if not df_year.empty:
            dfs.append(df_year)

    if not dfs:
        print("No data found in the specified date range.")
        return

    full_df = pd.concat(dfs, ignore_index=True)

    # Sort by begin_datetime for cleanliness
    full_df = full_df.sort_values("begin_datetime").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Aggregate to monthly state-level features (real USD) on full grid
    # -----------------------------------------------------------------
    print("Aggregating to monthly state-level features (real USD)...")
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
        print(f"  Wrote {out_path} with {len(df_state)} rows")

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
    print(f"  Wrote {us_path} with {len(us_agg)} rows")

    print("Done.")

if __name__ == "__main__":
    load_noaa_data()
