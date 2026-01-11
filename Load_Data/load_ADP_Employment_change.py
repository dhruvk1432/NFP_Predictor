"""
ADP Non-Farm Employment Change - Key NFP Predictor (2 Days Early)

Scrapes ADP employment data and converts to long format for NBEATSx pipeline integration.
Output format matches FRED snapshots: date, series_name, value, release_date
"""

import time
import csv
import pandas as pd
import re
import sys
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# --- Paths ---
EXOG_ADP_DIR = DATA_PATH / "Exogenous_data" / "ADP_data"
EXOG_ADP_DIR.mkdir(parents=True, exist_ok=True)

RAW_ADP_CSV = EXOG_ADP_DIR / "ADP_Employment_Change_raw.csv"
CLEAN_ADP_PARQUET = EXOG_ADP_DIR / "ADP_Employment_Change.parquet"

# --- Configuration ---
URL = "https://www.investing.com/economic-calendar/adp-nonfarm-employment-change-1"
WAIT_SEC = 10


def init_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--incognito")
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--window-size=1366,768")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.page_load_strategy = "eager"

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(60)  # Reasonable timeout for page load
    return driver


def click_show_more(driver, wait_sec: int = 10, max_clicks: int = 50, max_no_change_attempts: int = 3) -> None:
    """Keep clicking 'Show more' until all history is loaded or max_clicks reached.

    Args:
        driver: Selenium webdriver instance
        wait_sec: Seconds to wait for button visibility
        max_clicks: Maximum number of clicks to attempt
        max_no_change_attempts: Stop after this many consecutive failed row additions
    """
    clicks = 0
    no_change_count = 0

    while clicks < max_clicks:
        try:
            # First check if there are any rows at all
            current_rows = driver.find_elements(By.CSS_SELECTOR, "#eventHistoryTable1 tbody tr")
            logger.info(f"Current row count: {len(current_rows)}")

            # Try to find the "Show more" button
            show_more = WebDriverWait(driver, wait_sec).until(
                EC.visibility_of_element_located((By.ID, "showMoreHistory1"))
            )

            if not show_more.is_displayed():
                logger.info("'Show more' button is hidden - all history loaded")
                break

            rows_before = len(current_rows)

            # Scroll and click
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", show_more)
            driver.execute_script("arguments[0].click();", show_more)

            clicks += 1
            logger.info(f"Clicked 'Show more' (click #{clicks})")

            # Wait for new rows with shorter timeout (reduced from 5 to 3 seconds)
            try:
                WebDriverWait(driver, 3).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, "#eventHistoryTable1 tbody tr")) > rows_before
                )
                logger.info(f"✓ New rows loaded")
                no_change_count = 0  # Reset counter on success
            except:
                no_change_count += 1
                logger.warning(f"No new rows appeared after clicking (attempt {no_change_count}/{max_no_change_attempts})")

                # If we've failed to load new rows N times in a row, stop trying
                if no_change_count >= max_no_change_attempts:
                    logger.info(f"Stopped after {no_change_count} consecutive failed attempts to load new rows")
                    break

        except Exception as e:
            logger.info(f"Stopped clicking 'Show more': {type(e).__name__}")
            break

    final_rows = driver.find_elements(By.CSS_SELECTOR, "#eventHistoryTable1 tbody tr")
    logger.info(f"Final row count: {len(final_rows)} (after {clicks} clicks)")


def parse_table(html: str) -> List[List[str]]:
    """Return the ADP table rows as lists of text values."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="eventHistoryTable1")
    if not table:
        logger.error("ADP Employment table not found in HTML")
        raise RuntimeError("Table not found – check page structure")
    
    rows = []
    for tr in table.tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        rows.append([c.get_text(strip=True) for c in cells[:5]])
    
    logger.info(f"Parsed {len(rows):,d} rows from the table")
    return rows


def save_csv(rows: List[List[str]]) -> None:
    """Save raw CSV to ADP_data directory."""
    RAW_ADP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_ADP_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Release Date", "Time", "Actual", "Forecast", "Previous"])
        writer.writerows(rows)
    logger.info(f"Saved raw CSV: {RAW_ADP_CSV}")


def format_adp_data() -> None:
    """
    Format raw ADP CSV into long-format parquet matching FRED snapshot structure.

    Reference month determination (priority logic):
    1. Explicit format (with parentheses): Use the month specified in parentheses
       Example: "Aug 30, 2023 (Aug)" → August 2023
    2. Implicit format (no parentheses): Use day-based rule
       - Early release (day <= 10): Previous month
         Example: "Apr 02, 2008" → March 2008
       - Late release (day >= 20): Current month
         Example: "Apr 30, 2008" → April 2008

    Output format:
    - date: reference month (determined by logic above)
    - series_name: 'ADP_actual', 'ADP_forecast', or 'ADP_previous'
    - value: employment change in thousands
    - release_date: when ADP was published
    - series_type: 'adp' for filtering/identification
    """
    # Read raw CSV
    df_raw = pd.read_csv(RAW_ADP_CSV)
    
    data = []
    for _, row in df_raw.iterrows():
        try:
            release_date_str = row["Release Date"]
            actual_str = row["Actual"]
            forecast_str = row["Forecast"]
            previous_str = row["Previous"]
            
            # Parse release date - handle two formats:
            # Explicit: "Aug 30, 2023 (Aug)" - month in parentheses (present before 2008 and after 2014)
            # Implicit: "Apr 30, 2008" - no parentheses (March 2008 - Sept 2014 period)

            # Extract reference month from (Month) suffix if present
            month_match = re.search(r'\(([A-Za-z]+)\)\s*$', release_date_str)

            # Clean the release date string (remove suffix)
            release_date_clean = re.sub(r'\s*\([A-Za-z]+\)\s*$', '', release_date_str).strip()

            try:
                release_dt = pd.to_datetime(release_date_clean, format='%b %d, %Y')
            except:
                release_dt = pd.to_datetime(release_date_clean)  # Let pandas infer

            # Determine reference date using priority logic
            if month_match:
                # Explicit format: "Dec 03, 2025 (Nov)" → reference month is Nov 2025
                ref_month_str = month_match.group(1)
                # Combine the reference month with the release year
                # Handle year boundary: if release is Jan and ref month is Dec, use previous year
                release_month = release_dt.month
                release_year = release_dt.year

                # Parse reference month name to number
                ref_month_dt = pd.to_datetime(ref_month_str, format='%b')
                ref_month_num = ref_month_dt.month

                # If release is in Jan-Feb and reference month is Nov-Dec, use previous year
                if release_month <= 2 and ref_month_num >= 11:
                    ref_year = release_year - 1
                else:
                    ref_year = release_year

                reference_date = pd.Timestamp(year=ref_year, month=ref_month_num, day=1)
            else:
                # Implicit format: use day-based rule
                # Early release (day <= 10): previous month
                # Late release (day >= 20): current month
                release_day = release_dt.day

                if release_day <= 10:
                    # Early release: data refers to previous month
                    # Example: "Apr 02, 2008" → March 2008
                    reference_date = (release_dt - pd.DateOffset(months=1)).replace(day=1)
                elif release_day >= 20:
                    # Late release: data refers to current month
                    # Example: "Apr 30, 2008" → April 2008
                    reference_date = release_dt.replace(day=1)
                else:
                    # Days 11-19: ambiguous range, default to previous month
                    logger.warning(f"Release date {release_date_str} has day {release_day} in ambiguous range (11-19), defaulting to previous month")
                    reference_date = (release_dt - pd.DateOffset(months=1)).replace(day=1)
            
            # Clean numeric values (handle "145K" format)
            def clean_value(s):
                if not s or s == '--' or pd.isnull(s):
                    return None
                s = str(s).replace('K', '').replace(',', '').strip()
                try:
                    return float(s) * 1000
                except:
                    return None
            
            # Create 2 rows in long format (dropped ADP_previous due to multicollinearity)
            for series_name, value_str in [
                ('ADP_actual', actual_str),  # KEPT: more accurate timing than ADP_previous
                ('ADP_forecast', forecast_str),
                # ('ADP_previous', previous_str)  # DROPPED: highly correlated with ADP_actual
            ]:
                value = clean_value(value_str)
                if value is not None:  # Only add if we have a valid value
                    data.append({
                        'date': reference_date,
                        'series_name': series_name,
                        'value': value,
                        'release_date': release_dt,
                        'series_type': 'adp'
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to process row: {e}")
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'series_name']).reset_index(drop=True)
    
    logger.info(f"Processed {len(df)} ADP data points")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Series: {df['series_name'].unique().tolist()}")
    
    df.to_parquet(CLEAN_ADP_PARQUET, index=False)
    logger.info(f"Saved formatted parquet: {CLEAN_ADP_PARQUET}")


def main() -> None:
    # Force unbuffered output for terminal visibility
    print("Starting ADP scraper...", flush=True)
    logger.info("Started ADP scraper")

    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        driver = None
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Initializing Chrome driver...", flush=True)
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Initializing Chrome driver...")
            driver = init_driver()
            
            print(f"Loading page: {URL}", flush=True)
            logger.info(f"Loading page: {URL}")
            driver.get(URL)
            print("✓ Page loaded successfully", flush=True)
            logger.info("✓ Page loaded successfully")

            try:
                consent_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[class*='consent' i]"))
                )
                consent_btn.click()
                logger.info("✓ Clicked cookie consent banner")
            except Exception:
                logger.warning("No cookie consent banner found, continuing without it")

            print("Clicking 'Show more' to load historical data...", flush=True)
            logger.info("Clicking 'Show more' to load historical data...")
            click_show_more(driver)

            print("Parsing table data...", flush=True)
            logger.info("Parsing table data...")
            rows = parse_table(driver.page_source)

            print(f"Saving {len(rows)} rows to CSV...", flush=True)
            logger.info(f"Saving {len(rows)} rows to CSV...")
            save_csv(rows)
            print("✓ Scraping completed successfully", flush=True)
            logger.info("✓ Scraping completed successfully")
            
            # If we got here, scraping succeeded - break out of retry loop
            break
            
        except Exception as e:
            logger.error(f"✗ Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All retry attempts failed. Scraping aborted.")
                raise
                
        finally:
            if driver:
                logger.info("Closing Chrome driver...")
                driver.quit()
    
    # Format the data after scraping
    print("Formatting ADP data to long format...", flush=True)
    logger.info("Formatting ADP data to long format...")
    format_adp_data()
    print("✓ ADP data pipeline complete!", flush=True)
    logger.info("✓ ADP data pipeline complete!")


if __name__ == "__main__":
    main()
