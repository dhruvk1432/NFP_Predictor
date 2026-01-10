"""
BLS Employment Situation Release Date Scraper

Scrapes the BLS website to get official NFP release dates for future months.
Scrapes the BLS website to get official NFP release dates for future months.
Raises errors if scraping fails.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)




def parse_bls_date(date_str: str) -> Optional[pd.Timestamp]:
    """
    Parse BLS date formats:
    - "Nov. 01, 2024" (abbreviated month)
    - "January 9, 2026" (full month)
    - "Friday, January 9, 2026" (with day of week)

    Args:
        date_str: Date string from BLS website

    Returns:
        Timestamp or None if parsing fails
    """
    try:
        # Remove day of week prefix if present (e.g., "Friday, January 9, 2026")
        # Day of week would be at the start and followed by comma
        parts = date_str.split(',')
        if len(parts) == 3:
            # Format: "Friday, January 9, 2026"
            # Take everything after first comma
            date_str = ','.join(parts[1:]).strip()
        elif len(parts) == 2:
            # Could be "Month Day, Year" or "Friday, Month..."
            # Check if first part looks like a day of week
            first_part = parts[0].strip()
            days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if any(day in first_part for day in days_of_week):
                # It's "Friday, ..." format, skip first part
                date_str = parts[1].strip()
            # else it's already "Month Day, Year" format

        # Try parsing with abbreviated month first (Nov., Dec., etc.)
        for fmt in ["%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return pd.Timestamp(dt)
            except:
                continue

        return None
    except Exception as e:
        logger.warning(f"Failed to parse date '{date_str}': {e}")
        return None


def scrape_bls_employment_situation_schedule() -> pd.DataFrame:
    """
    Scrape BLS website for NFP (Employment Situation) release dates.

    The BLS publishes all Employment Situation release dates at a single URL.

    Returns:
        DataFrame with columns: ['observation_month', 'release_date']
        - observation_month: Month being reported (e.g., 2025-12-01 for December 2025)
        - release_date: Official release date (e.g., 2026-01-10)
    """
    url = "https://www.bls.gov/schedule/news_release/empsit.htm"

    try:
        logger.info(f"Scraping BLS Employment Situation schedule: {url}")

        # Add headers to avoid 403 errors - use macOS Safari user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all tables on the page
        tables = soup.find_all('table')

        if not tables:
            raise ValueError("No tables found on BLS Employment Situation page")

        results = []

        # The schedule is in the first table
        # Structure: Row 0 = headers, Row 1+ = data
        # Columns: Reference Month | Release Date | Release Time
        for table in tables:
            rows = table.find_all('tr')

            for row in rows:
                cells = row.find_all(['td', 'th'])

                # Need at least 2 cells (Reference Month, Release Date)
                if len(cells) < 2:
                    continue

                ref_month_text = cells[0].get_text(strip=True)
                release_date_text = cells[1].get_text(strip=True)

                # Skip header rows
                if ref_month_text in ['Reference Month', 'BY MONTH'] or 'ENTIRE YEAR' in ref_month_text:
                    continue

                # Parse reference month: "December 2025" format
                try:
                    # Add day 1 to make it parseable
                    obs_month = pd.to_datetime(ref_month_text + " 1")

                    # Parse release date: "Jan. 09, 2026" format
                    release_date = parse_bls_date(release_date_text)

                    if release_date is not None:
                        results.append({
                            'observation_month': obs_month,
                            'release_date': release_date
                        })
                        logger.info(f"Found: {obs_month.strftime('%B %Y')} -> {release_date.strftime('%Y-%m-%d')}")

                except Exception as e:
                    # Skip rows that don't parse (navigation, headers, etc.)
                    continue

        if not results:
            raise ValueError("No Employment Situation releases parsed from BLS page")

        df = pd.DataFrame(results)
        # Remove duplicates (in case same month appears multiple times)
        df = df.drop_duplicates(subset=['observation_month'])
        df = df.sort_values('observation_month')

        logger.info(f"Successfully scraped {len(df)} NFP release dates")
        return df

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch BLS Employment Situation schedule: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error scraping BLS schedule: {e}")


def get_future_nfp_dates(
    start_month: pd.Timestamp,
    end_month: pd.Timestamp
) -> pd.DataFrame:
    """
    Get NFP release dates for a date range by scraping BLS website.

    Args:
        start_month: First observation month to get (e.g., 2025-11-01)
        end_month: Last observation month to get (e.g., 2025-12-01)

    Returns:
        DataFrame with columns: ['observation_month', 'release_date']
    """
    # Scrape the Employment Situation schedule page
    all_dates = scrape_bls_employment_situation_schedule()



    # Filter to requested range
    filtered = all_dates[
        (all_dates['observation_month'] >= start_month) &
        (all_dates['observation_month'] <= end_month)
    ]

    filtered = filtered.sort_values('observation_month').reset_index(drop=True)

    return filtered





if __name__ == "__main__":
    # Test scraper
    print("Testing BLS Employment Situation scraper...")
    test_df = scrape_bls_employment_situation_schedule()
    print("\nScraped BLS Employment Situation Schedule:")
    print(test_df)

    # Test getting future dates for a specific range
    start = pd.Timestamp("2025-11-01")
    end = pd.Timestamp("2026-02-01")

    print(f"\nGetting dates for range {start.date()} to {end.date()}:")
    range_df = get_future_nfp_dates(start, end)
    print(range_df)

    print(range_df)
