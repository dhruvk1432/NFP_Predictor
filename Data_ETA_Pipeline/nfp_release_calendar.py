"""
NFP Release Calendar

Historical NFP release dates (first Friday of each month).
Used to determine if exogenous variables are available before NFP.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import START_DATE, END_DATE

def get_nfp_release_calendar(start = START_DATE, end = END_DATE):
    """
    Generate NFP release calendar.
    
    NFP is released on the first Friday of each month at 8:30 AM ET.
    
    Args:
        start: Starting date
        end: Ending date
        
    Returns:
        DataFrame with columns ['observation_month', 'nfp_release_date']
    """
    
    # Generate all months
    months = pd.date_range(
        start=start,
        end=end,
        freq='MS'  # Month start
    )
    
    calendar = []
    
    for month in months:
        # Find first Friday of next month
        next_month = month + pd.DateOffset(months=1)
        
        # Get first day of next month
        first_day = pd.Timestamp(year=next_month.year, month=next_month.month, day=1)
        
        # Find first Friday
        # Monday=0, Friday=4
        days_until_friday = (4 - first_day.weekday()) % 7
        if days_until_friday == 0 and first_day.weekday() != 4:
            days_until_friday = 7
        
        nfp_date = first_day + pd.Timedelta(days=days_until_friday)
        
        calendar.append({
            'observation_month': month,
            'nfp_release_date': nfp_date
        })
    
    return pd.DataFrame(calendar)


def get_nfp_release_for_month(observation_month):
    """
    Get NFP release date for a specific observation month.
    
    Args:
        observation_month: Date representing the observation month (e.g., '2024-07-01')
        
    Returns:
        NFP release date (first Friday of next month)
    """
    obs_date = pd.to_datetime(observation_month)
    
    # Next month
    next_month = obs_date + pd.DateOffset(months=1)
    first_day = pd.Timestamp(year=next_month.year, month=next_month.month, day=1)
    
    # First Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    if days_until_friday == 0 and first_day.weekday() != 4:
        days_until_friday = 7
    
    return first_day + pd.Timedelta(days=days_until_friday)


if __name__ == "__main__":
    # Generate and save calendar
    calendar = get_nfp_release_calendar(2020, 2025)
    
    output_path = Path(__file__).parent / "data" / "nfp_release_calendar.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(output_path, index=False)
    
    print("NFP Release Calendar")
    print("=" * 60)
    print(f"Generated {len(calendar)} months ({calendar['observation_month'].min().year} - {calendar['observation_month'].max().year})")
    print(f"\nSample:")
    print(calendar.tail(12))
    print(f"\nSaved to: {output_path}")
