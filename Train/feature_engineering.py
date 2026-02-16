"""
Feature Engineering for LightGBM NFP Model

Calendar feature functions for the NFP prediction pipeline.
Extracted from train_lightgbm_nfp.py for maintainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import calendar
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)


# =============================================================================
# CALENDAR FEATURES
# =============================================================================

def get_survey_week_date(year: int, month: int) -> datetime:
    """
    Get the date of the week containing the 12th of the month.
    NFP survey covers the pay period including the 12th.

    Returns the Sunday of that week (start of survey week).
    """
    # Find the 12th of the month
    day_12 = datetime(year, month, 12)
    
    # Get the day of week (0=Monday, 6=Sunday)
    dow = day_12.weekday()
    
    # Find the Sunday of that week (start of survey week)
    # Sunday is day 6, so we go back (dow + 1) days
    sunday = day_12 - pd.Timedelta(days=(dow + 1) % 7)
    
    return sunday


def calculate_weeks_between_surveys(target_month: pd.Timestamp) -> int:
    """
    Calculate weeks between current and previous month's survey weeks.

    This captures the "4 vs 5 weeks" logic that affects NFP seasonality.

    Args:
        target_month: The month being predicted (format: YYYY-MM-01)

    Returns:
        Number of weeks (typically 4 or 5)
    """
    # Get survey weeks for current and previous month
    current_survey = get_survey_week_date(target_month.year, target_month.month)
    
    # Previous month
    prev_month = target_month - pd.DateOffset(months=1)
    prev_survey = get_survey_week_date(prev_month.year, prev_month.month)
    
    # Calculate weeks between
    days_diff = (current_survey - prev_survey).days
    weeks = days_diff // 7
    
    return weeks


def add_calendar_features(df: pd.DataFrame, target_month: pd.Timestamp) -> pd.DataFrame:
    """
    Add comprehensive calendar-based features to a DataFrame.

    Includes:
    - Cyclical month encoding (sin/cos) - preserves December/January proximity
    - Quarter cyclical encoding
    - Survey interval (4 vs 5 weeks)
    - Seasonal and timing indicators

    Args:
        df: Input DataFrame to add features to
        target_month: Target month being predicted

    Returns:
        DataFrame with calendar features added as columns
    """
    df = df.copy()

    month = target_month.month

    # --- Cyclical Month Encoding (NO one-hot) ---
    # This preserves the cyclical nature: December is close to January
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Quarter cyclical encoding
    df['quarter_sin'] = np.sin(2 * np.pi * target_month.quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * target_month.quarter / 4)

    # --- Survey Interval Feature ---
    # The "4 vs 5 weeks" logic - critical for NSA prediction
    weeks_since_survey = calculate_weeks_between_surveys(target_month)
    df['weeks_since_last_survey'] = weeks_since_survey
    df['is_5_week_month'] = int(weeks_since_survey == 5)

    # --- Seasonal adjustment timing ---
    # BLS updates seasonal factors in January
    df['is_jan'] = int(month == 1)
    # Mid-year benchmark revision month
    df['is_july'] = int(month == 7)

    # Year (for capturing secular trends if needed)
    df['year'] = target_month.year

    # --- Additional seasonal indicators ---
    # Summer slowdown period
    df['is_summer'] = int(month in [6, 7, 8])

    # Holiday season (Nov-Dec hiring surge)
    df['is_holiday_season'] = int(month in [11, 12])

    # Beginning/end of year effects
    df['is_december'] = int(month == 12)

    return df


def get_calendar_features_dict(target_month: pd.Timestamp) -> Dict[str, float]:
    """
    Get calendar features as a dictionary (for merging into feature dict).

    Args:
        target_month: Target month being predicted

    Returns:
        Dictionary of calendar features
    """
    features = {}
    month = target_month.month

    # Cyclical encoding
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['quarter_sin'] = np.sin(2 * np.pi * target_month.quarter / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * target_month.quarter / 4)

    # Survey interval
    weeks_since_survey = calculate_weeks_between_surveys(target_month)
    features['weeks_since_last_survey'] = weeks_since_survey
    features['is_5_week_month'] = int(weeks_since_survey == 5)

    # Seasonal timing
    features['is_jan'] = int(month == 1)
    features['is_july'] = int(month == 7)
    features['year'] = target_month.year
    features['is_summer'] = int(month in [6, 7, 8])
    features['is_holiday_season'] = int(month in [11, 12])
    features['is_december'] = int(month == 12)

    return features
