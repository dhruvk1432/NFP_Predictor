"""
Feature Engineering for LightGBM NFP Model

This module provides specialized calendar and timing feature extraction functions 
crucial for the Non-Farm Payroll prediction pipeline. Since NFP data exhibits 
strong seasonality and depends heavily on specific calendrical quirks (like 
the varying number of weeks between survey periods), these functions capture those nuances.

These features are generally used to augment the macroeconomic feature set with 
structural time-based variables.
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
    Determine the specific date for the BLS employment survey week for a given month.
    
    The BLS defines the reference week for the establishment survey (Non-Farm Payrolls) 
    as the pay period that includes the 12th of the month. This function calculates the 
    Sunday that begins that specific week.

    Args:
        year (int): The year of the survey.
        month (int): The month of the survey.

    Returns:
        datetime: The exact date of the Sunday representing the start of the survey week.
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
    Calculate the elapsed weeks between the current month's survey and the previous month's.

    Due to the way the calendar falls, there are either 4 or 5 weeks between the
    12th-of-the-month survey periods. This is a critical structural feature for predicting
    the NSA (Non-Seasonally Adjusted) NFP variations. A 5-week interval usually allows 
    for more accumulated job growth/loss between reports.

    Args:
        target_month (pd.Timestamp): The month being predicted (expected format: YYYY-MM-01).

    Returns:
        int: Total number of full weeks (typically exactly 4 or 5) between the two surveys.
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
    Enrich an existing features DataFrame with comprehensive calendar-based metrics.

    This function appends several time-structural features beneficial for tree-based models:
    - Target month and quarter cyclical encoding (using sine and cosine to preserve proximity, 
      so December is mathematically adjacent to January).
    - Survey intervals representing whether a month had a 4 or 5 week gap since the last print.
    - Key seasonality indicators for structural BLS adjustments (e.g. January benchmark 
      revisions, mid-year July adjustments).
    - Cyclical business and hiring seasons (holidays, summer).

    Args:
        df (pd.DataFrame): The input DataFrame containing baseline features.
        target_month (pd.Timestamp): The target month being predicted.

    Returns:
        pd.DataFrame: A copy of the DataFrame with added calendar feature columns.
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

    return df


def get_calendar_features_dict(target_month: pd.Timestamp) -> Dict[str, float]:
    """
    Generate calendar features as a flat dictionary, useful for merging into a single sample's feature dict.

    This performs the same extraction as `add_calendar_features` but outputs a simple key-value structure
    rather than operating on a full Pandas DataFrame.

    Args:
        target_month (pd.Timestamp): The target month being predicted.

    Returns:
        Dict[str, float]: A dictionary containing all computed calendar features for the single month.
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
    return features
