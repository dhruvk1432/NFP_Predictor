"""
Feature Engineering for LightGBM NFP Model

Functions for creating features from employment data and exogenous indicators.
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
from Train.config import (
    KEY_EMPLOYMENT_SERIES,
)

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


# =============================================================================
# EMPLOYMENT FEATURES
# =============================================================================

def engineer_employment_features(
    fred_df: pd.DataFrame,
    target_month: pd.Timestamp,
    target_type: str = 'nsa'
) -> Dict[str, float]:
    """
    Engineer features from FRED employment snapshot data.

    Uses ALL available series (both NSA and SA) from the FRED snapshot.
    For each series, computes: latest, MoM, MoM%, 3m change, 6m change,
    YoY change, 12m pct change, rolling 3m mean, and 6m volatility.

    Args:
        fred_df: FRED employment snapshot DataFrame
        target_month: Month being predicted
        target_type: 'nsa' or 'sa' (kept for API compatibility, no longer filters series)

    Returns:
        Dictionary of engineered features
    """
    if fred_df is None or fred_df.empty:
        return {}

    features = {}

    # Filter to data available before target_month (avoid look-ahead)
    available_df = fred_df[fred_df['date'] < target_month].copy()

    if available_df.empty:
        return features

    # Use ALL series from the FRED snapshot (both NSA and SA)
    all_series = available_df['series_name'].unique()

    for series_name in all_series:
        series_data = available_df[available_df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        if series_data.empty:
            continue

        # Clean series name for feature naming
        clean_name = series_name.replace('.', '_').replace('total_', '')
        prefix = f'emp_{clean_name}'

        values = series_data['value']
        n = len(values)

        # Latest value
        latest_value = values.iloc[-1]
        features[f'{prefix}_latest'] = latest_value

        # MoM change
        if n >= 2:
            mom_change = values.iloc[-1] - values.iloc[-2]
            features[f'{prefix}_mom'] = mom_change
            # MoM percentage change
            if values.iloc[-2] != 0:
                features[f'{prefix}_mom_pct'] = (mom_change / abs(values.iloc[-2])) * 100

        # 3-month change
        if n >= 4:
            features[f'{prefix}_3m_chg'] = values.iloc[-1] - values.iloc[-4]

        # 6-month change
        if n >= 7:
            features[f'{prefix}_6m_chg'] = values.iloc[-1] - values.iloc[-7]

        # YoY change
        if n >= 13:
            features[f'{prefix}_yoy'] = values.iloc[-1] - values.iloc[-13]

        # 12-month percent change
        if n >= 13 and values.iloc[-13] != 0:
            features[f'{prefix}_12m_pct'] = (
                (values.iloc[-1] - values.iloc[-13]) / abs(values.iloc[-13])
            ) * 100

        # Rolling mean (3-month smoothed)
        if n >= 3:
            features[f'{prefix}_rolling_3m'] = values.iloc[-3:].mean()

        # Volatility (6-month std of changes)
        if n >= 7:
            changes = values.diff().iloc[-6:]
            features[f'{prefix}_volatility'] = changes.std()

    return features


# =============================================================================
# EXOGENOUS FEATURE AGGREGATION
# =============================================================================

def engineer_exogenous_features(
    snapshot_df: pd.DataFrame,
    target_month: pd.Timestamp
) -> Dict[str, float]:
    """
    Engineer features from exogenous master snapshot.

    Adds lagged values and momentum features for each series.

    Args:
        snapshot_df: Master snapshot DataFrame in long format
        target_month: Month being predicted

    Returns:
        Dictionary of features
    """
    if snapshot_df is None or snapshot_df.empty:
        return {}

    features = {}
    df = snapshot_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    for series_name in df['series_name'].unique():
        series_data = df[df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        # Get data available up to target_month
        available = series_data[series_data['date'] <= target_month]

        if available.empty:
            continue

        # Latest value
        features[f"{series_name}_latest"] = available['value'].iloc[-1]

        # Lags
        for lag in [1, 3, 6, 12]:
            if len(available) >= lag + 1:
                features[f"{series_name}_lag{lag}"] = available['value'].iloc[-(lag + 1)]

        # Month-over-month change
        if len(available) >= 2:
            mom = available['value'].iloc[-1] - available['value'].iloc[-2]
            features[f"{series_name}_mom_change"] = mom

        # 3-month rolling mean
        if len(available) >= 3:
            features[f"{series_name}_rolling_3m"] = available['value'].iloc[-3:].mean()

    return features


# =============================================================================
# SECTOR COMPOSITION
# =============================================================================

def calculate_sector_breadth(
    fred_df: pd.DataFrame,
    target_month: pd.Timestamp,
    target_type: str = 'nsa'
) -> Dict[str, float]:
    """
    Calculate sector breadth indicators.

    Measures how many sectors are expanding vs contracting.

    Args:
        fred_df: FRED employment snapshot
        target_month: Month being predicted
        target_type: 'nsa' or 'sa'

    Returns:
        Dictionary with breadth indicators
    """
    features = {}
    
    if fred_df is None or fred_df.empty:
        return features

    comp_suffix = '_nsa' if target_type == 'nsa' else ''
    
    available_df = fred_df[fred_df['date'] < target_month].copy()
    
    if available_df.empty:
        return features

    positive_sectors = 0
    negative_sectors = 0

    for series_base in KEY_EMPLOYMENT_SERIES['service_industries']:
        series_name = series_base + comp_suffix
        series_data = available_df[available_df['series_name'] == series_name]

        if not series_data.empty and len(series_data) >= 2:
            series_data = series_data.sort_values('date')
            mom = series_data['value'].iloc[-1] - series_data['value'].iloc[-2]
            if mom > 0:
                positive_sectors += 1
            elif mom < 0:
                negative_sectors += 1

    total_sectors = positive_sectors + negative_sectors
    if total_sectors > 0:
        features['emp_breadth'] = (positive_sectors - negative_sectors) / total_sectors
        features['emp_pct_positive'] = positive_sectors / total_sectors * 100

    return features
