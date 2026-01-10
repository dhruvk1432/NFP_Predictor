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
    ALL_KEY_EMPLOYMENT_SERIES,
    ALL_LAGS,
    SHORT_TERM_LAGS,
    MEDIUM_TERM_LAGS,
    LONG_TERM_LAGS,
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


def add_calendar_features(df: pd.DataFrame, target_month: pd.Timestamp) -> Dict[str, float]:
    """
    Add comprehensive calendar-based features.

    Includes:
    - Cyclical month encoding (sin/cos) - preserves December/January proximity
    - Survey interval (4 vs 5 weeks)
    - Quarter indicators
    """
    features = {}
    
    month = target_month.month
    
    # Cyclical encoding - preserves that December and January are close
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Quarter
    features['quarter'] = (month - 1) // 3 + 1
    
    # Survey week interval (4 vs 5 weeks logic)
    features['survey_weeks'] = calculate_weeks_between_surveys(target_month)
    
    # Beginning/end of year effects
    features['is_january'] = 1 if month == 1 else 0
    features['is_december'] = 1 if month == 12 else 0
    
    # Summer slowdown
    features['is_summer'] = 1 if month in [6, 7, 8] else 0
    
    # Holiday season
    features['is_holiday_season'] = 1 if month in [11, 12] else 0
    
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

    Separates NSA vs SA features based on target_type to avoid data leakage:
    - When predicting NSA: Use SA series as features (different adjustment)
    - When predicting SA: Use NSA series as features (raw data)
    - Always include sector composition and momentum indicators

    Args:
        fred_df: FRED employment snapshot DataFrame
        target_month: Month being predicted
        target_type: 'nsa' or 'sa' - determines which series to use as features

    Returns:
        Dictionary of engineered features
    """
    if fred_df is None or fred_df.empty:
        return {}

    features = {}

    # Determine which series to use based on target type
    if target_type == 'nsa':
        suffix = ''  # SA series don't have suffix
        feature_prefix = 'emp_sa'
    else:
        suffix = '_nsa'
        feature_prefix = 'emp_nsa'

    # Filter to data available before target_month (avoid look-ahead)
    available_df = fred_df[fred_df['date'] < target_month].copy()

    if available_df.empty:
        return features

    # Get key employment series
    for series_base in ALL_KEY_EMPLOYMENT_SERIES:
        series_name = series_base + suffix if suffix else series_base

        series_data = available_df[available_df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        if series_data.empty:
            continue

        # Clean series name for feature naming
        clean_name = series_base.replace('.', '_').replace('total_', '')

        # Latest value
        if len(series_data) >= 1:
            latest_value = series_data['value'].iloc[-1]
            features[f'{feature_prefix}_{clean_name}_latest'] = latest_value

        # MoM change (lag 1)
        if len(series_data) >= 2:
            mom_change = series_data['value'].iloc[-1] - series_data['value'].iloc[-2]
            features[f'{feature_prefix}_{clean_name}_mom'] = mom_change
            # MoM percentage change
            if series_data['value'].iloc[-2] != 0:
                mom_pct = (mom_change / series_data['value'].iloc[-2]) * 100
                features[f'{feature_prefix}_{clean_name}_mom_pct'] = mom_pct

        # 3-month change
        if len(series_data) >= 4:
            change_3m = series_data['value'].iloc[-1] - series_data['value'].iloc[-4]
            features[f'{feature_prefix}_{clean_name}_3m_chg'] = change_3m

        # 6-month change
        if len(series_data) >= 7:
            change_6m = series_data['value'].iloc[-1] - series_data['value'].iloc[-7]
            features[f'{feature_prefix}_{clean_name}_6m_chg'] = change_6m

        # YoY change
        if len(series_data) >= 13:
            yoy_change = series_data['value'].iloc[-1] - series_data['value'].iloc[-13]
            features[f'{feature_prefix}_{clean_name}_yoy'] = yoy_change

        # Rolling mean (3-month smoothed)
        if len(series_data) >= 3:
            rolling_3m = series_data['value'].iloc[-3:].mean()
            features[f'{feature_prefix}_{clean_name}_rolling_3m'] = rolling_3m

        # Volatility (6-month std of changes)
        if len(series_data) >= 7:
            changes = series_data['value'].diff().iloc[-6:]
            features[f'{feature_prefix}_{clean_name}_volatility'] = changes.std()

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
