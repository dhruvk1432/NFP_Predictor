"""
LightGBM NFP Prediction Model

Predicts NSA/SA NFP month-on-month change using master snapshots.
Uses snapshot of month M to predict month M (no look-ahead bias).
Outputs predictions with confidence intervals/error bounds.

Key Features:
- Handles data with varying start dates (some series start 1948, others 2008)
- Uses past target data (NFP levels, MoM changes) as features
- Survey interval features (4 vs 5 weeks logic)
- Momentum/divergence and acceleration features
- Cyclical month encoding (sin/cos)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import pickle
from datetime import datetime
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, BACKTEST_MONTHS
from Train.backtest_results import generate_backtest_report

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM/sklearn not available. Install with: pip install lightgbm scikit-learn")
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except ImportError:
    VIF_AVAILABLE = False

# Feature selection configuration
MAX_FEATURES = 80  # Maximum number of features to use in final model
VIF_THRESHOLD = 10.0  # Remove features with VIF above this
CORR_THRESHOLD = 0.95  # Remove one of a pair with correlation above this
MIN_TARGET_CORR = 0.05  # Minimum absolute correlation with target

# Linear baseline predictors for extrapolation capability
# These are chosen because they have strong linear relationships with NFP
# and can extrapolate beyond training ranges (unlike trees)
LINEAR_BASELINE_PREDICTORS = [
    # Weekly Claims - PRIMARY EXTRAPOLATION SIGNAL
    'CCSA_monthly_avg_latest',       # Current month average
    'CCSA_max_spike',                 # Maximum spike (NEW from Task 3)
    'CCSA_monthly_avg_lag1',          # Previous month
    'CCSA_monthly_avg_mom_change',    # Month-over-month change

    # Oil Prices - Economic stress indicator
    'Oil_Prices_mean_latest',         # Current level
    'Oil_Prices_30d_crash',           # Crash magnitude

    # Yield Curve - Leading indicator (12-month lag)
    'Yield_Curve_avg_lag12',          # Signal from year ago
    'Yield_Curve_monthly_chg_lag12',  # Change from year ago

    # Past NFP as autoregressive component
    'nfp_nsa_mom_lag1',               # Previous NFP MoM
    'nfp_sa_mom_lag1',                # Previous SA NFP MoM
]

# Binary regime flags that should never be removed by feature selection
# These capture critical extreme events (COVID-like scenarios)
PROTECTED_BINARY_FLAGS = [
    'VIX_panic_regime',       # VIX >50 (extreme panic)
    'VIX_high_regime',        # VIX >40 (high fear)
    'SP500_crash_month',      # Monthly return <-10%
    'SP500_bear_market',      # Drawdown <-20% from 52w high
    'SP500_circuit_breaker',  # Any day down >5%
]


# --- Path Configuration ---
MASTER_SNAPSHOTS_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"
FRED_SNAPSHOTS_DIR = DATA_PATH / "fred_data" / "decades"
FRED_PREPARED_DIR = DATA_PATH / "fred_data_prepared" / "decades"  # Preprocessed data
TARGET_PATH_NSA = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
TARGET_PATH_SA = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"
MODEL_SAVE_DIR = OUTPUT_DIR / "models" / "lightgbm_nfp"

# Use prepared data (SymLog transformed, scaled) or raw data
USE_PREPARED_FRED_DATA = True  # Set to False to use raw FRED data

# --- Key Employment Series for Feature Engineering ---
# These are the most predictive employment subsectors
KEY_EMPLOYMENT_SERIES = {
    # Level 1: Major Divisions (NSA only)
    'aggregates': [
        'total.private_nsa',
        'total.government_nsa',
    ],

    # Level 2: Private Sector Breakdown (NSA only)
    'goods_services': [
        'total.private.goods_nsa',
        'total.private.services_nsa',
    ],

    # Level 3: Goods-Producing Industries (NSA only)
    'goods_industries': [
        'total.private.goods.mining_logging_nsa',
        'total.private.goods.construction_nsa',
        'total.private.goods.manufacturing_nsa',
    ],

    # Level 4: Mining and Logging Breakdown (NSA only)
    'mining_logging_breakdown': [
        'total.private.goods.mining_logging.logging_nsa',
        'total.private.goods.mining_logging.mining_nsa',
    ],

    # Level 4: Construction Breakdown (NSA only)
    'construction_breakdown': [
        'total.private.goods.construction.buildings_nsa',
        'total.private.goods.construction.heavy_civil_nsa',
        'total.private.goods.construction.specialty_nsa',
    ],

    # Level 4: Manufacturing Breakdown (NSA only)
    'manufacturing_breakdown': [
        'total.private.goods.manufacturing.durable_nsa',
        'total.private.goods.manufacturing.nondurable_nsa',
    ],

    # Level 3: Service-Providing Industries (NSA only)
    'service_industries': [
        'total.private.services.trade_transportation_utilities_nsa',
        'total.private.services.information_nsa',
        'total.private.services.financial_nsa',
        'total.private.services.professional_business_nsa',
        'total.private.services.education_health_nsa',
        'total.private.services.leisure_hospitality_nsa',
        'total.private.services.other_nsa',
    ],

    # Level 4: Trade, Transportation, and Utilities Breakdown (NSA only)
    'trade_transport_utilities_breakdown': [
        'total.private.services.trade_transportation_utilities.wholesale_nsa',
        'total.private.services.trade_transportation_utilities.retail_nsa',
        'total.private.services.trade_transportation_utilities.transportation_warehousing_nsa',
        'total.private.services.trade_transportation_utilities.utilities_nsa',
    ],

    # Level 4: Information Breakdown (NSA only)
    'information_breakdown': [
        'total.private.services.information.motion_picture_nsa',
        'total.private.services.information.publishing_nsa',
        'total.private.services.information.broadcasting_nsa',
        'total.private.services.information.telecommunications_nsa',
        'total.private.services.information.data_processing_nsa',
        'total.private.services.information.other_nsa',
    ],

    # Level 4: Financial Activities Breakdown (NSA only)
    'financial_breakdown': [
        'total.private.services.financial.finance_insurance_nsa',
        'total.private.services.financial.real_estate_nsa',
    ],

    # Level 4: Professional and Business Services Breakdown (NSA only)
    'professional_business_breakdown': [
        'total.private.services.professional_business.professional_technical_nsa',
        'total.private.services.professional_business.management_companies_nsa',
        'total.private.services.professional_business.admin_waste_nsa',
    ],

    # Level 4: Education and Health Services Breakdown (NSA only)
    'education_health_breakdown': [
        'total.private.services.education_health.education_nsa',
        'total.private.services.education_health.health_social_nsa',
    ],

    # Level 4: Leisure and Hospitality Breakdown (NSA only)
    'leisure_hospitality_breakdown': [
        'total.private.services.leisure_hospitality.arts_nsa',
        'total.private.services.leisure_hospitality.accommodation_food_nsa',
    ],

    # Level 4: Other Services Breakdown (NSA only)
    'other_services_breakdown': [
        'total.private.services.other.repair_nsa',
        'total.private.services.other.personal_nsa',
        'total.private.services.other.religious_nsa',
    ],

    # Level 2: Government Breakdown (NSA only)
    'government_breakdown': [
        'total.government.federal_nsa',
        'total.government.state_nsa',
        'total.government.local_nsa',
    ],

    # Level 3: Federal Government Breakdown (NSA only)
    'federal_government_breakdown': [
        'total.government.federal.except_postal_nsa',
        'total.government.federal.postal_nsa',
    ],

    # Level 3: State Government Breakdown (NSA only)
    'state_government_breakdown': [
        'total.government.state.education_nsa',
        'total.government.state.excluding_education_nsa',
    ],

    # Level 3: Local Government Breakdown (NSA only)
    'local_government_breakdown': [
        'total.government.local.education_nsa',
        'total.government.local.excluding_education_nsa',
    ],
}

# Flatten for easy access
ALL_KEY_EMPLOYMENT_SERIES = (
    KEY_EMPLOYMENT_SERIES['aggregates'] +
    KEY_EMPLOYMENT_SERIES['goods_services'] +
    KEY_EMPLOYMENT_SERIES['goods_industries'] +
    KEY_EMPLOYMENT_SERIES['mining_logging_breakdown'] +
    KEY_EMPLOYMENT_SERIES['construction_breakdown'] +
    KEY_EMPLOYMENT_SERIES['manufacturing_breakdown'] +
    KEY_EMPLOYMENT_SERIES['service_industries'] +
    KEY_EMPLOYMENT_SERIES['trade_transport_utilities_breakdown'] +
    KEY_EMPLOYMENT_SERIES['information_breakdown'] +
    KEY_EMPLOYMENT_SERIES['financial_breakdown'] +
    KEY_EMPLOYMENT_SERIES['professional_business_breakdown'] +
    KEY_EMPLOYMENT_SERIES['education_health_breakdown'] +
    KEY_EMPLOYMENT_SERIES['leisure_hospitality_breakdown'] +
    KEY_EMPLOYMENT_SERIES['other_services_breakdown'] +
    KEY_EMPLOYMENT_SERIES['government_breakdown'] +
    KEY_EMPLOYMENT_SERIES['federal_government_breakdown'] +
    KEY_EMPLOYMENT_SERIES['state_government_breakdown'] +
    KEY_EMPLOYMENT_SERIES['local_government_breakdown']
)

# --- Lag Configuration for Macroeconomic Data ---
# Short-term lags: 1, 2, 3 months (business cycle dynamics)
# Medium-term lags: 6 months (semi-annual patterns)
# Long-term lags: 12, 18, 24 months (structural/secular trends)
SHORT_TERM_LAGS = [1, 2, 3]
MEDIUM_TERM_LAGS = [6]
LONG_TERM_LAGS = [12, 18, 24]
ALL_LAGS = SHORT_TERM_LAGS + MEDIUM_TERM_LAGS + LONG_TERM_LAGS


# =============================================================================
# ENDOGENOUS EMPLOYMENT DATA (FRED SNAPSHOTS)
# =============================================================================

def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """Build path to FRED employment snapshot file."""
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')

    # Use prepared data if configured
    base_dir = FRED_PREPARED_DIR if USE_PREPARED_FRED_DATA else FRED_SNAPSHOTS_DIR
    return base_dir / decade / year / f"{month_str}.parquet"


def load_fred_snapshot(snapshot_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Load FRED employment snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)

    Returns:
        DataFrame with columns: date, value, series_name, series_code, release_date
    """
    path = get_fred_snapshot_path(snapshot_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


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
    # If predicting NSA, use SA employment as features (and vice versa)
    # This provides complementary information without using the target itself
    if target_type == 'nsa':
        # Use SA employment data as features when predicting NSA
        suffix = ''  # SA series don't have suffix
        feature_prefix = 'emp_sa'
    else:
        # Use NSA employment data as features when predicting SA
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

    # --- Sector Composition Features ---
    # These show where job growth/decline is concentrated
    # Use SAME adjustment type for composition (internally consistent)
    comp_suffix = '_nsa' if target_type == 'nsa' else ''

    # Get latest private sector breakdown
    private_total_series = 'total.private' + comp_suffix
    private_data = available_df[available_df['series_name'] == private_total_series]

    if not private_data.empty and len(private_data) >= 2:
        private_latest = private_data['value'].iloc[-1]
        private_mom = private_data['value'].iloc[-1] - private_data['value'].iloc[-2]

        # Goods vs Services share
        goods_series = 'total.private.goods' + comp_suffix
        services_series = 'total.private.services' + comp_suffix

        goods_data = available_df[available_df['series_name'] == goods_series]
        services_data = available_df[available_df['series_name'] == services_series]

        if not goods_data.empty and len(goods_data) >= 2:
            goods_latest = goods_data['value'].iloc[-1]
            goods_mom = goods_data['value'].iloc[-1] - goods_data['value'].iloc[-2]
            if private_latest > 0:
                features['emp_goods_share'] = (goods_latest / private_latest) * 100
            if private_mom != 0:
                features['emp_goods_contribution'] = (goods_mom / private_mom) * 100 if private_mom != 0 else 0

        if not services_data.empty and len(services_data) >= 2:
            services_latest = services_data['value'].iloc[-1]
            services_mom = services_data['value'].iloc[-1] - services_data['value'].iloc[-2]
            if private_latest > 0:
                features['emp_services_share'] = (services_latest / private_latest) * 100
            if private_mom != 0:
                features['emp_services_contribution'] = (services_mom / private_mom) * 100 if private_mom != 0 else 0

    # --- Government vs Private Sector ---
    govt_series = 'total.government' + comp_suffix
    govt_data = available_df[available_df['series_name'] == govt_series]

    if not govt_data.empty and len(govt_data) >= 2:
        govt_mom = govt_data['value'].iloc[-1] - govt_data['value'].iloc[-2]
        features['emp_govt_mom'] = govt_mom

        if not private_data.empty and len(private_data) >= 2:
            features['emp_private_govt_ratio'] = private_latest / govt_data['value'].iloc[-1] if govt_data['value'].iloc[-1] > 0 else 0

    # --- Breadth Indicator ---
    # Count sectors with positive vs negative MoM changes
    positive_sectors = 0
    negative_sectors = 0

    for series_base in KEY_EMPLOYMENT_SERIES['service_industries']:
        series_name = series_base + comp_suffix
        series_data = available_df[available_df['series_name'] == series_name]

        if not series_data.empty and len(series_data) >= 2:
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


# =============================================================================
# CALENDAR AND SURVEY INTERVAL FEATURES
# =============================================================================

def get_survey_week_date(year: int, month: int) -> pd.Timestamp:
    """
    Get the date of the week containing the 12th of the month.
    NFP survey covers the pay period including the 12th.

    Returns the Sunday of that week (start of survey week).
    """
    # The 12th of the month
    twelfth = pd.Timestamp(year=year, month=month, day=12)
    # Get the Sunday of that week (weekday: Mon=0, Sun=6)
    # We want the start of the week containing the 12th
    days_since_sunday = (twelfth.weekday() + 1) % 7
    survey_week_start = twelfth - pd.Timedelta(days=days_since_sunday)
    return survey_week_start


def calculate_weeks_between_surveys(target_month: pd.Timestamp) -> int:
    """
    Calculate weeks between current and previous month's survey weeks.

    This captures the "4 vs 5 weeks" logic that affects NFP seasonality.

    Args:
        target_month: The month being predicted (format: YYYY-MM-01)

    Returns:
        Number of weeks (typically 4 or 5)
    """
    current_survey = get_survey_week_date(target_month.year, target_month.month)

    # Previous month
    prev_month = target_month - pd.DateOffset(months=1)
    prev_survey = get_survey_week_date(prev_month.year, prev_month.month)

    # Calculate weeks difference
    days_diff = (current_survey - prev_survey).days
    weeks = days_diff // 7

    return weeks


def add_calendar_features(df: pd.DataFrame, target_month: pd.Timestamp) -> pd.DataFrame:
    """
    Add comprehensive calendar-based features.

    Includes:
    - Cyclical month encoding (sin/cos) - preserves December/January proximity
    - Survey interval (4 vs 5 weeks)
    - Quarter indicators
    """
    df = df.copy()

    # --- Cyclical Month Encoding (NO one-hot) ---
    # This preserves the cyclical nature: December is close to January
    df['month_sin'] = np.sin(2 * np.pi * target_month.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * target_month.month / 12)

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
    df['is_jan'] = int(target_month.month == 1)
    # Mid-year benchmark revision month
    df['is_july'] = int(target_month.month == 7)

    # Year (for capturing secular trends if needed)
    df['year'] = target_month.year

    return df


# =============================================================================
# TARGET DATA FEATURES (PAST NFP VALUES)
# =============================================================================

def load_all_target_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both NSA and SA target data with all derived features.

    Returns:
        Tuple of (nsa_df, sa_df) with columns:
        - ds: date
        - y: level
        - y_mom: month-on-month change
        - Additional momentum/divergence features
    """
    nsa_df = pd.read_parquet(TARGET_PATH_NSA).sort_values('ds').reset_index(drop=True)
    sa_df = pd.read_parquet(TARGET_PATH_SA).sort_values('ds').reset_index(drop=True)

    for df, name in [(nsa_df, 'nsa'), (sa_df, 'sa')]:
        # MoM change
        df['y_mom'] = df['y'].diff()

        # Rolling averages (for momentum/divergence)
        df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
        df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
        df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()

        # MoM rolling averages
        df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
        df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()

        # Divergence: Current minus rolling average
        df['divergence_3m'] = df['y'] - df['y_rolling_3m']
        df['divergence_6m'] = df['y'] - df['y_rolling_6m']

        # Acceleration: Diff of diffs (second derivative)
        df['acceleration'] = df['y_mom'].diff()

        # YoY change
        df['y_yoy'] = df['y'].diff(12)
        df['y_mom_yoy'] = df['y_mom'].diff(12)

    return nsa_df, sa_df


def get_lagged_target_features(
    target_df: pd.DataFrame,
    target_month: pd.Timestamp,
    prefix: str = 'nfp'
) -> Dict[str, float]:
    """
    Get lagged target features for a specific prediction month.

    CRITICAL: Only use data from BEFORE target_month to avoid look-ahead bias.
    The target is released in the following month, so for month M prediction,
    we can only use data up to month M-1.

    Args:
        target_df: Target DataFrame with all features
        target_month: Month being predicted
        prefix: Prefix for feature names ('nfp_nsa' or 'nfp_sa')

    Returns:
        Dictionary of lagged features
    """
    features = {}

    # Filter to only data available BEFORE the target month
    # NFP for month M is released in month M+1, so we have M-1 data
    available = target_df[target_df['ds'] < target_month].copy()

    if available.empty:
        return features

    # Latest available data (this is M-1)
    latest = available.iloc[-1]

    # Level lags
    features[f'{prefix}_level_lag1'] = latest['y']
    if len(available) >= 2:
        features[f'{prefix}_level_lag2'] = available.iloc[-2]['y']
    if len(available) >= 3:
        features[f'{prefix}_level_lag3'] = available.iloc[-3]['y']
    if len(available) >= 12:
        features[f'{prefix}_level_lag12'] = available.iloc[-12]['y']

    # MoM change lags (critical features)
    if pd.notna(latest['y_mom']):
        features[f'{prefix}_mom_lag1'] = latest['y_mom']
    if len(available) >= 2 and pd.notna(available.iloc[-2]['y_mom']):
        features[f'{prefix}_mom_lag2'] = available.iloc[-2]['y_mom']
    if len(available) >= 3 and pd.notna(available.iloc[-3]['y_mom']):
        features[f'{prefix}_mom_lag3'] = available.iloc[-3]['y_mom']
    if len(available) >= 12 and pd.notna(available.iloc[-12]['y_mom']):
        features[f'{prefix}_mom_lag12'] = available.iloc[-12]['y_mom']

    # Rolling averages
    if pd.notna(latest.get('y_rolling_3m')):
        features[f'{prefix}_rolling_3m'] = latest['y_rolling_3m']
    if pd.notna(latest.get('y_rolling_6m')):
        features[f'{prefix}_rolling_6m'] = latest['y_rolling_6m']
    if pd.notna(latest.get('y_mom_rolling_3m')):
        features[f'{prefix}_mom_rolling_3m'] = latest['y_mom_rolling_3m']
    if pd.notna(latest.get('y_mom_rolling_6m')):
        features[f'{prefix}_mom_rolling_6m'] = latest['y_mom_rolling_6m']

    # Divergence features (momentum indicators)
    if pd.notna(latest.get('divergence_3m')):
        features[f'{prefix}_divergence_3m'] = latest['divergence_3m']
    if pd.notna(latest.get('divergence_6m')):
        features[f'{prefix}_divergence_6m'] = latest['divergence_6m']

    # Acceleration (second derivative)
    if pd.notna(latest.get('acceleration')):
        features[f'{prefix}_acceleration'] = latest['acceleration']

    # YoY features
    if pd.notna(latest.get('y_yoy')):
        features[f'{prefix}_yoy'] = latest['y_yoy']

    return features


def get_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """Build path to master snapshot file."""
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')
    return MASTER_SNAPSHOTS_DIR / decade / year / f"{month_str}.parquet"


def load_master_snapshot(snapshot_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Load master snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)

    Returns:
        DataFrame with columns: date, series_name, value, release_date, series_code, snapshot_date
    """
    path = get_snapshot_path(snapshot_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_target_data(target_type: str = 'nsa') -> pd.DataFrame:
    """
    Load and prepare target data (NSA or SA NFP levels) with derived features.

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted

    Returns:
        DataFrame with columns: ds, y (level), y_mom (month-on-month change),
        and additional momentum/divergence features
    """
    if target_type.lower() == 'sa':
        target_path = TARGET_PATH_SA
        logger.info("Loading SA (seasonally adjusted) target data")
    else:
        target_path = TARGET_PATH_NSA
        logger.info("Loading NSA (non-seasonally adjusted) target data")

    df = pd.read_parquet(target_path)
    df = df.sort_values('ds').reset_index(drop=True)

    # Calculate month-on-month change
    df['y_mom'] = df['y'].diff()

    # Rolling averages (for momentum/divergence features)
    df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
    df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
    df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()

    # MoM rolling averages
    df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()

    # Divergence: Current minus rolling average (momentum indicator)
    df['divergence_3m'] = df['y'] - df['y_rolling_3m']
    df['divergence_6m'] = df['y'] - df['y_rolling_6m']

    # Acceleration: Diff of diffs (second derivative - is hiring accelerating?)
    df['acceleration'] = df['y_mom'].diff()

    # YoY change
    df['y_yoy'] = df['y'].diff(12)
    df['y_mom_yoy'] = df['y_mom'].diff(12)

    # Drop first row (no MoM for first observation) but keep future rows with NaN y
    # Future rows will have y=NaN (unknown future value) and y_mom=NaN (can't calculate)
    # We drop only the first row where we CAN'T calculate MoM due to lack of prior value
    df = df[df.index != 0].reset_index(drop=True)

    logger.info(f"Loaded {target_type.upper()} target data: {len(df)} observations from {df['ds'].min()} to {df['ds'].max()}")
    return df


def pivot_snapshot_to_wide(snapshot_df: pd.DataFrame, target_month: pd.Timestamp) -> pd.DataFrame:
    """
    Convert long-format snapshot to wide format, taking the latest value for target month.

    Uses only data available at the snapshot date to predict month M.
    Includes short-term (1-3), medium-term (6), and long-term (12, 18, 24) lags
    for capturing macroeconomic cycles and structural trends.

    Args:
        snapshot_df: Long-format snapshot DataFrame
        target_month: Month we're predicting (format: YYYY-MM-01)

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    features = {}

    # Get unique series in the snapshot
    series_names = snapshot_df['series_name'].unique()

    for series_name in series_names:
        series_data = snapshot_df[snapshot_df['series_name'] == series_name].copy()
        series_data = series_data.sort_values('date')

        if series_data.empty:
            continue

        # Get the latest available value (most recent date <= target_month)
        # This ensures no look-ahead bias
        available_data = series_data[series_data['date'] <= target_month]

        if not available_data.empty:
            # Latest value
            latest_row = available_data.iloc[-1]
            features[f'{series_name}_latest'] = latest_row['value']

            # --- Short-term lags (1, 2, 3 months) ---
            # Capture recent momentum and business cycle dynamics
            if len(available_data) >= 2:
                features[f'{series_name}_lag1'] = available_data.iloc[-2]['value']
                features[f'{series_name}_mom_change'] = latest_row['value'] - available_data.iloc[-2]['value']

            if len(available_data) >= 3:
                features[f'{series_name}_lag2'] = available_data.iloc[-3]['value']

            if len(available_data) >= 4:
                features[f'{series_name}_lag3'] = available_data.iloc[-4]['value']

            # --- Medium-term lag (6 months) ---
            # Semi-annual patterns
            if len(available_data) >= 7:
                features[f'{series_name}_lag6'] = available_data.iloc[-7]['value']
                features[f'{series_name}_6m_change'] = latest_row['value'] - available_data.iloc[-7]['value']

            # --- Long-term lags (12, 18, 24 months) ---
            # Structural trends and secular patterns in macroeconomic data
            if len(available_data) >= 13:
                features[f'{series_name}_lag12'] = available_data.iloc[-13]['value']
                features[f'{series_name}_yoy_change'] = latest_row['value'] - available_data.iloc[-13]['value']

            if len(available_data) >= 19:
                features[f'{series_name}_lag18'] = available_data.iloc[-19]['value']
                features[f'{series_name}_18m_change'] = latest_row['value'] - available_data.iloc[-19]['value']

            if len(available_data) >= 25:
                features[f'{series_name}_lag24'] = available_data.iloc[-25]['value']
                features[f'{series_name}_2yr_change'] = latest_row['value'] - available_data.iloc[-25]['value']

            # --- Rolling means (smoothed signals) ---
            if len(available_data) >= 3:
                features[f'{series_name}_rolling_mean_3'] = available_data['value'].iloc[-3:].mean()

            if len(available_data) >= 6:
                features[f'{series_name}_rolling_mean_6'] = available_data['value'].iloc[-6:].mean()

            if len(available_data) >= 12:
                features[f'{series_name}_rolling_mean_12'] = available_data['value'].iloc[-12:].mean()

            if len(available_data) >= 24:
                features[f'{series_name}_rolling_mean_24'] = available_data['value'].iloc[-24:].mean()

            # --- Volatility features ---
            if len(available_data) >= 6:
                recent_values = available_data['value'].iloc[-6:]
                features[f'{series_name}_volatility_6m'] = recent_values.std()

            if len(available_data) >= 12:
                recent_values = available_data['value'].iloc[-12:]
                features[f'{series_name}_volatility_12m'] = recent_values.std()

            # --- Trend features ---
            if len(available_data) >= 12:
                # Linear trend slope over last 12 months
                recent_values = available_data['value'].iloc[-12:].values
                x = np.arange(12)
                if not np.isnan(recent_values).any():
                    try:
                        slope = np.polyfit(x, recent_values, 1)[0]
                        features[f'{series_name}_trend_12m'] = slope
                    except (np.linalg.LinAlgError, ValueError):
                        pass

    if not features:
        return pd.DataFrame()

    return pd.DataFrame([features])


def build_training_dataset(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset by joining snapshots with targets.

    Uses snapshot of month M to predict month M's MoM change.
    Includes lagged target features from BOTH NSA and SA data.

    Args:
        target_df: Target DataFrame with y_mom column (the prediction target)
        target_type: 'nsa' or 'sa' - determines which target we're predicting
        start_date: Start date for training data
        end_date: End date for training data

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    all_features = []
    all_targets = []
    valid_dates = []

    # Load both NSA and SA target data for feature engineering
    # This allows us to use both as predictors regardless of which we're predicting
    logger.info("Loading NSA and SA target data for feature engineering...")
    nsa_target_full = load_target_data('nsa')
    sa_target_full = load_target_data('sa')

    # Filter target data by date range
    filtered_df = target_df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['ds'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['ds'] <= end_date]

    logger.info(f"Building features for {len(filtered_df)} target months...")

    for idx, row in filtered_df.iterrows():
        target_month = row['ds']
        target_value = row['y_mom']

        # Get snapshot for this month (month-end date)
        snapshot_date = target_month + pd.offsets.MonthEnd(0)

        # Load and process snapshot
        snapshot_df = load_master_snapshot(snapshot_date)

        if snapshot_df is None or snapshot_df.empty:
            continue

        # Convert to wide format (exogenous features)
        features = pivot_snapshot_to_wide(snapshot_df, target_month)

        if features.empty:
            continue

        # Add calendar features (including survey interval)
        features = add_calendar_features(features, target_month)

        # Add lagged target features from BOTH NSA and SA data
        # These use only data BEFORE target_month to avoid look-ahead bias
        nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
        sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')

        # Merge target features into features DataFrame
        for k, v in {**nsa_target_features, **sa_target_features}.items():
            features[k] = v

        # Add FRED employment features (endogenous data)
        # Load FRED employment snapshot for this month
        # COMMENTED OUT: Testing model without employment subsector features
        # fred_df = load_fred_snapshot(snapshot_date)
        # if fred_df is not None and not fred_df.empty:
        #     employment_features = engineer_employment_features(fred_df, target_month, target_type)
        #     for k, v in employment_features.items():
        #         features[k] = v

        all_features.append(features)
        all_targets.append(target_value)
        valid_dates.append(target_month)

    if not all_features:
        logger.error("No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Combine all features
    X = pd.concat(all_features, ignore_index=True)
    y = pd.Series(all_targets, name='y_mom')

    # Add date index for reference
    X['ds'] = valid_dates

    # Handle missing values - fill with column mean (or 0 for truly missing)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    X = X.fillna(0)

    # Count feature categories
    calendar_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'weeks_since_last_survey', 'is_5_week_month', 'is_jan', 'is_july', 'year']
    target_features = [c for c in X.columns if c.startswith('nfp_')]
    employment_features = [c for c in X.columns if c.startswith('emp_')]
    exog_features = [c for c in X.columns if not c.startswith('nfp_') and not c.startswith('emp_') and c not in calendar_features and c != 'ds']

    logger.info(f"Built training dataset: {len(X)} samples, {len(X.columns)} total features")
    logger.info(f"  - Exogenous (macro) features: {len([c for c in exog_features if c in X.columns])}")
    logger.info(f"  - Employment (FRED) features: {len([c for c in employment_features if c in X.columns])}")
    logger.info(f"  - Calendar features: {len([c for c in calendar_features if c in X.columns])}")
    logger.info(f"  - Target (NFP) lag features: {len([c for c in target_features if c in X.columns])}")

    return X, y


def get_lgbm_params(
    use_huber_loss: bool = False,
    huber_delta: float = 1.0
) -> Dict:
    """
    Get LightGBM parameters with optional Huber loss.

    Args:
        use_huber_loss: If True, use Huber objective (robust to outliers)
        huber_delta: Huber delta parameter (transition point between L1 and L2)

    Returns:
        Dictionary of LightGBM parameters
    """
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
        'force_row_wise': True
    }

    if use_huber_loss:
        params['objective'] = 'huber'
        params['alpha'] = huber_delta  # Huber delta parameter
        params['metric'] = ['huber', 'mae']
        logger.info(f"Using Huber loss with delta={huber_delta}")
    else:
        params['objective'] = 'regression'
        params['metric'] = ['rmse', 'mae']
        logger.info("Using standard RMSE regression loss")

    return params


def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    use_huber_loss: bool = False,
    huber_delta: float = 1.0
) -> Tuple[lgb.Booster, Dict, List[float]]:
    """
    Train LightGBM model with time-series cross-validation.

    Args:
        X: Feature DataFrame (includes 'ds' column for date reference)
        y: Target Series
        n_splits: Number of time series CV splits
        num_boost_round: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience

    Returns:
        Tuple of (trained_model, feature_importance_dict, cv_residuals)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required")

    # Separate date column
    dates = X['ds'] if 'ds' in X.columns else None
    feature_cols = [c for c in X.columns if c != 'ds']
    X_train = X[feature_cols]

    logger.info(f"Training LightGBM on {len(X_train)} samples with {len(feature_cols)} features")

    # LightGBM parameters optimized for small datasets
    params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_residuals = []
    oof_predictions = np.zeros(len(X_train))

    logger.info(f"Running {n_splits}-fold time series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        evals_result = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=0),  # Suppress per-iteration output
            lgb.record_evaluation(evals_result)
        ]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # Get predictions and residuals
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        fold_residuals = y_val.values - val_pred
        cv_residuals.extend(fold_residuals.tolist())

        fold_rmse = np.sqrt(np.mean(fold_residuals ** 2))
        fold_mae = np.mean(np.abs(fold_residuals))
        cv_scores.append({'fold': fold + 1, 'rmse': fold_rmse, 'mae': fold_mae})

        logger.info(f"Fold {fold + 1}: RMSE = {fold_rmse:.2f}, MAE = {fold_mae:.2f}")

    # Overall CV performance
    mean_rmse = np.mean([s['rmse'] for s in cv_scores])
    mean_mae = np.mean([s['mae'] for s in cv_scores])
    logger.info(f"\nOverall CV Performance: RMSE = {mean_rmse:.2f}, MAE = {mean_mae:.2f}")

    # Train final model on all data
    logger.info("\nTraining final model on all data...")

    # Split for final early stopping
    train_size = int(len(X_train) * 0.85)
    X_final_train = X_train.iloc[:train_size]
    X_final_val = X_train.iloc[train_size:]
    y_final_train = y.iloc[:train_size]
    y_final_val = y.iloc[train_size:]

    train_data = lgb.Dataset(X_final_train, label=y_final_train)
    val_data = lgb.Dataset(X_final_val, label=y_final_val, reference=train_data)

    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result)
    ]
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importance(importance_type='gain')))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info(f"\nTop 15 most important features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:15], 1):
        logger.info(f"  {i}. {feat}: {imp:.1f}")

    return final_model, importance, cv_residuals


def calculate_prediction_intervals(
    residuals: List[float],
    prediction: float,
    confidence_levels: List[float] = [0.50, 0.80, 0.95]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate prediction intervals based on historical residuals.

    Uses empirical quantiles of residuals for non-parametric intervals.

    Args:
        residuals: List of historical prediction residuals
        prediction: Point prediction
        confidence_levels: List of confidence levels (e.g., [0.50, 0.80, 0.95])

    Returns:
        Dictionary with confidence level as key and (lower, upper) bounds as value
    """
    residuals = np.array(residuals)
    intervals = {}

    for level in confidence_levels:
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        lower_residual = np.percentile(residuals, lower_q * 100)
        upper_residual = np.percentile(residuals, upper_q * 100)

        # Prediction intervals: prediction + residual quantiles
        lower_bound = prediction + lower_residual
        upper_bound = prediction + upper_residual

        intervals[f'{int(level * 100)}%'] = (lower_bound, upper_bound)

    # Also add standard deviation based interval
    std_residual = np.std(residuals)
    intervals['std'] = std_residual
    intervals['mean_residual'] = np.mean(residuals)

    return intervals


def predict_with_intervals(
    model: lgb.Booster,
    features: pd.DataFrame,
    residuals: List[float],
    feature_cols: List[str]
) -> Dict:
    """
    Make prediction with confidence intervals.

    Args:
        model: Trained LightGBM model
        features: Feature DataFrame (single row)
        residuals: Historical residuals for interval calculation
        feature_cols: List of feature column names used in training

    Returns:
        Dictionary with prediction and interval information
    """
    # Align features with training columns
    missing_cols = set(feature_cols) - set(features.columns)
    for col in missing_cols:
        features[col] = 0

    # Select and order columns
    X = features[feature_cols]

    # Make prediction
    prediction = model.predict(X)[0]

    # Calculate intervals
    intervals = calculate_prediction_intervals(residuals, prediction)

    result = {
        'prediction': prediction,
        'intervals': {
            '50%': intervals['50%'],
            '80%': intervals['80%'],
            '95%': intervals['95%']
        },
        'std': intervals['std'],
        'mean_residual_bias': intervals['mean_residual']
    }

    return result


def save_model(
    model: lgb.Booster,
    feature_cols: List[str],
    residuals: List[float],
    importance: Dict,
    save_dir: Path = MODEL_SAVE_DIR,
    target_type: str = 'nsa',
    linear_model: Optional[Any] = None,
    linear_cols: Optional[List[str]] = None
):
    """Save model and associated metadata."""
    # Create subdirectory for target type
    type_save_dir = save_dir / target_type
    type_save_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM model
    model_path = type_save_dir / f"lightgbm_nfp_{target_type}_model.txt"
    model.save_model(str(model_path))

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'residuals': residuals,
        'importance': importance,
        'trained_at': datetime.now().isoformat(),
        'std_residual': np.std(residuals),
        'mean_residual': np.mean(residuals),
        'target_type': target_type,
        'linear_model': linear_model,
        'linear_cols': linear_cols
    }

    metadata_path = type_save_dir / f"model_metadata_{target_type}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"Saved {target_type.upper()} model to {model_path}")
    logger.info(f"Saved metadata to {metadata_path}")
    if linear_model is not None:
        logger.info(f"Saved linear baseline model with {len(linear_cols)} predictors")


def load_model(save_dir: Path = MODEL_SAVE_DIR, target_type: str = 'nsa') -> Tuple[lgb.Booster, Dict]:
    """Load model and associated metadata."""
    type_save_dir = save_dir / target_type
    model_path = type_save_dir / f"lightgbm_nfp_{target_type}_model.txt"
    metadata_path = type_save_dir / f"model_metadata_{target_type}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = lgb.Booster(model_file=str(model_path))

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded {target_type.upper()} model from {model_path}")
    return model, metadata


def run_backtest(
    model: lgb.Booster,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    residuals: List[float],
    backtest_start: pd.Timestamp,
    backtest_end: Optional[pd.Timestamp] = None,
    target_type: str = 'nsa'
) -> pd.DataFrame:
    """
    Run out-of-sample backtest on the model.

    Args:
        model: Trained LightGBM model
        target_df: Full target DataFrame
        feature_cols: Feature column names
        residuals: Historical residuals for intervals
        backtest_start: Start date for backtest
        backtest_end: End date for backtest
        target_type: 'nsa' or 'sa' - determines which employment features to use

    Returns:
        DataFrame with backtest results
    """
    backtest_df = target_df[target_df['ds'] >= backtest_start]
    if backtest_end:
        backtest_df = backtest_df[backtest_df['ds'] <= backtest_end]

    # Load target data for lagged features
    nsa_target_full = load_target_data('nsa')
    sa_target_full = load_target_data('sa')

    results = []

    for idx, row in backtest_df.iterrows():
        target_month = row['ds']
        actual = row['y_mom']

        # Get snapshot and create features
        snapshot_date = target_month + pd.offsets.MonthEnd(0)
        snapshot_df = load_master_snapshot(snapshot_date)

        if snapshot_df is None or snapshot_df.empty:
            continue

        features = pivot_snapshot_to_wide(snapshot_df, target_month)
        if features.empty:
            continue

        features = add_calendar_features(features, target_month)

        # Add lagged target features
        nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
        sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')
        for k, v in {**nsa_target_features, **sa_target_features}.items():
            features[k] = v

        # Add FRED employment features
        # COMMENTED OUT: Testing model without employment subsector features
        # fred_df = load_fred_snapshot(snapshot_date)
        # if fred_df is not None and not fred_df.empty:
        #     employment_features = engineer_employment_features(fred_df, target_month, target_type)
        #     for k, v in employment_features.items():
        #         features[k] = v

        # Make prediction with intervals
        pred_result = predict_with_intervals(model, features, residuals, feature_cols)

        results.append({
            'ds': target_month,
            'actual': actual,
            'predicted': pred_result['prediction'],
            'error': actual - pred_result['prediction'],
            'lower_50': pred_result['intervals']['50%'][0],
            'upper_50': pred_result['intervals']['50%'][1],
            'lower_80': pred_result['intervals']['80%'][0],
            'upper_80': pred_result['intervals']['80%'][1],
            'lower_95': pred_result['intervals']['95%'][0],
            'upper_95': pred_result['intervals']['95%'][1],
            'in_50_interval': pred_result['intervals']['50%'][0] <= actual <= pred_result['intervals']['50%'][1],
            'in_80_interval': pred_result['intervals']['80%'][0] <= actual <= pred_result['intervals']['80%'][1],
            'in_95_interval': pred_result['intervals']['95%'][0] <= actual <= pred_result['intervals']['95%'][1]
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Calculate metrics
        rmse = np.sqrt(np.mean(results_df['error'] ** 2))
        mae = np.mean(np.abs(results_df['error']))
        mape = np.mean(np.abs(results_df['error'] / results_df['actual'])) * 100

        # Coverage rates
        coverage_50 = results_df['in_50_interval'].mean() * 100
        coverage_80 = results_df['in_80_interval'].mean() * 100
        coverage_95 = results_df['in_95_interval'].mean() * 100

        logger.info(f"\nBacktest Results ({len(results_df)} predictions):")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"\nInterval Coverage:")
        logger.info(f"  50% CI: {coverage_50:.1f}% (target: 50%)")
        logger.info(f"  80% CI: {coverage_80:.1f}% (target: 80%)")
        logger.info(f"  95% CI: {coverage_95:.1f}% (target: 95%)")

    return results_df


def impute_with_expanding_window(X: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """
    Impute missing values using only data from the training window (no future leakage).

    Args:
        X: Full feature DataFrame
        train_idx: Indices of training data (expanding window)

    Returns:
        DataFrame with NaN values imputed using only training window statistics
    """
    X_imputed = X.copy()
    numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns

    # Calculate means from training window only
    train_means = X_imputed.iloc[train_idx][numeric_cols].mean()

    # Fill NaN using training window means
    X_imputed[numeric_cols] = X_imputed[numeric_cols].fillna(train_means)
    X_imputed = X_imputed.fillna(0)  # Fallback for remaining NaN

    return X_imputed


def run_expanding_window_backtest(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    use_feature_selection: bool = True,
    feature_selection_interval: int = 6,  # Re-run feature selection every N months
    use_huber_loss: bool = False,
    huber_delta: float = 1.0
) -> pd.DataFrame:
    """
    Run proper expanding window backtest with NO TIME-TRAVEL VIOLATIONS.

    Critical Design Principles:
    1. Feature selection is re-run inside the loop using only past data
    2. NaN imputation uses only expanding window means (no future data)
    3. Model is FULLY retrained from scratch at each step using only data available at that time
    4. No information from future time periods leaks into predictions

    Args:
        target_df: Target DataFrame with 'ds' and 'y_mom' columns
        target_type: 'nsa' or 'sa'
        use_feature_selection: Whether to run feature selection
        feature_selection_interval: How often to re-run feature selection (months)

    Returns:
        DataFrame with backtest results
    """
    logger.info("=" * 60)
    logger.info("EXPANDING WINDOW BACKTEST (No Time-Travel)")
    logger.info("=" * 60)
    logger.info(f"Feature selection re-runs every {feature_selection_interval} months")
    logger.info("Model is FULLY retrained from scratch at each step (no warm start)")

    # Determine backtest period
    backtest_start_idx = len(target_df) - BACKTEST_MONTHS
    backtest_months = target_df.iloc[backtest_start_idx:]['ds'].tolist()

    logger.info(f"\nBacktest period: {BACKTEST_MONTHS} months")
    logger.info(f"First prediction: {backtest_months[0]}")
    logger.info(f"Last prediction: {backtest_months[-1]}")

    # Load target data for lagged features (this is OK - we only use past values)
    nsa_target_full = load_target_data('nsa')
    sa_target_full = load_target_data('sa')

    # Build FULL feature dataset (we'll filter by date for each prediction)
    logger.info("\nBuilding full feature dataset...")
    X_full, y_full = build_training_dataset(target_df, target_type=target_type)

    if X_full.empty:
        logger.error("Failed to build training dataset")
        return pd.DataFrame()

    # Store results
    results = []
    all_residuals = []

    # Cache for feature selection results (to avoid redundant computation)
    cached_features = None
    last_feature_selection_idx = -feature_selection_interval  # Force first selection

    logger.info(f"\nRunning {len(backtest_months)} predictions...")

    for i, target_month in enumerate(backtest_months):
        # Get index of this target month in the full dataset
        target_idx = X_full[X_full['ds'] == target_month].index
        if len(target_idx) == 0:
            logger.warning(f"No data for {target_month}, skipping")
            continue
        target_idx = target_idx[0]

        # EXPANDING WINDOW: Training data is everything BEFORE the target month
        train_mask = X_full['ds'] < target_month
        train_idx = X_full[train_mask].index.tolist()

        if len(train_idx) < 24:  # Need at least 2 years of training data
            logger.warning(f"Insufficient training data for {target_month}, skipping")
            continue

        # Get training data (no future leakage)
        X_train_raw = X_full.iloc[train_idx].copy()
        y_train = y_full.iloc[train_idx].copy()

        # Filter out NaN targets from training data (but keep them for prediction)
        # This handles future months where y is unknown
        valid_train_mask = ~y_train.isna()
        train_idx_valid = [train_idx[i] for i in range(len(train_idx)) if valid_train_mask.iloc[i]]

        if len(train_idx_valid) < 24:
            logger.warning(f"Insufficient valid training data for {target_month}, skipping")
            continue

        # CRITICAL: Impute NaN using ONLY training window statistics (use all indices including NaN targets)
        X_train_imputed = impute_with_expanding_window(X_full, train_idx)
        X_train = X_train_imputed.iloc[train_idx].copy()

        # Now filter to only valid (non-NaN) targets for training
        X_train_valid = X_train.iloc[[i for i in range(len(train_idx)) if valid_train_mask.iloc[i]]].copy()
        y_train_valid = y_train[valid_train_mask].copy()

        # FEATURE SELECTION: Re-run periodically using only valid training data
        if use_feature_selection and (i - last_feature_selection_idx >= feature_selection_interval or cached_features is None):
            logger.info(f"\n[{target_month.strftime('%Y-%m')}] Re-running feature selection on {len(train_idx_valid)} samples...")

            # Feature selection on valid training data ONLY
            selected_features, _ = select_features_comprehensive(
                X=X_train_valid,
                y=y_train_valid,
                max_features=MAX_FEATURES,
                output_dir=None,  # Don't save intermediate results
                target_type=target_type
            )
            cached_features = selected_features
            last_feature_selection_idx = i
            logger.info(f"Selected {len(cached_features)} features")

        # Get feature columns for this iteration
        if use_feature_selection and cached_features:
            feature_cols = [c for c in cached_features if c in X_train_valid.columns and c != 'ds']
        else:
            feature_cols = [c for c in X_train_valid.columns if c != 'ds']

        # NEW: Train Linear Baseline and Add as Feature
        linear_model, linear_cols_used = train_linear_baseline(
            X_train_valid,
            y_train_valid,
            LINEAR_BASELINE_PREDICTORS
        )

        if linear_model is not None:
            # Add linear baseline prediction as feature to training data
            X_train_valid['linear_baseline_pred'] = create_linear_baseline_feature(
                X_train_valid,
                linear_model,
                linear_cols_used
            )

            # Ensure it's in feature_cols
            if 'linear_baseline_pred' not in feature_cols:
                feature_cols.append('linear_baseline_pred')

            logger.info(f"Added linear baseline feature using {len(linear_cols_used)} predictors")
        else:
            logger.warning("Linear baseline model failed, proceeding without it")

        # Prepare training data with selected features
        X_train_selected = X_train_valid[feature_cols]

        # LightGBM parameters
        params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

        # Train model (with optional warm start) using only valid targets
        train_size = int(len(X_train_selected) * 0.85)
        X_tr = X_train_selected.iloc[:train_size]
        X_val = X_train_selected.iloc[train_size:]
        y_tr = y_train_valid.iloc[:train_size]
        y_val = y_train_valid.iloc[train_size:]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)
        ]

        # Train model from scratch every time (no warm start)
        # This ensures full retraining with each new data point
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # Calculate residuals from training data for prediction intervals
        train_preds = model.predict(X_train_selected)
        train_residuals = (y_train_valid.values - train_preds).tolist()
        all_residuals.extend(train_residuals[-12:])  # Keep recent residuals

        # PREDICTION: Get features for target month using same imputation
        X_pred_imputed = impute_with_expanding_window(X_full, train_idx)
        X_pred = X_pred_imputed.iloc[[target_idx]].copy()

        # NEW: Add linear baseline feature to prediction
        if linear_model is not None:
            X_pred['linear_baseline_pred'] = create_linear_baseline_feature(
                X_pred,
                linear_model,
                linear_cols_used
            )

        X_pred = X_pred[feature_cols]

        # Make prediction
        prediction = model.predict(X_pred)[0]
        actual = y_full.iloc[target_idx]

        # Calculate prediction intervals from historical residuals
        if len(all_residuals) > 10:
            residual_array = np.array(all_residuals[-36:])  # Use last 36 months
            lower_50 = prediction + np.percentile(residual_array, 25)
            upper_50 = prediction + np.percentile(residual_array, 75)
            lower_80 = prediction + np.percentile(residual_array, 10)
            upper_80 = prediction + np.percentile(residual_array, 90)
            lower_95 = prediction + np.percentile(residual_array, 2.5)
            upper_95 = prediction + np.percentile(residual_array, 97.5)
        else:
            std_est = np.std(train_residuals) if train_residuals else 50
            lower_50, upper_50 = prediction - 0.67*std_est, prediction + 0.67*std_est
            lower_80, upper_80 = prediction - 1.28*std_est, prediction + 1.28*std_est
            lower_95, upper_95 = prediction - 1.96*std_est, prediction + 1.96*std_est

        # Handle NaN actuals (future predictions)
        is_future = pd.isna(actual)
        error = np.nan if is_future else actual - prediction
        in_50 = np.nan if is_future else (lower_50 <= actual <= upper_50)
        in_80 = np.nan if is_future else (lower_80 <= actual <= upper_80)
        in_95 = np.nan if is_future else (lower_95 <= actual <= upper_95)

        results.append({
            'ds': target_month,
            'actual': actual,
            'predicted': prediction,
            'error': error,
            'lower_50': lower_50,
            'upper_50': upper_50,
            'lower_80': lower_80,
            'upper_80': upper_80,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'in_50_interval': in_50,
            'in_80_interval': in_80,
            'in_95_interval': in_95,
            'n_train_samples': len(train_idx_valid),
            'n_features': len(feature_cols)
        })

        # Progress logging
        if (i + 1) % 6 == 0 or i == 0:
            if is_future:
                logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                           f"Pred={prediction:.0f} (FUTURE - no actual yet)")
            else:
                logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                           f"Actual={actual:.0f}, Pred={prediction:.0f}, Error={error:.0f}")

    results_df = pd.DataFrame(results)

    # Log summary statistics (excluding future predictions with NaN actuals)
    if not results_df.empty:
        # Filter to only rows with actual values (non-NaN) for metrics
        backtest_rows = results_df[~results_df['error'].isna()]
        future_rows = results_df[results_df['error'].isna()]

        if not backtest_rows.empty:
            rmse = np.sqrt(np.mean(backtest_rows['error'] ** 2))
            mae = np.mean(np.abs(backtest_rows['error']))

            logger.info("\n" + "=" * 60)
            logger.info("EXPANDING WINDOW BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Backtest predictions: {len(backtest_rows)}")
            logger.info(f"Future predictions: {len(future_rows)}")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"Coverage 50%: {backtest_rows['in_50_interval'].mean()*100:.1f}%")
            logger.info(f"Coverage 80%: {backtest_rows['in_80_interval'].mean()*100:.1f}%")
            logger.info(f"Coverage 95%: {backtest_rows['in_95_interval'].mean()*100:.1f}%")

        if not future_rows.empty:
            logger.info("\nFUTURE PREDICTIONS (No actuals yet):")
            for _, row in future_rows.iterrows():
                logger.info(f"  {row['ds'].strftime('%Y-%m')}: Pred={row['predicted']:.0f} "
                           f"[{row['lower_80']:.0f}, {row['upper_80']:.0f}]")

    return results_df


def train_and_evaluate(
    target_type: str = 'nsa',
    use_feature_selection: bool = True,
    use_huber_loss: bool = False,
    huber_delta: float = 1.0
):
    """
    Main training and evaluation function using EXPANDING WINDOW methodology.

    This function ensures NO TIME-TRAVEL VIOLATIONS:
    1. Feature selection is re-evaluated inside the expanding window loop
    2. NaN imputation uses only past data (expanding window means)
    3. Model training uses only data available at each prediction time

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        use_feature_selection: If True, run comprehensive feature selection
    """
    # Archive previous backtest results before starting new run
    # Only archive once per training session (when running NSA model)
    if target_type == 'nsa':
        try:
            sys.path.append(str(Path(__file__).resolve().parent))
            from backtest_archiver import prepare_new_backtest_run

            archive_path = prepare_new_backtest_run()
            if archive_path:
                logger.info(f"✓ Archived previous backtest to: {archive_path.name}")
            else:
                logger.info("No previous backtest results to archive")
        except Exception as e:
            logger.warning(f"Failed to archive backtest results: {e}")
            logger.info("Continuing with training...")

    logger.info("=" * 60)
    logger.info(f"LightGBM NFP Prediction Model - Training ({target_type.upper()})")
    logger.info("=" * 60)
    logger.info("Using EXPANDING WINDOW methodology (no time-travel)")

    # Load target data
    target_df = load_target_data(target_type=target_type)

    # Determine date ranges
    train_end = target_df['ds'].max() - pd.DateOffset(months=BACKTEST_MONTHS)

    logger.info(f"\nInitial training period: {target_df['ds'].min()} to {train_end}")
    logger.info(f"Backtest period: {train_end} to {target_df['ds'].max()}")

    # Run expanding window backtest (feature selection happens INSIDE the loop)
    backtest_results = run_expanding_window_backtest(
        target_df=target_df,
        target_type=target_type,
        use_feature_selection=use_feature_selection,
        feature_selection_interval=6,  # Re-select features every 6 months
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta
    )

    if backtest_results.empty:
        logger.error("Backtest produced no results")
        return

    # Train final production model on ALL data (for future predictions)
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING FINAL PRODUCTION MODEL")
    logger.info("=" * 60)

    X_full, y_full = build_training_dataset(target_df, target_type=target_type)

    # Filter out NaN targets for final model training (same as backtest logic)
    # Keep future months in X_full for potential predictions but exclude from training
    valid_mask = ~y_full.isna()
    X_full_valid = X_full[valid_mask].copy()
    y_full_valid = y_full[valid_mask].copy()

    logger.info(f"Total observations: {len(X_full)}, Valid for training: {len(X_full_valid)}")

    # NEW: Train linear baseline on full training data BEFORE feature selection
    logger.info("\nTraining Linear Baseline Model for Production...")
    linear_model_prod, linear_cols_prod = train_linear_baseline(
        X_full_valid,
        y_full_valid,
        LINEAR_BASELINE_PREDICTORS
    )

    if linear_model_prod is not None:
        X_full_valid['linear_baseline_pred'] = create_linear_baseline_feature(
            X_full_valid,
            linear_model_prod,
            linear_cols_prod
        )

    # Final feature selection on full training data (for production model only)
    if use_feature_selection:
        feature_output_dir = OUTPUT_DIR / "feature_selection" / target_type
        feature_output_dir.mkdir(parents=True, exist_ok=True)

        selected_features, selection_metadata = select_features_comprehensive(
            X=X_full_valid,
            y=y_full_valid,
            max_features=MAX_FEATURES,
            output_dir=feature_output_dir,
            target_type=target_type
        )
        feature_cols = selected_features
        X_train = X_full_valid[['ds'] + [c for c in selected_features if c in X_full_valid.columns]].copy()
    else:
        feature_cols = [c for c in X_full_valid.columns if c != 'ds']
        X_train = X_full_valid.copy()

    # Train final model (using only valid targets)
    model, importance, residuals = train_lightgbm_model(
        X_train,
        y_full_valid,
        n_splits=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta
    )

    # Save production model
    save_model(
        model,
        feature_cols,
        residuals,
        importance,
        target_type=target_type,
        linear_model=linear_model_prod,
        linear_cols=linear_cols_prod
    )

    # Save backtest results
    results_dir = OUTPUT_DIR / "backtest_results" / target_type
    results_dir.mkdir(parents=True, exist_ok=True)

    if not backtest_results.empty:
        results_path = results_dir / f"backtest_results_{target_type}.parquet"
        backtest_results.to_parquet(results_path, index=False)
        logger.info(f"\nSaved backtest results to {results_path}")

        # Also save as CSV for easy viewing
        csv_path = results_dir / f"backtest_results_{target_type}.csv"
        backtest_results.to_csv(csv_path, index=False)
        logger.info(f"Saved backtest CSV to {csv_path}")

        # Save summary statistics
        summary = {
            'target_type': target_type,
            'n_predictions': len(backtest_results),
            'rmse': np.sqrt(np.mean(backtest_results['error'] ** 2)),
            'mae': np.mean(np.abs(backtest_results['error'])),
            'mape': np.mean(np.abs(backtest_results['error'] / backtest_results['actual'])) * 100,
            'coverage_50': backtest_results['in_50_interval'].mean() * 100,
            'coverage_80': backtest_results['in_80_interval'].mean() * 100,
            'coverage_95': backtest_results['in_95_interval'].mean() * 100,
            'n_features': len(feature_cols),
            'train_end': str(train_end.date()),
            'backtest_start': str((train_end + pd.DateOffset(months=1)).date()),
            'backtest_end': str(target_df['ds'].max().date())
        }

        summary_path = results_dir / f"model_summary_{target_type}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        logger.info(f"Saved model summary to {summary_path}")

    # Save feature importance to output
    importance_dir = OUTPUT_DIR / "feature_importance" / target_type
    importance_dir.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    importance_path = importance_dir / f"feature_importance_{target_type}.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")

    # Generate comprehensive backtest report using dedicated module
    logger.info("\n" + "=" * 60)
    logger.info("Generating Backtest Report")
    logger.info("=" * 60)

    try:
        # Prepare data for backtest report
        report_df = backtest_results.copy()
        report_df = report_df.rename(columns={
            'ds': 'date',
            'predicted': 'pred'
        })

        # Generate comprehensive report (visualizations, metrics, tables)
        report_files = generate_backtest_report(
            predictions_df=report_df,
            selected_features=feature_cols,
            output_dir=results_dir
        )

        logger.info(f"✓ Generated {len(report_files)} report files")

    except Exception as e:
        logger.warning(f"Failed to generate backtest report: {e}")

    return model, feature_cols, residuals, backtest_results


def predict_nfp_mom(
    target_month: pd.Timestamp,
    model: Optional[lgb.Booster] = None,
    metadata: Optional[Dict] = None,
    target_type: str = 'nsa'
) -> Dict:
    """
    Make NFP MoM prediction for a specific month.

    Uses snapshot of month M to predict month M's MoM change.

    Args:
        target_month: Month to predict (format: YYYY-MM-01 or YYYY-MM-DD)
        model: Optional pre-loaded model. If None, loads from disk.
        metadata: Optional pre-loaded metadata. If None, loads from disk.
        target_type: 'nsa' or 'sa' - determines which model to load

    Returns:
        Dictionary with:
            - prediction: Point estimate of MoM change
            - intervals: Confidence intervals (50%, 80%, 95%)
            - std: Standard deviation of prediction error
            - target_month: The month being predicted
            - features_used: Number of features used
            - target_type: Type of prediction (nsa or sa)
    """
    # Normalize target_month to first of month
    target_month = pd.Timestamp(target_month).replace(day=1)

    # Load model if not provided
    if model is None or metadata is None:
        model, metadata = load_model(target_type=target_type)

    feature_cols = metadata['feature_cols']
    residuals = metadata['residuals']

    # Get snapshot for this month
    snapshot_date = target_month + pd.offsets.MonthEnd(0)
    snapshot_df = load_master_snapshot(snapshot_date)

    if snapshot_df is None or snapshot_df.empty:
        raise FileNotFoundError(f"No snapshot available for {snapshot_date}")

    # Create features
    features = pivot_snapshot_to_wide(snapshot_df, target_month)

    if features.empty:
        raise ValueError(f"Could not create features for {target_month}")

    features = add_calendar_features(features, target_month)

    # Add lagged target features
    nsa_target_full = load_target_data('nsa')
    sa_target_full = load_target_data('sa')
    nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
    sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')
    for k, v in {**nsa_target_features, **sa_target_features}.items():
        features[k] = v

    # Add FRED employment features
    # COMMENTED OUT: Testing model without employment subsector features
    # fred_df = load_fred_snapshot(snapshot_date)
    # if fred_df is not None and not fred_df.empty:
    #     employment_features = engineer_employment_features(fred_df, target_month, target_type)
    #     for k, v in employment_features.items():
    #         features[k] = v

    # Make prediction with intervals
    pred_result = predict_with_intervals(model, features, residuals, feature_cols)

    return {
        'target_month': target_month,
        'prediction': pred_result['prediction'],
        'intervals': pred_result['intervals'],
        'std': pred_result['std'],
        'mean_residual_bias': pred_result['mean_residual_bias'],
        'features_used': len(feature_cols),
        'target_type': target_type
    }


def train_linear_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    predictor_cols: List[str]
) -> Tuple[Any, List[str]]:
    """
    Train simple OLS baseline using key predictors for extrapolation.

    This linear model can extrapolate beyond training data ranges,
    unlike tree-based models which are bounded by training leaves.

    Args:
        X: Feature DataFrame
        y: Target Series
        predictor_cols: List of column names to use as predictors

    Returns:
        Tuple of (trained LinearRegression model, actual columns used)
    """
    if not LIGHTGBM_AVAILABLE:
        logger.warning("sklearn not available, cannot train linear baseline")
        return None, []

    # Filter to only requested predictors that exist
    available_cols = [c for c in predictor_cols if c in X.columns]

    if not available_cols:
        logger.warning("No linear baseline predictors found, returning None")
        return None, []

    X_linear = X[available_cols].copy()

    # Handle missing values (fill with 0 for simplicity)
    X_linear = X_linear.fillna(0)

    # Train OLS
    model = LinearRegression()
    model.fit(X_linear, y)

    # Log coefficients
    logger.info("Linear Baseline Model Coefficients:")
    for col, coef in zip(available_cols, model.coef_):
        logger.info(f"  {col}: {coef:.4f}")
    logger.info(f"  Intercept: {model.intercept_:.4f}")

    # Calculate R² on training data
    r2 = model.score(X_linear, y)
    logger.info(f"  Training R²: {r2:.4f}")

    return model, available_cols


def create_linear_baseline_feature(
    X: pd.DataFrame,
    linear_model: Any,
    predictor_cols: List[str]
) -> pd.Series:
    """
    Generate predictions from linear baseline model as a feature.

    Args:
        X: Feature DataFrame
        linear_model: Trained LinearRegression model
        predictor_cols: Column names used by linear model

    Returns:
        Series of linear baseline predictions
    """
    if linear_model is None or not predictor_cols:
        return pd.Series(0, index=X.index)

    X_linear = X[predictor_cols].copy()
    X_linear = X_linear.fillna(0)

    predictions = linear_model.predict(X_linear)
    return pd.Series(predictions, index=X.index, name='linear_baseline_pred')


def convert_mom_to_level(
    mom_prediction: float,
    previous_level: float
) -> float:
    """
    Convert MoM change prediction to level prediction.

    Args:
        mom_prediction: Month-on-month change prediction
        previous_level: Previous month's level

    Returns:
        Predicted level for current month
    """
    return previous_level + mom_prediction


def get_latest_prediction(target_type: str = 'nsa') -> Dict:
    """
    Get prediction for the most recent available month.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        Prediction dictionary
    """
    # Find latest snapshot available
    target_df = load_target_data(target_type=target_type)
    latest_target = target_df['ds'].max()

    logger.info(f"Making {target_type.upper()} prediction for latest available month: {latest_target}")

    return predict_nfp_mom(latest_target, target_type=target_type)


# =============================================================================
# VIF AND CORRELATION ANALYSIS
# =============================================================================

def compute_vif(X: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for feature multicollinearity analysis.

    VIF > 5 indicates moderate multicollinearity
    VIF > 10 indicates high multicollinearity (consider removing)

    Args:
        X: Feature DataFrame (numeric columns only)
        max_features: Maximum features to analyze (for performance)

    Returns:
        DataFrame with feature names and their VIF values, sorted by VIF descending
    """
    if not VIF_AVAILABLE:
        logger.warning("statsmodels not available for VIF calculation")
        return pd.DataFrame()

    # Select numeric columns only, exclude 'ds'
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'ds']

    # Handle inf/nan values
    X_clean = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.dropna(axis=1, how='any')

    # Limit features for performance
    if len(X_clean.columns) > max_features:
        # Select features with highest variance (most informative)
        variances = X_clean.var().sort_values(ascending=False)
        selected_cols = variances.head(max_features).index.tolist()
        X_clean = X_clean[selected_cols]
        logger.info(f"VIF analysis limited to top {max_features} features by variance")

    # Add constant for VIF calculation
    X_with_const = X_clean.copy()
    X_with_const['const'] = 1.0

    vif_data = []
    feature_cols = [c for c in X_with_const.columns if c != 'const']

    logger.info(f"Computing VIF for {len(feature_cols)} features...")

    for i, col in enumerate(feature_cols):
        try:
            vif_value = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(col))
            vif_data.append({'feature': col, 'VIF': vif_value})
        except Exception as e:
            logger.debug(f"Could not compute VIF for {col}: {e}")
            vif_data.append({'feature': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)

    # Summary statistics
    high_vif = vif_df[vif_df['VIF'] > 10]
    moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)]

    logger.info(f"\nVIF Analysis Summary:")
    logger.info(f"  Total features analyzed: {len(vif_df)}")
    logger.info(f"  High multicollinearity (VIF > 10): {len(high_vif)} features")
    logger.info(f"  Moderate multicollinearity (VIF 5-10): {len(moderate_vif)} features")

    if len(high_vif) > 0:
        logger.info(f"\nTop 10 features with highest VIF:")
        for idx, row in high_vif.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['VIF']:.2f}")

    return vif_df


def compute_correlation_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 30,
    high_corr_threshold: float = 0.9
) -> Dict:
    """
    Compute correlation analysis between features and target.

    Also identifies highly correlated feature pairs (potential redundancy).

    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of top correlated features to report
        high_corr_threshold: Threshold for identifying highly correlated pairs

    Returns:
        Dictionary with correlation analysis results
    """
    # Select numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'ds']
    X_numeric = X[numeric_cols].copy()

    # Handle inf/nan
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    # Feature-target correlations
    target_correlations = {}
    for col in X_numeric.columns:
        valid_mask = X_numeric[col].notna() & y.notna()
        if valid_mask.sum() > 10:  # Minimum samples
            corr = X_numeric.loc[valid_mask, col].corr(y[valid_mask])
            target_correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(target_correlations.items(), key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)

    logger.info(f"\nTop {top_n} Features by Target Correlation:")
    for i, (feat, corr) in enumerate(sorted_corrs[:top_n], 1):
        if pd.notna(corr):
            logger.info(f"  {i}. {feat}: {corr:.4f}")

    # Feature-feature correlations (identify redundancy)
    logger.info(f"\nComputing feature-feature correlation matrix...")
    corr_matrix = X_numeric.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

    if high_corr_pairs:
        logger.info(f"\nHighly Correlated Feature Pairs (|r| > {high_corr_threshold}):")
        for pair in high_corr_pairs[:20]:  # Show top 20
            logger.info(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.4f}")
        logger.info(f"  Total highly correlated pairs: {len(high_corr_pairs)}")
    else:
        logger.info(f"\nNo feature pairs with |correlation| > {high_corr_threshold}")

    return {
        'target_correlations': dict(sorted_corrs),
        'high_corr_pairs': high_corr_pairs,
        'correlation_matrix': corr_matrix
    }


# =============================================================================
# COMPREHENSIVE FEATURE SELECTION PIPELINE
# =============================================================================

def remove_high_vif_features(
    X: pd.DataFrame,
    vif_threshold: float = VIF_THRESHOLD,
    max_iterations: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with VIF above threshold.

    Args:
        X: Feature DataFrame
        vif_threshold: VIF threshold for removal
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    if not VIF_AVAILABLE:
        logger.warning("VIF not available, skipping VIF-based removal")
        return X, []

    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # Drop columns with any NaN
    X_clean = X_clean.dropna(axis=1, how='any')

    # Also drop columns with zero variance
    zero_var_cols = X_clean.columns[X_clean.var() == 0].tolist()
    X_clean = X_clean.drop(columns=zero_var_cols)

    removed_features = zero_var_cols.copy()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        if len(X_clean.columns) <= 10:
            break

        # Add constant for VIF
        X_with_const = X_clean.copy()
        X_with_const['const'] = 1.0

        # Calculate VIF for all features
        vif_values = {}
        for col in X_clean.columns:
            try:
                vif = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(col))
                vif_values[col] = vif
            except:
                vif_values[col] = np.inf

        # Find maximum VIF
        max_vif_col = max(vif_values, key=vif_values.get)
        max_vif = vif_values[max_vif_col]

        if max_vif <= vif_threshold:
            break

        # Remove feature with highest VIF
        X_clean = X_clean.drop(columns=[max_vif_col])
        removed_features.append(max_vif_col)

        # Log each removal to show iterative nature
        logger.info(f"  VIF iter {iteration}: removed '{max_vif_col}' (VIF={max_vif:.1f}), {len(X_clean.columns)} features remaining")

    logger.info(f"VIF removal complete: {len(removed_features)} features removed, {len(X_clean.columns)} remaining")
    return X_clean, removed_features


def remove_correlated_features(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = CORR_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features, keeping the one with higher target correlation.

    Args:
        X: Feature DataFrame
        y: Target series
        corr_threshold: Correlation threshold above which to remove

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')

    # Calculate target correlations
    target_corrs = {}
    for col in X_clean.columns:
        valid_mask = X_clean[col].notna() & y.notna()
        if valid_mask.sum() > 10:
            target_corrs[col] = abs(X_clean.loc[valid_mask, col].corr(y[valid_mask]))
        else:
            target_corrs[col] = 0

    # Calculate feature correlation matrix
    corr_matrix = X_clean.corr().abs()

    # Find pairs above threshold
    removed_features = []
    cols_to_keep = set(X_clean.columns)

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]

            if col_i not in cols_to_keep or col_j not in cols_to_keep:
                continue

            if corr_matrix.iloc[i, j] > corr_threshold:
                # Remove the one with lower target correlation
                if target_corrs.get(col_i, 0) >= target_corrs.get(col_j, 0):
                    cols_to_keep.discard(col_j)
                    removed_features.append(col_j)
                else:
                    cols_to_keep.discard(col_i)
                    removed_features.append(col_i)

    X_reduced = X_clean[list(cols_to_keep)]
    logger.info(f"Correlation removal: {len(removed_features)} features removed, {len(X_reduced.columns)} remaining")
    return X_reduced, removed_features


def select_by_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    min_corr: float = MIN_TARGET_CORR
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with negligible correlation to target.

    Args:
        X: Feature DataFrame
        y: Target series
        min_corr: Minimum absolute correlation required

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')

    target_corrs = {}
    for col in X_clean.columns:
        valid_mask = X_clean[col].notna() & y.notna()
        if valid_mask.sum() > 10:
            target_corrs[col] = X_clean.loc[valid_mask, col].corr(y[valid_mask])
        else:
            target_corrs[col] = 0

    # Keep features with correlation above threshold
    kept_cols = [col for col, corr in target_corrs.items() if abs(corr) >= min_corr]
    removed_cols = [col for col in X_clean.columns if col not in kept_cols]

    X_reduced = X_clean[kept_cols]
    logger.info(f"Target correlation filter: {len(removed_cols)} features removed, {len(X_reduced.columns)} remaining")
    return X_reduced, removed_cols


def compute_feature_importance_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100
) -> pd.DataFrame:
    """
    Compute feature importance using a quick LightGBM model.

    Args:
        X: Feature DataFrame
        y: Target series
        n_estimators: Number of estimators for quick training

    Returns:
        DataFrame with feature importance scores
    """
    if not LIGHTGBM_AVAILABLE:
        return pd.DataFrame()

    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Quick LightGBM training
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_clean, label=y)
    model = lgb.train(params, train_data, num_boost_round=n_estimators)

    # Get importance
    importance = pd.DataFrame({
        'feature': X_clean.columns,
        'importance_gain': model.feature_importance(importance_type='gain'),
        'importance_split': model.feature_importance(importance_type='split')
    })
    importance = importance.sort_values('importance_gain', ascending=False).reset_index(drop=True)

    return importance


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Compute mutual information between features and target.

    Args:
        X: Feature DataFrame
        y: Target series
        n_neighbors: Number of neighbors for MI estimation

    Returns:
        DataFrame with MI scores
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Standardize for better MI estimation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Compute MI
    mi_scores = mutual_info_regression(X_scaled, y, n_neighbors=n_neighbors, random_state=42)

    mi_df = pd.DataFrame({
        'feature': X_clean.columns,
        'mutual_info': mi_scores
    })
    mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

    return mi_df


def select_features_comprehensive(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = MAX_FEATURES,
    output_dir: Optional[Path] = None,
    target_type: str = 'nsa'
) -> Tuple[List[str], Dict]:
    """
    Comprehensive feature selection pipeline.

    Steps:
    1. Remove zero-variance and infinite features
    2. Remove features with negligible target correlation
    3. Remove highly correlated feature pairs (keep higher target correlation)
    4. Compute multiple importance metrics (LightGBM, MI, target correlation)
    5. Aggregate rankings and select top features
    6. (Optional) VIF check on final set

    Args:
        X: Full feature DataFrame
        y: Target series
        max_features: Maximum features to select
        output_dir: Directory to save selection results
        target_type: 'nsa' or 'sa' for labeling outputs

    Returns:
        Tuple of (selected feature list, selection metadata dict)
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Starting with {len(X.columns)} features, target: {max_features}")

    # Identify protected binary flags present in dataset
    protected_features_present = [f for f in PROTECTED_BINARY_FLAGS if f in X.columns]

    if protected_features_present:
        logger.info(f"Protecting {len(protected_features_present)} binary regime flags from removal:")
        for f in protected_features_present:
            logger.info(f"  - {f}")

    # Initialize metadata
    metadata = {
        'initial_features': len(X.columns),
        'target_features': max_features,
        'protected_features': protected_features_present,
        'steps': []
    }

    # Step 1: Basic cleaning
    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    # Remove columns with too many NaN (>50%)
    nan_pct = X_work.isna().mean()
    high_nan_cols = nan_pct[nan_pct > 0.5].index.tolist()

    # Don't remove protected features
    high_nan_cols = [c for c in high_nan_cols if c not in protected_features_present]

    X_work = X_work.drop(columns=high_nan_cols)
    logger.info(f"Step 1: Removed {len(high_nan_cols)} high-NaN columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'high_nan_removal', 'removed': len(high_nan_cols), 'remaining': len(X_work.columns)})

    # Fill remaining NaN with column mean
    X_work = X_work.fillna(X_work.mean())
    X_work = X_work.fillna(0)

    # Remove zero variance
    zero_var = X_work.columns[X_work.var() == 0].tolist()

    # Don't remove protected features
    zero_var = [c for c in zero_var if c not in protected_features_present]

    X_work = X_work.drop(columns=zero_var)
    logger.info(f"Step 2: Removed {len(zero_var)} zero-variance columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'zero_variance_removal', 'removed': len(zero_var), 'remaining': len(X_work.columns)})

    # Step 2: Target correlation filter
    logger.info("\nStep 3: Target Correlation Filter")
    X_work, removed = select_by_target_correlation(X_work, y, min_corr=MIN_TARGET_CORR)

    # Add back any protected features that were removed
    removed_protected = [f for f in protected_features_present if f not in X_work.columns and f in X.columns]
    if removed_protected:
        logger.info(f"Re-adding {len(removed_protected)} protected features removed by correlation filter")
        for f in removed_protected:
            X_work[f] = X[f]

    metadata['steps'].append({'step': 'target_correlation_filter', 'removed': len(removed), 'remaining': len(X_work.columns)})

    # Step 3: High correlation removal
    logger.info("\nStep 4: High Correlation Removal")
    X_work, removed = remove_correlated_features(X_work, y, corr_threshold=CORR_THRESHOLD)

    # Add back protected features
    removed_protected = [f for f in protected_features_present if f not in X_work.columns and f in X.columns]
    if removed_protected:
        logger.info(f"Re-adding {len(removed_protected)} protected features removed by correlation filter")
        for f in removed_protected:
            X_work[f] = X[f]

    metadata['steps'].append({'step': 'high_correlation_removal', 'removed': len(removed), 'remaining': len(X_work.columns)})

    # Step 4: Compute importance metrics
    logger.info("\nStep 5: Computing Feature Importance Metrics")

    # LightGBM importance
    logger.info("  Computing LightGBM importance...")
    lgbm_importance = compute_feature_importance_lgbm(X_work, y)

    # Mutual Information
    logger.info("  Computing Mutual Information...")
    mi_scores = compute_mutual_information(X_work, y)

    # Target correlation (absolute)
    target_corrs = {}
    for col in X_work.columns:
        valid_mask = X_work[col].notna() & y.notna()
        if valid_mask.sum() > 10:
            target_corrs[col] = abs(X_work.loc[valid_mask, col].corr(y[valid_mask]))
        else:
            target_corrs[col] = 0

    target_corr_df = pd.DataFrame([
        {'feature': k, 'target_corr': v} for k, v in target_corrs.items()
    ]).sort_values('target_corr', ascending=False).reset_index(drop=True)

    # Step 5: Aggregate rankings
    logger.info("\nStep 6: Aggregating Feature Rankings")

    # Create ranking DataFrame
    all_features = list(X_work.columns)
    ranking_df = pd.DataFrame({'feature': all_features})

    # Add LightGBM rank
    lgbm_importance['lgbm_rank'] = range(1, len(lgbm_importance) + 1)
    ranking_df = ranking_df.merge(lgbm_importance[['feature', 'importance_gain', 'lgbm_rank']], on='feature', how='left')

    # Add MI rank
    mi_scores['mi_rank'] = range(1, len(mi_scores) + 1)
    ranking_df = ranking_df.merge(mi_scores[['feature', 'mutual_info', 'mi_rank']], on='feature', how='left')

    # Add correlation rank
    target_corr_df['corr_rank'] = range(1, len(target_corr_df) + 1)
    ranking_df = ranking_df.merge(target_corr_df[['feature', 'target_corr', 'corr_rank']], on='feature', how='left')

    # Compute average rank (lower is better)
    ranking_df['avg_rank'] = ranking_df[['lgbm_rank', 'mi_rank', 'corr_rank']].mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank').reset_index(drop=True)

    # Step 6: Select top features
    selected_features = ranking_df['feature'].head(max_features).tolist()

    # Ensure all protected features are included (even if outside top-N)
    for f in protected_features_present:
        if f not in selected_features and f in X_work.columns:
            selected_features.append(f)
            logger.info(f"Force-included protected feature: {f}")

    logger.info(f"\nSelected {len(selected_features)} features based on aggregated ranking")

    # Step 7: Optional VIF check on final set
    if VIF_AVAILABLE and len(selected_features) > 10:
        logger.info("\nStep 7: VIF Check on Selected Features")
        X_selected = X_work[selected_features]
        X_selected, vif_removed = remove_high_vif_features(X_selected, vif_threshold=VIF_THRESHOLD, max_iterations=20)

        # Add back protected features if they were removed by VIF
        vif_removed_protected = [f for f in protected_features_present if f in vif_removed]
        if vif_removed_protected:
            logger.info(f"VIF removed {len(vif_removed_protected)} protected features - adding them back")
            for f in vif_removed_protected:
                if f in X_work.columns:
                    X_selected[f] = X_work[f]

        if vif_removed:
            selected_features = list(X_selected.columns)
            logger.info(f"After VIF check: {len(selected_features)} features")
            metadata['steps'].append({'step': 'vif_removal', 'removed': len(vif_removed), 'remaining': len(selected_features)})

    # Finalize
    metadata['final_features'] = len(selected_features)
    metadata['selected_features'] = selected_features

    # Log top 20 selected features
    logger.info("\n" + "=" * 40)
    logger.info("TOP 20 SELECTED FEATURES:")
    logger.info("=" * 40)
    for i, feat in enumerate(selected_features[:20], 1):
        row = ranking_df[ranking_df['feature'] == feat].iloc[0]
        logger.info(f"  {i:2d}. {feat}")
        logger.info(f"      LGBM: {row['importance_gain']:.2f} (rank {row['lgbm_rank']:.0f})")
        logger.info(f"      MI: {row['mutual_info']:.4f} (rank {row['mi_rank']:.0f})")
        logger.info(f"      Corr: {row['target_corr']:.4f} (rank {row['corr_rank']:.0f})")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full ranking
        ranking_path = output_dir / f"feature_ranking_{target_type}.csv"
        ranking_df.to_csv(ranking_path, index=False)
        logger.info(f"\nSaved feature ranking to {ranking_path}")

        # Save selected features
        selected_path = output_dir / f"selected_features_{target_type}.csv"
        selected_df = ranking_df[ranking_df['feature'].isin(selected_features)]
        selected_df.to_csv(selected_path, index=False)
        logger.info(f"Saved selected features to {selected_path}")

        # Save metadata
        metadata_path = output_dir / f"feature_selection_metadata_{target_type}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved selection metadata to {metadata_path}")

    return selected_features, metadata


def run_feature_diagnostics(target_type: str = 'nsa') -> Dict:
    """
    Run comprehensive feature diagnostics including VIF and correlation analysis.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        Dictionary with diagnostic results
    """
    logger.info("=" * 60)
    logger.info(f"Running Feature Diagnostics for {target_type.upper()} Model")
    logger.info("=" * 60)

    # Load target data
    target_df = load_target_data(target_type=target_type)

    # Build training dataset
    X, y = build_training_dataset(target_df, target_type=target_type)

    if X.empty:
        logger.error("Could not build training dataset for diagnostics")
        return {}

    logger.info(f"\nDataset: {len(X)} samples, {len(X.columns)} features")

    # VIF Analysis
    logger.info("\n" + "=" * 40)
    logger.info("VIF (Multicollinearity) Analysis")
    logger.info("=" * 40)
    vif_df = compute_vif(X)

    # Correlation Analysis
    logger.info("\n" + "=" * 40)
    logger.info("Correlation Analysis")
    logger.info("=" * 40)
    corr_results = compute_correlation_analysis(X, y)

    # Save results
    results_dir = MODEL_SAVE_DIR / target_type / "diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not vif_df.empty:
        vif_path = results_dir / f"vif_analysis_{target_type}.csv"
        vif_df.to_csv(vif_path, index=False)
        logger.info(f"\nSaved VIF analysis to {vif_path}")

    # Save target correlations
    corr_df = pd.DataFrame([
        {'feature': k, 'target_correlation': v}
        for k, v in corr_results['target_correlations'].items()
    ])
    corr_path = results_dir / f"target_correlations_{target_type}.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved target correlations to {corr_path}")

    # Save high correlation pairs
    if corr_results['high_corr_pairs']:
        pairs_df = pd.DataFrame(corr_results['high_corr_pairs'])
        pairs_path = results_dir / f"high_corr_pairs_{target_type}.csv"
        pairs_df.to_csv(pairs_path, index=False)
        logger.info(f"Saved high correlation pairs to {pairs_path}")

    return {
        'vif': vif_df,
        'correlations': corr_results,
        'num_features': len(X.columns),
        'num_samples': len(X)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LightGBM NFP Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Predict for a specific month (YYYY-MM)')
    parser.add_argument('--latest', action='store_true', help='Predict for latest available month')
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help='Target type: nsa (non-seasonally adjusted) or sa (seasonally adjusted)')
    parser.add_argument('--train-both', action='store_true', help='Train both NSA and SA models')
    parser.add_argument('--diagnostics', action='store_true', help='Run feature diagnostics (VIF, correlations)')
    parser.add_argument('--huber-loss', action='store_true',
                        help='Use Huber loss instead of RMSE (robust to outliers)')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Huber delta parameter (default: 1.0)')

    args = parser.parse_args()

    if args.diagnostics:
        # Run feature diagnostics (VIF and correlation analysis)
        run_feature_diagnostics(target_type=args.target)
    elif args.train_both:
        logger.info("Training both NSA and SA models...")
        train_and_evaluate(target_type='nsa', use_huber_loss=args.huber_loss, huber_delta=args.huber_delta)
        train_and_evaluate(target_type='sa', use_huber_loss=args.huber_loss, huber_delta=args.huber_delta)
    elif args.train:
        train_and_evaluate(target_type=args.target, use_huber_loss=args.huber_loss, huber_delta=args.huber_delta)
    elif args.predict:
        target_month = pd.Timestamp(args.predict + '-01')
        result = predict_nfp_mom(target_month, target_type=args.target)
        print(f"\n{'='*60}")
        print(f"NFP {args.target.upper()} MoM Change Prediction for {target_month.strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")
        print(f"Std: {result['std']:,.0f}")
        print(f"Features Used: {result['features_used']}")
    elif args.latest:
        result = get_latest_prediction(target_type=args.target)
        print(f"\n{'='*60}")
        print(f"NFP {args.target.upper()} MoM Change Prediction for {result['target_month'].strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")
    else:
        # Default: train and evaluate
        train_and_evaluate()