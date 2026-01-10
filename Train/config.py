"""
LightGBM NFP Model Configuration

Centralized configuration constants for the NFP prediction model.
Extracted from train_lightgbm_nfp.py for maintainability.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR


# =============================================================================
# FEATURE SELECTION THRESHOLDS
# =============================================================================

MAX_FEATURES = 80  # Maximum number of features to use in final model
VIF_THRESHOLD = 10.0  # Remove features with VIF above this
CORR_THRESHOLD = 0.95  # Remove one of a pair with correlation above this
MIN_TARGET_CORR = 0.05  # Minimum absolute correlation with target


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MASTER_SNAPSHOTS_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"
FRED_SNAPSHOTS_DIR = DATA_PATH / "fred_data" / "decades"
FRED_PREPARED_DIR = DATA_PATH / "fred_data_prepared" / "decades"  # Preprocessed data
TARGET_PATH_NSA = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
TARGET_PATH_SA = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"
MODEL_SAVE_DIR = OUTPUT_DIR / "models" / "lightgbm_nfp"

# Use prepared data (SymLog transformed, scaled) or raw data
USE_PREPARED_FRED_DATA = True


# =============================================================================
# LINEAR BASELINE PREDICTORS
# =============================================================================

# These have strong linear relationships with NFP and can extrapolate beyond
# training ranges (unlike trees which are bounded by training leaves).

LINEAR_BASELINE_PREDICTORS = [
    # Weekly Claims - PRIMARY EXTRAPOLATION SIGNAL
    'CCSA_monthly_avg_latest',       # Current month average
    'CCSA_max_spike',                 # Maximum spike
    'CCSA_weeks_high',                # Persistence metric (weeks above 95th percentile)
    'CCSA_monthly_avg_lag1',          # Previous month
    'CCSA_monthly_avg_mom_change',    # Month-over-month change

    # NEW: Average Weekly Hours - HOURS LEAD JOBS
    # Employers cut hours before cutting headcount - critical recession signal
    'AWH_All_Private_latest',         # Current hours level
    'AWH_All_Private_mom',            # MoM change (negative = layoffs coming)
    'AWH_Manufacturing_latest',       # Manufacturing hours (more volatile)
    'AWH_Manufacturing_mom',          # Manufacturing MoM change

    # High-velocity shock signals for extreme events
    'Financial_Stress_zscore_3m_max',  # Peak stress shock (3-month Z-score)
    'Oil_Prices_zscore_3m_min',        # Peak price collapse (3-month Z-score)
    'SP500_zscore_3m_min',             # Peak market crash (3-month Z-score)
    'Weekly_Econ_Index_latest',        # Real-time GDP signal

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


# =============================================================================
# PROTECTED BINARY FLAGS (Never removed by feature selection)
# =============================================================================

# These capture critical extreme events (COVID-like scenarios)
PROTECTED_BINARY_FLAGS = [
    'VIX_panic_regime',       # VIX >50 (extreme panic)
    'VIX_high_regime',        # VIX >40 (high fear)
    'SP500_crash_month',      # Monthly return <-10%
    'SP500_bear_market',      # Drawdown <-20% from 52w high
    'SP500_circuit_breaker',  # Any day down >5%
]


# =============================================================================
# KEY EMPLOYMENT SERIES
# =============================================================================

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


# =============================================================================
# LAG CONFIGURATION
# =============================================================================

# Short-term lags: 1, 2, 3 months (business cycle dynamics)
# Medium-term lags: 6 months (semi-annual patterns)
# Long-term lags: 12, 18, 24 months (structural/secular trends)
SHORT_TERM_LAGS = [1, 2, 3]
MEDIUM_TERM_LAGS = [6]
LONG_TERM_LAGS = [12, 18, 24]
ALL_LAGS = SHORT_TERM_LAGS + MEDIUM_TERM_LAGS + LONG_TERM_LAGS


# =============================================================================
# LIGHTGBM HYPERPARAMETERS
# =============================================================================

DEFAULT_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,  # Tuned for extreme events
    'min_data_in_leaf': 1,  # Allow single-sample leaves for rare events
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# Huber loss parameters
HUBER_DELTA = 1.0  # Transition point between L1 and L2 loss


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Time-series cross-validation
N_CV_SPLITS = 5

# Early stopping
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# Training weights for extreme events
PANIC_REGIME_WEIGHT = 5.0  # 5x weight for VIX_panic_regime or SP500_crash_month

# Feature selection interval (months)
FEATURE_SELECTION_INTERVAL = 6


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

CONFIDENCE_LEVELS = [0.50, 0.80, 0.95]
