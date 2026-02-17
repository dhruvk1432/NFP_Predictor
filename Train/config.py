"""
LightGBM NFP Model Configuration

Centralized configuration constants for the NFP prediction model.
Extracted from train_lightgbm_nfp.py for maintainability.

TARGET TYPES:
    - target_type: 'nsa' (non-seasonally adjusted) or 'sa' (seasonally adjusted)
    - release_type: 'first' (initial release) only - last release is disabled

This creates 2 model variants (first release only):
    - nsa_first: NSA with first release data
    - sa_first: SA with first release data

NOTE: Last release models (nsa_last, sa_last) are disabled.
"""

from pathlib import Path
from typing import List, Tuple
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, MODEL_TYPE


# =============================================================================
# VALID TARGET CONFIGURATIONS
# =============================================================================

VALID_TARGET_TYPES = ('nsa', 'sa')
VALID_RELEASE_TYPES = ('first', 'last')

# Target combinations - FIRST RELEASE ONLY
# Only training nsa_first and sa_first models.
# Last release models are disabled and commented out.
ALL_TARGET_CONFIGS = [
    ('nsa', 'first'),
    ('sa', 'first'),
    # DISABLED: Last release models - do not uncomment
    # ('nsa', 'last'),  # Disabled - last release not supported
    # ('sa', 'last'),   # Disabled - last release not supported
]


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MASTER_SNAPSHOTS_DIR = DATA_PATH / "Exogenous_data" / "master_snapshots" / "decades"
FRED_SNAPSHOTS_DIR = DATA_PATH / "fred_data" / "decades"
FRED_PREPARED_DIR = DATA_PATH / "fred_data_prepared" / "decades"  # Preprocessed data
NFP_TARGET_DIR = DATA_PATH / "NFP_target"
MODEL_SAVE_DIR = OUTPUT_DIR / "models" / "lightgbm_nfp"

# Use prepared data (SymLog transformed, scaled) or raw data
USE_PREPARED_FRED_DATA = True


def get_target_path(target_type: str, release_type: str = 'first') -> Path:
    """
    Get target file path based on type and release timing.

    Args:
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'

    Returns:
        Path to target parquet file

    Raises:
        ValueError: If invalid target_type or release_type
    """
    target_type = target_type.lower()
    release_type = release_type.lower()

    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: {target_type}. Must be one of {VALID_TARGET_TYPES}")
    if release_type not in VALID_RELEASE_TYPES:
        raise ValueError(f"Invalid release_type: {release_type}. Must be one of {VALID_RELEASE_TYPES}")

    # Use 'total_' prefix for univariate mode, 'y_' for multivariate mode
    prefix = "total" if MODEL_TYPE == "univariate" else "y"
    filename = f"{prefix}_{target_type}_{release_type}_release.parquet"
    return NFP_TARGET_DIR / filename


def get_model_id(target_type: str, release_type: str = 'first') -> str:
    """
    Get a unique model identifier string for the target configuration.

    Args:
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'

    Returns:
        Model identifier string (e.g., 'nsa_first', 'sa_last')
    """
    return f"{target_type.lower()}_{release_type.lower()}"


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """
    Parse a model identifier string into target_type and release_type.

    Args:
        model_id: Model identifier (e.g., 'nsa_first')

    Returns:
        Tuple of (target_type, release_type)

    Raises:
        ValueError: If invalid model_id format
    """
    parts = model_id.lower().split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid model_id format: {model_id}. Expected 'target_release' (e.g., 'nsa_first')")

    target_type, release_type = parts
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type in model_id: {target_type}")
    if release_type not in VALID_RELEASE_TYPES:
        raise ValueError(f"Invalid release_type in model_id: {release_type}")

    return target_type, release_type


# Legacy compatibility - point to first release files
TARGET_PATH_NSA = get_target_path('nsa', 'first')
TARGET_PATH_SA = get_target_path('sa', 'first')


# =============================================================================
# SELECTED FEATURES
# =============================================================================

SELECTED_FEATURES_DIR = Path(__file__).resolve().parent / "selected_features"

ALL_SOURCES = ['fred_employment', 'fred_exog', 'unifier', 'adp', 'noaa', 'prosper']


def load_selected_features(target_type: str) -> List[str]:
    """
    Load pre-selected feature names for a given target type (nsa or sa).

    Loads from all 6 source JSONs, sanitizes names to match
    pivot_snapshot_to_wide() output, and returns a deduplicated list.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        List of sanitized feature names selected for this target type
    """
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: {target_type}. Must be one of {VALID_TARGET_TYPES}")

    # Lazy import to avoid circular dependency (data_loader imports config)
    from Train.data_loader import sanitize_feature_name

    all_features = []

    for source in ALL_SOURCES:
        json_path = SELECTED_FEATURES_DIR / f"{source}_{target_type}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Selected features file not found: {json_path}")

        with open(json_path, 'r') as f:
            raw_features = json.load(f)

        sanitized = [sanitize_feature_name(name) for name in raw_features]
        all_features.extend(sanitized)

    # Deduplicate preserving order
    seen = set()
    unique_features = []
    for f in all_features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)

    return unique_features


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

# Huber loss parameters - ENABLED BY DEFAULT for outlier robustness
USE_HUBER_LOSS_DEFAULT = True  # Huber loss is more robust to COVID-like outliers
HUBER_DELTA = 1.0  # Transition point between L1 and L2 loss (lower = more robust)


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

# Optuna hyperparameter tuning
N_OPTUNA_TRIALS = 50        # Number of Optuna trials per tuning run
OPTUNA_TIMEOUT = 300        # Max seconds per tuning run
TUNE_EVERY_N_MONTHS = 12   # Re-tune hyperparameters every N months in backtest


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

CONFIDENCE_LEVELS = [0.50, 0.80, 0.95]
