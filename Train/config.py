"""
LightGBM NFP Model Configuration

Centralized configuration constants and path definitions for the NFP prediction model.
This file acts as the single source of truth for all hyperparameter bounds, valid
target configurations, and file paths used during the data loading and training phases.

TARGET TYPES:
    - target_type: 'nsa' (non-seasonally adjusted) or 'sa' (seasonally adjusted)
    - release_type: 'first' (initial release)
    - target_source: 'first_release' or 'revised'

This creates 4 model variants:
    - nsa_first: NSA target trained on first-release features
    - nsa_first_revised: NSA target trained on revised features
    - sa_first: SA target trained on first-release features
    - sa_first_revised: SA target trained on revised features

The first-release models are the operationally deployable variants (real-time actionable).
The revised models serve as an upper-bound benchmark for predictability using hindsight-
corrected data, useful for diagnosing whether model error comes from data noise vs.
structural model weakness.
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
"""Tuple of valid target types. 'nsa' for Non-Seasonally Adjusted, 'sa' for Seasonally Adjusted."""

VALID_RELEASE_TYPES = ('first', 'last')
"""Tuple of valid release types. 'first' is the initial release, 'last' is the final revised release."""

VALID_TARGET_SOURCES = ('first_release', 'revised')
"""Tuple of valid target sources. 'first_release' uses the initially reported number. 'revised' uses the revised number reported in the subsequent month."""

# FRED series names for revised target construction (raw snapshot levels)
REVISED_TARGET_SERIES = {'nsa': 'total_nsa', 'sa': 'total'}
"""Dictionary mapping target types to their corresponding raw FRED series names used for target construction."""

# Target combinations — all 4 variants trained and operationally deployable.
# first_release models: predict the initial BLS print (available on NFP release day).
# revised models: predict once-revised MoM (available ~1 month after first_release,
#   i.e., after the M+1 NFP release). MUST check operational_available_date before use.
#   predict_nfp_mom() enforces this via RuntimeError if called too early.
ALL_TARGET_CONFIGS = [
    ('nsa', 'first', 'first_release'),
    ('nsa', 'first', 'revised'),
    ('sa',  'first', 'first_release'),
    ('sa',  'first', 'revised'),
]
"""All 4 model variants (NSA/SA × first_release/revised). All are operationally deployable;
revised models require operational_available_date to have passed (enforced at inference)."""


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

MASTER_SNAPSHOTS_BASE = DATA_PATH / "master_snapshots"
"""Base directory for feature-selected master snapshots ({nsa,sa}/{first_release,revised}/decades/)."""

FRED_SNAPSHOTS_DIR = DATA_PATH / "fred_data" / "decades"
"""Directory containing raw FRED data snapshots (still needed for build_revised_target)."""


def get_master_snapshots_dir(target_type: str, target_source: str = 'first_release') -> Path:
    """
    Get the decades directory for master snapshots of a specific target configuration.

    Args:
        target_type: 'nsa' or 'sa'
        target_source: 'first_release' or 'revised'

    Returns:
        Path to the decades directory containing master snapshots.
    """
    return MASTER_SNAPSHOTS_BASE / target_type / target_source / "decades"

NFP_TARGET_DIR = DATA_PATH / "NFP_target"
"""Directory containing the target parquet files for both first and revised NFP prints."""

MODEL_SAVE_DIR = OUTPUT_DIR / "models" / "lightgbm_nfp"
"""Directory where the final trained LightGBM models are saved."""



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


def get_model_id(target_type: str, release_type: str = 'first',
                 target_source: str = 'first_release') -> str:
    """
    Get a unique model identifier string for the target configuration.

    Args:
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        target_source: 'first_release' or 'revised'

    Returns:
        Model identifier string (e.g., 'nsa_first', 'nsa_first_revised')
    """
    base = f"{target_type.lower()}_{release_type.lower()}"
    return f"{base}_revised" if target_source == 'revised' else base


def parse_model_id(model_id: str) -> Tuple[str, str, str]:
    """
    Parse a model identifier string into target_type, release_type, and target_source.

    Args:
        model_id: Model identifier (e.g., 'nsa_first' or 'nsa_first_revised')

    Returns:
        Tuple of (target_type, release_type, target_source)

    Raises:
        ValueError: If invalid model_id format
    """
    parts = model_id.lower().split('_')
    if len(parts) == 2:
        target_type, release_type = parts
        target_source = 'first_release'
    elif len(parts) == 3 and parts[2] == 'revised':
        target_type, release_type = parts[0], parts[1]
        target_source = 'revised'
    else:
        raise ValueError(
            f"Invalid model_id format: {model_id}. "
            f"Expected 'target_release' (e.g., 'nsa_first') or "
            f"'target_release_revised' (e.g., 'nsa_first_revised')"
        )

    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type in model_id: {target_type}")
    if release_type not in VALID_RELEASE_TYPES:
        raise ValueError(f"Invalid release_type in model_id: {release_type}")
    if target_source not in VALID_TARGET_SOURCES:
        raise ValueError(f"Invalid target_source in model_id: {target_source}")

    return target_type, release_type, target_source


# Legacy compatibility - point to first release files
TARGET_PATH_NSA = get_target_path('nsa', 'first')
TARGET_PATH_SA = get_target_path('sa', 'first')


# =============================================================================
# SELECTED FEATURES
# =============================================================================

def load_selected_features(target_type: str, target_source: str = 'first_release') -> List[str]:
    """
    Load pre-selected feature names from the master snapshots feature selection cache.

    The feature selection engine (Data_ETA_Pipeline/create_master_snapshots.py) saves
    a JSON cache of surviving features per {target_type, target_source} combination.
    This function reads that cache directly.

    Args:
        target_type: 'nsa' or 'sa'
        target_source: 'first_release' or 'revised'

    Returns:
        List of feature names selected for this target configuration
    """
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: {target_type}. Must be one of {VALID_TARGET_TYPES}")

    cache_path = MASTER_SNAPSHOTS_BASE / f"selected_features_{target_type}_{target_source}.json"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature selection cache not found: {cache_path}. "
            f"Run Data_ETA_Pipeline/create_master_snapshots.py first."
        )

    with open(cache_path, 'r') as f:
        data = json.load(f)

    return data.get("features", [])


# =============================================================================
# LIGHTGBM HYPERPARAMETERS
# =============================================================================

DEFAULT_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_data_in_leaf': 5,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# Huber loss parameters - ENABLED BY DEFAULT for outlier robustness
USE_HUBER_LOSS_DEFAULT = False  # Disabled: COVID outliers already handled by winsorize_covid_period()
HUBER_DELTA = 100.0  # Transition point between L1 and L2 loss; must match target scale (NFP MoM ~100-300)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Time-series cross-validation
N_CV_SPLITS = 5

# Early stopping
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# Exponential Decay Sample Weighting (Optuna bounds)
HALF_LIFE_MIN_MONTHS = 12
HALF_LIFE_MAX_MONTHS = 120

# Optuna hyperparameter tuning
N_OPTUNA_TRIALS = 25        # Number of Optuna trials per tuning run
OPTUNA_TIMEOUT = 300        # Max seconds per tuning run
TUNE_EVERY_N_MONTHS = 12   # Re-tune hyperparameters every N months in backtest


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

CONFIDENCE_LEVELS = [0.50, 0.80, 0.95]


# =============================================================================
# UNION-FIRST CANDIDATE POOL + SHORT-PASS SELECTION
# =============================================================================

# =============================================================================
# FEATURE SELECTION STAGE TIERS (Data_ETA_Pipeline)
# =============================================================================
# Controlled at ETL time via NFP_FS_STAGES env var (comma-separated ints).
# These constants document the available tier definitions.
#
# Stage 0 includes a vectorized Spearman pre-screen (BH-FDR α=0.30) that
# automatically activates when a source has >5,000 features (e.g. FRED
# Employment at 17k). This reduces Stage 1 input by ~78%, cutting its
# runtime from ~10+ min to ~2 min with zero signal loss.
#
# Stages 5 (Interaction Rescue) and 6 (SFS) are omitted from the default
# because their signal is redundant with the train-time short-pass, which
# re-derives a top-60 feature subset every backtest step via LightGBM gain.

FS_STAGES_DEFAULT = (0, 1, 2, 3, 4)
"""Default pipeline: Pre-screen → Dual Filter → Boruta → Vintage → Cluster.
Drops only Stages 5 (Interaction Rescue) and 6 (SFS), whose signal is
redundant with the train-time short-pass. ~5 min/source."""

FS_STAGES_FULL = (0, 1, 2, 3, 4, 5, 6)
"""All 7 stages. Use for benchmarking or validation only. ~10 min/source."""

FS_STAGES_FAST = (0, 1, 4)
"""Minimal: Pre-screen → Dual Filter → Cluster Redundancy. ~3 min/source.
Skips Boruta and Vintage — suitable for rapid iteration and A/B testing."""

FS_STAGES_FAST_VINTAGE = (0, 1, 3, 4)
"""Fast + Vintage Stability for temporal robustness. ~3 min/source."""

FS_STAGES_FAST_BORUTA = (0, 1, 2, 4)
"""Fast + Boruta for label-permutation robustness. ~4 min/source."""


# =============================================================================
# UNION-FIRST CANDIDATE POOL + SHORT-PASS SELECTION
# =============================================================================

USE_UNION_POOL = True               # Master toggle for union pool + short-pass
UNION_POOL_MAX = 200                # Max features in the global candidate pool
SHORTPASS_TOPK = 60                 # Features selected per backtest step (40-80 range)
SHORTPASS_METHOD = 'lgbm_gain'      # 'lgbm_gain' or 'weighted_corr'
SHORTPASS_HALF_LIFE = None          # None = reuse backtest step half_life

# Branch-target derived features (nfp_{target_type}_*) are selected separately
# and then merged on top of snapshot features.
USE_BRANCH_TARGET_FS = True         # Master toggle for branch-target feature selection
BRANCH_TARGET_FS_TOPK = 8           # Keep up to this many branch-target derived features
BRANCH_TARGET_FS_METHOD = 'weighted_corr'  # 'lgbm_gain' or 'weighted_corr'
BRANCH_TARGET_FS_CORR_THRESHOLD = 0.90     # Redundancy pruning threshold
BRANCH_TARGET_FS_MIN_OVERLAP = 24          # Minimum overlap for corr pruning


# =============================================================================
# BASELINE KEEP-RULE
# =============================================================================

ENABLE_BASELINE_TRACKING = True     # Compute baselines alongside model
BASELINE_ROLLING_WINDOW = 6         # Months for rolling_mean baseline
KEEP_RULE_ENABLED = True            # Enforce hard keep-rule gating
KEEP_RULE_WINDOW_M = 12             # Trailing OOS months to evaluate
KEEP_RULE_TOLERANCE = 0.0           # Max allowed MAE degradation vs best baseline
KEEP_RULE_ACTION = 'skip_save'      # 'fail' | 'fallback_to_baseline' | 'skip_save'
