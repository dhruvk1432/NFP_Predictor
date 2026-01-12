"""
LightGBM Model Training for NFP Prediction

Core model training, prediction, and interval calculation functions.
Extracted from train_lightgbm_nfp.py for maintainability.

MULTI-TARGET SUPPORT:
Model save/load functions support 4 target configurations:
- nsa_first, nsa_last, sa_first, sa_last
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.config import (
    DEFAULT_LGBM_PARAMS,
    HUBER_DELTA,
    N_CV_SPLITS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    PANIC_REGIME_WEIGHT,
    CONFIDENCE_LEVELS,
    MODEL_SAVE_DIR,
    PROTECTED_BINARY_FLAGS,
    get_model_id,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LinearRegression
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM/sklearn not available")
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

def get_lgbm_params(
    use_huber_loss: bool = False,
    huber_delta: float = HUBER_DELTA
) -> Dict:
    """
    Get LightGBM parameters with optional Huber loss.

    Args:
        use_huber_loss: If True, use Huber objective (robust to outliers)
        huber_delta: Huber delta parameter (transition point between L1 and L2)

    Returns:
        Dictionary of LightGBM parameters
    """
    params = DEFAULT_LGBM_PARAMS.copy()
    
    if use_huber_loss:
        params['objective'] = 'huber'
        params['alpha'] = huber_delta  # Huber delta parameter
    
    return params


# =============================================================================
# SAMPLE WEIGHTING
# =============================================================================

def calculate_sample_weights(X: pd.DataFrame) -> np.ndarray:
    """
    Calculate training sample weights based on regime.

    Assigns higher weights to extreme event periods (COVID-like scenarios).

    Args:
        X: Feature DataFrame with regime flag columns

    Returns:
        Array of sample weights
    """
    weights = np.ones(len(X))

    # Check for panic regime flags - try multiple column name patterns
    # Pattern 1: With _latest suffix (from pivot_snapshot_to_wide)
    # Pattern 2: Without suffix (direct column names)
    vix_panic_cols = ['VIX_panic_regime', 'VIX_panic_regime_latest']
    sp500_crash_cols = ['SP500_crash_month', 'SP500_crash_month_latest']
    vix_high_cols = ['VIX_high_regime', 'VIX_high_regime_latest']
    sp500_bear_cols = ['SP500_bear_market', 'SP500_bear_market_latest']
    sp500_circuit_cols = ['SP500_circuit_breaker', 'SP500_circuit_breaker_latest']

    panic_mask = np.zeros(len(X), dtype=bool)

    # Check VIX panic regime
    for col in vix_panic_cols:
        if col in X.columns:
            panic_mask |= (X[col] == 1)
            break

    # Check SP500 crash month
    for col in sp500_crash_cols:
        if col in X.columns:
            panic_mask |= (X[col] == 1)
            break

    # Check VIX high regime
    for col in vix_high_cols:
        if col in X.columns:
            panic_mask |= (X[col] == 1)
            break

    # Check SP500 bear market
    for col in sp500_bear_cols:
        if col in X.columns:
            panic_mask |= (X[col] == 1)
            break

    # Check SP500 circuit breaker
    for col in sp500_circuit_cols:
        if col in X.columns:
            panic_mask |= (X[col] == 1)
            break

    weights[panic_mask] = PANIC_REGIME_WEIGHT

    n_panic_samples = panic_mask.sum()
    if n_panic_samples > 0:
        logger.info(f"Applying {PANIC_REGIME_WEIGHT}x weight to {n_panic_samples} panic regime samples ({100*n_panic_samples/len(weights):.1f}%)")
    else:
        logger.warning("No regime indicators found or no panic samples detected - using equal weights")

    return weights


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    use_huber_loss: bool = False,
    huber_delta: float = HUBER_DELTA
) -> Tuple[lgb.Booster, Dict, List[float]]:
    """
    Train LightGBM model with time-series cross-validation.

    Args:
        X: Feature DataFrame (includes 'ds' column for date reference)
        y: Target Series
        n_splits: Number of time series CV splits
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
        use_huber_loss: If True, use Huber objective
        huber_delta: Huber delta parameter

    Returns:
        Tuple of (trained model, feature importance dict, residuals list)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    # Separate date column
    dates = X['ds'] if 'ds' in X.columns else None
    feature_cols = [c for c in X.columns if c != 'ds']
    X_train = X[feature_cols].copy()

    logger.info(f"Training LightGBM on {len(X_train)} samples with {len(feature_cols)} features")

    # Remove any remaining NaN/Inf from features
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    # Drop rows with any NaN in features or target
    valid_mask = ~(X_train.isna().any(axis=1) | y.isna())
    X_clean = X_train[valid_mask].copy()
    y_clean = y[valid_mask].copy()

    if len(X_clean) < 50:
        raise ValueError(f"Not enough valid training samples: {len(X_clean)}")

    logger.info(f"After cleaning: {len(X_clean)} valid samples")

    # Calculate sample weights based on regime indicators
    weights = calculate_sample_weights(X[valid_mask])

    # LightGBM parameters
    params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_residuals = []
    oof_predictions = np.zeros(len(X_clean))

    logger.info(f"Running {n_splits}-fold time series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
        X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_tr, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
        weights_tr = weights[train_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=weights_tr)
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
    train_size = int(len(X_clean) * 0.85)
    X_final_train = X_clean.iloc[:train_size]
    X_final_val = X_clean.iloc[train_size:]
    y_final_train = y_clean.iloc[:train_size]
    y_final_val = y_clean.iloc[train_size:]
    weights_final_train = weights[:train_size]

    train_data = lgb.Dataset(X_final_train, label=y_final_train, weight=weights_final_train)
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


# =============================================================================
# PREDICTION INTERVALS
# =============================================================================

def calculate_prediction_intervals(
    residuals: List[float],
    prediction: float,
    confidence_levels: List[float] = CONFIDENCE_LEVELS
) -> Dict[float, Tuple[float, float]]:
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
    intervals = {}
    
    if not residuals or len(residuals) < 10:
        # Not enough residuals, use placeholder intervals
        for level in confidence_levels:
            half_width = prediction * (1 - level)  # Rough estimate
            intervals[level] = (prediction - abs(half_width), prediction + abs(half_width))
        return intervals
    
    residuals_array = np.array(residuals)
    
    for level in confidence_levels:
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        lower_resid = np.quantile(residuals_array, lower_q)
        upper_resid = np.quantile(residuals_array, upper_q)
        
        intervals[level] = (prediction + lower_resid, prediction + upper_resid)
    
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
        feature_cols: Feature column names used by model

    Returns:
        Dictionary with prediction and interval information including:
        - prediction: Point estimate
        - intervals: Dict with both numeric (0.50) and string ('50%') keys
        - std: Standard deviation of residuals
        - mean_residual_bias: Mean bias from residuals
    """
    # Handle missing feature columns gracefully
    available_cols = [c for c in feature_cols if c in features.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing {len(missing)} feature columns: {list(missing)[:5]}...")

    X = features[available_cols].values.reshape(1, -1)
    prediction = model.predict(X)[0]

    intervals_numeric = calculate_prediction_intervals(residuals, prediction)

    # Calculate residual statistics
    residuals_array = np.array(residuals) if residuals else np.array([0])
    std = float(np.std(residuals_array)) if len(residuals_array) > 1 else 50.0
    mean_residual_bias = float(np.mean(residuals_array)) if len(residuals_array) > 0 else 0.0

    # Create intervals dict with both numeric and string keys for compatibility
    intervals = {}
    for level, bounds in intervals_numeric.items():
        intervals[level] = bounds  # Numeric key (0.50, 0.80, 0.95)
        intervals[f'{int(level * 100)}%'] = bounds  # String key ('50%', '80%', '95%')

    return {
        'prediction': prediction,
        'intervals': intervals,
        'lower_50': intervals.get(0.50, (np.nan, np.nan))[0],
        'upper_50': intervals.get(0.50, (np.nan, np.nan))[1],
        'lower_80': intervals.get(0.80, (np.nan, np.nan))[0],
        'upper_80': intervals.get(0.80, (np.nan, np.nan))[1],
        'lower_95': intervals.get(0.95, (np.nan, np.nan))[0],
        'upper_95': intervals.get(0.95, (np.nan, np.nan))[1],
        'std': std,
        'mean_residual_bias': mean_residual_bias,
    }


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    model: lgb.Booster,
    feature_cols: List[str],
    residuals: List[float],
    importance: Dict,
    save_dir: Path = MODEL_SAVE_DIR,
    target_type: str = 'nsa',
    release_type: str = 'first',
    linear_model: Optional[Any] = None,
    linear_cols: Optional[List[str]] = None
) -> None:
    """
    Save model and associated metadata.

    Args:
        model: Trained LightGBM model
        feature_cols: List of feature column names used
        residuals: List of residuals for prediction intervals
        importance: Feature importance dictionary
        save_dir: Base directory for model storage
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        linear_model: Optional linear baseline model
        linear_cols: Optional columns used by linear model
    """
    model_id = get_model_id(target_type, release_type)

    # Create subdirectory for this model variant
    model_dir = save_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM model
    model.save_model(str(model_dir / f"lightgbm_{model_id}_model.txt"))

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'residuals': residuals,
        'importance': importance,
        'target_type': target_type,
        'release_type': release_type,
        'model_id': model_id,
        'linear_model': linear_model,
        'linear_cols': linear_cols,
    }

    with open(model_dir / f"lightgbm_{model_id}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"Model {model_id.upper()} saved to {model_dir}")


def load_model(
    save_dir: Path = MODEL_SAVE_DIR,
    target_type: str = 'nsa',
    release_type: str = 'first'
) -> Tuple[lgb.Booster, Dict]:
    """
    Load model and associated metadata.

    Args:
        save_dir: Base directory for model storage
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'

    Returns:
        Tuple of (model, metadata)
    """
    model_id = get_model_id(target_type, release_type)
    model_dir = save_dir / model_id

    model_path = model_dir / f"lightgbm_{model_id}_model.txt"
    metadata_path = model_dir / f"lightgbm_{model_id}_metadata.pkl"

    # Fallback to legacy paths for backward compatibility
    if not model_path.exists():
        # Try legacy path format
        legacy_model_path = save_dir / f"lightgbm_{target_type}_model.txt"
        legacy_metadata_path = save_dir / f"lightgbm_{target_type}_metadata.pkl"

        if legacy_model_path.exists():
            logger.warning(f"Using legacy model path: {legacy_model_path}")
            model_path = legacy_model_path
            metadata_path = legacy_metadata_path
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    model = lgb.Booster(model_file=str(model_path))

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Model {model_id.upper()} loaded from {model_dir}")

    return model, metadata


def list_available_models(save_dir: Path = MODEL_SAVE_DIR) -> List[str]:
    """
    List all available trained models.

    Args:
        save_dir: Base directory for model storage

    Returns:
        List of model_ids that have trained models
    """
    available = []

    for target_type in VALID_TARGET_TYPES:
        for release_type in VALID_RELEASE_TYPES:
            model_id = get_model_id(target_type, release_type)
            model_dir = save_dir / model_id

            model_path = model_dir / f"lightgbm_{model_id}_model.txt"
            if model_path.exists():
                available.append(model_id)

    return available


# =============================================================================
# LINEAR BASELINE
# =============================================================================

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
        predictor_cols: Columns to use for linear model

    Returns:
        Tuple of (trained LinearRegression model, actual columns used)
    """
    # Find available columns
    available_cols = [c for c in predictor_cols if c in X.columns]
    
    if len(available_cols) < 3:
        logger.warning(f"Only {len(available_cols)} linear baseline predictors available")
        return None, []
    
    # Get clean data
    X_linear = X[available_cols].copy()
    valid_mask = ~(X_linear.isna().any(axis=1) | y.isna())
    
    X_fit = X_linear[valid_mask]
    y_fit = y[valid_mask]
    
    if len(X_fit) < 20:
        logger.warning("Not enough samples for linear baseline")
        return None, []
    
    # Fit model
    linear_model = LinearRegression()
    linear_model.fit(X_fit, y_fit)
    
    logger.info(f"Trained linear baseline with {len(available_cols)} predictors")
    
    return linear_model, available_cols


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
        return pd.Series(np.nan, index=X.index)
    
    X_linear = X[predictor_cols].copy()
    
    # Handle missing values
    X_linear = X_linear.fillna(X_linear.median())
    
    predictions = linear_model.predict(X_linear)
    
    return pd.Series(predictions, index=X.index, name='linear_baseline_pred')
