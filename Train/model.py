"""
LightGBM Model Training for NFP Prediction

This module encapsulates the core predictive infrastructure for the Non-Farm Payrolls model.
It handles:
1. Model hyperparameter retrieval (with optional Huber Loss for outlier robustness).
2. Dynamic sample weighting (using exponential decay to prioritize recent data regimes).
3. The main LightGBM training loop (including TimeSeriesSplit cross-validation and early stopping).
4. Prediction interval generation (using empirical non-parametric historical residuals).
5. Model persistence (saving and loading models alongside their feature states).

MULTI-TARGET SUPPORT:
Model save/load functions support 4 architectural target configurations depending on the 
chosen macro pipeline: nsa_first, nsa_last, sa_first, sa_last.
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
    USE_HUBER_LOSS_DEFAULT,
    N_CV_SPLITS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    CONFIDENCE_LEVELS,
    MODEL_SAVE_DIR,
    get_model_id,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM/sklearn not available")
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

def get_lgbm_params(
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
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

def calculate_sample_weights(
    X: pd.DataFrame, 
    target_month: pd.Timestamp, 
    half_life_months: float
) -> np.ndarray:
    """
    Calculate sample weights using exponential decay based on recency.
    
    More recent samples relative to the target prediction month are 
    assigned higher weights. The decay rate is determined by the half-life.

    Args:
        X: Feature DataFrame containing a 'ds' column with sample dates
        target_month: The month being predicted 
        half_life_months: The exponential decay half-life in months

    Returns:
        Array of sample weights
    """
    if 'ds' not in X.columns:
        logger.warning("'ds' date column not found in features - using equal weights")
        return np.ones(len(X))

    # Calculate distance in approx months (days / 30.44)
    # Ensure distance is non-negative (training data should be strictly before target)
    distance_days = (target_month - pd.to_datetime(X['ds'])).dt.days
    distance_months = np.maximum(0, distance_days / 30.436875)

    # Calculate exponential weights: w = exp(-ln(2) * distance_months / half_life)
    # The weight approaches 1.0 as distance approaches 0
    decay_rate = np.log(2) / half_life_months
    weights = np.exp(-decay_rate * distance_months)

    # Normalize weights so mean equals 1.0 (maintains LightGBM learning rate scale)
    weights = weights / np.mean(weights)
    
    return weights.values


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    params_override: Optional[Dict] = None,
) -> Tuple[lgb.Booster, Dict, List[float]]:
    """
    Train a LightGBM regression model using robust time-series cross-validation.

    This function performs a rigorously leakage-free training process:
    1. It drops invalid targets gracefully.
    2. It calculates point-in-time sample weights using exponential decay.
    3. It trains `n_splits` intermediate models across chronological folds to 
       calculate Out-Of-Fold (OOF) residuals and cross-validation performance.
    4. Finally, it trains one 'production' model on the entire provided dataset, 
       using a predefined 85/15 chronological split for early stopping to prevent 
       overfitting to the terminal edge of the window.

    Args:
        X (pd.DataFrame): Feature DataFrame (must include 'ds' column for dating).
        y (pd.Series): Target numerical Series.
        n_splits (int): Number of time series CV splits for error calculation.
        num_boost_round (int): Maximum number of boosting rounds for LightGBM.
        early_stopping_rounds (int): Early stopping patience parameter.
        use_huber_loss (bool): If True, use Huber objective function.
        huber_delta (float): Huber delta parameter if Huber loss is enabled.
        params_override (Optional[Dict]): Pre-tuned dictionary of Optuna hyperparameters.
            If None, the system falls back to default settings.

    Returns:
        Tuple[lgb.Booster, Dict, List[float]]: 
            - trained_model: The final LightGBM Booster trained on all available data.
            - importance: Dictionary mapping feature_name -> gain importance score.
            - final_residuals: List of Out-Of-Sample prediction errors from the final 15% holdout.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    # Separate date column
    dates = X['ds'] if 'ds' in X.columns else None
    feature_cols = [c for c in X.columns if c != 'ds']
    X_train = X[feature_cols].copy()

    logger.info(f"Training LightGBM on {len(X_train)} samples with {len(feature_cols)} features")

    # Replace inf with NaN (LightGBM handles NaN natively but not inf)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    # Only drop rows where the target is NaN (LightGBM handles NaN features natively)
    valid_mask = ~y.isna()
    X_clean = X_train[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    X_clean_with_ds = X[valid_mask].copy() # Keep ds for weight calculation

    if len(X_clean) < 50:
        raise ValueError(f"Not enough valid training samples: {len(X_clean)}")

    logger.info(f"After cleaning: {len(X_clean)} valid samples (dropped {(~valid_mask).sum()} NaN targets)")

    # Calculate sample weights (using target_month as the anchor, which should be passed via params_override or derived)
    # In normal CV, the "target_month" anchor should be the maximum date in the validation partition
    # But since this function is broadly used, we default to a flat weight if target_month isn't provided or we fallback.
    # Note: `calculate_sample_weights` requires target_month, so we derive it from the dataset's max date
    # if it's not provided in params_override.
    # Extract custom keys without mutating the caller's dict
    _NON_LGB_KEYS = {'target_month', 'half_life_months'}
    target_month = params_override.get('target_month', pd.to_datetime(X['ds'].max())) if params_override else pd.to_datetime(X['ds'].max())
    half_life_months = params_override.get('half_life_months', 60.0) if params_override else 60.0
    
    # Needs the full dataframe with 'ds' attached for weight calculation
    weights = calculate_sample_weights(X_clean_with_ds, target_month, half_life_months)

    # LightGBM parameters (use tuned params if provided, else static defaults)
    # Filter out non-LightGBM keys to prevent unknown parameter errors
    if params_override is not None:
        params = {k: v for k, v in params_override.items() if k not in _NON_LGB_KEYS}
        logger.info("Using Optuna-tuned hyperparameters")
    else:
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

    # Use the final model's validation residuals (not CV residuals from earlier models)
    # These are OOS residuals from the 15% holdout of the final model
    final_val_preds = final_model.predict(X_final_val)
    final_residuals = (y_final_val.values - final_val_preds).tolist()

    return final_model, importance, final_residuals


# =============================================================================
# PREDICTION INTERVALS
# =============================================================================

def calculate_prediction_intervals(
    residuals: List[float],
    prediction: float,
    confidence_levels: List[float] = CONFIDENCE_LEVELS
) -> Dict[float, Tuple[float, float]]:
    """
    Calculate confidence intervals for a point prediction based strictly on historical OOS errors.

    Rather than assuming a normal distribution (Gaussian errors) for the model's accuracy, 
    this function uses completely non-parametric empirical quantiles derived from the model's 
    actual historical residuals. 

    Args:
        residuals (List[float]): List of historical out-of-sample prediction residuals.
        prediction (float): The newly predicted point estimate.
        confidence_levels (List[float]): Desired probability coverage levels (e.g., [0.50, 0.95]).

    Returns:
        Dict[float, Tuple[float, float]]: Dictionary mapping the confidence level to its
        respective (lower_bound, upper_bound) tuple.
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
    # Ensure correct feature alignment: reindex to match training column order,
    # filling missing columns with NaN (LightGBM handles NaN natively).
    # This prevents column-shift bugs when some features are absent at predict time.
    available_cols = [c for c in feature_cols if c in features.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing {len(missing)} feature columns (filled with NaN): {list(missing)[:5]}...")

    X = features.reindex(columns=feature_cols).values.reshape(1, -1)
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
