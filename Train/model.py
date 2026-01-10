"""
LightGBM Model Training for NFP Prediction

Core model training, prediction, and interval calculation functions.
Extracted from train_lightgbm_nfp.py for maintainability.
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
    
    # Check for panic regime flags
    if 'VIX_panic_regime_latest' in X.columns:
        panic_mask = X['VIX_panic_regime_latest'] == 1
        weights[panic_mask] = PANIC_REGIME_WEIGHT
    
    if 'SP500_crash_month_latest' in X.columns:
        crash_mask = X['SP500_crash_month_latest'] == 1
        weights[crash_mask] = PANIC_REGIME_WEIGHT
    
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
) -> Tuple[lgb.Booster, List[float], Dict]:
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
        Tuple of (trained model, residuals list, feature importance dict)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    
    # Get feature columns (exclude date column)
    feature_cols = [c for c in X.columns if c != 'ds']
    
    # Remove any remaining NaN/Inf from features
    X_clean = X[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any NaN
    valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
    X_clean = X_clean[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_clean) < 50:
        raise ValueError(f"Not enough valid training samples: {len(X_clean)}")
    
    logger.info(f"Training on {len(X_clean)} samples with {len(feature_cols)} features")
    
    # Calculate sample weights
    weights = calculate_sample_weights(X[valid_mask])
    
    # Get parameters
    params = get_lgbm_params(use_huber_loss, huber_delta)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    residuals = []
    feature_importance = {col: 0 for col in feature_cols}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
        X_train = X_clean.iloc[train_idx]
        y_train = y_clean.iloc[train_idx]
        X_val = X_clean.iloc[val_idx]
        y_val = y_clean.iloc[val_idx]
        w_train = weights[train_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        
        # Collect residuals and importance
        preds = model.predict(X_val)
        fold_residuals = (y_val.values - preds).tolist()
        residuals.extend(fold_residuals)
        
        for i, col in enumerate(feature_cols):
            feature_importance[col] += model.feature_importance()[i]
    
    # Final model on all data
    final_train_data = lgb.Dataset(X_clean, label=y_clean, weight=weights)
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=num_boost_round // 2  # Use fewer rounds for final
    )
    
    # Normalize importance
    for col in feature_importance:
        feature_importance[col] /= n_splits
    
    # Sort by importance
    feature_importance = dict(sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return final_model, residuals, feature_importance


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
        Dictionary with prediction and interval information
    """
    X = features[feature_cols].values.reshape(1, -1)
    prediction = model.predict(X)[0]
    
    intervals = calculate_prediction_intervals(residuals, prediction)
    
    return {
        'prediction': prediction,
        'intervals': intervals,
        'lower_50': intervals.get(0.50, (np.nan, np.nan))[0],
        'upper_50': intervals.get(0.50, (np.nan, np.nan))[1],
        'lower_80': intervals.get(0.80, (np.nan, np.nan))[0],
        'upper_80': intervals.get(0.80, (np.nan, np.nan))[1],
        'lower_95': intervals.get(0.95, (np.nan, np.nan))[0],
        'upper_95': intervals.get(0.95, (np.nan, np.nan))[1],
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
    linear_model: Optional[Any] = None,
    linear_cols: Optional[List[str]] = None
) -> None:
    """Save model and associated metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LightGBM model
    model.save_model(str(save_dir / f"lightgbm_{target_type}_model.txt"))
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'residuals': residuals,
        'importance': importance,
        'target_type': target_type,
        'linear_model': linear_model,
        'linear_cols': linear_cols,
    }
    
    with open(save_dir / f"lightgbm_{target_type}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Model saved to {save_dir}")


def load_model(
    save_dir: Path = MODEL_SAVE_DIR, 
    target_type: str = 'nsa'
) -> Tuple[lgb.Booster, Dict]:
    """Load model and associated metadata."""
    model_path = save_dir / f"lightgbm_{target_type}_model.txt"
    metadata_path = save_dir / f"lightgbm_{target_type}_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = lgb.Booster(model_file=str(model_path))
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Model loaded from {save_dir}")
    
    return model, metadata


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
