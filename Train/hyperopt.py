"""
Hyperparameter Optimization for NFP LightGBM Model

Uses Optuna for efficient hyperparameter search with:
- Time-series cross-validation objective
- Regime-aware sample weighting
- Early stopping and pruning
- Best parameter persistence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import (
    MODEL_SAVE_DIR,
    DEFAULT_LGBM_PARAMS,
    N_CV_SPLITS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    PANIC_REGIME_WEIGHT,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# SEARCH SPACE DEFINITION
# =============================================================================

def get_search_space(trial: 'optuna.Trial') -> Dict:
    """
    Define hyperparameter search space.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters for this trial
    """
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        
        # Core tree parameters
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        
        # Regularization
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        
        # Additional regularization
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
    }
    
    # DART-specific parameters
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.4, 0.8)
    
    return params


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def calculate_sample_weights(X: pd.DataFrame) -> np.ndarray:
    """
    Calculate sample weights for regime-dependent training.
    
    Assigns higher weights to extreme event periods.
    """
    weights = np.ones(len(X))
    
    # Check for panic regime flags
    for col in X.columns:
        if 'panic_regime' in col.lower() or 'crash_month' in col.lower():
            panic_mask = X[col] == 1
            weights[panic_mask] = PANIC_REGIME_WEIGHT
            break
    
    return weights


def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS
) -> callable:
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of time-series CV splits
        
    Returns:
        Objective function that returns validation MAE
    """
    # Get feature columns (exclude date)
    feature_cols = [c for c in X.columns if c != 'ds']
    
    # Clean data
    X_clean = X[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
    X_clean = X_clean[valid_mask]
    y_clean = y[valid_mask]
    
    # Calculate weights
    weights = calculate_sample_weights(X[valid_mask])
    
    logger.info(f"Optimization data: {len(X_clean)} samples, {len(feature_cols)} features")
    
    def objective(trial: 'optuna.Trial') -> float:
        """Objective function to minimize: validation MAE."""
        params = get_search_space(trial)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_train = X_clean.iloc[train_idx]
            y_train = y_clean.iloc[train_idx]
            X_val = X_clean.iloc[val_idx]
            y_val = y_clean.iloc[val_idx]
            w_train = weights[train_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train with pruning callback
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
            if OPTUNA_AVAILABLE:
                callbacks.append(LightGBMPruningCallback(trial, 'l1'))
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[val_data],
                callbacks=callbacks
            )
            
            # Calculate validation MAE
            preds = model.predict(X_val)
            mae = np.mean(np.abs(y_val.values - preds))
            cv_scores.append(mae)
        
        mean_mae = np.mean(cv_scores)
        
        # Log progress
        trial.set_user_attr('cv_scores', cv_scores)
        
        return mean_mae
    
    return objective


# =============================================================================
# OPTIMIZATION RUNNER
# =============================================================================

def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    study_name: str = 'nfp_lgbm_optimization',
    save_dir: Optional[Path] = None
) -> Dict:
    """
    Run hyperparameter optimization.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (optional)
        study_name: Name for the Optuna study
        save_dir: Directory to save results
        
    Returns:
        Dictionary with best parameters and optimization history
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required. Install with: pip install optuna")
    
    if save_dir is None:
        save_dir = OUTPUT_DIR / "hyperopt"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting hyperparameter optimization: {n_trials} trials")
    
    # Create study with SQLite storage for persistence
    storage_path = save_dir / f"{study_name}.db"
    storage = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True
    )
    
    # Create objective
    objective = create_objective(X, y)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Get results
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best MAE: {best_value:.2f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Build complete params dict (merge with defaults)
    complete_params = DEFAULT_LGBM_PARAMS.copy()
    complete_params.update(best_params)
    
    # Save best parameters
    params_path = save_dir / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'complete_params': complete_params,
            'best_mae': best_value,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    logger.info(f"Best parameters saved to {params_path}")
    
    # Save trial history
    history_df = study.trials_dataframe()
    history_path = save_dir / "optimization_history.csv"
    history_df.to_csv(history_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Hyperparameter Optimization Complete")
    print(f"{'='*60}")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best MAE: {best_value:.2f}")
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {save_dir}")
    
    return {
        'best_params': best_params,
        'complete_params': complete_params,
        'best_mae': best_value,
        'n_trials': len(study.trials),
        'study': study
    }


def get_best_params(save_dir: Optional[Path] = None) -> Dict:
    """
    Load previously saved best parameters.
    
    Args:
        save_dir: Directory containing saved params
        
    Returns:
        Dictionary with best parameters
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR / "hyperopt"
    
    params_path = save_dir / "best_params.json"
    
    if not params_path.exists():
        logger.warning("No saved parameters found. Using defaults.")
        return DEFAULT_LGBM_PARAMS.copy()
    
    with open(params_path, 'r') as f:
        data = json.load(f)
    
    return data.get('complete_params', DEFAULT_LGBM_PARAMS.copy())


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run hyperparameter optimization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize LightGBM hyperparameters for NFP")
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help="Target type (default: nsa)")
    parser.add_argument('--trials', type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    parser.add_argument('--timeout', type=int, default=None,
                        help="Maximum time in seconds (optional)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load data for optimization
    from Train.data_loader import load_target_data, load_master_snapshot
    
    logger.info(f"Loading {args.target.upper()} target data...")
    target_df = load_target_data(args.target)
    
    # TODO: Load and prepare features (would need to call build_training_dataset)
    print("Note: Full optimization requires loading master snapshots.")
    print("Run with pre-built training data or call from train_lightgbm_nfp.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
