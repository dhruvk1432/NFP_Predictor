"""
Leakage-Safe Hyperparameter Tuning for LightGBM NFP Model

Uses Optuna with inner TimeSeriesSplit CV so that tuning never sees future data.
The outer expanding window backtest calls tune_hyperparameters() periodically
(every 12 months) on only the training data available at that point.

Requires: pip install optuna  (or pip install -e ".[hyperopt]")
Falls back to DEFAULT_LGBM_PARAMS if optuna is not installed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.config import (
    DEFAULT_LGBM_PARAMS,
    N_OPTUNA_TRIALS,
    OPTUNA_TIMEOUT,
    NUM_BOOST_ROUND,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    n_inner_splits: int = 3,
    use_huber_loss: bool = True,
    timeout: int = OPTUNA_TIMEOUT,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = 50,
) -> Dict:
    """
    Tune LightGBM hyperparameters using Optuna with inner TimeSeriesSplit CV.

    This is leakage-safe: it only uses the X/y data passed in (which should
    be strictly the training fold from the outer expanding window).

    Args:
        X: Training features (no 'ds' column)
        y: Training targets
        weights: Sample weights (from calculate_sample_weights)
        n_trials: Number of Optuna trials
        n_inner_splits: Number of inner TimeSeriesSplit folds
        use_huber_loss: Whether to use Huber objective
        timeout: Max seconds for the entire tuning run
        num_boost_round: Max boosting rounds per trial
        early_stopping_rounds: Early stopping patience per trial

    Returns:
        Best LightGBM parameter dict (ready to pass to lgb.train)
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed — using static defaults. "
                       "Install with: pip install optuna")
        params = DEFAULT_LGBM_PARAMS.copy()
        if use_huber_loss:
            params['objective'] = 'huber'
            params['alpha'] = 1.0
        return params

    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    # Replace inf with NaN (LightGBM handles NaN but not inf)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Filter NaN targets
    valid = ~y.isna()
    X = X[valid].copy()
    y = y[valid].copy()
    weights = weights[valid.values]

    inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'huber' if use_huber_loss else 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        if use_huber_loss:
            params['alpha'] = trial.suggest_float('huber_delta', 0.5, 5.0)

        fold_maes = []

        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = weights[train_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            pruning_callback = LightGBMPruningCallback(trial, 'valid', metric_name='l1')

            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),
                pruning_callback,
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                valid_names=['valid'],
                callbacks=callbacks,
            )

            preds = model.predict(X_val)
            fold_mae = np.mean(np.abs(y_val.values - preds))
            fold_maes.append(fold_mae)

        return np.mean(fold_maes)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import time as _time
    _tune_t0 = _time.time()
    logger.info(f"Starting Optuna: {n_trials} trials, timeout={timeout}s, "
                f"{len(X)} samples, {len(X.columns)} features, "
                f"{n_inner_splits}-fold inner CV")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    _tune_elapsed = _time.time() - _tune_t0
    _tune_str = f"{_tune_elapsed/60:.1f}m" if _tune_elapsed >= 60 else f"{_tune_elapsed:.0f}s"

    # Build final param dict from best trial
    best = study.best_trial
    best_params = {
        'objective': 'huber' if use_huber_loss else 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
    }
    best_params.update(best.params)

    # Rename huber_delta -> alpha for LightGBM
    if 'huber_delta' in best_params:
        best_params['alpha'] = best_params.pop('huber_delta')

    logger.info(f"Optuna tuning complete in {_tune_str}: {len(study.trials)} trials, "
                f"best MAE={best.value:.2f}")
    logger.info(f"Best params: lr={best_params.get('learning_rate', '?'):.4f}, "
                f"leaves={best_params.get('num_leaves', '?')}, "
                f"depth={best_params.get('max_depth', '?')}, "
                f"min_leaf={best_params.get('min_data_in_leaf', '?')}, "
                f"L1={best_params.get('lambda_l1', '?'):.2e}, "
                f"L2={best_params.get('lambda_l2', '?'):.2e}")

    return best_params
