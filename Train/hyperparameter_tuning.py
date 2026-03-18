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
from Data_ETA_Pipeline.perf_stats import profiled, perf_phase, inc_counter
from Train.config import (
    DEFAULT_LGBM_PARAMS,
    HUBER_DELTA,
    N_OPTUNA_TRIALS,
    OPTUNA_TIMEOUT,
    NUM_BOOST_ROUND,
    HALF_LIFE_MIN_MONTHS,
    HALF_LIFE_MAX_MONTHS,
    VARIANCE_TAIL_QUANTILE,
    TUNING_LAMBDA_STD_RATIO,
    TUNING_LAMBDA_DIFF_STD_RATIO,
    TUNING_LAMBDA_TAIL_MAE,
    TUNING_LAMBDA_CORR_DIFF,
    TUNING_LAMBDA_DIFF_SIGN,
    TUNING_LAMBDA_ACCEL,
    TUNING_LAMBDA_DIR,
)
from Train.variance_metrics import compute_variance_kpis, composite_objective_score

logger = setup_logger(__file__, TEMP_DIR)

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError as _optuna_import_err:
    OPTUNA_AVAILABLE = False
    _OPTUNA_IMPORT_ERROR = _optuna_import_err
    logger.error(f"Optuna import failed: {_optuna_import_err}. "
                 "Hyperparameter tuning will not be available. "
                 "Install with: pip install optuna")

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@profiled("train.tuning.total")
def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    target_month: pd.Timestamp,
    n_trials: int = N_OPTUNA_TRIALS,
    n_inner_splits: int = 5,
    use_huber_loss: bool = True,
    timeout: int = OPTUNA_TIMEOUT,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = 50,
    objective_mode: str = 'mae',
    lambda_std_ratio: float = TUNING_LAMBDA_STD_RATIO,
    lambda_diff_std_ratio: float = TUNING_LAMBDA_DIFF_STD_RATIO,
    lambda_tail_mae: float = TUNING_LAMBDA_TAIL_MAE,
    lambda_corr_diff: float = TUNING_LAMBDA_CORR_DIFF,
    lambda_diff_sign: float = TUNING_LAMBDA_DIFF_SIGN,
    tail_quantile: float = VARIANCE_TAIL_QUANTILE,
    tail_weighting: bool = False,
    tail_weight_abs_level_quantile: float = 0.80,
    tail_weight_abs_diff_quantile: float = 0.80,
    tail_weight_level_boost: float = 1.35,
    tail_weight_diff_boost: float = 1.35,
    tail_weight_max_multiplier: float = 2.50,
    warm_start_params: Optional[Dict] = None,
    lambda_accel: float = TUNING_LAMBDA_ACCEL,
    lambda_dir: float = TUNING_LAMBDA_DIR,
) -> Dict:
    """
    Tune LightGBM hyperparameters using Optuna to find the optimal model configuration
    for the current expanding window step.

    This function utilizes an inner TimeSeriesSplit cross-validation strategy, which is
    strictly forward-looking. This guarantees that hyperparameter search never leaks
    future target information into the training phase (a common point of failure in
    time-series ML models).

    Args:
        X: Training features (must include 'ds' column for weight calculation)
        y: Training targets
        target_month: The target month being predicted in the outer loop
        n_trials: Number of Optuna trials
        n_inner_splits: Number of inner TimeSeriesSplit folds
        use_huber_loss: Whether to use Huber objective
        timeout: Max seconds for the entire tuning run
        num_boost_round: Max boosting rounds per trial
        early_stopping_rounds: Early stopping patience per trial
        warm_start_params: Optional dict of previous best params to seed
            Trial 0 with.  Lets Optuna explore *around* a known-good prior
            rather than starting from scratch after feature reselection.

    Returns:
        Best LightGBM parameter dict (ready to pass to lgb.train)
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError(
            f"Optuna is required for hyperparameter tuning but failed to import: "
            f"{_OPTUNA_IMPORT_ERROR}. Install with: pip install optuna"
        )

    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    # Replace inf with NaN (LightGBM handles NaN but not inf)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Filter NaN targets
    valid = ~y.isna()
    X = X[valid].copy()
    y = y[valid].copy()

    inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)

    def objective(trial: optuna.Trial) -> float:
        """
        The Optuna objective function, defining the hyperparameter search space and 
        evaluation metrics for each trial round.

        Args:
            trial (optuna.Trial): The current Optuna trial object suggesting params.

        Returns:
            float: The mean fold score across inner CV folds.
        """
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
            params['alpha'] = trial.suggest_float('huber_delta', 25.0, 500.0)

        # Dynamic sample weighting half-life
        half_life_months = trial.suggest_float('half_life_months', HALF_LIFE_MIN_MONTHS, HALF_LIFE_MAX_MONTHS)

        # Tail-aware weight boosts (tuned when tail_weighting is enabled)
        if tail_weighting:
            _tw_level_boost = trial.suggest_float('tail_weight_level_boost', 1.0, 3.0)
            _tw_diff_boost = trial.suggest_float('tail_weight_diff_boost', 1.0, 3.0)
            _tw_max_mult = trial.suggest_float('tail_weight_max_multiplier', 1.5, 4.0)
        else:
            _tw_level_boost = tail_weight_level_boost
            _tw_diff_boost = tail_weight_diff_boost
            _tw_max_mult = tail_weight_max_multiplier

        # Import locally to avoid circular import since model.py imports config.py
        from Train.model import calculate_sample_weights

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Recalculate weights for this specific fold using the fold's own target month
            # (which is effectively the maximum date available in the fold to simulate true point-in-time)
            # or we could use the outer target_month. But using the true available max date prevents
            # the weights from overly decaying for early folds.
            fold_target_month = pd.to_datetime(X.iloc[val_idx]['ds'].max())

            w_tr = calculate_sample_weights(X_tr, fold_target_month, half_life_months)
            if tail_weighting and len(y_tr) > 0:
                y_arr = y_tr.values.astype(float)
                mult = np.ones_like(y_arr, dtype=float)

                abs_y = np.abs(y_arr)
                level_thr = float(np.quantile(abs_y, tail_weight_abs_level_quantile))
                mult[abs_y >= level_thr] *= _tw_level_boost

                abs_dy = np.abs(np.diff(y_arr, prepend=y_arr[0]))
                diff_thr = float(np.quantile(abs_dy, tail_weight_abs_diff_quantile))
                mult[abs_dy >= diff_thr] *= _tw_diff_boost

                mult = np.clip(mult, 1.0, _tw_max_mult)
                w_tr = w_tr * mult
                if float(np.mean(w_tr)) > 0:
                    w_tr = w_tr / float(np.mean(w_tr))

            # Drop 'ds' for LightGBM
            X_tr_feats = X_tr.drop(columns=['ds'])
            X_val_feats = X_val.drop(columns=['ds'])

            train_data = lgb.Dataset(X_tr_feats, label=y_tr, weight=w_tr)
            val_data = lgb.Dataset(X_val_feats, label=y_val, reference=train_data)

            pruning_callback = LightGBMPruningCallback(trial, 'l1', valid_name='valid')

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

            preds = model.predict(X_val_feats)
            y_val_arr = y_val.values.astype(float)
            pred_arr = np.asarray(preds, dtype=float)
            fold_mae = float(np.mean(np.abs(y_val_arr - pred_arr)))

            if objective_mode == 'composite':
                kpis = compute_variance_kpis(
                    y_val_arr, pred_arr, tail_quantile=tail_quantile
                )
                # Compute acceleration accuracy (need ≥3 samples for diff)
                accel_acc = 0.0
                if y_val_arr.size >= 3:
                    da = np.diff(y_val_arr)
                    dp = np.diff(pred_arr)
                    accel_acc = float(np.mean(np.sign(da) == np.sign(dp)))

                # Compute directional accuracy
                dir_acc = 0.0
                if y_val_arr.size >= 1:
                    dir_acc = float(np.mean(np.sign(y_val_arr) == np.sign(pred_arr)))

                fold_score = composite_objective_score(
                    mae=float(fold_mae),
                    std_ratio=float(kpis['std_ratio']),
                    diff_std_ratio=float(kpis['diff_std_ratio']),
                    tail_mae=float(kpis['tail_mae']),
                    corr_diff=float(kpis['corr_diff']),
                    diff_sign_accuracy=float(kpis['diff_sign_accuracy']),
                    lambda_std_ratio=lambda_std_ratio,
                    lambda_diff_std_ratio=lambda_diff_std_ratio,
                    lambda_tail_mae=lambda_tail_mae,
                    lambda_corr_diff=lambda_corr_diff,
                    lambda_diff_sign=lambda_diff_sign,
                    accel_accuracy=accel_acc,
                    lambda_accel=lambda_accel,
                    dir_accuracy=dir_acc,
                    lambda_dir=lambda_dir,
                )
            else:
                fold_score = float(fold_mae)
            fold_scores.append(fold_score)

        return float(np.mean(fold_scores))

    def _counted_objective(trial: optuna.Trial) -> float:
        inc_counter("train.tuning.trials")
        with perf_phase("train.tuning.trial", trial_number=trial.number):
            return objective(trial)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import time as _time
    _tune_t0 = _time.time()
    logger.info(f"Starting Optuna ({objective_mode} objective): {n_trials} trials, timeout={timeout}s, "
                f"{len(X)} samples, {len(X.columns)} features, "
                f"{n_inner_splits}-fold inner CV")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    # Warm-start: seed Trial 0 with previous best params so Optuna explores
    # around a known-good prior instead of starting cold after reselection.
    if warm_start_params is not None:
        # Extract only the tunable param keys from the previous best
        _tunable_keys = {
            'learning_rate', 'num_leaves', 'max_depth', 'min_data_in_leaf',
            'feature_fraction', 'bagging_fraction', 'bagging_freq',
            'lambda_l1', 'lambda_l2', 'half_life_months', 'huber_delta',
            'tail_weight_level_boost', 'tail_weight_diff_boost', 'tail_weight_max_multiplier',
        }
        # Map 'alpha' back to 'huber_delta' (LightGBM → Optuna naming)
        seed_params = {}
        for k, v in warm_start_params.items():
            if k == 'alpha' and use_huber_loss:
                seed_params['huber_delta'] = v
            elif k in _tunable_keys:
                seed_params[k] = v
        if seed_params:
            study.enqueue_trial(seed_params)
            logger.info(f"Warm-start: seeded Trial 0 with {len(seed_params)} params "
                        f"from previous best")

    study.optimize(_counted_objective, n_trials=n_trials, timeout=timeout)

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
        
    # Extract structural best_params for weights so they can be passed to model training
    if 'half_life_months' in best_params:
        # It's kept in best_params so model.py can extract it later via params_override
        pass

    logger.info(f"Optuna tuning complete in {_tune_str}: {len(study.trials)} trials, "
                f"best score={best.value:.2f}")
    logger.info(f"Best params: lr={best_params.get('learning_rate', '?'):.4f}, "
                f"leaves={best_params.get('num_leaves', '?')}, "
                f"depth={best_params.get('max_depth', '?')}, "
                f"min_leaf={best_params.get('min_data_in_leaf', '?')}, "
                f"L1={best_params.get('lambda_l1', '?'):.2e}, "
                f"L2={best_params.get('lambda_l2', '?'):.2e}, "
                f"half_life={best_params.get('half_life_months', '?'):.1f}m")

    return best_params
