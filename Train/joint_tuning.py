"""
Joint Optuna tuning over (NSA LightGBM params, adjustment half_life_years,
Kalman fusion params).

The pre-joint pipeline tuned three things separately:
  1. NSA LightGBM hyperparameters (per reselection step) -- scored on either
     plain NSA-MAE or the Kalman-fusion composite with a *frozen* adjustment
     half-life and *default* Kalman params.
  2. Adjustment half_life_years (post-training, by `_tune_kalman`) jointly
     with Kalman trailing_window and nsa_weight_scale.
  3. Kalman trailing_window + nsa_weight_scale (same `_tune_kalman` call).

That decoupling leaves a known stability gap: NSA hyperparams are chosen
under one HL/Kalman regime and then Kalman re-tunes under a different one,
which can leave the joint optimum slightly off.

This module collapses all three into ONE Optuna study so every NSA-param
configuration is scored under the HL and Kalman params it would actually
ship with. Per trial the score is:

    mean over expanding-window folds of
        MAE - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc

on the validation portion of each fold, where:

  * The NSA model is fit on the training fold with the trial's
    LightGBM params and exponential sample weights (decayed by the trial's
    ``half_life_months``).
  * For each ``ds`` in the combined train+val portion of the fold, the
    PIT-safe ``adj_pred[ds]`` is computed using the trial's
    ``half_life_years`` over the pre-cached adjustment history.
  * The champion channel ``nsa_pred + adj_pred`` is fused via
    ``kalman_fusion`` with the trial's ``trailing_window`` and
    ``nsa_weight_scale``.
  * Only the validation rows score the trial -- training-fold predictions
    are fed in as Kalman's noise-prior history, never as score targets.

PIT invariants
--------------
* SA actuals are the score target only; the model never sees them as
  features.
* ``adj_pred`` per ds uses adjustment-history rows with
  ``operational_available_date < ds``.
* Consensus per ds is loaded via ``_load_consensus_pit`` (master-snapshot
  PIT cache).
* Inner expanding-window CV folds are strictly chronological -- no fold's
  validation rows overlap or precede its training rows.

Usage
-----
Set ``Train.config.JOINT_OPTUNA = True``. ``train_lightgbm_nfp`` will then
call ``tune_full_pipeline_joint`` at each reselection step instead of the
separate ``tune_hyperparameters`` call, persist the chosen
(half_life_years, trailing_window, nsa_weight_scale) to
``_output/consensus_anchor/kalman_fusion/joint_tuned_params.json``, and skip
the post-hoc ``_tune_kalman`` (because the joint study already picked those
values).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Data_ETA_Pipeline.perf_stats import inc_counter, perf_phase, profiled
from Train.config import (
    DEFAULT_LGBM_PARAMS,
    HALF_LIFE_MAX_MONTHS,
    HALF_LIFE_MIN_MONTHS,
    HUBER_DELTA,
    KALMAN_LAMBDA_ACCEL,
    KALMAN_LAMBDA_DIR,
    LGBM_DETERMINISM,
    NUM_BOOST_ROUND,
    N_OPTUNA_TRIALS,
    OPTUNA_TIMEOUT,
    lgbm_n_jobs,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError as _optuna_import_err:
    OPTUNA_AVAILABLE = False
    _OPTUNA_IMPORT_ERROR = _optuna_import_err
    logger.error(f"Optuna unavailable for joint tuning: {_optuna_import_err}")

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public fusion-context builder
# ---------------------------------------------------------------------------

def build_joint_fusion_context(
    target_dates: pd.Series,
) -> Optional[Dict]:
    """Build the context needed for joint tuning.

    Differs from ``_build_fusion_tuning_context`` (in train_lightgbm_nfp.py)
    by ALSO exposing the raw ``adj_history`` and a precomputed PIT cache
    keyed by ``ds`` so each Optuna trial can recompute per-ds adjustments
    with its own ``half_life_years`` candidate -- without redoing the
    operational_available_date filter sweep.

    Returns None if any required input fails to load.
    """
    try:
        from Train.Output_code.consensus_anchor_runner import (
            _load_consensus_pit,
            _build_pit_adjustment_cache,
        )
        from Train.data_loader import load_target_data
        from Train.sandbox.experiment_predicted_adjustment import (
            load_adjustment_history,
        )
    except Exception as e:
        logger.warning(f"[JointTune] dependency import failed ({e}); cannot build context")
        return None

    try:
        consensus_monthly = _load_consensus_pit(
            target_type='sa', target_source='revised',
        )
        sa_target = load_target_data(
            target_type='sa', release_type='first', target_source='revised',
        )
        adj_history = load_adjustment_history()
    except Exception as e:
        logger.warning(f"[JointTune] context loading failed ({e})")
        return None

    sa_actuals_df = pd.DataFrame({
        'ds': pd.to_datetime(sa_target['ds']),
        'actual': sa_target['y_mom'].astype(float),
    })
    consensus_df = (
        consensus_monthly[['ds', 'consensus_pred']]
        .merge(sa_actuals_df, on='ds', how='outer')
        .sort_values('ds')
        .reset_index(drop=True)
    )

    consensus_by_ds: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): float(v)
        for d, v in zip(
            pd.to_datetime(consensus_monthly['ds']).tolist(),
            consensus_monthly['consensus_pred'].astype(float).tolist(),
        )
        if np.isfinite(v)
    }
    sa_actuals_by_ds: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): float(v)
        for d, v in zip(
            pd.to_datetime(sa_target['ds']).tolist(),
            sa_target['y_mom'].astype(float).tolist(),
        )
        if np.isfinite(v)
    }

    unique_dates = pd.Series(pd.to_datetime(pd.unique(target_dates.values))).sort_values()
    pit_cache = _build_pit_adjustment_cache(unique_dates, adj_history)

    logger.info(
        f"[JointTune] built joint context: consensus={len(consensus_by_ds)}, "
        f"sa_actuals={len(sa_actuals_by_ds)}, pit_cache_keys={len(pit_cache)}"
    )
    return {
        'consensus_df': consensus_df,
        'consensus_by_ds': consensus_by_ds,
        'sa_actuals_by_ds': sa_actuals_by_ds,
        'adj_history': adj_history,
        'pit_cache': pit_cache,
    }


# ---------------------------------------------------------------------------
# Per-trial scoring
# ---------------------------------------------------------------------------

def _build_adj_pred_for_dates(
    dates: pd.DatetimeIndex,
    pit_cache: Dict[pd.Timestamp, pd.DataFrame],
    half_life_years: float,
) -> Dict[pd.Timestamp, float]:
    """Compute per-ds adjustment predictions with the given half-life.

    Reuses the existing ExpWeightedMedianCovidExcludedPredictor implementation
    so the joint study sees the SAME adjustment surface that production uses.
    """
    from Train.sandbox.experiment_predicted_adjustment import (
        ExpWeightedMedianCovidExcludedPredictor,
    )
    predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=half_life_years)
    out: Dict[pd.Timestamp, float] = {}
    for ds in dates:
        target_ds = pd.Timestamp(ds)
        avail = pit_cache.get(target_ds)
        if avail is None or avail.empty:
            out[target_ds] = 0.0
            continue
        try:
            out[target_ds] = float(predictor.fit_predict(avail, target_ds))
        except Exception:
            out[target_ds] = 0.0
    return out


def _score_fold_joint(
    model,
    X_tr_feats: pd.DataFrame,
    X_val_feats: pd.DataFrame,
    ds_tr: np.ndarray,
    ds_val: np.ndarray,
    *,
    consensus_by_ds: Dict[pd.Timestamp, float],
    sa_actuals_by_ds: Dict[pd.Timestamp, float],
    adj_pred_by_ds: Dict[pd.Timestamp, float],
    consensus_df: pd.DataFrame,
    kalman_trailing_window: int,
    kalman_nsa_weight_scale: float,
) -> float:
    """Run the full fusion pipeline on one CV fold and score the composite.

    Mirrors the structure of ``_score_fold_kalman_fusion`` but uses the
    trial's (adj_pred, kalman_params) rather than frozen defaults.
    """
    from Train.Output_code.consensus_anchor_runner import (
        _composite_kalman_accel_objective,
        kalman_fusion,
    )

    try:
        nsa_tr = np.asarray(model.predict(X_tr_feats), dtype=float)
        nsa_val = np.asarray(model.predict(X_val_feats), dtype=float)
    except Exception:
        return float("inf")

    ds_all = np.concatenate([np.asarray(ds_tr), np.asarray(ds_val)])
    nsa_all = np.concatenate([nsa_tr, nsa_val])

    rows: List[Dict] = []
    for i, ds in enumerate(ds_all):
        ts = pd.Timestamp(ds)
        cons = consensus_by_ds.get(ts)
        if cons is None or not np.isfinite(cons):
            continue
        adj = adj_pred_by_ds.get(ts, 0.0)
        champ = float(nsa_all[i]) + float(adj)
        actual = sa_actuals_by_ds.get(ts, np.nan)
        rows.append({
            "ds": ts,
            "consensus_pred": float(cons),
            "champion_pred": champ,
            "nsa_pred": champ,
            "actual": float(actual) if actual is not None and np.isfinite(actual) else np.nan,
        })

    if not rows:
        return float("inf")

    overlap_df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    try:
        res_df, _ = kalman_fusion(
            overlap_df, consensus_df,
            trailing_window=int(kalman_trailing_window),
            use_nsa_accel=True,
            nsa_weight_scale=float(kalman_nsa_weight_scale),
        )
    except Exception:
        return float("inf")

    val_set = set(pd.Timestamp(d) for d in ds_val)
    val_mask = (
        res_df["ds"].isin(val_set)
        & res_df["actual"].notna()
        & res_df["predicted"].notna()
    )
    if int(val_mask.sum()) < 3:
        return float("inf")

    actual_arr = res_df.loc[val_mask, "actual"].to_numpy(dtype=float)
    pred_arr = res_df.loc[val_mask, "predicted"].to_numpy(dtype=float)
    return float(_composite_kalman_accel_objective(actual_arr, pred_arr))


# ---------------------------------------------------------------------------
# Joint tuner
# ---------------------------------------------------------------------------

@profiled("train.joint_tuning.total")
def tune_full_pipeline_joint(
    X: pd.DataFrame,
    y: pd.Series,
    target_month: pd.Timestamp,
    *,
    fusion_context: Dict,
    n_trials: int = N_OPTUNA_TRIALS,
    n_inner_splits: int = 5,
    use_huber_loss: bool = True,
    timeout: int = OPTUNA_TIMEOUT,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = 50,
    warm_start_params: Optional[Dict] = None,
) -> Dict:
    """Single Optuna study over NSA + HL + Kalman params.

    Args:
        X: training features (must include 'ds')
        y: training target (NSA y_mom)
        target_month: outer-loop target month
        fusion_context: output of ``build_joint_fusion_context``
        n_trials: Optuna trial budget
        n_inner_splits: inner expanding-window CV folds
        use_huber_loss: whether to use Huber objective
        timeout: per-study wall-clock budget
        num_boost_round / early_stopping_rounds: LGBM training control
        warm_start_params: optional dict of prior best params to seed Trial 0

    Returns:
        Dict with keys:
          * 'lgbm_params': dict ready to pass to ``lgb.train``
          * 'half_life_years': float
          * 'kalman_params': {'trailing_window': int, 'nsa_weight_scale': float}
          * 'best_score': float (composite from the best trial)
          * 'n_trials_run': int
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError(
            f"Optuna required for joint tuning: {_OPTUNA_IMPORT_ERROR}"
        )
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")

    required_keys = {'consensus_df', 'consensus_by_ds', 'sa_actuals_by_ds', 'pit_cache'}
    missing = required_keys - set(fusion_context.keys())
    if missing:
        raise ValueError(f"fusion_context missing keys: {missing}")

    consensus_df = fusion_context['consensus_df']
    consensus_by_ds = fusion_context['consensus_by_ds']
    sa_actuals_by_ds = fusion_context['sa_actuals_by_ds']
    pit_cache = fusion_context['pit_cache']

    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~y.isna()
    X = X[valid].copy()
    y = y[valid].copy()

    inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)

    # Cache adj_pred per (half_life_years rounded) so trials that pick the
    # same HL skip the inner ExpWeightedMedian sweep. HL is a continuous
    # float in [0.5, 8.0] but the median predictor is smooth -- rounding to
    # 3 decimals is enough.
    adj_cache: Dict[Tuple[float, Tuple[pd.Timestamp, ...]], Dict[pd.Timestamp, float]] = {}

    def _get_adj_pred(hl: float, dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, float]:
        hl_key = round(float(hl), 3)
        date_key = tuple(pd.Timestamp(d) for d in dates)
        cache_key = (hl_key, date_key)
        if cache_key in adj_cache:
            return adj_cache[cache_key]
        out = _build_adj_pred_for_dates(dates, pit_cache, hl_key)
        adj_cache[cache_key] = out
        return out

    def objective(trial: "optuna.Trial") -> float:
        # ── LightGBM params --------------------------------------------------
        params = {
            'objective': 'huber' if use_huber_loss else 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': lgbm_n_jobs(),
            **LGBM_DETERMINISM,
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

        half_life_months = trial.suggest_float(
            'half_life_months', HALF_LIFE_MIN_MONTHS, HALF_LIFE_MAX_MONTHS,
        )

        # ── Adjustment + Kalman params --------------------------------------
        hl_years = trial.suggest_float('half_life_years', 0.5, 8.0)
        kalman_tw = trial.suggest_int('kalman_trailing_window', 6, 36)
        kalman_ws = trial.suggest_float('kalman_nsa_weight_scale', 0.1, 3.0)

        # Local import (config.py imports kept above, sample_weights is internal)
        from Train.model import calculate_sample_weights

        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_target_month = pd.to_datetime(X.iloc[val_idx]['ds'].max())
            w_tr = calculate_sample_weights(X_tr, fold_target_month, half_life_months)

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
            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    callbacks=callbacks,
                )
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.debug(f"[JointTune] fold {fold_idx} LGBM train failed: {e}")
                fold_scores.append(float("inf"))
                continue

            ds_tr_arr = pd.to_datetime(X_tr['ds'].values)
            ds_val_arr = pd.to_datetime(X_val['ds'].values)
            ds_all = pd.DatetimeIndex(np.concatenate([ds_tr_arr, ds_val_arr]))
            adj_pred_by_ds = _get_adj_pred(hl_years, ds_all)

            fold_score = _score_fold_joint(
                model=model,
                X_tr_feats=X_tr_feats,
                X_val_feats=X_val_feats,
                ds_tr=ds_tr_arr.values,
                ds_val=ds_val_arr.values,
                consensus_by_ds=consensus_by_ds,
                sa_actuals_by_ds=sa_actuals_by_ds,
                adj_pred_by_ds=adj_pred_by_ds,
                consensus_df=consensus_df,
                kalman_trailing_window=kalman_tw,
                kalman_nsa_weight_scale=kalman_ws,
            )
            fold_scores.append(fold_score)

        return float(np.mean(fold_scores)) if fold_scores else float("inf")

    def _counted_objective(trial: "optuna.Trial") -> float:
        inc_counter("train.joint_tuning.trials")
        with perf_phase("train.joint_tuning.trial", trial_number=trial.number):
            return objective(trial)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    if warm_start_params is not None:
        # Seed Trial 0 with previously-validated params so the joint study
        # explores AROUND a known-good prior. We translate previous best
        # params (which may have been chosen by the separate tuners) into
        # this study's parameter space.
        _accepted = {
            'learning_rate', 'num_leaves', 'max_depth', 'min_data_in_leaf',
            'feature_fraction', 'bagging_fraction', 'bagging_freq',
            'lambda_l1', 'lambda_l2', 'half_life_months', 'huber_delta',
            'half_life_years', 'kalman_trailing_window', 'kalman_nsa_weight_scale',
        }
        seed = {}
        for k, v in warm_start_params.items():
            if k == 'alpha' and use_huber_loss:
                seed['huber_delta'] = v
            elif k in _accepted:
                seed[k] = v
        if seed:
            study.enqueue_trial(seed)
            logger.info(f"[JointTune] warm-start seeded Trial 0 with {len(seed)} params")

    t0 = time.time()
    logger.info(
        f"[JointTune] starting joint Optuna: trials={n_trials} timeout={timeout}s "
        f"samples={len(X)} features={len(X.columns)} inner_cv_folds={n_inner_splits} "
        f"lambda_accel={KALMAN_LAMBDA_ACCEL} lambda_dir={KALMAN_LAMBDA_DIR}"
    )
    study.optimize(_counted_objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.time() - t0

    best = study.best_trial
    best_params = best.params

    lgbm_params = {
        'objective': 'huber' if use_huber_loss else 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': lgbm_n_jobs(),
        **LGBM_DETERMINISM,
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'max_depth': best_params['max_depth'],
        'min_data_in_leaf': best_params['min_data_in_leaf'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'half_life_months': best_params['half_life_months'],
    }
    if use_huber_loss and 'huber_delta' in best_params:
        lgbm_params['alpha'] = best_params['huber_delta']

    result = {
        'lgbm_params': lgbm_params,
        'half_life_years': float(best_params['half_life_years']),
        'kalman_params': {
            'trailing_window': int(best_params['kalman_trailing_window']),
            'nsa_weight_scale': float(best_params['kalman_nsa_weight_scale']),
        },
        'best_score': float(best.value),
        'n_trials_run': len(study.trials),
        'elapsed_s': float(elapsed),
    }

    logger.info(
        f"[JointTune] done in {elapsed/60:.1f}m, {len(study.trials)} trials, "
        f"best_score={best.value:.2f}, "
        f"HL={result['half_life_years']:.2f}y, "
        f"kalman_tw={result['kalman_params']['trailing_window']}, "
        f"nsa_weight_scale={result['kalman_params']['nsa_weight_scale']:.2f}, "
        f"lr={lgbm_params['learning_rate']:.4f}, "
        f"leaves={lgbm_params['num_leaves']}, "
        f"half_life_months={lgbm_params['half_life_months']:.1f}"
    )
    return result


__all__ = [
    "build_joint_fusion_context",
    "tune_full_pipeline_joint",
]
