"""
LightGBM NFP Prediction Model

Predicts NSA NFP month-on-month change using master snapshots.
Uses release_date cutoff (not target_month) to match inference behavior.

Key Features:
- Handles data with varying start dates (some series start 1948, others 2008)
- Uses past target data (NFP levels, MoM changes) as features
- Survey interval features (4 vs 5 weeks logic)
- Momentum/divergence and acceleration features
- Cyclical month encoding (sin/cos)

MODULAR ARCHITECTURE:
This file is the main entry point. Core functionality is split into:
- Train/config.py: Configuration constants and hyperparameters
- Train/data_loader.py: Data loading functions
- Train/feature_engineering.py: Feature creation
- Train/model.py: Model training and prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, BACKTEST_MONTHS

# Import from modular components
from Train.config import (
    MODEL_SAVE_DIR,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    get_model_id,
    get_target_path,
    USE_HUBER_LOSS_DEFAULT,
    HUBER_DELTA,
    TUNE_EVERY_N_MONTHS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
)

from Train.data_loader import (
    load_master_snapshot,
    load_target_data,
    get_lagged_target_features,
    batch_lagged_target_features,
    pivot_snapshot_to_wide,
)

from joblib import Parallel, delayed

def _process_single_month_task(
    target_month: pd.Timestamp,
    target_value: float,
    cutoff_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
    target_type: str,
    release_type: str,
    target_source: str,
    nsa_lags_lookup: Dict[pd.Timestamp, Dict[str, float]],
    sa_lags_lookup: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """
    Helper function to process the feature engineering for a single month in parallel.
    Disables caching to avoid memory bloat in worker processes.

    Loads from the pre-merged, feature-selected master snapshot which already contains
    ALL data sources (FRED employment + exogenous). No separate FRED loading needed.

    This function sequentially:
    1. Extracts calendar metrics (survey weeks, seasonality).
    2. Collects historical NFP target lags (1m, 2m, 3m, 6m, 12m).
    3. Pivots the wide-format Master Snapshot for just this date cutoff.
    4. Computes short-term data revision metrics by comparing master[M] vs master[M-1].

    Args:
        target_month: The month being predicted.
        target_value: The actual target value (used simply for tracking).
        cutoff_date: The strict barrier preventing future data leakage (usually release date).
        snapshot_date: The date of the data snapshot being used.
        target_type: 'nsa' or 'sa'.
        release_type: 'first' or 'last'.
        target_source: 'first_release' or 'revised'.
        nsa_lags_lookup: Pre-computed dictionary of NSA target lags.
        sa_lags_lookup: Pre-computed dictionary of SA target lags.

    Returns:
        Tuple of (feature dict, target value). None if snapshot missing.
    """
    # Initialize dictionary for features
    features = {'ds': target_month}

    # 1. Add Calendar Features
    cal_features = get_calendar_features_dict(target_month)
    features.update(cal_features)

    # 2. Add Lagged Target Features (NSA & SA) — O(1) dict lookup
    features.update(nsa_lags_lookup.get(target_month, {}))
    features.update(sa_lags_lookup.get(target_month, {}))

    # 3. Load pre-merged, pre-selected master snapshot (contains ALL sources)
    snapshot_df = load_master_snapshot(snapshot_date, target_type=target_type,
                                       target_source=target_source, use_cache=False)

    if snapshot_df is None or snapshot_df.empty:
        return None, None

    features_wide = pivot_snapshot_to_wide(snapshot_df, target_month, cutoff_date=cutoff_date)
    if not features_wide.empty:
        features.update(features_wide.iloc[0].to_dict())

    # 4. Compute Revisions: compare master[M] vs master[M-1] for the PREVIOUS month
    prev_month_target = target_month - pd.DateOffset(months=1)
    prev_snapshot_date = prev_month_target + pd.offsets.MonthEnd(0)

    prev_snapshot = load_master_snapshot(prev_snapshot_date, target_type=target_type,
                                         target_source=target_source, use_cache=False)

    if prev_snapshot is not None and not prev_snapshot.empty:
        view_curr = pivot_snapshot_to_wide(snapshot_df, prev_month_target, cutoff_date=cutoff_date)
        view_prev = pivot_snapshot_to_wide(prev_snapshot, prev_month_target, cutoff_date=cutoff_date)
        revs = compute_revision_features(view_curr, view_prev, prefix='rev_master')
        if not revs.empty:
            features.update(revs.iloc[0].to_dict())

    return features, target_value


from Train.feature_engineering import (
    add_calendar_features,
    get_calendar_features_dict,
)

from Train.revision_features import (
    get_revision_features_for_month,
    compute_revision_features,
)
from utils.transforms import winsorize_covid_period
from Train.hyperparameter_tuning import tune_hyperparameters

from Train.model import (
    get_lgbm_params,
    train_lightgbm_model,
    predict_with_intervals,
    save_model,
    load_model,
    calculate_sample_weights,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


def clean_features(X: pd.DataFrame, y: pd.Series, min_non_nan: int = 100) -> List[str]:
    """
    Basic feature cleaning: only drop columns with too few data points.

    Data is already cleaned upstream so no other filtering is needed.
    LightGBM handles NaN natively, so NaN ratio is irrelevant.

    Args:
        X: Feature DataFrame
        y: Target series (unused, kept for API consistency)
        min_non_nan: Minimum number of non-NaN values required to keep a column

    Returns:
        List of cleaned feature column names
    """
    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    # Drop columns with fewer than min_non_nan non-NaN data points
    non_nan_counts = X_work.notna().sum()
    sparse_cols = non_nan_counts[non_nan_counts < min_non_nan].index.tolist()
    X_work = X_work.drop(columns=sparse_cols)

    logger.info(f"Feature cleaning: dropped {len(sparse_cols)} sparse (<{min_non_nan} non-NaN), "
                f"{len(X_work.columns)} remaining")

    return list(X_work.columns)


def build_training_dataset(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset by loading pre-merged master snapshots.

    Master snapshots already contain ALL data sources (FRED employment + exogenous),
    pre-filtered to selected features. No separate FRED loading or feature selection needed.

    Uses the NFP release_date as the data cutoff (e.g., May 3rd for April NFP),
    matching inference behavior so the model learns from intra-month data.
    Includes lagged target features from BOTH NSA and SA data.

    Args:
        target_df: Target DataFrame with y_mom column (the prediction target)
        target_type: 'nsa' or 'sa' - determines which target we're predicting
        release_type: 'first' or 'last' - determines which release to use for lagged features
        target_source: 'first_release' or 'revised' - determines which master snapshot variant
        start_date: Start date for training data
        end_date: End date for training data
        show_progress: Whether to show progress logging

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    all_features = []
    all_targets = []
    valid_dates = []

    model_id = get_model_id(target_type, release_type)

    # Load both NSA and SA target data ONCE (cached) - use same release_type and target_source
    source_label = "revised" if target_source == "revised" else f"{release_type} release"
    logger.info(f"Loading NSA and SA {source_label} target data for feature engineering...")
    nsa_target_full = load_target_data('nsa', release_type=release_type, target_source=target_source)
    sa_target_full = load_target_data('sa', release_type=release_type, target_source=target_source)

    # Pre-compute ALL lagged target features vectorized (shift/rolling) instead
    # of per-worker filtering. Produces lightweight dicts for O(1) worker lookup.
    logger.info("Pre-computing vectorized lagged target features...")
    nsa_lags_lookup = batch_lagged_target_features(nsa_target_full, prefix='nfp_nsa')
    sa_lags_lookup = batch_lagged_target_features(sa_target_full, prefix='nfp_sa')

    # Filter target data by date range
    filtered_df = target_df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['ds'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['ds'] <= end_date]

    n_months = len(filtered_df)
    logger.info(f"Building features for {n_months} target months...")

    import time as _time
    _build_t0 = _time.time()

    # Build release_date lookup (vectorized — no iterrows)
    target_ref = nsa_target_full if target_type == 'nsa' else sa_target_full
    release_date_map = {}
    if 'release_date' in target_ref.columns:
        valid_mask = target_ref['release_date'].notna()
        release_date_map = dict(zip(
            target_ref.loc[valid_mask, 'ds'],
            target_ref.loc[valid_mask, 'release_date'],
        ))

    # Prepare arguments for parallel execution (vectorized — no iterrows)
    target_months = filtered_df['ds'].values
    target_values = filtered_df['y_mom'].values
    tasks = []
    for i in range(len(filtered_df)):
        tm = pd.Timestamp(target_months[i])
        tasks.append((
            tm, target_values[i],
            release_date_map.get(tm, tm),
            tm + pd.offsets.MonthEnd(0),
            target_type, release_type, target_source,
            nsa_lags_lookup, sa_lags_lookup,
        ))

    logger.info(f"Starting parallel feature engineering for {len(tasks)} months using all processors...")

    # Execute in parallel
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_process_single_month_task)(*args) for args in tasks
    )

    # Filter out None results (skipped months)
    valid_results = [r for r in results if r[0] is not None]

    if not valid_results:
         logger.warning("No valid training samples generated!")
         return pd.DataFrame(), pd.Series(dtype=float)

    all_features_dicts, all_targets_list = zip(*valid_results)

    # Create final lists for DataFrame construction
    all_features = list(all_features_dicts)
    all_targets = list(all_targets_list)
    valid_dates = [f['ds'] for f in all_features]

    if not all_features:
        logger.error("No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Combine all features efficiently
    X = pd.DataFrame(all_features)
    y = pd.Series(all_targets, name='y_mom')

    # Add date index for reference
    X['ds'] = valid_dates

    # NOTE: COVID winsorization moved to backtest loop (per-fold) to avoid future data leakage.

    # Replace inf with NaN (LightGBM handles NaN natively but not inf)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop duplicate columns (can occur from overlapping source data)
    dupes = X.columns.duplicated()
    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning(f"Dropping {n_dupes} duplicate columns")
        X = X.loc[:, ~dupes]

    logger.info(f"Built training dataset: {len(X)} samples, {len(X.columns)} total features "
                f"(master snapshot features are pre-selected in ETL)")

    return X, y


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_expanding_window_backtest(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    tune: bool = True,
) -> pd.DataFrame:
    """
    Run proper expanding window backtest with strictly NO TIME-TRAVEL VIOLATIONS.

    This is the core loop evaluating the model's true real-world point-in-time performance.
    Instead of standard K-Fold CV (which randomly shuffles data and predicts the past using the future),
    this loop marches chronologically forward one month at a time. It uses strictly the available data 
    as of that specific historical month to predict it, precisely mirroring a real-time trading environment.

    Critical Design Principles:
    1. LightGBM handles NaN natively - no imputation needed preventing forward-filling bias.
    2. Model is FULLY retrained from scratch at each step.
    3. No information from future time periods leaks into predictions.
    4. Features are pre-selected per target_type (NSA or SA).
    5. Hyperparameters tuned via inner CV every `TUNE_EVERY_N_MONTHS` months.

    Args:
        target_df: Target DataFrame with 'ds' and 'y_mom' columns
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        target_source: 'first_release' or 'revised'
        use_huber_loss: Whether to use Huber loss
        huber_delta: Huber delta parameter
        tune: If True, run Optuna hyperparameter tuning periodically

    Returns:
        Tuple of (results_df, X_full, y_full) where X_full and y_full are the
        pre-built feature matrix and target series (reusable for production model).
    """
    model_id = get_model_id(target_type, release_type, target_source)

    logger.info("=" * 60)
    logger.info(f"EXPANDING WINDOW BACKTEST [{model_id.upper()}] (No Time-Travel)")
    logger.info("=" * 60)

    # Determine backtest period
    backtest_start_idx = len(target_df) - BACKTEST_MONTHS
    backtest_months = target_df.iloc[backtest_start_idx:]['ds'].tolist()

    logger.info(f"Backtest period: {BACKTEST_MONTHS} months ({backtest_months[0].strftime('%Y-%m')} to {backtest_months[-1].strftime('%Y-%m')})")

    # Load target data for lagged features (cached) - use same release_type and target_source
    load_target_data('nsa', release_type=release_type, target_source=target_source)
    load_target_data('sa', release_type=release_type, target_source=target_source)

    # Build FULL feature dataset once
    logger.info("Building full feature dataset...")
    X_full, y_full = build_training_dataset(
        target_df, target_type=target_type, release_type=release_type,
        target_source=target_source,
        show_progress=False
    )

    if X_full.empty:
        logger.error("Failed to build training dataset")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    # Pre-compute indices for faster lookup
    date_to_idx = {d: i for i, d in enumerate(X_full['ds'])}

    # Store results
    results = []
    all_residuals = []

    # Static fallback params (used when tune=False)
    static_params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    # Run clean_features once on early data to get initial feature set
    # (will be refined at each step)
    cleaned_features = None
    tuned_params = None  # Cached tuned hyperparameters

    logger.info(f"Running {len(backtest_months)} predictions "
                f"({'with' if tune else 'without'} hyperparameter tuning)...")

    import time as _time
    _backtest_t0 = _time.time()

    for i, target_month in enumerate(backtest_months):
        # Get index of this target month in the full dataset
        target_idx = date_to_idx.get(target_month)
        if target_idx is None:
            continue

        # EXPANDING WINDOW: Training data is everything BEFORE the target month
        train_mask = X_full['ds'] < target_month
        train_idx = X_full[train_mask].index.tolist()

        if len(train_idx) < 24:  # Need at least 2 years of training data
            continue

        # Get training data (no future leakage)
        y_train = y_full.iloc[train_idx]

        # Filter out NaN targets from training data
        valid_train_mask = ~y_train.isna()
        train_idx_valid = [train_idx[j] for j in range(len(train_idx)) if valid_train_mask.iloc[j]]

        if len(train_idx_valid) < 24:
            continue

        # Get valid training data (LightGBM handles NaN natively)
        X_train_valid = X_full.iloc[train_idx_valid].copy()
        y_train_valid = y_train[valid_train_mask].copy()

        # Sort by date to ensure time-ordered train/val split
        sort_order = X_train_valid['ds'].argsort()
        X_train_valid = X_train_valid.iloc[sort_order].reset_index(drop=True)
        y_train_valid = y_train_valid.iloc[sort_order].reset_index(drop=True)

        # COVID winsorization on training data only (no future leakage)
        X_indexed = X_train_valid.set_index('ds')
        numeric_cols = X_indexed.select_dtypes(include=[np.number]).columns
        X_indexed[numeric_cols] = winsorize_covid_period(X_indexed[numeric_cols])
        y_indexed = pd.Series(y_train_valid.values, index=X_indexed.index, name='y_mom')
        y_train_valid = winsorize_covid_period(y_indexed).reset_index(drop=True)
        X_train_valid = X_indexed.reset_index(names='ds')

        # Basic feature cleaning (no feature selection gates)
        if cleaned_features is None or i % TUNE_EVERY_N_MONTHS == 0:
            cleaned_features = clean_features(X_train_valid, y_train_valid)

        feature_cols = [c for c in cleaned_features if c in X_train_valid.columns and c != 'ds']

        # Compute default sample weights for initial/static use
        # (Optuna will dynamically test different half-lifes if tuning)
        # Using 60.0 months (5 years) as a reasonable default baseline if not tuning
        default_half_life = 60.0
        weights = calculate_sample_weights(X_train_valid, target_month, default_half_life)

        # Prepare training data with cleaned features
        X_train_selected = X_train_valid[feature_cols]
        # X_train_selected contains NO 'ds' column to avoid issues in lgb.train
        X_train_valid_with_ds = X_train_valid.copy() # keep one with ds

        # Tune hyperparameters periodically (same cadence as feature cleaning)
        if tune and (tuned_params is None or i % TUNE_EVERY_N_MONTHS == 0):
            logger.info(f"[{i+1}/{len(backtest_months)}] Tuning hyperparameters on "
                        f"{len(X_train_selected)} samples, {len(feature_cols)} features...")
            
            # Note: We pass the valid DataFrame WITH 'ds' to hyperparameter tuning
            # because the inner CV needs to recompute weights per inner-fold using 'ds'
            X_train_for_tuning = X_train_valid_with_ds[['ds'] + feature_cols]
            
            tuned_params = tune_hyperparameters(
                X_train_for_tuning, y_train_valid, target_month=target_month,
                use_huber_loss=use_huber_loss,
            )

        params = tuned_params if tune and tuned_params is not None else static_params
        
        # Determine the final half_life_months to use for this expanding window iteration model
        final_half_life = params.get('half_life_months', default_half_life)
        
        # Recompute final training weights for the main model fit using the chosen half_life
        weights = calculate_sample_weights(X_train_valid_with_ds, target_month, final_half_life)

        # Train-validation split (data already sorted by date above)
        train_size = int(len(X_train_selected) * 0.85)
        X_tr = X_train_selected.iloc[:train_size]
        X_val = X_train_selected.iloc[train_size:]
        y_tr = y_train_valid.iloc[:train_size]
        y_val = y_train_valid.iloc[train_size:]
        weights_tr = weights[:train_size]

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=weights_tr, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0)
        ]

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # OOS residuals accumulated after prediction below (not in-sample)

        # PREDICTION: Get features for target month
        X_pred = X_full.iloc[[target_idx]].copy()

        X_pred = X_pred[feature_cols]

        # Make prediction
        prediction = model.predict(X_pred)[0]
        actual = y_full.iloc[target_idx]

        # Calculate prediction intervals
        if len(all_residuals) > 10:
            residual_array = np.array(all_residuals[-36:])
            lower_50 = prediction + np.percentile(residual_array, 25)
            upper_50 = prediction + np.percentile(residual_array, 75)
            lower_80 = prediction + np.percentile(residual_array, 10)
            upper_80 = prediction + np.percentile(residual_array, 90)
            lower_95 = prediction + np.percentile(residual_array, 2.5)
            upper_95 = prediction + np.percentile(residual_array, 97.5)
        else:
            # Not enough OOS residuals yet; use validation-set residuals as initial estimate
            val_preds = model.predict(X_val)
            val_residuals = (y_val.values - val_preds).tolist()
            std_est = np.std(val_residuals) if val_residuals else 50
            lower_50, upper_50 = prediction - 0.67*std_est, prediction + 0.67*std_est
            lower_80, upper_80 = prediction - 1.28*std_est, prediction + 1.28*std_est
            lower_95, upper_95 = prediction - 1.96*std_est, prediction + 1.96*std_est

        # Handle NaN actuals (future predictions)
        is_future = pd.isna(actual)
        error = np.nan if is_future else actual - prediction

        # Accumulate OOS residuals for calibrated prediction intervals
        if not is_future:
            all_residuals.append(error)
        in_50 = np.nan if is_future else (lower_50 <= actual <= upper_50)
        in_80 = np.nan if is_future else (lower_80 <= actual <= upper_80)
        in_95 = np.nan if is_future else (lower_95 <= actual <= upper_95)

        results.append({
            'ds': target_month,
            'actual': actual,
            'predicted': prediction,
            'error': error,
            'lower_50': lower_50,
            'upper_50': upper_50,
            'lower_80': lower_80,
            'upper_80': upper_80,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'in_50_interval': in_50,
            'in_80_interval': in_80,
            'in_95_interval': in_95,
            'n_train_samples': len(train_idx_valid),
            'n_features': len(feature_cols)
        })

        # Progress logging — every prediction with elapsed/ETA
        _elapsed = _time.time() - _backtest_t0
        _avg_per_step = _elapsed / (i + 1)
        _eta = _avg_per_step * (len(backtest_months) - i - 1)
        _eta_str = f"{_eta/60:.1f}m" if _eta >= 60 else f"{_eta:.0f}s"
        _elapsed_str = f"{_elapsed/60:.1f}m" if _elapsed >= 60 else f"{_elapsed:.0f}s"

        if is_future:
            logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                        f"Pred={prediction:.0f} (FUTURE) | "
                        f"train={len(train_idx_valid)}, feats={len(feature_cols)} | "
                        f"elapsed={_elapsed_str}, ETA={_eta_str}")
        else:
            logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                        f"Actual={actual:.0f}, Pred={prediction:.0f}, Err={error:+.0f} | "
                        f"train={len(train_idx_valid)}, feats={len(feature_cols)} | "
                        f"elapsed={_elapsed_str}, ETA={_eta_str}")

    results_df = pd.DataFrame(results)

    # Log summary statistics
    if not results_df.empty:
        backtest_rows = results_df[~results_df['error'].isna()]
        future_rows = results_df[results_df['error'].isna()]

        if not backtest_rows.empty:
            rmse = np.sqrt(np.mean(backtest_rows['error'] ** 2))
            mae = np.mean(np.abs(backtest_rows['error']))

            logger.info("\n" + "=" * 60)
            logger.info("EXPANDING WINDOW BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Predictions: {len(backtest_rows)} backtest, {len(future_rows)} future")
            logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            logger.info(f"Coverage: 50%={backtest_rows['in_50_interval'].mean()*100:.1f}%, "
                       f"80%={backtest_rows['in_80_interval'].mean()*100:.1f}%, "
                       f"95%={backtest_rows['in_95_interval'].mean()*100:.1f}%")

        if not future_rows.empty:
            logger.info("\nFuture Predictions:")
            for _, row in future_rows.iterrows():
                logger.info(f"  {row['ds'].strftime('%Y-%m')}: {row['predicted']:.0f} [{row['lower_80']:.0f}, {row['upper_80']:.0f}]")

    return results_df, X_full, y_full


def train_and_evaluate(
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    tune: bool = True,
):
    """
    Main training and evaluation function using EXPANDING WINDOW methodology.

    This function ensures NO TIME-TRAVEL VIOLATIONS:
    1. LightGBM handles NaN natively - no imputation needed
    2. Model training uses only data available at each prediction time
    3. Hyperparameters tuned via inner CV (no future leakage)

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised
        target_source: 'first_release' or 'revised' (from M+1 FRED snapshot)
        use_huber_loss: If True, use Huber loss function
        huber_delta: Huber delta parameter
        tune: If True, run Optuna hyperparameter tuning
    """
    model_id = get_model_id(target_type, release_type, target_source)

    logger.info("=" * 60)
    logger.info(f"LightGBM NFP Prediction Model - Training [{model_id.upper()}]")
    logger.info("=" * 60)
    logger.info(f"Using EXPANDING WINDOW methodology (no time-travel, tune={tune})")

    # Load target data
    target_df = load_target_data(target_type=target_type, release_type=release_type,
                                 target_source=target_source)

    # Determine date ranges
    train_end = target_df['ds'].max() - pd.DateOffset(months=BACKTEST_MONTHS)

    logger.info(f"\nInitial training period: {target_df['ds'].min()} to {train_end}")
    logger.info(f"Backtest period: {train_end} to {target_df['ds'].max()}")

    # Run expanding window backtest (also returns pre-built X_full, y_full)
    backtest_results, X_full, y_full = run_expanding_window_backtest(
        target_df=target_df,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        tune=tune,
    )

    if backtest_results.empty:
        logger.error("Backtest produced no results")
        return

    # Train final production model on ALL data (for future predictions)
    # Reuses X_full/y_full from backtest to avoid redundant ~5min feature build
    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING FINAL PRODUCTION MODEL [{model_id.upper()}]")
    logger.info("=" * 60)

    # Filter out NaN targets for final model training
    valid_mask = ~y_full.isna()
    X_full_valid = X_full[valid_mask].copy()
    y_full_valid = y_full[valid_mask].copy()

    logger.info(f"Total observations: {len(X_full)}, Valid for training: {len(X_full_valid)}")

    # Basic feature cleaning (no feature selection)
    cleaned_feature_cols = clean_features(X_full_valid, y_full_valid)
    feature_cols = cleaned_feature_cols

    X_train = X_full_valid[['ds'] + [c for c in feature_cols if c in X_full_valid.columns]].copy()
    logger.info(f"Training final model with {len(feature_cols)} features")

    # Tune hyperparameters on all available data for the production model
    final_params = None
    # Final target_month anchor for the production model is simply the most recent date available
    # or equivalently, a future date where we plan to predict. But to be safe, we anchor it to 
    # the max date in the data (most recent NFP print).
    final_target_month = pd.to_datetime(X_train['ds'].max())

    if tune:
        logger.info("Tuning hyperparameters for final production model...")
        feature_only_cols = [c for c in feature_cols if c in X_train.columns and c != 'ds']
        
        # Pass the DF WITH 'ds' so inner tuning folds can manage their own point-in-time weights
        final_params = tune_hyperparameters(
            X_train[['ds'] + feature_only_cols], y_full_valid, target_month=final_target_month,
            use_huber_loss=use_huber_loss,
        )
        
        final_half_life = final_params.get('half_life_months', 60.0)
        # Reattach the targets and half_life so train_lightgbm_model can recompute internally
        final_params['target_month'] = final_target_month
        final_params['half_life_months'] = final_half_life
    else:
        final_params = {'target_month': final_target_month, 'half_life_months': 60.0}

    # Train final model
    model, importance, residuals = train_lightgbm_model(
        X_train,
        y_full_valid,
        n_splits=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        params_override=final_params,
    )

    # Save production model
    save_model(
        model,
        feature_cols,
        residuals,
        importance,
        target_type=target_type,
        release_type=release_type,
    )

    return model, feature_cols, residuals, backtest_results, X_full, y_full


def predict_nfp_mom(
    target_month: pd.Timestamp,
    model: Optional[lgb.Booster] = None,
    metadata: Optional[Dict] = None,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
) -> Dict:
    """
    Make NFP MoM prediction for a specific month.

    Uses snapshot of month M to predict month M's MoM change.

    Args:
        target_month: Month to predict (format: YYYY-MM-01 or YYYY-MM-DD)
        model: Optional pre-loaded model. If None, loads from disk.
        metadata: Optional pre-loaded metadata. If None, loads from disk.
        target_type: 'nsa' or 'sa' - determines which model to load
        release_type: 'first' or 'last' - determines which release model to load
        target_source: 'first_release' or 'revised' - determines which master snapshot variant

    Returns:
        Dictionary with prediction, intervals, and metadata
    """
    model_id = get_model_id(target_type, release_type, target_source)

    # Normalize target_month to first of month
    target_month = pd.Timestamp(target_month).replace(day=1)

    # Load model if not provided
    if model is None or metadata is None:
        model, metadata = load_model(target_type=target_type, release_type=release_type)

    feature_cols = metadata['feature_cols']
    residuals = metadata['residuals']

    # Get snapshot for this month (single pre-merged, pre-selected master snapshot)
    snapshot_date = target_month + pd.offsets.MonthEnd(0)
    snapshot_df = load_master_snapshot(
        snapshot_date, target_type=target_type, target_source=target_source
    )

    if snapshot_df is None or snapshot_df.empty:
        raise FileNotFoundError(f"No master snapshot available for {snapshot_date}")

    # Load target data for lagged features - use same release_type
    nsa_target_full = load_target_data('nsa', release_type=release_type)
    sa_target_full = load_target_data('sa', release_type=release_type)

    # Use NFP release date as strict cutoff when available
    target_ref = nsa_target_full if target_type == 'nsa' else sa_target_full
    cutoff_date = target_month
    if 'release_date' in target_ref.columns:
        match = target_ref[target_ref['ds'] == target_month]
        if not match.empty:
            rd = match['release_date'].iloc[0]
            if pd.notna(rd):
                cutoff_date = rd

    # Create features from master snapshot (already contains all data sources)
    features = pivot_snapshot_to_wide(snapshot_df, target_month, cutoff_date=cutoff_date)

    if features.empty:
        raise ValueError(f"Could not create features for {target_month}")

    features = add_calendar_features(features, target_month)

    # Add lagged target features - use same release_type
    nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
    sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')
    for k, v in {**nsa_target_features, **sa_target_features}.items():
        features[k] = v

    # Add cross-snapshot revision features (master[M] vs master[M-1])
    prev_month = target_month - pd.DateOffset(months=1)
    target_ref_df = nsa_target_full if target_type == 'nsa' else sa_target_full
    prev_row = target_ref_df[target_ref_df['ds'] == prev_month]
    if not prev_row.empty and 'release_date' in prev_row.columns:
        prev_release = prev_row.iloc[0]['release_date']
        prev_cutoff = prev_release if pd.notna(prev_release) else prev_month
    else:
        prev_cutoff = prev_month

    revision_feats = get_revision_features_for_month(
        target_month, prev_cutoff,
        target_type=target_type, target_source=target_source,
    )
    if not revision_feats.empty:
        for col in revision_feats.columns:
            features[col] = revision_feats[col].iloc[0]

    # Make prediction with intervals
    pred_result = predict_with_intervals(model, features, residuals, feature_cols)

    return {
        'target_month': target_month,
        'prediction': pred_result['prediction'],
        'intervals': pred_result['intervals'],
        'std': pred_result['std'],
        'mean_residual_bias': pred_result['mean_residual_bias'],
        'features_used': len(feature_cols),
        'target_type': target_type,
        'release_type': release_type,
        'model_id': model_id
    }


def convert_mom_to_level(
    mom_prediction: float,
    previous_level: float
) -> float:
    """Convert MoM change prediction to level prediction."""
    return previous_level + mom_prediction


def get_latest_prediction(target_type: str = 'nsa', release_type: str = 'first') -> Dict:
    """Get prediction for the most recent available month."""
    model_id = get_model_id(target_type, release_type)

    # Find latest snapshot available
    target_df = load_target_data(target_type=target_type, release_type=release_type)
    latest_target = target_df['ds'].max()

    logger.info(f"Making {model_id.upper()} prediction for latest available month: {latest_target}")

    return predict_nfp_mom(latest_target, target_type=target_type, release_type=release_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='LightGBM NFP Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model (default: nsa_first)
  python Train/train_lightgbm_nfp.py --train

  # Train with specific target/release
  python Train/train_lightgbm_nfp.py --train --target nsa --release first

  # Predict for a specific month
  python Train/train_lightgbm_nfp.py --predict 2024-12 --target nsa

  # Get latest prediction
  python Train/train_lightgbm_nfp.py --latest --target nsa
        """
    )
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', type=str, help='Predict for a specific month (YYYY-MM)')
    parser.add_argument('--latest', action='store_true', help='Predict for latest available month')
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help='Target type: nsa (non-seasonally adjusted) or sa (seasonally adjusted)')
    parser.add_argument('--release', type=str, default='first', choices=['first', 'last'],
                        help='Release type: first (initial) or last (final revised)')
    parser.add_argument('--no-huber-loss', action='store_true',
                        help='Disable Huber loss (uses MSE instead). Huber is enabled by default for outlier robustness.')
    parser.add_argument('--huber-delta', type=float, default=HUBER_DELTA,
                        help=f'Huber delta parameter (default: {HUBER_DELTA}). Lower = more robust to outliers.')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip Optuna hyperparameter tuning (use static defaults). Faster for debugging.')
    parser.add_argument('--revised', action='store_true',
                        help='Train on revised MoM target (from M+1 FRED snapshot instead of first release)')

    args = parser.parse_args()

    # Convert --no-* flags to booleans
    use_huber_loss = not args.no_huber_loss
    tune = not args.no_tune
    target_source = 'revised' if args.revised else 'first_release'

    model_id = get_model_id(args.target, args.release, target_source)

    if args.train:
        result = train_and_evaluate(
            target_type=args.target,
            release_type=args.release,
            target_source=target_source,
            use_huber_loss=use_huber_loss,
            huber_delta=args.huber_delta,
            tune=tune,
        )

        if result is not None:
            model, feature_cols, residuals, backtest_results, X_full, y_full = result
            metadata = {
                'feature_cols': feature_cols,
                'importance': dict(zip(feature_cols, model.feature_importance(importance_type='gain'))),
            }

            # If training NSA (default), also train SA and generate combined output
            if args.target == 'nsa':
                logger.info("\nNow training SA model to generate combined output...")
                sa_result = train_and_evaluate(
                    target_type='sa',
                    release_type=args.release,
                    target_source=target_source,
                    use_huber_loss=use_huber_loss,
                    huber_delta=args.huber_delta,
                    tune=tune,
                )
                if sa_result is not None:
                    sa_model, sa_feature_cols, sa_residuals, sa_backtest, sa_X_full, sa_y_full = sa_result
                    sa_metadata = {
                        'feature_cols': sa_feature_cols,
                        'importance': dict(zip(sa_feature_cols, sa_model.feature_importance(importance_type='gain'))),
                    }

                    from Train.Output_code.generate_output import generate_all_output
                    output_suffix = '_revised' if target_source == 'revised' else ''
                    generate_all_output(
                        nsa_results=backtest_results,
                        sa_results=sa_backtest,
                        nsa_model=model,
                        sa_model=sa_model,
                        nsa_metadata=metadata,
                        sa_metadata=sa_metadata,
                        nsa_X_full=X_full,
                        sa_X_full=sa_X_full,
                        nsa_y_full=y_full,
                        sa_y_full=sa_y_full,
                        nsa_residuals=residuals,
                        sa_residuals=sa_residuals,
                        suffix=output_suffix,
                    )

    elif args.predict:
        target_month = pd.Timestamp(args.predict + '-01')
        result = predict_nfp_mom(
            target_month,
            target_type=args.target,
            release_type=args.release
        )
        print(f"\n{'='*60}")
        print(f"NFP {model_id.upper()} MoM Change Prediction for {target_month.strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")
        print(f"Std: {result['std']:,.0f}")
        print(f"Features Used: {result['features_used']}")

    elif args.latest:
        result = get_latest_prediction(target_type=args.target, release_type=args.release)
        print(f"\n{'='*60}")
        print(f"NFP {model_id.upper()} MoM Change Prediction for {result['target_month'].strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")

    else:
        # Default: train and evaluate with defaults (nsa_first)
        train_and_evaluate(target_source=target_source)
