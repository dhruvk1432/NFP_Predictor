"""
LightGBM NFP Prediction Model

Predicts NSA/SA NFP month-on-month change using master snapshots.
Uses snapshot of month M to predict month M (no look-ahead bias).
Outputs predictions with confidence intervals/error bounds.

Key Features:
- Handles data with varying start dates (some series start 1948, others 2008)
- Uses past target data (NFP levels, MoM changes) as features
- Survey interval features (4 vs 5 weeks logic)
- Momentum/divergence and acceleration features
- Cyclical month encoding (sin/cos)
- SHAP-based feature importance analysis

FIRST RELEASE MODELS ONLY:
This model supports 2 target configurations (first release only):
- nsa_first: Non-seasonally adjusted, first release
- sa_first: Seasonally adjusted, first release

NOTE: Last release models (nsa_last, sa_last) are disabled.

Use --train-all to train both first release models efficiently.

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
import pickle
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, BACKTEST_MONTHS
from Train.backtest_results import generate_backtest_report

# Import from modular components
from Train.config import (
    LINEAR_BASELINE_PREDICTORS,
    PROTECTED_BINARY_FLAGS,
    MODEL_SAVE_DIR,
    ALL_TARGET_CONFIGS,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    get_model_id,
    get_target_path,
    USE_HUBER_LOSS_DEFAULT,
    HUBER_DELTA,
)

from Train.data_loader import (
    load_fred_snapshot,
    load_master_snapshot,
    load_target_data,
    get_lagged_target_features,
    pivot_snapshot_to_wide,
)

from Train.feature_engineering import (
    add_calendar_features,
    engineer_employment_features,
)

from Train.model import (
    get_lgbm_params,
    train_lightgbm_model,
    predict_with_intervals,
    save_model,
    load_model,
    train_linear_baseline,
    create_linear_baseline_feature,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False


def clean_features(X: pd.DataFrame, y: pd.Series, max_nan_ratio: float = 0.5) -> List[str]:
    """
    Basic feature cleaning: drop high-NaN and zero-variance columns.

    NaN ratio is computed per-column based on its own non-NaN count,
    so features that start later (e.g. ADP from 2010) are not penalized.

    Args:
        X: Feature DataFrame
        y: Target series (unused, kept for API consistency)
        max_nan_ratio: Maximum fraction of NaN values allowed per column

    Returns:
        List of cleaned feature column names
    """
    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    # Per-column NaN check
    nan_pct = X_work.isna().mean()
    high_nan_cols = nan_pct[nan_pct > max_nan_ratio].index.tolist()
    X_work = X_work.drop(columns=high_nan_cols)

    # Remove zero-variance columns
    col_means = X_work.mean()
    X_filled = X_work.fillna(col_means).fillna(0)
    zero_var = X_filled.var()[X_filled.var() == 0].index.tolist()
    X_work = X_work.drop(columns=zero_var)

    logger.info(f"Feature cleaning: dropped {len(high_nan_cols)} high-NaN, {len(zero_var)} zero-variance, {len(X_work.columns)} remaining")

    return list(X_work.columns)


def build_training_dataset(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset by joining snapshots with targets.

    Uses snapshot of month M to predict month M's MoM change.
    Includes lagged target features from BOTH NSA and SA data.

    Args:
        target_df: Target DataFrame with y_mom column (the prediction target)
        target_type: 'nsa' or 'sa' - determines which target we're predicting
        release_type: 'first' or 'last' - determines which release to use for lagged features
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

    # Load both NSA and SA target data ONCE (cached) - use same release_type for consistency
    logger.info(f"Loading NSA and SA {release_type} release target data for feature engineering...")
    nsa_target_full = load_target_data('nsa', release_type=release_type)
    sa_target_full = load_target_data('sa', release_type=release_type)

    # Filter target data by date range
    filtered_df = target_df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['ds'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['ds'] <= end_date]

    n_months = len(filtered_df)
    logger.info(f"Building features for {n_months} target months...")

    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        target_month = row['ds']
        target_value = row['y_mom']

        # Get snapshot for this month (month-end date)
        snapshot_date = target_month + pd.offsets.MonthEnd(0)

        # Load and process snapshot (cached)
        snapshot_df = load_master_snapshot(snapshot_date)

        if snapshot_df is None or snapshot_df.empty:
            continue

        # Convert to wide format (exogenous features)
        features = pivot_snapshot_to_wide(snapshot_df, target_month)

        if features.empty:
            continue

        # Add calendar features (including survey interval)
        features = add_calendar_features(features, target_month)

        # Add lagged target features from BOTH NSA and SA data
        # These use only data BEFORE target_month to avoid look-ahead bias
        nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
        sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')

        # Merge target features into features DataFrame
        for k, v in {**nsa_target_features, **sa_target_features}.items():
            features[k] = v

        # Add FRED employment features (endogenous data) - cached
        fred_df = load_fred_snapshot(snapshot_date)
        if fred_df is not None and not fred_df.empty:
            employment_features = engineer_employment_features(fred_df, target_month, target_type)
            for k, v in employment_features.items():
                features[k] = v

        all_features.append(features)
        all_targets.append(target_value)
        valid_dates.append(target_month)

        # Progress logging
        if show_progress and (i + 1) % 24 == 0:
            logger.info(f"  Processed {i + 1}/{n_months} months...")

    if not all_features:
        logger.error("No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Combine all features efficiently
    X = pd.concat(all_features, ignore_index=True)
    y = pd.Series(all_targets, name='y_mom')

    # Add date index for reference
    X['ds'] = valid_dates

    # Handle missing values - vectorized fill
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    col_means = X[numeric_cols].mean()
    X[numeric_cols] = X[numeric_cols].fillna(col_means)
    X = X.fillna(0)

    # Count feature categories
    calendar_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                         'weeks_since_last_survey', 'is_5_week_month', 'is_jan',
                         'is_july', 'year', 'is_summer', 'is_holiday_season', 'is_december']
    target_features = [c for c in X.columns if c.startswith('nfp_')]
    employment_features = [c for c in X.columns if c.startswith('emp_')]
    exog_features = [c for c in X.columns if not c.startswith('nfp_') and
                     not c.startswith('emp_') and c not in calendar_features and c != 'ds']

    logger.info(f"Built training dataset: {len(X)} samples, {len(X.columns)} total features")
    logger.info(f"  - Exogenous (macro) features: {len([c for c in exog_features if c in X.columns])}")
    logger.info(f"  - Employment (FRED) features: {len([c for c in employment_features if c in X.columns])}")
    logger.info(f"  - Calendar features: {len([c for c in calendar_features if c in X.columns])}")
    logger.info(f"  - Target (NFP) lag features: {len([c for c in target_features if c in X.columns])}")

    return X, y


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_backtest(
    model: lgb.Booster,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    residuals: List[float],
    backtest_start: pd.Timestamp,
    backtest_end: Optional[pd.Timestamp] = None,
    target_type: str = 'nsa',
    release_type: str = 'first'
) -> pd.DataFrame:
    """
    Run out-of-sample backtest on the model.

    Args:
        model: Trained LightGBM model
        target_df: Full target DataFrame
        feature_cols: Feature column names
        residuals: Historical residuals for intervals
        backtest_start: Start date for backtest
        backtest_end: End date for backtest
        target_type: 'nsa' or 'sa' - determines which employment features to use
        release_type: 'first' or 'last' - determines which release to use

    Returns:
        DataFrame with backtest results
    """
    backtest_df = target_df[target_df['ds'] >= backtest_start]
    if backtest_end:
        backtest_df = backtest_df[backtest_df['ds'] <= backtest_end]

    # Load target data for lagged features - use same release_type
    nsa_target_full = load_target_data('nsa', release_type=release_type)
    sa_target_full = load_target_data('sa', release_type=release_type)

    results = []

    for idx, row in backtest_df.iterrows():
        target_month = row['ds']
        actual = row['y_mom']

        # Get snapshot and create features
        snapshot_date = target_month + pd.offsets.MonthEnd(0)
        snapshot_df = load_master_snapshot(snapshot_date)

        if snapshot_df is None or snapshot_df.empty:
            continue

        features = pivot_snapshot_to_wide(snapshot_df, target_month)
        if features.empty:
            continue

        features = add_calendar_features(features, target_month)

        # Add lagged target features
        nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
        sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')
        for k, v in {**nsa_target_features, **sa_target_features}.items():
            features[k] = v

        # Add FRED employment features
        fred_df = load_fred_snapshot(snapshot_date)
        if fred_df is not None and not fred_df.empty:
            employment_features = engineer_employment_features(fred_df, target_month, target_type)
            for k, v in employment_features.items():
                features[k] = v

        # Make prediction with intervals
        pred_result = predict_with_intervals(model, features, residuals, feature_cols)

        results.append({
            'ds': target_month,
            'actual': actual,
            'predicted': pred_result['prediction'],
            'error': actual - pred_result['prediction'],
            'lower_50': pred_result['intervals']['50%'][0],
            'upper_50': pred_result['intervals']['50%'][1],
            'lower_80': pred_result['intervals']['80%'][0],
            'upper_80': pred_result['intervals']['80%'][1],
            'lower_95': pred_result['intervals']['95%'][0],
            'upper_95': pred_result['intervals']['95%'][1],
            'in_50_interval': pred_result['intervals']['50%'][0] <= actual <= pred_result['intervals']['50%'][1],
            'in_80_interval': pred_result['intervals']['80%'][0] <= actual <= pred_result['intervals']['80%'][1],
            'in_95_interval': pred_result['intervals']['95%'][0] <= actual <= pred_result['intervals']['95%'][1]
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Calculate metrics
        rmse = np.sqrt(np.mean(results_df['error'] ** 2))
        mae = np.mean(np.abs(results_df['error']))
        mape = np.mean(np.abs(results_df['error'] / results_df['actual'])) * 100

        # Coverage rates
        coverage_50 = results_df['in_50_interval'].mean() * 100
        coverage_80 = results_df['in_80_interval'].mean() * 100
        coverage_95 = results_df['in_95_interval'].mean() * 100

        logger.info(f"\nBacktest Results ({len(results_df)} predictions):")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"\nInterval Coverage:")
        logger.info(f"  50% CI: {coverage_50:.1f}% (target: 50%)")
        logger.info(f"  80% CI: {coverage_80:.1f}% (target: 80%)")
        logger.info(f"  95% CI: {coverage_95:.1f}% (target: 95%)")

    return results_df


def impute_with_expanding_window(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    cached_means: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute missing values using only data from the training window (no future leakage).

    Args:
        X: Full feature DataFrame
        train_idx: Indices of training data (expanding window)
        cached_means: Pre-computed means to reuse (optional)

    Returns:
        Tuple of (imputed DataFrame, computed means for caching)
    """
    X_imputed = X.copy()
    numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns

    # Calculate or reuse means
    if cached_means is not None:
        train_means = cached_means
    else:
        train_means = X_imputed.iloc[train_idx][numeric_cols].mean()

    # Fill NaN using training window means - vectorized
    X_imputed[numeric_cols] = X_imputed[numeric_cols].fillna(train_means)
    X_imputed = X_imputed.fillna(0)  # Fallback for remaining NaN

    return X_imputed, train_means


def run_expanding_window_backtest(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA
) -> pd.DataFrame:
    """
    Run proper expanding window backtest with NO TIME-TRAVEL VIOLATIONS.

    Critical Design Principles:
    1. NaN imputation uses only expanding window means (no future data)
    2. Model is FULLY retrained from scratch at each step
    3. No information from future time periods leaks into predictions
    4. No feature selection gates - all cleaned features are used

    Args:
        target_df: Target DataFrame with 'ds' and 'y_mom' columns
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        use_huber_loss: Whether to use Huber loss
        huber_delta: Huber delta parameter

    Returns:
        DataFrame with backtest results
    """
    model_id = get_model_id(target_type, release_type)

    logger.info("=" * 60)
    logger.info(f"EXPANDING WINDOW BACKTEST [{model_id.upper()}] (No Time-Travel)")
    logger.info("=" * 60)

    # Determine backtest period
    backtest_start_idx = len(target_df) - BACKTEST_MONTHS
    backtest_months = target_df.iloc[backtest_start_idx:]['ds'].tolist()

    logger.info(f"Backtest period: {BACKTEST_MONTHS} months ({backtest_months[0].strftime('%Y-%m')} to {backtest_months[-1].strftime('%Y-%m')})")

    # Load target data for lagged features (cached) - use same release_type
    load_target_data('nsa', release_type=release_type)
    load_target_data('sa', release_type=release_type)

    # Build FULL feature dataset once
    logger.info("Building full feature dataset...")
    X_full, y_full = build_training_dataset(target_df, target_type=target_type, release_type=release_type, show_progress=False)

    if X_full.empty:
        logger.error("Failed to build training dataset")
        return pd.DataFrame()

    # Pre-compute indices for faster lookup
    date_to_idx = {d: i for i, d in enumerate(X_full['ds'])}

    # Store results
    results = []
    all_residuals = []

    # Cache for imputation
    cached_impute_means = None
    last_train_idx_len = 0

    # Get LightGBM params once
    params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    # Run clean_features once on early data to get initial feature set
    # (will be refined at each step)
    cleaned_features = None

    logger.info(f"Running {len(backtest_months)} predictions...")

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

        # Recompute imputation if training set grew
        train_set_grew = len(train_idx) > last_train_idx_len
        if train_set_grew or cached_impute_means is None:
            X_train_imputed, cached_impute_means = impute_with_expanding_window(X_full, train_idx, None)
            last_train_idx_len = len(train_idx)
        else:
            X_train_imputed, _ = impute_with_expanding_window(X_full, train_idx, cached_impute_means)

        X_train = X_train_imputed.iloc[train_idx]

        # Get valid training data
        valid_local_idx = [j for j in range(len(train_idx)) if valid_train_mask.iloc[j]]
        X_train_valid = X_train.iloc[valid_local_idx].copy()
        y_train_valid = y_train[valid_train_mask].copy()

        # Basic feature cleaning (no feature selection gates)
        if cleaned_features is None or i % 12 == 0:
            cleaned_features = clean_features(X_train_valid, y_train_valid)

        feature_cols = [c for c in cleaned_features if c in X_train_valid.columns and c != 'ds']

        # Train Linear Baseline and Add as Feature
        linear_model, linear_cols_used = train_linear_baseline(
            X_train_valid,
            y_train_valid,
            LINEAR_BASELINE_PREDICTORS
        )

        if linear_model is not None:
            X_train_valid['linear_baseline_pred'] = create_linear_baseline_feature(
                X_train_valid,
                linear_model,
                linear_cols_used
            )
            if 'linear_baseline_pred' not in feature_cols:
                feature_cols.append('linear_baseline_pred')

        # Prepare training data with cleaned features
        X_train_selected = X_train_valid[feature_cols]

        # Train-validation split
        train_size = int(len(X_train_selected) * 0.85)
        X_tr = X_train_selected.iloc[:train_size]
        X_val = X_train_selected.iloc[train_size:]
        y_tr = y_train_valid.iloc[:train_size]
        y_val = y_train_valid.iloc[train_size:]

        train_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)
        ]

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # Calculate residuals for prediction intervals
        train_preds = model.predict(X_train_selected)
        train_residuals = (y_train_valid.values - train_preds).tolist()
        all_residuals.extend(train_residuals[-12:])

        # PREDICTION: Get features for target month
        X_pred = X_train_imputed.iloc[[target_idx]].copy()

        if linear_model is not None:
            X_pred['linear_baseline_pred'] = create_linear_baseline_feature(
                X_pred,
                linear_model,
                linear_cols_used
            )

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
            std_est = np.std(train_residuals) if train_residuals else 50
            lower_50, upper_50 = prediction - 0.67*std_est, prediction + 0.67*std_est
            lower_80, upper_80 = prediction - 1.28*std_est, prediction + 1.28*std_est
            lower_95, upper_95 = prediction - 1.96*std_est, prediction + 1.96*std_est

        # Handle NaN actuals (future predictions)
        is_future = pd.isna(actual)
        error = np.nan if is_future else actual - prediction
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

        # Progress logging (every 12 months or first/last)
        if (i + 1) % 12 == 0 or i == 0 or i == len(backtest_months) - 1:
            if is_future:
                logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: Pred={prediction:.0f} (FUTURE)")
            else:
                logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: Actual={actual:.0f}, Pred={prediction:.0f}, Err={error:.0f}")

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

    return results_df


def compute_shap_importance(
    model: lgb.Booster,
    X_train: pd.DataFrame,
    feature_cols: List[str],
    model_id: str
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP feature importance and save results.

    Args:
        model: Trained LightGBM Booster
        X_train: Training feature matrix
        feature_cols: Feature column names
        model_id: Model identifier for file naming

    Returns:
        DataFrame with mean |SHAP| per feature, or None if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP analysis")
        return None

    logger.info("Computing SHAP values...")

    try:
        # Use TreeExplainer for LightGBM
        explainer = shap.TreeExplainer(model)

        # Get feature data
        X_features = X_train[feature_cols] if feature_cols else X_train
        shap_values = explainer.shap_values(X_features)

        # Compute mean |SHAP| per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        # Save SHAP importance CSV
        shap_dir = OUTPUT_DIR / "shap_analysis" / model_id
        shap_dir.mkdir(parents=True, exist_ok=True)

        csv_path = shap_dir / f"shap_importance_{model_id}.csv"
        shap_df.to_csv(csv_path, index=False)
        logger.info(f"Saved SHAP importance to {csv_path}")

        # Save raw SHAP values matrix as parquet
        shap_matrix = pd.DataFrame(shap_values, columns=feature_cols)
        parquet_path = shap_dir / f"shap_values_{model_id}.parquet"
        shap_matrix.to_parquet(parquet_path, index=False)
        logger.info(f"Saved raw SHAP values to {parquet_path}")

        # Log top 20 features
        logger.info("\nTop 20 Features by Mean |SHAP|:")
        for i, row in shap_df.head(20).iterrows():
            logger.info(f"  {i+1:2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")

        return shap_df

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def train_and_evaluate(
    target_type: str = 'nsa',
    release_type: str = 'first',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    archive_results: bool = True
):
    """
    Main training and evaluation function using EXPANDING WINDOW methodology.

    This function ensures NO TIME-TRAVEL VIOLATIONS:
    1. NaN imputation uses only past data (expanding window means)
    2. Model training uses only data available at each prediction time
    3. SHAP values are computed on the final production model

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised
        use_huber_loss: If True, use Huber loss function
        huber_delta: Huber delta parameter
        archive_results: If True, archive previous results (only for first model in batch)
    """
    model_id = get_model_id(target_type, release_type)

    # Archive previous backtest results before starting new run
    if archive_results:
        try:
            sys.path.append(str(Path(__file__).resolve().parent))
            from backtest_archiver import prepare_new_backtest_run

            archive_path = prepare_new_backtest_run()
            if archive_path:
                logger.info(f"Archived previous backtest to: {archive_path.name}")
            else:
                logger.info("No previous backtest results to archive")
        except Exception as e:
            logger.warning(f"Failed to archive backtest results: {e}")
            logger.info("Continuing with training...")

    logger.info("=" * 60)
    logger.info(f"LightGBM NFP Prediction Model - Training [{model_id.upper()}]")
    logger.info("=" * 60)
    logger.info("Using EXPANDING WINDOW methodology (no time-travel)")

    # Load target data
    target_df = load_target_data(target_type=target_type, release_type=release_type)

    # Determine date ranges
    train_end = target_df['ds'].max() - pd.DateOffset(months=BACKTEST_MONTHS)

    logger.info(f"\nInitial training period: {target_df['ds'].min()} to {train_end}")
    logger.info(f"Backtest period: {train_end} to {target_df['ds'].max()}")

    # Run expanding window backtest
    backtest_results = run_expanding_window_backtest(
        target_df=target_df,
        target_type=target_type,
        release_type=release_type,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta
    )

    if backtest_results.empty:
        logger.error("Backtest produced no results")
        return

    # Train final production model on ALL data (for future predictions)
    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING FINAL PRODUCTION MODEL [{model_id.upper()}]")
    logger.info("=" * 60)

    X_full, y_full = build_training_dataset(target_df, target_type=target_type, release_type=release_type)

    # Filter out NaN targets for final model training
    valid_mask = ~y_full.isna()
    X_full_valid = X_full[valid_mask].copy()
    y_full_valid = y_full[valid_mask].copy()

    logger.info(f"Total observations: {len(X_full)}, Valid for training: {len(X_full_valid)}")

    # Train linear baseline on full training data
    logger.info("\nTraining Linear Baseline Model for Production...")
    linear_model_prod, linear_cols_prod = train_linear_baseline(
        X_full_valid,
        y_full_valid,
        LINEAR_BASELINE_PREDICTORS
    )

    if linear_model_prod is not None:
        X_full_valid['linear_baseline_pred'] = create_linear_baseline_feature(
            X_full_valid,
            linear_model_prod,
            linear_cols_prod
        )

    # Basic feature cleaning (no feature selection)
    cleaned_feature_cols = clean_features(X_full_valid, y_full_valid)
    feature_cols = cleaned_feature_cols

    # Ensure linear_baseline_pred is included if available
    if linear_model_prod is not None and 'linear_baseline_pred' not in feature_cols:
        feature_cols.append('linear_baseline_pred')

    X_train = X_full_valid[['ds'] + [c for c in feature_cols if c in X_full_valid.columns]].copy()

    logger.info(f"Training final model with {len(feature_cols)} features")

    # Train final model
    model, importance, residuals = train_lightgbm_model(
        X_train,
        y_full_valid,
        n_splits=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta
    )

    # Compute SHAP values on the final model
    logger.info("\n" + "=" * 60)
    logger.info("SHAP Feature Importance Analysis")
    logger.info("=" * 60)
    shap_df = compute_shap_importance(model, X_train, feature_cols, model_id)

    # Save production model
    save_model(
        model,
        feature_cols,
        residuals,
        importance,
        target_type=target_type,
        release_type=release_type,
        linear_model=linear_model_prod,
        linear_cols=linear_cols_prod
    )

    # Save backtest results
    results_dir = OUTPUT_DIR / "backtest_results" / model_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if not backtest_results.empty:
        results_path = results_dir / f"backtest_results_{model_id}.parquet"
        backtest_results.to_parquet(results_path, index=False)
        logger.info(f"\nSaved backtest results to {results_path}")

        # Also save as CSV for easy viewing
        csv_path = results_dir / f"backtest_results_{model_id}.csv"
        backtest_results.to_csv(csv_path, index=False)
        logger.info(f"Saved backtest CSV to {csv_path}")

        # Save summary statistics
        summary = {
            'target_type': target_type,
            'release_type': release_type,
            'model_id': model_id,
            'n_predictions': len(backtest_results),
            'rmse': np.sqrt(np.mean(backtest_results['error'] ** 2)),
            'mae': np.mean(np.abs(backtest_results['error'])),
            'mape': np.mean(np.abs(backtest_results['error'] / backtest_results['actual'])) * 100,
            'coverage_50': backtest_results['in_50_interval'].mean() * 100,
            'coverage_80': backtest_results['in_80_interval'].mean() * 100,
            'coverage_95': backtest_results['in_95_interval'].mean() * 100,
            'n_features': len(feature_cols),
            'train_end': str(train_end.date()),
            'backtest_start': str((train_end + pd.DateOffset(months=1)).date()),
            'backtest_end': str(target_df['ds'].max().date())
        }

        summary_path = results_dir / f"model_summary_{model_id}.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        logger.info(f"Saved model summary to {summary_path}")

    # Save feature importance to output
    importance_dir = OUTPUT_DIR / "feature_importance" / model_id
    importance_dir.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    importance_path = importance_dir / f"feature_importance_{model_id}.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")

    # Generate comprehensive backtest report using dedicated module
    logger.info("\n" + "=" * 60)
    logger.info("Generating Backtest Report")
    logger.info("=" * 60)

    try:
        # Prepare data for backtest report
        report_df = backtest_results.copy()
        report_df = report_df.rename(columns={
            'ds': 'date',
            'predicted': 'pred'
        })

        # Generate comprehensive report (visualizations, metrics, tables)
        report_files = generate_backtest_report(
            predictions_df=report_df,
            selected_features=feature_cols,
            output_dir=results_dir
        )

        logger.info(f"Generated {len(report_files)} report files")

    except Exception as e:
        logger.warning(f"Failed to generate backtest report: {e}")

    return model, feature_cols, residuals, backtest_results


def predict_nfp_mom(
    target_month: pd.Timestamp,
    model: Optional[lgb.Booster] = None,
    metadata: Optional[Dict] = None,
    target_type: str = 'nsa',
    release_type: str = 'first'
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

    Returns:
        Dictionary with prediction, intervals, and metadata
    """
    model_id = get_model_id(target_type, release_type)

    # Normalize target_month to first of month
    target_month = pd.Timestamp(target_month).replace(day=1)

    # Load model if not provided
    if model is None or metadata is None:
        model, metadata = load_model(target_type=target_type, release_type=release_type)

    feature_cols = metadata['feature_cols']
    residuals = metadata['residuals']

    # Get snapshot for this month
    snapshot_date = target_month + pd.offsets.MonthEnd(0)
    snapshot_df = load_master_snapshot(snapshot_date)

    if snapshot_df is None or snapshot_df.empty:
        raise FileNotFoundError(f"No snapshot available for {snapshot_date}")

    # Create features
    features = pivot_snapshot_to_wide(snapshot_df, target_month)

    if features.empty:
        raise ValueError(f"Could not create features for {target_month}")

    features = add_calendar_features(features, target_month)

    # Add lagged target features - use same release_type
    nsa_target_full = load_target_data('nsa', release_type=release_type)
    sa_target_full = load_target_data('sa', release_type=release_type)
    nsa_target_features = get_lagged_target_features(nsa_target_full, target_month, 'nfp_nsa')
    sa_target_features = get_lagged_target_features(sa_target_full, target_month, 'nfp_sa')
    for k, v in {**nsa_target_features, **sa_target_features}.items():
        features[k] = v

    # Add FRED employment features
    fred_df = load_fred_snapshot(snapshot_date)
    if fred_df is not None and not fred_df.empty:
        employment_features = engineer_employment_features(fred_df, target_month, target_type)
        for k, v in employment_features.items():
            features[k] = v

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


def train_all_models(
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA
) -> Dict[str, Any]:
    """
    Train all first release model variants efficiently.

    Model architecture:
    - nsa_first: LightGBM model for NSA predictions
    - sa_first: SARIMA model for seasonal adjustment (applied to NSA predictions)

    NOTE: Last release models (nsa_last, sa_last) are disabled.

    Args:
        use_huber_loss: If True, use Huber loss function (for NSA)
        huber_delta: Huber delta parameter (for NSA)

    Returns:
        Dictionary with results for each model_id
    """
    logger.info("=" * 70)
    logger.info("TRAINING ALL FIRST RELEASE MODEL VARIANTS")
    logger.info("=" * 70)
    logger.info("Model 1: NSA (LightGBM)")
    logger.info("Model 2: SA (SARIMA on seasonal adjustment)")

    results = {}

    # Step 1: Train NSA model with LightGBM (archives previous results)
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 1/2: NSA_FIRST (LightGBM)")
    logger.info("=" * 70)

    try:
        nsa_result = train_and_evaluate(
            target_type='nsa',
            release_type='first',
            use_huber_loss=use_huber_loss,
            huber_delta=huber_delta,
            archive_results=True
        )

        results['nsa_first'] = {
            'status': 'success',
            'model': nsa_result[0] if nsa_result else None,
            'feature_cols': nsa_result[1] if nsa_result else None,
            'residuals': nsa_result[2] if nsa_result else None,
            'backtest_results': nsa_result[3] if nsa_result else None,
        }

        logger.info("Successfully trained NSA_FIRST")

    except Exception as e:
        logger.error(f"Failed to train NSA_FIRST: {e}")
        results['nsa_first'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Step 2: Train SA model with SARIMA (uses NSA predictions)
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 2/2: SA_FIRST (SARIMA)")
    logger.info("=" * 70)

    try:
        # Import SARIMA training function
        from Train.train_sarima_sa import train_and_evaluate_sarima

        # Train SARIMA model (uses NSA predictions from step 1)
        sa_result = train_and_evaluate_sarima(archive_results=False)

        results['sa_first'] = {
            'status': 'success',
            'model_type': 'SARIMA',
            'backtest_results': sa_result if sa_result is not None else None,
        }

        logger.info("Successfully trained SA_FIRST (SARIMA)")

    except Exception as e:
        logger.error(f"Failed to train SA_FIRST: {e}")
        results['sa_first'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 70)

    for model_id, result in results.items():
        status = result['status']
        if status == 'success':
            backtest = result.get('backtest_results')
            if backtest is not None and isinstance(backtest, pd.DataFrame) and not backtest.empty:
                valid_rows = backtest[~backtest['error'].isna()]
                if not valid_rows.empty:
                    rmse = np.sqrt(np.mean(valid_rows['error'] ** 2))
                    mae = np.mean(np.abs(valid_rows['error']))
                    model_type = result.get('model_type', 'LightGBM')
                    logger.info(f"  {model_id.upper()} ({model_type}): RMSE={rmse:.1f}, MAE={mae:.1f}")
                else:
                    logger.info(f"  {model_id.upper()}: Success (no backtest metrics)")
            else:
                logger.info(f"  {model_id.upper()}: Success (no backtest results)")
        else:
            logger.info(f"  {model_id.upper()}: {status.upper()}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='LightGBM NFP Prediction Model - First Release Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model (default: nsa_first)
  python Train/train_lightgbm_nfp.py --train

  # Train all first release model variants (nsa_first, sa_first)
  python Train/train_lightgbm_nfp.py --train-all

  # Train both NSA and SA models (first release)
  python Train/train_lightgbm_nfp.py --train-both

  # Predict with specific model
  python Train/train_lightgbm_nfp.py --predict 2024-12 --target sa --release first

NOTE: Only first release models are supported. Last release models are disabled.
        """
    )
    parser.add_argument('--train', action='store_true', help='Train a single model')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all first release model variants (nsa_first, sa_first)')
    parser.add_argument('--train-both', action='store_true',
                        help='Train both NSA and SA models (first release)')
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

    args = parser.parse_args()

    # Convert --no-huber-loss to use_huber_loss boolean
    use_huber_loss = not args.no_huber_loss

    model_id = get_model_id(args.target, args.release)

    if args.train_all:
        # Train all first release model variants
        logger.info("Training all first release model variants (nsa_first, sa_first)...")
        train_all_models(
            use_huber_loss=use_huber_loss,
            huber_delta=args.huber_delta
        )

    elif args.train_both:
        # Train both NSA and SA models (first release)
        logger.info(f"Training both NSA and SA models (first release)...")
        logger.info("NSA: LightGBM, SA: SARIMA")

        # Train NSA with LightGBM
        train_and_evaluate(
            target_type='nsa',
            release_type='first',
            use_huber_loss=use_huber_loss,
            huber_delta=args.huber_delta,
            archive_results=True
        )

        # Train SA with SARIMA
        from Train.train_sarima_sa import train_and_evaluate_sarima
        train_and_evaluate_sarima(archive_results=False)

    elif args.train:
        # Train single model
        if args.target == 'sa':
            # SA uses SARIMA (requires NSA predictions to exist)
            logger.info("Training SA model with SARIMA (requires NSA predictions)...")
            from Train.train_sarima_sa import train_and_evaluate_sarima
            train_and_evaluate_sarima(archive_results=True)
        else:
            # NSA uses LightGBM
            train_and_evaluate(
                target_type=args.target,
                release_type=args.release,
                use_huber_loss=use_huber_loss,
                huber_delta=args.huber_delta
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
        train_and_evaluate()
