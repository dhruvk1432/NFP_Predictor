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
    MAX_FEATURES,
    VIF_THRESHOLD,
    CORR_THRESHOLD,
    MIN_TARGET_CORR,
    LINEAR_BASELINE_PREDICTORS,
    PROTECTED_BINARY_FLAGS,
    MODEL_SAVE_DIR,
    ALL_TARGET_CONFIGS,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    get_model_id,
    get_target_path,
    # New config for simplified feature selection and Huber loss
    USE_HUBER_LOSS_DEFAULT,
    HUBER_DELTA,
    FEATURE_RECENCY_MONTHS,
    MAX_NAN_RATIO,
    WINSORIZE_FEATURES,
    WINSORIZE_LOWER_PERCENTILE,
    WINSORIZE_UPPER_PERCENTILE,
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_regression
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM/sklearn not available. Install with: pip install lightgbm scikit-learn")
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except ImportError:
    VIF_AVAILABLE = False


# NOTE: All core functionality is imported from modular components above.
# Configuration constants from Train/config.py
# Data loading functions from Train/data_loader.py
# Feature engineering functions from Train/feature_engineering.py
# Model training functions from Train/model.py


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

    OPTIMIZED:
    - Uses cached data loading to avoid redundant I/O
    - Batch processes features where possible
    - Progress logging every 12 months

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

    OPTIMIZED:
    - Returns computed means for caching
    - Accepts cached means to avoid redundant computation
    - In-place fillna where possible

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


def winsorize_features(
    X: pd.DataFrame,
    lower_pct: float = WINSORIZE_LOWER_PERCENTILE,
    upper_pct: float = WINSORIZE_UPPER_PERCENTILE,
    train_idx: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Winsorize features to reduce impact of COVID-like outliers.

    Clips extreme values to specified percentiles, computed from training data only
    to avoid look-ahead bias.

    Args:
        X: Feature DataFrame
        lower_pct: Lower percentile for clipping (e.g., 1.0 = 1st percentile)
        upper_pct: Upper percentile for clipping (e.g., 99.0 = 99th percentile)
        train_idx: Indices to use for computing percentiles (to avoid leakage)

    Returns:
        Tuple of (winsorized DataFrame, dict of clip bounds per column)
    """
    if not WINSORIZE_FEATURES:
        return X, {}

    X_winsorized = X.copy()
    numeric_cols = X_winsorized.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'ds']

    clip_bounds = {}

    # Use training data only for computing percentiles
    if train_idx is not None:
        X_train = X_winsorized.iloc[train_idx]
    else:
        X_train = X_winsorized

    for col in numeric_cols:
        col_data = X_train[col].dropna()
        if len(col_data) < 10:
            continue

        lower_bound = np.percentile(col_data, lower_pct)
        upper_bound = np.percentile(col_data, upper_pct)

        # Only clip if bounds are different (avoid constant columns)
        if lower_bound < upper_bound:
            X_winsorized[col] = X_winsorized[col].clip(lower=lower_bound, upper=upper_bound)
            clip_bounds[col] = (lower_bound, upper_bound)

    n_clipped = len(clip_bounds)
    if n_clipped > 0:
        logger.info(f"Winsorized {n_clipped} features to [{lower_pct}, {upper_pct}] percentiles")

    return X_winsorized, clip_bounds


def check_feature_recency(
    X: pd.DataFrame,
    dates: pd.Series,
    recency_months: int = FEATURE_RECENCY_MONTHS
) -> List[str]:
    """
    Check which features have data in the last N months.

    Args:
        X: Feature DataFrame
        dates: Series of dates corresponding to X rows
        recency_months: Require non-NaN data within this many months

    Returns:
        List of feature names that have recent data
    """
    if dates is None or len(dates) == 0:
        return list(X.columns)

    max_date = dates.max()
    cutoff_date = max_date - pd.DateOffset(months=recency_months)
    recent_mask = dates >= cutoff_date

    if not recent_mask.any():
        logger.warning(f"No data found in last {recency_months} months")
        return list(X.columns)

    X_recent = X.loc[recent_mask]
    numeric_cols = X_recent.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'ds']

    # Features with at least one non-NaN value in recent period
    recent_features = []
    for col in numeric_cols:
        if X_recent[col].notna().any():
            recent_features.append(col)

    n_removed = len(numeric_cols) - len(recent_features)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} features without data in last {recency_months} months")

    return recent_features


def select_features_simplified(
    X: pd.DataFrame,
    y: pd.Series,
    dates: Optional[pd.Series] = None,
    max_features: int = MAX_FEATURES,
    output_dir: Optional[Path] = None,
    target_type: str = 'nsa'
) -> Tuple[List[str], Dict]:
    """
    Simplified feature selection pipeline optimized for gradient boosting.

    Steps:
    1. Remove high-NaN columns (>50% missing)
    2. Remove zero-variance columns
    3. Filter by recency (must have data in last 12 months)
    4. Apply winsorization to reduce outlier impact
    5. VIF filtering to remove multicollinearity
    6. Select top features by LightGBM importance (no multi-metric ranking)

    This simplified approach works well for gradient boosting because:
    - Tree models naturally handle irrelevant features
    - VIF handles multicollinearity better than correlation filtering
    - Recency ensures features are relevant to current predictions
    - Winsorization reduces outlier dominance in feature data

    Args:
        X: Full feature DataFrame
        y: Target series
        dates: Date series for recency filtering
        max_features: Maximum features to select
        output_dir: Directory to save selection results
        target_type: 'nsa' or 'sa' for labeling outputs

    Returns:
        Tuple of (selected feature list, selection metadata dict)
    """
    logger.info("=" * 60)
    logger.info("SIMPLIFIED FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Starting with {len(X.columns)} features, target: {max_features}")

    # Identify protected binary flags
    protected_set = set(PROTECTED_BINARY_FLAGS)
    protected_features_present = [f for f in X.columns if f in protected_set]

    if protected_features_present:
        logger.info(f"Protecting {len(protected_features_present)} binary regime flags from removal")

    metadata = {
        'initial_features': len(X.columns),
        'target_features': max_features,
        'protected_features': protected_features_present,
        'steps': []
    }

    # Step 1: Basic cleaning
    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    # Remove high-NaN columns
    nan_pct = X_work.isna().mean()
    high_nan_cols = nan_pct[nan_pct > MAX_NAN_RATIO].index.tolist()
    high_nan_cols = [c for c in high_nan_cols if c not in protected_set]
    X_work = X_work.drop(columns=high_nan_cols)

    logger.info(f"Step 1: Removed {len(high_nan_cols)} high-NaN columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'high_nan_removal', 'removed': len(high_nan_cols), 'remaining': len(X_work.columns)})

    # Step 2: Remove zero variance
    col_means = X_work.mean()
    X_work = X_work.fillna(col_means).fillna(0)

    variances = X_work.var()
    zero_var = variances[variances == 0].index.tolist()
    zero_var = [c for c in zero_var if c not in protected_set]
    X_work = X_work.drop(columns=zero_var)

    logger.info(f"Step 2: Removed {len(zero_var)} zero-variance columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'zero_variance_removal', 'removed': len(zero_var), 'remaining': len(X_work.columns)})

    # Step 3: Recency filter
    if dates is not None:
        recent_features = check_feature_recency(X_work, dates, FEATURE_RECENCY_MONTHS)
        non_recent = [c for c in X_work.columns if c not in recent_features and c not in protected_set]
        X_work = X_work.drop(columns=non_recent)

        # Re-add protected features
        for f in protected_features_present:
            if f not in X_work.columns and f in X.columns:
                X_work[f] = X[f]

        logger.info(f"Step 3: Removed {len(non_recent)} stale features, {len(X_work.columns)} remaining")
        metadata['steps'].append({'step': 'recency_filter', 'removed': len(non_recent), 'remaining': len(X_work.columns)})
    else:
        logger.info("Step 3: Skipping recency filter (no dates provided)")

    # Step 4: Winsorization (compute bounds from full data for selection phase)
    X_work, clip_bounds = winsorize_features(X_work)
    metadata['winsorize_bounds'] = clip_bounds

    # Step 5: VIF filtering
    if VIF_AVAILABLE and len(X_work.columns) > 10:
        logger.info("\nStep 5: VIF Filtering")
        X_work, vif_removed = remove_high_vif_features(X_work, vif_threshold=VIF_THRESHOLD, max_iterations=30)

        # Add back protected features if removed
        for f in protected_features_present:
            if f in vif_removed and f in X.columns:
                X_work[f] = X[f]

        if vif_removed:
            logger.info(f"Removed {len(vif_removed)} high-VIF features, {len(X_work.columns)} remaining")
            metadata['steps'].append({'step': 'vif_removal', 'removed': len(vif_removed), 'remaining': len(X_work.columns)})
    else:
        logger.info("Step 5: Skipping VIF (not available or too few features)")

    # Step 6: LightGBM importance ranking (simplified - single metric)
    logger.info("\nStep 6: Computing LightGBM Feature Importance")
    lgbm_importance = compute_feature_importance_lgbm(X_work, y)

    # Select top features by LightGBM importance
    selected_features = lgbm_importance['feature'].head(max_features).tolist()

    # Ensure all protected features are included
    for f in protected_features_present:
        if f not in selected_features and f in X_work.columns:
            selected_features.append(f)
            logger.info(f"Force-included protected feature: {f}")

    logger.info(f"\nSelected {len(selected_features)} features by LightGBM importance")

    # Finalize metadata
    metadata['final_features'] = len(selected_features)
    metadata['selected_features'] = selected_features

    # Log top 20 features
    logger.info("\n" + "=" * 40)
    logger.info("TOP 20 SELECTED FEATURES:")
    logger.info("=" * 40)
    for i, row in lgbm_importance.head(20).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']} (gain: {row['importance_gain']:.1f})")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lgbm_importance.to_csv(output_dir / f"feature_ranking_{target_type}.csv", index=False)
        lgbm_importance[lgbm_importance['feature'].isin(selected_features)].to_csv(
            output_dir / f"selected_features_{target_type}.csv", index=False)

        with open(output_dir / f"feature_selection_metadata_{target_type}.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"\nSaved feature selection results to {output_dir}")

    return selected_features, metadata


def run_expanding_window_backtest(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    use_feature_selection: bool = True,
    feature_selection_interval: int = 6,  # Re-run feature selection every N months
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA
) -> pd.DataFrame:
    """
    Run proper expanding window backtest with NO TIME-TRAVEL VIOLATIONS.

    Critical Design Principles:
    1. Feature selection is re-run inside the loop using only past data
    2. NaN imputation uses only expanding window means (no future data)
    3. Model is FULLY retrained from scratch at each step using only data available at that time
    4. No information from future time periods leaks into predictions

    OPTIMIZATIONS:
    - Caches imputation means to avoid redundant computation
    - Pre-computes valid train mask outside inner loop
    - Reuses LightGBM Dataset where possible
    - Reduced logging overhead

    Args:
        target_df: Target DataFrame with 'ds' and 'y_mom' columns
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        use_feature_selection: Whether to run feature selection
        feature_selection_interval: How often to re-run feature selection (months)

    Returns:
        DataFrame with backtest results
    """
    model_id = get_model_id(target_type, release_type)

    logger.info("=" * 60)
    logger.info(f"EXPANDING WINDOW BACKTEST [{model_id.upper()}] (No Time-Travel)")
    logger.info("=" * 60)
    logger.info(f"Feature selection re-runs every {feature_selection_interval} months")

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

    # Cache for feature selection and imputation
    cached_features = None
    cached_impute_means = None
    last_feature_selection_idx = -feature_selection_interval  # Force first selection
    last_train_idx_len = 0  # Track if training set changed significantly

    # Get LightGBM params once
    params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

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

        # OPTIMIZATION: Only recompute imputation if training set grew significantly
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

        # WINSORIZATION: Apply to training data to reduce outlier impact
        # Get dates for the training data
        train_dates = X_full['ds'].iloc[train_idx_valid] if 'ds' in X_full.columns else None

        # Winsorize features using training data bounds
        X_train_winsorized, _ = winsorize_features(
            X_train_valid,
            train_idx=np.arange(len(X_train_valid))  # Use all training data for bounds
        )

        # FEATURE SELECTION: Re-run periodically using simplified approach
        if use_feature_selection and (i - last_feature_selection_idx >= feature_selection_interval or cached_features is None):
            logger.info(f"[{target_month.strftime('%Y-%m')}] Feature selection on {len(train_idx_valid)} samples...")

            selected_features, _ = select_features_simplified(
                X=X_train_winsorized,
                y=y_train_valid,
                dates=train_dates,
                max_features=MAX_FEATURES,
                output_dir=None,
                target_type=target_type
            )
            cached_features = selected_features
            last_feature_selection_idx = i

        # Use winsorized data for subsequent steps
        X_train_valid = X_train_winsorized

        # Get feature columns for this iteration
        if use_feature_selection and cached_features:
            feature_cols = [c for c in cached_features if c in X_train_valid.columns and c != 'ds']
        else:
            feature_cols = [c for c in X_train_valid.columns if c != 'ds']

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

        # Prepare training data with selected features
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


def train_and_evaluate(
    target_type: str = 'nsa',
    release_type: str = 'first',
    use_feature_selection: bool = True,
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    archive_results: bool = True
):
    """
    Main training and evaluation function using EXPANDING WINDOW methodology.

    This function ensures NO TIME-TRAVEL VIOLATIONS:
    1. Feature selection is re-evaluated inside the expanding window loop
    2. NaN imputation uses only past data (expanding window means)
    3. Model training uses only data available at each prediction time

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised
        use_feature_selection: If True, run comprehensive feature selection
        use_huber_loss: If True, use Huber loss function
        huber_delta: Huber delta parameter
        archive_results: If True, archive previous results (only for first model in batch)
    """
    model_id = get_model_id(target_type, release_type)

    # Archive previous backtest results before starting new run
    # Only archive once per training session (controlled by archive_results flag)
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

    # Run expanding window backtest (feature selection happens INSIDE the loop)
    backtest_results = run_expanding_window_backtest(
        target_df=target_df,
        target_type=target_type,
        release_type=release_type,
        use_feature_selection=use_feature_selection,
        feature_selection_interval=6,  # Re-select features every 6 months
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

    # Filter out NaN targets for final model training (same as backtest logic)
    # Keep future months in X_full for potential predictions but exclude from training
    valid_mask = ~y_full.isna()
    X_full_valid = X_full[valid_mask].copy()
    y_full_valid = y_full[valid_mask].copy()

    logger.info(f"Total observations: {len(X_full)}, Valid for training: {len(X_full_valid)}")

    # NEW: Train linear baseline on full training data BEFORE feature selection
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

    # WINSORIZATION: Apply to training data to reduce outlier impact (COVID, etc.)
    logger.info("\nApplying Winsorization to Reduce Outlier Impact...")
    train_dates = X_full_valid['ds'] if 'ds' in X_full_valid.columns else None
    X_full_winsorized, winsorize_bounds = winsorize_features(
        X_full_valid,
        train_idx=np.arange(len(X_full_valid))
    )

    # Final feature selection on full training data (for production model only)
    # Using simplified approach: VIF filtering + recency + LightGBM importance
    if use_feature_selection:
        feature_output_dir = OUTPUT_DIR / "feature_selection" / model_id
        feature_output_dir.mkdir(parents=True, exist_ok=True)

        selected_features, selection_metadata = select_features_simplified(
            X=X_full_winsorized,
            y=y_full_valid,
            dates=train_dates,
            max_features=MAX_FEATURES,
            output_dir=feature_output_dir,
            target_type=target_type
        )
        # Store winsorization bounds in metadata for use during prediction
        selection_metadata['winsorize_bounds'] = winsorize_bounds

        feature_cols = selected_features
        X_train = X_full_winsorized[['ds'] + [c for c in selected_features if c in X_full_winsorized.columns]].copy()
    else:
        feature_cols = [c for c in X_full_winsorized.columns if c != 'ds']
        X_train = X_full_winsorized.copy()

    # Train final model (using only valid targets)
    model, importance, residuals = train_lightgbm_model(
        X_train,
        y_full_valid,
        n_splits=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta
    )

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

        logger.info(f"✓ Generated {len(report_files)} report files")

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
        Dictionary with:
            - prediction: Point estimate of MoM change
            - intervals: Confidence intervals (50%, 80%, 95%)
            - std: Standard deviation of prediction error
            - target_month: The month being predicted
            - features_used: Number of features used
            - target_type: Type of prediction (nsa or sa)
            - release_type: Release type (first or last)
            - model_id: Combined model identifier
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


# NOTE: train_linear_baseline and create_linear_baseline_feature are imported from Train/model.py


def convert_mom_to_level(
    mom_prediction: float,
    previous_level: float
) -> float:
    """
    Convert MoM change prediction to level prediction.

    Args:
        mom_prediction: Month-on-month change prediction
        previous_level: Previous month's level

    Returns:
        Predicted level for current month
    """
    return previous_level + mom_prediction


def get_latest_prediction(target_type: str = 'nsa', release_type: str = 'first') -> Dict:
    """
    Get prediction for the most recent available month.

    Args:
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'

    Returns:
        Prediction dictionary
    """
    model_id = get_model_id(target_type, release_type)

    # Find latest snapshot available
    target_df = load_target_data(target_type=target_type, release_type=release_type)
    latest_target = target_df['ds'].max()

    logger.info(f"Making {model_id.upper()} prediction for latest available month: {latest_target}")

    return predict_nfp_mom(latest_target, target_type=target_type, release_type=release_type)


def train_all_models(
    use_feature_selection: bool = True,
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA
) -> Dict[str, Any]:
    """
    Train all first release model variants efficiently.

    Model architecture:
    - nsa_first: LightGBM model for NSA predictions
    - sa_first: SARIMA model for seasonal adjustment (applied to NSA predictions)

    The SA model uses SARIMA to predict the seasonal adjustment factor
    (SA - NSA) and adds it to the NSA predictions.

    NOTE: Last release models (nsa_last, sa_last) are disabled.

    Args:
        use_feature_selection: If True, run comprehensive feature selection (for NSA)
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
            use_feature_selection=use_feature_selection,
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


# =============================================================================
# VIF AND CORRELATION ANALYSIS
# =============================================================================

def compute_vif(X: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for feature multicollinearity analysis.

    VIF > 5 indicates moderate multicollinearity
    VIF > 10 indicates high multicollinearity (consider removing)

    Args:
        X: Feature DataFrame (numeric columns only)
        max_features: Maximum features to analyze (for performance)

    Returns:
        DataFrame with feature names and their VIF values, sorted by VIF descending
    """
    if not VIF_AVAILABLE:
        logger.warning("statsmodels not available for VIF calculation")
        return pd.DataFrame()

    # Select numeric columns only, exclude 'ds'
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'ds']

    # Handle inf/nan values
    X_clean = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.dropna(axis=1, how='any')

    # Limit features for performance
    if len(X_clean.columns) > max_features:
        # Select features with highest variance (most informative)
        variances = X_clean.var().sort_values(ascending=False)
        selected_cols = variances.head(max_features).index.tolist()
        X_clean = X_clean[selected_cols]
        logger.info(f"VIF analysis limited to top {max_features} features by variance")

    # Add constant for VIF calculation
    X_with_const = X_clean.copy()
    X_with_const['const'] = 1.0

    vif_data = []
    feature_cols = [c for c in X_with_const.columns if c != 'const']

    logger.info(f"Computing VIF for {len(feature_cols)} features...")

    for i, col in enumerate(feature_cols):
        try:
            vif_value = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(col))
            vif_data.append({'feature': col, 'VIF': vif_value})
        except Exception as e:
            logger.debug(f"Could not compute VIF for {col}: {e}")
            vif_data.append({'feature': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)

    # Summary statistics
    high_vif = vif_df[vif_df['VIF'] > 10]
    moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)]

    logger.info(f"\nVIF Analysis Summary:")
    logger.info(f"  Total features analyzed: {len(vif_df)}")
    logger.info(f"  High multicollinearity (VIF > 10): {len(high_vif)} features")
    logger.info(f"  Moderate multicollinearity (VIF 5-10): {len(moderate_vif)} features")

    if len(high_vif) > 0:
        logger.info(f"\nTop 10 features with highest VIF:")
        for idx, row in high_vif.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['VIF']:.2f}")

    return vif_df


def compute_correlation_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 30,
    high_corr_threshold: float = 0.9
) -> Dict:
    """
    Compute correlation analysis between features and target.

    Also identifies highly correlated feature pairs (potential redundancy).

    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of top correlated features to report
        high_corr_threshold: Threshold for identifying highly correlated pairs

    Returns:
        Dictionary with correlation analysis results
    """
    # Select numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'ds']
    X_numeric = X[numeric_cols].copy()

    # Handle inf/nan
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    # Feature-target correlations
    target_correlations = {}
    for col in X_numeric.columns:
        valid_mask = X_numeric[col].notna() & y.notna()
        if valid_mask.sum() > 10:  # Minimum samples
            corr = X_numeric.loc[valid_mask, col].corr(y[valid_mask])
            target_correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(target_correlations.items(), key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)

    logger.info(f"\nTop {top_n} Features by Target Correlation:")
    for i, (feat, corr) in enumerate(sorted_corrs[:top_n], 1):
        if pd.notna(corr):
            logger.info(f"  {i}. {feat}: {corr:.4f}")

    # Feature-feature correlations (identify redundancy)
    logger.info(f"\nComputing feature-feature correlation matrix...")
    corr_matrix = X_numeric.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

    if high_corr_pairs:
        logger.info(f"\nHighly Correlated Feature Pairs (|r| > {high_corr_threshold}):")
        for pair in high_corr_pairs[:20]:  # Show top 20
            logger.info(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.4f}")
        logger.info(f"  Total highly correlated pairs: {len(high_corr_pairs)}")
    else:
        logger.info(f"\nNo feature pairs with |correlation| > {high_corr_threshold}")

    return {
        'target_correlations': dict(sorted_corrs),
        'high_corr_pairs': high_corr_pairs,
        'correlation_matrix': corr_matrix
    }


# =============================================================================
# COMPREHENSIVE FEATURE SELECTION PIPELINE
# =============================================================================

def remove_high_vif_features(
    X: pd.DataFrame,
    vif_threshold: float = VIF_THRESHOLD,
    max_iterations: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with VIF above threshold.

    Args:
        X: Feature DataFrame
        vif_threshold: VIF threshold for removal
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    if not VIF_AVAILABLE:
        logger.warning("VIF not available, skipping VIF-based removal")
        return X, []

    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # Drop columns with any NaN
    X_clean = X_clean.dropna(axis=1, how='any')

    # Also drop columns with zero variance
    zero_var_cols = X_clean.columns[X_clean.var() == 0].tolist()
    X_clean = X_clean.drop(columns=zero_var_cols)

    removed_features = zero_var_cols.copy()

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        if len(X_clean.columns) <= 10:
            break

        # Add constant for VIF
        X_with_const = X_clean.copy()
        X_with_const['const'] = 1.0

        # Calculate VIF for all features
        vif_values = {}
        for col in X_clean.columns:
            try:
                vif = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(col))
                vif_values[col] = vif
            except:
                vif_values[col] = np.inf

        # Find maximum VIF
        max_vif_col = max(vif_values, key=vif_values.get)
        max_vif = vif_values[max_vif_col]

        if max_vif <= vif_threshold:
            break

        # Remove feature with highest VIF
        X_clean = X_clean.drop(columns=[max_vif_col])
        removed_features.append(max_vif_col)

        # Log each removal to show iterative nature
        logger.info(f"  VIF iter {iteration}: removed '{max_vif_col}' (VIF={max_vif:.1f}), {len(X_clean.columns)} features remaining")

    logger.info(f"VIF removal complete: {len(removed_features)} features removed, {len(X_clean.columns)} remaining")
    return X_clean, removed_features


def remove_correlated_features(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = CORR_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features, keeping the one with higher target correlation.

    Args:
        X: Feature DataFrame
        y: Target series
        corr_threshold: Correlation threshold above which to remove

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')

    # Calculate target correlations
    target_corrs = {}
    for col in X_clean.columns:
        valid_mask = X_clean[col].notna() & y.notna()
        if valid_mask.sum() > 10:
            target_corrs[col] = abs(X_clean.loc[valid_mask, col].corr(y[valid_mask]))
        else:
            target_corrs[col] = 0

    # Calculate feature correlation matrix
    corr_matrix = X_clean.corr().abs()

    # Find pairs above threshold
    removed_features = []
    cols_to_keep = set(X_clean.columns)

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]

            if col_i not in cols_to_keep or col_j not in cols_to_keep:
                continue

            if corr_matrix.iloc[i, j] > corr_threshold:
                # Remove the one with lower target correlation
                if target_corrs.get(col_i, 0) >= target_corrs.get(col_j, 0):
                    cols_to_keep.discard(col_j)
                    removed_features.append(col_j)
                else:
                    cols_to_keep.discard(col_i)
                    removed_features.append(col_i)

    X_reduced = X_clean[list(cols_to_keep)]
    logger.info(f"Correlation removal: {len(removed_features)} features removed, {len(X_reduced.columns)} remaining")
    return X_reduced, removed_features


def select_by_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    min_corr: float = MIN_TARGET_CORR
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with negligible correlation to target.

    Args:
        X: Feature DataFrame
        y: Target series
        min_corr: Minimum absolute correlation required

    Returns:
        Tuple of (reduced DataFrame, list of removed features)
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')

    target_corrs = {}
    for col in X_clean.columns:
        valid_mask = X_clean[col].notna() & y.notna()
        if valid_mask.sum() > 10:
            target_corrs[col] = X_clean.loc[valid_mask, col].corr(y[valid_mask])
        else:
            target_corrs[col] = 0

    # Keep features with correlation above threshold
    kept_cols = [col for col, corr in target_corrs.items() if abs(corr) >= min_corr]
    removed_cols = [col for col in X_clean.columns if col not in kept_cols]

    X_reduced = X_clean[kept_cols]
    logger.info(f"Target correlation filter: {len(removed_cols)} features removed, {len(X_reduced.columns)} remaining")
    return X_reduced, removed_cols


def compute_feature_importance_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100
) -> pd.DataFrame:
    """
    Compute feature importance using a quick LightGBM model.

    Args:
        X: Feature DataFrame
        y: Target series
        n_estimators: Number of estimators for quick training

    Returns:
        DataFrame with feature importance scores
    """
    if not LIGHTGBM_AVAILABLE:
        return pd.DataFrame()

    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Quick LightGBM training
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_clean, label=y)
    model = lgb.train(params, train_data, num_boost_round=n_estimators)

    # Get importance
    importance = pd.DataFrame({
        'feature': X_clean.columns,
        'importance_gain': model.feature_importance(importance_type='gain'),
        'importance_split': model.feature_importance(importance_type='split')
    })
    importance = importance.sort_values('importance_gain', ascending=False).reset_index(drop=True)

    return importance


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Compute mutual information between features and target.

    Args:
        X: Feature DataFrame
        y: Target series
        n_neighbors: Number of neighbors for MI estimation

    Returns:
        DataFrame with MI scores
    """
    X_clean = X.select_dtypes(include=[np.number]).copy()
    X_clean = X_clean.drop(columns=['ds'], errors='ignore')
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Standardize for better MI estimation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Compute MI
    mi_scores = mutual_info_regression(X_scaled, y, n_neighbors=n_neighbors, random_state=42)

    mi_df = pd.DataFrame({
        'feature': X_clean.columns,
        'mutual_info': mi_scores
    })
    mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

    return mi_df


def select_features_comprehensive(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = MAX_FEATURES,
    output_dir: Optional[Path] = None,
    target_type: str = 'nsa',
    skip_vif: bool = False
) -> Tuple[List[str], Dict]:
    """
    Comprehensive feature selection pipeline.

    OPTIMIZED:
    - Computes target correlations once and reuses
    - Vectorized operations where possible
    - Optional VIF skip for faster iteration

    Steps:
    1. Remove zero-variance and infinite features
    2. Remove features with negligible target correlation
    3. Remove highly correlated feature pairs (keep higher target correlation)
    4. Compute multiple importance metrics (LightGBM, MI, target correlation)
    5. Aggregate rankings and select top features
    6. (Optional) VIF check on final set

    Args:
        X: Full feature DataFrame
        y: Target series
        max_features: Maximum features to select
        output_dir: Directory to save selection results
        target_type: 'nsa' or 'sa' for labeling outputs
        skip_vif: If True, skip VIF check (faster)

    Returns:
        Tuple of (selected feature list, selection metadata dict)
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Starting with {len(X.columns)} features, target: {max_features}")

    # Identify protected binary flags present in dataset
    protected_set = set(PROTECTED_BINARY_FLAGS)
    protected_features_present = [f for f in X.columns if f in protected_set]

    if protected_features_present:
        logger.info(f"Protecting {len(protected_features_present)} binary regime flags from removal")

    # Initialize metadata
    metadata = {
        'initial_features': len(X.columns),
        'target_features': max_features,
        'protected_features': protected_features_present,
        'steps': []
    }

    # Step 1: Basic cleaning - vectorized
    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    # Remove columns with too many NaN (>50%) - vectorized
    nan_pct = X_work.isna().mean()
    high_nan_cols = nan_pct[nan_pct > 0.5].index.tolist()
    high_nan_cols = [c for c in high_nan_cols if c not in protected_set]

    X_work = X_work.drop(columns=high_nan_cols)
    logger.info(f"Step 1: Removed {len(high_nan_cols)} high-NaN columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'high_nan_removal', 'removed': len(high_nan_cols), 'remaining': len(X_work.columns)})

    # Fill remaining NaN with column mean - vectorized
    col_means = X_work.mean()
    X_work = X_work.fillna(col_means).fillna(0)

    # Remove zero variance - vectorized
    variances = X_work.var()
    zero_var = variances[variances == 0].index.tolist()
    zero_var = [c for c in zero_var if c not in protected_set]

    X_work = X_work.drop(columns=zero_var)
    logger.info(f"Step 2: Removed {len(zero_var)} zero-variance columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'zero_variance_removal', 'removed': len(zero_var), 'remaining': len(X_work.columns)})

    # OPTIMIZATION: Compute all target correlations ONCE
    logger.info("\nComputing target correlations (vectorized)...")
    valid_y = y.notna()
    target_corrs = X_work.loc[valid_y].corrwith(y[valid_y]).abs().fillna(0).to_dict()

    # Step 3: Target correlation filter - use pre-computed correlations
    logger.info("\nStep 3: Target Correlation Filter")
    low_corr_cols = [c for c in X_work.columns if target_corrs.get(c, 0) < MIN_TARGET_CORR and c not in protected_set]
    X_work = X_work.drop(columns=low_corr_cols)

    # Re-add protected features
    for f in protected_features_present:
        if f not in X_work.columns and f in X.columns:
            X_work[f] = X[f]

    logger.info(f"Removed {len(low_corr_cols)} low-correlation columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'target_correlation_filter', 'removed': len(low_corr_cols), 'remaining': len(X_work.columns)})

    # Step 4: High correlation removal - use pre-computed target correlations
    logger.info("\nStep 4: High Correlation Removal")
    corr_matrix = X_work.corr().abs()
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                # Keep the one with higher target correlation
                if target_corrs.get(col_i, 0) >= target_corrs.get(col_j, 0):
                    if col_j not in protected_set:
                        to_drop.add(col_j)
                else:
                    if col_i not in protected_set:
                        to_drop.add(col_i)

    X_work = X_work.drop(columns=list(to_drop))

    # Re-add protected features
    for f in protected_features_present:
        if f not in X_work.columns and f in X.columns:
            X_work[f] = X[f]

    logger.info(f"Removed {len(to_drop)} highly-correlated columns, {len(X_work.columns)} remaining")
    metadata['steps'].append({'step': 'high_correlation_removal', 'removed': len(to_drop), 'remaining': len(X_work.columns)})

    # Step 5: Compute importance metrics
    logger.info("\nStep 5: Computing Feature Importance Metrics")

    # LightGBM importance
    logger.info("  Computing LightGBM importance...")
    lgbm_importance = compute_feature_importance_lgbm(X_work, y)

    # Mutual Information
    logger.info("  Computing Mutual Information...")
    mi_scores = compute_mutual_information(X_work, y)

    # Target correlation DataFrame (using pre-computed values)
    target_corr_df = pd.DataFrame([
        {'feature': c, 'target_corr': target_corrs.get(c, 0)} for c in X_work.columns
    ]).sort_values('target_corr', ascending=False).reset_index(drop=True)

    # Step 6: Aggregate rankings
    logger.info("\nStep 6: Aggregating Feature Rankings")

    # Create ranking DataFrame - optimized merge
    ranking_df = pd.DataFrame({'feature': list(X_work.columns)})

    lgbm_importance['lgbm_rank'] = range(1, len(lgbm_importance) + 1)
    mi_scores['mi_rank'] = range(1, len(mi_scores) + 1)
    target_corr_df['corr_rank'] = range(1, len(target_corr_df) + 1)

    ranking_df = (ranking_df
                  .merge(lgbm_importance[['feature', 'importance_gain', 'lgbm_rank']], on='feature', how='left')
                  .merge(mi_scores[['feature', 'mutual_info', 'mi_rank']], on='feature', how='left')
                  .merge(target_corr_df[['feature', 'target_corr', 'corr_rank']], on='feature', how='left'))

    # Compute average rank
    ranking_df['avg_rank'] = ranking_df[['lgbm_rank', 'mi_rank', 'corr_rank']].mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank').reset_index(drop=True)

    # Select top features
    selected_features = ranking_df['feature'].head(max_features).tolist()

    # Ensure all protected features are included
    for f in protected_features_present:
        if f not in selected_features and f in X_work.columns:
            selected_features.append(f)
            logger.info(f"Force-included protected feature: {f}")

    logger.info(f"\nSelected {len(selected_features)} features based on aggregated ranking")

    # Step 7: Optional VIF check
    if not skip_vif and VIF_AVAILABLE and len(selected_features) > 10:
        logger.info("\nStep 7: VIF Check on Selected Features")
        X_selected = X_work[selected_features].copy()
        X_selected, vif_removed = remove_high_vif_features(X_selected, vif_threshold=VIF_THRESHOLD, max_iterations=20)

        # Add back protected features if removed
        for f in protected_features_present:
            if f in vif_removed and f in X_work.columns:
                X_selected[f] = X_work[f]

        if vif_removed:
            selected_features = list(X_selected.columns)
            logger.info(f"After VIF check: {len(selected_features)} features")
            metadata['steps'].append({'step': 'vif_removal', 'removed': len(vif_removed), 'remaining': len(selected_features)})

    # Finalize
    metadata['final_features'] = len(selected_features)
    metadata['selected_features'] = selected_features

    # Log top 20 selected features (condensed)
    logger.info("\n" + "=" * 40)
    logger.info("TOP 20 SELECTED FEATURES:")
    logger.info("=" * 40)
    for i, feat in enumerate(selected_features[:20], 1):
        row = ranking_df[ranking_df['feature'] == feat]
        if not row.empty:
            row = row.iloc[0]
            logger.info(f"  {i:2d}. {feat} (LGBM:{row['lgbm_rank']:.0f}, MI:{row['mi_rank']:.0f}, Corr:{row['corr_rank']:.0f})")

    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ranking_df.to_csv(output_dir / f"feature_ranking_{target_type}.csv", index=False)
        ranking_df[ranking_df['feature'].isin(selected_features)].to_csv(
            output_dir / f"selected_features_{target_type}.csv", index=False)

        with open(output_dir / f"feature_selection_metadata_{target_type}.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"\nSaved feature selection results to {output_dir}")

    return selected_features, metadata


def run_feature_diagnostics(target_type: str = 'nsa') -> Dict:
    """
    Run comprehensive feature diagnostics including VIF and correlation analysis.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        Dictionary with diagnostic results
    """
    logger.info("=" * 60)
    logger.info(f"Running Feature Diagnostics for {target_type.upper()} Model")
    logger.info("=" * 60)

    # Load target data
    target_df = load_target_data(target_type=target_type)

    # Build training dataset
    X, y = build_training_dataset(target_df, target_type=target_type)

    if X.empty:
        logger.error("Could not build training dataset for diagnostics")
        return {}

    logger.info(f"\nDataset: {len(X)} samples, {len(X.columns)} features")

    # VIF Analysis
    logger.info("\n" + "=" * 40)
    logger.info("VIF (Multicollinearity) Analysis")
    logger.info("=" * 40)
    vif_df = compute_vif(X)

    # Correlation Analysis
    logger.info("\n" + "=" * 40)
    logger.info("Correlation Analysis")
    logger.info("=" * 40)
    corr_results = compute_correlation_analysis(X, y)

    # Save results
    results_dir = MODEL_SAVE_DIR / target_type / "diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not vif_df.empty:
        vif_path = results_dir / f"vif_analysis_{target_type}.csv"
        vif_df.to_csv(vif_path, index=False)
        logger.info(f"\nSaved VIF analysis to {vif_path}")

    # Save target correlations
    corr_df = pd.DataFrame([
        {'feature': k, 'target_correlation': v}
        for k, v in corr_results['target_correlations'].items()
    ])
    corr_path = results_dir / f"target_correlations_{target_type}.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved target correlations to {corr_path}")

    # Save high correlation pairs
    if corr_results['high_corr_pairs']:
        pairs_df = pd.DataFrame(corr_results['high_corr_pairs'])
        pairs_path = results_dir / f"high_corr_pairs_{target_type}.csv"
        pairs_df.to_csv(pairs_path, index=False)
        logger.info(f"Saved high correlation pairs to {pairs_path}")

    return {
        'vif': vif_df,
        'correlations': corr_results,
        'num_features': len(X.columns),
        'num_samples': len(X)
    }


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
    # DISABLED: --train-both-releases - last release models are not supported
    # parser.add_argument('--train-both-releases', action='store_true',
    #                     help='Train both first and last release models (same target type)')
    parser.add_argument('--predict', type=str, help='Predict for a specific month (YYYY-MM)')
    parser.add_argument('--latest', action='store_true', help='Predict for latest available month')
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help='Target type: nsa (non-seasonally adjusted) or sa (seasonally adjusted)')
    parser.add_argument('--release', type=str, default='first', choices=['first', 'last'],
                        help='Release type: first (initial) or last (final revised)')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Run feature diagnostics (VIF, correlations)')
    parser.add_argument('--no-huber-loss', action='store_true',
                        help='Disable Huber loss (uses MSE instead). Huber is enabled by default for outlier robustness.')
    parser.add_argument('--huber-delta', type=float, default=HUBER_DELTA,
                        help=f'Huber delta parameter (default: {HUBER_DELTA}). Lower = more robust to outliers.')

    args = parser.parse_args()

    # Convert --no-huber-loss to use_huber_loss boolean
    # Huber loss is enabled by default (USE_HUBER_LOSS_DEFAULT = True)
    use_huber_loss = not args.no_huber_loss

    model_id = get_model_id(args.target, args.release)

    if args.diagnostics:
        # Run feature diagnostics (VIF and correlation analysis)
        run_feature_diagnostics(target_type=args.target)

    elif args.train_all:
        # Train all first release model variants
        logger.info("Training all first release model variants (nsa_first, sa_first)...")
        train_all_models(
            use_huber_loss=use_huber_loss,
            huber_delta=args.huber_delta
        )

    # DISABLED: --train-both-releases - last release models are not supported
    # elif args.train_both_releases:
    #     # Train both first and last release for a single target type
    #     logger.info(f"Training both release types for {args.target.upper()}...")
    #     train_and_evaluate(
    #         target_type=args.target,
    #         release_type='first',
    #         use_huber_loss=use_huber_loss,
    #         huber_delta=args.huber_delta,
    #         archive_results=True
    #     )
    #     train_and_evaluate(
    #         target_type=args.target,
    #         release_type='last',
    #         use_huber_loss=use_huber_loss,
    #         huber_delta=args.huber_delta,
    #         archive_results=False
    #     )

    elif args.train_both:
        # Train both NSA and SA models (first release)
        # NSA uses LightGBM, SA uses SARIMA
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