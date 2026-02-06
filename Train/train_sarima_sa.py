"""
SARIMA-based SA NFP Prediction Model

Uses NSA predictions from LightGBM model and applies SARIMA to predict
the seasonal adjustment factor (SA - NSA in MoM changes).

Walk-forward methodology:
1. Load NSA predictions from backtest
2. For each prediction date, fit SARIMA using only historical adjustment data
3. Predict the adjustment for that month
4. Combine NSA prediction + SARIMA adjustment = SA prediction

The seasonal adjustment shows strong monthly patterns:
- January: ~+3000 (large positive adjustment)
- July: ~+1250 (moderate positive)
- Other months: mostly negative adjustments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import warnings
import itertools

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, BACKTEST_MONTHS
from Train.backtest_results import generate_backtest_report

logger = setup_logger(__file__, TEMP_DIR)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False


def load_adjustment_data() -> pd.DataFrame:
    """
    Load NSA and SA target data and calculate the MoM adjustment series.

    Returns:
        DataFrame with columns: ds, y_mom_nsa, y_mom_sa, adjustment_mom
    """
    data_nsa = pd.read_parquet("data/NFP_target/total_nsa_first_release.parquet")
    data_sa = pd.read_parquet("data/NFP_target/total_sa_first_release.parquet")

    # Merge on ds to ensure alignment
    merged = data_nsa.merge(data_sa, on='ds', suffixes=('_nsa', '_sa'))

    # Calculate MoM changes
    merged['y_mom_nsa'] = merged['y_nsa'].diff()
    merged['y_mom_sa'] = merged['y_sa'].diff()

    # Calculate the adjustment in MoM space: SA_MoM - NSA_MoM
    merged['adjustment_mom'] = merged['y_mom_sa'] - merged['y_mom_nsa']

    # Keep only rows with valid adjustment
    merged = merged.dropna(subset=['adjustment_mom'])

    return merged[['ds', 'y_mom_nsa', 'y_mom_sa', 'adjustment_mom']].copy()


def load_nsa_predictions(predictions_path: str = None) -> pd.DataFrame:
    """
    Load NSA predictions from the backtest results.

    Args:
        predictions_path: Path to predictions.csv. If None, uses default path.

    Returns:
        DataFrame with NSA predictions
    """
    if predictions_path is None:
        predictions_path = OUTPUT_DIR / "backtest_results" / "nsa_first" / "predictions.csv"

    df = pd.read_csv(predictions_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


def analyze_adjustment_stationarity(adjustment_series: pd.Series) -> Dict:
    """
    Analyze the stationarity of the adjustment series.

    Args:
        adjustment_series: Series of seasonal adjustments

    Returns:
        Dictionary with stationarity analysis results
    """
    results = {}

    # ADF test on original series
    adf_result = adfuller(adjustment_series.dropna())
    results['adf_statistic'] = adf_result[0]
    results['adf_pvalue'] = adf_result[1]
    results['is_stationary'] = adf_result[1] < 0.05

    # ADF test on differenced series
    diff_series = adjustment_series.diff().dropna()
    adf_diff = adfuller(diff_series)
    results['adf_diff_pvalue'] = adf_diff[1]
    results['diff_is_stationary'] = adf_diff[1] < 0.05

    logger.info(f"Stationarity Analysis:")
    logger.info(f"  ADF p-value: {results['adf_pvalue']:.4f} ({'stationary' if results['is_stationary'] else 'non-stationary'})")
    logger.info(f"  Differenced ADF p-value: {results['adf_diff_pvalue']:.4f}")

    return results


def grid_search_sarima(
    train_series: pd.Series,
    p_range: range = range(0, 3),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    D_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    seasonal_period: int = 12,
    n_cv_folds: int = 3,
    cv_horizon: int = 6
) -> Tuple[Tuple, Dict]:
    """
    Grid search for optimal SARIMA parameters using time-series cross-validation.

    Args:
        train_series: Historical adjustment series (indexed by date)
        p_range: Range for AR order
        d_range: Range for differencing order
        q_range: Range for MA order
        P_range: Range for seasonal AR order
        D_range: Range for seasonal differencing order
        Q_range: Range for seasonal MA order
        seasonal_period: Seasonal period (12 for monthly)
        n_cv_folds: Number of cross-validation folds
        cv_horizon: Prediction horizon for each fold

    Returns:
        Tuple of (best_order, search_results)
        best_order = ((p,d,q), (P,D,Q,s))
    """
    logger.info(f"Starting SARIMA grid search with {len(train_series)} observations...")
    logger.info(f"CV: {n_cv_folds} folds, {cv_horizon} month horizon each")

    # Generate parameter combinations
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [seasonal_period]))

    total_combinations = len(pdq) * len(seasonal_pdq)
    logger.info(f"Testing {total_combinations} parameter combinations...")

    results = []
    best_aic = np.inf
    best_cv_rmse = np.inf
    best_order = None
    best_seasonal_order = None

    # Time-series CV split points
    n = len(train_series)
    min_train_size = max(36, n - n_cv_folds * cv_horizon - cv_horizon)  # At least 3 years

    for i, (p, d, q) in enumerate(pdq):
        for (P, D, Q, s) in seasonal_pdq:
            try:
                cv_errors = []

                # Time-series cross-validation
                for fold in range(n_cv_folds):
                    # Training ends before validation period
                    train_end = n - (n_cv_folds - fold) * cv_horizon

                    if train_end < min_train_size:
                        continue

                    train_data = train_series.iloc[:train_end]
                    val_data = train_series.iloc[train_end:train_end + cv_horizon]

                    if len(val_data) < cv_horizon:
                        continue

                    # Fit model
                    model = SARIMAX(
                        train_data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fitted = model.fit(disp=False, maxiter=200)

                    # Predict validation period
                    forecast = fitted.forecast(steps=cv_horizon)

                    # Calculate errors
                    errors = val_data.values - forecast.values
                    cv_errors.extend(errors)

                if len(cv_errors) < n_cv_folds:
                    continue

                cv_rmse = np.sqrt(np.mean(np.array(cv_errors) ** 2))

                # Also fit on full data for AIC
                full_model = SARIMAX(
                    train_series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                full_fitted = full_model.fit(disp=False, maxiter=200)
                aic = full_fitted.aic

                results.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, s),
                    'cv_rmse': cv_rmse,
                    'aic': aic,
                    'n_cv_samples': len(cv_errors)
                })

                # Update best based on CV RMSE (primary) and AIC (secondary)
                if cv_rmse < best_cv_rmse:
                    best_cv_rmse = cv_rmse
                    best_aic = aic
                    best_order = (p, d, q)
                    best_seasonal_order = (P, D, Q, s)

            except Exception as e:
                continue

        # Progress logging
        if (i + 1) % 10 == 0:
            logger.info(f"  Tested {(i + 1) * len(seasonal_pdq)}/{total_combinations} combinations...")

    if best_order is None:
        # Fallback to simple model
        logger.warning("Grid search failed to find valid model, using defaults")
        best_order = (1, 0, 1)
        best_seasonal_order = (1, 0, 1, 12)
        best_cv_rmse = np.nan
        best_aic = np.nan

    logger.info(f"\nBest SARIMA parameters:")
    logger.info(f"  Order: {best_order}")
    logger.info(f"  Seasonal Order: {best_seasonal_order}")
    logger.info(f"  CV RMSE: {best_cv_rmse:.2f}")
    logger.info(f"  AIC: {best_aic:.2f}")

    return (best_order, best_seasonal_order), {
        'all_results': results,
        'best_cv_rmse': best_cv_rmse,
        'best_aic': best_aic
    }


def fit_sarima_and_predict(
    train_series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    steps: int = 1,
    return_conf_int: bool = True,
    alpha: float = 0.2
) -> Dict:
    """
    Fit SARIMA model and make prediction with confidence intervals.

    Args:
        train_series: Historical data to fit on
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        steps: Number of steps to forecast
        return_conf_int: Whether to return confidence intervals
        alpha: Significance level for confidence intervals

    Returns:
        Dictionary with prediction and optional confidence intervals
    """
    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False, maxiter=200)

        # Make forecast
        forecast_result = fitted.get_forecast(steps=steps)
        prediction = forecast_result.predicted_mean.iloc[0] if steps == 1 else forecast_result.predicted_mean

        result = {
            'prediction': prediction,
            'aic': fitted.aic,
            'bic': fitted.bic
        }

        if return_conf_int:
            # Get confidence intervals at different levels
            conf_int_50 = forecast_result.conf_int(alpha=0.50)
            conf_int_80 = forecast_result.conf_int(alpha=0.20)
            conf_int_95 = forecast_result.conf_int(alpha=0.05)

            if steps == 1:
                result['lower_50'] = conf_int_50.iloc[0, 0]
                result['upper_50'] = conf_int_50.iloc[0, 1]
                result['lower_80'] = conf_int_80.iloc[0, 0]
                result['upper_80'] = conf_int_80.iloc[0, 1]
                result['lower_95'] = conf_int_95.iloc[0, 0]
                result['upper_95'] = conf_int_95.iloc[0, 1]
            else:
                result['conf_int_50'] = conf_int_50
                result['conf_int_80'] = conf_int_80
                result['conf_int_95'] = conf_int_95

        return result

    except Exception as e:
        logger.warning(f"SARIMA fit failed: {e}")
        # Return simple historical mean as fallback
        month = train_series.index[-1].month + 1
        if month > 12:
            month = 1

        # Use historical mean for same month
        train_series_with_month = train_series.copy()
        train_series_with_month.index = pd.to_datetime(train_series_with_month.index)
        same_month_data = train_series_with_month[train_series_with_month.index.month == month]

        if len(same_month_data) > 0:
            prediction = same_month_data.mean()
            std = same_month_data.std() if len(same_month_data) > 1 else train_series.std()
        else:
            prediction = train_series.mean()
            std = train_series.std()

        return {
            'prediction': prediction,
            'lower_50': prediction - 0.67 * std,
            'upper_50': prediction + 0.67 * std,
            'lower_80': prediction - 1.28 * std,
            'upper_80': prediction + 1.28 * std,
            'lower_95': prediction - 1.96 * std,
            'upper_95': prediction + 1.96 * std,
            'aic': np.nan,
            'bic': np.nan,
            'fallback': True
        }


def run_sarima_backtest(
    nsa_predictions_path: str = None,
    rerun_grid_search: bool = False,
    fixed_order: Tuple = None,
    fixed_seasonal_order: Tuple = None
) -> pd.DataFrame:
    """
    Run walk-forward SARIMA backtest for SA predictions.

    For each prediction date:
    1. Use only adjustment data available before that date
    2. Fit SARIMA to predict the adjustment
    3. Combine NSA prediction + predicted adjustment = SA prediction

    Args:
        nsa_predictions_path: Path to NSA predictions CSV
        rerun_grid_search: If True, run grid search at each step (slow but optimal)
        fixed_order: Fixed ARIMA order to use (skip grid search)
        fixed_seasonal_order: Fixed seasonal order to use

    Returns:
        DataFrame with SA predictions and metrics
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for SARIMA. Install with: pip install statsmodels")

    logger.info("=" * 60)
    logger.info("SARIMA-BASED SA PREDICTION BACKTEST")
    logger.info("=" * 60)

    # Load adjustment data
    logger.info("Loading adjustment data...")
    adjustment_df = load_adjustment_data()
    adjustment_series = adjustment_df.set_index('ds')['adjustment_mom']

    logger.info(f"Adjustment data: {len(adjustment_series)} observations")
    logger.info(f"Date range: {adjustment_series.index.min()} to {adjustment_series.index.max()}")

    # Analyze stationarity
    analyze_adjustment_stationarity(adjustment_series)

    # Load NSA predictions
    logger.info("\nLoading NSA predictions...")
    nsa_preds = load_nsa_predictions(nsa_predictions_path)
    logger.info(f"NSA predictions: {len(nsa_preds)} observations")
    logger.info(f"Date range: {nsa_preds['date'].min()} to {nsa_preds['date'].max()}")

    # Initial grid search on data before first prediction
    first_pred_date = nsa_preds['date'].min()
    train_mask = adjustment_series.index < first_pred_date
    initial_train = adjustment_series[train_mask]

    if fixed_order is not None and fixed_seasonal_order is not None:
        best_order = fixed_order
        best_seasonal_order = fixed_seasonal_order
        logger.info(f"\nUsing fixed SARIMA parameters:")
        logger.info(f"  Order: {best_order}")
        logger.info(f"  Seasonal Order: {best_seasonal_order}")
    else:
        logger.info(f"\nRunning initial grid search on {len(initial_train)} observations...")
        (best_order, best_seasonal_order), search_results = grid_search_sarima(
            initial_train,
            p_range=range(0, 3),
            d_range=range(0, 2),
            q_range=range(0, 3),
            P_range=range(0, 2),
            D_range=range(0, 2),
            Q_range=range(0, 2),
            n_cv_folds=3,
            cv_horizon=6
        )

    # Walk-forward prediction
    logger.info("\nRunning walk-forward predictions...")
    results = []
    all_residuals = []

    for idx, row in nsa_preds.iterrows():
        target_date = row['date']
        nsa_pred = row['pred']
        nsa_actual = row['actual']  # This is the NSA actual

        # Get adjustment data up to (not including) target date
        train_mask = adjustment_series.index < target_date
        train_adjustment = adjustment_series[train_mask]

        if len(train_adjustment) < 24:  # Need at least 2 years
            logger.warning(f"Insufficient data for {target_date}, skipping")
            continue

        # Optionally re-run grid search (slow)
        if rerun_grid_search and (idx % 12 == 0):
            logger.info(f"Re-running grid search for {target_date}...")
            (best_order, best_seasonal_order), _ = grid_search_sarima(
                train_adjustment,
                p_range=range(0, 3),
                d_range=range(0, 2),
                q_range=range(0, 3),
                P_range=range(0, 2),
                D_range=range(0, 2),
                Q_range=range(0, 2),
                n_cv_folds=3,
                cv_horizon=6
            )

        # Fit SARIMA and predict adjustment
        sarima_result = fit_sarima_and_predict(
            train_adjustment,
            best_order,
            best_seasonal_order,
            steps=1,
            return_conf_int=True
        )

        predicted_adjustment = sarima_result['prediction']

        # SA prediction = NSA prediction + SARIMA adjustment
        sa_pred = nsa_pred + predicted_adjustment

        # Get actual adjustment if available
        if target_date in adjustment_series.index:
            actual_adjustment = adjustment_series[target_date]
            # SA actual = NSA actual + actual adjustment (or directly from data)
            sa_actual = nsa_actual + actual_adjustment if pd.notna(nsa_actual) else np.nan
            adjustment_error = actual_adjustment - predicted_adjustment
            all_residuals.append(adjustment_error)
        else:
            actual_adjustment = np.nan
            sa_actual = np.nan
            adjustment_error = np.nan

        # Calculate prediction error
        error = sa_actual - sa_pred if pd.notna(sa_actual) else np.nan

        # Propagate intervals: Combine NSA and SARIMA uncertainties
        # Since SA = NSA_pred + Adj_pred, and assuming independence:
        # Var(SA) = Var(NSA_pred) + Var(Adj_pred)
        # We estimate variance from interval widths: SE ≈ (upper - lower) / (2 * z)
        # For 80% CI: z ≈ 1.28, so SE ≈ (upper_80 - lower_80) / 2.56

        # Get NSA interval widths (if available)
        # Note: row is a pandas Series from iterrows(), use [] access
        nsa_lower_50 = row['lower_50'] if 'lower_50' in row.index else nsa_pred
        nsa_upper_50 = row['upper_50'] if 'upper_50' in row.index else nsa_pred
        nsa_lower_80 = row['lower_80'] if 'lower_80' in row.index else nsa_pred
        nsa_upper_80 = row['upper_80'] if 'upper_80' in row.index else nsa_pred
        nsa_lower_95 = row['lower_95'] if 'lower_95' in row.index else nsa_pred
        nsa_upper_95 = row['upper_95'] if 'upper_95' in row.index else nsa_pred

        # Calculate combined intervals using variance addition
        # For 50% CI (z ≈ 0.67): half-width
        nsa_hw_50 = (nsa_upper_50 - nsa_lower_50) / 2
        adj_hw_50 = (sarima_result['upper_50'] - sarima_result['lower_50']) / 2
        combined_hw_50 = np.sqrt(nsa_hw_50**2 + adj_hw_50**2)

        # For 80% CI (z ≈ 1.28)
        nsa_hw_80 = (nsa_upper_80 - nsa_lower_80) / 2
        adj_hw_80 = (sarima_result['upper_80'] - sarima_result['lower_80']) / 2
        combined_hw_80 = np.sqrt(nsa_hw_80**2 + adj_hw_80**2)

        # For 95% CI (z ≈ 1.96)
        nsa_hw_95 = (nsa_upper_95 - nsa_lower_95) / 2
        adj_hw_95 = (sarima_result['upper_95'] - sarima_result['lower_95']) / 2
        combined_hw_95 = np.sqrt(nsa_hw_95**2 + adj_hw_95**2)

        lower_50 = sa_pred - combined_hw_50
        upper_50 = sa_pred + combined_hw_50
        lower_80 = sa_pred - combined_hw_80
        upper_80 = sa_pred + combined_hw_80
        lower_95 = sa_pred - combined_hw_95
        upper_95 = sa_pred + combined_hw_95

        # Coverage checks
        in_50 = lower_50 <= sa_actual <= upper_50 if pd.notna(sa_actual) else np.nan
        in_80 = lower_80 <= sa_actual <= upper_80 if pd.notna(sa_actual) else np.nan
        in_95 = lower_95 <= sa_actual <= upper_95 if pd.notna(sa_actual) else np.nan

        results.append({
            'date': target_date,
            'actual': sa_actual,
            'pred': sa_pred,
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
            'nsa_pred': nsa_pred,
            'predicted_adjustment': predicted_adjustment,
            'actual_adjustment': actual_adjustment,
            'adjustment_error': adjustment_error,
            'n_train_samples': len(train_adjustment),
            'sarima_order': str(best_order),
            'sarima_seasonal_order': str(best_seasonal_order)
        })

        # Progress logging
        if (idx + 1) % 6 == 0 or idx == 0 or idx == len(nsa_preds) - 1:
            if pd.notna(sa_actual):
                logger.info(f"[{idx + 1}/{len(nsa_preds)}] {target_date.strftime('%Y-%m')}: "
                           f"SA_pred={sa_pred:.0f}, SA_actual={sa_actual:.0f}, "
                           f"Adj_pred={predicted_adjustment:.0f}, Adj_actual={actual_adjustment:.0f}")
            else:
                logger.info(f"[{idx + 1}/{len(nsa_preds)}] {target_date.strftime('%Y-%m')}: "
                           f"SA_pred={sa_pred:.0f} (FUTURE), Adj_pred={predicted_adjustment:.0f}")

    results_df = pd.DataFrame(results)

    # Log summary statistics
    if not results_df.empty:
        backtest_rows = results_df[~results_df['error'].isna()]
        future_rows = results_df[results_df['error'].isna()]

        if not backtest_rows.empty:
            rmse = np.sqrt(np.mean(backtest_rows['error'] ** 2))
            mae = np.mean(np.abs(backtest_rows['error']))
            adj_rmse = np.sqrt(np.mean(backtest_rows['adjustment_error'] ** 2))
            adj_mae = np.mean(np.abs(backtest_rows['adjustment_error']))

            logger.info("\n" + "=" * 60)
            logger.info("SARIMA SA BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Predictions: {len(backtest_rows)} backtest, {len(future_rows)} future")
            logger.info(f"\nSA Prediction Metrics:")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"\nAdjustment Prediction Metrics:")
            logger.info(f"  RMSE: {adj_rmse:.2f}")
            logger.info(f"  MAE: {adj_mae:.2f}")
            logger.info(f"\nInterval Coverage:")
            # Convert object dtype to numeric for proper mean calculation
            cov_50 = backtest_rows['in_50_interval'].astype(float).mean() * 100
            cov_80 = backtest_rows['in_80_interval'].astype(float).mean() * 100
            cov_95 = backtest_rows['in_95_interval'].astype(float).mean() * 100
            logger.info(f"  50% CI: {cov_50:.1f}% (target: 50%)")
            logger.info(f"  80% CI: {cov_80:.1f}% (target: 80%)")
            logger.info(f"  95% CI: {cov_95:.1f}% (target: 95%)")

        if not future_rows.empty:
            logger.info("\nFuture Predictions:")
            for _, row in future_rows.iterrows():
                logger.info(f"  {row['date'].strftime('%Y-%m')}: SA={row['pred']:.0f} "
                           f"[{row['lower_80']:.0f}, {row['upper_80']:.0f}]")

    return results_df


def train_and_evaluate_sarima(
    nsa_predictions_path: str = None,
    archive_results: bool = False
):
    """
    Main function to train SARIMA model and evaluate SA predictions.

    Note: archive_results defaults to False because this model depends on
    NSA predictions already existing in the output directory.

    Args:
        nsa_predictions_path: Path to NSA predictions
        archive_results: If True, archive previous results (default False)
    """
    logger.info("=" * 60)
    logger.info("SARIMA SA PREDICTION MODEL - Training & Evaluation")
    logger.info("=" * 60)

    # Archive previous results if needed (disabled by default for SARIMA)
    if archive_results:
        try:
            sys.path.append(str(Path(__file__).resolve().parent))
            from backtest_archiver import prepare_new_backtest_run

            archive_path = prepare_new_backtest_run()
            if archive_path:
                logger.info(f"Archived previous backtest to: {archive_path.name}")
        except Exception as e:
            logger.warning(f"Failed to archive: {e}")

    # Run backtest
    results_df = run_sarima_backtest(nsa_predictions_path)

    if results_df.empty:
        logger.error("Backtest produced no results")
        return

    # Save results
    results_dir = OUTPUT_DIR / "backtest_results" / "sa_first"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions CSV (same format as NSA)
    predictions_output = results_df[[
        'date', 'actual', 'pred', 'error',
        'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_95', 'upper_95',
        'in_50_interval', 'in_80_interval', 'in_95_interval',
        'n_train_samples'
    ]].copy()

    # Add n_features column (for compatibility - SARIMA doesn't use features in same way)
    predictions_output['n_features'] = 0

    predictions_path = results_dir / "predictions.csv"
    predictions_output.to_csv(predictions_path, index=False)
    logger.info(f"\nSaved predictions to {predictions_path}")

    # Save detailed results (including adjustment info)
    detailed_path = results_dir / "predictions_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results to {detailed_path}")

    # Save backtest results parquet
    backtest_results = results_df.rename(columns={'date': 'ds', 'pred': 'predicted'})
    backtest_path = results_dir / "backtest_results_sa_first.parquet"
    backtest_results.to_parquet(backtest_path, index=False)
    logger.info(f"Saved backtest parquet to {backtest_path}")

    # Calculate and save summary statistics
    backtest_rows = results_df[~results_df['error'].isna()]

    if not backtest_rows.empty:
        summary = {
            'target_type': 'sa',
            'release_type': 'first',
            'model_type': 'SARIMA',
            'model_id': 'sa_first_sarima',
            'n_predictions': len(backtest_rows),
            'rmse': np.sqrt(np.mean(backtest_rows['error'] ** 2)),
            'mae': np.mean(np.abs(backtest_rows['error'])),
            'adjustment_rmse': np.sqrt(np.mean(backtest_rows['adjustment_error'] ** 2)),
            'adjustment_mae': np.mean(np.abs(backtest_rows['adjustment_error'])),
            'coverage_50': backtest_rows['in_50_interval'].astype(float).mean() * 100,
            'coverage_80': backtest_rows['in_80_interval'].astype(float).mean() * 100,
            'coverage_95': backtest_rows['in_95_interval'].astype(float).mean() * 100,
            'backtest_start': str(backtest_rows['date'].min().date()),
            'backtest_end': str(backtest_rows['date'].max().date())
        }

        summary_path = results_dir / "model_summary_sa_first.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        logger.info(f"Saved model summary to {summary_path}")

    # Generate backtest report
    logger.info("\n" + "=" * 60)
    logger.info("Generating Backtest Report")
    logger.info("=" * 60)

    try:
        report_df = results_df[['date', 'actual', 'pred', 'error',
                                'lower_50', 'upper_50', 'lower_80', 'upper_80',
                                'lower_95', 'upper_95', 'in_50_interval',
                                'in_80_interval', 'in_95_interval']].copy()

        report_files = generate_backtest_report(
            predictions_df=report_df,
            selected_features=[],  # SARIMA doesn't have feature importance
            output_dir=results_dir
        )

        logger.info(f"Generated {len(report_files)} report files")

    except Exception as e:
        logger.warning(f"Failed to generate backtest report: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("SARIMA SA Training Complete")
    logger.info("=" * 60)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='SARIMA-based SA NFP Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SARIMA model for SA predictions
  python Train/train_sarima_sa.py --train

  # Run with custom NSA predictions path
  python Train/train_sarima_sa.py --train --nsa-path path/to/predictions.csv

  # Run grid search at each step (slow but optimal)
  python Train/train_sarima_sa.py --train --rerun-grid-search

  # Use fixed SARIMA parameters
  python Train/train_sarima_sa.py --train --order 1,0,1 --seasonal-order 1,0,1,12
        """
    )

    parser.add_argument('--train', action='store_true', help='Train and evaluate model')
    parser.add_argument('--nsa-path', type=str, default=None, help='Path to NSA predictions CSV')
    parser.add_argument('--rerun-grid-search', action='store_true',
                        help='Re-run grid search periodically during backtest')
    parser.add_argument('--order', type=str, default=None,
                        help='Fixed ARIMA order (p,d,q), e.g., 1,0,1')
    parser.add_argument('--seasonal-order', type=str, default=None,
                        help='Fixed seasonal order (P,D,Q,s), e.g., 1,0,1,12')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze adjustment data (no backtest)')

    args = parser.parse_args()

    if args.analyze_only:
        adjustment_df = load_adjustment_data()
        adjustment_series = adjustment_df.set_index('ds')['adjustment_mom']

        print("\nAdjustment Statistics by Month:")
        adjustment_df['month'] = pd.to_datetime(adjustment_df['ds']).dt.month
        print(adjustment_df.groupby('month')['adjustment_mom'].agg(['mean', 'std', 'count']))

        analyze_adjustment_stationarity(adjustment_series)

    elif args.train:
        fixed_order = None
        fixed_seasonal_order = None

        if args.order:
            fixed_order = tuple(map(int, args.order.split(',')))
        if args.seasonal_order:
            parts = list(map(int, args.seasonal_order.split(',')))
            fixed_seasonal_order = tuple(parts)

        if fixed_order and fixed_seasonal_order:
            results = run_sarima_backtest(
                args.nsa_path,
                rerun_grid_search=False,
                fixed_order=fixed_order,
                fixed_seasonal_order=fixed_seasonal_order
            )
        elif args.rerun_grid_search:
            results = run_sarima_backtest(
                args.nsa_path,
                rerun_grid_search=True
            )
        else:
            results = train_and_evaluate_sarima(args.nsa_path)
    else:
        # Default: run full training and evaluation
        train_and_evaluate_sarima()
