"""
Predicted seasonal adjustment walk-forward backtest.

Tests multiple models for predicting the seasonal adjustment factor
(SA_MoM - NSA_MoM) without lookahead bias, then combines NSA predictions
with the predicted adjustment to produce SA-space predictions.

Models tested:
  1. SARIMA(1,0,1)x(1,1,1,12)
  2. Monthly average (same calendar month, all prior years)
  3. 12-month complement (last 11 months summed, negated)
  4. Same-month last year (naive seasonal)
  5. Exponentially-weighted monthly average (recent years weighted more)
  6. Linear regression (month dummies + trend)
"""

from __future__ import annotations

import argparse
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, DATA_PATH, setup_logger
from Train.variance_metrics import compute_variance_kpis, composite_objective_score
from Train.sandbox.output_utils import write_sandbox_output_bundle

logger = setup_logger(__file__, TEMP_DIR)
OUT_DIR = OUTPUT_DIR / "sandbox" / "nsa_predicted_adjustment_revised"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adjustment_history() -> pd.DataFrame:
    """
    Load full historical adjustment series from revised target parquets.

    Returns DataFrame with columns:
        [ds, nsa_mom, sa_mom, adjustment, operational_available_date]
    """
    nsa = pd.read_parquet(DATA_PATH / "NFP_target" / "y_nsa_revised.parquet")
    sa = pd.read_parquet(DATA_PATH / "NFP_target" / "y_sa_revised.parquet")

    merged = pd.merge(
        nsa[["ds", "y_mom", "operational_available_date"]].rename(
            columns={"y_mom": "nsa_mom"}
        ),
        sa[["ds", "y_mom"]].rename(columns={"y_mom": "sa_mom"}),
        on="ds",
        how="inner",
    )
    merged["adjustment"] = merged["sa_mom"] - merged["nsa_mom"]
    merged = merged.dropna(subset=["nsa_mom", "sa_mom"]).sort_values("ds").reset_index(drop=True)
    logger.info("Loaded adjustment history: %d months (%s to %s)",
                len(merged), merged["ds"].min().date(), merged["ds"].max().date())
    return merged


def load_backtest_inputs() -> pd.DataFrame:
    """
    Load NSA revised backtest predictions and SA revised actuals.

    Returns DataFrame with columns: [ds, nsa_predicted, sa_actual]
    """
    nsa_path = OUTPUT_DIR / "NSA_prediction_revised" / "backtest_results.csv"
    sa_path = OUTPUT_DIR / "SA_prediction_revised" / "backtest_results.csv"

    nsa = pd.read_csv(nsa_path, parse_dates=["ds"])[["ds", "predicted"]].rename(
        columns={"predicted": "nsa_predicted"}
    )
    sa = pd.read_csv(sa_path, parse_dates=["ds"])[["ds", "actual"]].rename(
        columns={"actual": "sa_actual"}
    )

    merged = pd.merge(nsa, sa, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    logger.info("Loaded backtest inputs: %d months", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Adjustment prediction models
# ---------------------------------------------------------------------------

class AdjustmentPredictor(ABC):
    """Base class for seasonal adjustment prediction models."""

    @abstractmethod
    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        """Predict adjustment for target_ds using only PIT-filtered history."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SARIMAPredictor(AdjustmentPredictor):
    """SARIMA(1,0,1)x(1,1,1,12) on the adjustment series."""

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12)):
        self._order = order
        self._seasonal_order = seasonal_order

    @property
    def name(self) -> str:
        return "sarima"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        if len(history) < 36:
            return _monthly_avg_fallback(history, target_ds)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            series = history.set_index("ds")["adjustment"].asfreq("MS")
            series = series.interpolate(method="linear")  # fill any freq gaps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    series,
                    order=self._order,
                    seasonal_order=self._seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                forecast = result.get_forecast(steps=1)
                return float(forecast.predicted_mean.iloc[0])
        except Exception as e:
            logger.debug("SARIMA failed for %s: %s — falling back to monthly avg", target_ds.date(), e)
            return _monthly_avg_fallback(history, target_ds)


class MonthlyAveragePredictor(AdjustmentPredictor):
    """Mean adjustment for the target calendar month from all prior years."""

    @property
    def name(self) -> str:
        return "monthly_avg"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        return _monthly_avg_fallback(history, target_ds)


class TwelveMonthComplementPredictor(AdjustmentPredictor):
    """Sum last 11 months of adjustments, predict the negative."""

    def __init__(self, constant: float = 0.0):
        self._constant = constant

    @property
    def name(self) -> str:
        return "12m_complement"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        if len(history) < 11:
            return _monthly_avg_fallback(history, target_ds)
        last_11 = history.tail(11)["adjustment"].values
        return float(-last_11.sum() + self._constant)


class SameMonthLastYearPredictor(AdjustmentPredictor):
    """Use the adjustment from the same calendar month one year ago."""

    @property
    def name(self) -> str:
        return "same_month_ly"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        month = target_ds.month
        same_month = history[history["ds"].dt.month == month]
        if same_month.empty:
            return _monthly_avg_fallback(history, target_ds)
        return float(same_month.iloc[-1]["adjustment"])


class ExpWeightedMonthlyAvgPredictor(AdjustmentPredictor):
    """Exponentially-weighted average of same-calendar-month adjustments."""

    def __init__(self, half_life_years: float = 3.0):
        self._half_life = half_life_years

    @property
    def name(self) -> str:
        return "exp_weighted_avg"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        month = target_ds.month
        same_month = history[history["ds"].dt.month == month].copy()
        if same_month.empty:
            return _monthly_avg_fallback(history, target_ds)
        years_ago = (target_ds - same_month["ds"]).dt.days / 365.25
        decay = np.exp(-np.log(2) * years_ago / self._half_life)
        weights = decay / decay.sum()
        return float((same_month["adjustment"].values * weights.values).sum())


class LinearRegressionPredictor(AdjustmentPredictor):
    """OLS: adjustment ~ month_dummies + linear_trend."""

    @property
    def name(self) -> str:
        return "linreg_month_trend"

    def fit_predict(self, history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
        from sklearn.linear_model import LinearRegression

        df = history.copy()
        for m in range(1, 13):
            df[f"m{m}"] = (df["ds"].dt.month == m).astype(float)
        df["trend"] = (df["ds"] - df["ds"].min()).dt.days / 30.44

        feat_cols = [f"m{m}" for m in range(1, 13)] + ["trend"]
        X = df[feat_cols].values
        y = df["adjustment"].values

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        target_feat = np.zeros(13)
        target_feat[target_ds.month - 1] = 1.0
        target_feat[12] = (target_ds - history["ds"].min()).days / 30.44
        return float(model.predict(target_feat.reshape(1, -1))[0])


def _monthly_avg_fallback(history: pd.DataFrame, target_ds: pd.Timestamp) -> float:
    """Shared fallback: mean of same-calendar-month adjustments."""
    month = target_ds.month
    same = history[history["ds"].dt.month == month]
    if same.empty:
        return float(history["adjustment"].mean()) if not history.empty else 0.0
    return float(same["adjustment"].mean())


# ---------------------------------------------------------------------------
# Walk-forward backtest engine
# ---------------------------------------------------------------------------

def run_walkforward_backtest(
    adj_history: pd.DataFrame,
    backtest_inputs: pd.DataFrame,
    models: List[AdjustmentPredictor],
) -> Dict[str, pd.DataFrame]:
    """
    Walk-forward expanding window backtest for all adjustment prediction models.

    Returns dict mapping model_name -> DataFrame with columns:
        [ds, actual, nsa_predicted, predicted_adjustment, perfect_adjustment, predicted, error]
    """
    results: Dict[str, list] = {m.name: [] for m in models}

    for _, row in backtest_inputs.iterrows():
        target_ds = row["ds"]
        nsa_pred = row["nsa_predicted"]
        sa_actual = row["sa_actual"]

        # PIT filter: only use adjustments available before target month
        if "operational_available_date" in adj_history.columns:
            pit_mask = adj_history["operational_available_date"].notna() & (
                adj_history["operational_available_date"] < target_ds
            )
        else:
            pit_mask = adj_history["ds"] < target_ds
        avail_history = adj_history[pit_mask].copy()

        if avail_history.empty:
            for m in models:
                results[m.name].append({
                    "ds": target_ds, "actual": sa_actual,
                    "nsa_predicted": nsa_pred,
                    "predicted_adjustment": np.nan,
                    "perfect_adjustment": np.nan,
                    "predicted": np.nan, "error": np.nan,
                })
            continue

        # Compute perfect adjustment for this month (for diagnostic only)
        nsa_actual_row = adj_history[adj_history["ds"] == target_ds]
        perfect_adj = float(nsa_actual_row["adjustment"].iloc[0]) if not nsa_actual_row.empty and pd.notna(nsa_actual_row["adjustment"].iloc[0]) else np.nan

        for m in models:
            pred_adj = m.fit_predict(avail_history, target_ds)
            sa_pred = nsa_pred + pred_adj
            err = np.nan if pd.isna(sa_actual) else float(sa_actual - sa_pred)

            results[m.name].append({
                "ds": target_ds,
                "actual": sa_actual,
                "nsa_predicted": nsa_pred,
                "predicted_adjustment": pred_adj,
                "perfect_adjustment": perfect_adj,
                "predicted": sa_pred,
                "error": err,
            })

    return {name: pd.DataFrame(rows) for name, rows in results.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _composite_score(actual: np.ndarray, pred: np.ndarray) -> float:
    """Compute composite objective score."""
    if actual.size < 3:
        return float("inf")
    mae = float(np.mean(np.abs(actual - pred)))
    kpis = compute_variance_kpis(actual, pred)
    return composite_objective_score(
        mae=mae,
        std_ratio=float(kpis["std_ratio"]),
        diff_std_ratio=float(kpis["diff_std_ratio"]),
        tail_mae=float(kpis["tail_mae"]),
        corr_diff=float(kpis["corr_diff"]),
        diff_sign_accuracy=float(kpis["diff_sign_accuracy"]),
        lambda_std_ratio=18.0,
        lambda_diff_std_ratio=18.0,
        lambda_tail_mae=0.20,
        lambda_corr_diff=14.0,
        lambda_diff_sign=10.0,
    )


def evaluate_models(model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare all models on adjustment accuracy and final SA prediction accuracy.

    Returns comparison DataFrame with one row per model.
    """
    rows = []
    for name, df in model_results.items():
        valid = df[df["error"].notna()].copy()
        if valid.empty:
            continue

        actual = valid["actual"].values.astype(float)
        pred = valid["predicted"].values.astype(float)

        # Adjustment accuracy (where perfect is known)
        adj_valid = valid[valid["perfect_adjustment"].notna()]
        if not adj_valid.empty:
            adj_err = adj_valid["predicted_adjustment"].values - adj_valid["perfect_adjustment"].values
            adj_mae = float(np.mean(np.abs(adj_err)))
            adj_rmse = float(np.sqrt(np.mean(adj_err ** 2)))
        else:
            adj_mae = adj_rmse = np.nan

        # SA prediction accuracy
        sa_mae = float(np.mean(np.abs(actual - pred)))
        sa_rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        sa_composite = _composite_score(actual, pred)

        kpis = compute_variance_kpis(actual, pred)
        dir_correct = np.sum(np.sign(actual) == np.sign(pred))
        dir_acc = float(dir_correct / len(actual))

        rows.append({
            "model_name": name,
            "adj_mae": adj_mae,
            "adj_rmse": adj_rmse,
            "sa_mae": sa_mae,
            "sa_rmse": sa_rmse,
            "sa_composite": sa_composite,
            "std_ratio": float(kpis["std_ratio"]),
            "diff_std_ratio": float(kpis["diff_std_ratio"]),
            "corr_diff": float(kpis["corr_diff"]),
            "diff_sign_accuracy": float(kpis["diff_sign_accuracy"]),
            "directional_accuracy": dir_acc,
            "n_backtest": len(valid),
        })

    comp = pd.DataFrame(rows).sort_values("sa_composite").reset_index(drop=True)
    return comp


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    best_model_name: str,
    model_results: Dict[str, pd.DataFrame],
    comparison_df: pd.DataFrame,
) -> None:
    """Save best model bundle, comparison CSV, and all individual backtests."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Best model output bundle
    best_df = model_results[best_model_name].copy()
    write_sandbox_output_bundle(
        results_df=best_df[["ds", "actual", "predicted", "error"]],
        out_dir=OUT_DIR,
        model_id="nsa_predicted_adjustment_revised",
        diagnostics_label=f"NSA + predicted adj ({best_model_name})",
        n_features=0,
    )

    # Comparison CSV
    comparison_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model comparison (%d models) to %s",
                len(comparison_df), OUT_DIR / "model_comparison.csv")

    # Individual model backtests
    all_dir = OUT_DIR / "all_models"
    all_dir.mkdir(parents=True, exist_ok=True)
    for name, df in model_results.items():
        df.to_csv(all_dir / f"{name}_backtest.csv", index=False)

    # Config JSON
    config = {
        "selected_model": best_model_name,
        "selection_metric": "sa_composite",
        "models_evaluated": list(model_results.keys()),
        "n_backtest_months": int(comparison_df["n_backtest"].iloc[0])
            if not comparison_df.empty else 0,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved all outputs to %s", OUT_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predicted seasonal adjustment walk-forward backtest."
    )
    parser.add_argument(
        "--best-by", type=str, choices=["sa_mae", "sa_composite"],
        default="sa_composite",
        help="Metric to select the best model.",
    )
    parser.add_argument(
        "--exp-half-life", type=float, default=3.0,
        help="Half-life in years for exponential weighting model.",
    )
    parser.add_argument(
        "--complement-constant", type=float, default=0.0,
        help="Additive constant for 12-month complement model.",
    )
    args = parser.parse_args()

    adj_history = load_adjustment_history()
    backtest_inputs = load_backtest_inputs()

    models: List[AdjustmentPredictor] = [
        SARIMAPredictor(),
        MonthlyAveragePredictor(),
        TwelveMonthComplementPredictor(constant=args.complement_constant),
        SameMonthLastYearPredictor(),
        ExpWeightedMonthlyAvgPredictor(half_life_years=args.exp_half_life),
        LinearRegressionPredictor(),
    ]

    logger.info("Running walk-forward backtest with %d models over %d months",
                len(models), len(backtest_inputs))
    model_results = run_walkforward_backtest(adj_history, backtest_inputs, models)

    comparison = evaluate_models(model_results)
    logger.info("\n=== Model Comparison ===\n%s", comparison.to_string(index=False))

    # Select best
    best_col = args.best_by
    best_name = comparison.iloc[0]["model_name"]  # already sorted by sa_composite
    if best_col == "sa_mae":
        best_name = comparison.sort_values("sa_mae").iloc[0]["model_name"]
    logger.info("Selected best model: %s (by %s)", best_name, best_col)

    save_outputs(best_name, model_results, comparison)


if __name__ == "__main__":
    main()
