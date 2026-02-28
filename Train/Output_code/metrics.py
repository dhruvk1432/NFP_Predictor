"""
Backtest summary statistics computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from Train.variance_metrics import compute_variance_kpis


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE, MAE, MSE from backtest results.

    Filters out future predictions (NaN actuals) before computing.

    Args:
        results_df: DataFrame with 'error' column from walk-forward backtest.

    Returns:
        Dict with keys 'RMSE', 'MAE', 'MSE'.
    """
    backtest = results_df[results_df["error"].notna()].copy()
    errors = backtest["error"].values

    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    out = {"RMSE": rmse, "MAE": mae, "MSE": mse}

    if {'actual', 'predicted'}.issubset(backtest.columns):
        kpis = compute_variance_kpis(
            backtest['actual'].values,
            backtest['predicted'].values,
        )
        out.update({
            "STD_Actual": float(kpis["std_actual"]),
            "STD_Pred": float(kpis["std_pred"]),
            "STD_Ratio": float(kpis["std_ratio"]),
            "Diff_STD_Actual": float(kpis["diff_std_actual"]),
            "Diff_STD_Pred": float(kpis["diff_std_pred"]),
            "Diff_STD_Ratio": float(kpis["diff_std_ratio"]),
            "Corr_Level": float(kpis["corr_level"]),
            "Corr_Diff": float(kpis["corr_diff"]),
            "Diff_Sign_Accuracy": float(kpis["diff_sign_accuracy"]),
            "Tail_MAE": float(kpis["tail_mae"]),
            "Extreme_Hit_Rate": float(kpis["extreme_hit_rate"]),
        })

    return out


def save_metrics_csv(metrics: Dict[str, float], save_path: Path) -> None:
    """Write metrics dict to a single-row CSV."""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
