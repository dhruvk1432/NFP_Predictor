"""
Backtest summary statistics computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


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

    return {"RMSE": rmse, "MAE": mae, "MSE": mse}


def save_metrics_csv(metrics: Dict[str, float], save_path: Path) -> None:
    """Write metrics dict to a single-row CSV."""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
