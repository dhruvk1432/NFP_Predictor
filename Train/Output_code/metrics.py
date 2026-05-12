"""
Backtest summary statistics computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from Train.variance_metrics import compute_variance_kpis
from utils.transforms import is_covid_month


def _metric_block(df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """
    Compute RMSE/MAE/MSE + variance KPIs over an arbitrary stratum of a
    backtest results DataFrame.

    Args:
        df: DataFrame with at least an 'error' column. If 'actual' and
            'predicted' are also present, the variance KPIs are added.
        prefix: String prepended to every output key (e.g. "NonCovid_").
            Empty string preserves the legacy schema.

    Returns:
        Dict of metric_name -> float. Returns NaN-filled entries when df is
        empty so the consumer always sees the expected keys.
    """
    metric_keys = (
        "RMSE", "MAE", "MSE",
        "STD_Actual", "STD_Pred", "STD_Ratio",
        "Diff_STD_Actual", "Diff_STD_Pred", "Diff_STD_Ratio",
        "Corr_Level", "Corr_Diff", "Diff_Sign_Accuracy",
        "Tail_MAE", "Extreme_Hit_Rate",
    )
    if df.empty:
        return {f"{prefix}{k}": float("nan") for k in metric_keys}

    errors = df["error"].values
    block = {
        f"{prefix}RMSE": float(np.sqrt(np.mean(errors ** 2))),
        f"{prefix}MAE": float(np.mean(np.abs(errors))),
        f"{prefix}MSE": float(np.mean(errors ** 2)),
    }
    if {'actual', 'predicted'}.issubset(df.columns):
        kpis = compute_variance_kpis(
            df['actual'].values, df['predicted'].values,
        )
        block.update({
            f"{prefix}STD_Actual": float(kpis["std_actual"]),
            f"{prefix}STD_Pred": float(kpis["std_pred"]),
            f"{prefix}STD_Ratio": float(kpis["std_ratio"]),
            f"{prefix}Diff_STD_Actual": float(kpis["diff_std_actual"]),
            f"{prefix}Diff_STD_Pred": float(kpis["diff_std_pred"]),
            f"{prefix}Diff_STD_Ratio": float(kpis["diff_std_ratio"]),
            f"{prefix}Corr_Level": float(kpis["corr_level"]),
            f"{prefix}Corr_Diff": float(kpis["corr_diff"]),
            f"{prefix}Diff_Sign_Accuracy": float(kpis["diff_sign_accuracy"]),
            f"{prefix}Tail_MAE": float(kpis["tail_mae"]),
            f"{prefix}Extreme_Hit_Rate": float(kpis["extreme_hit_rate"]),
        })
    return block


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE, MAE, MSE (and variance KPIs) from backtest results,
    stratified into all / non-COVID / COVID-only buckets.

    The unprefixed keys (RMSE, MAE, MSE, STD_*, Corr_*, etc.) preserve the
    legacy summary_statistics.csv schema. The new NonCovid_* and CovidOnly_*
    prefixes expose the same metrics on subsets so a downstream reader can
    judge model quality independently of the COVID winsorized period (which
    can attenuate the visible error and distort the unprefixed averages).

    Args:
        results_df: DataFrame with 'error' column from walk-forward backtest.
            Future predictions (NaN error) are filtered out.

    Returns:
        Dict with metric keys plus 'N', 'N_NonCovid', 'N_Covid' counts.
    """
    backtest = results_df[results_df["error"].notna()].copy()
    if backtest.empty:
        out = _metric_block(backtest, prefix="")
        out.update(_metric_block(backtest, prefix="NonCovid_"))
        out.update(_metric_block(backtest, prefix="CovidOnly_"))
        out["N"] = 0
        out["N_NonCovid"] = 0
        out["N_Covid"] = 0
        return out

    backtest["ds"] = pd.to_datetime(backtest["ds"])
    covid_mask = is_covid_month(backtest["ds"])

    out = _metric_block(backtest, prefix="")
    out.update(_metric_block(backtest[~covid_mask], prefix="NonCovid_"))
    out.update(_metric_block(backtest[covid_mask], prefix="CovidOnly_"))
    out["N"] = int(len(backtest))
    out["N_NonCovid"] = int((~covid_mask).sum())
    out["N_Covid"] = int(covid_mask.sum())
    return out


def save_metrics_csv(metrics: Dict[str, float], save_path: Path) -> None:
    """Write metrics dict to a single-row CSV."""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
