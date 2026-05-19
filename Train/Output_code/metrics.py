"""
Backtest summary statistics computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Optional
from Train.variance_metrics import compute_variance_kpis
from utils.transforms import is_covid_month


CONSENSUS_MEAN_REF_COLS = (
    "consensus_pred",
    "consensus_mean_pred",
    "Consensus_Mean",
    "NFP_Consensus_Mean",
)
CONSENSUS_MEDIAN_REF_COLS = (
    "consensus_median_pred",
    "Consensus_Median",
    "NFP_Consensus_Median",
)


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _reference_hit_block(
    df: pd.DataFrame,
    *,
    pred_col: str,
    ref_col: str,
    label: str,
    prefix: str = "",
) -> Dict[str, float]:
    """Consensus-relative hit-rate metrics for one reference forecast.

    A "hit" means the model forecast is strictly closer to actual than the
    reference forecast for the same month. Ties are counted separately.
    """
    keys = (
        f"{prefix}HitRate_vs_{label}",
        f"{prefix}TieRate_vs_{label}",
        f"{prefix}LossRate_vs_{label}",
        f"{prefix}MeanAbsErrorDelta_vs_{label}",
        f"{prefix}HitWins_vs_{label}",
        f"{prefix}HitLosses_vs_{label}",
        f"{prefix}HitTies_vs_{label}",
        f"{prefix}HitN_vs_{label}",
    )
    if df.empty or pred_col not in df.columns or ref_col not in df.columns:
        return {k: float("nan") for k in keys}

    work = df[["actual", pred_col, ref_col]].copy()
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work[ref_col] = pd.to_numeric(work[ref_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return {k: float("nan") for k in keys}

    model_abs = (work[pred_col] - work["actual"]).abs()
    ref_abs = (work[ref_col] - work["actual"]).abs()
    wins = model_abs < ref_abs
    ties = model_abs == ref_abs
    losses = model_abs > ref_abs
    n = int(len(work))
    return {
        f"{prefix}HitRate_vs_{label}": float(wins.mean()),
        f"{prefix}TieRate_vs_{label}": float(ties.mean()),
        f"{prefix}LossRate_vs_{label}": float(losses.mean()),
        f"{prefix}MeanAbsErrorDelta_vs_{label}": float((ref_abs - model_abs).mean()),
        f"{prefix}HitWins_vs_{label}": int(wins.sum()),
        f"{prefix}HitLosses_vs_{label}": int(losses.sum()),
        f"{prefix}HitTies_vs_{label}": int(ties.sum()),
        f"{prefix}HitN_vs_{label}": n,
    }


def add_consensus_hit_rate_metrics(
    metrics: Dict[str, float],
    results_df: pd.DataFrame,
    *,
    pred_col: str = "predicted",
    exclude_covid_for_hitrate: bool = True,
) -> Dict[str, float]:
    """Attach hit-rate metrics versus consensus mean/median references.

    The non-COVID block excludes only ``utils.transforms.COVID_EXCLUDE_MONTHS``.
    Missing reference columns are tolerated so older artifacts remain readable.
    """
    if results_df.empty or "actual" not in results_df.columns or pred_col not in results_df.columns:
        return metrics

    refs = {
        "ConsensusMean": _first_existing_col(results_df, CONSENSUS_MEAN_REF_COLS),
        "ConsensusMedian": _first_existing_col(results_df, CONSENSUS_MEDIAN_REF_COLS),
    }
    for label, ref_col in refs.items():
        if ref_col is None:
            continue
        metrics.update(_reference_hit_block(
            results_df,
            pred_col=pred_col,
            ref_col=ref_col,
            label=label,
            prefix="",
        ))
        if exclude_covid_for_hitrate and "ds" in results_df.columns:
            ds = pd.to_datetime(results_df["ds"], errors="coerce")
            non_covid = results_df[~is_covid_month(ds)].copy()
            metrics.update(_reference_hit_block(
                non_covid,
                pred_col=pred_col,
                ref_col=ref_col,
                label=f"{label}_NonCovid",
                prefix="",
            ))
    return metrics


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
    out = add_consensus_hit_rate_metrics(out, backtest)
    return out


def save_metrics_csv(metrics: Dict[str, float], save_path: Path) -> None:
    """Write metrics dict to a single-row CSV."""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
