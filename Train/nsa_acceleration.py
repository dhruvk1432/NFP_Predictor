"""
NSA Acceleration Feature Extractor for SA Branch

Computes point-in-time-safe features from NSA OOS backtest predictions
AND NSA revised target actuals for injection into the SA LightGBM model.

Key insight: NSA and SA acceleration have low direct correlation (~0.22)
because seasonal adjustment fundamentally transforms the dynamics. So we
provide features that let the SA model learn the NSA→SA mapping:

1. NSA model's predicted acceleration (what the NSA model thinks)
2. NSA revised target actuals' acceleration (ground truth NSA dynamics)
3. Historical NSA→SA acceleration relationship (bridging signal)
4. NSA-SA gap dynamics (seasonal adjustment pattern)

The SA model's goal is to predict SA acceleration correctly.  These
features give it the NSA signal + context to translate it.

All features use only actuals through t-1 — strict PIT safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_nsa_revised_target() -> pd.DataFrame:
    """Load NSA revised target with y_mom and acceleration columns."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from settings import DATA_PATH

    path = DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["ds", "y_mom", "acceleration"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)


def _load_sa_revised_target() -> pd.DataFrame:
    """Load SA revised target with y_mom and acceleration columns."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from settings import DATA_PATH

    path = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["ds", "y_mom", "acceleration"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)


# Module-level cache (loaded once)
_NSA_TARGET_CACHE: Optional[pd.DataFrame] = None
_SA_TARGET_CACHE: Optional[pd.DataFrame] = None

_NAN_FEATURES = {
    "nsa_pred_delta": np.nan,
    "nsa_pred_accel": np.nan,
    "nsa_pred_direction": np.nan,
    "nsa_actual_accel": np.nan,
    "nsa_accel_accuracy_12m": np.nan,
    "nsa_residual_trend_6m": np.nan,
    "nsa_sa_accel_corr_12m": np.nan,
    "nsa_sa_gap_delta": np.nan,
}


def compute_nsa_acceleration_features(
    nsa_backtest_df: pd.DataFrame,
    target_month: pd.Timestamp,
) -> Dict[str, float]:
    """
    Compute NSA acceleration features for a single SA backtest step.

    Uses NSA backtest predictions + NSA/SA revised target actuals,
    all strictly before ``target_month`` (PIT-safe).

    Features produced:
        nsa_pred_delta:         NSA model's predicted MoM change (pred[t] - actual_nsa[t-1])
        nsa_pred_accel:         NSA model's predicted acceleration (pred_delta[t] - actual_delta[t-1])
        nsa_pred_direction:     Sign of nsa_pred_delta (+1/-1/0)
        nsa_actual_accel:       Actual NSA acceleration at t-1 (from revised target)
        nsa_accel_accuracy_12m: Rolling 12m accuracy of NSA model's acceleration predictions
        nsa_residual_trend_6m:  Slope of NSA residuals over last 6 months
        nsa_sa_accel_corr_12m:  Rolling 12m correlation of NSA vs SA acceleration (bridging signal)
        nsa_sa_gap_delta:       Change in (SA_y_mom - NSA_y_mom) gap at t-1 vs t-2

    Args:
        nsa_backtest_df: NSA backtest results with 'ds', 'actual', 'predicted'.
        target_month: The month being predicted by the SA model.

    Returns:
        Dict of feature_name -> value. NaN when insufficient history.
    """
    global _NSA_TARGET_CACHE, _SA_TARGET_CACHE

    features: Dict[str, float] = {}

    # Load revised targets (cached)
    if _NSA_TARGET_CACHE is None:
        _NSA_TARGET_CACHE = _load_nsa_revised_target()
    if _SA_TARGET_CACHE is None:
        _SA_TARGET_CACHE = _load_sa_revised_target()

    nsa_target = _NSA_TARGET_CACHE
    sa_target = _SA_TARGET_CACHE

    # Filter backtest to rows before target_month
    df = nsa_backtest_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    hist = df[df["ds"] < target_month].sort_values("ds").reset_index(drop=True)

    # Current NSA prediction for target_month
    current = df[df["ds"] == target_month]
    has_current_pred = not current.empty and pd.notna(current.iloc[0].get("predicted"))

    if hist.empty or not has_current_pred:
        return dict(_NAN_FEATURES)

    nsa_pred_now = float(current.iloc[0]["predicted"])

    # NSA revised actuals available before target_month (PIT-safe)
    nsa_hist = nsa_target[nsa_target["ds"] < target_month].sort_values("ds")
    nsa_hist = nsa_hist[nsa_hist["y_mom"].notna()]

    # SA revised actuals available before target_month (PIT-safe)
    sa_hist = sa_target[sa_target["ds"] < target_month].sort_values("ds")
    sa_hist = sa_hist[sa_hist["y_mom"].notna()]

    if nsa_hist.empty:
        return dict(_NAN_FEATURES)

    last_nsa_actual = float(nsa_hist.iloc[-1]["y_mom"])

    # ── 1. NSA predicted delta: pred[t] - actual_nsa[t-1] ──
    nsa_pred_delta = nsa_pred_now - last_nsa_actual
    features["nsa_pred_delta"] = nsa_pred_delta

    # ── 2. NSA predicted acceleration (second derivative) ──
    if len(nsa_hist) >= 2:
        prev_nsa_actual = float(nsa_hist.iloc[-2]["y_mom"])
        actual_delta_prev = last_nsa_actual - prev_nsa_actual
        features["nsa_pred_accel"] = nsa_pred_delta - actual_delta_prev
    else:
        features["nsa_pred_accel"] = np.nan

    # ── 3. NSA predicted direction ──
    features["nsa_pred_direction"] = float(np.sign(nsa_pred_delta))

    # ── 4. Actual NSA acceleration at t-1 (from revised target) ──
    if "acceleration" in nsa_hist.columns and pd.notna(nsa_hist.iloc[-1].get("acceleration")):
        features["nsa_actual_accel"] = float(nsa_hist.iloc[-1]["acceleration"])
    elif len(nsa_hist) >= 2:
        features["nsa_actual_accel"] = float(
            nsa_hist.iloc[-1]["y_mom"] - nsa_hist.iloc[-2]["y_mom"]
        )
    else:
        features["nsa_actual_accel"] = np.nan

    # ── 5. Rolling 12m NSA acceleration accuracy (credibility weight) ──
    # How often did the NSA model correctly predict the DIRECTION of NSA acceleration?
    hist_with_actual = hist[hist["actual"].notna()]
    recent = hist_with_actual.tail(13)
    if len(recent) >= 3:
        actuals = recent["actual"].values.astype(float)
        preds = recent["predicted"].values.astype(float)
        actual_diffs = np.diff(actuals)
        pred_diffs = np.diff(preds)
        n = min(len(actual_diffs), len(pred_diffs))
        if n >= 2:
            accel_correct = np.sign(actual_diffs[:n]) == np.sign(pred_diffs[:n])
            features["nsa_accel_accuracy_12m"] = float(np.mean(accel_correct))
        else:
            features["nsa_accel_accuracy_12m"] = np.nan
    else:
        features["nsa_accel_accuracy_12m"] = np.nan

    # ── 6. NSA residual trend over last 6 months ──
    recent_6m = hist_with_actual.tail(6)
    if len(recent_6m) >= 3:
        residuals = (
            recent_6m["actual"].values.astype(float)
            - recent_6m["predicted"].values.astype(float)
        )
        x = np.arange(len(residuals), dtype=float)
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered ** 2))
        if denom > 1e-12:
            features["nsa_residual_trend_6m"] = float(
                np.sum(x_centered * residuals) / denom
            )
        else:
            features["nsa_residual_trend_6m"] = 0.0
    else:
        features["nsa_residual_trend_6m"] = np.nan

    # ── 7. Rolling 12m correlation of NSA vs SA acceleration (bridging signal) ──
    # This tells the SA model how predictive NSA acceleration has been for SA acceleration recently
    if not nsa_hist.empty and not sa_hist.empty:
        merged = nsa_hist[["ds", "y_mom"]].merge(
            sa_hist[["ds", "y_mom"]], on="ds", suffixes=("_nsa", "_sa")
        ).sort_values("ds").tail(13)

        if len(merged) >= 4:
            nsa_accels = np.diff(merged["y_mom_nsa"].values.astype(float))
            sa_accels = np.diff(merged["y_mom_sa"].values.astype(float))
            n = min(len(nsa_accels), len(sa_accels))
            if n >= 3:
                nsa_a = nsa_accels[:n]
                sa_a = sa_accels[:n]
                s_nsa = np.std(nsa_a)
                s_sa = np.std(sa_a)
                if s_nsa > 1e-12 and s_sa > 1e-12:
                    c = np.corrcoef(nsa_a, sa_a)[0, 1]
                    features["nsa_sa_accel_corr_12m"] = float(c) if np.isfinite(c) else 0.0
                else:
                    features["nsa_sa_accel_corr_12m"] = 0.0
            else:
                features["nsa_sa_accel_corr_12m"] = np.nan
        else:
            features["nsa_sa_accel_corr_12m"] = np.nan
    else:
        features["nsa_sa_accel_corr_12m"] = np.nan

    # ── 8. NSA-SA gap delta (seasonal adjustment dynamics) ──
    # Change in the gap = (SA_y_mom - NSA_y_mom) at t-1 vs t-2
    if not nsa_hist.empty and not sa_hist.empty:
        merged_gap = nsa_hist[["ds", "y_mom"]].merge(
            sa_hist[["ds", "y_mom"]], on="ds", suffixes=("_nsa", "_sa")
        ).sort_values("ds")

        if len(merged_gap) >= 2:
            gaps = (merged_gap["y_mom_sa"] - merged_gap["y_mom_nsa"]).values.astype(float)
            features["nsa_sa_gap_delta"] = float(gaps[-1] - gaps[-2])
        else:
            features["nsa_sa_gap_delta"] = np.nan
    else:
        features["nsa_sa_gap_delta"] = np.nan

    return features


def build_nsa_features_for_training(
    nsa_backtest_df: pd.DataFrame,
    training_months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build NSA acceleration features for all training months.

    Used to populate X_train with NSA features before the SA model trains.
    Each row uses only data available before that month (PIT-safe).

    Args:
        nsa_backtest_df: Full NSA backtest results.
        training_months: DatetimeIndex of months in the SA training set.

    Returns:
        DataFrame indexed by month with NSA acceleration feature columns.
    """
    rows = []
    for month in training_months:
        feats = compute_nsa_acceleration_features(nsa_backtest_df, month)
        feats["ds"] = month
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ds")
