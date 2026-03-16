"""
Variance-capture diagnostics and composite objective utilities.

These helpers are used in tuning and walk-forward evaluation to prevent
variance-collapse where predictions track central drift but miss month-to-month
amplitude.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x.astype(float), ddof=0))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def compute_variance_kpis(
    actual: np.ndarray,
    predicted: np.ndarray,
    tail_quantile: float = 0.75,
    extreme_quantile: float = 0.90,
) -> Dict[str, float]:
    """
    Compute amplitude/swing diagnostics for OOS predictions.

    Args:
        actual: Actual target values.
        predicted: Predicted target values.
        tail_quantile: Quantile on |actual| used for tail MAE.
        extreme_quantile: Quantile on |actual| used for extreme-hit recall.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a = a[mask]
    p = p[mask]

    if a.size == 0:
        return {
            "std_actual": 0.0,
            "std_pred": 0.0,
            "std_ratio": 0.0,
            "diff_std_actual": 0.0,
            "diff_std_pred": 0.0,
            "diff_std_ratio": 0.0,
            "corr_level": 0.0,
            "corr_diff": 0.0,
            "diff_sign_accuracy": 0.0,
            "tail_mae": 0.0,
            "extreme_hit_rate": 0.0,
            "n_tail": 0,
            "n_extreme": 0,
        }

    err = a - p
    std_a = _safe_std(a)
    std_p = _safe_std(p)
    std_ratio = float(std_p / std_a) if std_a > 1e-12 else 0.0

    da = np.diff(a)
    dp = np.diff(p)
    dstd_a = _safe_std(da)
    dstd_p = _safe_std(dp)
    diff_std_ratio = float(dstd_p / dstd_a) if dstd_a > 1e-12 else 0.0

    corr_level = _safe_corr(a, p)
    corr_diff = _safe_corr(da, dp) if da.size >= 2 else 0.0
    if da.size > 0:
        diff_sign_accuracy = float(np.mean(np.sign(da) == np.sign(dp)))
    else:
        diff_sign_accuracy = 0.0

    abs_a = np.abs(a)
    abs_err = np.abs(err)

    tail_thr = float(np.quantile(abs_a, tail_quantile))
    tail_mask = abs_a >= tail_thr
    if tail_mask.any():
        tail_mae = float(abs_err[tail_mask].mean())
        n_tail = int(tail_mask.sum())
    else:
        tail_mae = float(abs_err.mean())
        n_tail = 0

    ext_thr = float(np.quantile(abs_a, extreme_quantile))
    ext_actual = abs_a >= ext_thr
    ext_pred = np.abs(p) >= ext_thr
    n_extreme = int(ext_actual.sum())
    if n_extreme > 0:
        extreme_hit_rate = float((ext_actual & ext_pred).sum() / n_extreme)
    else:
        extreme_hit_rate = 0.0

    return {
        "std_actual": std_a,
        "std_pred": std_p,
        "std_ratio": std_ratio,
        "diff_std_actual": dstd_a,
        "diff_std_pred": dstd_p,
        "diff_std_ratio": diff_std_ratio,
        "corr_level": corr_level,
        "corr_diff": corr_diff,
        "diff_sign_accuracy": diff_sign_accuracy,
        "tail_mae": tail_mae,
        "extreme_hit_rate": extreme_hit_rate,
        "n_tail": n_tail,
        "n_extreme": n_extreme,
    }


def composite_objective_score(
    mae: float,
    std_ratio: float,
    diff_std_ratio: float,
    tail_mae: float,
    lambda_std_ratio: float,
    lambda_diff_std_ratio: float,
    lambda_tail_mae: float,
    corr_diff: float = 0.0,
    diff_sign_accuracy: float = 0.0,
    lambda_corr_diff: float = 0.0,
    lambda_diff_sign: float = 0.0,
    accel_accuracy: float = 0.0,
    lambda_accel: float = 0.0,
    dir_accuracy: float = 0.0,
    lambda_dir: float = 0.0,
) -> float:
    """
    Composite score minimizing MAE while penalizing variance collapse
    and rewarding acceleration/directional accuracy.
    """
    return float(
        mae
        + lambda_std_ratio * abs(1.0 - std_ratio)
        + lambda_diff_std_ratio * abs(1.0 - diff_std_ratio)
        + lambda_tail_mae * tail_mae
        + lambda_corr_diff * (1.0 - max(min(corr_diff, 1.0), -1.0))
        + lambda_diff_sign * (1.0 - max(min(diff_sign_accuracy, 1.0), 0.0))
        + lambda_accel * (1.0 - max(min(accel_accuracy, 1.0), 0.0))
        + lambda_dir * (1.0 - max(min(dir_accuracy, 1.0), 0.0))
    )
