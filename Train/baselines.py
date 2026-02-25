"""
Naive baseline predictors for NFP MoM change.

Used inside the expanding-window backtest to track whether the LightGBM model
actually adds value over simple heuristics. Baselines are computed using
ONLY data available at the prediction point (strictly before target_month).
"""

from typing import Dict

import numpy as np
import pandas as pd


def baseline_last_y(y_train: pd.Series) -> float:
    """
    Predict the next NFP MoM change as the most recent observed value.

    Args:
        y_train: Historical target values (ordered chronologically).

    Returns:
        The last non-NaN value in y_train, or NaN if no valid observations.
    """
    valid = y_train.dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def baseline_rolling_mean(y_train: pd.Series, window: int = 6) -> float:
    """
    Predict the next NFP MoM change as the rolling mean of the last N observations.

    Args:
        y_train: Historical target values (ordered chronologically).
        window: Number of trailing months to average.

    Returns:
        Mean of last ``window`` non-NaN values, or NaN if fewer than
        ``window`` valid observations are available.
    """
    valid = y_train.dropna()
    if len(valid) < window:
        return np.nan
    tail = valid.tail(window)
    return float(tail.mean())


def compute_all_baselines(
    y_train: pd.Series,
    rolling_window: int = 6,
) -> Dict[str, float]:
    """
    Compute all baseline predictions for a single backtest step.

    Args:
        y_train: Historical target values (ordered chronologically).
        rolling_window: Number of trailing months for rolling mean baseline.

    Returns:
        Dict with keys ``baseline_last_y`` and ``baseline_rolling_mean_{N}``.
    """
    return {
        'baseline_last_y': baseline_last_y(y_train),
        f'baseline_rolling_mean_{rolling_window}': baseline_rolling_mean(
            y_train, rolling_window
        ),
    }
