"""Tests for Train/baselines.py — naive baseline predictors."""

import numpy as np
import pandas as pd
import pytest

from Train.baselines import baseline_last_y, baseline_rolling_mean, compute_all_baselines


def test_last_y_returns_final():
    y = pd.Series([100.0, 200.0, 300.0])
    assert baseline_last_y(y) == 300.0


def test_last_y_single_obs():
    y = pd.Series([42.0])
    assert baseline_last_y(y) == 42.0


def test_last_y_empty_returns_nan():
    y = pd.Series([], dtype=float)
    assert np.isnan(baseline_last_y(y))


def test_last_y_all_nan_returns_nan():
    y = pd.Series([np.nan, np.nan])
    assert np.isnan(baseline_last_y(y))


def test_rolling_mean():
    y = pd.Series([100.0, 200.0, 300.0, 400.0])
    assert baseline_rolling_mean(y, window=3) == pytest.approx(300.0)


def test_rolling_mean_insufficient_returns_nan():
    y = pd.Series([100.0, 200.0])
    assert np.isnan(baseline_rolling_mean(y, window=6))


def test_rolling_mean_exact_window():
    y = pd.Series([10.0, 20.0, 30.0])
    assert baseline_rolling_mean(y, window=3) == pytest.approx(20.0)


def test_handles_nan():
    y = pd.Series([100.0, np.nan, 300.0])
    assert baseline_last_y(y) == 300.0
    # Only 2 valid obs, window=2 -> mean of last 2 valid (100, 300)
    assert baseline_rolling_mean(y, window=2) == pytest.approx(200.0)


def test_compute_all_baselines_keys():
    y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    result = compute_all_baselines(y, rolling_window=6)
    assert 'baseline_last_y' in result
    assert 'baseline_rolling_mean_6' in result
    assert result['baseline_last_y'] == 700.0
    # Mean of last 6: 200..700 = 450
    assert result['baseline_rolling_mean_6'] == pytest.approx(450.0)
