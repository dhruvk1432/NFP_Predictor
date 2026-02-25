"""Tests for Train/short_pass_selection.py — nested short-pass feature selection."""

import numpy as np
import pandas as pd
import pytest

from Train.short_pass_selection import (
    select_features_for_step,
    short_pass_lgbm_gain,
    short_pass_weighted_corr,
)


def _make_synthetic_data(n_rows=100, n_features=20, seed=42):
    """Create synthetic training data with known signal features."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2010-01-01", periods=n_rows, freq="MS")
    y = rng.randn(n_rows) * 100

    cols = {}
    for i in range(n_features):
        if i < 3:
            # Strong signal: correlated with y
            cols[f"signal_{i}"] = y + rng.randn(n_rows) * 20
        else:
            # Noise
            cols[f"noise_{i}"] = rng.randn(n_rows) * 100

    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y_mom"), dates


def test_no_lookahead_invariance_to_future_corruption():
    """Corrupting future data must not change features selected using only past data."""
    X, y, dates = _make_synthetic_data(n_rows=100, n_features=20, seed=42)

    T_idx = 70
    X_past = X.iloc[:T_idx].copy()
    y_past = y.iloc[:T_idx].copy()
    candidates = list(X.columns)

    selected_A = select_features_for_step(
        X_past, y_past, candidate_features=candidates, top_k=10, method='lgbm_gain', seed=42
    )

    # Corrupt future data (rows >= T_idx) — should have no effect
    rng = np.random.RandomState(99)
    X_corrupted = X.copy()
    X_corrupted.iloc[T_idx:] = rng.randn(100 - T_idx, 20) * 999
    y_corrupted = y.copy()
    y_corrupted.iloc[T_idx:] = rng.randn(100 - T_idx) * 999

    # Re-select using only past data (unchanged)
    selected_B = select_features_for_step(
        X_past, y_past, candidate_features=candidates, top_k=10, method='lgbm_gain', seed=42
    )

    assert selected_A == selected_B


def test_determinism():
    """Same inputs + seed should produce identical output."""
    X, y, _ = _make_synthetic_data(seed=42)
    candidates = list(X.columns)

    sel_1 = select_features_for_step(
        X, y, candidate_features=candidates, top_k=10, method='lgbm_gain', seed=42
    )
    sel_2 = select_features_for_step(
        X, y, candidate_features=candidates, top_k=10, method='lgbm_gain', seed=42
    )
    assert sel_1 == sel_2


def test_top_k_exact():
    """Should return exactly top_k features when candidates > top_k."""
    X, y, _ = _make_synthetic_data(n_features=150, seed=42)
    candidates = list(X.columns)

    selected = select_features_for_step(
        X, y, candidate_features=candidates, top_k=60, method='lgbm_gain', seed=42
    )
    assert len(selected) == 60


def test_handles_all_nan_column():
    """A candidate column that is entirely NaN should not appear in top_k."""
    X, y, _ = _make_synthetic_data(n_features=10, seed=42)
    X["all_nan_col"] = np.nan
    candidates = list(X.columns)

    selected = select_features_for_step(
        X, y, candidate_features=candidates, top_k=5, method='lgbm_gain', seed=42
    )
    assert "all_nan_col" not in selected


def test_both_methods_run():
    """Both lgbm_gain and weighted_corr should execute without error."""
    X, y, _ = _make_synthetic_data(n_features=20, seed=42)
    candidates = list(X.columns)

    sel_lgbm = select_features_for_step(
        X, y, candidate_features=candidates, top_k=10, method='lgbm_gain', seed=42
    )
    sel_corr = select_features_for_step(
        X, y, candidate_features=candidates, top_k=10, method='weighted_corr'
    )

    assert len(sel_lgbm) == 10
    assert len(sel_corr) == 10
    assert isinstance(sel_lgbm, list)
    assert isinstance(sel_corr, list)


def test_returns_empty_for_no_candidates():
    """Zero candidate features should return empty list."""
    X, y, _ = _make_synthetic_data(seed=42)
    selected = select_features_for_step(
        X, y, candidate_features=[], top_k=10, method='lgbm_gain', seed=42
    )
    assert selected == []


def test_weighted_corr_with_weights():
    """Weighted correlation should run correctly with sample weights."""
    X, y, _ = _make_synthetic_data(n_rows=50, n_features=10, seed=42)
    weights = np.linspace(0.1, 1.0, len(X))
    candidates = list(X.columns)

    selected = short_pass_weighted_corr(
        X, y, candidate_features=candidates, top_k=5, sample_weights=weights
    )
    assert len(selected) == 5


def test_lgbm_gain_with_weights():
    """LightGBM gain ranker should work correctly with sample weights."""
    X, y, _ = _make_synthetic_data(n_rows=50, n_features=10, seed=42)
    weights = np.linspace(0.1, 1.0, len(X))
    candidates = list(X.columns)

    selected = short_pass_lgbm_gain(
        X, y, candidate_features=candidates, top_k=5,
        sample_weights=weights, seed=42
    )
    assert len(selected) == 5


def test_unknown_method_raises():
    """Unknown method should raise ValueError."""
    X, y, _ = _make_synthetic_data(seed=42)
    with pytest.raises(ValueError, match="Unknown short-pass method"):
        select_features_for_step(
            X, y, candidate_features=list(X.columns),
            top_k=5, method='nonexistent'
        )
