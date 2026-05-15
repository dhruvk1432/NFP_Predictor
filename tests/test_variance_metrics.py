"""Tests for Train/variance_metrics.py."""

import numpy as np

from Train.variance_metrics import (
    compute_variance_kpis,
    composite_objective_score,
)


def test_compute_variance_kpis_detects_variance_collapse():
    rng = np.random.RandomState(0)
    actual = rng.randn(200) * 100.0
    predicted = np.full_like(actual, actual.mean())  # collapsed

    k = compute_variance_kpis(actual, predicted)

    # Variance collapse is caught by std_ratio / diff_std_ratio.
    assert k["std_ratio"] < 0.2
    assert k["diff_std_ratio"] < 0.2
    # diff_sign_accuracy uses the operational "vs last actual" formula now
    # (sign(p[m] - a[m-1]) vs sign(a[m] - a[m-1])). A flat-mean predictor
    # actually scores ~0.7 on a zero-mean random series because mean
    # reversion lines up directionally, so we no longer assert it as a
    # collapse signal — the std-based KPIs catch collapse on their own.


def test_diff_sign_accuracy_hostile_predictor_scores_low():
    """A predictor that systematically *opposes* the direction-from-prior-actual
    should score below 0.5 under the new formula."""
    rng = np.random.RandomState(2)
    actual = np.cumsum(rng.randn(200)) * 10.0  # random walk
    # Predict actual_prev minus the true change → guaranteed-wrong direction.
    predicted = np.empty_like(actual)
    predicted[0] = actual[0]
    predicted[1:] = actual[:-1] - (actual[1:] - actual[:-1])

    k = compute_variance_kpis(actual, predicted)
    assert k["diff_sign_accuracy"] <= 0.2


def test_compute_variance_kpis_well_calibrated_series():
    rng = np.random.RandomState(1)
    actual = rng.randn(300) * 50.0
    predicted = actual + rng.randn(300) * 5.0

    k = compute_variance_kpis(actual, predicted)

    assert 0.7 <= k["std_ratio"] <= 1.3
    assert k["corr_level"] > 0.9
    assert k["diff_sign_accuracy"] >= 0.6


def test_composite_objective_penalizes_collapse():
    mae = 50.0
    score_good = composite_objective_score(
        mae=mae,
        std_ratio=0.95,
        diff_std_ratio=0.90,
        tail_mae=60.0,
        corr_diff=0.8,
        diff_sign_accuracy=0.8,
        lambda_std_ratio=25.0,
        lambda_diff_std_ratio=25.0,
        lambda_tail_mae=0.2,
        lambda_corr_diff=10.0,
        lambda_diff_sign=10.0,
    )
    score_bad = composite_objective_score(
        mae=mae,
        std_ratio=0.15,
        diff_std_ratio=0.10,
        tail_mae=60.0,
        corr_diff=-0.2,
        diff_sign_accuracy=0.4,
        lambda_std_ratio=25.0,
        lambda_diff_std_ratio=25.0,
        lambda_tail_mae=0.2,
        lambda_corr_diff=10.0,
        lambda_diff_sign=10.0,
    )

    assert score_bad > score_good
