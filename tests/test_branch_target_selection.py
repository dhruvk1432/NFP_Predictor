"""Tests for Train/branch_target_selection.py."""

import numpy as np
import pandas as pd

from Train.branch_target_selection import (
    partition_feature_columns,
    prune_redundant_by_correlation,
    select_branch_target_features_for_step,
)


def test_partition_feature_columns_splits_groups_correctly():
    cols = [
        "snapshot_feature_a",
        "nfp_nsa_mom_lag1",
        "nfp_sa_mom_lag1",
        "month_sin",
        "year",
        "rev_master_mean_abs",
        "is_jan",
    ]

    out = partition_feature_columns(cols, target_type="nsa")

    assert out["snapshot_features"] == ["snapshot_feature_a"]
    assert out["target_branch_features"] == ["nfp_nsa_mom_lag1"]
    assert out["other_target_features"] == ["nfp_sa_mom_lag1"]
    assert set(out["calendar_features"]) == {"month_sin", "year", "is_jan"}
    assert out["revision_features"] == ["rev_master_mean_abs"]


def test_prune_redundant_by_correlation_drops_highly_correlated_duplicate():
    rng = np.random.RandomState(7)
    n = 140
    y = pd.Series(rng.randn(n), name="y_mom")

    x1 = y.values + rng.randn(n) * 0.05
    x2 = x1 + rng.randn(n) * 0.01  # almost duplicate of x1
    x3 = rng.randn(n)              # mostly independent

    X = pd.DataFrame(
        {
            "nfp_nsa_signal_1": x1,
            "nfp_nsa_signal_2": x2,
            "nfp_nsa_noise": x3,
        }
    )

    survivors = prune_redundant_by_correlation(
        X_train=X,
        y_train=y,
        candidate_features=list(X.columns),
        corr_threshold=0.95,
        min_overlap=24,
    )

    # One of the near-duplicates should be dropped.
    assert len(survivors) == 2
    assert "nfp_nsa_noise" in survivors
    assert {"nfp_nsa_signal_1", "nfp_nsa_signal_2"} & set(survivors)


def test_select_branch_target_features_for_step_keeps_signal_features():
    rng = np.random.RandomState(11)
    n = 160
    y = pd.Series(rng.randn(n) * 100.0, name="y_mom")

    X = pd.DataFrame(
        {
            "nfp_nsa_signal_1": y.values + rng.randn(n) * 10.0,
            "nfp_nsa_signal_2": -0.5 * y.values + rng.randn(n) * 12.0,
            "nfp_nsa_noise_1": rng.randn(n) * 90.0,
            "nfp_nsa_noise_2": rng.randn(n) * 100.0,
            "snapshot_feature_a": rng.randn(n),  # non-target feature
        }
    )

    selected = select_branch_target_features_for_step(
        X_train=X,
        y_train=y,
        target_type="nsa",
        candidate_features=[
            "nfp_nsa_signal_1",
            "nfp_nsa_signal_2",
            "nfp_nsa_noise_1",
            "nfp_nsa_noise_2",
        ],
        top_k=2,
        method="weighted_corr",
        corr_threshold=0.98,
        min_overlap=24,
    )

    assert len(selected) == 2
    assert "nfp_nsa_signal_1" in selected


def test_select_branch_target_features_dynamics_composite_prefers_turning_signal():
    rng = np.random.RandomState(17)
    n = 180

    # Build a wavy target with turning points.
    t = np.arange(n)
    y = pd.Series(80.0 * np.sin(t / 6.0) + rng.randn(n) * 8.0, name="y_mom")
    y_diff = y.diff().fillna(0.0).values

    X = pd.DataFrame(
        {
            # Strong on direction / changes.
            "nfp_nsa_turn_signal": y_diff + rng.randn(n) * 0.6,
            # Mostly level-like.
            "nfp_nsa_level_signal": y.values + rng.randn(n) * 10.0,
            # Noise.
            "nfp_nsa_noise_a": rng.randn(n) * 60.0,
            "nfp_nsa_noise_b": rng.randn(n) * 70.0,
        }
    )

    selected = select_branch_target_features_for_step(
        X_train=X,
        y_train=y,
        target_type="nsa",
        candidate_features=list(X.columns),
        top_k=2,
        method="dynamics_composite",
        corr_threshold=0.98,
        min_overlap=24,
        dynamics_weight_level=0.2,
        dynamics_weight_diff=0.65,
        dynamics_weight_dir=0.15,
    )

    assert len(selected) == 2
    assert "nfp_nsa_turn_signal" in selected


def test_select_branch_target_features_dynamics_composite_uses_variance_terms():
    rng = np.random.RandomState(29)
    n = 200
    vol_regime = (np.sin(np.arange(n) / 13.0) > 0).astype(float)
    y_diff = rng.randn(n) * (2.0 + 5.0 * vol_regime)
    y = pd.Series(np.cumsum(y_diff), name="y_mom")

    random_sign = rng.choice([-1.0, 1.0], size=n)
    amp_diff = random_sign * (np.abs(y_diff) + rng.randn(n) * 0.25)
    amp_signal = np.cumsum(amp_diff)

    X = pd.DataFrame(
        {
            # Strongly linked to change amplitude (|diff(y)|), weak on level.
            "nfp_sa_amp_signal": amp_signal,
            # Mostly level-like.
            "nfp_sa_level_signal": y.values + rng.randn(n) * 8.0,
            # Noise.
            "nfp_sa_noise": rng.randn(n) * 60.0,
        }
    )

    selected = select_branch_target_features_for_step(
        X_train=X,
        y_train=y,
        target_type="sa",
        candidate_features=list(X.columns),
        top_k=1,
        method="dynamics_composite",
        corr_threshold=0.98,
        min_overlap=24,
        dynamics_weight_level=0.0,
        dynamics_weight_diff=0.10,
        dynamics_weight_dir=0.0,
        dynamics_weight_amp=0.75,
        dynamics_weight_sign=0.10,
        dynamics_weight_tail=0.05,
    )

    assert selected == ["nfp_sa_amp_signal"]
