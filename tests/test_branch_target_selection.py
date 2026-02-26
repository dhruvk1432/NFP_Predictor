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
