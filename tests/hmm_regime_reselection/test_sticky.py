import numpy as np
import pandas as pd

from Train.hmm_regime_reselection import (
    StickySelectionConfig,
    apply_sticky_feature_selection,
)


def _frame():
    rng = np.random.default_rng(42)
    y = pd.Series(np.linspace(-2.0, 2.0, 80), name="y")
    X = pd.DataFrame({
        "weak_old": rng.normal(size=80),
        "strong_old": y + rng.normal(scale=0.02, size=80),
        "weak_new": rng.normal(size=80),
        "strong_new": y * 2.0 + rng.normal(scale=0.01, size=80),
        "another_new": -y + rng.normal(scale=0.01, size=80),
    })
    return X, y


def test_sticky_selection_preserves_incumbents_unless_margin_clears():
    X, y = _frame()
    result = apply_sticky_feature_selection(
        candidate_features=["weak_new", "strong_old"],
        incumbent_features=["weak_old", "strong_old"],
        X_train=X,
        y_train=y,
        regime_label="stable",
        config=StickySelectionConfig(
            enabled=True,
            max_features=2,
            margin_stable=10.0,
            stable_max_replacement_share=0.50,
        ),
    )

    assert "strong_old" in result.features
    assert "weak_old" in result.features
    assert "weak_new" not in result.features
    assert result.replacements == 0


def test_sticky_selection_replaces_when_challenger_beats_margin():
    X, y = _frame()
    result = apply_sticky_feature_selection(
        candidate_features=["strong_new", "strong_old"],
        incumbent_features=["weak_old", "strong_old"],
        X_train=X,
        y_train=y,
        regime_label="stable",
        config=StickySelectionConfig(
            enabled=True,
            max_features=2,
            margin_stable=0.10,
            stable_max_replacement_share=0.50,
        ),
    )

    assert "strong_new" in result.features
    assert "weak_old" not in result.features
    assert result.replacements == 1


def test_downside_regime_has_minimum_replacement_budget():
    X, y = _frame()
    result = apply_sticky_feature_selection(
        candidate_features=["another_new", "strong_new"],
        incumbent_features=["weak_old", "strong_old"],
        X_train=X,
        y_train=y,
        regime_label="volatile_down",
        config=StickySelectionConfig(
            enabled=True,
            max_features=2,
            margin_volatile=10.0,
            volatile_min_replacement_share=0.50,
        ),
    )

    assert result.replacements >= 1
    assert any(feature in result.features for feature in {"another_new", "strong_new"})


def test_sticky_selection_enforces_replacement_cap_share():
    X, y = _frame()
    result = apply_sticky_feature_selection(
        candidate_features=["strong_new", "another_new"],
        incumbent_features=["weak_old", "strong_old"],
        X_train=X,
        y_train=y,
        regime_label="stable",
        config=StickySelectionConfig(
            enabled=True,
            max_features=2,
            margin_stable=0.0,
            replacement_cap_share=0.0,
        ),
    )

    assert result.features == ["weak_old", "strong_old"]
    assert result.replacements == 0
    assert result.replacement_cap == 0


def test_challenger_slots_reserve_capacity_for_new_features():
    X, y = _frame()
    result = apply_sticky_feature_selection(
        candidate_features=["strong_new", "another_new", "strong_old"],
        incumbent_features=["weak_old", "strong_old"],
        X_train=X,
        y_train=y,
        regime_label="stable",
        config=StickySelectionConfig(
            enabled=True,
            max_features=2,
            challenger_slots=1,
            margin_stable=10.0,
            replacement_cap_share=0.0,
        ),
    )

    assert len(result.features) == 2
    assert "weak_old" in result.features
    assert any(feature in result.features for feature in {"strong_new", "another_new"})
    assert result.challenger_slots == 1
