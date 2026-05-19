import numpy as np
import pandas as pd

from Train.hmm_regime_reselection import (
    HMMRegimeConfig,
    HMMRegimeSnapshot,
    decide_hmm_reselection,
    evaluate_hmm_reselection_trigger,
    known_economic_window,
    select_hmm_observation_columns,
)
from Train.hmm_regime_reselection.trigger import _pit_same_month_zscore


def _snapshot(**overrides):
    base = dict(
        step_date=pd.Timestamp("2020-04-01"),
        trained_through=pd.Timestamp("2020-03-01"),
        state=1,
        label="volatile_down",
        confidence=0.9,
        entropy=0.1,
        transition_risk=0.7,
        expected_duration=3.0,
        n_components=3,
        n_features=5,
        surprise=60.0,
        surprise_threshold=40.0,
        surprise_q90=40.0,
        surprise_q975=50.0,
        surprise_ratio=1.5,
        state_support_n=30.0,
        state_support_share=0.20,
        month_of_year=4,
    )
    base.update(overrides)
    return HMMRegimeSnapshot(**base)


def test_select_hmm_observation_columns_prefers_macro_prefixes():
    dates = pd.date_range("2010-01-01", periods=80, freq="MS")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "ds": dates,
            "random_snapshot_feature": rng.normal(size=len(dates)),
            "VIX_latest": rng.normal(size=len(dates)),
            "nfp_nsa_lag1": rng.normal(size=len(dates)),
            "always_nan": np.nan,
        }
    )

    cols = select_hmm_observation_columns(
        X,
        max_features=2,
        min_non_nan=60,
        min_features=2,
    )

    assert cols == ["nfp_nsa_lag1", "VIX_latest"]


def test_decide_hmm_reselection_fires_on_confident_label_change():
    prev = _snapshot(
        step_date=pd.Timestamp("2020-01-01"),
        trained_through=pd.Timestamp("2019-12-01"),
        state=0,
        label="stable",
        confidence=0.8,
        entropy=0.2,
        transition_risk=0.1,
        expected_duration=10.0,
        surprise=20.0,
        surprise_ratio=0.5,
    )
    current = _snapshot(
        step_date=pd.Timestamp("2020-04-01"),
        trained_through=pd.Timestamp("2020-03-01"),
        state=1,
        label="crash",
        confidence=0.9,
        entropy=0.1,
        transition_risk=0.2,
        expected_duration=5.0,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=12,
        config=HMMRegimeConfig(min_gap_months=3, trigger_score_threshold=1.0),
    )

    assert decision.should_reselect is True
    assert "regime_label_change:stable->crash" in decision.reason


def test_decide_hmm_reselection_respects_cooldown():
    prev = _snapshot(
        step_date=pd.Timestamp("2020-01-01"),
        trained_through=pd.Timestamp("2019-12-01"),
        state=0,
        label="stable",
        confidence=0.8,
        entropy=0.2,
        transition_risk=0.1,
        expected_duration=10.0,
        surprise=20.0,
        surprise_ratio=0.5,
    )
    current = _snapshot(
        step_date=pd.Timestamp("2020-02-01"),
        trained_through=pd.Timestamp("2020-01-01"),
        state=1,
        label="volatile_down",
        confidence=0.9,
        entropy=0.1,
        transition_risk=0.2,
        expected_duration=5.0,
        surprise=50.0,
        surprise_ratio=1.25,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=1,
        config=HMMRegimeConfig(min_gap_months=6),
    )

    assert decision.should_reselect is False
    assert decision.cooldown_active is True
    assert decision.trigger_class == "cooldown_suppressed"


def test_decide_hmm_reselection_allows_high_risk_downside_cooldown_override():
    prev = _snapshot(
        step_date=pd.Timestamp("2020-01-01"),
        trained_through=pd.Timestamp("2019-12-01"),
        state=0,
        label="stable",
        confidence=0.8,
        entropy=0.2,
        transition_risk=0.1,
        expected_duration=10.0,
        surprise=20.0,
        surprise_ratio=0.5,
    )
    current = _snapshot(
        step_date=pd.Timestamp("2020-02-01"),
        trained_through=pd.Timestamp("2020-01-01"),
        state=1,
        label="volatile_down",
        confidence=0.9,
        entropy=0.1,
        transition_risk=0.7,
        expected_duration=2.0,
        surprise=70.0,
        surprise_ratio=1.75,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=1,
        config=HMMRegimeConfig(min_gap_months=6, force_reselect_risk=0.6),
    )

    assert decision.cooldown_active is True
    assert decision.force_override is True
    assert decision.should_reselect is True


def test_pit_same_month_residual_uses_only_past_same_month_rows():
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    values = pd.Series(np.arange(36, dtype=float))

    z = _pit_same_month_zscore(values, pd.Series(dates))

    jan_2020_idx = 24
    jan_history = values.iloc[[0, 12]]
    # With fewer than three prior same-month observations, the function must
    # fall back to the expanding history rather than use current/future Januaries.
    expanding = values.iloc[:jan_2020_idx]
    expected = (values.iloc[jan_2020_idx] - expanding.mean()) / expanding.std(ddof=1)
    assert np.isclose(z.iloc[jan_2020_idx], expected)
    assert jan_history.max() < values.iloc[jan_2020_idx]


def test_january_downside_penalty_suppresses_weak_seasonal_crash():
    prev = _snapshot(
        step_date=pd.Timestamp("2019-12-01"),
        label="stable",
        transition_risk=0.1,
        surprise_ratio=0.8,
    )
    current = _snapshot(
        step_date=pd.Timestamp("2020-01-01"),
        label="crash",
        transition_risk=0.8,
        surprise=42.0,
        surprise_ratio=1.05,
        month_of_year=1,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=12,
        config=HMMRegimeConfig(trigger_score_threshold=2.0),
    )

    assert decision.should_reselect is False
    assert decision.trigger_class in {"low_score_suppressed", "structural_suppressed"}


def test_state_support_gate_rejects_small_transient_state():
    prev = _snapshot(label="stable", transition_risk=0.1, surprise_ratio=0.8)
    current = _snapshot(
        label="volatile_down",
        state_support_n=4.0,
        state_support_share=0.02,
        expected_duration=3.0,
        surprise_ratio=1.3,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=12,
        config=HMMRegimeConfig(trigger_score_threshold=1.0),
    )

    assert decision.should_reselect is False
    assert "low_state_support" in decision.structural_gate_reason


def test_expected_duration_gate_rejects_one_month_state():
    prev = _snapshot(label="stable", transition_risk=0.1, surprise_ratio=0.8)
    current = _snapshot(
        label="volatile_down",
        expected_duration=1.0,
        surprise_ratio=1.3,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=12,
        config=HMMRegimeConfig(trigger_score_threshold=1.0),
    )

    assert decision.should_reselect is False
    assert "short_expected_duration" in decision.structural_gate_reason


def test_severe_downside_surprise_bypasses_short_duration_gate():
    prev = _snapshot(label="stable", transition_risk=0.1, surprise_ratio=0.8)
    current = _snapshot(
        label="volatile_down",
        expected_duration=1.0,
        state_support_n=4.0,
        state_support_share=0.02,
        surprise=100.0,
        surprise_ratio=2.5,
    )

    decision = decide_hmm_reselection(
        snapshot=current,
        previous_snapshot=prev,
        months_since_reselection=12,
        config=HMMRegimeConfig(trigger_score_threshold=1.0),
    )

    assert decision.structural_gate_passed is True
    assert decision.should_reselect is True


def test_evaluate_hmm_reselection_trigger_is_pit_bounded():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-01", periods=132, freq="MS")
    y = pd.Series(
        np.r_[rng.normal(80, 15, 66), rng.normal(-120, 30, 66)],
        name="y_mom",
    )
    X = pd.DataFrame(
        {
            "ds": dates,
            "nfp_nsa_lag1": y.shift(1).fillna(0.0),
            "VIX_latest": np.r_[rng.normal(15, 2, 66), rng.normal(35, 5, 66)],
            "Yield_10y_latest": rng.normal(size=len(dates)),
            "ISM_PMI_latest": np.r_[rng.normal(54, 1, 66), rng.normal(45, 2, 66)],
        }
    )
    step_date = pd.Timestamp("2020-06-01")
    train = X[X["ds"] < step_date].reset_index(drop=True)
    y_train = y.iloc[: len(train)].reset_index(drop=True)

    decision = evaluate_hmm_reselection_trigger(
        X_train=train,
        y_train=y_train,
        step_date=step_date,
        cleaned_features=[c for c in train.columns if c != "ds"],
        previous_snapshot=None,
        last_reselection_date=pd.Timestamp("2020-03-01"),
        config=HMMRegimeConfig(
            min_train_months=72,
            min_non_nan=48,
            min_features=3,
            n_components=3,
            max_features=4,
        ),
        X_current=X[X["ds"] == step_date],
    )

    assert decision.snapshot is not None
    assert decision.snapshot.trained_through < step_date
    assert decision.snapshot.n_features >= 3
    assert np.isfinite(decision.snapshot.surprise)
    assert np.isfinite(decision.snapshot.surprise_threshold)


def test_known_economic_window_labels_verification_periods():
    assert known_economic_window(pd.Timestamp("2020-04-01")) == "covid_crash"
    assert known_economic_window(pd.Timestamp("2015-01-01")) is None
