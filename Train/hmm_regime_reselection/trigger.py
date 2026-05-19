"""PIT-safe HMM regime trigger for dynamic feature reselection.

The trigger is deliberately separate from the feature-selection engine. It
only decides whether the current expanding-window training history looks like
a materially different macro regime than the prior step. The expensive Boruta
reselection remains in ``Train.train_lightgbm_nfp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REGIME_PRIORITY_PREFIXES: tuple[tuple[str, int], ...] = (
    ("nfp_", 6),
    ("sidecar_meta__", 6),
    ("sidecar_", 5),
    ("VIX_", 5),
    ("WEI_", 5),
    ("Weekly_", 5),
    ("CCSA_", 5),
    ("CCNSA_", 5),
    ("Yield_", 4),
    ("Treasury_", 4),
    ("FedFunds_", 4),
    ("SOFR_", 4),
    ("SP500_", 4),
    ("SP500_Futures_", 4),
    ("Oil_", 3),
    ("WTI_Crude_", 3),
    ("DollarIndex_", 3),
    ("ISM_", 3),
    ("Challenger_", 3),
    ("ADP_", 3),
    ("NFP_Consensus", 3),
    ("sanagap_", 3),
    ("total_", 2),
)


KNOWN_ECONOMIC_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("1990-07", "1991-03", "gulf_war_recession"),
    ("2001-03", "2001-11", "dotcom_recession"),
    ("2007-12", "2009-06", "global_financial_crisis"),
    ("2020-03", "2020-05", "covid_crash"),
    ("2020-06", "2021-12", "covid_reopening_recovery"),
    ("2022-03", "2023-07", "inflation_tightening"),
)


@dataclass(frozen=True)
class HMMRegimeConfig:
    min_train_months: int = 96
    n_components: int = 3
    covariance_type: str = "diag"
    max_features: int = 32
    min_features: int = 4
    min_non_nan: int = 72
    min_gap_months: int = 9
    max_gap_months: int = 0
    min_state_prob: float = 0.55
    transition_risk_threshold: float = 0.35
    transition_jump: float = 0.15
    entropy_threshold: float = 0.70
    entropy_jump: float = 0.18
    surprise_quantile: float = 0.95
    force_reselect_labels: tuple[str, ...] = ("crash", "volatile_down")
    force_reselect_risk: float = 0.60
    min_prob_margin: float = 0.20
    emission_profile: str = "seasonal_resid"
    trigger_score_threshold: float = 2.0
    surprise_low_quantile: float = 0.90
    surprise_high_quantile: float = 0.975
    severe_surprise_ratio: float = 2.0
    downside_surprise_ratio: float = 1.05
    seasonal_penalty_surprise_ratio: float = 1.50
    min_state_support_n: int = 12
    min_state_support_share: float = 0.05
    min_expected_duration: float = 2.0
    episode_suppression_months: int = 6
    episode_clear_months: int = 3
    random_state: int = 42
    n_iter: int = 200


@dataclass(frozen=True)
class HMMRegimeSnapshot:
    step_date: pd.Timestamp
    trained_through: pd.Timestamp
    state: int
    label: str
    confidence: float
    entropy: float
    transition_risk: float
    expected_duration: float
    n_components: int
    n_features: int
    surprise: float = np.nan
    surprise_threshold: float = np.nan
    surprise_q90: float = np.nan
    surprise_q975: float = np.nan
    surprise_ratio: float = np.nan
    prob_margin: float = 1.0
    state_support_n: float = np.nan
    state_support_share: float = np.nan
    month_of_year: int = 0
    is_seasonal_false_positive_candidate: bool = False
    event_window: Optional[str] = None
    state_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "step_date": self.step_date.strftime("%Y-%m"),
            "trained_through": self.trained_through.strftime("%Y-%m"),
            "hmm_state": int(self.state),
            "hmm_regime_label": self.label,
            "hmm_confidence": _finite_or_none(self.confidence),
            "hmm_entropy": _finite_or_none(self.entropy),
            "hmm_transition_risk": _finite_or_none(self.transition_risk),
            "hmm_expected_duration": _finite_or_none(self.expected_duration),
            "hmm_surprise": _finite_or_none(self.surprise),
            "hmm_surprise_threshold": _finite_or_none(self.surprise_threshold),
            "hmm_surprise_q90": _finite_or_none(self.surprise_q90),
            "hmm_surprise_q975": _finite_or_none(self.surprise_q975),
            "hmm_surprise_ratio": _finite_or_none(self.surprise_ratio),
            "hmm_prob_margin": _finite_or_none(self.prob_margin),
            "hmm_state_support_n": _finite_or_none(self.state_support_n),
            "hmm_state_support_share": _finite_or_none(self.state_support_share),
            "hmm_month_of_year": int(self.month_of_year),
            "hmm_is_seasonal_false_positive_candidate": bool(self.is_seasonal_false_positive_candidate),
            "hmm_n_components": int(self.n_components),
            "hmm_n_features": int(self.n_features),
            "event_window": self.event_window,
            "hmm_features": list(self.feature_names),
            "hmm_state_stats": {
                str(k): {name: _finite_or_none(v) for name, v in stats.items()}
                for k, stats in self.state_stats.items()
            },
        }


@dataclass(frozen=True)
class HMMRegimeDecision:
    available: bool
    should_reselect: bool
    reason: str
    reasons: tuple[str, ...]
    snapshot: Optional[HMMRegimeSnapshot] = None
    months_since_reselection: Optional[int] = None
    cooldown_active: bool = False
    label_changed: bool = False
    high_risk: bool = False
    force_override: bool = False
    trigger_class: str = "no_shift"
    trigger_score: float = np.nan
    structural_gate_passed: bool = True
    structural_gate_reason: str = ""
    episode_suppressed: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        payload = {
            "hmm_available": bool(self.available),
            "hmm_should_reselect": bool(self.should_reselect),
            "hmm_trigger_reason": self.reason,
            "hmm_trigger_reasons": list(self.reasons),
            "hmm_months_since_reselection": self.months_since_reselection,
            "hmm_cooldown_active": bool(self.cooldown_active),
            "hmm_label_changed": bool(self.label_changed),
            "hmm_high_risk": bool(self.high_risk),
            "hmm_force_override": bool(self.force_override),
            "hmm_trigger_class": self.trigger_class,
            "hmm_trigger_score": _finite_or_none(self.trigger_score),
            "hmm_structural_gate_passed": bool(self.structural_gate_passed),
            "hmm_structural_gate_reason": self.structural_gate_reason,
            "hmm_episode_suppressed": bool(self.episode_suppressed),
        }
        if self.snapshot is not None:
            payload.update(self.snapshot.to_log_dict())
        return payload


def months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    later = pd.Timestamp(later)
    earlier = pd.Timestamp(earlier)
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def known_economic_window(month: pd.Timestamp) -> Optional[str]:
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    for start, end, label in KNOWN_ECONOMIC_WINDOWS:
        if pd.Timestamp(start) <= month <= pd.Timestamp(end):
            return label
    return None


def select_hmm_observation_columns(
    X_train: pd.DataFrame,
    candidate_features: Optional[Iterable[str]] = None,
    *,
    max_features: int = 32,
    min_non_nan: int = 72,
    min_features: int = 4,
) -> list[str]:
    """Select a compact, high-coverage macro panel for the HMM emissions."""
    if candidate_features is None:
        candidates = [c for c in X_train.columns if c != "ds"]
    else:
        candidates = [c for c in candidate_features if c in X_train.columns and c != "ds"]

    numeric = [
        c for c in candidates
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    if not numeric:
        return []

    n_rows = max(1, len(X_train))
    strict_min = min(int(min_non_nan), max(12, int(n_rows * 0.65)))
    fallback_min = min(strict_min, max(12, int(n_rows * 0.35)))

    scored = _score_columns(X_train, numeric, min_non_nan=strict_min)
    if len(scored) < int(min_features):
        scored = _score_columns(X_train, numeric, min_non_nan=fallback_min)

    scored.sort(reverse=True)
    return [col for *_rest, col in scored[: int(max_features)]]


def evaluate_hmm_reselection_trigger(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    step_date: pd.Timestamp,
    cleaned_features: Optional[Iterable[str]],
    previous_snapshot: Optional[HMMRegimeSnapshot],
    last_reselection_date: Optional[pd.Timestamp],
    config: HMMRegimeConfig,
    X_current: Optional[pd.DataFrame | pd.Series] = None,
) -> HMMRegimeDecision:
    """Fit an expanding-window HMM and decide whether to refresh features.

    PIT contract: callers pass only the training slice whose labels and
    features are operationally available before ``step_date``. This function
    never reads target-month labels, future rows, or external state.
    """
    step_date = pd.Timestamp(step_date).to_period("M").to_timestamp()
    months_since = (
        months_between(step_date, last_reselection_date)
        if last_reselection_date is not None
        else None
    )
    max_due = _max_gap_due(months_since, config)
    if len(X_train) < int(config.min_train_months):
        return _unavailable_decision(
            "insufficient_hmm_history",
            months_since,
            max_due,
            config,
        )

    profile_candidates = _filter_emission_profile_candidates(
        cleaned_features,
        profile=config.emission_profile,
    )
    cols = select_hmm_observation_columns(
        X_train,
        candidate_features=profile_candidates,
        max_features=config.max_features,
        min_non_nan=config.min_non_nan,
        min_features=config.min_features,
    )
    if len(cols) < int(config.min_features):
        return _unavailable_decision(
            "insufficient_hmm_features",
            months_since,
            max_due,
            config,
        )

    try:
        snapshot = fit_hmm_regime_snapshot(
            X_train=X_train,
            y_train=y_train,
            step_date=step_date,
            feature_cols=cols,
            config=config,
            X_current=X_current,
        )
    except Exception as exc:
        return _unavailable_decision(
            f"hmm_fit_failed:{type(exc).__name__}",
            months_since,
            max_due,
            config,
        )

    return decide_hmm_reselection(
        snapshot=snapshot,
        previous_snapshot=previous_snapshot,
        months_since_reselection=months_since,
        config=config,
    )


def fit_hmm_regime_snapshot(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    step_date: pd.Timestamp,
    feature_cols: list[str],
    config: HMMRegimeConfig,
    X_current: Optional[pd.DataFrame | pd.Series] = None,
) -> HMMRegimeSnapshot:
    from hmmlearn.hmm import GaussianHMM

    frame = X_train[["ds"] + feature_cols].copy()
    frame["__y__"] = pd.to_numeric(pd.Series(y_train).reset_index(drop=True), errors="coerce").values
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame = frame.dropna(subset=["ds", "__y__"]).sort_values("ds").reset_index(drop=True)
    if len(frame) < int(config.min_train_months):
        raise ValueError("insufficient non-null HMM rows")

    raw_emissions = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    current_raw = None
    if X_current is not None:
        current_raw = _current_emission_row(X_current, feature_cols)
    emissions, current_emissions = _prepare_hmm_emissions(
        raw_emissions=raw_emissions,
        train_dates=frame["ds"],
        current_raw=current_raw,
        current_date=step_date,
        profile=config.emission_profile,
    )
    med = emissions.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    emissions = emissions.fillna(med)
    if current_emissions is not None:
        current_emissions = current_emissions.reindex(columns=emissions.columns).fillna(med)
    variances = emissions.var(skipna=True)
    keep_cols = [c for c in emissions.columns if np.isfinite(variances.get(c, np.nan)) and variances[c] > 1e-12]
    if len(keep_cols) < int(config.min_features):
        raise ValueError("insufficient non-constant HMM features")
    emissions = emissions[keep_cols]
    med = med.reindex(keep_cols).fillna(0.0)
    if current_emissions is not None:
        current_emissions = current_emissions[keep_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(emissions.to_numpy(dtype=float))
    n_components = min(
        int(config.n_components),
        max(2, len(frame) // 24),
        max(2, len(frame) // max(8, int(config.min_features))),
    )
    if n_components < 2:
        raise ValueError("insufficient rows for multiple HMM states")

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=config.covariance_type,
        n_iter=int(config.n_iter),
        random_state=int(config.random_state),
        min_covar=1e-3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled)
        states = model.predict(X_scaled)
        posterior_train = model.predict_proba(X_scaled)

    y_values = frame["__y__"].to_numpy(dtype=float)
    accel_values = np.diff(y_values, prepend=np.nan)
    state_stats = _state_stats(states, y_values, accel_values, n_components)
    state_labels = _label_states(state_stats, y_values, accel_values)

    if current_emissions is not None:
        current_emissions = current_emissions.replace([np.inf, -np.inf], np.nan).fillna(med)
        X_probe = scaler.transform(current_emissions.to_numpy(dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior = model.predict_proba(np.vstack([X_scaled, X_probe]))
        latest_prob = posterior[-1]
        surprise = _predictive_surprise(model, X_scaled, posterior_train, X_probe)
    else:
        latest_prob = posterior_train[-1]
        surprise = _predictive_surprise(model, X_scaled[:-1], posterior_train[:-1], X_scaled[-1:])

    state = int(np.argmax(latest_prob))
    confidence = float(np.max(latest_prob))
    prob_margin = _prob_margin(latest_prob)
    entropy = _entropy(latest_prob)
    train_surprises = _training_predictive_surprises(model, X_scaled, posterior_train)
    surprise_threshold = _finite_quantile(train_surprises, config.surprise_quantile)
    surprise_q90 = _finite_quantile(train_surprises, config.surprise_low_quantile)
    surprise_q975 = _finite_quantile(train_surprises, config.surprise_high_quantile)
    surprise_ratio = _safe_ratio(surprise, surprise_threshold)
    p_stay = _safe_transmat_value(model, state)
    transition_risk = 1.0 - p_stay if np.isfinite(p_stay) else np.nan
    expected_duration = 1.0 / max(1.0 - p_stay, 1e-6) if np.isfinite(p_stay) else np.nan
    trained_through = pd.Timestamp(frame["ds"].iloc[-1])
    support = state_stats.get(state, {})
    support_n = float(support.get("n", np.nan))
    support_share = float(support.get("frequency", np.nan))
    event_window = known_economic_window(step_date)
    month_of_year = int(pd.Timestamp(step_date).month)
    seasonal_false_positive_candidate = (
        month_of_year in {1, 7}
        and state_labels.get(state, "unknown") in {"crash", "volatile_down"}
        and (not np.isfinite(surprise_ratio) or surprise_ratio < float(config.seasonal_penalty_surprise_ratio))
        and event_window is None
    )

    return HMMRegimeSnapshot(
        step_date=pd.Timestamp(step_date).to_period("M").to_timestamp(),
        trained_through=trained_through,
        state=state,
        label=state_labels.get(state, "unknown"),
        confidence=confidence,
        entropy=entropy,
        transition_risk=float(transition_risk),
        expected_duration=float(expected_duration),
        surprise=float(surprise),
        surprise_threshold=float(surprise_threshold),
        surprise_q90=float(surprise_q90),
        surprise_q975=float(surprise_q975),
        surprise_ratio=float(surprise_ratio),
        prob_margin=float(prob_margin),
        state_support_n=float(support_n),
        state_support_share=float(support_share),
        month_of_year=month_of_year,
        is_seasonal_false_positive_candidate=seasonal_false_positive_candidate,
        n_components=n_components,
        n_features=len(keep_cols),
        event_window=event_window,
        state_stats=state_stats,
        feature_names=tuple(keep_cols),
    )


def decide_hmm_reselection(
    *,
    snapshot: HMMRegimeSnapshot,
    previous_snapshot: Optional[HMMRegimeSnapshot],
    months_since_reselection: Optional[int],
    config: HMMRegimeConfig,
) -> HMMRegimeDecision:
    reasons: list[str] = []
    max_due = _max_gap_due(months_since_reselection, config)
    cooldown = (
        months_since_reselection is not None
        and months_since_reselection < int(config.min_gap_months)
        and not max_due
    )

    label_changed = False
    risk_jump = np.nan
    if previous_snapshot is not None:
        label_changed = snapshot.label != previous_snapshot.label
        confident_change = (
            snapshot.confidence >= float(config.min_state_prob)
            and snapshot.prob_margin >= float(config.min_prob_margin)
        )
        if label_changed and confident_change:
            reasons.append(f"regime_label_change:{previous_snapshot.label}->{snapshot.label}")

        risk_jump = snapshot.transition_risk - previous_snapshot.transition_risk
        if (
            np.isfinite(risk_jump)
            and snapshot.transition_risk >= float(config.transition_risk_threshold)
            and risk_jump >= float(config.transition_jump)
        ):
            reasons.append("transition_risk_jump")

        entropy_jump = snapshot.entropy - previous_snapshot.entropy
        if (
            np.isfinite(entropy_jump)
            and snapshot.entropy >= float(config.entropy_threshold)
            and entropy_jump >= float(config.entropy_jump)
        ):
            reasons.append("posterior_entropy_jump")

    if (
        np.isfinite(snapshot.surprise)
        and np.isfinite(snapshot.surprise_threshold)
        and snapshot.surprise >= snapshot.surprise_threshold
    ):
        reasons.append("emission_surprise")

    if max_due:
        reasons.append("max_gap_fallback")

    trigger_score = _trigger_score(
        snapshot=snapshot,
        risk_jump=risk_jump,
        config=config,
    )
    structural_passed, structural_reason = _structural_gate(
        snapshot=snapshot,
        previous_snapshot=previous_snapshot,
        label_changed=label_changed,
        max_due=max_due,
        config=config,
    )
    high_risk = (
        np.isfinite(snapshot.transition_risk)
        and snapshot.transition_risk >= float(config.transition_risk_threshold)
    )
    severe_downside = (
        snapshot.label in set(config.force_reselect_labels)
        and np.isfinite(snapshot.surprise_ratio)
        and snapshot.surprise_ratio >= float(config.severe_surprise_ratio)
    )
    force_risk = (
        snapshot.label in set(config.force_reselect_labels)
        and np.isfinite(snapshot.transition_risk)
        and snapshot.transition_risk >= float(config.force_reselect_risk)
        and np.isfinite(snapshot.surprise_ratio)
        and snapshot.surprise_ratio >= float(config.seasonal_penalty_surprise_ratio)
    )
    force_override = (
        cooldown
        and (severe_downside or force_risk)
    )
    score_passed = (
        np.isfinite(trigger_score)
        and trigger_score >= float(config.trigger_score_threshold)
    )
    should = (
        bool(reasons)
        and (max_due or (score_passed and structural_passed))
        and (not cooldown or force_override or max_due)
    )
    reason = ";".join(reasons) if reasons else "no_hmm_shift"
    if structural_reason and reasons and not max_due:
        reason = f"{reason};gate:{structural_reason}"
    if reasons and not max_due and not score_passed:
        reason = f"{reason};low_trigger_score:{trigger_score:.3f}"
    if cooldown and reasons and not force_override:
        reason = f"cooldown:{reason}"
    trigger_class = _trigger_class(
        reasons,
        cooldown=cooldown,
        force_override=force_override,
        structural_passed=structural_passed,
        score_passed=score_passed,
        max_due=max_due,
    )
    return HMMRegimeDecision(
        available=True,
        should_reselect=should,
        reason=reason,
        reasons=tuple(reasons),
        snapshot=snapshot,
        months_since_reselection=months_since_reselection,
        cooldown_active=cooldown,
        label_changed=label_changed,
        high_risk=high_risk,
        force_override=force_override,
        trigger_class=trigger_class,
        trigger_score=float(trigger_score),
        structural_gate_passed=bool(structural_passed),
        structural_gate_reason=structural_reason,
    )


def _filter_emission_profile_candidates(
    candidate_features: Optional[Iterable[str]],
    *,
    profile: str,
) -> Optional[list[str]]:
    if candidate_features is None:
        return None
    profile = _normal_emission_profile(profile)
    out: list[str] = []
    for col in candidate_features:
        if profile == "macro_only" and _macro_only_excludes(col):
            continue
        out.append(col)
    return out


def _normal_emission_profile(profile: str) -> str:
    profile = (profile or "raw").strip().lower()
    if profile not in {"raw", "seasonal_resid", "hybrid", "macro_only"}:
        return "raw"
    return profile


def _prepare_hmm_emissions(
    *,
    raw_emissions: pd.DataFrame,
    train_dates: pd.Series,
    current_raw: Optional[pd.DataFrame],
    current_date: pd.Timestamp,
    profile: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    profile = _normal_emission_profile(profile)
    raw = raw_emissions.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_dates = pd.to_datetime(train_dates, errors="coerce").dt.to_period("M").dt.to_timestamp()
    current = None
    if current_raw is not None:
        current = current_raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        current = current.reindex(columns=raw.columns)

    if profile in {"raw", "macro_only"}:
        return raw, current

    transformed = pd.DataFrame(index=raw.index)
    current_values: dict[str, float] = {}
    for col in raw.columns:
        use_seasonal = profile == "seasonal_resid" or (
            profile == "hybrid" and _seasonal_residual_feature(col)
        )
        if use_seasonal:
            transformed[col] = _pit_same_month_zscore(raw[col], train_dates)
            if current is not None:
                current_values[col] = _current_same_month_zscore(
                    raw[col],
                    train_dates,
                    current[col].iloc[0],
                    pd.Timestamp(current_date),
                )
        else:
            transformed[col] = _pit_expanding_zscore(raw[col])
            if current is not None:
                current_values[col] = _current_expanding_zscore(raw[col], current[col].iloc[0])

    current_frame = None
    if current is not None:
        current_frame = pd.DataFrame([current_values], columns=raw.columns)
    return transformed, current_frame


def _pit_same_month_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    months = pd.to_datetime(dates, errors="coerce").dt.month.reset_index(drop=True)
    out = np.full(len(vals), np.nan, dtype=float)
    for i, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        prior_same = vals.iloc[:i][months.iloc[:i].eq(months.iloc[i])]
        mean, std = _history_mean_std(prior_same)
        if not np.isfinite(mean) or not np.isfinite(std):
            mean, std = _history_mean_std(vals.iloc[:i])
        if np.isfinite(mean) and np.isfinite(std):
            out[i] = (float(value) - mean) / std
    return pd.Series(out, index=values.index)


def _pit_expanding_zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    out = np.full(len(vals), np.nan, dtype=float)
    for i, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        mean, std = _history_mean_std(vals.iloc[:i])
        if np.isfinite(mean) and np.isfinite(std):
            out[i] = (float(value) - mean) / std
    return pd.Series(out, index=values.index)


def _current_same_month_zscore(
    history: pd.Series,
    history_dates: pd.Series,
    value: Any,
    current_date: pd.Timestamp,
) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return np.nan
    dates = pd.to_datetime(history_dates, errors="coerce").dt.to_period("M").dt.to_timestamp()
    vals = pd.to_numeric(history, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    same_month = vals[dates.dt.month.reset_index(drop=True).eq(pd.Timestamp(current_date).month)]
    mean, std = _history_mean_std(same_month)
    if not np.isfinite(mean) or not np.isfinite(std):
        mean, std = _history_mean_std(vals)
    return float((float(value) - mean) / std) if np.isfinite(mean) and np.isfinite(std) else np.nan


def _current_expanding_zscore(history: pd.Series, value: Any) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return np.nan
    mean, std = _history_mean_std(history)
    return float((float(value) - mean) / std) if np.isfinite(mean) and np.isfinite(std) else np.nan


def _history_mean_std(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size < 3:
        return np.nan, np.nan
    std = float(np.std(arr, ddof=1))
    if not np.isfinite(std) or std <= 1e-6:
        return float(np.mean(arr)), np.nan
    return float(np.mean(arr)), std


def _seasonal_residual_feature(col: str) -> bool:
    lower = str(col).lower()
    return lower.startswith("nfp_") or "nsa" in lower or "same_month" in lower or "seasonal" in lower


def _macro_only_excludes(col: str) -> bool:
    lower = str(col).lower()
    return (
        lower.startswith("nfp_nsa_mom_")
        or "same_month" in lower
        or "seasonal" in lower
    )


def _trigger_score(
    *,
    snapshot: HMMRegimeSnapshot,
    risk_jump: float,
    config: HMMRegimeConfig,
) -> float:
    denom = snapshot.surprise_q975 - snapshot.surprise_q90
    if not np.isfinite(denom) or denom <= 1e-9:
        denom = max(abs(snapshot.surprise_threshold), 1.0)
    surprise_component = 0.0
    if np.isfinite(snapshot.surprise) and np.isfinite(snapshot.surprise_q90):
        surprise_component = max(0.0, (snapshot.surprise - snapshot.surprise_q90) / denom)
    jump_component = max(0.0, float(risk_jump)) if np.isfinite(risk_jump) else 0.0
    downside_component = 1.0 if snapshot.label in set(config.force_reselect_labels) else 0.0
    seasonal_penalty = (
        1.0
        if snapshot.month_of_year in {1, 7}
        and (
            not np.isfinite(snapshot.surprise_ratio)
            or snapshot.surprise_ratio < float(config.seasonal_penalty_surprise_ratio)
        )
        else 0.0
    )
    return float(
        1.5 * surprise_component
        + jump_component
        + 0.75 * downside_component
        - seasonal_penalty
    )


def _structural_gate(
    *,
    snapshot: HMMRegimeSnapshot,
    previous_snapshot: Optional[HMMRegimeSnapshot],
    label_changed: bool,
    max_due: bool,
    config: HMMRegimeConfig,
) -> tuple[bool, str]:
    if max_due:
        return True, ""

    reasons: list[str] = []
    severe_surprise = (
        np.isfinite(snapshot.surprise_ratio)
        and snapshot.surprise_ratio >= float(config.severe_surprise_ratio)
    )
    support_n = snapshot.state_support_n
    support_share = snapshot.state_support_share
    estimated_history_n = support_n / support_share if np.isfinite(support_n) and np.isfinite(support_share) and support_share > 0 else np.nan
    required_support = float(config.min_state_support_n)
    if np.isfinite(estimated_history_n):
        required_support = max(required_support, float(config.min_state_support_share) * estimated_history_n)
    if np.isfinite(support_n) and support_n < required_support and not severe_surprise:
        reasons.append("low_state_support")

    if (
        np.isfinite(snapshot.expected_duration)
        and snapshot.expected_duration < float(config.min_expected_duration)
        and not severe_surprise
    ):
        reasons.append("short_expected_duration")

    downside_labels = set(config.force_reselect_labels)
    recovery_labels = {"recovery", "volatile_up"}
    previous_downside = previous_snapshot is not None and previous_snapshot.label in downside_labels
    eligible_label = snapshot.label in downside_labels or (
        snapshot.label in recovery_labels and previous_downside
    )
    if not eligible_label:
        reasons.append("ineligible_label")

    downside_transition = label_changed and snapshot.label in downside_labels
    if (
        downside_transition
        and (
            not np.isfinite(snapshot.surprise_ratio)
            or snapshot.surprise_ratio < float(config.downside_surprise_ratio)
        )
        and not severe_surprise
    ):
        reasons.append("weak_downside_surprise")

    return (not reasons), ";".join(reasons)


def _score_columns(
    X_train: pd.DataFrame,
    cols: list[str],
    *,
    min_non_nan: int,
) -> list[tuple[int, int, float, str]]:
    scored: list[tuple[int, int, float, str]] = []
    for col in cols:
        s = pd.to_numeric(X_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(s.notna().sum())
        if count < int(min_non_nan):
            continue
        var = float(s.var(skipna=True))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        scored.append((_prefix_priority(col), count, math.log1p(var), col))
    return scored


def _training_predictive_surprises(
    model: Any,
    X_scaled: np.ndarray,
    posterior_train: np.ndarray,
) -> np.ndarray:
    """Approximate one-step emission surprises using prior filtered states."""
    if len(X_scaled) == 0:
        return np.array([], dtype=float)
    try:
        emission_ll = np.asarray(model._compute_log_likelihood(X_scaled), dtype=float)
        transmat = np.asarray(model.transmat_, dtype=float)
        startprob = np.asarray(model.startprob_, dtype=float)
    except Exception:
        return np.array([], dtype=float)

    out = np.full(len(X_scaled), np.nan, dtype=float)
    prior = _safe_prob_vector(startprob, emission_ll.shape[1])
    for i in range(len(X_scaled)):
        out[i] = -_logsumexp(np.log(prior + 1e-300) + emission_ll[i])
        if i < len(posterior_train):
            posterior = _safe_prob_vector(posterior_train[i], emission_ll.shape[1])
            prior = _safe_prob_vector(posterior @ transmat, emission_ll.shape[1])
    return out


def _predictive_surprise(
    model: Any,
    X_history: np.ndarray,
    posterior_history: np.ndarray,
    X_probe: np.ndarray,
) -> float:
    try:
        emission_ll = np.asarray(model._compute_log_likelihood(X_probe), dtype=float)[0]
        n_states = int(emission_ll.shape[0])
        if len(X_history) and len(posterior_history):
            prior = _safe_prob_vector(posterior_history[-1], n_states) @ np.asarray(model.transmat_, dtype=float)
        else:
            prior = np.asarray(model.startprob_, dtype=float)
        prior = _safe_prob_vector(prior, n_states)
        return float(-_logsumexp(np.log(prior + 1e-300) + emission_ll))
    except Exception:
        return np.nan


def _safe_prob_vector(values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != int(n):
        arr = np.ones(int(n), dtype=float)
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(int(n), dtype=float) / max(int(n), 1)
    return arr / total


def _logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -np.inf
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(arr, q))


def _safe_ratio(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or abs(float(denom)) <= 1e-12:
        return np.nan
    return float(num) / float(denom)


def _prob_margin(prob: np.ndarray) -> float:
    arr = np.asarray(prob, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 1.0 if arr.size == 1 else 0.0
    top = np.sort(arr)[-2:]
    return float(top[-1] - top[-2])


def _trigger_class(
    reasons: list[str],
    *,
    cooldown: bool,
    force_override: bool,
    structural_passed: bool,
    score_passed: bool,
    max_due: bool,
) -> str:
    if not reasons:
        return "no_shift"
    if max_due or "max_gap_fallback" in reasons:
        return "max_gap"
    if not structural_passed:
        return "structural_suppressed"
    if not score_passed:
        return "low_score_suppressed"
    if cooldown and not force_override:
        return "cooldown_suppressed"
    if any(reason.startswith("regime_label_change:") for reason in reasons):
        return "regime_shift"
    if "transition_risk_jump" in reasons:
        return "risk_jump"
    if "emission_surprise" in reasons:
        return "surprise"
    return "regime_shift"


def _current_emission_row(
    X_current: pd.DataFrame | pd.Series,
    keep_cols: list[str],
) -> pd.DataFrame:
    if isinstance(X_current, pd.Series):
        row = X_current.to_frame().T
    else:
        row = X_current.copy()
    for col in keep_cols:
        if col not in row.columns:
            row[col] = np.nan
    return row[keep_cols].iloc[[0]].apply(pd.to_numeric, errors="coerce")


def _prefix_priority(col: str) -> int:
    for prefix, score in REGIME_PRIORITY_PREFIXES:
        if col.startswith(prefix):
            return score
    return 1


def _entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(max(len(prob), 2)))


def _state_stats(
    states: np.ndarray,
    y_values: np.ndarray,
    accel_values: np.ndarray,
    n_components: int,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for state in range(int(n_components)):
        mask = states == state
        y_state = y_values[mask]
        a_state = accel_values[mask]
        out[state] = {
            "frequency": float(np.mean(mask)) if len(mask) else 0.0,
            "n": float(np.sum(mask)),
            "mean_y": _nanmean(y_state),
            "std_y": _nanstd(y_state),
            "mean_accel": _nanmean(a_state),
            "std_accel": _nanstd(a_state),
        }
    return out


def _label_states(
    state_stats: dict[int, dict[str, float]],
    y_values: np.ndarray,
    accel_values: np.ndarray,
) -> dict[int, str]:
    y_mean = _nanmean(y_values)
    y_std = max(_nanstd(y_values), 1e-6)
    accel_std = max(_nanstd(accel_values), 1e-6)
    means = np.array([v["mean_y"] for v in state_stats.values()], dtype=float)
    vols = np.array([v["std_y"] for v in state_stats.values()], dtype=float)
    finite_means = means[np.isfinite(means)]
    finite_vols = vols[np.isfinite(vols)]
    low_mean = float(np.quantile(finite_means, 0.20)) if finite_means.size else y_mean
    high_vol = float(np.quantile(finite_vols, 0.70)) if finite_vols.size else _nanstd(y_values)
    low_vol = float(np.quantile(finite_vols, 0.35)) if finite_vols.size else _nanstd(y_values)

    labels: dict[int, str] = {}
    for state, stats in state_stats.items():
        mean_y = stats["mean_y"]
        vol_y = stats["std_y"]
        mean_accel = stats["mean_accel"]
        mean_z = (mean_y - y_mean) / y_std if np.isfinite(mean_y) else 0.0
        accel_z = mean_accel / accel_std if np.isfinite(mean_accel) else 0.0
        high_vol_state = np.isfinite(vol_y) and vol_y >= high_vol
        low_vol_state = np.isfinite(vol_y) and vol_y <= low_vol

        if np.isfinite(mean_y) and mean_y <= low_mean and (mean_z <= -0.35 or accel_z <= -0.50):
            label = "crash"
        elif high_vol_state and (mean_z >= 0.0 or accel_z >= 0.35):
            label = "volatile_up"
        elif high_vol_state and (mean_z < 0.0 or accel_z <= -0.35):
            label = "volatile_down"
        elif mean_z < -0.20 and accel_z >= 0.55:
            label = "recovery"
        elif mean_z > 0.20 and accel_z <= -0.55:
            label = "cooling"
        elif low_vol_state and abs(mean_z) <= 0.45 and abs(accel_z) <= 0.45:
            label = "stable"
        elif mean_z >= 0.35 or accel_z >= 0.35:
            label = "upward"
        elif mean_z <= -0.35 or accel_z <= -0.35:
            label = "downward"
        else:
            label = "stable"
        labels[state] = label
    return labels


def _safe_transmat_value(model: Any, state: int) -> float:
    try:
        value = float(model.transmat_[state, state])
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _max_gap_due(months_since: Optional[int], config: HMMRegimeConfig) -> bool:
    return (
        months_since is not None
        and int(config.max_gap_months) > 0
        and months_since >= int(config.max_gap_months)
    )


def _unavailable_decision(
    reason: str,
    months_since: Optional[int],
    max_due: bool,
    config: HMMRegimeConfig,
) -> HMMRegimeDecision:
    cooldown = (
        months_since is not None
        and months_since < int(config.min_gap_months)
        and not max_due
    )
    should = bool(max_due) and not cooldown
    reasons = ("max_gap_fallback",) if max_due else (reason,)
    out_reason = "max_gap_fallback" if should else reason
    return HMMRegimeDecision(
        available=False,
        should_reselect=should,
        reason=out_reason,
        reasons=reasons,
        snapshot=None,
        months_since_reselection=months_since,
        cooldown_active=cooldown,
        high_risk=False,
        force_override=False,
        trigger_class="max_gap" if should else "no_shift",
    )


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _nanstd(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
