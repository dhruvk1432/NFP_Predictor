"""Sticky feature-selection helpers for HMM-gated reselection.

The dynamic feature-selection engine returns a ranked survivor list, but each
fresh Boruta/cap run can churn features on small score differences. This module
adds an incumbency layer: existing features stay unless a challenger clears a
regime-specific score margin, with wider replacement allowances in downside
shock regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StickySelectionConfig:
    enabled: bool = False
    max_features: int = 120
    margin_stable: float = 0.20
    margin_upward: float = 0.15
    margin_volatile: float = 0.05
    margin_default: float = 0.10
    stable_max_replacement_share: float = 0.20
    volatile_min_replacement_share: float = 0.10
    replacement_cap_share: Optional[float] = None
    challenger_slots: int = 0
    rank_weight: float = 0.05
    correlation_weight: float = 1.0


@dataclass(frozen=True)
class StickySelectionResult:
    features: list[str]
    enabled: bool
    regime_label: Optional[str]
    margin: float
    replacements: int
    replacement_cap: Optional[int]
    replacement_floor: int
    jaccard_vs_previous: Optional[float]
    incumbent_count: int
    candidate_count: int
    challenger_slots: int = 0

    def to_log_dict(self) -> dict[str, object]:
        return {
            "sticky_enabled": bool(self.enabled),
            "sticky_regime_label": self.regime_label,
            "sticky_margin": float(self.margin),
            "sticky_replacements": int(self.replacements),
            "sticky_replacement_cap": self.replacement_cap,
            "sticky_replacement_floor": int(self.replacement_floor),
            "sticky_jaccard_vs_previous": self.jaccard_vs_previous,
            "sticky_incumbent_count": int(self.incumbent_count),
            "sticky_candidate_count": int(self.candidate_count),
            "sticky_challenger_slots": int(self.challenger_slots),
        }


def apply_sticky_feature_selection(
    *,
    candidate_features: Iterable[str],
    incumbent_features: Optional[Iterable[str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    regime_label: Optional[str],
    config: StickySelectionConfig,
) -> StickySelectionResult:
    candidates = _unique_existing(candidate_features, X_train.columns)
    incumbents = _unique_existing(incumbent_features or [], X_train.columns)
    cap = max(1, int(config.max_features))
    regime = (regime_label or "").strip().lower() or None
    margin = _margin_for_regime(regime, config)

    if not config.enabled or not incumbents:
        features = candidates[:cap]
        return StickySelectionResult(
            features=features,
            enabled=bool(config.enabled),
            regime_label=regime,
            margin=margin,
            replacements=_replacement_count(features, incumbents),
            replacement_cap=None,
            replacement_floor=0,
            jaccard_vs_previous=_jaccard(features, incumbents),
            incumbent_count=len(incumbents),
            candidate_count=len(candidates),
            challenger_slots=0,
        )

    scores = _score_features(
        X_train=X_train,
        y_train=y_train,
        features=_merge_unique(candidates, incumbents),
        candidate_rank={f: i for i, f in enumerate(candidates)},
        rank_weight=float(config.rank_weight),
        correlation_weight=float(config.correlation_weight),
    )
    challenger_slots = max(0, min(int(config.challenger_slots), cap))
    stable_budget = cap - challenger_slots if challenger_slots > 0 else cap
    selected = incumbents[:stable_budget]
    selected_set = set(selected)
    challengers = [f for f in candidates if f not in selected_set]
    challengers.sort(key=lambda f: (scores.get(f, 0.0), -candidates.index(f)), reverse=True)

    replacement_cap = _replacement_cap(regime, cap, config)
    replacement_floor = _replacement_floor(regime, cap, challengers, config)
    replacements = 0

    for challenger in challengers:
        if len(selected) < cap:
            selected.append(challenger)
            selected_set.add(challenger)
            if challenger not in incumbents:
                replacements += 1
            continue
        if replacement_cap is not None and replacements >= replacement_cap:
            break
        weakest = min(selected, key=lambda f: (scores.get(f, 0.0), -selected.index(f)))
        weak_score = max(float(scores.get(weakest, 0.0)), 0.0)
        challenger_score = max(float(scores.get(challenger, 0.0)), 0.0)
        clears_margin = challenger_score > weak_score * (1.0 + margin)
        force_floor = replacements < replacement_floor and challenger_score >= weak_score
        if clears_margin or force_floor:
            idx = selected.index(weakest)
            selected_set.remove(weakest)
            selected[idx] = challenger
            selected_set.add(challenger)
            replacements += 1

    if len(selected) < cap:
        for feature in candidates:
            if feature not in selected_set:
                selected.append(feature)
                selected_set.add(feature)
            if len(selected) >= cap:
                break

    return StickySelectionResult(
        features=selected[:cap],
        enabled=True,
        regime_label=regime,
        margin=margin,
        replacements=replacements,
        replacement_cap=replacement_cap,
        replacement_floor=replacement_floor,
        jaccard_vs_previous=_jaccard(selected, incumbents),
        incumbent_count=len(incumbents),
        candidate_count=len(candidates),
        challenger_slots=challenger_slots,
    )


def _score_features(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    candidate_rank: dict[str, int],
    rank_weight: float,
    correlation_weight: float,
) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(y_train).reset_index(drop=True), errors="coerce")
    scores: dict[str, float] = {}
    max_rank = max(len(candidate_rank), 1)
    for feature in features:
        if feature not in X_train.columns:
            continue
        x = pd.to_numeric(X_train[feature].reset_index(drop=True), errors="coerce")
        valid = x.notna() & y.notna()
        corr = 0.0
        if int(valid.sum()) >= 12:
            xv = x[valid].to_numpy(dtype=float)
            yv = y[valid].to_numpy(dtype=float)
            if np.nanstd(xv) > 1e-12 and np.nanstd(yv) > 1e-12:
                corr = abs(float(np.corrcoef(xv, yv)[0, 1]))
                if not math.isfinite(corr):
                    corr = 0.0
        rank_score = 0.0
        if feature in candidate_rank:
            rank_score = float(max_rank - candidate_rank[feature]) / max_rank
        scores[feature] = float(correlation_weight) * corr + float(rank_weight) * rank_score
    return scores


def _margin_for_regime(regime: Optional[str], config: StickySelectionConfig) -> float:
    if regime == "stable":
        return float(config.margin_stable)
    if regime in {"upward", "recovery"}:
        return float(config.margin_upward)
    if regime in {"volatile_up", "volatile_down", "crash"}:
        return float(config.margin_volatile)
    return float(config.margin_default)


def _replacement_cap(
    regime: Optional[str],
    cap: int,
    config: StickySelectionConfig,
) -> Optional[int]:
    if config.replacement_cap_share is not None:
        share = max(0.0, min(1.0, float(config.replacement_cap_share)))
        return max(0, int(math.floor(cap * share)))
    if regime == "stable":
        return max(1, int(math.floor(cap * float(config.stable_max_replacement_share))))
    return None


def _replacement_floor(
    regime: Optional[str],
    cap: int,
    challengers: list[str],
    config: StickySelectionConfig,
) -> int:
    if regime in {"volatile_down", "crash"} and challengers:
        return max(1, int(math.ceil(cap * float(config.volatile_min_replacement_share))))
    return 0


def _unique_existing(features: Iterable[str], columns: Iterable[str]) -> list[str]:
    available = set(columns)
    out: list[str] = []
    seen: set[str] = set()
    for feature in features:
        if feature in available and feature not in seen:
            seen.add(feature)
            out.append(feature)
    return out


def _merge_unique(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for feature in group:
            if feature not in seen:
                seen.add(feature)
                out.append(feature)
    return out


def _replacement_count(features: list[str], incumbents: list[str]) -> int:
    incumbent_set = set(incumbents)
    return sum(1 for feature in features if feature not in incumbent_set)


def _jaccard(features: list[str], incumbents: list[str]) -> Optional[float]:
    if not incumbents:
        return None
    a = set(features)
    b = set(incumbents)
    union = a | b
    if not union:
        return None
    return float(len(a & b) / len(union))
