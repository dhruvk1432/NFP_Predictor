"""
Branch-target derived feature selection utilities.

This module keeps branch-specific target-derived features (``nfp_{target_type}_*``)
compact and signal-focused before they are merged with snapshot features.
Selection is intentionally lightweight:
1) correlation-based redundancy pruning
2) fast ranking via existing short-pass rankers
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.short_pass_selection import select_features_for_step

logger = setup_logger(__file__, TEMP_DIR)

TARGET_PREFIXES = ("nfp_nsa_", "nfp_sa_")
CALENDAR_PREFIXES = ("is_", "month_", "quarter_", "weeks_")
REVISION_PREFIXES = ("rev_master_", "rev_")
CALENDAR_EXACT = {"year", "snapshot_date"}


def partition_feature_columns(
    feature_cols: Sequence[str],
    target_type: str,
) -> Dict[str, List[str]]:
    """
    Split feature columns into logical groups for branch training.

    Args:
        feature_cols: Iterable of feature names.
        target_type: ``'nsa'`` or ``'sa'``.

    Returns:
        Dict with keys:
            - ``snapshot_features``
            - ``target_branch_features`` (nfp_{target_type}_*)
            - ``other_target_features`` (cross-target nfp_* features)
            - ``calendar_features``
            - ``revision_features``
    """
    target_type = target_type.lower()
    if target_type not in ("nsa", "sa"):
        raise ValueError(f"Invalid target_type: {target_type!r}. Expected 'nsa' or 'sa'.")

    branch_prefix = f"nfp_{target_type}_"
    out = {
        "snapshot_features": [],
        "target_branch_features": [],
        "other_target_features": [],
        "calendar_features": [],
        "revision_features": [],
    }

    for col in feature_cols:
        if col.startswith(branch_prefix):
            out["target_branch_features"].append(col)
        elif col.startswith(TARGET_PREFIXES):
            out["other_target_features"].append(col)
        elif col.startswith(REVISION_PREFIXES):
            out["revision_features"].append(col)
        elif col in CALENDAR_EXACT or col.startswith(CALENDAR_PREFIXES):
            out["calendar_features"].append(col)
        else:
            out["snapshot_features"].append(col)

    return out


def _weighted_abs_corr(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    min_overlap: int = 24,
) -> float:
    """Return weighted absolute Pearson correlation with pairwise-complete rows."""
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )
    if mask.sum() < min_overlap:
        return 0.0

    x_valid = x[mask].astype(float)
    y_valid = y[mask].astype(float)
    if sample_weights is None:
        w = np.ones_like(x_valid, dtype=float)
    else:
        w = sample_weights[mask].astype(float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return 0.0
    w = w / w_sum

    mx = np.sum(w * x_valid)
    my = np.sum(w * y_valid)
    vx = np.sum(w * (x_valid - mx) ** 2)
    vy = np.sum(w * (y_valid - my) ** 2)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0

    cov = np.sum(w * (x_valid - mx) * (y_valid - my))
    return float(abs(cov / np.sqrt(vx * vy)))


def prune_redundant_by_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: Sequence[str],
    corr_threshold: float = 0.90,
    min_overlap: int = 24,
    sample_weights: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Greedy redundancy pruning for branch-target candidates.

    Features are ranked by weighted |corr(feature, y)|, then selected greedily
    while dropping later features highly correlated with already-selected ones.
    """
    available = [c for c in candidate_features if c in X_train.columns]
    if len(available) <= 1:
        return available

    X = X_train[available].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(y_train, errors="coerce")
    y_arr = y.values.astype(float)
    w = sample_weights

    # 1) Rank by relevance to target
    relevance = {}
    for col in available:
        relevance[col] = _weighted_abs_corr(
            X[col].values,
            y_arr,
            sample_weights=w,
            min_overlap=min_overlap,
        )
    ranked = sorted(available, key=lambda c: relevance.get(c, 0.0), reverse=True)

    # 2) Precompute pairwise feature correlation + overlap
    corr = X[ranked].corr(method="spearman").abs().fillna(0.0)
    notna = X[ranked].notna().values.astype(np.float32)
    overlap = notna.T @ notna
    idx = {f: i for i, f in enumerate(ranked)}

    selected: List[str] = []
    for feat in ranked:
        i = idx[feat]
        keep = True
        for kept in selected:
            j = idx[kept]
            if overlap[i, j] >= min_overlap and corr.iat[i, j] >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(feat)

    return selected


def _direction_separation_score(
    x: np.ndarray,
    y_diff: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    min_overlap: int = 24,
) -> float:
    """
    Measure how well feature values separate up vs down target-diff regimes.

    Returns a bounded [0, 1) score via tanh(effect_size) to keep scales stable.
    """
    mask = np.isfinite(x) & np.isfinite(y_diff)
    if mask.sum() < min_overlap:
        return 0.0

    xv = x[mask].astype(float)
    yv = y_diff[mask].astype(float)
    if sample_weights is None:
        w = np.ones_like(xv, dtype=float)
    else:
        w = sample_weights[mask].astype(float)
    if float(w.sum()) <= 0:
        return 0.0
    w = w / float(w.sum())

    up = yv > 0
    down = yv < 0
    if up.sum() < 5 or down.sum() < 5:
        return 0.0

    w_up = w[up]
    w_down = w[down]
    if float(w_up.sum()) <= 0 or float(w_down.sum()) <= 0:
        return 0.0
    w_up = w_up / float(w_up.sum())
    w_down = w_down / float(w_down.sum())

    m_up = float(np.sum(w_up * xv[up]))
    m_down = float(np.sum(w_down * xv[down]))
    v_all = float(np.sum(w * (xv - float(np.sum(w * xv))) ** 2))
    if v_all <= 1e-12:
        return 0.0
    effect = abs(m_up - m_down) / np.sqrt(v_all)
    return float(np.tanh(effect))


def _weighted_sign_agreement_score(
    x_diff: np.ndarray,
    y_diff: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    min_overlap: int = 24,
) -> float:
    """
    Weighted sign-coherence between feature and target first differences.

    Returns 0 when agreement is near random (50%), and approaches 1 as
    agreement or anti-agreement becomes consistent. Anti-agreement can still
    be predictive for tree models via sign inversion splits.
    """
    mask = np.isfinite(x_diff) & np.isfinite(y_diff)
    if mask.sum() < min_overlap:
        return 0.0

    xv = x_diff[mask].astype(float)
    yv = y_diff[mask].astype(float)
    nz = (np.abs(xv) > 1e-12) & (np.abs(yv) > 1e-12)
    if nz.sum() < max(8, min_overlap // 2):
        return 0.0

    xv = xv[nz]
    yv = yv[nz]
    if sample_weights is None:
        w = np.ones_like(xv, dtype=float)
    else:
        w = sample_weights[mask][nz].astype(float)
    if float(w.sum()) <= 0:
        return 0.0
    w = w / float(w.sum())

    same_sign = np.sign(xv) == np.sign(yv)
    agreement = float(np.sum(w * same_sign.astype(float)))
    return float(np.clip(2.0 * abs(agreement - 0.5), 0.0, 1.0))


def _tail_amplitude_alignment_score(
    x: np.ndarray,
    y_abs_diff: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    min_overlap: int = 24,
    tail_quantile: float = 0.80,
) -> float:
    """
    Score whether |feature| separates high-volatility target-diff regimes.
    """
    mask = np.isfinite(x) & np.isfinite(y_abs_diff)
    if mask.sum() < min_overlap:
        return 0.0

    xv = np.abs(x[mask].astype(float))
    yv = y_abs_diff[mask].astype(float)
    if sample_weights is None:
        w = np.ones_like(xv, dtype=float)
    else:
        w = sample_weights[mask].astype(float)
    if float(w.sum()) <= 0:
        return 0.0
    w = w / float(w.sum())

    thr = float(np.quantile(yv, tail_quantile))
    hi = yv >= thr
    lo = yv < thr
    if hi.sum() < 5 or lo.sum() < 5:
        return 0.0

    w_hi = w[hi]
    w_lo = w[lo]
    if float(w_hi.sum()) <= 0 or float(w_lo.sum()) <= 0:
        return 0.0
    w_hi = w_hi / float(w_hi.sum())
    w_lo = w_lo / float(w_lo.sum())

    m_hi = float(np.sum(w_hi * xv[hi]))
    m_lo = float(np.sum(w_lo * xv[lo]))
    m_all = float(np.sum(w * xv))
    v_all = float(np.sum(w * (xv - m_all) ** 2))
    if v_all <= 1e-12:
        return 0.0

    effect = abs(m_hi - m_lo) / np.sqrt(v_all)
    return float(np.tanh(effect))


def rank_branch_target_features_dynamics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: Sequence[str],
    sample_weights: Optional[np.ndarray] = None,
    min_overlap: int = 24,
    weight_level: float = 0.45,
    weight_diff: float = 0.40,
    weight_dir: float = 0.15,
    weight_amp: float = 0.0,
    weight_sign: float = 0.0,
    weight_tail: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Rank branch-target features by a dynamics composite score.

    Base score terms:
    - |corr(x, y)|
    - |corr(diff(x), diff(y))|
    - direction_separation(x, diff(y))

    Optional variance-focused terms:
    - |corr(|diff(x)|, |diff(y)|)|
    - sign-agreement(diff(x), diff(y))
    - tail-amplitude alignment of |x| to |diff(y)|
    """
    available = [c for c in candidate_features if c in X_train.columns]
    if not available:
        return []

    X = X_train[available].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(y_train, errors="coerce").values.astype(float)
    y_diff = np.diff(y)
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)
        w_diff = w[1:] if w.shape[0] >= 2 else None
    else:
        w = None
        w_diff = None

    weights = np.array(
        [weight_level, weight_diff, weight_dir, weight_amp, weight_sign, weight_tail],
        dtype=float,
    )
    weights = np.clip(weights, 0.0, None)
    w_sum = float(weights.sum())
    if w_sum <= 0:
        weights = np.array([0.45, 0.40, 0.15, 0.0, 0.0, 0.0], dtype=float)
        w_sum = float(weights.sum())
    weights /= w_sum

    ranked: List[Tuple[str, float]] = []
    for col in available:
        x = X[col].values.astype(float)
        level_corr = _weighted_abs_corr(
            x, y, sample_weights=w, min_overlap=min_overlap
        )
        if x.size >= 2 and y_diff.size >= 1:
            diff_corr = _weighted_abs_corr(
                np.diff(x), y_diff, sample_weights=w_diff,
                min_overlap=max(12, min_overlap - 1),
            )
            dir_score = _direction_separation_score(
                x[1:], y_diff, sample_weights=w_diff,
                min_overlap=max(12, min_overlap - 1),
            )
            amp_score = _weighted_abs_corr(
                np.abs(np.diff(x)), np.abs(y_diff), sample_weights=w_diff,
                min_overlap=max(12, min_overlap - 1),
            )
            sign_score = _weighted_sign_agreement_score(
                np.diff(x), y_diff, sample_weights=w_diff,
                min_overlap=max(12, min_overlap - 1),
            )
            tail_score = _tail_amplitude_alignment_score(
                x[1:], np.abs(y_diff), sample_weights=w_diff,
                min_overlap=max(12, min_overlap - 1),
            )
        else:
            diff_corr = 0.0
            dir_score = 0.0
            amp_score = 0.0
            sign_score = 0.0
            tail_score = 0.0
        score = (
            weights[0] * level_corr
            + weights[1] * diff_corr
            + weights[2] * dir_score
            + weights[3] * amp_score
            + weights[4] * sign_score
            + weights[5] * tail_score
        )
        ranked.append((col, float(score)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def select_branch_target_features_for_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_type: str,
    candidate_features: Optional[Sequence[str]] = None,
    top_k: int = 8,
    method: str = "weighted_corr",
    corr_threshold: float = 0.90,
    min_overlap: int = 24,
    sample_weights: Optional[np.ndarray] = None,
    dynamics_weight_level: float = 0.45,
    dynamics_weight_diff: float = 0.40,
    dynamics_weight_dir: float = 0.15,
    dynamics_weight_amp: float = 0.0,
    dynamics_weight_sign: float = 0.0,
    dynamics_weight_tail: float = 0.0,
    seed: int = 42,
) -> List[str]:
    """
    Select branch-target derived features quickly for one training window.

    Args:
        X_train: Training feature matrix (no look-ahead rows).
        y_train: Training target aligned with ``X_train``.
        target_type: ``'nsa'`` or ``'sa'``.
        candidate_features: Optional pre-filtered candidate list. If None, uses
            ``partition_feature_columns(...)[\"target_branch_features\"]``.
        top_k: Maximum number of branch-target features to keep.
        method: Ranking method. Supports ``'lgbm_gain'``, ``'weighted_corr'``,
            and ``'dynamics_composite'``.
        corr_threshold: Redundancy pruning threshold.
        min_overlap: Minimum overlap for correlations.
        sample_weights: Optional point-in-time sample weights.
        dynamics_weight_level: Weight on level correlation term.
        dynamics_weight_diff: Weight on differenced correlation term.
        dynamics_weight_dir: Weight on directional separation term.
        dynamics_weight_amp: Weight on |diff(feature)| vs |diff(target)| term.
        dynamics_weight_sign: Weight on diff-sign coherence term.
        dynamics_weight_tail: Weight on tail-amplitude alignment term.
        seed: Random seed for lgbm-based rankers.
    """
    if top_k <= 0:
        return []

    if candidate_features is None:
        groups = partition_feature_columns(list(X_train.columns), target_type=target_type)
        candidates = groups["target_branch_features"]
    else:
        candidates = [c for c in candidate_features if c in X_train.columns]

    if not candidates:
        return []

    survivors = prune_redundant_by_correlation(
        X_train=X_train,
        y_train=y_train,
        candidate_features=candidates,
        corr_threshold=corr_threshold,
        min_overlap=min_overlap,
        sample_weights=sample_weights,
    )

    if len(survivors) <= top_k:
        return survivors

    if method == "dynamics_composite":
        ranked = rank_branch_target_features_dynamics(
            X_train=X_train,
            y_train=y_train,
            candidate_features=survivors,
            sample_weights=sample_weights,
            min_overlap=min_overlap,
            weight_level=dynamics_weight_level,
            weight_diff=dynamics_weight_diff,
            weight_dir=dynamics_weight_dir,
            weight_amp=dynamics_weight_amp,
            weight_sign=dynamics_weight_sign,
            weight_tail=dynamics_weight_tail,
        )
        return [feat for feat, _ in ranked[:top_k]]

    try:
        selected = select_features_for_step(
            X_train=X_train[survivors],
            y_train=y_train,
            candidate_features=survivors,
            top_k=top_k,
            method=method,
            sample_weights=sample_weights,
            seed=seed,
        )
        return selected[:top_k]
    except Exception as exc:
        logger.warning(
            "Branch-target selector fallback: ranking failed (%s). "
            "Using first %d corr-pruned features.",
            exc,
            top_k,
        )
        return survivors[:top_k]
