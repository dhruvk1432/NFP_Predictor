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
from typing import Dict, List, Optional, Sequence
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
        method: Ranking method passed to ``select_features_for_step``.
        corr_threshold: Redundancy pruning threshold.
        min_overlap: Minimum overlap for correlations.
        sample_weights: Optional point-in-time sample weights.
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
