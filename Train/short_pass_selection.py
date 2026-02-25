"""
Fast nested feature selection for the expanding-window backtest.

Called once per backtest step on ONLY the training window data.
Reduces the candidate pool (~200 features) down to a short list (~60)
using a quick LightGBM gain ranker or weighted correlation ranking.

LEAKAGE SAFETY: This module NEVER sees test-month data.  It receives
``X_train`` and ``y_train`` which are strictly ``< target_month``.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Data_ETA_Pipeline.perf_stats import profiled

logger = setup_logger(__file__, TEMP_DIR)

# JSON-forbidden characters that break LightGBM column names
_BAD_CHARS_RE = re.compile(r'[\[\]{}:,"]+')


def _sanitize_col_names(cols: List[str]):
    """Return a mapping from original to sanitized column names."""
    mapping = {}
    for col in cols:
        safe = _BAD_CHARS_RE.sub('_', col)
        mapping[col] = safe
    return mapping


@profiled("train.short_pass.lgbm_gain")
def short_pass_lgbm_gain(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: List[str],
    top_k: int = 60,
    sample_weights: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[str]:
    """Quick LightGBM ranker: trains a shallow model and returns top_k features by gain.

    Args:
        X_train: Training features (must NOT contain ``ds``).
        y_train: Training target.
        candidate_features: Features to rank (subset of ``X_train`` columns).
        top_k: Number of features to select.
        sample_weights: Optional exponential decay weights.
        seed: Random seed for reproducibility.

    Returns:
        List of up to ``top_k`` feature names sorted by gain descending.
    """
    import lightgbm as lgb

    # Restrict to candidate columns present in X_train
    available = [c for c in candidate_features if c in X_train.columns]
    if not available:
        return []

    X = X_train[available].copy()

    # Replace inf with NaN (LightGBM handles NaN but not inf)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows where target is NaN
    valid = ~y_train.isna()
    X = X[valid]
    y = y_train[valid]
    w = sample_weights[valid.values] if sample_weights is not None else None

    if len(X) < 10:
        logger.warning("Short-pass: fewer than 10 valid rows, returning all candidates")
        return available[:top_k]

    # Sanitize column names for LightGBM
    name_map = _sanitize_col_names(available)
    reverse_map = {v: k for k, v in name_map.items()}
    X = X.rename(columns=name_map)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'max_depth': 4,
        'num_leaves': 15,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': 1,
        'random_state': seed,
    }

    ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=100)

    # Feature importance by gain
    importances = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()

    ranked = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )

    # Map sanitized names back to originals
    selected = []
    for safe_name, _ in ranked:
        original = reverse_map.get(safe_name, safe_name)
        selected.append(original)
        if len(selected) >= top_k:
            break

    return selected


def short_pass_weighted_corr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: List[str],
    top_k: int = 60,
    sample_weights: Optional[np.ndarray] = None,
) -> List[str]:
    """Weighted absolute correlation ranking.

    For each candidate feature, compute the weighted Pearson correlation
    with ``y_train``.  Rank by ``|r|`` descending, return top_k.

    Handles NaN by pairwise-complete-case weighting.

    Args:
        X_train: Training features (must NOT contain ``ds``).
        y_train: Training target.
        candidate_features: Features to rank.
        top_k: Number of features to select.
        sample_weights: Optional sample weights.

    Returns:
        List of up to ``top_k`` feature names sorted by ``|r|`` descending.
    """
    available = [c for c in candidate_features if c in X_train.columns]
    if not available:
        return []

    valid = ~y_train.isna()
    y = y_train[valid].values.astype(float)
    w = sample_weights[valid.values] if sample_weights is not None else np.ones(len(y))

    correlations = []
    for col in available:
        x = X_train.loc[valid, col].values.astype(float)
        # Pairwise complete cases
        mask = ~(np.isnan(x) | np.isinf(x))
        if mask.sum() < 10:
            correlations.append((col, 0.0))
            continue
        xm = x[mask]
        ym = y[mask]
        wm = w[mask]
        wm = wm / wm.sum()
        # Weighted means
        mx = np.average(xm, weights=wm)
        my = np.average(ym, weights=wm)
        # Weighted correlation
        cov = np.sum(wm * (xm - mx) * (ym - my))
        sx = np.sqrt(np.sum(wm * (xm - mx) ** 2))
        sy = np.sqrt(np.sum(wm * (ym - my) ** 2))
        if sx < 1e-12 or sy < 1e-12:
            correlations.append((col, 0.0))
        else:
            correlations.append((col, abs(cov / (sx * sy))))

    correlations.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in correlations[:top_k]]


@profiled("train.short_pass.select_features_for_step")
def select_features_for_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: List[str],
    top_k: int = 60,
    method: str = 'lgbm_gain',
    sample_weights: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[str]:
    """Dispatch to the appropriate short-pass method.

    Args:
        X_train: Training features (must NOT contain ``ds``).
        y_train: Training target.
        candidate_features: Features to rank.
        top_k: Number of features to select.
        method: ``'lgbm_gain'`` or ``'weighted_corr'``.
        sample_weights: Optional sample weights.
        seed: Random seed (used by lgbm_gain only).

    Returns:
        List of selected feature names (length <= ``top_k``).
    """
    if not candidate_features:
        return []

    if method == 'lgbm_gain':
        return short_pass_lgbm_gain(
            X_train, y_train, candidate_features, top_k, sample_weights, seed
        )
    elif method == 'weighted_corr':
        return short_pass_weighted_corr(
            X_train, y_train, candidate_features, top_k, sample_weights
        )
    else:
        raise ValueError(f"Unknown short-pass method: {method!r}")
