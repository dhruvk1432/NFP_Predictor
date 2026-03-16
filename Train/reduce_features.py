"""
Post-Selection Feature Reduction

Standalone script that reduces the 7-stage ETL output (~400-500 features)
down to ~60 features per branch using SHAP-based methods and correlation analysis.

Runs BETWEEN the ETL feature selection engine and the union-pool/short-pass
layers. Uses data up to a configurable PIT cutoff (default: Jan 2022, pre-backtest)
to maintain strict point-in-time correctness.

Pipeline placement:
    ETL (400-500 features) -> [THIS SCRIPT] (~60) -> overwrites JSON files
    -> candidate_pool.py -> short_pass_selection.py -> training

Usage:
    python -m Train.reduce_features
    python -m Train.reduce_features --target-n 60 --corr-threshold 0.90
    python -m Train.reduce_features --branches nsa_revised sa_revised
    python -m Train.reduce_features --dry-run
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import binomtest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import (
    ALL_TARGET_CONFIGS,
    DEFAULT_LGBM_PARAMS,
    MASTER_SNAPSHOTS_BASE,
)
from Train.data_loader import load_target_data, sanitize_feature_name
from Train.variance_metrics import compute_variance_kpis, composite_objective_score

logger = setup_logger(__file__, TEMP_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_N = 60
DEFAULT_CORR_THRESHOLD = 0.90
DEFAULT_BORUTA_ITERATIONS = 50
DEFAULT_BORUTA_ALPHA = 0.05
DEFAULT_MIN_OVERLAP = 30
DEFAULT_PIT_CUTOFF = '2022-01-01'
DEFAULT_STAGE3_REDUNDANCY_SA = 0.88
DEFAULT_STAGE3_REDUNDANCY_OTHER = 0.90
DEFAULT_STAGE3_MIN_OVERLAP = 24

# LightGBM params for quick Boruta models (shallow, fast)
BORUTA_LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'max_depth': 4,
    'num_leaves': 15,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'verbose': -1,
    'n_jobs': -1,
}

BORUTA_NUM_ROUNDS = 100
NOAA_STALENESS_SUFFIX = "__staleness_months"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Hierarchical Correlation Clustering
# ─────────────────────────────────────────────────────────────────────────────

def stage_correlation_clustering(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = DEFAULT_CORR_THRESHOLD,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    output_dir: Optional[Path] = None,
    branch_label: str = '',
) -> List[str]:
    """
    Remove redundant features via hierarchical clustering on Spearman correlation.

    For each cluster of correlated features (|r| > threshold), keeps only the
    single feature with highest |Spearman correlation with y|.

    Args:
        X: Feature matrix (may contain NaN).
        y: Target series.
        threshold: Correlation threshold for clustering (features with |r| > threshold
            are considered redundant).
        min_overlap: Minimum non-NaN observation overlap for a valid pairwise correlation.
        output_dir: Directory to save diagnostic CSV.
        branch_label: Label for logging.

    Returns:
        List of surviving feature names.
    """
    t0 = time.time()
    features = list(X.columns)
    n_features = len(features)
    logger.info(f"[{branch_label}] Stage 1: Correlation clustering on {n_features} features "
                f"(threshold={threshold})")

    if n_features <= 1:
        return features

    # Spearman correlation matrix (pairwise complete observations)
    corr = X.corr(method='spearman')

    # Overlap matrix: count of shared non-NaN observations per pair
    notna = X.notna().values.astype(np.float32)
    overlap = notna.T @ notna

    # Mask low-overlap pairs and NaN correlations: treat as uncorrelated
    low_overlap_mask = overlap < min_overlap
    corr_values = corr.values.copy()
    corr_values[low_overlap_mask] = 0.0
    corr_values = np.nan_to_num(corr_values, nan=0.0)

    # Force symmetry and zero diagonal for distance computation
    corr_values = (corr_values + corr_values.T) / 2
    np.fill_diagonal(corr_values, 1.0)

    # Distance matrix: 1 - |r|
    dist_matrix = 1.0 - np.abs(corr_values)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0, None)
    # Ensure no NaN/inf values for scipy linkage
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=0.0)

    # Hierarchical clustering
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    # Compute each feature's |correlation with y| for representative selection
    y_corr = {}
    for col in features:
        valid = X[col].notna() & y.notna()
        if valid.sum() < min_overlap:
            y_corr[col] = 0.0
        else:
            y_corr[col] = abs(X.loc[valid, col].corr(y[valid], method='spearman'))

    # Select representative from each cluster
    cluster_map: Dict[int, List[str]] = {}
    for feat, cid in zip(features, cluster_labels):
        cluster_map.setdefault(cid, []).append(feat)

    survivors = []
    diagnostic_rows = []
    for cid, members in sorted(cluster_map.items()):
        # Pick member with highest |corr with y|
        representative = max(members, key=lambda f: y_corr.get(f, 0.0))
        survivors.append(representative)
        for feat in members:
            diagnostic_rows.append({
                'feature': feat,
                'cluster_id': cid,
                'cluster_size': len(members),
                'is_representative': feat == representative,
                'abs_corr_with_y': round(y_corr.get(feat, 0.0), 4),
            })

    # Save diagnostics
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / 'stage1_clusters.csv', index=False)

    elapsed = time.time() - t0
    n_clusters = len(cluster_map)
    n_multi = sum(1 for m in cluster_map.values() if len(m) > 1)
    logger.info(f"[{branch_label}] Stage 1 done: {n_features} -> {len(survivors)} features "
                f"({n_clusters} clusters, {n_multi} multi-member, {elapsed:.1f}s)")

    return survivors


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: BorutaSHAP
# ─────────────────────────────────────────────────────────────────────────────

def _create_shadow_features(
    X: pd.DataFrame,
    rng: np.random.Generator,
    max_shadows: int = 200,
) -> pd.DataFrame:
    """
    Create shadow (permuted) features for Boruta comparison.

    For each feature, shuffles the non-NaN values while preserving NaN positions.
    If n_features > max_shadows, randomly samples a subset of features to shadow.

    Args:
        X: Real feature matrix.
        rng: NumPy random generator.
        max_shadows: Maximum number of shadow features to create.

    Returns:
        DataFrame of shadow features with 'shadow_' prefix.
    """
    cols = list(X.columns)
    if len(cols) > max_shadows:
        cols = list(rng.choice(cols, size=max_shadows, replace=False))

    shadow_data = {}
    for col in cols:
        values = X[col].values.copy()
        mask = ~np.isnan(values)
        valid_vals = values[mask].copy()
        rng.shuffle(valid_vals)
        values[mask] = valid_vals
        shadow_data[f'shadow_{col}'] = values

    return pd.DataFrame(shadow_data, index=X.index)


def stage_boruta_shap(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = DEFAULT_BORUTA_ITERATIONS,
    alpha: float = DEFAULT_BORUTA_ALPHA,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    branch_label: str = '',
) -> List[str]:
    """
    SHAP-based Boruta feature selection.

    Trains LightGBM models with real + shadow features, compares SHAP importances.
    Features that consistently beat shadow features are confirmed as important.

    Args:
        X: Feature matrix (may contain NaN).
        y: Target series.
        n_iterations: Number of Boruta iterations.
        alpha: Significance level for binomial test.
        seed: Random seed.
        output_dir: Directory to save diagnostic CSV.
        branch_label: Label for logging.

    Returns:
        List of confirmed + tentative feature names.
    """
    t0 = time.time()
    features = list(X.columns)
    n_features = len(features)
    logger.info(f"[{branch_label}] Stage 2: BorutaSHAP on {n_features} features "
                f"({n_iterations} iterations, alpha={alpha})")

    if n_features == 0:
        return []

    rng = np.random.default_rng(seed)
    hits = np.zeros(n_features, dtype=int)

    # Replace inf with NaN for LightGBM
    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Early stopping tracking
    half_point = n_iterations // 2
    active_mask = np.ones(n_features, dtype=bool)  # features still in consideration

    for iteration in range(n_iterations):
        # Create shadow features
        shadow_df = _create_shadow_features(X_clean, rng)
        X_combined = pd.concat([X_clean, shadow_df], axis=1)

        # Train LightGBM
        params = {**BORUTA_LGB_PARAMS, 'random_state': seed + iteration}
        ds = lgb.Dataset(X_combined, label=y, free_raw_data=False)

        try:
            model = lgb.train(params, ds, num_boost_round=BORUTA_NUM_ROUNDS)
        except Exception as e:
            logger.warning(f"[{branch_label}] Boruta iteration {iteration} failed: {e}")
            continue

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_combined)

        # Mean |SHAP| per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Split into real and shadow importances
        n_real = n_features
        real_importances = mean_abs_shap[:n_real]
        shadow_importances = mean_abs_shap[n_real:]

        # Shadow threshold: use percentile of shadow importances for robustness.
        # Max is too strict when there are many shadow features (one can get lucky).
        # The 75th percentile matches the approach in feature_selection_engine._boruta_core().
        if len(shadow_importances) > 0:
            shadow_threshold = np.percentile(shadow_importances, 75)
        else:
            shadow_threshold = 0.0

        # Record hits for real features that beat the shadow threshold
        for j in range(n_features):
            if active_mask[j] and real_importances[j] > shadow_threshold:
                hits[j] += 1

        # Early stopping at halfway point
        if iteration == half_point and half_point > 0:
            for j in range(n_features):
                if active_mask[j] and hits[j] == 0:
                    active_mask[j] = False

            n_active = active_mask.sum()
            n_early_rejected = n_features - n_active
            if n_early_rejected > 0:
                logger.info(f"[{branch_label}] Boruta early stop: rejected {n_early_rejected} "
                            f"features with 0 hits at iteration {iteration}")

        if (iteration + 1) % 10 == 0:
            logger.info(f"[{branch_label}] Boruta iteration {iteration + 1}/{n_iterations} "
                        f"({active_mask.sum()} active features)")

    # Classify features via binomial test
    confirmed = []
    tentative = []
    rejected = []
    diagnostic_rows = []

    for j, feat in enumerate(features):
        n_hits = int(hits[j])
        # binomial test: is this feature significantly better than random (p=0.5)?
        if n_hits > 0:
            p_val = binomtest(n_hits, n_iterations, 0.5, alternative='greater').pvalue
        else:
            p_val = 1.0

        if p_val < alpha:
            status = 'confirmed'
            confirmed.append(feat)
        elif p_val < 2 * alpha:
            status = 'tentative'
            tentative.append(feat)
        else:
            status = 'rejected'
            rejected.append(feat)

        diagnostic_rows.append({
            'feature': feat,
            'hits': n_hits,
            'n_iterations': n_iterations,
            'p_value': round(p_val, 6),
            'status': status,
        })

    # Save diagnostics
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        diag_df = pd.DataFrame(diagnostic_rows).sort_values('p_value')
        diag_df.to_csv(output_dir / 'stage2_boruta_hits.csv', index=False)

    survivors = confirmed + tentative
    elapsed = time.time() - t0
    logger.info(f"[{branch_label}] Stage 2 done: {n_features} -> {len(survivors)} features "
                f"({len(confirmed)} confirmed, {len(tentative)} tentative, "
                f"{len(rejected)} rejected, {elapsed:.1f}s)")

    return survivors


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Final SHAP Ranking
# ─────────────────────────────────────────────────────────────────────────────

def _safe_abs_spearman(
    x: pd.Series,
    y: pd.Series,
    min_overlap: int = DEFAULT_STAGE3_MIN_OVERLAP,
) -> float:
    """Absolute Spearman correlation with overlap guardrails."""
    xv = pd.to_numeric(x, errors='coerce')
    yv = pd.to_numeric(y, errors='coerce')
    valid = xv.notna() & yv.notna()
    if int(valid.sum()) < min_overlap:
        return 0.0
    corr = xv[valid].corr(yv[valid], method='spearman')
    if pd.isna(corr):
        return 0.0
    return float(abs(corr))


def _safe_abs_diff_spearman(
    x: pd.Series,
    y: pd.Series,
    min_overlap: int = DEFAULT_STAGE3_MIN_OVERLAP,
) -> float:
    """Absolute Spearman correlation on first differences."""
    dx = pd.to_numeric(x, errors='coerce').diff()
    dy = pd.to_numeric(y, errors='coerce').diff()
    return _safe_abs_spearman(dx, dy, min_overlap=max(12, min_overlap - 1))


def _direction_separation_score(
    x: pd.Series,
    y: pd.Series,
    min_overlap: int = DEFAULT_STAGE3_MIN_OVERLAP,
) -> float:
    """
    Score how strongly feature values separate up-vs-down target-diff regimes.
    Returns a bounded [0, 1) score.
    """
    xv = pd.to_numeric(x, errors='coerce')
    dy = pd.to_numeric(y, errors='coerce').diff()
    valid = xv.notna() & dy.notna()
    if int(valid.sum()) < min_overlap:
        return 0.0

    x_valid = xv[valid].values.astype(float)
    dy_valid = dy[valid].values.astype(float)
    up = dy_valid > 0
    down = dy_valid < 0
    if int(up.sum()) < 6 or int(down.sum()) < 6:
        return 0.0

    mean_up = float(np.mean(x_valid[up]))
    mean_down = float(np.mean(x_valid[down]))
    std_all = float(np.std(x_valid))
    if std_all <= 1e-12:
        return 0.0
    effect = abs(mean_up - mean_down) / std_all
    return float(np.tanh(effect))


def _amplitude_link_score(
    x: pd.Series,
    y: pd.Series,
    min_overlap: int = DEFAULT_STAGE3_MIN_OVERLAP,
) -> float:
    """
    Correlation between absolute feature changes and absolute target changes.
    Useful for variance-amplitude capture.
    """
    ax = pd.to_numeric(x, errors='coerce').diff().abs()
    ay = pd.to_numeric(y, errors='coerce').diff().abs()
    return _safe_abs_spearman(ax, ay, min_overlap=max(12, min_overlap - 1))


def _rank_features_variance_aware(
    X: pd.DataFrame,
    y: pd.Series,
    shap_scores: Dict[str, float],
    branch_label: str,
) -> pd.DataFrame:
    """
    Build a variance-aware feature ranking by combining:
    SHAP importance + level corr + diff corr + direction + amplitude linkage.
    """
    features = list(X.columns)
    if not features:
        return pd.DataFrame(columns=['feature', 'combined_score'])

    shap_series = pd.Series(shap_scores, dtype=float).reindex(features).fillna(0.0)
    shap_rank = shap_series.rank(method='average', ascending=False)
    shap_rank_score = 1.0 - ((shap_rank - 1.0) / max(1.0, float(len(features) - 1)))

    level_corr = []
    diff_corr = []
    dir_score = []
    amp_score = []

    for feat in features:
        x = X[feat]
        level_corr.append(_safe_abs_spearman(x, y))
        diff_corr.append(_safe_abs_diff_spearman(x, y))
        dir_score.append(_direction_separation_score(x, y))
        amp_score.append(_amplitude_link_score(x, y))

    out = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': shap_series.values.astype(float),
        'shap_rank_score': shap_rank_score.values.astype(float),
        'level_corr': np.asarray(level_corr, dtype=float),
        'diff_corr': np.asarray(diff_corr, dtype=float),
        'dir_score': np.asarray(dir_score, dtype=float),
        'amp_score': np.asarray(amp_score, dtype=float),
    })

    is_sa_branch = str(branch_label).startswith('sa_')
    if is_sa_branch:
        # SA branches are variance-priority in your objective.
        w_shap, w_level, w_diff, w_dir, w_amp = 0.35, 0.15, 0.25, 0.15, 0.10
    else:
        w_shap, w_level, w_diff, w_dir, w_amp = 0.45, 0.30, 0.15, 0.05, 0.05

    out['combined_score'] = (
        w_shap * out['shap_rank_score']
        + w_level * out['level_corr']
        + w_diff * out['diff_corr']
        + w_dir * out['dir_score']
        + w_amp * out['amp_score']
    )
    out = out.sort_values('combined_score', ascending=False).reset_index(drop=True)
    out['variance_rank'] = np.arange(1, len(out) + 1)
    return out


def _redundancy_prune_ranked(
    X: pd.DataFrame,
    ranked_features: List[str],
    target_n: int,
    corr_threshold: float,
    min_overlap: int = DEFAULT_STAGE3_MIN_OVERLAP,
) -> List[str]:
    """Greedy redundancy pruning following ranking priority."""
    ranked = [f for f in ranked_features if f in X.columns]
    if target_n <= 0 or not ranked:
        return []

    X_rank = X[ranked]
    corr = X_rank.corr(method='spearman').abs().fillna(0.0)
    notna = X_rank.notna().values.astype(np.float32)
    overlap = notna.T @ notna
    idx = {f: i for i, f in enumerate(ranked)}

    selected: List[str] = []
    for feat in ranked:
        i = idx[feat]
        keep = True
        for kept in selected:
            j = idx[kept]
            if overlap[i, j] >= min_overlap and float(corr.iat[i, j]) >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(feat)
            if len(selected) >= target_n:
                break

    return selected


def _evaluate_feature_set_objective(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate one feature set on a chronological holdout using
    error + variance-capture composite objective.
    """
    feat = [f for f in features if f in X.columns]
    if len(feat) < 2:
        return {"objective": float("inf"), "rmse": float("inf"), "mae": float("inf")}

    X_set = X[feat].replace([np.inf, -np.inf], np.nan)
    n_train = int(len(X_set) * 0.8)
    if n_train < 36 or len(X_set) - n_train < 12:
        return {"objective": float("inf"), "rmse": float("inf"), "mae": float("inf")}

    X_train, X_val = X_set.iloc[:n_train], X_set.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]

    params = {
        **DEFAULT_LGBM_PARAMS,
        'random_state': seed,
    }
    ds_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, free_raw_data=False)
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=400,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict(X_val)
    actual = y_val.values.astype(float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    if int(valid.sum()) < 12:
        return {"objective": float("inf"), "rmse": float("inf"), "mae": float("inf")}

    actual = actual[valid]
    pred = pred[valid]
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    kpis = compute_variance_kpis(actual, pred)

    comp = composite_objective_score(
        mae=mae,
        std_ratio=float(kpis["std_ratio"]),
        diff_std_ratio=float(kpis["diff_std_ratio"]),
        tail_mae=float(kpis["tail_mae"]),
        corr_diff=float(kpis["corr_diff"]),
        diff_sign_accuracy=float(kpis["diff_sign_accuracy"]),
        lambda_std_ratio=20.0,
        lambda_diff_std_ratio=22.0,
        lambda_tail_mae=0.15,
        lambda_corr_diff=14.0,
        lambda_diff_sign=10.0,
    )
    objective = float(comp + 0.35 * rmse)
    return {
        "objective": objective,
        "rmse": rmse,
        "mae": mae,
        "std_ratio": float(kpis["std_ratio"]),
        "diff_std_ratio": float(kpis["diff_std_ratio"]),
        "corr_diff": float(kpis["corr_diff"]),
        "diff_sign_accuracy": float(kpis["diff_sign_accuracy"]),
        "tail_mae": float(kpis["tail_mae"]),
    }


def stage_final_shap_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    target_n: int = DEFAULT_TARGET_N,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    branch_label: str = '',
) -> List[str]:
    """Final ranking that balances drift accuracy and variance capture."""
    t0 = time.time()
    features = list(X.columns)
    n_features = len(features)
    logger.info(f"[{branch_label}] Stage 3: Variance-aware final ranking on {n_features} features "
                f"(target_n={target_n})")

    if n_features <= target_n:
        logger.info(f"[{branch_label}] Already at or below target, returning all {n_features}")
        return features

    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Chronological train/val split (80/20)
    n_train = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X_clean.iloc[n_train:]
    y_val = y.iloc[n_train:]

    params = {
        **DEFAULT_LGBM_PARAMS,
        'random_state': seed,
    }

    ds_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, free_raw_data=False)

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Compute SHAP values on full dataset for stable importance estimates
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[0]
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
    shap_score_map = {feat: float(score) for feat, score in zip(features, mean_abs_shap)}

    # Baseline SHAP ranking
    ranked_indices = np.argsort(mean_abs_shap)[::-1]
    ranked_features = [(features[i], float(mean_abs_shap[i])) for i in ranked_indices]
    selected_shap = [feat for feat, _ in ranked_features[:target_n]]

    # Variance-aware re-ranking
    variance_rank_df = _rank_features_variance_aware(
        X=X_clean,
        y=y,
        shap_scores=shap_score_map,
        branch_label=branch_label,
    )
    variance_ranked = variance_rank_df["feature"].tolist()
    redundancy_threshold = (
        DEFAULT_STAGE3_REDUNDANCY_SA
        if str(branch_label).startswith('sa_')
        else DEFAULT_STAGE3_REDUNDANCY_OTHER
    )
    selected_variance = _redundancy_prune_ranked(
        X=X_clean,
        ranked_features=variance_ranked,
        target_n=target_n,
        corr_threshold=redundancy_threshold,
    )
    if len(selected_variance) < target_n:
        selected_set = set(selected_variance)
        for feat, _ in ranked_features:
            if feat not in selected_set:
                selected_variance.append(feat)
                selected_set.add(feat)
                if len(selected_variance) >= target_n:
                    break
        selected_variance = selected_variance[:target_n]

    # Compare candidate sets with a variance-aware objective; select better.
    eval_shap = _evaluate_feature_set_objective(X_clean, y, selected_shap, seed=seed)
    eval_variance = _evaluate_feature_set_objective(X_clean, y, selected_variance, seed=seed)
    shap_obj = float(eval_shap.get("objective", np.inf))
    variance_obj = float(eval_variance.get("objective", np.inf))
    if variance_obj <= shap_obj:
        selected = selected_variance
        chosen = "variance_aware"
        chosen_eval = eval_variance
    else:
        selected = selected_shap
        chosen = "pure_shap"
        chosen_eval = eval_shap

    # Save diagnostics
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        variance_rank_map = {
            row["feature"]: row for _, row in variance_rank_df.iterrows()
        }
        shap_selected_set = set(selected_shap)
        variance_selected_set = set(selected_variance)
        final_selected_set = set(selected)
        diag_rows = []
        for rank, (feat, importance) in enumerate(ranked_features, 1):
            vrow = variance_rank_map.get(feat, {})
            diag_rows.append({
                'rank': rank,
                'feature': feat,
                'mean_abs_shap': round(importance, 6),
                'variance_rank': int(vrow.get('variance_rank', 0)),
                'shap_rank_score': round(float(vrow.get('shap_rank_score', 0.0)), 6),
                'level_corr': round(float(vrow.get('level_corr', 0.0)), 6),
                'diff_corr': round(float(vrow.get('diff_corr', 0.0)), 6),
                'dir_score': round(float(vrow.get('dir_score', 0.0)), 6),
                'amp_score': round(float(vrow.get('amp_score', 0.0)), 6),
                'combined_score': round(float(vrow.get('combined_score', 0.0)), 6),
                'selected_shap': feat in shap_selected_set,
                'selected_variance': feat in variance_selected_set,
                'selected_final': feat in final_selected_set,
            })
        pd.DataFrame(diag_rows).to_csv(output_dir / 'stage3_shap_ranking.csv', index=False)
        objective_payload = {
            "branch": branch_label,
            "selected_strategy": chosen,
            "target_n": int(target_n),
            "redundancy_threshold": float(redundancy_threshold),
            "eval_pure_shap": eval_shap,
            "eval_variance_aware": eval_variance,
            "eval_selected": chosen_eval,
            "selected_features": selected,
        }
        with open(output_dir / 'stage3_objective_comparison.json', 'w') as f:
            json.dump(objective_payload, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"[{branch_label}] Stage 3 done: {n_features} -> {len(selected)} features "
        f"(strategy={chosen}, obj_shap={shap_obj:.3f}, obj_var={variance_obj:.3f}, {elapsed:.1f}s)"
    )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_data(
    target_type: str,
    release_type: str,
    target_source: str,
    pit_cutoff: str,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load or build training dataset for a branch, restricted to pre-cutoff data.

    Optionally caches the result to disk to speed up re-runs.

    Returns:
        (X, y) where X has feature columns only (no ds), y is non-NaN target.
    """
    cache_x = None
    cache_y = None
    if cache_dir:
        cache_x = cache_dir / f"X_{target_type}_{target_source}.parquet"
        cache_y = cache_dir / f"y_{target_type}_{target_source}.parquet"

    # Check for cached version
    if cache_dir and not force_rebuild:
        if cache_x.exists() and cache_y.exists():
            logger.info(f"Loading cached training data from {cache_dir}")
            X = pd.read_parquet(cache_x)
            y = pd.read_parquet(cache_y).squeeze()
            return X, y

    # Build from scratch using the training pipeline
    from Train.train_lightgbm_nfp import build_training_dataset

    target_df = load_target_data(target_type, release_type, target_source)

    X, y = build_training_dataset(
        target_df,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
        end_date=pd.Timestamp(pit_cutoff),
        show_progress=True,
    )

    # Drop ds column
    if 'ds' in X.columns:
        X = X.drop(columns=['ds'])

    # Filter NaN targets
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    # Cache for re-runs
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache_x)
        y.to_frame().to_parquet(cache_y)
        logger.info(f"Cached training data to {cache_dir}")

    return X, y


def _collect_snapshot_schema_features_pre_cutoff(
    target_type: str,
    target_source: str,
    pit_cutoff: str,
) -> set[str]:
    """
    Collect sanitized feature names present in branch snapshot schemas up to pit_cutoff month.

    Uses parquet schema reads only (fast, no table load).
    """
    base_dir = MASTER_SNAPSHOTS_BASE / target_type / target_source / "decades"
    cutoff_month = pd.Timestamp(pit_cutoff).replace(day=1)
    features: set[str] = set()
    scanned = 0

    for path in sorted(base_dir.rglob("*.parquet")):
        try:
            month = pd.Timestamp(path.stem).replace(day=1)
        except Exception:
            continue

        if month > cutoff_month:
            continue

        try:
            cols = pq.ParquetFile(path).schema_arrow.names
        except Exception as e:
            logger.warning(f"Failed to read schema for {path}: {e}")
            continue

        scanned += 1
        for col in cols:
            if col in {"date", "snapshot_date"}:
                continue
            features.add(sanitize_feature_name(str(col)))

    logger.info(
        f"[{target_type}_{target_source}] Pre-cutoff snapshot schema universe: "
        f"{len(features)} features from {scanned} files (<= {cutoff_month.strftime('%Y-%m')})"
    )
    return features


def _load_feature_universe_for_reduction(
    target_type: str,
    target_source: str,
    target_n: int,
) -> tuple[list[str], str]:
    """
    Load the feature universe to reduce from.

    Prefer the branch cache JSON by default. If that file already appears reduced
    (e.g., output of this script), fallback to the latest regime cache snapshot,
    which preserves the original 400-500 ETL feature universe.
    """
    branch_cache = MASTER_SNAPSHOTS_BASE / f"selected_features_{target_type}_{target_source}.json"
    with open(branch_cache, "r") as f:
        branch_payload = json.load(f)
    branch_features = branch_payload.get("features", [])
    if not isinstance(branch_features, list) or not all(isinstance(x, str) for x in branch_features):
        raise ValueError(f"Invalid feature payload in {branch_cache}")

    reduction_method = str(branch_payload.get("reduction_method", "") or "")
    appears_reduced = reduction_method in {
        "boruta_shap_3stage",
        "boruta_shap_3stage_variance_aware",
    } or len(branch_features) <= target_n
    if not appears_reduced:
        return branch_features, str(branch_cache)

    regime_dir = MASTER_SNAPSHOTS_BASE / "regime_caches"
    regime_candidates = sorted(regime_dir.glob(f"selected_features_{target_type}_{target_source}_*.json"))
    if not regime_candidates:
        return branch_features, str(branch_cache)

    latest_regime = regime_candidates[-1]
    with open(latest_regime, "r") as f:
        regime_payload = json.load(f)
    regime_features = regime_payload.get("features", [])
    if isinstance(regime_features, list) and all(isinstance(x, str) for x in regime_features):
        if len(regime_features) > len(branch_features):
            return regime_features, str(latest_regime)

    return branch_features, str(branch_cache)


def reduce_features_for_branch(
    target_type: str,
    target_source: str,
    target_n: int = DEFAULT_TARGET_N,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    boruta_iterations: int = DEFAULT_BORUTA_ITERATIONS,
    boruta_alpha: float = DEFAULT_BORUTA_ALPHA,
    pit_cutoff: str = DEFAULT_PIT_CUTOFF,
    dry_run: bool = False,
    output_base: Optional[Path] = None,
    refresh_training_cache: bool = False,
) -> List[str]:
    """
    Full 3-stage feature reduction for a single branch.

    Args:
        target_type: 'nsa' or 'sa'
        target_source: 'revised'
        target_n: Target number of features after reduction.
        corr_threshold: Correlation clustering threshold.
        boruta_iterations: Number of BorutaSHAP iterations.
        boruta_alpha: Significance level for Boruta binomial test.
        pit_cutoff: Point-in-time cutoff date string (YYYY-MM-DD).
        dry_run: If True, don't overwrite JSON files.
        output_base: Base directory for diagnostic outputs.

    Returns:
        List of final selected feature names (raw/original names).
    """
    branch_label = f"{target_type}_{target_source}"
    release_type = 'first'  # Always 'first' per ALL_TARGET_CONFIGS

    if output_base is None:
        output_base = OUTPUT_DIR / "feature_reduction"
    output_dir = output_base / branch_label
    cache_dir = output_base / "training_cache"

    logger.info("=" * 70)
    logger.info(f"FEATURE REDUCTION: {branch_label.upper()}")
    logger.info("=" * 70)

    # ── Load feature universe for reduction (prefer full pre-reduction universe) ──
    current_features, current_feature_source = _load_feature_universe_for_reduction(
        target_type=target_type, target_source=target_source, target_n=target_n
    )
    logger.info(
        f"[{branch_label}] Reduction feature universe count: {len(current_features)} "
        f"(source: {current_feature_source})"
    )

    # ── Load training data ──
    t0 = time.time()
    X, y = _load_training_data(
        target_type,
        release_type,
        target_source,
        pit_cutoff,
        cache_dir,
        force_rebuild=refresh_training_cache,
    )
    logger.info(f"[{branch_label}] Training data: {X.shape[0]} samples, {X.shape[1]} columns "
                f"({time.time() - t0:.1f}s)")

    # ── Identify data features vs meta/target columns ──
    # Exclude: lagged target features (nfp_*), revision features (rev_*),
    # calendar features (is_*, month_*, quarter_*, year, weeks_*), and snapshot_date
    _META_PREFIXES = ('nfp_nsa_', 'nfp_sa_', 'rev_master_', 'rev_', 'is_', 'month_',
                      'quarter_', 'weeks_')
    _META_EXACT = {'year', 'snapshot_date'}

    feature_cols = []
    excluded = []
    for c in X.columns:
        if c in _META_EXACT or any(c.startswith(p) for p in _META_PREFIXES):
            excluded.append(c)
        else:
            feature_cols.append(c)

    logger.info(f"[{branch_label}] Data features in X: {len(feature_cols)}, "
                f"excluded meta/target cols: {len(excluded)} "
                f"({', '.join(sorted(excluded)[:5])}...)")

    if not feature_cols:
        logger.error(f"[{branch_label}] No data features found in X! Aborting.")
        return []

    # ── Strict candidate universe: original branch feature set ∩ pre-cutoff snapshot schema ──
    # Normalize feature-universe names to the same sanitized namespace used by X columns.
    current_feature_set = {sanitize_feature_name(str(f)) for f in current_features}
    schema_feature_set = _collect_snapshot_schema_features_pre_cutoff(
        target_type=target_type, target_source=target_source, pit_cutoff=pit_cutoff
    )

    dropped_not_in_json = [c for c in feature_cols if c not in current_feature_set]
    dropped_not_in_schema = [c for c in feature_cols if c in current_feature_set and c not in schema_feature_set]
    dropped_synthetic = [c for c in feature_cols if c.endswith(NOAA_STALENESS_SUFFIX)]

    eligible_features = [
        c for c in feature_cols
        if c in current_feature_set
        and c in schema_feature_set
        and not c.endswith(NOAA_STALENESS_SUFFIX)
    ]

    if len(current_feature_set) != len(current_features):
        logger.warning(
            f"[{branch_label}] Sanitization collapsed feature-universe names: "
            f"raw={len(current_features)} -> sanitized={len(current_feature_set)}"
        )
    logger.info(
        f"[{branch_label}] Eligible feature universe: {len(eligible_features)} "
        f"(from X={len(feature_cols)}, in_json_sanitized={len(current_feature_set)}, "
        f"in_schema={len(schema_feature_set)})"
    )
    if dropped_not_in_json:
        logger.info(
            f"[{branch_label}] Dropped not in current JSON universe: {len(dropped_not_in_json)} "
            f"(sample: {dropped_not_in_json[:5]})"
        )
    if dropped_not_in_schema:
        logger.info(
            f"[{branch_label}] Dropped not in pre-cutoff snapshot schema: {len(dropped_not_in_schema)} "
            f"(sample: {dropped_not_in_schema[:5]})"
        )
    if dropped_synthetic:
        logger.info(
            f"[{branch_label}] Dropped synthetic (non-snapshot) features: {len(dropped_synthetic)} "
            f"(sample: {dropped_synthetic[:5]})"
        )

    if len(eligible_features) < target_n:
        raise ValueError(
            f"[{branch_label}] Only {len(eligible_features)} eligible features available, "
            f"cannot select target_n={target_n}"
        )

    X_feat = X[eligible_features]

    # ── Stage 1: Correlation Clustering ──
    survivors_s1 = stage_correlation_clustering(
        X_feat, y, threshold=corr_threshold, output_dir=output_dir, branch_label=branch_label,
    )
    X_s1 = X_feat[survivors_s1]

    # ── Stage 2: BorutaSHAP ──
    survivors_s2 = stage_boruta_shap(
        X_s1, y, n_iterations=boruta_iterations, alpha=boruta_alpha,
        output_dir=output_dir, branch_label=branch_label,
    )
    X_s2 = X_s1[survivors_s2]

    # ── Stage 3: Final SHAP Ranking ──
    # If Boruta is overly strict (< target_n), fallback to Stage-1 survivors to still output target_n.
    stage3_input = X_s2
    stage3_source = "stage2"
    if X_s2.shape[1] < target_n:
        logger.warning(
            f"[{branch_label}] Stage 2 returned {X_s2.shape[1]} features (< {target_n}); "
            f"falling back to Stage 1 survivors for final ranking."
        )
        stage3_input = X_s1
        stage3_source = "stage1_fallback"

    final_features = stage_final_shap_ranking(
        stage3_input, y, target_n=target_n, output_dir=output_dir, branch_label=branch_label,
    )
    final_features.sort()

    # Final safety check: selected features must be inside original branch feature universe.
    invalid_final = [f for f in final_features if f not in current_feature_set]
    if invalid_final:
        raise ValueError(
            f"[{branch_label}] Final features not in original JSON universe: {invalid_final[:10]}"
        )
    invalid_schema = [f for f in final_features if f not in schema_feature_set]
    if invalid_schema:
        raise ValueError(
            f"[{branch_label}] Final features not in pre-cutoff snapshot schema: {invalid_schema[:10]}"
        )

    # ── Save summary ──
    summary = {
        'branch': branch_label,
        'pit_cutoff': pit_cutoff,
        'original_json_count': len(current_features),
        'stage1_input': len(eligible_features),
        'stage1_output': len(survivors_s1),
        'stage2_output': len(survivors_s2),
        'stage3_input_source': stage3_source,
        'stage3_input_count': int(stage3_input.shape[1]),
        'final_count': len(final_features),
        'target_n': target_n,
        'corr_threshold': corr_threshold,
        'boruta_iterations': boruta_iterations,
        'boruta_alpha': boruta_alpha,
        'final_features': final_features,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    logger.info(f"[{branch_label}] REDUCTION COMPLETE: "
                f"{len(feature_cols)} -> {len(survivors_s1)} -> {len(survivors_s2)} "
                f"-> {len(final_features)} features")

    # ── Overwrite JSON ──
    if dry_run:
        logger.info(f"[{branch_label}] DRY RUN — not overwriting JSON")
    else:
        json_path = MASTER_SNAPSHOTS_BASE / f"selected_features_{target_type}_{target_source}.json"
        payload = {
            "last_run_date": str(date.today()),
            "target_source": target_source,
            "target_cat": target_type,
            "reduction_method": "boruta_shap_3stage_variance_aware",
            "original_json_count": len(current_features),
            "stage1_count": len(survivors_s1),
            "stage2_count": len(survivors_s2),
            "final_count": len(final_features),
            "features": final_features,
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info(f"[{branch_label}] Wrote {len(final_features)} features to {json_path}")

    return final_features


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Post-selection feature reduction via BorutaSHAP (400-500 -> ~60 features)',
    )
    parser.add_argument(
        '--target-n', type=int, default=DEFAULT_TARGET_N,
        help=f'Target number of features (default: {DEFAULT_TARGET_N})',
    )
    parser.add_argument(
        '--corr-threshold', type=float, default=DEFAULT_CORR_THRESHOLD,
        help=f'Correlation clustering threshold (default: {DEFAULT_CORR_THRESHOLD})',
    )
    parser.add_argument(
        '--boruta-iterations', type=int, default=DEFAULT_BORUTA_ITERATIONS,
        help=f'Number of BorutaSHAP iterations (default: {DEFAULT_BORUTA_ITERATIONS})',
    )
    parser.add_argument(
        '--boruta-alpha', type=float, default=DEFAULT_BORUTA_ALPHA,
        help=f'BorutaSHAP significance level (default: {DEFAULT_BORUTA_ALPHA})',
    )
    parser.add_argument(
        '--branches', nargs='+', default=None,
        help='Specific branches to process (e.g., nsa_revised sa_revised). '
             'Default: all branches.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run without overwriting JSON files',
    )
    parser.add_argument(
        '--pit-cutoff', type=str, default=DEFAULT_PIT_CUTOFF,
        help=f'Point-in-time cutoff date YYYY-MM-DD (default: {DEFAULT_PIT_CUTOFF})',
    )
    parser.add_argument(
        '--refresh-training-cache', action='store_true',
        help='Rebuild X/y training caches from current master snapshots instead of reusing cached parquet.',
    )

    args = parser.parse_args()

    # Determine which branches to process
    all_branches = [
        ('nsa', 'revised'),
        ('sa', 'revised'),
    ]

    if args.branches:
        branches = []
        for b in args.branches:
            parts = b.split('_', 1)
            if len(parts) == 2 and parts[0] in ('nsa', 'sa'):
                branches.append((parts[0], parts[1]))
            else:
                logger.error(f"Invalid branch format: {b}. Expected nsa_revised, etc.")
                sys.exit(1)
    else:
        branches = all_branches

    logger.info(f"Feature reduction: {len(branches)} branches, target_n={args.target_n}, "
                f"pit_cutoff={args.pit_cutoff}")

    results = {}
    total_t0 = time.time()

    for target_type, target_source in branches:
        branch_label = f"{target_type}_{target_source}"
        try:
            features = reduce_features_for_branch(
                target_type=target_type,
                target_source=target_source,
                target_n=args.target_n,
                corr_threshold=args.corr_threshold,
                boruta_iterations=args.boruta_iterations,
                boruta_alpha=args.boruta_alpha,
                pit_cutoff=args.pit_cutoff,
                dry_run=args.dry_run,
                refresh_training_cache=args.refresh_training_cache,
            )
            results[branch_label] = len(features)
        except Exception as e:
            logger.error(f"[{branch_label}] Failed: {e}", exc_info=True)
            results[branch_label] = f"ERROR: {e}"

    # Final summary
    total_elapsed = time.time() - total_t0
    logger.info("=" * 70)
    logger.info("FEATURE REDUCTION SUMMARY")
    logger.info("=" * 70)
    for branch, count in results.items():
        logger.info(f"  {branch}: {count} features")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    if args.dry_run:
        logger.info("DRY RUN — no JSON files were modified")


if __name__ == '__main__':
    main()
