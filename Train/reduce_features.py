"""
Post-Selection Feature Reduction

Standalone script that reduces the 7-stage ETL output (~400-500 features)
down to ~50 features per branch using SHAP-based methods and correlation analysis.

Runs BETWEEN the ETL feature selection engine and the union-pool/short-pass
layers. Uses data up to a configurable PIT cutoff (default: Jan 2022, pre-backtest)
to maintain strict point-in-time correctness.

Pipeline placement:
    ETL (400-500 features) -> [THIS SCRIPT] (~50) -> overwrites JSON files
    -> candidate_pool.py -> short_pass_selection.py -> training

Usage:
    python -m Train.reduce_features
    python -m Train.reduce_features --target-n 50 --corr-threshold 0.90
    python -m Train.reduce_features --branches nsa_first_release sa_revised
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
    load_selected_features,
)
from Train.data_loader import load_target_data, sanitize_feature_name

logger = setup_logger(__file__, TEMP_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_N = 50
DEFAULT_CORR_THRESHOLD = 0.90
DEFAULT_BORUTA_ITERATIONS = 50
DEFAULT_BORUTA_ALPHA = 0.05
DEFAULT_MIN_OVERLAP = 30
DEFAULT_PIT_CUTOFF = '2022-01-01'

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

def stage_final_shap_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    target_n: int = DEFAULT_TARGET_N,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    branch_label: str = '',
) -> List[str]:
    """
    Train a full LightGBM model and rank features by mean |SHAP| importance.

    Uses chronological train/validation split with early stopping for a
    well-calibrated model, then takes the top N features by SHAP importance.

    Args:
        X: Feature matrix (may contain NaN).
        y: Target series.
        target_n: Number of features to keep.
        seed: Random seed.
        output_dir: Directory to save diagnostic CSV.
        branch_label: Label for logging.

    Returns:
        List of top features by SHAP importance.
    """
    t0 = time.time()
    features = list(X.columns)
    n_features = len(features)
    logger.info(f"[{branch_label}] Stage 3: Final SHAP ranking on {n_features} features "
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
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Rank features
    ranked_indices = np.argsort(mean_abs_shap)[::-1]
    ranked_features = [(features[i], float(mean_abs_shap[i])) for i in ranked_indices]

    # Take top N
    selected = [feat for feat, _ in ranked_features[:target_n]]

    # Save diagnostics
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        diag_rows = []
        for rank, (feat, importance) in enumerate(ranked_features, 1):
            diag_rows.append({
                'rank': rank,
                'feature': feat,
                'mean_abs_shap': round(importance, 6),
                'selected': feat in set(selected),
            })
        pd.DataFrame(diag_rows).to_csv(output_dir / 'stage3_shap_ranking.csv', index=False)

    elapsed = time.time() - t0
    logger.info(f"[{branch_label}] Stage 3 done: {n_features} -> {len(selected)} features "
                f"({elapsed:.1f}s)")

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
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load or build training dataset for a branch, restricted to pre-cutoff data.

    Optionally caches the result to disk to speed up re-runs.

    Returns:
        (X, y) where X has feature columns only (no ds), y is non-NaN target.
    """
    # Check for cached version
    if cache_dir:
        cache_x = cache_dir / f"X_{target_type}_{target_source}.parquet"
        cache_y = cache_dir / f"y_{target_type}_{target_source}.parquet"
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
) -> List[str]:
    """
    Full 3-stage feature reduction for a single branch.

    Args:
        target_type: 'nsa' or 'sa'
        target_source: 'first_release' or 'revised'
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

    # ── Load current feature list from JSON (for metadata only) ──
    current_features = load_selected_features(target_type, target_source)
    logger.info(f"[{branch_label}] Current JSON feature count: {len(current_features)}")

    # ── Load training data ──
    t0 = time.time()
    X, y = _load_training_data(target_type, release_type, target_source, pit_cutoff, cache_dir)
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

    X_feat = X[feature_cols]

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
    final_features = stage_final_shap_ranking(
        X_s2, y, target_n=target_n, output_dir=output_dir, branch_label=branch_label,
    )
    final_features.sort()

    # ── Save summary ──
    summary = {
        'branch': branch_label,
        'pit_cutoff': pit_cutoff,
        'original_json_count': len(current_features),
        'stage1_input': len(feature_cols),
        'stage1_output': len(survivors_s1),
        'stage2_output': len(survivors_s2),
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
            "reduction_method": "boruta_shap_3stage",
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
        description='Post-selection feature reduction via BorutaSHAP (400-500 -> ~50 features)',
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
        help='Specific branches to process (e.g., nsa_first_release sa_revised). '
             'Default: all 4 branches.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run without overwriting JSON files',
    )
    parser.add_argument(
        '--pit-cutoff', type=str, default=DEFAULT_PIT_CUTOFF,
        help=f'Point-in-time cutoff date YYYY-MM-DD (default: {DEFAULT_PIT_CUTOFF})',
    )

    args = parser.parse_args()

    # Determine which branches to process
    all_branches = [
        ('nsa', 'first_release'),
        ('nsa', 'revised'),
        ('sa', 'first_release'),
        ('sa', 'revised'),
    ]

    if args.branches:
        branches = []
        for b in args.branches:
            parts = b.split('_', 1)
            if len(parts) == 2 and parts[0] in ('nsa', 'sa'):
                branches.append((parts[0], parts[1]))
            else:
                logger.error(f"Invalid branch format: {b}. Expected nsa_first_release, etc.")
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
