"""
6-Stage Feature Selection Engine
================================
Exact replica of the per-source Jupyter notebook pipelines.
Each stage matches the notebook implementation precisely.

Stages:
1. Group-wise Dual Filter (Purged Expanding Correlation + LightGBM)
2. Boruta Feature Selection (shadow features, 100 runs)
3. Vintage Stability (exponential recency weighting across historical snapshots)
4. Cluster Redundancy (NaN-aware Spearman hierarchical clustering)
5. Interaction Rescue (two-phase: single-feature + split-pair detection)
6. Sequential Forward Selection (walk-forward CV with embargo)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import binomtest, t as t_dist
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Lock
import re
import time
import logging
import os
import gc
import psutil

from utils.transforms import winsorize_covid_period
from Data_ETA_Pipeline.perf_stats import (
    inc_counter,
    install_hooks,
    perf_phase,
    profiled,
)

logger = logging.getLogger(__name__)
install_hooks()

SEED = 42
MIN_VALID_OBS = 60
TRIAL_EVAL_MAX_WORKERS_DEFAULT = 4
BORUTA_SHADOW_CAP_MAX = 500
DEFAULT_RECENCY_MONTHS = 3
NOAA_RECENCY_MONTHS = 6
USE_STAGE1_PURGED_SEARCHSORTED = (
    os.getenv("NFP_STAGE1_PURGED_SEARCHSORTED", "0").strip() == "1"
)
USE_STAGE1_REUSE_TARGET_INDEX = (
    os.getenv("NFP_STAGE1_REUSE_TARGET_INDEX", "0").strip() == "1"
)
USE_STAGE1_REUSE_FEATURE_INDEX = (
    os.getenv("NFP_STAGE1_REUSE_FEATURE_INDEX", "0").strip() == "1"
)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    # MUST BE 1. LightGBM + ProcessPoolExecutor + n_jobs=-1 = OOM deadlock on macOS
    'n_jobs': 1,
    'random_state': SEED,
}

# =============================================================================
# LightGBM Safe Helpers
# =============================================================================

_LGB_COL_SCHEMA_CACHE: dict[tuple[object, ...], tuple[tuple[str, ...], bool]] = {}
_LGB_COL_SCHEMA_CACHE_LOCK = Lock()
_LGB_COL_SCHEMA_CACHE_MAX = 4096


def _sanitize_lgb_col_name(name) -> str:
    """Sanitize LightGBM JSON-forbidden characters."""
    return re.sub(r'[\[\]\{\}:,]', '_', str(name))


def _get_lgb_column_schema(columns) -> tuple[tuple[str, ...], bool]:
    """
    Return (sanitized_columns, changed) for a given column sequence.

    Results are cached by the exact column tuple to avoid repeated regex work
    in CV-heavy loops.
    """
    key = tuple(columns)
    with _LGB_COL_SCHEMA_CACHE_LOCK:
        cached = _LGB_COL_SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached

    seen = set()
    safe_cols = []
    changed = False

    for col in key:
        raw = str(col)
        safe = _sanitize_lgb_col_name(raw)
        base = safe
        suffix = 2
        while safe in seen:
            safe = f"{base}_{suffix}"
            suffix += 1
        if safe != raw:
            changed = True
        seen.add(safe)
        safe_cols.append(safe)

    result = (tuple(safe_cols), changed)
    with _LGB_COL_SCHEMA_CACHE_LOCK:
        if key not in _LGB_COL_SCHEMA_CACHE:
            if len(_LGB_COL_SCHEMA_CACHE) >= _LGB_COL_SCHEMA_CACHE_MAX:
                _LGB_COL_SCHEMA_CACHE.pop(next(iter(_LGB_COL_SCHEMA_CACHE)))
            _LGB_COL_SCHEMA_CACHE[key] = result
    return result


def _prepare_lgb_frame(
    X: pd.DataFrame,
    *,
    expected_raw_columns: tuple[object, ...] | None = None,
    expected_safe_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[object, ...], tuple[str, ...]]:
    """
    Align feature columns for LightGBM without copying underlying data.
    """
    raw_cols = tuple(X.columns)

    if expected_raw_columns is not None and expected_safe_columns is not None:
        if raw_cols == expected_raw_columns:
            if tuple(expected_safe_columns) == raw_cols:
                return X, raw_cols, tuple(expected_safe_columns)
            return (
                X.set_axis(list(expected_safe_columns), axis=1, copy=False),
                raw_cols,
                tuple(expected_safe_columns),
            )

    safe_cols, changed = _get_lgb_column_schema(raw_cols)
    if not changed:
        return X, raw_cols, safe_cols
    return X.set_axis(list(safe_cols), axis=1, copy=False), raw_cols, safe_cols


def _safe_lgb_fit(model, X, y):
    """Safely fit LightGBM with cached column sanitization and no data copy."""
    if X.empty:
        return model
    X_safe, raw_cols, safe_cols = _prepare_lgb_frame(X)
    model.fit(X_safe, y)
    model._safe_lgb_raw_cols = raw_cols
    model._safe_lgb_safe_cols = safe_cols
    return model


def _safe_lgb_predict(model, X):
    """Safely predict LightGBM matching sanitized feature names from fit."""
    if X.empty:
        return np.array([])
    raw_cols = getattr(model, '_safe_lgb_raw_cols', None)
    safe_cols = getattr(model, '_safe_lgb_safe_cols', None)
    X_safe, _, _ = _prepare_lgb_frame(
        X,
        expected_raw_columns=raw_cols,
        expected_safe_columns=safe_cols,
    )
    return model.predict(X_safe)

# =============================================================================
# Source-Specific Grouping Taxonomies
# =============================================================================

UNIFIER_PREFIXES = [
    ('ISM_Manufacturing',         'ISM Surveys'),
    ('ISM_NonManufacturing',      'ISM Surveys'),
    ('CB_Consumer_Confidence',    'Consumer Sentiment'),
    ('UMich_Expectations',        'Consumer Sentiment'),
    ('AWH_All_Private',           'Hours & Wages'),
    ('AWH_Manufacturing',         'Hours & Wages'),
    ('AHE_Private',               'Hours & Wages'),
    ('Housing_Starts',            'Housing'),
    ('Retail_Sales',              'Consumer Demand'),
    ('Industrial_Production',     'Industrial Activity'),
    ('Empire_State_Mfg',          'Regional PMI'),
    ('Challenger_Job_Cuts',       'Labor Market'),
]

FRED_EXOG_PREFIXES = [
    ('VIX_panic_regime',        'VIX (Binary Regimes)'),
    ('VIX_high_regime',         'VIX (Binary Regimes)'),
    ('SP500_bear_market',       'SP500 (Binary Regimes)'),
    ('SP500_crash_month',       'SP500 (Binary Regimes)'),
    ('SP500_circuit_breaker',   'SP500 (Binary Regimes)'),
    ('Credit_Spreads',          'Credit Spreads'),
    ('Yield_Curve',             'Yield Curve'),
    ('Oil_Prices',              'Oil Prices'),
    ('Oil_worst_day',           'Oil Prices'),
    ('VIX',                     'VIX'),
    ('SP500',                   'SP500'),
    ('CCNSA',                   'CCNSA (Continued Claims NSA)'),
    ('CCSA',                    'CCSA (Continued Claims SA)'),
    ('Financial_Stress',        'Financial Stress'),
    ('Weekly_Econ_Index',       'Weekly Economic Index'),
]

NOAA_PREFIXES = [
    ('NOAA_Economic_Damage_Index', 'Economic Damage'),
    ('NOAA_Human_Impact_Index', 'Human Impact'),
]

PROSPER_MAP = {
    'Consumer Mood Index': 'Consumer Mood',
    'Prosper Consumer Spending Forecast': 'Spending Forecast',
    'Regarding the U.S. employment environment, over the next six (6) months, do you think that there will be more, the same or fewer layoffs than at present?': 'Layoff Expectations',
    'Which of the following most accurately describes your employment environment? (Check all that apply)': 'Employment Status',
}


def _classify_series(col: str, source_name: str) -> str:
    """Classify a given feature column into its mathematically logical group."""
    if source_name == 'ADP':
        return 'ADP Symlog' if '_symlog' in col else 'ADP Raw'

    elif source_name == 'NOAA':
        for prefix, grp in NOAA_PREFIXES:
            if col.startswith(prefix): return grp
        return 'Other'

    elif source_name == 'Prosper':
        for prefix, grp in PROSPER_MAP.items():
            if col.startswith(prefix): return grp
        return 'Other'

    elif source_name == 'Unifier':
        for prefix, grp in UNIFIER_PREFIXES:
            if col.startswith(prefix): return grp
        return 'Other'

    elif source_name == 'FRED_Exogenous':
        for prefix, grp in FRED_EXOG_PREFIXES:
            if col.startswith(prefix): return grp
        return 'Other'

    elif source_name.startswith('FRED_Employment'):
        parts = col.split('.')
        if col.startswith('total.private.services'):
            if len(parts) >= 4:
                return f"total.private.services.{parts[3].split('_')[0]}"
            return 'total.private.services'
        elif col.startswith('total.private.goods'):
            if len(parts) >= 4:
                return f"total.private.goods.{parts[3].split('_')[0]}"
            return 'total.private.goods'
        elif col.startswith('total.government'):
            if len(parts) >= 3:
                return f"total.government.{parts[2].split('_')[0]}"
            return 'total.government'
        elif col.startswith('total.private'):
            return 'total.private'
        return 'total'

    return 'Global'


# =============================================================================
# Snapshot Helpers
# =============================================================================

def load_snapshot_wide(path, feature_filter=None):
    """Load a snapshot parquet and return in wide format (date index, feature columns).

    Handles both long-format (series_name/value columns) and pre-pivoted wide format.
    Also handles the case where 'date' is stored as the DataFrame index (e.g., output
    from compute_features_wide).
    """
    df = pd.read_parquet(path)
    if 'series_name' in df.columns and 'value' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if feature_filter:
            df = df[df['series_name'].isin(feature_filter)]
        df = df.drop_duplicates(subset=['date', 'series_name'], keep='last')
        wide = df.pivot(index='date', columns='series_name', values='value')
    else:
        # Handle 'date' being either a column or the index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            wide = df.set_index('date')
        elif df.index.name == 'date':
            df.index = pd.to_datetime(df.index)
            wide = df
        else:
            # Fallback: assume the index IS the date
            df.index = pd.to_datetime(df.index)
            wide = df
        wide = wide.drop(columns=['snapshot_date'], errors='ignore')
        wide = wide.select_dtypes(include=[np.number])
        if feature_filter:
            available = [c for c in feature_filter if c in wide.columns]
            wide = wide[available]
            
    # Cutoff analysis to post-1990 for cleaner, more plentiful data points
    # (Removes massive NaN sparsity from earlier decades)
    wide = wide[wide.index >= '1990-01-01']
            
    # OOM/Freeze Protection: pandas `dropna(how='all')` freezes on 50k cols. 
    # Use vectorized boolean indexing instead.
    valid_cols = wide.columns[wide.notna().any(axis=0)]
    wide = wide[valid_cols]
    valid_rows = wide.index[wide.notna().any(axis=1)]
    wide = wide.loc[valid_rows].sort_index()
    
    return wide


def _select_wide(snap_wide, series_list):
    """Select a subset of columns from a wide DataFrame, dropping all-NaN rows/cols using fast vectorized masks."""
    available = [s for s in series_list if s in snap_wide.columns]
    if not available:
        return pd.DataFrame()
    subset = snap_wide[available].copy()
    
    # Fast vectorized dropna equivalent
    valid_cols = subset.columns[subset.notna().any(axis=0)]
    subset = subset[valid_cols]
    valid_rows = subset.index[subset.notna().any(axis=1)]
    
    return subset.loc[valid_rows]


# =============================================================================
# LightGBM Helpers (used by global pipeline / Boruta-SHAP variant)
# =============================================================================

def _sanitize_col(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))


def _sanitize_df(df):
    clean_to_orig = {}
    new_cols = []
    for c in df.columns:
        clean = _sanitize_col(c)
        base = clean
        i = 2
        while clean in clean_to_orig:
            clean = f"{base}_{i}"
            i += 1
        clean_to_orig[clean] = c
        new_cols.append(clean)
    out = df.copy()
    out.columns = new_cols
    return out, clean_to_orig


def _lgb_fit(X, y, params=None):
    if params is None:
        params = LGB_PARAMS
    X_clean, mapping = _sanitize_df(X)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_clean, y)
    return model, mapping


def _lgb_predict(model, X, mapping):
    orig_to_clean = {v: k for k, v in mapping.items()}
    X_clean = X.rename(columns=orig_to_clean)
    return _safe_lgb_predict(model, X_clean)


def _lgb_importances(model, mapping):
    clean_names = model.feature_name_
    imp_vals = model.feature_importances_
    return pd.Series({mapping.get(c, c): v for c, v in zip(clean_names, imp_vals)})


# =============================================================================
# Stage 1: Group-wise Dual Filter (Purged Correlation + LightGBM + VIF)
# =============================================================================

def _purged_expanding_corr(feature_series, target_series,
                           min_window=60, purge_months=3,
                           target_non_na_index=None,
                           feature_non_na_index=None):
    """
    Compute weighted-average correlation using expanding windows with a purge gap.
    Handles staggered start dates: only uses dates where the feature has data.
    """
    with perf_phase("fs.stage1.purged_corr.total"):
        with perf_phase("fs.stage1.purged_corr.target_index_prep"):
            # Stage-1 target-index reuse chunk (guarded at call-site): when a
            # precomputed non-NaN target index is provided, reuse it across all
            # feature columns in this group. Intersection and all downstream
            # correlation math remain unchanged.
            if target_non_na_index is None:
                target_non_na_index = target_series.dropna().index
            if feature_non_na_index is None:
                with perf_phase("fs.stage1.purged_corr.feature_index_prep"):
                    feature_non_na_index = feature_series.dropna().index
            common = feature_non_na_index.intersection(target_non_na_index)
            common = common.sort_values()

        if len(common) < min_window:
            return np.nan, 0

        eval_points = np.linspace(
            min_window, len(common) - 1,
            min(5, len(common) - min_window), dtype=int
        )
        eval_points = sorted(set(eval_points))

        if not eval_points:
            return np.nan, 0

        correlations = []
        window_sizes = []
        use_searchsorted_windows = USE_STAGE1_PURGED_SEARCHSORTED
        for end_idx in eval_points:
            eval_date = common[end_idx]
            purge_cutoff = eval_date - pd.DateOffset(months=purge_months)
            with perf_phase("fs.stage1.purged_corr.window_index_prep"):
                if use_searchsorted_windows:
                    # Stage-1 purged-corr chunk (guarded): build the train window via
                    # index boundary lookup instead of per-iteration boolean masking.
                    # This preserves the exact "< purge_cutoff" date semantics.
                    cutoff_pos = common.searchsorted(purge_cutoff, side='left')
                    train_dates = common[:cutoff_pos]
                else:
                    train_dates = common[common < purge_cutoff]

            if len(train_dates) < min_window // 2:
                continue
            x_train = feature_series.loc[train_dates]
            y_train = target_series.loc[train_dates]

            if len(x_train) >= 30 and x_train.std() > 0:
                r = x_train.corr(y_train)
                if not np.isnan(r):
                    correlations.append(r)
                    window_sizes.append(len(x_train))

        if not correlations:
            return np.nan, 0

        weights = np.sqrt(window_sizes)
        weighted_corr = np.average(correlations, weights=weights)
        effective_n = int(np.mean(window_sizes))

        return weighted_corr, effective_n


def _variance_filter(wide_df, max_same_frac=0.97, min_obs=30):
    """
    Drop near-constant features where >= max_same_frac of non-NaN values
    are identical. Uses a two-tier approach for speed on 50k+ features:

    Tier 1 (fast): pandas nunique() to classify columns:
      - nunique <= 1  → definitely constant, drop
      - nunique > 5   → can't be 97% one value with 6+ uniques, keep
      - nunique 2-5   → ambiguous, need exact check
    Tier 2 (exact): np.unique per-column loop only on the ambiguous set.
    """
    n_original = wide_df.shape[1]

    # 1. Count valid observations per column
    valid_counts = wide_df.notna().sum()
    valid_cols = valid_counts[valid_counts >= min_obs].index
    if len(valid_cols) == 0:
        return wide_df[[]]

    df_valid = wide_df[valid_cols]

    # 2. Fast pre-pass using nunique() (vectorized C-level operation)
    nuniq = df_valid.nunique()

    # Columns with <= 1 unique value → constant, always dropped
    definitely_drop = nuniq[nuniq <= 1].index
    # Columns with > 5 unique values → extremely unlikely to have 97% in one bin
    definitely_keep = nuniq[nuniq > 5].index
    # Ambiguous: 2-5 unique values → need exact per-column mode fraction check
    ambiguous_cols = nuniq[(nuniq >= 2) & (nuniq <= 5)].index

    logger.info(f"    Variance pre-pass: {len(definitely_drop)} constant, "
                f"{len(definitely_keep)} variable, {len(ambiguous_cols)} ambiguous")

    # 3. Exact check only on the ambiguous set
    keep = list(definitely_keep)
    for col in ambiguous_cols:
        arr = df_valid[col].values
        arr = arr[~np.isnan(arr)]
        if len(arr) < min_obs:
            continue
        _, counts = np.unique(arr, return_counts=True)
        mode_frac = counts.max() / len(arr)
        if mode_frac < max_same_frac:
            keep.append(col)

    n_dropped = n_original - len(keep)
    if n_dropped > 0:
        logger.info(f"    Variance filter: {n_original} → {len(keep)} "
                    f"(dropped {n_dropped} near-constant features)")
    return wide_df[keep]


def _deduplicate_group(wide_df, threshold=0.95):
    """
    Agglomerative clustering to collapse near-identical features.
    Keeps the feature with the most non-NaN observations per cluster.
    Auto-chunks massive groups to guarantee OOM-safety.
    """
    n_features = wide_df.shape[1]
    if n_features <= 3:
        return wide_df

    def _cluster_chunk(df_chunk, t):
        if df_chunk.shape[1] <= 1:
            return list(df_chunk.columns)
        corr = df_chunk.corr(method='spearman').abs().fillna(0)
        dist = 1 - corr.values
        np.fill_diagonal(dist, 0)
        dist = np.maximum(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        Z = hierarchy.linkage(condensed, method='average')
        clusters = hierarchy.fcluster(Z, t=1 - t, criterion='distance')
        
        keep = []
        cols = list(df_chunk.columns)
        for cid in np.unique(clusters):
            members = [cols[i] for i, c in enumerate(clusters) if c == cid]
            best = max(members, key=lambda m: df_chunk[m].notna().sum())
            keep.append(best)
        return keep

    try:
        # If massive (>5000), chunk the deduplication to save RAM
        if n_features > 5000:
            logger.info(f"    Massive group ({n_features} cols). Using chunked deduplication...")
            cols = list(wide_df.columns)
            chunk_size = 2500
            survivors = []

            # Phase 1: Dedup chunks individually
            for i in range(0, len(cols), chunk_size):
                chunk = cols[i:i+chunk_size]
                chunk_survivors = _cluster_chunk(wide_df[chunk], threshold)
                survivors.extend(chunk_survivors)

            logger.info(f"    Chunk dedup Phase 1: {n_features} → {len(survivors)} features.")

            # Phase 2: Iterative merge — keep deduplicating until convergence
            # Shuffle between iterations to expose cross-chunk correlations
            max_iterations = 5
            for iteration in range(max_iterations):
                n_before = len(survivors)
                if n_before <= chunk_size:
                    survivors = _cluster_chunk(wide_df[survivors], threshold)
                    break

                # Shuffle to break chunk boundary artifacts
                rng = np.random.RandomState(SEED + iteration)
                shuffled = list(survivors)
                rng.shuffle(shuffled)

                new_survivors = []
                for i in range(0, len(shuffled), chunk_size):
                    chunk = shuffled[i:i + chunk_size]
                    chunk_survivors = _cluster_chunk(wide_df[chunk], threshold)
                    new_survivors.extend(chunk_survivors)

                survivors = new_survivors
                n_after = len(survivors)
                logger.info(f"    Chunk dedup iteration {iteration + 2}: "
                            f"{n_before} → {n_after} features")

                if n_after == n_before:
                    break  # Converged

            final_survivors = survivors
            n_dropped = n_features - len(final_survivors)
            logger.info(f"    Chunk dedup complete: {n_features} → {len(final_survivors)} "
                        f"(dropped {n_dropped})")
            return wide_df[final_survivors]
            
        else:
            # Normal dedup for <= 5000 features
            keep = _cluster_chunk(wide_df, threshold)
            n_dropped = n_features - len(keep)
            if n_dropped > 0:
                logger.info(f"    Dedup (|r|>{threshold}): {n_features} → {len(keep)} "
                            f"(collapsed {n_dropped} near-identical features)")
            return wide_df[keep]
            
    except MemoryError:
        logger.warning(f"    Dedup OOM on {n_features} cols. Skipping completely to save pipeline.")
        return wide_df


def _is_noaa_feature_name(col_name: str) -> bool:
    """Heuristic NOAA feature detector for raw/sanitized column names."""
    name = str(col_name).upper()
    return name.startswith('NOAA') or '_NOAA_' in name


def _source_aware_recency_mask(
    X: pd.DataFrame,
    default_months: int = DEFAULT_RECENCY_MONTHS,
    noaa_months: int = NOAA_RECENCY_MONTHS,
) -> pd.Series:
    """
    Build a recency mask with source-aware windows.

    Non-NOAA features must have data in the last `default_months`.
    NOAA features get a wider window (`noaa_months`) due publication lag.
    """
    if X.empty:
        return pd.Series(False, index=X.columns, dtype=bool)

    last_date = X.index.max()
    default_cutoff = last_date - pd.DateOffset(months=default_months)
    default_rows = X.loc[X.index >= default_cutoff]
    if default_rows.empty:
        recency_mask = pd.Series(False, index=X.columns, dtype=bool)
    else:
        recency_mask = default_rows.notna().any(axis=0)

    if noaa_months <= default_months:
        return recency_mask

    noaa_cols = pd.Index([col for col in X.columns if _is_noaa_feature_name(col)])
    if noaa_cols.empty:
        return recency_mask

    noaa_cutoff = last_date - pd.DateOffset(months=noaa_months)
    noaa_rows = X.loc[X.index >= noaa_cutoff]
    if noaa_rows.empty:
        recency_mask.loc[noaa_cols] = False
        return recency_mask

    noaa_mask = noaa_rows.notna().any(axis=0)
    recency_mask.loc[noaa_cols] = noaa_mask.reindex(noaa_cols, fill_value=False)
    return recency_mask


def _lgb_screen_group(wide_df, target_series, top_k=15,
                      n_subspaces=10, subspace_frac=0.3):
    """
    Random Subspace LightGBM screening: trains multiple models on random
    column subsets to catch features that a single model might miss.
    Falls back to single model for small groups (<= 200 features).
    """
    common = wide_df.index.intersection(target_series.index)
    if len(common) < 50:
        return []

    X = wide_df.loc[common]
    y = target_series.loc[common]

    # Recency check (source-aware: NOAA gets a wider window due release lag)
    recency_mask = _source_aware_recency_mask(X)
    X = X.loc[:, recency_mask]

    if X.shape[1] == 0:
        return []

    # For small groups, fall back to single model (fast enough)
    if X.shape[1] <= 200:
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        _safe_lgb_fit(model, X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        nonzero = imp[imp > 0].sort_values(ascending=False)
        return nonzero.head(top_k).index.tolist()

    # Random Subspace ensemble for large groups
    all_cols = list(X.columns)
    n_select = max(200, int(len(all_cols) * subspace_frac))
    rng = np.random.RandomState(SEED)

    aggregated_imp = pd.Series(0.0, index=X.columns)

    for i in range(n_subspaces):
        subset_cols = list(rng.choice(all_cols,
                                     size=min(n_select, len(all_cols)),
                                     replace=False))
        X_sub = X[subset_cols]
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        logger.debug(f"      [LGB Subspace {i+1}/{n_subspaces}] fitting {X_sub.shape[1]} features...")
        _safe_lgb_fit(model, X_sub, y)
        imp = pd.Series(model.feature_importances_, index=subset_cols)
        aggregated_imp[imp.index] += imp.values

    nonzero = aggregated_imp[aggregated_imp > 0].sort_values(ascending=False)
    # Wider net (top_k * 2): downstream VIF and Boruta will prune excess
    logger.debug(f"      [LGB Subspace Done] generated {len(nonzero)} nonzero features")
    return nonzero.head(top_k * 2).index.tolist()


def _is_binary_feature(series, threshold=5):
    """Check if a feature is effectively binary (<=threshold unique non-NaN values)."""
    return series.dropna().nunique() <= threshold


def _compute_purged_corr_for_col(args):
    """Worker for threaded purged correlation computation."""
    with perf_phase("fs.stage1.purged_corr.compute_for_col"):
        if len(args) == 5:
            col_name, col_values, y_local, min_window, target_non_na_index = args
            feature_non_na_index = None
        elif len(args) == 6:
            (
                col_name,
                col_values,
                y_local,
                min_window,
                target_non_na_index,
                feature_non_na_index,
            ) = args
        else:
            col_name, col_values, y_local, min_window = args
            target_non_na_index = None
            feature_non_na_index = None
        avg_corr, eff_n = _purged_expanding_corr(
            col_values,
            y_local,
            min_window=min_window,
            target_non_na_index=target_non_na_index,
            feature_non_na_index=feature_non_na_index,
        )
        if np.isnan(avg_corr) or eff_n < min_window:
            return None
        denom = max(1 - avg_corr ** 2, 1e-8)
        t_stat = avg_corr * np.sqrt((eff_n - 2) / denom)
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=eff_n - 2))
        return (col_name, p_value)


def _bh_fdr_select(feature_pvalues: dict[str, float], fdr_alpha: float) -> list[str]:
    """Benjamini-Hochberg FDR selection using the largest passing rank."""
    if not feature_pvalues:
        return []

    sorted_features = sorted(feature_pvalues.items(), key=lambda x: x[1])
    n_tests = len(sorted_features)
    max_pass_rank = 0

    for rank, (_, pval) in enumerate(sorted_features, 1):
        bh_threshold = (rank / n_tests) * fdr_alpha
        if pval <= bh_threshold:
            max_pass_rank = rank

    if max_pass_rank == 0:
        return []

    return [feat for feat, _ in sorted_features[:max_pass_rank]]


def _vectorized_target_prescreen(
    snap_wide: pd.DataFrame,
    y_target: pd.Series,
    fdr_alpha: float = 0.30,
    min_corr_obs: int = 30,
) -> list[str]:
    """
    Fast O(n×p) vectorized Spearman pre-screen against the prediction target.

    Designed as a coarse pre-filter before Stage 1's expensive per-feature
    purged expanding correlation. Rank-transforms the full feature matrix once
    (vectorized), then calls corrwith() — completing in ~5-15s even at 17k
    features vs. the ~10+ min Stage 1 would need on the same input.

    Uses BH-FDR at a deliberately permissive alpha to minimize false negatives;
    Stage 1 (purged expanding corr + LGB subspace) remains the rigorous gate.

    Only triggered when n_cols > 5000 (guarded in run_pipeline Stage 0).
    Small sources (NOAA, Prosper, ADP, Unifier) are never affected.

    Args:
        snap_wide: Wide feature DataFrame (DatetimeIndex × features),
            post-variance-filter.
        y_target: Target Series (DatetimeIndex), already winsorized.
        fdr_alpha: BH-FDR threshold. Default 0.30 is intentionally permissive
            to minimize false negatives in this coarse pre-screen stage.
        min_corr_obs: Minimum non-NaN overlap with target required for a
            feature to be eligible for correlation scoring. Features below
            this threshold receive p-value = 1.0 and are excluded.

    Returns:
        List of surviving feature names (order not guaranteed).
    """
    with perf_phase("fs.stage0.prescreen"):
        inc_counter("fs.stage0.prescreen.calls", 1)

        # Align on dates where y_target is valid (no NaN in y after this)
        y_valid_idx = y_target.dropna().index
        common = snap_wide.index.intersection(y_valid_idx)

        if len(common) < min_corr_obs:
            logger.warning(
                f"   Pre-screen: only {len(common)} common dates with target "
                f"(< {min_corr_obs} floor). Returning all features."
            )
            return list(snap_wide.columns)

        X = snap_wide.loc[common]
        y = y_target.loc[common]

        # Per-column non-NaN count — denominator for per-feature t-stats.
        # y has no NaN here (restricted to y_valid_idx), so this is the
        # correct effective-n for each pairwise correlation.
        n_obs = X.notna().sum()

        # Rank-transform for Spearman via rank-Pearson.
        # na_option='keep' preserves NaN positions; corrwith handles them
        # pairwise (excludes NaN rows per column, matching n_obs above).
        with perf_phase("fs.stage0.prescreen.rank"):
            X_ranked = X.rank(pct=True, na_option='keep')
            y_ranked = y.rank(pct=True)

        # Vectorized pairwise Spearman: one corrwith call for all p columns.
        with perf_phase("fs.stage0.prescreen.corrwith"):
            corrs = X_ranked.corrwith(y_ranked)

        # Vectorized t-stat → two-tailed p-value.
        # corrs.fillna(0) handles constant columns (undefined correlation).
        r = corrs.fillna(0.0).values
        n = n_obs.reindex(corrs.index).fillna(0).values.astype(float)

        # Flag features whose overlap with target is insufficient
        insufficient = n < min_corr_obs

        denom = np.maximum(1.0 - r ** 2, 1e-8)
        df = np.maximum(n - 2.0, 1.0)
        t_stat = np.nan_to_num(r * np.sqrt(df) / np.sqrt(denom), nan=0.0)
        # sf (survival function) is numerically superior to 1-cdf in tails
        p_vals = 2.0 * t_dist.sf(np.abs(t_stat), df=df)
        p_vals[insufficient] = 1.0

        # BH-FDR selection using existing pipeline helper
        feature_pvalues = dict(zip(corrs.index.tolist(), p_vals.tolist()))
        survivors = _bh_fdr_select(feature_pvalues, fdr_alpha=fdr_alpha)

        n_total = len(corrs)
        abs_r = np.abs(r)
        logger.info(
            f"   Pre-screen stats: |ρ| mean={abs_r.mean():.3f} "
            f"median={np.median(abs_r):.3f} max={abs_r.max():.3f} | "
            f"dropped {n_total - len(survivors)}/{n_total}"
        )
        inc_counter("fs.stage0.prescreen.features_dropped", n_total - len(survivors))
        inc_counter("fs.stage0.prescreen.features_surviving", len(survivors))

        return survivors


def _feature_set_cache_key(feature_set) -> tuple[str, ...]:
    """Deterministic, order-insensitive cache key for a feature set."""
    return tuple(sorted(dict.fromkeys(feature_set)))


def _memoized_score(
    feature_set,
    scorer,
    cache: dict[tuple[str, ...], float],
    cache_lock=None,
) -> float:
    """
    Evaluate scorer(feature_set) with memoization.

    Feature order is normalized in the key so semantically identical sets reuse
    the same cached MAE evaluation.
    """
    key = _feature_set_cache_key(feature_set)

    if cache_lock is not None:
        with cache_lock:
            if key in cache:
                return cache[key]
    elif key in cache:
        return cache[key]

    value = float(scorer(list(key)))

    if cache_lock is not None:
        with cache_lock:
            if key not in cache:
                cache[key] = value
            return cache[key]

    cache.setdefault(key, value)
    return cache[key]


def _parallel_trial_scores(
    trial_defs: list[tuple[object, list[str]]],
    scorer,
    max_workers: int,
    executor: ThreadPoolExecutor | None = None,
) -> list[tuple[object, float]]:
    """
    Evaluate (label, feature_set) trials, parallelizing when beneficial.
    """
    if not trial_defs:
        return []
    max_workers = max(1, int(max_workers))
    if len(trial_defs) <= 1 or max_workers == 1:
        return [(label, float(scorer(feature_set))) for label, feature_set in trial_defs]

    def _score_item(item):
        label, feature_set = item
        return label, float(scorer(feature_set))

    if executor is not None:
        return list(executor.map(_score_item, trial_defs))

    workers = min(max_workers, len(trial_defs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_score_item, trial_defs))


def filter_group_data_purged(wide_df, target_series, group_name,
                             fdr_alpha=0.10):
    """
    Group-wise filter with DUAL screening paths:

    Path A (Linear): Purged expanding-window correlation + BH-FDR significance
    Path B (Non-linear): LightGBM gain importance pre-screen

    Returns the UNION of both paths.
    Binary regime groups are handled with relaxed min_window.
    """
    common = wide_df.index.intersection(target_series.index)
    use_copyless_xlocal = os.getenv("NFP_STAGE1_COPYLESS_XLOCAL", "0").strip() == "1"
    if use_copyless_xlocal:
        inc_counter("fs.stage1.copyless_xlocal_enabled", 1)
        # Stage-1 chunk optimization: avoid eager DataFrame copy during the
        # initial common-index alignment. Downstream logic rebinds X_local and
        # does not mutate wide_df in place, so this should preserve parity.
        X_local = wide_df.loc[common]
    else:
        inc_counter("fs.stage1.copyless_xlocal_disabled", 1)
        X_local = wide_df.loc[common].copy()
    y_local = target_series.loc[common]

    if X_local.empty:
        return []

    # 1. Recency Check (source-aware: NOAA gets a wider window due release lag)
    recency_mask = _source_aware_recency_mask(X_local)
    X_local = X_local.loc[:, recency_mask]

    if X_local.shape[1] == 0:
        return []

    # Detect if this is a binary-regime group
    binary_cols = [c for c in X_local.columns if _is_binary_feature(X_local[c])]
    is_binary_group = len(binary_cols) > len(X_local.columns) * 0.5

    # Relaxed min_window for binary groups (fewer events = fewer valid obs)
    min_window = 30 if is_binary_group else 60

    # === FAST PRE-FILTER (Path A only): vectorized correlation to eliminate ===
    # === obviously uncorrelated features before expensive purged-corr      ===
    # Keep Path B on the full recency-filtered matrix so non-linear signals
    # are not dropped before LightGBM can evaluate them.
    X_path_a = X_local
    n_before_prefilter = X_path_a.shape[1]
    if n_before_prefilter > 200:  # Only worth it for large groups
        fast_corrs = X_path_a.corrwith(y_local).abs()
        # Tiered threshold: tighter for very large groups (safe because
        # Path B runs on full X_local and catches non-linear features.
        corr_threshold = 0.05 if n_before_prefilter > 500 else 0.02
        keep_mask = (fast_corrs > corr_threshold) | fast_corrs.isna()
        X_path_a = X_path_a.loc[:, keep_mask]
        n_eliminated = n_before_prefilter - X_path_a.shape[1]
        if n_eliminated > 0:
            logger.info(f"    [{group_name}] Pre-filter (|r|>{corr_threshold}): "
                        f"{n_before_prefilter} → {X_path_a.shape[1]} features "
                        f"(eliminated {n_eliminated})")

    # === Path A: Purged Expanding-Window Correlation + BH-FDR ===
    # Use ThreadPoolExecutor for parallel correlation computation
    n_cols = X_path_a.shape[1]
    n_threads = min(os.cpu_count() or 4, max(1, n_cols // 10))
    feature_pvalues = {}
    use_reuse_target_index = USE_STAGE1_REUSE_TARGET_INDEX
    if use_reuse_target_index:
        inc_counter("fs.stage1.reuse_target_index.enabled", 1)
        target_non_na_index = y_local.dropna().index
    else:
        inc_counter("fs.stage1.reuse_target_index.disabled", 1)
        target_non_na_index = None
    use_reuse_feature_index = USE_STAGE1_REUSE_FEATURE_INDEX
    if use_reuse_feature_index:
        inc_counter("fs.stage1.reuse_feature_index.enabled", 1)
        # Stage-1 feature-index reuse chunk (guarded): precompute per-feature
        # non-NaN indices once for this group and reuse inside purged-corr.
        # This preserves exact row alignment because each index is derived from
        # the same Series used by the legacy in-function dropna() path.
        feature_non_na_indices = {
            col: X_path_a[col].dropna().index for col in X_path_a.columns
        }
    else:
        inc_counter("fs.stage1.reuse_feature_index.disabled", 1)
        feature_non_na_indices = None
    if n_cols > 0:
        if n_cols > 50 and n_threads > 1:
            # Parallel path
            if use_reuse_feature_index:
                args_list = [
                    (
                        col,
                        X_path_a[col],
                        y_local,
                        min_window,
                        target_non_na_index if use_reuse_target_index else None,
                        feature_non_na_indices[col],
                    )
                    for col in X_path_a.columns
                ]
            elif use_reuse_target_index:
                args_list = [
                    (col, X_path_a[col], y_local, min_window, target_non_na_index)
                    for col in X_path_a.columns
                ]
            else:
                args_list = [
                    (col, X_path_a[col], y_local, min_window)
                    for col in X_path_a.columns
                ]
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for result in pool.map(_compute_purged_corr_for_col, args_list):
                    if result is not None:
                        feature_pvalues[result[0]] = result[1]
        else:
            # Sequential path for small groups
            for col in X_path_a.columns:
                avg_corr, eff_n = _purged_expanding_corr(
                    X_path_a[col],
                    y_local,
                    min_window=min_window,
                    target_non_na_index=target_non_na_index if use_reuse_target_index else None,
                    feature_non_na_index=(
                        feature_non_na_indices[col] if use_reuse_feature_index else None
                    ),
                )
                if np.isnan(avg_corr) or eff_n < min_window:
                    continue
                denom = max(1 - avg_corr ** 2, 1e-8)
                t_stat = avg_corr * np.sqrt((eff_n - 2) / denom)
                p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=eff_n - 2))
                feature_pvalues[col] = p_value

    corr_features = _bh_fdr_select(feature_pvalues, fdr_alpha=fdr_alpha)

    # === Path B: LightGBM Gain Importance Pre-Screen ===
    # Use full recency-filtered matrix (no correlation prefilter).
    lgb_features = _lgb_screen_group(X_local, y_local, top_k=15)

    # === Union of both paths ===
    passed_features = sorted(set(corr_features) | set(lgb_features))

    if not passed_features:
        return []

    if is_binary_group:
        logger.info(f"    [{group_name}] Binary group — "
                     f"keeping {len(passed_features)} features")
        return passed_features

    logger.info(f"    [{group_name}] Dual filter passed {len(passed_features)} features")
    return passed_features


# =============================================================================
# Stage 2: Boruta Feature Selection (Tournament Architecture)
# =============================================================================

def _adaptive_boruta_shadow_cap(n_features: int) -> int:
    """
    Adaptive cap for shadow features.

    Keeps full shadows for small/medium sets and progressively caps very large
    sets to control memory/runtime without collapsing the null distribution.
    """
    if n_features <= 300:
        return n_features
    if n_features <= 1200:
        return min(n_features, 450)
    if n_features <= 3000:
        return min(n_features, 350)
    return min(n_features, 250)


def _adaptive_boruta_runs(
    n_features: int,
    n_rows: int,
) -> dict[str, int]:
    """
    Choose Boruta run counts based on feature-set size and sample size.
    """
    if n_features <= 300:
        standard_runs = 100
        heat_runs = 35
        final_runs = 80
    elif n_features <= 1000:
        standard_runs = 85
        heat_runs = 30
        final_runs = 70
    elif n_features <= 3000:
        standard_runs = 70
        heat_runs = 24
        final_runs = 60
    else:
        standard_runs = 60
        heat_runs = 20
        final_runs = 50

    # Slightly more runs when data is plentiful, slightly fewer when scarce.
    if n_rows >= 320:
        standard_runs += 10
        heat_runs += 4
        final_runs += 10
    elif n_rows < 120:
        standard_runs = max(50, standard_runs - 10)
        heat_runs = max(16, heat_runs - 4)
        final_runs = max(40, final_runs - 10)

    return {
        'standard_runs': int(standard_runs),
        'heat_runs': int(heat_runs),
        'final_runs': int(final_runs),
    }


def _boruta_core(X_curr, y_curr, n_runs=100, alpha=0.05, max_shadows=None):
    """
    Core Boruta logic: shadow feature comparison with early stopping.

    Compares each real feature to the highest percentiles of shadow importances.
    Features proportional capping of shadow features to prevent OOMs.

    Early stopping: after 50% of runs, features that are clearly confirmed
    (p < alpha/10) or clearly rejected (0 hits) stop being tracked,
    and if all features are resolved the loop exits entirely.

    Shadow array is pre-allocated and reused across runs to avoid
    repeated DataFrame construction overhead.
    """
    n_features = X_curr.shape[1]
    n_rows = len(X_curr)
    feature_names = X_curr.columns.tolist()

    with perf_phase(
        "fs.boruta.core",
        n_features=n_features,
        n_rows=n_rows,
        n_runs=n_runs,
    ):
        inc_counter("fs.boruta.core.calls", 1)
        inc_counter("fs.boruta.core.features_total", n_features)

        hits = np.zeros(n_features, dtype=int)

        # Adaptive shadow cap limits memory and runtime for very large pools.
        if max_shadows is None:
            max_shadows = _adaptive_boruta_shadow_cap(n_features)
        max_shadows = max(50, min(int(max_shadows), BORUTA_SHADOW_CAP_MAX))
        n_shadows = min(n_features, max_shadows)
        shadow_ratio = n_shadows / max(n_features, 1)
        if shadow_ratio >= 0.75:
            percentile_thresh = 75
        elif shadow_ratio >= 0.5:
            percentile_thresh = 78
        else:
            percentile_thresh = 80

        # Pre-allocate shadow array and build combined DataFrame structure once
        X_values = X_curr.values
        shadow_arr = np.empty((n_rows, n_shadows), dtype=np.float64)
        shadow_col_names = [f'_shadow_{i}' for i in range(n_shadows)]
        combined_cols = feature_names + shadow_col_names

        # Pre-compute valid masks for all features (reused each run)
        valid_masks = []
        for j in range(n_features):
            col_vals = X_values[:, j]
            if np.issubdtype(col_vals.dtype, np.floating):
                valid_masks.append(~np.isnan(col_vals))
            else:
                valid_masks.append(np.ones(n_rows, dtype=bool))

        # Early stopping state
        confirmed_early = set()
        rejected_early = set()
        half_runs = n_runs // 2
        actual_runs = n_runs

        for run in range(n_runs):
            inc_counter("fs.boruta.core.iterations_total", 1)

            with perf_phase("fs.boruta.core.iteration"):
                rng = np.random.RandomState(SEED + run)

                # Determine active features (not yet early-stopped)
                active_indices = [i for i in range(n_features)
                                  if i not in confirmed_early and i not in rejected_early]

                # If all features resolved early, stop
                if not active_indices:
                    actual_runs = run
                    inc_counter("fs.boruta.core.early_stop_all_resolved", 1)
                    logger.info(f"   Boruta early stop: all features resolved after {run} runs")
                    break

                # Select shadow source columns
                if n_features > max_shadows:
                    source_indices = rng.choice(n_features, size=n_shadows, replace=False)
                else:
                    source_indices = np.arange(n_features)

                with perf_phase("fs.boruta.core.shadow_build"):
                    # Shuffle valid values into pre-allocated shadow array
                    for j, src_idx in enumerate(source_indices):
                        col_vals = X_values[:, src_idx].copy()
                        mask = valid_masks[src_idx]
                        valid_vals = col_vals[mask].copy()
                        rng.shuffle(valid_vals)
                        col_vals[mask] = valid_vals
                        shadow_arr[:, j] = col_vals

                with perf_phase("fs.boruta.core.model_fit"):
                    # Build combined array (avoids pd.concat overhead)
                    combined_arr = np.hstack([X_values, shadow_arr])
                    X_combined = pd.DataFrame(combined_arr, index=X_curr.index,
                                              columns=combined_cols)

                    model = lgb.LGBMRegressor(**LGB_PARAMS)
                    _safe_lgb_fit(model, X_combined, y_curr)

                    importances = pd.Series(
                        model.feature_importances_, index=X_combined.columns
                    )

                with perf_phase("fs.boruta.core.hit_update"):
                    shadow_threshold = np.percentile(
                        importances[shadow_col_names].values, percentile_thresh
                    )

                    # Only count hits for active features
                    for i in active_indices:
                        feat = feature_names[i]
                        if importances[feat] > shadow_threshold:
                            hits[i] += 1

                # Early stopping check after 50% of runs
                if run == half_runs and half_runs > 0:
                    runs_so_far = run + 1
                    for i in range(n_features):
                        if i in confirmed_early or i in rejected_early:
                            continue
                        # Confirmed early: very significant binomial test
                        p_confirm = binomtest(
                            hits[i], n=runs_so_far, p=0.25, alternative='greater'
                        ).pvalue
                        if p_confirm < alpha / 10:
                            confirmed_early.add(i)
                        # Rejected early: zero hits after 50% of runs
                        elif hits[i] == 0:
                            rejected_early.add(i)

                    if confirmed_early or rejected_early:
                        inc_counter("fs.boruta.core.early_stop_partial", 1)
                        logger.info(f"   Boruta early stop at run {runs_so_far}: "
                                    f"{len(confirmed_early)} confirmed, "
                                    f"{len(rejected_early)} rejected early")

        # Final classification
        confirmed = []
        tentative = []

        for i, feat in enumerate(feature_names):
            if i in confirmed_early:
                confirmed.append(feat)
                continue
            if i in rejected_early:
                continue

            # Standard binomial test for features that ran the full course
            p_val = binomtest(
                hits[i], n=actual_runs, p=0.25, alternative='greater'
            ).pvalue
            if p_val < alpha:
                confirmed.append(feat)
            elif p_val < alpha * 5:
                tentative.append(feat)

        inc_counter("fs.boruta.core.confirmed_total", len(confirmed))
        inc_counter("fs.boruta.core.tentative_total", len(tentative))

        logger.info(f"   Boruta: {len(confirmed)} confirmed, {len(tentative)} tentative "
                    f"(hit rates: min={hits.min()}/{actual_runs}, "
                    f"max={hits.max()}/{actual_runs})")

        return confirmed + tentative


def get_boruta_importance(X, y, n_runs=None, block_size=6, alpha=0.05,
                          tournament_chunk=100, tournament_threshold=150):
    """
    Tournament Boruta: two-round elimination for large feature sets.

    Round 1 (Heats): Split features into chunks. Run fast Boruta per chunk.
    Round 2 (Final): Pool survivors. Run rigorous Boruta on combined pool.

    For small feature sets (<= tournament_threshold), runs standard Boruta.
    """
    common = X.index.intersection(y.index)
    X_curr = X.loc[common]
    y_curr = y.loc[common]
    n_features = X_curr.shape[1]

    with perf_phase("fs.boruta.get_importance", n_features=n_features):
        inc_counter("fs.boruta.get_importance.calls", 1)

        run_plan = _adaptive_boruta_runs(
            n_features=n_features,
            n_rows=len(common),
        )
        shadow_cap = _adaptive_boruta_shadow_cap(n_features)

        if n_features <= tournament_threshold:
            # Small enough for standard single-round Boruta
            if n_runs is None:
                n_runs = run_plan['standard_runs']
            logger.info(f"   Boruta (standard): {n_features} features × {n_runs} runs")
            with perf_phase("fs.boruta.standard", n_features=n_features, n_runs=n_runs):
                return _boruta_core(
                    X_curr,
                    y_curr,
                    n_runs=n_runs,
                    alpha=alpha,
                    max_shadows=shadow_cap,
                )

        # ==========================================
        # TOURNAMENT MODE for large feature sets
        # ==========================================
        logger.info(f"   Boruta Tournament: {n_features} features → "
                    f"chunked into groups of {tournament_chunk}")

        cols = list(X_curr.columns)
        rng = np.random.RandomState(SEED)
        rng.shuffle(cols)

        chunks = [cols[i:i + tournament_chunk]
                  for i in range(0, len(cols), tournament_chunk)]

        # Round 1: Fast heats, relaxed alpha
        heat_survivors = []
        heat_alpha = alpha * 2  # More permissive in heats to avoid dropping good features
        heat_runs = run_plan['heat_runs']

        for ci, chunk in enumerate(chunks, 1):
            chunk_shadow_cap = _adaptive_boruta_shadow_cap(len(chunk))
            logger.info(f"   Heat {ci}/{len(chunks)}: "
                        f"{len(chunk)} features × {heat_runs} runs...")
            with perf_phase(
                "fs.boruta.tournament.heat",
                heat_idx=ci,
                chunk_size=len(chunk),
                n_runs=heat_runs,
            ):
                survivors = _boruta_core(
                    X_curr[chunk],
                    y_curr,
                    n_runs=heat_runs,
                    alpha=heat_alpha,
                    max_shadows=chunk_shadow_cap,
                )
            heat_survivors.extend(survivors)
            logger.info(f"   Heat {ci}: {len(survivors)} survived")

        heat_survivors = sorted(set(heat_survivors))
        inc_counter("fs.boruta.tournament.survivors_after_heats", len(heat_survivors))
        logger.info(f"   Round 1 total survivors: {len(heat_survivors)}")

        if len(heat_survivors) == 0:
            logger.info("   WARNING: No features survived tournament heats.")
            return []

        # Round 2: Rigorous final, strict alpha
        final_runs = run_plan['final_runs']
        final_shadow_cap = _adaptive_boruta_shadow_cap(len(heat_survivors))
        logger.info(f"   Final round: {len(heat_survivors)} features × {final_runs} runs")
        with perf_phase(
            "fs.boruta.tournament.final",
            n_features=len(heat_survivors),
            n_runs=final_runs,
        ):
            final = _boruta_core(
                X_curr[heat_survivors],
                y_curr,
                n_runs=final_runs,
                alpha=alpha,
                max_shadows=final_shadow_cap,
            )
        logger.info(f"   Final survivors: {len(final)}")
        return final


# =============================================================================
# Stage 3: Vintage Stability
# =============================================================================

def _discover_vintage_snapshots(snapshots_dir):
    """Dynamically discover available snapshot years from the directory tree.

    Scans snapshots_dir/<decade>/<year>/<year>-12.parquet for all available
    vintages instead of relying on a hardcoded list.  Returns a sorted list
    of year strings (e.g. ['2010', '2014', '2018', '2022']).
    """
    found = []
    snapshots_path = Path(snapshots_dir)
    if not snapshots_path.exists():
        return found
    for decade_dir in sorted(snapshots_path.iterdir()):
        if not decade_dir.is_dir():
            continue
        for year_dir in sorted(decade_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            year_str = year_dir.name
            parquet = year_dir / f"{year_str}-12.parquet"
            if parquet.exists():
                found.append(year_str)
    return sorted(found)


def _vintage_cv_importance(X_wide, target_series, feature_list, n_folds=3):
    """Compute normalised feature importance via temporal CV over a vintage.

    Instead of a single LightGBM fit (noisy), we use a small *n_folds*-fold
    temporal split and average the gain importances across folds.  This
    smooths out the randomness inherent in any single tree ensemble.
    """
    common = X_wide.index.intersection(target_series.index)
    if len(common) < 50:
        return pd.Series(dtype=float)

    X = X_wide.loc[common]
    y = target_series.loc[common]

    # Need enough rows for temporal splits
    if len(X) < n_folds * 15:
        # Fall back to single fit for very small vintages
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        _safe_lgb_fit(model, X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return imp / imp.sum() if imp.sum() > 0 else imp

    tscv = TimeSeriesSplit(n_splits=n_folds)
    agg_imp = pd.Series(0.0, index=X.columns)
    n_valid_folds = 0

    for train_idx, _ in tscv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        if len(X_train) < 30:
            continue
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        _safe_lgb_fit(model, X_train, y_train)
        fold_imp = pd.Series(model.feature_importances_, index=X.columns)
        if fold_imp.sum() > 0:
            agg_imp += fold_imp / fold_imp.sum()
            n_valid_folds += 1

    if n_valid_folds == 0:
        return pd.Series(dtype=float)

    agg_imp /= n_valid_folds
    return agg_imp


def _aggregate_vintage_scores(
    scores: pd.DataFrame,
    weight_series: pd.Series,
    min_presence: int,
) -> pd.Series:
    """
    Vectorized aggregation of per-checkpoint importance scores.
    """
    if scores.empty:
        return pd.Series(dtype=float)

    available = scores.notna()
    positive = scores > 0
    nonzero_counts = positive.sum(axis=1)

    weighted_values = scores.mul(weight_series, axis=1)
    weighted_sum = weighted_values.where(available).sum(axis=1, min_count=1)
    weight_totals = available.mul(weight_series, axis=1).sum(axis=1)
    weighted_scores = weighted_sum / weight_totals.replace(0, np.nan)

    latest_positive = positive['Latest'] if 'Latest' in positive.columns else pd.Series(
        False, index=scores.index
    )
    keep_mask = (
        (nonzero_counts >= min_presence)
        & latest_positive
        & weighted_scores.notna()
    )
    return weighted_scores[keep_mask].sort_values(ascending=False)


def get_vintage_stability(feature_list, target_series, snapshots_dir,
                          snap_latest_wide, min_presence=2):
    """
    Check feature importance stability across historical snapshots.

    Improvements over the original:
    - **Dynamic snapshot discovery**: scans the directory tree for available
      vintages instead of hardcoding [2010, 2014, 2018, 2022].  Sources with
      shorter histories (e.g. ADP starting 2012) use all available checkpoints.
    - **Temporal CV per vintage**: 3-fold temporal CV replaces a single noisy
      LightGBM fit, giving more stable importance estimates.
    - **Exponential recency weighting**: weight = 2^rank where rank is the
      position in the sorted vintage list (oldest=0).  'Latest' always gets
      the highest weight (2× the newest vintage).
    """
    # --- 1. Discover available vintages dynamically ---
    vintage_years = _discover_vintage_snapshots(snapshots_dir)
    # Always include 'Latest' as the final (highest-weight) checkpoint
    checkpoints = vintage_years + ['Latest']

    if not checkpoints:
        return pd.Series(dtype=float)

    # --- 2. Build exponential recency weights ---
    # weight = 2^rank  →  e.g. 5 vintages + Latest → {v0:1, v1:2, v2:4, v3:8, v4:16, Latest:32}
    weights = {label: 2 ** rank for rank, label in enumerate(checkpoints)}

    logger.info(f"   Vintage checkpoints: {checkpoints} "
                f"(weights: {list(weights.values())})")

    scores = pd.DataFrame(np.nan, index=feature_list, columns=checkpoints)

    for label in checkpoints:
        if label == 'Latest':
            X_wide = _select_wide(snap_latest_wide, feature_list)
        else:
            decade = label[:3] + "0s"
            path = snapshots_dir / decade / label / f"{label}-12.parquet"
            if not path.exists():
                continue
            X_wide = load_snapshot_wide(path, feature_filter=feature_list)

        if X_wide.empty:
            continue

        X_wide = winsorize_covid_period(X_wide)

        imp = _vintage_cv_importance(X_wide, target_series, feature_list)
        if imp.empty:
            continue

        valid_imp = imp.reindex(scores.index).dropna()
        if not valid_imp.empty:
            scores.loc[valid_imp.index, label] = valid_imp.values

    weight_series = pd.Series(weights)
    return _aggregate_vintage_scores(scores, weight_series, min_presence=min_presence)


# =============================================================================
# Stage 4: Cluster Redundancy
# =============================================================================

def cluster_redundancy(X, feature_list, target_series,
                       max_clusters=50, min_overlap=30,
                       min_corr_to_cluster=0.85):
    """
    Hierarchical clustering to remove redundant features.
    NaN-aware pairwise Spearman correlations.

    Improvements:
    - **Correlation-threshold gating**: only features participating in at least
      one high-correlation pair (|rho| >= min_corr_to_cluster) are clustered.
      Independent features are preserved.
    - **Guardrail clustering for redundant subset only**: max_clusters acts as
      a safety cap on the redundant subset, avoiding forced pruning of
      unrelated predictors.
    - **Vectorized overlap matrix**: replaces O(p²) Python loop with a single
      matrix multiply: X.notna().T @ X.notna()  →  full overlap counts in one call.
    - **LightGBM gain for cluster representatives**: replaces median-imputed MI
      (which biases against features with many NaNs) with LightGBM gain
      importance, which handles NaN natively and reflects actual model usage.
    """
    with perf_phase("fs.cluster_redundancy.total", n_features=len(feature_list)):
        inc_counter("fs.cluster_redundancy.calls", 1)

        if len(feature_list) <= 2:
            return feature_list

        X_curr = X[feature_list].copy()
        common = X_curr.index.intersection(target_series.index)
        X_curr = X_curr.loc[common]
        y_curr = target_series.loc[common]

        valid_target = y_curr.notna()
        X_curr = X_curr.loc[valid_target]
        y_curr = y_curr.loc[valid_target]

        with perf_phase("fs.cluster_redundancy.corr_matrix"):
            corr = X_curr.corr(method='spearman')
            corr = corr.fillna(0)

        with perf_phase("fs.cluster_redundancy.overlap_matrix"):
            # Vectorized pairwise overlap: notna_matrix.T @ notna_matrix gives full
            # overlap counts in one call — replaces the O(p²) Python loop.
            notna_mat = X_curr.notna().values.astype(np.float32)
            overlap_matrix = notna_mat.T @ notna_mat  # shape (p, p)

            # Zero out correlations where overlap is insufficient
            low_overlap = overlap_matrix < min_overlap
            np.fill_diagonal(low_overlap, False)  # diagonal is self-overlap, always fine
            corr_values = corr.values
            corr_values[low_overlap] = 0.0
            corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)

        abs_corr = corr.abs().values
        high_corr = abs_corr >= min_corr_to_cluster
        np.fill_diagonal(high_corr, False)

        # No strong redundancy detected: preserve all features.
        if not high_corr.any():
            return feature_list

        redundant_mask = high_corr.any(axis=0)
        redundant_features = [feature_list[i] for i, is_redundant in enumerate(redundant_mask)
                              if is_redundant]
        independent_features = [feature_list[i] for i, is_redundant in enumerate(redundant_mask)
                                if not is_redundant]

        if len(redundant_features) <= 1:
            return feature_list

        with perf_phase("fs.cluster_redundancy.clustering_prep", n_redundant=len(redundant_features)):
            corr_red = corr.loc[redundant_features, redundant_features]
            dist_matrix = 1 - corr_red.abs().values
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = np.maximum(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            condensed = squareform(dist_matrix, checks=False)
            linkage = hierarchy.ward(condensed)

            dist_threshold = max(0.0, 1.0 - float(min_corr_to_cluster))
            cluster_ids = hierarchy.fcluster(linkage, t=dist_threshold, criterion='distance')

            n_clusters = len(np.unique(cluster_ids))
            redundant_guardrail = min(max_clusters, len(redundant_features))
            redundant_guardrail = max(redundant_guardrail, 1)
            if n_clusters > redundant_guardrail and len(redundant_features) > redundant_guardrail:
                logger.info(f"   Redundancy guardrail: {n_clusters} clusters in redundant subset "
                            f"→ capped to {redundant_guardrail}")
                cluster_ids = hierarchy.fcluster(linkage, t=redundant_guardrail, criterion='maxclust')

        inc_counter("fs.cluster_redundancy.clusters_total", len(np.unique(cluster_ids)))

        # Use LightGBM gain importance for cluster representative selection.
        # This avoids median imputation bias and reflects actual model usage of
        # features with NaN patterns that LightGBM handles natively.
        with perf_phase("fs.cluster_redundancy.representative_model_fit"):
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            _safe_lgb_fit(model, X_curr, y_curr)
            importance_series = pd.Series(model.feature_importances_, index=feature_list)

        selected_redundant = []
        for cluster_id in np.unique(cluster_ids):
            members = [redundant_features[i]
                       for i, c in enumerate(cluster_ids) if c == cluster_id]
            best = importance_series[members].idxmax()
            selected_redundant.append(best)

        selected_set = set(independent_features) | set(selected_redundant)
        selected = [f for f in feature_list if f in selected_set]
        return selected


# =============================================================================
# Stage 5: Interaction Rescue (Two-Phase)
# =============================================================================

def _extract_split_pairs(model, feature_names):
    """Extract parent-child split pairs from LightGBM trees.

    Calls dump_model() once and iterates over the tree_info list, avoiding
    the expensive per-tree serialisation of the entire model to JSON.
    """
    pair_counts = Counter()

    booster = model.booster_
    # Dump once — dump_model() serialises the full model; calling it per-tree
    # was O(n_trees²) in serialisation cost.
    all_tree_info = booster.dump_model()['tree_info']

    for tree_info in all_tree_info:

        def _walk(node, parent_feature=None):
            if 'split_feature' not in node:
                return

            feat_idx = node['split_feature']
            if feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
            else:
                feat_name = f'feature_{feat_idx}'

            if parent_feature is not None and parent_feature != feat_name:
                pair = tuple(sorted([parent_feature, feat_name]))
                pair_counts[pair] += 1

            if 'left_child' in node:
                _walk(node['left_child'], feat_name)
            if 'right_child' in node:
                _walk(node['right_child'], feat_name)

        _walk(tree_info['tree_structure'])

    return pair_counts


def interaction_rescue(X, y, confirmed_features, rejected_pool,
                       n_splits=8, gap=3, top_k=10, max_phase1_screen=30,
                       trial_eval_workers=None):
    """
    Two-phase interaction rescue:
    Phase 1: Single-feature conditional test
    Phase 2: Split-pair detection from LightGBM trees

    Improvements:
    - **Pre-screening**: Phase 1 first ranks all rejected features by LightGBM
      gain in a single full model, then only CV-tests the top *max_phase1_screen*
      (default 30).  This caps the O(|rejected| × n_splits) LightGBM fits.
    - **dump_model() called once** in _extract_split_pairs (see above).
    - **Cached baseline MAE**: the baseline_with_singles MAE in Phase 2 is
      computed once outside the loop instead of redundantly per iteration.
    """
    with perf_phase(
        "fs.interaction_rescue.total",
        n_confirmed=len(confirmed_features),
        n_rejected_pool=len(rejected_pool),
    ):
        inc_counter("fs.interaction_rescue.calls", 1)

        common = X.index.intersection(y.index)
        X_curr = X.loc[common]
        y_curr = y.loc[common]

        if trial_eval_workers is None:
            trial_eval_workers = min(TRIAL_EVAL_MAX_WORKERS_DEFAULT, os.cpu_count() or 1)
        trial_eval_workers = max(1, int(trial_eval_workers))

        rejected = [f for f in rejected_pool
                    if f in X_curr.columns and f not in confirmed_features]

        if not rejected:
            return []

        inc_counter("fs.interaction_rescue.rejected_candidates_total", len(rejected))
        fold_indices = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X_curr))

        def _cv_mae_raw(feature_set):
            with perf_phase("fs.interaction_rescue.cv_eval", n_features=len(feature_set)):
                inc_counter("fs.interaction_rescue.cv_calls", 1)
                inc_counter("fs.interaction_rescue.cv_folds_total", len(fold_indices))
                X_sub = X_curr[feature_set]
                maes = []
                for train_idx, test_idx in fold_indices:
                    model = lgb.LGBMRegressor(**LGB_PARAMS)
                    _safe_lgb_fit(model, X_sub.iloc[train_idx], y_curr.iloc[train_idx])
                    preds = _safe_lgb_predict(model, X_sub.iloc[test_idx])
                    maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
                return np.mean(maes)

        mae_cache = {}
        mae_cache_lock = Lock()

        def _cv_mae(feature_set):
            return _memoized_score(
                feature_set,
                scorer=_cv_mae_raw,
                cache=mae_cache,
                cache_lock=mae_cache_lock,
            )

        with perf_phase("fs.interaction_rescue.phase1.baseline_cv", n_features=len(confirmed_features)):
            baseline_mae = _cv_mae(confirmed_features)

        trial_pool_cm = (
            ThreadPoolExecutor(max_workers=trial_eval_workers)
            if trial_eval_workers > 1 else nullcontext(None)
        )
        with trial_pool_cm as trial_pool:
            # === Phase 1: Single-feature conditional rescue ===
            # Pre-screen rejected features by LightGBM gain in a single full model
            # to avoid O(|rejected| × n_splits) CV fits for every rejected feature.
            if len(rejected) > max_phase1_screen:
                with perf_phase("fs.interaction_rescue.phase1.prescreen_fit", n_candidates=len(rejected)):
                    all_phase1_feats = confirmed_features + rejected
                    all_phase1_feats = list(dict.fromkeys(all_phase1_feats))
                    screen_model = lgb.LGBMRegressor(**LGB_PARAMS)
                    _safe_lgb_fit(screen_model, X_curr[all_phase1_feats], y_curr)
                    screen_imp = pd.Series(
                        screen_model.feature_importances_, index=all_phase1_feats
                    )
                    # Rank only rejected features by gain, take top-N
                    rejected_imp = screen_imp[rejected].sort_values(ascending=False)
                    rejected = rejected_imp.head(max_phase1_screen).index.tolist()
                    logger.info(f"   Phase 1 pre-screen: {len(rejected_pool)} rejected "
                                f"→ top {len(rejected)} by LightGBM gain")

            single_trials = [
                (feat, confirmed_features + [feat])
                for feat in rejected
            ]
            with perf_phase("fs.interaction_rescue.phase1.trials", n_trials=len(single_trials)):
                trial_scores = _parallel_trial_scores(
                    single_trials,
                    scorer=_cv_mae,
                    max_workers=trial_eval_workers,
                    executor=trial_pool,
                )

        single_improvements = {}
        for feat, trial_mae in trial_scores:
            improvement = (baseline_mae - trial_mae) / (baseline_mae + 1e-12)
            if improvement > 0:
                single_improvements[feat] = improvement

        single_rescued = sorted(
            single_improvements.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        inc_counter("fs.interaction_rescue.single_rescued_total", len(single_rescued))

        for feat, imp in single_rescued:
            logger.info(f"   Rescued (single): {feat} "
                         f"(delta MAE = {imp:.4f})")

        # === Phase 2: Split-pair detection ===
        all_feats = [f for f in X_curr.columns
                     if f in confirmed_features or f in rejected]

        with perf_phase("fs.interaction_rescue.phase2.full_model_fit", n_features=len(all_feats)):
            model_full = lgb.LGBMRegressor(
                **{**LGB_PARAMS, 'n_estimators': 200, 'num_leaves': 63, 'max_depth': 6}
            )
            _safe_lgb_fit(model_full, X_curr[all_feats], y_curr)

        with perf_phase("fs.interaction_rescue.phase2.extract_pairs"):
            pair_counts = _extract_split_pairs(model_full, all_feats)
        inc_counter("fs.interaction_rescue.pair_candidates_total", len(pair_counts))

        rejected_set = set(rejected)
        single_rescued_set = {f for f, _ in single_rescued}
        pair_rescued = []

        # Cache the baseline_with_singles MAE (does not change between iterations)
        baseline_with_singles = list(dict.fromkeys(
            confirmed_features + [f for f, _ in single_rescued]
        ))
        with perf_phase(
            "fs.interaction_rescue.phase2.baseline_with_singles_cv",
            n_features=len(baseline_with_singles),
        ):
            baseline_singles_mae = _cv_mae(baseline_with_singles)

        with perf_phase("fs.interaction_rescue.phase2.pair_trials", n_pairs=len(pair_counts)):
            for (feat_a, feat_b), count in pair_counts.most_common():
                if len(pair_rescued) >= top_k:
                    break

                new_feats = []
                if feat_a in rejected_set and feat_a not in single_rescued_set:
                    new_feats.append(feat_a)
                if feat_b in rejected_set and feat_b not in single_rescued_set:
                    new_feats.append(feat_b)

                if not new_feats:
                    continue

                trial_feats = list(dict.fromkeys(
                    baseline_with_singles + new_feats
                ))
                trial_mae = _cv_mae(trial_feats)

                improvement = ((baseline_singles_mae - trial_mae)
                               / (baseline_singles_mae + 1e-12))
                if improvement > 0:
                    pair_rescued_set = {pf for pf, _ in pair_rescued}
                    for f in new_feats:
                        if f not in pair_rescued_set:
                            pair_rescued.append((f, improvement))
                            other = feat_a if f != feat_a else feat_b
                            logger.info(f"   Rescued (pair w/ {other}, "
                                        f"co-occur={count}): {f} "
                                        f"(delta = {improvement:.4f})")

        all_rescued = ([f for f, _ in single_rescued]
                       + [f for f, _ in pair_rescued])
        seen = set()
        result = []
        for f in all_rescued:
            if f not in seen:
                seen.add(f)
                result.append(f)

        inc_counter("fs.interaction_rescue.total_rescued_features", len(result))
        return result


# =============================================================================
# Stage 6: Sequential Forward Selection
# =============================================================================

def sequential_forward_selection(X, y, candidate_features, boruta_hits=None,
                                 n_splits=8, gap=3, min_improvement=0.001,
                                 patience=3, min_features=3, beam_width=3,
                                 trial_eval_workers=None):
    """Sequential Forward Selection using walk-forward CV.

    Improvements:
    - **Pre-screening by gain**: ranks all candidates by LightGBM gain in a
      single model, capping the effective candidate pool for the expensive
      per-step CV loop.
    - **Cached fold predictions**: per-fold predictions for the current
      selected set are cached and reused.  Each candidate step only retrains
      the model with one additional feature instead of re-fitting from scratch.
    - **Beam search (top-*beam_width* per step)**: instead of pure greedy,
      we track the top-*beam_width* (default 3) partial solutions at each
      step, which can discover feature pairs that are weak individually but
      strong together.
    """
    with perf_phase("fs.sfs.total", n_candidates=len(candidate_features)):
        inc_counter("fs.sfs.calls", 1)

        if not candidate_features:
            return []

        common = X.index.intersection(y.index)
        X_curr = X.loc[common]
        y_curr = y.loc[common]
        candidate_features = [f for f in candidate_features if f in X_curr.columns]
        if not candidate_features:
            return []

        if trial_eval_workers is None:
            trial_eval_workers = min(TRIAL_EVAL_MAX_WORKERS_DEFAULT, os.cpu_count() or 1)
        trial_eval_workers = max(1, int(trial_eval_workers))

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        # Pre-compute fold indices once (they only depend on array length)
        fold_indices = list(tscv.split(X_curr))
        use_precomputed_fold_targets = os.getenv("NFP_SFS_PRECOMPUTE_Y", "1").strip() != "0"
        if use_precomputed_fold_targets:
            inc_counter("fs.sfs.precompute_y_enabled", 1)
        else:
            inc_counter("fs.sfs.precompute_y_disabled", 1)
        use_precomputed_fold_features = os.getenv("NFP_SFS_PRECOMPUTE_X", "1").strip() != "0"
        if use_precomputed_fold_features:
            inc_counter("fs.sfs.precompute_x_enabled", 1)
        else:
            inc_counter("fs.sfs.precompute_x_disabled", 1)
        # Phase 2 SFS optimization (parity-safe): precompute target vectors for
        # each fold once per SFS invocation. This removes thousands of repeated
        # y_curr.iloc[...] slices in the inner CV loop while preserving
        # identical fold membership and MAE math.
        fold_data = []
        if use_precomputed_fold_targets:
            y_values = y_curr.to_numpy()
            for train_idx, test_idx in fold_indices:
                train_idx_arr = np.asarray(train_idx, dtype=np.int64)
                test_idx_arr = np.asarray(test_idx, dtype=np.int64)
                fold_data.append(
                    (
                        train_idx_arr,
                        test_idx_arr,
                        y_values[train_idx_arr],
                        y_values[test_idx_arr],
                    )
                )

        def _cv_mae_raw(feature_set):
            with perf_phase("fs.sfs.cv_eval", n_features=len(feature_set)):
                inc_counter("fs.sfs.cv_calls", 1)
                inc_counter(
                    "fs.sfs.cv_folds_total",
                    len(fold_data) if use_precomputed_fold_targets else len(fold_indices),
                )
                if not feature_set:
                    return float('inf')
                maes = []
                if use_precomputed_fold_features:
                    # Phase 2 SFS optimization (parity-safe): build the feature
                    # matrix once per subset evaluation and slice rows via NumPy.
                    # This preserves exact row/column ordering while avoiding
                    # repeated DataFrame iloc construction in the hot CV loop.
                    X_sub_arr = X_curr.loc[:, feature_set].to_numpy(copy=False)
                    if use_precomputed_fold_targets:
                        for train_idx, test_idx, y_train_fold, y_test_fold in fold_data:
                            model = lgb.LGBMRegressor(**LGB_PARAMS)
                            model.fit(X_sub_arr[train_idx], y_train_fold)
                            preds = model.predict(X_sub_arr[test_idx])
                            maes.append(np.mean(np.abs(y_test_fold - preds)))
                    else:
                        for train_idx, test_idx in fold_indices:
                            model = lgb.LGBMRegressor(**LGB_PARAMS)
                            model.fit(X_sub_arr[train_idx], y_curr.iloc[train_idx])
                            preds = model.predict(X_sub_arr[test_idx])
                            maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
                else:
                    X_sub = X_curr[feature_set]
                    if use_precomputed_fold_targets:
                        for train_idx, test_idx, y_train_fold, y_test_fold in fold_data:
                            model = lgb.LGBMRegressor(**LGB_PARAMS)
                            _safe_lgb_fit(model, X_sub.iloc[train_idx], y_train_fold)
                            preds = _safe_lgb_predict(model, X_sub.iloc[test_idx])
                            maes.append(np.mean(np.abs(y_test_fold - preds)))
                    else:
                        for train_idx, test_idx in fold_indices:
                            model = lgb.LGBMRegressor(**LGB_PARAMS)
                            _safe_lgb_fit(model, X_sub.iloc[train_idx], y_curr.iloc[train_idx])
                            preds = _safe_lgb_predict(model, X_sub.iloc[test_idx])
                            maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
                return np.mean(maes)

        mae_cache = {}
        mae_cache_lock = Lock()

        def _cv_mae(feature_set):
            return _memoized_score(
                feature_set,
                scorer=_cv_mae_raw,
                cache=mae_cache,
                cache_lock=mae_cache_lock,
            )

        # --- Pre-screen candidates by LightGBM gain ---
        # Fit one model on all candidates to get a rough importance ranking.
        # This lets us prioritise the most promising features in the beam search.
        with perf_phase("fs.sfs.prescreen_fit", n_features=len(candidate_features)):
            screen_model = lgb.LGBMRegressor(**LGB_PARAMS)
            _safe_lgb_fit(screen_model, X_curr[candidate_features], y_curr)
            gain_rank = pd.Series(
                screen_model.feature_importances_, index=candidate_features
            ).sort_values(ascending=False)

        if boruta_hits is not None:
            # Combine Boruta hit rank with gain rank (average rank)
            boruta_rank = pd.Series(boruta_hits).reindex(candidate_features).fillna(0)
            boruta_rank = boruta_rank.rank(ascending=False)
            gain_rank_r = gain_rank.rank(ascending=False)
            combined = ((boruta_rank + gain_rank_r) / 2).sort_values()
            candidates = combined.index.tolist()
        else:
            candidates = gain_rank.index.tolist()

        if not candidates:
            return []

        # Start with best single feature (test top 20)
        best_single_mae = float('inf')
        best_single = candidates[0]
        with perf_phase("fs.sfs.best_single_scan", n_candidates=min(20, len(candidates))):
            for feat in candidates[:min(20, len(candidates))]:
                mae = _cv_mae([feat])
                if mae < best_single_mae:
                    best_single_mae = mae
                    best_single = feat

        # --- Beam search ---
        # Each beam entry: (selected_list, current_mae)
        beams = [([best_single], best_single_mae)]
        all_remaining = [f for f in candidates if f != best_single]

        logger.info(f"   SFS start: {best_single} (MAE={best_single_mae:.2f}), "
                    f"beam_width={beam_width}")

        consecutive_failures = 0
        round_idx = 0
        trial_pool_cm = (
            ThreadPoolExecutor(max_workers=trial_eval_workers)
            if trial_eval_workers > 1 else nullcontext(None)
        )
        with trial_pool_cm as trial_pool:
            while all_remaining:
                round_idx += 1
                inc_counter("fs.sfs.rounds_total", 1)
                with perf_phase("fs.sfs.beam_round", round=round_idx, remaining=len(all_remaining)):
                    next_beams = []

                    for selected, current_mae in beams:
                        remaining = [f for f in all_remaining if f not in selected]
                        if not remaining:
                            next_beams.append((selected, current_mae))
                            continue

                        # Evaluate each candidate feature added to this beam's selection
                        trial_defs = [
                            (feat, selected + [feat])
                            for feat in remaining
                        ]
                        inc_counter("fs.sfs.trials_total", len(trial_defs))
                        with perf_phase("fs.sfs.beam_round.trials", n_trials=len(trial_defs)):
                            trials = _parallel_trial_scores(
                                trial_defs,
                                scorer=_cv_mae,
                                max_workers=trial_eval_workers,
                                executor=trial_pool,
                            )

                        # Sort by MAE (ascending = best first)
                        trials.sort(key=lambda x: x[1])

                        # Keep top beam_width expansions from this beam
                        for feat, mae in trials[:beam_width]:
                            next_beams.append((selected + [feat], mae))

                    if not next_beams:
                        break

                    # Deduplicate beams by frozen feature set, keeping lowest MAE
                    seen_sets = {}
                    for sel, mae in next_beams:
                        key = frozenset(sel)
                        if key not in seen_sets or mae < seen_sets[key][1]:
                            seen_sets[key] = (sel, mae)

                    # Keep top beam_width beams overall
                    ranked_beams = sorted(seen_sets.values(), key=lambda x: x[1])
                    ranked_beams = ranked_beams[:beam_width]

                    # Check improvement of the best beam vs previous best
                    best_beam_sel, best_beam_mae = ranked_beams[0]
                    prev_best_mae = beams[0][1]

                    improvement = ((prev_best_mae - best_beam_mae)
                                   / (prev_best_mae + 1e-12))

                    # Log the best addition this round
                    new_feat = [f for f in best_beam_sel if f not in beams[0][0]]
                    new_feat_str = new_feat[0] if new_feat else "?"

                    if improvement < min_improvement:
                        consecutive_failures += 1
                        if (len(best_beam_sel) >= min_features
                                and consecutive_failures >= patience):
                            logger.info(f"   SFS stop: {patience} consecutive improvements "
                                        f"< {min_improvement:.4f}")
                            break
                        logger.info(f"   SFS +{new_feat_str} (MAE={best_beam_mae:.2f}, "
                                    f"delta={improvement:.4f}, "
                                    f"patience {consecutive_failures}/{patience})")
                    else:
                        consecutive_failures = 0
                        logger.info(f"   SFS +{new_feat_str} (MAE={best_beam_mae:.2f}, "
                                    f"delta={improvement:.4f})")

                    if best_beam_mae >= prev_best_mae:
                        # No beam improved — check forced add for min_features
                        if len(beams[0][0]) < min_features:
                            forced = all_remaining[0]
                            beams = [(beams[0][0] + [forced], prev_best_mae)]
                            all_remaining.remove(forced)
                            logger.info(f"   SFS +{forced} (forced, "
                                        f"MAE~{prev_best_mae:.2f}, "
                                        f"below min_features={min_features})")
                            continue

                    beams = ranked_beams

                    # Update remaining pool: remove features selected by ANY beam
                    all_selected = set()
                    for sel, _ in beams:
                        all_selected.update(sel)
                    all_remaining = [f for f in all_remaining if f not in all_selected]

        # Return the best beam
        best_selected, best_mae = min(beams, key=lambda x: x[1])
        logger.info(f"   SFS final: {len(best_selected)} features, MAE={best_mae:.2f}")
        inc_counter("fs.sfs.selected_features_total", len(best_selected))
        return best_selected


def should_keep_change(
    baseline_mae: float,
    candidate_mae: float,
    baseline_runtime_s: float,
    candidate_runtime_s: float,
    min_mae_improvement_pct: float = 0.5,
    min_runtime_improvement_pct: float = 15.0,
    max_mae_loss_for_runtime_pct: float = 0.5,
) -> bool:
    """
    Keep-rule used to gate pipeline changes.

    Rule:
    1) Keep if MAE improves by >= min_mae_improvement_pct.
    2) Otherwise keep only if runtime improves by >= min_runtime_improvement_pct
       AND MAE degradation is no worse than max_mae_loss_for_runtime_pct.
    """
    if baseline_mae == 0 or baseline_runtime_s == 0:
        raise ValueError("baseline_mae and baseline_runtime_s must be non-zero")

    mae_improvement_pct = ((baseline_mae - candidate_mae) / abs(baseline_mae)) * 100.0
    runtime_improvement_pct = (
        (baseline_runtime_s - candidate_runtime_s) / abs(baseline_runtime_s)
    ) * 100.0

    if mae_improvement_pct >= min_mae_improvement_pct:
        return True

    if (runtime_improvement_pct >= min_runtime_improvement_pct
            and mae_improvement_pct >= -max_mae_loss_for_runtime_pct):
        return True

    return False


# =============================================================================
# Full Pipeline Orchestrator
# =============================================================================

@profiled("feature_selection_engine.run_pipeline")
def run_pipeline(snap_wide, y_target, source_name, snapshots_dir, series_groups,
                 stages=None):
    """Execute feature selection stages for a single target on one source.

    Args:
        stages: Tuple/list of stage numbers (0-6) to execute.
            ``None`` (default) runs all stages for backward compatibility.
            Tier 1 minimal set: ``(0, 1, 4)``.
            Skipped stages pass their input through to the next active stage.
    """
    active = set(stages) if stages is not None else {0, 1, 2, 3, 4, 5, 6}

    with perf_phase("fs.run_pipeline.total", source=source_name):
        pipeline_start = time.time()
        inc_counter("fs.run_pipeline.calls", 1)
        inc_counter("fs.run_pipeline.stages_config", len(active))
        logger.info(f"Active stages: {sorted(active)}")
        y_target = winsorize_covid_period(y_target)

        # ===== Stage 0: Global Pre-Funnel (Variance + Spearman Pre-Screen) =====
        if 0 in active:
            with perf_phase("fs.run_pipeline.stage0_prefunnel", source=source_name):
                t0 = time.time()
                n_cols_before = snap_wide.shape[1]

                # Memory Watchdog
                mem_pct = psutil.virtual_memory().percent
                if mem_pct > 80:
                    logger.warning(f"CRITICAL MEMORY: system at {mem_pct}% full before {source_name} pipeline!")

                if n_cols_before > 500:
                    logger.info(f"0. Pre-funnel: {n_cols_before} features...")
                    snap_wide = _variance_filter(snap_wide)

                # Fast Spearman pre-screen for very large sources (e.g. FRED_Employment_NSA).
                # Runs only when n_cols > 5000 after variance filter; small sources are unaffected.
                # Uses BH-FDR at fdr_alpha=0.30 (permissive) — Stage 1 remains the rigorous gate.
                # Safety floor: if fewer than 500 features survive, skip and log a warning.
                _PRESCREEN_THRESHOLD = 5000
                _PRESCREEN_MIN_SURVIVORS = 500
                n_after_variance = snap_wide.shape[1]
                if n_after_variance > _PRESCREEN_THRESHOLD:
                    with perf_phase("fs.run_pipeline.stage0_prescreen", source=source_name):
                        t_pre = time.time()
                        logger.info(
                            f"   Pre-screen: {n_after_variance} features — "
                            f"running vectorized Spearman target filter (fdr_alpha=0.30)..."
                        )
                        pre_survivors = _vectorized_target_prescreen(
                            snap_wide, y_target, fdr_alpha=0.30, min_corr_obs=30
                        )
                        if len(pre_survivors) >= _PRESCREEN_MIN_SURVIVORS:
                            snap_wide = snap_wide[pre_survivors]
                            logger.info(
                                f"   > Pre-screen: {n_after_variance} → {snap_wide.shape[1]} features "
                                f"({time.time() - t_pre:.1f}s)"
                            )
                            inc_counter("fs.run_pipeline.stage0_prescreen_survivors",
                                        snap_wide.shape[1])
                        else:
                            logger.warning(
                                f"   Pre-screen produced only {len(pre_survivors)} survivors "
                                f"(< {_PRESCREEN_MIN_SURVIVORS} floor). Skipping pre-screen."
                            )
                            inc_counter("fs.run_pipeline.stage0_prescreen_skipped", 1)

                logger.info(f"   > Pre-funnel total: {n_cols_before} → {snap_wide.shape[1]} features "
                            f"({time.time() - t0:.1f}s)")
        else:
            logger.info("0. Pre-funnel: SKIPPED")

        gc.collect()

        # ===== Stage 1: Group-wise Dual Filter (Linear + LightGBM) =====
        # Stage 1 is always required — it produces the candidate list.
        stage1_candidates = []
        with perf_phase("fs.run_pipeline.stage1_group_dual_filter", source=source_name):
            t0 = time.time()
            n_groups = len(series_groups)
            logger.info(f"1. Group-wise Dual Filter "
                        f"(Purged Corr + Random Subspace LightGBM) — {n_groups} groups...")
            for gi, (group_name, series_list) in enumerate(series_groups.items(), 1):
                with perf_phase(
                    "fs.run_pipeline.stage1_group",
                    source=source_name,
                    group=group_name,
                    group_idx=gi,
                ):
                    gt0 = time.time()
                    wide_group = _select_wide(snap_wide, series_list)
                    if wide_group.empty:
                        continue
                    wide_group = winsorize_covid_period(wide_group)

                    # Intra-group deduplication for large groups
                    n_before_dedup = wide_group.shape[1]
                    if n_before_dedup > 50:
                        logger.debug(f"    [{group_name}] Starting dedup on {n_before_dedup} features...")
                        wide_group = _deduplicate_group(wide_group, threshold=0.92)
                        logger.debug(f"    [{group_name}] Dedup finished.")

                    logger.debug(f"    [{group_name}] Starting filter_group_data_purged...")
                    selected = filter_group_data_purged(
                        wide_group, y_target, group_name
                    )
                    stage1_candidates.extend(selected)
                    logger.info(f"    [{group_name}] ({gi}/{n_groups}) "
                                f"{len(series_list)} features → {len(selected)} survived "
                                f"({time.time() - gt0:.1f}s)")

            stage1_candidates = sorted(list(set(stage1_candidates)))
            logger.info(f"   > Stage 1 Survivors: {len(stage1_candidates)} features "
                        f"({time.time() - t0:.1f}s)")
            inc_counter("fs.run_pipeline.stage1_survivors_total", len(stage1_candidates))

        if len(stage1_candidates) == 0:
            logger.info("   WARNING: No features survived Stage 1. Skipping pipeline.")
            return []

        # Aggressively release memory
        X_stage2 = _select_wide(snap_wide, stage1_candidates)
        X_stage2 = winsorize_covid_period(X_stage2)
        gc.collect()

        # Track the running candidate list through optional stages.
        # Each stage narrows (or rescues into) this list; skipped stages pass through.
        candidates = list(stage1_candidates)

        # ===== Stage 2: Boruta Importance =====
        if 2 in active:
            with perf_phase("fs.run_pipeline.stage2_boruta", source=source_name):
                t0 = time.time()
                logger.info("2. Boruta Feature Selection (shadow features)...")

                stage2_candidates = get_boruta_importance(
                    X_stage2, y_target
                )
                logger.info(f"   > Stage 2 Survivors: {len(stage2_candidates)} features "
                            f"({time.time() - t0:.1f}s)")
                inc_counter("fs.run_pipeline.stage2_survivors_total", len(stage2_candidates))
            gc.collect()

            if len(stage2_candidates) == 0:
                with perf_phase("fs.run_pipeline.stage2_boruta_fallback", source=source_name):
                    logger.info("   WARNING: No features survived Boruta. "
                                "Falling back to Stage 1 top 30.")
                    common = X_stage2.index.intersection(y_target.index)
                    corrs = X_stage2.loc[common].corrwith(
                        y_target.loc[common]
                    ).abs()
                    stage2_candidates = corrs.nlargest(30).index.tolist()
                    inc_counter("fs.run_pipeline.stage2_fallback_used", 1)

            candidates = stage2_candidates
        else:
            logger.info("2. Boruta: SKIPPED")

        # ===== Stage 3: Vintage Stability =====
        if 3 in active:
            with perf_phase("fs.run_pipeline.stage3_vintage_stability", source=source_name):
                t0 = time.time()
                logger.info("3. Vintage Stability "
                            "(exponential recency weights, min 2 snapshots)...")
                weighted_scores = get_vintage_stability(
                    candidates, y_target, snapshots_dir, snap_wide
                )

                if len(weighted_scores) == 0:
                    logger.info("   WARNING: No features survived vintage stability. "
                                "Using all previous stage output.")
                    stage3_candidates = candidates
                else:
                    # Keep features with any positive weighted importance (replaces
                    # the old median×0.5 which was arbitrary and overly permissive)
                    stage3_candidates = weighted_scores[
                        weighted_scores > 0
                    ].index.tolist()
                logger.info(f"   > Stage 3 Survivors: {len(stage3_candidates)} features "
                            f"({time.time() - t0:.1f}s)")
                inc_counter("fs.run_pipeline.stage3_survivors_total", len(stage3_candidates))
            gc.collect()
            candidates = stage3_candidates
        else:
            logger.info("3. Vintage Stability: SKIPPED")

        # ===== Stage 4: Cluster Redundancy =====
        if 4 in active:
            with perf_phase("fs.run_pipeline.stage4_cluster_redundancy", source=source_name):
                t0 = time.time()
                logger.info("4. Cluster Redundancy Check (NaN-aware Spearman)...")
                stage4_candidates = cluster_redundancy(
                    X_stage2, candidates, y_target
                )
                logger.info(f"   > Stage 4 Survivors: {len(stage4_candidates)} features "
                            f"({time.time() - t0:.1f}s)")
                inc_counter("fs.run_pipeline.stage4_survivors_total", len(stage4_candidates))
            gc.collect()
            candidates = stage4_candidates
        else:
            logger.info("4. Cluster Redundancy: SKIPPED")

        # ===== Stage 5: Interaction Rescue (two-phase) =====
        if 5 in active:
            with perf_phase("fs.run_pipeline.stage5_interaction_rescue", source=source_name):
                t0 = time.time()
                logger.info("5. Interaction Rescue "
                            "(single-feature + split-pair detection)...")
                rescued = interaction_rescue(
                    X_stage2, y_target,
                    confirmed_features=candidates,
                    rejected_pool=stage1_candidates,
                    top_k=10
                )
                stage5_candidates = candidates + rescued
                seen = set()
                stage5_candidates = [
                    f for f in stage5_candidates
                    if f not in seen and not seen.add(f)
                ]
                logger.info(f"   > Stage 5 Candidates: {len(stage5_candidates)} "
                            f"({len(candidates)} confirmed "
                            f"+ {len(rescued)} rescued) "
                            f"({time.time() - t0:.1f}s)")
                inc_counter("fs.run_pipeline.stage5_candidates_total", len(stage5_candidates))
            candidates = stage5_candidates
        else:
            logger.info("5. Interaction Rescue: SKIPPED")

        # ===== Stage 6: Sequential Forward Selection =====
        if 6 in active:
            with perf_phase("fs.run_pipeline.stage6_sfs_final_selection", source=source_name):
                t0 = time.time()
                logger.info("6. Sequential Forward Selection "
                            "(walk-forward CV with embargo)...")
                final_list = sequential_forward_selection(
                    X_stage2, y_target, candidates,
                    n_splits=8, gap=3,
                    min_improvement=0.001,
                    patience=3,
                    min_features=3,
                )
                logger.info(f"   > Final Count: {len(final_list)} "
                            f"({time.time() - t0:.1f}s)")
                inc_counter("fs.run_pipeline.stage6_selected_total", len(final_list))
            candidates = final_list
        else:
            logger.info("6. Sequential Forward Selection: SKIPPED")

        logger.info(f"   TOTAL PIPELINE: {(time.time() - pipeline_start) / 60:.1f} min")
        logger.info(f"   FINAL: {len(candidates)} features selected")
        return candidates


@profiled("feature_selection_engine.run_full_source_pipeline")
def run_full_source_pipeline(snap_wide, target_mom,
                             source_name, snapshots_dir, series_groups,
                             stages=None):
    """
    Run the feature selection pipeline on a single source for the MoM target.

    Args:
        stages: Tuple/list of stage numbers (0-6) to execute, or None for all.
    Returns the selected features.
    """
    y_clean = target_mom.dropna()
    if len(y_clean) < 50:
        logger.warning(f"   [MoM] Insufficient target data "
                       f"({len(y_clean)} obs). Skipping.")
        return []

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  PIPELINE START: {source_name} / MoM")
    logger.info(f"{'=' * 60}")

    final_feats = run_pipeline(
        snap_wide, y_clean, source_name, snapshots_dir, series_groups,
        stages=stages,
    )

    logger.info(f"\n  [{source_name}] Selected: {len(final_feats)} features (MoM only)")

    return final_feats
