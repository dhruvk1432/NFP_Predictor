"""
7-Stage Feature Selection Engine
================================
Exact replica of the per-source Jupyter notebook pipelines.
Each stage matches the notebook implementation precisely.

Stages:
1. Group-wise Dual Filter (Purged Expanding Correlation + LightGBM + VIF)
2. Boruta Feature Selection (shadow features, 100 runs)
3. Vintage Stability (exponential recency weighting across historical snapshots)
4. Cluster Redundancy (NaN-aware Spearman hierarchical clustering)
5. Interaction Rescue (two-phase: single-feature + split-pair detection)
6. Sequential Forward Selection (walk-forward CV with embargo)
7. Union of MoM + Acceleration targets (handled by caller)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import binomtest, t as t_dist
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import defaultdict, Counter
import re
import logging

from utils.transforms import winsorize_covid_period

logger = logging.getLogger(__name__)

SEED = 42
MIN_VALID_OBS = 60

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': 1,
    'random_state': SEED,
}

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

    elif source_name == 'FRED_Employment':
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
    """
    df = pd.read_parquet(path)
    if 'series_name' in df.columns and 'value' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if feature_filter:
            df = df[df['series_name'].isin(feature_filter)]
        df = df.drop_duplicates(subset=['date', 'series_name'], keep='last')
        wide = df.pivot(index='date', columns='series_name', values='value')
    else:
        df['date'] = pd.to_datetime(df['date'])
        wide = df.set_index('date')
        wide = wide.drop(columns=['snapshot_date'], errors='ignore')
        wide = wide.select_dtypes(include=[np.number])
        if feature_filter:
            available = [c for c in feature_filter if c in wide.columns]
            wide = wide[available]
    wide = wide.sort_index().dropna(axis=1, how='all').dropna(axis=0, how='all')
    return wide


def _select_wide(snap_wide, series_list):
    """Select a subset of columns from a wide DataFrame, dropping all-NaN rows/cols."""
    available = [s for s in series_list if s in snap_wide.columns]
    if not available:
        return pd.DataFrame()
    subset = snap_wide[available].copy()
    subset = subset.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return subset


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
    return model.predict(X_clean)


def _lgb_importances(model, mapping):
    clean_names = model.feature_name_
    imp_vals = model.feature_importances_
    return pd.Series({mapping.get(c, c): v for c, v in zip(clean_names, imp_vals)})


# =============================================================================
# Stage 1: Group-wise Dual Filter (Purged Correlation + LightGBM + VIF)
# =============================================================================

def _purged_expanding_corr(feature_series, target_series,
                           min_window=60, purge_months=3):
    """
    Compute weighted-average correlation using expanding windows with a purge gap.
    Handles staggered start dates: only uses dates where the feature has data.
    """
    common = feature_series.dropna().index.intersection(target_series.dropna().index)
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
    for end_idx in eval_points:
        eval_date = common[end_idx]
        purge_cutoff = eval_date - pd.DateOffset(months=purge_months)
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


def _lgb_screen_group(wide_df, target_series, top_k=15):
    """
    LightGBM-based group screening: catches features useful for tree splits
    even if they have zero marginal correlation with the target.
    """
    common = wide_df.index.intersection(target_series.index)
    if len(common) < 50:
        return []

    X = wide_df.loc[common]
    y = target_series.loc[common]

    # Recency check
    last_date = X.index.max()
    recent_cutoff = last_date - pd.DateOffset(months=3)
    recency_mask = X.apply(lambda col: col.last_valid_index() >= recent_cutoff)
    X = X.loc[:, recency_mask]

    if X.shape[1] == 0:
        return []

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    nonzero = imp[imp > 0].sort_values(ascending=False)
    return nonzero.head(top_k).index.tolist()


def _is_binary_feature(series, threshold=5):
    """Check if a feature is effectively binary (<=threshold unique non-NaN values)."""
    return series.dropna().nunique() <= threshold


def filter_group_data_purged(wide_df, target_series, group_name,
                             vif_threshold=10, fdr_alpha=0.10):
    """
    Group-wise filter with DUAL screening paths:

    Path A (Linear): Purged expanding-window correlation + BH-FDR significance
    Path B (Non-linear): LightGBM gain importance pre-screen

    The UNION of both paths goes through VIF reduction.
    Binary regime groups skip VIF (meaningless for 0/1 variables).
    """
    common = wide_df.index.intersection(target_series.index)
    X_local = wide_df.loc[common].copy()
    y_local = target_series.loc[common]

    if X_local.empty:
        return []

    # 1. Recency Check
    last_date = X_local.index.max()
    recent_cutoff = last_date - pd.DateOffset(months=3)
    recency_mask = X_local.apply(
        lambda col: col.last_valid_index() >= recent_cutoff
    )
    X_local = X_local.loc[:, recency_mask]

    if X_local.shape[1] == 0:
        return []

    # Detect if this is a binary-regime group
    binary_cols = [c for c in X_local.columns if _is_binary_feature(X_local[c])]
    is_binary_group = len(binary_cols) > len(X_local.columns) * 0.5

    # Relaxed min_window for binary groups (fewer events = fewer valid obs)
    min_window = 30 if is_binary_group else 60

    # === Path A: Purged Expanding-Window Correlation + BH-FDR ===
    feature_pvalues = {}
    for col in X_local.columns:
        avg_corr, eff_n = _purged_expanding_corr(
            X_local[col], y_local, min_window=min_window
        )

        if np.isnan(avg_corr) or eff_n < min_window:
            continue

        denom = max(1 - avg_corr ** 2, 1e-8)
        t_stat = avg_corr * np.sqrt((eff_n - 2) / denom)
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=eff_n - 2))
        feature_pvalues[col] = p_value

    corr_features = []
    if feature_pvalues:
        sorted_features = sorted(feature_pvalues.items(), key=lambda x: x[1])
        n_tests = len(sorted_features)
        for rank, (feat, pval) in enumerate(sorted_features, 1):
            bh_threshold = (rank / n_tests) * fdr_alpha
            if pval <= bh_threshold:
                corr_features.append(feat)
            else:
                break

    # === Path B: LightGBM Gain Importance Pre-Screen ===
    lgb_features = _lgb_screen_group(X_local, y_local, top_k=15)

    # === Union of both paths ===
    passed_features = sorted(set(corr_features) | set(lgb_features))

    if not passed_features:
        return []

    # Skip VIF for binary-only groups (VIF is meaningless for 0/1 variables)
    if is_binary_group:
        logger.info(f"    [{group_name}] Binary group - skipping VIF, "
                     f"keeping {len(passed_features)} features")
        return passed_features

    # 3. Iterative Pruning for VIF (need dense matrix)
    X_vif = X_local[passed_features].copy()

    target_rows = 100
    current_rows = X_vif.dropna()
    while len(current_rows) < target_rows and X_vif.shape[1] > 1:
        nan_counts = X_vif.isna().sum()
        worst_col = nan_counts.idxmax()
        X_vif = X_vif.drop(columns=[worst_col])
        current_rows = X_vif.dropna()

    X_vif = X_vif.dropna()
    y_vif = y_local.loc[X_vif.index]

    if len(X_vif) < 50 or X_vif.shape[1] == 0:
        return passed_features[:15]

    if X_vif.shape[1] > 60:
        corrs = X_vif.corrwith(y_vif).abs()
        top_60 = corrs.nlargest(60).index
        X_vif = X_vif[top_60]

    # 4. VIF Iterative Reduction
    while X_vif.shape[1] > 1:
        try:
            vifs = [variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])]
            max_vif = max(vifs)
            if max_vif <= vif_threshold:
                break

            max_idx = vifs.index(max_vif)
            feat_max = X_vif.columns[max_idx]

            curr_corr = X_vif.corr().abs()
            partner = curr_corr[feat_max].drop(feat_max).idxmax()

            if abs(X_vif[feat_max].corr(y_vif)) < abs(X_vif[partner].corr(y_vif)):
                drop_col = feat_max
            else:
                drop_col = partner

            X_vif = X_vif.drop(columns=[drop_col])
        except Exception as e:
            logger.warning(f"  [Group {group_name}] VIF Error: {e}")
            break

    return X_vif.columns.tolist()


# =============================================================================
# Stage 2: Boruta Feature Selection
# =============================================================================

def get_boruta_importance(X, y, n_runs=100, block_size=6, alpha=0.05):
    """
    Boruta-style feature selection with shadow features.
    Compares each real feature to the 75th percentile of all shadow importances.
    """
    common = X.index.intersection(y.index)
    X_curr = X.loc[common]
    y_curr = y.loc[common]

    n_features = X_curr.shape[1]
    feature_names = X_curr.columns.tolist()

    hits = np.zeros(n_features, dtype=int)

    for run in range(n_runs):
        rng = np.random.RandomState(SEED + run)

        shadow_data = {}
        for col in feature_names:
            col_vals = X_curr[col].values.copy()
            valid_mask = (
                ~np.isnan(col_vals)
                if np.issubdtype(col_vals.dtype, np.floating)
                else np.ones(len(col_vals), dtype=bool)
            )
            valid_vals = col_vals[valid_mask].copy()
            rng.shuffle(valid_vals)
            col_vals[valid_mask] = valid_vals
            shadow_data[f'_shadow_{col}'] = col_vals

        shadow_df = pd.DataFrame(shadow_data, index=X_curr.index)
        X_combined = pd.concat([X_curr, shadow_df], axis=1)

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_combined, y_curr)

        importances = pd.Series(
            model.feature_importances_, index=X_combined.columns
        )

        shadow_cols = [c for c in X_combined.columns if c.startswith('_shadow_')]
        shadow_threshold = np.percentile(importances[shadow_cols].values, 75)

        for i, feat in enumerate(feature_names):
            if importances[feat] > shadow_threshold:
                hits[i] += 1

    confirmed = []
    tentative = []

    for i, feat in enumerate(feature_names):
        p_val = binomtest(
            hits[i], n=n_runs, p=0.25, alternative='greater'
        ).pvalue
        if p_val < alpha:
            confirmed.append(feat)
        elif p_val < alpha * 5:
            tentative.append(feat)

    logger.info(f"   Boruta: {len(confirmed)} confirmed, {len(tentative)} tentative "
                f"(hit rates: min={hits.min()}/{n_runs}, max={hits.max()}/{n_runs})")

    return confirmed + tentative


# =============================================================================
# Stage 3: Vintage Stability
# =============================================================================

def get_vintage_stability(feature_list, target_series, snapshots_dir,
                          snap_latest_wide, min_presence=2):
    """
    Check feature importance stability across historical snapshots.
    Exponential recency weighting: {2010:1, 2014:2, 2018:4, 2022:8, Latest:16}.
    """
    years = ['2010', '2014', '2018', '2022', 'Latest']
    weights = {'2010': 1, '2014': 2, '2018': 4, '2022': 8, 'Latest': 16}

    scores = pd.DataFrame(np.nan, index=feature_list, columns=years)

    for year in years:
        if year == 'Latest':
            X_wide = _select_wide(snap_latest_wide, feature_list)
        else:
            decade = year[:3] + "0s"
            path = snapshots_dir / decade / year / f"{year}-12.parquet"
            if not path.exists():
                continue
            X_wide = load_snapshot_wide(path, feature_filter=feature_list)

        if X_wide.empty:
            continue

        X_wide = winsorize_covid_period(X_wide)

        common = X_wide.index.intersection(target_series.index)
        if len(common) < 50:
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_wide.loc[common], target_series.loc[common])

        imp = pd.Series(model.feature_importances_, index=X_wide.columns)
        if imp.sum() > 0:
            imp = imp / imp.sum()
            for feat in imp.index:
                if feat in scores.index:
                    scores.loc[feat, year] = imp[feat]

    weight_series = pd.Series(weights)

    results = {}
    for feat in feature_list:
        feat_scores = scores.loc[feat].dropna()
        if len(feat_scores) == 0:
            continue

        nonzero_count = (feat_scores > 0).sum()
        if nonzero_count < min_presence:
            continue

        available_weights = weight_series[feat_scores.index]
        weighted_score = ((feat_scores * available_weights).sum()
                          / available_weights.sum())
        results[feat] = weighted_score

    result_series = pd.Series(results)

    latest_scores = scores['Latest'].dropna()
    latest_nonzero = latest_scores[latest_scores > 0].index
    result_series = result_series[result_series.index.isin(latest_nonzero)]

    return result_series.sort_values(ascending=False)


# =============================================================================
# Stage 4: Cluster Redundancy
# =============================================================================

def cluster_redundancy(X, feature_list, target_series,
                       max_clusters=50, min_overlap=30):
    """
    Hierarchical clustering to remove redundant features.
    NaN-aware pairwise Spearman correlations.
    """
    if len(feature_list) <= max_clusters:
        return feature_list

    X_curr = X[feature_list].copy()
    common = X_curr.index.intersection(target_series.index)
    X_curr = X_curr.loc[common]
    y_curr = target_series.loc[common]

    valid_target = y_curr.notna()
    X_curr = X_curr.loc[valid_target]
    y_curr = y_curr.loc[valid_target]

    corr = X_curr.corr(method='spearman')
    corr = corr.fillna(0)

    for i, col_i in enumerate(feature_list):
        for j, col_j in enumerate(feature_list):
            if i >= j:
                continue
            overlap = (X_curr[col_i].notna() & X_curr[col_j].notna()).sum()
            if overlap < min_overlap:
                corr.loc[col_i, col_j] = 0
                corr.loc[col_j, col_i] = 0

    dist_matrix = 1 - corr.abs().values
    np.fill_diagonal(dist_matrix, 0)

    dist_matrix = np.maximum(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    condensed = squareform(dist_matrix, checks=False)

    linkage = hierarchy.ward(condensed)
    cluster_ids = hierarchy.fcluster(linkage, t=max_clusters, criterion='maxclust')

    X_imputed = X_curr.fillna(X_curr.median())
    mi = mutual_info_regression(X_imputed, y_curr, random_state=SEED)
    mi_series = pd.Series(mi, index=feature_list)

    selected = []
    for cluster_id in np.unique(cluster_ids):
        members = [feature_list[i]
                   for i, c in enumerate(cluster_ids) if c == cluster_id]
        best = mi_series[members].idxmax()
        selected.append(best)

    return selected


# =============================================================================
# Stage 5: Interaction Rescue (Two-Phase)
# =============================================================================

def _extract_split_pairs(model, feature_names):
    """Extract parent-child split pairs from LightGBM trees."""
    pair_counts = Counter()

    booster = model.booster_
    for tree_idx in range(booster.num_trees()):
        tree_info = booster.dump_model()['tree_info'][tree_idx]

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
                       n_splits=8, gap=3, top_k=10):
    """
    Two-phase interaction rescue:
    Phase 1: Single-feature conditional test
    Phase 2: Split-pair detection from LightGBM trees
    """
    common = X.index.intersection(y.index)
    X_curr = X.loc[common]
    y_curr = y.loc[common]

    rejected = [f for f in rejected_pool
                if f in X_curr.columns and f not in confirmed_features]

    if not rejected:
        return []

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def _cv_mae(feature_set):
        X_sub = X_curr[feature_set]
        maes = []
        for train_idx, test_idx in tscv.split(X_sub):
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_sub.iloc[train_idx], y_curr.iloc[train_idx])
            preds = model.predict(X_sub.iloc[test_idx])
            maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
        return np.mean(maes)

    baseline_mae = _cv_mae(confirmed_features)

    # === Phase 1: Single-feature conditional rescue ===
    single_improvements = {}
    for feat in rejected:
        trial_mae = _cv_mae(confirmed_features + [feat])
        improvement = (baseline_mae - trial_mae) / (baseline_mae + 1e-12)
        if improvement > 0:
            single_improvements[feat] = improvement

    single_rescued = sorted(
        single_improvements.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    for feat, imp in single_rescued:
        logger.info(f"   Rescued (single): {feat} "
                     f"(delta MAE = {imp:.4f})")

    # === Phase 2: Split-pair detection ===
    all_feats = [f for f in X_curr.columns
                 if f in confirmed_features or f in rejected]

    model_full = lgb.LGBMRegressor(
        **{**LGB_PARAMS, 'n_estimators': 200, 'num_leaves': 63, 'max_depth': 6}
    )
    model_full.fit(X_curr[all_feats], y_curr)

    pair_counts = _extract_split_pairs(model_full, all_feats)

    rejected_set = set(rejected)
    pair_rescued = []

    for (feat_a, feat_b), count in pair_counts.most_common():
        if len(pair_rescued) >= top_k:
            break

        new_feats = []
        if feat_a in rejected_set and feat_a not in [f for f, _ in single_rescued]:
            new_feats.append(feat_a)
        if feat_b in rejected_set and feat_b not in [f for f, _ in single_rescued]:
            new_feats.append(feat_b)

        if not new_feats:
            continue

        trial_feats = (confirmed_features
                       + [f for f, _ in single_rescued]
                       + new_feats)
        trial_feats = list(dict.fromkeys(trial_feats))

        baseline_with_singles = (confirmed_features
                                 + [f for f, _ in single_rescued])
        baseline_with_singles = list(dict.fromkeys(baseline_with_singles))
        baseline_singles_mae = _cv_mae(baseline_with_singles)
        trial_mae = _cv_mae(trial_feats)

        improvement = ((baseline_singles_mae - trial_mae)
                       / (baseline_singles_mae + 1e-12))
        if improvement > 0:
            for f in new_feats:
                if f not in [pf for pf, _ in pair_rescued]:
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

    return result


# =============================================================================
# Stage 6: Sequential Forward Selection
# =============================================================================

def sequential_forward_selection(X, y, candidate_features, boruta_hits=None,
                                 n_splits=8, gap=3, min_improvement=0.001,
                                 patience=3, min_features=3):
    """Sequential Forward Selection using walk-forward CV."""
    common = X.index.intersection(y.index)
    X_curr = X.loc[common]
    y_curr = y.loc[common]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def _cv_mae(feature_set):
        if not feature_set:
            return float('inf')

        X_sub = X_curr[feature_set]
        maes = []
        for train_idx, test_idx in tscv.split(X_sub):
            X_train = X_sub.iloc[train_idx]
            X_test = X_sub.iloc[test_idx]
            y_train = y_curr.iloc[train_idx]
            y_test = y_curr.iloc[test_idx]

            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(np.mean(np.abs(y_test.values - preds)))

        return np.mean(maes)

    if boruta_hits is not None:
        candidates = sorted(
            candidate_features,
            key=lambda f: boruta_hits.get(f, 0), reverse=True
        )
    else:
        candidates = list(candidate_features)

    # Start with best single feature
    best_single_mae = float('inf')
    best_single = candidates[0]
    for feat in candidates[:min(20, len(candidates))]:
        mae = _cv_mae([feat])
        if mae < best_single_mae:
            best_single_mae = mae
            best_single = feat

    selected = [best_single]
    current_mae = best_single_mae
    remaining = [f for f in candidates if f != best_single]
    consecutive_failures = 0

    logger.info(f"   SFS start: {best_single} (MAE={current_mae:.2f})")

    while remaining:
        best_next = None
        best_next_mae = current_mae

        for feat in remaining:
            trial = selected + [feat]
            mae = _cv_mae(trial)
            if mae < best_next_mae:
                best_next_mae = mae
                best_next = feat

        if best_next is None:
            consecutive_failures += 1
            if (len(selected) >= min_features
                    and consecutive_failures >= patience):
                logger.info(f"   SFS stop: no improvement for {patience} rounds")
                break
            forced = remaining[0]
            selected.append(forced)
            remaining.remove(forced)
            logger.info(f"   SFS +{forced} (forced, "
                        f"MAE~{current_mae:.2f}, "
                        f"below min_features={min_features})")
            continue

        improvement = ((current_mae - best_next_mae)
                       / (current_mae + 1e-12))

        if improvement < min_improvement:
            consecutive_failures += 1
            if (len(selected) >= min_features
                    and consecutive_failures >= patience):
                logger.info(f"   SFS stop: {patience} consecutive improvements "
                            f"< {min_improvement:.4f}")
                break
            selected.append(best_next)
            remaining.remove(best_next)
            current_mae = best_next_mae
            logger.info(f"   SFS +{best_next} (MAE={current_mae:.2f}, "
                        f"delta={improvement:.4f}, "
                        f"patience {consecutive_failures}/{patience})")
        else:
            consecutive_failures = 0
            selected.append(best_next)
            remaining.remove(best_next)
            current_mae = best_next_mae
            logger.info(f"   SFS +{best_next} (MAE={current_mae:.2f}, "
                        f"delta={improvement:.4f})")

    logger.info(f"   SFS final: {len(selected)} features, MAE={current_mae:.2f}")
    return selected


# =============================================================================
# Full Pipeline Orchestrator
# =============================================================================

def run_pipeline(snap_wide, y_target, source_name, snapshots_dir, series_groups):
    """Execute stages 1-6 for a single target on one source."""
    y_target = winsorize_covid_period(y_target)

    # ===== Stage 1: Group-wise Dual Filter (Linear + LightGBM) =====
    stage1_candidates = []
    logger.info("1. Group-wise Dual Filter "
                "(Purged Corr + LightGBM screen + VIF)...")
    for group_name, series_list in series_groups.items():
        wide_group = _select_wide(snap_wide, series_list)
        if wide_group.empty:
            continue
        wide_group = winsorize_covid_period(wide_group)

        selected = filter_group_data_purged(
            wide_group, y_target, group_name
        )
        stage1_candidates.extend(selected)

    stage1_candidates = sorted(list(set(stage1_candidates)))
    logger.info(f"   > Stage 1 Survivors: {len(stage1_candidates)} features")

    if len(stage1_candidates) == 0:
        logger.info("   WARNING: No features survived Stage 1. Skipping pipeline.")
        return []

    # ===== Stage 2: Boruta Importance =====
    logger.info("2. Boruta Feature Selection (shadow features)...")
    X_stage2 = _select_wide(snap_wide, stage1_candidates)
    X_stage2 = winsorize_covid_period(X_stage2)

    stage2_candidates = get_boruta_importance(
        X_stage2, y_target, n_runs=100
    )
    logger.info(f"   > Stage 2 Survivors: {len(stage2_candidates)} features")

    if len(stage2_candidates) == 0:
        logger.info("   WARNING: No features survived Boruta. "
                     "Falling back to Stage 1 top 30.")
        common = X_stage2.index.intersection(y_target.index)
        corrs = X_stage2.loc[common].corrwith(
            y_target.loc[common]
        ).abs()
        stage2_candidates = corrs.nlargest(30).index.tolist()

    # ===== Stage 3: Vintage Stability =====
    logger.info("3. Vintage Stability "
                "(exponential recency weights, min 2 snapshots)...")
    weighted_scores = get_vintage_stability(
        stage2_candidates, y_target, snapshots_dir, snap_wide
    )

    if len(weighted_scores) == 0:
        logger.info("   WARNING: No features survived vintage stability. "
                     "Using all Stage 2.")
        stage3_candidates = stage2_candidates
    else:
        threshold = weighted_scores.median() * 0.5
        stage3_candidates = weighted_scores[
            weighted_scores > threshold
        ].index.tolist()
    logger.info(f"   > Stage 3 Survivors: {len(stage3_candidates)} features")

    # ===== Stage 4: Cluster Redundancy =====
    logger.info("4. Cluster Redundancy Check (NaN-aware Spearman)...")
    stage4_candidates = cluster_redundancy(
        X_stage2, stage3_candidates, y_target, max_clusters=50
    )
    logger.info(f"   > Stage 4 Survivors: {len(stage4_candidates)} features")

    # ===== Stage 5: Interaction Rescue (two-phase) =====
    logger.info("5. Interaction Rescue "
                "(single-feature + split-pair detection)...")
    rescued = interaction_rescue(
        X_stage2, y_target,
        confirmed_features=stage4_candidates,
        rejected_pool=stage1_candidates,
        top_k=10
    )
    stage5_candidates = stage4_candidates + rescued
    seen = set()
    stage5_candidates = [
        f for f in stage5_candidates
        if f not in seen and not seen.add(f)
    ]
    logger.info(f"   > Stage 5 Candidates: {len(stage5_candidates)} "
                f"({len(stage4_candidates)} confirmed "
                f"+ {len(rescued)} rescued)")

    # ===== Stage 6: Sequential Forward Selection =====
    logger.info("6. Sequential Forward Selection "
                "(walk-forward CV with embargo)...")
    final_list = sequential_forward_selection(
        X_stage2, y_target, stage5_candidates,
        n_splits=8, gap=3,
        min_improvement=0.001,
        patience=3,
        min_features=3,
    )
    logger.info(f"   > Final Count: {len(final_list)}")

    return final_list


def run_full_source_pipeline(snap_wide, target_mom, target_acc,
                             source_name, snapshots_dir, series_groups):
    """
    Run the 7-stage pipeline on a single source for both MoM and Acc targets.
    Returns the union of MoM + Acc selected features (Stage 7).
    """
    targets = {
        'MoM': target_mom,
        'Acc': target_acc,
    }

    final_results = {}
    for t_name, y_target in targets.items():
        y_clean = y_target.dropna()
        if len(y_clean) < 50:
            logger.warning(f"   [{t_name}] Insufficient target data "
                           f"({len(y_clean)} obs). Skipping.")
            final_results[t_name] = []
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  PIPELINE START: {source_name} / {t_name}")
        logger.info(f"{'=' * 60}")

        final_results[t_name] = run_pipeline(
            snap_wide, y_clean, source_name, snapshots_dir, series_groups
        )

    # ===== Stage 7: Union MoM + Acc =====
    mom_feats = set(final_results.get('MoM', []))
    acc_feats = set(final_results.get('Acc', []))
    union_feats = sorted(mom_feats | acc_feats)

    logger.info(f"\n  [{source_name}] Stage 7 Union: "
                f"{len(mom_feats)} MoM + {len(acc_feats)} Acc "
                f"-> {len(union_feats)} union "
                f"({len(mom_feats & acc_feats)} overlap)")

    return union_feats
