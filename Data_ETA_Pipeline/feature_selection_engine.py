from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import binomtest
from scipy.cluster import hierarchy
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
import re
from tqdm import tqdm
from collections import defaultdict

from utils.transforms import winsorize_covid_period

SEED = 42

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
# Helper Functions
# =============================================================================

def _sanitize_col(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

def _sanitize_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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

def _lgb_fit(X: pd.DataFrame, y: pd.Series, params: dict = None) -> tuple[lgb.LGBMRegressor, dict]:
    if params is None: params = LGB_PARAMS
    X_clean, mapping = _sanitize_df(X)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_clean, y)
    return model, mapping

def _lgb_importances(model: lgb.LGBMRegressor, mapping: dict) -> pd.Series:
    clean_names = model.feature_name_
    imp_vals = model.feature_importances_
    return pd.Series({mapping.get(c, c): v for c, v in zip(clean_names, imp_vals)})


# =============================================================================
# 7-Stage Pipeline Components
# =============================================================================

def _stage1_group_dual_filter(X: pd.DataFrame, y: pd.Series, source_name: str) -> list[str]:
    """Segment features into API-specific chunks, screen each chunk with LightGBM, and compile survivors."""
    common = X.index.intersection(y.index)
    X_c, y_c = X.loc[common], y.loc[common]
    
    # 1. Map columns to groups
    groups = defaultdict(list)
    for col in X_c.columns:
        grp = _classify_series(col, source_name)
        groups[grp].append(col)
        
    # 2. Process groups independently
    survivors = []
    for grp, cols in groups.items():
        X_grp = X_c[cols]
        if len(X_grp) < 50:
            continue
            
        model, mapping = _lgb_fit(X_grp, y_c)
        imp = _lgb_importances(model, mapping)
        top = imp[imp > 0].sort_values(ascending=False).index.tolist()
        survivors.extend(top[:min(50, len(top))])
        
    return list(set(survivors))


def _stage2_boruta(X: pd.DataFrame, y: pd.Series, candidate_features: list[str], n_runs: int = 100, alpha: float = 0.05) -> list[str]:
    common = X.index.intersection(y.index)
    X_c = X.loc[common][candidate_features]
    y_c = y.loc[common]

    hits = np.zeros(len(candidate_features), dtype=int)

    for run in range(n_runs):
        rng = np.random.RandomState(SEED + run)
        shadow_data = {}
        for col in candidate_features:
            col_vals = X_c[col].values.copy()
            valid_mask = ~np.isnan(col_vals) if np.issubdtype(col_vals.dtype, np.floating) else np.ones(len(col_vals), dtype=bool)
            valid_vals = col_vals[valid_mask].copy()
            rng.shuffle(valid_vals)
            col_vals[valid_mask] = valid_vals
            shadow_data[f'_shadow_{col}'] = col_vals

        shadow_df = pd.DataFrame(shadow_data, index=X_c.index)
        X_combined = pd.concat([X_c, shadow_df], axis=1)

        model, mapping = _lgb_fit(X_combined, y_c)
        importances = _lgb_importances(model, mapping)

        shadow_cols = [c for c in importances.index if c.startswith('_shadow_')]
        shadow_threshold = np.percentile(importances[shadow_cols].values, 75) if shadow_cols else 0

        for i, feat in enumerate(candidate_features):
            if importances.get(feat, 0) > shadow_threshold:
                hits[i] += 1

    confirmed = []
    tentative = []
    for i, feat in enumerate(candidate_features):
        p_val = binomtest(hits[i], n=n_runs, p=0.25, alternative='greater').pvalue
        if p_val < alpha:
            confirmed.append(feat)
        elif p_val < alpha * 5:
            tentative.append(feat)

    return confirmed + tentative


def _stage3_vintage_stability(X: pd.DataFrame, y: pd.Series, feature_list: list[str], min_presence: int = 2) -> list[str]:
    years = ['2010', '2014', '2018', '2022', 'Latest']
    weights = {'2010': 1, '2014': 2, '2018': 4, '2022': 8, 'Latest': 16}
    scores = pd.DataFrame(np.nan, index=feature_list, columns=years)

    for year in years:
        cutoff = X.index.max() if year == 'Latest' else pd.to_datetime(f"{year}-12-01")
        X_vintage = X[X.index <= cutoff].copy()
        y_vintage = y[y.index <= cutoff].copy()

        if len(X_vintage) < 50: continue

        X_vintage = winsorize_covid_period(X_vintage)
        common = X_vintage.index.intersection(y_vintage.index)

        if len(common) < 50: continue

        model, mapping = _lgb_fit(X_vintage.loc[common][feature_list], y_vintage.loc[common])
        imp = _lgb_importances(model, mapping)
        
        if imp.sum() > 0:
            imp = imp / imp.sum()
            for feat in imp.index:
                if feat in scores.index:
                    scores.loc[feat, year] = imp[feat]

    weight_series = pd.Series(weights)
    results = {}
    
    for feat in feature_list:
        feat_scores = scores.loc[feat].dropna()
        if len(feat_scores) == 0: continue
        if (feat_scores > 0).sum() < min_presence: continue

        available_weights = weight_series[feat_scores.index]
        weighted_score = (feat_scores * available_weights).sum() / available_weights.sum()
        results[feat] = weighted_score

    result_series = pd.Series(results)
    latest_scores = scores['Latest'].dropna()
    latest_nonzero = latest_scores[latest_scores > 0].index
    filtered = result_series[result_series.index.isin(latest_nonzero)]

    if len(filtered) == 0: return feature_list
    threshold = filtered.median() * 0.5
    return filtered[filtered > threshold].index.tolist()


def _stage4_cluster_redundancy(X: pd.DataFrame, y: pd.Series, feature_list: list[str], max_clusters: int = 50, min_overlap: int = 30) -> list[str]:
    if len(feature_list) <= max_clusters: return feature_list

    X_curr = X[feature_list]
    common = X_curr.index.intersection(y.index)
    X_curr, y_curr = X_curr.loc[common], y.loc[common]

    valid_target = y_curr.notna()
    X_curr, y_curr = X_curr.loc[valid_target], y_curr.loc[valid_target]

    corr = X_curr.corr(method='spearman').fillna(0)

    for i, col_i in enumerate(feature_list):
        for j, col_j in enumerate(feature_list):
            if i >= j: continue
            overlap = (X_curr[col_i].notna() & X_curr[col_j].notna()).sum()
            if overlap < min_overlap:
                corr.loc[col_i, col_j] = 0
                corr.loc[col_j, col_i] = 0

    dist_matrix = 1 - corr.abs().values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)

    linkage = hierarchy.ward(condensed)
    cluster_ids = hierarchy.fcluster(linkage, t=max_clusters, criterion='maxclust')

    X_imputed = X_curr.fillna(X_curr.median())
    mi = mutual_info_regression(X_imputed.fillna(0), y_curr, random_state=SEED)
    mi_series = pd.Series(mi, index=feature_list)

    selected = []
    for cluster_id in np.unique(cluster_ids):
        members = [feature_list[i] for i, c in enumerate(cluster_ids) if c == cluster_id]
        best = mi_series[members].idxmax()
        selected.append(best)

    return selected


def _stage5_interaction_rescue(X: pd.DataFrame, y: pd.Series, confirmed: list[str], rejected_pool: list[str], n_splits: int = 4, top_k: int = 5) -> list[str]:
    common = X.index.intersection(y.index)
    X_curr, y_curr = X.loc[common], y.loc[common]

    rejected = [f for f in rejected_pool if f in X_curr.columns and f not in confirmed]
    if not rejected: return []

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=3)

    def _cv_mae(feature_set):
        if not feature_set: return float('inf')
        X_sub = X_curr[feature_set]
        maes = []
        for train_idx, test_idx in tscv.split(X_sub):
            model, mapping = _lgb_fit(X_sub.iloc[train_idx], y_curr.iloc[train_idx])
            preds = _lgb_predict(model, X_sub.iloc[test_idx], mapping)
            maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
        return np.mean(maes)

    baseline_mae = _cv_mae(confirmed)

    single_imp = {}
    for feat in rejected:
        mae = _cv_mae(confirmed + [feat])
        imp = (baseline_mae - mae) / (baseline_mae + 1e-12)
        if imp > 0:
            single_imp[feat] = imp

    return [feat for feat, _ in sorted(single_imp.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def _stage6_sfs(X: pd.DataFrame, y: pd.Series, candidates: list[str], min_improvement: float = 0.001, patience: int = 3, min_features: int = 5) -> list[str]:
    if not candidates: return []
        
    common = X.index.intersection(y.index)
    X_curr, y_curr = X.loc[common], y.loc[common]

    tscv = TimeSeriesSplit(n_splits=4, gap=3)

    def _cv_mae(feature_set):
        if not feature_set: return float('inf')
        X_sub = X_curr[feature_set]
        maes = []
        for train_idx, test_idx in tscv.split(X_sub):
            model, mapping = _lgb_fit(X_sub.iloc[train_idx], y_curr.iloc[train_idx])
            preds = _lgb_predict(model, X_sub.iloc[test_idx], mapping)
            maes.append(np.mean(np.abs(y_curr.iloc[test_idx].values - preds)))
        return np.mean(maes)

    best_single_mae = float('inf')
    best_single = candidates[0]
    for feat in candidates[:min(10, len(candidates))]:
        mae = _cv_mae([feat])
        if mae < best_single_mae:
            best_single_mae = mae
            best_single = feat

    selected = [best_single]
    current_mae = best_single_mae
    remaining = [f for f in candidates if f != best_single]
    fails = 0

    while remaining:
        best_next = None
        best_next_mae = current_mae

        for feat in remaining:
            trial = selected + [feat]
            mae = _cv_mae(trial)
            if mae < best_next_mae:
                best_next_mae, best_next = mae, feat

        if best_next is None:
            fails += 1
            if len(selected) >= min_features and fails >= patience: break
            forced = remaining[0]
            selected.append(forced)
            remaining.remove(forced)
            continue

        improvement = (current_mae - best_next_mae) / (current_mae + 1e-12)
        if improvement < min_improvement:
            fails += 1
            if len(selected) >= min_features and fails >= patience: break
        else:
            fails = 0
            
        selected.append(best_next)
        remaining.remove(best_next)
        current_mae = best_next_mae

    return selected


def run_pipeline(X: pd.DataFrame, y: pd.Series, source_name: str) -> list[str]:
    """Execute the full 7-stage engine on a given source matrix and target."""
    common = X.index.intersection(y.index)
    X_c, y_c = X.loc[common], y.loc[common]
    
    stage1 = _stage1_group_dual_filter(X_c, y_c, source_name)
    if not stage1: return []
        
    stage2 = _stage2_boruta(X_c, y_c, stage1, n_runs=50)
    if not stage2: stage2 = stage1[:20] 
        
    stage3 = _stage3_vintage_stability(X_c, y_c, stage2, min_presence=1)
    if not stage3: stage3 = stage2
        
    stage4 = _stage4_cluster_redundancy(X_c, y_c, stage3, max_clusters=min(30, max(10, len(stage3))))
    
    stage5_rescue = _stage5_interaction_rescue(X_c, y_c, stage4, stage1, top_k=5)
    stage5 = list(dict.fromkeys(stage4 + stage5_rescue))
    
    stage6 = _stage6_sfs(X_c, y_c, stage5, patience=2, min_features=3)
    
    return stage6
