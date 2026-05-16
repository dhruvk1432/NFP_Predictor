"""
LightGBM NFP Prediction Model

Predicts NSA NFP month-on-month change using master snapshots.
Uses release_date cutoff (not target_month) to match inference behavior.

Key Features:
- Handles data with varying start dates (some series start 1948, others 2008)
- Uses past target data (NFP levels, MoM changes) as features
- Survey interval features (4 vs 5 weeks logic)
- Momentum/divergence and acceleration features
- Cyclical month encoding (sin/cos)

MODULAR ARCHITECTURE:
This file is the main entry point. Core functionality is split into:
- Train/config.py: Configuration constants and hyperparameters
- Train/data_loader.py: Data loading functions
- Train/feature_engineering.py: Feature creation
- Train/model.py: Model training and prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import json
import random
import warnings

# ── Global determinism ──
# Pin every global RNG before anything else imports numpy/random under us.
# LightGBM's own seeds are set per-param-dict (see Train.config.LGBM_DETERMINISM);
# this block covers numpy, the stdlib `random` module, and the Python hash seed
# (which only takes effect for child processes — informative for joblib workers).
_GLOBAL_RNG_SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(_GLOBAL_RNG_SEED))
random.seed(_GLOBAL_RNG_SEED)
np.random.seed(_GLOBAL_RNG_SEED)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import (
    DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger, BACKTEST_MONTHS,
    RESELECT_EVERY_N_MONTHS, USE_PER_WINDOW_FEATURES,
)
from Data_ETA_Pipeline.perf_stats import profiled, perf_phase, inc_counter, dump_perf_json, register_atexit_dump
from experiments.sidecars.integration import attach_sidecar_features

# Import from modular components
from Train.config import (
    MODEL_SAVE_DIR,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    get_model_id,
    get_target_path,
    load_selected_features,
    USE_HUBER_LOSS_DEFAULT,
    HUBER_DELTA,
    TUNE_EVERY_N_MONTHS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    LGBM_DETERMINISM,
)

from Train.data_loader import (
    load_master_snapshot,
    load_target_data,
    get_lagged_target_features,
    batch_lagged_target_features,
    pivot_snapshot_to_wide,
)

from joblib import Parallel, delayed

def _process_single_month_task(
    target_month: pd.Timestamp,
    target_value: float,
    cutoff_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
    target_type: str,
    release_type: str,
    target_source: str,
    target_lags_lookup: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """
    Helper function to process the feature engineering for a single month in parallel.
    Disables caching to avoid memory bloat in worker processes.

    Loads from the pre-merged, feature-selected master snapshot which already contains
    ALL data sources (FRED employment + exogenous). No separate FRED loading needed.

    This function sequentially:
    1. Extracts calendar metrics (survey weeks, seasonality).
    2. Collects historical branch-target lag/trend/regime features.
    3. Pivots the wide-format Master Snapshot for just this date cutoff.
    4. Computes short-term data revision metrics by comparing master[M] vs master[M-1].

    Args:
        target_month: The month being predicted.
        target_value: The actual target value (used simply for tracking).
        cutoff_date: The strict barrier preventing future data leakage (usually release date).
        snapshot_date: The date of the data snapshot being used.
        target_type: 'nsa' or 'sa'.
        release_type: 'first' or 'last'.
        target_source: 'revised'.
        target_lags_lookup: Pre-computed dictionary of branch-target lags.

    Returns:
        Tuple of (feature dict, target value). None if snapshot missing.
    """
    # Initialize dictionary for features
    features = {'ds': target_month}

    # 1. Add Calendar Features
    cal_features = get_calendar_features_dict(target_month)
    features.update(cal_features)

    # 2. Add branch-target lagged features — O(1) dict lookup
    features.update(target_lags_lookup.get(target_month, {}))

    # 3. Load pre-merged, pre-selected master snapshot (contains ALL sources)
    snapshot_df = load_master_snapshot(snapshot_date, target_type=target_type,
                                       target_source=target_source, use_cache=False)

    if snapshot_df is None or snapshot_df.empty:
        return None, None

    features_wide = pivot_snapshot_to_wide(snapshot_df, target_month, cutoff_date=cutoff_date)
    if not features_wide.empty:
        features.update(features_wide.iloc[0].to_dict())

    # 4. Compute Revisions: compare master[M] vs master[M-1] for the PREVIOUS month
    # NFP_PERF_SKIP_REVISIONS=1 skips this block during profiling-only runs
    # (useful when master snapshots contain non-numeric columns that trigger type errors)
    _skip_revisions = os.getenv("NFP_PERF_SKIP_REVISIONS", "").strip() == "1"
    if not _skip_revisions:
        prev_month_target = target_month - pd.DateOffset(months=1)
        prev_snapshot_date = prev_month_target + pd.offsets.MonthEnd(0)

        prev_snapshot = load_master_snapshot(prev_snapshot_date, target_type=target_type,
                                             target_source=target_source, use_cache=False)

        if prev_snapshot is not None and not prev_snapshot.empty:
            view_curr = pivot_snapshot_to_wide(snapshot_df, prev_month_target, cutoff_date=cutoff_date)
            view_prev = pivot_snapshot_to_wide(prev_snapshot, prev_month_target, cutoff_date=cutoff_date)
            revs = compute_revision_features(view_curr, view_prev, prefix='rev_master')
            if not revs.empty:
                features.update(revs.iloc[0].to_dict())

    return features, target_value


from Train.feature_engineering import (
    add_calendar_features,
    get_calendar_features_dict,
)

from Train.revision_features import (
    get_revision_features_for_month,
    compute_revision_features,
)
from utils.transforms import winsorize_covid_period
from Train.hyperparameter_tuning import tune_hyperparameters

from Train.config import (
    N_OPTUNA_TRIALS, OPTUNA_TIMEOUT,
    USE_UNION_POOL, UNION_POOL_MAX, SHORTPASS_TOPK, SHORTPASS_METHOD,
    SHORTPASS_HALF_LIFE, ENABLE_BASELINE_TRACKING, BASELINE_ROLLING_WINDOW,
    KEEP_RULE_ENABLED, KEEP_RULE_WINDOW_M, KEEP_RULE_TOLERANCE, KEEP_RULE_ACTION,
    USE_BRANCH_TARGET_FS, BRANCH_TARGET_FS_TOPK, BRANCH_TARGET_FS_TOPK_VARIANCE,
    BRANCH_TARGET_FS_METHOD, BRANCH_TARGET_FS_METHOD_VARIANCE,
    BRANCH_TARGET_FS_CORR_THRESHOLD, BRANCH_TARGET_FS_MIN_OVERLAP,
    BRANCH_TARGET_FS_WEIGHT_LEVEL, BRANCH_TARGET_FS_WEIGHT_DIFF, BRANCH_TARGET_FS_WEIGHT_DIR,
    BRANCH_TARGET_FS_WEIGHT_AMP, BRANCH_TARGET_FS_WEIGHT_SIGN, BRANCH_TARGET_FS_WEIGHT_TAIL,
    VARIANCE_PRIORITY_TARGETS, VARIANCE_TAIL_QUANTILE, VARIANCE_EXTREME_QUANTILE,
    ENABLE_VARIANCE_GATE, VARIANCE_GATE_MIN_STD_RATIO, VARIANCE_GATE_MIN_DIFF_STD_RATIO,
    VARIANCE_GATE_MIN_CORR_DIFF, VARIANCE_GATE_MIN_DIFF_SIGN_ACC, VARIANCE_GATE_MIN_EXTREME_HIT_RATE,
    TUNING_OBJECTIVE_MODE_DEFAULT, TUNING_OBJECTIVE_MODE_VARIANCE,
    NSA_TUNE_USE_KALMAN_FUSION, JOINT_OPTUNA,
    TUNING_LAMBDA_STD_RATIO, TUNING_LAMBDA_DIFF_STD_RATIO, TUNING_LAMBDA_TAIL_MAE,
    TUNING_LAMBDA_CORR_DIFF, TUNING_LAMBDA_DIFF_SIGN,
    ENABLE_VARIANCE_ENHANCEMENTS, ENHANCEMENT_SEQUENCE, ENHANCEMENT_MIN_IMPROVEMENT,
    ENABLE_AMPLITUDE_CALIBRATION, AMPLITUDE_CAL_MIN_SAMPLES,
    AMPLITUDE_CAL_SLOPE_MIN, AMPLITUDE_CAL_SLOPE_MAX,
    ENABLE_SHOCK_MODEL, SHOCK_MODEL_NUM_BOOST_ROUND, SHOCK_MODEL_MAX_DEPTH, SHOCK_MODEL_NUM_LEAVES,
    ENABLE_ACCELERATION_MODEL, ACCEL_MODEL_NUM_BOOST_ROUND, ACCEL_MODEL_MAX_DEPTH, ACCEL_MODEL_NUM_LEAVES,
    ENABLE_MULTI_TARGET_DYNAMICS, DYNAMICS_MODEL_NUM_BOOST_ROUND, DYNAMICS_MODEL_MAX_DEPTH,
    DYNAMICS_MODEL_NUM_LEAVES, DYNAMICS_DELTA_BLEND, DYNAMICS_DIRECTION_CONFIDENCE,
    DYNAMICS_DIRECTION_BLEND, DYNAMICS_MAGNITUDE_FLOOR,
    ENABLE_TAIL_AWARE_WEIGHTING, TAIL_WEIGHT_ABS_LEVEL_QUANTILE, TAIL_WEIGHT_ABS_DIFF_QUANTILE,
    TAIL_WEIGHT_LEVEL_BOOST, TAIL_WEIGHT_DIFF_BOOST, TAIL_WEIGHT_MAX_MULTIPLIER,
    ENABLE_REGIME_ROUTER, REGIME_HIGHVOL_QUANTILE, REGIME_MIN_CLASS_SAMPLES,
    REGIME_MODEL_NUM_BOOST_ROUND,
    SA_CALENDAR_FEATURES_KEEP,
    ENHANCEMENT_EXEMPT_TARGETS,
    RESELECTION_START_DATE,
)

from Train.branch_target_selection import (
    partition_feature_columns,
    select_branch_target_features_for_step,
)
from Train.variance_metrics import compute_variance_kpis, composite_objective_score

from Train.model import (
    get_lgbm_params,
    train_lightgbm_model,
    predict_with_intervals,
    save_model,
    load_model,
    calculate_sample_weights,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


def _feature_engineering_parallel_settings() -> Tuple[int, Optional[str]]:
    raw_jobs = os.getenv("NFP_TRAIN_FEATURE_N_JOBS", "").strip()
    if raw_jobs:
        try:
            n_jobs = int(raw_jobs)
        except ValueError:
            logger.warning(
                "Invalid NFP_TRAIN_FEATURE_N_JOBS=%r; using all processors",
                raw_jobs,
            )
            n_jobs = -1
    else:
        n_jobs = -1

    backend = os.getenv("NFP_TRAIN_FEATURE_BACKEND", "").strip() or None
    if backend and backend not in {"loky", "threading", "multiprocessing"}:
        logger.warning(
            "Invalid NFP_TRAIN_FEATURE_BACKEND=%r; using joblib default",
            backend,
        )
        backend = None
    return n_jobs, backend


def _coverage_protected_sources() -> set[str]:
    """Sources allowed to survive sparse-history cleanup with a lower floor.

    Futures and economist forecasts can be valuable in live inference even when
    local historical collection was spotty.  This guardrail keeps those already
    culled source families in the candidate universe without relaxing the rule
    globally.
    """
    raw = os.getenv(
        "NFP_TRAIN_COVERAGE_PROTECTED_SOURCES",
        "Futures,EconomistPanel",
    ).strip()
    if not raw or raw.lower() in {"off", "none", "false", "0"}:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _coverage_protected_min_non_nan() -> int:
    raw = os.getenv("NFP_TRAIN_COVERAGE_PROTECTED_MIN_NON_NAN", "48").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid NFP_TRAIN_COVERAGE_PROTECTED_MIN_NON_NAN=%r; using 48",
            raw,
        )
        return 48


def _coverage_protected_max_staleness_months() -> Optional[int]:
    raw = os.getenv(
        "NFP_TRAIN_COVERAGE_PROTECTED_MAX_STALENESS_MONTHS",
        "12",
    ).strip()
    if not raw or raw.lower() in {"off", "none", "false"}:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid NFP_TRAIN_COVERAGE_PROTECTED_MAX_STALENESS_MONTHS=%r; using 12",
            raw,
        )
        return 12


def clean_features(
    X: pd.DataFrame,
    y: pd.Series,
    min_non_nan: int = 100,
    nan_eval_start: Optional[str] = None,
    nan_max_rate: Optional[float] = None,
) -> List[str]:
    """
    Feature cleaning with optional post-2010 NaN evaluation.

    Two-stage filter:
    1. Global sparsity: drop columns with fewer than ``min_non_nan`` non-NaN values
       across the entire history (catches features with almost no data at all).
    2. Modern-era NaN rate: if ``nan_eval_start`` is set, evaluate NaN rate only
       from that date onward. Features with NaN rate > ``nan_max_rate`` in the
       modern window are dropped.  This allows features that start later (ADP from
       2001, Prosper from 2006) to survive as long as they have good coverage in
       the evaluation window.

    LightGBM handles NaN natively, so moderate NaN is fine — this filter only
    removes features that are too sparse to contribute signal.

    Args:
        X: Feature DataFrame (must contain a 'ds' datetime column).
        y: Target series (unused, kept for API consistency).
        min_non_nan: Minimum number of non-NaN values required globally.
        nan_eval_start: If set, evaluate NaN rate from this date onward
            (e.g. '2010-01-01').  Defaults to ``DYNAMIC_FS_NAN_EVAL_START``
            from config when called during dynamic reselection.
        nan_max_rate: Maximum acceptable NaN rate in the eval window.
            Defaults to ``DYNAMIC_FS_NAN_MAX_RATE`` from config.

    Returns:
        List of cleaned feature column names
    """
    from Train.config import DYNAMIC_FS_NAN_EVAL_START, DYNAMIC_FS_NAN_MAX_RATE

    # Profiling-only override: lower threshold when running with sparse data
    _perf_override = os.getenv("NFP_PERF_MIN_NON_NAN", "").strip()
    if _perf_override:
        try:
            min_non_nan = int(_perf_override)
        except ValueError:
            pass

    X_work = X.select_dtypes(include=[np.number]).copy()
    X_work = X_work.drop(columns=['ds'], errors='ignore')
    X_work = X_work.replace([np.inf, -np.inf], np.nan)

    protected_sources = _coverage_protected_sources()
    protected_min_non_nan = _coverage_protected_min_non_nan()
    protected_max_staleness = _coverage_protected_max_staleness_months()
    protected_cols: set[str] = set()
    if protected_sources:
        ds_values = (
            pd.to_datetime(X['ds'], errors='coerce')
            if 'ds' in X.columns
            else pd.Series(pd.NaT, index=X_work.index)
        )
        max_ds = ds_values.max()
        for col in X_work.columns:
            source = _coverage_source_for_column(col)
            if source not in protected_sources:
                continue
            observed = X_work[col].notna()
            observed_count = int(observed.sum())
            if observed_count < protected_min_non_nan:
                continue
            if protected_max_staleness is not None and pd.notna(max_ds) and 'ds' in X.columns:
                last_observed = ds_values.loc[observed].max()
                if pd.isna(last_observed):
                    continue
                months_stale = (
                    (max_ds.year - last_observed.year) * 12
                    + (max_ds.month - last_observed.month)
                )
                if months_stale > protected_max_staleness:
                    continue
            protected_cols.add(col)

    # Stage 1: Global sparsity filter (same as before)
    non_nan_counts = X_work.notna().sum()
    sparse_candidates = non_nan_counts[non_nan_counts < min_non_nan].index.tolist()
    sparse_cols = [col for col in sparse_candidates if col not in protected_cols]
    sparse_protected = len(sparse_candidates) - len(sparse_cols)
    X_work = X_work.drop(columns=sparse_cols)

    # Stage 2: Post-2010 NaN rate filter
    eval_start = nan_eval_start or DYNAMIC_FS_NAN_EVAL_START
    max_rate = nan_max_rate if nan_max_rate is not None else DYNAMIC_FS_NAN_MAX_RATE

    modern_nan_dropped = []
    if eval_start and 'ds' in X.columns:
        eval_ts = pd.Timestamp(eval_start)
        modern_mask = X['ds'] >= eval_ts
        n_modern = modern_mask.sum()

        if n_modern > 0:
            X_modern = X.loc[modern_mask, X_work.columns].replace([np.inf, -np.inf], np.nan)
            nan_rates = X_modern.isna().sum() / n_modern
            modern_nan_candidates = nan_rates[nan_rates > max_rate].index.tolist()
            modern_nan_dropped = [
                col for col in modern_nan_candidates if col not in protected_cols
            ]
            modern_nan_protected = len(modern_nan_candidates) - len(modern_nan_dropped)
            X_work = X_work.drop(columns=modern_nan_dropped)
        else:
            modern_nan_protected = 0
    else:
        modern_nan_protected = 0

    logger.info(
        f"Feature cleaning: dropped {len(sparse_cols)} sparse (<{min_non_nan} non-NaN), "
        f"{len(modern_nan_dropped)} high-NaN post-{eval_start}, "
        f"{len(X_work.columns)} remaining "
        f"(coverage-protected: {sparse_protected} sparse, {modern_nan_protected} high-NaN)"
    )

    return list(X_work.columns)


# =============================================================================
# DYNAMIC FEATURE RE-SELECTION (two-pass, per-source then global)
# =============================================================================

# ── Source classification patterns ──
# After sanitize_feature_name(), column prefixes that identify each data source.
# NSA employment: "total_*_nsa" (dot→underscore, _nsa suffix).
# SA employment: "total_*" without _nsa suffix.
# Exogenous FRED: known macro-series prefixes.
# Other exogenous: unique first-word prefixes.

_FRED_EXOG_PREFIXES = (
    'CCNSA_', 'CCSA_', 'Credit_', 'Financial_', 'Oil_', 'SP500_',
    'VIX_', 'Weekly_', 'Yield_', 'WEI_',
    # Binary regime features
    'regime_',
)
_UNIFIER_PREFIXES = (
    'AHE_', 'AWH_', 'CB_', 'Challenger_', 'Empire_', 'Housing_',
    'ISM_', 'Industrial_', 'NFP_Consensus', 'Retail_', 'UMich_',
)
_ADP_PREFIXES = ('ADP_', 'adp_')
_NOAA_PREFIXES = ('NOAA_', 'noaa_', 'storm_', 'hurricane_')
_PROSPER_PREFIXES = ('Consumer_Mood', 'Prosper_', 'Consumer_Spending')
_FUTURES_PREFIXES = (
    'Treasury_', 'FedFunds_', 'SOFR_', 'WTI_Crude_', 'NatGas_', 'Gold_',
    'Copper_', 'DollarIndex_', 'EuroFX_', 'YenFX_', 'SP500_Futures_',
)
_ECONOMIST_PREFIXES = ('NFP_Forecast_', 'Economist_', 'economist_')
_SA_NSA_GAP_PREFIXES = ('sanagap_',)
_DERIVED_CONTROL_PREFIXES = (
    'is_', 'month_', 'quarter_', 'year', 'weeks_since_', 'nfp_', 'rev_master_',
)


def _coverage_source_for_column(col: str) -> Optional[str]:
    """Return source labels used by sparse-history coverage protections."""
    if col.startswith(_FUTURES_PREFIXES):
        return 'Futures'
    if col.startswith(_ECONOMIST_PREFIXES):
        return 'EconomistPanel'
    return None


def _classify_columns_by_source(
    snapshot_cols: List[str],
) -> Dict[str, List[str]]:
    """Partition master-snapshot feature columns into source groups.

    The classifier uses column-name prefixes that are stable after
    ``sanitize_feature_name()`` is applied during snapshot generation.

    Returns:
        Dict mapping source name → list of column names. Columns that
        cannot be matched are placed in an ``'Unknown'`` bucket.
    """
    groups: Dict[str, List[str]] = {
        'FRED_Employment_NSA': [],
        'FRED_Employment_SA': [],
        'FRED_Exogenous': [],
        'Unifier': [],
        'ADP': [],
        'NOAA': [],
        'Prosper': [],
        'Futures': [],
        'EconomistPanel': [],
        'SA_NSA_Gap': [],
        'DerivedControls': [],
        'Unknown': [],
    }

    for col in snapshot_cols:
        if col.startswith('total_'):
            # Employment series start with "total_".
            # NSA columns have "_nsa" in the base name (before transform suffixes).
            if '_nsa' in col:
                groups['FRED_Employment_NSA'].append(col)
            else:
                groups['FRED_Employment_SA'].append(col)
        elif col.startswith(_FRED_EXOG_PREFIXES):
            groups['FRED_Exogenous'].append(col)
        elif col.startswith(_UNIFIER_PREFIXES):
            groups['Unifier'].append(col)
        elif col.startswith(_ADP_PREFIXES):
            groups['ADP'].append(col)
        elif col.startswith(_NOAA_PREFIXES):
            groups['NOAA'].append(col)
        elif col.startswith(_PROSPER_PREFIXES):
            groups['Prosper'].append(col)
        elif col.startswith(_FUTURES_PREFIXES):
            groups['Futures'].append(col)
        elif col.startswith(_ECONOMIST_PREFIXES):
            groups['EconomistPanel'].append(col)
        elif col.startswith(_SA_NSA_GAP_PREFIXES):
            groups['SA_NSA_Gap'].append(col)
        elif col.startswith(_DERIVED_CONTROL_PREFIXES):
            groups['DerivedControls'].append(col)
        else:
            groups['Unknown'].append(col)

    return groups


def _load_per_window_feature_sets(
    target_type: str,
    target_source: str,
) -> List[Tuple[pd.Timestamp, List[str]]]:
    """Load saved per-window feature JSONs for replay mode.

    Reads ``_output/dynamic_selection/{target_type}_{target_source}/*.json``
    and filters to the most recent run cohort (JSONs whose mtime is within
    6 hours of the latest). Returns a list of ``(step_date, features)``
    tuples sorted ascending by step_date.

    Used by USE_PER_WINDOW_FEATURES mode to reproduce the predictions of a
    prior dynamic-reselection backtest run without re-running the slow
    feature-selection stage.
    """
    sel_dir = OUTPUT_DIR / "dynamic_selection" / f"{target_type}_{target_source}"
    if not sel_dir.exists():
        return []

    json_paths = list(sel_dir.glob("*.json"))
    if not json_paths:
        return []

    mtimes = [(p, p.stat().st_mtime) for p in json_paths]
    latest_mtime = max(m for _, m in mtimes)
    cohort_window_seconds = 6 * 3600
    cohort = [p for p, m in mtimes if (latest_mtime - m) <= cohort_window_seconds]

    sets: List[Tuple[pd.Timestamp, List[str]]] = []
    for p in sorted(cohort):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            sets.append((pd.Timestamp(data["step_date"]), list(data["features"])))
        except Exception as e:
            logger.warning(f"_load_per_window_feature_sets: failed to load {p}: {e}")

    sets.sort(key=lambda x: x[0])

    # Stale-cohort warning. When users flip RESELECT_EVERY_N_MONTHS back on
    # without removing old JSONs, the 6-hour cohort filter above silently
    # snaps to the most-recent ad-hoc run, which can be months old. Surface
    # this explicitly so we don't quietly replay 2021-vintage features.
    import time as _time
    age_days = (_time.time() - latest_mtime) / 86400.0
    if age_days > 30:
        logger.warning(
            f"_load_per_window_feature_sets: latest cohort under {sel_dir} is "
            f"{age_days:.0f} days old. If you intended fresh reselection, "
            f"clear or back up this directory and set RESELECT_EVERY_N_MONTHS>0."
        )
    if len(sets) == 1:
        logger.warning(
            f"_load_per_window_feature_sets: per-window replay is using a "
            f"single feature set at step_date={sets[0][0].strftime('%Y-%m')} "
            f"for ALL backtest steps. Verify this is intentional."
        )

    return sets


def _get_features_for_step(
    per_window_sets: List[Tuple[pd.Timestamp, List[str]]],
    target_month: pd.Timestamp,
) -> Optional[List[str]]:
    """Return the feature list with the most recent step_date <= target_month."""
    chosen: Optional[List[str]] = None
    for step_date, features in per_window_sets:
        if step_date <= target_month:
            chosen = features
        else:
            break
    return chosen


def _save_joint_tuned_params(joint_result: Dict, step_date: pd.Timestamp) -> None:
    """Persist joint-Optuna params to joint_tuned_params.json.

    Format mirrors the existing ``tuned_params.json`` so downstream readers
    (``consensus_anchor_runner._tune_kalman`` skip-logic) can consume it
    without translation. ``step_date`` is logged so a stale file from a
    previous reselection is easy to spot.
    """
    import json as _json
    out_dir = OUTPUT_DIR / 'consensus_anchor' / 'kalman_fusion'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'source': 'joint_tuning',
        'step_date': pd.Timestamp(step_date).strftime('%Y-%m-%d'),
        'half_life_years': float(joint_result['half_life_years']),
        'trailing_window': int(joint_result['kalman_params']['trailing_window']),
        'nsa_weight_scale': float(joint_result['kalman_params']['nsa_weight_scale']),
        'best_score': float(joint_result.get('best_score', float('nan'))),
        'n_trials_run': int(joint_result.get('n_trials_run', 0)),
    }
    with open(out_dir / 'joint_tuned_params.json', 'w') as f:
        _json.dump(payload, f, indent=2)


def _build_fusion_tuning_context(target_dates: pd.Series) -> Optional[Dict]:
    """Build the fusion-CV context used by ``tune_hyperparameters`` when
    its ``objective_mode='kalman_fusion'``.

    Loads (once per call):
      * Consensus PIT predictions + SA-revised actuals merged into a single
        ``consensus_df`` shaped for ``kalman_fusion``'s noise-prior init.
      * ``consensus_by_ds`` and ``sa_actuals_by_ds`` fast-lookup dicts.
      * Per-``ds`` PIT-safe adjustment predictions, using the same
        ``half_life_years`` that ``_load_fusion_selection_target`` reads (so
        the selection target and the tuning objective agree on the adjustment
        baseline).

    Returns ``None`` if any required input (consensus, SA actuals, adjustment
    history) cannot be loaded — the caller falls back to its non-fusion
    objective mode in that case.
    """
    try:
        from Train.Output_code.consensus_anchor_runner import (
            _load_consensus_pit,
            _build_pit_adjustment_cache,
            _compute_adjustment_series,
        )
        from Train.data_loader import load_target_data
        from Train.sandbox.experiment_predicted_adjustment import (
            load_adjustment_history,
        )
    except Exception as e:  # pragma: no cover - guard against import errors
        logger.warning(f"[FusionTune] Could not import fusion deps ({e}); "
                       f"falling back to non-fusion tuning")
        return None

    # half_life_years: prefer tuned value, fall back to 3.0.
    hl_used = 3.0
    tuned_path = OUTPUT_DIR / 'consensus_anchor' / 'kalman_fusion' / 'tuned_params.json'
    if tuned_path.exists():
        try:
            import json as _json
            with open(tuned_path, 'r') as f:
                _tuned = _json.load(f)
            if isinstance(_tuned, dict) and 'half_life_years' in _tuned:
                hl_used = float(_tuned['half_life_years'])
        except Exception:
            pass

    try:
        consensus_monthly = _load_consensus_pit(
            target_type='sa', target_source='revised',
        )
        sa_target = load_target_data(
            target_type='sa', release_type='first', target_source='revised',
        )
        adj_history = load_adjustment_history()
    except Exception as e:
        logger.warning(
            f"[FusionTune] Could not load fusion context ({e}); falling back "
            f"to non-fusion tuning"
        )
        return None

    # consensus_df expected by kalman_fusion has [ds, consensus_pred, actual]
    # so it can compute the prior-history variance for noise initialization.
    sa_actuals_df = pd.DataFrame({
        'ds': pd.to_datetime(sa_target['ds']),
        'actual': sa_target['y_mom'].astype(float),
    })
    consensus_df = (
        consensus_monthly[['ds', 'consensus_pred']]
        .merge(sa_actuals_df, on='ds', how='outer')
        .sort_values('ds')
        .reset_index(drop=True)
    )

    consensus_by_ds: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): float(v)
        for d, v in zip(
            pd.to_datetime(consensus_monthly['ds']).tolist(),
            consensus_monthly['consensus_pred'].astype(float).tolist(),
        )
        if np.isfinite(v)
    }
    sa_actuals_by_ds: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): float(v)
        for d, v in zip(
            pd.to_datetime(sa_target['ds']).tolist(),
            sa_target['y_mom'].astype(float).tolist(),
        )
        if np.isfinite(v)
    }

    # Pre-compute PIT-safe adj_pred for every date the tuner may see.
    unique_dates = pd.Series(pd.to_datetime(pd.unique(target_dates.values))).sort_values()
    pit_cache = _build_pit_adjustment_cache(unique_dates, adj_history)
    adj_values = _compute_adjustment_series(unique_dates, pit_cache, hl_used)
    adj_pred_by_ds: Dict[pd.Timestamp, float] = {
        pd.Timestamp(d): float(v) for d, v in zip(unique_dates, adj_values)
    }

    logger.info(
        f"[FusionTune] Built fusion-CV context: consensus={len(consensus_by_ds)}, "
        f"sa_actuals={len(sa_actuals_by_ds)}, adj_pred={len(adj_pred_by_ds)}, "
        f"half_life_years={hl_used:.3f}"
    )
    return {
        'consensus_df': consensus_df,
        'consensus_by_ds': consensus_by_ds,
        'sa_actuals_by_ds': sa_actuals_by_ds,
        'adj_pred_by_ds': adj_pred_by_ds,
        'half_life_years': hl_used,
    }


def _load_fusion_selection_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_type: str,
    step_date: pd.Timestamp,
) -> Tuple[pd.Series, Optional[float]]:
    """Build the fusion-aligned selection target for the NSA branch.

    Returns ``(y_sel, half_life_years_used)``.

    For the NSA branch, the published forecast is produced by the Kalman
    fusion of ``consensus + (NSA_LGBM_pred + adj_pred) + NSA_accel``. The
    *model channel*'s actual target is ``SA_revised − adj_pred(ds)``
    (because at fusion time ``adj_pred`` is added back). So if we want the
    feature selection to prefer features that help the fusion (rather than
    features that help stand-alone NSA prediction), the selection's notion
    of ``y`` should be ``SA_revised − adj_pred``. The LightGBM model is
    still trained on the original NSA y_mom — only the *selection*'s sense
    of "good feature" shifts.

    PIT-safe: for each ``ds`` the predicted adjustment is built from
    history filtered by ``operational_available_date < ds``. SA_revised
    rows with ``ds >= step_date`` never enter this construction because
    ``X_train['ds'] < step_date`` is the caller's invariant.

    The half-life is read from
    ``_output/consensus_anchor/kalman_fusion/tuned_params.json`` if
    present (the value the most recent Kalman tune chose); otherwise it
    falls back to 3.0. The value used is logged and persisted to
    ``_output/consensus_anchor/dynamic_fs_selection_hl.json`` so the
    next Kalman re-tune can emit a drift warning.

    Falls back to the legacy NSA-y_mom target (and returns
    ``half_life_years_used=None``) when:
      - target_type is not 'nsa'
      - SA target / adjustment history loading fails
      - the produced fusion target has zero finite rows
    """
    legacy_y_sel = pd.Series(
        y_train.values,
        index=pd.to_datetime(X_train['ds'].values),
        name='y_mom',
    ).dropna()

    # FS-A REVERTED (2026-05-15): empirically the fusion-aligned target
    # SA_revised − adj_pred(HL) collapsed feature count from 80→17 and bumped
    # fusion MAE from 93.96→100.61 vs the simpler NSA y_mom target. The
    # smaller residual signal causes SFS stopping (min 0.5% MAE improvement,
    # patience=3) to terminate too early. Restored NSA y_mom as the selection
    # target. Fusion-aligned construction below is kept as dead code for
    # future revival (would require non-zero KALMAN_LAMBDA_* to give the
    # objective curvature first).
    return legacy_y_sel, None

    if str(target_type).lower() != 'nsa':
        return legacy_y_sel, None

    from Train.data_loader import load_target_data
    from Train.sandbox.experiment_predicted_adjustment import (
        load_adjustment_history,
    )
    from Train.Output_code.consensus_anchor_runner import (
        _build_pit_adjustment_cache,
        _compute_adjustment_series,
    )

    # ── half_life_years: read tuned value, fallback to 3.0 ──
    hl_used = 3.0
    tuned_path = OUTPUT_DIR / 'consensus_anchor' / 'kalman_fusion' / 'tuned_params.json'
    if tuned_path.exists():
        try:
            import json as _json
            with open(tuned_path, 'r') as f:
                _tuned = _json.load(f)
            if isinstance(_tuned, dict) and 'half_life_years' in _tuned:
                hl_used = float(_tuned['half_life_years'])
        except Exception as e:
            logger.warning(
                f"[DynFS] Could not read tuned half_life_years from "
                f"{tuned_path} ({e}); using default 3.0"
            )

    # ── Load SA revised actuals and adjustment history ──
    try:
        sa_target = load_target_data(
            target_type='sa', release_type='first', target_source='revised',
        )
        sa_y_mom_by_ds = pd.Series(
            sa_target['y_mom'].astype(float).values,
            index=pd.to_datetime(sa_target['ds'].values),
        )
        adj_history = load_adjustment_history()
    except Exception as e:
        logger.warning(
            f"[DynFS] Could not load SA target / adjustment history for "
            f"fusion-aligned selection target ({e}); falling back to NSA y_mom"
        )
        return legacy_y_sel, None

    # ── Build PIT-filtered cache and compute per-ds adjustment values ──
    train_dates = pd.to_datetime(X_train['ds'].values)
    train_dates_series = pd.Series(train_dates)
    pit_cache = _build_pit_adjustment_cache(train_dates_series, adj_history)
    adj_values = _compute_adjustment_series(train_dates_series, pit_cache, hl_used)

    # ── Assemble fusion target with row-level fallbacks ──
    # If SA actual missing → drop row (matches existing dropna).
    # If PIT history empty for a ds → fall back to NSA y_mom for that row.
    out_dates: List[pd.Timestamp] = []
    out_values: List[float] = []
    n_pit_fallback = 0
    y_train_arr = y_train.values
    for i, ds in enumerate(train_dates):
        sa_val = sa_y_mom_by_ds.get(ds, np.nan)
        avail = pit_cache.get(ds)
        if avail is None or avail.empty:
            nsa_val = y_train_arr[i] if i < len(y_train_arr) else np.nan
            if pd.isna(nsa_val):
                continue
            out_dates.append(ds)
            out_values.append(float(nsa_val))
            n_pit_fallback += 1
            continue
        if pd.isna(sa_val):
            continue
        out_dates.append(ds)
        out_values.append(float(sa_val) - float(adj_values[i]))

    if not out_dates:
        logger.warning(
            "[DynFS] Fusion-aligned target produced zero rows; "
            "falling back to NSA y_mom"
        )
        return legacy_y_sel, None

    fusion_y_sel = pd.Series(
        out_values,
        index=pd.DatetimeIndex(out_dates),
        name='y_fusion',
    )

    # ── Persist the half-life used so the post-tune drift warning can read it ──
    try:
        hl_path = OUTPUT_DIR / 'consensus_anchor' / 'dynamic_fs_selection_hl.json'
        hl_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(hl_path, 'w') as f:
            _json.dump({
                'half_life_years': hl_used,
                'step_date': step_date.strftime('%Y-%m'),
                'n_rows': len(fusion_y_sel),
                'n_pit_fallback': n_pit_fallback,
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"[DynFS] Could not write {hl_path.name}: {e}")

    logger.info(
        f"[DynFS] Selection target = SA_revised − adj_pred "
        f"(half_life_years={hl_used:.3f}, n={len(fusion_y_sel)}, "
        f"NSA-fallback rows={n_pit_fallback})"
    )
    return fusion_y_sel, hl_used


def _dynamic_reselection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    step_date: pd.Timestamp,
    target_type: str,
    target_source: str,
) -> List[str]:
    """Two-pass recency-weighted dynamic feature re-selection.

    Called at the scheduled reselection months during the expanding-window
    backtest. The schedule is anchored on the LAST backtest month and steps
    backward in ``RESELECT_EVERY_N_MONTHS`` increments: ``{anchor, anchor-N,
    anchor-2N, ...}`` (gated by ``RESELECTION_START_DATE``).  Uses exponential
    decay sample weights (half-life = ``RESELECTION_HALF_LIFE_MONTHS``) to
    bias feature importance toward recent observations, adapting to
    structural changes.

    **Pass 1 (Per-Source):** Partition ``X_train`` columns by data source.
    Run lightweight feature selection (stages 0, 2, 4, 5) independently per
    source with recency weights.  FRED Employment runs sequentially;
    smaller sources run in parallel.

    **Pass 2 (Global Cross-Source):** Combine Pass-1 survivors with
    target-derived features.  Run global reduction (stages 0, 2, 4) with
    recency weights to ≤ ``DYNAMIC_FS_PASS2_MAX_FEATURES``.

    Returns:
        List of selected feature names (≤ DYNAMIC_FS_PASS2_MAX_FEATURES).
        **Raises RuntimeError** if selection produces zero features.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc

    from Train.config import (
        DYNAMIC_FS_PASS2_MAX_FEATURES,
        DYNAMIC_FS_BORUTA_RUNS,
        RESELECTION_HALF_LIFE_MONTHS,
        RESELECTION_STAGES_PASS1,
        RESELECTION_STAGES_PASS2,
    )
    from Data_ETA_Pipeline.feature_selection_engine import (
        run_full_source_pipeline,
        _classify_series,
    )

    # Use recency-weighted stages
    stages_pass1 = RESELECTION_STAGES_PASS1
    stages_pass2 = RESELECTION_STAGES_PASS2

    label = f"DynFS/{target_type.upper()}/{target_source}"
    logger.info(
        f"[{label}] Dynamic re-selection triggered at {step_date.strftime('%Y-%m')}. "
        f"Pass-1 stages={stages_pass1}, Pass-2 stages={stages_pass2}, "
        f"max_features={DYNAMIC_FS_PASS2_MAX_FEATURES}, "
        f"half_life={RESELECTION_HALF_LIFE_MONTHS}mo"
    )

    # ── Partition X_train columns by type (numeric only) ──
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    all_cols = [c for c in numeric_cols if c != 'ds']
    groups = partition_feature_columns(all_cols, target_type=target_type)
    snapshot_cols = groups['snapshot_features']
    non_snapshot_cols = (
        groups['target_branch_features']
        + groups['calendar_features']
        + groups['revision_features']
    )

    # ── Classify snapshot columns by data source ──
    source_groups = _classify_columns_by_source(snapshot_cols)
    for src_name, cols in source_groups.items():
        if cols:
            logger.info(f"[{label}] Source {src_name}: {len(cols)} features")

    # ── Build date-indexed target for the selection engine ──
    # For NSA branch, swap to the fusion-aligned target so every target-aware
    # stage (Boruta, cluster, vintage, interaction rescue, SFS) is tilted
    # toward "features whose information makes the fusion model channel
    # close to SA_revised", rather than "features that just predict NSA".
    # The trained LightGBM model still uses y_train (NSA y_mom); only the
    # selection's notion of "good feature" shifts. See
    # _load_fusion_selection_target for PIT details and the fallback path.
    y_sel, _hl_used_for_selection = _load_fusion_selection_target(
        X_train, y_train, target_type, step_date,
    )

    # ── Build date-indexed feature DataFrame from X_train (numeric only) ──
    X_dated = X_train[list(all_cols)].copy()
    X_dated.index = pd.to_datetime(X_train['ds'].values)
    X_dated.index.name = 'date'

    # ── Compute recency weights for feature selection ──
    # Exponential decay: recent months count more for feature importance
    dates = pd.to_datetime(X_train['ds'])
    distance_months = np.maximum(0, (step_date - dates).dt.days.values / 30.436875)
    decay_rate = np.log(2) / RESELECTION_HALF_LIFE_MONTHS
    reselect_weights = np.exp(-decay_rate * distance_months)
    reselect_weights = reselect_weights / np.mean(reselect_weights)
    # Create date-indexed Series for the feature selection engine
    sw_series = pd.Series(reselect_weights, index=X_dated.index, name='sample_weight')

    # ── Pass 1: Per-source feature selection ──
    # Tier-A universe-distillation cache: when USE_UNIVERSE_CACHE is enabled,
    # try to load a recent enough Pass-1 result from disk and skip the heavy
    # per-source Boruta loop entirely. PIT invariant (enforced inside
    # load_latest_universe): the cache's asof must be ≤ step_date.
    massive_sources = ['FRED_Employment_NSA', 'FRED_Employment_SA']
    small_sources = [
        'FRED_Exogenous', 'Unifier', 'ADP', 'NOAA', 'Prosper',
        'Futures', 'EconomistPanel', 'SA_NSA_Gap',
    ]
    pass1_survivors: Dict[str, List[str]] = {}

    from Train.config import USE_UNIVERSE_CACHE, UNIVERSE_REFRESH_MONTHS

    _universe_cache_hit = False
    if USE_UNIVERSE_CACHE:
        from Train.universe_cache import (
            is_cache_fresh,
            load_latest_universe,
            save_universe,
        )
        _cached_universe = load_latest_universe(
            target_type=target_type,
            target_source=target_source,
            step_date=step_date,
        )
        if _cached_universe is not None and is_cache_fresh(
            _cached_universe["asof"], step_date, UNIVERSE_REFRESH_MONTHS,
        ):
            # Restrict cached survivors to columns that exist in this slice
            # (the universe might have been built when extra cols existed).
            _x_cols = set(X_dated.columns)
            pass1_survivors = {
                src: [c for c in feats if c in _x_cols]
                for src, feats in _cached_universe["survivors"].items()
            }
            _universe_cache_hit = True
            logger.info(
                f"[{label}] Tier-A HIT: universe_asof="
                f"{_cached_universe['asof'].strftime('%Y-%m')} step_date="
                f"{step_date.strftime('%Y-%m')} → "
                f"{sum(len(v) for v in pass1_survivors.values())} survivors "
                f"across {len(pass1_survivors)} sources. Skipping Pass-1."
            )

    def _run_source_pass1(source_name: str, cols: List[str]) -> Tuple[str, List[str]]:
        """Run Pass-1 selection on one source's columns from X_train."""
        if not cols:
            return source_name, []

        snap_wide = X_dated[cols].copy()

        # Drop zero-variance columns
        zero_var = snap_wide.std() == 0
        if zero_var.any():
            snap_wide = snap_wide.loc[:, ~zero_var]
        if snap_wide.empty:
            return source_name, []

        # Build series groups (using the FS engine's classifier)
        series_groups_local = defaultdict(list)
        for col in snap_wide.columns:
            grp = _classify_series(col, source_name)
            series_groups_local[grp].append(col)

        logger.info(
            f"[{label}] {source_name}: {snap_wide.shape[1]} features, "
            f"{len(series_groups_local)} groups → Pass-1"
        )

        try:
            survivors = run_full_source_pipeline(
                snap_wide, y_sel, source_name, Path("/dev/null"),
                series_groups_local, stages=stages_pass1,
                sample_weight=sw_series,
            )
            logger.info(f"[{label}] {source_name}: Pass-1 → {len(survivors)} survivors")
            return source_name, survivors
        except Exception as e:
            logger.error(f"[{label}] {source_name}: Pass-1 failed: {e}")
            return source_name, []

    if not _universe_cache_hit:
        # Run massive sources (FRED Employment) sequentially
        for src_name in massive_sources:
            cols = source_groups.get(src_name, [])
            if cols:
                name, feats = _run_source_pass1(src_name, cols)
                pass1_survivors[name] = feats
                gc.collect()

        # Run small sources in parallel
        small_to_run = [(s, source_groups.get(s, [])) for s in small_sources
                        if source_groups.get(s)]
        if small_to_run:
            max_workers = min(len(small_to_run), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_source_pass1, name, cols): name
                    for name, cols in small_to_run
                }
                for future in as_completed(futures):
                    src_name, feats = future.result()
                    pass1_survivors[src_name] = feats

        # Include deterministic derived controls directly; these were
        # previously in the Unknown bucket before source attribution was
        # tightened for feature-culling reports.
        for direct_source in ('DerivedControls', 'Unknown'):
            direct_cols = source_groups.get(direct_source, [])
            if direct_cols:
                pass1_survivors[direct_source] = direct_cols

    total_pass1 = sum(len(v) for v in pass1_survivors.values())
    logger.info(
        f"[{label}] Pass-1 complete: {total_pass1} total survivors across "
        f"{len([k for k, v in pass1_survivors.items() if v])} sources"
        f"{' (from cache)' if _universe_cache_hit else ''}"
    )

    # Persist a freshly-computed Pass-1 result so future step_dates inside
    # the UNIVERSE_REFRESH_MONTHS window can reuse it. We only save when we
    # actually recomputed (cache miss); cache hits don't re-save the same data.
    if USE_UNIVERSE_CACHE and not _universe_cache_hit and total_pass1 > 0:
        try:
            save_universe(
                survivors=pass1_survivors,
                target_type=target_type,
                target_source=target_source,
                asof=step_date,
                stages=tuple(stages_pass1),
            )
        except Exception as _e:
            logger.warning(f"[{label}] universe_cache save failed: {_e}")

    if total_pass1 == 0:
        raise RuntimeError(
            f"[{label}] Dynamic re-selection Pass-1 returned zero features. "
            f"Cannot proceed without features."
        )

    # ── Pass 2: Global cross-source reduction ──
    # Combine Pass-1 survivors + target-derived features
    all_pass1_cols = []
    for feats in pass1_survivors.values():
        all_pass1_cols.extend(feats)

    pass2_cols = list(set(all_pass1_cols + non_snapshot_cols))
    # Only keep columns that exist in X_dated
    pass2_cols = [c for c in pass2_cols if c in X_dated.columns]

    pass2_wide = X_dated[pass2_cols].copy()

    logger.info(
        f"[{label}] Pass-2 input: {pass2_wide.shape[1]} features × "
        f"{pass2_wide.shape[0]} dates"
    )

    pass2_groups = defaultdict(list)
    for col in pass2_wide.columns:
        pass2_groups["Global"].append(col)

    try:
        pass2_survivors = run_full_source_pipeline(
            pass2_wide, y_sel, "Global_Pass2", Path("/dev/null"),
            pass2_groups, stages=stages_pass2,
            sample_weight=sw_series,
        )
    except Exception as e:
        logger.error(f"[{label}] Pass-2 pipeline failed: {e}")
        raise RuntimeError(
            f"[{label}] Dynamic re-selection Pass-2 failed: {e}"
        )

    # Hard cap: if more than max, take top-N by Boruta importance
    if len(pass2_survivors) > DYNAMIC_FS_PASS2_MAX_FEATURES:
        logger.info(
            f"[{label}] Pass-2 returned {len(pass2_survivors)} features; "
            f"applying hard cap at {DYNAMIC_FS_PASS2_MAX_FEATURES}"
        )
        try:
            from Data_ETA_Pipeline.feature_selection_engine import get_boruta_importance
            boruta_hits = get_boruta_importance(
                pass2_wide[pass2_survivors], y_sel,
                n_runs=DYNAMIC_FS_BORUTA_RUNS,
                sample_weight=sw_series,
            )
            if len(boruta_hits) >= DYNAMIC_FS_PASS2_MAX_FEATURES:
                pass2_survivors = boruta_hits[:DYNAMIC_FS_PASS2_MAX_FEATURES]
            else:
                remaining = [f for f in pass2_survivors if f not in set(boruta_hits)]
                pass2_survivors = boruta_hits + remaining[
                    : DYNAMIC_FS_PASS2_MAX_FEATURES - len(boruta_hits)
                ]
        except Exception as e:
            logger.warning(f"[{label}] Boruta hard-cap failed ({e}); truncating by column order.")
            pass2_survivors = pass2_survivors[:DYNAMIC_FS_PASS2_MAX_FEATURES]

    # Only return features that exist in X_train
    available_in_train = set(X_train.columns)
    final_features = [f for f in pass2_survivors if f in available_in_train]

    if not final_features:
        raise RuntimeError(
            f"[{label}] Dynamic re-selection produced zero usable features "
            f"(Pass-1: {total_pass1}, Pass-2: {len(pass2_survivors)}, "
            f"available in X_train: 0)."
        )

    logger.info(
        f"[{label}] Dynamic re-selection complete: "
        f"{total_pass1} → {len(pass2_survivors)} → {len(final_features)} features"
    )

    return final_features


def _merge_unique_feature_lists(*groups: List[str]) -> List[str]:
    """Merge feature name groups while preserving input order and uniqueness."""
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for feature in group:
            if feature not in seen:
                seen.add(feature)
                merged.append(feature)
    return merged


def _is_variance_priority_target(target_type: str, target_source: str) -> bool:
    return (target_type, target_source) in set(VARIANCE_PRIORITY_TARGETS)


def _get_tuning_objective_mode(target_type: str, target_source: str) -> str:
    # NSA branch optimizes directly for the fusion composite when the flag is
    # on (this is the published-output mode). Variance-priority targets and
    # other branches keep the legacy composite objective.
    if str(target_type).lower() == 'nsa' and NSA_TUNE_USE_KALMAN_FUSION:
        return 'kalman_fusion'
    if _is_variance_priority_target(target_type, target_source):
        return TUNING_OBJECTIVE_MODE_VARIANCE
    return TUNING_OBJECTIVE_MODE_DEFAULT


def _get_branch_target_fs_method(target_type: str, target_source: str) -> str:
    if _is_variance_priority_target(target_type, target_source):
        return BRANCH_TARGET_FS_METHOD_VARIANCE
    return BRANCH_TARGET_FS_METHOD


def _get_branch_target_fs_top_k(target_type: str, target_source: str) -> int:
    if _is_variance_priority_target(target_type, target_source):
        return BRANCH_TARGET_FS_TOPK_VARIANCE
    return BRANCH_TARGET_FS_TOPK


def _validation_composite_score(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    yv = y_true.values.astype(float)
    pv = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(yv - pv)))
    kpis = compute_variance_kpis(
        yv, pv, tail_quantile=VARIANCE_TAIL_QUANTILE, extreme_quantile=VARIANCE_EXTREME_QUANTILE
    )
    score = composite_objective_score(
        mae=mae,
        std_ratio=float(kpis['std_ratio']),
        diff_std_ratio=float(kpis['diff_std_ratio']),
        tail_mae=float(kpis['tail_mae']),
        corr_diff=float(kpis['corr_diff']),
        diff_sign_accuracy=float(kpis['diff_sign_accuracy']),
        lambda_std_ratio=TUNING_LAMBDA_STD_RATIO,
        lambda_diff_std_ratio=TUNING_LAMBDA_DIFF_STD_RATIO,
        lambda_tail_mae=TUNING_LAMBDA_TAIL_MAE,
        lambda_corr_diff=TUNING_LAMBDA_CORR_DIFF,
        lambda_diff_sign=TUNING_LAMBDA_DIFF_SIGN,
    )
    kpis['mae'] = mae
    kpis['composite_score'] = score
    return score, kpis


def _compute_tail_weight_multiplier(y_values: np.ndarray) -> np.ndarray:
    """
    Build a multiplicative weight vector emphasizing large levels and changes.
    """
    y = np.asarray(y_values, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)

    mult = np.ones_like(y, dtype=float)
    abs_y = np.abs(y)
    level_thr = float(np.quantile(abs_y, TAIL_WEIGHT_ABS_LEVEL_QUANTILE))
    mult[abs_y >= level_thr] *= TAIL_WEIGHT_LEVEL_BOOST

    abs_dy = np.abs(np.diff(y, prepend=y[0]))
    diff_thr = float(np.quantile(abs_dy, TAIL_WEIGHT_ABS_DIFF_QUANTILE))
    mult[abs_dy >= diff_thr] *= TAIL_WEIGHT_DIFF_BOOST

    return np.clip(mult, 1.0, TAIL_WEIGHT_MAX_MULTIPLIER)


def _apply_tail_aware_weighting(
    weights: np.ndarray,
    y_values: np.ndarray,
    target_type: str,
    target_source: str,
) -> np.ndarray:
    """
    Apply tail-aware weighting only for variance-priority targets.
    """
    base = np.asarray(weights, dtype=float)
    if (
        not ENABLE_TAIL_AWARE_WEIGHTING
        or not _is_variance_priority_target(target_type, target_source)
        or base.size == 0
    ):
        return base

    mult = _compute_tail_weight_multiplier(np.asarray(y_values, dtype=float))
    if mult.size != base.size:
        return base
    out = base * mult
    m = float(np.mean(out))
    if m > 0:
        out = out / m
    return out


def _fit_amplitude_calibration(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Optional[Tuple[float, float]]:
    yv = y_true.values.astype(float)
    pv = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yv) & np.isfinite(pv)
    if mask.sum() < AMPLITUDE_CAL_MIN_SAMPLES:
        return None
    p_valid = pv[mask]
    y_valid = yv[mask]
    if np.std(p_valid) < 1e-12:
        return None
    slope, intercept = np.polyfit(p_valid, y_valid, 1)
    slope = float(np.clip(slope, AMPLITUDE_CAL_SLOPE_MIN, AMPLITUDE_CAL_SLOPE_MAX))
    return float(intercept), slope


def _apply_amplitude_calibration(
    y_pred: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return intercept + slope * np.asarray(y_pred, dtype=float)


def _train_simple_regressor(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    num_boost_round: int,
    max_depth: int,
    num_leaves: int,
    objective: str = 'regression',
    train_weights: Optional[np.ndarray] = None,
):
    params = {
        'objective': objective,
        'metric': 'mae' if objective != 'binary' else 'binary_logloss',
        'learning_rate': 0.05,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_data_in_leaf': 5,
        'verbosity': -1,
        'n_jobs': 1,
        **LGBM_DETERMINISM,
    }
    w = None if train_weights is None else np.asarray(train_weights, dtype=float)
    if objective == 'binary':
        # Rebalance minority direction class so the classifier does not collapse
        # to mostly-positive sign predictions.
        yb = np.asarray(y_tr, dtype=float)
        pos = float(np.sum(yb > 0.5))
        neg = float(np.sum(yb <= 0.5))
        if pos > 0 and neg > 0:
            pos_w = (pos + neg) / (2.0 * pos)
            neg_w = (pos + neg) / (2.0 * neg)
            class_w = np.where(yb > 0.5, pos_w, neg_w)
            w = class_w if w is None else (w * class_w)

    train_data = lgb.Dataset(X_tr, label=y_tr, weight=w, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=0)],
    )
    return model


def _build_multi_target_dynamics_candidate(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_pred: pd.DataFrame,
    y_train_valid: pd.Series,
    current_best_val: np.ndarray,
    current_best_pred: float,
) -> Optional[Tuple[np.ndarray, float, Dict[str, float]]]:
    """
    Build a candidate forecast using three targets:
    1) level (current best prediction stream),
    2) magnitude of delta (|MoM acceleration|),
    3) direction of delta (sign).
    """
    if len(X_tr) < 24 or len(X_val) < 3:
        return None

    ytr = y_tr.values.astype(float)
    yval = y_val.values.astype(float)
    if ytr.size < 3:
        return None

    # Delta target: dy[t] = y[t] - y[t-1]
    y_tr_diff = np.diff(ytr)
    if y_tr_diff.size < 12:
        return None
    X_tr_diff = X_tr.iloc[1:]
    prev_for_val = np.concatenate([np.array([ytr[-1]]), yval[:-1]])
    y_val_diff = yval - prev_for_val

    # Tail-aware emphasis on larger delta regimes for dynamics auxiliary models.
    aux_w = _compute_tail_weight_multiplier(y_tr_diff)

    # Magnitude model: |dy|
    mag_train = np.abs(y_tr_diff)
    mag_val_target = np.abs(y_val_diff)
    mag_model = _train_simple_regressor(
        X_tr_diff, mag_train, X_val, mag_val_target,
        num_boost_round=DYNAMICS_MODEL_NUM_BOOST_ROUND,
        max_depth=DYNAMICS_MODEL_MAX_DEPTH,
        num_leaves=DYNAMICS_MODEL_NUM_LEAVES,
        objective='regression',
        train_weights=aux_w,
    )
    mag_val = np.maximum(np.asarray(mag_model.predict(X_val), dtype=float), DYNAMICS_MAGNITUDE_FLOOR)
    mag_pred = float(max(float(mag_model.predict(X_pred)[0]), DYNAMICS_MAGNITUDE_FLOOR))

    # Direction model: sign(dy)
    dir_train = (y_tr_diff > 0.0).astype(float)
    if np.unique(dir_train).size < 2:
        sign_val = np.sign(np.asarray(current_best_val, dtype=float) - prev_for_val)
        sign_val[sign_val == 0.0] = 1.0
        sign_pred = np.sign(float(current_best_pred - float(y_train_valid.iloc[-1])))
        if sign_pred == 0.0:
            sign_pred = 1.0
        p_up_val = np.where(sign_val > 0, 0.51, 0.49)
        p_up_pred = 1.0 if sign_pred > 0 else 0.0
    else:
        dir_val_label = (y_val_diff > 0.0).astype(float)
        dir_model = _train_simple_regressor(
            X_tr_diff, dir_train, X_val, dir_val_label,
            num_boost_round=DYNAMICS_MODEL_NUM_BOOST_ROUND,
            max_depth=DYNAMICS_MODEL_MAX_DEPTH,
            num_leaves=DYNAMICS_MODEL_NUM_LEAVES,
            objective='binary',
            train_weights=aux_w,
        )
        p_up_val = np.clip(np.asarray(dir_model.predict(X_val), dtype=float), 0.0, 1.0)
        p_up_pred = float(np.clip(dir_model.predict(X_pred)[0], 0.0, 1.0))
        sign_val = np.where(p_up_val >= 0.5, 1.0, -1.0)
        sign_pred = 1.0 if p_up_pred >= 0.5 else -1.0

    delta_signedmag_val = sign_val * mag_val
    delta_signedmag_pred = float(sign_pred * mag_pred)

    # Blend signed-magnitude delta with current best implied delta.
    current_delta_val = np.asarray(current_best_val, dtype=float) - prev_for_val
    last_known = float(y_train_valid.iloc[-1])
    current_delta_pred = float(current_best_pred - last_known)
    delta_core_val = (
        DYNAMICS_DELTA_BLEND * delta_signedmag_val
        + (1.0 - DYNAMICS_DELTA_BLEND) * current_delta_val
    )
    delta_core_pred = float(
        DYNAMICS_DELTA_BLEND * delta_signedmag_pred
        + (1.0 - DYNAMICS_DELTA_BLEND) * current_delta_pred
    )

    # Confidence-weighted sign enforcement from direction classifier.
    conf_val = np.abs(p_up_val - 0.5)
    conf_pred = abs(p_up_pred - 0.5)
    denom = max(1e-9, 0.5 - DYNAMICS_DIRECTION_CONFIDENCE)
    blend_val = np.clip((conf_val - DYNAMICS_DIRECTION_CONFIDENCE) / denom, 0.0, 1.0)
    blend_val = DYNAMICS_DIRECTION_BLEND * blend_val
    blend_pred = float(
        DYNAMICS_DIRECTION_BLEND
        * np.clip((conf_pred - DYNAMICS_DIRECTION_CONFIDENCE) / denom, 0.0, 1.0)
    )

    delta_enforced_val = sign_val * np.abs(delta_core_val)
    delta_enforced_pred = float(sign_pred * abs(delta_core_pred))
    delta_final_val = (1.0 - blend_val) * delta_core_val + blend_val * delta_enforced_val
    delta_final_pred = float((1.0 - blend_pred) * delta_core_pred + blend_pred * delta_enforced_pred)

    cand_val = prev_for_val + delta_final_val
    cand_pred = float(last_known + delta_final_pred)
    return cand_val, cand_pred, {
        "direction_enforced_share_val": float(np.mean(blend_val > 0)) if blend_val.size else 0.0,
        "direction_conf_pred": float(conf_pred),
        "direction_model_used": 1.0,
        "delta_signedmag_mean_abs": float(np.mean(np.abs(delta_signedmag_val))),
    }


def _run_variance_enhancement_sequence(
    base_model: "lgb.Booster",
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_pred: pd.DataFrame,
    y_train_valid: pd.Series,
    target_type: str,
    target_source: str,
) -> Tuple[float, np.ndarray, str, Dict[str, Any]]:
    """
    Sequentially apply variance-enhancement stages and keep only improving stages.

    Sequence: base -> amplitude -> shock -> dynamics -> acceleration -> regime.
    """
    base_val = base_model.predict(X_val)
    base_pred = float(base_model.predict(X_pred)[0])
    best_val = np.asarray(base_val, dtype=float)
    best_pred = float(base_pred)
    best_stage = 'base'
    best_score, best_kpis = _validation_composite_score(y_val, best_val)

    stage_report: Dict[str, Any] = {
        'applied': [],
        'scores': {'base': float(best_score)},
        'kpis_base': best_kpis,
    }

    use_stack = (
        ENABLE_VARIANCE_ENHANCEMENTS
        and _is_variance_priority_target(target_type, target_source)
        and (target_type, target_source) not in set(ENHANCEMENT_EXEMPT_TARGETS)
    )
    if not use_stack:
        return best_pred, best_val, best_stage, stage_report

    for stage in ENHANCEMENT_SEQUENCE:
        cand_val = None
        cand_pred = None

        try:
            if stage == 'amplitude' and ENABLE_AMPLITUDE_CALIBRATION:
                cal = _fit_amplitude_calibration(y_val, best_val)
                if cal is not None:
                    intercept, slope = cal
                    cand_val = _apply_amplitude_calibration(best_val, intercept, slope)
                    cand_pred = float(_apply_amplitude_calibration(np.array([best_pred]), intercept, slope)[0])

            elif stage == 'shock' and ENABLE_SHOCK_MODEL:
                base_tr = base_model.predict(X_tr)
                base_val_now = base_model.predict(X_val)
                residual_tr = y_tr.values.astype(float) - base_tr
                residual_val = y_val.values.astype(float) - base_val_now
                shock_model = _train_simple_regressor(
                    X_tr, residual_tr, X_val, residual_val,
                    num_boost_round=SHOCK_MODEL_NUM_BOOST_ROUND,
                    max_depth=SHOCK_MODEL_MAX_DEPTH,
                    num_leaves=SHOCK_MODEL_NUM_LEAVES,
                    objective='regression',
                )
                shock_val = shock_model.predict(X_val)
                shock_pred = float(shock_model.predict(X_pred)[0])
                cand_val = best_val + shock_val
                cand_pred = float(best_pred + shock_pred)

            elif stage == 'dynamics' and ENABLE_MULTI_TARGET_DYNAMICS:
                dynamic_candidate = _build_multi_target_dynamics_candidate(
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_val=X_val,
                    y_val=y_val,
                    X_pred=X_pred,
                    y_train_valid=y_train_valid,
                    current_best_val=best_val,
                    current_best_pred=best_pred,
                )
                if dynamic_candidate is not None:
                    cand_val, cand_pred, dyn_meta = dynamic_candidate
                    stage_report['dynamics_meta'] = dyn_meta

            elif stage == 'acceleration' and ENABLE_ACCELERATION_MODEL:
                if len(X_tr) >= 12 and len(X_val) >= 2:
                    ytr = y_tr.values.astype(float)
                    yval = y_val.values.astype(float)
                    y_tr_acc = np.diff(ytr)
                    X_tr_acc = X_tr.iloc[1:]
                    prev_for_val = np.concatenate([np.array([ytr[-1]]), yval[:-1]])
                    y_val_acc = yval - prev_for_val

                    acc_model = _train_simple_regressor(
                        X_tr_acc, y_tr_acc, X_val, y_val_acc,
                        num_boost_round=ACCEL_MODEL_NUM_BOOST_ROUND,
                        max_depth=ACCEL_MODEL_MAX_DEPTH,
                        num_leaves=ACCEL_MODEL_NUM_LEAVES,
                        objective='regression',
                    )
                    acc_val = acc_model.predict(X_val)
                    acc_pred = float(acc_model.predict(X_pred)[0])
                    cand_val = prev_for_val + acc_val
                    last_known = float(y_train_valid.iloc[-1])
                    cand_pred = float(last_known + acc_pred)

            elif stage == 'regime' and ENABLE_REGIME_ROUTER:
                ytr = y_tr.values.astype(float)
                thr = float(np.quantile(np.abs(ytr), REGIME_HIGHVOL_QUANTILE))
                high_mask = np.abs(ytr) >= thr
                low_mask = ~high_mask
                if high_mask.sum() >= REGIME_MIN_CLASS_SAMPLES and low_mask.sum() >= REGIME_MIN_CLASS_SAMPLES:
                    low_model = _train_simple_regressor(
                        X_tr[low_mask], ytr[low_mask], X_val, y_val.values.astype(float),
                        num_boost_round=REGIME_MODEL_NUM_BOOST_ROUND,
                        max_depth=3,
                        num_leaves=15,
                        objective='regression',
                    )
                    high_model = _train_simple_regressor(
                        X_tr[high_mask], ytr[high_mask], X_val, y_val.values.astype(float),
                        num_boost_round=REGIME_MODEL_NUM_BOOST_ROUND,
                        max_depth=3,
                        num_leaves=15,
                        objective='regression',
                    )
                    router = _train_simple_regressor(
                        X_tr, high_mask.astype(float), X_val, (np.abs(y_val.values.astype(float)) >= thr).astype(float),
                        num_boost_round=REGIME_MODEL_NUM_BOOST_ROUND,
                        max_depth=3,
                        num_leaves=15,
                        objective='binary',
                    )
                    p_high_val = np.clip(router.predict(X_val), 0.0, 1.0)
                    p_high_pred = float(np.clip(router.predict(X_pred)[0], 0.0, 1.0))
                    low_val = low_model.predict(X_val)
                    high_val = high_model.predict(X_val)
                    low_pred = float(low_model.predict(X_pred)[0])
                    high_pred = float(high_model.predict(X_pred)[0])
                    cand_val = (1.0 - p_high_val) * low_val + p_high_val * high_val
                    cand_pred = float((1.0 - p_high_pred) * low_pred + p_high_pred * high_pred)

        except Exception as exc:
            logger.warning(f"Variance stage '{stage}' failed: {exc}")
            cand_val = None
            cand_pred = None

        if cand_val is None or cand_pred is None:
            stage_report['scores'][stage] = None
            continue

        cand_score, cand_kpis = _validation_composite_score(y_val, cand_val)
        stage_report['scores'][stage] = float(cand_score)
        improvement = best_score - cand_score
        if improvement >= ENHANCEMENT_MIN_IMPROVEMENT:
            best_score = cand_score
            best_val = np.asarray(cand_val, dtype=float)
            best_pred = float(cand_pred)
            best_stage = stage
            stage_report['applied'].append(stage)
            stage_report[f'kpis_{stage}'] = cand_kpis

    stage_report['selected_stage'] = best_stage
    stage_report['selected_score'] = float(best_score)
    return best_pred, best_val, best_stage, stage_report


@profiled("train.build_training_dataset")
def build_training_dataset(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'revised',
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset by loading pre-merged master snapshots.

    Master snapshots already contain ALL data sources (FRED employment + exogenous),
    pre-filtered to selected features. No separate FRED loading or feature selection needed.

    Uses the NFP release_date as the data cutoff (e.g., May 3rd for April NFP),
    matching inference behavior so the model learns from intra-month data.
    Includes lagged target features from the branch's own target history.

    Args:
        target_df: Target DataFrame with y_mom column (the prediction target)
        target_type: 'nsa' or 'sa' - determines which target we're predicting
        release_type: 'first' or 'last' - determines which release to use for lagged features
        target_source: 'revised' - determines which master snapshot variant
        start_date: Start date for training data
        end_date: End date for training data
        show_progress: Whether to show progress logging

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    all_features = []
    all_targets = []
    valid_dates = []

    model_id = get_model_id(target_type, release_type, target_source)

    # Load branch target data once (cached) for lag feature engineering
    source_label = "revised" if target_source == "revised" else f"{release_type} release"
    logger.info(f"Loading {target_type.upper()} {source_label} target data for feature engineering...")
    branch_target_full = load_target_data(
        target_type, release_type=release_type, target_source=target_source
    )

    # Pre-compute branch-target lagged features vectorized (shift/rolling)
    # instead of per-worker filtering. Produces O(1) worker lookups.
    target_prefix = f"nfp_{target_type}"
    logger.info(f"Pre-computing vectorized lagged target features ({target_prefix})...")
    target_lags_lookup = batch_lagged_target_features(branch_target_full, prefix=target_prefix)

    # Filter target data by date range
    filtered_df = target_df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['ds'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['ds'] <= end_date]

    n_months = len(filtered_df)
    logger.info(f"Building features for {n_months} target months...")

    import time as _time
    _build_t0 = _time.time()

    # Build release_date lookup (vectorized — no iterrows)
    target_ref = branch_target_full
    release_date_map = {}
    if 'release_date' in target_ref.columns:
        valid_mask = target_ref['release_date'].notna()
        release_date_map = dict(zip(
            target_ref.loc[valid_mask, 'ds'],
            target_ref.loc[valid_mask, 'release_date'],
        ))

    # Prepare arguments for parallel execution (vectorized — no iterrows)
    target_months = filtered_df['ds'].values
    target_values = filtered_df['y_mom'].values
    tasks = []
    for i in range(len(filtered_df)):
        tm = pd.Timestamp(target_months[i])
        tasks.append((
            tm, target_values[i],
            release_date_map.get(tm, tm),
            tm + pd.offsets.MonthEnd(0),
            target_type, release_type, target_source,
            target_lags_lookup,
        ))

    n_jobs, backend = _feature_engineering_parallel_settings()
    backend_msg = f" with {backend} backend" if backend else ""
    if n_jobs == 1:
        logger.info(
            f"Starting sequential feature engineering for {len(tasks)} months"
        )
        results = [_process_single_month_task(*args) for args in tasks]
    else:
        logger.info(
            f"Starting parallel feature engineering for {len(tasks)} months "
            f"with n_jobs={n_jobs}{backend_msg}..."
        )
        parallel_kwargs = {"n_jobs": n_jobs, "verbose": 5}
        if backend:
            parallel_kwargs["backend"] = backend
        results = Parallel(**parallel_kwargs)(
            delayed(_process_single_month_task)(*args) for args in tasks
        )

    # Filter out None results (skipped months)
    valid_results = [r for r in results if r[0] is not None]

    if not valid_results:
         logger.warning("No valid training samples generated!")
         return pd.DataFrame(), pd.Series(dtype=float)

    all_features_dicts, all_targets_list = zip(*valid_results)

    # Create final lists for DataFrame construction
    all_features = list(all_features_dicts)
    all_targets = list(all_targets_list)
    valid_dates = [f['ds'] for f in all_features]

    if not all_features:
        logger.error("No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=float)

    # Combine all features efficiently
    X = pd.DataFrame(all_features)
    y = pd.Series(all_targets, name='y_mom')

    # Add date index for reference
    X['ds'] = valid_dates

    # NOTE: COVID winsorization moved to backtest loop (per-fold) to avoid future data leakage.

    # Replace inf with NaN (LightGBM handles NaN natively but not inf)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop duplicate columns (can occur from overlapping source data)
    dupes = X.columns.duplicated()
    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning(f"Dropping {n_dupes} duplicate columns")
        X = X.loc[:, ~dupes]

    logger.info(f"Built training dataset: {len(X)} samples, {len(X.columns)} total features "
                f"(master snapshot features are pre-selected in ETL)")

    return X, y


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

@profiled("train.backtest.total")
def run_expanding_window_backtest(
    target_df: pd.DataFrame,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'revised',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    tune: bool = True,
    nsa_backtest_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run proper expanding window backtest with strictly NO TIME-TRAVEL VIOLATIONS.

    This is the core loop evaluating the model's true real-world point-in-time performance.
    Instead of standard K-Fold CV (which randomly shuffles data and predicts the past using the future),
    this loop marches chronologically forward one month at a time. It uses strictly the available data
    as of that specific historical month to predict it, precisely mirroring a real-time trading environment.

    Critical Design Principles:
    1. LightGBM handles NaN natively - no imputation needed preventing forward-filling bias.
    2. Model is FULLY retrained from scratch at each step.
    3. No information from future time periods leaks into predictions.
    4. Features are pre-selected per target_type (NSA or SA).
    5. Hyperparameters tuned via inner CV every `TUNE_EVERY_N_MONTHS` months.

    Args:
        target_df: Target DataFrame with 'ds' and 'y_mom' columns
        target_type: 'nsa' or 'sa'
        release_type: 'first' or 'last'
        target_source: 'revised'
        use_huber_loss: Whether to use Huber loss
        huber_delta: Huber delta parameter
        tune: If True, run Optuna hyperparameter tuning periodically

    Returns:
        Tuple of (results_df, X_full, y_full) where X_full and y_full are the
        pre-built feature matrix and target series (reusable for production model).
    """
    model_id = get_model_id(target_type, release_type, target_source)

    logger.info("=" * 60)
    logger.info(f"EXPANDING WINDOW BACKTEST [{model_id.upper()}] (No Time-Travel)")
    logger.info("=" * 60)

    # Determine backtest period
    backtest_start_idx = len(target_df) - BACKTEST_MONTHS
    backtest_months = target_df.iloc[backtest_start_idx:]['ds'].tolist()

    logger.info(f"Backtest period: {BACKTEST_MONTHS} months ({backtest_months[0].strftime('%Y-%m')} to {backtest_months[-1].strftime('%Y-%m')})")

    # Warm branch-target cache used in lagged feature engineering
    load_target_data(target_type, release_type=release_type, target_source=target_source)

    # Build a ds -> release_date map from the current target. Used by the SA
    # branch's NSA-acceleration injection block to filter revised actuals by
    # operational_available_date < cutoff_date (Issue 10 fix). For the NSA
    # branch this map is built but not consumed.
    target_release_date_map: Dict[pd.Timestamp, pd.Timestamp] = {}
    if 'release_date' in target_df.columns:
        valid_rd = target_df['release_date'].notna()
        target_release_date_map = dict(zip(
            pd.to_datetime(target_df.loc[valid_rd, 'ds']),
            pd.to_datetime(target_df.loc[valid_rd, 'release_date']),
        ))

    # Build FULL feature dataset once.
    # First check the persistent content-hashed parquet cache — build_training_dataset
    # is a deterministic function of the master snapshots + target parquet, so when
    # neither has changed since the last run, we skip the parallel build entirely.
    from Train.training_dataset_cache import (
        load_cached_dataset,
        save_cached_dataset,
    )
    _cached = load_cached_dataset(
        target_df, target_type, release_type, target_source,
        start_date=None, end_date=None,
    )
    if _cached is not None:
        X_full, y_full = _cached
    else:
        logger.info("Building full feature dataset...")
        X_full, y_full = build_training_dataset(
            target_df, target_type=target_type, release_type=release_type,
            target_source=target_source,
            show_progress=False
        )
        if not X_full.empty:
            save_cached_dataset(
                X_full, y_full, target_df,
                target_type, release_type, target_source,
                start_date=None, end_date=None,
            )

    if X_full.empty:
        logger.error("Failed to build training dataset")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    X_full = attach_sidecar_features(X_full, output_dir=OUTPUT_DIR, logger=logger)

    # COVID-winsorize X_full and y_full ONCE upfront so training, prediction
    # (X_pred = X_full.iloc[[target_idx]]), AND the production model fit (which
    # also reuses X_full via train_and_evaluate) all see consistent winsorized
    # values for Mar-May 2020. This closes the asymmetry where the prior
    # per-step winsorize on X_train_valid only touched the training slice and
    # left X_pred at COVID target months exposed to raw out-of-distribution
    # feature values (e.g., NFP_Consensus_Mean = -14,448 at Apr 2020). The
    # clipping bounds use full-history non-COVID quantiles, which are
    # empirically stable across decades for NFP-related features.
    logger.info("Applying COVID winsorization to X_full and y_full upfront...")
    _x_indexed = X_full.set_index('ds')
    _numeric = _x_indexed.select_dtypes(include=[np.number]).columns
    _x_indexed[_numeric] = winsorize_covid_period(_x_indexed[_numeric])
    X_full = _x_indexed.reset_index(names='ds')

    _y_indexed = pd.Series(
        y_full.values, index=pd.to_datetime(X_full['ds']), name='y_mom',
    )
    y_full = winsorize_covid_period(_y_indexed).reset_index(drop=True)

    # Pre-compute indices for faster lookup
    date_to_idx = {d: i for i, d in enumerate(X_full['ds'])}

    # Store results
    results = []
    all_residuals = []
    # Tracking var for acceleration accuracy (vs-last-actual formula)
    last_actual: float = np.nan

    # Static fallback params (used when tune=False)
    static_params = get_lgbm_params(use_huber_loss=use_huber_loss, huber_delta=huber_delta)

    # ── Feature selection mode ──
    # Check if master snapshots are in all-features mode (no ETL-time selection).
    # When all-features mode, dynamic reselection is mandatory and the only path.
    master_snapshot_base_features = load_selected_features(
        target_type=target_type, target_source=target_source
    )
    _all_features_mode = master_snapshot_base_features is None

    if _all_features_mode:
        logger.info(
            f"[{model_id}] ALL-FEATURES mode: master snapshots contain all lean features. "
            f"Dynamic reselection is the ONLY feature selection path."
        )
    else:
        logger.info(
            f"Loaded master snapshot base feature set for {model_id}: "
            f"{len(master_snapshot_base_features)} features"
        )

    # Load candidate pool only in legacy (selected-features) mode
    candidate_pool = None
    if not _all_features_mode and USE_UNION_POOL:
        from Train.candidate_pool import load_or_build_union_pool
        candidate_pool = load_or_build_union_pool(target_type, target_source, UNION_POOL_MAX)
        logger.info(f"Union candidate pool loaded: {len(candidate_pool)} features")

    tuned_params = None  # Cached tuned hyperparameters
    _warm_start_params = None  # Previous best params for Optuna warm-start after reselection

    # ── Fusion-CV tuning context (NSA branch only) ──
    # Built once per train_and_evaluate. Each tune call reuses this same
    # context — consensus PIT / SA actuals / adjustment history don't
    # change inside the backtest loop. None when NSA fusion-CV is disabled
    # or its dependencies fail to load (the tuner then falls back to the
    # legacy composite objective).
    _fusion_tuning_context: Optional[Dict] = None
    _fusion_objective_active = (
        str(target_type).lower() == 'nsa' and NSA_TUNE_USE_KALMAN_FUSION
    )
    if _fusion_objective_active and tune:
        _fusion_tuning_context = _build_fusion_tuning_context(X_full['ds'])
        if _fusion_tuning_context is None:
            logger.warning(
                "[FusionTune] Fusion context unavailable — NSA tuning falls "
                "back to the legacy composite objective."
            )

    # ── Joint Optuna context (NSA branch only, requires JOINT_OPTUNA=True) ──
    # When enabled, every reselection runs a single study over NSA params +
    # adjustment HL + Kalman params, then writes joint_tuned_params.json so
    # the post-hoc Kalman tune skips its own search.
    _joint_tuning_context: Optional[Dict] = None
    _joint_active = (
        JOINT_OPTUNA
        and str(target_type).lower() == 'nsa'
        and tune
    )
    if _joint_active:
        try:
            from Train.joint_tuning import build_joint_fusion_context
            _joint_tuning_context = build_joint_fusion_context(X_full['ds'])
        except Exception as e:
            logger.warning(f"[JointTune] context build failed ({e}); disabling joint tuning")
            _joint_tuning_context = None
        if _joint_tuning_context is None:
            logger.warning(
                "[JointTune] Joint Optuna requested but context unavailable — "
                "falling back to separate NSA + Kalman tunes."
            )
            _joint_active = False

    # ── Dynamic re-selection state ──
    dynamic_features: Optional[List[str]] = None   # None = not yet selected
    last_reselection_date: Optional[pd.Timestamp] = None
    _dynamic_selection_logs: List[Dict] = []  # JSON logs per reselection window

    def _months_since(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
        """Whole calendar-month count from ``earlier`` to ``later`` (signed).

        Used as the reselection cadence metric so that ``RESELECT_EVERY_N_MONTHS``
        means clean calendar months rather than 30-day approximations
        (the old ``days * 30`` heuristic drifted ~5 days per year, producing
        the off-by-one spacing visible in older ``_output/dynamic_selection``).
        """
        return (later.year - earlier.year) * 12 + (later.month - earlier.month)

    # ── Frozen-features mode ──
    # When master_snapshot_base_features is loaded (mode="selected") AND
    # reselection is disabled (RESELECT_EVERY_N_MONTHS == 0), use the saved
    # feature list verbatim. Skips both dynamic reselection and the legacy
    # static short-pass / branch-target FS extras — every backtest step uses
    # exactly the same pre-selected feature set.
    _frozen_features_mode = (
        not _all_features_mode
        and master_snapshot_base_features
        and RESELECT_EVERY_N_MONTHS == 0
    )
    if _frozen_features_mode:
        dynamic_features = list(master_snapshot_base_features)
        logger.info(
            f"[{model_id}] FROZEN-FEATURES mode: using {len(dynamic_features)} "
            f"pre-selected features verbatim. No reselection, no extras."
        )

    # ── Per-window features replay mode ──
    # When enabled, replay a prior dynamic-reselection run by loading saved
    # per-window JSONs and applying them at the matching backtest step.
    # Overrides frozen-features AND all-features modes when JSONs are present —
    # an explicit replay schedule is treated as a deliberate user choice that
    # takes precedence over automatic feature selection.
    _per_window_features_sets: List[Tuple[pd.Timestamp, List[str]]] = []
    _per_window_mode = USE_PER_WINDOW_FEATURES
    if _per_window_mode:
        _per_window_features_sets = _load_per_window_feature_sets(target_type, target_source)
        if not _per_window_features_sets:
            logger.warning(
                f"[{model_id}] PER-WINDOW mode requested but no JSONs found at "
                f"{OUTPUT_DIR / 'dynamic_selection' / f'{target_type}_{target_source}'}. "
                f"Falling back to frozen-features mode."
            )
            _per_window_mode = False
        else:
            _frozen_features_mode = False
            dynamic_features = None  # set per step inside loop
            _step_dates_str = ", ".join(d.strftime("%Y-%m") for d, _ in _per_window_features_sets)
            logger.info(
                f"[{model_id}] PER-WINDOW mode: loaded {len(_per_window_features_sets)} "
                f"feature sets at step_dates [{_step_dates_str}]. Backtest will replay "
                f"these sets at the matching steps; no fresh reselection."
                + (" (overriding ALL-FEATURES mode)" if _all_features_mode else "")
            )

    # ── Short-pass stability tracking ──
    step_feature_sets: list[set] = []
    step_jaccards: list[float] = []
    strategy_counts: Dict[str, int] = {}

    logger.info(f"Running {len(backtest_months)} predictions "
                f"({'with' if tune else 'without'} hyperparameter tuning)...")

    import time as _time
    _backtest_t0 = _time.time()

    for i, target_month in enumerate(backtest_months):
      with perf_phase("train.backtest.step", step=i, month=str(target_month)):
        # Get index of this target month in the full dataset
        target_idx = date_to_idx.get(target_month)
        if target_idx is None:
            continue

        with perf_phase("train.backtest.step.split_mask", step=i):
            # EXPANDING WINDOW: Training data is everything BEFORE the target month
            train_mask = X_full['ds'] < target_month
            train_idx = X_full[train_mask].index.tolist()

            if len(train_idx) < 24:  # Need at least 2 years of training data
                continue

            # Get training data (no future leakage)
            y_train = y_full.iloc[train_idx]

            # Filter out NaN targets from training data
            valid_train_mask = ~y_train.isna()
            train_idx_valid = [train_idx[j] for j in range(len(train_idx)) if valid_train_mask.iloc[j]]

            if len(train_idx_valid) < 24:
                continue

            # Get valid training data (LightGBM handles NaN natively)
            X_train_valid = X_full.iloc[train_idx_valid].copy()
            y_train_valid = y_train[valid_train_mask].copy()

            # Sort by date to ensure time-ordered train/val split
            sort_order = X_train_valid['ds'].argsort()
            X_train_valid = X_train_valid.iloc[sort_order].reset_index(drop=True)
            y_train_valid = y_train_valid.iloc[sort_order].reset_index(drop=True)

        # NOTE: COVID winsorization is now applied ONCE upfront on X_full / y_full
        # (just after build_training_dataset) so X_train_valid is already winsorized
        # by virtue of being a slice of X_full. The prior per-step winsorize block
        # was removed because (a) it duplicated work on every step and (b) it left
        # X_pred (taken directly from X_full) unwinsorized, creating an
        # asymmetry between training inputs and prediction inputs at COVID
        # target months.

        # Compute baseline predictions using only training data
        baseline_preds = {}
        if ENABLE_BASELINE_TRACKING:
            from Train.baselines import compute_all_baselines
            baseline_preds = compute_all_baselines(y_train_valid, BASELINE_ROLLING_WINDOW)

        with perf_phase("train.backtest.step.clean_features", step=i):
            # Recompute clean_features every step (feature availability changes as window expands)
            cleaned_features = clean_features(X_train_valid, y_train_valid)
            cleaned_features = [c for c in cleaned_features if c in X_train_valid.columns and c != 'ds']

        # ── Per-window feature swap (replay) ──
        # In per-window mode, pick the saved feature set whose step_date is the
        # most recent <= the current target_month. When the set changes vs the
        # previous step, invalidate Optuna params (preserve previous best as
        # warm-start seed) — mirrors what fresh reselection does.
        if _per_window_mode:
            new_features = _get_features_for_step(_per_window_features_sets, target_month)
            if new_features is None:
                raise RuntimeError(
                    f"[Step {i}] PER-WINDOW mode: no feature set found with "
                    f"step_date <= {target_month.strftime('%Y-%m')}. Earliest "
                    f"available: {_per_window_features_sets[0][0].strftime('%Y-%m')}"
                )
            if dynamic_features != new_features:
                if dynamic_features is not None:
                    logger.info(
                        f"[Step {i}] PER-WINDOW switch at "
                        f"{target_month.strftime('%Y-%m')}: feature set changed "
                        f"(now {len(new_features)} features). Invalidating Optuna "
                        f"params (warm-start preserved)."
                    )
                    _warm_start_params = tuned_params
                    tuned_params = None
                dynamic_features = list(new_features)

        # ── Dynamic re-selection (anchored on the latest backtest month) ──
        # The reselection schedule is anchored on the LAST backtest month
        # (the prediction we care about most) and extends backward through
        # history at N-month intervals: {anchor, anchor-N, anchor-2N, ...}.
        # This makes the live prediction month itself a reselection step,
        # so features are freshly selected and re-tuned for the target we
        # are actually about to predict, instead of using a feature set that
        # was last refreshed at an arbitrary point N months earlier.
        #
        # In all-features mode: mandatory bootstrap on step 0 (so the model
        # has features at all); subsequent reselections fire only when
        # target_month is on the schedule AND >= RESELECTION_START_DATE.
        # In legacy mode: optional, triggered only when RESELECT_EVERY_N_MONTHS > 0.
        # Per-window replay mode short-circuits this entirely — the user-curated
        # schedule from disk is authoritative.
        _reselection_start = pd.Timestamp(RESELECTION_START_DATE)
        _reselection_anchor = backtest_months[-1]

        def _is_scheduled_reselection_month(month: pd.Timestamp) -> bool:
            """True iff ``month`` lies on {anchor - k*N : k=0,1,2,...}."""
            if RESELECT_EVERY_N_MONTHS <= 0:
                return False
            diff = _months_since(_reselection_anchor, month)
            return diff >= 0 and diff % RESELECT_EVERY_N_MONTHS == 0

        _trigger_reselection = False
        if _per_window_mode:
            _trigger_reselection = False
        elif _all_features_mode:
            _trigger_reselection = (
                dynamic_features is None  # First step — bootstrap select
                or (
                    target_month >= _reselection_start
                    and _is_scheduled_reselection_month(target_month)
                )
            )
        elif RESELECT_EVERY_N_MONTHS > 0:
            _trigger_reselection = (
                last_reselection_date is None
                or (
                    target_month >= _reselection_start
                    and _is_scheduled_reselection_month(target_month)
                )
            )

        if _trigger_reselection:
            with perf_phase("train.backtest.step.dynamic_reselection", step=i):
                _dyn_result = _dynamic_reselection(
                    X_train=X_train_valid,
                    y_train=y_train_valid,
                    step_date=target_month,
                    target_type=target_type,
                    target_source=target_source,
                )
                # _dynamic_reselection raises RuntimeError on failure in all-features mode
                dynamic_features = _dyn_result
                last_reselection_date = target_month
                _next_scheduled = target_month + pd.DateOffset(months=RESELECT_EVERY_N_MONTHS)
                _next_str = (
                    _next_scheduled.strftime('%Y-%m')
                    if _next_scheduled <= _reselection_anchor
                    else f"none — {target_month.strftime('%Y-%m')} is the anchor"
                )
                logger.info(
                    f"[Step {i}] Dynamic re-selection: {len(dynamic_features)} features "
                    f"(anchor={_reselection_anchor.strftime('%Y-%m')}, "
                    f"next scheduled re-selection: {_next_str})"
                )

                # Write JSON log for this reselection window
                _sel_log = {
                    "target_type": target_type,
                    "target_source": target_source,
                    "step_date": target_month.strftime('%Y-%m'),
                    "step_index": i,
                    "n_features": len(dynamic_features),
                    "features": dynamic_features,
                }
                _dynamic_selection_logs.append(_sel_log)
                _sel_log_dir = OUTPUT_DIR / "dynamic_selection" / f"{target_type}_{target_source}"
                _sel_log_dir.mkdir(parents=True, exist_ok=True)
                _sel_log_path = _sel_log_dir / f"{target_month.strftime('%Y-%m')}.json"
                try:
                    import json as _json
                    with open(_sel_log_path, 'w') as _f:
                        _json.dump(_sel_log, _f, indent=2)
                    logger.info(f"[Step {i}] Selection log → {_sel_log_path}")
                except Exception as _e:
                    logger.warning(f"[Step {i}] Failed to write selection log: {_e}")

                # Invalidate Optuna params — feature set changed, must re-tune.
                # Preserve previous best as warm-start seed for faster convergence.
                _warm_start_params = tuned_params  # may be None on first reselection
                tuned_params = None
                logger.info(f"[Step {i}] Optuna params invalidated after reselection"
                            f"{' (warm-start from prior best)' if _warm_start_params else ''}")

        # Compute sample weights (needed by both dynamic and static paths)
        default_half_life = 60.0
        weights = calculate_sample_weights(X_train_valid, target_month, default_half_life)
        weights = _apply_tail_aware_weighting(
            weights,
            y_train_valid.values.astype(float),
            target_type,
            target_source,
        )

        # ── Feature selection for this step ──
        if dynamic_features is not None:
            # Dynamic features are active — use them directly
            feature_cols = [
                c for c in dynamic_features
                if c in X_train_valid.columns and c in cleaned_features
            ]
            if not feature_cols:
                if _all_features_mode:
                    raise RuntimeError(
                        f"[Step {i}] Dynamic features have zero overlap with cleaned features. "
                        f"Dynamic: {len(dynamic_features)}, Cleaned: {len(cleaned_features)}. "
                        f"Cannot proceed in all-features mode."
                    )
                logger.warning(
                    f"[Step {i}] Dynamic features have no overlap with cleaned features; "
                    f"falling back to static pipeline."
                )
                dynamic_features = None  # Fall through to static pipeline

        if dynamic_features is None:
            if _all_features_mode:
                # Should never reach here — dynamic reselection is mandatory
                raise RuntimeError(
                    f"[Step {i}] All-features mode requires dynamic reselection, "
                    f"but no features are available. This is a bug."
                )

            # ── Legacy static feature pipeline ──
            groups = partition_feature_columns(cleaned_features, target_type=target_type)
            snapshot_candidates = groups['snapshot_features']
            branch_target_candidates = groups['target_branch_features']
            dropped_cross_target = groups['other_target_features']
            cal_feats = groups['calendar_features']
            if target_type == 'sa':
                cal_feats = [f for f in cal_feats if f in SA_CALENDAR_FEATURES_KEEP]
            always_keep = _merge_unique_feature_lists(
                cal_feats,
                groups['revision_features'],
            )

            snapshot_base_features = [
                f for f in master_snapshot_base_features if f in X_train_valid.columns and f != 'ds'
            ]
            if not snapshot_base_features:
                snapshot_base_features = [c for c in snapshot_candidates if c in X_train_valid.columns]

            _current_params = tuned_params if (tune and tuned_params is not None) else static_params
            sp_half_life = SHORTPASS_HALF_LIFE or _current_params.get('half_life_months', default_half_life)
            sp_weights = calculate_sample_weights(X_train_valid, target_month, sp_half_life)

            snapshot_selected = snapshot_base_features
            snapshot_base_set = set(snapshot_base_features)
            snapshot_extra_candidates = [
                c for c in snapshot_candidates if c in X_train_valid.columns and c not in snapshot_base_set
            ]

            if USE_UNION_POOL and candidate_pool is not None:
                candidate_in_data = [f for f in candidate_pool if f in snapshot_extra_candidates]
                if candidate_in_data:
                    if len(candidate_in_data) > SHORTPASS_TOPK:
                        with perf_phase("train.backtest.step.short_pass", step=i):
                            from Train.short_pass_selection import select_features_for_step
                            snapshot_extra_selected = select_features_for_step(
                                X_train_valid[candidate_in_data], y_train_valid,
                                candidate_features=candidate_in_data,
                                top_k=SHORTPASS_TOPK,
                                method=SHORTPASS_METHOD,
                                sample_weights=sp_weights,
                            )
                    else:
                        snapshot_extra_selected = candidate_in_data
                    snapshot_selected = _merge_unique_feature_lists(snapshot_base_features, snapshot_extra_selected)

            branch_fs_top_k = _get_branch_target_fs_top_k(target_type, target_source)
            branch_fs_method = _get_branch_target_fs_method(target_type, target_source)
            if USE_BRANCH_TARGET_FS and branch_target_candidates:
                with perf_phase("train.backtest.step.branch_target_fs", step=i):
                    branch_target_selected = select_branch_target_features_for_step(
                        X_train=X_train_valid,
                        y_train=y_train_valid,
                        target_type=target_type,
                        candidate_features=branch_target_candidates,
                        top_k=branch_fs_top_k,
                        method=branch_fs_method,
                        corr_threshold=BRANCH_TARGET_FS_CORR_THRESHOLD,
                        min_overlap=BRANCH_TARGET_FS_MIN_OVERLAP,
                        sample_weights=sp_weights,
                        dynamics_weight_level=BRANCH_TARGET_FS_WEIGHT_LEVEL,
                        dynamics_weight_diff=BRANCH_TARGET_FS_WEIGHT_DIFF,
                        dynamics_weight_dir=BRANCH_TARGET_FS_WEIGHT_DIR,
                        dynamics_weight_amp=BRANCH_TARGET_FS_WEIGHT_AMP,
                        dynamics_weight_sign=BRANCH_TARGET_FS_WEIGHT_SIGN,
                        dynamics_weight_tail=BRANCH_TARGET_FS_WEIGHT_TAIL,
                    )
            else:
                branch_target_selected = branch_target_candidates

            feature_cols = _merge_unique_feature_lists(
                snapshot_selected,
                branch_target_selected,
                always_keep,
            )

        logger.info(f"[Step {i}] Feature pipeline: {len(feature_cols)} features")

        if not feature_cols:
            logger.warning("No features selected for this step; skipping month.")
            continue

        # ── Inject NSA acceleration features for SA branch (PIT-safe) ──
        if target_type == 'sa' and nsa_backtest_results is not None:
            from Train.nsa_acceleration import (
                compute_nsa_acceleration_features,
                build_nsa_features_for_training,
            )
            # Cutoff = SA's first NFP release date for this target_month, matching
            # the strict-< cutoff that build_training_dataset already uses for all
            # other features. Closes Issue 10's same-day leak by filtering revised
            # actuals via operational_available_date < cutoff_date.
            cutoff_for_target = target_release_date_map.get(target_month, target_month)
            nsa_feats_pred = compute_nsa_acceleration_features(
                nsa_backtest_results, target_month, cutoff_date=cutoff_for_target,
            )
            _nsa_accel_cols = list(nsa_feats_pred.keys())

            training_months = pd.to_datetime(X_train_valid['ds'])
            nsa_train_feats = build_nsa_features_for_training(
                nsa_backtest_results, training_months,
                cutoff_dates=target_release_date_map,
            )
            for col in _nsa_accel_cols:
                if col in nsa_train_feats.columns:
                    X_train_valid[col] = nsa_train_feats[col].reindex(training_months).values
                else:
                    X_train_valid[col] = np.nan
            for col, val in nsa_feats_pred.items():
                X_full.at[target_idx, col] = val

            feature_cols = _merge_unique_feature_lists(feature_cols, _nsa_accel_cols)
            logger.info(f"[Step {i}] Injected {len(_nsa_accel_cols)} NSA acceleration features")

        # ── Track short-pass stability (Jaccard similarity) ──
        current_set = set(feature_cols)
        step_feature_sets.append(current_set)
        if len(step_feature_sets) >= 2:
            prev = step_feature_sets[-2]
            union_size = len(prev | current_set)
            jaccard = len(prev & current_set) / union_size if union_size > 0 else 1.0
            step_jaccards.append(jaccard)
            logger.info(f"Short-pass Jaccard vs prev step: {jaccard:.3f}")

        # Prepare training data with cleaned features
        X_train_selected = X_train_valid[feature_cols]
        # X_train_selected contains NO 'ds' column to avoid issues in lgb.train
        X_train_valid_with_ds = X_train_valid.copy() # keep one with ds

        # NOTE: Feature selection (short-pass) is frozen for this step.
        # Optuna tunes hyperparams on the selected features. Inner CV scores may be
        # slightly optimistic due to selection-validation overlap. Only outer walk-forward
        # OOS metrics are trusted for model quality evaluation.
        if tune and (tuned_params is None or i % TUNE_EVERY_N_MONTHS == 0):
            logger.info(f"[{i+1}/{len(backtest_months)}] Tuning hyperparameters on "
                        f"{len(X_train_selected)} samples, {len(feature_cols)} features...")

            X_train_for_tuning = X_train_valid_with_ds[['ds'] + feature_cols]

            with perf_phase("train.backtest.step.tuning", step=i):
                tuning_mode = _get_tuning_objective_mode(target_type, target_source)
                # Fall back to composite if the fusion context didn't load.
                if tuning_mode == 'kalman_fusion' and _fusion_tuning_context is None:
                    tuning_mode = TUNING_OBJECTIVE_MODE_DEFAULT

                if _joint_active and _joint_tuning_context is not None and tuning_mode == 'kalman_fusion':
                    # Joint Optuna: single study over NSA + HL + Kalman params.
                    from Train.joint_tuning import tune_full_pipeline_joint
                    joint_result = tune_full_pipeline_joint(
                        X_train_for_tuning, y_train_valid, target_month=target_month,
                        fusion_context=_joint_tuning_context,
                        use_huber_loss=use_huber_loss,
                        warm_start_params=_warm_start_params,
                    )
                    tuned_params = joint_result['lgbm_params']
                    _save_joint_tuned_params(joint_result, step_date=target_month)
                else:
                    tuned_params = tune_hyperparameters(
                        X_train_for_tuning, y_train_valid, target_month=target_month,
                        use_huber_loss=use_huber_loss,
                        objective_mode=tuning_mode,
                        lambda_std_ratio=TUNING_LAMBDA_STD_RATIO,
                        lambda_diff_std_ratio=TUNING_LAMBDA_DIFF_STD_RATIO,
                        lambda_tail_mae=TUNING_LAMBDA_TAIL_MAE,
                        lambda_corr_diff=TUNING_LAMBDA_CORR_DIFF,
                        lambda_diff_sign=TUNING_LAMBDA_DIFF_SIGN,
                        tail_quantile=VARIANCE_TAIL_QUANTILE,
                        tail_weighting=(
                            ENABLE_TAIL_AWARE_WEIGHTING
                            and _is_variance_priority_target(target_type, target_source)
                        ),
                        tail_weight_abs_level_quantile=TAIL_WEIGHT_ABS_LEVEL_QUANTILE,
                        tail_weight_abs_diff_quantile=TAIL_WEIGHT_ABS_DIFF_QUANTILE,
                        tail_weight_level_boost=TAIL_WEIGHT_LEVEL_BOOST,
                        tail_weight_diff_boost=TAIL_WEIGHT_DIFF_BOOST,
                        tail_weight_max_multiplier=TAIL_WEIGHT_MAX_MULTIPLIER,
                        warm_start_params=_warm_start_params,
                        fusion_context=(
                            _fusion_tuning_context if tuning_mode == 'kalman_fusion' else None
                        ),
                    )
                _warm_start_params = None  # consumed

        params = tuned_params if tune and tuned_params is not None else static_params
        
        # Determine the final half_life_months to use for this expanding window iteration model
        final_half_life = params.get('half_life_months', default_half_life)
        
        # Recompute final training weights for the main model fit using the chosen half_life
        weights = calculate_sample_weights(X_train_valid_with_ds, target_month, final_half_life)
        weights = _apply_tail_aware_weighting(
            weights,
            y_train_valid.values.astype(float),
            target_type,
            target_source,
        )

        with perf_phase("train.backtest.step.fit", step=i):
            # Train-validation split (data already sorted by date above)
            train_size = int(len(X_train_selected) * 0.85)
            X_tr = X_train_selected.iloc[:train_size]
            X_val = X_train_selected.iloc[train_size:]
            y_tr = y_train_valid.iloc[:train_size]
            y_val = y_train_valid.iloc[train_size:]
            weights_tr = weights[:train_size]

            train_data = lgb.Dataset(X_tr, label=y_tr, weight=weights_tr, free_raw_data=False)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=0)
            ]

            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )

        # OOS residuals accumulated after prediction below (not in-sample)

        with perf_phase("train.backtest.step.predict", step=i):
            # PREDICTION: Get features for target month
            X_pred = X_full.iloc[[target_idx]].copy()
            X_pred = X_pred[feature_cols]

            # Base prediction
            base_prediction = float(model.predict(X_pred)[0])
            actual = y_full.iloc[target_idx]

            # Sequential variance enhancement stack (selected by validation composite score)
            final_prediction = base_prediction
            selected_val_preds = model.predict(X_val)
            selected_strategy = 'base'
            stage_report = {}
            if len(X_val) >= 5:
                final_prediction, selected_val_preds, selected_strategy, stage_report = _run_variance_enhancement_sequence(
                    base_model=model,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_val=X_val,
                    y_val=y_val,
                    X_pred=X_pred,
                    y_train_valid=y_train_valid,
                    target_type=target_type,
                    target_source=target_source,
                )
            prediction = final_prediction
            strategy_counts[selected_strategy] = strategy_counts.get(selected_strategy, 0) + 1

        with perf_phase("train.backtest.step.intervals", step=i):
            # Calculate prediction intervals
            if len(all_residuals) > 10:
                residual_array = np.array(all_residuals[-36:])
                lower_50 = prediction + np.percentile(residual_array, 25)
                upper_50 = prediction + np.percentile(residual_array, 75)
                lower_80 = prediction + np.percentile(residual_array, 10)
                upper_80 = prediction + np.percentile(residual_array, 90)
                lower_95 = prediction + np.percentile(residual_array, 2.5)
                upper_95 = prediction + np.percentile(residual_array, 97.5)
            else:
                # Not enough OOS residuals yet; use validation residuals from selected strategy
                val_residuals = (y_val.values - selected_val_preds).tolist()
                std_est = np.std(val_residuals) if val_residuals else 50
                lower_50, upper_50 = prediction - 0.67*std_est, prediction + 0.67*std_est
                lower_80, upper_80 = prediction - 1.28*std_est, prediction + 1.28*std_est
                lower_95, upper_95 = prediction - 1.96*std_est, prediction + 1.96*std_est

            # Handle NaN actuals (future predictions)
            is_future = pd.isna(actual)
            error = np.nan if is_future else actual - prediction

            # Accumulate OOS residuals for calibrated prediction intervals
            if not is_future:
                all_residuals.append(error)
            in_50 = np.nan if is_future else (lower_50 <= actual <= upper_50)
            in_80 = np.nan if is_future else (lower_80 <= actual <= upper_80)
            in_95 = np.nan if is_future else (lower_95 <= actual <= upper_95)

            # ── P1-1: Directional accuracy ──
            dir_correct = np.nan if is_future else int(np.sign(actual) == np.sign(prediction))

            # ── P1-2: Directional acceleration accuracy ──
            # Operational "vs last actual" formula: at trade time the prior
            # month's actual is known, so the metric is whether the predicted
            # MoM CHANGE — sign(prediction - last_actual) — matches the actual
            # MoM change sign(actual - last_actual). Replaces the older
            # "sign(pred - last_pred)" formulation which was contaminated by
            # the prior month's prediction error.
            if not is_future and not np.isnan(last_actual):
                accel_actual = float(actual) - last_actual
                accel_pred   = float(prediction) - last_actual
                accel_correct = int(np.sign(accel_actual) == np.sign(accel_pred))
            else:
                accel_correct = np.nan

        result_row = {
            'ds': target_month,
            'actual': actual,
            'predicted': prediction,
            'error': error,
            'lower_50': lower_50,
            'upper_50': upper_50,
            'lower_80': lower_80,
            'upper_80': upper_80,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'in_50_interval': in_50,
            'in_80_interval': in_80,
            'in_95_interval': in_95,
            'n_train_samples': len(train_idx_valid),
            'n_features': len(feature_cols),
            'dir_correct': dir_correct,
            'accel_correct': accel_correct,
            'prediction_strategy': selected_strategy,
        }
        if stage_report:
            result_row['strategy_selected_score'] = stage_report.get('selected_score', np.nan)
            result_row['strategy_applied_count'] = len(stage_report.get('applied', []))

        # Add baseline predictions and errors (error = actual - baseline_pred)
        if ENABLE_BASELINE_TRACKING:
            for bname, bpred in baseline_preds.items():
                result_row[bname] = bpred
                if not is_future and not np.isnan(bpred):
                    result_row[f'{bname}_error'] = actual - bpred
                else:
                    result_row[f'{bname}_error'] = np.nan

        results.append(result_row)

        # Update acceleration tracking var
        if not is_future:
            last_actual = float(actual)

        # Progress logging — every prediction with elapsed/ETA
        _elapsed = _time.time() - _backtest_t0
        _avg_per_step = _elapsed / (i + 1)
        _eta = _avg_per_step * (len(backtest_months) - i - 1)
        _eta_str = f"{_eta/60:.1f}m" if _eta >= 60 else f"{_eta:.0f}s"
        _elapsed_str = f"{_elapsed/60:.1f}m" if _elapsed >= 60 else f"{_elapsed:.0f}s"

        if is_future:
            logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                        f"Pred={prediction:.0f} (FUTURE, strategy={selected_strategy}) | "
                        f"train={len(train_idx_valid)}, feats={len(feature_cols)} | "
                        f"elapsed={_elapsed_str}, ETA={_eta_str}")
        else:
            logger.info(f"[{i+1}/{len(backtest_months)}] {target_month.strftime('%Y-%m')}: "
                        f"Actual={actual:.0f}, Pred={prediction:.0f}, Err={error:+.0f}, "
                        f"strategy={selected_strategy} | "
                        f"train={len(train_idx_valid)}, feats={len(feature_cols)} | "
                        f"elapsed={_elapsed_str}, ETA={_eta_str}")

    results_df = pd.DataFrame(results)

    # Log summary statistics
    if not results_df.empty:
        backtest_rows = results_df[~results_df['error'].isna()].copy()
        future_rows = results_df[results_df['error'].isna()]

        if not backtest_rows.empty:
            rmse = np.sqrt(np.mean(backtest_rows['error'] ** 2))
            mae  = np.mean(np.abs(backtest_rows['error']))
            bias = backtest_rows['error'].mean()
            dir_acc   = backtest_rows['dir_correct'].dropna().mean()
            accel_acc = backtest_rows['accel_correct'].dropna().mean()
            variance_kpis = compute_variance_kpis(
                backtest_rows['actual'].values.astype(float),
                backtest_rows['predicted'].values.astype(float),
                tail_quantile=VARIANCE_TAIL_QUANTILE,
                extreme_quantile=VARIANCE_EXTREME_QUANTILE,
            )

            logger.info("\n" + "=" * 60)
            logger.info("EXPANDING WINDOW BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Predictions: {len(backtest_rows)} backtest, {len(future_rows)} future")
            logger.info(f"RMSE: {rmse:.2f}  MAE: {mae:.2f}  Bias: {bias:+.2f}")
            logger.info(f"Directional Accuracy:      {dir_acc:.1%}")
            logger.info(f"Acceleration Accuracy:     {accel_acc:.1%}")
            logger.info(
                f"Variance KPIs: std_ratio={variance_kpis['std_ratio']:.3f}, "
                f"diff_std_ratio={variance_kpis['diff_std_ratio']:.3f}, "
                f"corr_diff={variance_kpis['corr_diff']:.3f}, "
                f"diff_sign_accuracy={variance_kpis['diff_sign_accuracy']:.1%}, "
                f"tail_mae={variance_kpis['tail_mae']:.2f}, "
                f"extreme_hit_rate={variance_kpis['extreme_hit_rate']:.1%}"
            )
            logger.info(f"Coverage: 50%={backtest_rows['in_50_interval'].mean()*100:.1f}%  "
                        f"80%={backtest_rows['in_80_interval'].mean()*100:.1f}%  "
                        f"95%={backtest_rows['in_95_interval'].mean()*100:.1f}%")

            if strategy_counts:
                logger.info(f"Strategy usage counts: {strategy_counts}")

            # ── P1-4: Bias warning ──
            if abs(bias) > 0.2 * mae:
                logger.warning(f"BIAS WARNING: |bias|/MAE = {abs(bias)/mae:.0%} — model is systematically "
                               f"{'under' if bias < 0 else 'over'}-predicting")

            # ── P1-7: Calibration warnings ──
            for nominal, col in [
                (0.50, 'in_50_interval'), (0.80, 'in_80_interval'), (0.95, 'in_95_interval')
            ]:
                empirical = backtest_rows[col].dropna().mean()
                if abs(nominal - empirical) > 0.10:
                    logger.warning(
                        f"CALIBRATION WARNING: {nominal:.0%} nominal → {empirical:.1%} empirical coverage "
                        f"({'anti-conservative' if empirical < nominal else 'conservative'})"
                    )

            # ── P1-3: Stratified metrics ──
            logger.info("\nStratified Metrics:")
            strata = [
                ('ALL',                    pd.Series(True,  index=backtest_rows.index)),
                ('non-COVID',              backtest_rows['ds'].dt.year != 2020),
                ('post-2021',              backtest_rows['ds'] >= pd.Timestamp('2021-01-01')),
                ('large-print (|y|>200K)', backtest_rows['actual'].abs() > 200),
            ]
            for label, mask in strata:
                sub = backtest_rows[mask]
                if sub.empty:
                    continue
                s_rmse  = np.sqrt((sub['error'] ** 2).mean())
                s_mae   = sub['error'].abs().mean()
                s_dir   = sub['dir_correct'].dropna().mean()
                s_accel = sub['accel_correct'].dropna().mean()
                logger.info(
                    f"  [{label}] n={len(sub)}  RMSE={s_rmse:.1f}  MAE={s_mae:.1f}  "
                    f"Dir={s_dir:.1%}  Accel={s_accel:.1%}"
                )

            # ── P1-5: Rolling 12-month MAE ──
            backtest_rows['rolling_12m_mae'] = (
                backtest_rows['error'].abs().rolling(12, min_periods=6).mean()
            )
            trailing_mae = backtest_rows['rolling_12m_mae'].iloc[-1]
            logger.info(f"\nTrailing-12m MAE: {trailing_mae:.1f}K")
            results_df = results_df.merge(
                backtest_rows[['ds', 'rolling_12m_mae']], on='ds', how='left'
            )

            # Store aggregate metrics for JSON persistence (accessed in train_and_evaluate)
            results_df.attrs['summary_metrics'] = {
                'rmse': float(rmse), 'mae': float(mae), 'bias': float(bias),
                'dir_acc': float(dir_acc), 'accel_acc': float(accel_acc),
                'coverage_50': float(backtest_rows['in_50_interval'].mean()),
                'coverage_80': float(backtest_rows['in_80_interval'].mean()),
                'coverage_95': float(backtest_rows['in_95_interval'].mean()),
                'trailing_12m_mae': float(trailing_mae),
                'n_backtest': int(len(backtest_rows)),
                'std_ratio': float(variance_kpis['std_ratio']),
                'diff_std_ratio': float(variance_kpis['diff_std_ratio']),
                'corr_level': float(variance_kpis['corr_level']),
                'corr_diff': float(variance_kpis['corr_diff']),
                'diff_sign_accuracy': float(variance_kpis['diff_sign_accuracy']),
                'tail_mae': float(variance_kpis['tail_mae']),
                'extreme_hit_rate': float(variance_kpis['extreme_hit_rate']),
                'strategy_counts': dict(strategy_counts),
            }

        if not future_rows.empty:
            logger.info("\nFuture Predictions:")
            for _, row in future_rows.iterrows():
                logger.info(f"  {row['ds'].strftime('%Y-%m')}: {row['predicted']:.0f} [{row['lower_80']:.0f}, {row['upper_80']:.0f}]")

    # ── Keep-rule enforcement ──
    if KEEP_RULE_ENABLED and ENABLE_BASELINE_TRACKING and not results_df.empty:
        backtest_only = results_df[~results_df['error'].isna()].copy()

        if len(backtest_only) >= KEEP_RULE_WINDOW_M:
            trailing = backtest_only.iloc[-KEEP_RULE_WINDOW_M:]
            trailing_indices = trailing.index

            trailing_model_mae = np.mean(np.abs(trailing['error']))

            baseline_error_cols = [c for c in trailing.columns
                                   if c.startswith('baseline_') and c.endswith('_error')]

            for col in baseline_error_cols:
                bname = col.replace('_error', '')
                valid_errors = trailing[col].dropna()
                if valid_errors.empty:
                    continue

                trailing_baseline_mae = np.mean(np.abs(valid_errors))
                degradation = trailing_model_mae - trailing_baseline_mae

                if degradation > KEEP_RULE_TOLERANCE:
                    logger.warning(
                        f"KEEP-RULE TRIGGERED for '{bname}': "
                        f"Model MAE={trailing_model_mae:.2f} vs Baseline MAE={trailing_baseline_mae:.2f} "
                        f"over last {KEEP_RULE_WINDOW_M} OOS months "
                        f"(degradation={degradation:+.2f} > tolerance={KEEP_RULE_TOLERANCE})"
                    )

                    if KEEP_RULE_ACTION == 'fail':
                        raise RuntimeError(
                            f"Keep-rule failed: model underperforms '{bname}' by "
                            f"{degradation:.2f} MAE. Action='fail'. Aborting."
                        )
                    elif KEEP_RULE_ACTION == 'fallback_to_baseline':
                        logger.warning(
                            f"Action='fallback_to_baseline': replacing predictions "
                            f"for trailing {KEEP_RULE_WINDOW_M} rows with '{bname}'"
                        )
                        results_df.loc[trailing_indices, 'predicted'] = results_df.loc[trailing_indices, bname]
                        results_df.loc[trailing_indices, 'error'] = (
                            results_df.loc[trailing_indices, 'actual']
                            - results_df.loc[trailing_indices, 'predicted']
                        )
                        results_df.loc[trailing_indices, 'keep_rule_fallback'] = True
                    elif KEEP_RULE_ACTION == 'skip_save':
                        logger.warning(
                            f"Action='skip_save': production model will NOT be saved"
                        )
                        results_df.attrs['skip_save'] = True
                        results_df.attrs['keep_rule_failed'] = True

                    break  # Enforce against first triggering baseline only
                else:
                    logger.info(
                        f"Keep-rule OK for '{bname}': Model MAE={trailing_model_mae:.2f} "
                        f"vs Baseline MAE={trailing_baseline_mae:.2f} "
                        f"(improvement={-degradation:.2f})"
                    )
        else:
            logger.info(
                f"Keep-rule skipped: only {len(backtest_only)} OOS months "
                f"(need >= {KEEP_RULE_WINDOW_M})"
            )

    # ── Variance gate enforcement (priority targets) ──
    if ENABLE_VARIANCE_GATE and not results_df.empty and _is_variance_priority_target(target_type, target_source):
        backtest_only = results_df[~results_df['error'].isna()].copy()
        if not backtest_only.empty:
            vk = compute_variance_kpis(
                backtest_only['actual'].values.astype(float),
                backtest_only['predicted'].values.astype(float),
                tail_quantile=VARIANCE_TAIL_QUANTILE,
                extreme_quantile=VARIANCE_EXTREME_QUANTILE,
            )
            gate_fail_reasons = []
            if vk['std_ratio'] < VARIANCE_GATE_MIN_STD_RATIO:
                gate_fail_reasons.append(
                    f"std_ratio={vk['std_ratio']:.3f} < {VARIANCE_GATE_MIN_STD_RATIO:.3f}"
                )
            if vk['diff_std_ratio'] < VARIANCE_GATE_MIN_DIFF_STD_RATIO:
                gate_fail_reasons.append(
                    f"diff_std_ratio={vk['diff_std_ratio']:.3f} < {VARIANCE_GATE_MIN_DIFF_STD_RATIO:.3f}"
                )
            if vk['corr_diff'] < VARIANCE_GATE_MIN_CORR_DIFF:
                gate_fail_reasons.append(
                    f"corr_diff={vk['corr_diff']:.3f} < {VARIANCE_GATE_MIN_CORR_DIFF:.3f}"
                )
            if vk['diff_sign_accuracy'] < VARIANCE_GATE_MIN_DIFF_SIGN_ACC:
                gate_fail_reasons.append(
                    f"diff_sign_accuracy={vk['diff_sign_accuracy']:.3f} < {VARIANCE_GATE_MIN_DIFF_SIGN_ACC:.3f}"
                )
            if vk['extreme_hit_rate'] < VARIANCE_GATE_MIN_EXTREME_HIT_RATE:
                gate_fail_reasons.append(
                    f"extreme_hit_rate={vk['extreme_hit_rate']:.3f} < {VARIANCE_GATE_MIN_EXTREME_HIT_RATE:.3f}"
                )

            if gate_fail_reasons:
                logger.warning(
                    "VARIANCE GATE FAILED for %s: %s",
                    model_id,
                    "; ".join(gate_fail_reasons),
                )
                results_df.attrs['skip_save'] = True
                results_df.attrs['variance_gate_failed'] = True
                results_df.attrs['variance_gate_reasons'] = gate_fail_reasons
            else:
                logger.info(
                    "Variance gate passed for %s "
                    "(std_ratio=%.3f, diff_std_ratio=%.3f, corr_diff=%.3f, diff_sign_accuracy=%.3f, extreme_hit_rate=%.3f)",
                    model_id,
                    vk['std_ratio'],
                    vk['diff_std_ratio'],
                    vk['corr_diff'],
                    vk['diff_sign_accuracy'],
                    vk['extreme_hit_rate'],
                )

    # ── Short-pass stability summary ──
    if step_jaccards:
        mean_j = np.mean(step_jaccards)
        min_j = np.min(step_jaccards)
        max_j = np.max(step_jaccards)
        logger.info(f"Short-pass stability: Jaccard mean={mean_j:.3f}, "
                    f"min={min_j:.3f}, max={max_j:.3f} "
                    f"({len(step_jaccards)} transitions)")

        # Feature tenure: count how many steps each feature was selected
        from collections import Counter as _Counter
        tenure = _Counter()
        for fs in step_feature_sets:
            tenure.update(fs)
        top_10 = tenure.most_common(10)
        logger.info(f"Most stable features (top 10 by tenure): "
                    f"{[(f, c) for f, c in top_10]}")

        # Save stability report
        import json as _json
        stability_dir = OUTPUT_DIR / "models" / "lightgbm_nfp" / model_id
        stability_dir.mkdir(parents=True, exist_ok=True)
        stability_path = stability_dir / "shortpass_stability.json"
        stability_data = {
            "jaccard_mean": round(mean_j, 4),
            "jaccard_min": round(min_j, 4),
            "jaccard_max": round(max_j, 4),
            "n_steps": len(step_feature_sets),
            "top_10_by_tenure": [(f, c) for f, c in top_10],
            "full_tenure": dict(tenure.most_common()),
        }
        with open(stability_path, 'w') as _f:
            _json.dump(stability_data, _f, indent=2)
        logger.info(f"Saved stability report to {stability_path}")

    return results_df, X_full, y_full


@profiled("train.train_and_evaluate")
def train_and_evaluate(
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'revised',
    use_huber_loss: bool = USE_HUBER_LOSS_DEFAULT,
    huber_delta: float = HUBER_DELTA,
    tune: bool = True,
    nsa_backtest_results: Optional[pd.DataFrame] = None,
):
    """
    Main training and evaluation function using EXPANDING WINDOW methodology.

    This function ensures NO TIME-TRAVEL VIOLATIONS:
    1. LightGBM handles NaN natively - no imputation needed
    2. Model training uses only data available at each prediction time
    3. Hyperparameters tuned via inner CV (no future leakage)

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised
        target_source: 'revised' (from M+1 FRED snapshot)
        use_huber_loss: If True, use Huber loss function
        huber_delta: Huber delta parameter
        tune: If True, run Optuna hyperparameter tuning
        nsa_backtest_results: Optional NSA backtest results for SA branch
            NSA acceleration feature injection (PIT-safe).
    """
    model_id = get_model_id(target_type, release_type, target_source)

    # Register atexit perf dump so data is captured even if training exits early
    register_atexit_dump("train", output_dir=TEMP_DIR / "perf")

    logger.info("=" * 60)
    logger.info(f"LightGBM NFP Prediction Model - Training [{model_id.upper()}]")
    logger.info("=" * 60)
    logger.info(f"Using EXPANDING WINDOW methodology (no time-travel, tune={tune})")

    # Load target data
    target_df = load_target_data(target_type=target_type, release_type=release_type,
                                 target_source=target_source)

    # Determine date ranges
    train_end = target_df['ds'].max() - pd.DateOffset(months=BACKTEST_MONTHS)

    logger.info(f"\nInitial training period: {target_df['ds'].min()} to {train_end}")
    logger.info(f"Backtest period: {train_end} to {target_df['ds'].max()}")

    # Run expanding window backtest (also returns pre-built X_full, y_full)
    backtest_results, X_full, y_full = run_expanding_window_backtest(
        target_df=target_df,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        tune=tune,
        nsa_backtest_results=nsa_backtest_results,
    )

    if backtest_results.empty:
        logger.error("Backtest produced no results")
        return

    # Train final production model on ALL data (for future predictions)
    # Reuses X_full/y_full from backtest to avoid redundant ~5min feature build
    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING FINAL PRODUCTION MODEL [{model_id.upper()}]")
    logger.info("=" * 60)

    # Filter out NaN targets for final model training
    valid_mask = ~y_full.isna()
    X_full_valid = X_full[valid_mask].copy()
    y_full_valid = y_full[valid_mask].copy()

    logger.info(f"Total observations: {len(X_full)}, Valid for training: {len(X_full_valid)}")

    # Final target_month anchor for the production model is simply the most recent date available
    # or equivalently, a future date where we plan to predict. But to be safe, we anchor it to 
    # the max date in the data (most recent NFP print).
    final_target_month = pd.to_datetime(X_full_valid['ds'].max())

    # Final feature selection mirrors backtest policy.
    # In all-features mode, use the post-backtest selected features (now written to JSON).
    # In legacy mode, use snapshot base features + union pool + branch target FS.
    master_snapshot_base_features = load_selected_features(
        target_type=target_type, target_source=target_source
    )
    _prod_all_features_mode = master_snapshot_base_features is None

    cleaned_feature_cols = clean_features(X_full_valid, y_full_valid)
    cleaned_feature_cols = [c for c in cleaned_feature_cols if c in X_full_valid.columns and c != 'ds']

    # Frozen-features mode for production: same gate as backtest.
    _prod_frozen_features_mode = (
        not _prod_all_features_mode
        and master_snapshot_base_features
        and RESELECT_EVERY_N_MONTHS == 0
    )

    # Per-window mode for production: use the LATEST per-window feature set
    # (most recent step_date in the cohort). Mirrors the backtest's final-step
    # behavior so the production model stays consistent with the last step's
    # OOS predictions. Takes precedence over all-features mode — an explicit
    # replay schedule is a deliberate user choice.
    _prod_per_window_sets: List[Tuple[pd.Timestamp, List[str]]] = []
    _prod_per_window_mode = USE_PER_WINDOW_FEATURES
    if _prod_per_window_mode:
        _prod_per_window_sets = _load_per_window_feature_sets(target_type, target_source)
        if not _prod_per_window_sets:
            logger.warning(
                f"[{model_id}] PER-WINDOW mode requested for production but no "
                f"JSONs found. Falling back to frozen-features mode."
            )
            _prod_per_window_mode = False
        else:
            _prod_frozen_features_mode = False

    if _prod_per_window_mode:
        latest_step_date, latest_features = _prod_per_window_sets[-1]
        feature_cols = [
            c for c in latest_features
            if c in X_full_valid.columns and c in cleaned_feature_cols
        ]
        if not feature_cols:
            raise RuntimeError(
                f"[{model_id}] PER-WINDOW production features have zero overlap "
                f"with X_full_valid / cleaned_feature_cols. Latest set "
                f"({latest_step_date.strftime('%Y-%m')}): {len(latest_features)} features."
            )
        logger.info(
            f"[{model_id}] PER-WINDOW production features: {len(feature_cols)} "
            f"(from latest set {latest_step_date.strftime('%Y-%m')})"
            + (" (overriding ALL-FEATURES mode)" if _prod_all_features_mode else "")
        )
    elif _prod_all_features_mode:
        # All-features mode: run dynamic reselection on the full dataset for production
        logger.info(
            f"[{model_id}] ALL-FEATURES mode for production model: "
            f"running dynamic reselection on full dataset"
        )
        _prod_features = _dynamic_reselection(
            X_train=X_full_valid,
            y_train=y_full_valid,
            step_date=final_target_month,
            target_type=target_type,
            target_source=target_source,
        )
        feature_cols = [
            c for c in _prod_features
            if c in X_full_valid.columns and c in cleaned_feature_cols
        ]
        if not feature_cols:
            raise RuntimeError(
                f"[{model_id}] Production dynamic reselection returned no usable features. "
                f"Dynamic: {len(_prod_features)}, Cleaned: {len(cleaned_feature_cols)}."
            )
        logger.info(
            f"[{model_id}] Production features: {len(feature_cols)} "
            f"(from dynamic reselection: {len(_prod_features)})"
        )
    elif _prod_frozen_features_mode:
        # Frozen mode: production model uses the same pre-selected features as the backtest.
        feature_cols = [
            c for c in master_snapshot_base_features
            if c in X_full_valid.columns and c in cleaned_feature_cols
        ]
        if not feature_cols:
            raise RuntimeError(
                f"[{model_id}] Frozen production features have zero overlap with "
                f"X_full_valid / cleaned_feature_cols. Selected={len(master_snapshot_base_features)}."
            )
        logger.info(
            f"[{model_id}] FROZEN production features: {len(feature_cols)} "
            f"(from selected_features_{target_type}_{target_source}.json)"
        )
    else:
        groups = partition_feature_columns(cleaned_feature_cols, target_type=target_type)

        snapshot_candidates = groups['snapshot_features']
        snapshot_base_features = [
            f for f in master_snapshot_base_features if f in X_full_valid.columns and f != 'ds'
        ]
        if not snapshot_base_features:
            snapshot_base_features = [c for c in snapshot_candidates if c in X_full_valid.columns]

        snapshot_base_set = set(snapshot_base_features)
        snapshot_extra_candidates = [
            c for c in snapshot_candidates if c in X_full_valid.columns and c not in snapshot_base_set
        ]

        branch_target_candidates = groups['target_branch_features']
        cal_feats_final = groups['calendar_features']
        if target_type == 'sa':
            cal_feats_final = [f for f in cal_feats_final if f in SA_CALENDAR_FEATURES_KEEP]
        always_keep = _merge_unique_feature_lists(
            cal_feats_final,
            groups['revision_features'],
        )
        dropped_cross_target = groups['other_target_features']

        fs_half_life = SHORTPASS_HALF_LIFE or 60.0
        fs_weights = calculate_sample_weights(X_full_valid, final_target_month, fs_half_life)

        snapshot_selected = snapshot_base_features
        if USE_UNION_POOL:
            from Train.candidate_pool import load_or_build_union_pool
            from Train.short_pass_selection import select_features_for_step

            candidate_pool = load_or_build_union_pool(target_type, target_source, UNION_POOL_MAX)
            candidate_in_data = [f for f in candidate_pool if f in snapshot_extra_candidates]
            logger.info(
                f"Final snapshot pool (extras): base={len(snapshot_base_features)}, "
                f"extra_candidates={len(snapshot_extra_candidates)}, pool={len(candidate_pool)}, "
                f"intersection={len(candidate_in_data)}"
            )
            if candidate_in_data:
                if len(candidate_in_data) > SHORTPASS_TOPK:
                    snapshot_extra_selected = select_features_for_step(
                        X_full_valid[candidate_in_data], y_full_valid,
                        candidate_features=candidate_in_data,
                        top_k=SHORTPASS_TOPK,
                        method=SHORTPASS_METHOD,
                        sample_weights=fs_weights,
                    )
                else:
                    snapshot_extra_selected = candidate_in_data
                snapshot_selected = _merge_unique_feature_lists(snapshot_base_features, snapshot_extra_selected)
            else:
                logger.warning(
                    "Final union pool had no overlap with snapshot extra candidates; using base snapshot features."
                )

        final_branch_fs_top_k = _get_branch_target_fs_top_k(target_type, target_source)
        final_branch_fs_method = _get_branch_target_fs_method(target_type, target_source)
        if USE_BRANCH_TARGET_FS and branch_target_candidates:
            branch_target_selected = select_branch_target_features_for_step(
                X_train=X_full_valid,
                y_train=y_full_valid,
                target_type=target_type,
                candidate_features=branch_target_candidates,
                top_k=final_branch_fs_top_k,
                method=final_branch_fs_method,
                corr_threshold=BRANCH_TARGET_FS_CORR_THRESHOLD,
                min_overlap=BRANCH_TARGET_FS_MIN_OVERLAP,
                sample_weights=fs_weights,
                dynamics_weight_level=BRANCH_TARGET_FS_WEIGHT_LEVEL,
                dynamics_weight_diff=BRANCH_TARGET_FS_WEIGHT_DIFF,
                dynamics_weight_dir=BRANCH_TARGET_FS_WEIGHT_DIR,
                dynamics_weight_amp=BRANCH_TARGET_FS_WEIGHT_AMP,
                dynamics_weight_sign=BRANCH_TARGET_FS_WEIGHT_SIGN,
                dynamics_weight_tail=BRANCH_TARGET_FS_WEIGHT_TAIL,
            )
        else:
            branch_target_selected = branch_target_candidates

        feature_cols = _merge_unique_feature_lists(
            snapshot_selected,
            branch_target_selected,
            always_keep,
        )
        logger.info(
            f"Final feature set: snapshot={len(snapshot_selected)} "
            f"(base={len(snapshot_base_features)}, extras_from_clean={len(snapshot_extra_candidates)}), "
            f"branch_target={len(branch_target_selected)}/{len(branch_target_candidates)} "
            f"(method={final_branch_fs_method}, top_k={final_branch_fs_top_k}), "
            f"always_keep={len(always_keep)}, dropped_cross_target={len(dropped_cross_target)}, "
            f"final={len(feature_cols)}"
        )

    if not feature_cols:
        raise ValueError("Final feature selection produced no features.")

    X_train = X_full_valid[['ds'] + [c for c in feature_cols if c in X_full_valid.columns]].copy()
    logger.info(f"Training final model with {len(feature_cols)} features")

    # Tune hyperparameters on all available data for the production model
    final_params = None

    if tune:
        logger.info("Tuning hyperparameters for final production model...")
        feature_only_cols = [c for c in feature_cols if c in X_train.columns and c != 'ds']
        tuning_mode = _get_tuning_objective_mode(target_type, target_source)
        # Build the fusion-CV context locally — it's built fresh inside
        # run_expanding_window_backtest but that variable is out of scope
        # here. The call is cheap (~1s) and reuses the same disk artifacts.
        _fusion_tuning_context: Optional[Dict] = None
        if tuning_mode == 'kalman_fusion':
            _fusion_tuning_context = _build_fusion_tuning_context(X_train['ds'])
            if _fusion_tuning_context is None:
                logger.warning(
                    "[FusionTune] Fusion context unavailable for final "
                    "production tune — falling back to legacy composite."
                )
                tuning_mode = TUNING_OBJECTIVE_MODE_DEFAULT

        _joint_final_active = (
            JOINT_OPTUNA
            and str(target_type).lower() == 'nsa'
            and tuning_mode == 'kalman_fusion'
        )
        _joint_final_ctx: Optional[Dict] = None
        if _joint_final_active:
            try:
                from Train.joint_tuning import build_joint_fusion_context
                _joint_final_ctx = build_joint_fusion_context(X_train['ds'])
            except Exception as e:
                logger.warning(f"[JointTune] final-tune context build failed ({e})")
                _joint_final_ctx = None
            if _joint_final_ctx is None:
                _joint_final_active = False
                logger.warning(
                    "[JointTune] joint final tune requested but context unavailable — "
                    "falling back to separate tunes."
                )

        if _joint_final_active and _joint_final_ctx is not None:
            from Train.joint_tuning import tune_full_pipeline_joint
            joint_result = tune_full_pipeline_joint(
                X_train[['ds'] + feature_only_cols], y_full_valid,
                target_month=final_target_month,
                fusion_context=_joint_final_ctx,
                use_huber_loss=use_huber_loss,
            )
            final_params = joint_result['lgbm_params']
            _save_joint_tuned_params(joint_result, step_date=final_target_month)
        else:
            # Pass the DF WITH 'ds' so inner tuning folds can manage their own point-in-time weights
            final_params = tune_hyperparameters(
                X_train[['ds'] + feature_only_cols], y_full_valid, target_month=final_target_month,
                use_huber_loss=use_huber_loss,
                objective_mode=tuning_mode,
                lambda_std_ratio=TUNING_LAMBDA_STD_RATIO,
                lambda_diff_std_ratio=TUNING_LAMBDA_DIFF_STD_RATIO,
                lambda_tail_mae=TUNING_LAMBDA_TAIL_MAE,
                lambda_corr_diff=TUNING_LAMBDA_CORR_DIFF,
                lambda_diff_sign=TUNING_LAMBDA_DIFF_SIGN,
                tail_quantile=VARIANCE_TAIL_QUANTILE,
                tail_weighting=(
                    ENABLE_TAIL_AWARE_WEIGHTING
                    and _is_variance_priority_target(target_type, target_source)
                ),
                tail_weight_abs_level_quantile=TAIL_WEIGHT_ABS_LEVEL_QUANTILE,
                tail_weight_abs_diff_quantile=TAIL_WEIGHT_ABS_DIFF_QUANTILE,
                tail_weight_level_boost=TAIL_WEIGHT_LEVEL_BOOST,
                tail_weight_diff_boost=TAIL_WEIGHT_DIFF_BOOST,
                tail_weight_max_multiplier=TAIL_WEIGHT_MAX_MULTIPLIER,
                fusion_context=(
                    _fusion_tuning_context if tuning_mode == 'kalman_fusion' else None
                ),
            )
        
        final_half_life = final_params.get('half_life_months', 60.0)
        # Reattach the targets and half_life so train_lightgbm_model can recompute internally
        final_params['target_month'] = final_target_month
        final_params['half_life_months'] = final_half_life
        final_params['tail_weighting'] = (
            ENABLE_TAIL_AWARE_WEIGHTING
            and _is_variance_priority_target(target_type, target_source)
        )
        final_params['tail_weight_abs_level_quantile'] = TAIL_WEIGHT_ABS_LEVEL_QUANTILE
        final_params['tail_weight_abs_diff_quantile'] = TAIL_WEIGHT_ABS_DIFF_QUANTILE
        final_params['tail_weight_level_boost'] = TAIL_WEIGHT_LEVEL_BOOST
        final_params['tail_weight_diff_boost'] = TAIL_WEIGHT_DIFF_BOOST
        final_params['tail_weight_max_multiplier'] = TAIL_WEIGHT_MAX_MULTIPLIER
    else:
        final_params = {
            'target_month': final_target_month,
            'half_life_months': 60.0,
            'tail_weighting': (
                ENABLE_TAIL_AWARE_WEIGHTING
                and _is_variance_priority_target(target_type, target_source)
            ),
            'tail_weight_abs_level_quantile': TAIL_WEIGHT_ABS_LEVEL_QUANTILE,
            'tail_weight_abs_diff_quantile': TAIL_WEIGHT_ABS_DIFF_QUANTILE,
            'tail_weight_level_boost': TAIL_WEIGHT_LEVEL_BOOST,
            'tail_weight_diff_boost': TAIL_WEIGHT_DIFF_BOOST,
            'tail_weight_max_multiplier': TAIL_WEIGHT_MAX_MULTIPLIER,
        }

    # Train final model
    model, importance, residuals = train_lightgbm_model(
        X_train,
        y_full_valid,
        n_splits=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        params_override=final_params,
    )

    # Persist current-run artifacts even if keep-rule failed to avoid stale files.
    # production_eligible communicates deployment readiness.
    keep_rule_failed = bool(backtest_results.attrs.get('keep_rule_failed', False))
    variance_gate_failed = bool(backtest_results.attrs.get('variance_gate_failed', False))
    production_eligible = not (keep_rule_failed or variance_gate_failed)

    if keep_rule_failed or variance_gate_failed:
        logger.warning(
            f"Model {model_id.upper()} flagged as non-production-eligible; "
            f"keep_rule_failed={keep_rule_failed}, variance_gate_failed={variance_gate_failed}. "
            "Saving artifacts with production_eligible=False."
        )

    save_model(
        model,
        feature_cols,
        residuals,
        importance,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
        extra_metadata={
            'production_eligible': production_eligible,
            'keep_rule_failed': keep_rule_failed,
            'variance_gate_failed': variance_gate_failed,
            'variance_gate_reasons': backtest_results.attrs.get('variance_gate_reasons', []),
        },
    )

    # ── P1-6: Persist OOS metrics as structured JSON ──
    _metrics_dir = OUTPUT_DIR / "backtest"
    _metrics_dir.mkdir(parents=True, exist_ok=True)
    _metrics_path = _metrics_dir / f"{model_id}_metrics.json"

    try:
        _summary = backtest_results.attrs.get('summary_metrics', {})
        _per_month = []
        for _, _row in backtest_results.iterrows():
            if pd.isna(_row.get('error')):
                continue
            _per_month.append({
                'ds':           _row['ds'].strftime('%Y-%m'),
                'actual':       None if pd.isna(_row['actual']) else float(_row['actual']),
                'predicted':    float(_row['predicted']),
                'error':        None if pd.isna(_row['error']) else float(_row['error']),
                'dir_correct':  None if pd.isna(_row.get('dir_correct', np.nan)) else int(_row['dir_correct']),
                'accel_correct':None if pd.isna(_row.get('accel_correct', np.nan)) else int(_row['accel_correct']),
            })

        _metrics_payload = {
            'model_id':        model_id,
            'run_date':        str(pd.Timestamp.now().date()),
            'backtest_months': BACKTEST_MONTHS,
            'overall':         _summary,
            'per_month':       _per_month,
        }
        with open(_metrics_path, 'w') as _f:
            json.dump(_metrics_payload, _f, indent=2)
        logger.info(f"OOS metrics saved → {_metrics_path}")
    except Exception as _e:
        logger.warning(f"Could not save metrics JSON: {_e}")

    # ── Perf dump (env-gated: NFP_PERF=1 only) ──
    dump_perf_json(
        stage_name="train",
        output_dir=TEMP_DIR / "perf",
        extra={
            "target_type": target_type,
            "release_type": release_type,
            "target_source": target_source,
            "tune": tune,
            "use_huber_loss": use_huber_loss,
            "backtest_months": BACKTEST_MONTHS,
            "model_id": model_id,
        },
    )

    return model, feature_cols, residuals, backtest_results, X_full, y_full


def predict_nfp_mom(
    target_month: pd.Timestamp,
    model: Optional[lgb.Booster] = None,
    metadata: Optional[Dict] = None,
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'revised',
) -> Dict:
    """
    Make NFP MoM prediction for a specific month.

    Uses snapshot of month M to predict month M's MoM change.

    Args:
        target_month: Month to predict (format: YYYY-MM-01 or YYYY-MM-DD)
        model: Optional pre-loaded model. If None, loads from disk.
        metadata: Optional pre-loaded metadata. If None, loads from disk.
        target_type: 'nsa' or 'sa' - determines which model to load
        release_type: 'first' or 'last' - determines which release model to load
        target_source: 'revised' - determines which master snapshot variant

    Returns:
        Dictionary with prediction, intervals, and metadata
    """
    model_id = get_model_id(target_type, release_type, target_source)

    # Normalize target_month to first of month
    target_month = pd.Timestamp(target_month).replace(day=1)

    # ── P2-4: Revised-model timing guard ──
    # The revised label for month M is only available after the M+1 NFP release.
    # Calling this with target_source='revised' before that date is a data error.
    if target_source == 'revised':
        try:
            rev_df = load_target_data(target_type, release_type=release_type, target_source='revised')
            match = rev_df[rev_df['ds'] == target_month]
            if not match.empty and 'operational_available_date' in match.columns:
                op_date = pd.to_datetime(match['operational_available_date'].iloc[0])
                if pd.notna(op_date) and pd.Timestamp.now() < op_date:
                    raise RuntimeError(
                        f"Revised target for {target_month:%Y-%m} is not yet observable "
                        f"(available from {op_date:%Y-%m-%d}). Revised data not yet available."
                    )
        except FileNotFoundError:
            pass  # Revised cache missing — let load_target_data raise below

    # Load model if not provided
    if model is None or metadata is None:
        model, metadata = load_model(
            target_type=target_type,
            release_type=release_type,
            target_source=target_source,
        )

    feature_cols = metadata['feature_cols']
    residuals = metadata['residuals']

    # Get snapshot for this month (single pre-merged, pre-selected master snapshot)
    snapshot_date = target_month + pd.offsets.MonthEnd(0)
    snapshot_df = load_master_snapshot(
        snapshot_date, target_type=target_type, target_source=target_source
    )

    if snapshot_df is None or snapshot_df.empty:
        raise FileNotFoundError(f"No master snapshot available for {snapshot_date}")

    # Load branch target data for lagged features
    target_full = load_target_data(
        target_type, release_type=release_type, target_source=target_source
    )

    # Use NFP release date as strict cutoff when available
    target_ref = target_full
    cutoff_date = target_month
    if 'release_date' in target_ref.columns:
        match = target_ref[target_ref['ds'] == target_month]
        if not match.empty:
            rd = match['release_date'].iloc[0]
            if pd.notna(rd):
                cutoff_date = rd

    # Create features from master snapshot (already contains all data sources)
    features = pivot_snapshot_to_wide(snapshot_df, target_month, cutoff_date=cutoff_date)

    if features.empty:
        raise ValueError(f"Could not create features for {target_month}")

    features = add_calendar_features(features, target_month)

    # Add branch-target lagged features
    target_lag_features = get_lagged_target_features(
        target_full, target_month, f'nfp_{target_type}'
    )
    for k, v in target_lag_features.items():
        features[k] = v

    # Add cross-snapshot revision features (master[M] vs master[M-1])
    prev_month = target_month - pd.DateOffset(months=1)
    target_ref_df = target_full
    prev_row = target_ref_df[target_ref_df['ds'] == prev_month]
    if not prev_row.empty and 'release_date' in prev_row.columns:
        prev_release = prev_row.iloc[0]['release_date']
        prev_cutoff = prev_release if pd.notna(prev_release) else prev_month
    else:
        prev_cutoff = prev_month

    revision_feats = get_revision_features_for_month(
        target_month, prev_cutoff,
        target_type=target_type, target_source=target_source,
    )
    if not revision_feats.empty:
        for col in revision_feats.columns:
            features[col] = revision_feats[col].iloc[0]

    # Make prediction with intervals
    pred_result = predict_with_intervals(model, features, residuals, feature_cols)

    return {
        'target_month': target_month,
        'prediction': pred_result['prediction'],
        'intervals': pred_result['intervals'],
        'std': pred_result['std'],
        'mean_residual_bias': pred_result['mean_residual_bias'],
        'features_used': len(feature_cols),
        'target_type': target_type,
        'release_type': release_type,
        'model_id': model_id
    }


def convert_mom_to_level(
    mom_prediction: float,
    previous_level: float
) -> float:
    """Convert MoM change prediction to level prediction."""
    return previous_level + mom_prediction


def get_latest_prediction(
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'revised',
) -> Dict:
    """Get prediction for the most recent available month."""
    model_id = get_model_id(target_type, release_type, target_source)

    # Find latest snapshot available
    target_df = load_target_data(
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
    )
    latest_target = target_df['ds'].max()

    logger.info(f"Making {model_id.upper()} prediction for latest available month: {latest_target}")

    return predict_nfp_mom(
        latest_target,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
    )


# `--train-all` runs ONLY the NSA branch.
# The SA branch has been retired from the published pipeline as of 2026-05:
# the canonical SA-revised forecast is produced by the Kalman fusion of
# (consensus, NSA + predicted adjustment, NSA acceleration), not by a
# stand-alone SA LightGBM. Tuning is therefore directed at the fusion
# objective; running an SA LightGBM model in --train-all would burn compute
# on a branch no one publishes.
ALL_COMBOS = [
    ('nsa', 'first', 'revised'),
]


def validate_post_train_all_artifacts(
    run_model_ids: Optional[List[str]] = None,
    combos: List[Tuple[str, str, str]] = ALL_COMBOS,
    model_save_dir: Path = MODEL_SAVE_DIR,
    metrics_dir: Path = OUTPUT_DIR / "backtest",
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate that train-all produced complete production artifacts.

    Required artifacts:
      1) 2 model files + 2 metadata files
      2) 2 per-model metrics JSON files
      3) model_comparison.csv/html
      4) revised output bundles under OUTPUT_DIR (_output/)

    Optional artifacts (post-training pipeline):
      5) sandbox predicted adjustment + SA blend walk-forward
      6) consensus anchor (baseline consensus + Kalman fusion)
    """
    if output_root is None:
        output_root = OUTPUT_DIR

    required: List[Tuple[str, Path]] = []
    optional: List[Tuple[str, Path]] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []

    expected_model_ids = [get_model_id(tt, rt, ts) for tt, rt, ts in combos]
    if run_model_ids is not None:
        missing_run_ids = sorted(set(expected_model_ids) - set(run_model_ids))
        for mid in missing_run_ids:
            missing_required.append(f"run_result::{mid}")

    for model_id in expected_model_ids:
        model_dir = model_save_dir / model_id
        required.append((f"model::{model_id}", model_dir / f"lightgbm_{model_id}_model.txt"))
        required.append((f"metadata::{model_id}", model_dir / f"lightgbm_{model_id}_metadata.pkl"))
        required.append((f"metrics::{model_id}", metrics_dir / f"{model_id}_metrics.json"))

    required.append(("comparison::csv", model_save_dir / "model_comparison.csv"))
    required.append(("comparison::html", model_save_dir / "model_comparison.html"))

    required.append(("output::revised::NSA_prediction",
                     output_root / "NSA_prediction" / "backtest_results.csv"))
    required.append(("output::revised::NSA_metrics",
                     output_root / "NSA_prediction" / "summary_statistics.csv"))
    required.append(("output::revised::NSA_importance",
                     output_root / "NSA_prediction" / "feature_importance.csv"))
    # SA LightGBM artifacts are no longer required — the SA branch is retired.
    # The fusion output (consensus_anchor/kalman_fusion) is the canonical
    # SA-revised forecast.
    required.append(("output::revised::adjustment_prediction",
                     output_root / "NSA_plus_adjustment" / "backtest_results.csv"))
    required.append(("output::revised::adjustment_metrics",
                     output_root / "NSA_plus_adjustment" / "summary_statistics.csv"))

    # predictions.csv can be absent when there are no future (NaN) months left.
    optional.append(("output::revised::forward_predictions",
                     output_root / "Predictions" / "predictions.csv"))

    # Post-training sandbox artifacts (optional)
    optional.append(("sandbox::predicted_adjustment",
                     output_root / "sandbox" / "nsa_predicted_adjustment_revised" / "backtest_results.csv"))
    optional.append(("sandbox::sa_blend_walkforward",
                     output_root / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv"))

    # Consensus anchor: Baseline_Consensus + Kalman_Fusion are the production
    # artifact set. AccelOverride and Kalman+AccelPostFilter were dropped
    # 2026-05-11 (both underperformed Consensus on the backtest window).
    consensus_dir = output_root / "consensus_anchor"
    required.append(("consensus_anchor::baseline_consensus",
                     consensus_dir / "baseline_consensus" / "backtest_results.csv"))
    required.append(("consensus_anchor::kalman_fusion",
                     consensus_dir / "kalman_fusion" / "backtest_results.csv"))
    required.append(("consensus_anchor::comparison_metrics",
                     consensus_dir / "comparison_metrics.csv"))
    required.append(("consensus_anchor::comparison_overlay",
                     consensus_dir / "comparison_overlay.png"))
    required.append(("consensus_anchor::comparison_metrics_png",
                     consensus_dir / "comparison_metrics.png"))
    required.append(("consensus_anchor::comparison_scorecard",
                     consensus_dir / "comparison_scorecard.html"))

    for label, path in required:
        if not path.exists():
            missing_required.append(f"{label}::{path}")

    for label, path in optional:
        if not path.exists():
            missing_optional.append(f"{label}::{path}")

    return {
        "ok": len(missing_required) == 0,
        "required_checked": len(required),
        "optional_checked": len(optional),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def _read_half_life_json(path: Path) -> Optional[float]:
    """Read ``half_life_years`` from a JSON file. Returns None if missing or unreadable.

    Used by the iterative fusion-tune orchestrator to compare HL state across
    passes. The same JSON files are written by `_load_fusion_selection_target`
    (dynamic_fs_selection_hl.json) and `_tune_kalman` (tuned_params.json).
    """
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    hl = data.get("half_life_years") if isinstance(data, dict) else None
    try:
        return float(hl) if hl is not None else None
    except (TypeError, ValueError):
        return None


def _clear_universe_cache() -> int:
    """Delete all universe cache files. Returns count removed.

    The cache is keyed by (target_type, target_source, asof_month) — NOT by
    half_life_years. So when iterating the fusion tune, we must invalidate
    between passes; otherwise Pass 2's dynamic FS would silently reuse Pass 1's
    cached features and the selection target swap (`SA_revised − adj_pred(HL_1)`)
    would be ignored.
    """
    from Train.config import UNIVERSE_CACHE_DIR
    if not UNIVERSE_CACHE_DIR.exists():
        return 0
    removed = 0
    for f in UNIVERSE_CACHE_DIR.glob("universe_*.json"):
        try:
            f.unlink()
            removed += 1
        except Exception as e:
            logger.warning(f"Failed to remove universe cache file {f}: {e}")
    return removed


def _read_kalman_fusion_metrics() -> Dict[str, Optional[float]]:
    """Read MAE/RMSE/AccelAcc/DirAcc from the current Kalman fusion summary CSV.

    Returns a dict with NaN/None for missing fields; never raises.
    """
    out: Dict[str, Optional[float]] = {
        "MAE": None, "RMSE": None,
        "Acceleration_Accuracy": None, "Directional_Accuracy": None,
    }
    summary_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "summary_statistics.csv"
    if not summary_path.exists():
        return out
    try:
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        for k in list(out.keys()):
            if k in row:
                try:
                    out[k] = float(row[k])
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='LightGBM NFP Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model (default: nsa_first)
  python Train/train_lightgbm_nfp.py --train

  # Train all 4 model variants and generate comparison
  python Train/train_lightgbm_nfp.py --train-all

  # Train with specific target/release
  python Train/train_lightgbm_nfp.py --train --target nsa --release first

  # Joint iterative tune: full pipeline runs up to N times, alternating
  # NSA tuning and Kalman half-life tuning until they agree (Δhl < 0.25y).
  # Clears universe cache between passes to force fresh FS each iteration.
  python Train/train_lightgbm_nfp.py --iterate-fusion-tune

  # Predict for a specific month
  python Train/train_lightgbm_nfp.py --predict 2024-12 --target nsa

  # Get latest prediction
  python Train/train_lightgbm_nfp.py --latest --target nsa
        """
    )
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--train-all', action='store_true',
                        help='Train both model variants (NSA/SA × revised) and generate comparison')
    parser.add_argument('--iterate-fusion-tune', action='store_true',
                        help='Run --train-all repeatedly until NSA hyperparameters and Kalman '
                             'half-life converge (|Δhl| < 0.25y). Clears universe cache between '
                             'passes so each iteration re-runs feature selection with the new HL.')
    parser.add_argument('--max-fusion-passes', type=int, default=3,
                        help='Max passes for --iterate-fusion-tune (default: 3).')
    parser.add_argument('--fusion-converge-threshold', type=float, default=0.25,
                        help='Stop iterating when |Δhalf_life_years| is below this (default: 0.25y).')
    parser.add_argument('--predict', type=str, help='Predict for a specific month (YYYY-MM)')
    parser.add_argument('--latest', action='store_true', help='Predict for latest available month')
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help='Target type: nsa (non-seasonally adjusted) or sa (seasonally adjusted)')
    parser.add_argument('--release', type=str, default='first', choices=['first', 'last'],
                        help='Release type: first (initial) or last (final revised)')
    parser.add_argument('--no-huber-loss', action='store_true',
                        help='Disable Huber loss (uses MSE instead). Huber is enabled by default for outlier robustness.')
    parser.add_argument('--huber-delta', type=float, default=HUBER_DELTA,
                        help=f'Huber delta parameter (default: {HUBER_DELTA}). Lower = more robust to outliers.')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip Optuna hyperparameter tuning (use static defaults). Faster for debugging.')
    args = parser.parse_args()

    # Convert --no-* flags to booleans
    use_huber_loss = not args.no_huber_loss
    tune = not args.no_tune
    target_source = 'revised'

    model_id = get_model_id(args.target, args.release, target_source)

    if args.iterate_fusion_tune:
        # =====================================================================
        # ITERATIVE FUSION TUNING — joint NSA ↔ Kalman convergence
        # =====================================================================
        # Runs --train-all up to N times as a subprocess. After each pass we
        # compare the half-life that the dynamic FS USED in this pass (from
        # dynamic_fs_selection_hl.json) against the half-life that Kalman
        # PRODUCED at the end of this pass (from tuned_params.json). When
        # |Δhl| < threshold, the system is internally consistent — FS picked
        # features for a HL that Kalman now confirms, and we stop. Otherwise
        # the next pass starts with the new HL and re-runs FS + NSA Optuna
        # + backtest + Kalman tune against that HL. Each pass gets its own
        # archive folder so all snapshots are preserved.
        import subprocess as _sub
        import sys as _sys

        if args.no_tune:
            logger.error(
                "--iterate-fusion-tune is incompatible with --no-tune. "
                "Iteration requires Kalman's HL tune to provide a feedback "
                "signal; with --no-tune the HL is fixed at the fallback. "
                "Drop --no-tune or drop --iterate-fusion-tune."
            )
            sys.exit(2)

        max_passes = max(1, int(args.max_fusion_passes))
        converge_threshold = float(args.fusion_converge_threshold)

        tuned_hl_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"
        fs_hl_path = OUTPUT_DIR / "consensus_anchor" / "dynamic_fs_selection_hl.json"
        iter_log_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "fusion_iteration_log.json"
        iter_log_path.parent.mkdir(parents=True, exist_ok=True)

        passes: List[Dict[str, Any]] = []
        converged = False
        iteration_start_hl = _read_half_life_json(tuned_hl_path)
        logger.info("=" * 70)
        logger.info(f"ITERATIVE FUSION TUNE — up to {max_passes} passes, "
                    f"converge |Δhl| < {converge_threshold:.3f}y")
        logger.info(f"  Starting HL state (tuned_params.json): {iteration_start_hl}")
        logger.info("=" * 70)

        for pass_idx in range(1, max_passes + 1):
            hl_prior_tuned = _read_half_life_json(tuned_hl_path)
            hl_prior_fs = _read_half_life_json(fs_hl_path)

            logger.info(f"\n{'-' * 70}")
            logger.info(f"PASS {pass_idx}/{max_passes}")
            logger.info(f"{'-' * 70}")
            logger.info(f"  HL prior (tuned_params.json): {hl_prior_tuned}")
            logger.info(f"  HL prior (FS used last pass): {hl_prior_fs}")

            removed = _clear_universe_cache()
            logger.info(f"  Cleared {removed} universe cache files "
                        f"(forces fresh FS for this pass)")

            cmd = [_sys.executable, str(Path(__file__).resolve()), "--train-all"]
            if args.no_huber_loss:
                cmd.append("--no-huber-loss")
            cmd.extend(["--huber-delta", str(args.huber_delta)])
            logger.info(f"  Launching subprocess: {' '.join(cmd)}")

            proc = _sub.run(cmd)
            if proc.returncode != 0:
                logger.error(f"  Pass {pass_idx} subprocess failed (exit {proc.returncode}); "
                             f"aborting iteration")
                passes.append({
                    "pass": pass_idx,
                    "hl_prior_tuned": hl_prior_tuned,
                    "hl_prior_fs": hl_prior_fs,
                    "hl_used_by_fs": _read_half_life_json(fs_hl_path),
                    "hl_new_tuned": _read_half_life_json(tuned_hl_path),
                    "delta": None,
                    "fusion_metrics": _read_kalman_fusion_metrics(),
                    "status": f"failed_exit_{proc.returncode}",
                })
                break

            hl_new_tuned = _read_half_life_json(tuned_hl_path)
            hl_new_fs = _read_half_life_json(fs_hl_path)
            fusion_metrics = _read_kalman_fusion_metrics()

            delta: Optional[float] = None
            if hl_new_fs is not None and hl_new_tuned is not None:
                delta = float(abs(hl_new_tuned - hl_new_fs))

            passes.append({
                "pass": pass_idx,
                "hl_prior_tuned": hl_prior_tuned,
                "hl_prior_fs": hl_prior_fs,
                "hl_used_by_fs": hl_new_fs,
                "hl_new_tuned": hl_new_tuned,
                "delta": delta,
                "fusion_metrics": fusion_metrics,
                "status": "ok",
            })

            logger.info(f"\n  PASS {pass_idx} COMPLETE")
            logger.info(f"    HL used by FS in this pass:   {hl_new_fs}")
            logger.info(f"    HL produced by Kalman tune:   {hl_new_tuned}")
            logger.info(f"    Δ = |Kalman_new − FS_used|:   "
                        f"{delta if delta is not None else 'n/a'}")
            logger.info(f"    Fusion MAE / RMSE / AccelAcc: "
                        f"{fusion_metrics.get('MAE')} / "
                        f"{fusion_metrics.get('RMSE')} / "
                        f"{fusion_metrics.get('Acceleration_Accuracy')}")

            if delta is not None and delta < converge_threshold:
                logger.info(f"  ★ CONVERGED — Δ {delta:.3f}y < {converge_threshold:.3f}y threshold")
                converged = True
                break

            if pass_idx == max_passes:
                logger.warning(
                    f"  Reached max_passes={max_passes} without convergence "
                    f"(last Δ={delta}). Increase --max-fusion-passes if needed."
                )

        with open(iter_log_path, "w") as f:
            json.dump({
                "max_passes": max_passes,
                "converge_threshold_years": converge_threshold,
                "converged": converged,
                "iteration_start_hl": iteration_start_hl,
                "passes": passes,
            }, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info(f"ITERATIVE FUSION TUNE COMPLETE  (converged={converged}, "
                    f"passes_run={len(passes)})")
        logger.info(f"  Log: {iter_log_path}")
        if passes:
            best = min(
                (p for p in passes if p.get("fusion_metrics", {}).get("MAE") is not None),
                key=lambda p: p["fusion_metrics"]["MAE"],
                default=None,
            )
            if best is not None:
                bm = best["fusion_metrics"]
                logger.info(
                    f"  Best pass by fusion MAE: pass {best['pass']} → "
                    f"MAE={bm.get('MAE')}, RMSE={bm.get('RMSE')}, "
                    f"AccelAcc={bm.get('Acceleration_Accuracy')}"
                )
        logger.info("=" * 70)

    elif args.train_all:
        # =====================================================================
        # TRAIN ALL 4 MODEL VARIANTS AND GENERATE COMPARISON
        # =====================================================================
        import time as _time
        from Train.Output_code.model_comparison import generate_comparison_scorecard

        logger.info("=" * 70)
        logger.info("TRAINING BOTH MODEL VARIANTS (NSA/SA × revised)")
        logger.info("=" * 70)

        all_comparison_results = {}
        all_train_results = {}  # Store full results for output generation
        _train_all_t0 = _time.time()

        # Sequential NSA → SA: NSA runs first, its backtest results provide
        # acceleration features for the SA model via nsa_acceleration.py.
        _nsa_backtest_for_sa: Optional[pd.DataFrame] = None

        for combo_idx, (tt, rt, ts) in enumerate(ALL_COMBOS, 1):
            combo_id = get_model_id(tt, rt, ts)
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# [{combo_idx}/{len(ALL_COMBOS)}] TRAINING: {combo_id.upper()}")
            logger.info(f"{'#' * 70}")

            try:
                # Pass NSA backtest results to SA branch for acceleration features
                _nsa_bt = _nsa_backtest_for_sa if tt == 'sa' else None
                result = train_and_evaluate(
                    target_type=tt,
                    release_type=rt,
                    target_source=ts,
                    use_huber_loss=use_huber_loss,
                    huber_delta=args.huber_delta,
                    tune=tune,
                    nsa_backtest_results=_nsa_bt,
                )

                if result is not None:
                    model, feature_cols, residuals, backtest_results, X_full, y_full = result
                    all_comparison_results[combo_id] = {
                        'backtest_results': backtest_results,
                        'n_features': len(feature_cols),
                        'n_train_obs': len(X_full),
                    }
                    all_train_results[combo_id] = {
                        'model': model,
                        'feature_cols': feature_cols,
                        'residuals': residuals,
                        'backtest_results': backtest_results,
                        'X_full': X_full,
                        'y_full': y_full,
                        'target_type': tt,
                        'release_type': rt,
                        'target_source': ts,
                    }
                    # Capture NSA backtest results for SA acceleration feature injection
                    if tt == 'nsa':
                        _nsa_backtest_for_sa = backtest_results.copy()
                        logger.info(f"[{combo_idx}] Captured NSA backtest results "
                                    f"({len(_nsa_backtest_for_sa)} months) for SA acceleration features")
                    logger.info(f"[{combo_idx}/{len(ALL_COMBOS)}] {combo_id.upper()} completed successfully")
                else:
                    logger.warning(f"[{combo_idx}/{len(ALL_COMBOS)}] {combo_id.upper()} returned None")

            except Exception as e:
                logger.error(f"[{combo_idx}/{len(ALL_COMBOS)}] {combo_id.upper()} FAILED: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Generate comparative scorecard
        if all_comparison_results:
            logger.info("\nGenerating comparative scorecard...")
            scorecard = generate_comparison_scorecard(all_comparison_results)
        else:
            logger.error("No models completed successfully")

        # Generate combined output for revised (NSA-only LightGBM; SA from
        # the target file is just the actual series used by the adjustment +
        # fusion stages).
        from Train.Output_code.generate_output import generate_all_output
        from Train.data_loader import load_target_data
        nsa_id = get_model_id('nsa', 'first', 'revised')
        if nsa_id in all_train_results:
            nsa_r = all_train_results[nsa_id]
            nsa_metadata = {
                'feature_cols': nsa_r['feature_cols'],
                'importance': dict(zip(
                    nsa_r['feature_cols'],
                    nsa_r['model'].feature_importance(importance_type='gain')
                )),
            }
            # SA revised actuals for the adjustment / fusion folders. Only
            # [ds, actual] are needed downstream — we do NOT train an SA LGBM.
            _sa_target = load_target_data(
                target_type='sa', release_type='first', target_source='revised',
            )
            sa_results_actuals = pd.DataFrame({
                'ds': pd.to_datetime(_sa_target['ds']),
                'actual': _sa_target['y_mom'].astype(float),
            })

            logger.info("\nGenerating combined output for revised "
                        "(NSA LightGBM + fusion-driven SA forecast)...")
            generate_all_output(
                    nsa_results=nsa_r['backtest_results'],
                    sa_results=sa_results_actuals,
                    nsa_model=nsa_r['model'],
                    sa_model=None,
                    nsa_metadata=nsa_metadata,
                    sa_metadata=None,
                    nsa_X_full=nsa_r['X_full'],
                    sa_X_full=None,
                    nsa_y_full=nsa_r['y_full'],
                    sa_y_full=None,
                    nsa_residuals=nsa_r['residuals'],
                    sa_residuals=None,
                    output_base=OUTPUT_DIR,
                    suffix='',
                    # Archive happens at the very end of --train-all so the
                    # snapshot includes the consensus-anchor outputs.
                    archive=False,
                )

        # ── Post-output: Predicted adjustment, SA blend, Consensus anchor ──
        logger.info("\n" + "=" * 70)
        logger.info("POST-TRAINING: Sandbox experiments + Consensus anchor integration")
        logger.info("=" * 70)

        # Step 1: NSA predicted seasonal adjustment (PIT-safe walk-forward)
        try:
            from Train.sandbox.experiment_predicted_adjustment import (
                load_adjustment_history,
                load_backtest_inputs,
                run_walkforward_backtest,
                evaluate_models,
                save_outputs,
                SARIMAPredictor,
                MonthlyAveragePredictor,
                TwelveMonthComplementPredictor,
                SameMonthLastYearPredictor,
                ExpWeightedMonthlyAvgPredictor,
                LinearRegressionPredictor,
            )
            logger.info("\n[Post-1] NSA predicted seasonal adjustment...")
            adj_history = load_adjustment_history()
            backtest_inputs = load_backtest_inputs()
            adj_models = [
                SARIMAPredictor(),
                MonthlyAveragePredictor(),
                TwelveMonthComplementPredictor(),
                SameMonthLastYearPredictor(),
                ExpWeightedMonthlyAvgPredictor(half_life_years=3.0),
                LinearRegressionPredictor(),
            ]
            model_results = run_walkforward_backtest(adj_history, backtest_inputs, adj_models)
            comparison = evaluate_models(model_results)
            best_name = comparison.iloc[0]["model_name"]
            save_outputs(best_name, model_results, comparison)
            logger.info("  Predicted adjustment complete (best model: %s)", best_name)
        except Exception as e:
            logger.warning("  Predicted adjustment failed (non-fatal): %s", e)

        # Step 2: SA blend walk-forward — retired alongside the SA LightGBM
        # branch. The blend combined SA-LGBM predictions with NSA+adjustment;
        # with SA LGBM no longer produced, the blend has nothing to blend.
        # Kalman fusion is the canonical replacement.
        logger.info("\n[Post-2] SA blend walk-forward — SKIPPED (SA LightGBM retired)")

        # Step 3: Consensus anchor (Kalman fusion + AccelOverride with Optuna)
        # Required artifact: failing this step fails the whole pipeline, since the
        # 4-way scorecard (baseline_consensus, kalman_fusion, accel_override,
        # kalman_accel_postfilter) is part of the production output set.
        from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline
        logger.info("\n[Post-3] Consensus anchor integration...")
        run_consensus_anchor_pipeline(
            output_base=OUTPUT_DIR,
            tune=tune,
            n_trials=N_OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT,
        )
        logger.info("  Consensus anchor integration complete")

        # Step 4: Archive the full final state (deferred from generate_all_output
        # so the snapshot includes the consensus anchor 4-way scorecard).
        from Train.Output_code.generate_output import archive_outputs
        logger.info("\n[Post-4] Archiving final outputs...")
        archive_outputs(OUTPUT_DIR)

        _total_elapsed = _time.time() - _train_all_t0
        validation = validate_post_train_all_artifacts(
            run_model_ids=list(all_train_results.keys()),
        )
        if not validation["ok"]:
            logger.error("Post-train artifact validation FAILED.")
            for missing in validation["missing_required"]:
                logger.error(f"  MISSING REQUIRED: {missing}")
            raise RuntimeError("Post-train artifact validation failed")
        logger.info(
            "Post-train artifact validation passed "
            f"({validation['required_checked']} required artifacts found)."
        )
        for missing_opt in validation["missing_optional"]:
            logger.warning(f"  MISSING OPTIONAL: {missing_opt}")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"ALL {len(ALL_COMBOS)} MODELS COMPLETE ({_total_elapsed / 60:.1f} minutes total)")
        logger.info(f"{'=' * 70}")

    elif args.train:
        result = train_and_evaluate(
            target_type=args.target,
            release_type=args.release,
            target_source=target_source,
            use_huber_loss=use_huber_loss,
            huber_delta=args.huber_delta,
            tune=tune,
        )

        if result is not None:
            model, feature_cols, residuals, backtest_results, X_full, y_full = result
            metadata = {
                'feature_cols': feature_cols,
                'importance': dict(zip(feature_cols, model.feature_importance(importance_type='gain'))),
            }
            # Always refresh single-branch visualization artifacts for direct runs.
            try:
                from Train.Output_code.generate_output import generate_single_branch_output
                generate_single_branch_output(
                    results_df=backtest_results,
                    model=model,
                    metadata=metadata,
                    X_full=X_full,
                    target_type=args.target,
                    target_source=target_source,
                    output_base=OUTPUT_DIR,
                )
            except Exception as e:
                logger.warning(f"Single-branch output generation failed: {e}")

            # If training NSA (default), generate the combined output using SA
            # actuals from the target file (SA LightGBM is retired — fusion is
            # the canonical SA-revised forecast).
            if args.target == 'nsa':
                from Train.Output_code.generate_output import generate_all_output
                from Train.data_loader import load_target_data
                _sa_target = load_target_data(
                    target_type='sa',
                    release_type=args.release,
                    target_source=target_source,
                )
                sa_results_actuals = pd.DataFrame({
                    'ds': pd.to_datetime(_sa_target['ds']),
                    'actual': _sa_target['y_mom'].astype(float),
                })
                generate_all_output(
                    nsa_results=backtest_results,
                    sa_results=sa_results_actuals,
                    nsa_model=model,
                    sa_model=None,
                    nsa_metadata=metadata,
                    sa_metadata=None,
                    nsa_X_full=X_full,
                    sa_X_full=None,
                    nsa_y_full=y_full,
                    sa_y_full=None,
                    nsa_residuals=residuals,
                    sa_residuals=None,
                    output_base=OUTPUT_DIR,
                    suffix='',
                )

    elif args.predict:
        target_month = pd.Timestamp(args.predict + '-01')
        result = predict_nfp_mom(
            target_month,
            target_type=args.target,
            release_type=args.release,
            target_source=target_source,
        )
        print(f"\n{'='*60}")
        print(f"NFP {model_id.upper()} MoM Change Prediction for {target_month.strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")
        print(f"Std: {result['std']:,.0f}")
        print(f"Features Used: {result['features_used']}")

    elif args.latest:
        result = get_latest_prediction(
            target_type=args.target,
            release_type=args.release,
            target_source=target_source,
        )
        print(f"\n{'='*60}")
        print(f"NFP {model_id.upper()} MoM Change Prediction for {result['target_month'].strftime('%Y-%m')}")
        print(f"{'='*60}")
        print(f"Point Prediction: {result['prediction']:,.0f}")
        print(f"50% CI: [{result['intervals']['50%'][0]:,.0f}, {result['intervals']['50%'][1]:,.0f}]")
        print(f"80% CI: [{result['intervals']['80%'][0]:,.0f}, {result['intervals']['80%'][1]:,.0f}]")
        print(f"95% CI: [{result['intervals']['95%'][0]:,.0f}, {result['intervals']['95%'][1]:,.0f}]")

    else:
        # Default: train and evaluate with defaults (nsa_first)
        train_and_evaluate(target_source=target_source)
