"""
Master Snapshot Generation Pipeline (Auto-Feature Selection)
===================================
This module constructs the final, unified "Master Snapshots" used by the Machine Learning
model. It acts as the grand aggregator, combining independently generated datasets
(FRED, Unifier, ADP, NOAA, Prosper) into a single wide-format parquet file per prediction month.

Critical Architecture:
- Point-in-time accuracy is strictly maintained.
- It operates in a quad-track mode: {nsa, sa} x {first_release, revised}.
- Prior to concatenation, if a valid JSON cache is not found for the specific combo,
  it spins up parallel workers to run a rigorous 6-stage LightGBM feature selection engine on ALL sources.
- It strictly filters the final master outputs to ONLY include the surviving features,
  preventing massive horizontal array bloat.
- Per-source caching: each source's feature selection results are cached independently,
  so a crash in one source doesn't waste completed work from others.
- Source-level batch loading: for the master generation phase, each source's snapshots
  are loaded once and cached in memory to avoid redundant file reads across months.
"""

import pandas as pd
import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to FRONT of path so project-level packages (utils/, settings)
# take priority over local files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger, START_DATE, END_DATE
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map
from Data_ETA_Pipeline.perf_stats import (
    inc_counter,
    install_hooks,
    perf_phase,
    profiled,
    register_atexit_dump,
)
from Data_ETA_Pipeline.feature_selection_engine import (
    load_snapshot_wide, _classify_series, run_full_source_pipeline, MIN_VALID_OBS
)
from Train.data_loader import load_target_data

logger = setup_logger(__file__, TEMP_DIR)
install_hooks()
register_atexit_dump("create_master_snapshots", output_dir=TEMP_DIR / "perf")
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

MASTER_BASE = DATA_PATH / "master_snapshots"
REGIME_CACHE_SCHEMA_VERSION = "2026-02-24-regime-cache-v1"
FEATURE_SELECTION_REGIME_REFRESH_MONTHS = 12
HARD_CODED_REGIME_STARTS = [
    ("Pre-GFC Great Moderation", pd.Timestamp("1998-01-01")),
    ("GFC Shock + Repair", pd.Timestamp("2008-01-01")),
    ("Late-Cycle Long Expansion", pd.Timestamp("2015-01-01")),
    ("COVID Shock + Great Resignation", pd.Timestamp("2020-03-01")),
    ("Inflation Tightening & Soft Landing", pd.Timestamp("2022-03-01")),
    ("AI and Trump Era with More Volatility", pd.Timestamp("2025-02-01")),
]

SOURCES = {
    'FRED_Employment_NSA': DATA_PATH / "fred_data_prepared_nsa" / "decades",
    'FRED_Employment_SA':  DATA_PATH / "fred_data_prepared_sa"  / "decades",
    'FRED_Exogenous': DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades",
    'Unifier': DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades",
    'ADP': DATA_PATH / "Exogenous_data" / "ADP_snapshots" / "decades",
    'NOAA': DATA_PATH / "Exogenous_data" / "exogenous_noaa_snapshots" / "decades",
    'Prosper': DATA_PATH / "Exogenous_data" / "prosper" / "decades",
}

# Ordered by typical execution time (longest to shortest) to optimize ProcessPool scheduling
SOURCE_EXEC_ORDER = ['FRED_Employment_NSA', 'FRED_Employment_SA', 'FRED_Exogenous',
                     'Unifier', 'Prosper', 'NOAA', 'ADP']

# All 4 combinations of target category and target source
TARGET_COMBOS = [
    ('nsa', 'first_release'),
    ('nsa', 'revised'),
    ('sa', 'first_release'),
    ('sa', 'revised'),
]
DEFAULT_TARGET_COMBOS = tuple(TARGET_COMBOS)
FAST_VERIFY_MONTH_WINDOW = 36


def _resolve_target_combos(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    target_source_scope: str = "auto",
) -> list[tuple[str, str]]:
    """
    Resolve which target branches to run for this invocation.

    Scopes:
    - all: run all branches in TARGET_COMBOS
    - revised: run only revised branches (nsa+sa)
    - first_release: run only first-release branches (nsa+sa)
    - auto: for short verification windows (<= FAST_VERIFY_MONTH_WINDOW), run
      revised-only for speed. For longer windows, run all branches.

    If TARGET_COMBOS was explicitly overridden (e.g., in tests), auto mode
    respects that override directly.
    """
    scope = (target_source_scope or "auto").strip().lower()

    if scope not in {"auto", "all", "revised", "first_release"}:
        raise ValueError(
            f"Invalid target_source_scope='{target_source_scope}'. "
            "Expected one of: auto, all, revised, first_release."
        )

    # Respect explicit combo overrides in auto mode (used by tests/advanced runs).
    if scope == "auto" and tuple(TARGET_COMBOS) != DEFAULT_TARGET_COMBOS:
        return list(TARGET_COMBOS)

    if scope == "auto":
        start_m = pd.Timestamp(start_dt).replace(day=1)
        end_m = pd.Timestamp(end_dt).replace(day=1)
        months_in_window = _month_distance(start_m, end_m) + 1
        scope = "revised" if months_in_window <= FAST_VERIFY_MONTH_WINDOW else "all"

    if scope == "all":
        return list(TARGET_COMBOS)
    if scope == "revised":
        return [(cat, src) for cat, src in TARGET_COMBOS if src == "revised"]
    return [(cat, src) for cat, src in TARGET_COMBOS if src == "first_release"]

# Source-specific minimum observation thresholds used before expensive selection.
# Long-history sources can tolerate stricter cutoffs, while shorter-history
# datasets keep a lower bar to avoid dropping all signal candidates.
SOURCE_MIN_VALID_OBS = {
    'FRED_Employment_NSA': 96,
    'FRED_Employment_SA': 96,
    'FRED_Exogenous': 84,
    'Unifier': 72,
    'ADP': 48,
    'NOAA': 36,
    'Prosper': 24,
}

def _month_key(value: pd.Timestamp | str) -> str:
    """Normalize a timestamp-like value to YYYY-MM string key."""
    return pd.Timestamp(value).strftime('%Y-%m')


def _month_distance(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Whole-month distance between two timestamps (b - a)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def _min_valid_obs_for_source(source_name: str) -> int:
    """Return source-specific minimum history requirement."""
    return int(SOURCE_MIN_VALID_OBS.get(source_name, MIN_VALID_OBS))


def _latest_snapshot_file(source_dir: Path, asof_month: pd.Timestamp | None = None) -> Path | None:
    """Return latest parquet snapshot for a source (optionally <= asof month)."""
    latest_files = sorted(source_dir.rglob('*.parquet'))
    if not latest_files:
        return None
    if asof_month is None:
        return latest_files[-1]

    asof_key = _month_key(asof_month)
    eligible = [p for p in latest_files if p.stem <= asof_key]
    if not eligible:
        return None
    return eligible[-1]


# =============================================================================
# CACHE LOGIC
# =============================================================================

def _get_cache_path(target_cat: str, target_source: str) -> Path:
    MASTER_BASE.mkdir(parents=True, exist_ok=True)
    return MASTER_BASE / f"selected_features_{target_cat}_{target_source}.json"


def _check_cache(target_cat: str, target_source: str) -> list[str]:
    """Return cached features if generated within the last 30 days."""
    inc_counter("master_snapshot.cache.branch.lookup", 1)
    cache_path = _get_cache_path(target_cat, target_source)
    if not cache_path.exists():
        inc_counter("master_snapshot.cache.branch.miss", 1)
        return None

    try:
        with perf_phase(
            "master_snapshot.cache.branch.read",
            target_cat=target_cat,
            target_source=target_source,
        ):
            with open(cache_path, 'r') as f:
                data = json.load(f)

        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days

        if (data.get("target_source") == target_source
                and data.get("target_cat") == target_cat
                and days_old < 30):
            logger.info(f"[{target_cat.upper()}/{target_source}] Using cached features "
                        f"(Age: {days_old} days).")
            inc_counter("master_snapshot.cache.branch.hit", 1)
            return data.get("features", [])

        logger.info(f"[{target_cat.upper()}/{target_source}] Cache expired "
                    f"(Age: {days_old} days). Rebuilding...")
        inc_counter("master_snapshot.cache.branch.miss", 1)
        inc_counter("master_snapshot.cache.branch.expired", 1)
        return None

    except Exception as e:
        logger.warning(f"Failed to read feature cache: {e}")
        inc_counter("master_snapshot.cache.branch.miss", 1)
        inc_counter("master_snapshot.cache.branch.read_error", 1)
        return None


def _save_cache(features: list[str], target_cat: str, target_source: str) -> None:
    cache_path = _get_cache_path(target_cat, target_source)
    data = {
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "target_source": target_source,
        "target_cat": target_cat,
        "features": sorted(list(set(features))),
    }
    with perf_phase(
        "master_snapshot.cache.branch.write",
        target_cat=target_cat,
        target_source=target_source,
        n_features=len(features),
    ):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=4)
    inc_counter("master_snapshot.cache.branch.write", 1)
    logger.info(f"Saved {len(features)} selected features to cache: {cache_path}")


def _get_regime_cache_path(target_cat: str, target_source: str, regime_cutoff: pd.Timestamp) -> Path:
    """Cache path for one branch-level regime cutoff."""
    cache_dir = MASTER_BASE / "regime_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cutoff_key = _month_key(regime_cutoff)
    return cache_dir / f"selected_features_{target_cat}_{target_source}_{cutoff_key}.json"


def _check_regime_cache(
    target_cat: str,
    target_source: str,
    regime_cutoff: pd.Timestamp,
) -> list[str] | None:
    """Load regime cache if TTL checks pass."""
    inc_counter("master_snapshot.cache.regime.lookup", 1)
    cache_path = _get_regime_cache_path(target_cat, target_source, regime_cutoff)
    if not cache_path.exists():
        inc_counter("master_snapshot.cache.regime.miss", 1)
        return None

    try:
        with perf_phase(
            "master_snapshot.cache.regime.read",
            target_cat=target_cat,
            target_source=target_source,
            regime_cutoff=_month_key(regime_cutoff),
        ):
            with open(cache_path, 'r') as f:
                data = json.load(f)

        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days

        if (days_old < 30
                and data.get("regime_cutoff_month") == _month_key(regime_cutoff)):
            feats = data.get("features", [])
            logger.info(f"[{target_cat.upper()}/{target_source}] Regime {data.get('regime_cutoff_month')} "
                        f"loaded from cache ({len(feats)} features, {days_old} days old).")
            inc_counter("master_snapshot.cache.regime.hit", 1)
            return feats

        inc_counter("master_snapshot.cache.regime.miss", 1)
        inc_counter("master_snapshot.cache.regime.expired", 1)
        return None

    except Exception:
        inc_counter("master_snapshot.cache.regime.miss", 1)
        inc_counter("master_snapshot.cache.regime.read_error", 1)
        return None


def _save_regime_cache(
    features: list[str],
    target_cat: str,
    target_source: str,
    regime_cutoff: pd.Timestamp,
) -> None:
    """Persist regime-selected features for one cutoff month."""
    cache_path = _get_regime_cache_path(target_cat, target_source, regime_cutoff)
    data = {
        "schema_version": REGIME_CACHE_SCHEMA_VERSION,
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "target_source": target_source,
        "target_cat": target_cat,
        "regime_cutoff_month": _month_key(regime_cutoff),
        "features": sorted(list(set(features))),
    }
    with perf_phase(
        "master_snapshot.cache.regime.write",
        target_cat=target_cat,
        target_source=target_source,
        regime_cutoff=_month_key(regime_cutoff),
        n_features=len(features),
    ):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=4)
    inc_counter("master_snapshot.cache.regime.write", 1)
    logger.info(f"[{target_cat.upper()}/{target_source}] Saved regime cache "
                f"{data['regime_cutoff_month']} with {len(features)} features.")


# =============================================================================
# PER-SOURCE CACHE LOGIC
# =============================================================================

def _get_source_cache_path(
    source_name: str,
    target_cat: str,
    target_source: str,
    asof_month: pd.Timestamp | None = None,
) -> Path:
    """Cache path for a single source's feature selection results."""
    cache_dir = MASTER_BASE / "source_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if asof_month is None:
        return cache_dir / f"source_{source_name}_{target_cat}_{target_source}.json"
    cutoff_key = _month_key(asof_month)
    return cache_dir / f"source_{source_name}_{target_cat}_{target_source}_asof_{cutoff_key}.json"


def _check_source_cache(
    source_name: str,
    target_cat: str,
    target_source: str,
    asof_month: pd.Timestamp | None = None,
) -> list[str] | None:
    """Return source cache if TTL checks pass."""
    inc_counter("master_snapshot.cache.source.lookup", 1)
    cache_path = _get_source_cache_path(source_name, target_cat, target_source, asof_month=asof_month)
    if not cache_path.exists():
        inc_counter("master_snapshot.cache.source.miss", 1)
        return None
    try:
        with perf_phase(
            "master_snapshot.cache.source.read",
            source=source_name,
            target_cat=target_cat,
            target_source=target_source,
            asof_month=_month_key(asof_month) if asof_month is not None else "latest",
        ):
            with open(cache_path, 'r') as f:
                data = json.load(f)
        last_run = datetime.strptime(data.get("last_run_date", "2000-01-01"), "%Y-%m-%d")
        days_old = (datetime.now() - last_run).days
        if days_old < 30:
            feats = data.get("features", [])
            logger.info(f"[{source_name}] Using cached source features "
                        f"({len(feats)} features, {days_old} days old)")
            inc_counter("master_snapshot.cache.source.hit", 1)
            return feats
        inc_counter("master_snapshot.cache.source.miss", 1)
        inc_counter("master_snapshot.cache.source.expired", 1)
        return None
    except Exception:
        inc_counter("master_snapshot.cache.source.miss", 1)
        inc_counter("master_snapshot.cache.source.read_error", 1)
        return None


def _save_source_cache(features: list[str], source_name: str,
                       target_cat: str, target_source: str,
                       asof_month: pd.Timestamp | None = None) -> None:
    """Save a single source's feature selection results."""
    cache_path = _get_source_cache_path(source_name, target_cat, target_source, asof_month=asof_month)
    data = {
        "last_run_date": datetime.now().strftime("%Y-%m-%d"),
        "source_name": source_name,
        "target_cat": target_cat,
        "target_source": target_source,
        "features": sorted(list(set(features))),
    }
    with perf_phase(
        "master_snapshot.cache.source.write",
        source=source_name,
        target_cat=target_cat,
        target_source=target_source,
        asof_month=_month_key(asof_month) if asof_month is not None else "latest",
        n_features=len(features),
    ):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=4)
    inc_counter("master_snapshot.cache.source.write", 1)
    logger.info(f"[{source_name}] Saved {len(features)} features to source cache")


# =============================================================================
# FEATURE SELECTION PIPELINE (PARALLEL WORKERS)
# =============================================================================

def _snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    return base_dir / decade / year / f"{date_ts.strftime('%Y-%m')}.parquet"


@profiled("create_master_snapshots._process_source_features")
def _process_source_features(source_name: str, source_dir: Path,
                             target_cat: str, target_source: str,
                             asof_month: pd.Timestamp | None = None) -> list[str]:
    """Worker function for ProcessPoolExecutor to run the 6-stage engine on one source."""
    label = f"{target_cat.upper()}/{target_source}"
    asof_label = _month_key(asof_month) if asof_month is not None else "latest"

    # 1. Find and load latest snapshot
    logger.info(f"[{label}] [{source_name}] Finding latest snapshot (as-of {asof_label})...")
    latest_path = _latest_snapshot_file(source_dir, asof_month=asof_month)
    if latest_path is None:
        logger.warning(f"[{label}] [{source_name}] No snapshot files found.")
        return []
    snap_wide = load_snapshot_wide(latest_path)
    if snap_wide.empty:
        logger.warning(f"[{label}] [{source_name}] Empty latest snapshot.")
        return []

    logger.info(f"[{label}] [{source_name}] Loaded {latest_path.stem}: {snap_wide.shape}")

    # 2. Apply source-specific min-history filter + zero-variance filter
    min_valid_obs = _min_valid_obs_for_source(source_name)
    valid_counts = snap_wide.count()
    short_features = valid_counts[valid_counts < min_valid_obs].index
    if len(short_features) > 0:
        snap_wide = snap_wide.drop(columns=short_features)
        logger.info(f"[{label}] [{source_name}] Dropped {len(short_features)} "
                     f"short-history features (<{min_valid_obs} obs)")

    zero_var = snap_wide.std() == 0
    if zero_var.any():
        snap_wide = snap_wide.loc[:, ~zero_var]
        logger.info(f"[{label}] [{source_name}] Dropped {zero_var.sum()} zero-variance features")

    if snap_wide.empty:
        logger.warning(f"[{label}] [{source_name}] No features remain after filtering.")
        return []

    # 3. Build series groups using source-specific taxonomy
    series_groups = defaultdict(list)
    for col in snap_wide.columns:
        grp = _classify_series(col, source_name)
        series_groups[grp].append(col)

    logger.info(f"[{label}] [{source_name}] {len(series_groups)} groups, "
                f"{snap_wide.shape[1]} total features")

    # 4. Load targets (MoM only)
    logger.info(f"[{label}] [{source_name}] Loading targets...")
    if target_source == 'revised':
        from Train.data_loader import build_revised_target
        inc_counter("master_snapshot.target.build_revised.invoked", 1)
        inc_counter("master_snapshot.cache.target_load.miss", 1)
        with perf_phase(
            "master_snapshot.fs.target.build_revised_target",
            source=source_name,
            target_cat=target_cat,
            target_source=target_source,
        ):
            target_df = build_revised_target(target_cat)
            target_df['ds'] = pd.to_datetime(target_df['ds'])
            target_df = target_df.sort_values('ds')
    else:
        # use_cache=False is intentional for strict point-in-time behavior in FS prep path.
        inc_counter("master_snapshot.cache.target_load.miss", 1)
        with perf_phase(
            "master_snapshot.fs.target.load_target_data",
            source=source_name,
            target_cat=target_cat,
            target_source=target_source,
        ):
            target_df = load_target_data(target_type=target_cat, release_type='first', use_cache=False)

    if target_df.empty or 'y_mom' not in target_df.columns:
        logger.error(f"[{label}] [{source_name}] Failed to load valid targets.")
        return []

    target_indexed = target_df.dropna(subset=['y_mom']).set_index('ds')
    y_mom = target_indexed['y_mom']

    # 5. Run full pipeline (Stages 1-6: MoM only)
    logger.info(f"[{label}] [{source_name}] Matrix shape {snap_wide.shape}. "
                f"Starting 6-stage pipeline...")
    try:
        survivors = run_full_source_pipeline(
            snap_wide, y_mom, source_name, source_dir, series_groups
        )
        logger.info(f"[{label}] [{source_name}] Engine returned {len(survivors)} features.")
        return survivors
    except Exception as e:
        logger.error(f"[{label}] [{source_name}] Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


@profiled("create_master_snapshots._run_parallel_feature_selection")
def _run_parallel_feature_selection(
    target_cat: str,
    target_source: str,
    asof_month: pd.Timestamp | None = None,
) -> list[str]:
    label = f"{target_cat.upper()}/{target_source}"
    asof_label = _month_key(asof_month) if asof_month is not None else "latest"
    logger.info(f"[{label}] Starting Feature Selection Across All Sources (as-of {asof_label})...")
    all_selected = []

    force_serial_small = (
        os.getenv("NFP_PERF_SERIAL_FS", "").strip() == "1"
        or os.getenv("NFP_PREPARE_FORCE_SERIAL", "").strip() == "1"
    )
    if force_serial_small:
        logger.info(f"[{label}] Profiling serial fallback enabled for small-source FS section.")
        inc_counter("master_snapshot.fs.serial_fallback_enabled", 1)

    # OOM Protection Strategy:
    # Massive datasets (FRED) chew up ~15-20GB of RAM *per worker*.
    # If we run them in parallel with others, macOS will instantly SIGKILL
    # the process for RAM exhaustion (Exit Code 137).
    # Therefore, we run the massive datasets sequentially first, then
    # parallelize the smaller ones.

    massive_sources = ['FRED_Employment_NSA', 'FRED_Employment_SA', 'FRED_Exogenous']
    small_sources = [s for s in SOURCE_EXEC_ORDER if s not in massive_sources]

    # Route FRED Employment: only process the branch matching target_cat
    # (nsa model only needs NSA features, sa model only needs SA features)
    skip_source = 'FRED_Employment_SA' if target_cat == 'nsa' else 'FRED_Employment_NSA'

    def _run_or_load_source(name):
        """Check per-source cache first; run pipeline only if cache miss."""
        cached = _check_source_cache(name, target_cat, target_source, asof_month=asof_month)
        if cached is not None:
            return cached
        feats = _process_source_features(
            name, SOURCES[name], target_cat, target_source, asof_month=asof_month
        )
        _save_source_cache(feats, name, target_cat, target_source, asof_month=asof_month)
        return feats

    # 1. Run massive sources sequentially (skip the irrelevant FRED branch)
    with perf_phase("master_snapshot.fs.massive_sources_sequential", target_cat=target_cat, target_source=target_source):
        for name in massive_sources:
            if name == skip_source:
                logger.info(f"[{label}] Skipping '{name}' (not needed for {target_cat} branch)")
                continue
            if name in SOURCES:
                logger.info(f"[{label}] Executing massive dataset '{name}' sequentially to prevent OOM...")
                try:
                    with perf_phase(
                        "master_snapshot.fs.massive_source",
                        source=name,
                        target_cat=target_cat,
                        target_source=target_source,
                    ):
                        feats = _run_or_load_source(name)
                    all_selected.extend(feats)
                    logger.info(f"+++ [{label}] {name} completed successfully. Added {len(feats)} features.")
                except Exception as e:
                    logger.error(f"--- [{label}] {name} failed: {e}")

    # 2. Run small sources in parallel (cache-aware)
    # Separate cached vs uncached sources to avoid spawning workers unnecessarily
    uncached_small = []
    with perf_phase("master_snapshot.fs.small_sources_cache_scan", target_cat=target_cat, target_source=target_source):
        for name in small_sources:
            cached = _check_source_cache(name, target_cat, target_source, asof_month=asof_month)
            if cached is not None:
                all_selected.extend(cached)
                logger.info(f"+++ [{label}] {name} loaded from cache. Added {len(cached)} features.")
            else:
                uncached_small.append(name)

    if uncached_small:
        inc_counter("master_snapshot.fs.small_sources_uncached", len(uncached_small))
        if force_serial_small:
            logger.info(f"[{label}] Executing {len(uncached_small)} uncached smaller datasets serially (profiling fallback)...")
            with perf_phase("master_snapshot.fs.small_sources_serial", n_sources=len(uncached_small)):
                for name in uncached_small:
                    try:
                        with perf_phase(
                            "master_snapshot.fs.small_source_serial",
                            source=name,
                            target_cat=target_cat,
                            target_source=target_source,
                        ):
                            feats = _process_source_features(
                                name,
                                SOURCES[name],
                                target_cat,
                                target_source,
                                asof_month=asof_month,
                            )
                        _save_source_cache(
                            feats, name, target_cat, target_source, asof_month=asof_month
                        )
                        all_selected.extend(feats)
                        logger.info(f"+++ [{label}] {name} completed successfully. Added {len(feats)} features.")
                    except Exception as e:
                        logger.error(f"--- [{label}] {name} failed in serial fallback: {e}")
        else:
            logger.info(f"[{label}] Executing {len(uncached_small)} uncached smaller datasets in parallel...")
            with perf_phase("master_snapshot.fs.small_sources_parallel", n_sources=len(uncached_small)):
                with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                    futures = {
                        executor.submit(_process_source_features, name, SOURCES[name],
                                        target_cat, target_source, asof_month): name
                        for name in uncached_small
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            with perf_phase(
                                "master_snapshot.fs.small_source_future_result",
                                source=name,
                                target_cat=target_cat,
                                target_source=target_source,
                            ):
                                feats = future.result()
                            _save_source_cache(
                                feats, name, target_cat, target_source, asof_month=asof_month
                            )
                            all_selected.extend(feats)
                            logger.info(f"+++ [{label}] {name} completed successfully. Added {len(feats)} features.")
                        except Exception as e:
                            logger.error(f"--- [{label}] {name} spawned an exception: {e}")

    all_selected = list(set(all_selected))
    logger.info(f"[{label}] Feature Selection Complete. Total surviving features: {len(all_selected)}")
    return all_selected


# =============================================================================
# MASTER GENERATION
# =============================================================================

def _normalize_to_wide(df):
    """Convert a raw source DataFrame to wide format with 'date' as a column."""
    if 'series_name' in df.columns and 'value' in df.columns:
        wide = df.pivot(index='date', columns='series_name', values='value').reset_index()
    else:
        wide = df

    if 'date' not in wide.columns:
        if isinstance(wide.index, pd.DatetimeIndex) or wide.index.name == 'date':
            wide = wide.reset_index()
        elif hasattr(wide.index, 'name') and wide.index.name is not None:
            wide = wide.reset_index()

    return wide


@profiled("create_master_snapshots._batch_load_source")
def _batch_load_source(source_name: str, source_dir: Path,
                       snapshot_months: list[pd.Timestamp],
                       allowed_features: set) -> dict[str, pd.DataFrame]:
    """Pre-load all snapshots for one source, filtered to allowed features.

    Returns a dict mapping month-key (YYYY-MM) to a filtered wide DataFrame.
    This eliminates redundant file reads when iterating over months.
    """
    result = {}
    for obs_month in snapshot_months:
        path = _snapshot_path(source_dir, obs_month)
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"[{source_name}] Failed to read {path}: {e}")
            continue
        if df.empty:
            continue

        wide = _normalize_to_wide(df)
        keep_cols = [c for c in wide.columns
                     if c in allowed_features or c in ['date', 'snapshot_date']]
        wide = wide[keep_cols]

        if not wide.drop(columns=['date', 'snapshot_date'], errors='ignore').empty:
            result[obs_month.strftime('%Y-%m')] = wide

    return result


@profiled("create_master_snapshots._load_all_sources_from_cache")
def _load_all_sources_from_cache(obs_month: pd.Timestamp,
                                 source_caches: dict[str, dict],
                                 ) -> pd.DataFrame:
    """Assemble a master row from pre-loaded source caches (O(1) lookup per source)."""
    month_key = obs_month.strftime('%Y-%m')
    results = []
    for source_name, month_dict in source_caches.items():
        wide = month_dict.get(month_key)
        if wide is not None:
            results.append(wide)

    if not results:
        return pd.DataFrame()

    master = results[0]
    for nxt in results[1:]:
        master = pd.merge(master, nxt, on='date', how='outer')
    return master


def _get_progress_path(target_cat: str, target_source: str) -> Path:
    """Path for the branch progress checkpoint file."""
    progress_dir = MASTER_BASE / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    return progress_dir / f"progress_{target_cat}_{target_source}.json"


def _load_progress(target_cat: str, target_source: str) -> set[str]:
    """Load the set of successfully completed month-keys for a branch."""
    path = _get_progress_path(target_cat, target_source)
    if not path.exists():
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return set(data.get("completed_months", []))
    except Exception:
        return set()


def _save_progress(target_cat: str, target_source: str,
                   completed_months: set[str]) -> None:
    """Persist the set of completed month-keys for crash-resume."""
    path = _get_progress_path(target_cat, target_source)
    data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_cat": target_cat,
        "target_source": target_source,
        "completed_months": sorted(completed_months)
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def _clear_progress(target_cat: str, target_source: str) -> None:
    """Remove the progress file when a branch completes fully."""
    path = _get_progress_path(target_cat, target_source)
    if path.exists():
        path.unlink()


def _build_feature_selection_regimes(
    snapshot_months: list[pd.Timestamp],
    refresh_months: int = FEATURE_SELECTION_REGIME_REFRESH_MONTHS,
) -> list[pd.Timestamp]:
    """Build ascending hard-coded feature-selection regime cutoffs.

    refresh_months is kept for backward compatibility but ignored.
    """
    _ = refresh_months
    if not snapshot_months:
        return []

    months = sorted(pd.Timestamp(m).replace(day=1) for m in snapshot_months)
    first_month = months[0]
    latest_month = months[-1]
    starts = [start for _, start in HARD_CODED_REGIME_STARTS]

    eligible = [start for start in starts if start <= latest_month]
    if not eligible:
        return []

    # For short run windows, keep only the regime active at window start plus
    # any newer regime starts inside the window.
    anchor_candidates = [start for start in eligible if start <= first_month]
    if anchor_candidates:
        anchor = max(anchor_candidates)
    else:
        anchor = min(eligible)

    return [start for start in eligible if start >= anchor]


def _resolve_regime_cutoff(
    obs_month: pd.Timestamp,
    regime_cutoffs: list[pd.Timestamp],
) -> pd.Timestamp:
    """Map observation month to the latest regime cutoff not in the future."""
    if not regime_cutoffs:
        raise ValueError("regime_cutoffs must not be empty")

    obs_key = _month_key(obs_month)
    selected = regime_cutoffs[0]
    for cutoff in regime_cutoffs:
        if _month_key(cutoff) <= obs_key:
            selected = cutoff
        else:
            break
    return selected


@profiled("create_master_snapshots.create_master_snapshots")
def create_master_snapshots(
    skip_existing: bool = False,
    target_source_scope: str | None = None,
    asof_start: str | None = None,
    asof_end: str | None = None,
):
    """
    Args:
        skip_existing: Skip months where master snapshot already exists.
        target_source_scope: Branch scope ('auto', 'all', 'revised', 'first_release').
        asof_start: Optional YYYY-MM lower bound for as-of months to process (inclusive).
                    Default None = no lower bound.
        asof_end: Optional YYYY-MM upper bound for as-of months to process (inclusive).
                  Default None = no upper bound.
    """
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    nfp_map = get_nfp_release_map(start_date=start_dt, end_date=end_dt)
    snapshot_pairs = sorted(nfp_map.items(), key=lambda x: x[0])

    # Bounded-run filtering: restrict which as-of months are processed.
    # Outputs for included months are identical to an unbounded run.
    if asof_start:
        asof_start_ts = pd.Timestamp(asof_start)
        before_count = len(snapshot_pairs)
        snapshot_pairs = [(obs, snap) for obs, snap in snapshot_pairs if obs >= asof_start_ts]
        logger.info(f"Bounded start filter: {before_count} -> {len(snapshot_pairs)} months (>= {asof_start})")
    if asof_end:
        asof_end_ts = pd.Timestamp(asof_end) + pd.offsets.MonthEnd(0)
        before_count = len(snapshot_pairs)
        snapshot_pairs = [(obs, snap) for obs, snap in snapshot_pairs if obs <= asof_end_ts]
        logger.info(f"Bounded end filter: {before_count} -> {len(snapshot_pairs)} months (<= {asof_end})")

    if not snapshot_pairs:
        logger.info("No NFP release dates found.")
        return

    scope = target_source_scope or os.getenv("MASTER_TARGET_SOURCE_SCOPE", "auto")
    target_combos = _resolve_target_combos(start_dt, end_dt, scope)

    logger.info(f"Target scope resolved to '{scope}'. Running {len(target_combos)} branch(es): "
                f"{target_combos}")

    # Each branch runs its own independent feature selection
    for target_cat, target_source in target_combos:
        label = f"{target_cat.upper()}/{target_source}"
        logger.info(f"========== COMMENCING BRANCH: {label} ==========")
        target_master_dir = MASTER_BASE / target_cat / target_source / "decades"

        # 1. Build leakage-safe regime feature sets (as-of cutoffs only).
        all_branch_months = [obs for obs, _ in snapshot_pairs]
        regime_cutoffs = _build_feature_selection_regimes(all_branch_months)
        if not regime_cutoffs:
            logger.error(f"[{label}] No regime cutoffs could be constructed. Skipping branch.")
            continue

        cutoff_keys = [_month_key(c) for c in regime_cutoffs]
        logger.info(f"[{label}] Building {len(regime_cutoffs)} hard-coded feature regimes: "
                    f"{cutoff_keys}")
        regime_feature_sets: dict[str, set[str]] = {}
        last_nonempty: set[str] | None = None

        for cutoff in regime_cutoffs:
            cutoff_key = _month_key(cutoff)
            allowed_list = _check_regime_cache(target_cat, target_source, cutoff)
            if allowed_list is None:
                logger.info(f"[{label}] Executing selection for regime cutoff {cutoff_key}...")
                allowed_list = _run_parallel_feature_selection(
                    target_cat, target_source, asof_month=cutoff
                )
                _save_regime_cache(allowed_list, target_cat, target_source, cutoff)

            allowed_set = set(allowed_list)
            if not allowed_set and last_nonempty:
                logger.warning(f"[{label}] Regime {cutoff_key} produced 0 features. "
                               f"Reusing prior regime set ({len(last_nonempty)} features).")
                allowed_set = set(last_nonempty)

            if not allowed_set:
                logger.error(f"[{label}] Regime {cutoff_key} produced no usable features.")
                continue

            regime_feature_sets[cutoff_key] = allowed_set
            last_nonempty = allowed_set
            logger.info(f"[{label}] Regime {cutoff_key}: {len(allowed_set)} whitelisted features.")

        if not regime_feature_sets:
            logger.error(f"[{label}] All regimes failed to produce features. Skipping branch.")
            continue

        # Preserve legacy cache contract for downstream consumers that load one list.
        latest_regime_key = sorted(regime_feature_sets.keys())[-1]
        _save_cache(sorted(regime_feature_sets[latest_regime_key]), target_cat, target_source)

        # 2. Determine which months to process (skip_existing + progress resume)
        completed_months = _load_progress(target_cat, target_source)
        if completed_months:
            logger.info(f"[{label}] Resuming from checkpoint: "
                        f"{len(completed_months)} months already completed")

        branch_snapshot_pairs = []
        for obs, snap in snapshot_pairs:
            month_key = obs.strftime('%Y-%m')
            # Skip if already in progress tracker
            if month_key in completed_months:
                continue
            # Skip if file exists and skip_existing is set
            if skip_existing and _snapshot_path(target_master_dir, obs).exists():
                completed_months.add(month_key)
                continue
            branch_snapshot_pairs.append((obs, snap))

        if not branch_snapshot_pairs:
            logger.info(f"[{label}] All months already completed. Skipping.")
            _clear_progress(target_cat, target_source)
            logger.info(f"========== BRANCH COMPLETE: {label} ==========\n")
            continue

        logger.info(f"[{label}] {len(branch_snapshot_pairs)} months to generate...")

        # 3. Route months to the most recent available feature regime.
        regime_to_pairs: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
        backfilled_to_first_regime = 0
        first_cutoff_key = _month_key(regime_cutoffs[0])
        for obs_month, snap_date in branch_snapshot_pairs:
            cutoff = _resolve_regime_cutoff(obs_month, regime_cutoffs)
            if _month_key(obs_month) < first_cutoff_key:
                backfilled_to_first_regime += 1
            cutoff_key = _month_key(cutoff)
            if cutoff_key not in regime_feature_sets:
                # Fall back to the nearest available past regime (never future).
                obs_key = _month_key(obs_month)
                eligible = sorted(
                    [k for k in regime_feature_sets.keys() if k <= obs_key]
                )
                if eligible:
                    cutoff_key = eligible[-1]
            if cutoff_key not in regime_feature_sets:
                logger.error(f"[{label}] Missing feature regime for {obs_month.strftime('%Y-%m')}.")
                continue
            regime_to_pairs[cutoff_key].append((obs_month, snap_date))

        if backfilled_to_first_regime:
            logger.info(
                f"[{label}] Backfilled {backfilled_to_first_regime} month(s) before first regime "
                f"cutoff {first_cutoff_key} using that regime's selected features."
            )

        if not regime_to_pairs:
            logger.error(f"[{label}] No months could be assigned to valid regimes. Skipping branch.")
            continue

        # 4. Batch-load sources and build masters per regime.
        skip_source = None
        if target_cat == 'nsa':
            skip_source = 'FRED_Employment_SA'
        elif target_cat == 'sa':
            skip_source = 'FRED_Employment_NSA'

        total = len(branch_snapshot_pairs)
        processed = 0

        for regime_key in sorted(regime_to_pairs.keys()):
            regime_pairs = regime_to_pairs[regime_key]
            regime_pairs = sorted(regime_pairs, key=lambda x: x[0])
            allowed_set = regime_feature_sets[regime_key]
            regime_months = [obs for obs, _ in regime_pairs]
            source_caches = {}

            logger.info(f"[{label}] Regime {regime_key}: {len(regime_pairs)} months, "
                        f"{len(allowed_set)} selected features.")

            for name, sdir in SOURCES.items():
                if name == skip_source:
                    continue
                logger.info(f"[{label}] [{regime_key}] Batch-loading {name}...")
                source_caches[name] = _batch_load_source(
                    name, sdir, regime_months, allowed_set
                )
                loaded_count = len(source_caches[name])
                logger.info(f"[{label}] [{regime_key}] {name}: "
                            f"{loaded_count}/{len(regime_months)} months loaded")

            for obs_month, snap_date in regime_pairs:
                try:
                    master = _load_all_sources_from_cache(obs_month, source_caches)
                    if master.empty:
                        logger.debug(f"[{label}] Skipped {obs_month.date()} (no valid data)")
                        completed_months.add(obs_month.strftime('%Y-%m'))
                        processed += 1
                        continue

                    master['date'] = pd.to_datetime(master['date'])
                    master['snapshot_date'] = snap_date

                    save_dir = _snapshot_path(target_master_dir, obs_month).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    master.to_parquet(save_dir / f"{obs_month.strftime('%Y-%m')}.parquet", index=False)

                    completed_months.add(obs_month.strftime('%Y-%m'))
                    processed += 1

                    if processed % 12 == 0 or processed == total:
                        logger.info(f"[{label}] Generated {obs_month.date()} "
                                    f"({processed}/{total}, Cols: {master.shape[1]})")
                        # Checkpoint every 12 months
                        _save_progress(target_cat, target_source, completed_months)

                except Exception as e:
                    logger.error(f"[{label}] Error processing {obs_month.date()}: {e}")
                    # Save progress so we can resume past this point
                    _save_progress(target_cat, target_source, completed_months)

            # Free memory between regimes.
            del source_caches

        # Branch complete — clean up progress file
        _clear_progress(target_cat, target_source)
        logger.info(f"========== BRANCH COMPLETE: {label} ==========\n")

    logger.info("Master Snapshots generation completely finished for all branches.")

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument(
        '--target-source-scope',
        choices=['auto', 'all', 'revised', 'first_release'],
        default=None,
        help=("Branch scope: auto (short windows -> revised-only), all, "
              "revised, or first_release"),
    )
    parser.add_argument(
        '--asof-start',
        default=None,
        help='Lower bound for as-of months (YYYY-MM, inclusive). Default: no bound.',
    )
    parser.add_argument(
        '--asof-end',
        default=None,
        help='Upper bound for as-of months (YYYY-MM, inclusive). Default: no bound.',
    )
    args = parser.parse_args()

    start_time = time.time()
    create_master_snapshots(
        skip_existing=args.skip_existing,
        target_source_scope=args.target_source_scope,
        asof_start=args.asof_start,
        asof_end=args.asof_end,
    )
    logger.info(f"Total Master Time: {(time.time() - start_time)/60:.1f} minutes")
