"""
Data Loading Utilities for LightGBM NFP Model

Functions for loading snapshots, target data, and building training datasets.
Extracted from train_lightgbm_nfp.py for maintainability.

OPTIMIZATIONS:
- LRU cache for snapshot loading (avoids redundant I/O)
- Vectorized pivot_snapshot_to_wide using pandas native operations
- Pre-computed lag indices for batch feature generation

MULTI-TARGET SUPPORT:
- Supports 4 target configurations: (nsa/sa) x (first/last release)
- Cache keys include both target_type and release_type
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger
from utils.transforms import winsorize_covid_period
from Data_ETA_Pipeline.perf_stats import inc_counter, install_hooks, profiled, perf_phase
from Train.config import (
    MASTER_SNAPSHOTS_BASE,
    FRED_SNAPSHOTS_DIR,
    NFP_TARGET_DIR,  # needed for revised-target cache fast-path in load_target_data()
    get_master_snapshots_dir,
    get_target_path,
    get_model_id,
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    VALID_TARGET_SOURCES,
    REVISED_TARGET_SERIES,
)

logger = setup_logger(__file__, TEMP_DIR)
install_hooks()

# Module-level cache for loaded data
_snapshot_cache: Dict[str, pd.DataFrame] = {}
_target_cache: Dict[str, pd.DataFrame] = {}
NOAA_MAX_FFILL_MONTHS = 6
NOAA_STALENESS_SUFFIX = "__staleness_months"


# =============================================================================
# FEATURE NAME SANITIZATION (for LightGBM compatibility)
# =============================================================================

# Pre-compiled patterns for sanitize_feature_name (avoid recompilation per call)
_SANITIZE_MULTI_CHAR = {
    '+': 'plus', '%': 'pct', '&': '_and_', '<': '_lt_', '>': '_gt_',
}
_SANITIZE_MULTI_RE = re.compile('|'.join(re.escape(k) for k in _SANITIZE_MULTI_CHAR))
_SANITIZE_TO_UNDERSCORE = re.compile(r"[|\s\[\]{}\\,()\?/:;!@#$*=.<>]")
_SANITIZE_STRIP_QUOTES = re.compile(r"[\"']")
_SANITIZE_INTERIOR_HYPHEN = re.compile(r'(?<!^)-(?!$)')
_SANITIZE_COLLAPSE = re.compile(r'_+')


def sanitize_feature_name(name: str) -> str:
    """
    Sanitize feature name for LightGBM compatibility.

    LightGBM doesn't support special JSON characters in feature names.
    This function replaces problematic characters with safe alternatives.

    Args:
        name: Original feature name

    Returns:
        Sanitized feature name
    """
    # Multi-char replacements first (%, &, +, <, >)
    name = _SANITIZE_MULTI_RE.sub(lambda m: _SANITIZE_MULTI_CHAR[m.group()], name)
    # Strip quotes entirely
    name = _SANITIZE_STRIP_QUOTES.sub('', name)
    # Replace interior hyphens with underscores
    name = _SANITIZE_INTERIOR_HYPHEN.sub('_', name)
    # Replace all single-char specials with underscores in one pass
    name = _SANITIZE_TO_UNDERSCORE.sub('_', name)
    # Collapse multiple underscores and strip
    name = _SANITIZE_COLLAPSE.sub('_', name).strip('_')
    return name


def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize all column names in a DataFrame for LightGBM compatibility.

    Args:
        df: DataFrame with potentially problematic column names

    Returns:
        DataFrame with sanitized column names
    """
    df = df.copy()
    df.columns = [sanitize_feature_name(str(c)) for c in df.columns]
    return df


def _is_noaa_feature_name(name: str) -> bool:
    """Heuristic NOAA feature detector for raw/sanitized column names."""
    col = str(name).upper()
    return col.startswith("NOAA") or "_NOAA_" in col


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_fred_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    """
    Constructs path to raw FRED employment snapshot for the specified month.

    Note: Training no longer loads FRED snapshots directly (they are included in
    master snapshots). This function is kept for build_revised_target() which needs
    raw FRED levels.

    Args:
        snapshot_date (pd.Timestamp): The vintage/snapshot month-end date.

    Returns:
        Path: The fully resolved filesystem path.
    """
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')
    return FRED_SNAPSHOTS_DIR / decade / year / f"{month_str}.parquet"




def get_master_snapshot_path(snapshot_date: pd.Timestamp,
                            target_type: str = 'nsa',
                            target_source: str = 'first_release') -> Path:
    """
    Constructs the file path to the feature-selected master snapshot. This file
    combines all data sources (FRED employment, FRED exog, Unifier, ADP, NOAA, Prosper)
    already merged and filtered to selected features.

    Args:
        snapshot_date (pd.Timestamp): The vintage/snapshot month-end date.
        target_type: 'nsa' or 'sa' — determines which feature-selected variant.
        target_source: 'first_release' or 'revised' — determines target used for selection.

    Returns:
        Path: The fully resolved filesystem path to the master snapshot.
    """
    base_dir = get_master_snapshots_dir(target_type, target_source)
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime('%Y-%m')
    return base_dir / decade / year / f"{month_str}.parquet"


@lru_cache(maxsize=1)
def _load_revised_target_audit_cache() -> Dict[str, pd.DataFrame]:
    """
    Load the latest audit file and retain only revised-target series.

    Revised targets need values as-of the M+1 release cutoff inclusive (<= cutoff).
    Monthly snapshots are strict pre-cutoff views (< cutoff), which can miss
    boundary vintages for some months.
    """
    fred_root = DATA_PATH / "fred_data"
    audit_files = list(fred_root.glob("_audit_asof_*.parquet"))
    if not audit_files:
        logger.warning("No audit file found under data/fred_data; revised target will fallback to snapshots.")
        return {}

    def _audit_asof_key(path: Path) -> pd.Timestamp:
        m = re.match(r"_audit_asof_(\d{4}-\d{2}-\d{2})\.parquet$", path.name)
        if not m:
            return pd.Timestamp.min
        return pd.to_datetime(m.group(1), errors="coerce")

    latest_audit = max(audit_files, key=_audit_asof_key)
    wanted = set(REVISED_TARGET_SERIES.values())

    try:
        audit = pd.read_parquet(
            latest_audit,
            columns=["unique_id", "ds", "y", "realtime_start"],
        )
    except Exception as exc:
        logger.warning(
            f"Failed to read audit file {latest_audit.name}; revised target will fallback to snapshots. Error: {exc}"
        )
        return {}

    audit = audit[audit["unique_id"].isin(wanted)].copy()
    if audit.empty:
        logger.warning(
            f"Audit file {latest_audit.name} contains no revised-target series; "
            "revised target will fallback to snapshots."
        )
        return {}

    audit["ds"] = pd.to_datetime(audit["ds"], errors="coerce")
    audit["realtime_start"] = pd.to_datetime(audit["realtime_start"], errors="coerce")
    audit["y"] = pd.to_numeric(audit["y"], errors="coerce")
    audit = audit.dropna(subset=["ds", "realtime_start", "y"])

    out: Dict[str, pd.DataFrame] = {}
    for uid in wanted:
        sub = audit[audit["unique_id"] == uid][["ds", "y", "realtime_start"]]
        if not sub.empty:
            out[uid] = sub.sort_values(["ds", "realtime_start"]).reset_index(drop=True)

    logger.info(
        f"Loaded revised-target audit cache from {latest_audit.name} "
        f"for series: {sorted(out.keys())}"
    )
    return out


# =============================================================================
# DATA LOADING (with caching)
# =============================================================================

def load_fred_snapshot(snapshot_date: pd.Timestamp, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load FRED employment snapshot for a given date.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: date, value, series_name, series_code, release_date
    """
    cache_key = f"fred_{snapshot_date.strftime('%Y-%m')}"

    if use_cache and cache_key in _snapshot_cache:
        cached = _snapshot_cache[cache_key]
        return cached.copy() if cached is not None else None

    path = get_fred_snapshot_path(snapshot_date)
    if not path.exists():
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    # Check for empty or corrupt file
    if path.stat().st_size == 0:
        logger.warning(f"Skipping empty FRED snapshot: {path}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to read FRED snapshot {path}: {e}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    if use_cache:
        _snapshot_cache[cache_key] = df

    return df.copy() if use_cache else df


@profiled("train.data_loader.load_master_snapshot")
def load_master_snapshot(snapshot_date: pd.Timestamp,
                        target_type: str = 'nsa',
                        target_source: str = 'first_release',
                        use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load feature-selected master snapshot for a given date.

    The master snapshots are pre-merged wide-format files containing ALL data sources
    (FRED employment + exogenous) filtered to the features selected by the 7-stage engine.

    Args:
        snapshot_date: Month-end timestamp (e.g., 2024-10-31)
        target_type: 'nsa' or 'sa'
        target_source: 'first_release' or 'revised'
        use_cache: Whether to use/populate the module cache

    Returns:
        Wide-format DataFrame with columns: date, snapshot_date, + feature columns
    """
    cache_key = f"master_{target_type}_{target_source}_{snapshot_date.strftime('%Y-%m')}"

    if use_cache and cache_key in _snapshot_cache:
        cached = _snapshot_cache[cache_key]
        return cached.copy() if cached is not None else None

    path = get_master_snapshot_path(snapshot_date, target_type, target_source)
    if not path.exists():
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    # Check for empty or corrupt file
    if path.stat().st_size == 0:
        logger.warning(f"Skipping empty Master snapshot: {path}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to read Master snapshot {path}: {e}")
        if use_cache:
            _snapshot_cache[cache_key] = None
        return None

    if use_cache:
        _snapshot_cache[cache_key] = df

    return df.copy() if use_cache else df


def clear_snapshot_cache() -> None:
    """Clear the snapshot cache to free memory."""
    global _snapshot_cache
    _snapshot_cache.clear()
    _load_revised_target_audit_cache.cache_clear()
    logger.info("Snapshot cache cleared")


@profiled("data_loader.load_target_data")
def load_target_data(
    target_type: str = 'nsa',
    release_type: str = 'first',
    target_source: str = 'first_release',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load and prepare target data (NSA or SA NFP levels) with derived features.

    Args:
        target_type: 'nsa' for non-seasonally adjusted, 'sa' for seasonally adjusted
        release_type: 'first' for initial release, 'last' for final revised release
        target_source: 'first_release' for original release, 'revised' for once-revised
            (from M+1 FRED snapshot). Revised targets use raw FRED snapshot levels.
        use_cache: Whether to use/populate the module cache

    Returns:
        DataFrame with columns: ds, y (level), y_mom (month-on-month change),
        and additional momentum/divergence features
    """
    # Validate inputs
    target_type = target_type.lower()
    release_type = release_type.lower()

    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: {target_type}. Must be one of {VALID_TARGET_TYPES}")
    if release_type not in VALID_RELEASE_TYPES:
        raise ValueError(f"Invalid release_type: {release_type}. Must be one of {VALID_RELEASE_TYPES}")
    if target_source not in VALID_TARGET_SOURCES:
        raise ValueError(f"Invalid target_source: {target_source}. Must be one of {VALID_TARGET_SOURCES}")

    cache_key = f"target_{get_model_id(target_type, release_type, target_source)}"

    if use_cache and cache_key in _target_cache:
        inc_counter("data_loader.target_cache.hit", 1)
        return _target_cache[cache_key].copy()
    if use_cache:
        inc_counter("data_loader.target_cache.miss", 1)
    else:
        inc_counter("data_loader.target_cache.bypass", 1)

    # Revised target: REQUIRE pre-built parquet cache (no on-the-fly rebuild).
    # Files are generated during data load by fred_employment_pipeline.py via
    # Prepare_Data/build_revised_targets.py and stored under data/NFP_target.
    if target_source == 'revised':
        revised_cache_path = NFP_TARGET_DIR / f"y_{target_type}_revised.parquet"

        if not revised_cache_path.exists():
            inc_counter("data_loader.revised_target.cache_miss", 1)
            raise FileNotFoundError(
                f"Required revised target cache missing: {revised_cache_path}. "
                "Run the data load stage (fred_employment) to materialize revised targets."
            )

        inc_counter("data_loader.revised_target.cache_hit", 1)
        logger.info(f"Loading cached revised {target_type.upper()} target from {revised_cache_path.name}")
        df = pd.read_parquet(revised_cache_path)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)

        model_id = get_model_id(target_type, release_type, target_source)
        if use_cache:
            _target_cache[cache_key] = df
            logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations (cached)")
        return df.copy() if use_cache else df

    # Load appropriate target file using dynamic path
    target_path = get_target_path(target_type, release_type)
    logger.info(f"Loading {target_type.upper()} {release_type} release target data")

    if not target_path.exists():
        if use_cache:
            _target_cache[cache_key] = None
        raise FileNotFoundError(f"Target file not found: {target_path}")

    df = pd.read_parquet(target_path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # Calculate MoM change (our prediction target)
    df['y_mom'] = df['y'].diff()

    # ── P2-1: Winsorize COVID months at load time for all targets ──
    # Applied BEFORE rolling stats so all downstream features see clipped values.
    # This affects all 4 variants: NSA/SA × first_release/revised.
    df_indexed = df.set_index('ds')
    df_indexed['y']     = winsorize_covid_period(df_indexed['y'])
    df_indexed['y_mom'] = winsorize_covid_period(df_indexed['y_mom'])
    df = df_indexed.reset_index()

    # Rolling averages (for momentum/divergence features) - vectorized
    df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
    df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
    df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()

    # MoM rolling averages
    df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()

    # Divergence: Current minus rolling average (momentum indicator)
    df['divergence_3m'] = df['y'] - df['y_rolling_3m']
    df['divergence_6m'] = df['y'] - df['y_rolling_6m']

    # Acceleration: Diff of diffs (second derivative - is hiring accelerating?)
    df['acceleration'] = df['y_mom'].diff()

    # YoY change
    df['y_yoy'] = df['y'].diff(12)
    df['y_mom_yoy'] = df['y_mom'].diff(12)

    # Drop first row (no MoM for first observation) but keep future rows with NaN y
    df = df[df.index != 0].reset_index(drop=True)

    model_id = get_model_id(target_type, release_type)
    if use_cache:
        _target_cache[cache_key] = df
        logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations (cached)")
    else:
        logger.info(f"Loaded {model_id.upper()} target data: {len(df)} observations")

    return df.copy() if use_cache else df


def load_all_target_data(release_type: str = 'first',
                        target_source: str = 'first_release') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both NSA and SA target data with all derived features for a given release type.

    Args:
        release_type: 'first' for initial release, 'last' for final revised
        target_source: 'first_release' or 'revised'

    Returns:
        Tuple of (nsa_df, sa_df) with columns:
        - ds: date
        - y: level
        - y_mom: month-on-month change
        - Additional momentum/divergence features
    """
    nsa_df = load_target_data('nsa', release_type=release_type, target_source=target_source)
    sa_df = load_target_data('sa', release_type=release_type, target_source=target_source)
    return nsa_df, sa_df


@profiled("data_loader.build_revised_target")
def build_revised_target(target_type: str = 'nsa') -> pd.DataFrame:
    """
    Build revised target data from raw FRED snapshots with audit fallbacks.

    For each month M:
    1) Use the strict pre-cutoff M+1 snapshot (historical pipeline behavior)
    2) If levels are missing at the boundary, fallback to audit vintages using
       <= cutoff for that month only
    3) Compute:
        revised_mom[M] = level[M] - level[M-1]

    This captures the "once-revised" MoM change, which includes BLS revisions
    to the M-1 level that occur between the M and M+1 NFP releases.

    Args:
        target_type: 'nsa' or 'sa'

    Returns:
        DataFrame matching load_target_data() output shape:
        ds, y, y_mom, release_date, and derived momentum features.
    """
    series_name = REVISED_TARGET_SERIES[target_type]

    # Load first-release target to get the list of months and release_dates
    first_release_path = get_target_path(target_type, 'first')
    first_release_df = pd.read_parquet(first_release_path)
    first_release_df['ds'] = pd.to_datetime(first_release_df['ds'])
    first_release_df = first_release_df.sort_values('ds').reset_index(drop=True)

    if 'release_date' in first_release_df.columns:
        first_release_df['release_date'] = pd.to_datetime(first_release_df['release_date'], errors='coerce')
    else:
        first_release_df['release_date'] = pd.NaT
    first_release_release_map = first_release_df.set_index('ds')['release_date']

    audit_cache = _load_revised_target_audit_cache()
    audit_series = audit_cache.get(series_name)
    use_audit = audit_series is not None and not audit_series.empty
    if not use_audit:
        logger.warning(
            f"Audit cache unavailable for {series_name}; revised target boundary fallbacks are disabled."
        )

    def _level_asof_inclusive(
        series_df: pd.DataFrame,
        obs_month: pd.Timestamp,
        cutoff: pd.Timestamp
    ) -> Optional[float]:
        """Return latest level for obs_month where realtime_start <= cutoff."""
        if series_df is None or series_df.empty or pd.isna(cutoff):
            return None
        rows = series_df[
            (series_df['ds'] == obs_month) &
            (series_df['realtime_start'] <= cutoff)
        ]
        if rows.empty:
            return None
        val = rows.iloc[-1]['y']
        return None if pd.isna(val) else float(val)

    records = []
    for _, row in first_release_df.iterrows():
        m = row['ds']
        release_date = row.get('release_date', pd.NaT)
        m_minus1 = m - pd.DateOffset(months=1)
        m_plus1 = m + pd.DateOffset(months=1)
        cutoff = first_release_release_map.get(m_plus1, pd.NaT)

        level_m = None
        level_m1 = None

        # Primary path: strict pre-cutoff snapshot (historical behavior).
        snapshot_date = m_plus1
        snap = load_fred_snapshot(snapshot_date, use_cache=True)
        if snap is not None:
            series_data = snap[snap['series_name'] == series_name]
            if not series_data.empty:
                levels = (series_data[['date', 'value']]
                          .drop_duplicates('date')
                          .set_index('date')['value']
                          .sort_index())
                val_m = levels.get(m)
                val_m1 = levels.get(m_minus1)
                if val_m is not None and not pd.isna(val_m):
                    level_m = float(val_m)
                if val_m1 is not None and not pd.isna(val_m1):
                    level_m1 = float(val_m1)

        # Boundary fallback: if snapshot missed one/both levels, try inclusive
        # cutoff from audit vintages to capture same-day release entries.
        if use_audit and (level_m is None or level_m1 is None):
            audit_m = _level_asof_inclusive(audit_series, m, cutoff)
            audit_m1 = _level_asof_inclusive(audit_series, m_minus1, cutoff)

            if audit_m is not None and audit_m1 is not None:
                level_m, level_m1 = audit_m, audit_m1
            else:
                if level_m is None and audit_m is not None:
                    level_m = audit_m
                if level_m1 is None and audit_m1 is not None:
                    level_m1 = audit_m1

        if level_m is not None and level_m1 is not None:
            records.append({'ds': m, 'y': float(level_m),
                            'y_mom': float(level_m) - float(level_m1),
                            'release_date': release_date})
        elif level_m is not None:
            records.append({'ds': m, 'y': float(level_m), 'y_mom': np.nan,
                            'release_date': release_date})
        else:
            records.append({'ds': m, 'y': np.nan, 'y_mom': np.nan,
                            'release_date': release_date})

    df = pd.DataFrame(records)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # ── P2-1: Winsorize COVID months (same as load_target_data) ──
    df_indexed = df.set_index('ds')
    df_indexed['y']     = winsorize_covid_period(df_indexed['y'])
    df_indexed['y_mom'] = winsorize_covid_period(df_indexed['y_mom'])
    df = df_indexed.reset_index()

    # ── P2-2: Add operational_available_date ──
    # For month M: the revised label is only observable after the M+1 NFP release.
    # operational_available_date = release_date of month M+1 (the next print).
    first_release_df_lookup = first_release_df.set_index('ds')
    def _next_release_date(m: pd.Timestamp) -> pd.Timestamp:
        """Return the release_date for the month after m, or NaT if unknown."""
        next_m = m + pd.DateOffset(months=1)
        row = first_release_df_lookup.get('release_date')
        if row is None:
            return pd.NaT
        return pd.to_datetime(first_release_df_lookup.loc[next_m, 'release_date'])\
            if next_m in first_release_df_lookup.index else pd.NaT

    df['operational_available_date'] = df['ds'].apply(_next_release_date)

    # Derived features (same as load_target_data)
    df['y_rolling_3m'] = df['y'].rolling(window=3, min_periods=1).mean()
    df['y_rolling_6m'] = df['y'].rolling(window=6, min_periods=1).mean()
    df['y_rolling_12m'] = df['y'].rolling(window=12, min_periods=1).mean()
    df['y_mom_rolling_3m'] = df['y_mom'].rolling(window=3, min_periods=1).mean()
    df['y_mom_rolling_6m'] = df['y_mom'].rolling(window=6, min_periods=1).mean()
    df['divergence_3m'] = df['y'] - df['y_rolling_3m']
    df['divergence_6m'] = df['y'] - df['y_rolling_6m']
    df['acceleration'] = df['y_mom'].diff()
    df['y_yoy'] = df['y'].diff(12)
    df['y_mom_yoy'] = df['y_mom'].diff(12)

    # Drop first row (no MoM for first observation)
    df = df[df.index != 0].reset_index(drop=True)

    # Diagnostic guardrail: a long contiguous NaN run usually means upstream
    # snapshot/calendar issues and should be surfaced loudly.
    null_mask = df['y_mom'].isna().to_numpy()
    longest_run = 0
    current_run = 0
    for is_null in null_mask:
        if is_null:
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0
    if longest_run >= 12:
        logger.warning(
            f"Revised {target_type.upper()} target has a long contiguous NaN y_mom run "
            f"({longest_run} months). Check FRED snapshot coverage and release-date imputation."
        )

    logger.info(f"Built revised {target_type.upper()} target: {len(df)} months, "
                f"{df['y_mom'].notna().sum()} with valid MoM")
    return df


def clear_target_cache() -> None:
    """Clear the target data cache to free memory."""
    global _target_cache
    _target_cache.clear()
    logger.info("Target cache cleared")


def _build_lagged_target_feature_frame(
    target_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Build PIT-safe lagged target features indexed by target month.

    A value at month t only uses information available strictly before t.
    """
    df = target_df.sort_values('ds').set_index('ds')
    mom = df['y_mom']
    level = df['y']

    result = pd.DataFrame(index=df.index)

    # Canonical lag set
    result[f'{prefix}_mom_lag1'] = mom.shift(1)
    result[f'{prefix}_level_lag1'] = level.shift(1)
    result[f'{prefix}_mom_lag2'] = mom.shift(2)
    result[f'{prefix}_mom_lag3'] = mom.shift(3)
    result[f'{prefix}_mom_lag6'] = mom.shift(6)
    result[f'{prefix}_mom_lag12'] = mom.shift(12)

    # Acceleration lags
    accel = mom.diff(1)
    result[f'{prefix}_accel_lag1'] = accel.shift(1)
    result[f'{prefix}_accel_lag3'] = accel.shift(3)
    result[f'{prefix}_accel_lag6'] = accel.shift(6)

    # Rolling momentum trend
    result[f'{prefix}_mom_rolling_3m'] = mom.rolling(3, min_periods=3).mean().shift(1)
    result[f'{prefix}_mom_rolling_6m'] = mom.rolling(6, min_periods=6).mean().shift(1)
    result[f'{prefix}_mom_rolling_12m'] = mom.rolling(12, min_periods=12).mean().shift(1)

    # Rolling volatility regime
    result[f'{prefix}_mom_vol_6m'] = mom.rolling(6, min_periods=6).std().shift(1)
    result[f'{prefix}_mom_vol_12m'] = mom.rolling(12, min_periods=12).std().shift(1)

    # Last print vs recent trend
    result[f'{prefix}_mom_vs_trend'] = (
        result[f'{prefix}_mom_lag1'] - result[f'{prefix}_mom_rolling_3m']
    )

    # Year-over-year change in momentum
    result[f'{prefix}_mom_yoy'] = mom.diff(12).shift(1)

    # Expansion/stall regime indicator
    result[f'{prefix}_positive_months_6m'] = (
        (mom > 0).astype(float).rolling(6, min_periods=6).sum().shift(1)
    )

    return result


def get_lagged_target_features(
    target_df: pd.DataFrame,
    target_month: pd.Timestamp,
    prefix: str = 'nfp'
) -> Dict[str, float]:
    """
    Get lagged target features for a specific prediction month.

    CRITICAL: Only use data from BEFORE target_month to avoid look-ahead bias.
    The target is released in the following month, so for month M prediction,
    we only know NFP values through month M-1.

    Args:
        target_df: DataFrame with target values
        target_month: Month we're predicting
        prefix: Prefix for feature names ('nfp_nsa' or 'nfp_sa')

    Returns:
        Dictionary of lagged features
    """
    if target_df.empty:
        return {}

    target_month = pd.Timestamp(target_month).replace(day=1)

    # Ensure target_month exists in the index so we can retrieve a PIT row even
    # for future months that are not yet present in target_df.
    working_df = target_df
    if not (target_df['ds'] == target_month).any():
        extra_row = pd.DataFrame({'ds': [target_month], 'y': [np.nan], 'y_mom': [np.nan]})
        working_df = pd.concat([target_df, extra_row], ignore_index=True)

    feature_frame = _build_lagged_target_feature_frame(working_df, prefix)
    if target_month not in feature_frame.index:
        return {}

    row = feature_frame.loc[target_month]
    features: Dict[str, float] = {}
    for col, val in row.items():
        if pd.notna(val):
            features[col] = float(val)

    return features


@profiled("train.data_loader.batch_lagged_target_features")
def batch_lagged_target_features(
    target_df: pd.DataFrame,
    prefix: str = 'nfp',
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Vectorized batch computation of lagged target features for ALL months at once.

    Replaces per-month get_lagged_target_features() calls with a single pass
    using pandas shift/rolling. Returns a lookup dict for O(1) access per month.

    Semantics match get_lagged_target_features exactly:
    - shift(1) = "most recent value before this month" (lag 1)
    - rolling(N, min_periods=N) on shifted data = rolling average of N prior months

    Args:
        target_df: DataFrame with ds, y, y_mom columns
        prefix: Feature name prefix ('nfp_nsa' or 'nfp_sa')

    Returns:
        Dict mapping target_month -> {feature_name: value} (NaN features omitted)
    """
    result = _build_lagged_target_feature_frame(target_df, prefix)

    # Convert to dict-of-dicts, dropping NaN values per row
    lookup: Dict[pd.Timestamp, Dict[str, float]] = {}
    cols = result.columns.tolist()
    arr = result.values
    for i, ts in enumerate(result.index):
        row_vals = arr[i]
        features = {}
        for j, col in enumerate(cols):
            v = row_vals[j]
            if not np.isnan(v):
                features[col] = float(v)
        lookup[ts] = features

    return lookup


def prefilter_snapshot(
    df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Pre-filter a loaded snapshot to only include series matching selected features.

    Call this ONCE right after loading a snapshot, before passing it to
    multiple pivot_snapshot_to_wide() calls. This avoids repeated filtering
    inside each pivot call, reducing DataFrame copies and groupby operations
    on the full snapshot.

    Args:
        df: Snapshot DataFrame (wide or long format)
        selected_features: List of sanitized feature names to keep

    Returns:
        Filtered DataFrame with only relevant series/columns
    """
    if df is None or df.empty or not selected_features:
        return df

    selected_set = set(selected_features)
    is_wide = 'series_name' not in df.columns

    if is_wide:
        # Wide format: filter columns (keep 'date' index/column + selected)
        cols_to_keep = []
        for col in df.columns:
            if col == 'date':
                cols_to_keep.append(col)
            elif sanitize_feature_name(str(col)) in selected_set:
                cols_to_keep.append(col)
        if not cols_to_keep:
            return pd.DataFrame()
        return df[cols_to_keep]
    else:
        # Long format: filter rows by series_name
        unique_raw = df['series_name'].unique()
        allowed_raw = [raw for raw in unique_raw if sanitize_feature_name(str(raw)) in selected_set]
        if not allowed_raw:
            return pd.DataFrame()
        return df[df['series_name'].isin(set(allowed_raw))]


@profiled("train.data_loader.pivot_snapshot_to_wide")
def pivot_snapshot_to_wide(
    snapshot_df: pd.DataFrame,
    target_month: pd.Timestamp,
    cutoff_date: Optional[pd.Timestamp] = None,
    selected_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert snapshot to wide format, taking the latest value for target month.

    Supports both formats:
    - Wide-format (from compute_features_wide): DatetimeIndex + feature columns.
      Skips the expensive pivot and just takes the last valid row per column.
    - Long-format (legacy): columns [date, series_name, value, ...].
      Pivots first, then takes the last valid row per column.

    Args:
        snapshot_df: Snapshot DataFrame (wide or long format)
        target_month: Month we're predicting (format: YYYY-MM-01)
        cutoff_date: Strict cutoff for data availability (e.g., NFP release date)
        selected_features: List of sanitized feature names to keep. If provided,
            filters data early for performance.

    Returns:
        Single-row DataFrame with features as columns
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    # Detect format: wide-format has no 'series_name' column
    is_wide = 'series_name' not in snapshot_df.columns

    if is_wide:
        # Wide-format: index=date (or 'date' column), columns=feature names
        wide_df = snapshot_df.copy()
        
        # Ensure 'date' is available as column or index for filtering
        if 'date' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['date'])
            wide_df = wide_df.set_index('date')
        
        if not isinstance(wide_df.index, pd.DatetimeIndex):
             # Try to convert index if not datetime
             try:
                 wide_df.index = pd.to_datetime(wide_df.index)
             except:
                 pass
        
        wide_df = wide_df.sort_index()

        # Apply cutoff
        cutoff = pd.to_datetime(cutoff_date) if cutoff_date is not None else pd.to_datetime(target_month)
        wide_df = wide_df[wide_df.index < cutoff]

        if wide_df.empty:
            return pd.DataFrame()

        # EARLY FILTERING (Wide): Filter columns
        if selected_features is not None:
            # Match columns to selected features
            # Columns in wide format are series names (potentially raw)
            # We need to sanitize them to check against selected_features
            
            # Optimization: 
            # 1. Sanitize all columns (once) -> Map sanitized to raw
            # 2. Keep only raw columns where sanitized version is in selected_features
            
            selected_set = set(selected_features)
            cols_to_keep = []
            
            for col in wide_df.columns:
                sanitized = sanitize_feature_name(str(col))
                if sanitized in selected_set:
                    cols_to_keep.append(col)
            
            if not cols_to_keep:
                return pd.DataFrame()
                
            wide_df = wide_df[cols_to_keep]

        # Take last valid value per column (vectorized)
        if wide_df.empty:
             return pd.DataFrame()

        # NOAA release lag can leave trailing NaNs; capture staleness explicitly.
        noaa_cols = [col for col in wide_df.columns if _is_noaa_feature_name(col)]
        noaa_staleness = {}
        if noaa_cols:
            noaa_notna = wide_df[noaa_cols].notna()
            has_obs = noaa_notna.any(axis=0)
            if has_obs.any():
                last_obs = noaa_notna.iloc[::-1].idxmax()
                last_obs = pd.to_datetime(last_obs.where(has_obs, pd.NaT), errors='coerce')
                staleness_months = (
                    (cutoff.year - last_obs.dt.year) * 12
                    + (cutoff.month - last_obs.dt.month)
                )
                noaa_staleness = {
                    f"{sanitize_feature_name(str(col))}{NOAA_STALENESS_SUFFIX}": float(months)
                    for col, months in staleness_months.items()
                    if pd.notna(months)
                }

        if noaa_cols:
            other_cols = [col for col in wide_df.columns if col not in noaa_cols]
            filled_parts = []
            if other_cols:
                filled_parts.append(wide_df[other_cols].ffill())
            filled_parts.append(wide_df[noaa_cols].ffill(limit=NOAA_MAX_FFILL_MONTHS))
            wide_filled = pd.concat(filled_parts, axis=1).reindex(columns=wide_df.columns)
        else:
            wide_filled = wide_df.ffill()

        last_valid = wide_filled.iloc[-1].dropna()

        # Sanitize feature names for LightGBM compatibility
        features = {sanitize_feature_name(str(k)): v for k, v in last_valid.items()}
        if noaa_staleness:
            features.update(noaa_staleness)
        if not features:
            return pd.DataFrame()
        return pd.DataFrame([features])

    # Valid "Long" format has 'series_name' and 'value'
    # Optimize: Filter -> GroupBy -> Last (Avoids creating sparse wide DF)
    df = snapshot_df.copy()
    
    # Ensure 'date' is a column
    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff = pd.to_datetime(cutoff_date) if cutoff_date is not None else pd.to_datetime(target_month)
        df = df[df['date'] < cutoff]
    else:
        # If no date column and no DatetimeIndex, we can't filter by time
        # This shouldn't happen for valid snapshots
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
        
    # EARLY FILTERING: Filter by selected features if provided
    if selected_features is not None:
        try:
            # Get unique raw series names present in this filtered slice
            unique_raw = df['series_name'].unique()
        except KeyError as e:
            logger.error(f"KeyError accessing 'series_name'. Columns: {df.columns.tolist()}")
            logger.error(f"Original is_wide detection: {is_wide}")
            # Fallback: maybe it's in the index now?
            if 'series_name' in df.index.names:
                df = df.reset_index()
                unique_raw = df['series_name'].unique()
            else:
                raise e
        
        # Build mapping: raw -> sanitized
        # Only keep those that are in selected_features
        allowed_raw = []
        selected_set = set(selected_features)
        
        for raw in unique_raw:
            sanitized = sanitize_feature_name(str(raw))
            if sanitized in selected_set:
                allowed_raw.append(raw)
        
        if not allowed_raw:
            return pd.DataFrame()
            
        # Filter DataFrame to only allowed raw series
        df = df[df['series_name'].isin(allowed_raw)]
        
        if df.empty:
            return pd.DataFrame()

    # Sort by date so "last" is truly the latest
    df = df.sort_values('date')
    
    # Take last value for each series
    latest_values = df.groupby('series_name')['value'].last()
    
    if latest_values.empty:
        return pd.DataFrame()
        
    # Convert to dict -> DataFrame (1 row)
    features = {sanitize_feature_name(str(k)): v for k, v in latest_values.to_dict().items()}
    
    # Final check: if selected_features provided, ensure we only return those
    # (The pre-filtering above handled the bulk, this is just cleanup)
    if selected_features is not None:
         # Dict comprehension above already sanitized keys.
         # Just filter the dict.
         features = {k: v for k, v in features.items() if k in selected_set}
         
    return pd.DataFrame([features])


def pivot_snapshot_to_wide_batch(
    snapshot_df: pd.DataFrame,
    target_months: List[pd.Timestamp],
    cutoff_dates: Optional[Dict[pd.Timestamp, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Batch version of pivot_snapshot_to_wide for multiple target months.

    Supports both wide-format (from compute_features_wide) and long-format
    (legacy) snapshots. Instead of loading and pivoting the same snapshot
    multiple times, this processes all target months in a single pass.

    Args:
        snapshot_df: Snapshot DataFrame (wide or long format)
        target_months: List of target months to generate features for
        cutoff_dates: Optional mapping of target_month -> cutoff_date (e.g., NFP release date)

    Returns:
        DataFrame with one row per target month, features as columns
    """
    if snapshot_df is None or snapshot_df.empty or not target_months:
        return pd.DataFrame()

    # Detect format and get wide_df
    is_wide = 'series_name' not in snapshot_df.columns

    if is_wide:
        wide_df = snapshot_df.copy()
        if 'date' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['date'])
            wide_df = wide_df.set_index('date')
        wide_df.index = pd.to_datetime(wide_df.index)
        wide_df = wide_df.sort_index()
    else:
        df = snapshot_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        wide_df = df.pivot_table(
            index='date',
            columns='series_name',
            values='value',
            aggfunc='first'
        ).sort_index()

    if wide_df.empty:
        return pd.DataFrame()

    noaa_cols = [col for col in wide_df.columns if _is_noaa_feature_name(col)]
    other_cols = [col for col in wide_df.columns if col not in noaa_cols]

    all_features = []

    for target_month in target_months:
        if cutoff_dates is not None and target_month in cutoff_dates:
            cutoff = pd.to_datetime(cutoff_dates[target_month])
        else:
            cutoff = pd.to_datetime(target_month)
        available = wide_df[wide_df.index < cutoff]

        if available.empty:
            all_features.append({'target_month': target_month})
            continue

        features = {'target_month': target_month}
        noaa_staleness = {}
        if noaa_cols:
            noaa_notna = available[noaa_cols].notna()
            has_obs = noaa_notna.any(axis=0)
            if has_obs.any():
                last_obs = noaa_notna.iloc[::-1].idxmax()
                last_obs = pd.to_datetime(last_obs.where(has_obs, pd.NaT), errors='coerce')
                staleness_months = (
                    (cutoff.year - last_obs.dt.year) * 12
                    + (cutoff.month - last_obs.dt.month)
                )
                noaa_staleness = {
                    f"{sanitize_feature_name(str(col))}{NOAA_STALENESS_SUFFIX}": float(months)
                    for col, months in staleness_months.items()
                    if pd.notna(months)
                }

        if noaa_cols:
            filled_parts = []
            if other_cols:
                filled_parts.append(available[other_cols].ffill())
            filled_parts.append(available[noaa_cols].ffill(limit=NOAA_MAX_FFILL_MONTHS))
            available_filled = pd.concat(filled_parts, axis=1).reindex(columns=available.columns)
        else:
            available_filled = available.ffill()

        last_valid = available_filled.iloc[-1].dropna()
        for raw_series_name, value in last_valid.items():
            series_name = sanitize_feature_name(str(raw_series_name))
            features[series_name] = value

        if noaa_staleness:
            features.update(noaa_staleness)

        all_features.append(features)

    return pd.DataFrame(all_features)
