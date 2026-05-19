"""
Persistent content-hashed cache for ``build_training_dataset`` output.

``build_training_dataset`` is a deterministic function of:
  * the master snapshot parquets on disk
  * the branch target parquet
  * the (target_type, release_type, target_source) tuple
  * the date range covered by the passed ``target_df``

When none of those have changed between runs, the resulting
``(X_full, y_full)`` is byte-identical. This module hashes those inputs
to a short key and stores the result under
``_output/cache/training_dataset/`` so subsequent runs skip the
parallel-feature-engineering build entirely.

The cache key includes ``SCHEMA_VERSION`` because feature-builder semantics
can change without any upstream parquet mtime changing. Bump the version
whenever generated feature values or PIT masking behavior changes.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import (
    get_master_snapshots_dir,
    get_target_path,
)

logger = setup_logger(__file__, TEMP_DIR)

# v2: branch-target lag features mask revised targets by each snapshot's
# operational cutoff before shift/rolling construction.
SCHEMA_VERSION = 2

CACHE_DIR = OUTPUT_DIR / "cache" / "training_dataset"


def _hash_file_signature(h, path: Path, label: str) -> None:
    """Fold a file's (name, mtime, size) into the running hash. Missing = noop."""
    if not path.exists():
        h.update(f"{label}|MISSING|{path.name}".encode())
        return
    st = path.stat()
    h.update(f"{label}|{path.name}|{st.st_mtime_ns}|{st.st_size}".encode())


def compute_cache_key(
    target_df: pd.DataFrame,
    target_type: str,
    release_type: str,
    target_source: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> str:
    """Deterministic 16-hex-char key over all inputs that drive the output."""
    h = hashlib.sha256()
    h.update(
        (
            f"SCHEMA_VERSION={SCHEMA_VERSION}|"
            f"target_type={target_type}|"
            f"release_type={release_type}|"
            f"target_source={target_source}|"
            f"start_date={start_date}|"
            f"end_date={end_date}"
        ).encode()
    )

    # Fold the effective target-month range carried by target_df. We don't
    # hash the full content because target_df itself is derived from the
    # branch target parquet (whose signature we hash below), so the range +
    # length is a sufficient distinguisher.
    if 'ds' in target_df.columns and len(target_df) > 0:
        ds = pd.to_datetime(target_df['ds'])
        h.update(
            (
                f"|target_df_len={len(target_df)}"
                f"|first_ds={ds.min().strftime('%Y-%m-%d')}"
                f"|last_ds={ds.max().strftime('%Y-%m-%d')}"
            ).encode()
        )
    else:
        h.update(b"|target_df_empty")

    # Master snapshot files — every parquet that ``load_master_snapshot``
    # could potentially read for this branch.
    snap_dir = get_master_snapshots_dir(target_type, target_source)
    if snap_dir.exists():
        for p in sorted(snap_dir.rglob("*.parquet")):
            _hash_file_signature(h, p, "master")
    else:
        h.update(b"|master_dir_missing")

    # Branch target parquet — drives lag features and target alignment.
    _hash_file_signature(h, get_target_path(target_type, release_type), "target")

    return h.hexdigest()[:16]


def _normalize_cache_read_root(raw: str) -> Optional[Path]:
    """Resolve an env-provided cache root to its training_dataset directory."""
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.name == "training_dataset" and path.parent.name == "cache":
        return path
    return path / "cache" / "training_dataset"


def _candidate_cache_dirs() -> Iterable[Path]:
    """Primary output cache plus optional read-through roots from prior output dirs."""
    seen: set[Path] = set()
    primary = CACHE_DIR.resolve()
    seen.add(primary)
    yield CACHE_DIR

    raw = os.getenv("NFP_TRAIN_DATASET_CACHE_READ_ROOTS", "").strip()
    if not raw:
        return
    for token in raw.replace(os.pathsep, ",").split(","):
        extra = _normalize_cache_read_root(token)
        if extra is None:
            continue
        resolved = extra.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield extra


def _paths_for(
    target_type: str,
    release_type: str,
    target_source: str,
    key: str,
    cache_dir: Path = CACHE_DIR,
) -> Tuple[Path, Path]:
    base = cache_dir / f"{target_type}_{release_type}_{target_source}__{key}"
    return base.with_suffix(".X.parquet"), base.with_suffix(".y.parquet")


def load_cached_dataset(
    target_df: pd.DataFrame,
    target_type: str,
    release_type: str,
    target_source: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Return cached ``(X_full, y_full)`` if present, else ``None``."""
    key = compute_cache_key(
        target_df, target_type, release_type, target_source, start_date, end_date,
    )
    x_path = y_path = None
    for cache_dir in _candidate_cache_dirs():
        cand_x, cand_y = _paths_for(target_type, release_type, target_source, key, cache_dir)
        if cand_x.exists() and cand_y.exists():
            x_path, y_path = cand_x, cand_y
            break
    if x_path is None or y_path is None:
        return None

    try:
        X = pd.read_parquet(x_path)
        y_df = pd.read_parquet(y_path)
    except Exception as e:
        logger.warning(f"training_dataset_cache: failed to read cache {x_path}: {e}")
        return None

    if 'y_mom' not in y_df.columns:
        logger.warning(f"training_dataset_cache: cache {y_path} missing y_mom column")
        return None

    y = y_df['y_mom']
    y.name = 'y_mom'
    logger.info(
        f"training_dataset_cache HIT [{target_type}/{release_type}/{target_source}] "
        f"key={key} source={x_path.parent} → {len(X)} rows × {len(X.columns)} cols"
    )
    return X, y


def save_cached_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    target_df: pd.DataFrame,
    target_type: str,
    release_type: str,
    target_source: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> None:
    """Persist ``(X, y)`` under the deterministic cache key. Best-effort."""
    key = compute_cache_key(
        target_df, target_type, release_type, target_source, start_date, end_date,
    )
    x_path, y_path = _paths_for(target_type, release_type, target_source, key)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_path)
        y.to_frame(name='y_mom').to_parquet(y_path)
        logger.info(
            f"training_dataset_cache SAVE [{target_type}/{release_type}/{target_source}] "
            f"key={key} → {x_path.name}"
        )
    except Exception as e:
        logger.warning(f"training_dataset_cache: failed to save cache: {e}")
