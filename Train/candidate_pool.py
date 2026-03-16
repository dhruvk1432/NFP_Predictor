"""
Union-first candidate pool builder.

Reads per-source and per-regime cached feature selection JSONs produced by
Data_ETA_Pipeline/create_master_snapshots.py, unions survivors across all
sources and regimes, applies source-balanced ranking, and outputs a bounded
candidate pool (<= UNION_POOL_MAX features).

Cache invalidation uses SHA-256 of upstream file contents so the pool
automatically rebuilds whenever any ETL cache changes.
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger
from Train.config import MASTER_SNAPSHOTS_BASE, UNION_POOL_MAX

logger = setup_logger(__file__, TEMP_DIR)

# Regex for source cache filenames produced by create_master_snapshots.py
# Pattern: source_{SOURCE_NAME}_{CAT}_{SRC_TYPE}[_asof_{YYYY-MM}].json
_SOURCE_CACHE_RE = re.compile(
    r'^source_(.+?)_(nsa|sa)_(revised)(?:_asof_\d{4}-\d{2})?\.json$'
)


def _extract_source_name(filepath: Path) -> str:
    """Extract source name from a source_cache filename.

    Returns ``'UNKNOWN'`` when the filename does not match the expected pattern.
    """
    m = _SOURCE_CACHE_RE.match(filepath.name)
    if m:
        return m.group(1)
    return 'UNKNOWN'


def _scan_cache_files(target_type: str, target_source: str) -> List[Path]:
    """Find all source-cache and regime-cache JSON files for a target config."""
    files: List[Path] = []

    source_dir = MASTER_SNAPSHOTS_BASE / "source_caches"
    if source_dir.is_dir():
        for p in source_dir.glob("*.json"):
            # Match files for this target_type and target_source
            if f"_{target_type}_{target_source}" in p.name:
                files.append(p)

    regime_dir = MASTER_SNAPSHOTS_BASE / "regime_caches"
    if regime_dir.is_dir():
        for p in regime_dir.glob("*.json"):
            if p.name.startswith(f"selected_features_{target_type}_{target_source}_"):
                files.append(p)

    # Legacy branch-level cache (fallback)
    legacy = MASTER_SNAPSHOTS_BASE / f"selected_features_{target_type}_{target_source}.json"
    if legacy.is_file() and legacy not in files:
        files.append(legacy)

    return sorted(files)


def _compute_cache_key(
    target_type: str,
    target_source: str,
    max_candidates: int,
    cache_files: List[Path],
) -> str:
    """Deterministic cache key from config params + upstream file contents."""
    h = hashlib.sha256()
    h.update(f"{target_type}|{target_source}|{max_candidates}".encode())
    for path in sorted(cache_files):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _read_features_from_json(path: Path) -> List[str]:
    """Read the ``features`` key from a cache JSON, handling both dict and list schemas."""
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        # Legacy schema: plain list of feature names
        return data
    if isinstance(data, dict):
        return data.get('features', [])
    return []


def load_all_cached_features(
    target_type: str,
    target_source: str = 'revised',
) -> Dict[str, List[str]]:
    """Load features from all cache JSONs for a target config.

    Returns:
        Dict mapping ``source_label`` to list of features. Source labels are
        extracted from filenames for source caches (e.g. ``FRED_Employment_NSA``)
        and labelled ``REGIME_{YYYY-MM}`` for regime caches.
    """
    result: Dict[str, List[str]] = {}

    source_dir = MASTER_SNAPSHOTS_BASE / "source_caches"
    if source_dir.is_dir():
        for p in sorted(source_dir.glob("*.json")):
            if f"_{target_type}_{target_source}" not in p.name:
                continue
            label = _extract_source_name(p)
            features = _read_features_from_json(p)
            if features:
                key = f"{label}|{p.stem}"
                result[key] = features

    regime_dir = MASTER_SNAPSHOTS_BASE / "regime_caches"
    if regime_dir.is_dir():
        for p in sorted(regime_dir.glob("*.json")):
            if not p.name.startswith(f"selected_features_{target_type}_{target_source}_"):
                continue
            features = _read_features_from_json(p)
            if features:
                # Extract regime month from filename tail
                regime_key = p.stem.split('_')[-1]  # e.g. "2025-01"
                result[f"REGIME_{regime_key}"] = features

    # Legacy branch cache
    legacy = MASTER_SNAPSHOTS_BASE / f"selected_features_{target_type}_{target_source}.json"
    if legacy.is_file():
        features = _read_features_from_json(legacy)
        if features:
            result["LEGACY_BRANCH"] = features

    return result


def build_union_pool(
    target_type: str,
    target_source: str = 'revised',
    max_candidates: int = UNION_POOL_MAX,
) -> List[str]:
    """Build the global union candidate pool with source-balanced ranking.

    Steps:
        1. Load all source + regime caches.
        2. Union all feature names.
        3. Group features by source and rank within each source by frequency
           (features appearing in more cache files rank higher = more stable).
        4. Round-robin across sources to fill the pool up to ``max_candidates``,
           prioritising sources with the fewest features already selected.

    Returns:
        Sorted list of up to ``max_candidates`` feature names.
    """
    all_caches = load_all_cached_features(target_type, target_source)

    if not all_caches:
        logger.warning("No feature selection caches found — returning empty pool")
        return []

    # --- Group features by source ---
    # For source caches, the label is "SOURCE_NAME|stem"; extract the source part.
    # For regime caches, label is "REGIME_*"; these are cross-source.
    source_features: Dict[str, Counter] = defaultdict(Counter)

    for label, features in all_caches.items():
        if label.startswith("REGIME_") or label == "LEGACY_BRANCH":
            source = "CROSS_SOURCE"
        else:
            source = label.split('|')[0]
        for feat in features:
            source_features[source][feat] += 1

    # --- Track UNKNOWN share ---
    unknown_count = sum(source_features.get('UNKNOWN', Counter()).values())
    total_count = sum(sum(c.values()) for c in source_features.values())
    if total_count > 0 and unknown_count / total_count > 0.10:
        logger.warning(
            f"UNKNOWN source share is {unknown_count / total_count:.0%} "
            f"({unknown_count}/{total_count} feature-occurrences)"
        )

    # --- Build per-source ranked lists ---
    # Within each source, sort by frequency descending (stability proxy)
    source_ranked: Dict[str, List[str]] = {}
    for source, counter in source_features.items():
        source_ranked[source] = [
            feat for feat, _ in counter.most_common()
        ]

    # --- Round-robin across sources ---
    pool: List[str] = []
    pool_set: set = set()
    # Track how many features each source has contributed
    source_contributed: Counter = Counter()
    # Create iterators for each source
    source_iters = {s: iter(ranked) for s, ranked in source_ranked.items()}
    sources = sorted(source_ranked.keys())

    while len(pool) < max_candidates and source_iters:
        # Pick the source with the fewest contributions so far
        sources_alive = [s for s in sources if s in source_iters]
        if not sources_alive:
            break
        sources_alive.sort(key=lambda s: source_contributed[s])

        exhausted = []
        for source in sources_alive:
            if len(pool) >= max_candidates:
                break
            it = source_iters[source]
            # Pull features from this source until we add one new feature
            added = False
            while not added:
                feat = next(it, None)
                if feat is None:
                    exhausted.append(source)
                    break
                if feat not in pool_set:
                    pool.append(feat)
                    pool_set.add(feat)
                    source_contributed[source] += 1
                    added = True

        for s in exhausted:
            del source_iters[s]

    # --- Log summary ---
    source_counts = dict(source_contributed)
    logger.info(
        f"Union pool built: {len(pool)} features from {len(all_caches)} caches. "
        f"Source breakdown: {source_counts}"
    )

    return pool


def _get_pool_cache_path(target_type: str, target_source: str) -> Path:
    """Path for the cached union pool JSON."""
    return MASTER_SNAPSHOTS_BASE / f"candidate_pool_{target_type}_{target_source}.json"


def load_or_build_union_pool(
    target_type: str,
    target_source: str = 'revised',
    max_candidates: int = UNION_POOL_MAX,
) -> List[str]:
    """Load a cached candidate pool if valid, otherwise rebuild.

    Cache validity is checked via a SHA-256 key computed from
    ``target_type``, ``target_source``, ``max_candidates``, and the
    contents of all upstream cache JSON files.
    """
    upstream_files = _scan_cache_files(target_type, target_source)
    current_key = _compute_cache_key(
        target_type, target_source, max_candidates, upstream_files
    )

    pool_path = _get_pool_cache_path(target_type, target_source)

    # Try loading cached pool
    if pool_path.is_file():
        try:
            with open(pool_path, 'r') as f:
                cached = json.load(f)
            if cached.get('cache_key') == current_key:
                features = cached.get('features', [])
                logger.info(
                    f"Loaded cached union pool ({len(features)} features, "
                    f"key={current_key[:8]})"
                )
                return features
            else:
                logger.info("Union pool cache key mismatch — rebuilding")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt union pool cache — rebuilding")

    # Build fresh
    pool = build_union_pool(target_type, target_source, max_candidates)

    # Compute source_counts for the output
    all_caches = load_all_cached_features(target_type, target_source)
    source_counts: Counter = Counter()
    for label in all_caches:
        if label.startswith("REGIME_") or label == "LEGACY_BRANCH":
            source_counts["CROSS_SOURCE"] += 1
        else:
            source_counts[label.split('|')[0]] += 1

    # Save
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'features': pool,
        'cache_key': current_key,
        'n_caches_scanned': len(upstream_files),
        'source_counts': dict(source_counts),
        'last_run_date': str(date.today()),
    }
    with open(pool_path, 'w') as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved union pool to {pool_path} ({len(pool)} features)")
    return pool
