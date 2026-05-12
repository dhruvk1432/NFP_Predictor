"""
Tier-A universe-distillation cache.

What this stores
----------------
For each (target_type, target_source, universe_asof) tuple, the per-source
Pass-1 survivor lists produced by the feature-selection engine. Pass-1 is
the wall-clock-dominant work in dynamic reselection (Boruta on the full
~17k-column FRED universe), so persisting its output amortises the cost
across every Pass-2-only refresh that happens inside the universe window.

PIT invariant
-------------
A cache file with ``universe_asof = T`` is built using only training data
with ``ds < T``. It may therefore be consumed at any backtest step_date
``t`` where ``t >= T`` — never before. ``load_latest_universe`` enforces
this by filtering candidate cache files on filename (``asof`` is embedded
in the filename, not just inside the JSON) before reading.

This module is read/write only — it does not run feature selection itself.
The caller (``_dynamic_reselection`` in train_lightgbm_nfp.py) decides
when to recompute and call ``save_universe``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.config import UNIVERSE_CACHE_DIR

logger = setup_logger(__file__, TEMP_DIR)


_FILENAME_RE = re.compile(
    r"^universe_(?P<tt>nsa|sa)_(?P<src>revised)_(?P<asof>\d{4}-\d{2})\.json$"
)


def universe_cache_filename(
    target_type: str, target_source: str, asof: pd.Timestamp,
) -> str:
    """Deterministic filename embedding the PIT-relevant ``asof`` month."""
    return f"universe_{target_type}_{target_source}_{asof.strftime('%Y-%m')}.json"


def universe_cache_path(
    target_type: str, target_source: str, asof: pd.Timestamp,
) -> Path:
    return UNIVERSE_CACHE_DIR / universe_cache_filename(target_type, target_source, asof)


def _list_eligible_caches(
    cache_dir: Path,
    target_type: str,
    target_source: str,
    step_date: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, Path]]:
    """Return ``(asof, path)`` tuples for caches with ``asof <= step_date``."""
    if not cache_dir.exists():
        return []
    out: List[Tuple[pd.Timestamp, Path]] = []
    step_month = pd.Timestamp(step_date).normalize().replace(day=1)
    for p in cache_dir.glob(f"universe_{target_type}_{target_source}_*.json"):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        if m.group("tt") != target_type or m.group("src") != target_source:
            continue
        asof = pd.Timestamp(m.group("asof"))
        # PIT filter: month-aligned cache files with asof month-start strictly
        # before-or-equal to the step_date's month are eligible. We compare
        # at month granularity because backtest step_dates are month-anchored.
        if asof <= step_month:
            out.append((asof, p))
    out.sort(key=lambda x: x[0])
    return out


def load_latest_universe(
    target_type: str,
    target_source: str,
    step_date: pd.Timestamp,
    cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """Return the most-recent PIT-eligible universe cache, or ``None``.

    Returns a dict with keys ``asof`` (pd.Timestamp), ``survivors`` (Dict[str,
    List[str]]), and ``stages`` (list[int]). Returns ``None`` if no eligible
    cache exists.

    Hard-asserts the PIT invariant on the loaded asof so a bug in caller
    logic cannot silently leak future information.
    """
    cdir = cache_dir if cache_dir is not None else UNIVERSE_CACHE_DIR
    eligible = _list_eligible_caches(cdir, target_type, target_source, step_date)
    if not eligible:
        return None
    asof, path = eligible[-1]

    # Belt-and-braces PIT check: the asof from the filename must be ≤ step_date.
    if asof > pd.Timestamp(step_date).normalize().replace(day=1):
        raise RuntimeError(
            f"universe_cache PIT VIOLATION: cache {path.name} has "
            f"asof={asof} > step_date={step_date}"
        )

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"universe_cache: failed to read {path}: {e}")
        return None

    survivors = data.get("survivors", {})
    if not isinstance(survivors, dict) or not survivors:
        return None

    cleaned = {str(k): list(v) for k, v in survivors.items() if isinstance(v, list)}
    return {
        "asof": asof,
        "survivors": cleaned,
        "stages": data.get("stages", []),
        "path": str(path),
    }


def save_universe(
    survivors: Dict[str, List[str]],
    target_type: str,
    target_source: str,
    asof: pd.Timestamp,
    stages: Tuple[int, ...] = (),
    cache_dir: Optional[Path] = None,
) -> Path:
    """Persist Pass-1 survivors. Returns the written path."""
    cdir = cache_dir if cache_dir is not None else UNIVERSE_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / universe_cache_filename(target_type, target_source, pd.Timestamp(asof))
    payload = {
        "asof": pd.Timestamp(asof).strftime("%Y-%m-%d"),
        "target_type": target_type,
        "target_source": target_source,
        "stages": list(stages),
        "survivors": {str(k): list(v) for k, v in survivors.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info(
        f"universe_cache SAVE [{target_type}/{target_source}] asof={asof.strftime('%Y-%m')} "
        f"→ {path.name} ({sum(len(v) for v in survivors.values())} survivors across "
        f"{len(survivors)} sources)"
    )
    return path


def is_cache_fresh(
    cache_asof: pd.Timestamp,
    step_date: pd.Timestamp,
    refresh_months: int,
) -> bool:
    """Return True iff ``step_date`` is within the ``refresh_months`` window."""
    months = (step_date.year - cache_asof.year) * 12 + (step_date.month - cache_asof.month)
    return 0 <= months < refresh_months
