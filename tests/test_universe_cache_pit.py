"""
PIT (Point-In-Time) safety tests for ``Train/universe_cache.py``.

The hierarchical feature-reselection scheme persists Pass-1 survivors to
disk so later backtest steps reuse them. The scheme is only safe if the
loader never returns a cache built using data from after the requested
``step_date``. These tests verify that invariant directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Train import universe_cache as uc


def _write_cache(
    cache_dir: Path,
    target_type: str,
    target_source: str,
    asof: pd.Timestamp,
    survivors,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / uc.universe_cache_filename(target_type, target_source, asof)
    payload = {
        "asof": pd.Timestamp(asof).strftime("%Y-%m-%d"),
        "target_type": target_type,
        "target_source": target_source,
        "stages": [0, 1, 2, 4, 5, 6],
        "survivors": survivors,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_load_returns_none_when_only_future_cache_exists(tmp_path: Path) -> None:
    """A cache built at 2022-06 must not be visible at step_date 2022-05."""
    _write_cache(
        tmp_path, "nsa", "revised",
        pd.Timestamp("2022-06-01"),
        {"FRED_Employment_NSA": ["payems"]},
    )

    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2022-05-15"),
        cache_dir=tmp_path,
    )
    assert result is None, (
        "PIT violation: cache asof=2022-06 should be invisible at step_date=2022-05"
    )


def test_load_returns_latest_eligible_cache(tmp_path: Path) -> None:
    """Given 2020-06 and 2022-06 caches, step_date 2023-01 must pick 2022-06."""
    _write_cache(
        tmp_path, "nsa", "revised",
        pd.Timestamp("2020-06-01"),
        {"FRED_Employment_NSA": ["payems_old"]},
    )
    _write_cache(
        tmp_path, "nsa", "revised",
        pd.Timestamp("2022-06-01"),
        {"FRED_Employment_NSA": ["payems_recent"]},
    )

    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2023-01-15"),
        cache_dir=tmp_path,
    )
    assert result is not None
    assert result["asof"] == pd.Timestamp("2022-06-01"), (
        f"Expected latest eligible cache (2022-06), got {result['asof']}"
    )
    assert result["survivors"] == {"FRED_Employment_NSA": ["payems_recent"]}


def test_load_returns_none_for_empty_directory(tmp_path: Path) -> None:
    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2023-01-15"),
        cache_dir=tmp_path,
    )
    assert result is None


def test_load_ignores_other_branch_caches(tmp_path: Path) -> None:
    """An SA cache must not be returned when nsa is requested."""
    _write_cache(
        tmp_path, "sa", "revised",
        pd.Timestamp("2022-06-01"),
        {"FRED_Employment_SA": ["payems_sa"]},
    )
    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2023-01-15"),
        cache_dir=tmp_path,
    )
    assert result is None, "NSA load should ignore SA caches"


def test_save_round_trip(tmp_path: Path) -> None:
    survivors = {
        "FRED_Employment_NSA": ["a", "b", "c"],
        "Unifier": ["d"],
    }
    asof = pd.Timestamp("2024-01-01")
    written = uc.save_universe(
        survivors=survivors,
        target_type="nsa", target_source="revised",
        asof=asof, stages=(0, 1, 2, 4, 5, 6),
        cache_dir=tmp_path,
    )
    assert written.exists()
    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2024-06-01"),
        cache_dir=tmp_path,
    )
    assert result is not None
    assert result["asof"] == asof
    assert result["survivors"] == survivors
    assert result["stages"] == [0, 1, 2, 4, 5, 6]


def test_is_cache_fresh_window() -> None:
    asof = pd.Timestamp("2022-06-01")
    # within the 24-month window (inclusive of asof month → exclusive of asof+24)
    assert uc.is_cache_fresh(asof, pd.Timestamp("2022-06-01"), 24) is True
    assert uc.is_cache_fresh(asof, pd.Timestamp("2023-01-01"), 24) is True
    assert uc.is_cache_fresh(asof, pd.Timestamp("2024-05-01"), 24) is True
    # outside the window
    assert uc.is_cache_fresh(asof, pd.Timestamp("2024-06-01"), 24) is False
    assert uc.is_cache_fresh(asof, pd.Timestamp("2024-12-01"), 24) is False
    # step_date before asof — never fresh (would imply PIT violation)
    assert uc.is_cache_fresh(asof, pd.Timestamp("2022-05-01"), 24) is False


def test_load_skips_unparseable_filenames(tmp_path: Path) -> None:
    """Random JSON files in the cache dir must not break the loader."""
    (tmp_path / "junk.json").write_text("{}")
    (tmp_path / "universe_nope.json").write_text("{}")
    _write_cache(
        tmp_path, "nsa", "revised",
        pd.Timestamp("2022-06-01"),
        {"FRED_Employment_NSA": ["payems"]},
    )
    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2023-01-01"),
        cache_dir=tmp_path,
    )
    assert result is not None
    assert result["asof"] == pd.Timestamp("2022-06-01")


def test_month_boundary_asof_equal_to_step_date_is_eligible(tmp_path: Path) -> None:
    """asof == step_date (same month) is still PIT-safe."""
    _write_cache(
        tmp_path, "nsa", "revised",
        pd.Timestamp("2022-06-01"),
        {"FRED_Employment_NSA": ["payems"]},
    )
    result = uc.load_latest_universe(
        target_type="nsa", target_source="revised",
        step_date=pd.Timestamp("2022-06-15"),
        cache_dir=tmp_path,
    )
    assert result is not None
    assert result["asof"] == pd.Timestamp("2022-06-01")
