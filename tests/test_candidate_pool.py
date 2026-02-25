"""Tests for Train/candidate_pool.py — union-first candidate pool builder."""

import json

import pytest

from Train.candidate_pool import (
    _compute_cache_key,
    _extract_source_name,
    _read_features_from_json,
    build_union_pool,
    load_all_cached_features,
    load_or_build_union_pool,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# _extract_source_name
# ---------------------------------------------------------------------------


def test_extract_source_name_fred_employment_nsa():
    p = Path("source_FRED_Employment_NSA_nsa_first_release.json")
    assert _extract_source_name(p) == "FRED_Employment_NSA"


def test_extract_source_name_adp():
    p = Path("source_ADP_nsa_first_release_asof_2025-01.json")
    assert _extract_source_name(p) == "ADP"


def test_extract_source_name_noaa():
    p = Path("source_NOAA_sa_revised.json")
    assert _extract_source_name(p) == "NOAA"


def test_extract_source_name_unknown():
    p = Path("some_random_file.json")
    assert _extract_source_name(p) == "UNKNOWN"


def test_extract_source_name_unifier_asof():
    p = Path("source_Unifier_nsa_first_release_asof_2024-06.json")
    assert _extract_source_name(p) == "Unifier"


# ---------------------------------------------------------------------------
# _read_features_from_json
# ---------------------------------------------------------------------------


def test_read_features_dict_schema(tmp_path):
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"features": ["a", "b"], "last_run_date": "2026-01-01"}))
    assert _read_features_from_json(p) == ["a", "b"]


def test_read_features_list_schema(tmp_path):
    p = tmp_path / "test.json"
    p.write_text(json.dumps(["x", "y", "z"]))
    assert _read_features_from_json(p) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# _compute_cache_key
# ---------------------------------------------------------------------------


def test_cache_key_deterministic(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text('{"features": ["a"]}')
    key1 = _compute_cache_key("nsa", "first_release", 200, [f1])
    key2 = _compute_cache_key("nsa", "first_release", 200, [f1])
    assert key1 == key2


def test_cache_key_changes_with_max_candidates(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text('{"features": ["a"]}')
    key_200 = _compute_cache_key("nsa", "first_release", 200, [f1])
    key_100 = _compute_cache_key("nsa", "first_release", 100, [f1])
    assert key_200 != key_100


def test_cache_key_changes_with_content(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text('{"features": ["a"]}')
    key1 = _compute_cache_key("nsa", "first_release", 200, [f1])
    f1.write_text('{"features": ["a", "b"]}')
    key2 = _compute_cache_key("nsa", "first_release", 200, [f1])
    assert key1 != key2


# ---------------------------------------------------------------------------
# build_union_pool (using monkeypatched cache dirs)
# ---------------------------------------------------------------------------


def test_union_deduplicates(tmp_path, monkeypatch):
    """Features appearing in multiple caches should be deduplicated."""
    source_dir = tmp_path / "source_caches"
    source_dir.mkdir()
    (source_dir / "source_ADP_nsa_first_release.json").write_text(
        json.dumps({"features": ["feat_a", "feat_b", "feat_c"]})
    )
    (source_dir / "source_NOAA_nsa_first_release.json").write_text(
        json.dumps({"features": ["feat_b", "feat_c", "feat_d"]})
    )
    regime_dir = tmp_path / "regime_caches"
    regime_dir.mkdir()

    monkeypatch.setattr("Train.candidate_pool.MASTER_SNAPSHOTS_BASE", tmp_path)

    pool = build_union_pool("nsa", "first_release", max_candidates=200)
    assert len(pool) == len(set(pool))
    assert set(pool) == {"feat_a", "feat_b", "feat_c", "feat_d"}


def test_respects_max_candidates(tmp_path, monkeypatch):
    """Pool should be capped at max_candidates."""
    source_dir = tmp_path / "source_caches"
    source_dir.mkdir()
    features = [f"feat_{i}" for i in range(300)]
    (source_dir / "source_ADP_nsa_first_release.json").write_text(
        json.dumps({"features": features})
    )
    regime_dir = tmp_path / "regime_caches"
    regime_dir.mkdir()

    monkeypatch.setattr("Train.candidate_pool.MASTER_SNAPSHOTS_BASE", tmp_path)

    pool = build_union_pool("nsa", "first_release", max_candidates=200)
    assert len(pool) <= 200


def test_source_balanced_ranking(tmp_path, monkeypatch):
    """Two sources with unequal sizes should both be represented."""
    source_dir = tmp_path / "source_caches"
    source_dir.mkdir()
    # Source A has 100 features, source B has 10
    (source_dir / "source_ADP_nsa_first_release.json").write_text(
        json.dumps({"features": [f"adp_{i}" for i in range(100)]})
    )
    (source_dir / "source_NOAA_nsa_first_release.json").write_text(
        json.dumps({"features": [f"noaa_{i}" for i in range(10)]})
    )
    regime_dir = tmp_path / "regime_caches"
    regime_dir.mkdir()

    monkeypatch.setattr("Train.candidate_pool.MASTER_SNAPSHOTS_BASE", tmp_path)

    pool = build_union_pool("nsa", "first_release", max_candidates=20)
    adp_count = sum(1 for f in pool if f.startswith("adp_"))
    noaa_count = sum(1 for f in pool if f.startswith("noaa_"))
    # Both sources should have features in the pool
    assert adp_count >= 5
    assert noaa_count >= 5


def test_cache_invalidation_on_content_change(tmp_path, monkeypatch):
    """Modifying upstream JSON content should cause pool to auto-rebuild."""
    source_dir = tmp_path / "source_caches"
    source_dir.mkdir()
    regime_dir = tmp_path / "regime_caches"
    regime_dir.mkdir()

    src = source_dir / "source_ADP_nsa_first_release.json"
    src.write_text(json.dumps({"features": ["feat_a", "feat_b"]}))

    monkeypatch.setattr("Train.candidate_pool.MASTER_SNAPSHOTS_BASE", tmp_path)

    pool1 = load_or_build_union_pool("nsa", "first_release", max_candidates=200)
    assert "feat_a" in pool1

    # Modify upstream file
    src.write_text(json.dumps({"features": ["feat_x", "feat_y"]}))

    pool2 = load_or_build_union_pool("nsa", "first_release", max_candidates=200)
    assert "feat_x" in pool2
    assert "feat_a" not in pool2
