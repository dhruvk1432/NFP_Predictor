"""
Pytest suite for create_master_snapshots.py:
  - Per-source caching (independent crash recovery)
  - Batch source loading (reduced file reads)
  - Progress tracking with checkpoint/resume
"""
import pytest
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.create_master_snapshots import (
    _check_source_cache,
    _save_source_cache,
    _get_source_cache_path,
    _normalize_to_wide,
    _batch_load_source,
    _load_all_sources_from_cache,
    _snapshot_path,
    _load_progress,
    _save_progress,
    _clear_progress,
    _get_progress_path,
    MASTER_BASE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def source_cache_dir(tmp_path, monkeypatch):
    """Redirect MASTER_BASE to tmp_path so cache files don't pollute real data."""
    monkeypatch.setattr(
        'Data_ETA_Pipeline.create_master_snapshots.MASTER_BASE', tmp_path
    )
    return tmp_path


@pytest.fixture
def sample_wide_parquet(tmp_path):
    """Create a sample wide-format parquet file and return its path."""
    dates = pd.date_range('2010-01-01', periods=60, freq='MS')
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'date': dates,
        'feat_a': rng.randn(60),
        'feat_b': rng.randn(60),
        'feat_c': rng.randn(60),
        'snapshot_date': pd.Timestamp('2024-12-31'),
    })
    path = tmp_path / 'test.parquet'
    df.to_parquet(path, index=False)
    return path, df


@pytest.fixture
def sample_long_parquet(tmp_path):
    """Create a sample long-format parquet file."""
    dates = pd.date_range('2010-01-01', periods=60, freq='MS')
    rng = np.random.RandomState(42)
    rows = []
    for d in dates:
        for series in ['feat_a', 'feat_b', 'feat_c']:
            rows.append({
                'date': d,
                'series_name': series,
                'value': rng.randn(),
                'snapshot_date': pd.Timestamp('2024-12-31'),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / 'test_long.parquet'
    df.to_parquet(path, index=False)
    return path, df


# =============================================================================
# Per-Source Cache
# =============================================================================

class TestPerSourceCache:
    """Tests for per-source feature selection caching."""

    def test_cache_miss_returns_none(self, source_cache_dir):
        """No cache file should return None."""
        result = _check_source_cache('FRED_Exogenous', 'nsa', 'first_release')
        assert result is None

    def test_save_and_load_round_trip(self, source_cache_dir):
        """Saved features should be retrievable."""
        features = ['feat_a', 'feat_b', 'feat_c']
        _save_source_cache(features, 'ADP', 'nsa', 'first_release')
        result = _check_source_cache('ADP', 'nsa', 'first_release')
        assert sorted(result) == sorted(features)

    def test_expired_cache_returns_none(self, source_cache_dir):
        """Cache older than 30 days should return None."""
        features = ['feat_a']
        _save_source_cache(features, 'NOAA', 'sa', 'revised')

        # Manually backdate the cache
        cache_path = _get_source_cache_path('NOAA', 'sa', 'revised')
        with open(cache_path, 'r') as f:
            data = json.load(f)
        data['last_run_date'] = (datetime.now() - timedelta(days=31)).strftime('%Y-%m-%d')
        with open(cache_path, 'w') as f:
            json.dump(data, f)

        result = _check_source_cache('NOAA', 'sa', 'revised')
        assert result is None

    def test_different_sources_independent(self, source_cache_dir):
        """Caches for different sources should not interfere."""
        _save_source_cache(['feat_a'], 'ADP', 'nsa', 'first_release')
        _save_source_cache(['feat_x', 'feat_y'], 'Unifier', 'nsa', 'first_release')

        adp = _check_source_cache('ADP', 'nsa', 'first_release')
        unifier = _check_source_cache('Unifier', 'nsa', 'first_release')

        assert adp == ['feat_a']
        assert sorted(unifier) == ['feat_x', 'feat_y']

    def test_different_branches_independent(self, source_cache_dir):
        """Same source with different target combos should have separate caches."""
        _save_source_cache(['feat_nsa'], 'ADP', 'nsa', 'first_release')
        _save_source_cache(['feat_sa'], 'ADP', 'sa', 'first_release')

        nsa = _check_source_cache('ADP', 'nsa', 'first_release')
        sa = _check_source_cache('ADP', 'sa', 'first_release')

        assert nsa == ['feat_nsa']
        assert sa == ['feat_sa']

    def test_corrupted_cache_returns_none(self, source_cache_dir):
        """Corrupted JSON should return None gracefully."""
        _save_source_cache(['feat_a'], 'ADP', 'nsa', 'first_release')
        cache_path = _get_source_cache_path('ADP', 'nsa', 'first_release')
        with open(cache_path, 'w') as f:
            f.write("NOT VALID JSON{{{")

        result = _check_source_cache('ADP', 'nsa', 'first_release')
        assert result is None

    def test_empty_features_list_cached(self, source_cache_dir):
        """A source that produced 0 features should still cache (avoid re-running)."""
        _save_source_cache([], 'Prosper', 'nsa', 'first_release')
        result = _check_source_cache('Prosper', 'nsa', 'first_release')
        assert result == []


# =============================================================================
# Batch Loading
# =============================================================================

class TestNormalizeToWide:
    """Tests for _normalize_to_wide format conversion."""

    def test_wide_format_passthrough(self, sample_wide_parquet):
        """Already-wide DataFrames should pass through with 'date' as column."""
        path, df = sample_wide_parquet
        result = _normalize_to_wide(df)
        assert 'date' in result.columns

    def test_long_format_pivots(self, sample_long_parquet):
        """Long-format DataFrames should be pivoted to wide."""
        path, df = sample_long_parquet
        result = _normalize_to_wide(df)
        assert 'date' in result.columns
        assert 'feat_a' in result.columns
        assert 'feat_b' in result.columns

    def test_index_based_date(self):
        """DataFrames with date as index should be reset."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'val': range(12)}, index=dates)
        df.index.name = 'date'
        result = _normalize_to_wide(df)
        assert 'date' in result.columns


class TestBatchLoadSource:
    """Tests for _batch_load_source pre-loading."""

    def test_loads_available_months(self, tmp_path):
        """Should load parquets for months that exist."""
        rng = np.random.RandomState(42)
        months = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01'),
                  pd.Timestamp('2020-03-01')]

        # Create parquet files in the expected directory structure
        for m in months[:2]:  # Only create 2 of 3
            path = _snapshot_path(tmp_path, m)
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                'date': pd.date_range('2019-01-01', periods=12, freq='MS'),
                'feat_a': rng.randn(12),
                'feat_b': rng.randn(12),
            })
            df.to_parquet(path, index=False)

        result = _batch_load_source(
            'TestSource', tmp_path, months,
            allowed_features={'feat_a', 'feat_b'}
        )

        assert '2020-01' in result
        assert '2020-02' in result
        assert '2020-03' not in result  # File doesn't exist

    def test_filters_to_allowed_features(self, tmp_path):
        """Only allowed features should be kept."""
        rng = np.random.RandomState(42)
        month = pd.Timestamp('2020-06-01')
        path = _snapshot_path(tmp_path, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=12, freq='MS'),
            'feat_a': rng.randn(12),
            'feat_b': rng.randn(12),
            'feat_c': rng.randn(12),
        })
        df.to_parquet(path, index=False)

        result = _batch_load_source(
            'TestSource', tmp_path, [month],
            allowed_features={'feat_a'}  # Only allow feat_a
        )

        wide = result['2020-06']
        data_cols = [c for c in wide.columns if c not in ['date', 'snapshot_date']]
        assert data_cols == ['feat_a']

    def test_empty_source_returns_empty_dict(self, tmp_path):
        """Source directory with no matching files returns empty dict."""
        months = [pd.Timestamp('2020-01-01')]
        result = _batch_load_source(
            'EmptySource', tmp_path, months,
            allowed_features={'feat_a'}
        )
        assert result == {}


class TestLoadAllSourcesFromCache:
    """Tests for _load_all_sources_from_cache assembly."""

    def test_merges_multiple_sources(self):
        """Multiple source caches should be merged on 'date'."""
        dates = pd.date_range('2019-01-01', periods=12, freq='MS')
        cache_a = {
            '2020-01': pd.DataFrame({
                'date': dates, 'feat_a': range(12)
            })
        }
        cache_b = {
            '2020-01': pd.DataFrame({
                'date': dates, 'feat_b': range(12, 24)
            })
        }

        result = _load_all_sources_from_cache(
            pd.Timestamp('2020-01-01'),
            {'SourceA': cache_a, 'SourceB': cache_b}
        )

        assert 'feat_a' in result.columns
        assert 'feat_b' in result.columns
        assert len(result) == 12

    def test_missing_month_returns_empty(self):
        """Month not in any source cache returns empty DataFrame."""
        result = _load_all_sources_from_cache(
            pd.Timestamp('2099-01-01'),
            {'SourceA': {}}
        )
        assert result.empty

    def test_partial_source_coverage(self):
        """If only some sources have the month, result includes available ones."""
        dates = pd.date_range('2019-01-01', periods=12, freq='MS')
        cache_a = {
            '2020-01': pd.DataFrame({'date': dates, 'feat_a': range(12)})
        }
        cache_b = {}  # Source B has no data for this month

        result = _load_all_sources_from_cache(
            pd.Timestamp('2020-01-01'),
            {'SourceA': cache_a, 'SourceB': cache_b}
        )

        assert 'feat_a' in result.columns
        assert len(result) == 12


# =============================================================================
# Progress Tracking
# =============================================================================

class TestProgressTracking:
    """Tests for checkpoint/resume progress tracking."""

    def test_no_progress_returns_empty(self, source_cache_dir):
        """No progress file should return empty set."""
        result = _load_progress('nsa', 'first_release')
        assert result == set()

    def test_save_and_load_round_trip(self, source_cache_dir):
        """Saved progress should be retrievable."""
        months = {'2020-01', '2020-02', '2020-03'}
        _save_progress('nsa', 'first_release', months)
        result = _load_progress('nsa', 'first_release')
        assert result == months

    def test_clear_removes_file(self, source_cache_dir):
        """Clear should delete the progress file."""
        _save_progress('sa', 'revised', {'2020-01'})
        path = _get_progress_path('sa', 'revised')
        assert path.exists()

        _clear_progress('sa', 'revised')
        assert not path.exists()

    def test_clear_nonexistent_is_safe(self, source_cache_dir):
        """Clearing non-existent progress should not raise."""
        _clear_progress('nsa', 'revised')  # No file to clear

    def test_incremental_progress(self, source_cache_dir):
        """Progress should accumulate across saves."""
        _save_progress('nsa', 'first_release', {'2020-01'})
        loaded = _load_progress('nsa', 'first_release')
        loaded.add('2020-02')
        _save_progress('nsa', 'first_release', loaded)

        result = _load_progress('nsa', 'first_release')
        assert result == {'2020-01', '2020-02'}

    def test_different_branches_independent(self, source_cache_dir):
        """Progress for different branches should not interfere."""
        _save_progress('nsa', 'first_release', {'2020-01'})
        _save_progress('sa', 'revised', {'2020-06'})

        nsa = _load_progress('nsa', 'first_release')
        sa = _load_progress('sa', 'revised')

        assert nsa == {'2020-01'}
        assert sa == {'2020-06'}

    def test_corrupted_progress_returns_empty(self, source_cache_dir):
        """Corrupted JSON should return empty set (safe restart)."""
        _save_progress('nsa', 'first_release', {'2020-01'})
        path = _get_progress_path('nsa', 'first_release')
        with open(path, 'w') as f:
            f.write("CORRUPTED{{{")

        result = _load_progress('nsa', 'first_release')
        assert result == set()
