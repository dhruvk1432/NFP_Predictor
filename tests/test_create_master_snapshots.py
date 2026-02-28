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
    _check_regime_cache,
    _save_regime_cache,
    _check_source_cache,
    _save_source_cache,
    _get_source_cache_path,
    _build_feature_selection_regimes,
    _resolve_regime_cutoff,
    _min_valid_obs_for_source,
    _process_source_features,
    _normalize_to_wide,
    _batch_load_source,
    _load_all_sources_from_cache,
    _snapshot_path,
    _load_progress,
    _save_progress,
    _clear_progress,
    _get_progress_path,
    _resolve_target_combos,
    _apply_target_combo_filters,
    _resolve_selection_target_mode,
    _build_selection_target,
    _parse_fs_stages_arg,
    MASTER_BASE,
    HARD_CODED_REGIME_STARTS,
)
from Train.data_loader import sanitize_feature_name


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def source_cache_dir(tmp_path, monkeypatch):
    """Redirect cache root to tmp_path so cache files don't pollute real data."""
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

    def test_cache_mode_mismatch_returns_none(self, source_cache_dir):
        """Cache built for one selection-target mode must not leak into another mode."""
        _save_source_cache(
            ['feat_a'], 'ADP', 'sa', 'revised',
            selection_target_mode='mom',
        )
        result = _check_source_cache(
            'ADP', 'sa', 'revised',
            selection_target_mode='model_aligned',
        )
        assert result is None

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

    def test_matches_sanitized_feature_names_and_outputs_sanitized_columns(self, tmp_path):
        """Selected sanitized names should map to raw columns and output sanitized columns."""
        month = pd.Timestamp('2020-06-01')
        path = _snapshot_path(tmp_path, month)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw_1 = "Prosper Consumer Spending Forecast|Consumer Spending Forecast (18-34)"
        raw_2 = (
            "Regarding the U.S. employment environment over the next six (6) months, "
            "do you think that there will be more, the same, or fewer layoffs than at "
            "present? Same (Males) diff"
        )

        df = pd.DataFrame(
            {
                'date': pd.date_range('2019-01-01', periods=3, freq='MS'),
                raw_1: [1.0, 2.0, 3.0],
                raw_2: [10.0, 11.0, 12.0],
                'other_feature': [99.0, 98.0, 97.0],
            }
        )
        df.to_parquet(path, index=False)

        s1 = sanitize_feature_name(raw_1)
        s2 = sanitize_feature_name(raw_2)
        out = _batch_load_source(
            'TestSource',
            tmp_path,
            [month],
            allowed_features={s1, s2},
        )

        assert '2020-06' in out
        wide = out['2020-06']
        assert s1 in wide.columns
        assert s2 in wide.columns
        assert raw_1 not in wide.columns
        assert raw_2 not in wide.columns
        assert 'other_feature' not in wide.columns
        assert wide[s1].tolist() == [1.0, 2.0, 3.0]
        assert wide[s2].tolist() == [10.0, 11.0, 12.0]

    def test_handles_sanitized_name_collisions_with_first_non_null(self, tmp_path):
        """If two raw columns sanitize to same key, output should use first non-null value."""
        month = pd.Timestamp('2020-07-01')
        path = _snapshot_path(tmp_path, month)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw_a = "a.b"
        raw_b = "a|b"
        sanitized = sanitize_feature_name(raw_a)
        assert sanitized == sanitize_feature_name(raw_b)

        df = pd.DataFrame(
            {
                'date': pd.date_range('2019-01-01', periods=2, freq='MS'),
                raw_a: [np.nan, 2.0],
                raw_b: [1.0, np.nan],
            }
        )
        df.to_parquet(path, index=False)

        out = _batch_load_source(
            'TestSource',
            tmp_path,
            [month],
            allowed_features={sanitized},
        )
        wide = out['2020-07']
        assert sanitized in wide.columns
        assert wide[sanitized].tolist() == [1.0, 2.0]


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


class TestRegimeSelection:
    """Tests for leakage-safe regime feature selection and routing."""

    def test_build_feature_selection_regimes_uses_window_relevant_hard_coded_cutoffs(self):
        months = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-01"),
            pd.Timestamp("2026-02-01"),
        ]
        cutoffs = _build_feature_selection_regimes(months)
        assert cutoffs == [
            pd.Timestamp("2022-03-01"),
            pd.Timestamp("2025-02-01"),
        ]

    def test_build_feature_selection_regimes_returns_all_for_full_history_window(self):
        months = [
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("2026-02-01"),
        ]
        cutoffs = _build_feature_selection_regimes(months)
        assert cutoffs == [
            pd.Timestamp("1998-01-01"),
            pd.Timestamp("2008-01-01"),
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2022-03-01"),
            pd.Timestamp("2025-02-01"),
        ]

    def test_build_feature_selection_regimes_returns_empty_before_first_regime(self):
        months = [
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("1997-12-01"),
        ]
        cutoffs = _build_feature_selection_regimes(months)
        assert cutoffs == []

    def test_hard_coded_regimes_start_in_1998(self):
        assert HARD_CODED_REGIME_STARTS[0][1] == pd.Timestamp("1998-01-01")

    def test_regime_cache_round_trip(self, source_cache_dir):
        cutoff = pd.Timestamp("2020-01-01")
        _save_regime_cache(["feat_a", "feat_b"], "nsa", "first_release", cutoff)
        loaded = _check_regime_cache("nsa", "first_release", cutoff)
        assert sorted(loaded) == ["feat_a", "feat_b"]

    def test_resolve_regime_cutoff_uses_latest_past_cutoff(self):
        cutoffs = [
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2021-01-01"),
        ]
        obs_month = pd.Timestamp("2020-11-01")
        resolved = _resolve_regime_cutoff(obs_month, cutoffs)
        assert resolved == pd.Timestamp("2020-01-01")

    def test_resolve_regime_cutoff_backfills_to_first_cutoff(self):
        cutoffs = [
            pd.Timestamp("1998-01-01"),
            pd.Timestamp("2008-01-01"),
        ]
        obs_month = pd.Timestamp("1997-12-01")
        resolved = _resolve_regime_cutoff(obs_month, cutoffs)
        assert resolved == pd.Timestamp("1998-01-01")

    def test_create_master_snapshots_routes_months_to_past_regime(self, tmp_path, monkeypatch):
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, "MASTER_BASE", master_base)
        monkeypatch.setattr(cms, "TARGET_COMBOS", [("nsa", "first_release")])
        monkeypatch.setattr(cms, "SOURCES", {"ADP": tmp_path / "mock_sources" / "adp" / "decades"})
        monkeypatch.setattr(cms, "SOURCE_EXEC_ORDER", ["ADP"])
        monkeypatch.setattr(cms, "START_DATE", "2020-01-01")
        monkeypatch.setattr(cms, "END_DATE", "2021-02-01")

        month_pairs = {
            pd.Timestamp("2019-12-01"): pd.Timestamp("2019-12-06"),
            pd.Timestamp("2020-01-01"): pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-02-01"): pd.Timestamp("2020-02-07"),
            pd.Timestamp("2021-01-01"): pd.Timestamp("2021-01-08"),
            pd.Timestamp("2021-02-01"): pd.Timestamp("2021-02-05"),
        }
        monkeypatch.setattr(cms, "get_nfp_release_map", lambda start_date, end_date: month_pairs)
        monkeypatch.setattr(
            cms,
            "_build_feature_selection_regimes",
            lambda months, refresh_months=cms.FEATURE_SELECTION_REGIME_REFRESH_MONTHS: [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2021-01-01"),
            ],
        )
        monkeypatch.setattr(cms, "_check_regime_cache", lambda *args, **kwargs: None)

        def fake_parallel_selection(target_cat, target_source, asof_month=None, stages=None, selection_target_mode='auto'):
            asof_key = pd.Timestamp(asof_month).strftime("%Y-%m")
            return ["feat_old"] if asof_key == "2020-01" else ["feat_new"]

        def fake_batch_load(source_name, source_dir, snapshot_months, allowed_features):
            keep_col = "feat_new" if "feat_new" in allowed_features else "feat_old"
            result = {}
            for month in snapshot_months:
                result[month.strftime("%Y-%m")] = pd.DataFrame(
                    {
                        "date": pd.date_range(month, periods=2, freq="MS"),
                        keep_col: [1.0, 2.0],
                    }
                )
            return result

        monkeypatch.setattr(cms, "_run_parallel_feature_selection", fake_parallel_selection)
        monkeypatch.setattr(cms, "_batch_load_source", fake_batch_load)

        cms.create_master_snapshots(skip_existing=False)

        target_dir = master_base / "nsa" / "first_release" / "decades"
        pre_path = cms._snapshot_path(target_dir, pd.Timestamp("2019-12-01"))
        old_path = cms._snapshot_path(target_dir, pd.Timestamp("2020-02-01"))
        new_path = cms._snapshot_path(target_dir, pd.Timestamp("2021-02-01"))
        assert pre_path.exists()
        assert old_path.exists()
        assert new_path.exists()

        pre_df = pd.read_parquet(pre_path)
        old_df = pd.read_parquet(old_path)
        new_df = pd.read_parquet(new_path)
        assert "feat_old" in pre_df.columns
        assert "feat_new" not in pre_df.columns
        assert "feat_old" in old_df.columns
        assert "feat_new" not in old_df.columns
        assert "feat_new" in new_df.columns
        assert "feat_old" not in new_df.columns


class TestTargetScopeResolution:
    """Tests for auto/all/revised/first_release branch selection."""

    def test_auto_uses_revised_for_short_window(self):
        combos = _resolve_target_combos(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2026-02-01"),
            target_source_scope="auto",
        )
        assert combos == [("nsa", "revised"), ("sa", "revised")]

    def test_auto_uses_all_for_long_window(self):
        combos = _resolve_target_combos(
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("2026-02-01"),
            target_source_scope="auto",
        )
        assert ("nsa", "first_release") in combos
        assert ("nsa", "revised") in combos
        assert ("sa", "first_release") in combos
        assert ("sa", "revised") in combos
        assert len(combos) == 4

    def test_explicit_scope_revised(self):
        combos = _resolve_target_combos(
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("2026-02-01"),
            target_source_scope="revised",
        )
        assert combos == [("nsa", "revised"), ("sa", "revised")]


class TestTargetComboFilters:
    def test_target_type_scope_sa_filters_to_sa_branches(self):
        filtered = _apply_target_combo_filters(
            target_combos=[
                ("nsa", "first_release"),
                ("nsa", "revised"),
                ("sa", "first_release"),
                ("sa", "revised"),
            ],
            target_type_scope="sa",
        )
        assert filtered == [("sa", "first_release"), ("sa", "revised")]

    def test_explicit_branches_override_scope(self):
        filtered = _apply_target_combo_filters(
            target_combos=[
                ("nsa", "first_release"),
                ("nsa", "revised"),
                ("sa", "first_release"),
                ("sa", "revised"),
            ],
            target_type_scope="all",
            branches=["sa_revised"],
        )
        assert filtered == [("sa", "revised")]


class TestSelectionTargetModes:
    def test_auto_mode_prefers_model_aligned_for_sa(self):
        assert _resolve_selection_target_mode("sa", "revised", "auto") == "model_aligned"

    def test_auto_mode_prefers_mom_for_nsa(self):
        assert _resolve_selection_target_mode("nsa", "first_release", "auto") == "mom"

    def test_build_selection_target_model_aligned_shape(self):
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        y = pd.Series(np.linspace(-100.0, 200.0, len(idx)), index=idx)
        target, mode = _build_selection_target(
            y, target_cat="sa", target_source="revised", selection_target_mode="model_aligned"
        )
        assert mode == "model_aligned"
        assert target.index.equals(y.index)
        assert target.notna().sum() >= len(y) - 1  # first diff row can be NaN


class TestFsStageParsing:
    def test_parse_fs_stages_arg(self):
        assert _parse_fs_stages_arg("0,1,2,4") == (0, 1, 2, 4)
        assert _parse_fs_stages_arg("0,1,1,2") == (0, 1, 2)
        assert _parse_fs_stages_arg(None) is None


class TestSkipFeatureSelectionFastPath:
    """Tests for skip-feature-selection parallel fast-path activation/fallback."""

    def test_dispatches_to_fast_path_when_enabled(self, tmp_path, monkeypatch):
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, "MASTER_BASE", master_base)
        monkeypatch.setattr(cms, "START_DATE", "2020-01-01")
        monkeypatch.setattr(cms, "END_DATE", "2020-01-01")
        monkeypatch.setattr(
            cms,
            "get_nfp_release_map",
            lambda start_date, end_date: {
                pd.Timestamp("2020-01-01"): pd.Timestamp("2020-01-10")
            },
        )
        monkeypatch.setattr(
            cms,
            "_resolve_target_combos",
            lambda start_dt, end_dt, scope: list(cms.DEFAULT_TARGET_COMBOS),
        )

        called = {}

        def fake_fast_path(target_combos, snapshot_pairs, skip_existing):
            called["target_combos"] = list(target_combos)
            called["snapshot_pairs"] = list(snapshot_pairs)
            called["skip_existing"] = skip_existing
            return True

        monkeypatch.setattr(cms, "_run_skip_fs_parallel_fast_path", fake_fast_path)

        cms.create_master_snapshots(skip_feature_selection=True, skip_existing=True)

        assert called["target_combos"] == list(cms.DEFAULT_TARGET_COMBOS)
        assert called["snapshot_pairs"] == [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))
        ]
        assert called["skip_existing"] is True

    def test_falls_back_to_legacy_skip_fs_when_fast_path_unavailable(self, tmp_path, monkeypatch):
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        source_dir = tmp_path / "mock_sources" / "adp" / "decades"
        target_obs = pd.Timestamp("2020-01-01")
        target_snap = pd.Timestamp("2020-01-10")

        monkeypatch.setattr(cms, "MASTER_BASE", master_base)
        monkeypatch.setattr(cms, "TARGET_COMBOS", [("nsa", "first_release")])
        monkeypatch.setattr(cms, "SOURCES", {"ADP": source_dir})
        monkeypatch.setattr(cms, "SOURCE_EXEC_ORDER", ["ADP"])
        monkeypatch.setattr(cms, "START_DATE", "2020-01-01")
        monkeypatch.setattr(cms, "END_DATE", "2020-01-01")
        monkeypatch.setattr(
            cms,
            "get_nfp_release_map",
            lambda start_date, end_date: {target_obs: target_snap},
        )
        monkeypatch.setattr(
            cms,
            "_resolve_target_combos",
            lambda start_dt, end_dt, scope: [("nsa", "first_release")],
        )
        monkeypatch.setattr(cms, "_run_skip_fs_parallel_fast_path", lambda *args, **kwargs: False)

        cache_path = master_base / "selected_features_nsa_first_release.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "last_run_date": "2026-01-01",
                    "target_source": "first_release",
                    "target_cat": "nsa",
                    "features": ["feat_a"],
                },
                f,
            )

        def fake_batch_load(source_name, source_dir, snapshot_months, allowed_features):
            result = {}
            for month in snapshot_months:
                result[month.strftime("%Y-%m")] = pd.DataFrame(
                    {
                        "date": pd.date_range(month, periods=2, freq="MS"),
                        "feat_a": [1.0, 2.0],
                    }
                )
            return result

        monkeypatch.setattr(cms, "_batch_load_source", fake_batch_load)

        cms.create_master_snapshots(skip_feature_selection=True, skip_existing=False)

        out_path = cms._snapshot_path(
            master_base / "nsa" / "first_release" / "decades",
            target_obs,
        )
        assert out_path.exists()
        out_df = pd.read_parquet(out_path)
        assert "feat_a" in out_df.columns


class TestSourceSpecificMinObs:
    """Tests for source-specific short-history filtering."""

    def test_process_source_features_uses_source_threshold(self, tmp_path, monkeypatch):
        import Data_ETA_Pipeline.create_master_snapshots as cms

        source_dir = tmp_path / "adp" / "decades"
        obs_month = pd.Timestamp("2020-01-01")
        path = _snapshot_path(source_dir, obs_month)
        path.parent.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range("2010-01-01", periods=90, freq="MS")
        feat_short = np.arange(90, dtype=float)
        feat_short[:25] = np.nan  # 65 valid observations
        feat_long = np.arange(90, dtype=float) + 10.0
        df = pd.DataFrame(
            {
                "date": dates,
                "feat_short": feat_short,
                "feat_long": feat_long,
            }
        )
        df.to_parquet(path, index=False)

        monkeypatch.setitem(cms.SOURCE_MIN_VALID_OBS, "ADP", 70)
        assert _min_valid_obs_for_source("ADP") == 70

        target_df = pd.DataFrame({"ds": dates, "y_mom": np.linspace(0.0, 1.0, len(dates))})
        monkeypatch.setattr(
            cms,
            "load_target_data",
            lambda target_type, release_type, use_cache=False: target_df,
        )

        captured = {}

        def fake_pipeline(snap_wide, y_mom, source_name, snapshots_dir, series_groups, stages=None):
            captured["cols"] = list(snap_wide.columns)
            return list(snap_wide.columns)

        monkeypatch.setattr(cms, "run_full_source_pipeline", fake_pipeline)

        selected = _process_source_features(
            "ADP",
            source_dir,
            "nsa",
            "first_release",
            asof_month=obs_month,
        )

        assert "feat_long" in selected
        assert "feat_short" not in selected
        assert captured["cols"] == ["feat_long"]


class TestSingleAsOfSelection:
    def test_single_selection_asof_and_stage_override(self, tmp_path, monkeypatch):
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, "MASTER_BASE", master_base)
        monkeypatch.setattr(cms, "SOURCES", {"ADP": tmp_path / "mock_sources" / "adp" / "decades"})
        monkeypatch.setattr(cms, "SOURCE_EXEC_ORDER", ["ADP"])
        monkeypatch.setattr(cms, "START_DATE", "2020-01-01")
        monkeypatch.setattr(cms, "END_DATE", "2020-02-01")
        monkeypatch.setattr(
            cms,
            "get_nfp_release_map",
            lambda start_date, end_date: {
                pd.Timestamp("2020-01-01"): pd.Timestamp("2020-01-10"),
                pd.Timestamp("2020-02-01"): pd.Timestamp("2020-02-07"),
            },
        )

        captured = {"calls": []}

        def fake_parallel_selection(
            target_cat,
            target_source,
            asof_month=None,
            stages=None,
            selection_target_mode='auto',
        ):
            captured["calls"].append(
                (
                    target_cat,
                    target_source,
                    pd.Timestamp(asof_month),
                    tuple(stages),
                    selection_target_mode,
                )
            )
            return ["feat_static"]

        def fake_batch_load(source_name, source_dir, snapshot_months, allowed_features):
            result = {}
            for month in snapshot_months:
                result[month.strftime("%Y-%m")] = pd.DataFrame(
                    {
                        "date": pd.date_range(month, periods=2, freq="MS"),
                        "feat_static": [1.0, 2.0],
                    }
                )
            return result

        monkeypatch.setattr(cms, "_run_parallel_feature_selection", fake_parallel_selection)
        monkeypatch.setattr(cms, "_batch_load_source", fake_batch_load)

        cms.create_master_snapshots(
            target_type_scope="sa",
            branches=["sa_revised"],
            selection_target_mode="model_aligned",
            fs_stages_override=(0, 1, 2, 4),
            single_selection_asof="2021-12",
            skip_existing=False,
        )

        assert len(captured["calls"]) == 1
        call = captured["calls"][0]
        assert call[0] == "sa"
        assert call[1] == "revised"
        assert call[2] == pd.Timestamp("2021-12-01")
        assert call[3] == (0, 1, 2, 4)
        assert call[4] == "model_aligned"
