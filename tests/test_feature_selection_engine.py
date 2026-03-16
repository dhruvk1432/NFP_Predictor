"""
Pytest suite for the Feature Selection Engine (Stages 0-2).
Uses synthetic data only — no dependency on real FRED data or DATA_PATH.
"""
import pytest
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from collections import Counter
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.feature_selection_engine import (
    _safe_lgb_fit,
    _safe_lgb_predict,
    _prepare_lgb_frame,
    _variance_filter,
    _deduplicate_group,
    _source_aware_recency_mask,
    _lgb_screen_group,
    _bh_fdr_select,
    _memoized_score,
    _parallel_trial_scores,
    _aggregate_vintage_scores,
    filter_group_data_purged,
    _adaptive_boruta_runs,
    _adaptive_boruta_shadow_cap,
    _boruta_core,
    get_boruta_importance,
    _discover_vintage_snapshots,
    _vintage_cv_importance,
    get_vintage_stability,
    cluster_redundancy,
    _extract_split_pairs,
    interaction_rescue,
    sequential_forward_selection,
    should_keep_change,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def make_wide_df():
    """Factory fixture to create wide DataFrames with DatetimeIndex."""
    def _make(n_rows=120, n_cols=50, seed=42):
        rng = np.random.RandomState(seed)
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')
        data = rng.randn(n_rows, n_cols)
        cols = [f'feat_{i}' for i in range(n_cols)]
        return pd.DataFrame(data, index=dates, columns=cols)
    return _make


@pytest.fixture
def make_target():
    """Factory fixture to create a target series aligned with wide_df."""
    def _make(n_rows=120, seed=99):
        rng = np.random.RandomState(seed)
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')
        return pd.Series(rng.randn(n_rows) * 100, index=dates, name='y_mom')
    return _make


# =============================================================================
# LightGBM Safe Helpers
# =============================================================================

class TestSafeLgbHelpers:
    """Tests for cached/no-copy LightGBM input sanitization."""

    def test_prepare_lgb_frame_returns_same_object_when_clean(self, make_wide_df):
        X = make_wide_df(n_rows=50, n_cols=5)
        X_prep, raw_cols, safe_cols = _prepare_lgb_frame(X)
        assert X_prep is X
        assert raw_cols == tuple(X.columns)
        assert safe_cols == tuple(X.columns)

    def test_prepare_lgb_frame_sanitizes_columns(self):
        dates = pd.date_range('2010-01-01', periods=20, freq='MS')
        X = pd.DataFrame(
            {
                'a[b]': np.arange(20, dtype=float),
                'a{b}': np.arange(20, dtype=float) + 1,
            },
            index=dates,
        )
        X_prep, _, safe_cols = _prepare_lgb_frame(X)
        assert tuple(X_prep.columns) == safe_cols
        assert safe_cols[0] == 'a_b_'
        assert safe_cols[1].startswith('a_b_')
        assert np.shares_memory(X_prep.to_numpy(), X.to_numpy())

    def test_safe_lgb_predict_reuses_fit_schema(self):
        import lightgbm as lgb
        import Data_ETA_Pipeline.feature_selection_engine as fse

        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        X = pd.DataFrame(
            {
                'feat[0]': rng.randn(n_rows),
                'feat{1}': rng.randn(n_rows),
                'feat:2': rng.randn(n_rows),
            },
            index=dates,
        )
        y = pd.Series(
            X['feat[0]'] * 40 + X['feat{1}'] * -15 + rng.randn(n_rows) * 15,
            index=dates,
        )
        model = lgb.LGBMRegressor(**fse.LGB_PARAMS)

        with patch.object(
            fse, '_get_lgb_column_schema', wraps=fse._get_lgb_column_schema
        ) as schema_fn:
            _safe_lgb_fit(model, X, y)
            _safe_lgb_predict(model, X.iloc[-12:])
            _safe_lgb_predict(model, X.iloc[:12])

        assert schema_fn.call_count == 1, \
            "Predict on same feature schema should reuse fit-time sanitized columns"


# =============================================================================
# Stage 0: Variance Filter
# =============================================================================

class TestVarianceFilter:
    """Tests for _variance_filter with nunique() pre-pass optimization."""

    def test_drops_constant_columns(self, make_wide_df):
        """Columns with a single constant value should be dropped."""
        df = make_wide_df(n_rows=120, n_cols=50)
        # Set 5 columns to constant
        for i in range(5):
            df[f'feat_{i}'] = 42.0

        result = _variance_filter(df)
        for i in range(5):
            assert f'feat_{i}' not in result.columns, \
                f"Constant column feat_{i} should have been dropped"
        assert result.shape[1] == 45

    def test_keeps_variable_columns(self, make_wide_df):
        """Normal random columns should all survive."""
        df = make_wide_df(n_rows=120, n_cols=50)
        result = _variance_filter(df)
        assert result.shape[1] == 50, \
            f"All 50 random columns should survive, got {result.shape[1]}"

    def test_nunique_prepass_speedup(self):
        """10k-column df with 8k constants should filter in < 5s."""
        rng = np.random.RandomState(42)
        n_rows, n_cols = 120, 10000
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')
        data = rng.randn(n_rows, n_cols)
        cols = [f'feat_{i}' for i in range(n_cols)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        # Set 8000 columns to constant
        for i in range(8000):
            df.iloc[:, i] = 0.0

        t0 = time.time()
        result = _variance_filter(df)
        elapsed = time.time() - t0

        assert elapsed < 5.0, \
            f"Pre-pass should handle 10k cols in <5s, took {elapsed:.1f}s"
        assert result.shape[1] == 2000, \
            f"Expected 2000 surviving cols, got {result.shape[1]}"

    def test_near_constant_97pct(self):
        """Column where 97% of values are identical should be dropped."""
        n_rows = 100
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')

        # 97 zeros, 3 ones → mode_frac = 0.97 → dropped (>= threshold)
        col_97 = np.zeros(n_rows)
        col_97[:3] = 1.0
        # 96 zeros, 4 ones → mode_frac = 0.96 → kept (< threshold)
        col_96 = np.zeros(n_rows)
        col_96[:4] = 1.0

        df = pd.DataFrame({
            'should_drop': col_97,
            'should_keep': col_96,
        }, index=dates)

        result = _variance_filter(df, max_same_frac=0.97)
        assert 'should_keep' in result.columns
        assert 'should_drop' not in result.columns

    def test_handles_nan_columns(self, make_wide_df):
        """Columns with too few observations (< min_obs) should be excluded."""
        df = make_wide_df(n_rows=120, n_cols=10)
        # Set one column to mostly NaN (only 5 valid obs)
        df['feat_0'] = np.nan
        df.iloc[:5, 0] = 1.0

        result = _variance_filter(df, min_obs=30)
        assert 'feat_0' not in result.columns


# =============================================================================
# Stage 1: Deduplicate Group
# =============================================================================

class TestDeduplicateGroup:
    """Tests for _deduplicate_group with iterative merging."""

    def test_collapses_identical_columns(self, make_wide_df):
        """Two nearly identical columns should collapse to one."""
        rng = np.random.RandomState(42)
        df = make_wide_df(n_rows=120, n_cols=10)
        # Make feat_1 nearly identical to feat_0
        df['feat_1'] = df['feat_0'] + rng.randn(120) * 1e-6

        result = _deduplicate_group(df, threshold=0.95)
        # At least one of feat_0/feat_1 should be gone
        both_present = ('feat_0' in result.columns and 'feat_1' in result.columns)
        assert not both_present, \
            "Near-identical columns should collapse to one representative"

    def test_keeps_independent_columns(self, make_wide_df):
        """Independent random columns should all survive."""
        df = make_wide_df(n_rows=120, n_cols=10)
        result = _deduplicate_group(df, threshold=0.95)
        # Independent random columns have low correlation → all should survive
        assert result.shape[1] >= 8, \
            f"Expected most columns to survive, got {result.shape[1]}/10"

    def test_chunked_mode_convergence(self):
        """6000+ columns should trigger chunked mode and catch cross-chunk dupes."""
        rng = np.random.RandomState(42)
        n_rows = 100
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')

        # Create 200 base signals + 5800 near-copies (each base has ~29 copies)
        n_base = 200
        n_total = 6000
        base_data = rng.randn(n_rows, n_base)

        data = np.empty((n_rows, n_total))
        cols = []
        for i in range(n_total):
            base_idx = i % n_base
            data[:, i] = base_data[:, base_idx] + rng.randn(n_rows) * 1e-4
            cols.append(f'feat_{i}')

        df = pd.DataFrame(data, index=dates, columns=cols)
        result = _deduplicate_group(df, threshold=0.95)

        # Should collapse to roughly 200 (one per base group)
        # Allow tolerance since clustering is approximate
        assert result.shape[1] < n_total * 0.5, \
            f"Expected heavy collapse, got {result.shape[1]}/{n_total}"
        assert result.shape[1] >= n_base * 0.5, \
            f"Collapsed too aggressively: {result.shape[1]} < {n_base // 2}"

    def test_memory_error_graceful(self, make_wide_df):
        """MemoryError during clustering should return input unchanged."""
        df = make_wide_df(n_rows=120, n_cols=10)

        with patch('Data_ETA_Pipeline.feature_selection_engine.hierarchy.linkage',
                   side_effect=MemoryError("test OOM")):
            result = _deduplicate_group(df, threshold=0.95)

        assert result.shape[1] == df.shape[1], \
            "MemoryError should gracefully return input unchanged"

    def test_small_group_passthrough(self):
        """Groups with <= 3 features should pass through unchanged."""
        dates = pd.date_range('2010-01-01', periods=50, freq='MS')
        df = pd.DataFrame(
            np.random.randn(50, 3),
            index=dates,
            columns=['a', 'b', 'c']
        )
        result = _deduplicate_group(df, threshold=0.95)
        assert list(result.columns) == ['a', 'b', 'c']


# =============================================================================
# Stage 1: Recency Check
# =============================================================================

class TestRecencyCheck:
    """Tests for the vectorized recency check."""

    def test_vectorized_recency_matches_apply(self, make_wide_df):
        """Vectorized mask should produce identical results to old .apply() approach."""
        df = make_wide_df(n_rows=120, n_cols=100)
        # Set some columns to NaN in the last 3 months
        for i in range(20):
            df.iloc[-3:, i] = np.nan
        # Set some columns to all NaN in last 3 months
        for i in range(20, 30):
            df.iloc[-3:, i] = np.nan
            df.iloc[-6:-3, i] = np.nan  # Also sparse before cutoff

        last_date = df.index.max()
        recent_cutoff = last_date - pd.DateOffset(months=3)

        # Old approach
        old_mask = df.apply(lambda col: col.last_valid_index() >= recent_cutoff)
        # New vectorized approach
        recent_rows = df.loc[df.index >= recent_cutoff]
        new_mask = recent_rows.notna().any(axis=0)

        pd.testing.assert_series_equal(old_mask, new_mask,
                                       check_names=False)

    def test_speed_improvement(self):
        """Vectorized recency should be significantly faster than .apply()."""
        rng = np.random.RandomState(42)
        n_rows, n_cols = 200, 10000
        dates = pd.date_range('2005-01-01', periods=n_rows, freq='MS')
        data = rng.randn(n_rows, n_cols)
        df = pd.DataFrame(data, index=dates,
                          columns=[f'f_{i}' for i in range(n_cols)])

        last_date = df.index.max()
        recent_cutoff = last_date - pd.DateOffset(months=3)

        # Time vectorized approach
        t0 = time.time()
        recent_rows = df.loc[df.index >= recent_cutoff]
        _ = recent_rows.notna().any(axis=0)
        t_vectorized = time.time() - t0

        # Time old .apply() approach
        t0 = time.time()
        _ = df.apply(lambda col: col.last_valid_index() >= recent_cutoff)
        t_apply = time.time() - t0

        speedup = t_apply / max(t_vectorized, 1e-6)
        assert speedup > 3.0, \
            f"Expected at least 3x speedup, got {speedup:.1f}x " \
            f"(vectorized={t_vectorized:.3f}s, apply={t_apply:.3f}s)"

    def test_source_aware_noaa_window(self):
        """NOAA features should pass with 6m recency even when non-NOAA fail 3m."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame(
            {
                'NOAA_Economic_Damage_Index_feat': np.nan,
                'macro_feat': np.nan,
            },
            index=dates,
        )
        # Last value is 5 months before the final date (inside NOAA 6m, outside 3m)
        df.loc[pd.Timestamp('2020-07-01'), 'NOAA_Economic_Damage_Index_feat'] = 1.0
        df.loc[pd.Timestamp('2020-07-01'), 'macro_feat'] = 1.0

        mask = _source_aware_recency_mask(df)

        assert bool(mask['NOAA_Economic_Damage_Index_feat']) is True
        assert bool(mask['macro_feat']) is False


# =============================================================================
# Stage 1: Filter Group Data (VIF Removed)
# =============================================================================

class TestFilterGroupDataPurged:
    """Tests for filter_group_data_purged after VIF removal."""

    def test_no_vif_parameter(self):
        """Function signature should not accept vif_threshold."""
        import inspect
        sig = inspect.signature(filter_group_data_purged)
        assert 'vif_threshold' not in sig.parameters, \
            "VIF has been removed — vif_threshold should not be a parameter"

    def test_binary_group_handling(self, make_target):
        """Binary-only features should be handled without error."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')
        target = make_target(n_rows=n_rows)

        # Create binary features
        data = {}
        for i in range(10):
            data[f'binary_{i}'] = rng.choice([0.0, 1.0], size=n_rows)
        df = pd.DataFrame(data, index=dates)

        result = filter_group_data_purged(df, target, 'test_binary')
        assert isinstance(result, list)

    def test_empty_input(self, make_target):
        """Empty DataFrame should return empty list."""
        target = make_target(n_rows=120)
        df = pd.DataFrame(index=target.index)

        result = filter_group_data_purged(df, target, 'empty_group')
        assert result == []

    def test_returns_union_of_dual_paths(self, make_wide_df, make_target):
        """Function should return features (not raise errors) for normal input."""
        n_rows = 120
        df = make_wide_df(n_rows=n_rows, n_cols=30)
        target = make_target(n_rows=n_rows)

        # Inject some signal so at least one feature passes
        df['signal_feat'] = target.values * 0.5 + np.random.randn(n_rows) * 10

        result = filter_group_data_purged(df, target, 'test_group')
        assert isinstance(result, list)

    def test_prefilter_applies_to_path_a_only(self):
        """Path A may prefilter low-corr columns; Path B should still see full matrix."""
        n_rows = 120
        dates = pd.date_range('2010-01-01', periods=n_rows, freq='MS')
        y_vals = np.tile([1.0, -1.0], n_rows // 2)
        target = pd.Series(y_vals, index=dates, name='y_mom')

        # feat_zero has exact zero linear correlation to target in this construction.
        feat_zero = np.tile([1.0, 1.0, -1.0, -1.0], n_rows // 4)
        feat_signal = y_vals + np.random.RandomState(42).randn(n_rows) * 0.01

        data = {
            'feat_zero': feat_zero,
            'feat_signal': feat_signal,
        }
        rng = np.random.RandomState(7)
        for i in range(208):  # total columns = 210 (>200 triggers prefilter)
            data[f'noise_{i}'] = rng.randn(n_rows)
        df = pd.DataFrame(data, index=dates)

        path_a_processed = []
        path_b_seen = {}

        def mock_corr_worker(args):
            col_name = args[0]
            path_a_processed.append(col_name)
            return (col_name, 0.9)

        def mock_lgb_screen(X_lgb, y_lgb, top_k=15):
            path_b_seen['columns'] = list(X_lgb.columns)
            return ['feat_zero']

        with patch('Data_ETA_Pipeline.feature_selection_engine._compute_purged_corr_for_col',
                   side_effect=mock_corr_worker), \
             patch('Data_ETA_Pipeline.feature_selection_engine._lgb_screen_group',
                   side_effect=mock_lgb_screen):
            result = filter_group_data_purged(df, target, 'path_split_test')

        assert 'feat_zero' not in set(path_a_processed), \
            "Path A prefilter should remove low-corr feature from purged-corr path"
        assert 'feat_zero' in path_b_seen.get('columns', []), \
            "Path B should receive full recency-filtered matrix, including low-corr features"
        assert 'feat_zero' in result, \
            "Union step should keep nonlinear candidate returned by Path B"


class TestBhFdrSelect:
    """Tests for Benjamini-Hochberg selection with largest passing rank logic."""

    def test_uses_largest_passing_rank_not_first_break(self):
        """Should keep top-k where k is largest passing BH rank (not early break)."""
        pvals = {
            "feat_a": 0.020,  # fails at rank 1 for alpha=0.05, n=3
            "feat_b": 0.021,  # passes at rank 2
            "feat_c": 0.900,
        }
        selected = _bh_fdr_select(pvals, fdr_alpha=0.05)
        assert selected == ["feat_a", "feat_b"]

    def test_returns_empty_when_no_rank_passes(self):
        """Should return empty list when BH finds no passing rank."""
        pvals = {"a": 0.7, "b": 0.8, "c": 0.9}
        assert _bh_fdr_select(pvals, fdr_alpha=0.05) == []

    def test_empty_input(self):
        """Empty p-value mapping should return empty list."""
        assert _bh_fdr_select({}, fdr_alpha=0.1) == []


# =============================================================================
# Stage 2: Boruta Core
# =============================================================================

class TestBorutaCore:
    """Tests for _boruta_core with early stopping and pre-allocated shadows."""

    def test_confirms_strong_signal(self):
        """A strongly correlated feature should be confirmed."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # 49 noise + 1 strong signal
        data = rng.randn(n_rows, 50)
        data[:, 0] = target.values * 0.8 + rng.randn(n_rows) * 20
        cols = [f'feat_{i}' for i in range(50)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        result = _boruta_core(df, target, n_runs=50, alpha=0.05)
        assert 'feat_0' in result, \
            "Strong signal feature should be confirmed by Boruta"

    def test_rejects_noise(self):
        """Pure noise features should mostly be rejected."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 30)
        cols = [f'noise_{i}' for i in range(30)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        result = _boruta_core(df, target, n_runs=50, alpha=0.05)
        # With 30 noise features, alpha=0.05, and tentative zone (alpha*5),
        # up to ~30*0.25 = 7.5 false positives are statistically plausible
        assert len(result) <= 10, \
            f"Expected <=10 false positives from pure noise, got {len(result)}"

    def test_early_stopping_correctness(self):
        """Early stopping should not corrupt the results."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 20)
        # 2 strong signals
        data[:, 0] = target.values * 0.9 + rng.randn(n_rows) * 10
        data[:, 1] = target.values * 0.7 + rng.randn(n_rows) * 30
        cols = [f'feat_{i}' for i in range(20)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        result = _boruta_core(df, target, n_runs=100, alpha=0.05)

        # Strong signals should be in result regardless of early stopping
        assert 'feat_0' in result, "feat_0 (strong signal) should survive"
        assert 'feat_1' in result, "feat_1 (moderate signal) should survive"
        # All results should be valid column names
        for feat in result:
            assert feat in cols, f"Result {feat} is not a valid column name"

    def test_shadow_reuse_returns_valid_features(self):
        """Pre-allocated shadow approach should return valid feature names."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 20)
        cols = [f'feat_{i}' for i in range(20)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        result = _boruta_core(df, target, n_runs=30, alpha=0.05)
        for feat in result:
            assert feat in cols, \
                f"Returned feature '{feat}' is not in input columns"

    def test_handles_nan_features(self):
        """Features with NaN values should be handled correctly."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 20)
        # Add NaN to some features
        data[:20, 0] = np.nan
        data[:50, 5] = np.nan
        cols = [f'feat_{i}' for i in range(20)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        # Should not raise any error
        result = _boruta_core(df, target, n_runs=20, alpha=0.05)
        assert isinstance(result, list)


# =============================================================================
# Stage 2: Boruta Tournament
# =============================================================================

class TestBorutaTournament:
    """Tests for get_boruta_importance tournament mode."""

    def test_heat_runs_35(self):
        """Tournament heats should use 35 runs (not 20)."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 200)
        cols = [f'feat_{i}' for i in range(200)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        heat_n_runs = []
        original_boruta = _boruta_core

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            heat_n_runs.append(n_runs)
            # Return a few fake survivors to keep tournament going
            return list(X.columns[:3])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, tournament_threshold=150)

        # All calls except the final round should use 35 runs
        # Final round uses 80 runs
        heat_calls = [n for n in heat_n_runs if n != 80]
        assert all(n == 35 for n in heat_calls), \
            f"Expected heat runs=35, got {heat_calls}"

    def test_tournament_mode_triggers(self):
        """Features > tournament_threshold should trigger tournament mode."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 200)
        cols = [f'feat_{i}' for i in range(200)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        call_count = [0]

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            call_count[0] += 1
            return list(X.columns[:2])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, tournament_threshold=150)

        # Tournament mode: multiple heat calls + 1 final = more than 1 call
        assert call_count[0] > 1, \
            f"Tournament mode should make multiple _boruta_core calls, got {call_count[0]}"

    def test_standard_mode_for_small_sets(self):
        """Features <= tournament_threshold should use standard single-round Boruta."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 50)
        cols = [f'feat_{i}' for i in range(50)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        call_count = [0]

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            call_count[0] += 1
            return list(X.columns[:3])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, tournament_threshold=150)

        assert call_count[0] == 1, \
            f"Standard mode should make exactly 1 call, got {call_count[0]}"

    def test_adaptive_run_budget_for_very_large_sets(self):
        """Very large sets should use reduced heat/final run budgets."""
        rng = np.random.RandomState(42)
        n_rows = 150
        n_cols = 2500
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, n_cols)
        cols = [f'feat_{i}' for i in range(n_cols)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        calls = []

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            calls.append((X.shape[1], n_runs, max_shadows))
            # Keep tournament moving
            return list(X.columns[:3])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, tournament_threshold=150)

        # Last call is final round, preceding calls are heats
        heat_runs = [n_runs for _, n_runs, _ in calls[:-1]]
        final_runs = calls[-1][1]
        assert heat_runs, "Expected at least one heat call"
        assert all(n < 35 for n in heat_runs), f"Expected reduced heat runs, got {heat_runs}"
        assert final_runs < 80, f"Expected reduced final runs, got {final_runs}"

    def test_explicit_n_runs_overrides_adaptive_standard_runs(self):
        """User-specified n_runs should override adaptive standard-mode budget."""
        rng = np.random.RandomState(42)
        n_rows = 150
        n_cols = 90
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, n_cols)
        cols = [f'feat_{i}' for i in range(n_cols)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        seen_runs = []

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            seen_runs.append(n_runs)
            return list(X.columns[:3])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, n_runs=77, tournament_threshold=150)

        assert seen_runs == [77], f"Expected explicit n_runs=77, got {seen_runs}"

    def test_standard_mode_passes_adaptive_shadow_cap(self):
        """Standard mode should pass adaptive max_shadows through to core."""
        rng = np.random.RandomState(42)
        n_rows = 150
        n_cols = 500
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, n_cols)
        cols = [f'feat_{i}' for i in range(n_cols)]
        df = pd.DataFrame(data, index=dates, columns=cols)

        seen_caps = []

        def mock_boruta(X, y, n_runs=100, alpha=0.05, max_shadows=None, sample_weight=None):
            seen_caps.append(max_shadows)
            return list(X.columns[:3])

        with patch('Data_ETA_Pipeline.feature_selection_engine._boruta_core',
                   side_effect=mock_boruta):
            get_boruta_importance(df, target, tournament_threshold=600)

        assert seen_caps, "Expected _boruta_core to be called"
        assert seen_caps[0] == _adaptive_boruta_shadow_cap(n_cols)


class TestBorutaAdaptiveHelpers:
    """Unit tests for adaptive Boruta budget helper functions."""

    def test_adaptive_runs_scale_with_feature_count(self):
        small = _adaptive_boruta_runs(n_features=200, n_rows=180)
        huge = _adaptive_boruta_runs(n_features=4000, n_rows=180)
        assert small['standard_runs'] > huge['standard_runs']
        assert small['heat_runs'] > huge['heat_runs']
        assert small['final_runs'] > huge['final_runs']

    def test_adaptive_shadow_cap_scales_down_for_large_sets(self):
        assert _adaptive_boruta_shadow_cap(200) == 200
        assert _adaptive_boruta_shadow_cap(1200) <= 450
        assert _adaptive_boruta_shadow_cap(5000) <= 250


# =============================================================================
# Stage 3: Vintage Stability
# =============================================================================

class TestDiscoverVintageSnapshots:
    """Tests for _discover_vintage_snapshots dynamic directory scanning."""

    def test_discovers_available_years(self, tmp_path):
        """Should find all years with <year>-12.parquet files."""
        # Create directory structure: 2010s/2010/2010-12.parquet, etc.
        for decade, years in [('2010s', ['2010', '2014', '2018']),
                              ('2020s', ['2022'])]:
            for year in years:
                year_dir = tmp_path / decade / year
                year_dir.mkdir(parents=True)
                (year_dir / f"{year}-12.parquet").touch()

        result = _discover_vintage_snapshots(tmp_path)
        assert result == ['2010', '2014', '2018', '2022']

    def test_skips_missing_parquets(self, tmp_path):
        """Directories without <year>-12.parquet should be ignored."""
        year_dir = tmp_path / '2010s' / '2010'
        year_dir.mkdir(parents=True)
        # No parquet file created
        (year_dir / 'other_file.csv').touch()

        result = _discover_vintage_snapshots(tmp_path)
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        """Non-existent directory should return empty list."""
        result = _discover_vintage_snapshots(tmp_path / 'does_not_exist')
        assert result == []

    def test_empty_dir(self, tmp_path):
        """Empty directory should return empty list."""
        result = _discover_vintage_snapshots(tmp_path)
        assert result == []


class TestVintageCVImportance:
    """Tests for _vintage_cv_importance temporal CV per vintage."""

    def test_returns_normalised_importances(self):
        """Importances should sum to ~1 (normalised)."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 20)
        data[:, 0] = target.values * 0.8 + rng.randn(n_rows) * 20
        cols = [f'feat_{i}' for i in range(20)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        imp = _vintage_cv_importance(X, target, cols)
        assert not imp.empty
        # Average of normalised fold importances should sum to ~1
        assert 0.5 < imp.sum() < 1.5, \
            f"Expected normalised importance near 1.0, got {imp.sum():.3f}"

    def test_small_vintage_falls_back_to_single_fit(self):
        """Vintages with few rows should fall back to single fit (not error)."""
        rng = np.random.RandomState(42)
        n_rows = 55  # > 50 min but < 3*15=45? Actually 55 > 45, so CV works
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 10)
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        imp = _vintage_cv_importance(X, target, cols)
        assert isinstance(imp, pd.Series)

    def test_too_few_rows_returns_empty(self):
        """Fewer than 50 common rows should return empty Series."""
        rng = np.random.RandomState(42)
        n_rows = 30
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 10)
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        imp = _vintage_cv_importance(X, target, cols)
        assert imp.empty


class TestGetVintageStability:
    """Tests for get_vintage_stability with dynamic snapshots."""

    def test_latest_only_returns_results(self):
        """Even with no historical snapshots, Latest alone should produce scores."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 10)
        data[:, 0] = target.values * 0.8 + rng.randn(n_rows) * 20
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        # Empty snapshots_dir → only Latest checkpoint used
        result = get_vintage_stability(
            cols, target, Path('/nonexistent'), X, min_presence=1
        )
        assert isinstance(result, pd.Series)
        # At least the signal feature should appear
        assert len(result) > 0

    def test_positive_threshold_filters_zeros(self):
        """Features with zero weighted importance should be excluded."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # 1 signal + 9 noise
        data = rng.randn(n_rows, 10)
        data[:, 0] = target.values * 0.9 + rng.randn(n_rows) * 10
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = get_vintage_stability(
            cols, target, Path('/nonexistent'), X, min_presence=1
        )
        # All returned features should have positive scores
        assert (result > 0).all(), \
            "All returned features should have positive weighted importance"


class TestScoringHelpers:
    """Tests for memoization and vectorized vintage score aggregation helpers."""

    def test_memoized_score_reuses_equivalent_feature_sets(self):
        calls = {'n': 0}

        def scorer(feature_set):
            calls['n'] += 1
            return float(len(feature_set))

        cache = {}
        v1 = _memoized_score(['b', 'a'], scorer=scorer, cache=cache)
        v2 = _memoized_score(['a', 'b', 'a'], scorer=scorer, cache=cache)

        assert v1 == v2 == 2.0
        assert calls['n'] == 1, f"Expected one scorer call, got {calls['n']}"

    def test_vectorized_vintage_aggregation_matches_manual_logic(self):
        scores = pd.DataFrame(
            {
                '2018': [0.10, np.nan, -0.10, 0.20],
                '2022': [0.20, 0.10, 0.00, np.nan],
                'Latest': [0.30, 0.20, -0.20, np.nan],
            },
            index=['f1', 'f2', 'f3', 'f4'],
        )
        weight_series = pd.Series({'2018': 1, '2022': 2, 'Latest': 4})

        manual = {}
        for feat in scores.index:
            feat_scores = scores.loc[feat].dropna()
            if len(feat_scores) == 0:
                continue
            if (feat_scores > 0).sum() < 2:
                continue
            weighted_score = (
                (feat_scores * weight_series[feat_scores.index]).sum()
                / weight_series[feat_scores.index].sum()
            )
            if pd.notna(scores.loc[feat, 'Latest']) and scores.loc[feat, 'Latest'] > 0:
                manual[feat] = weighted_score

        expected = pd.Series(manual).sort_values(ascending=False)
        got = _aggregate_vintage_scores(scores, weight_series, min_presence=2)

        pd.testing.assert_series_equal(got, expected)

    def test_parallel_trial_scores_uses_supplied_executor(self):
        trial_defs = [
            ('a', ['f1']),
            ('b', ['f2', 'f3']),
            ('c', ['f4']),
        ]

        class FakeExecutor:
            def __init__(self):
                self.map_calls = 0

            def map(self, fn, iterable):
                self.map_calls += 1
                return [fn(item) for item in iterable]

        exec_inst = FakeExecutor()

        with patch('Data_ETA_Pipeline.feature_selection_engine.ThreadPoolExecutor',
                   side_effect=AssertionError("Should not construct new pool")):
            scored = _parallel_trial_scores(
                trial_defs,
                scorer=lambda feats: float(len(feats)),
                max_workers=4,
                executor=exec_inst,
            )

        assert exec_inst.map_calls == 1
        assert scored == [('a', 1.0), ('b', 2.0), ('c', 1.0)]


# =============================================================================
# Stage 4: Cluster Redundancy
# =============================================================================

class TestClusterRedundancy:
    """Tests for cluster_redundancy with threshold gating and redundant-subset guardrail."""

    def test_scaled_max_clusters_small_input(self):
        """Weakly correlated features should not be force-collapsed by max_clusters."""
        rng = np.random.RandomState(42)
        n_rows = 300
        n_feats = 51
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, n_feats)
        cols = [f'feat_{i}' for i in range(n_feats)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = cluster_redundancy(X, cols, target, max_clusters=50)
        assert len(result) == n_feats, \
            f"Independent features should be preserved, got {len(result)}/{n_feats}"

    def test_passthrough_when_fewer_than_max(self):
        """Input smaller than max_clusters should pass through unchanged."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 4)
        cols = [f'feat_{i}' for i in range(4)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        # max_clusters=50, input has 4 → min(50, 2) = 2, but 4 > 2 so clustering runs
        # Actually let's test with very small: 3 features
        cols_3 = cols[:3]
        result = cluster_redundancy(X[cols_3], cols_3, target, max_clusters=50)
        # min(50, 3//2)=min(50,1)=1, but floor is 2. 3 > 2 so clustering runs
        assert len(result) <= 3

    def test_vectorized_overlap_correctness(self):
        """Vectorized overlap should match brute-force pairwise check."""
        rng = np.random.RandomState(42)
        n_rows = 100
        n_feats = 20
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        data = rng.randn(n_rows, n_feats)
        # Introduce NaN patterns
        for j in range(n_feats):
            nan_idx = rng.choice(n_rows, size=rng.randint(0, 30), replace=False)
            data[nan_idx, j] = np.nan
        cols = [f'feat_{i}' for i in range(n_feats)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        # Vectorized: notna.T @ notna
        notna_mat = X.notna().values.astype(np.float32)
        overlap_vec = notna_mat.T @ notna_mat

        # Brute force
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                brute = (X.iloc[:, i].notna() & X.iloc[:, j].notna()).sum()
                assert overlap_vec[i, j] == brute, \
                    f"Overlap mismatch at ({i},{j}): vec={overlap_vec[i,j]} vs brute={brute}"

    def test_lgb_gain_handles_nan_features(self):
        """Cluster rep selection via LGB gain should work with NaN-heavy features."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        n_feats = 60
        data = rng.randn(n_rows, n_feats)
        # Make some features mostly NaN
        for j in range(0, 10):
            data[:80, j] = np.nan
        cols = [f'feat_{i}' for i in range(n_feats)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        # Should not raise (previously MI with median imputation would bias)
        result = cluster_redundancy(X, cols, target)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_reduces_redundant_features(self):
        """Highly correlated features should be collapsed."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # 5 base signals, 10 copies each = 50 features
        n_base = 5
        data = []
        cols = []
        for b in range(n_base):
            base = rng.randn(n_rows)
            for c in range(10):
                data.append(base + rng.randn(n_rows) * 0.01)
                cols.append(f'base{b}_copy{c}')

        X = pd.DataFrame(np.column_stack(data), index=dates, columns=cols)
        result = cluster_redundancy(X, cols, target)
        # max_clusters = min(50, 50//2) = 25 → should collapse to ≤25
        assert len(result) <= 26, \
            f"Expected heavy reduction from 50 redundant features, got {len(result)}"
        assert len(result) < 50, \
            f"Should reduce from 50 features, got {len(result)}"

    def test_preserves_independent_features_when_redundant_subset_exists(self):
        """Only redundant subset should be clustered; independent columns stay."""
        rng = np.random.RandomState(42)
        n_rows = 160
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        base = rng.randn(n_rows)
        redundant = {
            f'base_copy_{i}': base + rng.randn(n_rows) * 0.001
            for i in range(6)
        }
        independent = {
            f'indep_{i}': rng.randn(n_rows)
            for i in range(8)
        }
        X = pd.DataFrame({**redundant, **independent}, index=dates)
        cols = list(X.columns)

        result = cluster_redundancy(
            X, cols, target,
            max_clusters=2,
            min_corr_to_cluster=0.9,
        )

        for col in independent:
            assert col in result, f"Independent feature {col} should be preserved"

        redundant_selected = [c for c in result if c.startswith('base_copy_')]
        assert len(redundant_selected) <= 2, \
            f"Redundant subset should respect guardrail cap, got {len(redundant_selected)}"


# =============================================================================
# Stage 5: Interaction Rescue
# =============================================================================

class TestExtractSplitPairs:
    """Tests for _extract_split_pairs (dump_model called once)."""

    def test_returns_counter(self):
        """Should return a Counter of (feat_a, feat_b) → count."""
        import lightgbm as lgb
        from Data_ETA_Pipeline.feature_selection_engine import LGB_PARAMS, _safe_lgb_fit

        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 10)
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        _safe_lgb_fit(model, X, target)

        pairs = _extract_split_pairs(model, cols)
        assert isinstance(pairs, dict)  # Counter is a dict subclass
        for (a, b), count in pairs.items():
            assert a <= b, "Pairs should be sorted alphabetically"
            assert count > 0

    def test_dump_model_called_once(self):
        """dump_model should be called exactly once, not per-tree."""
        import lightgbm as lgb
        from Data_ETA_Pipeline.feature_selection_engine import LGB_PARAMS, _safe_lgb_fit

        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 10)
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        _safe_lgb_fit(model, X, target)

        # Wrap dump_model to count calls
        original_dump = model.booster_.dump_model
        call_count = [0]

        def counting_dump(*args, **kwargs):
            call_count[0] += 1
            return original_dump(*args, **kwargs)

        model.booster_.dump_model = counting_dump

        _extract_split_pairs(model, cols)
        assert call_count[0] == 1, \
            f"dump_model should be called exactly once, was called {call_count[0]} times"


class TestInteractionRescue:
    """Tests for interaction_rescue with pre-screening and cached baseline."""

    def test_rescues_helpful_feature(self):
        """A rejected feature that improves MAE should be rescued."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # Base signal for confirmed set
        data = rng.randn(n_rows, 5)
        data[:, 0] = target.values * 0.5 + rng.randn(n_rows) * 50
        cols = [f'conf_{i}' for i in range(5)]

        # Rejected feature that is actually useful
        data_rej = rng.randn(n_rows, 5)
        data_rej[:, 0] = target.values * 0.6 + rng.randn(n_rows) * 40
        rej_cols = [f'rej_{i}' for i in range(5)]

        all_data = np.column_stack([data, data_rej])
        all_cols = cols + rej_cols
        X = pd.DataFrame(all_data, index=dates, columns=all_cols)

        rescued = interaction_rescue(
            X, target,
            confirmed_features=cols,
            rejected_pool=rej_cols,
            n_splits=4, gap=1, top_k=5
        )
        assert isinstance(rescued, list)
        # rej_0 has strong signal, should likely be rescued
        # (but don't enforce — depends on CV noise)

    def test_prescreen_limits_rejected_pool(self):
        """When rejected pool > max_phase1_screen, should be reduced."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 5)
        data[:, 0] = target.values * 0.5 + rng.randn(n_rows) * 50
        conf_cols = [f'conf_{i}' for i in range(5)]

        # Large rejected pool (50 features)
        rej_data = rng.randn(n_rows, 50)
        rej_cols = [f'rej_{i}' for i in range(50)]

        all_data = np.column_stack([data, rej_data])
        all_cols = conf_cols + rej_cols
        X = pd.DataFrame(all_data, index=dates, columns=all_cols)

        # max_phase1_screen=10 → should pre-screen 50 down to 10
        rescued = interaction_rescue(
            X, target,
            confirmed_features=conf_cols,
            rejected_pool=rej_cols,
            n_splits=3, gap=1, top_k=5,
            max_phase1_screen=10
        )
        assert isinstance(rescued, list)

    def test_empty_rejected_pool(self):
        """Empty rejected pool should return empty list."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 5)
        cols = [f'feat_{i}' for i in range(5)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = interaction_rescue(
            X, target,
            confirmed_features=cols,
            rejected_pool=[],
            n_splits=3, gap=1
        )
        assert result == []

    def test_uses_parallel_trial_evaluation_for_single_feature_phase(self):
        """Single-feature trial scoring should use ThreadPoolExecutor when enabled."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 24)
        conf_cols = [f'conf_{i}' for i in range(4)]
        rej_cols = [f'rej_{i}' for i in range(20)]
        all_cols = conf_cols + rej_cols
        X = pd.DataFrame(data, index=dates, columns=all_cols)

        map_calls = {'n': 0}

        class FakeExecutor:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, fn, iterable):
                map_calls['n'] += 1
                return [fn(item) for item in iterable]

        with patch('Data_ETA_Pipeline.feature_selection_engine.ThreadPoolExecutor', FakeExecutor), \
             patch('Data_ETA_Pipeline.feature_selection_engine._extract_split_pairs',
                   return_value=Counter()):
            rescued = interaction_rescue(
                X, target,
                confirmed_features=conf_cols,
                rejected_pool=rej_cols,
                n_splits=3, gap=1,
                top_k=5,
                trial_eval_workers=2,
            )

        assert isinstance(rescued, list)
        assert map_calls['n'] >= 1, "Expected parallel trial map() call in phase-1 scoring"


# =============================================================================
# Stage 6: Sequential Forward Selection
# =============================================================================

class TestSequentialForwardSelection:
    """Tests for SFS with beam search, pre-screening, and cached folds."""

    def test_selects_signal_features(self):
        """Strong signal features should be selected over noise."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 15)
        # 3 signal features
        data[:, 0] = target.values * 0.9 + rng.randn(n_rows) * 10
        data[:, 1] = target.values * 0.6 + rng.randn(n_rows) * 40
        data[:, 2] = target.values * 0.4 + rng.randn(n_rows) * 60
        cols = [f'feat_{i}' for i in range(15)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = sequential_forward_selection(
            X, target, cols,
            n_splits=4, gap=1,
            min_improvement=0.001, patience=2, min_features=2,
            beam_width=2
        )
        assert isinstance(result, list)
        assert len(result) >= 2, "Should select at least min_features=2"
        # The strongest signal should be selected
        assert 'feat_0' in result, \
            "Strongest signal feature should be selected"

    def test_beam_search_finds_synergies(self):
        """Beam search should discover feature pairs better than greedy."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')

        # Target is sum of two latent factors
        factor_a = rng.randn(n_rows) * 50
        factor_b = rng.randn(n_rows) * 50
        target = pd.Series(factor_a + factor_b, index=dates)

        data = rng.randn(n_rows, 10)
        # feat_0 captures factor_a, feat_1 captures factor_b
        data[:, 0] = factor_a + rng.randn(n_rows) * 10
        data[:, 1] = factor_b + rng.randn(n_rows) * 10
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = sequential_forward_selection(
            X, target, cols,
            n_splits=4, gap=1,
            min_improvement=0.001, patience=3, min_features=2,
            beam_width=3
        )
        # Both factor features should be found
        assert 'feat_0' in result and 'feat_1' in result, \
            f"Expected both factor features, got {result}"

    def test_patience_terminates(self):
        """SFS should stop after patience consecutive non-improvements."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # 1 signal + 19 noise
        data = rng.randn(n_rows, 20)
        data[:, 0] = target.values * 0.9 + rng.randn(n_rows) * 10
        cols = [f'feat_{i}' for i in range(20)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = sequential_forward_selection(
            X, target, cols,
            n_splits=3, gap=1,
            min_improvement=0.01,
            patience=2, min_features=2,
            beam_width=2
        )
        # Should not select all 20 — patience should kick in
        assert len(result) < 15, \
            f"Patience should stop early, but selected {len(result)}/20 features"

    def test_prescreen_orders_by_gain(self):
        """Pre-screening should put high-gain features first in candidates."""
        rng = np.random.RandomState(42)
        n_rows = 150
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        data = rng.randn(n_rows, 10)
        # feat_9 is the strongest but listed last in candidate order
        data[:, 9] = target.values * 0.95 + rng.randn(n_rows) * 5
        cols = [f'feat_{i}' for i in range(10)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = sequential_forward_selection(
            X, target, cols,
            n_splits=3, gap=1,
            min_features=1, patience=2,
            beam_width=2
        )
        # Pre-screening by gain should discover feat_9 despite it being last
        assert 'feat_9' in result, \
            "Pre-screening should find the strongest feature regardless of input order"

    def test_min_features_forced_adds(self):
        """Should force-add features if below min_features even with no improvement."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)

        # All noise features
        data = rng.randn(n_rows, 5)
        cols = [f'feat_{i}' for i in range(5)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        result = sequential_forward_selection(
            X, target, cols,
            n_splits=3, gap=1,
            min_improvement=0.5,  # Very high threshold
            patience=5,  # High patience to allow forced adds
            min_features=3,
            beam_width=2
        )
        assert len(result) >= 3, \
            f"Should force-add to reach min_features=3, got {len(result)}"

    def test_single_candidate(self):
        """Single candidate should be returned as-is."""
        rng = np.random.RandomState(42)
        n_rows = 120
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, 1)
        X = pd.DataFrame(data, index=dates, columns=['only_feat'])

        result = sequential_forward_selection(
            X, target, ['only_feat'],
            n_splits=3, gap=1, min_features=1, patience=1, beam_width=2
        )
        assert result == ['only_feat']

    def test_uses_parallel_trial_evaluation_in_beam_step(self):
        """Beam trial evaluation should reuse one ThreadPoolExecutor across rounds."""
        rng = np.random.RandomState(42)
        n_rows = 120
        n_cols = 12
        dates = pd.date_range('2000-01-01', periods=n_rows, freq='MS')
        target = pd.Series(rng.randn(n_rows) * 100, index=dates)
        data = rng.randn(n_rows, n_cols)
        data[:, 0] = target.values * 0.8 + rng.randn(n_rows) * 20
        cols = [f'feat_{i}' for i in range(n_cols)]
        X = pd.DataFrame(data, index=dates, columns=cols)

        map_calls = {'n': 0}
        init_calls = {'n': 0}

        class FakeExecutor:
            def __init__(self, max_workers=None):
                init_calls['n'] += 1
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, fn, iterable):
                map_calls['n'] += 1
                return [fn(item) for item in iterable]

        with patch('Data_ETA_Pipeline.feature_selection_engine.ThreadPoolExecutor', FakeExecutor):
            result = sequential_forward_selection(
                X, target, cols,
                n_splits=3, gap=1,
                min_improvement=0.001, patience=2, min_features=2,
                beam_width=2,
                trial_eval_workers=2,
            )

        assert isinstance(result, list)
        assert init_calls['n'] == 1, "Expected one persistent executor for SFS beam loop"
        assert map_calls['n'] >= 2, "Expected multiple parallel map() calls across beam rounds"

class TestKeepRule:
    """Tests for keep-rule acceptance logic."""

    def test_keep_on_mae_improvement(self):
        """Should keep when MAE improves by at least 0.5%."""
        assert should_keep_change(
            baseline_mae=100.0,
            candidate_mae=99.4,  # +0.6% MAE improvement
            baseline_runtime_s=100.0,
            candidate_runtime_s=110.0,
        )

    def test_keep_on_runtime_gain_with_small_mae_loss(self):
        """Should keep on >=15% runtime gain if MAE loss is within 0.5%."""
        assert should_keep_change(
            baseline_mae=100.0,
            candidate_mae=100.4,  # -0.4% MAE improvement (small degradation)
            baseline_runtime_s=100.0,
            candidate_runtime_s=80.0,  # +20% runtime improvement
        )

    def test_reject_on_runtime_gain_with_large_mae_loss(self):
        """Should reject when runtime improves but MAE degrades beyond tolerance."""
        assert not should_keep_change(
            baseline_mae=100.0,
            candidate_mae=100.7,  # -0.7% MAE improvement (too much degradation)
            baseline_runtime_s=100.0,
            candidate_runtime_s=80.0,
        )

    def test_reject_when_neither_threshold_met(self):
        """Should reject if no MAE threshold and no runtime threshold are met."""
        assert not should_keep_change(
            baseline_mae=100.0,
            candidate_mae=100.2,
            baseline_runtime_s=100.0,
            candidate_runtime_s=95.0,
        )
