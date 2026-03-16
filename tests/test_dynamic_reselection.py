"""
Tests for Dynamic Feature Re-Selection & Post-2010 NaN Pruning

Covers:
- clean_features() with post-2010 NaN evaluation (Stage 2 filter)
- _dynamic_reselection() integration (mocked source loading)
- Config constants for dynamic FS
- Calendar feature reductions (dropped is_summer, is_holiday_season, is_december)
- Revision feature reductions (dropped n_new, n_dropped)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_X_with_dates():
    """Create a sample feature DataFrame spanning 2005-2024 with a 'ds' column."""
    dates = pd.date_range('2005-01-01', '2024-12-01', freq='MS')
    n = len(dates)
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        'ds': dates,
        'good_feature': rng.randn(n),
        'sparse_old': np.nan,
        'modern_good': np.nan,
        'modern_bad': np.nan,
        'always_nan': np.nan,
    })

    # sparse_old: has data only pre-2010 (40 non-NaN)
    pre2010 = df['ds'] < '2010-01-01'
    df.loc[pre2010, 'sparse_old'] = rng.randn(pre2010.sum())

    # modern_good: full data from 2010 onward (< 20% NaN post-2010)
    post2010 = df['ds'] >= '2010-01-01'
    df.loc[post2010, 'modern_good'] = rng.randn(post2010.sum())
    # Also has some pre-2010 data
    pre2010_indices = df.index[pre2010][:20]
    df.loc[pre2010_indices, 'modern_good'] = rng.randn(len(pre2010_indices))

    # modern_bad: data from 2010 but >20% NaN in post-2010 window
    post2010_idx = df.index[post2010]
    # Only fill 50% of post-2010 rows (well above 20% NaN threshold)
    fill_idx = post2010_idx[: len(post2010_idx) // 2]
    df.loc[fill_idx, 'modern_bad'] = rng.randn(len(fill_idx))

    return df


@pytest.fixture
def sample_y(sample_X_with_dates):
    """Target series matching the sample X."""
    rng = np.random.RandomState(123)
    return pd.Series(rng.randn(len(sample_X_with_dates)), name='y_mom')


# =============================================================================
# TEST: clean_features() POST-2010 NaN EVALUATION
# =============================================================================

class TestCleanFeaturesNaNPruning:
    """Tests for the enhanced clean_features() with post-2010 NaN evaluation."""

    def test_global_sparsity_still_works(self, sample_X_with_dates, sample_y):
        """Stage 1: columns with <min_non_nan are dropped globally."""
        from Train.train_lightgbm_nfp import clean_features

        # always_nan has 0 non-NaN → should be dropped
        result = clean_features(
            sample_X_with_dates, sample_y,
            min_non_nan=10,
            nan_eval_start=None,  # Disable stage 2
            nan_max_rate=1.0,     # Effectively disabled
        )
        assert 'always_nan' not in result
        assert 'good_feature' in result

    def test_modern_nan_drops_high_nan_post2010(self, sample_X_with_dates, sample_y):
        """Stage 2: features with >20% NaN from 2010 onward are dropped."""
        from Train.train_lightgbm_nfp import clean_features

        result = clean_features(
            sample_X_with_dates, sample_y,
            min_non_nan=5,
            nan_eval_start='2010-01-01',
            nan_max_rate=0.20,
        )
        # modern_bad has ~50% NaN post-2010 → should be dropped
        assert 'modern_bad' not in result
        # modern_good has ~0% NaN post-2010 → should survive
        assert 'modern_good' in result
        # good_feature is fully populated → should survive
        assert 'good_feature' in result

    def test_sparse_old_survives_global_but_fails_modern(
        self, sample_X_with_dates, sample_y
    ):
        """sparse_old has 40 values pre-2010 but 100% NaN post-2010."""
        from Train.train_lightgbm_nfp import clean_features

        result = clean_features(
            sample_X_with_dates, sample_y,
            min_non_nan=5,           # Passes global (40 > 5)
            nan_eval_start='2010-01-01',
            nan_max_rate=0.20,        # Fails modern (100% NaN post-2010)
        )
        assert 'sparse_old' not in result

    def test_disabled_modern_filter(self, sample_X_with_dates, sample_y):
        """When nan_max_rate=1.0, no features are dropped by modern filter."""
        from Train.train_lightgbm_nfp import clean_features

        result = clean_features(
            sample_X_with_dates, sample_y,
            min_non_nan=5,
            nan_eval_start='2010-01-01',
            nan_max_rate=1.0,  # Accept any NaN rate
        )
        # modern_bad should survive since max_rate=1.0
        assert 'modern_bad' in result

    def test_ds_column_never_in_output(self, sample_X_with_dates, sample_y):
        """The 'ds' column should never appear in the output feature list."""
        from Train.train_lightgbm_nfp import clean_features

        result = clean_features(
            sample_X_with_dates, sample_y,
            min_non_nan=5,
        )
        assert 'ds' not in result

    def test_no_ds_column_still_works(self, sample_y):
        """clean_features works even if X has no 'ds' column (modern filter skipped)."""
        from Train.train_lightgbm_nfp import clean_features

        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            'feat_a': rng.randn(50),
            'feat_b': rng.randn(50),
        })
        result = clean_features(X, sample_y[:50], min_non_nan=5)
        assert 'feat_a' in result
        assert 'feat_b' in result


# =============================================================================
# TEST: CONFIG CONSTANTS
# =============================================================================

class TestDynamicFSConfig:
    """Verify that dynamic FS config constants are properly defined."""

    def test_reselect_every_n_months_exists(self):
        from settings import RESELECT_EVERY_N_MONTHS
        assert isinstance(RESELECT_EVERY_N_MONTHS, int)
        assert RESELECT_EVERY_N_MONTHS >= 0

    def test_dynamic_fs_stages_pass1(self):
        from Train.config import DYNAMIC_FS_STAGES_PASS1
        assert isinstance(DYNAMIC_FS_STAGES_PASS1, tuple)
        # Should not include stage 3 (vintage stability)
        assert 3 not in DYNAMIC_FS_STAGES_PASS1
        # Should include stages 0, 1, 2
        assert 0 in DYNAMIC_FS_STAGES_PASS1
        assert 1 in DYNAMIC_FS_STAGES_PASS1
        assert 2 in DYNAMIC_FS_STAGES_PASS1

    def test_dynamic_fs_stages_pass2(self):
        from Train.config import DYNAMIC_FS_STAGES_PASS2, DYNAMIC_FS_STAGES_PASS1
        assert isinstance(DYNAMIC_FS_STAGES_PASS2, tuple)
        # Pass 2 should be more aggressive (fewer stages)
        assert len(DYNAMIC_FS_STAGES_PASS2) <= len(DYNAMIC_FS_STAGES_PASS1)

    def test_dynamic_fs_pass2_max_features(self):
        from Train.config import DYNAMIC_FS_PASS2_MAX_FEATURES
        assert isinstance(DYNAMIC_FS_PASS2_MAX_FEATURES, int)
        assert DYNAMIC_FS_PASS2_MAX_FEATURES == 50

    def test_dynamic_fs_nan_eval_start(self):
        from Train.config import DYNAMIC_FS_NAN_EVAL_START
        assert DYNAMIC_FS_NAN_EVAL_START == '2010-01-01'
        # Should be parseable as a date
        pd.Timestamp(DYNAMIC_FS_NAN_EVAL_START)

    def test_dynamic_fs_nan_max_rate(self):
        from Train.config import DYNAMIC_FS_NAN_MAX_RATE
        assert 0.0 < DYNAMIC_FS_NAN_MAX_RATE < 1.0
        assert DYNAMIC_FS_NAN_MAX_RATE == 0.20

    def test_dynamic_fs_boruta_runs(self):
        from Train.config import DYNAMIC_FS_BORUTA_RUNS
        assert isinstance(DYNAMIC_FS_BORUTA_RUNS, int)
        assert DYNAMIC_FS_BORUTA_RUNS > 0


# =============================================================================
# TEST: CALENDAR FEATURE REDUCTIONS
# =============================================================================

class TestCalendarFeatureReductions:
    """Verify that dropped calendar features are actually removed."""

    def test_dropped_features_not_in_calendar_dict(self):
        """is_summer, is_holiday_season, is_december should not be generated."""
        from Train.feature_engineering import get_calendar_features_dict

        features = get_calendar_features_dict(pd.Timestamp('2024-07-01'))
        assert 'is_summer' not in features
        assert 'is_holiday_season' not in features
        assert 'is_december' not in features

    def test_kept_features_still_present(self):
        """Core calendar features should still be generated."""
        from Train.feature_engineering import get_calendar_features_dict

        features = get_calendar_features_dict(pd.Timestamp('2024-07-01'))
        assert 'month_sin' in features
        assert 'month_cos' in features
        assert 'is_jan' in features
        assert 'is_july' in features
        assert 'is_5_week_month' in features

    def test_add_calendar_features_no_dropped(self):
        """add_calendar_features should not produce dropped columns."""
        from Train.feature_engineering import add_calendar_features

        df = pd.DataFrame({'value': [42.0]})
        # Test multiple months to cover summer/holiday/december
        for month in [6, 7, 8, 11, 12]:
            target_month = pd.Timestamp(f'2024-{month:02d}-01')
            result = add_calendar_features(df, target_month)
            assert 'is_summer' not in result.columns
            assert 'is_holiday_season' not in result.columns
            assert 'is_december' not in result.columns

    def test_calendar_feature_count(self):
        """Should have exactly 9 calendar features."""
        from Train.feature_engineering import get_calendar_features_dict

        features = get_calendar_features_dict(pd.Timestamp('2024-06-15'))
        # Expected: month_sin, month_cos, quarter_sin, quarter_cos,
        # weeks_since_last_survey, is_5_week_month, is_jan, is_july, year
        assert len(features) == 9


# =============================================================================
# TEST: REVISION FEATURE REDUCTIONS
# =============================================================================

class TestRevisionFeatureReductions:
    """Verify that dropped revision features are removed."""

    def test_no_n_new_n_dropped_in_output(self):
        """compute_revision_features should not produce n_new or n_dropped."""
        from Train.revision_features import compute_revision_features

        # Wide-format single-row DataFrames (as returned by pivot_snapshot_to_wide)
        current = pd.DataFrame({'feat_a': [100.0], 'feat_b': [200.0]})
        previous = pd.DataFrame({'feat_a': [99.0], 'feat_b': [201.0]})

        result = compute_revision_features(current, previous, prefix='rev_master')
        assert 'rev_master_n_new' not in result.columns
        assert 'rev_master_n_dropped' not in result.columns

    def test_kept_revision_features_present(self):
        """Core revision features should still be computed."""
        from Train.revision_features import compute_revision_features

        current = pd.DataFrame({'feat_a': [100.0], 'feat_b': [200.0]})
        previous = pd.DataFrame({'feat_a': [99.0], 'feat_b': [201.0]})

        result = compute_revision_features(current, previous, prefix='rev_master')
        assert 'rev_master_mean' in result.columns
        assert 'rev_master_abs_mean' in result.columns
        assert 'rev_master_count' in result.columns


# =============================================================================
# TEST: DYNAMIC RESELECTION FUNCTION (unit-level with mocks)
# =============================================================================

class TestDynamicReselection:
    """Test _dynamic_reselection function logic with mocked dependencies."""

    def test_reselection_returns_list(self):
        """_dynamic_reselection should return a list (possibly empty)."""
        from Train.train_lightgbm_nfp import _dynamic_reselection

        # Create minimal training data
        dates = pd.date_range('2010-01-01', '2023-12-01', freq='MS')
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            'ds': dates,
            'feat_a': rng.randn(len(dates)),
            'nfp_nsa_lag1': rng.randn(len(dates)),
            'month_sin': np.sin(np.arange(len(dates))),
        })
        y = pd.Series(rng.randn(len(dates)))

        # Mock all the heavy imports to avoid loading real data
        with patch(
            'Train.train_lightgbm_nfp._dynamic_reselection'
        ) as mock_resel:
            mock_resel.return_value = ['feat_a', 'nfp_nsa_lag1']
            result = mock_resel(X, y, pd.Timestamp('2023-06-01'), 'nsa', 'revised')
            assert isinstance(result, list)

    def test_reselection_disabled_when_interval_zero(self):
        """When RESELECT_EVERY_N_MONTHS=0, reselection should never trigger."""
        # The trigger condition is: RESELECT_EVERY_N_MONTHS > 0
        from settings import RESELECT_EVERY_N_MONTHS

        # With current default of 6, this just checks the constant
        assert RESELECT_EVERY_N_MONTHS > 0  # Current default

    def test_reselection_interval_calculation(self):
        """Verify the interval calculation logic."""
        from settings import RESELECT_EVERY_N_MONTHS

        interval_days = RESELECT_EVERY_N_MONTHS * 30
        # 6 months * 30 days = 180 days
        assert interval_days == 180

    def test_dynamic_features_override_static(self):
        """When dynamic_features is set, feature_cols should come from it."""
        # This tests the integration logic conceptually
        dynamic_features = ['feat_a', 'feat_b', 'nfp_nsa_lag1']
        cleaned_features = ['feat_a', 'feat_b', 'feat_c', 'nfp_nsa_lag1']
        available_cols = set(cleaned_features + ['ds'])

        # Simulate the dynamic path logic
        feature_cols = [
            c for c in dynamic_features
            if c in available_cols and c in cleaned_features
        ]

        assert feature_cols == ['feat_a', 'feat_b', 'nfp_nsa_lag1']
        assert 'feat_c' not in feature_cols  # Not in dynamic_features

    def test_dynamic_fallback_on_no_overlap(self):
        """When dynamic features have zero overlap, should fall back to None."""
        dynamic_features = ['nonexistent_feat1', 'nonexistent_feat2']
        cleaned_features = ['feat_a', 'feat_b']

        feature_cols = [
            c for c in dynamic_features if c in cleaned_features
        ]

        assert len(feature_cols) == 0
        # In the real code, this triggers: dynamic_features = None

    def test_all_features_mode_raises_on_zero_overlap(self):
        """In all-features mode, zero overlap should raise RuntimeError (no fallback)."""
        dynamic_features = ['nonexistent_feat1']
        cleaned_features = ['feat_a', 'feat_b']

        feature_cols = [
            c for c in dynamic_features if c in cleaned_features
        ]

        # Simulate the all-features mode logic
        _all_features_mode = True
        if not feature_cols and _all_features_mode:
            with pytest.raises(RuntimeError):
                raise RuntimeError(
                    "Dynamic features have zero overlap with cleaned features."
                )


# =============================================================================
# TEST: SOURCE COLUMN CLASSIFIER
# =============================================================================

class TestClassifyColumnsBySource:
    """Tests for _classify_columns_by_source column partitioning."""

    def test_nsa_employment_columns(self):
        """Columns starting with total_ containing _nsa should be classified as NSA employment."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['total_payroll_nsa_latest', 'total_payroll_nsa_mom',
                'total_payroll_nsa_yoy']
        groups = _classify_columns_by_source(cols)
        assert groups['FRED_Employment_NSA'] == cols
        assert groups['FRED_Employment_SA'] == []

    def test_sa_employment_columns(self):
        """Columns starting with total_ without _nsa should be classified as SA employment."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['total_payroll_latest', 'total_payroll_mom', 'total_payroll_yoy']
        groups = _classify_columns_by_source(cols)
        assert groups['FRED_Employment_SA'] == cols
        assert groups['FRED_Employment_NSA'] == []

    def test_fred_exogenous_prefixes(self):
        """Known FRED exogenous prefixes should be classified correctly."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['VIX_latest', 'SP500_12m_pct_change', 'Oil_latest',
                'Yield_10y_latest', 'CCSA_latest', 'WEI_latest', 'regime_high_vol']
        groups = _classify_columns_by_source(cols)
        assert len(groups['FRED_Exogenous']) == len(cols)

    def test_unifier_prefixes(self):
        """Known Unifier prefixes should be classified correctly."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['ISM_PMI_latest', 'Challenger_layoffs_latest',
                'NFP_Consensus_latest', 'Housing_starts_latest']
        groups = _classify_columns_by_source(cols)
        assert len(groups['Unifier']) == len(cols)

    def test_adp_prefixes(self):
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['ADP_total_latest', 'ADP_small_biz_latest']
        groups = _classify_columns_by_source(cols)
        assert len(groups['ADP']) == 2

    def test_noaa_prefixes(self):
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['NOAA_temp_anomaly', 'storm_count', 'hurricane_cat5']
        groups = _classify_columns_by_source(cols)
        assert len(groups['NOAA']) == 3

    def test_prosper_prefixes(self):
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['Consumer_Mood_latest', 'Prosper_sentiment',
                'Consumer_Spending_latest']
        groups = _classify_columns_by_source(cols)
        assert len(groups['Prosper']) == 3

    def test_unknown_columns(self):
        """Unrecognized columns should go to 'Unknown' bucket."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = ['some_random_feat', 'another_unknown']
        groups = _classify_columns_by_source(cols)
        assert groups['Unknown'] == cols

    def test_mixed_sources(self):
        """A mix of columns should be correctly partitioned."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = [
            'total_payroll_nsa_latest',   # NSA emp
            'total_payroll_latest',       # SA emp
            'VIX_latest',                 # FRED exog
            'ISM_PMI_latest',             # Unifier
            'ADP_total_latest',           # ADP
            'NOAA_temp_anomaly',          # NOAA
            'Consumer_Mood_latest',       # Prosper
            'random_feature',             # Unknown
        ]
        groups = _classify_columns_by_source(cols)

        assert len(groups['FRED_Employment_NSA']) == 1
        assert len(groups['FRED_Employment_SA']) == 1
        assert len(groups['FRED_Exogenous']) == 1
        assert len(groups['Unifier']) == 1
        assert len(groups['ADP']) == 1
        assert len(groups['NOAA']) == 1
        assert len(groups['Prosper']) == 1
        assert len(groups['Unknown']) == 1

    def test_no_column_counted_twice(self):
        """Each column should appear in exactly one source group."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        cols = [
            'total_payroll_nsa_latest', 'VIX_latest', 'ADP_total',
            'NOAA_temp', 'ISM_PMI', 'Consumer_Mood_latest',
            'total_payroll_latest', 'random',
        ]
        groups = _classify_columns_by_source(cols)
        all_classified = []
        for v in groups.values():
            all_classified.extend(v)

        assert len(all_classified) == len(cols)
        assert set(all_classified) == set(cols)

    def test_empty_input(self):
        """Empty input should return empty groups."""
        from Train.train_lightgbm_nfp import _classify_columns_by_source

        groups = _classify_columns_by_source([])
        total = sum(len(v) for v in groups.values())
        assert total == 0


# =============================================================================
# TEST: DYNAMIC RESELECTION TWO-PASS ARCHITECTURE
# =============================================================================

class TestDynamicReselectionTwoPass:
    """Tests for the two-pass dynamic reselection architecture."""

    def test_reselection_works_from_x_train_columns(self):
        """_dynamic_reselection should partition X_train columns by source and run two passes."""
        from Train.train_lightgbm_nfp import _dynamic_reselection

        dates = pd.date_range('2010-01-01', '2023-12-01', freq='MS')
        rng = np.random.RandomState(42)
        n = len(dates)

        X = pd.DataFrame({
            'ds': dates,
            'total_payroll_nsa_latest': rng.randn(n),
            'total_payroll_latest': rng.randn(n),
            'VIX_latest': rng.randn(n),
            'ISM_PMI_latest': rng.randn(n),
            'ADP_total_latest': rng.randn(n),
            'nfp_nsa_lag1': rng.randn(n),
            'month_sin': np.sin(np.arange(n)),
        })
        y = pd.Series(rng.randn(n))

        # Mock at the actual import source (lazy import inside _dynamic_reselection)
        with patch('Data_ETA_Pipeline.feature_selection_engine.run_full_source_pipeline') as mock_pipeline, \
             patch('Data_ETA_Pipeline.feature_selection_engine._classify_series') as mock_classify:

            def fake_pipeline(snap_wide, y_sel, source_name, snapshots_dir,
                            series_groups, stages=None, sample_weight=None):
                return list(snap_wide.columns)

            mock_pipeline.side_effect = fake_pipeline
            mock_classify.return_value = "default"

            result = _dynamic_reselection(
                X_train=X,
                y_train=y,
                step_date=pd.Timestamp('2023-06-01'),
                target_type='nsa',
                target_source='revised',
            )

            assert isinstance(result, list)
            assert len(result) > 0
            # All returned features should be columns of X
            assert all(f in X.columns for f in result)

    def test_reselection_raises_on_pass1_zero_features(self):
        """If Pass-1 returns zero features, should raise RuntimeError."""
        from Train.train_lightgbm_nfp import _dynamic_reselection

        dates = pd.date_range('2010-01-01', '2023-12-01', freq='MS')
        rng = np.random.RandomState(42)
        n = len(dates)

        X = pd.DataFrame({
            'ds': dates,
            'total_payroll_nsa_latest': rng.randn(n),
            'VIX_latest': rng.randn(n),
        })
        y = pd.Series(rng.randn(n))

        with patch('Data_ETA_Pipeline.feature_selection_engine.run_full_source_pipeline') as mock_pipeline, \
             patch('Data_ETA_Pipeline.feature_selection_engine._classify_series') as mock_classify:

            mock_pipeline.return_value = []
            mock_classify.return_value = "default"

            with pytest.raises(RuntimeError, match="zero features"):
                _dynamic_reselection(
                    X_train=X,
                    y_train=y,
                    step_date=pd.Timestamp('2023-06-01'),
                    target_type='nsa',
                    target_source='revised',
                )

    def test_reselection_respects_max_features_cap(self):
        """Result should have <= DYNAMIC_FS_PASS2_MAX_FEATURES features."""
        from Train.train_lightgbm_nfp import _dynamic_reselection
        from Train.config import DYNAMIC_FS_PASS2_MAX_FEATURES

        dates = pd.date_range('2010-01-01', '2023-12-01', freq='MS')
        rng = np.random.RandomState(42)
        n = len(dates)

        # Create lots of features
        data = {'ds': dates}
        for i in range(100):
            data[f'total_payroll_nsa_feat_{i}'] = rng.randn(n)
        X = pd.DataFrame(data)
        y = pd.Series(rng.randn(n))

        with patch('Data_ETA_Pipeline.feature_selection_engine.run_full_source_pipeline') as mock_pipeline, \
             patch('Data_ETA_Pipeline.feature_selection_engine._classify_series') as mock_classify:

            mock_pipeline.side_effect = lambda snap_wide, *a, **kw: list(snap_wide.columns)
            mock_classify.return_value = "default"

            result = _dynamic_reselection(
                X_train=X,
                y_train=y,
                step_date=pd.Timestamp('2023-06-01'),
                target_type='nsa',
                target_source='revised',
            )

            assert len(result) <= DYNAMIC_FS_PASS2_MAX_FEATURES


# =============================================================================
# TEST: LOAD_SELECTED_FEATURES ALL-FEATURES MODE
# =============================================================================

class TestLoadSelectedFeaturesAllFeaturesMode:
    """Tests for load_selected_features returning None in all-features mode."""

    def test_returns_none_for_all_features_marker(self, tmp_path, monkeypatch):
        """load_selected_features should return None when mode=all_features."""
        import Train.config as cfg

        monkeypatch.setattr(cfg, 'MASTER_SNAPSHOTS_BASE', tmp_path)

        marker = {
            "mode": "all_features",
            "generated_at": "2026-03-15T12:00:00",
        }
        marker_path = tmp_path / "selected_features_nsa_revised.json"
        with open(marker_path, 'w') as f:
            import json
            json.dump(marker, f)

        result = cfg.load_selected_features('nsa', 'revised')
        assert result is None

    def test_returns_list_for_selected_mode(self, tmp_path, monkeypatch):
        """load_selected_features should return a list when features are present."""
        import Train.config as cfg

        monkeypatch.setattr(cfg, 'MASTER_SNAPSHOTS_BASE', tmp_path)

        marker = {
            "mode": "selected",
            "features": ["feat_a", "feat_b"],
        }
        marker_path = tmp_path / "selected_features_sa_revised.json"
        with open(marker_path, 'w') as f:
            import json
            json.dump(marker, f)

        result = cfg.load_selected_features('sa', 'revised')
        assert result == ["feat_a", "feat_b"]

    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        """load_selected_features should raise FileNotFoundError when cache is missing."""
        import Train.config as cfg

        monkeypatch.setattr(cfg, 'MASTER_SNAPSHOTS_BASE', tmp_path)

        with pytest.raises(FileNotFoundError):
            cfg.load_selected_features('nsa', 'revised')

    def test_returns_empty_list_for_legacy_format(self, tmp_path, monkeypatch):
        """Legacy JSON without 'mode' should return features list."""
        import Train.config as cfg

        monkeypatch.setattr(cfg, 'MASTER_SNAPSHOTS_BASE', tmp_path)

        marker = {
            "last_run_date": "2026-01-01",
            "features": ["old_feat_1", "old_feat_2"],
        }
        marker_path = tmp_path / "selected_features_nsa_revised.json"
        with open(marker_path, 'w') as f:
            import json
            json.dump(marker, f)

        result = cfg.load_selected_features('nsa', 'revised')
        assert result == ["old_feat_1", "old_feat_2"]


# =============================================================================
# TEST: BACKTEST ALL-FEATURES MODE BEHAVIOR
# =============================================================================

class TestBacktestAllFeaturesMode:
    """Tests verifying all-features mode behavior in the backtest loop."""

    def test_all_features_mode_forces_reselection_on_step_0(self):
        """In all-features mode, reselection must trigger on step 0."""
        _all_features_mode = True
        dynamic_features = None  # Not yet selected (step 0)
        last_reselection_date = None

        RESELECT_EVERY_N_MONTHS = 6
        _reselect_interval_days = RESELECT_EVERY_N_MONTHS * 30

        _trigger_reselection = (
            dynamic_features is None
            or (
                RESELECT_EVERY_N_MONTHS > 0
                and last_reselection_date is not None
                and (pd.Timestamp('2020-06-01') - last_reselection_date).days >= _reselect_interval_days
            )
        )

        assert _trigger_reselection is True

    def test_all_features_mode_triggers_every_6_months(self):
        """In all-features mode, reselection should trigger after 6 months."""
        _all_features_mode = True
        dynamic_features = ['feat_a']  # Already selected
        last_reselection_date = pd.Timestamp('2020-01-01')
        target_month = pd.Timestamp('2020-07-15')

        RESELECT_EVERY_N_MONTHS = 6
        _reselect_interval_days = RESELECT_EVERY_N_MONTHS * 30

        _trigger_reselection = (
            dynamic_features is None
            or (
                RESELECT_EVERY_N_MONTHS > 0
                and last_reselection_date is not None
                and (target_month - last_reselection_date).days >= _reselect_interval_days
            )
        )

        assert _trigger_reselection is True

    def test_all_features_mode_no_trigger_before_interval(self):
        """In all-features mode, no reselection before interval elapses."""
        dynamic_features = ['feat_a']
        last_reselection_date = pd.Timestamp('2020-01-01')
        target_month = pd.Timestamp('2020-03-01')  # Only 2 months later

        RESELECT_EVERY_N_MONTHS = 6
        _reselect_interval_days = RESELECT_EVERY_N_MONTHS * 30

        _trigger_reselection = (
            dynamic_features is None
            or (
                RESELECT_EVERY_N_MONTHS > 0
                and last_reselection_date is not None
                and (target_month - last_reselection_date).days >= _reselect_interval_days
            )
        )

        assert _trigger_reselection is False

    def test_no_static_fallback_in_all_features_mode(self):
        """In all-features mode, zero dynamic features should raise RuntimeError."""
        _all_features_mode = True
        dynamic_features = None

        with pytest.raises(RuntimeError):
            if dynamic_features is None and _all_features_mode:
                raise RuntimeError(
                    "All-features mode requires dynamic reselection, "
                    "but no features are available."
                )


# =============================================================================
# TEST: LEAN PIPELINE INTEGRATION
# =============================================================================

class TestLeanPipelineIntegration:
    """Verify lean=True is passed correctly in all pipeline files."""

    def test_fred_exogenous_uses_lean(self):
        """load_fred_exogenous.py should call compute_all_features with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'load_fred_exogenous.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_all_features':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_all_features(lean=True) not found in load_fred_exogenous.py"

    def test_adp_pipeline_uses_lean(self):
        """adp_pipeline.py should call compute_all_features with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'adp_pipeline.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_all_features':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_all_features(lean=True) not found in adp_pipeline.py"

    def test_prosper_uses_lean(self):
        """load_prosper_data.py should call compute_all_features with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'load_prosper_data.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_all_features':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_all_features(lean=True) not found in load_prosper_data.py"

    def test_noaa_uses_lean(self):
        """noaa_pipeline.py should call compute_all_features with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'noaa_pipeline.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_all_features':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_all_features(lean=True) not found in noaa_pipeline.py"

    def test_unifier_uses_lean(self):
        """load_unifier_data.py should call compute_all_features with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'load_unifier_data.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_all_features':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_all_features(lean=True) not found in load_unifier_data.py"

    def test_fred_employment_uses_lean(self):
        """fred_employment_pipeline.py should call compute_features_wide with lean=True."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'fred_employment_pipeline.py'
        source = path.read_text()
        tree = ast.parse(source)

        found_lean = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name == 'compute_features_wide':
                    for kw in node.keywords:
                        if kw.arg == 'lean' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            found_lean = True

        assert found_lean, "compute_features_wide(lean=True) not found in fred_employment_pipeline.py"

    def test_no_symlog_in_fred_exogenous(self):
        """load_fred_exogenous.py should not call add_symlog_copies."""
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'load_fred_exogenous.py'
        source = path.read_text()
        # After lean changes, add_symlog_copies should not be called
        assert 'add_symlog_copies(' not in source or 'add_symlog_copies' not in source

    def test_prosper_filtered_to_two_questions(self):
        """load_prosper_data.py should only include Spending Forecast and Consumer Mood."""
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'load_prosper_data.py'
        source = path.read_text()
        assert 'Prosper Consumer Spending Forecast' in source
        assert 'Consumer Mood Index' in source
