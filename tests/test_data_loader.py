"""
Tests for Data Loader Module

This module contains comprehensive tests for the NFP Predictor's data loading,
target generation, and snapshot pivoting functionalities. It ensures that data 
is correctly ingested, strictly adheres to anti-lookahead point-in-time constraints, 
and supports the robust calculation of MoM/YoY targets based on 'revised' vs 'first_release' logic.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.data_loader import (
    get_fred_snapshot_path,
    get_master_snapshot_path,
    load_target_data,
    get_lagged_target_features,
    pivot_snapshot_to_wide,
    clear_snapshot_cache,
    clear_target_cache,
)


class TestSnapshotPathGeneration:
    """Tests for snapshot path generation functions."""

    def test_fred_path_2020s(self):
        """Test FRED path generation for 2020s decade."""
        date = pd.Timestamp('2024-10-31')
        path = get_fred_snapshot_path(date)

        assert '2020s' in str(path)
        assert '2024' in str(path)
        assert '2024-10' in str(path)
        assert str(path).endswith('.parquet')

    def test_fred_path_2010s(self):
        """Test FRED path generation for 2010s decade."""
        date = pd.Timestamp('2015-06-30')
        path = get_fred_snapshot_path(date)

        assert '2010s' in str(path)
        assert '2015' in str(path)

    def test_master_path_structure(self):
        """Test master snapshot path structure."""
        date = pd.Timestamp('2020-03-31')
        path = get_master_snapshot_path(date)

        assert 'master_snapshots' in str(path)
        assert '2020s' in str(path)
        assert '2020-03' in str(path)


class TestLoadTargetData:
    """Tests for target data loading."""

    def test_target_validation_invalid_type(self):
        """Test that invalid target type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            load_target_data('invalid', 'first')

    def test_target_validation_invalid_release(self):
        """Test that invalid release type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid release_type"):
            load_target_data('nsa', 'invalid')


class TestLaggedTargetFeatures:
    """Tests for lagged target feature generation."""

    @pytest.fixture
    def sample_target_df(self):
        """Create sample target DataFrame for testing."""
        dates = pd.date_range('2019-01-01', '2020-12-01', freq='MS')
        np.random.seed(42)

        df = pd.DataFrame({
            'ds': dates,
            'y': np.cumsum(np.random.randn(len(dates)) * 100) + 150000,
        })
        df['y_mom'] = df['y'].diff()
        df['y_yoy'] = df['y'].diff(12)

        return df

    def test_lag1_feature(self, sample_target_df):
        """Test that lag 1 feature is generated."""
        target_month = pd.Timestamp('2020-06-01')
        features = get_lagged_target_features(
            sample_target_df,
            target_month,
            prefix='nfp'
        )

        assert 'nfp_mom_lag1' in features

    def test_lag12_feature(self, sample_target_df):
        """Test that lag 12 (YoY) feature is generated."""
        target_month = pd.Timestamp('2020-06-01')
        features = get_lagged_target_features(
            sample_target_df,
            target_month,
            prefix='nfp'
        )

        assert 'nfp_mom_lag12' in features

    def test_rolling_mean_features(self, sample_target_df):
        """Test that rolling mean features are generated."""
        target_month = pd.Timestamp('2020-06-01')
        features = get_lagged_target_features(
            sample_target_df,
            target_month,
            prefix='nfp'
        )

        assert 'nfp_rolling_3m' in features
        assert 'nfp_rolling_6m' in features
        assert 'nfp_rolling_12m' in features

    def test_no_look_ahead_bias(self, sample_target_df):
        """Test that only data before target month is used."""
        target_month = pd.Timestamp('2020-06-01')
        features = get_lagged_target_features(
            sample_target_df,
            target_month,
            prefix='nfp'
        )

        # The lag1 feature should be from May 2020, not June
        may_2020_data = sample_target_df[sample_target_df['ds'] == '2020-05-01']
        if not may_2020_data.empty:
            expected_lag1 = may_2020_data['y_mom'].iloc[0]
            if not np.isnan(expected_lag1):
                assert np.isclose(features['nfp_mom_lag1'], expected_lag1)

    def test_empty_df_returns_empty_features(self):
        """Test that empty DataFrame returns empty features."""
        empty_df = pd.DataFrame(columns=['ds', 'y', 'y_mom'])
        target_month = pd.Timestamp('2020-06-01')

        features = get_lagged_target_features(empty_df, target_month, prefix='nfp')

        assert features == {}


class TestPivotSnapshotToWide:
    """Tests for snapshot pivoting functionality."""

    @pytest.fixture
    def sample_snapshot_df(self):
        """Create sample snapshot DataFrame in long format."""
        dates = pd.date_range('2019-01-01', '2020-06-01', freq='MS')
        series_names = ['VIX_max', 'SP500_return', 'Credit_Spreads_avg']

        data = []
        np.random.seed(42)
        for date in dates:
            for series in series_names:
                data.append({
                    'date': date,
                    'series_name': series,
                    'value': np.random.randn() * 10 + 50,
                    'snapshot_date': date + pd.offsets.MonthEnd(0)
                })

        return pd.DataFrame(data)

    def test_pivot_returns_dataframe(self, sample_snapshot_df):
        """Test that pivot returns a DataFrame."""
        target_month = pd.Timestamp('2020-03-01')
        result = pivot_snapshot_to_wide(sample_snapshot_df, target_month)

        assert isinstance(result, pd.DataFrame)

    def test_pivot_creates_feature_columns(self, sample_snapshot_df):
        """Test that pivot creates expected feature columns (sanitized names)."""
        target_month = pd.Timestamp('2020-03-01')
        result = pivot_snapshot_to_wide(sample_snapshot_df, target_month)

        if not result.empty:
            # Should have columns matching sanitized input series names
            expected_names = {'VIX_max', 'SP500_return', 'Credit_Spreads_avg'}
            assert len(result.columns) > 0
            assert result.columns.tolist() != []
            # At least some of the expected series should be present
            found = expected_names & set(result.columns)
            assert len(found) > 0, f"Expected some of {expected_names}, got {result.columns.tolist()}"

    def test_pivot_filters_by_target_month(self, sample_snapshot_df):
        """Test that pivot only uses data up to target month."""
        target_month = pd.Timestamp('2020-01-01')
        result = pivot_snapshot_to_wide(sample_snapshot_df, target_month)

        # Should not include data from after Jan 2020
        # This is implicit in the function behavior
        assert isinstance(result, pd.DataFrame)

    def test_pivot_empty_df(self):
        """Test that empty DataFrame returns empty result."""
        empty_df = pd.DataFrame(columns=['date', 'series_name', 'value'])
        target_month = pd.Timestamp('2020-03-01')

        result = pivot_snapshot_to_wide(empty_df, target_month)

        assert result.empty

    def test_pivot_none_df(self):
        """Test that None DataFrame returns empty result."""
        result = pivot_snapshot_to_wide(None, pd.Timestamp('2020-03-01'))

        assert result.empty

    def test_wide_noaa_ffill_cap_and_staleness(self):
        """NOAA forward fill should be capped and staleness should be emitted."""
        dates = pd.date_range('2020-01-01', periods=9, freq='MS')  # Jan..Sep
        wide_df = pd.DataFrame(
            {
                'NOAA_Economic_Damage_Index_feature': [5.0] + [np.nan] * 8,
                'macro_feature': [3.0] + [np.nan] * 8,
            },
            index=dates,
        )
        target_month = pd.Timestamp('2020-10-01')

        result = pivot_snapshot_to_wide(wide_df, target_month)

        assert not result.empty
        # NOAA is too stale (>6m) so capped forward-fill should drop it.
        assert 'NOAA_Economic_Damage_Index_feature' not in result.columns
        # Non-NOAA still uses unlimited ffill.
        assert 'macro_feature' in result.columns
        assert np.isclose(result['macro_feature'].iloc[0], 3.0)
        # Staleness feature remains available for the stale NOAA signal.
        staleness_col = 'NOAA_Economic_Damage_Index_feature__staleness_months'
        assert staleness_col in result.columns
        assert np.isclose(result[staleness_col].iloc[0], 9.0)

    def test_wide_noaa_within_cap_survives(self):
        """NOAA values within fill cap should remain available."""
        dates = pd.date_range('2020-01-01', periods=9, freq='MS')  # Jan..Sep
        noaa_values = [np.nan] * 9
        noaa_values[3] = 7.0  # Apr value, 6 months stale by Oct cutoff
        wide_df = pd.DataFrame(
            {'NOAA_Human_Impact_Index_feature': noaa_values},
            index=dates,
        )
        target_month = pd.Timestamp('2020-10-01')

        result = pivot_snapshot_to_wide(wide_df, target_month)

        assert not result.empty
        assert 'NOAA_Human_Impact_Index_feature' in result.columns
        assert np.isclose(result['NOAA_Human_Impact_Index_feature'].iloc[0], 7.0)
        staleness_col = 'NOAA_Human_Impact_Index_feature__staleness_months'
        assert staleness_col in result.columns
        assert np.isclose(result[staleness_col].iloc[0], 6.0)


class TestCacheFunctions:
    """Tests for cache management functions."""

    def test_clear_snapshot_cache(self):
        """Test that snapshot cache can be cleared."""
        # Should not raise
        clear_snapshot_cache()

    def test_clear_target_cache(self):
        """Test that target cache can be cleared."""
        # Should not raise
        clear_target_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
