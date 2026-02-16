
import pytest
import pandas as pd
import numpy as np
from utils.transforms import compute_all_features, add_pct_change_copies, add_symlog_copies

class TestGlobalLags:
    """Tests for the global application of lags to all features in compute_all_features."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame with a simple linear series."""
        dates = pd.date_range('2020-01-01', '2022-01-01', freq='MS')
        df = pd.DataFrame({
            'date': dates,
            'series_name': 'TestSeries',
            'value': np.arange(len(dates), dtype=float), # 0, 1, 2...
            'release_date': dates + pd.Timedelta(days=10),
            'series_code': 'TEST',
            'snapshot_date': pd.Timestamp('2022-02-01')
        })
        return df

    def test_lags_applied_to_base_level(self, sample_data):
        """Test that base level features get all expected lags."""
        result = compute_all_features(sample_data)
        series_names = result['series_name'].unique()
        
        expected_lags = [1, 3, 6, 12, 18]
        for lag in expected_lags:
            expected_name = f"TestSeries_lag_{lag}m"
            assert expected_name in series_names, f"Missing expected base lag: {expected_name}"

    def test_lags_applied_to_derived_features(self, sample_data):
        """Test that derived features (e.g. z-scores) get all expected lags."""
        result = compute_all_features(sample_data)
        series_names = result['series_name'].unique()
        
        # Check a derived feature like 3m z-score
        base_derived = "TestSeries_zscore_3m"
        
        # Ensure base derived feature exists first (might be filtered out if too short, but here length is 25)
        if base_derived in series_names:
            expected_lags = [1, 3, 6, 12, 18]
            for lag in expected_lags:
                expected_name = f"{base_derived}_lag_{lag}m"
                assert expected_name in series_names, f"Missing expected derived lag: {expected_name}"
        else:
            pytest.fail(f"Base derived feature {base_derived} was not generated, cannot test lags.")

    def test_lag_18m_value_correctness(self, sample_data):
        """Test that the 18-month lag contains the correct shifted values."""
        result = compute_all_features(sample_data)
        
        lag_18_name = "TestSeries_lag_18m"
        target_date = sample_data['date'].iloc[18] # Index 18
        
        # Value at index 18 for lag 18 should be value at index 0
        lag_val = result.loc[(result['series_name'] == lag_18_name) & (result['date'] == target_date), 'value']
        
        assert len(lag_val) == 1, "Expected exactly one value for lag check"
        assert lag_val.iloc[0] == 0.0, f"Expected 0.0, got {lag_val.iloc[0]}"

    def test_lags_on_pct_change_variants(self, sample_data):
        """Test that lags are applied to _pct_chg and _symlog_pct_chg variants."""
        df = add_symlog_copies(sample_data)
        df = add_pct_change_copies(df)
        result = compute_all_features(df)
        series_names = result['series_name'].unique()
        
        variants = ['TestSeries_pct_chg', 'TestSeries_symlog_pct_chg']
        expected_lags = [1, 3, 6, 12, 18]
        
        for variant in variants:
            # Check if base variant exists (it should)
            if variant in series_names:
                for lag in expected_lags:
                    expected_name = f"{variant}_lag_{lag}m"
                    assert expected_name in series_names, f"Missing lag for {variant}: {expected_name}"
