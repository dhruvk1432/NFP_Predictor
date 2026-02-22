"""
Tests for Transform Utilities Module

This module ensures that statistical transformations (symlog for extreme values, 
z-scores for normalized volatility, log1p, and winsorization) act predictably on both
positive and negative economic data. It also validates the _symlog_pct_chg variant
logic introduced for the Branch-and-Expand pipeline, ensuring no double-transforms occur.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.transforms import (
    apply_symlog,
    inverse_symlog,
    apply_log1p,
    inverse_log1p,
    calculate_z_score,
    winsorize,
    winsorize_covid_period,
    add_symlog_copies,
    add_pct_change_copies,
    compute_all_features,
)


class TestSymlog:
    """Tests for symlog transform."""
    
    def test_symlog_positive(self):
        """Test symlog on positive values."""
        result = apply_symlog(1000)
        assert result > 0
        assert result == np.log1p(1000)
    
    def test_symlog_negative(self):
        """Test symlog on negative values."""
        result = apply_symlog(-1000)
        assert result < 0
        assert result == -np.log1p(1000)
    
    def test_symlog_zero(self):
        """Test symlog on zero."""
        result = apply_symlog(0)
        assert result == 0
    
    def test_symlog_invertibility(self, sample_values):
        """Test that inverse_symlog recovers original values."""
        transformed = apply_symlog(sample_values)
        recovered = inverse_symlog(transformed)
        np.testing.assert_array_almost_equal(sample_values, recovered)
    
    def test_symlog_series(self):
        """Test symlog on pandas Series."""
        series = pd.Series([-100, 0, 100])
        result = apply_symlog(series)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
    
    def test_symlog_compression(self):
        """Test that symlog compresses extreme values."""
        small = apply_symlog(100)
        large = apply_symlog(1000000)
        
        # Large value should not be 10000x the small value
        assert large / small < 100


class TestLog1p:
    """Tests for log1p transform."""
    
    def test_log1p_positive(self):
        """Test log1p on positive values."""
        result = apply_log1p(100)
        assert result == np.log1p(100)
    
    def test_log1p_zero(self):
        """Test log1p on zero."""
        result = apply_log1p(0)
        assert result == 0
    
    def test_log1p_invertibility(self):
        """Test that inverse_log1p recovers original values."""
        values = np.array([0, 1, 10, 100, 1000])
        transformed = apply_log1p(values)
        recovered = inverse_log1p(transformed)
        np.testing.assert_array_almost_equal(values, recovered)


class TestZScore:
    """Tests for z-score calculation."""
    
    def test_zscore_basic(self):
        """Test basic z-score calculation."""
        series = pd.Series([1, 2, 3, 4, 5] * 20)  # 100 values
        result = calculate_z_score(series, window=20, min_periods=10)
        
        # After warmup, z-scores should exist
        assert not result.iloc[-1:].isna().all()
    
    def test_zscore_output_shape(self):
        """Test that z-score output has same shape as input."""
        series = pd.Series(np.random.randn(100))
        result = calculate_z_score(series, window=20, min_periods=10)
        
        assert len(result) == len(series)


class TestWinsorize:
    """Tests for winsorization."""
    
    def test_winsorize_clips_extremes(self):
        """Test that winsorize clips extreme values."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        result = winsorize(series, lower_percentile=0.1, upper_percentile=0.9)
        
        assert result.max() < 100
    
    def test_winsorize_preserves_middle(self):
        """Test that winsorize preserves middle values."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = winsorize(series, lower_percentile=0.1, upper_percentile=0.9)
        
        # Middle value should be unchanged
        assert result.iloc[2] == 3


class TestSymlogPctChg:
    """Tests for the _symlog_pct_chg variant in the Branch-and-Expand pipeline."""

    @pytest.fixture
    def sample_long_df(self):
        """Create sample long-format DataFrame with multiple series and dates."""
        dates = pd.date_range('2019-01-01', '2020-12-01', freq='MS')
        np.random.seed(42)
        data = []
        for date in dates:
            for series in ['VIX_max', 'Oil_Prices_mean']:
                data.append({
                    'date': date,
                    'series_name': series,
                    'value': np.random.rand() * 50 + 10,
                })
        return pd.DataFrame(data)

    def test_creates_symlog_pct_chg_series(self, sample_long_df):
        """Test that add_pct_change_copies creates _symlog_pct_chg from _symlog series."""
        # Step 1: Add symlog copies (creates VIX_max_symlog, Oil_Prices_mean_symlog)
        df = add_symlog_copies(sample_long_df)
        symlog_series = [s for s in df['series_name'].unique() if s.endswith('_symlog')]
        assert len(symlog_series) == 2, f"Expected 2 symlog series, got {len(symlog_series)}"

        # Step 2: Add pct_change copies (should create both _pct_chg and _symlog_pct_chg)
        df = add_pct_change_copies(df)
        series_names = sorted(df['series_name'].unique())

        # Check that symlog_pct_chg variants exist
        symlog_pct_chg = [s for s in series_names if s.endswith('_symlog_pct_chg')]
        assert len(symlog_pct_chg) == 2, (
            f"Expected 2 _symlog_pct_chg series, got {len(symlog_pct_chg)}: {symlog_pct_chg}"
        )

        # Check that regular pct_chg variants also exist
        regular_pct_chg = [s for s in series_names if s.endswith('_pct_chg') and '_symlog_pct_chg' not in s]
        assert len(regular_pct_chg) == 2, (
            f"Expected 2 _pct_chg series, got {len(regular_pct_chg)}: {regular_pct_chg}"
        )

    def test_no_double_symlog(self, sample_long_df):
        """Test that the standard pipeline order doesn't produce _symlog_symlog or _pct_chg_symlog."""
        # Standard pipeline order: symlog first, then pct_change
        df = add_symlog_copies(sample_long_df)
        df = add_pct_change_copies(df)

        series_names = df['series_name'].unique()
        # There should be no _symlog_symlog (double symlog transform)
        double_symlog = [s for s in series_names if '_symlog_symlog' in s]
        assert len(double_symlog) == 0, (
            f"Found unwanted double-symlog series: {double_symlog}"
        )

    def test_compute_features_symlog_pct_chg_no_diffs(self, sample_long_df):
        """Test that compute_all_features treats _symlog_pct_chg like _pct_chg (no diffs)."""
        df = add_symlog_copies(sample_long_df)
        df = add_pct_change_copies(df)
        result = compute_all_features(df)

        series_names = result['series_name'].unique()

        # _symlog_pct_chg should NOT have _diff series (it's already a rate-of-change)
        symlog_pct_chg_diffs = [
            s for s in series_names
            if '_symlog_pct_chg' in s and '_diff' in s
        ]
        assert len(symlog_pct_chg_diffs) == 0, (
            f"Found unexpected diff series for _symlog_pct_chg: {symlog_pct_chg_diffs}"
        )

        # But _symlog_pct_chg SHOULD have lag and zscore series
        symlog_pct_chg_lags = [
            s for s in series_names
            if '_symlog_pct_chg' in s and '_lag' in s
        ]
        assert len(symlog_pct_chg_lags) > 0, (
            "Expected lag series for _symlog_pct_chg but found none"
        )

class TestWinsorizeCovidPeriod:
    """Tests for COVID-period winsorization."""

    @pytest.fixture
    def wide_df(self):
        """Wide-format DataFrame spanning 2019–2021 with a COVID spike."""
        dates = pd.date_range('2019-01-01', '2021-06-01', freq='MS')
        np.random.seed(0)
        df = pd.DataFrame(
            {'A': np.random.randn(len(dates)) * 10,
             'B': np.random.randn(len(dates)) * 5},
            index=dates,
        )
        # Inject extreme COVID-period values
        df.loc['2020-03-01', 'A'] = 500
        df.loc['2020-04-01', 'A'] = -600
        df.loc['2020-05-01', 'B'] = 400
        return df

    def test_clips_covid_values_dataframe(self, wide_df):
        """COVID-period values should be clipped in a DataFrame."""
        result = winsorize_covid_period(wide_df)
        assert result.loc['2020-03-01', 'A'] < 500
        assert result.loc['2020-04-01', 'A'] > -600
        assert result.loc['2020-05-01', 'B'] < 400

    def test_clips_covid_values_series(self, wide_df):
        """COVID-period values should be clipped in a Series."""
        series = wide_df['A'].copy()
        result = winsorize_covid_period(series)
        assert result.loc['2020-03-01'] < 500
        assert result.loc['2020-04-01'] > -600

    def test_non_covid_unchanged(self, wide_df):
        """Values outside the COVID period should remain unchanged."""
        result = winsorize_covid_period(wide_df)
        non_covid_mask = ~(
            (wide_df.index >= '2020-03-01') & (wide_df.index <= '2020-05-01')
        )
        pd.testing.assert_frame_equal(
            result.loc[non_covid_mask], wide_df.loc[non_covid_mask]
        )

    def test_custom_date_range(self, wide_df):
        """Custom start/end should change which rows are clipped."""
        # Only winsorize March
        result = winsorize_covid_period(
            wide_df, covid_start='2020-03-01', covid_end='2020-03-01'
        )
        assert result.loc['2020-03-01', 'A'] < 500   # clipped
        assert result.loc['2020-04-01', 'A'] == -600  # untouched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
