"""
Tests for Transform Utilities

Tests for symlog, z-score, and other transformation functions.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
