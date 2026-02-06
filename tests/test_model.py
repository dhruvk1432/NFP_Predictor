"""
Tests for Model Module

Tests for LightGBM model training, prediction, and persistence functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.model import (
    get_lgbm_params,
    calculate_sample_weights,
    calculate_prediction_intervals,
    get_model_id,
    LIGHTGBM_AVAILABLE,
)

from Train.config import (
    DEFAULT_LGBM_PARAMS,
    PANIC_REGIME_WEIGHT,
    CONFIDENCE_LEVELS,
)


class TestGetLgbmParams:
    """Tests for LightGBM parameter configuration."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        params = get_lgbm_params()
        assert isinstance(params, dict)

    def test_default_objective_is_huber(self):
        """Test default objective is Huber (for outlier robustness)."""
        params = get_lgbm_params()
        assert params['objective'] == 'huber'
        assert 'alpha' in params

    def test_disable_huber_loss(self):
        """Test disabling Huber loss returns regression objective."""
        params = get_lgbm_params(use_huber_loss=False)
        assert params['objective'] == 'regression'
        assert 'alpha' not in params

    def test_huber_delta_customization(self):
        """Test custom Huber delta value."""
        params = get_lgbm_params(use_huber_loss=True, huber_delta=2.0)
        assert params['alpha'] == 2.0

    def test_does_not_modify_default(self):
        """Test that default params are not modified."""
        original = DEFAULT_LGBM_PARAMS.copy()
        get_lgbm_params(use_huber_loss=True)
        assert DEFAULT_LGBM_PARAMS == original


class TestCalculateSampleWeights:
    """Tests for sample weight calculation."""

    @pytest.fixture
    def normal_features(self):
        """Create features for normal market conditions."""
        return pd.DataFrame({
            'VIX_panic_regime': [0, 0, 0, 0, 0],
            'SP500_crash_month': [0, 0, 0, 0, 0],
            'VIX_high_regime': [0, 0, 0, 0, 0],
            'feature1': [1, 2, 3, 4, 5],
        })

    @pytest.fixture
    def panic_features(self):
        """Create features with panic regime indicators."""
        return pd.DataFrame({
            'VIX_panic_regime': [0, 1, 0, 1, 0],  # 2 panic months
            'SP500_crash_month': [0, 0, 1, 0, 0],  # 1 crash month
            'VIX_high_regime': [0, 0, 0, 0, 0],
            'feature1': [1, 2, 3, 4, 5],
        })

    def test_returns_array(self, normal_features):
        """Test that function returns numpy array."""
        weights = calculate_sample_weights(normal_features)
        assert isinstance(weights, np.ndarray)

    def test_correct_length(self, normal_features):
        """Test that weights have correct length."""
        weights = calculate_sample_weights(normal_features)
        assert len(weights) == len(normal_features)

    def test_normal_weights_are_one(self, normal_features):
        """Test that normal samples get weight of 1."""
        weights = calculate_sample_weights(normal_features)
        assert np.all(weights == 1.0)

    def test_panic_weights_elevated(self, panic_features):
        """Test that panic samples get elevated weights."""
        weights = calculate_sample_weights(panic_features)

        # Rows 1, 2, 3 have panic indicators
        assert weights[1] == PANIC_REGIME_WEIGHT
        assert weights[2] == PANIC_REGIME_WEIGHT
        assert weights[3] == PANIC_REGIME_WEIGHT

        # Rows 0, 4 are normal
        assert weights[0] == 1.0
        assert weights[4] == 1.0

    def test_handles_latest_suffix_columns(self):
        """Test handling of columns with _latest suffix."""
        df = pd.DataFrame({
            'VIX_panic_regime_latest': [0, 1, 0],
            'SP500_crash_month_latest': [0, 0, 1],
            'feature1': [1, 2, 3],
        })
        weights = calculate_sample_weights(df)

        assert weights[0] == 1.0
        assert weights[1] == PANIC_REGIME_WEIGHT
        assert weights[2] == PANIC_REGIME_WEIGHT

    def test_handles_missing_columns(self):
        """Test handling when regime columns are missing."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
        })
        weights = calculate_sample_weights(df)

        # Should return all ones when no indicators found
        assert np.all(weights == 1.0)


class TestCalculatePredictionIntervals:
    """Tests for prediction interval calculation."""

    @pytest.fixture
    def sample_residuals(self):
        """Create sample residuals for testing."""
        np.random.seed(42)
        return list(np.random.randn(100) * 50)  # Std dev ~50

    def test_returns_dict(self, sample_residuals):
        """Test that function returns a dictionary."""
        intervals = calculate_prediction_intervals(sample_residuals, 100.0)
        assert isinstance(intervals, dict)

    def test_contains_all_levels(self, sample_residuals):
        """Test that all confidence levels are present."""
        intervals = calculate_prediction_intervals(sample_residuals, 100.0)

        for level in CONFIDENCE_LEVELS:
            assert level in intervals

    def test_intervals_are_tuples(self, sample_residuals):
        """Test that intervals are (lower, upper) tuples."""
        intervals = calculate_prediction_intervals(sample_residuals, 100.0)

        for level, bounds in intervals.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] <= bounds[1]  # lower <= upper

    def test_wider_intervals_for_higher_confidence(self, sample_residuals):
        """Test that higher confidence means wider intervals."""
        intervals = calculate_prediction_intervals(sample_residuals, 100.0)

        width_50 = intervals[0.50][1] - intervals[0.50][0]
        width_80 = intervals[0.80][1] - intervals[0.80][0]
        width_95 = intervals[0.95][1] - intervals[0.95][0]

        assert width_50 < width_80 < width_95

    def test_prediction_in_50_interval(self, sample_residuals):
        """Test that prediction is typically within 50% interval."""
        prediction = 100.0
        intervals = calculate_prediction_intervals(sample_residuals, prediction)

        lower, upper = intervals[0.50]
        # The prediction should be reasonably close to the interval center
        center = (lower + upper) / 2
        assert abs(prediction - center) < (upper - lower)

    def test_handles_empty_residuals(self):
        """Test handling of empty residuals list."""
        intervals = calculate_prediction_intervals([], 100.0)

        # Should still return intervals (with placeholder values)
        assert len(intervals) > 0

    def test_handles_few_residuals(self):
        """Test handling of very few residuals."""
        intervals = calculate_prediction_intervals([10, 20, 30], 100.0)

        # Should return intervals
        assert len(intervals) > 0


class TestModelIdFunctions:
    """Tests for model ID utility functions."""

    def test_get_model_id_nsa_first(self):
        """Test model ID for NSA first release."""
        from Train.config import get_model_id
        model_id = get_model_id('nsa', 'first')
        assert model_id == 'nsa_first'

    def test_get_model_id_sa_last(self):
        """Test model ID for SA last release."""
        from Train.config import get_model_id
        model_id = get_model_id('sa', 'last')
        assert model_id == 'sa_last'

    def test_model_id_lowercase(self):
        """Test that model IDs are always lowercase."""
        from Train.config import get_model_id
        model_id = get_model_id('NSA', 'FIRST')
        assert model_id == 'nsa_first'


class TestLightGBMAvailability:
    """Tests for LightGBM availability check."""

    def test_lightgbm_available_is_bool(self):
        """Test that LIGHTGBM_AVAILABLE is a boolean."""
        assert isinstance(LIGHTGBM_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
