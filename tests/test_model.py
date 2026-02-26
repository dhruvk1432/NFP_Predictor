"""
Tests for Model Module

Tests for LightGBM model training, prediction, and persistence functions.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
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
    save_model,
)

from Train.config import (
    DEFAULT_LGBM_PARAMS,
    CONFIDENCE_LEVELS,
    HALF_LIFE_MIN_MONTHS,
    HALF_LIFE_MAX_MONTHS
)


class TestGetLgbmParams:
    """Tests for LightGBM parameter configuration."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        params = get_lgbm_params()
        assert isinstance(params, dict)

    def test_default_objective_is_regression(self):
        """Test that default objective is regression (Huber disabled by default)."""
        params = get_lgbm_params()
        assert params['objective'] == 'regression'
        assert 'alpha' not in params

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
    """Tests for sample weight calculation using exponential decay."""

    @pytest.fixture
    def sample_features(self):
        """Create features for different dates to test decay."""
        return pd.DataFrame({
            'ds': [
                pd.Timestamp('2020-01-01'),  # Oldest (furthest)
                pd.Timestamp('2022-01-01'),  # Middle
                pd.Timestamp('2024-01-01'),  # Newest (closest)
            ],
            'feature1': [1, 2, 3],
        })

    def test_returns_array(self, sample_features):
        """Test that function returns numpy array."""
        target_month = pd.Timestamp('2024-02-01')
        weights = calculate_sample_weights(sample_features, target_month, 60.0)
        assert isinstance(weights, np.ndarray)

    def test_correct_length(self, sample_features):
        """Test that weights have correct length."""
        target_month = pd.Timestamp('2024-02-01')
        weights = calculate_sample_weights(sample_features, target_month, 60.0)
        assert len(weights) == len(sample_features)

    def test_more_recent_gets_higher_weight(self, sample_features):
        """Test that more recent samples get exponentially higher weights."""
        target_month = pd.Timestamp('2024-02-01')
        weights = calculate_sample_weights(sample_features, target_month, 60.0)
        
        # weights[2] is 2024, weights[1] is 2022, weights[0] is 2020
        assert weights[2] > weights[1] > weights[0]

    def test_mean_weight_is_one(self, sample_features):
        """Test that weights are normalized to mean 1.0 to preserve learning rate."""
        target_month = pd.Timestamp('2024-02-01')
        weights = calculate_sample_weights(sample_features, target_month, 60.0)
        assert np.isclose(np.mean(weights), 1.0)
        
    def test_shorter_halflife_creates_steeper_decay(self, sample_features):
        """Test that a shorter half-life penalizes older dates more heavily."""
        target_month = pd.Timestamp('2024-02-01')
        weights_short = calculate_sample_weights(sample_features, target_month, 12.0)
        weights_long = calculate_sample_weights(sample_features, target_month, 120.0)
        
        # The ratio of newest weight to oldest weight should be much larger for the short half-life
        ratio_short = weights_short[2] / weights_short[0]
        ratio_long = weights_long[2] / weights_long[0]
        
        assert ratio_short > ratio_long

    def test_handles_missing_ds_column(self):
        """Test fallback to equal weights if date column is missing."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        target_month = pd.Timestamp('2024-02-01')
        weights = calculate_sample_weights(df, target_month, 60.0)
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


class TestSaveModelMetadata:
    """Tests for save_model metadata payload behavior."""

    def test_save_model_persists_extra_metadata(self, tmp_path):
        """extra_metadata fields should be saved into metadata.pkl."""

        class DummyModel:
            def save_model(self, path):
                Path(path).write_text("dummy-model")

        save_model(
            model=DummyModel(),
            feature_cols=["f1", "f2"],
            residuals=[1.0, -1.0],
            importance={"f1": 10.0, "f2": 0.0},
            save_dir=tmp_path,
            target_type="nsa",
            release_type="first",
            target_source="revised",
            extra_metadata={"production_eligible": False, "keep_rule_failed": True},
        )

        meta_path = tmp_path / "nsa_first_revised" / "lightgbm_nsa_first_revised_metadata.pkl"
        assert meta_path.exists()
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)

        assert payload["production_eligible"] is False
        assert payload["keep_rule_failed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
