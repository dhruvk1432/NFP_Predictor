"""
Tests for Train Configuration Module

Tests for model configuration, target path generation, and validation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.config import (
    VALID_TARGET_TYPES,
    VALID_RELEASE_TYPES,
    ALL_TARGET_CONFIGS,
    get_target_path,
    get_model_id,
    parse_model_id,
    MAX_FEATURES,
    VIF_THRESHOLD,
    CORR_THRESHOLD,
    MIN_TARGET_CORR,
    DEFAULT_LGBM_PARAMS,
    PROTECTED_BINARY_FLAGS,
    LINEAR_BASELINE_PREDICTORS,
    KEY_EMPLOYMENT_SERIES,
    ALL_KEY_EMPLOYMENT_SERIES,
    ALL_LAGS,
)


class TestTargetConfiguration:
    """Tests for target type and release type configuration."""

    def test_valid_target_types(self):
        """Test that valid target types are defined."""
        assert 'nsa' in VALID_TARGET_TYPES
        assert 'sa' in VALID_TARGET_TYPES
        assert len(VALID_TARGET_TYPES) == 2

    def test_valid_release_types(self):
        """Test that valid release types are defined."""
        assert 'first' in VALID_RELEASE_TYPES
        assert 'last' in VALID_RELEASE_TYPES
        assert len(VALID_RELEASE_TYPES) == 2

    def test_all_target_configs(self):
        """Test that all 4 target configurations are defined."""
        assert len(ALL_TARGET_CONFIGS) == 4
        assert ('nsa', 'first') in ALL_TARGET_CONFIGS
        assert ('nsa', 'last') in ALL_TARGET_CONFIGS
        assert ('sa', 'first') in ALL_TARGET_CONFIGS
        assert ('sa', 'last') in ALL_TARGET_CONFIGS


class TestGetTargetPath:
    """Tests for target path generation."""

    def test_nsa_first_path(self):
        """Test NSA first release path generation."""
        path = get_target_path('nsa', 'first')
        assert 'y_nsa_first_release.parquet' in str(path)
        assert 'NFP_target' in str(path)

    def test_sa_last_path(self):
        """Test SA last release path generation."""
        path = get_target_path('sa', 'last')
        assert 'y_sa_last_release.parquet' in str(path)

    def test_case_insensitivity(self):
        """Test that target and release types are case insensitive."""
        path1 = get_target_path('NSA', 'FIRST')
        path2 = get_target_path('nsa', 'first')
        assert str(path1) == str(path2)

    def test_invalid_target_type(self):
        """Test that invalid target type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            get_target_path('invalid', 'first')

    def test_invalid_release_type(self):
        """Test that invalid release type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid release_type"):
            get_target_path('nsa', 'invalid')


class TestGetModelId:
    """Tests for model ID generation."""

    def test_nsa_first_id(self):
        """Test NSA first model ID."""
        model_id = get_model_id('nsa', 'first')
        assert model_id == 'nsa_first'

    def test_sa_last_id(self):
        """Test SA last model ID."""
        model_id = get_model_id('sa', 'last')
        assert model_id == 'sa_last'

    def test_case_normalization(self):
        """Test that model IDs are lowercase."""
        model_id = get_model_id('NSA', 'FIRST')
        assert model_id == 'nsa_first'


class TestParseModelId:
    """Tests for model ID parsing."""

    def test_parse_nsa_first(self):
        """Test parsing nsa_first model ID."""
        target_type, release_type = parse_model_id('nsa_first')
        assert target_type == 'nsa'
        assert release_type == 'first'

    def test_parse_sa_last(self):
        """Test parsing sa_last model ID."""
        target_type, release_type = parse_model_id('sa_last')
        assert target_type == 'sa'
        assert release_type == 'last'

    def test_parse_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            parse_model_id('invalid')

    def test_parse_invalid_target(self):
        """Test that invalid target in model ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            parse_model_id('invalid_first')


class TestFeatureSelectionThresholds:
    """Tests for feature selection thresholds."""

    def test_max_features(self):
        """Test that MAX_FEATURES is reasonable."""
        assert MAX_FEATURES > 0
        assert MAX_FEATURES <= 200

    def test_vif_threshold(self):
        """Test that VIF threshold is reasonable."""
        assert VIF_THRESHOLD > 1.0  # VIF >= 1 always
        assert VIF_THRESHOLD <= 20.0  # Common thresholds: 5-10

    def test_correlation_threshold(self):
        """Test that correlation threshold is reasonable."""
        assert 0 < CORR_THRESHOLD <= 1.0
        assert CORR_THRESHOLD >= 0.8  # Should be high

    def test_min_target_correlation(self):
        """Test that min target correlation is reasonable."""
        assert 0 < MIN_TARGET_CORR <= 0.5


class TestLightGBMParams:
    """Tests for default LightGBM parameters."""

    def test_required_params_present(self):
        """Test that required parameters are present."""
        assert 'objective' in DEFAULT_LGBM_PARAMS
        assert 'metric' in DEFAULT_LGBM_PARAMS
        assert 'learning_rate' in DEFAULT_LGBM_PARAMS

    def test_learning_rate_reasonable(self):
        """Test that learning rate is reasonable."""
        lr = DEFAULT_LGBM_PARAMS['learning_rate']
        assert 0.001 <= lr <= 0.3

    def test_num_leaves_reasonable(self):
        """Test that num_leaves is reasonable."""
        num_leaves = DEFAULT_LGBM_PARAMS['num_leaves']
        assert 2 <= num_leaves <= 256


class TestProtectedFeatures:
    """Tests for protected binary flags configuration."""

    def test_protected_flags_defined(self):
        """Test that protected binary flags are defined."""
        assert len(PROTECTED_BINARY_FLAGS) > 0

    def test_vix_panic_in_protected(self):
        """Test that VIX panic regime is protected."""
        assert 'VIX_panic_regime' in PROTECTED_BINARY_FLAGS

    def test_sp500_crash_in_protected(self):
        """Test that SP500 crash indicators are protected."""
        assert 'SP500_crash_month' in PROTECTED_BINARY_FLAGS


class TestLinearBaselinePredictors:
    """Tests for linear baseline predictor configuration."""

    def test_predictors_defined(self):
        """Test that linear baseline predictors are defined."""
        assert len(LINEAR_BASELINE_PREDICTORS) > 0

    def test_contains_claims_data(self):
        """Test that claims data is included."""
        claims_predictors = [p for p in LINEAR_BASELINE_PREDICTORS if 'CCSA' in p]
        assert len(claims_predictors) > 0


class TestEmploymentSeries:
    """Tests for employment series configuration."""

    def test_key_series_categories(self):
        """Test that key employment series categories are defined."""
        assert 'aggregates' in KEY_EMPLOYMENT_SERIES
        assert 'goods_services' in KEY_EMPLOYMENT_SERIES
        assert 'goods_industries' in KEY_EMPLOYMENT_SERIES
        assert 'service_industries' in KEY_EMPLOYMENT_SERIES

    def test_all_series_flattened(self):
        """Test that ALL_KEY_EMPLOYMENT_SERIES contains all series."""
        assert len(ALL_KEY_EMPLOYMENT_SERIES) > 20  # Should have many series

    def test_series_naming_convention(self):
        """Test that series follow naming convention."""
        for series in ALL_KEY_EMPLOYMENT_SERIES:
            # All should end with _nsa (NSA series)
            assert series.endswith('_nsa'), f"Series {series} should end with _nsa"


class TestLagConfiguration:
    """Tests for lag configuration."""

    def test_lags_defined(self):
        """Test that lag periods are defined."""
        assert len(ALL_LAGS) > 0

    def test_lag_values(self):
        """Test that expected lag values are present."""
        assert 1 in ALL_LAGS  # Short-term
        assert 12 in ALL_LAGS  # Annual


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
