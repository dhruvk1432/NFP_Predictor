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
    VARIANCE_PRIORITY_TARGETS,
    get_target_path,
    get_model_id,
    parse_model_id,
    DEFAULT_LGBM_PARAMS,
    HALF_LIFE_MIN_MONTHS,
    HALF_LIFE_MAX_MONTHS
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
        """Test that all 4 branch target configurations are defined."""
        assert len(ALL_TARGET_CONFIGS) == 4
        assert ('nsa', 'first', 'first_release') in ALL_TARGET_CONFIGS
        assert ('nsa', 'first', 'revised') in ALL_TARGET_CONFIGS
        assert ('sa', 'first', 'first_release') in ALL_TARGET_CONFIGS
        assert ('sa', 'first', 'revised') in ALL_TARGET_CONFIGS


class TestGetTargetPath:
    """Tests for target path generation."""

    def test_nsa_first_path(self):
        """Test NSA first release path."""
        path = get_target_path('nsa', 'first')
        assert 'total_nsa_first_release.parquet' in str(path)
        assert 'NFP_target' in str(path)

    def test_sa_last_path(self):
        """Test SA last release path."""
        path = get_target_path('sa', 'last')
        assert 'total_sa_last_release.parquet' in str(path)

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
        target_type, release_type, target_source = parse_model_id('nsa_first')
        assert target_type == 'nsa'
        assert release_type == 'first'
        assert target_source == 'first_release'

    def test_parse_sa_last(self):
        """Test parsing sa_last model ID."""
        target_type, release_type, target_source = parse_model_id('sa_last')
        assert target_type == 'sa'
        assert release_type == 'last'
        assert target_source == 'first_release'

    def test_parse_nsa_first_revised(self):
        """Test parsing nsa_first_revised model ID."""
        target_type, release_type, target_source = parse_model_id('nsa_first_revised')
        assert target_type == 'nsa'
        assert release_type == 'first'
        assert target_source == 'revised'

    def test_parse_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            parse_model_id('invalid')

    def test_parse_invalid_three_parts(self):
        """Test that invalid 3-part format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            parse_model_id('nsa_first_bogus')

    def test_parse_invalid_target(self):
        """Test that invalid target in model ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            parse_model_id('invalid_first')


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


class TestHyperparameters:
    """Tests for hyperparameter bounds and defaults."""
    
    def test_halflife_bounds_exist(self):
        """Test that exponential decay bounds exist."""
        assert HALF_LIFE_MIN_MONTHS > 0
        assert HALF_LIFE_MAX_MONTHS > HALF_LIFE_MIN_MONTHS
        
    def test_halflife_values(self):
        """Test that the default half-life bounds are reasonable."""
        assert HALF_LIFE_MIN_MONTHS >= 6  # At least 6 months
        assert HALF_LIFE_MAX_MONTHS <= 240  # Max 20 years


class TestVariancePriorityTargets:
    """Tests for variance-priority branch configuration."""

    def test_sa_branches_are_variance_priority(self):
        """SA first-release and revised should share variance-focused modeling."""
        assert ('sa', 'first_release') in VARIANCE_PRIORITY_TARGETS
        assert ('sa', 'revised') in VARIANCE_PRIORITY_TARGETS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
