"""
Tests for Feature Engineering Module

Tests for calendar features, employment features, and exogenous feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.feature_engineering import (
    get_survey_week_date,
    calculate_weeks_between_surveys,
    add_calendar_features,
    get_calendar_features_dict,
    engineer_employment_features,
    engineer_exogenous_features,
    calculate_sector_breadth,
)


class TestSurveyWeekDate:
    """Tests for survey week date calculation."""

    def test_survey_week_returns_sunday(self):
        """Test that survey week returns a Sunday."""
        result = get_survey_week_date(2020, 3)
        # The 12th of March 2020 was a Thursday
        # Sunday of that week would be March 8, 2020
        assert result.weekday() == 6  # Sunday is 6 in Python's weekday()

    def test_survey_week_near_12th(self):
        """Test that returned date is near the 12th."""
        result = get_survey_week_date(2020, 3)
        # Should be within a week of the 12th
        day_12 = pd.Timestamp(2020, 3, 12)
        diff = abs((pd.Timestamp(result) - day_12).days)
        assert diff <= 6


class TestWeeksBetweenSurveys:
    """Tests for weeks between surveys calculation."""

    def test_returns_integer(self):
        """Test that function returns an integer."""
        target_month = pd.Timestamp('2020-03-01')
        result = calculate_weeks_between_surveys(target_month)
        assert isinstance(result, int)

    def test_returns_4_or_5(self):
        """Test that result is typically 4 or 5 weeks."""
        # Test multiple months
        for month in range(1, 13):
            target_month = pd.Timestamp(f'2020-{month:02d}-01')
            result = calculate_weeks_between_surveys(target_month)
            assert result in [4, 5], f"Month {month} returned {result} weeks"


class TestAddCalendarFeatures:
    """Tests for calendar feature addition."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})

    def test_adds_cyclical_month_encoding(self, sample_df):
        """Test that cyclical month encoding is added."""
        target_month = pd.Timestamp('2020-03-01')
        result = add_calendar_features(sample_df, target_month)

        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns

    def test_cyclical_encoding_bounds(self, sample_df):
        """Test that cyclical encoding values are bounded [-1, 1]."""
        target_month = pd.Timestamp('2020-06-01')
        result = add_calendar_features(sample_df, target_month)

        assert -1 <= result['month_sin'].iloc[0] <= 1
        assert -1 <= result['month_cos'].iloc[0] <= 1

    def test_adds_quarter_encoding(self, sample_df):
        """Test that quarter encoding is added."""
        target_month = pd.Timestamp('2020-03-01')
        result = add_calendar_features(sample_df, target_month)

        assert 'quarter_sin' in result.columns
        assert 'quarter_cos' in result.columns

    def test_adds_survey_interval(self, sample_df):
        """Test that survey interval features are added."""
        target_month = pd.Timestamp('2020-03-01')
        result = add_calendar_features(sample_df, target_month)

        assert 'weeks_since_last_survey' in result.columns
        assert 'is_5_week_month' in result.columns

    def test_adds_seasonal_indicators(self, sample_df):
        """Test that seasonal indicators are added."""
        target_month = pd.Timestamp('2020-07-01')
        result = add_calendar_features(sample_df, target_month)

        assert 'is_summer' in result.columns
        assert 'is_holiday_season' in result.columns

    def test_january_indicator(self, sample_df):
        """Test that January indicator is correct."""
        jan_result = add_calendar_features(sample_df, pd.Timestamp('2020-01-01'))
        mar_result = add_calendar_features(sample_df, pd.Timestamp('2020-03-01'))

        assert jan_result['is_jan'].iloc[0] == 1
        assert mar_result['is_jan'].iloc[0] == 0

    def test_does_not_modify_original(self, sample_df):
        """Test that original DataFrame is not modified."""
        original_columns = list(sample_df.columns)
        target_month = pd.Timestamp('2020-03-01')
        add_calendar_features(sample_df, target_month)

        assert list(sample_df.columns) == original_columns


class TestGetCalendarFeaturesDict:
    """Tests for calendar features dictionary generation."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        target_month = pd.Timestamp('2020-03-01')
        result = get_calendar_features_dict(target_month)

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that dictionary contains required keys."""
        target_month = pd.Timestamp('2020-03-01')
        result = get_calendar_features_dict(target_month)

        required_keys = [
            'month_sin', 'month_cos',
            'quarter_sin', 'quarter_cos',
            'weeks_since_last_survey', 'is_5_week_month',
            'is_jan', 'is_summer'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_values_are_numeric(self):
        """Test that all values are numeric."""
        target_month = pd.Timestamp('2020-03-01')
        result = get_calendar_features_dict(target_month)

        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"


class TestEngineerEmploymentFeatures:
    """Tests for employment feature engineering."""

    @pytest.fixture
    def sample_fred_df(self):
        """Create sample FRED DataFrame for testing."""
        dates = pd.date_range('2019-01-01', '2020-06-01', freq='MS')
        series_names = [
            'total.private_nsa',
            'total.government_nsa',
            'total.private.goods_nsa'
        ]

        data = []
        np.random.seed(42)
        for date in dates:
            for series in series_names:
                data.append({
                    'date': date,
                    'series_name': series,
                    'value': np.random.randn() * 100 + 50000,
                })

        return pd.DataFrame(data)

    def test_returns_dict(self, sample_fred_df):
        """Test that function returns a dictionary."""
        target_month = pd.Timestamp('2020-03-01')
        result = engineer_employment_features(sample_fred_df, target_month, 'nsa')

        assert isinstance(result, dict)

    def test_empty_df_returns_empty(self):
        """Test that empty DataFrame returns empty dict."""
        empty_df = pd.DataFrame()
        target_month = pd.Timestamp('2020-03-01')

        result = engineer_employment_features(empty_df, target_month, 'nsa')

        assert result == {}

    def test_none_df_returns_empty(self):
        """Test that None DataFrame returns empty dict."""
        result = engineer_employment_features(None, pd.Timestamp('2020-03-01'), 'nsa')

        assert result == {}


class TestEngineerExogenousFeatures:
    """Tests for exogenous feature engineering."""

    @pytest.fixture
    def sample_snapshot_df(self):
        """Create sample snapshot DataFrame."""
        dates = pd.date_range('2019-01-01', '2020-06-01', freq='MS')
        series_names = ['VIX_max', 'Credit_Spreads_avg', 'Oil_Prices_mean']

        data = []
        np.random.seed(42)
        for date in dates:
            for series in series_names:
                data.append({
                    'date': date,
                    'series_name': series,
                    'value': np.random.randn() * 10 + 50,
                })

        return pd.DataFrame(data)

    def test_returns_dict(self, sample_snapshot_df):
        """Test that function returns a dictionary."""
        target_month = pd.Timestamp('2020-03-01')
        result = engineer_exogenous_features(sample_snapshot_df, target_month)

        assert isinstance(result, dict)

    def test_creates_latest_features(self, sample_snapshot_df):
        """Test that _latest features are created."""
        target_month = pd.Timestamp('2020-03-01')
        result = engineer_exogenous_features(sample_snapshot_df, target_month)

        latest_features = [k for k in result.keys() if '_latest' in k]
        assert len(latest_features) > 0

    def test_creates_lag_features(self, sample_snapshot_df):
        """Test that lag features are created."""
        target_month = pd.Timestamp('2020-03-01')
        result = engineer_exogenous_features(sample_snapshot_df, target_month)

        lag_features = [k for k in result.keys() if '_lag' in k]
        assert len(lag_features) > 0

    def test_empty_df_returns_empty(self):
        """Test that empty DataFrame returns empty dict."""
        empty_df = pd.DataFrame()
        result = engineer_exogenous_features(empty_df, pd.Timestamp('2020-03-01'))
        assert result == {}


class TestCalculateSectorBreadth:
    """Tests for sector breadth calculation."""

    @pytest.fixture
    def sample_sector_df(self):
        """Create sample DataFrame with sector data."""
        dates = pd.date_range('2020-01-01', '2020-06-01', freq='MS')

        # Create data for service industries
        service_series = [
            'total.private.services.trade_transportation_utilities_nsa',
            'total.private.services.information_nsa',
            'total.private.services.financial_nsa',
        ]

        data = []
        np.random.seed(42)
        base_values = {s: np.random.rand() * 10000 + 50000 for s in service_series}

        for i, date in enumerate(dates):
            for series in service_series:
                # Add some growth/decline
                value = base_values[series] + i * np.random.randn() * 100
                data.append({
                    'date': date,
                    'series_name': series,
                    'value': value,
                })

        return pd.DataFrame(data)

    def test_returns_dict(self, sample_sector_df):
        """Test that function returns a dictionary."""
        target_month = pd.Timestamp('2020-06-01')
        result = calculate_sector_breadth(sample_sector_df, target_month, 'nsa')

        assert isinstance(result, dict)

    def test_empty_df_returns_empty(self):
        """Test that empty DataFrame returns empty dict."""
        result = calculate_sector_breadth(
            pd.DataFrame(),
            pd.Timestamp('2020-03-01'),
            'nsa'
        )
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
