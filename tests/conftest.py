"""
Pytest Configuration and Fixtures

Shared fixtures for NFP Predictor tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# DATE FIXTURES
# =============================================================================

@pytest.fixture
def sample_snapshot_date():
    """Sample snapshot date for testing."""
    return pd.Timestamp('2020-03-31')


@pytest.fixture
def sample_covid_date():
    """COVID crash date for extreme event testing."""
    return pd.Timestamp('2020-03-01')


@pytest.fixture
def sample_normal_date():
    """Normal period date for baseline testing."""
    return pd.Timestamp('2019-06-01')


@pytest.fixture
def sample_target_month():
    """Sample target month for prediction testing."""
    return pd.Timestamp('2020-06-01')


# =============================================================================
# VALUE FIXTURES
# =============================================================================

@pytest.fixture
def sample_values():
    """Sample employment values for transform testing."""
    return np.array([-20000, -1000, -100, 0, 100, 1000, 20000], dtype=float)


@pytest.fixture
def sample_positive_values():
    """Sample positive values for log1p transform testing."""
    return np.array([0, 1, 10, 100, 1000, 10000], dtype=float)


@pytest.fixture
def sample_residuals():
    """Sample residuals for interval testing."""
    np.random.seed(42)
    return list(np.random.randn(100) * 50)


# =============================================================================
# DATAFRAME FIXTURES
# =============================================================================

@pytest.fixture
def sample_time_series():
    """Sample time series DataFrame for feature engineering testing."""
    dates = pd.date_range('2019-01-01', '2020-12-01', freq='MS')
    np.random.seed(42)

    return pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates)) * 100 + 150000,
        'series_name': 'test_series'
    })


@pytest.fixture
def sample_snapshot_df():
    """Sample snapshot DataFrame in long format."""
    dates = pd.date_range('2019-01-01', '2020-03-01', freq='MS')
    series_names = ['VIX_max', 'SP500_crash_month', 'Credit_Spreads_avg']

    np.random.seed(42)
    data = []
    for date in dates:
        for series in series_names:
            data.append({
                'date': date,
                'series_name': series,
                'value': np.random.randn() * 10 + 50,
                'snapshot_date': date + pd.offsets.MonthEnd(0)
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_target_df():
    """Sample target DataFrame with MoM calculations."""
    dates = pd.date_range('2018-01-01', '2020-12-01', freq='MS')
    np.random.seed(42)

    df = pd.DataFrame({
        'ds': dates,
        'y': np.cumsum(np.random.randn(len(dates)) * 100) + 150000,
    })
    df['y_mom'] = df['y'].diff()
    df['y_yoy'] = df['y'].diff(12)

    return df


@pytest.fixture
def sample_fred_df():
    """Sample FRED employment DataFrame."""
    dates = pd.date_range('2019-01-01', '2020-06-01', freq='MS')
    series_names = [
        'total.private_nsa',
        'total.government_nsa',
        'total.private.goods_nsa',
        'total.private.services_nsa',
    ]

    np.random.seed(42)
    data = []
    for date in dates:
        for series in series_names:
            data.append({
                'date': date,
                'series_name': series,
                'value': np.random.randn() * 1000 + 50000,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_exogenous_df():
    """Sample exogenous indicators DataFrame."""
    dates = pd.date_range('2019-01-01', '2020-06-01', freq='MS')
    series_names = [
        'VIX_max', 'VIX_mean', 'VIX_volatility',
        'SP500_monthly_return', 'SP500_volatility',
        'Credit_Spreads_avg', 'Oil_Prices_mean',
    ]

    np.random.seed(42)
    data = []
    for date in dates:
        for series in series_names:
            data.append({
                'date': date,
                'series_name': series,
                'value': np.random.randn() * 10 + 50,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_panic_features():
    """Sample features with panic regime indicators."""
    return pd.DataFrame({
        'VIX_panic_regime': [0, 1, 0, 1, 0],
        'SP500_crash_month': [0, 0, 1, 0, 0],
        'VIX_high_regime': [0, 1, 1, 0, 0],
        'SP500_bear_market': [0, 0, 0, 0, 0],
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
    })


@pytest.fixture
def sample_normal_features():
    """Sample features with no panic indicators."""
    return pd.DataFrame({
        'VIX_panic_regime': [0, 0, 0, 0, 0],
        'SP500_crash_month': [0, 0, 0, 0, 0],
        'VIX_high_regime': [0, 0, 0, 0, 0],
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
    })


# =============================================================================
# MODEL CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def all_target_configs():
    """All valid target configurations."""
    return [
        ('nsa', 'first'),
        ('nsa', 'last'),
        ('sa', 'first'),
        ('sa', 'last'),
    ]


@pytest.fixture
def sample_feature_cols():
    """Sample feature column names."""
    return [
        'VIX_max_latest', 'VIX_mean_latest', 'VIX_volatility_latest',
        'SP500_monthly_return_latest', 'Credit_Spreads_avg_latest',
        'month_sin', 'month_cos', 'is_5_week_month',
        'nfp_mom_lag1', 'nfp_mom_lag12',
    ]
