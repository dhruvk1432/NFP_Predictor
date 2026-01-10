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
def sample_values():
    """Sample employment values for transform testing."""
    return np.array([-20000, -1000, -100, 0, 100, 1000, 20000], dtype=float)


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
