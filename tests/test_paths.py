"""
Tests for Path Utilities

Tests for snapshot path generation functions.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    get_fred_snapshot_path,
    get_master_snapshot_path,
    get_exogenous_snapshot_path,
    get_target_path,
    get_decade_year_path,
)


class TestDecadeYearPath:
    """Tests for decade/year path generation."""
    
    def test_1990s_path(self):
        """Test path generation for 1990s."""
        date = pd.Timestamp('1995-06-30')
        base = Path('/test')
        result = get_decade_year_path(base, date)
        
        assert '1990s' in str(result)
        assert '1995' in str(result)
        assert '1995-06' in str(result)
    
    def test_2020s_path(self):
        """Test path generation for 2020s."""
        date = pd.Timestamp('2024-12-31')
        base = Path('/test')
        result = get_decade_year_path(base, date)
        
        assert '2020s' in str(result)
        assert '2024' in str(result)
        assert '2024-12' in str(result)
    
    def test_parquet_extension(self):
        """Test that paths have .parquet extension."""
        date = pd.Timestamp('2020-03-31')
        base = Path('/test')
        result = get_decade_year_path(base, date)
        
        assert str(result).endswith('.parquet')


class TestFredSnapshotPath:
    """Tests for FRED snapshot paths."""
    
    def test_contains_fred_data(self, sample_snapshot_date):
        """Test that path contains fred_data directory."""
        result = get_fred_snapshot_path(sample_snapshot_date)
        
        assert 'fred_data' in str(result)
    
    def test_correct_date_format(self, sample_snapshot_date):
        """Test that date is formatted correctly."""
        result = get_fred_snapshot_path(sample_snapshot_date)
        
        assert '2020-03' in str(result)


class TestMasterSnapshotPath:
    """Tests for master snapshot paths."""
    
    def test_contains_master_snapshots(self, sample_snapshot_date):
        """Test that path contains master_snapshots directory."""
        result = get_master_snapshot_path(sample_snapshot_date)
        
        assert 'master_snapshots' in str(result)
    
    def test_contains_exogenous_data(self, sample_snapshot_date):
        """Test that path contains Exogenous_data directory."""
        result = get_master_snapshot_path(sample_snapshot_date)
        
        assert 'Exogenous_data' in str(result)


class TestTargetPath:
    """Tests for target data paths."""
    
    def test_nsa_first_release(self):
        """Test NSA first release path."""
        result = get_target_path('nsa', 'first')
        
        assert 'y_nsa_first_release.parquet' in str(result)
    
    def test_sa_last_release(self):
        """Test SA last release path."""
        result = get_target_path('sa', 'last')
        
        assert 'y_sa_last_release.parquet' in str(result)
    
    def test_contains_nfp_target(self):
        """Test that path contains NFP_target directory."""
        result = get_target_path('nsa', 'first')
        
        assert 'NFP_target' in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
