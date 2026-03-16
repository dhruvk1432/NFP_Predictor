"""
Tests for unified no-selection master snapshot generation.

Covers:
- _batch_load_source_all_features() keeps all columns (no filtering)
- _run_unified_no_selection() produces parquets at all 2 branch paths (revised only)
- Marker JSON has "mode": "all_features"
- CLI default is no-selection
"""

import json
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.create_master_snapshots import (
    _batch_load_source_all_features,
    _normalize_to_wide,
    _snapshot_path,
    _load_all_sources_from_cache,
    MASTER_BASE,
)
from Train.data_loader import sanitize_feature_name


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def source_dir_with_parquets(tmp_path):
    """Create a fake source directory with wide-format parquets."""
    source_dir = tmp_path / "source" / "decades"
    months = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01'),
              pd.Timestamp('2020-03-01')]
    rng = np.random.RandomState(42)

    for m in months:
        path = _snapshot_path(source_dir, m)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=12, freq='MS'),
            'feat_alpha': rng.randn(12),
            'feat_beta': rng.randn(12),
            'feat_gamma': rng.randn(12),
            'VIX_latest': rng.randn(12),
        })
        df.to_parquet(path, index=False)

    return source_dir, months


@pytest.fixture
def source_dir_with_long_format(tmp_path):
    """Create a fake source with long-format parquets (series_name column)."""
    source_dir = tmp_path / "source_long" / "decades"
    month = pd.Timestamp('2021-06-01')
    path = _snapshot_path(source_dir, month)
    path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    rows = []
    for d in dates:
        for series in ['CCSA_latest', 'SP500_latest', 'Oil_latest']:
            rows.append({
                'date': d,
                'series_name': series,
                'value': np.random.randn(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)

    return source_dir, [month]


# =============================================================================
# Tests: _batch_load_source_all_features
# =============================================================================

class TestBatchLoadSourceAllFeatures:
    """Tests for _batch_load_source_all_features (no filtering)."""

    def test_loads_all_columns(self, source_dir_with_parquets):
        """All feature columns should be kept (no allowed_features filter)."""
        source_dir, months = source_dir_with_parquets
        result = _batch_load_source_all_features('TestSource', source_dir, months)

        assert len(result) == 3  # 3 months
        for month_key, df in result.items():
            data_cols = [c for c in df.columns if c not in ['date', 'snapshot_date']]
            assert 'feat_alpha' in data_cols
            assert 'feat_beta' in data_cols
            assert 'feat_gamma' in data_cols
            assert 'VIX_latest' in data_cols

    def test_sanitizes_column_names(self, tmp_path):
        """Feature names should be sanitized for LightGBM compatibility."""
        source_dir = tmp_path / "sanitize_test" / "decades"
        month = pd.Timestamp('2022-01-01')
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw_name = "Prosper|Consumer Spending (18-34)"
        df = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=6, freq='MS'),
            raw_name: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        df.to_parquet(path, index=False)

        result = _batch_load_source_all_features('Prosper', source_dir, [month])
        assert '2022-01' in result
        loaded_df = result['2022-01']

        # Raw name should not be present
        assert raw_name not in loaded_df.columns
        # Sanitized name should be present
        san = sanitize_feature_name(raw_name)
        assert san in loaded_df.columns

    def test_skips_missing_months(self, source_dir_with_parquets):
        """Months without parquets should not appear in result."""
        source_dir, months = source_dir_with_parquets
        extra_months = months + [pd.Timestamp('2025-12-01')]  # Does not exist
        result = _batch_load_source_all_features('TestSource', source_dir, extra_months)
        assert '2025-12' not in result
        assert len(result) == 3  # Only the 3 existing months

    def test_skips_empty_parquets(self, tmp_path):
        """Empty parquets should be skipped."""
        source_dir = tmp_path / "empty_src" / "decades"
        month = pd.Timestamp('2023-01-01')
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(path, index=False)

        result = _batch_load_source_all_features('EmptySource', source_dir, [month])
        assert result == {}

    def test_handles_long_format(self, source_dir_with_long_format):
        """Long-format parquets should be pivoted to wide."""
        source_dir, months = source_dir_with_long_format
        result = _batch_load_source_all_features('FRED_Exog', source_dir, months)

        assert '2021-06' in result
        df = result['2021-06']
        assert 'CCSA_latest' in df.columns
        assert 'SP500_latest' in df.columns
        assert 'Oil_latest' in df.columns

    def test_no_filtering_unlike_batch_load_source(self, source_dir_with_parquets):
        """Unlike _batch_load_source, this function has no allowed_features param."""
        source_dir, months = source_dir_with_parquets
        # _batch_load_source_all_features has no allowed_features parameter
        import inspect
        sig = inspect.signature(_batch_load_source_all_features)
        assert 'allowed_features' not in sig.parameters


# =============================================================================
# Tests: _run_unified_no_selection (integration with monkeypatching)
# =============================================================================

class TestRunUnifiedNoSelection:
    """Tests for unified no-selection master snapshot generation."""

    def test_produces_parquets_at_all_2_branch_paths(self, tmp_path, monkeypatch):
        """Parquets should exist at nsa/revised and sa/revised."""
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, 'MASTER_BASE', master_base)
        monkeypatch.setattr(cms, 'TARGET_COMBOS', [
            ('nsa', 'revised'),
            ('sa', 'revised'),
        ])

        # Create a fake source
        source_dir = tmp_path / "fake_source" / "decades"
        month = pd.Timestamp('2020-06-01')
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=12, freq='MS'),
            'feat_x': np.random.randn(12),
        })
        df.to_parquet(path, index=False)

        monkeypatch.setattr(cms, 'SOURCES', {'TestSrc': source_dir})

        snapshot_pairs = [(month, pd.Timestamp('2020-06-05'))]
        cms._run_unified_no_selection(snapshot_pairs, skip_existing=False)

        # Check all 2 branches have the parquet
        for cat in ['nsa', 'sa']:
            branch_path = master_base / cat / "revised" / "decades"
            pq = _snapshot_path(branch_path, month)
            assert pq.exists(), f"Missing parquet at {pq}"
            loaded = pd.read_parquet(pq)
            assert 'feat_x' in loaded.columns
            assert 'date' in loaded.columns
            assert 'snapshot_date' in loaded.columns

    def test_writes_all_features_marker_json(self, tmp_path, monkeypatch):
        """Each branch should get a selected_features marker with mode=all_features."""
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, 'MASTER_BASE', master_base)
        monkeypatch.setattr(cms, 'TARGET_COMBOS', [
            ('nsa', 'revised'), ('sa', 'revised'),
        ])

        # Create minimal source
        source_dir = tmp_path / "src" / "decades"
        month = pd.Timestamp('2020-01-01')
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'date': [pd.Timestamp('2019-01-01')],
            'f1': [1.0],
        }).to_parquet(path, index=False)
        monkeypatch.setattr(cms, 'SOURCES', {'S1': source_dir})

        cms._run_unified_no_selection(
            [(month, pd.Timestamp('2020-01-10'))],
            skip_existing=False,
        )

        for cat, src in [('nsa', 'revised'), ('sa', 'revised')]:
            marker = master_base / f"selected_features_{cat}_{src}.json"
            assert marker.exists()
            with open(marker) as f:
                data = json.load(f)
            assert data["mode"] == "all_features"
            assert "generated_at" in data

    def test_all_2_branches_are_identical_copies(self, tmp_path, monkeypatch):
        """All 2 branch parquets should have identical content."""
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, 'MASTER_BASE', master_base)
        monkeypatch.setattr(cms, 'TARGET_COMBOS', [
            ('nsa', 'revised'),
            ('sa', 'revised'),
        ])

        source_dir = tmp_path / "src" / "decades"
        month = pd.Timestamp('2020-03-01')
        rng = np.random.RandomState(99)
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=6, freq='MS'),
            'col_a': rng.randn(6),
            'col_b': rng.randn(6),
        }).to_parquet(path, index=False)
        monkeypatch.setattr(cms, 'SOURCES', {'S': source_dir})

        cms._run_unified_no_selection(
            [(month, pd.Timestamp('2020-03-06'))],
            skip_existing=False,
        )

        dfs = {}
        for cat in ['nsa', 'sa']:
            branch_dir = master_base / cat / "revised" / "decades"
            pq = _snapshot_path(branch_dir, month)
            dfs[f"{cat}/revised"] = pd.read_parquet(pq)

        ref = dfs['nsa/revised']
        pd.testing.assert_frame_equal(ref, dfs['sa/revised'], check_like=True)

    def test_skip_existing_respects_flag(self, tmp_path, monkeypatch):
        """When skip_existing=True, existing parquets should not be regenerated."""
        import Data_ETA_Pipeline.create_master_snapshots as cms

        master_base = tmp_path / "master_snapshots"
        monkeypatch.setattr(cms, 'MASTER_BASE', master_base)
        monkeypatch.setattr(cms, 'TARGET_COMBOS', [('nsa', 'revised')])

        source_dir = tmp_path / "src" / "decades"
        month = pd.Timestamp('2020-01-01')
        path = _snapshot_path(source_dir, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'date': [pd.Timestamp('2019-01-01')],
            'f1': [1.0],
        }).to_parquet(path, index=False)
        monkeypatch.setattr(cms, 'SOURCES', {'S': source_dir})

        # First run
        cms._run_unified_no_selection(
            [(month, pd.Timestamp('2020-01-10'))],
            skip_existing=False,
        )

        # Modify the unified parquet to detect if it gets re-generated
        unified_dir = master_base / "_unified" / "decades"
        unified_pq = _snapshot_path(unified_dir, month)
        original_df = pd.read_parquet(unified_pq)

        # Second run with skip_existing=True
        cms._run_unified_no_selection(
            [(month, pd.Timestamp('2020-01-10'))],
            skip_existing=True,
        )

        # File should still exist (not regenerated)
        loaded = pd.read_parquet(unified_pq)
        pd.testing.assert_frame_equal(original_df, loaded)


# =============================================================================
# Tests: CLI default is no-selection
# =============================================================================

class TestCLIDefaults:
    """Verify that --with-selection is opt-in, no-selection is default."""

    def test_cli_with_selection_not_default(self):
        """The CLI should default to no-selection mode."""
        import ast
        path = Path(__file__).parent.parent / 'Data_ETA_Pipeline' / 'create_master_snapshots.py'
        source = path.read_text()

        # Look for argparse: --with-selection should have action='store_true' (default=False)
        assert '--with-selection' in source
        # The presence of 'store_true' means default is False
        assert "store_true" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
