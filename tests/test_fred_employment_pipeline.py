"""
Tests for FRED employment release-date imputation and target window validation.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.fred_employment_pipeline import (
    _target_file_covers_window,
    calculate_complex_release_date,
)


def test_calculate_complex_release_date_uses_strictly_prior_candidate():
    """
    For pre-2009 stale vintages, Option 3 should pick a date strictly before
    snapshot_date (not equal), so rows survive strict '< as_of_cutoff' filtering.
    """
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("1998-10-01")],
            "release_date": [pd.Timestamp("2008-09-23")],
        }
    )

    out, imputed_count = calculate_complex_release_date(df, pd.Timestamp("1998-12-04"))

    assert imputed_count == 1
    assert out.loc[0, "release_date"] == pd.Timestamp("1998-11-06")


def test_calculate_complex_release_date_imputes_pre_2009_late_release():
    """
    Pre-2009 releases far beyond the normal first-Friday window should be
    imputed even when lag is less than 1 year.
    """
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2008-07-01")],
            "release_date": [pd.Timestamp("2008-09-23")],
        }
    )

    out, imputed_count = calculate_complex_release_date(df, pd.Timestamp("2008-09-05"))

    assert imputed_count == 1
    assert out.loc[0, "release_date"] == pd.Timestamp("2008-08-01")


def test_calculate_complex_release_date_preserves_post_2009_actual_release():
    """Post-2009 releases should not be rewritten by pre-2009 imputation rules."""
    original_release = pd.Timestamp("2010-03-15")
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2010-01-01")],
            "release_date": [original_release],
        }
    )

    out, imputed_count = calculate_complex_release_date(df, pd.Timestamp("2010-04-01"))

    assert imputed_count == 0
    assert out.loc[0, "release_date"] == original_release


def test_target_file_covers_window_detects_stale_start_date(tmp_path: Path):
    """Coverage helper should reject target files that start after requested window."""
    stale_path = tmp_path / "stale.parquet"
    stale_df = pd.DataFrame({"ds": pd.date_range("2001-01-01", "2026-02-01", freq="MS")})
    stale_df.to_parquet(stale_path, index=False)

    assert not _target_file_covers_window(stale_path, "1990-01-01", "2026-02-01")

    full_path = tmp_path / "full.parquet"
    full_df = pd.DataFrame({"ds": pd.date_range("1990-01-01", "2026-02-01", freq="MS")})
    full_df.to_parquet(full_path, index=False)

    assert _target_file_covers_window(full_path, "1990-01-01", "2026-02-01")

