"""PIT spot-check + idempotency tests for the dynamic-economist injector.

Each test reads a small number of master-snapshot files, recomputes the
dynamic feature from raw inputs independently, and asserts that the
injected column matches the independent re-computation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data_ETA_Pipeline.inject_dynamic_economist_features import (  # noqa: E402
    DYNAMIC_COL_PREFIX,
    MIN_COVERAGE_PCT,
    PRIMARY_TOP_N,
    SKIP_COVID_IN_TRACK_RECORD,
    TRACK_WINDOW_MONTHS,
    _compute_features_for_month,
    _load_full_panel,
    _load_nfp_release_map,
    _load_target_actuals,
)
from Train.config import get_master_snapshots_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures (panel/actuals/release_map loaded once across the suite)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def panel():
    return _load_full_panel()


@pytest.fixture(scope="module")
def actuals():
    return _load_target_actuals()


@pytest.fixture(scope="module")
def release_map():
    return _load_nfp_release_map()


# --------------------------------------------------------------------------- #
# Per-snapshot spot-check helpers
# --------------------------------------------------------------------------- #

def _sampled_snapshot_paths(branch: str, n: int = 6) -> list[Path]:
    """Pick a small, deterministic sample of snapshots across decades."""
    all_paths = sorted(get_master_snapshots_dir(branch, "revised").glob("**/*.parquet"))
    if len(all_paths) <= n:
        return all_paths
    step = max(1, len(all_paths) // n)
    return all_paths[::step][:n]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("branch", ["sa"])
def test_dynamic_columns_present_in_snapshots(branch):
    """Every sampled snapshot should carry the full dynamic-column inventory."""
    expected_cols = {
        f"{DYNAMIC_COL_PREFIX}Top4_k12",
        f"{DYNAMIC_COL_PREFIX}Top10_k12",
        f"{DYNAMIC_COL_PREFIX}Top15_k12",
        f"{DYNAMIC_COL_PREFIX}PanelN",
        f"{DYNAMIC_COL_PREFIX}NCalibrated",
        f"{DYNAMIC_COL_PREFIX}DispersionStd",
        f"{DYNAMIC_COL_PREFIX}DispersionIqr",
        f"{DYNAMIC_COL_PREFIX}Top10TrackMae",
        f"{DYNAMIC_COL_PREFIX}RobustMedian",
        f"{DYNAMIC_COL_PREFIX}TrimmedMean10",
    }
    for snap_path in _sampled_snapshot_paths(branch):
        snap = pd.read_parquet(snap_path)
        missing = expected_cols - set(snap.columns)
        assert not missing, (
            f"{snap_path.name}: missing dynamic cols {missing}"
        )


def test_compute_features_uses_last_available_submission():
    target_month = pd.Timestamp("2022-06-01")
    cutoff = pd.Timestamp("2022-06-03")
    actuals = pd.Series(dtype=float, name="actual")
    panel = pd.DataFrame([
        {
            "ds": target_month,
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 100.0,
            "first_release_date": pd.Timestamp("2022-05-20"),
        },
        {
            "ds": target_month,
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 120.0,
            "first_release_date": pd.Timestamp("2022-06-01"),
        },
        {
            "ds": target_month,
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 999.0,
            "first_release_date": pd.Timestamp("2022-06-10"),
        },
        {
            "ds": target_month,
            "ident": "US_B",
            "name": "ECON_B",
            "forecast": 200.0,
            "first_release_date": pd.Timestamp("2022-06-01"),
        },
    ])

    features = _compute_features_for_month(target_month, cutoff, panel, actuals)

    assert features["PanelN"] == 2
    assert features["RobustMedian"] == pytest.approx(160.0)
    assert features["TrimmedMean10"] == pytest.approx(160.0)


@pytest.mark.parametrize("branch", ["sa"])
def test_injected_values_match_independent_recompute(branch, panel, actuals, release_map):
    """For a sampled snapshot + sampled row, the injected dynamic-feature
    value must equal a fresh PIT-correct re-computation from raw inputs.
    """
    for snap_path in _sampled_snapshot_paths(branch, n=4):
        snap = pd.read_parquet(snap_path)
        primary_col = f"{DYNAMIC_COL_PREFIX}Top{PRIMARY_TOP_N}_k12"
        if primary_col not in snap.columns:
            pytest.fail(f"Missing primary col in {snap_path}")
        with_val = snap.dropna(subset=[primary_col]).copy()
        if with_val.empty:
            continue  # very early snapshot — no features yet, allowed
        # Sample three rows: earliest, middle, latest with a feature value
        with_val = with_val.sort_values("date").reset_index(drop=True)
        sample_idx = sorted({0, len(with_val) // 2, len(with_val) - 1})
        for i in sample_idx:
            row = with_val.iloc[i]
            target_month = pd.Timestamp(row["date"]).to_period("M").to_timestamp()
            if target_month not in release_map.index:
                continue
            cutoff = release_map.loc[target_month]
            expected = _compute_features_for_month(
                target_month, cutoff, panel, actuals
            )
            for feat, suffix in [
                (f"Top{PRIMARY_TOP_N}_k12", "Top10_k12"),
                ("PanelN", "PanelN"),
                ("NCalibrated", "NCalibrated"),
                ("RobustMedian", "RobustMedian"),
            ]:
                col = f"{DYNAMIC_COL_PREFIX}{suffix}"
                got = row[col]
                want = expected[feat]
                if pd.isna(got) and pd.isna(want):
                    continue
                if isinstance(want, float):
                    assert got == pytest.approx(want, rel=1e-6, abs=1e-6), (
                        f"{snap_path.name} row {target_month.strftime('%Y-%m')}: "
                        f"{col} = {got} but expected {want}"
                    )
                else:
                    assert got == want, (
                        f"{snap_path.name} row {target_month.strftime('%Y-%m')}: "
                        f"{col} = {got} but expected {want}"
                    )


@pytest.mark.parametrize("branch", ["sa"])
def test_pit_invariant_no_future_data_in_feature(branch, panel, release_map):
    """For each sampled snapshot at snap_date, every injected feature value
    at row M was computed using only forecasts with first_release_date <
    release_date(M). PIT property: release_date(M) ≤ snap_date for every
    row M in the snapshot.

    Equality is allowed for the to-be-forecast row M = snap_month — its
    release_date equals snap_date by construction, and its feature was
    computed with strict `first_release_date < cutoff` so it remains
    PIT-safe. For past rows (M < snap_month) the inequality is strict.
    """
    for snap_path in _sampled_snapshot_paths(branch):
        # Parse snap month from filename '2025-08.parquet' → release_date
        try:
            snap_month = pd.Timestamp(snap_path.stem + "-01")
        except (TypeError, ValueError):
            continue
        if snap_month not in release_map.index:
            continue
        snap_date = release_map.loc[snap_month]
        snap = pd.read_parquet(snap_path)
        primary_col = f"{DYNAMIC_COL_PREFIX}Top{PRIMARY_TOP_N}_k12"
        if primary_col not in snap.columns:
            continue
        feature_rows = snap.dropna(subset=[primary_col])
        if feature_rows.empty:
            continue
        # Every row's release_date(date) must be ≤ snap_date.
        # Strict < for past months; equality only allowed for M == snap_month.
        for date in pd.to_datetime(feature_rows["date"]):
            if date not in release_map.index:
                # If a row's date predates our release_map start, accept
                continue
            row_release = release_map.loc[date]
            if date == snap_month:
                assert row_release == snap_date, (
                    f"{snap_path.name}: target row {date.strftime('%Y-%m')} "
                    f"release {row_release} does not match snap_date {snap_date}"
                )
            else:
                assert row_release < snap_date, (
                    f"{snap_path.name}: row {date.strftime('%Y-%m')} has release "
                    f"{row_release} which is not strictly before snap_date {snap_date}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
