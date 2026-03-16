"""Tests for lean transform mode in utils/transforms.py."""
import numpy as np
import pandas as pd
import pytest

from utils.transforms import compute_all_features, compute_features_wide


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_long_df(n_months: int = 36, n_series: int = 3) -> pd.DataFrame:
    """Create a synthetic long-format DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n_series):
        name = f"series_{i}"
        vals = rng.randn(n_months).cumsum() + 100  # random walk
        for d, v in zip(dates, vals):
            rows.append({"date": d, "series_name": name, "value": v})
    return pd.DataFrame(rows)


def _make_long_with_symlog_and_pctchg(n_months: int = 36) -> pd.DataFrame:
    """Create long-format DF that mimics post-add_symlog + add_pct_change pipeline."""
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rng = np.random.RandomState(42)
    rows = []
    for name_base in ["A", "B"]:
        vals = rng.randn(n_months).cumsum() + 50
        symlog_vals = np.sign(vals) * np.log1p(np.abs(vals))
        pct_vals = np.concatenate([[np.nan], np.diff(vals) / vals[:-1] * 100])
        for d, v, sv, pv in zip(dates, vals, symlog_vals, pct_vals):
            rows.append({"date": d, "series_name": name_base, "value": v})
            rows.append({"date": d, "series_name": f"{name_base}_symlog", "value": sv})
            if not np.isnan(pv):
                rows.append({"date": d, "series_name": f"{name_base}_pct_chg", "value": pv})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_all_features — lean mode
# ---------------------------------------------------------------------------

class TestComputeAllFeaturesLean:
    """Tests for compute_all_features(lean=True)."""

    def test_lean_has_fewer_features(self):
        df = _make_long_with_symlog_and_pctchg()
        full = compute_all_features(df, lean=False)
        lean = compute_all_features(df, lean=True)
        full_names = set(full["series_name"].unique())
        lean_names = set(lean["series_name"].unique())
        assert len(lean_names) < len(full_names), (
            f"Lean ({len(lean_names)}) should have fewer features than full ({len(full_names)})"
        )

    def test_lean_excludes_symlog_series(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = lean["series_name"].unique()
        symlog_names = [n for n in lean_names if "_symlog" in n]
        assert len(symlog_names) == 0, f"Lean mode should not contain symlog features: {symlog_names}"

    def test_lean_excludes_diff_zscore_12m(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = lean["series_name"].unique()
        zscore12 = [n for n in lean_names if "diff_zscore_12m" in n]
        assert len(zscore12) == 0, f"Lean mode should not contain diff_zscore_12m: {zscore12}"

    def test_lean_excludes_level_zscores(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = lean["series_name"].unique()
        level_zs = [n for n in lean_names if "_zscore_3m" in n and "_diff_" not in n]
        assert len(level_zs) == 0, f"Lean mode should not contain level z-scores: {level_zs}"

    def test_lean_keeps_diff_zscore_3m(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = set(lean["series_name"].unique())
        assert any("_diff_zscore_3m" in n for n in lean_names), "Lean mode must keep diff_zscore_3m"

    def test_lean_keeps_rolling_stats(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = set(lean["series_name"].unique())
        assert any("_rolling_mean_3m" in n for n in lean_names), "Lean must keep rolling_mean_3m"
        assert any("_rolling_std_6m" in n for n in lean_names), "Lean must keep rolling_std_6m"

    def test_lean_keeps_multi_period_changes(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = set(lean["series_name"].unique())
        for period in [3, 6, 12]:
            assert any(f"_chg_{period}m" in n for n in lean_names), f"Lean must keep chg_{period}m"

    def test_lean_keeps_all_four_lags(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = set(lean["series_name"].unique())
        for lag in [1, 3, 6, 12]:
            assert any(f"_lag_{lag}m" in n for n in lean_names), f"Lean must keep lag_{lag}m"

    def test_lean_keeps_pct_chg_level(self):
        df = _make_long_with_symlog_and_pctchg()
        lean = compute_all_features(df, lean=True)
        lean_names = set(lean["series_name"].unique())
        # pct_chg level should be present (un-lagged)
        pct_chg_base = [n for n in lean_names if n.endswith("_pct_chg")]
        assert len(pct_chg_base) > 0, "Lean must keep base pct_chg features"

    def test_full_mode_keeps_symlog(self):
        df = _make_long_with_symlog_and_pctchg()
        full = compute_all_features(df, lean=False)
        full_names = full["series_name"].unique()
        symlog_names = [n for n in full_names if "_symlog" in n]
        assert len(symlog_names) > 0, "Full mode must keep symlog features"

    def test_lean_skip_series_still_works(self):
        df = _make_long_with_symlog_and_pctchg()
        skip = frozenset({"A"})
        lean = compute_all_features(df, skip_series=skip, lean=True)
        lean_names = set(lean["series_name"].unique())
        # Series "A" should only appear as level (+ lags), no diff/rolling
        a_features = [n for n in lean_names if n == "A" or n.startswith("A_lag_")]
        a_derived = [n for n in lean_names
                     if n.startswith("A_") and n not in a_features
                     and "_pct_chg" not in n and "_symlog" not in n]
        assert len(a_derived) == 0, f"Skipped series should have no derived features: {a_derived}"


# ---------------------------------------------------------------------------
# compute_features_wide — lean mode
# ---------------------------------------------------------------------------

class TestComputeFeaturesWideLean:
    """Tests for compute_features_wide(lean=True)."""

    def test_lean_has_fewer_columns(self):
        df = _make_long_df()
        full = compute_features_wide(df, apply_mom=True, lean=False)
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        assert lean.shape[1] < full.shape[1], (
            f"Lean ({lean.shape[1]} cols) should have fewer columns than full ({full.shape[1]} cols)"
        )

    def test_lean_no_symlog_columns(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        symlog_cols = [c for c in lean.columns if "_symlog" in c]
        assert len(symlog_cols) == 0, f"Lean should have no symlog columns: {symlog_cols[:5]}..."

    def test_lean_no_diff_zscore_12m(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        bad = [c for c in lean.columns if "diff_zscore_12m" in c]
        assert len(bad) == 0, f"Lean should have no diff_zscore_12m: {bad[:5]}..."

    def test_lean_no_level_zscores(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        bad = [c for c in lean.columns if "_zscore_3m" in c and "_diff_" not in c]
        assert len(bad) == 0, f"Lean should have no level z-scores: {bad[:5]}..."

    def test_lean_has_diff_zscore_3m(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        good = [c for c in lean.columns if "diff_zscore_3m" in c]
        assert len(good) > 0, "Lean must have diff_zscore_3m columns"

    def test_lean_has_pct_chg(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        good = [c for c in lean.columns if "_pct_chg" in c]
        assert len(good) > 0, "Lean must have pct_chg columns"

    def test_lean_has_all_lags(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        for lag in [1, 3, 6, 12]:
            lagged = [c for c in lean.columns if f"_lag_{lag}m" in c]
            assert len(lagged) > 0, f"Lean must have lag_{lag}m columns"

    def test_lean_has_rolling_stats(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        rm = [c for c in lean.columns if "_rolling_mean_3m" in c]
        rs = [c for c in lean.columns if "_rolling_std_6m" in c]
        assert len(rm) > 0, "Lean must have rolling_mean_3m"
        assert len(rs) > 0, "Lean must have rolling_std_6m"

    def test_lean_has_multi_period_changes(self):
        df = _make_long_df()
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        for period in [3, 6, 12]:
            chg = [c for c in lean.columns if f"_chg_{period}m" in c]
            assert len(chg) > 0, f"Lean must have chg_{period}m"

    def test_lean_feature_count_per_series(self):
        """Verify lean mode produces ~45 features per series (9 base × 5 lag depths)."""
        df = _make_long_df(n_months=36, n_series=1)
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        n_cols = lean.shape[1]
        # 1 series → base features: level(1) + pct_chg(1) + diff(1) + diff_zscore_3m(1)
        #   + rolling_mean_3m(1) + rolling_std_6m(1) + chg_3m(1) + chg_6m(1) + chg_12m(1) = 9
        # × 5 lag depths (0, 1m, 3m, 6m, 12m) = 45
        assert n_cols == 45, f"Expected 45 features for 1 series in lean mode, got {n_cols}"

    def test_full_feature_count_sanity(self):
        """Verify full mode has substantially more features than lean."""
        df = _make_long_df(n_months=36, n_series=1)
        full = compute_features_wide(df, apply_mom=True, lean=False)
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        # Full should have at least 2x more features per series
        assert full.shape[1] >= lean.shape[1] * 2, (
            f"Full ({full.shape[1]}) should be at least 2x lean ({lean.shape[1]})"
        )

    def test_lean_same_index_as_full(self):
        """Lean and full should produce the same date index."""
        df = _make_long_df()
        full = compute_features_wide(df, apply_mom=True, lean=False)
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        pd.testing.assert_index_equal(full.index, lean.index)

    def test_lean_values_match_full_for_shared_columns(self):
        """Where column names overlap, lean and full values must be identical."""
        df = _make_long_df(n_months=36, n_series=2)
        full = compute_features_wide(df, apply_mom=True, lean=False)
        lean = compute_features_wide(df, apply_mom=True, lean=True)
        shared = set(lean.columns) & set(full.columns)
        assert len(shared) > 0, "Should have shared columns between lean and full"
        for col in sorted(shared):
            pd.testing.assert_series_equal(
                lean[col], full[col], check_names=False,
                obj=f"lean[{col}]",
            )
