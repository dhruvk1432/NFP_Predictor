"""
Tests for VIX_monthly feature computation and aggregation.

VIX_monthly = VIX^2 / 12 (annualized variance -> monthly variance)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.load_fred_exogenous import (
    compute_vix_daily_features,
    aggregate_vix_to_monthly,
)


def _make_vix_df(n_days=300, start="2020-01-01", seed=42):
    """Create synthetic daily VIX DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="B")
    values = 15 + rng.standard_normal(n_days).cumsum() * 0.5
    values = np.clip(values, 10, 80)
    return pd.DataFrame({"date": dates, "value": values})


# ---------------------------------------------------------------------------
# compute_vix_daily_features
# ---------------------------------------------------------------------------

class TestComputeVixDailyFeatures:
    def setup_method(self):
        self.df = _make_vix_df()
        self.daily = compute_vix_daily_features(self.df)

    def test_vix_monthly_column_exists(self):
        assert "vix_monthly" in self.daily.columns

    def test_vix_monthly_formula(self):
        """VIX_monthly must equal VIX^2 / 12."""
        expected = self.df.set_index("date").sort_index()["value"] ** 2 / 12
        pd.testing.assert_series_equal(
            self.daily["vix_monthly"],
            expected.rename("vix_monthly"),
            check_names=True,
        )

    def test_vix_monthly_always_positive(self):
        assert (self.daily["vix_monthly"] > 0).all()

    def test_vix_monthly_daily_chg_is_diff(self):
        expected = self.daily["vix_monthly"].diff()
        pd.testing.assert_series_equal(
            self.daily["vix_monthly_daily_chg"], expected, check_names=False
        )

    def test_vix_monthly_spike_ratio_columns_exist(self):
        assert "vix_monthly_spike_ratio" in self.daily.columns
        assert "vix_monthly_spike_5d" in self.daily.columns

    def test_vix_monthly_spike_ratio_formula(self):
        """spike_ratio = vix_monthly / vix_monthly_21_days_ago."""
        expected = self.daily["vix_monthly"] / self.daily["vix_monthly"].shift(21)
        pd.testing.assert_series_equal(
            self.daily["vix_monthly_spike_ratio"], expected, check_names=False
        )

    def test_vix_monthly_spike_5d_formula(self):
        """spike_5d = vix_monthly / vix_monthly_5_days_ago."""
        expected = self.daily["vix_monthly"] / self.daily["vix_monthly"].shift(5)
        pd.testing.assert_series_equal(
            self.daily["vix_monthly_spike_5d"], expected, check_names=False
        )

    def test_original_vix_columns_still_present(self):
        """Ensure existing VIX columns are unaffected."""
        for col in ["daily_chg", "vix_spike_ratio", "vix_spike_5d"]:
            assert col in self.daily.columns

    def test_intermediate_shift_columns_present(self):
        """Intermediate shift columns are retained (consistent with original VIX behavior)."""
        for col in ["vix_monthly_30d_ago", "vix_monthly_5d_ago", "vix_30d_ago", "vix_5d_ago"]:
            assert col in self.daily.columns, f"Expected intermediate column '{col}' to be present"


# ---------------------------------------------------------------------------
# aggregate_vix_to_monthly
# ---------------------------------------------------------------------------

class TestAggregateVixToMonthly:
    def setup_method(self):
        df = _make_vix_df(n_days=400)
        daily = compute_vix_daily_features(df)
        self.result = aggregate_vix_to_monthly(daily)
        self.series_names = self.result["series_name"].unique().tolist()

    def test_vix_monthly_series_present(self):
        expected = {
            "VIX_monthly_mean",
            "VIX_monthly_max",
            "VIX_monthly_volatility",
            "VIX_monthly_30d_spike",
            "VIX_monthly_max_5d_spike",
        }
        missing = expected - set(self.series_names)
        assert not missing, f"Missing series: {missing}"

    def test_original_vix_series_still_present(self):
        original = {"VIX_mean", "VIX_max", "VIX_volatility", "VIX_30d_spike", "VIX_max_5d_spike",
                    "VIX_panic_regime", "VIX_high_regime"}
        missing = original - set(self.series_names)
        assert not missing, f"Original VIX series missing: {missing}"

    def test_result_has_required_columns(self):
        for col in ["date", "series_name", "value", "release_date"]:
            assert col in self.result.columns

    def test_vix_monthly_mean_geq_zero(self):
        """VIX_monthly = VIX^2/12 is always positive, so mean must be > 0."""
        monthly_mean = self.result[self.result["series_name"] == "VIX_monthly_mean"]["value"]
        assert (monthly_mean.dropna() > 0).all()

    def test_vix_monthly_max_geq_mean(self):
        """Monthly max must be >= monthly mean for the same month."""
        mean_df = self.result[self.result["series_name"] == "VIX_monthly_mean"].set_index("date")["value"]
        max_df = self.result[self.result["series_name"] == "VIX_monthly_max"].set_index("date")["value"]
        common = mean_df.index.intersection(max_df.index)
        assert (max_df.loc[common].values >= mean_df.loc[common].values - 1e-9).all()

    def test_release_date_is_month_end(self):
        """release_date should be the last calendar day of the observation month."""
        sub = self.result[self.result["series_name"] == "VIX_monthly_mean"].dropna(subset=["value"])
        for _, row in sub.head(5).iterrows():
            assert row["release_date"] == row["date"] + pd.offsets.MonthEnd(0)

    def test_vix_monthly_values_scale_with_vix(self):
        """VIX_monthly_mean should be close to mean(VIX^2)/12 per month."""
        df = _make_vix_df(n_days=400)
        daily = compute_vix_daily_features(df)
        result = aggregate_vix_to_monthly(daily)

        # Compute expected: resample raw vix^2/12 by month
        vix_sq_monthly = (daily["value"] ** 2 / 12).resample("MS").mean()

        monthly_mean = result[result["series_name"] == "VIX_monthly_mean"].set_index("date")["value"]
        common = vix_sq_monthly.index.intersection(monthly_mean.index)

        np.testing.assert_allclose(
            monthly_mean.loc[common].values,
            vix_sq_monthly.loc[common].values,
            rtol=1e-6,
        )

    def test_no_duplicate_series_names(self):
        """Each series_name/date pair should be unique."""
        dupes = self.result.groupby(["date", "series_name"]).size()
        assert (dupes == 1).all(), "Duplicate (date, series_name) rows found"
