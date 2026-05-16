"""
PIT-safe SA-NSA gap (realized seasonal adjustment) features for master snapshots.

The Kalman fusion's model channel is ``champion_pred = nsa_pred + adj_pred``
where ``adj_pred`` is an ExpWeightedMedian over historical realized adjustments
``adj[m] = SA_revised[m] - NSA_revised[m]``. Feature-engineering the *shape* of
that realized adjustment series lets the NSA LightGBM see what the seasonal
adjustment is doing -- trend, volatility, seasonal strength -- and potentially
compensate for the median-smoother's lag.

PIT invariants
--------------
For a master snapshot dated ``snapshot_date`` (the ``snapshot_date`` column of
the assembled wide DataFrame), a feature value at row ``date = d`` may only
use adjustment-history rows with both:

  * ``operational_available_date < snapshot_date`` -- the adjustment was
    publicly available before the snapshot was taken.
  * ``ds < d``                                     -- no future leakage for
    the calendar month being modelled.

The combined effect: for old rows ``d << snapshot_date`` features are stable;
for the most recent rows the realised-adjustment lookback may be partly NaN
because that month's revised adjustment was not yet published when the
snapshot was sampled. That NaN behaviour mirrors how the rest of the master
snapshot already handles unpublished values, and is exactly what we want.

Columns produced (all prefixed ``sanagap_`` so they flow through the
sanitiser unchanged and are easy to filter out for diagnostics):

  * ``sanagap_adj_lag1``                -- realised adjustment at d-1 month
  * ``sanagap_adj_lag12``               -- realised adjustment at d-12 months
  * ``sanagap_adj_mom``                 -- (adj[d-1] - adj[d-2])
  * ``sanagap_adj_seasonal_strength``   -- var(calendar-month means) / var(all)
                                           over the lookback up to d-1
  * ``sanagap_adj_trend_slope_24m``     -- OLS slope of adj over the 24 months
                                           ending at d-1
  * ``sanagap_adj_vol_12m``             -- std of adj over the 12 months
                                           ending at d-1

The output DataFrame's ``date`` column is monotonically increasing month-starts
so callers can merge directly on ``date`` against an existing master snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

# COVID months excluded from the realized-adjustment series so a single
# pandemic shock does not dominate the rolling-window features. Mirrors
# the exclusion used by ExpWeightedMedianCovidExcludedPredictor.
_COVID_EXCLUSION_START = pd.Timestamp("2020-03-01")
_COVID_EXCLUSION_END = pd.Timestamp("2020-12-31")

FEATURE_PREFIX = "sanagap_"

FEATURE_COLS: tuple[str, ...] = (
    f"{FEATURE_PREFIX}adj_lag1",
    f"{FEATURE_PREFIX}adj_lag12",
    f"{FEATURE_PREFIX}adj_mom",
    f"{FEATURE_PREFIX}adj_seasonal_strength",
    f"{FEATURE_PREFIX}adj_trend_slope_24m",
    f"{FEATURE_PREFIX}adj_vol_12m",
)


def _drop_covid_rows(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Drop COVID months from a long-format adjustment series."""
    if df.empty:
        return df
    mask = (df[date_col] >= _COVID_EXCLUSION_START) & (df[date_col] <= _COVID_EXCLUSION_END)
    return df.loc[~mask].copy()


def _pit_filter(
    adj_history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict the adjustment series to rows that were publicly available
    before ``snapshot_date``. Sorted by ``ds`` ascending.
    """
    if "operational_available_date" not in adj_history.columns:
        raise ValueError(
            "adj_history is missing 'operational_available_date'; cannot enforce PIT."
        )

    pit = adj_history[
        adj_history["operational_available_date"].notna()
        & (adj_history["operational_available_date"] < snapshot_date)
    ].copy()
    return pit.sort_values("ds").reset_index(drop=True)


def _build_features_from_series(
    adj_by_ds: pd.Series,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Vectorised feature construction.

    ``adj_by_ds`` is a Series indexed by month-start ``ds`` whose values are
    the realised adjustment (SA-NSA). It is assumed to already be
    PIT-filtered and COVID-excluded.

    For each ``d`` in ``target_dates`` we look up adj_by_ds with strict
    ``ds < d`` -- features at ``d`` never see the calendar-month-``d``
    adjustment itself.
    """
    if adj_by_ds.empty or len(target_dates) == 0:
        empty = pd.DataFrame({c: pd.Series(dtype=float) for c in FEATURE_COLS})
        empty.insert(0, "date", pd.to_datetime([]))
        return empty

    # Reindex to a contiguous monthly grid so lag/rolling ops are well-defined.
    full_index = pd.date_range(
        start=adj_by_ds.index.min().to_period("M").to_timestamp(),
        end=adj_by_ds.index.max().to_period("M").to_timestamp(),
        freq="MS",
    )
    series = adj_by_ds.reindex(full_index)

    # ── lag features --------------------------------------------------------
    # adj[d-1] and adj[d-12] are the value AT calendar months one and twelve
    # before d. shift(1)/shift(12) on a Series indexed by date yields exactly
    # that. The strict-less-than (ds < d) PIT requirement is automatically
    # satisfied because shift uses earlier indices than d.
    lag1 = series.shift(1)
    lag2 = series.shift(2)
    lag12 = series.shift(12)
    mom = lag1 - lag2

    # ── trailing-window stats (ending at d-1) -----------------------------
    # Build off ``lag1`` so the latest sample is adj[d-1]. The rolling window
    # then naturally covers [d-N, ..., d-1] without ever touching adj[d].
    vol_12m = lag1.rolling(window=12, min_periods=6).std()

    # 24-month trend slope: OLS slope of values vs an integer index. Closed
    # form: cov(x, y) / var(x) for x = 0..23.
    def _slope_24(window: np.ndarray) -> float:
        valid = ~np.isnan(window)
        if int(valid.sum()) < 8:
            return np.nan
        y = window[valid]
        x = np.arange(window.size, dtype=float)[valid]
        x_mean = x.mean()
        y_mean = y.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 0.0:
            return np.nan
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    trend_slope_24m = lag1.rolling(window=24, min_periods=8).apply(_slope_24, raw=True)

    # ── seasonal strength (expanding, ending at d-1) ----------------------
    # For each d, take all observations with ds < d, drop NaNs, and compute
    # var(calendar-month means) / var(all). Bounded in [0, 1]. We
    # implement it manually with a running per-calendar-month accumulator so
    # the cost is O(N) over the entire series, not O(N * window_size).
    seasonal_strength = pd.Series(np.nan, index=series.index, dtype=float)

    # Per calendar month: running sum, sum-of-squares, count over ds < d.
    month_sum = np.zeros(12, dtype=float)
    month_sumsq = np.zeros(12, dtype=float)
    month_count = np.zeros(12, dtype=np.int64)

    running_sum = 0.0
    running_sumsq = 0.0
    running_count = 0

    values = series.values  # numpy aligned to series.index
    months_of_year = series.index.month.values - 1  # 0..11

    for i in range(len(values)):
        # First, compute seasonal_strength using stats accumulated from ds < d
        # (i.e. observations at index 0..i-1).
        if running_count >= 24:
            # Per-calendar-month means, weighted by count (matches plain
            # within-group mean).
            means = np.where(month_count > 0, month_sum / np.maximum(month_count, 1), np.nan)
            valid_means = means[~np.isnan(means)]
            if valid_means.size >= 6:
                # Variance of group means weighted by group size.
                # (We use the simple unweighted mean variance — the calendar
                # months are by construction roughly equally represented.)
                between = float(np.nanvar(means, ddof=0))
                overall = float(running_sumsq / running_count - (running_sum / running_count) ** 2)
                if overall > 0.0:
                    seasonal_strength.iat[i] = float(np.clip(between / overall, 0.0, 1.0))

        # Now incorporate observation i so it contributes to ds < d for the
        # NEXT iteration (i.e. d > current index).
        v = values[i]
        if np.isfinite(v):
            m = int(months_of_year[i])
            month_sum[m] += v
            month_sumsq[m] += v * v
            month_count[m] += 1
            running_sum += v
            running_sumsq += v * v
            running_count += 1

    # Assemble the feature frame, indexed by the full monthly grid, then
    # reindex to the requested target_dates.
    feat = pd.DataFrame(
        {
            f"{FEATURE_PREFIX}adj_lag1": lag1,
            f"{FEATURE_PREFIX}adj_lag12": lag12,
            f"{FEATURE_PREFIX}adj_mom": mom,
            f"{FEATURE_PREFIX}adj_seasonal_strength": seasonal_strength,
            f"{FEATURE_PREFIX}adj_trend_slope_24m": trend_slope_24m,
            f"{FEATURE_PREFIX}adj_vol_12m": vol_12m,
        }
    )
    feat.index.name = "date"

    feat = feat.reindex(pd.DatetimeIndex(target_dates).normalize())
    feat = feat.reset_index().rename(columns={"index": "date"})
    return feat


def build_sa_nsa_gap_features_for_snapshot(
    snapshot_date: pd.Timestamp,
    *,
    target_dates: Optional[pd.DatetimeIndex] = None,
    history_start: pd.Timestamp = pd.Timestamp("1990-01-01"),
    adj_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return PIT-safe SA-NSA gap features for a master snapshot.

    Args:
        snapshot_date: The ``snapshot_date`` of the master snapshot being
            built. Any adjustment value with
            ``operational_available_date >= snapshot_date`` is excluded.
        target_dates: The calendar months (``date`` column) to emit features
            for. If None, emits a contiguous monthly grid from ``history_start``
            up to and including ``snapshot_date - 1 month``.
        history_start: Lower bound on emitted ``date`` rows. Older rows are
            dropped because the master snapshot already floors to 1990.
        adj_history: Optional pre-loaded adjustment history (DataFrame with
            columns ``[ds, nsa_mom, sa_mom, adjustment,
            operational_available_date]``). If omitted, it is loaded fresh
            via ``Train.sandbox.experiment_predicted_adjustment.load_adjustment_history``.

    Returns:
        DataFrame with one row per target date, columns
        ``[date, sanagap_adj_lag1, sanagap_adj_lag12, sanagap_adj_mom,
        sanagap_adj_seasonal_strength, sanagap_adj_trend_slope_24m,
        sanagap_adj_vol_12m]``. Rows with all-NaN features are still emitted
        so the caller can merge on ``date`` without losing rows.
    """
    snapshot_date = pd.Timestamp(snapshot_date)

    if adj_history is None:
        from Train.sandbox.experiment_predicted_adjustment import load_adjustment_history
        adj_history = load_adjustment_history()

    pit_adj = _pit_filter(adj_history, snapshot_date)
    pit_adj = _drop_covid_rows(pit_adj, date_col="ds")
    pit_adj["ds"] = pd.to_datetime(pit_adj["ds"]).dt.normalize()

    if pit_adj.empty:
        # No usable adjustment history yet (very early snapshots). Return an
        # all-NaN frame so the merge downstream still works.
        if target_dates is None:
            return pd.DataFrame({"date": pd.to_datetime([])} | {c: [] for c in FEATURE_COLS})
        out = pd.DataFrame({"date": pd.DatetimeIndex(target_dates).normalize()})
        for c in FEATURE_COLS:
            out[c] = np.nan
        return out

    adj_by_ds = pit_adj.set_index("ds")["adjustment"].astype(float)
    # Collapse duplicates (one revision per ds expected, but defensive).
    adj_by_ds = adj_by_ds[~adj_by_ds.index.duplicated(keep="last")]

    if target_dates is None:
        # Default span: history_start .. snapshot_date - 1 month. The master
        # snapshot's most recent ``date`` row is the month just before
        # ``snapshot_date`` (the observation month), and rows further out are
        # OOS placeholders. Returning a generous range keeps the merge simple.
        end = (snapshot_date - pd.DateOffset(months=1)).to_period("M").to_timestamp()
        target_dates = pd.date_range(
            start=pd.Timestamp(history_start).to_period("M").to_timestamp(),
            end=end,
            freq="MS",
        )

    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    target_dates = target_dates[target_dates >= pd.Timestamp(history_start)]

    feat = _build_features_from_series(adj_by_ds, target_dates)
    return feat


__all__ = [
    "FEATURE_COLS",
    "FEATURE_PREFIX",
    "build_sa_nsa_gap_features_for_snapshot",
]
