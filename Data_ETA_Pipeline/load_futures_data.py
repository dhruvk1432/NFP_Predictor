"""
PIT-correct futures feature pipeline for NFP prediction.

Selection rationale (literature-driven; see Andersen et al. 2003; Fleming &
Remolona 1997; Kuttner 2001; Gürkaynak/Sack/Swanson 2005; Faust et al. 2007;
Bekaert et al. 2013):

    Rate expectations (direct Fed reaction function):
      - 30-day Fed Funds (ZQ), 3-month SOFR (SR3)
    Treasury curve (growth/inflation pricing):
      - 2Y/5Y/10Y/30Y T-Notes/Bonds (ZT/ZF/ZN/ZB)
    Equity sentiment (small-cap = labor-sensitive):
      - S&P 500 (ES), Nasdaq 100 (NQ), Russell 2000 (RTY)
    FX (rate-differential channel + safe haven):
      - Dollar Index (DX), Euro (6E), Yen (6J)
    Volatility / risk premium:
      - VIX futures (VX)
    Industrial commodities (real-activity proxies):
      - Copper (HG, "Dr. Copper"), WTI (CL), Brent (BRN), Nat Gas (NG)
    Precious metals (real-rate / risk-off complement):
      - Gold (GC), Silver (SI)

Variant choice (plain vs _CCB back-adjusted):
    - Returns / momentum / realized vol: derived from `_CCB` daily series.
      Back-adjustment removes roll-induced price discontinuities that would
      otherwise inject spurious returns at contract rolls.
    - Levels (price/yield interpretation): use the plain (non-back-adjusted)
      series. For rate futures (ZQ, SR3, ZT-ZB), the plain close is in the
      `100 - implied_rate` convention; back-adjustment destroys that.

Output:
    DATA_PATH/Exogenous_data/exogenous_futures_data/decades/{decade}s/{year}/{YYYY-MM}.parquet

Each monthly snapshot contains long-format rows:
    columns: date, release_date, value, series_name, series_code, snapshot_date
    + pct_change copies and lean engineered features (consistent with
      load_unifier_data.py).

Per-future monthly observations are anchored to the *last trading day of the
calendar month*. For a snapshot at snap_date (NFP release for that target
month), only monthly rows with release_date < snap_date are included
(strictly PIT-correct).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import (
    DATA_PATH,
    TEMP_DIR,
    setup_logger,
    START_DATE,
    END_DATE,
)
from Data_ETA_Pipeline.fred_employment_pipeline import get_nfp_release_map
from Data_ETA_Pipeline.perf_stats import (
    install_hooks,
    profiled,
    register_atexit_dump,
)
from Data_ETA_Pipeline.utils import get_snapshot_path
from utils.transforms import add_pct_change_copies, compute_all_features

logger = setup_logger(__file__, TEMP_DIR)
install_hooks()
register_atexit_dump("load_futures_data", output_dir=TEMP_DIR / "perf")


# (display_name, ticker, asset_class). Asset class controls feature emission:
#   - "rate": emit implied_rate (100 - close) on the plain series (no CCB level).
#   - others: standard close + return + realized vol.
FUTURES: Dict[str, Tuple[str, str]] = {
    "FedFunds":     ("&ZQ",  "rate"),
    "SOFR_3M":      ("&SR3", "rate"),
    "Treasury_2Y":  ("&ZT",  "rate"),
    "Treasury_5Y":  ("&ZF",  "rate"),
    "Treasury_10Y": ("&ZN",  "rate"),
    "Treasury_30Y": ("&ZB",  "rate"),
    "SP500":        ("&ES",  "equity"),
    "Nasdaq100":    ("&NQ",  "equity"),
    "Russell2000":  ("&RTY", "equity"),
    "DollarIndex":  ("&DX",  "fx"),
    "EUR_USD":      ("&6E",  "fx"),
    "JPY_USD":      ("&6J",  "fx"),
    "VIX":          ("&VX",  "vol"),
    "Copper":       ("&HG",  "commodity"),
    "WTI_Crude":    ("&CL",  "commodity"),
    "Brent_Crude":  ("&BRN", "commodity"),
    "NatGas":       ("&NG",  "commodity"),
    "Gold":         ("&GC",  "commodity"),
    "Silver":       ("&SI",  "commodity"),
}

# Realized vol can sit near zero in calm regimes; pct_change of vol crossing zero
# produces noisy signal. Add their monthly_vol series_names to the skip set.
ZERO_CENTERED_PREFIXES = frozenset({
    "VIX_close",  # already a vol level — pct_change handles non-zero
})


FUTURES_DIR = Path(__file__).resolve().parent.parent / "continuous_futures"
SNAPSHOT_BASE = DATA_PATH / "Exogenous_data" / "exogenous_futures_data"


# ---------- Daily CSV loading -----------------------------------------------

def _read_futures_csv(ticker: str, suffix: str = "") -> pd.DataFrame:
    """Read one continuous-futures CSV and return tidy daily OHLCV."""
    path = FUTURES_DIR / f"{ticker}{suffix}.csv"
    if not path.exists():
        logger.warning(f"Futures CSV missing: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }, inplace=True)
    return df


# ---------- Monthly feature aggregation -------------------------------------

def _monthly_from_daily(plain: pd.DataFrame, ccb: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly per-future features.

    Returns a DataFrame with columns:
        month_start (1st of calendar month, normalized)
        release_date (last trading day of that calendar month)
        close          - last close from PLAIN series (level)
        log_return     - log(close_CCB / prev_close_CCB), close-to-close monthly
        realized_vol   - sqrt(252) * stdev(daily log returns from CCB intra-month)
        log_range      - log(monthly_high / monthly_low) from CCB
    """
    if plain.empty or ccb.empty:
        return pd.DataFrame()

    plain = plain.copy()
    ccb = ccb.copy()
    plain["month"] = plain["date"].dt.to_period("M")
    ccb["month"] = ccb["date"].dt.to_period("M")

    # Daily log returns on CCB (back-adjusted, so roll-clean)
    ccb["log_ret_d"] = np.log(ccb["close"]).diff()

    monthly_plain = plain.groupby("month").agg(
        close=("close", "last"),
        last_date=("date", "last"),
    )
    monthly_ccb = ccb.groupby("month").agg(
        close_ccb=("close", "last"),
        high_ccb=("high", "max"),
        low_ccb=("low", "min"),
        sigma_d=("log_ret_d", "std"),
        n_days=("log_ret_d", "count"),
    )

    out = monthly_plain.join(monthly_ccb, how="inner")
    if out.empty:
        return out

    # Close-to-close monthly log return (uses CCB to avoid roll artifacts)
    out["log_return"] = np.log(out["close_ccb"]).diff()
    # Annualized realized vol from intra-month daily log returns
    out["realized_vol"] = out["sigma_d"] * np.sqrt(252)
    # Log range — must be guarded against zero/neg
    safe_lo = out["low_ccb"].where(out["low_ccb"] > 0)
    out["log_range"] = np.log(out["high_ccb"] / safe_lo)

    # Drop early rows with insufficient samples for vol (<5 daily obs in month)
    out = out[out["n_days"] >= 5]

    out = out.reset_index().rename(columns={"month": "month_p"})
    out["month_start"] = out["month_p"].dt.to_timestamp()
    out["release_date"] = out["last_date"]

    return out[["month_start", "release_date", "close", "log_return", "realized_vol", "log_range"]]


def _long_format_rows(monthly: pd.DataFrame, display_name: str, ticker: str,
                      asset_class: str) -> pd.DataFrame:
    """Reshape per-future monthly features into snapshot long-format rows."""
    if monthly.empty:
        return pd.DataFrame()

    pieces = []
    base = monthly[["month_start", "release_date"]].rename(
        columns={"month_start": "date"}
    )

    def _emit(value_series: pd.Series, name: str) -> pd.DataFrame:
        chunk = base.copy()
        chunk["value"] = value_series.values
        chunk["series_name"] = f"{display_name}_{name}"
        chunk["series_code"] = ticker
        return chunk.dropna(subset=["value"])

    pieces.append(_emit(monthly["close"], "close"))
    pieces.append(_emit(monthly["log_return"], "log_return"))
    pieces.append(_emit(monthly["realized_vol"], "realized_vol"))
    pieces.append(_emit(monthly["log_range"], "log_range"))

    # Rate futures: implied rate = 100 - close (CME convention).
    if asset_class == "rate":
        implied = 100.0 - monthly["close"]
        pieces.append(_emit(implied, "implied_rate"))
        # Monthly change in implied rate (rate-expectation shift)
        pieces.append(_emit(implied.diff(), "implied_rate_diff_1m"))

    return pd.concat(pieces, ignore_index=True)


def _yield_curve_spreads(monthly_by_name: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build 2s10s and 5s30s spread series from monthly close levels."""
    rows = []

    def _spread(name_long: str, name_short: str, short_label: str, long_label: str,
                series_label: str) -> pd.DataFrame:
        m_long = monthly_by_name.get(name_long)
        m_short = monthly_by_name.get(name_short)
        if m_long is None or m_short is None or m_long.empty or m_short.empty:
            return pd.DataFrame()
        a = m_long[["month_start", "release_date", "close"]].rename(
            columns={"close": "close_long", "release_date": "release_long"}
        )
        b = m_short[["month_start", "close", "release_date"]].rename(
            columns={"close": "close_short", "release_date": "release_short"}
        )
        merged = a.merge(b, on="month_start", how="inner")
        if merged.empty:
            return pd.DataFrame()
        merged["value"] = merged["close_long"] - merged["close_short"]
        # The spread is only known once BOTH legs have settled — use the later of the two.
        merged["release_date"] = merged[["release_long", "release_short"]].max(axis=1)
        merged["date"] = merged["month_start"]
        merged["series_name"] = f"YieldCurve_{series_label}"
        merged["series_code"] = f"SPREAD:{short_label}/{long_label}"
        return merged[["date", "release_date", "value", "series_name", "series_code"]]

    rows.append(_spread("Treasury_10Y", "Treasury_2Y",  "ZT", "ZN", "spread_2s10s"))
    rows.append(_spread("Treasury_30Y", "Treasury_5Y",  "ZF", "ZB", "spread_5s30s"))

    rows = [r for r in rows if not r.empty]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------- Snapshot writer --------------------------------------------------

@profiled("load_futures_data.fetch_futures_snapshots")
def fetch_futures_snapshots(start_date=START_DATE, end_date=END_DATE):
    """
    Master orchestrator. Builds monthly per-future feature time series once,
    then emits one PIT-correct snapshot per NFP release date.
    """
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)
    if not nfp_release_map:
        raise RuntimeError(
            f"NFP release map empty for window {start_date}..{end_date}"
        )

    # Skip-if-exists: gate on snapshot file presence
    existing = 0
    for obs_month in nfp_release_map.keys():
        if get_snapshot_path(SNAPSHOT_BASE, pd.Timestamp(obs_month)).exists():
            existing += 1
    if existing == len(nfp_release_map):
        print(f"✓ Futures data already exists: {existing} monthly snapshots", flush=True)
        logger.info("Futures snapshots already exist, skipping")
        return

    # Load every future once and build its monthly time series
    logger.info("Loading futures daily CSVs and building monthly features")
    monthly_by_name: Dict[str, pd.DataFrame] = {}
    per_future_long: list[pd.DataFrame] = []

    for display_name, (ticker, asset_class) in FUTURES.items():
        plain = _read_futures_csv(ticker)
        ccb = _read_futures_csv(ticker, suffix="_CCB")
        if plain.empty or ccb.empty:
            logger.warning(f"{display_name} ({ticker}): missing data, skipping")
            continue

        monthly = _monthly_from_daily(plain, ccb)
        if monthly.empty:
            logger.warning(f"{display_name} ({ticker}): no monthly aggregates, skipping")
            continue

        monthly_by_name[display_name] = monthly
        long_rows = _long_format_rows(monthly, display_name, ticker, asset_class)
        if not long_rows.empty:
            per_future_long.append(long_rows)
        logger.info(
            f"  {display_name:14s} ({ticker}): {len(monthly)} monthly rows "
            f"[{monthly['month_start'].min().date()} .. {monthly['month_start'].max().date()}]"
        )

    if not per_future_long:
        raise RuntimeError("No futures monthly data could be built")

    # Cross-sectional spreads (computed from already-built monthly closes)
    spread_long = _yield_curve_spreads(monthly_by_name)
    if not spread_long.empty:
        per_future_long.append(spread_long)
        logger.info(f"  YieldCurve spreads: {len(spread_long)} rows")

    full_long = pd.concat(per_future_long, ignore_index=True)
    full_long["date"] = pd.to_datetime(full_long["date"])
    full_long["release_date"] = pd.to_datetime(full_long["release_date"])

    # For each NFP release date, emit the PIT-filtered snapshot
    snapshots_written = 0
    for obs_month, nfp_release_date in nfp_release_map.items():
        snap_date = pd.Timestamp(nfp_release_date)
        save_path = get_snapshot_path(SNAPSHOT_BASE, pd.Timestamp(obs_month))

        snap = full_long[full_long["release_date"] < snap_date].copy()
        if snap.empty:
            continue

        snap["snapshot_date"] = snap_date

        # Skip pct_change on series that may hover near zero (vol/spreads can flip sign)
        # The transforms layer already handles symlog; lean=True skips most level z-scores.
        skip_pct = {s for s in snap["series_name"].unique()
                    if any(s.endswith(suf) for suf in
                           ("_realized_vol", "_log_return", "_implied_rate_diff_1m"))
                    or s.startswith("YieldCurve_")}

        snap = add_pct_change_copies(snap, skip_series=skip_pct, source_name="Futures")
        snap = compute_all_features(snap, lean=True, source_name="Futures")

        snap.to_parquet(save_path)
        snapshots_written += 1

        if obs_month.month == 12:
            logger.info(f"Saved {obs_month.year} futures snapshots")

    if snapshots_written == 0:
        raise RuntimeError("No futures snapshots were written")

    logger.info(f"✓ Futures pipeline complete: {snapshots_written} snapshots written")


if __name__ == "__main__":
    fetch_futures_snapshots(start_date=START_DATE, end_date=END_DATE)
