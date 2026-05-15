"""
Economist panel snapshots: top-4 forecasters as PIT-correct features.

The 4 economists are hardcoded based on head-to-head validation against the
announced first-release MoM (y_sa_revised.y_mom) on the 36 shared months
each of them filed for in Apr 2022 → Sep 2025 (see methodology notes below
in TOP_4_ECONOMISTS). This is a deliberate, deterministic feature list — not
the auto-ranked union — so downstream features are stable across reruns even
as the underlying ranking shifts month-by-month.

Inputs (project-root):
    economist_panel/by_economist/US_XXXXX.parquet  — one parquet per economist
    economist_panel/contributors.parquet           — economist name <-> ident
    NFP_target/y_sa_revised.parquet                — has y_mom = announced first-release MoM

PIT correctness: scoring/storage uses `y_sa_revised.y_mom`, which is the
seasonally-adjusted first-release MoM as published on BLS release day. This
matches what the economists were trying to forecast (their published values
are also SA, MoM, in thousands). The earlier `total_sa_first_release.y.diff()`
choice was incorrect — that's a vintage-mismatch artifact around annual
benchmark revisions.

Outputs:
    1. _output/economist_panel/rankings_full.csv   — full per-economist x per-window
                                                      RMSE/MAE table (for transparency)
    2. _output/economist_panel/top_economists.csv  — the 4 hardcoded picks with their
                                                      metrics on the standard windows
    3. DATA_PATH/Exogenous_data/exogenous_economist_data/decades/.../{YYYY-MM}.parquet
       PIT-correct monthly long-format snapshots. Each row:
            date          = forecast event_month
            release_date  = first_release_date of that forecast
            value         = forecast (thousands of SA MoM jobs)
            series_name   = NFP_Forecast_<EconShortName>  or  NFP_Forecast_Top4Mean
            series_code   = economist ident (US_XXX)        or  TOP4_MEAN
       Snapshot filter: release_date < snap_date (NFP release date of target month).
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
register_atexit_dump("load_economist_panel", output_dir=TEMP_DIR / "perf")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ECONOMIST_DIR = PROJECT_ROOT / "economist_panel" / "by_economist"
CONTRIBUTORS_PATH = PROJECT_ROOT / "economist_panel" / "contributors.parquet"

# Truth = announced first-release SA MoM (PIT-correct: what economists were paid to nail).
# y_sa_revised.y_mom stores exactly the SA first-release MoM as it hit the wire on each
# release day. (For COVID months it's winsorized, but our scoring window is post-Apr-2021.)
NFP_TARGET_PATH = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"

SNAPSHOT_BASE = DATA_PATH / "Exogenous_data" / "exogenous_economist_data"
OUTPUT_DIR = PROJECT_ROOT / "_output" / "economist_panel"
RANKINGS_PATH = OUTPUT_DIR / "rankings_full.csv"
TOP_PATH = OUTPUT_DIR / "top_economists.csv"


# ---------------------------------------------------------------------------
# Hardcoded top-4 economist feature set.
# Selected by head-to-head RMSE/MAE on 36 shared months (Apr 2022 → Sep 2025)
# against the announced first-release MoM:
#
#   Rank | Economist           | RMSE | MAE  | corr | Theil U | Accel acc
#   -----|---------------------|------|------|------|---------|----------
#     1  | Continuum Econ      | 67.5 | 54.2 | 0.72 |  0.747  |  71.4%
#     2  | Nationwide Insur    | 70.2 | 56.4 | 0.73 |  0.777  |  68.6%
#     3  | Danske Bank         | 73.9 | 57.2 | 0.71 |  0.817  |  62.9%
#     4  | AIB                 | 76.0 | 60.1 | 0.69 |  0.841  |  65.7%
#   (LSEG median: RMSE 81.5 — beaten by all 4)
#   (Equal-weight ensemble of the 4: RMSE 68.2, MAE 52.6, corr 0.746)
#
# Error-correlation pairs (vs announced MoM):
#   Danske ↔ Continuum  = 0.655 (least correlated → diversification benefit)
#   AIB ↔ Nationwide    = 0.925 (essentially same signal)
# ---------------------------------------------------------------------------
TOP_4_ECONOMISTS: List[str] = [
    "CONTINUUM ECON",
    "NATIONWIDE INSUR",
    "DANSKE BK",
    "AIB",
]
TOP4_ENSEMBLE_NAME = "NFP_Forecast_Top4Mean"
TOP4_ENSEMBLE_CODE = "TOP4_MEAN"


# ---------- Loaders ---------------------------------------------------------

def _load_nfp_truth() -> pd.DataFrame:
    """
    Return DataFrame with cols: ds, mom_change (announced first-release SA MoM, thousands).

    Truth: y_sa_revised.y_mom — what BLS first put on the wire on each release day.
    """
    df = pd.read_parquet(NFP_TARGET_PATH)[["ds", "y_mom"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.rename(columns={"y_mom": "mom_change"})
    return df


def _load_contributors() -> pd.DataFrame:
    return pd.read_parquet(CONTRIBUTORS_PATH)[["name", "ident"]].copy()


def _safe_econ_name(name: str) -> str:
    """Sanitize an economist name into a feature-friendly identifier."""
    s = str(name).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "UNKNOWN"


def _load_economist_forecasts(parquet_path: Path) -> pd.DataFrame:
    """
    Return per-row forecast: (event_month, forecast, first_release_date, last_revision_date).
    Drops rows with NaN forecast or NaN first_release_date.
    """
    df = pd.read_parquet(parquet_path)
    cols_needed = {"timestamp", "first_release_value", "first_release_date"}
    if not cols_needed.issubset(df.columns):
        return pd.DataFrame()

    out = df[["timestamp", "first_release_value", "first_release_date",
              "last_revision_date"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["first_release_date"] = pd.to_datetime(out["first_release_date"], errors="coerce")
    out["last_revision_date"] = pd.to_datetime(out["last_revision_date"], errors="coerce")
    out["first_release_value"] = pd.to_numeric(out["first_release_value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "first_release_value", "first_release_date"])
    if out.empty:
        return out

    out["event_month"] = out["timestamp"].dt.to_period("M").dt.to_timestamp()
    out = out.rename(columns={"first_release_value": "forecast"})
    # Most recent forecast per event_month (in case of duplicates filed at different dates)
    out = out.sort_values("first_release_date") \
             .drop_duplicates(subset=["event_month"], keep="last")
    return out[["event_month", "forecast", "first_release_date", "last_revision_date"]]


def _resolve_idents(contributors: pd.DataFrame, names: List[str]) -> Dict[str, str]:
    """Map display name → ident, raising if a name isn't present."""
    lookup = dict(zip(contributors["name"], contributors["ident"]))
    out = {}
    for n in names:
        if n not in lookup:
            raise KeyError(
                f"Top-4 economist name not found in contributors.parquet: {n!r}. "
                f"Available: {sorted(lookup.keys())[:10]}..."
            )
        out[n] = lookup[n]
    return out


# ---------- Snapshot rows ---------------------------------------------------

def _build_economist_long(top_idents: Dict[str, str]) -> pd.DataFrame:
    """
    Concatenate the forecast histories of the top economists into long format.

    Returns long-format DataFrame with columns:
        date, release_date, value, series_name, series_code

    Plus an equal-weight ensemble row per event_month (where ≥ 2 of the 4 have
    published a forecast that month). Ensemble release_date = MAX of constituent
    first_release_dates so the ensemble is only "known" once all available
    members for that month have filed.
    """
    per_econ = []
    for display_name, ident in top_idents.items():
        safe_ident = ident.replace("&", "_")
        path = ECONOMIST_DIR / f"{safe_ident}.parquet"
        if not path.exists():
            logger.warning(f"Missing economist parquet for {display_name}: {path.name}")
            continue
        f = _load_economist_forecasts(path)
        if f.empty:
            continue
        clean_name = _safe_econ_name(display_name)
        out = pd.DataFrame({
            "date": f["event_month"].values,
            "release_date": f["first_release_date"].values,
            "value": f["forecast"].astype(float).values,
            "first_release_date_orig": f["first_release_date"].values,
        })
        out["series_name"] = f"NFP_Forecast_{clean_name}"
        out["series_code"] = ident
        out["_econ"] = display_name
        per_econ.append(out)
        logger.info(
            f"  {display_name}: {len(out)} forecasts  "
            f"[{f['event_month'].min().strftime('%Y-%m')} … "
            f"{f['event_month'].max().strftime('%Y-%m')}]"
        )

    if not per_econ:
        return pd.DataFrame()

    individuals = pd.concat(per_econ, ignore_index=True)

    # Equal-weight ensemble — one row per event_month with ≥ 2 constituents.
    ens = (
        individuals.groupby("date")
        .agg(value=("value", "mean"),
             release_date=("first_release_date_orig", "max"),
             n_constituents=("value", "size"))
        .reset_index()
    )
    ens = ens[ens["n_constituents"] >= 2].drop(columns=["n_constituents"])
    ens["series_name"] = TOP4_ENSEMBLE_NAME
    ens["series_code"] = TOP4_ENSEMBLE_CODE

    out = pd.concat([
        individuals[["date", "release_date", "value", "series_name", "series_code"]],
        ens[["date", "release_date", "value", "series_name", "series_code"]],
    ], ignore_index=True)
    return out


# ---------- Optional ranking analytics (for transparency only) -------------

def _rank_all_economists(actuals_lookup: pd.Series,
                         contributors: pd.DataFrame,
                         end_date: pd.Timestamp = pd.Timestamp("2026-05-14"),
                         windows_years: Tuple[int, ...] = (3, 5, 7, 10)
                         ) -> pd.DataFrame:
    """Full rankings table written to rankings_full.csv for transparency."""
    rows = []
    for _, contrib in contributors.iterrows():
        ident = contrib["ident"]
        name = contrib["name"]
        path = ECONOMIST_DIR / f"{ident.replace('&', '_')}.parquet"
        if not path.exists():
            continue
        f = _load_economist_forecasts(path)
        if f.empty:
            continue
        f = f.copy()
        f["actual"] = f["event_month"].map(actuals_lookup)
        for years in windows_years:
            start = end_date - pd.DateOffset(years=years)
            sub = f[(f["event_month"] >= start) & (f["event_month"] <= end_date)] \
                    .dropna(subset=["actual"])
            n = len(sub)
            if n == 0:
                rmse = mae = bias = np.nan
            else:
                err = sub["forecast"] - sub["actual"]
                rmse = float(np.sqrt(np.mean(err.values ** 2)))
                mae = float(np.mean(np.abs(err.values)))
                bias = float(np.mean(err.values))
            rows.append({"ident": ident, "name": name, "window_years": years,
                         "n": n, "rmse": rmse, "mae": mae, "bias": bias})
    return pd.DataFrame(rows)


# ---------- Snapshot writer --------------------------------------------------

@profiled("load_economist_panel.fetch_economist_snapshots")
def fetch_economist_snapshots(start_date=START_DATE, end_date=END_DATE):
    """
    Master orchestrator: build & persist PIT-correct snapshots of the top-4
    economist forecasts plus their equal-weight ensemble.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    actuals = _load_nfp_truth()
    logger.info(
        f"Loaded NFP truth (y_sa_revised.y_mom = announced first-release SA MoM): "
        f"{len(actuals)} rows  [{actuals['ds'].min().date()} .. {actuals['ds'].max().date()}]"
    )
    actuals_lookup = actuals.set_index("ds")["mom_change"]
    contributors = _load_contributors()
    logger.info(f"Loaded {len(contributors)} contributors")

    # Optional transparency: full rankings table
    rankings = _rank_all_economists(actuals_lookup, contributors)
    rankings.sort_values(["window_years", "rmse"]).to_csv(RANKINGS_PATH, index=False)
    logger.info(f"Wrote full rankings: {RANKINGS_PATH}")

    # Resolve the 4 hardcoded picks
    top_idents = _resolve_idents(contributors, TOP_4_ECONOMISTS)
    logger.info(f"Selected (hardcoded) top-4 economists: {list(top_idents.keys())}")

    # Save the top-4 summary CSV with their key metrics
    top_summary_rows = []
    for name, ident in top_idents.items():
        for years in (3, 5, 7, 10):
            r = rankings[(rankings["ident"] == ident) & (rankings["window_years"] == years)]
            if r.empty:
                continue
            top_summary_rows.append({
                "name": name, "ident": ident, "window_years": years,
                "n": int(r["n"].iloc[0]),
                "rmse": float(r["rmse"].iloc[0]),
                "mae": float(r["mae"].iloc[0]),
                "bias": float(r["bias"].iloc[0]),
            })
    pd.DataFrame(top_summary_rows).to_csv(TOP_PATH, index=False)
    logger.info(f"Wrote top-4 summary: {TOP_PATH}")

    # Build the long-format forecast frame for the top 4 + ensemble
    full_long = _build_economist_long(top_idents)
    if full_long.empty:
        raise RuntimeError("No forecast rows assembled for selected economists")
    full_long["date"] = pd.to_datetime(full_long["date"])
    full_long["release_date"] = pd.to_datetime(full_long["release_date"])
    logger.info(
        f"Built long-format forecast frame: {len(full_long)} rows "
        f"across {full_long['series_name'].nunique()} series"
    )

    # Snapshot writer
    nfp_release_map = get_nfp_release_map(start_date=start_date, end_date=end_date)
    if not nfp_release_map:
        raise RuntimeError(
            f"NFP release map empty for window {start_date}..{end_date}"
        )

    existing = 0
    for obs_month in nfp_release_map.keys():
        if get_snapshot_path(SNAPSHOT_BASE, pd.Timestamp(obs_month)).exists():
            existing += 1
    if existing == len(nfp_release_map):
        print(f"✓ Economist snapshots already exist: {existing} months", flush=True)
        logger.info("Economist snapshots already exist, skipping write")
        return

    snapshots_written = 0
    for obs_month, nfp_release_date in nfp_release_map.items():
        snap_date = pd.Timestamp(nfp_release_date)
        save_path = get_snapshot_path(SNAPSHOT_BASE, pd.Timestamp(obs_month))

        # PIT filter: only forecasts published before snap_date
        snap = full_long[full_long["release_date"] < snap_date].copy()
        if snap.empty:
            continue

        snap["snapshot_date"] = snap_date

        # Forecast values are MoM levels in thousands — pct_change copies cross zero
        # (recessionary months can be negative), so skip pct_change on all series here.
        skip_pct = set(snap["series_name"].unique())
        snap = add_pct_change_copies(snap, skip_series=skip_pct)
        snap = compute_all_features(snap, lean=True)

        snap.to_parquet(save_path)
        snapshots_written += 1

        if obs_month.month == 12:
            logger.info(f"Saved {obs_month.year} economist snapshots")

    if snapshots_written == 0:
        raise RuntimeError("No economist snapshots were written")

    logger.info(f"✓ Economist pipeline complete: {snapshots_written} snapshots written")


if __name__ == "__main__":
    fetch_economist_snapshots(start_date=START_DATE, end_date=END_DATE)
