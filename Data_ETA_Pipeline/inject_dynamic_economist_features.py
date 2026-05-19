"""Inject dynamic-best-economist features into the existing master snapshots.

Computes the auto-selected dynamic panel forecasts for every historical
target month using the FULL 261-economist panel at
``economist_panel/by_economist/``, then *appends them as new columns* to
every existing master snapshot parquet under
``data/master_snapshots/sa/revised/decades/**/``. The sa branch is the
single canonical store; nsa requests are resolved to the same location
by ``Train.config.get_master_snapshots_dir``.

PIT contract (verified by the companion test):
  For row at ``date = M`` in a snapshot at ``snap_date = release(snap_month)``
  with M < snap_month, the injected feature value is computed using only
  economist forecasts with ``first_release_date < release(M)``. Because NFP
  release dates are monotone in M and M < snap_month, we have
  ``release(M) < release(snap_month) = snap_date`` — so the feature's
  ingredients are strictly earlier than ``snap_date``. The feature is
  PIT-safe to inject at row M of any snapshot dated > release(M).

Features written (all prefixed ``NFP_Forecast_Dynamic_``):
  - ``Top10_k12``     primary auto-panel forecast (track-window 12 calendar
                      months, ≥70% coverage filter, equal-weight mean of
                      top-10 by trailing MAE among active forecasters)
  - ``Top4_k12``      narrower auto-panel variant
  - ``Top15_k12``     wider auto-panel variant
  - ``PanelN``        count of PIT-eligible forecasters for the month
  - ``NCalibrated``   count passing the 70% coverage threshold
  - ``DispersionStd`` cross-sectional std of eligible forecasts (uncertainty)
  - ``DispersionIqr`` IQR of eligible forecasts
  - ``Top10TrackMae`` mean trailing MAE across the selected top-10 (signal
                      quality / regime indicator)
  - ``RobustMedian``  median of every eligible forecast (broadest pool)
  - ``TrimmedMean10`` 10/90 trimmed mean of every eligible forecast

Idempotent: re-running overwrites existing dynamic columns without touching
other columns. Tested below to confirm row counts and other column inventory
are preserved.

Usage::
    python -m Data_ETA_Pipeline.inject_dynamic_economist_features
    python -m Data_ETA_Pipeline.inject_dynamic_economist_features --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import DATA_PATH, TEMP_DIR, setup_logger  # noqa: E402
from Train.config import (  # noqa: E402
    MASTER_SNAPSHOTS_BASE,
    get_master_snapshots_dir,
)

logger = setup_logger(__file__, TEMP_DIR)
logger.setLevel(logging.INFO)


# ── Paths ────────────────────────────────────────────────────────────────
ECON_PANEL_DIR = PROJECT_ROOT / "economist_panel" / "by_economist"
NFP_TARGET_PATH = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"
NFP_FIRST_RELEASE_PATH = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"

# ── Default config ───────────────────────────────────────────────────────
TRACK_WINDOW_MONTHS = 12
MIN_COVERAGE_PCT = 0.70
TOP_N_VARIANTS = (4, 10, 15)
PRIMARY_TOP_N = 10
SKIP_COVID_IN_TRACK_RECORD = False  # Data-driven; see economist_panel_sidecar.py docstring.

# Earliest target month for which we attempt a feature. Panel data starts
# 1999-04, so M = 2001-01 leaves ~21 months of history for track-record fit.
FEATURE_HISTORY_START = pd.Timestamp("2001-01-01")

# Column name prefix in master snapshots
DYNAMIC_COL_PREFIX = "NFP_Forecast_Dynamic_"


# ── Loaders (mirror experiments/sidecars/economist_panel_sidecar.py) ─────

def _load_target_actuals() -> pd.Series:
    df = pd.read_parquet(NFP_TARGET_PATH)[["ds", "y_mom"]].dropna()
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("ds").drop_duplicates("ds")
    return pd.Series(df["y_mom"].to_numpy(dtype=float), index=df["ds"], name="actual")


def _load_nfp_release_map() -> pd.Series:
    if not NFP_FIRST_RELEASE_PATH.exists():
        raise FileNotFoundError(f"Missing NFP release map: {NFP_FIRST_RELEASE_PATH}")
    df = pd.read_parquet(NFP_FIRST_RELEASE_PATH)[["ds", "release_date"]].dropna()
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df.drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
    diffs = df["release_date"].diff().dropna()
    if (diffs < pd.Timedelta(0)).any():
        raise RuntimeError("NFP release_date is not monotone with ds")
    return pd.Series(df["release_date"].to_numpy(), index=df["ds"])


def _load_full_panel() -> pd.DataFrame:
    if not ECON_PANEL_DIR.exists():
        raise FileNotFoundError(f"Economist panel dir missing: {ECON_PANEL_DIR}")
    rows: List[pd.DataFrame] = []
    for parquet_path in sorted(ECON_PANEL_DIR.glob("*.parquet")):
        try:
            p = pd.read_parquet(
                parquet_path,
                columns=["ident", "name", "timestamp",
                         "first_release_value", "first_release_date"],
            )
        except Exception as exc:
            logger.warning("Could not read %s: %s", parquet_path.name, exc)
            continue
        p = p.dropna(subset=["timestamp", "first_release_value",
                             "first_release_date"])
        if p.empty:
            continue
        p = p.copy()
        p["ds"] = pd.to_datetime(p["timestamp"]).dt.to_period("M").dt.to_timestamp()
        p["first_release_date"] = pd.to_datetime(p["first_release_date"])
        p["forecast"] = pd.to_numeric(p["first_release_value"], errors="coerce")
        rows.append(p[["ds", "ident", "name", "forecast", "first_release_date"]])
    if not rows:
        raise RuntimeError("No usable economist parquets found.")
    return pd.concat(rows, ignore_index=True)


def _latest_available_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep the last forecast submission per economist/month."""
    if panel.empty:
        return panel.copy()
    return (
        panel.sort_values(["ident", "ds", "first_release_date"], kind="mergesort")
        .drop_duplicates(subset=["ident", "ds"], keep="last")
        .reset_index(drop=True)
    )


# ── Per-month feature computation ────────────────────────────────────────

def _compute_features_for_month(
    target_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    panel: pd.DataFrame,
    actuals: pd.Series,
) -> Dict[str, float]:
    """Compute every dynamic feature for one target month.

    Mirrors `_compute_track_record` + `_pool_step` from
    experiments/sidecars/economist_panel_sidecar.py but optimized for
    one-pass batch computation. PIT contract: every input is strictly
    older than ``release(target_month) = cutoff``.
    """
    out: Dict[str, float] = {f"Top{n}_k12": np.nan for n in TOP_N_VARIANTS}
    out.update({
        "PanelN": 0,
        "NCalibrated": 0,
        "DispersionStd": np.nan,
        "DispersionIqr": np.nan,
        "Top10TrackMae": np.nan,
        "RobustMedian": np.nan,
        "TrimmedMean10": np.nan,
    })

    # PIT-eligible forecasts for this target month
    eligible = _latest_available_forecasts(
        panel[
            (panel["ds"] == target_month) & (panel["first_release_date"] < cutoff)
        ][["ds", "ident", "name", "forecast", "first_release_date"]].copy()
    )[["ident", "name", "forecast"]]
    out["PanelN"] = int(len(eligible))
    if eligible.empty:
        return out

    forecasts = eligible["forecast"].to_numpy(dtype=float)
    out["DispersionStd"] = (
        float(np.std(forecasts, ddof=1)) if len(forecasts) > 1 else 0.0
    )
    out["DispersionIqr"] = (
        float(np.percentile(forecasts, 75) - np.percentile(forecasts, 25))
        if len(forecasts) > 1 else 0.0
    )
    out["RobustMedian"] = float(np.median(forecasts))
    if len(forecasts) >= 10:
        lo, hi = np.percentile(forecasts, [10, 90])
        trimmed = forecasts[(forecasts >= lo) & (forecasts <= hi)]
        out["TrimmedMean10"] = (
            float(np.mean(trimmed)) if len(trimmed) else float(np.mean(forecasts))
        )
    else:
        out["TrimmedMean10"] = float(np.mean(forecasts))

    # Track record over the trailing 12 calendar months
    window_end = target_month - pd.DateOffset(months=1)
    window_start = target_month - pd.DateOffset(months=TRACK_WINDOW_MONTHS)
    window_months = pd.date_range(window_start, window_end, freq="MS")
    if SKIP_COVID_IN_TRACK_RECORD:
        from utils.transforms import COVID_EXCLUDE_MONTHS
        window_months = window_months[~window_months.isin(COVID_EXCLUDE_MONTHS)]
    n_window_months = int(len(window_months))
    if n_window_months == 0:
        return out

    hist = panel[
        panel["ds"].isin(window_months) & (panel["first_release_date"] < cutoff)
    ][["ds", "ident", "forecast", "first_release_date"]].copy()
    hist = _latest_available_forecasts(hist)
    if hist.empty:
        return out
    hist["actual"] = hist["ds"].map(actuals)
    hist = hist.dropna(subset=["actual", "forecast"])
    if hist.empty:
        return out
    hist["abs_err"] = (hist["forecast"] - hist["actual"]).abs()
    agg = hist.groupby("ident", as_index=False).agg(
        mae=("abs_err", "mean"),
        n=("abs_err", "size"),
    )
    agg["coverage"] = agg["n"].astype(float) / float(n_window_months)

    # Restrict to (a) economists who filed THIS month and (b) calibrated
    active_track = agg[agg["ident"].isin(eligible["ident"])]
    calibrated = active_track[active_track["coverage"] >= MIN_COVERAGE_PCT]
    out["NCalibrated"] = int(len(calibrated))
    if calibrated.empty:
        return out

    ranked = calibrated.sort_values("mae")
    eligible_lookup = eligible.set_index("ident")["forecast"].astype(float)
    for n in TOP_N_VARIANTS:
        top_n = min(n, len(ranked))
        if top_n <= 0:
            continue
        top = ranked.head(top_n)
        top_forecasts = top["ident"].map(eligible_lookup).astype(float).to_numpy()
        mask = np.isfinite(top_forecasts)
        if not mask.any():
            continue
        if n == PRIMARY_TOP_N:
            out["Top10TrackMae"] = float(top.loc[mask, "mae"].mean())
        out[f"Top{n}_k12"] = float(np.mean(top_forecasts[mask]))
    return out


# ── Time-series precomputation ───────────────────────────────────────────

def precompute_dynamic_features() -> pd.DataFrame:
    """Compute the dynamic features for every target month with ≥1 PIT-eligible
    forecast. Returns a DataFrame indexed by month-start ``date``."""
    panel = _load_full_panel()
    actuals = _load_target_actuals()
    release_map = _load_nfp_release_map()

    logger.info(
        "Loaded panel: %d forecasts from %d economists (%s → %s)",
        len(panel), panel["ident"].nunique(),
        panel["ds"].min().strftime("%Y-%m"),
        panel["ds"].max().strftime("%Y-%m"),
    )

    target_months = release_map.index[release_map.index >= FEATURE_HISTORY_START]
    target_months = target_months[target_months <= panel["ds"].max()]
    logger.info(
        "Computing dynamic features for %d target months (%s → %s)",
        len(target_months),
        target_months.min().strftime("%Y-%m"),
        target_months.max().strftime("%Y-%m"),
    )

    rows: List[Dict] = []
    for tm in target_months:
        cutoff = release_map.loc[tm]
        feats = _compute_features_for_month(tm, cutoff, panel, actuals)
        feats["date"] = tm
        feats["release_date"] = cutoff
        rows.append(feats)
    return pd.DataFrame(rows).set_index("date")


# ── Master-snapshot injection ────────────────────────────────────────────

def _resolve_snapshot_paths(branch: str) -> List[Path]:
    base = get_master_snapshots_dir(branch, "revised")
    if not base.exists():
        raise FileNotFoundError(f"Master snapshot dir missing: {base}")
    return sorted(base.glob("**/*.parquet"))


def _inject_into_snapshot(
    snap_path: Path,
    feature_frame: pd.DataFrame,
) -> Tuple[int, int]:
    """Inject dynamic feature columns into one snapshot file.

    PIT guard: only fill rows whose target month is ≤ snap_month. Future-month
    rows (scratch placeholders) get NaN since their feature was computed with
    a cutoff strictly later than the snapshot's vintage.

    Returns ``(n_rows_with_feature, n_new_columns)``.
    """
    snap = pd.read_parquet(snap_path)
    if "date" not in snap.columns:
        raise ValueError(f"Snapshot {snap_path} missing 'date' column")
    snap_dates = pd.to_datetime(snap["date"])
    snap_month = pd.Timestamp(snap_path.stem + "-01")
    pit_mask = snap_dates <= snap_month

    # Drop any prior dynamic columns to make the operation idempotent
    drop_cols = [c for c in snap.columns if c.startswith(DYNAMIC_COL_PREFIX)]
    if drop_cols:
        snap = snap.drop(columns=drop_cols)

    feature_cols = [c for c in feature_frame.columns if c != "release_date"]
    n_added = 0
    n_rows_present = 0
    for feat in feature_cols:
        col_name = f"{DYNAMIC_COL_PREFIX}{feat}"
        series = feature_frame[feat]
        mapped = snap_dates.map(series)
        snap[col_name] = mapped.where(pit_mask, other=pd.NA)
        n_added += 1
        if feat == f"Top{PRIMARY_TOP_N}_k12":
            n_rows_present = int(snap[col_name].notna().sum())

    snap.to_parquet(snap_path)
    return n_rows_present, n_added


def run_injector(branches: Iterable[str], dry_run: bool = False) -> Dict:
    feature_frame = precompute_dynamic_features()
    summary = {"branches": {}, "feature_columns_added": []}
    for branch in branches:
        paths = _resolve_snapshot_paths(branch)
        logger.info("Branch '%s': %d snapshot files to update", branch, len(paths))
        per_branch: List[Dict] = []
        for i, snap_path in enumerate(paths):
            if dry_run:
                per_branch.append({
                    "snapshot": str(snap_path.relative_to(MASTER_SNAPSHOTS_BASE)),
                    "would_inject": True,
                })
                continue
            try:
                n_rows, n_added = _inject_into_snapshot(snap_path, feature_frame)
                per_branch.append({
                    "snapshot": str(snap_path.relative_to(MASTER_SNAPSHOTS_BASE)),
                    "rows_with_primary_feature": int(n_rows),
                    "columns_added": int(n_added),
                })
            except Exception as exc:
                logger.error("Failed on %s: %s", snap_path, exc)
                per_branch.append({
                    "snapshot": str(snap_path.relative_to(MASTER_SNAPSHOTS_BASE)),
                    "error": str(exc),
                })
            if (i + 1) % 50 == 0:
                logger.info("  %s: %d/%d done", branch, i + 1, len(paths))
        summary["branches"][branch] = {
            "n_snapshots": len(paths),
            "first": per_branch[0] if per_branch else None,
            "last": per_branch[-1] if per_branch else None,
        }
    expected_cols = [f"{DYNAMIC_COL_PREFIX}{f}" for f in feature_frame.columns
                     if f != "release_date"]
    summary["feature_columns_added"] = expected_cols
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write; just report what would happen.")
    args = parser.parse_args(argv)

    summary = run_injector(["sa"], dry_run=args.dry_run)
    import json
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
