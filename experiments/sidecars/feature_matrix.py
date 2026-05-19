"""Feature matrix helpers for local sidecar models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


try:
    from settings import DATA_PATH
    from Train.config import get_master_snapshots_dir
except RuntimeError:
    # Sidecar tests sometimes run before settings/env is fully initialised.
    # Mirror the canonical resolver: any target_type → sa/revised/decades.
    DATA_PATH = Path("data")

    def get_master_snapshots_dir(target_type: str, target_source: str = "revised") -> Path:
        return DATA_PATH / "master_snapshots" / "sa" / "revised" / "decades"


BLOCK_PREFIXES: dict[str, tuple[str, ...]] = {
    "consensus": ("NFP_Consensus", "rev_master_"),
    "economist": ("NFP_Forecast_", "Economist_", "economist_"),
    "futures": (
        "Treasury_",
        "FedFunds_",
        "SOFR_",
        "WTI_Crude_",
        "NatGas_",
        "Gold_",
        "Copper_",
        "DollarIndex_",
        "EuroFX_",
        "YenFX_",
        "SP500_Futures_",
    ),
    "labor": ("total_",),
    "unifier": (
        "AHE_",
        "AWH_",
        "Challenger_",
        "ISM_",
        "UMich_",
        "CB_",
        "Retail_",
        "Industrial_",
    ),
    "stress": ("Financial_", "VIX_", "Credit_", "Yield_", "Oil_", "SP500_"),
    "gap": ("sanagap_",),
}


def target_path_for_space(target_space: str) -> Path:
    key = target_space.lower()
    if key.startswith("sa"):
        return DATA_PATH / "NFP_target" / "y_sa_revised.parquet"
    if key.startswith("nsa"):
        return DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    raise ValueError(f"Unknown target_space={target_space!r}")


def load_target(target_space: str = "sa_revised", target_path: Path | None = None) -> pd.DataFrame:
    path = target_path or target_path_for_space(target_space)
    df = pd.read_parquet(path)
    if "ds" not in df.columns:
        raise ValueError(f"{path} lacks ds")
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "y_mom" not in out.columns:
        if "y" not in out.columns:
            raise ValueError(f"{path} must contain y_mom or y")
        out["y_mom"] = pd.to_numeric(out["y"], errors="coerce").diff()
    else:
        out["y_mom"] = pd.to_numeric(out["y_mom"], errors="coerce")
    keep = ["ds", "y_mom"]
    if "y" in out.columns:
        keep.append("y")
    if "release_date" in out.columns:
        keep.append("release_date")
    if "operational_available_date" in out.columns:
        keep.append("operational_available_date")
    return out[keep].dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)


def add_target_dynamics(target: pd.DataFrame, prefix: str = "target") -> pd.DataFrame:
    target = target.copy().sort_values("ds").reset_index(drop=True)
    has_availability = {
        "release_date",
        "operational_available_date",
    }.issubset(target.columns)
    if has_availability:
        return _add_target_dynamics_pit(target, prefix=prefix)

    df = target[["ds", "y_mom"]].copy().sort_values("ds").reset_index(drop=True)
    df["actual_mom"] = df["y_mom"]
    df["prev_mom"] = df["y_mom"].shift(1)
    df["actual_accel"] = df["y_mom"] - df["prev_mom"]
    for lag in (1, 2, 3, 6, 12):
        df[f"{prefix}_mom_lag{lag}"] = df["y_mom"].shift(lag)
        df[f"{prefix}_accel_lag{lag}"] = df["y_mom"].diff().shift(lag)
    for window in (3, 6, 12):
        shifted = df["y_mom"].shift(1)
        df[f"{prefix}_mom_mean_{window}m"] = shifted.rolling(window, min_periods=max(2, window // 2)).mean()
        df[f"{prefix}_mom_std_{window}m"] = shifted.rolling(window, min_periods=max(2, window // 2)).std()
        mean = df[f"{prefix}_mom_mean_{window}m"]
        std = df[f"{prefix}_mom_std_{window}m"].replace(0.0, np.nan)
        df[f"{prefix}_mom_z_{window}m"] = (shifted - mean) / std
    df[f"{prefix}_mom_vs_12m_trend"] = df[f"{prefix}_mom_lag1"] - df[f"{prefix}_mom_mean_12m"]
    return df


def _history_values_before(row: pd.Series, target: pd.DataFrame) -> np.ndarray:
    cutoff = pd.Timestamp(row["release_date"])
    current_ds = pd.Timestamp(row["ds"])
    hist = target[
        (pd.to_datetime(target["ds"]) < current_ds)
        & pd.to_numeric(target["y_mom"], errors="coerce").notna()
    ].copy()
    avail = pd.to_datetime(hist["operational_available_date"], errors="coerce")
    hist = hist[avail.notna() & (avail < cutoff)]
    return hist.sort_values("ds")["y_mom"].astype(float).to_numpy()


def _add_target_dynamics_pit(target: pd.DataFrame, prefix: str = "target") -> pd.DataFrame:
    """Target dynamics using only actuals available before each row's release.

    Revised NFP actuals for the immediately previous month are commonly
    published on the same date as the current target release, so a plain
    ``shift(1)`` would leak same-day information into sidecar features.
    """
    keep_cols = ["ds", "y_mom", "release_date", "operational_available_date"]
    if "y" in target.columns:
        keep_cols.insert(2, "y")
    df = target[keep_cols].copy().sort_values("ds").reset_index(drop=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["operational_available_date"] = pd.to_datetime(
        df["operational_available_date"],
        errors="coerce",
    )
    df["actual_mom"] = pd.to_numeric(df["y_mom"], errors="coerce")

    for col in ["prev_mom", "actual_accel"]:
        df[col] = np.nan
    for lag in (1, 2, 3, 6, 12):
        df[f"{prefix}_mom_lag{lag}"] = np.nan
        df[f"{prefix}_accel_lag{lag}"] = np.nan
    for window in (3, 6, 12):
        df[f"{prefix}_mom_mean_{window}m"] = np.nan
        df[f"{prefix}_mom_std_{window}m"] = np.nan
        df[f"{prefix}_mom_z_{window}m"] = np.nan
    df[f"{prefix}_mom_vs_12m_trend"] = np.nan

    for idx, row in df.iterrows():
        if pd.isna(row["release_date"]):
            continue
        values = _history_values_before(row, df)
        if values.size:
            df.at[idx, "prev_mom"] = float(values[-1])
            if pd.notna(row["actual_mom"]):
                df.at[idx, "actual_accel"] = float(row["actual_mom"]) - float(values[-1])
        diffs = np.diff(values) if values.size >= 2 else np.array([])
        for lag in (1, 2, 3, 6, 12):
            if values.size >= lag:
                df.at[idx, f"{prefix}_mom_lag{lag}"] = float(values[-lag])
            if diffs.size >= lag:
                df.at[idx, f"{prefix}_accel_lag{lag}"] = float(diffs[-lag])
        for window in (3, 6, 12):
            min_periods = max(2, window // 2)
            recent = values[-window:]
            if recent.size >= min_periods:
                mean = float(np.mean(recent))
                std = float(np.std(recent, ddof=1)) if recent.size >= 2 else np.nan
                df.at[idx, f"{prefix}_mom_mean_{window}m"] = mean
                df.at[idx, f"{prefix}_mom_std_{window}m"] = std
                if np.isfinite(std) and std > 0 and values.size:
                    df.at[idx, f"{prefix}_mom_z_{window}m"] = (float(values[-1]) - mean) / std
        if pd.notna(df.at[idx, f"{prefix}_mom_lag1"]) and pd.notna(df.at[idx, f"{prefix}_mom_mean_12m"]):
            df.at[idx, f"{prefix}_mom_vs_12m_trend"] = (
                float(df.at[idx, f"{prefix}_mom_lag1"])
                - float(df.at[idx, f"{prefix}_mom_mean_12m"])
            )
    return df


def _source_for_col(col: str) -> str:
    for block, prefixes in BLOCK_PREFIXES.items():
        if col.startswith(prefixes):
            return block
    if col.startswith(("target_", "sa_", "nsa_", "prev_", "actual_")):
        return "target"
    return "snapshot"


def source_map_for_columns(cols: Iterable[str]) -> dict[str, str]:
    return {col: _source_for_col(col) for col in cols}


def _columns_for_blocks(columns: Iterable[str], blocks: set[str]) -> list[str]:
    selected = []
    for col in columns:
        if col == "date":
            continue
        for block in blocks:
            if col.startswith(BLOCK_PREFIXES.get(block, ())):
                selected.append(col)
                break
    return selected


def load_snapshot_matrix(
    *,
    target_type: str = "sa",
    target_source: str = "revised",
    blocks: Iterable[str] = ("consensus", "economist", "futures", "unifier", "stress", "gap"),
    max_columns: int = 250,
    min_non_nan: int = 24,
) -> pd.DataFrame:
    """Load one PIT row per master snapshot for selected source blocks."""
    base = get_master_snapshots_dir(target_type, target_source)
    if not base.exists():
        return pd.DataFrame(columns=["ds"])
    blocks_set = set(blocks)
    rows: list[pd.Series] = []
    for path in sorted(base.glob("**/*.parquet")):
        try:
            ds = pd.Timestamp(path.stem + "-01")
        except Exception:
            continue
        try:
            probe = pd.read_parquet(path)
        except Exception:
            continue
        if "date" not in probe.columns:
            continue
        cols = _columns_for_blocks(probe.columns, blocks_set)
        if not cols:
            continue
        probe["date"] = pd.to_datetime(probe["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        match = probe[probe["date"] == ds]
        if match.empty:
            continue
        row = match[cols].iloc[-1].copy()
        row["ds"] = ds
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ds"])
    mat = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    numeric = [c for c in mat.columns if c != "ds"]
    mat[numeric] = mat[numeric].apply(pd.to_numeric, errors="coerce")
    counts = mat[numeric].notna().sum()
    variances = mat[numeric].var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [
        c for c in numeric
        if int(counts.get(c, 0)) >= int(min_non_nan) and float(variances.get(c, 0.0)) > 1e-12
    ]
    keep = sorted(keep, key=lambda c: (int(counts[c]), float(variances[c])), reverse=True)[: int(max_columns)]
    return mat[["ds"] + keep]


def build_sidecar_design(
    *,
    target_space: str = "sa_revised",
    target_path: Path | None = None,
    include_snapshots: bool = True,
    snapshot_blocks: Iterable[str] = ("consensus", "economist", "futures", "unifier", "stress", "gap"),
    max_snapshot_columns: int = 250,
) -> pd.DataFrame:
    target = load_target(target_space, target_path=target_path)
    prefix = "sa" if target_space.lower().startswith("sa") else "nsa"
    design = add_target_dynamics(target, prefix=prefix)
    if include_snapshots:
        snap_type = "sa" if target_space.lower().startswith("sa") else "nsa"
        snap = load_snapshot_matrix(
            target_type=snap_type,
            blocks=snapshot_blocks,
            max_columns=max_snapshot_columns,
        )
        if not snap.empty:
            design = design.merge(snap, on="ds", how="left")
    return design.sort_values("ds").reset_index(drop=True)


def select_numeric_feature_cols(
    frame: pd.DataFrame,
    *,
    exclude: Iterable[str] = ("ds", "y_mom", "actual_mom", "actual_accel"),
) -> list[str]:
    excluded = set(exclude)
    out = []
    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out.append(col)
    return out


def rank_features_by_correlation(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    min_non_nan: int = 36,
    top_n: int = 50,
) -> list[str]:
    scores: list[tuple[float, int, str]] = []
    y = pd.to_numeric(train[target_col], errors="coerce")
    for col in feature_cols:
        x = pd.to_numeric(train[col], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < int(min_non_nan):
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
            continue
        corr = np.corrcoef(xv, yv)[0, 1]
        if np.isfinite(corr):
            scores.append((abs(float(corr)), int(mask.sum()), col))
    scores.sort(reverse=True)
    return [col for _, _, col in scores[: int(top_n)]]
