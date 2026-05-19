"""Mini study of PIT-safe alternatives for SA revised NFP MoM.

This script intentionally stays outside the production LightGBM/Kalman path.
It builds one row per target month from the canonical master snapshot:

    data/master_snapshots/sa/revised/decades/**/<YYYY-MM>.parquet

For month M it uses the row where snapshot row ``date == M`` and joins the
once-revised SA target from ``data/NFP_target/y_sa_revised.parquet``.  Walk-forward
models only train on actuals with ``actual_available_date < target_release_date``
for the forecast month, which avoids same-release-day revised-target leakage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import COVID_EXCLUDE_MONTHS, winsorize_covid_period  # noqa: E402


SNAPSHOT_BASE = PROJECT_ROOT / "data" / "master_snapshots" / "sa" / "revised" / "decades"
TARGET_PATH = PROJECT_ROOT / "data" / "NFP_target" / "y_sa_revised.parquet"
DEFAULT_OUT = PROJECT_ROOT / "_output_sa_alternative_methods_mini"

FORECAST_COLS = [
    "NFP_Consensus_Mean",
    "NFP_Consensus_Median",
    "NFP_Consensus_High",
    "NFP_Consensus_Low",
    "NFP_Consensus_Good",
    "NFP_Forecast_Dynamic_Top4_k12",
    "NFP_Forecast_Dynamic_Top10_k12",
    "NFP_Forecast_Dynamic_Top15_k12",
    "NFP_Forecast_Dynamic_RobustMedian",
    "NFP_Forecast_Dynamic_TrimmedMean10",
    "NFP_Forecast_Dynamic_DispersionStd",
    "NFP_Forecast_Dynamic_DispersionIqr",
    "NFP_Forecast_Dynamic_PanelN",
    "NFP_Forecast_Dynamic_NCalibrated",
    "NFP_Forecast_Dynamic_Top10TrackMae",
]

CORE_PREFIXES = (
    "ADP_",
    "ISM_",
    "Challenger_",
    "AWH_",
    "AHE_",
    "UMich_",
    "CB_",
    "Retail_",
    "Industrial_",
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
    "Financial_",
    "VIX_",
    "Credit_",
    "Yield_",
    "Oil_",
    "SP500_",
    "sanagap_",
)

TOTAL_SUFFIXES = (
    "_diff",
    "_diff_lag_1m",
    "_chg_3m",
    "_chg_3m_lag_1m",
    "_chg_6m",
    "_chg_6m_lag_1m",
    "_rolling_mean_3m",
    "_rolling_std_6m",
    "_lag_1m",
)

LITE_SUFFIXES = (
    "",
    "_diff",
    "_chg_3m",
    "_chg_6m",
    "_lag_1m",
    "_diff_lag_1m",
    "_rolling_mean_3m",
    "_rolling_std_6m",
)

LITE_PREFIX_LIMITS = {
    "ADP_": 14,
    "ISM_": 24,
    "Challenger_": 10,
    "AWH_": 18,
    "AHE_": 10,
    "UMich_": 8,
    "CB_": 8,
    "Retail_": 10,
    "Industrial_": 10,
    "Treasury_": 14,
    "FedFunds_": 10,
    "SP500_Futures_": 8,
    "DollarIndex_": 6,
    "VIX_": 8,
    "Yield_": 8,
}

LITE_TOTAL_BASES = (
    "total_private",
    "total_government",
    "total_private_goods",
    "total_private_services",
    "total_private_goods_construction",
    "total_private_goods_manufacturing",
    "total_private_services_professional_business",
    "total_private_services_education_health",
    "total_private_services_leisure_hospitality",
    "total_private_services_trade_transportation_utilities",
)

ALPHAS = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


@dataclass
class Prediction:
    ds: pd.Timestamp
    model: str
    predicted: float
    actual: float
    prev_actual: float
    accel_correct: float
    n_train: int
    detail: str = ""


def parse_month(raw: str | None) -> pd.Timestamp | None:
    if raw is None:
        return None
    return pd.Timestamp(raw).to_period("M").to_timestamp()


def snapshot_month_from_path(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(path.stem + "-01")
    except Exception:
        return None


def is_candidate_feature(col: str) -> bool:
    if col in {"date", "snapshot_date"}:
        return False
    if col in FORECAST_COLS:
        return True
    if col.startswith(("NFP_Consensus_", "NFP_Forecast_Dynamic_")):
        return True
    if col.startswith(CORE_PREFIXES):
        return True
    if col.startswith("total_"):
        return col in {
            "total",
            "total_private",
            "total_government",
            "total_private_goods",
            "total_private_services",
        } or col.endswith(TOTAL_SUFFIXES)
    return False


def latest_schema_columns(snapshot_files: list[Path]) -> list[str]:
    for path in reversed(snapshot_files):
        try:
            return pq.ParquetFile(path).schema_arrow.names
        except Exception:
            continue
    raise RuntimeError(f"No readable snapshot schemas under {SNAPSHOT_BASE}")


def choose_lite_columns(names: list[str], max_pool: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(col: str) -> None:
        if col in names and col not in seen:
            selected.append(col)
            seen.add(col)

    for col in FORECAST_COLS:
        add(col)
    for col in names:
        if col.startswith("NFP_Consensus_") and any(col.endswith(s) for s in LITE_SUFFIXES):
            add(col)
    for base in LITE_TOTAL_BASES:
        for suffix in LITE_SUFFIXES:
            add(base + suffix)
    for col in names:
        if col.startswith("sanagap_"):
            add(col)

    for prefix, limit in LITE_PREFIX_LIMITS.items():
        n = 0
        for col in names:
            if n >= limit:
                break
            if col in seen:
                continue
            if col.startswith(prefix) and any(col.endswith(s) for s in LITE_SUFFIXES):
                add(col)
                n += 1

    return selected[: int(max_pool)]


def choose_columns(snapshot_files: list[Path], max_pool: int, *, lite: bool = False) -> list[str]:
    names = latest_schema_columns(snapshot_files)
    if lite:
        return choose_lite_columns(names, max_pool=max_pool)
    candidates = [c for c in names if is_candidate_feature(c)]
    forecast = [c for c in FORECAST_COLS if c in names]
    rest = [c for c in candidates if c not in forecast]
    return forecast + rest[: max(0, int(max_pool) - len(forecast))]


def load_snapshot_design(
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    max_pool: int,
    lite: bool,
    progress_every: int,
) -> tuple[pd.DataFrame, list[str]]:
    snapshot_files = sorted(SNAPSHOT_BASE.glob("**/*.parquet"))
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshots found under {SNAPSHOT_BASE}")

    selected_cols = choose_columns(snapshot_files, max_pool=max_pool, lite=lite)
    requested = ["date", "snapshot_date", *selected_cols]
    rows: list[dict] = []

    for idx, path in enumerate(snapshot_files, start=1):
        ds = snapshot_month_from_path(path)
        if ds is None:
            continue
        if start is not None and ds < start:
            continue
        if end is not None and ds > end:
            continue
        try:
            available = set(pq.ParquetFile(path).schema_arrow.names)
            cols = [c for c in requested if c in available]
            snap = pd.read_parquet(path, columns=cols)
        except Exception as exc:
            print(f"skip unreadable snapshot {path}: {exc}")
            continue
        if "date" not in snap.columns:
            continue
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        match = snap[snap["date"] == ds]
        if match.empty:
            continue
        row = match.iloc[-1].to_dict()
        row["ds"] = ds
        rows.append(row)
        if progress_every > 0 and len(rows) % progress_every == 0:
            print(f"loaded {len(rows)} target-month snapshot rows through {ds:%Y-%m}", flush=True)

    if not rows:
        raise RuntimeError("No target-month rows were loaded from master snapshots")

    out = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c not in {"ds", "date", "snapshot_date"}]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out = add_engineered_anchor_features(out)

    winsor_cols = [
        c for c in out.columns
        if c not in {"ds", "date", "snapshot_date"} and pd.api.types.is_numeric_dtype(out[c])
    ]
    indexed = out.set_index("ds")
    indexed[winsor_cols] = winsorize_covid_period(indexed[winsor_cols])
    out = indexed.reset_index()
    return out, selected_cols


def add_engineered_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean = pd.to_numeric(out.get("NFP_Consensus_Mean"), errors="coerce")
    median = pd.to_numeric(out.get("NFP_Consensus_Median"), errors="coerce")
    high = pd.to_numeric(out.get("NFP_Consensus_High"), errors="coerce")
    low = pd.to_numeric(out.get("NFP_Consensus_Low"), errors="coerce")
    panel = pd.to_numeric(out.get("NFP_Forecast_Dynamic_Top4_k12"), errors="coerce")
    robust = pd.to_numeric(out.get("NFP_Forecast_Dynamic_RobustMedian"), errors="coerce")

    out["anchor_mean_minus_median"] = mean - median
    out["anchor_range"] = high - low
    out["panel_top4_minus_consensus_median"] = panel - median
    out["panel_robust_minus_consensus_median"] = robust - median
    out["month_sin"] = np.sin(2.0 * np.pi * pd.to_datetime(out["ds"]).dt.month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * pd.to_datetime(out["ds"]).dt.month / 12.0)
    return out


def load_target() -> pd.DataFrame:
    target = pd.read_parquet(TARGET_PATH)
    target["ds"] = pd.to_datetime(target["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    target["actual"] = pd.to_numeric(target["y_mom"], errors="coerce")
    idx = target.set_index("ds")
    idx["actual"] = winsorize_covid_period(idx["actual"])
    target = idx.reset_index()
    target = target.sort_values("ds").reset_index(drop=True)
    target["prev_actual"] = target["actual"].shift(1)
    keep = ["ds", "actual", "release_date", "operational_available_date"]
    keep.append("prev_actual")
    out = target[keep].copy()
    out["target_release_date"] = pd.to_datetime(out["release_date"], errors="coerce")
    out["actual_available_date"] = pd.to_datetime(out["operational_available_date"], errors="coerce")
    return out.drop(columns=["release_date", "operational_available_date"])


def add_target_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("ds").reset_index(drop=True)
    for col in [
        "sa_mom_lag1",
        "sa_mom_lag2",
        "sa_mom_lag3",
        "sa_mom_lag6",
        "sa_mom_lag12",
        "sa_mom_mean3",
        "sa_mom_mean6",
        "sa_mom_mean12",
        "sa_mom_std6",
        "sa_mom_accel_lag1",
    ]:
        out[col] = np.nan

    actuals = out[["ds", "actual", "actual_available_date"]].copy()
    for idx, row in out.iterrows():
        cutoff = row.get("target_release_date", pd.NaT)
        if pd.isna(cutoff):
            continue
        hist = actuals[
            (actuals["ds"] < row["ds"])
            & actuals["actual"].notna()
            & actuals["actual_available_date"].notna()
            & (actuals["actual_available_date"] < cutoff)
        ].sort_values("ds")
        vals = hist["actual"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        for lag in (1, 2, 3, 6, 12):
            if vals.size >= lag:
                out.at[idx, f"sa_mom_lag{lag}"] = vals[-lag]
        for window in (3, 6, 12):
            recent = vals[-window:]
            if recent.size >= max(2, window // 2):
                out.at[idx, f"sa_mom_mean{window}"] = float(np.mean(recent))
        if vals.size >= 6:
            out.at[idx, "sa_mom_std6"] = float(np.std(vals[-6:], ddof=1))
        if vals.size >= 2:
            out.at[idx, "sa_mom_accel_lag1"] = vals[-1] - vals[-2]
    return out


def available_history(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    cutoff = row.get("target_release_date", pd.NaT)
    if pd.isna(cutoff):
        return frame.iloc[0:0].copy()
    hist = frame[
        (frame["ds"] < row["ds"])
        & frame["actual"].notna()
        & frame["actual_available_date"].notna()
        & (frame["actual_available_date"] < cutoff)
    ].copy()
    return hist.sort_values("ds").reset_index(drop=True)


def numeric_feature_cols(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "ds",
        "date",
        "snapshot_date",
        "actual",
        "prev_actual",
        "accel_correct",
        "target_release_date",
        "actual_available_date",
    }
    return [
        c for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]


def rank_features(
    train: pd.DataFrame,
    features: Iterable[str],
    target: pd.Series,
    *,
    min_obs: int,
    top_n: int,
) -> list[str]:
    cols = list(dict.fromkeys(features))
    if not cols:
        return []
    y = pd.to_numeric(target, errors="coerce")
    x = train[cols].apply(pd.to_numeric, errors="coerce")
    valid_counts = x.notna().mul(y.notna(), axis=0).sum()
    keep = valid_counts[valid_counts >= min_obs].index.tolist()
    if not keep:
        return []
    corr = x[keep].corrwith(y).abs().replace([np.inf, -np.inf], np.nan).dropna()
    if corr.empty:
        return []
    ranked = corr.sort_values(ascending=False).index.tolist()
    return ranked[: int(top_n)]


def ridge_predict(
    train: pd.DataFrame,
    row: pd.Series,
    features: list[str],
    target_col: str,
) -> tuple[float, str]:
    features = list(dict.fromkeys(features))
    if not features:
        return np.nan, "no_features"
    y = pd.to_numeric(train[target_col], errors="coerce")
    train_mask = y.notna()
    x_raw = train.loc[train_mask, features].apply(pd.to_numeric, errors="coerce")
    y_fit = y.loc[train_mask].to_numpy(dtype=float)
    non_nan = x_raw.notna().sum()
    variances = x_raw.var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = [c for c in features if non_nan.get(c, 0) >= 24 and variances.get(c, 0.0) > 1e-12]
    if len(cols) == 0 or len(y_fit) < 24:
        return np.nan, "insufficient"
    x_raw = x_raw[cols]
    med = x_raw.median()
    x_fit = x_raw.fillna(med).to_numpy(dtype=float)
    x_test = pd.DataFrame([row[cols]], columns=cols).apply(pd.to_numeric, errors="coerce").fillna(med)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_fit_s = scaler.fit_transform(x_fit)
    x_test_s = scaler.transform(x_test.to_numpy(dtype=float))
    model = RidgeCV(alphas=ALPHAS)
    model.fit(x_fit_s, y_fit)
    pred = float(model.predict(x_test_s)[0])
    return pred, f"n_features={len(cols)} alpha={float(model.alpha_):.3g}"


def residual_cap(train_resid: pd.Series) -> float:
    vals = pd.to_numeric(train_resid, errors="coerce").dropna().abs()
    if vals.empty:
        return 125.0
    return float(np.clip(vals.quantile(0.80), 50.0, 175.0))


def pca_residual_predict(
    train: pd.DataFrame,
    row: pd.Series,
    features: list[str],
    *,
    target_col: str,
    n_components: int = 5,
) -> tuple[float, str]:
    features = list(dict.fromkeys(features))
    if len(features) < 5:
        return np.nan, "too_few_features"
    y = pd.to_numeric(train[target_col], errors="coerce")
    mask = y.notna()
    x_raw = train.loc[mask, features].apply(pd.to_numeric, errors="coerce")
    y_fit = y.loc[mask].to_numpy(dtype=float)
    non_nan = x_raw.notna().sum()
    variances = x_raw.var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = [c for c in features if non_nan.get(c, 0) >= 36 and variances.get(c, 0.0) > 1e-12]
    if len(cols) < 5 or len(y_fit) < 60:
        return np.nan, "insufficient"
    x_raw = x_raw[cols]
    med = x_raw.median()
    x_fit = x_raw.fillna(med).to_numpy(dtype=float)
    x_test = pd.DataFrame([row[cols]], columns=cols).apply(pd.to_numeric, errors="coerce").fillna(med)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_fit_s = scaler.fit_transform(x_fit)
    x_test_s = scaler.transform(x_test.to_numpy(dtype=float))
    n_comp = min(int(n_components), x_fit_s.shape[0] - 1, x_fit_s.shape[1])
    if n_comp < 1:
        return np.nan, "no_components"
    pca = PCA(n_components=n_comp, random_state=42)
    z_fit = pca.fit_transform(x_fit_s)
    z_test = pca.transform(x_test_s)
    model = RidgeCV(alphas=ALPHAS)
    model.fit(z_fit, y_fit)
    pred = float(model.predict(z_test)[0])
    explained = float(np.sum(pca.explained_variance_ratio_))
    return pred, f"n_features={len(cols)} n_components={n_comp} pca_var={explained:.3f}"


def analog_residual_predict(
    train: pd.DataFrame,
    row: pd.Series,
    features: list[str],
    *,
    target_col: str,
    k: int = 8,
) -> tuple[float, str]:
    features = list(dict.fromkeys(features))
    if len(features) < 5:
        return np.nan, "too_few_features"
    y = pd.to_numeric(train[target_col], errors="coerce")
    mask = y.notna()
    x_raw = train.loc[mask, features].apply(pd.to_numeric, errors="coerce")
    y_fit = y.loc[mask].to_numpy(dtype=float)
    non_nan = x_raw.notna().sum()
    variances = x_raw.var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = [c for c in features if non_nan.get(c, 0) >= 36 and variances.get(c, 0.0) > 1e-12]
    if len(cols) < 5 or len(y_fit) < 60:
        return np.nan, "insufficient"
    x_raw = x_raw[cols]
    med = x_raw.median()
    x_fit = x_raw.fillna(med).to_numpy(dtype=float)
    x_test = pd.DataFrame([row[cols]], columns=cols).apply(pd.to_numeric, errors="coerce").fillna(med)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x_fit_s = scaler.fit_transform(x_fit)
    x_test_s = scaler.transform(x_test.to_numpy(dtype=float))[0]
    dist = np.sqrt(np.mean((x_fit_s - x_test_s) ** 2, axis=1))
    order = np.argsort(dist)[: min(k, len(dist))]
    nearest_dist = dist[order]
    weights = 1.0 / (nearest_dist + 1e-6)
    weights = weights / weights.sum()
    pred = float(np.dot(weights, y_fit[order]))
    return pred, f"n_features={len(cols)} k={len(order)} avg_dist={float(nearest_dist.mean()):.3f}"


def add_prediction(
    rows: list[Prediction],
    row: pd.Series,
    model: str,
    pred: float,
    n_train: int,
    detail: str = "",
) -> None:
    if pd.isna(pred):
        return
    actual = float(row["actual"]) if pd.notna(row.get("actual")) else np.nan
    prev_actual = float(row["prev_actual"]) if pd.notna(row.get("prev_actual")) else np.nan
    if np.isfinite(actual) and np.isfinite(prev_actual):
        accel_correct = float(int(np.sign(actual - prev_actual) == np.sign(float(pred) - prev_actual)))
    else:
        accel_correct = np.nan
    rows.append(
        Prediction(
            ds=pd.Timestamp(row["ds"]),
            model=model,
            predicted=float(pred),
            actual=actual,
            prev_actual=prev_actual,
            accel_correct=accel_correct,
            n_train=int(n_train),
            detail=detail,
        )
    )


def run_walk_forward(
    frame: pd.DataFrame,
    *,
    score_start: pd.Timestamp,
    min_train: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = numeric_feature_cols(frame)
    anchor_cols = [
        c for c in [
            "NFP_Consensus_Mean",
            "NFP_Consensus_Median",
            "NFP_Consensus_High",
            "NFP_Consensus_Low",
            "NFP_Consensus_Good",
            "NFP_Forecast_Dynamic_Top4_k12",
            "NFP_Forecast_Dynamic_Top10_k12",
            "NFP_Forecast_Dynamic_Top15_k12",
            "NFP_Forecast_Dynamic_RobustMedian",
            "NFP_Forecast_Dynamic_TrimmedMean10",
            "NFP_Forecast_Dynamic_DispersionStd",
            "NFP_Forecast_Dynamic_DispersionIqr",
            "NFP_Forecast_Dynamic_PanelN",
            "NFP_Forecast_Dynamic_NCalibrated",
            "NFP_Forecast_Dynamic_Top10TrackMae",
            "anchor_mean_minus_median",
            "anchor_range",
            "panel_top4_minus_consensus_median",
            "panel_robust_minus_consensus_median",
            "month_sin",
            "month_cos",
        ] if c in frame.columns
    ]
    target_dyn_cols = [c for c in feature_cols if c.startswith("sa_mom_")]
    residual_pool = list(dict.fromkeys([
        c for c in feature_cols
        if c not in set(anchor_cols) and not c.startswith("NFP_Consensus_")
    ] + target_dyn_cols))

    preds: list[Prediction] = []
    feature_usage: list[dict] = []

    for _, row in frame[frame["ds"] >= score_start].sort_values("ds").iterrows():
        hist = available_history(frame, row)
        hist = hist[hist["actual"].notna()].reset_index(drop=True)
        n_train = len(hist)

        consensus_median = row.get("NFP_Consensus_Median", np.nan)
        consensus_mean = row.get("NFP_Consensus_Mean", np.nan)
        panel_top4 = row.get("NFP_Forecast_Dynamic_Top4_k12", np.nan)
        panel_top10 = row.get("NFP_Forecast_Dynamic_Top10_k12", np.nan)
        panel_robust = row.get("NFP_Forecast_Dynamic_RobustMedian", np.nan)

        add_prediction(preds, row, "consensus_mean", consensus_mean, n_train)
        add_prediction(preds, row, "consensus_median", consensus_median, n_train)
        add_prediction(preds, row, "dynamic_panel_top4", panel_top4, n_train)
        add_prediction(preds, row, "dynamic_panel_top10", panel_top10, n_train)
        add_prediction(preds, row, "dynamic_panel_robust_median", panel_robust, n_train)

        if n_train < min_train:
            continue

        base = consensus_median if pd.notna(consensus_median) else consensus_mean
        if pd.isna(base):
            continue

        hist = hist.copy()
        hist_base = pd.to_numeric(hist.get("NFP_Consensus_Median"), errors="coerce")
        hist_alt_base = pd.to_numeric(hist.get("NFP_Consensus_Mean"), errors="coerce")
        hist["base_forecast"] = hist_base.combine_first(hist_alt_base)
        hist["residual_to_base"] = hist["actual"] - hist["base_forecast"]
        cap = residual_cap(hist["residual_to_base"])

        bias_hist = hist["residual_to_base"].dropna().tail(24)
        if len(bias_hist) >= 6:
            shrink = min(1.0, len(bias_hist) / 18.0)
            bias = float(bias_hist.mean()) * shrink
            add_prediction(preds, row, "consensus_median_rolling_bias", base + np.clip(bias, -cap, cap), n_train)

        if panel_top4 == panel_top4 and "NFP_Forecast_Dynamic_Top4_k12" in hist.columns:
            panel_err = (hist["actual"] - pd.to_numeric(hist["NFP_Forecast_Dynamic_Top4_k12"], errors="coerce")).abs()
            cons_err = (hist["actual"] - hist["base_forecast"]).abs()
            recent = pd.DataFrame({"panel": panel_err, "cons": cons_err}).dropna().tail(24)
            if len(recent) >= 8:
                pred = panel_top4 if recent["panel"].mean() < recent["cons"].mean() else base
                source = "panel" if pred == panel_top4 else "consensus"
                add_prediction(preds, row, "rolling_best_anchor_panel_vs_consensus", pred, n_train, source)

        direct_features = [c for c in anchor_cols + target_dyn_cols if c in frame.columns]
        direct_pred, detail = ridge_predict(hist, row, direct_features, "actual")
        add_prediction(preds, row, "forecast_stack_ridge", direct_pred, n_train, detail)

        ranked_resid = rank_features(
            hist,
            residual_pool,
            hist["residual_to_base"],
            min_obs=36,
            top_n=top_n,
        )
        if ranked_resid:
            feature_usage.append({
                "ds": row["ds"],
                "model": "residual_ridge_top_features",
                "features": "|".join(ranked_resid[:20]),
                "n_ranked": len(ranked_resid),
            })
        resid_pred, detail = ridge_predict(hist, row, ranked_resid[:top_n], "residual_to_base")
        if pd.notna(resid_pred):
            add_prediction(
                preds,
                row,
                "residual_ridge_top_features",
                base + np.clip(resid_pred, -cap, cap),
                n_train,
                detail,
            )

        factor_features = ranked_resid[: min(max(60, top_n), 180)]
        factor_pred, detail = pca_residual_predict(
            hist,
            row,
            factor_features,
            target_col="residual_to_base",
            n_components=5,
        )
        if pd.notna(factor_pred):
            add_prediction(
                preds,
                row,
                "factor_residual_ridge",
                base + np.clip(factor_pred, -cap, cap),
                n_train,
                detail,
            )

        analog_features = ranked_resid[: min(50, top_n)]
        analog_pred, detail = analog_residual_predict(
            hist,
            row,
            analog_features,
            target_col="residual_to_base",
            k=8,
        )
        if pd.notna(analog_pred):
            add_prediction(
                preds,
                row,
                "analog_residual_knn",
                base + np.clip(analog_pred, -cap, cap),
                n_train,
                detail,
            )

    return pd.DataFrame([p.__dict__ for p in preds]), pd.DataFrame(feature_usage)


def metrics_for(group: pd.DataFrame) -> dict:
    g = group.dropna(subset=["actual", "predicted"]).copy()
    if g.empty:
        return {}
    err = g["predicted"] - g["actual"]
    out = {
        "n": int(len(g)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }
    noncovid = g[~pd.to_datetime(g["ds"]).isin(COVID_EXCLUDE_MONTHS)]
    if len(noncovid):
        e = noncovid["predicted"] - noncovid["actual"]
        out["noncovid_n"] = int(len(noncovid))
        out["noncovid_mae"] = float(np.mean(np.abs(e)))
        out["noncovid_rmse"] = float(np.sqrt(np.mean(e**2)))
    last24 = g.sort_values("ds").tail(24)
    if len(last24):
        e = last24["predicted"] - last24["actual"]
        out["last24_mae"] = float(np.mean(np.abs(e)))
        out["last24_rmse"] = float(np.sqrt(np.mean(e**2)))
    last60 = g.sort_values("ds").tail(60)
    if len(last60):
        e = last60["predicted"] - last60["actual"]
        out["last60_mae"] = float(np.mean(np.abs(e)))
        out["last60_rmse"] = float(np.sqrt(np.mean(e**2)))
    if "accel_correct" in g.columns:
        accel = pd.to_numeric(g["accel_correct"], errors="coerce").dropna()
        if not accel.empty:
            out["accel_acc"] = float(accel.mean())
    return out


def build_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in preds.groupby("model"):
        row = {"model": model}
        row.update(metrics_for(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["last60_mae", "mae"], na_position="last").reset_index(drop=True)


def summarize_feature_usage(feature_usage: pd.DataFrame) -> pd.DataFrame:
    if feature_usage.empty:
        return feature_usage
    counts: dict[str, int] = {}
    for features in feature_usage["features"].dropna():
        for feature in str(features).split("|"):
            if feature:
                counts[feature] = counts.get(feature, 0) + 1
    return (
        pd.DataFrame([{"feature": k, "selected_count": v} for k, v in counts.items()])
        .sort_values("selected_count", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1999-01", help="First month to load from snapshots.")
    parser.add_argument("--end", default=None, help="Last month to load from snapshots.")
    parser.add_argument("--score-start", default="2010-01", help="First month included in reported backtest.")
    parser.add_argument("--min-train", type=int, default=72)
    parser.add_argument("--max-feature-pool", type=int, default=2200)
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--lite", action="store_true", help="Use a hand-picked compact nowcast feature pool.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = parse_month(args.start)
    end = parse_month(args.end)
    score_start = parse_month(args.score_start)
    if score_start is None:
        raise ValueError("--score-start is required")

    design, selected_cols = load_snapshot_design(
        start=start,
        end=end,
        max_pool=args.max_feature_pool,
        lite=args.lite,
        progress_every=args.progress_every,
    )
    target = load_target()
    frame = design.merge(target, on="ds", how="left")
    frame = add_target_dynamics(frame)
    frame = frame.sort_values("ds").reset_index(drop=True)

    preds, feature_usage = run_walk_forward(
        frame,
        score_start=score_start,
        min_train=args.min_train,
        top_n=args.top_n,
    )
    metrics = build_metrics(preds)
    usage_summary = summarize_feature_usage(feature_usage)

    frame.to_parquet(out_dir / "design_frame.parquet", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    feature_usage.to_csv(out_dir / "feature_usage_by_month.csv", index=False)
    usage_summary.to_csv(out_dir / "feature_usage_summary.csv", index=False)

    manifest = {
        "snapshot_base": str(SNAPSHOT_BASE),
        "target_path": str(TARGET_PATH),
        "target": "SA revised y_mom",
        "pit_rule": "Each model trains on actual_available_date < target_release_date for the scored month.",
        "start": str(start.date()) if start is not None else None,
        "end": str(end.date()) if end is not None else None,
        "score_start": str(score_start.date()),
        "min_train": args.min_train,
        "max_feature_pool": args.max_feature_pool,
        "top_n": args.top_n,
        "lite": bool(args.lite),
        "selected_snapshot_columns": len(selected_cols),
        "loaded_rows": int(len(frame)),
        "scored_prediction_rows": int(len(preds)),
        "covid_note": "COVID months are winsorized via utils.transforms.winsorize_covid_period; treat 2020-03..2020-05 as diagnostic, not hard truth.",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {out_dir}")
    print(metrics.to_string(index=False))
    if not usage_summary.empty:
        print("\nTop selected residual features:")
        print(usage_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
