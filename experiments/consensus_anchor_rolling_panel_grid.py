"""PIT rolling economist-panel, Panel-Kalman fusion, and router grid.

This is a local experimental harness. It does not change train-all defaults,
production outputs, or AWS. It replaces the forward-biased fixed Top-4
economist panel with walk-forward panels selected from only prior, operationally
available evidence, then evaluates several ways to combine those panels with
the existing Kalman final layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sidecars.economist_panel_sidecar import (  # noqa: E402
    PanelConfig,
    _load_full_panel,
    _load_nfp_release_map,
)
from settings import DATA_PATH  # noqa: E402
from Train.Output_code.consensus_anchor_runner import (  # noqa: E402
    full_metrics,
    kalman_fusion,
)
from utils.transforms import COVID_EXCLUDE_MONTHS, is_covid_month  # noqa: E402


DEFAULT_WINDOWS = (6, 8, 12, 18, 24, 36)
DEFAULT_TOP_NS = (4, 6, 8, 10, 15, 20)
DEFAULT_COVERAGES = (0.70, 0.80)
DEFAULT_POOLING = (
    "equal_mean",
    "median",
    "trimmed_mean",
    "bias_corrected_mean",
    "inv_mae_weighted_mean",
    "inv_rmse_weighted_mean",
)
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "_output_pairing_baseline_pitfix"
DEFAULT_OUT_DIR = PROJECT_ROOT / "_output_panel_kalman_experiments"
TARGET_PATH = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"


@dataclass(frozen=True)
class GridSpec:
    track_window: int
    top_n: int
    min_coverage_pct: float
    pooling: str = "equal_mean"
    skip_covid: bool = False

    @property
    def label(self) -> str:
        cov = int(round(self.min_coverage_pct * 100))
        return f"w{self.track_window}_top{self.top_n}_cov{cov}_{self.pooling}"


def _parse_int_list(raw: str) -> Tuple[int, ...]:
    values = tuple(sorted({int(x) for x in str(raw).split(",") if x.strip()}))
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}")
    return values


def _parse_float_list(raw: str) -> Tuple[float, ...]:
    values = tuple(sorted({float(x) for x in str(raw).split(",") if x.strip()}))
    if not values:
        raise ValueError(f"Expected at least one float in {raw!r}")
    return values


def _parse_str_list(raw: str) -> Tuple[str, ...]:
    values = tuple(x.strip() for x in str(raw).split(",") if x.strip())
    if not values:
        raise ValueError(f"Expected at least one value in {raw!r}")
    return values


def _load_actuals_with_availability() -> pd.DataFrame:
    target = pd.read_parquet(TARGET_PATH)
    required = {"ds", "y_mom", "operational_available_date"}
    missing = required.difference(target.columns)
    if missing:
        raise RuntimeError(f"{TARGET_PATH} missing required columns: {sorted(missing)}")

    out = target[["ds", "y_mom", "operational_available_date"]].copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.to_period("M").dt.to_timestamp()
    out["actual"] = pd.to_numeric(out["y_mom"], errors="coerce")
    out["actual_available_date"] = pd.to_datetime(
        out["operational_available_date"],
        errors="coerce",
    )
    return (
        out[["ds", "actual", "actual_available_date"]]
        .drop_duplicates("ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _compute_track_record_pit(
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    target_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    track_window: int,
    *,
    skip_covid: bool = False,
) -> Tuple[pd.DataFrame, pd.Timestamp, int]:
    """Rank economist history using only values knowable before ``cutoff``."""
    window_end = target_month - pd.DateOffset(months=1)
    window_start = target_month - pd.DateOffset(months=track_window)
    window_months = pd.date_range(window_start, window_end, freq="MS")
    if skip_covid:
        window_months = window_months[~window_months.isin(COVID_EXCLUDE_MONTHS)]

    actual_window = actuals[
        actuals["ds"].isin(window_months)
        & actuals["actual"].notna()
        & actuals["actual_available_date"].notna()
        & (actuals["actual_available_date"] < cutoff)
    ][["ds", "actual"]].copy()

    n_scorable_months = int(actual_window["ds"].nunique())
    empty_cols = [
        "ident", "name", "mae", "rmse", "bias", "n", "coverage",
        "n_scorable_months", "trained_through",
    ]
    if n_scorable_months == 0:
        return pd.DataFrame(columns=empty_cols), target_month - pd.DateOffset(months=1), 0

    hist = panel[
        panel["ds"].isin(actual_window["ds"])
        & (panel["first_release_date"] < cutoff)
    ][["ds", "ident", "name", "forecast"]].copy()
    if hist.empty:
        return pd.DataFrame(columns=empty_cols), pd.Timestamp(actual_window["ds"].max()), n_scorable_months

    hist = hist.merge(actual_window, on="ds", how="inner")
    hist = hist.dropna(subset=["forecast", "actual"])
    if hist.empty:
        return pd.DataFrame(columns=empty_cols), pd.Timestamp(actual_window["ds"].max()), n_scorable_months

    hist["err"] = hist["forecast"].astype(float) - hist["actual"].astype(float)
    track = (
        hist.groupby(["ident", "name"], as_index=False)
        .agg(
            mae=("err", lambda s: float(np.mean(np.abs(s)))),
            rmse=("err", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("err", lambda s: float(np.mean(s))),
            n=("err", "size"),
        )
    )
    track["n"] = track["n"].astype(int)
    track["coverage"] = track["n"].astype(float) / float(n_scorable_months)
    track["n_scorable_months"] = n_scorable_months
    track["trained_through"] = pd.Timestamp(actual_window["ds"].max())
    return track, pd.Timestamp(actual_window["ds"].max()), n_scorable_months


def _pool_prediction(selected: pd.DataFrame, pooling: str) -> float:
    if selected.empty:
        return float("nan")
    raw = selected["forecast"].astype(float).to_numpy()
    if pooling == "equal_mean":
        return float(np.mean(raw))
    if pooling == "median":
        return float(np.median(raw))
    if pooling == "trimmed_mean":
        if raw.size >= 10:
            lo, hi = np.percentile(raw, [10, 90])
            raw = raw[(raw >= lo) & (raw <= hi)]
        return float(np.mean(raw))
    if pooling == "bias_corrected_mean":
        bias = selected["bias"].astype(float).to_numpy()
        return float(np.mean(raw - bias))
    if pooling in {"inv_mae_weighted_mean", "inv_rmse_weighted_mean"}:
        err_col = "mae" if pooling == "inv_mae_weighted_mean" else "rmse"
        denom = np.maximum(selected[err_col].astype(float).to_numpy(), 1.0) ** 2
        weights = 1.0 / denom
        weights = weights / weights.sum()
        return float(np.sum(weights * raw))
    raise ValueError(f"Unknown pooling variant: {pooling}")


def _panel_common_row(
    *,
    spec: GridSpec,
    target_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    actual: float,
    eligible: pd.DataFrame,
    calibrated: pd.DataFrame,
    selected: pd.DataFrame,
    pred: float,
    trained_through: pd.Timestamp,
    n_scorable: int,
) -> Dict[str, object]:
    selected_names = selected["name"].astype(str).tolist() if not selected.empty else []
    selected_idents = selected["ident"].astype(str).tolist() if not selected.empty else []
    selected_raw = selected["forecast"].astype(float).to_numpy() if not selected.empty else np.array([])
    missing_reason = ""
    if eligible.empty:
        missing_reason = "no_current_forecasts_before_release"
    elif calibrated.empty:
        missing_reason = "no_active_forecasters_pass_coverage"
    elif selected.empty or not np.isfinite(pred):
        missing_reason = "pooling_no_prediction"

    return {
        "model_id": f"panel_{spec.label}",
        "family": "panel",
        "track_window": int(spec.track_window),
        "top_n": int(spec.top_n),
        "min_coverage_pct": float(spec.min_coverage_pct),
        "pooling": spec.pooling,
        "ds": target_month,
        "actual": actual,
        "panel_pred_raw": pred,
        "predicted": pred,
        "panel_size": int(len(selected)),
        "eligible_count": int(len(eligible)),
        "calibrated_active_count": int(len(calibrated)),
        "n_scorable_months": int(n_scorable),
        "trained_through": trained_through,
        "latest_forecast_release": (
            pd.to_datetime(selected["first_release_date"]).max()
            if not selected.empty else pd.NaT
        ),
        "selected_names": "|".join(selected_names),
        "selected_idents": "|".join(selected_idents),
        "selected_mean_mae": float(selected["mae"].mean()) if not selected.empty else np.nan,
        "selected_mean_rmse": float(selected["rmse"].mean()) if not selected.empty else np.nan,
        "selected_mean_bias": float(selected["bias"].mean()) if not selected.empty else np.nan,
        "selected_mean_coverage": float(selected["coverage"].mean()) if not selected.empty else np.nan,
        "panel_dispersion_std": (
            float(np.std(selected_raw, ddof=1)) if selected_raw.size > 1 else 0.0
            if selected_raw.size == 1 else np.nan
        ),
        "panel_dispersion_iqr": (
            float(np.percentile(selected_raw, 75) - np.percentile(selected_raw, 25))
            if selected_raw.size > 1 else 0.0 if selected_raw.size == 1 else np.nan
        ),
        "missing_panel_reason": missing_reason,
        "target_release_date": cutoff,
    }


def _validate_panel_pit(panel_df: pd.DataFrame, release_map: pd.Series) -> None:
    if panel_df.empty:
        return
    tmp = panel_df.copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    tmp["trained_through"] = pd.to_datetime(tmp["trained_through"])
    leaked = tmp[tmp["trained_through"].notna() & (tmp["trained_through"] >= tmp["ds"])]
    if not leaked.empty:
        raise RuntimeError(
            "Rolling panel PIT validation failed: trained_through >= ds for "
            f"{leaked[['model_id', 'ds', 'trained_through']].head().to_dict('records')}"
        )
    cutoff = tmp["ds"].map(release_map).map(pd.Timestamp)
    release_leak = tmp[
        tmp["latest_forecast_release"].notna()
        & (pd.to_datetime(tmp["latest_forecast_release"]) >= cutoff)
    ]
    if not release_leak.empty:
        raise RuntimeError(
            "Rolling panel PIT validation failed: selected forecast released after cutoff for "
            f"{release_leak[['model_id', 'ds', 'latest_forecast_release']].head().to_dict('records')}"
        )


def build_rolling_panel_family(
    *,
    track_window: int,
    min_coverage_pct: float,
    top_ns: Sequence[int],
    pooling_variants: Sequence[str],
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    release_map: pd.Series,
    target_months: Sequence[pd.Timestamp],
    skip_covid: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build all Top-N/pooling variants for a window with shared PIT ranking."""
    actual_lookup = actuals.set_index("ds")["actual"]
    max_top = max(int(n) for n in top_ns)
    rows_by_id: Dict[str, List[Dict[str, object]]] = {}

    for target_month in target_months:
        cutoff = release_map.get(target_month)
        if pd.isna(cutoff):
            continue
        cutoff = pd.Timestamp(cutoff)
        eligible = panel[
            (panel["ds"] == target_month)
            & (panel["first_release_date"] < cutoff)
        ][["ident", "name", "forecast", "first_release_date"]].copy()

        track, trained_through, n_scorable = _compute_track_record_pit(
            panel,
            actuals,
            target_month,
            cutoff,
            int(track_window),
            skip_covid=skip_covid,
        )
        active_track = track[track["ident"].isin(eligible["ident"])] if not track.empty else track
        if not active_track.empty:
            calibrated = active_track[
                active_track["coverage"] >= float(min_coverage_pct)
            ].copy()
            ranked = calibrated.sort_values(
                ["mae", "rmse", "coverage", "n", "ident"],
                ascending=[True, True, False, False, True],
            ).head(max_top).reset_index(drop=True)
            ranked = ranked.merge(
                eligible[["ident", "forecast", "first_release_date"]],
                on="ident",
                how="left",
                validate="one_to_one",
            )
        else:
            calibrated = active_track
            ranked = pd.DataFrame()

        for top_n in top_ns:
            top = ranked.head(int(top_n)).copy() if not ranked.empty else ranked.copy()
            for pooling in pooling_variants:
                spec = GridSpec(
                    track_window=int(track_window),
                    top_n=int(top_n),
                    min_coverage_pct=float(min_coverage_pct),
                    pooling=str(pooling),
                    skip_covid=skip_covid,
                )
                pred = _pool_prediction(top, str(pooling)) if not top.empty else float("nan")
                row = _panel_common_row(
                    spec=spec,
                    target_month=pd.Timestamp(target_month),
                    cutoff=cutoff,
                    actual=actual_lookup.get(target_month, np.nan),
                    eligible=eligible,
                    calibrated=calibrated,
                    selected=top,
                    pred=pred,
                    trained_through=pd.Timestamp(trained_through),
                    n_scorable=int(n_scorable),
                )
                rows_by_id.setdefault(row["model_id"], []).append(row)

    out: Dict[str, pd.DataFrame] = {}
    for model_id, rows in rows_by_id.items():
        df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
        _validate_panel_pit(df, release_map)
        out[model_id] = df
    return out


def build_rolling_panel(
    spec: GridSpec,
    *,
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    release_map: pd.Series,
    target_months: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    family = build_rolling_panel_family(
        track_window=spec.track_window,
        min_coverage_pct=spec.min_coverage_pct,
        top_ns=(spec.top_n,),
        pooling_variants=(spec.pooling,),
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        target_months=target_months,
        skip_covid=spec.skip_covid,
    )
    return family[f"panel_{spec.label}"]


def _score_objective(actual: np.ndarray, pred: np.ndarray, objective: str) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("inf")
    a = actual[mask]
    p = pred[mask]
    mae = float(np.mean(np.abs(a - p)))
    if objective == "mae":
        return mae
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    if objective == "rmse":
        return rmse
    if objective == "hybrid":
        return mae + 0.15 * rmse
    raise ValueError(f"Unknown objective: {objective}")


def _hac_se(values: np.ndarray, max_lag: int = 3) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return float("nan")
    x = x - np.mean(x)
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        var += 2.0 * weight * gamma
    return float(np.sqrt(max(var, 0.0) / n))


def _block_bootstrap_ci(
    values: np.ndarray,
    *,
    block_size: int = 6,
    n_boot: int = 200,
    seed: int = 1729,
) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block_size + 1))
    means = []
    for _ in range(int(n_boot)):
        parts = []
        while sum(len(p) for p in parts) < n:
            s = int(rng.choice(starts))
            parts.append(x[s:s + block_size])
        sample = np.concatenate(parts)[:n]
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    actual_rows = df[df["actual"].notna()].sort_values("ds")
    if window == "all":
        return df["actual"].notna()
    n = 60 if window == "last60" else 36 if window == "last36" else None
    if n is None:
        raise ValueError(f"Unknown metric window: {window}")
    keep = set(actual_rows["ds"].tail(n))
    return df["actual"].notna() & df["ds"].isin(keep)


def _evaluate_candidate(
    *,
    model_id: str,
    family: str,
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    baseline = baseline.loc[:, ~baseline.columns.duplicated()].copy()
    frame = frame[["ds", "actual", "predicted"]].copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    baseline = baseline[["ds", "actual", "predicted"]].copy()
    baseline["ds"] = pd.to_datetime(baseline["ds"])

    row: Dict[str, object] = {"model_id": model_id, "family": family}
    if extra:
        row.update(extra)

    merged = frame.merge(
        baseline.rename(columns={"predicted": "baseline_pred"}),
        on=["ds", "actual"],
        how="inner",
    )
    for window in ("all", "last60", "last36"):
        mask = _window_mask(frame, window)
        sub = frame[mask & frame["predicted"].notna()]
        prefix = "" if window == "all" else f"{window}_"
        row[f"{prefix}N"] = int(len(sub))
        if sub.empty:
            for k in ("MAE", "RMSE", "Directional_Accuracy", "Acceleration_Accuracy"):
                row[f"{prefix}{k}"] = float("nan")
            continue
        metrics = full_metrics(
            sub["actual"].to_numpy(dtype=float),
            sub["predicted"].to_numpy(dtype=float),
            model_id,
            ds=sub["ds"],
        )
        for k in ("MAE", "RMSE", "Directional_Accuracy", "Acceleration_Accuracy", "Tail_MAE"):
            row[f"{prefix}{k}"] = metrics.get(k)

    comp = merged[merged["actual"].notna() & merged["predicted"].notna() & merged["baseline_pred"].notna()]
    if not comp.empty:
        abs_delta = (
            (comp["actual"] - comp["baseline_pred"]).abs()
            - (comp["actual"] - comp["predicted"]).abs()
        ).to_numpy(dtype=float)
        sq_delta = (
            (comp["actual"] - comp["baseline_pred"]) ** 2
            - (comp["actual"] - comp["predicted"]) ** 2
        ).to_numpy(dtype=float)
        lo, hi = _block_bootstrap_ci(abs_delta)
        row.update({
            "delta_MAE_vs_baseline": float(np.mean(abs_delta)),
            "delta_RMSE2_vs_baseline": float(np.mean(sq_delta)),
            "hac_se_delta_abs": _hac_se(abs_delta),
            "hac_se_delta_sq": _hac_se(sq_delta),
            "block_bootstrap_delta_abs_ci_low": lo,
            "block_bootstrap_delta_abs_ci_high": hi,
            "beats_baseline_abs_loss_months": int((abs_delta > 0).sum()),
            "comparison_months": int(len(comp)),
        })
    return row


def _panel_for_model(panel_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ds", "actual", "consensus_pred", "champion_pred", "nsa_pred",
        "nsa_raw_pred",
    ]
    base_cols = [c for c in cols if c in merged.columns]
    out = merged[base_cols].merge(
        panel_df.drop(columns=["actual"], errors="ignore"),
        on="ds",
        how="left",
        validate="one_to_one",
    )
    out["panel_consensus_mean"] = out["panel_pred_raw"]
    out["panel_consensus_count"] = out["panel_size"]
    out["panel_consensus_std"] = out["panel_dispersion_std"]
    return out.sort_values("ds").reset_index(drop=True)


def _classify_source(row: pd.Series) -> str:
    pred = row.get("predicted")
    if pd.isna(pred):
        return "missing"
    for source, col in (
        ("kalman", "kalman_pred"),
        ("panel", "panel_consensus_mean"),
        ("consensus", "consensus_pred"),
    ):
        val = row.get(col)
        if pd.notna(val) and abs(float(pred) - float(val)) < 1e-8:
            return source
    return "blend_or_fusion"


def adaptive_panel_kalman_fusion(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    *,
    trailing_window: int,
    nsa_weight_scale: float,
    panel_weight_scale: float,
    panel_max_precision_share: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Experiment-local Kalman fusion with panel quality-adjusted precision."""
    keep_cols = ["ds", "actual", "consensus_pred", "champion_pred"]
    for col in (
        "nsa_pred", "panel_consensus_mean", "panel_size",
        "panel_dispersion_std", "selected_mean_coverage",
        "selected_mean_mae", "selected_mean_rmse",
    ):
        if col in overlap_df.columns:
            keep_cols.append(col)
    df = overlap_df[keep_cols].copy()
    df = df.dropna(subset=["consensus_pred", "champion_pred"])
    df = df.sort_values("ds").reset_index(drop=True)

    has_nsa = "nsa_pred" in df.columns and df["nsa_pred"].notna().any()
    first_ds = df.iloc[0]["ds"] if not df.empty else pd.Timestamp.max
    prior_cons = consensus_df[
        consensus_df["consensus_pred"].notna()
        & consensus_df["actual"].notna()
        & (consensus_df["ds"] < first_ds)
    ]
    prior_err = (prior_cons["actual"] - prior_cons["consensus_pred"]).values
    r_c = float(np.var(prior_err[-60:], ddof=1)) if len(prior_err) >= 2 else 1.0
    prior_actuals = prior_cons["actual"].dropna().values[-60:]
    q = float(np.var(np.diff(prior_actuals), ddof=1)) if len(prior_actuals) >= 2 else 1.0
    r_m = r_c * 1.5
    r_a = r_c * 2.0

    x_hat = float(df.iloc[0]["consensus_pred"])
    p_var = q
    rows: List[Dict[str, object]] = []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        hist_valid = hist[hist["actual"].notna()]
        if len(hist_valid) >= 6:
            hist_clean = hist_valid[~is_covid_month(hist_valid["ds"])]
            if len(hist_clean) >= 4:
                cons_err = (hist_clean["actual"] - hist_clean["consensus_pred"]).values[-trailing_window:]
                model_err = (hist_clean["actual"] - hist_clean["champion_pred"]).values[-trailing_window:]
                actual_diff = np.diff(hist_clean["actual"].values[-trailing_window:])
                r_c = float(np.var(cons_err, ddof=1)) + 1e-6
                r_m = float(np.var(model_err, ddof=1)) + 1e-6
                if actual_diff.size >= 2:
                    q = float(np.var(actual_diff, ddof=1)) + 1e-6

        x_prior = x_hat
        p_prior = p_var + q
        info_prior = 1.0 / p_prior
        info_c = 1.0 / r_c
        info_m = 1.0 / r_m

        info_a = 0.0
        nsa_level = 0.0
        if has_nsa and pd.notna(row.get("nsa_pred")) and len(hist_valid) >= 2:
            nsa_level = float(row["nsa_pred"])
            nsa_hist = hist_valid[hist_valid["nsa_pred"].notna()]
            nsa_hist = nsa_hist[~is_covid_month(nsa_hist["ds"])] if not nsa_hist.empty else nsa_hist
            if len(nsa_hist) >= 4:
                nsa_err = (nsa_hist["actual"] - nsa_hist["nsa_pred"]).values[-trailing_window:]
                r_a = float(np.var(nsa_err, ddof=1)) + 1e-6
            info_a = float(nsa_weight_scale) / r_a

        info_panel = 0.0
        panel_obs = row.get("panel_consensus_mean")
        panel_quality = 0.0
        if pd.notna(panel_obs):
            hist_panel = (
                hist_valid[hist_valid["panel_consensus_mean"].notna()]
                if "panel_consensus_mean" in hist_valid.columns else pd.DataFrame()
            )
            hist_panel = hist_panel[~is_covid_month(hist_panel["ds"])] if not hist_panel.empty else hist_panel
            if len(hist_panel) >= 4:
                panel_err = (
                    hist_panel["actual"] - hist_panel["panel_consensus_mean"]
                ).values[-trailing_window:]
                r_panel = float(np.var(panel_err, ddof=1)) + 1e-6
            else:
                r_panel = max(r_c, r_m, r_a)
            size_factor = np.clip(float(row.get("panel_size", 0.0)) / 10.0, 0.15, 1.0)
            coverage_factor = np.clip(float(row.get("selected_mean_coverage", 0.0)), 0.15, 1.0)
            dispersion = row.get("panel_dispersion_std")
            if pd.isna(dispersion):
                dispersion_factor = 0.35
            else:
                dispersion_factor = 1.0 / (1.0 + max(float(dispersion), 0.0) / 75.0)
            panel_quality = float(np.clip(size_factor * coverage_factor * dispersion_factor, 0.05, 1.0))
            info_panel = max(float(panel_weight_scale), 0.0) * panel_quality / r_panel
            base_info = info_prior + info_c + info_m + info_a
            cap_share = max(0.0, min(0.95, float(panel_max_precision_share)))
            cap_info = (cap_share / max(1.0 - cap_share, 1e-6)) * base_info
            info_panel = min(info_panel, cap_info)

        total_info = info_prior + info_c + info_m + info_a + info_panel
        p_post = 1.0 / total_info
        x_post = p_post * (
            info_prior * x_prior
            + info_c * float(row["consensus_pred"])
            + info_m * float(row["champion_pred"])
            + (info_a * nsa_level if info_a > 0 else 0.0)
            + (info_panel * float(panel_obs) if info_panel > 0 else 0.0)
        )

        if pd.notna(row["actual"]):
            x_hat = float(row["actual"])
            p_var = 1e-6
        else:
            x_hat = float(x_post)
            p_var = float(p_post)
        rows.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": float(x_post),
            "consensus_pred": row["consensus_pred"],
            "panel_consensus_mean": panel_obs,
            "panel_precision_share": info_panel / total_info if total_info else 0.0,
            "panel_quality": panel_quality,
            "error": row["actual"] - x_post if pd.notna(row["actual"]) else np.nan,
        })

    res = pd.DataFrame(rows)
    metrics = full_metrics(res["actual"].values, res["predicted"].values, "Adaptive_Panel_Kalman", ds=res["ds"])
    return res, metrics


def _run_fusion_variant(
    *,
    mode: str,
    overlap: pd.DataFrame,
    consensus_df: pd.DataFrame,
    trailing_window: int,
    nsa_weight_scale: float,
    panel_weight_scale: float,
    panel_max_precision_share: float,
) -> pd.DataFrame:
    use_nsa = "nsa_pred" in overlap.columns and overlap["nsa_pred"].notna().any()
    if mode == "consensus_kalman_baseline":
        res, _ = kalman_fusion(
            overlap,
            consensus_df,
            trailing_window=trailing_window,
            use_nsa_accel=use_nsa,
            nsa_weight_scale=nsa_weight_scale,
        )
        return res
    if mode == "panel_replaces_consensus_kalman":
        modified = overlap.copy()
        modified["consensus_pred"] = modified["panel_consensus_mean"].combine_first(
            modified["consensus_pred"]
        )
        res, _ = kalman_fusion(
            modified,
            consensus_df,
            trailing_window=trailing_window,
            use_nsa_accel=use_nsa,
            nsa_weight_scale=nsa_weight_scale,
        )
        return res
    if mode == "panel_plus_consensus_kalman":
        res, _ = kalman_fusion(
            overlap,
            consensus_df,
            trailing_window=trailing_window,
            use_nsa_accel=use_nsa,
            nsa_weight_scale=nsa_weight_scale,
            use_panel_observation=True,
            panel_weight_scale=panel_weight_scale,
            panel_max_precision_share=panel_max_precision_share,
        )
        return res
    if mode == "adaptive_panel_kalman":
        res, _ = adaptive_panel_kalman_fusion(
            overlap,
            consensus_df,
            trailing_window=trailing_window,
            nsa_weight_scale=nsa_weight_scale,
            panel_weight_scale=panel_weight_scale,
            panel_max_precision_share=panel_max_precision_share,
        )
        return res
    raise ValueError(f"Unknown fusion mode: {mode}")


def _candidate_score(actual: pd.Series, pred: pd.Series, objective: str) -> float:
    return _score_objective(actual.to_numpy(dtype=float), pred.to_numpy(dtype=float), objective)


def _build_router_candidates(
    base: pd.DataFrame,
    extra_candidates: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, pd.Series]:
    panel_raw = pd.to_numeric(base["panel_consensus_mean"], errors="coerce")
    consensus = pd.to_numeric(base["consensus_pred"], errors="coerce")
    kalman = pd.to_numeric(base["kalman_pred"], errors="coerce")
    panel = panel_raw.combine_first(consensus)
    candidates: Dict[str, pd.Series] = {
        "panel": panel,
        "kalman": kalman,
        "consensus": consensus,
        "panel_missing_else_kalman": panel_raw.combine_first(kalman),
    }
    for w in (0.25, 0.35, 0.50, 0.65, 0.75):
        candidates[f"blend_panel_kalman_{w:.2f}"] = w * panel + (1.0 - w) * kalman
        candidates[f"blend_panel_consensus_{w:.2f}"] = w * panel + (1.0 - w) * consensus
    for t in (25, 50, 75, 100, 125, 150, 200):
        candidates[f"gate_kalman_cons_gt_{t}"] = panel.mask((kalman - consensus).abs() > t, kalman)
        candidates[f"gate_panel_kalman_gt_{t}"] = panel.mask((panel_raw - kalman).abs() > t, kalman)
    if "panel_dispersion_std" in base.columns:
        dispersion = pd.to_numeric(base["panel_dispersion_std"], errors="coerce")
        for t in (20, 35, 50, 75):
            candidates[f"gate_panel_disp_gt_{t}"] = panel.mask(dispersion > t, kalman)
    if "panel_size" in base.columns:
        size = pd.to_numeric(base["panel_size"], errors="coerce").fillna(0)
        for n in (4, 6, 8):
            candidates[f"gate_panel_count_lt_{n}"] = panel.mask(size < n, kalman)
    if extra_candidates:
        candidates.update(extra_candidates)
    return candidates


def build_panel_router_v2(
    base: pd.DataFrame,
    *,
    panel_model_id: str,
    extra_candidates: Optional[Dict[str, pd.Series]] = None,
    min_history: int = 24,
    objective: str = "mae",
) -> pd.DataFrame:
    df = base.sort_values("ds").reset_index(drop=True).copy()
    candidates = _build_router_candidates(df, extra_candidates)
    pred_matrix = pd.DataFrame({name: s.reset_index(drop=True) for name, s in candidates.items()})
    preds: List[float] = []
    rules: List[str] = []
    scores_out: List[float] = []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        hist_idx = hist.index[hist["actual"].notna()]
        if len(hist_idx) < int(min_history):
            chosen = "panel"
            score = float("nan")
        else:
            actual = df.loc[hist_idx, "actual"]
            best_name = ""
            best_score = float("inf")
            best_kalman_share = -1.0
            for name in pred_matrix.columns:
                cand = pred_matrix.loc[hist_idx, name]
                score_i = _candidate_score(actual, cand, objective)
                kalman_hist = df.loc[hist_idx, "kalman_pred"]
                kalman_share = float(np.nanmean(np.isclose(cand, kalman_hist, atol=1e-8)))
                if (
                    score_i < best_score - 1e-9
                    or (abs(score_i - best_score) <= 1e-9 and kalman_share > best_kalman_share)
                ):
                    best_name = name
                    best_score = score_i
                    best_kalman_share = kalman_share
            chosen = best_name
            score = best_score
        pred = pred_matrix.loc[i, chosen]
        if pd.isna(pred):
            pred = row["kalman_pred"] if pd.notna(row.get("kalman_pred")) else row["consensus_pred"]
            chosen = f"{chosen}_fallback"
        preds.append(float(pred))
        rules.append(chosen)
        scores_out.append(float(score) if np.isfinite(score) else np.nan)

    out = df.copy()
    out["model_id"] = f"router_v2_{panel_model_id}"
    out["family"] = "router_v2"
    out["predicted"] = preds
    out["selected_model"] = rules
    out["selection_score"] = scores_out
    out["effective_source"] = out.apply(_classify_source, axis=1)
    out["hindsight_winner"] = np.where(
        out["actual"].notna() & out["panel_consensus_mean"].notna()
        & ((out["actual"] - out["kalman_pred"]).abs() < (out["actual"] - out["panel_consensus_mean"]).abs()),
        "kalman",
        np.where(out["actual"].notna(), "panel_or_other", ""),
    )
    out["error"] = np.where(out["actual"].notna(), out["actual"] - out["predicted"], np.nan)
    return out


def _router_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.to_numeric(df["panel_consensus_mean"], errors="coerce")
    kalman = pd.to_numeric(df["kalman_pred"], errors="coerce")
    consensus = pd.to_numeric(df["consensus_pred"], errors="coerce")
    out = pd.DataFrame({
        "abs_kalman_minus_consensus": (kalman - consensus).abs(),
        "abs_panel_minus_consensus": (panel - consensus).abs(),
        "abs_panel_minus_kalman": (panel - kalman).abs(),
        "panel_minus_consensus": panel - consensus,
        "kalman_minus_consensus": kalman - consensus,
        "panel_size": pd.to_numeric(df.get("panel_size", 0.0), errors="coerce"),
        "panel_dispersion_std": pd.to_numeric(df.get("panel_dispersion_std", 0.0), errors="coerce"),
        "selected_mean_coverage": pd.to_numeric(df.get("selected_mean_coverage", 0.0), errors="coerce"),
        "selected_mean_mae": pd.to_numeric(df.get("selected_mean_mae", 0.0), errors="coerce"),
        "selected_mean_rmse": pd.to_numeric(df.get("selected_mean_rmse", 0.0), errors="coerce"),
    })
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_ridge_probability(X: np.ndarray, y: np.ndarray, x_now: np.ndarray) -> float:
    if X.shape[0] < X.shape[1] + 4 or len(np.unique(y)) < 2:
        return 0.5
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd <= 1e-9, 1.0, sd)
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)
    xs = np.nan_to_num((x_now - mu) / sd, nan=0.0)
    X_design = np.column_stack([np.ones(Xs.shape[0]), Xs])
    x_design = np.concatenate([[1.0], xs])
    alpha = 2.0
    penalty = np.eye(X_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y
    score = float(x_design @ beta)
    return float(1.0 / (1.0 + math.exp(-np.clip(score, -20.0, 20.0))))


def build_learned_router(
    base: pd.DataFrame,
    *,
    panel_model_id: str,
    threshold: float,
    min_history: int = 24,
    min_expected_edge: float = 0.0,
) -> pd.DataFrame:
    df = base.sort_values("ds").reset_index(drop=True).copy()
    features = _router_feature_frame(df)
    panel = pd.to_numeric(df["panel_consensus_mean"], errors="coerce").combine_first(df["consensus_pred"])
    kalman = pd.to_numeric(df["kalman_pred"], errors="coerce")
    actual = pd.to_numeric(df["actual"], errors="coerce")
    kalman_edge = (actual - panel).abs() - (actual - kalman).abs()
    label = (kalman_edge > 0).astype(float)

    preds: List[float] = []
    probs: List[float] = []
    selected: List[str] = []
    expected_edges: List[float] = []
    for i, row in df.iterrows():
        hist_idx = np.where(
            np.isfinite(actual.iloc[:i].to_numpy(dtype=float))
            & np.isfinite(panel.iloc[:i].to_numpy(dtype=float))
            & np.isfinite(kalman.iloc[:i].to_numpy(dtype=float))
        )[0]
        if len(hist_idx) < int(min_history) or pd.isna(panel.iloc[i]):
            prob = 0.5
            expected_edge = 0.0
            use_kalman = pd.isna(panel.iloc[i])
        else:
            X = features.iloc[hist_idx].to_numpy(dtype=float)
            y = label.iloc[hist_idx].to_numpy(dtype=float)
            prob = _fit_ridge_probability(X, y, features.iloc[i].to_numpy(dtype=float))
            hist_prob = []
            for j in hist_idx:
                prior_idx = hist_idx[hist_idx < j]
                if len(prior_idx) < int(min_history):
                    hist_prob.append(np.nan)
                else:
                    hist_prob.append(
                        _fit_ridge_probability(
                            features.iloc[prior_idx].to_numpy(dtype=float),
                            label.iloc[prior_idx].to_numpy(dtype=float),
                            features.iloc[j].to_numpy(dtype=float),
                        )
                    )
            hist_prob_arr = np.asarray(hist_prob, dtype=float)
            edge_hist = kalman_edge.iloc[hist_idx].to_numpy(dtype=float)
            edge_mask = np.isfinite(hist_prob_arr) & (hist_prob_arr >= float(threshold))
            expected_edge = float(np.nanmean(edge_hist[edge_mask])) if edge_mask.any() else 0.0
            use_kalman = bool(prob > float(threshold) and expected_edge > float(min_expected_edge))
        pred = kalman.iloc[i] if use_kalman else panel.iloc[i]
        if pd.isna(pred):
            pred = row["consensus_pred"]
        preds.append(float(pred))
        probs.append(float(prob))
        expected_edges.append(float(expected_edge))
        selected.append("kalman" if use_kalman else "panel")

    out = df.copy()
    out["model_id"] = f"learned_router_{panel_model_id}_thr{threshold:.2f}"
    out["family"] = "learned_router"
    out["predicted"] = preds
    out["kalman_win_probability"] = probs
    out["expected_kalman_edge"] = expected_edges
    out["selected_model"] = selected
    out["effective_source"] = out.apply(_classify_source, axis=1)
    out["hindsight_winner"] = np.where(
        actual.notna() & (kalman_edge > 0),
        "kalman",
        np.where(actual.notna(), "panel_or_other", ""),
    )
    out["error"] = np.where(out["actual"].notna(), out["actual"] - out["predicted"], np.nan)
    return out


def _canonical_prediction_frame(
    model_id: str,
    family: str,
    df: pd.DataFrame,
    *,
    source: Optional[str] = None,
) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    cols = ["ds", "actual", "predicted"]
    out = df[cols].copy()
    out["model_id"] = model_id
    out["family"] = family
    if source is not None:
        out["source"] = source
    return out[["model_id", "family", "ds", "actual", "predicted"] + (["source"] if source else [])]


def _make_report(metrics: pd.DataFrame, fixed_summary: Dict[str, object], out_dir: Path) -> None:
    top = metrics.sort_values(["MAE", "RMSE"], na_position="last").head(20)
    non_diagnostic = metrics[metrics["family"] != "fixed_panel_diagnostic"]
    promoted = non_diagnostic[
        (non_diagnostic["MAE"] < float(fixed_summary["baseline_mae"]))
        & (non_diagnostic["RMSE"] < float(fixed_summary["baseline_rmse"]))
        & (non_diagnostic["last36_RMSE"] <= float(fixed_summary["baseline_last36_rmse"]))
    ].sort_values(["MAE", "RMSE"])
    lines = [
        "# PIT Panel/Kalman Experiment Report",
        "",
        "## Baseline",
        f"- Consensus Kalman MAE/RMSE: {fixed_summary['baseline_mae']:.3f} / {fixed_summary['baseline_rmse']:.3f}",
        f"- Consensus Kalman last36 RMSE: {fixed_summary['baseline_last36_rmse']:.3f}",
        f"- Fixed forward-biased panel router MAE/RMSE: {fixed_summary.get('fixed_router_mae', np.nan):.3f} / {fixed_summary.get('fixed_router_rmse', np.nan):.3f}",
        "",
        "## Promotion Result",
    ]
    if promoted.empty:
        lines.append("- No PIT candidate beat consensus Kalman on full-window MAE and RMSE without degrading last36 RMSE.")
    else:
        best = promoted.iloc[0]
        lines.append(f"- Best promotable candidate: `{best['model_id']}` ({best['family']}) MAE/RMSE {best['MAE']:.3f} / {best['RMSE']:.3f}.")
    lines.extend([
        "",
        "## Top Candidates By MAE",
        top[["model_id", "family", "N", "MAE", "RMSE", "last36_RMSE", "delta_MAE_vs_baseline"]]
        .to_markdown(index=False),
        "",
        "## Forward-Biased Panel Note",
        "- The fixed Top-4 panel remains diagnostic only. Rolling PIT selection does not reproduce the hardcoded panel membership, so fixed-panel performance is not a valid promotion target.",
        "",
        "## Artifact Index",
        "- `grid_metrics.csv`: ranked panel/fusion/router metrics.",
        "- `monthly_predictions.csv`: long predictions table.",
        "- `router_decisions.csv`: selected router model/source per month.",
        "- `pit_audit.csv`: panel cutoff and trained-through checks.",
    ])
    (out_dir / "model_report.md").write_text("\n".join(lines) + "\n")


def _load_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0]
    return {k: float(row[k]) for k in ("MAE", "RMSE") if k in row and pd.notna(row[k])}


def run_grid(
    *,
    output_base: Path,
    out_dir: Path,
    windows: Iterable[int],
    top_ns: Iterable[int],
    coverages: Iterable[float],
    pooling_variants: Sequence[str],
    router_min_history: int,
    objective: str,
    max_panels_for_fusion: int,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    panel_raw = _load_full_panel()
    actuals = _load_actuals_with_availability()
    release_map = _load_nfp_release_map()
    merged_path = output_base / "consensus_anchor" / "merged_consensus_model.csv"
    kalman_path = output_base / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv"
    fixed_router_path = output_base / "consensus_anchor" / "panel_kalman_router" / "backtest_results.csv"
    if not merged_path.exists():
        raise FileNotFoundError(merged_path)
    if not kalman_path.exists():
        raise FileNotFoundError(kalman_path)

    merged = pd.read_csv(merged_path, parse_dates=["ds"]).sort_values("ds")
    baseline = pd.read_csv(kalman_path, parse_dates=["ds"]).sort_values("ds")
    baseline = baseline.rename(columns={"predicted": "baseline_pred"})
    baseline_frame = baseline[["ds", "actual", "baseline_pred"]].rename(columns={"baseline_pred": "predicted"})
    actual_months = baseline_frame[baseline_frame["actual"].notna()]["ds"]
    if start is None:
        start = pd.Timestamp(actual_months.min())
    if end is None:
        end = pd.Timestamp(baseline_frame["ds"].max())
    target_months = [
        pd.Timestamp(m)
        for m in baseline_frame["ds"].sort_values().unique()
        if start <= pd.Timestamp(m) <= end
    ]
    overlap_base = merged[merged["ds"].isin(target_months)].copy()
    consensus_df = merged[["ds", "actual", "consensus_pred"]].copy()
    kalman_base = baseline_frame.rename(columns={"predicted": "kalman_pred"})

    metrics_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    router_frames: List[pd.DataFrame] = []
    audit_frames: List[pd.DataFrame] = []

    metrics_rows.append(_evaluate_candidate(
        model_id="consensus_kalman_baseline",
        family="baseline",
        frame=baseline_frame,
        baseline=baseline_frame,
    ))
    prediction_frames.append(_canonical_prediction_frame(
        "consensus_kalman_baseline", "baseline", baseline_frame
    ))

    panel_frames: Dict[str, pd.DataFrame] = {}
    for window in windows:
        for coverage in coverages:
            family = build_rolling_panel_family(
                track_window=int(window),
                min_coverage_pct=float(coverage),
                top_ns=tuple(int(n) for n in top_ns),
                pooling_variants=pooling_variants,
                panel=panel_raw,
                actuals=actuals,
                release_map=release_map,
                target_months=target_months,
            )
            panel_frames.update(family)

    for model_id, panel_df in panel_frames.items():
        panel_df.to_csv(details_dir / f"{model_id}.csv", index=False)
        metrics_rows.append(_evaluate_candidate(
            model_id=model_id,
            family="panel",
            frame=panel_df.rename(columns={"panel_pred_raw": "predicted"}),
            baseline=baseline_frame,
            extra={
                "track_window": int(panel_df["track_window"].iloc[0]),
                "top_n": int(panel_df["top_n"].iloc[0]),
                "min_coverage_pct": float(panel_df["min_coverage_pct"].iloc[0]),
                "pooling": panel_df["pooling"].iloc[0],
                "months_with_panel": int(panel_df["panel_pred_raw"].notna().sum()),
                "months_missing_panel": int(panel_df["panel_pred_raw"].isna().sum()),
                "mean_panel_size": float(panel_df["panel_size"].mean()),
                "mean_calibrated_active_count": float(panel_df["calibrated_active_count"].mean()),
            },
        ))
        prediction_frames.append(_canonical_prediction_frame(
            model_id,
            "panel",
            panel_df.rename(columns={"panel_pred_raw": "predicted"}),
        ))
        audit_cols = [
            "model_id", "ds", "trained_through", "target_release_date",
            "latest_forecast_release", "panel_size", "eligible_count",
            "calibrated_active_count", "n_scorable_months",
            "missing_panel_reason", "selected_names",
        ]
        audit_frames.append(panel_df[audit_cols].copy())

    if fixed_router_path.exists():
        fixed_router = pd.read_csv(fixed_router_path, parse_dates=["ds"])
        fixed_router["model_id"] = "fixed_forward_biased_panel_router"
        fixed_router["family"] = "fixed_panel_diagnostic"
        fixed_router["effective_source"] = fixed_router.apply(_classify_source, axis=1)
        metrics_rows.append(_evaluate_candidate(
            model_id="fixed_forward_biased_panel_router",
            family="fixed_panel_diagnostic",
            frame=fixed_router,
            baseline=baseline_frame,
        ))
        prediction_frames.append(_canonical_prediction_frame(
            "fixed_forward_biased_panel_router", "fixed_panel_diagnostic", fixed_router
        ))
        router_frames.append(fixed_router)

    panel_metric_df = pd.DataFrame([r for r in metrics_rows if r["family"] == "panel"])
    selected_panel_ids = (
        panel_metric_df.sort_values(["MAE", "RMSE"], na_position="last")["model_id"]
        .head(int(max_panels_for_fusion))
        .tolist()
    )

    fusion_modes = (
        "panel_replaces_consensus_kalman",
        "panel_plus_consensus_kalman",
        "adaptive_panel_kalman",
    )
    trailing_windows = (8, 12, 18, 24)
    panel_weights = (0.25, 0.5, 1.0, 1.5)
    panel_caps = (0.15, 0.25, 0.40, 0.65)
    nsa_scales = (0.25, 0.4, 0.75, 1.0, 1.5)
    fusion_by_panel: Dict[str, Dict[str, pd.DataFrame]] = {}
    for panel_id in selected_panel_ids:
        panel_df = panel_frames[panel_id]
        overlap = _panel_for_model(panel_df, overlap_base)
        panel_fusions: Dict[str, pd.DataFrame] = {}
        for mode in fusion_modes:
            for tw in trailing_windows:
                for nsa_scale in nsa_scales:
                    weight_iter = (0.0,) if mode == "panel_replaces_consensus_kalman" else panel_weights
                    cap_iter = (0.0,) if mode == "panel_replaces_consensus_kalman" else panel_caps
                    for panel_weight in weight_iter:
                        for panel_cap in cap_iter:
                            model_id = (
                                f"{mode}__{panel_id}__tw{tw}"
                                f"__nsa{nsa_scale:.2f}__pw{panel_weight:.2f}__cap{panel_cap:.2f}"
                            )
                            res = _run_fusion_variant(
                                mode=mode,
                                overlap=overlap,
                                consensus_df=consensus_df,
                                trailing_window=int(tw),
                                nsa_weight_scale=float(nsa_scale),
                                panel_weight_scale=float(panel_weight),
                                panel_max_precision_share=float(panel_cap),
                            )
                            res["model_id"] = model_id
                            res["family"] = mode
                            panel_fusions[model_id] = res
                            metrics_rows.append(_evaluate_candidate(
                                model_id=model_id,
                                family=mode,
                                frame=res,
                                baseline=baseline_frame,
                                extra={
                                    "source_panel_model_id": panel_id,
                                    "trailing_window": int(tw),
                                    "nsa_weight_scale": float(nsa_scale),
                                    "panel_weight_scale": float(panel_weight),
                                    "panel_max_precision_share": float(panel_cap),
                                },
                            ))
                            prediction_frames.append(_canonical_prediction_frame(model_id, mode, res))
        fusion_by_panel[panel_id] = panel_fusions

    for panel_id in selected_panel_ids:
        panel_df = panel_frames[panel_id]
        base = _panel_for_model(panel_df, overlap_base).merge(
            kalman_base[["ds", "kalman_pred"]],
            on="ds",
            how="left",
            validate="one_to_one",
        )
        metrics_so_far = pd.DataFrame(metrics_rows)
        if "source_panel_model_id" in metrics_so_far.columns:
            top_fusion_ids = set(
                metrics_so_far[
                    (metrics_so_far["source_panel_model_id"] == panel_id)
                    & (metrics_so_far["family"].isin(fusion_modes))
                ]
                .sort_values(["MAE", "RMSE"], na_position="last")
                ["model_id"]
                .head(8)
                .tolist()
            )
        else:
            top_fusion_ids = set()
        fusion_candidates = {
            model_id: df.set_index("ds").reindex(base["ds"])["predicted"].reset_index(drop=True)
            for model_id, df in fusion_by_panel.get(panel_id, {}).items()
            if model_id in top_fusion_ids
        }
        router = build_panel_router_v2(
            base,
            panel_model_id=panel_id,
            extra_candidates=fusion_candidates,
            min_history=int(router_min_history),
            objective=objective,
        )
        metrics_rows.append(_evaluate_candidate(
            model_id=f"router_v2_{panel_id}",
            family="router_v2",
            frame=router,
            baseline=baseline_frame,
            extra={
                "source_panel_model_id": panel_id,
                "router_min_history": int(router_min_history),
                "router_objective": objective,
                "router_kalman_count": int((router["effective_source"] == "kalman").sum()),
                "router_panel_count": int((router["effective_source"] == "panel").sum()),
                "router_blend_or_fusion_count": int((router["effective_source"] == "blend_or_fusion").sum()),
            },
        ))
        prediction_frames.append(_canonical_prediction_frame(f"router_v2_{panel_id}", "router_v2", router))
        router_frames.append(router)

        for threshold in (0.50, 0.55, 0.60, 0.65):
            learned = build_learned_router(
                base,
                panel_model_id=panel_id,
                threshold=float(threshold),
                min_history=int(router_min_history),
                min_expected_edge=0.0,
            )
            model_id = f"learned_router_{panel_id}_thr{threshold:.2f}"
            metrics_rows.append(_evaluate_candidate(
                model_id=model_id,
                family="learned_router",
                frame=learned,
                baseline=baseline_frame,
                extra={
                    "source_panel_model_id": panel_id,
                    "router_min_history": int(router_min_history),
                    "learned_threshold": float(threshold),
                    "router_kalman_count": int((learned["effective_source"] == "kalman").sum()),
                    "router_panel_count": int((learned["effective_source"] == "panel").sum()),
                },
            ))
            prediction_frames.append(_canonical_prediction_frame(model_id, "learned_router", learned))
            router_frames.append(learned)

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values(["MAE", "RMSE"], na_position="last").reset_index(drop=True)
    monthly = pd.concat(prediction_frames, ignore_index=True)
    routers = pd.concat(router_frames, ignore_index=True) if router_frames else pd.DataFrame()
    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)
    monthly.to_csv(out_dir / "monthly_predictions.csv", index=False)
    routers.to_csv(out_dir / "router_decisions.csv", index=False)
    audit.to_csv(out_dir / "pit_audit.csv", index=False)

    fixed_summary = {
        "baseline_mae": float(metrics.loc[metrics["model_id"] == "consensus_kalman_baseline", "MAE"].iloc[0]),
        "baseline_rmse": float(metrics.loc[metrics["model_id"] == "consensus_kalman_baseline", "RMSE"].iloc[0]),
        "baseline_last36_rmse": float(metrics.loc[metrics["model_id"] == "consensus_kalman_baseline", "last36_RMSE"].iloc[0]),
    }
    fixed_stats = _load_summary(output_base / "consensus_anchor" / "panel_kalman_router" / "summary_statistics.csv")
    if fixed_stats:
        fixed_summary["fixed_router_mae"] = fixed_stats.get("MAE")
        fixed_summary["fixed_router_rmse"] = fixed_stats.get("RMSE")

    manifest = {
        "output_base": str(output_base),
        "windows": list(windows),
        "top_ns": list(top_ns),
        "coverages": list(coverages),
        "pooling_variants": list(pooling_variants),
        "selected_panel_ids_for_fusion": selected_panel_ids,
        "router_min_history": int(router_min_history),
        "objective": objective,
        "target_start": str(start.date()) if start is not None else None,
        "target_end": str(end.date()) if end is not None else None,
        "pit_validation": {
            "forecast_cutoff": "current-month economist first_release_date < target NFP release_date",
            "ranking_actual_cutoff": "prior revised actual operational_available_date < target NFP release_date",
            "ranking_window": "strictly prior target months only",
            "learned_router": "expanding-window fit uses only rows before prediction month",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    _make_report(metrics, fixed_summary, out_dir)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--windows", default=",".join(str(x) for x in DEFAULT_WINDOWS))
    parser.add_argument("--top-ns", default=",".join(str(x) for x in DEFAULT_TOP_NS))
    parser.add_argument("--coverages", default=",".join(str(x) for x in DEFAULT_COVERAGES))
    parser.add_argument("--pooling", default=",".join(DEFAULT_POOLING))
    parser.add_argument("--router-min-history", type=int, default=24)
    parser.add_argument("--objective", default="mae", choices=["mae", "rmse", "hybrid"])
    parser.add_argument("--max-panels-for-fusion", type=int, default=8)
    parser.add_argument("--start", default=None, help="YYYY-MM first target month")
    parser.add_argument("--end", default=None, help="YYYY-MM last target month")
    args = parser.parse_args(argv)

    windows = _parse_int_list(args.windows)
    top_ns = _parse_int_list(args.top_ns)
    coverages = _parse_float_list(args.coverages)
    pooling = _parse_str_list(args.pooling)
    start = pd.Timestamp(args.start).to_period("M").to_timestamp() if args.start else None
    end = pd.Timestamp(args.end).to_period("M").to_timestamp() if args.end else None

    metrics = run_grid(
        output_base=args.output_base,
        out_dir=args.out_dir,
        windows=windows,
        top_ns=top_ns,
        coverages=coverages,
        pooling_variants=pooling,
        router_min_history=int(args.router_min_history),
        objective=str(args.objective),
        max_panels_for_fusion=int(args.max_panels_for_fusion),
        start=start,
        end=end,
    )
    display_cols = [
        "model_id", "family", "N", "MAE", "RMSE", "last36_RMSE",
        "delta_MAE_vs_baseline",
    ]
    print(metrics[display_cols].head(25).to_string(index=False))
    print(f"\nWrote experiment artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
