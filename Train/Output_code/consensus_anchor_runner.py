"""
Post-training consensus-anchor integration.

Runs after the main train-all pipeline to produce the final SA-revised
forecast layer. The two first-class outputs are:

  - Kalman Fusion: consensus + NSA+adjustment + NSA acceleration signals fused
    through a random-walk information filter with adaptive measurement noise.
  - Panel/Kalman Router: a PIT walk-forward rule layer that chooses between the
    PIT economist-panel forecast, the Kalman forecast, and conservative blends
    using only earlier months with actuals.
  - Optional experiment: Panel-Replaces-Consensus Kalman, enabled only by
    ``NFP_ENABLE_PANEL_REPLACES_CONSENSUS_KALMAN=1``. This swaps the consensus
    level observation for a PIT rolling economist panel when available, then
    falls back to consensus when the rolling panel is missing.

Optuna with nested expanding-window CV tunes the Kalman parameters while the
router stays deterministic and PIT-selected from prior realized misses.

The merged consensus+model dataset is built on-the-fly from:
  - Master snapshots         (NFP_Consensus_Mean, PIT-correct per target month)
  - Economist panel features (PIT target-month snapshot row)
  - NSA + adjustment         (_output/NSA_plus_adjustment/backtest_results.csv)
  - NSA raw predictions      (_output/NSA_prediction/backtest_results.csv)

Outputs:
  _output/consensus_anchor/
  ├── main_models.json
  ├── merged_consensus_model.csv
  ├── baseline_consensus/        (raw analyst consensus mean, benchmark)
  ├── panel_consensus_mean/      (panel-only diagnostic)
  ├── kalman_fusion/             (main output)
  ├── panel_kalman_router/       (main output)
  ├── panel_replaces_consensus_kalman/ (optional gated experiment)
  └── comparison_metrics.csv

Note: AccelOverride and Kalman+AccelPostFilter were removed (2026-05-11)
because both consistently underperformed the Consensus baseline on the
60-month backtest window.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, DATA_PATH, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT, get_master_snapshots_dir
from Train.variance_metrics import compute_variance_kpis
from Train.Output_code.metrics import add_consensus_hit_rate_metrics
from Train.sandbox.output_utils import write_sandbox_output_bundle
from utils.transforms import (
    winsorize_covid_period,
    is_covid_month,
    COVID_EXCLUDE_MONTHS,
)
from experiments.sidecars.integration import (
    merge_sidecar_observations,
    sidecar_max_precision_share,
    sidecar_observation_columns,
    sidecar_router_enabled,
    sidecar_router_values,
)

logger = setup_logger(__file__, TEMP_DIR)

OUT_BASE = OUTPUT_DIR / "consensus_anchor"

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

TARGET_PARQUET = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"
CONSENSUS_FEATURE_COL = "NFP_Consensus_Mean"
CONSENSUS_MEDIAN_FEATURE_COL = "NFP_Consensus_Median"
# Hardcoded 4-economist + Top4Mean entries were removed: they were hand-picked
# off a future window (selection leakage). With this list empty, the panel
# loader below falls back to the PIT-safe dynamic panel aggregate columns under
# NFP_Forecast_Dynamic_*.
ECONOMIST_PANEL_FORECAST_COLS: List[str] = []
DYNAMIC_PANEL_PRIMARY_COL = "NFP_Forecast_Dynamic_Top10_k12"
DYNAMIC_PANEL_FORECAST_COLS: List[str] = [
    "NFP_Forecast_Dynamic_Top10_k12",
    "NFP_Forecast_Dynamic_Top4_k12",
    "NFP_Forecast_Dynamic_Top15_k12",
    "NFP_Forecast_Dynamic_RobustMedian",
    "NFP_Forecast_Dynamic_TrimmedMean10",
]
DYNAMIC_PANEL_META_COLS: List[str] = [
    "NFP_Forecast_Dynamic_PanelN",
    "NFP_Forecast_Dynamic_NCalibrated",
    "NFP_Forecast_Dynamic_DispersionStd",
    "NFP_Forecast_Dynamic_DispersionIqr",
    "NFP_Forecast_Dynamic_Top10TrackMae",
]

PANEL_REPLACES_CONSENSUS_ENV = "NFP_ENABLE_PANEL_REPLACES_CONSENSUS_KALMAN"
PANEL_ROUTER_SELECTION_LOOKBACK_ENV = "NFP_PANEL_ROUTER_SELECTION_LOOKBACK"
PANEL_ROUTER_OBJECTIVE_ENV = "NFP_ROUTER_OBJECTIVE"
PANEL_ROUTER_LEARNED_ENV = "NFP_ENABLE_LEARNED_FINAL_ROUTER"
PANEL_REPLACES_CONSENSUS_DIR = "panel_replaces_consensus_kalman"
PANEL_REPLACES_CONSENSUS_MODEL = "consensus_anchor_panel_replaces_consensus_kalman_experiment"
PANEL_REPLACES_CONSENSUS_FORECAST = "Panel_Replaces_Consensus_Kalman"

DEFAULT_PANEL_REPLACEMENT_CONFIG: Dict[str, object] = {
    # Experiment defaults only. Economist identities are selected dynamically
    # each target month from PIT trailing track records, never fixed here.
    "track_window": 8,
    "top_n": 8,
    "min_coverage_pct": 0.80,
    "pooling": "median",
    "skip_covid_track_record": False,
    "trailing_window": 18,
    "nsa_weight_scale": 0.40,
}

PANEL_ROUTER_TRAILING_EDGE_WINDOWS = (6, 8, 12, 18, 24, 36)
PANEL_ROUTER_TRAILING_EDGE_MARGINS = (-50, -25, -10, 0, 10, 25, 50)
PANEL_ROUTER_MIN_LOCAL_HISTORY = 6
PANEL_ROUTER_SUPPORTED_OBJECTIVES = {
    "mae", "rmse", "composite", "mae_hit", "rmse_hit", "composite_non_covid",
}
HMM_REGIME_METADATA_COLS = (
    "hmm_regime_label",
    "hmm_transition_risk",
    "hmm_surprise",
    "hmm_should_reselect",
    "hmm_trigger_class",
    "hmm_force_override",
)

# Minimum expanding-window history before producing a prediction
MIN_HISTORY = 12

PIT_DATE_COLS = ("target_release_date", "actual_available_date")


def _row_prediction_cutoff(row: "pd.Series") -> pd.Timestamp:
    """Operational cutoff for a forecast row.

    The target parquet carries the NFP release date for the row's target
    month. Final-layer history must be known strictly before that date.
    Older synthetic tests/artifacts do not have the column, so the fallback is
    a conservative month-start timestamp that preserves legacy chronology-only
    behavior for those frames.
    """
    cutoff = row.get("target_release_date", pd.NaT)
    if pd.notna(cutoff):
        return pd.Timestamp(cutoff)
    return pd.Timestamp(row["ds"])


def _actual_history_available_before(
    hist: pd.DataFrame,
    cutoff: pd.Timestamp,
    current_ds: pd.Timestamp,
) -> pd.DataFrame:
    """Rows with actuals operationally available before the current forecast."""
    if hist.empty or "actual" not in hist.columns:
        return hist.iloc[0:0].copy()
    out = hist[hist["actual"].notna()].copy()
    if out.empty:
        return out
    if "ds" in out.columns:
        out = out[pd.to_datetime(out["ds"]) < pd.Timestamp(current_ds)]
    if "actual_available_date" in out.columns and pd.notna(cutoff):
        avail = pd.to_datetime(out["actual_available_date"], errors="coerce")
        out = out[avail.notna() & (avail < pd.Timestamp(cutoff))]
    return out


def _panel_router_selection_lookback(default: int = 24) -> Optional[int]:
    """Recent strict-history window used to pick the router rule.

    ``0`` or a negative value restores the old all-history scoring behavior.
    The default is intentionally modest because the panel/Kalman edge changes
    over time as panel coverage and the model-side signal quality drift.
    """
    raw = str(os.getenv(PANEL_ROUTER_SELECTION_LOOKBACK_ENV, str(default))).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %d",
            PANEL_ROUTER_SELECTION_LOOKBACK_ENV,
            raw,
            default,
        )
        value = int(default)
    return None if value <= 0 else value


def _panel_router_objective(default: str = "mae_hit") -> str:
    raw = str(os.getenv(PANEL_ROUTER_OBJECTIVE_ENV, default)).strip().lower()
    if raw not in PANEL_ROUTER_SUPPORTED_OBJECTIVES:
        logger.warning(
            "Invalid %s=%r; using %s",
            PANEL_ROUTER_OBJECTIVE_ENV,
            raw,
            default,
        )
        return default
    return raw


def _learned_panel_router_enabled() -> bool:
    return _env_bool(PANEL_ROUTER_LEARNED_ENV, default=False)


def _available_actual_indices_before(df: pd.DataFrame, row_idx: int) -> np.ndarray:
    if row_idx <= 0:
        return np.array([], dtype=int)
    row = df.iloc[row_idx]
    hist = df.iloc[:row_idx]
    hist_valid = _actual_history_available_before(
        hist,
        _row_prediction_cutoff(row),
        pd.Timestamp(row["ds"]),
    )
    return hist_valid.index.to_numpy(dtype=int)


def _latest_available_actual_row(hist_valid: pd.DataFrame) -> Optional[pd.Series]:
    if hist_valid.empty:
        return None
    return hist_valid.sort_values("ds").iloc[-1]


def _month_gap(later, earlier) -> int:
    if pd.isna(later) or pd.isna(earlier):
        return 1
    later_p = pd.Timestamp(later).to_period("M")
    earlier_p = pd.Timestamp(earlier).to_period("M")
    return max(1, (later_p.year - earlier_p.year) * 12 + (later_p.month - earlier_p.month))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _full_metric_block(
    a: np.ndarray, p: np.ndarray, prefix: str = "",
) -> Dict[str, float]:
    """Compute the consensus_anchor metric suite over an arbitrary stratum.

    Returns NaN-filled entries when the stratum is empty so the comparison
    CSV's column set is stable across forecasts.
    """
    keys = (
        "RMSE", "MAE", "MSE", "ME_Bias", "MedAE", "SMAPE_pct",
        "Directional_Accuracy", "Acceleration_Accuracy",
        "STD_Ratio", "Diff_STD_Ratio", "Corr_Level", "Corr_Diff",
        "Diff_Sign_Accuracy", "Tail_MAE", "Extreme_Hit_Rate",
    )
    if a.size == 0:
        return {f"{prefix}{k}": float("nan") for k in keys}

    e = a - p
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mse = float(np.mean(e ** 2))
    me = float(np.mean(e))
    medae = float(np.median(np.abs(e)))

    dir_acc = float(np.mean(np.sign(a) == np.sign(p)))
    # Acceleration accuracy uses the operational "vs last actual" formula:
    # sign(p[m] - a[m-1]) vs sign(a[m] - a[m-1]).
    from Train.variance_metrics import acceleration_accuracy
    accel_acc = float(acceleration_accuracy(a, p))

    denom = (np.abs(a) + np.abs(p))
    smape = float(np.mean(2 * np.abs(e) / np.where(denom == 0, 1, denom)) * 100)

    vk = compute_variance_kpis(a, p)

    return {
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}MSE": mse,
        f"{prefix}ME_Bias": me,
        f"{prefix}MedAE": medae,
        f"{prefix}SMAPE_pct": smape,
        f"{prefix}Directional_Accuracy": dir_acc,
        f"{prefix}Acceleration_Accuracy": accel_acc,
        f"{prefix}STD_Ratio": float(vk["std_ratio"]),
        f"{prefix}Diff_STD_Ratio": float(vk["diff_std_ratio"]),
        f"{prefix}Corr_Level": float(vk["corr_level"]),
        f"{prefix}Corr_Diff": float(vk["corr_diff"]),
        f"{prefix}Diff_Sign_Accuracy": float(vk["diff_sign_accuracy"]),
        f"{prefix}Tail_MAE": float(vk["tail_mae"]),
        f"{prefix}Extreme_Hit_Rate": float(vk["extreme_hit_rate"]),
    }


def full_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    label: str,
    ds: "pd.Series | pd.DatetimeIndex | None" = None,
) -> Dict:
    """Compute the full metric suite for consensus anchor experiments,
    stratified by all / non-COVID / COVID-only when ``ds`` is provided.

    Args:
        actual: Actual values (NaN entries are filtered out together with the
            corresponding pred and ds entries).
        pred: Predicted values.
        label: Forecast name written into the 'Forecast' column.
        ds: Optional aligned datestamps. When provided, the output gains
            NonCovid_* and CovidOnly_* prefixed metric blocks plus
            N_NonCovid / N_Covid counts. When None, only the unprefixed
            metrics are returned (preserves legacy schema for any caller
            that does not pass ds).
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)

    if ds is not None:
        ds_arr = pd.to_datetime(pd.Series(ds).reset_index(drop=True))
    else:
        ds_arr = None

    finite_mask = np.isfinite(a) & np.isfinite(p)
    a = a[finite_mask]
    p = p[finite_mask]
    if ds_arr is not None:
        ds_arr = ds_arr[finite_mask].reset_index(drop=True)

    out: Dict[str, "float | int | str"] = {"Forecast": label, "N": int(a.size)}
    if a.size == 0:
        return out

    out.update(_full_metric_block(a, p, prefix=""))

    if ds_arr is not None:
        covid_mask = is_covid_month(ds_arr).to_numpy()
        non_covid_mask = ~covid_mask
        out.update(_full_metric_block(a[non_covid_mask], p[non_covid_mask],
                                      prefix="NonCovid_"))
        out.update(_full_metric_block(a[covid_mask], p[covid_mask],
                                      prefix="CovidOnly_"))
        out["N_NonCovid"] = int(non_covid_mask.sum())
        out["N_Covid"] = int(covid_mask.sum())

    return out


def _eval_exclude_covid_hitrate() -> bool:
    return _env_bool("NFP_EVAL_EXCLUDE_COVID_HITRATE", default=True)


def _metrics_with_consensus_hits(metrics: Dict, results_df: pd.DataFrame) -> Dict:
    """Attach consensus-relative hit rates when reference columns are present."""
    return add_consensus_hit_rate_metrics(
        metrics,
        results_df,
        exclude_covid_for_hitrate=_eval_exclude_covid_hitrate(),
    )


# ---------------------------------------------------------------------------
# Gated PIT rolling-panel replacement experiment
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def panel_replaces_consensus_enabled() -> bool:
    """Whether to emit the local Panel-Replaces-Consensus Kalman experiment."""
    return _env_bool(PANEL_REPLACES_CONSENSUS_ENV, default=False)


def kalman_hmm_regime_noise_enabled() -> bool:
    return _env_bool("NFP_KALMAN_HMM_REGIME_NOISE", default=False)


def _hmm_regime_alpha(label: object) -> float:
    regime = str(label or "").strip().lower()
    return {
        "stable": 0.25,
        "upward": 0.50,
        "recovery": 0.50,
        "volatile_up": 0.75,
        "volatile_down": 1.00,
        "crash": 1.50,
    }.get(regime, 0.50)


def _hmm_reset_multiplier(label: object, risk: float, reselected: bool) -> float:
    if not reselected:
        return 1.0
    regime = str(label or "").strip().lower()
    if regime in {"crash", "volatile_down"} or risk >= 0.60:
        return 5.0
    return 2.0


def _panel_replacement_config() -> Dict[str, object]:
    cfg = dict(DEFAULT_PANEL_REPLACEMENT_CONFIG)
    cfg["track_window"] = int(os.getenv("NFP_PANEL_REPLACE_WINDOW", cfg["track_window"]))
    cfg["top_n"] = int(os.getenv("NFP_PANEL_REPLACE_TOP_N", cfg["top_n"]))
    cfg["min_coverage_pct"] = float(
        os.getenv("NFP_PANEL_REPLACE_MIN_COVERAGE", cfg["min_coverage_pct"])
    )
    cfg["pooling"] = str(os.getenv("NFP_PANEL_REPLACE_POOLING", cfg["pooling"]))
    cfg["skip_covid_track_record"] = _env_bool(
        "NFP_PANEL_REPLACE_SKIP_COVID_TRACK",
        bool(cfg["skip_covid_track_record"]),
    )
    cfg["trailing_window"] = int(
        os.getenv("NFP_PANEL_REPLACE_TRAILING_WINDOW", cfg["trailing_window"])
    )
    cfg["nsa_weight_scale"] = float(
        os.getenv("NFP_PANEL_REPLACE_NSA_WEIGHT_SCALE", cfg["nsa_weight_scale"])
    )
    if cfg["pooling"] not in {
        "equal_mean", "median", "trimmed_mean", "bias_corrected_mean",
        "inv_mae_weighted_mean", "inv_rmse_weighted_mean",
    }:
        raise ValueError(f"Unsupported NFP_PANEL_REPLACE_POOLING={cfg['pooling']!r}")
    return cfg


def _load_sa_actuals_with_availability() -> pd.DataFrame:
    target = pd.read_parquet(TARGET_PARQUET)
    required = {"ds", "y_mom", "operational_available_date"}
    missing = required.difference(target.columns)
    if missing:
        raise RuntimeError(
            f"{TARGET_PARQUET} missing required columns for PIT panel ranking: "
            f"{sorted(missing)}"
        )
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


def _compute_panel_track_record_pit(
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    target_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    track_window: int,
    *,
    skip_covid: bool = False,
) -> Tuple[pd.DataFrame, pd.Timestamp, int]:
    """Rank economists using only prior actuals operationally known at cutoff."""
    window_end = target_month - pd.DateOffset(months=1)
    window_start = target_month - pd.DateOffset(months=int(track_window))
    window_months = pd.date_range(window_start, window_end, freq="MS")
    if skip_covid:
        window_months = window_months[~window_months.isin(COVID_EXCLUDE_MONTHS)]

    actual_window = actuals[
        actuals["ds"].isin(window_months)
        & actuals["actual"].notna()
        & actuals["actual_available_date"].notna()
        & (actuals["actual_available_date"] < cutoff)
    ][["ds", "actual"]].copy()
    n_scorable = int(actual_window["ds"].nunique())
    empty_cols = [
        "ident", "name", "mae", "rmse", "bias", "n", "coverage",
        "n_scorable_months", "trained_through",
    ]
    if n_scorable == 0:
        return pd.DataFrame(columns=empty_cols), target_month - pd.DateOffset(months=1), 0

    hist = panel[
        panel["ds"].isin(actual_window["ds"])
        & (panel["first_release_date"] < cutoff)
    ][["ds", "ident", "name", "forecast"]].copy()
    if hist.empty:
        return pd.DataFrame(columns=empty_cols), pd.Timestamp(actual_window["ds"].max()), n_scorable

    hist = hist.merge(actual_window, on="ds", how="inner").dropna(subset=["forecast", "actual"])
    if hist.empty:
        return pd.DataFrame(columns=empty_cols), pd.Timestamp(actual_window["ds"].max()), n_scorable

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
    track["coverage"] = track["n"].astype(float) / float(n_scorable)
    track["n_scorable_months"] = int(n_scorable)
    track["trained_through"] = pd.Timestamp(actual_window["ds"].max())
    return track, pd.Timestamp(actual_window["ds"].max()), n_scorable


def _pool_panel_replacement(selected: pd.DataFrame, pooling: str) -> float:
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
        return float(np.mean(raw - selected["bias"].astype(float).to_numpy()))
    if pooling in {"inv_mae_weighted_mean", "inv_rmse_weighted_mean"}:
        err_col = "mae" if pooling == "inv_mae_weighted_mean" else "rmse"
        denom = np.maximum(selected[err_col].astype(float).to_numpy(), 1.0) ** 2
        weights = 1.0 / denom
        weights = weights / weights.sum()
        return float(np.sum(weights * raw))
    raise ValueError(f"Unknown panel replacement pooling={pooling!r}")


def _build_rolling_panel_replacement(
    *,
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    release_map: pd.Series,
    target_months: List[pd.Timestamp],
    config: Dict[str, object],
) -> pd.DataFrame:
    """Build the gated rolling panel used to replace consensus in one experiment."""
    rows: List[Dict[str, object]] = []
    actual_lookup = actuals.set_index("ds")["actual"]
    track_window = int(config["track_window"])
    top_n = int(config["top_n"])
    min_coverage = float(config["min_coverage_pct"])
    pooling = str(config["pooling"])
    skip_covid = bool(config.get("skip_covid_track_record", False))

    for raw_month in sorted(pd.to_datetime(pd.Series(target_months)).dropna().unique()):
        target_month = pd.Timestamp(raw_month).to_period("M").to_timestamp()
        cutoff = release_map.get(target_month)
        if pd.isna(cutoff):
            rows.append({
                "ds": target_month,
                "actual": actual_lookup.get(target_month, np.nan),
                "panel_replacement_pred": np.nan,
                "panel_replacement_size": 0,
                "panel_replacement_eligible_count": 0,
                "panel_replacement_calibrated_count": 0,
                "panel_replacement_n_scorable_months": 0,
                "panel_replacement_trained_through": pd.NaT,
                "panel_replacement_latest_forecast_release": pd.NaT,
                "panel_replacement_selected_names": "",
                "panel_replacement_selected_idents": "",
                "panel_replacement_selected_mean_mae": np.nan,
                "panel_replacement_selected_mean_rmse": np.nan,
                "panel_replacement_selected_mean_coverage": np.nan,
                "panel_replacement_dispersion_std": np.nan,
                "panel_replacement_missing_reason": "missing_target_release_date",
                "panel_replacement_target_release_date": pd.NaT,
            })
            continue
        cutoff = pd.Timestamp(cutoff)
        eligible = panel[
            (panel["ds"] == target_month)
            & (panel["first_release_date"] < cutoff)
        ][["ident", "name", "forecast", "first_release_date"]].copy()

        track, trained_through, n_scorable = _compute_panel_track_record_pit(
            panel,
            actuals,
            target_month,
            cutoff,
            track_window,
            skip_covid=skip_covid,
        )
        active_track = track[track["ident"].isin(eligible["ident"])] if not track.empty else track
        if not active_track.empty:
            calibrated = active_track[active_track["coverage"] >= min_coverage].copy()
            ranked = (
                calibrated
                .sort_values(
                    ["mae", "rmse", "coverage", "n", "ident"],
                    ascending=[True, True, False, False, True],
                )
                .head(top_n)
                .reset_index(drop=True)
            )
            selected = ranked.merge(
                eligible[["ident", "forecast", "first_release_date"]],
                on="ident",
                how="left",
                validate="one_to_one",
            )
        else:
            calibrated = active_track
            selected = pd.DataFrame()

        pred = _pool_panel_replacement(selected, pooling) if not selected.empty else float("nan")
        selected_raw = (
            selected["forecast"].astype(float).to_numpy()
            if not selected.empty else np.array([])
        )
        if eligible.empty:
            missing_reason = "no_current_forecasts_before_release"
        elif calibrated.empty:
            missing_reason = "no_active_forecasters_pass_coverage"
        elif selected.empty or not np.isfinite(pred):
            missing_reason = "pooling_no_prediction"
        else:
            missing_reason = ""

        rows.append({
            "ds": target_month,
            "actual": actual_lookup.get(target_month, np.nan),
            "panel_replacement_pred": pred,
            "panel_replacement_size": int(len(selected)),
            "panel_replacement_eligible_count": int(len(eligible)),
            "panel_replacement_calibrated_count": int(len(calibrated)),
            "panel_replacement_n_scorable_months": int(n_scorable),
            "panel_replacement_trained_through": pd.Timestamp(trained_through),
            "panel_replacement_latest_forecast_release": (
                pd.to_datetime(selected["first_release_date"]).max()
                if not selected.empty else pd.NaT
            ),
            "panel_replacement_selected_names": (
                "|".join(selected["name"].astype(str).tolist()) if not selected.empty else ""
            ),
            "panel_replacement_selected_idents": (
                "|".join(selected["ident"].astype(str).tolist()) if not selected.empty else ""
            ),
            "panel_replacement_selected_mean_mae": (
                float(selected["mae"].mean()) if not selected.empty else np.nan
            ),
            "panel_replacement_selected_mean_rmse": (
                float(selected["rmse"].mean()) if not selected.empty else np.nan
            ),
            "panel_replacement_selected_mean_coverage": (
                float(selected["coverage"].mean()) if not selected.empty else np.nan
            ),
            "panel_replacement_dispersion_std": (
                float(np.std(selected_raw, ddof=1)) if selected_raw.size > 1
                else 0.0 if selected_raw.size == 1 else np.nan
            ),
            "panel_replacement_missing_reason": missing_reason,
            "panel_replacement_target_release_date": cutoff,
        })

    out = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    _validate_rolling_panel_replacement(out)
    return out


def _validate_rolling_panel_replacement(panel_df: pd.DataFrame) -> None:
    if panel_df.empty:
        return
    tmp = panel_df.copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    tmp["panel_replacement_trained_through"] = pd.to_datetime(
        tmp["panel_replacement_trained_through"],
        errors="coerce",
    )
    trained_leak = tmp[
        tmp["panel_replacement_trained_through"].notna()
        & (tmp["panel_replacement_trained_through"] >= tmp["ds"])
    ]
    if not trained_leak.empty:
        raise RuntimeError(
            "Panel-replacement PIT validation failed: trained_through >= ds for "
            f"{trained_leak[['ds', 'panel_replacement_trained_through']].head().to_dict('records')}"
        )

    tmp["panel_replacement_latest_forecast_release"] = pd.to_datetime(
        tmp["panel_replacement_latest_forecast_release"],
        errors="coerce",
    )
    tmp["panel_replacement_target_release_date"] = pd.to_datetime(
        tmp["panel_replacement_target_release_date"],
        errors="coerce",
    )
    release_leak = tmp[
        tmp["panel_replacement_latest_forecast_release"].notna()
        & tmp["panel_replacement_target_release_date"].notna()
        & (
            tmp["panel_replacement_latest_forecast_release"]
            >= tmp["panel_replacement_target_release_date"]
        )
    ]
    if not release_leak.empty:
        raise RuntimeError(
            "Panel-replacement PIT validation failed: selected forecast released "
            "after cutoff for "
            f"{release_leak[['ds', 'panel_replacement_latest_forecast_release', 'panel_replacement_target_release_date']].head().to_dict('records')}"
        )


def _attach_rolling_panel_replacement(
    merged: pd.DataFrame,
    *,
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Attach the full-panel rolling replacement signal to merged train-all data."""
    from experiments.sidecars.economist_panel_sidecar import (
        _load_full_panel,
        _load_nfp_release_map,
    )

    panel = _load_full_panel()
    release_map = _load_nfp_release_map()
    actuals = _load_sa_actuals_with_availability()
    target_months = pd.to_datetime(merged["ds"]).dropna().dt.to_period("M").dt.to_timestamp()
    target_month_list = list(target_months.drop_duplicates())
    panel_df = _build_rolling_panel_replacement(
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        target_months=target_month_list,
        config=config,
    )

    drop_cols = [
        c for c in merged.columns
        if c.startswith("panel_replacement_")
    ]
    merged_clean = merged.drop(columns=drop_cols, errors="ignore")
    attached = merged_clean.merge(
        panel_df.drop(columns=["actual"], errors="ignore"),
        on="ds",
        how="left",
    )
    manifest = {
        "enabled_by": PANEL_REPLACES_CONSENSUS_ENV,
        "config": dict(config),
        "panel_rows": int(len(panel_df)),
        "panel_available_rows": int(panel_df["panel_replacement_pred"].notna().sum()),
        "panel_missing_rows": int(panel_df["panel_replacement_pred"].isna().sum()),
        "missing_reasons": (
            panel_df["panel_replacement_missing_reason"]
            .replace("", "available")
            .value_counts(dropna=False)
            .to_dict()
        ),
        "pit_validation": (
            "current-month forecasts require first_release_date < target NFP "
            "release_date; economist ranking uses only prior actuals with "
            "operational_available_date < target release_date; trained_through < ds"
        ),
    }
    logger.info(
        "Panel replacement attached: %d available / %d rows (%s)",
        manifest["panel_available_rows"],
        manifest["panel_rows"],
        dict(manifest["missing_reasons"]),
    )
    return attached, panel_df, manifest


def build_panel_replaces_consensus_kalman(
    overlap_with_oos: pd.DataFrame,
    consensus_df: pd.DataFrame,
    *,
    config: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, Dict, Dict[str, object]]:
    """Run the gated experiment where PIT rolling panel replaces consensus.

    Missing rolling-panel months fall back to the original consensus anchor.
    This preserves production-safe behavior for recent months where local panel
    collection was historically spotty.
    """
    if config is None:
        config = _panel_replacement_config()
    if "panel_replacement_pred" not in overlap_with_oos.columns:
        raise RuntimeError("panel_replacement_pred missing; attach rolling panel first")

    modified = overlap_with_oos.copy()
    modified["consensus_original"] = modified["consensus_pred"]
    modified["anchor_source"] = np.where(
        modified["panel_replacement_pred"].notna(),
        "rolling_panel",
        "consensus_fallback",
    )
    modified["consensus_pred"] = modified["panel_replacement_pred"].combine_first(
        modified["consensus_pred"]
    )
    has_nsa = "nsa_pred" in modified.columns and modified["nsa_pred"].notna().any()
    res_df, metrics = kalman_fusion(
        modified,
        consensus_df,
        trailing_window=int(config["trailing_window"]),
        use_nsa_accel=has_nsa,
        nsa_weight_scale=float(config["nsa_weight_scale"]),
        use_panel_observation=False,
    )
    metrics["Forecast"] = PANEL_REPLACES_CONSENSUS_FORECAST

    audit_cols = [
        "ds", "consensus_original", "panel_replacement_pred", "anchor_source",
        "panel_replacement_size", "panel_replacement_eligible_count",
        "panel_replacement_calibrated_count",
        "panel_replacement_n_scorable_months",
        "panel_replacement_trained_through",
        "panel_replacement_latest_forecast_release",
        "panel_replacement_target_release_date",
        "panel_replacement_selected_names",
        "panel_replacement_selected_idents",
        "panel_replacement_selected_mean_mae",
        "panel_replacement_selected_mean_rmse",
        "panel_replacement_selected_mean_coverage",
        "panel_replacement_dispersion_std",
        "panel_replacement_missing_reason",
    ]
    audit_cols = [c for c in audit_cols if c in modified.columns]
    res_df = res_df.merge(modified[audit_cols], on="ds", how="left")
    if "consensus_original" in res_df.columns:
        res_df["consensus_pred"] = res_df["consensus_original"].combine_first(
            res_df["consensus_pred"]
        )
        metrics = _metrics_with_consensus_hits(metrics, res_df)

    manifest = {
        "enabled_by": PANEL_REPLACES_CONSENSUS_ENV,
        "model_id": PANEL_REPLACES_CONSENSUS_MODEL,
        "forecast": PANEL_REPLACES_CONSENSUS_FORECAST,
        "artifact_dir": f"consensus_anchor/{PANEL_REPLACES_CONSENSUS_DIR}",
        "config": dict(config),
        "kalman_params": {
            "trailing_window": int(config["trailing_window"]),
            "nsa_weight_scale": float(config["nsa_weight_scale"]),
            "use_nsa_accel": bool(has_nsa),
            "use_panel_observation": False,
        },
        "anchor_source_counts": res_df["anchor_source"].value_counts(dropna=False).to_dict(),
        "pit_validation": (
            "Panel values are built before fusion with current forecasts released "
            "before the target NFP release and prior ranking actuals only. "
            "Kalman/router history requires actual_available_date < target_release_date, "
            "so same-day prior revised actuals are excluded."
        ),
    }
    return res_df, metrics, manifest


# ---------------------------------------------------------------------------
# Data Loading (reuses logic from build_consensus_anchor_merged_variants.py)
# ---------------------------------------------------------------------------

def _load_consensus_pit(
    target_type: str = "sa",
    target_source: str = "revised",
) -> pd.DataFrame:
    """Load PIT consensus mean/median per ds via master snapshots.

    For each target month M, the master snapshot at M was built at ETL time with the
    strict filter ``release_date < M's NFP release date``, so any column value at row
    ``ds=M`` is PIT-correct by construction. We extract ``NFP_Consensus_Mean`` and,
    when present, ``NFP_Consensus_Median`` at that row.

    Replaces the prior "read latest Unifier snapshot + groupby('value','last')"
    approach, which carried no PIT enforcement and was vulnerable to silent leak if
    the upstream survey ever published post-NFP-release updates.
    """
    base_dir = get_master_snapshots_dir(target_type, target_source)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Master snapshots directory not found: {base_dir}. "
            "Run the ETL pipeline first."
        )

    snapshot_files = sorted(base_dir.glob("**/*.parquet"))
    if not snapshot_files:
        raise FileNotFoundError(f"No master snapshot parquet files under {base_dir}")

    rows: List[Dict] = []
    skipped_no_column = 0
    for snap_path in snapshot_files:
        try:
            obs_month = pd.to_datetime(snap_path.stem + "-01")
        except (ValueError, TypeError):
            continue

        try:
            snap = pd.read_parquet(
                snap_path,
                columns=["date", CONSENSUS_FEATURE_COL, CONSENSUS_MEDIAN_FEATURE_COL],
            )
        except (KeyError, ValueError):
            try:
                snap = pd.read_parquet(snap_path, columns=["date", CONSENSUS_FEATURE_COL])
            except (KeyError, ValueError):
                # NFP_Consensus_Mean was not selected into this snapshot.
                skipped_no_column += 1
                continue
        except Exception as exc:
            logger.warning("Failed to read consensus from %s: %s", snap_path, exc)
            continue

        snap["date"] = pd.to_datetime(snap["date"], errors="coerce")
        match = snap[snap["date"] == obs_month]
        if match.empty:
            continue

        val = match[CONSENSUS_FEATURE_COL].iloc[0]
        if pd.isna(val):
            continue

        row = {"ds": obs_month, "consensus_pred": float(val)}
        if CONSENSUS_MEDIAN_FEATURE_COL in match.columns:
            median_val = match[CONSENSUS_MEDIAN_FEATURE_COL].iloc[0]
            if pd.notna(median_val):
                row["consensus_median_pred"] = float(median_val)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No consensus values found via master snapshots under {base_dir} "
            f"({skipped_no_column} snapshots lacked the {CONSENSUS_FEATURE_COL} column)."
        )

    if skipped_no_column:
        logger.info(
            "Consensus PIT load: %d snapshots lacked %s and were skipped",
            skipped_no_column, CONSENSUS_FEATURE_COL,
        )

    out = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    # Apply COVID winsorization so the consensus_anchor stage sees consensus
    # on the same scale as the LightGBM training pipeline (which winsorizes
    # X_full upfront) and the SA actuals (which are pre-winsorized at parquet
    # write time). Without this, raw consensus -14,448 for Apr 2020 produces
    # an artificial +13,911 error vs the winsorized SA actual -537, inflating
    # the Consensus baseline's MAE in comparison_metrics.csv and leaking into
    # Kalman R_c estimation as a regime-shift bias.
    out_indexed = out.set_index("ds")
    value_cols = [
        c for c in ("consensus_pred", "consensus_median_pred")
        if c in out_indexed.columns
    ]
    out_indexed[value_cols] = winsorize_covid_period(out_indexed[value_cols])
    out = out_indexed.reset_index()

    return out


def _load_economist_panel_pit(
    target_type: str = "sa",
    target_source: str = "revised",
) -> pd.DataFrame:
    """Load PIT economist panel forecasts from each monthly master snapshot.

    ``NFP_Consensus_Mean`` is the broad analyst anchor. The curated
    ``NFP_Forecast_*`` panel is smaller and spottier, but local replay shows
    its simple cross-sectional mean is a strong final-level forecast when
    available. Values are read from the target month's own PIT snapshot row,
    so this has the same release cutoff discipline as ``_load_consensus_pit``.
    """
    base_dir = get_master_snapshots_dir(target_type, target_source)
    snapshot_files = sorted(base_dir.glob("**/*.parquet"))
    rows: List[Dict] = []
    skipped_no_columns = 0
    use_dynamic_panel = len(ECONOMIST_PANEL_FORECAST_COLS) == 0
    requested_cols = (
        DYNAMIC_PANEL_FORECAST_COLS + DYNAMIC_PANEL_META_COLS
        if use_dynamic_panel
        else ECONOMIST_PANEL_FORECAST_COLS
    )

    for snap_path in snapshot_files:
        try:
            obs_month = pd.to_datetime(snap_path.stem + "-01")
        except (ValueError, TypeError):
            continue

        columns = ["date", *requested_cols]
        try:
            snap = pd.read_parquet(snap_path, columns=columns)
        except (KeyError, ValueError):
            skipped_no_columns += 1
            continue
        except Exception as exc:
            logger.warning("Failed to read economist panel from %s: %s", snap_path, exc)
            continue

        snap["date"] = pd.to_datetime(snap["date"], errors="coerce")
        match = snap[snap["date"] == obs_month]
        if match.empty:
            continue

        if use_dynamic_panel:
            values = pd.to_numeric(
                match[DYNAMIC_PANEL_FORECAST_COLS].iloc[0],
                errors="coerce",
            )
            if not values.notna().any():
                continue

            primary = values.get(DYNAMIC_PANEL_PRIMARY_COL, np.nan)
            if pd.isna(primary):
                primary = values.dropna().iloc[0] if values.notna().any() else np.nan
            if pd.isna(primary):
                continue

            median = values.get("NFP_Forecast_Dynamic_RobustMedian", np.nan)
            if pd.isna(median):
                median = values.median()

            meta = pd.to_numeric(
                match[[c for c in DYNAMIC_PANEL_META_COLS if c in match.columns]].iloc[0],
                errors="coerce",
            )
            count = meta.get("NFP_Forecast_Dynamic_NCalibrated", np.nan)
            if pd.isna(count):
                count = meta.get("NFP_Forecast_Dynamic_PanelN", values.notna().sum())
            dispersion = meta.get("NFP_Forecast_Dynamic_DispersionStd", np.nan)

            row = {
                "ds": obs_month,
                "panel_consensus_mean": float(primary),
                "panel_consensus_median": float(median) if pd.notna(median) else np.nan,
                "panel_consensus_count": int(count) if pd.notna(count) else int(values.notna().sum()),
                "panel_consensus_std": float(dispersion) if pd.notna(dispersion) else np.nan,
                "panel_source": "dynamic_panel_aggregates",
            }
            for col, val in values.items():
                row[f"panel_{col}"] = float(val) if pd.notna(val) else np.nan
            for col, val in meta.items():
                row[f"panel_{col}"] = float(val) if pd.notna(val) else np.nan
        else:
            values = pd.to_numeric(
                match[ECONOMIST_PANEL_FORECAST_COLS].iloc[0],
                errors="coerce",
            )
            if not values.notna().any():
                continue

            row = {
                "ds": obs_month,
                "panel_consensus_mean": float(values.mean()),
                "panel_consensus_median": float(values.median()),
                "panel_consensus_count": int(values.notna().sum()),
                "panel_consensus_std": (
                    float(values.std()) if int(values.notna().sum()) > 1 else np.nan
                ),
                "panel_source": "fixed_economist_columns",
            }
            for col, val in values.items():
                row[f"panel_{col}"] = float(val) if pd.notna(val) else np.nan
        rows.append(row)

    if not rows:
        logger.warning(
            "No economist panel forecasts found via master snapshots under %s "
            "(%d snapshots lacked the panel columns)",
            base_dir, skipped_no_columns,
        )
        return pd.DataFrame(columns=["ds"])

    out = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    value_cols = [
        c for c in out.columns
        if c != "ds" and pd.api.types.is_numeric_dtype(out[c])
    ]
    out_indexed = out.set_index("ds")
    out_indexed[value_cols] = winsorize_covid_period(out_indexed[value_cols])
    out = out_indexed.reset_index()

    logger.info(
        "Economist panel PIT load: %d months (%s to %s), mean coverage %.1f forecasters",
        len(out),
        out["ds"].min().strftime("%Y-%m"),
        out["ds"].max().strftime("%Y-%m"),
        float(out["panel_consensus_count"].mean()),
    )
    return out


def _load_model_backtest(path: Path, pred_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["ds"])
    out = df[["ds", "actual", "predicted"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["predicted"] = pd.to_numeric(out["predicted"], errors="coerce")
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = out.rename(columns={"actual": f"actual_{pred_name}", "predicted": pred_name})
    return out


def _load_hmm_regime_metadata(output_base: Path) -> pd.DataFrame:
    path = output_base / "models" / "lightgbm_nfp" / "nsa_first_revised" / "hmm_regime_reselection.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ds"])
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read HMM regime metadata at %s: %s", path, exc)
        return pd.DataFrame(columns=["ds"])
    if "step_date" not in df.columns:
        return pd.DataFrame(columns=["ds"])
    df["ds"] = pd.to_datetime(df["step_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    rename = {
        "hmm_should_reselect": "hmm_reselected_this_month",
    }
    keep = ["ds"]
    for col in HMM_REGIME_METADATA_COLS:
        if col in df.columns:
            keep.append(col)
    out = df[keep].rename(columns=rename).dropna(subset=["ds"]).copy()
    for col in ("hmm_transition_risk", "hmm_surprise"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("hmm_reselected_this_month", "hmm_force_override"):
        if col in out.columns:
            out[col] = out[col].astype(bool)
    logger.info("Loaded HMM regime metadata: %d months", len(out))
    return out


def build_merged_dataset(output_base: Optional[Path] = None) -> pd.DataFrame:
    """
    Build merged consensus + model predictions dataset.

    Uses:
      - champion: SA blend walk-forward (sandbox)
      - challenger: SA revised direct
      - nsa_pred: NSA backtest predictions (for acceleration channel)
    """
    if output_base is None:
        output_base = OUTPUT_DIR

    consensus_monthly = _load_consensus_pit(target_type="sa", target_source="revised")
    panel_monthly = _load_economist_panel_pit(target_type="sa", target_source="revised")
    if not panel_monthly.empty:
        consensus_monthly = consensus_monthly.merge(panel_monthly, on="ds", how="left")
    logger.info(
        "Consensus loaded PIT-correctly via master snapshots: %d months (%s to %s)",
        len(consensus_monthly),
        consensus_monthly["ds"].min().strftime("%Y-%m"),
        consensus_monthly["ds"].max().strftime("%Y-%m"),
    )

    # Champion: NSA+Adjustment (best acceleration signal for SA target).
    # NSA+Adj outperforms SA blend as Kalman model channel because its
    # acceleration dynamics translate better to the SA target.
    champion_path = output_base / "NSA_plus_adjustment" / "backtest_results.csv"
    if not champion_path.exists():
        champion_path = output_base / "NSA_plus_adjustment_revised" / "backtest_results.csv"
    if not champion_path.exists():
        # Fallback to SA blend
        champion_path = output_base / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv"
        logger.warning("NSA+Adj not found for champion; falling back to SA blend")

    # SA LightGBM "challenger" is only a diagnostic overlay — Kalman fusion
    # itself does not consume it. With the SA branch retired the file may not
    # exist; fall back to the champion's actuals.
    challenger_path = output_base / "SA_prediction" / "backtest_results.csv"
    if not challenger_path.exists():
        challenger_path = output_base / "SA_prediction_revised" / "backtest_results.csv"

    champion_df = _load_model_backtest(champion_path, "champion_pred")
    if challenger_path.exists():
        challenger_df = _load_model_backtest(challenger_path, "challenger_pred")
        merged = (
            consensus_monthly
            .merge(champion_df, on="ds", how="outer")
            .merge(challenger_df, on="ds", how="outer")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        merged["actual"] = merged["actual_champion_pred"].combine_first(
            merged["actual_challenger_pred"]
        )
    else:
        logger.info("SA LightGBM challenger backtest not present — fusion will run without it")
        merged = (
            consensus_monthly
            .merge(champion_df, on="ds", how="outer")
            .sort_values("ds")
            .reset_index(drop=True)
        )
        merged["challenger_pred"] = np.nan
        merged["actual_challenger_pred"] = np.nan
        merged["actual"] = merged["actual_champion_pred"]
    logger.info("Champion: %s (%d months)", champion_path.parent.name,
                merged["champion_pred"].notna().sum())

    # NSA+Adjustment for the Kalman 3rd channel (same as champion if champion is NSA+adj)
    nsa_adj_path = output_base / "NSA_plus_adjustment" / "backtest_results.csv"
    if not nsa_adj_path.exists():
        nsa_adj_path = output_base / "NSA_plus_adjustment_revised" / "backtest_results.csv"
    if nsa_adj_path.exists():
        nsa_df = _load_model_backtest(nsa_adj_path, "nsa_pred")
        merged = merged.merge(nsa_df[["ds", "nsa_pred"]], on="ds", how="outer")
        logger.info("Loaded NSA+adjustment: %d months", merged["nsa_pred"].notna().sum())
    else:
        merged["nsa_pred"] = np.nan
        logger.warning("NSA+adjustment not found")

    # NSA raw predictions are retained for diagnostics and sidecar/router
    # experiments; the production Kalman channel uses NSA+adjustment above.
    nsa_raw_path = output_base / "NSA_prediction" / "backtest_results.csv"
    if not nsa_raw_path.exists():
        nsa_raw_path = output_base / "NSA_prediction_revised" / "backtest_results.csv"
    if nsa_raw_path.exists():
        nsa_raw_df = _load_model_backtest(nsa_raw_path, "nsa_raw_pred")
        merged = merged.merge(nsa_raw_df[["ds", "nsa_raw_pred"]], on="ds", how="outer")
        logger.info("Loaded NSA raw: %d months", merged["nsa_raw_pred"].notna().sum())
    else:
        merged["nsa_raw_pred"] = np.nan
        logger.warning("NSA raw not found; diagnostics will use fewer signals")

    hmm_meta = _load_hmm_regime_metadata(output_base)
    if not hmm_meta.empty:
        merged = merged.merge(hmm_meta, on="ds", how="left")

    # Backfill actuals and operational availability from the target parquet.
    # The dates are required by the final Kalman/router layer: a prior revised
    # actual is usable only when actual_available_date < current target release.
    if TARGET_PARQUET.exists():
        target_raw = pd.read_parquet(TARGET_PARQUET)
        keep = ["ds", "y_mom"]
        for col in ("release_date", "operational_available_date"):
            if col in target_raw.columns:
                keep.append(col)
        target = target_raw[keep].copy()
        target["ds"] = pd.to_datetime(target["ds"]).dt.to_period("M").dt.to_timestamp()
        target = target.rename(columns={"y_mom": "actual_from_target"})
        if "release_date" in target.columns:
            target["target_release_date"] = pd.to_datetime(
                target["release_date"],
                errors="coerce",
            )
        if "operational_available_date" in target.columns:
            target["actual_available_date"] = pd.to_datetime(
                target["operational_available_date"],
                errors="coerce",
            )
        target = target.drop(
            columns=["release_date", "operational_available_date"],
            errors="ignore",
        )
        merged = merged.merge(target, on="ds", how="left")
        merged["actual"] = merged["actual"].combine_first(merged["actual_from_target"])
        merged = merged.drop(columns=["actual_from_target"])

    merged = merge_sidecar_observations(merged, output_dir=output_base, logger=logger)

    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged


def split_datasets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into consensus-full (train), overlap (eval), and overlap+oos datasets.

    Returns:
        consensus_full: rows with consensus + actual (for Kalman noise init).
        overlap: rows with consensus + model + actual (for backtest eval).
        overlap_with_oos: overlap + future rows where actual is NaN but
            consensus and model predictions exist (for OOS prediction).
    """
    consensus_full = df[df["consensus_pred"].notna() & df["actual"].notna()].copy()
    overlap = df[
        df["consensus_pred"].notna()
        & df["actual"].notna()
        & df["champion_pred"].notna()
    ].copy()
    overlap_with_oos = df[
        df["consensus_pred"].notna()
        & df["champion_pred"].notna()
    ].copy()
    return consensus_full, overlap, overlap_with_oos


# ---------------------------------------------------------------------------
# Approach: Kalman Filter Fusion
# ---------------------------------------------------------------------------

def kalman_fusion(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    trailing_window: int = 18,
    use_model: bool = True,
    use_nsa_accel: bool = True,
    nsa_weight_scale: float = 1.0,
    use_panel_observation: bool = False,
    panel_weight_scale: float = 1.0,
    panel_max_precision_share: float = 0.65,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kalman filter fusing consensus, model, and NSA acceleration predictions.

    State-space model: x_t = x_{t-1} + w_t  (random walk)
    Observations:
      - consensus_t = x_t + v_c  (consensus prediction)
      - model_t = x_t + v_m      (SA blend champion)
      - nsa_accel_t: NSA-implied delta observation, converted to level
    Multi-observation update via information filter form.

    Args:
        overlap_df: Merged dataset with consensus, champion, nsa_pred, actual.
        consensus_df: Full consensus history for noise initialization.
        trailing_window: Trailing months for adaptive noise estimation.
        use_model: Whether to include the SA blend champion channel.
        use_nsa_accel: Whether to include the NSA acceleration channel.
        nsa_weight_scale: Multiplier for NSA channel precision (>1 = more trust).
        use_panel_observation: Whether to add the PIT economist panel mean as
            another observation channel when present.
        panel_weight_scale: Multiplier for the panel channel precision.
        panel_max_precision_share: Maximum share of posterior precision the
            panel channel may contribute at a step.
    """
    # Keep rows where consensus + model exist; actual can be NaN (OOS)
    keep_cols = ["ds", "actual", "consensus_pred", "champion_pred"]
    if "consensus_median_pred" in overlap_df.columns:
        keep_cols.append("consensus_median_pred")
    keep_cols.extend([c for c in PIT_DATE_COLS if c in overlap_df.columns])
    if "nsa_pred" in overlap_df.columns:
        keep_cols.append("nsa_pred")
    panel_cols = [c for c in ("panel_consensus_mean", "panel_consensus_count") if c in overlap_df.columns]
    keep_cols.extend(panel_cols)
    hmm_cols = [c for c in (
        "hmm_regime_label",
        "hmm_transition_risk",
        "hmm_surprise",
        "hmm_reselected_this_month",
        "hmm_trigger_class",
        "hmm_force_override",
    ) if c in overlap_df.columns]
    keep_cols.extend(hmm_cols)
    sidecar_cols = [c for c in overlap_df.columns if c.startswith("sidecar_")]
    keep_cols.extend(sidecar_cols)
    df = overlap_df[keep_cols].copy()
    df = df.dropna(subset=["consensus_pred", "champion_pred"])
    df = df.sort_values("ds").reset_index(drop=True)

    has_nsa = "nsa_pred" in df.columns and use_nsa_accel
    sidecar_obs_cols = sidecar_observation_columns(df)
    sidecar_active = bool(sidecar_obs_cols) or (bool(sidecar_cols) and sidecar_router_enabled())

    # Initialize noise parameters from consensus history strictly BEFORE the first
    # backtest month so the noise prior cannot peek into months that will later be
    # evaluated. Once `len(hist_valid) >= 6` inside the loop the per-step path takes
    # over and these inits no longer matter.
    first_backtest_ds = df.iloc[0]["ds"] if not df.empty else pd.Timestamp.max
    first_cutoff = _row_prediction_cutoff(df.iloc[0]) if not df.empty else pd.Timestamp.max
    prior_cons = consensus_df[consensus_df["consensus_pred"].notna()].copy()
    prior_cons = _actual_history_available_before(
        prior_cons,
        first_cutoff,
        pd.Timestamp(first_backtest_ds),
    )
    prior_cons_err = (prior_cons["actual"] - prior_cons["consensus_pred"]).values
    R_c_init = float(np.var(prior_cons_err[-60:], ddof=1)) if len(prior_cons_err) >= 2 else 1.0
    prior_actuals = prior_cons["actual"].dropna().values[-60:]
    Q_init = float(np.var(np.diff(prior_actuals), ddof=1)) if len(prior_actuals) >= 2 else 1.0

    x_hat = float(df.iloc[0]["consensus_pred"])
    P = Q_init

    # Most-recent variance estimates — re-used across the loop when a step's
    # COVID-clean trailing window is too small to re-estimate. Initialized to
    # the prior-history defaults computed above.
    R_c = R_c_init
    R_m = R_c_init * 1.5 if use_model else 1e12
    Q = Q_init
    R_a = R_c_init * 2.0  # NSA channel default

    last_state_actual_ds: Optional[pd.Timestamp] = None
    hmm_noise_active = kalman_hmm_regime_noise_enabled()
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        current_ds = pd.Timestamp(row["ds"])
        current_cutoff = _row_prediction_cutoff(row)
        # Use only historical rows whose actuals were operationally known
        # before the target release cutoff. The previous month's revised
        # actual is often released on the same date as the current target, so
        # chronological df.iloc[:i] alone is not strict enough.
        hist = df.iloc[:i]
        hist_valid = _actual_history_available_before(hist, current_cutoff, current_ds)
        latest_actual_row = _latest_available_actual_row(hist_valid)
        if latest_actual_row is not None:
            latest_actual_ds = pd.Timestamp(latest_actual_row["ds"])
            if last_state_actual_ds is None or latest_actual_ds > last_state_actual_ds:
                x_hat = float(latest_actual_row["actual"])
                P = 1e-6
                last_state_actual_ds = latest_actual_ds
        state_gap_months = (
            _month_gap(current_ds, last_state_actual_ds)
            if last_state_actual_ds is not None else 1
        )

        # COVID-clean view for variance computation: Mar/Apr/May 2020 are
        # winsorized (Fix 3) so values are flat and constant — including them
        # in var(...) collapses the trailing-window noise estimate. Excluding
        # them gives an honest estimate of the post-winsor noise floor for
        # non-COVID months.
        if len(hist_valid) >= 6:
            hist_clean = hist_valid[~is_covid_month(hist_valid["ds"])]
            if len(hist_clean) >= 4:
                recent_cons_err = (
                    hist_clean["actual"] - hist_clean["consensus_pred"]
                ).values[-trailing_window:]
                R_c = float(np.var(recent_cons_err, ddof=1)) + 1e-6
                if use_model:
                    recent_model_err = (
                        hist_clean["actual"] - hist_clean["champion_pred"]
                    ).values[-trailing_window:]
                    R_m = float(np.var(recent_model_err, ddof=1)) + 1e-6
                else:
                    R_m = 1e12
                recent_actual_diff = np.diff(
                    hist_clean["actual"].values[-trailing_window:]
                )
                Q = float(np.var(recent_actual_diff, ddof=1)) + 1e-6
            # else: keep R_c, R_m, Q at the most recent estimates (or inits)
        # else: keep R_c, R_m, Q at the most recent estimates (or inits)

        router_risk, router_confidence = sidecar_router_values(row)
        Q_step = Q * float(state_gap_months) * (1.0 + 0.75 * router_risk)
        R_m_step = R_m
        hmm_transition_risk = (
            float(row.get("hmm_transition_risk"))
            if pd.notna(row.get("hmm_transition_risk", np.nan))
            else 0.0
        )
        hmm_reselected_raw = row.get("hmm_reselected_this_month", False)
        hmm_reselected = bool(hmm_reselected_raw) if pd.notna(hmm_reselected_raw) else False
        hmm_q_multiplier = 1.0
        hmm_r_multiplier = 1.0
        if hmm_noise_active:
            alpha = _hmm_regime_alpha(row.get("hmm_regime_label"))
            hmm_r_multiplier = 1.0 + alpha * max(0.0, hmm_transition_risk)
            hmm_q_multiplier = _hmm_reset_multiplier(
                row.get("hmm_regime_label"),
                hmm_transition_risk,
                hmm_reselected,
            )
            Q_step *= hmm_q_multiplier
            R_m_step *= hmm_r_multiplier
        if router_confidence > 0 and use_model and len(hist_valid) >= 1:
            prev_actual = float(hist_valid["actual"].iloc[-1])
            champion_delta = float(row["champion_pred"]) - prev_actual
            side_sign_cols = [c for c in df.columns if c.startswith("sidecar_") and c.endswith("__predicted_accel_sign")]
            signs = [
                float(row[c]) for c in side_sign_cols
                if pd.notna(row.get(c)) and float(row[c]) != 0.0
            ]
            if signs and np.sign(np.mean(signs)) != np.sign(champion_delta):
                R_m_step = R_m * (1.0 + router_confidence)

        # Prediction step
        x_prior = x_hat
        P_prior = P + Q_step

        # Update step: multi-observation Kalman (information filter)
        info_prior = 1.0 / P_prior
        info_c = 1.0 / R_c
        info_m = 1.0 / R_m_step if use_model else 0.0

        # NSA acceleration channel: observes delta, converted to level
        info_a = 0.0
        nsa_level_implied = 0.0
        if has_nsa and pd.notna(row.get("nsa_pred")) and len(hist_valid) >= 2:
            prev_actual = float(hist_valid["actual"].iloc[-1])
            nsa_delta = float(row["nsa_pred"]) - prev_actual
            nsa_level_implied = prev_actual + nsa_delta

            # Estimate NSA delta noise from trailing window — also COVID-clean.
            if has_nsa and len(hist_valid) >= 6:
                nsa_hist = (
                    hist_valid[hist_valid["nsa_pred"].notna()]
                    if "nsa_pred" in hist_valid.columns
                    else pd.DataFrame()
                )
                nsa_hist_clean = (
                    nsa_hist[~is_covid_month(nsa_hist["ds"])]
                    if not nsa_hist.empty else nsa_hist
                )
                if len(nsa_hist_clean) >= 4:
                    # NSA delta error: (actual[t] - actual[t-1]) - (nsa_pred[t] - actual[t-1])
                    # = actual[t] - nsa_pred[t]
                    recent_nsa_err = (
                        nsa_hist_clean["actual"] - nsa_hist_clean["nsa_pred"]
                    ).values[-trailing_window:]
                    R_a = float(np.var(recent_nsa_err, ddof=1)) + 1e-6
                    info_a = nsa_weight_scale / R_a
                else:
                    # Insufficient COVID-clean NSA history — keep most-recent R_a
                    info_a = nsa_weight_scale / R_a
            else:
                info_a = nsa_weight_scale / R_a

        # Economist panel channel: a PIT cross-sectional economist mean read
        # from the target month's master snapshot. Its measurement noise is
        # estimated from prior panel misses only.
        info_panel = 0.0
        panel_obs = 0.0
        if (
            use_panel_observation
            and "panel_consensus_mean" in df.columns
            and pd.notna(row.get("panel_consensus_mean"))
        ):
            panel_obs = float(row["panel_consensus_mean"])
            hist_panel = (
                hist_valid[hist_valid["panel_consensus_mean"].notna()]
                if "panel_consensus_mean" in hist_valid.columns
                else pd.DataFrame()
            )
            hist_panel_clean = (
                hist_panel[~is_covid_month(hist_panel["ds"])]
                if not hist_panel.empty else hist_panel
            )
            if len(hist_panel_clean) >= 4:
                recent_panel_err = (
                    hist_panel_clean["actual"] - hist_panel_clean["panel_consensus_mean"]
                ).values[-trailing_window:]
                R_panel = float(np.var(recent_panel_err, ddof=1)) + 1e-6
            else:
                R_panel = max(R_c, R_m_step if use_model else R_c, R_a)
            info_panel = max(float(panel_weight_scale), 0.0) / R_panel

            base_info_for_panel = info_prior + info_c + info_m + info_a
            cap_share_panel = max(0.0, min(0.95, float(panel_max_precision_share)))
            if info_panel > 0 and cap_share_panel < 1.0:
                cap_info = (
                    cap_share_panel / max(1.0 - cap_share_panel, 1e-6)
                ) * base_info_for_panel
                info_panel = min(info_panel, cap_info)

        sidecar_terms: List[Tuple[float, float]] = []
        if sidecar_obs_cols and len(hist_valid) >= 6:
            for col in sidecar_obs_cols:
                obs = row.get(col)
                if pd.isna(obs):
                    continue
                hist_side = hist_valid[hist_valid[col].notna()] if col in hist_valid.columns else pd.DataFrame()
                hist_side = hist_side[~is_covid_month(hist_side["ds"])] if not hist_side.empty else hist_side
                if len(hist_side) >= 4:
                    recent_err = (hist_side["actual"] - hist_side[col]).values[-trailing_window:]
                    R_s = float(np.var(recent_err, ddof=1)) + 1e-6
                else:
                    R_s = max(R_c, R_m if use_model else R_c) * 4.0
                conf_col = col.replace("__predicted_mom", "__confidence")
                confidence = row.get(conf_col, 0.5)
                confidence = float(np.clip(confidence, 0.05, 1.0)) if pd.notna(confidence) else 0.25
                sidecar_terms.append((confidence / R_s, float(obs)))

        sidecar_info_raw = float(sum(info for info, _ in sidecar_terms))
        sidecar_info = sidecar_info_raw
        base_info = info_prior + info_c + info_m + info_a + info_panel
        cap_share = sidecar_max_precision_share()
        if sidecar_info > 0 and cap_share < 1.0:
            cap_info = (cap_share / max(1.0 - cap_share, 1e-6)) * base_info
            if sidecar_info > cap_info > 0:
                scale = cap_info / sidecar_info
                sidecar_terms = [(info * scale, obs) for info, obs in sidecar_terms]
                sidecar_info = cap_info

        total_info = info_prior + info_c + info_m + info_a + info_panel + sidecar_info
        P_post = 1.0 / total_info
        x_post = P_post * (
            info_prior * x_prior
            + info_c * row["consensus_pred"]
            + (info_m * row["champion_pred"] if use_model else 0.0)
            + (info_a * nsa_level_implied if info_a > 0 else 0.0)
            + (info_panel * panel_obs if info_panel > 0 else 0.0)
            + sum(info * obs for info, obs in sidecar_terms)
        )

        pred = x_post

        row_actual_available_now = (
            pd.notna(row["actual"])
            and (
                "actual_available_date" not in df.columns
                or (
                    pd.notna(row.get("actual_available_date"))
                    and pd.Timestamp(row["actual_available_date"]) < current_cutoff
                )
            )
        )
        if row_actual_available_now:
            x_hat = row["actual"]
            P = 1e-6
            last_state_actual_ds = current_ds
        else:
            x_hat = x_post
            P = P_post

        result_row = {
            "ds": row["ds"],
            "actual": row["actual"],
            "target_release_date": row.get("target_release_date", pd.NaT),
            "actual_available_date": row.get("actual_available_date", pd.NaT),
            "history_available_n": int(len(hist_valid)),
            "latest_available_actual_ds": (
                pd.Timestamp(latest_actual_row["ds"]) if latest_actual_row is not None else pd.NaT
            ),
            "state_gap_months": int(state_gap_months),
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "error": row["actual"] - pred if pd.notna(row["actual"]) else np.nan,
        }
        if "consensus_median_pred" in df.columns:
            result_row["consensus_median_pred"] = row.get("consensus_median_pred")
        if hmm_cols:
            result_row.update({
                "hmm_regime_label": row.get("hmm_regime_label"),
                "hmm_transition_risk": row.get("hmm_transition_risk"),
                "hmm_surprise": row.get("hmm_surprise"),
                "hmm_reselected_this_month": row.get("hmm_reselected_this_month"),
                "hmm_trigger_class": row.get("hmm_trigger_class"),
                "hmm_force_override": row.get("hmm_force_override"),
                "hmm_model_R_multiplier": hmm_r_multiplier,
                "hmm_process_Q_multiplier": hmm_q_multiplier,
            })
        if use_panel_observation and "panel_consensus_mean" in df.columns:
            result_row.update({
                "panel_consensus_mean": row.get("panel_consensus_mean"),
                "panel_precision_share": (
                    info_panel / total_info if total_info > 0 else 0.0
                ),
            })
        if sidecar_active:
            result_row.update({
                "sidecar_precision_share": sidecar_info / total_info if total_info > 0 else 0.0,
                "sidecar_router_risk": router_risk,
                "sidecar_router_confidence": router_confidence,
            })
        results.append(result_row)

    res_df = pd.DataFrame(results)
    label = "Kalman_Fusion" if use_model else "Kalman_Consensus_Only"
    if has_nsa:
        label += "_NSA"
    metrics = full_metrics(
        res_df["actual"].values, res_df["predicted"].values, label,
        ds=res_df["ds"],
    )
    metrics = _metrics_with_consensus_hits(metrics, res_df)
    return res_df, metrics


def _kalman_fold_runner(
    full_overlap: pd.DataFrame,
    consensus_df: pd.DataFrame,
    fn_kwargs: Dict,
    eval_end: int,
) -> pd.DataFrame:
    """Run kalman_fusion on overlap_df.iloc[:eval_end] and return its full res_df.

    `kalman_fusion` is per-step PIT-correct (uses df.iloc[:i] for noise estimation),
    so passing a chronological prefix preserves PIT for every row of the prefix.
    """
    res_df, _ = kalman_fusion(
        full_overlap.iloc[:eval_end],
        consensus_df,
        **fn_kwargs,
    )
    return res_df


def _walkforward_cv_score(
    overlap_df: pd.DataFrame,
    fold_runner,
    composite_objective_fn,
    n_splits: int = 5,
    min_train: int = 60,
) -> float:
    """Nested expanding-window CV score for a tuner trial.

    Splits ``overlap_df`` into ``n_splits`` chronological folds at the tail. For
    each fold, runs ``fold_runner(overlap_df, eval_end)``  on a prefix that
    includes that fold's eval window. Scores only on the fold's eval rows
    (which are strictly future relative to all earlier folds). Returns the mean
    composite score across folds.

    This breaks the Optuna meta-leak (Issue 9): hyperparameters are chosen by
    averaging composite scores over windows that each function has not "trained
    on" via earlier-fold actuals — every score is OOS w.r.t. the params being
    tuned.

    Args:
        overlap_df: Sorted DataFrame with [ds, actual, ...] (consensus + model).
        fold_runner: Callable(overlap_df, eval_end) -> res_df, where res_df has
            columns [ds, actual, predicted] and len(res_df) == eval_end.
        composite_objective_fn: Callable(actual, pred) -> float (lower is better).
        n_splits: Number of chronological folds at the tail.
        min_train: Minimum training prefix size (months) before the first fold.
            Set to ensure each fold's per-step noise estimation has enough
            history to switch off the init fallbacks.

    Returns:
        Mean composite score over chronological validation folds. If the
        requested ``min_train`` is too large for the available overlap window,
        it is reduced to leave at least one future validation row per split.
        This deliberately avoids the old full-window fallback, which tuned
        Kalman params on the same rows later reported as backtest performance.
    """
    overlap_df = overlap_df.sort_values("ds").reset_index(drop=True)
    n = len(overlap_df)

    if n <= MIN_HISTORY + 1:
        return float("inf")

    effective_splits = max(1, min(int(n_splits), n - MIN_HISTORY))
    preferred_eval_size = 3 if n >= MIN_HISTORY + effective_splits * 3 else 1
    effective_min_train = min(int(min_train), n - effective_splits * preferred_eval_size)
    effective_min_train = max(MIN_HISTORY, effective_min_train)
    if effective_min_train >= n:
        effective_min_train = n - 1
    eval_size = max(1, (n - effective_min_train) // effective_splits)

    fold_scores: List[float] = []
    for k in range(effective_splits):
        eval_start = effective_min_train + k * eval_size
        if eval_start >= n:
            break
        eval_end = n if k == effective_splits - 1 else min(
            n, effective_min_train + (k + 1) * eval_size
        )
        res_df = fold_runner(overlap_df, eval_end=eval_end)
        # Score ONLY on this fold's eval window (strictly future w.r.t. all
        # earlier folds; never overlaps with future folds either).
        eval_rows = res_df.iloc[eval_start:eval_end]
        actual = eval_rows["actual"].values.astype(float)
        pred = eval_rows["predicted"].values.astype(float)
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            continue
        fold_scores.append(float(composite_objective_fn(actual[mask], pred[mask])))

    if not fold_scores:
        return float("inf")
    return float(np.mean(fold_scores))


def _composite_kalman_accel_objective(
    actual: np.ndarray, pred: np.ndarray,
) -> float:
    """MAE - λ_accel * accel_acc - λ_dir * dir_acc — the composite objective
    used by `_tune_kalman` for nested expanding-window CV scoring."""
    from Train.config import KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if a.size == 0:
        return float("inf")
    mae = float(np.mean(np.abs(a - p)))
    if not np.isfinite(mae):
        return float("inf")
    dir_acc = float(np.mean(np.sign(a) == np.sign(p))) if a.size >= 1 else 0.0
    # Operational "vs last actual" accel formula.
    from Train.variance_metrics import acceleration_accuracy
    _acc = float(acceleration_accuracy(a, p))
    accel_acc = 0.0 if not np.isfinite(_acc) else _acc
    return float(mae - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc)


def _build_pit_adjustment_cache(
    target_dates: pd.Series,
    adj_history: pd.DataFrame,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Pre-compute PIT-filtered adjustment history for each target date.

    Done once outside the Optuna inner loop so per-trial cost is just the
    weight computation in ``ExpWeightedMedianCovidExcludedPredictor.fit_predict``
    (a function of ``half_life_years``), not the full filter sweep.
    """
    cache: Dict[pd.Timestamp, pd.DataFrame] = {}
    if "operational_available_date" in adj_history.columns:
        op_col = "operational_available_date"
    else:
        op_col = "ds"
    for ds in target_dates:
        target_ds = pd.Timestamp(ds)
        if op_col == "operational_available_date":
            mask = (
                adj_history["operational_available_date"].notna()
                & (adj_history["operational_available_date"] < target_ds)
            )
        else:
            mask = adj_history["ds"] < target_ds
        cache[target_ds] = adj_history[mask].reset_index(drop=True)
    return cache


def _compute_adjustment_series(
    target_dates: pd.Series,
    pit_cache: Dict[pd.Timestamp, pd.DataFrame],
    half_life_years: float,
) -> np.ndarray:
    """Compute the per-date predicted adjustment using ExpWeightedMedian with
    the given ``half_life_years``. Uses the pre-built PIT cache."""
    from Train.sandbox.experiment_predicted_adjustment import (
        ExpWeightedMedianCovidExcludedPredictor,
    )
    predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=half_life_years)
    out = np.zeros(len(target_dates), dtype=float)
    for i, ds in enumerate(target_dates):
        target_ds = pd.Timestamp(ds)
        avail = pit_cache.get(target_ds)
        if avail is None or avail.empty:
            out[i] = 0.0
        else:
            out[i] = float(predictor.fit_predict(avail, target_ds))
    return out


def _tune_kalman(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    n_trials: int = N_OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    n_splits: int = 5,
    *,
    adj_history: Optional[pd.DataFrame] = None,
    nsa_raw_by_ds: Optional[Dict[pd.Timestamp, float]] = None,
    tune_adjustment: bool = False,
) -> Dict:
    """Optuna-tune Kalman fusion params (and optionally the adjustment
    half-life) against the fusion-level composite objective.

    Uses nested expanding-window CV (n_splits chronological folds) so each
    trial's score is averaged over windows that the trial's params did not
    "train on" — closes Issue 9's meta-leak.

    Composite objective: MAE - λ_accel * accel_acc - λ_dir * dir_acc.

    When ``tune_adjustment=True`` and the raw NSA predictions + adjustment
    history are provided, ``half_life_years`` is sampled jointly with the
    Kalman params and the champion / nsa_pred columns of ``overlap_df`` are
    rebuilt in-memory for each trial. This drives the adjustment toward the
    same fusion objective rather than leaving it at a hard-coded 3-year
    half-life.
    """
    from Train.config import KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available; using default Kalman params")
        return {"trailing_window": 18, "nsa_weight_scale": 1.0, "half_life_years": 3.0}

    has_nsa = "nsa_pred" in overlap_df.columns and overlap_df["nsa_pred"].notna().any()

    _adj_enabled = (
        tune_adjustment
        and adj_history is not None
        and nsa_raw_by_ds is not None
        and not overlap_df.empty
    )
    pit_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    if _adj_enabled:
        pit_cache = _build_pit_adjustment_cache(overlap_df["ds"], adj_history)

    logger.info("Optuna tuning Kalman fusion: trials=%d timeout=%ds nsa=%s "
                "tune_adj=%s λ_accel=%.1f λ_dir=%.1f n_splits=%d (nested walkforward CV)",
                n_trials, timeout, has_nsa, _adj_enabled, KALMAN_LAMBDA_ACCEL,
                KALMAN_LAMBDA_DIR, n_splits)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        tw = trial.suggest_int("trailing_window", 6, 36)
        nsa_ws = trial.suggest_float("nsa_weight_scale", 0.1, 3.0) if has_nsa else 1.0

        if _adj_enabled:
            hl = trial.suggest_float("half_life_years", 0.5, 8.0)
            adj_vals = _compute_adjustment_series(overlap_df["ds"], pit_cache, hl)
            nsa_raw_vals = np.array([
                nsa_raw_by_ds.get(pd.Timestamp(d), np.nan) for d in overlap_df["ds"]
            ], dtype=float)
            champion_new = nsa_raw_vals + adj_vals
            modified_overlap = overlap_df.copy()
            modified_overlap["champion_pred"] = champion_new
            if "nsa_pred" in modified_overlap.columns:
                modified_overlap["nsa_pred"] = champion_new
        else:
            modified_overlap = overlap_df

        fn_kwargs = {
            "trailing_window": tw,
            "use_nsa_accel": has_nsa,
            "nsa_weight_scale": nsa_ws,
        }

        def _runner(full_overlap, eval_end):
            return _kalman_fold_runner(full_overlap, consensus_df, fn_kwargs, eval_end)

        score = _walkforward_cv_score(
            overlap_df=modified_overlap,
            fold_runner=_runner,
            composite_objective_fn=_composite_kalman_accel_objective,
            n_splits=n_splits,
        )
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial

    result = {"trailing_window": int(best.params["trailing_window"])}
    if has_nsa:
        result["nsa_weight_scale"] = float(best.params["nsa_weight_scale"])
    if _adj_enabled and "half_life_years" in best.params:
        result["half_life_years"] = float(best.params["half_life_years"])

    logger.info(
        "Kalman Optuna: best_obj=%.1f trailing_window=%d nsa_weight_scale=%.2f "
        "half_life_years=%s",
        best.value, result["trailing_window"],
        result.get("nsa_weight_scale", 1.0),
        f"{result['half_life_years']:.2f}" if "half_life_years" in result else "n/a (untuned)",
    )
    return result


def _kalman_tune_mode() -> str:
    mode = os.getenv("NFP_KALMAN_TUNE_MODE", "adaptive_grid").strip().lower()
    if mode not in {"adaptive_grid", "optuna_cv"}:
        return "adaptive_grid"
    return mode


def _load_nsa_adjustment_tuning_inputs(
    output_base: Path,
) -> Tuple[Optional[Dict[pd.Timestamp, float]], Optional[pd.DataFrame], Optional[Path]]:
    """Load raw NSA predictions and adjustment history for final-layer tuning."""
    from Train.sandbox.experiment_predicted_adjustment import load_adjustment_history

    nsa_raw_path = output_base / "NSA_prediction" / "backtest_results.csv"
    if not nsa_raw_path.exists():
        nsa_raw_path = output_base / "NSA_prediction_revised" / "backtest_results.csv"

    if not nsa_raw_path.exists():
        logger.warning(
            "NSA raw backtest not found at %s; half_life_years will NOT be tuned",
            nsa_raw_path,
        )
        return None, None, None

    _nsa_raw_df = pd.read_csv(nsa_raw_path, parse_dates=["ds"])
    _nsa_raw_df = _nsa_raw_df.dropna(subset=["predicted"])
    nsa_raw_by_ds = dict(zip(
        pd.to_datetime(_nsa_raw_df["ds"]).tolist(),
        _nsa_raw_df["predicted"].astype(float).tolist(),
    ))
    try:
        adj_history = load_adjustment_history()
    except Exception as e:
        logger.warning(
            "Failed to load adjustment history; half_life_years will NOT be tuned: %s",
            e,
        )
        adj_history = None
    return nsa_raw_by_ds, adj_history, nsa_raw_path


def _pit_adaptive_candidate_grid(
    *,
    has_nsa: bool,
    has_panel: bool,
    tune_adjustment: bool,
) -> List[Dict[str, object]]:
    """Small deterministic final-layer grid for prior-only PIT tuning.

    This replaces post-hoc global Optuna for reported backtests. The grid is
    intentionally compact: all candidates can be precomputed once, then each
    month selects among them using only earlier realized months.
    """
    hls: List[Optional[float]]
    if tune_adjustment:
        hls = [1.0, 1.25, 1.5, 2.0, 3.0]
    else:
        hls = [None]
    trailing_windows = [12, 14, 18, 23]
    nsa_scales = [0.25, 0.4, 0.75, 1.0, 1.5] if has_nsa else [1.0]
    panel_options: List[Tuple[bool, float]] = [(False, 0.0)]
    if has_panel:
        panel_options.extend([(True, 0.5), (True, 1.0)])

    out: List[Dict[str, object]] = []
    for hl in hls:
        for tw in trailing_windows:
            for nsa_ws in nsa_scales:
                for use_panel, panel_ws in panel_options:
                    out.append({
                        "half_life_years": hl,
                        "trailing_window": int(tw),
                        "nsa_weight_scale": float(nsa_ws),
                        "use_panel_observation": bool(use_panel),
                        "panel_weight_scale": float(panel_ws),
                    })
    return out


def _apply_adjustment_candidate(
    overlap_df: pd.DataFrame,
    *,
    half_life_years: Optional[float],
    pit_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]],
    nsa_raw_by_ds: Optional[Dict[pd.Timestamp, float]],
) -> pd.DataFrame:
    if half_life_years is None or pit_cache is None or nsa_raw_by_ds is None:
        return overlap_df

    modified = overlap_df.copy()
    adj_vals = _compute_adjustment_series(
        modified["ds"],
        pit_cache,
        float(half_life_years),
    )
    nsa_raw_vals = np.array([
        nsa_raw_by_ds.get(pd.Timestamp(d), np.nan) for d in modified["ds"]
    ], dtype=float)
    champion_new = nsa_raw_vals + adj_vals
    finite = np.isfinite(champion_new)
    modified.loc[finite, "champion_pred"] = champion_new[finite]
    if "nsa_pred" in modified.columns:
        modified.loc[finite, "nsa_pred"] = champion_new[finite]
    return modified


def _score_candidate_history(
    actual: np.ndarray,
    pred: np.ndarray,
    idx: np.ndarray,
    objective: str,
) -> float:
    if idx.size == 0:
        return float("inf")
    a = np.asarray(actual[idx], dtype=float)
    p = np.asarray(pred[idx], dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() == 0:
        return float("inf")
    a = a[mask]
    p = p[mask]
    mae = float(np.mean(np.abs(a - p)))
    if objective == "mae":
        return mae
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    if objective == "rmse":
        return rmse
    if objective == "hybrid":
        return mae + 0.15 * rmse
    raise ValueError(f"Unknown PIT adaptive Kalman objective: {objective}")


def pit_adaptive_kalman_fusion(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    *,
    adj_history: Optional[pd.DataFrame] = None,
    nsa_raw_by_ds: Optional[Dict[pd.Timestamp, float]] = None,
    min_history: int = 24,
    selection_lookback: Optional[int] = None,
    objective: str = "rmse",
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """PIT-safe adaptive final-layer tuner.

    For every row ``t``, this precomputes all candidate Kalman prediction
    streams, then selects the candidate using only rows with known actuals
    strictly before ``t``. Historical metrics therefore do not benefit from
    future hyperparameter outcomes. The OOS/live row still uses all known
    historical actuals, which is valid for a real forecast.
    """
    overlap_df = overlap_df.sort_values("ds").reset_index(drop=True)
    has_nsa = "nsa_pred" in overlap_df.columns and overlap_df["nsa_pred"].notna().any()
    has_panel = (
        "panel_consensus_mean" in overlap_df.columns
        and overlap_df["panel_consensus_mean"].notna().any()
    )
    tune_adjustment = adj_history is not None and nsa_raw_by_ds is not None
    pit_cache: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    if tune_adjustment:
        pit_cache = _build_pit_adjustment_cache(overlap_df["ds"], adj_history)

    candidates = _pit_adaptive_candidate_grid(
        has_nsa=has_nsa,
        has_panel=has_panel,
        tune_adjustment=tune_adjustment,
    )
    candidate_frames: List[pd.DataFrame] = []
    candidate_preds: List[np.ndarray] = []
    for candidate in candidates:
        candidate_overlap = _apply_adjustment_candidate(
            overlap_df,
            half_life_years=candidate["half_life_years"],
            pit_cache=pit_cache,
            nsa_raw_by_ds=nsa_raw_by_ds,
        )
        res_df, _ = kalman_fusion(
            candidate_overlap,
            consensus_df,
            trailing_window=int(candidate["trailing_window"]),
            use_nsa_accel=has_nsa,
            nsa_weight_scale=float(candidate["nsa_weight_scale"]),
            use_panel_observation=bool(candidate["use_panel_observation"]),
            panel_weight_scale=float(candidate["panel_weight_scale"]),
        )
        candidate_frames.append(res_df)
        candidate_preds.append(res_df["predicted"].to_numpy(dtype=float))

    actual = overlap_df["actual"].to_numpy(dtype=float)
    default_idx = 0
    for i, candidate in enumerate(candidates):
        if (
            (candidate["half_life_years"] in {None, 3.0})
            and int(candidate["trailing_window"]) == 18
            and abs(float(candidate["nsa_weight_scale"]) - 1.0) < 1e-12
            and not bool(candidate["use_panel_observation"])
        ):
            default_idx = i
            break

    selected_rows: List[pd.Series] = []
    selected_candidates: List[Dict[str, object]] = []
    selected_scores: List[float] = []
    for i in range(len(overlap_df)):
        hist_idx = _available_actual_indices_before(overlap_df, i)
        if hist_idx.size < int(min_history):
            best_idx = default_idx
            best_score = float("nan")
        else:
            score_idx = hist_idx[-int(selection_lookback):] if selection_lookback else hist_idx
            scores = [
                _score_candidate_history(actual, pred, score_idx, objective)
                for pred in candidate_preds
            ]
            best_idx = int(np.argmin(scores))
            best_score = float(scores[best_idx])

        row = candidate_frames[best_idx].iloc[i].copy()
        candidate = candidates[best_idx]
        row["selected_half_life_years"] = candidate["half_life_years"]
        row["selected_trailing_window"] = int(candidate["trailing_window"])
        row["selected_nsa_weight_scale"] = float(candidate["nsa_weight_scale"])
        row["selected_use_panel_observation"] = bool(candidate["use_panel_observation"])
        row["selected_panel_weight_scale"] = float(candidate["panel_weight_scale"])
        row["selection_history_n"] = int(hist_idx.size)
        row["selection_score"] = best_score
        selected_rows.append(row)
        selected_candidates.append(candidate)
        selected_scores.append(best_score)

    out = pd.DataFrame(selected_rows).sort_values("ds").reset_index(drop=True)
    metrics = full_metrics(
        out["actual"].values,
        out["predicted"].values,
        "Kalman_Fusion_NSA" if has_nsa else "Kalman_Fusion",
        ds=out["ds"],
    )
    metrics = _metrics_with_consensus_hits(metrics, out)

    live_params = selected_candidates[-1] if selected_candidates else {}
    manifest = {
        "mode": "adaptive_grid",
        "pit_validation": (
            "each row selects final-layer params using only earlier rows with actuals; "
            "the OOS row uses all known historical actuals"
        ),
        "candidate_count": len(candidates),
        "min_history": int(min_history),
        "selection_lookback": selection_lookback,
        "selection_objective": objective,
        "live_params": live_params,
        "has_panel_observation_candidates": bool(has_panel),
        "tunes_adjustment_half_life": bool(tune_adjustment),
    }
    return out, metrics, manifest


# ---------------------------------------------------------------------------
# Comparison Visualization
# ---------------------------------------------------------------------------

# Stable colors per forecast across the overlay + bar chart.
_FORECAST_COLORS = {
    "Baseline_Consensus": "#DC2626",        # red
    "Panel_Consensus_Mean": "#059669",      # green
    "Panel_Kalman_Router": "#7C3AED",       # purple
    PANEL_REPLACES_CONSENSUS_FORECAST: "#EA580C",  # orange
    "Kalman_Fusion_NSA": "#2563EB",         # blue
    "Kalman_Fusion": "#2563EB",
    "Baseline_Champion": "#9CA3AF",         # gray
}


def _color_for(label: str) -> str:
    if label in _FORECAST_COLORS:
        return _FORECAST_COLORS[label]
    if label.startswith("Kalman_Fusion"):
        return _FORECAST_COLORS["Kalman_Fusion"]
    return "#374151"


def write_comparison_visualization(
    out_dir: Path,
    forecast_dfs: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
) -> None:
    """
    Produce a unified comparison view across consensus-anchor forecasts:
      - comparison_overlay.png  (actual vs each forecast, full backtest)
      - comparison_metrics.png  (MAE / RMSE / DirAcc / AccelAcc bar chart)
      - comparison_scorecard.html  (sortable metrics table + image grid)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Overlay plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actuals once from whichever forecast has them
    actual_series = None
    for df in forecast_dfs.values():
        if "actual" in df.columns and df["actual"].notna().any():
            actual_series = df[["ds", "actual"]].dropna(subset=["actual"]).sort_values("ds")
            break
    if actual_series is not None:
        ax.plot(actual_series["ds"], actual_series["actual"],
                color="black", linewidth=1.8, marker="o", markersize=3, label="Actual",
                zorder=10)

    for label, df in forecast_dfs.items():
        plot_df = df[["ds", "predicted"]].dropna().sort_values("ds")
        if plot_df.empty:
            continue
        ax.plot(plot_df["ds"], plot_df["predicted"],
                color=_color_for(label), linewidth=1.4, marker="s", markersize=2.5,
                alpha=0.9, label=label)

    ax.set_title("Consensus Anchor: Forecast Comparison (SA Revised MoM)",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NFP MoM Change (thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Metrics bar chart — split into two panels because MAE/RMSE (~100s)
    # and accuracies (~0.5) live on incompatible scales.
    error_metrics = [("MAE", "MAE"), ("RMSE", "RMSE")]
    accuracy_metrics = [
        ("Acceleration_Accuracy", "AccelAcc"),
        ("Directional_Accuracy", "DirAcc"),
    ]
    bar_df = metrics_df[metrics_df["Forecast"].isin(forecast_dfs.keys())].copy()
    if not bar_df.empty:
        ordered = [f for f in forecast_dfs.keys() if f in set(bar_df["Forecast"])]
        bar_df = bar_df.set_index("Forecast").loc[ordered]

        n_models = len(bar_df)
        width = 0.8 / max(n_models, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, panel in zip(axes, [error_metrics, accuracy_metrics]):
            cols, names = zip(*panel)
            x = np.arange(len(names))
            for i, (label, row) in enumerate(bar_df.iterrows()):
                vals = [float(row[c]) for c in cols]
                offset = (i - (n_models - 1) / 2) * width
                bars = ax.bar(
                    x + offset, vals, width,
                    label=label, color=_color_for(label), alpha=0.88,
                )
                for bar in bars:
                    h = bar.get_height()
                    fmt = f"{h:.1f}" if abs(h) > 2 else f"{h:.3f}"
                    ax.annotate(
                        fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.grid(True, axis="y", alpha=0.3)

        axes[0].set_title("Error metrics (lower is better)", fontweight="bold")
        axes[0].set_ylabel("Thousands of jobs")
        axes[1].set_title("Accuracy metrics (higher is better)", fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

        fig.suptitle("Consensus Anchor: Backtest Metrics by Forecast", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_metrics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3) HTML scorecard
    table_html = (
        metrics_df.round(3)
        .to_html(index=False, classes="metrics", border=0)
    )
    experiment_html = ""
    if PANEL_REPLACES_CONSENSUS_FORECAST in forecast_dfs:
        experiment_html = (
            "  <div><h3>Panel-Replaces-Consensus Kalman</h3>"
            f"<img src=\"{PANEL_REPLACES_CONSENSUS_DIR}/backtest_predictions.png\"/></div>\n"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Consensus Anchor Scorecard</title>
<style>
 body {{ font-family: -apple-system, system-ui, Helvetica, Arial, sans-serif;
        margin: 24px; color: #111; }}
 h1 {{ font-size: 22px; margin-bottom: 8px; }}
 h2 {{ font-size: 16px; margin-top: 28px; }}
 table.metrics {{ border-collapse: collapse; font-size: 13px; }}
 table.metrics th, table.metrics td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; }}
 table.metrics th:first-child, table.metrics td:first-child {{ text-align: left; }}
 table.metrics tr:hover td {{ background: #f6f8fa; }}
 img {{ max-width: 100%; border: 1px solid #eee; margin-top: 8px; }}
 .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
</style></head>
<body>
 <h1>Consensus Anchor — Forecast Scorecard</h1>
 <p>Sorted by MAE. Backtest period: {len(actual_series) if actual_series is not None else "?"} months.</p>
 {table_html}
 <h2>Forecast overlay</h2>
 <img src="comparison_overlay.png" alt="overlay" />
 <h2>Metrics comparison</h2>
 <img src="comparison_metrics.png" alt="metrics" />
 <h2>Per-forecast diagnostics</h2>
 <div class="grid">
  <div><h3>Baseline Consensus</h3><img src="baseline_consensus/backtest_predictions.png"/></div>
  <div><h3>Panel Consensus Mean</h3><img src="panel_consensus_mean/backtest_predictions.png"/></div>
  <div><h3>Panel/Kalman Router</h3><img src="panel_kalman_router/backtest_predictions.png"/></div>
  <div><h3>Kalman Fusion (NSA)</h3><img src="kalman_fusion/backtest_predictions.png"/></div>
{experiment_html.rstrip()}
 </div>
</body></html>
"""
    (out_dir / "comparison_scorecard.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# predictions.csv augmentation
# ---------------------------------------------------------------------------

# Map predictions.csv `model` column → relative path of the model's
# summary_statistics.csv under output_base. Used to attach a backtest RMSE
# to each row so the file can be sorted best-to-worst.
_MODEL_RMSE_PATHS: Dict[str, str] = {
    "NSA":                                       "NSA_prediction/summary_statistics.csv",
    "SA":                                        "SA_prediction/summary_statistics.csv",
    "NSA_plus_adjustment":                       "NSA_plus_adjustment/summary_statistics.csv",
    "Consensus":                                 "consensus_anchor/baseline_consensus/summary_statistics.csv",
    "consensus_anchor_panel_mean":               "consensus_anchor/panel_consensus_mean/summary_statistics.csv",
    "consensus_anchor_panel_kalman_router":      "consensus_anchor/panel_kalman_router/summary_statistics.csv",
    "consensus_anchor_kalman_fusion":            "consensus_anchor/kalman_fusion/summary_statistics.csv",
    PANEL_REPLACES_CONSENSUS_MODEL:              f"consensus_anchor/{PANEL_REPLACES_CONSENSUS_DIR}/summary_statistics.csv",
}


def _load_model_rmses(output_base: Path) -> Dict[str, float]:
    """Read backtest RMSE from each model's summary_statistics.csv.

    Missing files are skipped (the row will get NaN RMSE and sort last).
    """
    out: Dict[str, float] = {}
    for model, rel in _MODEL_RMSE_PATHS.items():
        path = output_base / rel
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "RMSE" in df.columns and len(df) > 0:
                out[model] = float(df["RMSE"].iloc[0])
        except Exception as exc:
            logger.warning("Could not read RMSE from %s: %s", path, exc)
    return out


def _panel_router_candidate(
    data: pd.DataFrame,
    kind: str,
    param: Optional[float] = None,
) -> pd.Series:
    """Candidate final forecasts for the PIT panel/Kalman router."""
    panel = pd.to_numeric(data["panel_pred"], errors="coerce")
    kalman = pd.to_numeric(data["kalman_pred"], errors="coerce")
    if kind == "panel":
        return panel
    if kind == "panel_median":
        raw_panel_median = pd.to_numeric(data.get("panel_consensus_median"), errors="coerce")
        return raw_panel_median.combine_first(panel)
    if kind == "consensus":
        return pd.to_numeric(data["consensus_pred"], errors="coerce")
    if kind == "consensus_median":
        median = pd.to_numeric(data.get("consensus_median_pred"), errors="coerce")
        return median.combine_first(pd.to_numeric(data["consensus_pred"], errors="coerce"))
    if kind == "kalman":
        return kalman
    if kind == "panel_missing_else_kalman":
        raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")
        return raw_panel.combine_first(kalman)
    if kind == "blend":
        weight = 0.5 if param is None else float(param)
        return weight * panel + (1.0 - weight) * kalman
    if kind == "panel_consensus_blend":
        weight = 0.5 if param is None else float(param)
        consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
        return weight * panel + (1.0 - weight) * consensus
    if kind == "gate":
        threshold = 100.0 if param is None else float(param)
        raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")
        consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
        pred = panel.copy()
        use_kalman = raw_panel.isna() | ((kalman - consensus).abs() > threshold)
        pred[use_kalman] = kalman[use_kalman]
        return pred
    raise ValueError(f"Unknown panel router candidate: {kind}")


def _panel_router_static_candidates(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Static router candidates that do not depend on realized errors."""
    candidates: Dict[str, pd.Series] = {
        "panel": _panel_router_candidate(data, "panel"),
        "kalman": _panel_router_candidate(data, "kalman"),
        "consensus": _panel_router_candidate(data, "consensus"),
        "panel_missing_else_kalman": _panel_router_candidate(
            data, "panel_missing_else_kalman",
        ),
    }
    if "consensus_median_pred" in data.columns:
        candidates["consensus_median"] = _panel_router_candidate(data, "consensus_median")
    if "panel_consensus_median" in data.columns:
        candidates["panel_median"] = _panel_router_candidate(data, "panel_median")
    for weight in np.linspace(0.35, 0.95, 13):
        candidates[f"blend:{weight:.2f}"] = _panel_router_candidate(
            data, "blend", float(weight),
        )
    for weight in np.linspace(0.35, 0.95, 13):
        candidates[f"panel_consensus_blend:{weight:.2f}"] = _panel_router_candidate(
            data, "panel_consensus_blend", float(weight),
        )
    for threshold in [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]:
        candidates[f"gate:{threshold:.2f}"] = _panel_router_candidate(
            data, "gate", float(threshold),
        )
    return candidates


def _panel_router_trailing_edge_candidate(
    data: pd.DataFrame,
    *,
    lookback: Optional[int],
    margin: float,
    min_local_history: int = PANEL_ROUTER_MIN_LOCAL_HISTORY,
) -> pd.Series:
    """Replay a recent-performance gate without current-row leakage.

    For each row, use Kalman only if its trailing strict-PIT MAE beat the panel
    by more than ``margin``. Positive margins require a Kalman edge; negative
    margins allow Kalman when the panel's recent advantage is small. Missing
    live panel rows still fall back to Kalman.
    """
    panel = pd.to_numeric(data["panel_pred"], errors="coerce")
    kalman = pd.to_numeric(data["kalman_pred"], errors="coerce")
    consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
    raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")

    preds: List[float] = []
    for row_pos, (_, row) in enumerate(data.iterrows()):
        use_kalman = bool(pd.isna(raw_panel.iloc[row_pos]) or pd.isna(panel.iloc[row_pos]))
        if not use_kalman:
            hist = _actual_history_available_before(
                data.iloc[:row_pos],
                _row_prediction_cutoff(row),
                pd.Timestamp(row["ds"]),
            )
            hist = hist.dropna(subset=["actual", "panel_pred", "kalman_pred"])
            if lookback is not None:
                hist = hist.tail(int(lookback))
            if len(hist) >= int(min_local_history):
                panel_mae = float(np.mean(np.abs(hist["actual"] - hist["panel_pred"])))
                kalman_mae = float(np.mean(np.abs(hist["actual"] - hist["kalman_pred"])))
                use_kalman = bool((panel_mae - kalman_mae) > float(margin))

        pred = kalman.iloc[row_pos] if use_kalman else panel.iloc[row_pos]
        if pd.isna(pred):
            pred = kalman.iloc[row_pos] if pd.notna(kalman.iloc[row_pos]) else consensus.iloc[row_pos]
        preds.append(float(pred) if pd.notna(pred) else np.nan)
    return pd.Series(preds, index=data.index)


def _router_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    panel = pd.to_numeric(data["panel_pred"], errors="coerce")
    kalman = pd.to_numeric(data["kalman_pred"], errors="coerce")
    consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
    panel_count = (
        pd.to_numeric(data["panel_consensus_count"], errors="coerce")
        if "panel_consensus_count" in data.columns else pd.Series(np.nan, index=data.index)
    )
    panel_disp = (
        pd.to_numeric(data["panel_consensus_std"], errors="coerce")
        if "panel_consensus_std" in data.columns else pd.Series(np.nan, index=data.index)
    )
    out = pd.DataFrame({
        "abs_panel_kalman_gap": (panel - kalman).abs(),
        "abs_panel_consensus_gap": (panel - consensus).abs(),
        "abs_kalman_consensus_gap": (kalman - consensus).abs(),
        "panel_count": panel_count,
        "panel_dispersion": panel_disp,
        "panel_missing": panel.isna().astype(float),
    }, index=data.index)
    return out.replace([np.inf, -np.inf], np.nan)


def _panel_router_learned_edge_candidate(
    data: pd.DataFrame,
    *,
    min_history: int = 36,
    threshold: float = 0.58,
) -> pd.Series:
    """Expanding-window learned router candidate.

    The target is whether Kalman beat the panel on prior months. Each target
    row trains only on rows whose actuals were operationally available before
    that target's release cutoff.
    """
    panel = pd.to_numeric(data["panel_pred"], errors="coerce")
    kalman = pd.to_numeric(data["kalman_pred"], errors="coerce")
    consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
    raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")
    features = _router_feature_frame(data)
    preds: List[float] = []

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        logger.warning("sklearn unavailable; learned panel router candidate falls back to panel")
        return panel.combine_first(kalman).combine_first(consensus)

    for row_pos, (_, row) in enumerate(data.iterrows()):
        use_kalman = bool(pd.isna(raw_panel.iloc[row_pos]) or pd.isna(panel.iloc[row_pos]))
        if not use_kalman:
            hist = _actual_history_available_before(
                data.iloc[:row_pos],
                _row_prediction_cutoff(row),
                pd.Timestamp(row["ds"]),
            )
            hist = hist.dropna(subset=["actual", "panel_pred", "kalman_pred"])
            if len(hist) >= int(min_history):
                hist_idx = hist.index
                y = (
                    (hist["actual"] - hist["kalman_pred"]).abs()
                    < (hist["actual"] - hist["panel_pred"]).abs()
                ).astype(int)
                if y.nunique() >= 2:
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=200, random_state=42),
                    )
                    model.fit(features.loc[hist_idx], y)
                    p_kalman = float(model.predict_proba(features.iloc[[row_pos]])[0, 1])
                    use_kalman = p_kalman >= float(threshold)

        pred = kalman.iloc[row_pos] if use_kalman else panel.iloc[row_pos]
        if pd.isna(pred):
            pred = kalman.iloc[row_pos] if pd.notna(kalman.iloc[row_pos]) else consensus.iloc[row_pos]
        preds.append(float(pred) if pd.notna(pred) else np.nan)
    return pd.Series(preds, index=data.index)


def _panel_router_candidate_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """All router candidates, including PIT-replayed recent-edge gates."""
    candidates = _panel_router_static_candidates(data)
    for lookback in PANEL_ROUTER_TRAILING_EDGE_WINDOWS:
        for margin in PANEL_ROUTER_TRAILING_EDGE_MARGINS:
            name = f"trailing_edge:w{lookback}:margin{margin:+.0f}"
            candidates[name] = _panel_router_trailing_edge_candidate(
                data,
                lookback=int(lookback),
                margin=float(margin),
            )
    if _learned_panel_router_enabled():
        candidates["learned_consensus_edge"] = _panel_router_learned_edge_candidate(data)
    return pd.DataFrame(candidates, index=data.index)


def _panel_router_source(row: pd.Series) -> str:
    pred = row.get("predicted")
    if pd.isna(pred):
        return "missing"
    for source, col in (
        ("kalman", "kalman_pred"),
        ("panel", "panel_pred"),
        ("panel_median", "panel_consensus_median"),
        ("consensus", "consensus_pred"),
        ("consensus_median", "consensus_median_pred"),
    ):
        val = row.get(col)
        if pd.notna(val) and abs(float(pred) - float(val)) < 1e-8:
            return source
    return "blend"


def _panel_router_score(
    actual: np.ndarray,
    pred: np.ndarray,
    objective: str,
    *,
    ds: Optional[np.ndarray] = None,
    consensus: Optional[np.ndarray] = None,
) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if consensus is not None:
        consensus = np.asarray(consensus, dtype=float)
        mask = mask & np.isfinite(consensus)
    if mask.sum() == 0:
        return float("inf")
    a = actual[mask]
    p = pred[mask]
    c = np.asarray(consensus, dtype=float)[mask] if consensus is not None else None
    ds_masked = pd.to_datetime(pd.Series(ds[mask])) if ds is not None else None
    if objective == "rmse":
        return float(np.sqrt(np.mean((a - p) ** 2)))
    if objective == "mae":
        return float(np.mean(np.abs(a - p)))
    if objective in {"mae_hit", "rmse_hit", "composite_non_covid"}:
        model_abs = np.abs(a - p)
        base = (
            float(np.sqrt(np.mean((a - p) ** 2)))
            if objective == "rmse_hit"
            else float(np.mean(model_abs))
        )
        if c is None:
            return base
        ref_abs = np.abs(a - c)
        if objective == "composite_non_covid" and ds_masked is not None:
            keep = ~is_covid_month(ds_masked).to_numpy()
            if keep.any():
                a_eval = a[keep]
                p_eval = p[keep]
                model_abs = model_abs[keep]
                ref_abs = ref_abs[keep]
            else:
                a_eval = a
                p_eval = p
        else:
            a_eval = a
            p_eval = p
        wins = model_abs < ref_abs
        losses = model_abs > ref_abs
        hit_rate = float(np.mean(wins)) if wins.size else 0.0
        loss_rate = float(np.mean(losses)) if losses.size else 0.0
        mean_edge = float(np.mean(ref_abs - model_abs)) if model_abs.size else 0.0
        if objective == "mae_hit":
            return float(base + 25.0 * loss_rate - 10.0 * hit_rate - 0.15 * mean_edge)
        if objective == "rmse_hit":
            return float(base + 40.0 * loss_rate - 15.0 * hit_rate - 0.10 * mean_edge)
        accel_acc = float(compute_variance_kpis(a_eval, p_eval)["diff_sign_accuracy"])
        rmse = float(np.sqrt(np.mean((a_eval - p_eval) ** 2))) if a_eval.size else base
        mae = float(np.mean(np.abs(a_eval - p_eval))) if a_eval.size else base
        return float(0.55 * mae + 0.35 * rmse + 30.0 * loss_rate - 20.0 * hit_rate - 15.0 * accel_acc)
    if objective == "composite":
        from Train.config import KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR
        dir_acc = float(np.mean(np.sign(a) == np.sign(p)))
        accel_acc = float(compute_variance_kpis(a, p)["diff_sign_accuracy"])
        mae = float(np.mean(np.abs(a - p)))
        return mae - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc
    raise ValueError(f"Unknown panel router objective: {objective}")


def build_panel_kalman_router(
    panel_results: pd.DataFrame,
    kalman_df: pd.DataFrame,
    *,
    min_history: int = 24,
    objective: str = "mae",
    selection_lookback: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Walk-forward router between the PIT economist panel and Kalman.

    For every month, the routing rule is selected using only earlier months
    with actuals. This lets the final layer exploit the panel's lower level
    error while falling back to Kalman when the disagreement pattern has
    historically favored Kalman.
    """
    panel = panel_results.copy()
    panel = panel.rename(columns={"predicted": "panel_pred"})
    kalman_cols = ["ds", "predicted"]
    kalman_cols.extend([c for c in PIT_DATE_COLS if c in kalman_df.columns and c not in panel.columns])
    kalman_cols.extend([c for c in kalman_df.columns if c.startswith("hmm_")])
    kalman = kalman_df[kalman_cols].rename(columns={"predicted": "kalman_pred"})
    df = panel.merge(kalman, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    if selection_lookback is None:
        selection_lookback = _panel_router_selection_lookback()
    objective = (objective or _panel_router_objective()).strip().lower()
    if objective not in PANEL_ROUTER_SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported panel router objective={objective!r}")
    candidate_matrix = _panel_router_candidate_matrix(df)

    preds: List[float] = []
    rules: List[str] = []
    selection_scores: List[float] = []
    history_counts: List[int] = []
    scored_history_counts: List[int] = []
    latest_history_ds: List[pd.Timestamp] = []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        hist = _actual_history_available_before(
            hist,
            _row_prediction_cutoff(row),
            pd.Timestamp(row["ds"]),
        )
        history_counts.append(int(len(hist)))
        latest_history_ds.append(hist["ds"].max() if not hist.empty else pd.NaT)
        score_hist = hist.tail(int(selection_lookback)) if selection_lookback else hist
        scored_history_counts.append(int(len(score_hist)))
        if len(hist) < min_history or len(score_hist) < min_history:
            pred = float(row["panel_pred"]) if pd.notna(row.get("panel_pred")) else float(row["kalman_pred"])
            preds.append(pred)
            rules.append("warmup_panel")
            selection_scores.append(float("nan"))
            continue

        actual = score_hist["actual"].to_numpy(dtype=float)
        score_ds = pd.to_datetime(score_hist["ds"]).to_numpy()
        consensus = pd.to_numeric(score_hist["consensus_pred"], errors="coerce").to_numpy(dtype=float)
        best: Optional[Tuple[float, float, str]] = None
        for name in candidate_matrix.columns:
            cand = candidate_matrix.loc[score_hist.index, name].to_numpy(dtype=float)
            score = _panel_router_score(
                actual,
                cand,
                objective,
                ds=score_ds,
                consensus=consensus,
            )
            kalman_score_share = float(np.nanmean(
                np.isclose(
                    cand,
                    pd.to_numeric(score_hist["kalman_pred"], errors="coerce").to_numpy(dtype=float),
                    atol=1e-8,
                )
            ))
            if (
                best is None
                or score < best[0] - 1e-9
                or (abs(score - best[0]) <= 1e-9 and kalman_score_share > best[1])
            ):
                best = (score, kalman_score_share, name)

        assert best is not None
        pred_value = candidate_matrix.loc[i, best[2]]
        if pd.isna(pred_value):
            pred_value = (
                row["kalman_pred"]
                if pd.notna(row.get("kalman_pred"))
                else row.get("consensus_pred")
            )
        pred = float(pred_value)
        preds.append(pred)
        rules.append(best[2])
        selection_scores.append(float(best[0]))

    out = df.copy()
    out["predicted"] = preds
    out["selected_rule"] = rules
    out["selected_source"] = out.apply(_panel_router_source, axis=1)
    out["router_selection_score"] = selection_scores
    out["router_history_available_n"] = history_counts
    out["router_scored_history_n"] = scored_history_counts
    out["router_latest_available_actual_ds"] = latest_history_ds
    out["router_selection_lookback"] = int(selection_lookback) if selection_lookback else 0
    out["error"] = np.where(out["actual"].notna(), out["actual"] - out["predicted"], np.nan)
    out["panel_abs_error"] = np.where(
        out["actual"].notna(),
        np.abs(out["actual"] - out["panel_pred"]),
        np.nan,
    )
    out["kalman_abs_error"] = np.where(
        out["actual"].notna(),
        np.abs(out["actual"] - out["kalman_pred"]),
        np.nan,
    )
    out["router_abs_error"] = np.where(out["actual"].notna(), np.abs(out["error"]), np.nan)
    out["kalman_beats_panel"] = np.where(
        out["actual"].notna() & out["panel_abs_error"].notna() & out["kalman_abs_error"].notna(),
        out["kalman_abs_error"] < out["panel_abs_error"],
        np.nan,
    )
    metrics = full_metrics(
        out["actual"].values,
        out["predicted"].values,
        "Panel_Kalman_Router",
        ds=out["ds"],
    )
    metrics = _metrics_with_consensus_hits(metrics, out)
    metrics["Router_Kalman_Count"] = int((out["selected_source"] == "kalman").sum())
    metrics["Router_Live_Panel_Kalman_Count"] = int(
        ((out["selected_source"] == "kalman") & out["panel_consensus_mean"].notna()).sum()
    )
    metrics["Router_Selection_Lookback"] = int(selection_lookback) if selection_lookback else 0
    metrics["Router_Objective"] = objective
    out.attrs["candidate_count"] = int(candidate_matrix.shape[1])
    return out, metrics


def _quantile_ci_row(model_label: str, ds, pred: float, residuals: np.ndarray) -> Dict:
    """Build a predictions.csv row with quantile-based CIs from residuals."""
    if residuals.size > 2:
        return {
            "model": model_label,
            "ds": ds,
            "predicted": pred,
            "lower_50": pred + np.percentile(residuals, 25),
            "upper_50": pred + np.percentile(residuals, 75),
            "lower_80": pred + np.percentile(residuals, 10),
            "upper_80": pred + np.percentile(residuals, 90),
            "lower_95": pred + np.percentile(residuals, 2.5),
            "upper_95": pred + np.percentile(residuals, 97.5),
        }
    return {
        "model": model_label,
        "ds": ds,
        "predicted": pred,
        "lower_50": np.nan, "upper_50": np.nan,
        "lower_80": np.nan, "upper_80": np.nan,
        "lower_95": np.nan, "upper_95": np.nan,
    }


def _augment_predictions_csv(
    output_base: Path,
    cons_results: pd.DataFrame,
    kalman_df: pd.DataFrame,
    panel_results: Optional[pd.DataFrame] = None,
    panel_router_df: Optional[pd.DataFrame] = None,
    panel_replacement_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Append Consensus + final consensus-anchor rows to _output/Predictions/predictions.csv.

    For each OOS month (actual is NaN) in the consensus-anchor result frames,
    add rows for the benchmark consensus, the diagnostic panel mean, and the
    two main final models:
      - consensus_anchor_kalman_fusion
      - consensus_anchor_panel_kalman_router

    AccelOverride and Kalman+AccelPostFilter were dropped (2026-05-11) because
    both consistently underperformed the Consensus baseline.

    CIs are derived from each forecast's historical residuals (last 36).
    """
    pred_path = output_base / "Predictions" / "predictions.csv"
    if not pred_path.exists():
        logger.warning("predictions.csv not found at %s; skipping augmentation", pred_path)
        return

    base_df = pd.read_csv(pred_path, parse_dates=["ds"])
    # Drop any consensus-anchor / Consensus rows from prior runs to keep the
    # file idempotent. Also strip the deprecated AccelOverride and
    # Kalman+AccelPostFilter rows in case they linger from earlier runs.
    keep_models = {"NSA", "SA", "NSA_plus_adjustment"}
    base_df = base_df[base_df["model"].isin(keep_models)].copy()

    new_rows: List[Dict] = []

    def _residuals(df: pd.DataFrame) -> np.ndarray:
        if "error" not in df.columns:
            return np.array([])
        return df["error"].dropna().to_numpy()[-36:]

    variant_specs = [
        ("consensus_anchor_panel_mean", panel_results),
        ("consensus_anchor_panel_kalman_router", panel_router_df),
        ("consensus_anchor_kalman_fusion", kalman_df),
    ]
    if panel_replacement_df is not None and not panel_replacement_df.empty:
        variant_specs.append((PANEL_REPLACES_CONSENSUS_MODEL, panel_replacement_df))

    # Restrict to the next-to-release month only. predictions.csv is the
    # next-NFP forecast bundle, not a multi-month forward strip — the base
    # NSA/SA/NSA_plus_adjustment rows already contain only that month, and
    # consensus_anchor rows must match.
    target_ds: Optional[pd.Timestamp] = None
    if not base_df.empty:
        target_ds = pd.Timestamp(base_df["ds"].min())

    def _is_target(row_ds) -> bool:
        if target_ds is None:
            return True
        return pd.Timestamp(row_ds) == target_ds

    # Consensus row (the analyst consensus anchor). No CI — it's a single number.
    cons_oos = cons_results[cons_results["actual"].isna()].copy().sort_values("ds")
    if target_ds is None and not cons_oos.empty:
        target_ds = pd.Timestamp(cons_oos.iloc[0]["ds"])
    for _, row in cons_oos.iterrows():
        if not _is_target(row["ds"]):
            continue
        new_rows.append({
            "model": "Consensus",
            "ds": row["ds"],
            "predicted": float(row["predicted"]),
            "lower_50": np.nan, "upper_50": np.nan,
            "lower_80": np.nan, "upper_80": np.nan,
            "lower_95": np.nan, "upper_95": np.nan,
        })
        logger.info(
                "  Consensus %s -> %.0f (analyst consensus anchor)",
            pd.Timestamp(row["ds"]).strftime("%Y-%m"), float(row["predicted"]),
        )

    # consensus_anchor variants
    for label, df in variant_specs:
        if df is None or df.empty:
            continue
        res = _residuals(df)
        oos = df[df["actual"].isna()].copy().sort_values("ds")
        for _, row in oos.iterrows():
            if not _is_target(row["ds"]):
                continue
            new_rows.append(_quantile_ci_row(
                label, row["ds"], float(row["predicted"]), res,
            ))
            logger.info(
                "  %s %s -> %.0f",
                label, pd.Timestamp(row["ds"]).strftime("%Y-%m"), float(row["predicted"]),
            )

    if not new_rows:
        logger.info("No OOS consensus-anchor rows to add to predictions.csv")
        return

    augmented = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)

    # Attach backtest RMSE per model and sort best→worst so a reader of
    # predictions.csv immediately sees which forecasts to trust most.
    rmse_map = _load_model_rmses(output_base)
    augmented["rmse"] = augmented["model"].map(rmse_map)
    # Stable order: by RMSE ascending (NaN last), tie-break on model name.
    augmented = augmented.sort_values(
        ["rmse", "model"], na_position="last"
    ).reset_index(drop=True)

    # Reorder columns so `rmse` sits next to `predicted` for readability.
    cols = list(augmented.columns)
    if "rmse" in cols and "predicted" in cols:
        cols.remove("rmse")
        insert_at = cols.index("predicted") + 1
        cols = cols[:insert_at] + ["rmse"] + cols[insert_at:]
        augmented = augmented[cols]

    augmented.to_csv(pred_path, index=False)
    logger.info(
        "Augmented %s with %d Consensus / consensus_anchor rows",
        pred_path, len(new_rows),
    )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_consensus_anchor_pipeline(
    output_base: Optional[Path] = None,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    require_panel_router: bool = True,
) -> Dict[str, Dict]:
    """
    Run the full consensus-anchor post-training pipeline.

    Args:
        output_base: Base output directory (default: settings.OUTPUT_DIR).
        tune: Enable Optuna hyperparameter tuning.
        n_trials: Number of Optuna trials per approach.
        timeout: Optuna timeout in seconds per approach.
        require_panel_router: If True, fail the post-train stage when the
            Panel/Kalman Router cannot be built. This is the train-all default
            because the router is now a first-class final forecast.

    Returns:
        Dict mapping approach names to their metrics dicts.
    """
    if output_base is None:
        output_base = OUTPUT_DIR

    out_dir = output_base / "consensus_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CONSENSUS ANCHOR INTEGRATION")
    logger.info("=" * 60)

    # 1) Build merged dataset
    logger.info("Building merged consensus+model dataset...")
    merged = build_merged_dataset(output_base=output_base)
    panel_replacement_df: Optional[pd.DataFrame] = None
    panel_replacement_manifest: Optional[Dict[str, object]] = None
    panel_replacement_config: Optional[Dict[str, object]] = None
    panel_replacement_enabled = panel_replaces_consensus_enabled()
    if panel_replacement_enabled:
        panel_replacement_config = _panel_replacement_config()
        logger.info(
            "Gated Panel-Replaces-Consensus Kalman experiment enabled via %s "
            "(window=%s top_n=%s coverage=%.2f pooling=%s tw=%s nsa_ws=%.2f)",
            PANEL_REPLACES_CONSENSUS_ENV,
            panel_replacement_config["track_window"],
            panel_replacement_config["top_n"],
            float(panel_replacement_config["min_coverage_pct"]),
            panel_replacement_config["pooling"],
            panel_replacement_config["trailing_window"],
            float(panel_replacement_config["nsa_weight_scale"]),
        )
        merged, panel_replacement_df, panel_replacement_manifest = _attach_rolling_panel_replacement(
            merged,
            config=panel_replacement_config,
        )
    consensus_df, overlap_df, overlap_with_oos = split_datasets(merged)

    n_oos = int(overlap_with_oos["actual"].isna().sum())
    logger.info("Consensus history: %d months (%s to %s)",
                len(consensus_df),
                consensus_df["ds"].min().strftime("%Y-%m"),
                consensus_df["ds"].max().strftime("%Y-%m"))
    logger.info("Overlap (consensus+model+actual): %d months (%s to %s)",
                len(overlap_df),
                overlap_df["ds"].min().strftime("%Y-%m"),
                overlap_df["ds"].max().strftime("%Y-%m"))
    if n_oos > 0:
        oos_dates = overlap_with_oos[overlap_with_oos["actual"].isna()]["ds"]
        logger.info("OOS future months: %d (%s)",
                    n_oos, ", ".join(d.strftime("%Y-%m") for d in oos_dates))

    # Save merged dataset
    merged.to_csv(out_dir / "merged_consensus_model.csv", index=False)

    all_metrics = []

    # 2) Baselines
    logger.info("Computing baselines...")

    cons_base = full_metrics(
        overlap_df["actual"].values,
        overlap_df["consensus_pred"].values,
        "Baseline_Consensus",
        ds=overlap_df["ds"],
    )
    logger.info("  Consensus: MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                cons_base["MAE"], cons_base["RMSE"], cons_base["Acceleration_Accuracy"])

    # Write a full diagnostics bundle for the consensus baseline so it has the
    # same plot/CSV/ACF artifacts as the three model approaches.
    cons_keep_cols = ["ds", "actual", "consensus_pred"]
    if "consensus_median_pred" in overlap_with_oos.columns:
        cons_keep_cols.append("consensus_median_pred")
    cons_results = overlap_with_oos[cons_keep_cols].copy()
    cons_results = cons_results.dropna(subset=["consensus_pred"]).sort_values("ds").reset_index(drop=True)
    cons_results = cons_results.rename(columns={"consensus_pred": "predicted"})
    cons_results["consensus_pred"] = cons_results["predicted"]
    cons_base = _metrics_with_consensus_hits(cons_base, cons_results)
    all_metrics.append(cons_base)
    cons_results["error"] = np.where(
        cons_results["actual"].notna(),
        cons_results["actual"] - cons_results["predicted"],
        np.nan,
    )
    write_sandbox_output_bundle(
        results_df=cons_results,
        out_dir=out_dir / "baseline_consensus",
        model_id="baseline_consensus",
        diagnostics_label="Baseline Consensus (Bloomberg/Reuters mean)",
    )

    panel_results: Optional[pd.DataFrame] = None
    panel_metrics: Optional[Dict] = None
    if "panel_consensus_mean" in overlap_with_oos.columns:
        panel_keep_cols = [
            "ds", "actual", "consensus_pred",
            "panel_consensus_mean", "panel_consensus_median",
            "panel_consensus_count", "panel_consensus_std",
            "target_release_date", "actual_available_date",
        ]
        if "consensus_median_pred" in overlap_with_oos.columns:
            panel_keep_cols.append("consensus_median_pred")
        panel_keep_cols = [c for c in panel_keep_cols if c in overlap_with_oos.columns]
        panel_results = overlap_with_oos[panel_keep_cols].copy()
        panel_results["predicted"] = panel_results["panel_consensus_mean"].combine_first(
            panel_results["consensus_pred"]
        )
        panel_results["error"] = np.where(
            panel_results["actual"].notna(),
            panel_results["actual"] - panel_results["predicted"],
            np.nan,
        )
        panel_results = (
            panel_results
            .dropna(subset=["predicted"])
            .sort_values("ds")
            .reset_index(drop=True)
        )
        panel_eval = panel_results[panel_results["actual"].notna()].copy()
        if not panel_eval.empty:
            panel_metrics = full_metrics(
                panel_eval["actual"].values,
                panel_eval["predicted"].values,
                "Panel_Consensus_Mean",
                ds=panel_eval["ds"],
            )
            panel_metrics = _metrics_with_consensus_hits(panel_metrics, panel_eval)
            all_metrics.append(panel_metrics)
            logger.info(
                "  Panel mean: MAE=%.1f RMSE=%.1f AccelAcc=%.3f "
                "(fallback to consensus when panel missing)",
                panel_metrics["MAE"], panel_metrics["RMSE"],
                panel_metrics["Acceleration_Accuracy"],
            )
            write_sandbox_output_bundle(
                results_df=panel_results,
                out_dir=out_dir / "panel_consensus_mean",
                model_id="panel_consensus_mean",
                diagnostics_label="PIT Economist Panel Mean",
            )
    elif require_panel_router:
        raise RuntimeError(
            "Panel/Kalman Router is required, but no PIT economist panel columns "
            "were found in merged_consensus_model. Rebuild master snapshots with "
            "the Economist Panel source before running train-all."
        )

    champ_cols = ["ds", "actual", "champion_pred", "consensus_pred"]
    if "consensus_median_pred" in overlap_df.columns:
        champ_cols.append("consensus_median_pred")
    champ_ov = overlap_df[champ_cols].dropna(subset=["actual", "champion_pred"])
    champ_base = full_metrics(
        champ_ov["actual"].values,
        champ_ov["champion_pred"].values,
        "Baseline_Champion",
        ds=champ_ov["ds"],
    )
    champ_base = _metrics_with_consensus_hits(
        champ_base,
        champ_ov.rename(columns={"champion_pred": "predicted"}),
    )
    all_metrics.append(champ_base)
    logger.info("  Champion:  MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                champ_base["MAE"], champ_base["RMSE"], champ_base["Acceleration_Accuracy"])

    # 3) Kalman Fusion (jointly tune Kalman params + adjustment half-life)
    logger.info("Running Kalman Fusion...")
    kalman_df: Optional[pd.DataFrame] = None
    kalman_metrics: Optional[Dict] = None
    kalman_manifest: Dict[str, object] = {}
    _nsa_raw_by_ds: Optional[Dict[pd.Timestamp, float]] = None
    _adj_history: Optional[pd.DataFrame] = None
    nsa_raw_path: Optional[Path] = None
    if tune:
        tune_mode = _kalman_tune_mode()
        _nsa_raw_by_ds, _adj_history, nsa_raw_path = _load_nsa_adjustment_tuning_inputs(output_base)

        if tune_mode == "adaptive_grid":
            logger.info(
                "PIT-adaptive Kalman grid tuning enabled "
                "(set NFP_KALMAN_TUNE_MODE=optuna_cv for the legacy CV tuner)"
            )
            kalman_df, kalman_metrics, kalman_manifest = pit_adaptive_kalman_fusion(
                overlap_with_oos,
                consensus_df,
                adj_history=_adj_history,
                nsa_raw_by_ds=_nsa_raw_by_ds,
                min_history=24,
                selection_lookback=None,
                objective="rmse",
            )
            kalman_params = dict(kalman_manifest.get("live_params", {}))
        else:
            kalman_params = None
    else:
        kalman_params = {"trailing_window": 18}

    if tune and kalman_df is None:
        # ── Joint-tune short-circuit ────────────────────────────────────────
        # If train_lightgbm_nfp ran with JOINT_OPTUNA=True it has already
        # chosen (half_life_years, trailing_window, nsa_weight_scale) via a
        # single joint study. Reuse those params here and skip the post-hoc
        # _tune_kalman Optuna call entirely. The downstream HL-regen logic
        # below still runs.
        _joint_params_path = output_base / "consensus_anchor" / "kalman_fusion" / "joint_tuned_params.json"
        if _joint_params_path.exists():
            try:
                with open(_joint_params_path, "r") as _fp:
                    _jp = json.load(_fp)
                kalman_params = {
                    "trailing_window": int(_jp["trailing_window"]),
                    "nsa_weight_scale": float(_jp["nsa_weight_scale"]),
                    "half_life_years": float(_jp["half_life_years"]),
                }
                logger.info(
                    "[JointTune] reusing joint_tuned_params.json: "
                    "HL=%.2fy tw=%d ws=%.2f (step_date=%s, best_score=%.2f)",
                    kalman_params["half_life_years"],
                    kalman_params["trailing_window"],
                    kalman_params["nsa_weight_scale"],
                    _jp.get("step_date", "?"),
                    float(_jp.get("best_score", float("nan"))),
                )
            except Exception as e:
                logger.warning("[JointTune] could not read %s (%s); falling back to _tune_kalman",
                               _joint_params_path.name, e)
                kalman_params = None

        if kalman_params is None:
            kalman_params = _tune_kalman(
                overlap_df, consensus_df,
                n_trials=n_trials, timeout=timeout,
                adj_history=_adj_history,
                nsa_raw_by_ds=_nsa_raw_by_ds,
                tune_adjustment=(_adj_history is not None and _nsa_raw_by_ds is not None),
            )

        # ── Half-life drift warning (retired 2026-05-15) ──
        # Used to compare ``tuned_params.json`` HL against a value written
        # by the FS-side fusion-aligned selection target
        # (``dynamic_fs_selection_hl.json``). That selection variant was
        # reverted, so the FS-side HL file is no longer produced and this
        # consumer is gone. The iterative-fusion-tune orchestrator now
        # compares the Kalman-tuned HL across consecutive passes instead.

        # If half_life_years was tuned, regenerate the NSA+adjustment CSV with
        # the tuned value so the final Kalman fusion (and downstream consumers
        # like the predictions CSV) see the optimal champion.
        tuned_hl = kalman_params.get("half_life_years")
        if tuned_hl is not None and _adj_history is not None and abs(tuned_hl - 3.0) > 1e-6:
            try:
                from Train.Output_code.generate_output import _generate_adjustment_folder
                logger.info("Regenerating NSA+adjustment with tuned half_life_years=%.3f", tuned_hl)
                _adj_folder = output_base / "NSA_plus_adjustment"
                _nsa_results = pd.read_csv(nsa_raw_path, parse_dates=["ds"])
                from Train.data_loader import load_target_data
                _sa_target = load_target_data(
                    target_type="sa", release_type="first", target_source="revised",
                )
                _sa_results = pd.DataFrame({
                    "ds": pd.to_datetime(_sa_target["ds"]),
                    "actual": _sa_target["y_mom"].astype(float),
                })
                _generate_adjustment_folder(
                    _nsa_results, _sa_results, _adj_folder,
                    half_life_years=tuned_hl,
                )
                # Rebuild merged dataset so the rest of the pipeline reads
                # the freshly-tuned champion.
                merged = build_merged_dataset(output_base=output_base)
                if panel_replacement_enabled:
                    merged, panel_replacement_df, panel_replacement_manifest = _attach_rolling_panel_replacement(
                        merged,
                        config=panel_replacement_config or _panel_replacement_config(),
                    )
                consensus_df, overlap_df, overlap_with_oos = split_datasets(merged)
            except Exception as e:
                logger.warning(
                    "Could not regenerate NSA+adjustment with tuned half_life "
                    "(%.3f): %s. Falling back to existing CSV (half_life=3.0).",
                    tuned_hl, e,
                )

    # Final run includes OOS future rows
    if kalman_df is None or kalman_metrics is None:
        has_nsa = "nsa_pred" in overlap_with_oos.columns and overlap_with_oos["nsa_pred"].notna().any()
        kalman_df, kalman_metrics = kalman_fusion(
            overlap_with_oos, consensus_df,
            trailing_window=kalman_params["trailing_window"],
            use_nsa_accel=has_nsa,
            nsa_weight_scale=kalman_params.get("nsa_weight_scale", 1.0),
        )
        kalman_manifest = {
            "mode": "fixed_params",
            "params": kalman_params,
        }
    all_metrics.append(kalman_metrics)
    logger.info("  Kalman Fusion: MAE=%.1f RMSE=%.1f AccelAcc=%.3f (window=%d)",
                kalman_metrics["MAE"], kalman_metrics["RMSE"],
                kalman_metrics["Acceleration_Accuracy"],
                int(kalman_params.get("trailing_window", 18)))

    # Log OOS predictions
    kalman_oos = kalman_df[kalman_df["actual"].isna()]
    if not kalman_oos.empty:
        for _, r in kalman_oos.iterrows():
            logger.info("  [OOS] Kalman Fusion %s -> predicted=%.1f",
                        r["ds"].strftime("%Y-%m"), r["predicted"])

    # Save Kalman output bundle
    kalman_dir = out_dir / "kalman_fusion"
    write_sandbox_output_bundle(
        results_df=kalman_df,
        out_dir=kalman_dir,
        model_id="kalman_fusion",
        diagnostics_label="Kalman Fusion (Consensus + Model)",
    )
    with open(kalman_dir / "tuned_params.json", "w") as f:
        json.dump(kalman_params, f, indent=2)
    with open(kalman_dir / "tuning_manifest.json", "w") as f:
        json.dump(kalman_manifest, f, indent=2, default=str)

    panel_router_df: Optional[pd.DataFrame] = None
    panel_router_metrics: Optional[Dict] = None
    if panel_results is not None and not panel_results.empty:
        try:
            panel_router_selection_lookback = _panel_router_selection_lookback()
            panel_router_objective = _panel_router_objective()
            panel_router_df, panel_router_metrics = build_panel_kalman_router(
                panel_results,
                kalman_df,
                min_history=24,
                objective=panel_router_objective,
                selection_lookback=panel_router_selection_lookback,
            )
            all_metrics.append(panel_router_metrics)
            logger.info(
                "  Panel/Kalman Router: MAE=%.1f RMSE=%.1f AccelAcc=%.3f "
                "(objective=%s, selection_lookback=%s, live_panel_kalman=%d)",
                panel_router_metrics["MAE"], panel_router_metrics["RMSE"],
                panel_router_metrics["Acceleration_Accuracy"],
                panel_router_objective,
                panel_router_selection_lookback or "all",
                int(panel_router_metrics.get("Router_Live_Panel_Kalman_Count", 0)),
            )
            write_sandbox_output_bundle(
                results_df=panel_router_df,
                out_dir=out_dir / "panel_kalman_router",
                model_id="panel_kalman_router",
                diagnostics_label="PIT Panel/Kalman Router",
            )
            rule_counts = panel_router_df["selected_rule"].value_counts().to_dict()
            source_counts = panel_router_df["selected_source"].value_counts(dropna=False).to_dict()
            with open(out_dir / "panel_kalman_router" / "router_manifest.json", "w") as f:
                json.dump({
                    "objective": panel_router_objective,
                    "min_history": 24,
                    "selection_lookback": panel_router_selection_lookback or "all",
                    "selection_lookback_env": PANEL_ROUTER_SELECTION_LOOKBACK_ENV,
                    "objective_env": PANEL_ROUTER_OBJECTIVE_ENV,
                    "learned_router_enabled": _learned_panel_router_enabled(),
                    "candidate_count": int(panel_router_df.attrs.get("candidate_count", 0)),
                    "rule_counts": rule_counts,
                    "source_counts": source_counts,
                    "live_panel_kalman_count": int(
                        ((panel_router_df["selected_source"] == "kalman")
                         & panel_router_df["panel_consensus_mean"].notna()).sum()
                    ),
                    "pit_validation": "routing rule selected from prior actual months only",
                }, f, indent=2)
        except Exception as exc:
            if require_panel_router:
                raise RuntimeError("Required Panel/Kalman router failed") from exc
            logger.warning("Panel/Kalman router failed: %s", exc)
    elif require_panel_router:
        raise RuntimeError(
            "Panel/Kalman Router is required, but the PIT economist panel "
            "diagnostic frame is empty."
        )

    panel_replace_kalman_df: Optional[pd.DataFrame] = None
    panel_replace_kalman_metrics: Optional[Dict] = None
    panel_replace_kalman_manifest: Optional[Dict[str, object]] = None
    if panel_replacement_enabled:
        try:
            panel_replace_kalman_df, panel_replace_kalman_metrics, panel_replace_kalman_manifest = (
                build_panel_replaces_consensus_kalman(
                    overlap_with_oos,
                    consensus_df,
                    config=panel_replacement_config or _panel_replacement_config(),
                )
            )
            all_metrics.append(panel_replace_kalman_metrics)
            logger.info(
                "  Panel-Replaces-Consensus Kalman: MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                panel_replace_kalman_metrics["MAE"],
                panel_replace_kalman_metrics["RMSE"],
                panel_replace_kalman_metrics["Acceleration_Accuracy"],
            )
            panel_replace_dir = out_dir / PANEL_REPLACES_CONSENSUS_DIR
            write_sandbox_output_bundle(
                results_df=panel_replace_kalman_df,
                out_dir=panel_replace_dir,
                model_id=PANEL_REPLACES_CONSENSUS_DIR,
                diagnostics_label="PIT Rolling Panel-Replaces-Consensus Kalman",
            )
            if panel_replacement_df is not None:
                panel_replacement_df.to_csv(
                    panel_replace_dir / "panel_replacement_pit_audit.csv",
                    index=False,
                )
            manifest_payload = {
                "rolling_panel": panel_replacement_manifest,
                "kalman_experiment": panel_replace_kalman_manifest,
            }
            with open(panel_replace_dir / "experiment_manifest.json", "w") as f:
                json.dump(manifest_payload, f, indent=2, default=str)
        except Exception as exc:
            raise RuntimeError("Panel-Replaces-Consensus Kalman experiment failed") from exc

    # AccelOverride and Kalman+AccelPostFilter were removed (2026-05-11) because
    # both consistently underperformed the analyst Consensus baseline on the
    # 60-month backtest window. Kalman_Fusion and Panel_Kalman_Router are the
    # two main final forecasts; raw Consensus and panel mean remain diagnostics.

    # 4) Comparison metrics CSV
    metrics_df = pd.DataFrame(all_metrics).sort_values("MAE").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "comparison_metrics.csv", index=False)

    def _metric_snapshot(metrics: Optional[Dict]) -> Optional[Dict]:
        if metrics is None:
            return None
        keep = [
            "Forecast", "N", "MAE", "RMSE", "Directional_Accuracy",
            "Acceleration_Accuracy", "Tail_MAE", "Extreme_Hit_Rate",
            "NonCovid_MAE", "NonCovid_RMSE",
            "HitRate_vs_ConsensusMean_NonCovid",
            "HitRate_vs_ConsensusMedian_NonCovid",
            "MeanAbsErrorDelta_vs_ConsensusMean_NonCovid",
            "Router_Objective",
        ]
        return {k: metrics.get(k) for k in keep if k in metrics}

    main_models_manifest = {
        "main_models": [
            {
                "model_id": "consensus_anchor_kalman_fusion",
                "forecast": kalman_metrics.get("Forecast"),
                "artifact_dir": "consensus_anchor/kalman_fusion",
                "role": "existing Kalman fusion final forecast",
                "metrics": _metric_snapshot(kalman_metrics),
            },
            {
                "model_id": "consensus_anchor_panel_kalman_router",
                "forecast": (
                    panel_router_metrics.get("Forecast")
                    if panel_router_metrics is not None else None
                ),
                "artifact_dir": "consensus_anchor/panel_kalman_router",
                "role": "PIT walk-forward router over economist panel and Kalman",
                "metrics": _metric_snapshot(panel_router_metrics),
            },
        ],
        "experimental_outputs": (
            [
                {
                    "model_id": PANEL_REPLACES_CONSENSUS_MODEL,
                    "forecast": (
                        panel_replace_kalman_metrics.get("Forecast")
                        if panel_replace_kalman_metrics is not None else None
                    ),
                    "artifact_dir": f"consensus_anchor/{PANEL_REPLACES_CONSENSUS_DIR}",
                    "role": (
                        "gated experiment: PIT rolling economist panel replaces "
                        "the consensus level observation, with consensus fallback "
                        "when panel is missing"
                    ),
                    "enabled_by": PANEL_REPLACES_CONSENSUS_ENV,
                    "metrics": _metric_snapshot(panel_replace_kalman_metrics),
                }
            ]
            if panel_replace_kalman_metrics is not None else []
        ),
        "diagnostic_outputs": [
            "consensus_anchor/baseline_consensus",
            "consensus_anchor/panel_consensus_mean",
            "NSA_plus_adjustment",
        ],
        "pit_validation": {
            "kalman": (
                "measurement noise and router sidecar effects are estimated "
                "inside the monthly loop from df.iloc[:i] historical actuals only"
            ),
            "panel_router": (
                "for each target month, selected_rule is chosen by scoring "
                "candidate rules only on earlier rows with actuals"
            ),
            "panel_data": (
                "economist panel values are read from each target month's PIT "
                "master snapshot row"
            ),
            "panel_replaces_consensus_experiment": (
                "when enabled, the rolling full-economist panel is selected "
                "per month from current forecasts released before that month "
                "NFP release and ranked using only prior actuals available "
                "before the same cutoff"
            ),
        },
    }
    with open(out_dir / "main_models.json", "w") as f:
        json.dump(main_models_manifest, f, indent=2, default=str)

    logger.info("\nComparison (sorted by MAE):")
    for _, row in metrics_df.iterrows():
        logger.info("  %-25s MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                     row["Forecast"], row["MAE"], row["RMSE"],
                     row["Acceleration_Accuracy"])

    # 5) Unified comparison visualization across the surviving forecasts
    forecast_dfs = {
        cons_base["Forecast"]: cons_results,
        kalman_metrics["Forecast"]: kalman_df,
    }
    if panel_results is not None and panel_metrics is not None:
        forecast_dfs[panel_metrics["Forecast"]] = panel_results
    if panel_router_df is not None and panel_router_metrics is not None:
        forecast_dfs[panel_router_metrics["Forecast"]] = panel_router_df
    if panel_replace_kalman_df is not None and panel_replace_kalman_metrics is not None:
        forecast_dfs[panel_replace_kalman_metrics["Forecast"]] = panel_replace_kalman_df
    try:
        write_comparison_visualization(out_dir, forecast_dfs, metrics_df)
        logger.info("Wrote unified comparison visualization (overlay + bar + HTML)")
    except Exception as exc:
        logger.warning("Comparison visualization failed: %s", exc)

    # 6) Augment _output/Predictions/predictions.csv with the consensus anchor
    # OOS rows + the analyst Consensus we are anchoring to. The base file is
    # written by generate_all_output (NSA, SA, NSA_plus_adjustment rows).
    try:
        _augment_predictions_csv(
            output_base=output_base,
            cons_results=cons_results,
            kalman_df=kalman_df,
            panel_results=panel_results,
            panel_router_df=panel_router_df,
            panel_replacement_df=panel_replace_kalman_df,
        )
    except Exception as exc:
        logger.warning("Augmenting predictions.csv failed: %s", exc)

    logger.info("Consensus anchor outputs saved to %s", out_dir)
    logger.info("=" * 60)

    return {
        "kalman_fusion": kalman_metrics,
        "baselines": {
            "consensus": cons_base,
            "champion": champ_base,
            "panel_consensus_mean": panel_metrics,
            "panel_kalman_router": panel_router_metrics,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run consensus-anchor integration pipeline."
    )
    parser.add_argument(
        "--no-tune", dest="tune", action="store_false", default=True,
        help="Disable Optuna hyperparameter tuning.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=N_OPTUNA_TRIALS,
        help=f"Optuna trials per approach (default: {N_OPTUNA_TRIALS}).",
    )
    parser.add_argument(
        "--timeout", type=int, default=OPTUNA_TIMEOUT,
        help=f"Optuna timeout per approach in seconds (default: {OPTUNA_TIMEOUT}).",
    )
    args = parser.parse_args()

    run_consensus_anchor_pipeline(
        tune=args.tune,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
