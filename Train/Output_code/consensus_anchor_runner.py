"""
Post-training consensus-anchor integration.

Runs after the main train-all pipeline to produce a consensus-anchored
forecast via Kalman Filter Fusion (consensus + model signals fused via a
state-space random walk with adaptive trailing-window noise estimation).

Optuna with nested expanding-window CV tunes `trailing_window` and
`nsa_weight_scale` to maintain strict PIT safety.

The merged consensus+model dataset is built on-the-fly from:
  - Master snapshots         (NFP_Consensus_Mean, PIT-correct per target month)
  - SA blend champion        (_output/sandbox/sa_blend_walkforward/backtest_results.csv)
  - SA revised challenger    (_output/SA_prediction/backtest_results.csv)
  - NSA + adjustment         (_output/NSA_plus_adjustment/backtest_results.csv)

Outputs:
  _output/consensus_anchor/
  ├── merged_consensus_model.csv
  ├── baseline_consensus/        (raw analyst median, for benchmarking)
  ├── kalman_fusion/
  │   ├── backtest_results.csv
  │   ├── summary_statistics.csv
  │   ├── backtest_predictions.png
  │   └── summary_table.png
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
ECONOMIST_PANEL_FORECAST_COLS = [
    "NFP_Forecast_AIB",
    "NFP_Forecast_CONTINUUM_ECON",
    "NFP_Forecast_DANSKE_BK",
    "NFP_Forecast_NATIONWIDE_INSUR",
    "NFP_Forecast_Top4Mean",
]

# Minimum expanding-window history before producing a prediction
MIN_HISTORY = 12


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


# ---------------------------------------------------------------------------
# Data Loading (reuses logic from build_consensus_anchor_merged_variants.py)
# ---------------------------------------------------------------------------

def _load_consensus_pit(
    target_type: str = "sa",
    target_source: str = "revised",
) -> pd.DataFrame:
    """Load NFP_Consensus_Mean per ds via the master-snapshot infrastructure.

    For each target month M, the master snapshot at M was built at ETL time with the
    strict filter ``release_date < M's NFP release date``, so any column value at row
    ``ds=M`` is PIT-correct by construction. We extract ``NFP_Consensus_Mean`` at
    that row.

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

        rows.append({"ds": obs_month, "consensus_pred": float(val)})

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
    out_indexed["consensus_pred"] = winsorize_covid_period(
        out_indexed[["consensus_pred"]]
    )["consensus_pred"]
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

    for snap_path in snapshot_files:
        try:
            obs_month = pd.to_datetime(snap_path.stem + "-01")
        except (ValueError, TypeError):
            continue

        columns = ["date", *ECONOMIST_PANEL_FORECAST_COLS]
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

        values = pd.to_numeric(
            match[ECONOMIST_PANEL_FORECAST_COLS].iloc[0],
            errors="coerce",
        )
        if not values.notna().any():
            continue

        row: Dict[str, object] = {
            "ds": obs_month,
            "panel_consensus_mean": float(values.mean()),
            "panel_consensus_median": float(values.median()),
            "panel_consensus_count": int(values.notna().sum()),
            "panel_consensus_std": (
                float(values.std()) if int(values.notna().sum()) > 1 else np.nan
            ),
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
    value_cols = [c for c in out.columns if c != "ds"]
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

    # NSA Raw predictions for AccelOverride direction voting
    nsa_raw_path = output_base / "NSA_prediction" / "backtest_results.csv"
    if not nsa_raw_path.exists():
        nsa_raw_path = output_base / "NSA_prediction_revised" / "backtest_results.csv"
    if nsa_raw_path.exists():
        nsa_raw_df = _load_model_backtest(nsa_raw_path, "nsa_raw_pred")
        merged = merged.merge(nsa_raw_df[["ds", "nsa_raw_pred"]], on="ds", how="outer")
        logger.info("Loaded NSA raw: %d months", merged["nsa_raw_pred"].notna().sum())
    else:
        merged["nsa_raw_pred"] = np.nan
        logger.warning("NSA raw not found; AccelOverride will use fewer signals")

    # Backfill actuals from target parquet for full consensus history
    if TARGET_PARQUET.exists():
        target = pd.read_parquet(TARGET_PARQUET, columns=["ds", "y_mom"])
        target["ds"] = pd.to_datetime(target["ds"])
        target = target.rename(columns={"y_mom": "actual_from_target"})
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
    if "nsa_pred" in overlap_df.columns:
        keep_cols.append("nsa_pred")
    panel_cols = [c for c in ("panel_consensus_mean", "panel_consensus_count") if c in overlap_df.columns]
    keep_cols.extend(panel_cols)
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
    prior_cons = consensus_df[
        consensus_df["consensus_pred"].notna()
        & consensus_df["actual"].notna()
        & (consensus_df["ds"] < first_backtest_ds)
    ]
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

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        # Use only historical rows with known actuals for noise estimation
        hist = df.iloc[:i]
        hist_valid = hist[hist["actual"].notna()]

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
        Q_step = Q * (1.0 + 0.75 * router_risk)
        R_m_step = R_m
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

        if pd.notna(row["actual"]):
            x_hat = row["actual"]
            P = 1e-6
        else:
            x_hat = x_post
            P = P_post

        result_row = {
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "error": row["actual"] - pred if pd.notna(row["actual"]) else np.nan,
        }
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
        hist_idx = np.where(np.isfinite(actual[:i]))[0]
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
    Produce a unified comparison view across all 4 consensus-anchor forecasts:
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

    ax.set_title("Consensus Anchor: 4-Way Forecast Comparison (SA Revised MoM)",
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
 <h1>Consensus Anchor — 4-Way Forecast Scorecard</h1>
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
    if kind == "kalman":
        return kalman
    if kind == "panel_missing_else_kalman":
        raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")
        return raw_panel.combine_first(kalman)
    if kind == "blend":
        weight = 0.5 if param is None else float(param)
        return weight * panel + (1.0 - weight) * kalman
    if kind == "gate":
        threshold = 100.0 if param is None else float(param)
        raw_panel = pd.to_numeric(data["panel_consensus_mean"], errors="coerce")
        consensus = pd.to_numeric(data["consensus_pred"], errors="coerce")
        pred = panel.copy()
        use_kalman = raw_panel.isna() | ((kalman - consensus).abs() > threshold)
        pred[use_kalman] = kalman[use_kalman]
        return pred
    raise ValueError(f"Unknown panel router candidate: {kind}")


def _panel_router_score(actual: np.ndarray, pred: np.ndarray, objective: str) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("inf")
    a = actual[mask]
    p = pred[mask]
    if objective == "rmse":
        return float(np.sqrt(np.mean((a - p) ** 2)))
    if objective == "mae":
        return float(np.mean(np.abs(a - p)))
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
) -> Tuple[pd.DataFrame, Dict]:
    """Walk-forward router between the PIT economist panel and Kalman.

    For every month, the routing rule is selected using only earlier months
    with actuals. This lets the final layer exploit the panel's lower level
    error while falling back to Kalman when the disagreement pattern has
    historically favored Kalman.
    """
    panel = panel_results.copy()
    panel = panel.rename(columns={"predicted": "panel_pred"})
    kalman = kalman_df[["ds", "predicted"]].rename(columns={"predicted": "kalman_pred"})
    df = panel.merge(kalman, on="ds", how="inner").sort_values("ds").reset_index(drop=True)

    candidates: List[Tuple[str, Optional[float]]] = [
        ("panel", None),
        ("kalman", None),
        ("panel_missing_else_kalman", None),
    ]
    candidates.extend(("blend", float(w)) for w in np.linspace(0.35, 0.95, 13))
    candidates.extend(("gate", float(t)) for t in [25, 50, 75, 100, 125, 150, 175, 200, 250, 300])

    preds: List[float] = []
    rules: List[str] = []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        hist = hist[hist["actual"].notna()].copy()
        if len(hist) < min_history:
            pred = float(row["panel_pred"]) if pd.notna(row.get("panel_pred")) else float(row["kalman_pred"])
            preds.append(pred)
            rules.append("warmup_panel")
            continue

        actual = hist["actual"].to_numpy(dtype=float)
        best: Optional[Tuple[float, str, Optional[float]]] = None
        for kind, param in candidates:
            cand = _panel_router_candidate(hist, kind, param).to_numpy(dtype=float)
            score = _panel_router_score(actual, cand, objective)
            if best is None or score < best[0]:
                best = (score, kind, param)

        assert best is not None
        pred = float(_panel_router_candidate(row.to_frame().T, best[1], best[2]).iloc[0])
        preds.append(pred)
        rules.append(best[1] if best[2] is None else f"{best[1]}:{best[2]:.2f}")

    out = df.copy()
    out["predicted"] = preds
    out["selected_rule"] = rules
    out["error"] = np.where(out["actual"].notna(), out["actual"] - out["predicted"], np.nan)
    metrics = full_metrics(
        out["actual"].values,
        out["predicted"].values,
        "Panel_Kalman_Router",
        ds=out["ds"],
    )
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
) -> None:
    """
    Append Consensus + Kalman_Fusion rows to _output/Predictions/predictions.csv.

    For each OOS month (actual is NaN) in the consensus-anchor result frames,
    add two rows:
      - Consensus  (the analyst median we are anchoring to)
      - consensus_anchor_kalman_fusion

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

    # Consensus row (the analyst median anchor). No CI — it's a single number.
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
            "  Consensus %s -> %.0f (analyst median anchor)",
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
) -> Dict[str, Dict]:
    """
    Run the full consensus-anchor post-training pipeline.

    Args:
        output_base: Base output directory (default: settings.OUTPUT_DIR).
        tune: Enable Optuna hyperparameter tuning.
        n_trials: Number of Optuna trials per approach.
        timeout: Optuna timeout in seconds per approach.

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
    all_metrics.append(cons_base)
    logger.info("  Consensus: MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                cons_base["MAE"], cons_base["RMSE"], cons_base["Acceleration_Accuracy"])

    # Write a full diagnostics bundle for the consensus baseline so it has the
    # same plot/CSV/ACF artifacts as the three model approaches.
    cons_results = overlap_with_oos[["ds", "actual", "consensus_pred"]].copy()
    cons_results = cons_results.dropna(subset=["consensus_pred"]).sort_values("ds").reset_index(drop=True)
    cons_results = cons_results.rename(columns={"consensus_pred": "predicted"})
    cons_results["consensus_pred"] = cons_results["predicted"]
    cons_results["error"] = np.where(
        cons_results["actual"].notna(),
        cons_results["actual"] - cons_results["predicted"],
        np.nan,
    )
    write_sandbox_output_bundle(
        results_df=cons_results,
        out_dir=out_dir / "baseline_consensus",
        model_id="baseline_consensus",
        diagnostics_label="Baseline Consensus (Bloomberg/Reuters median)",
    )

    panel_results: Optional[pd.DataFrame] = None
    panel_metrics: Optional[Dict] = None
    if "panel_consensus_mean" in overlap_with_oos.columns:
        panel_keep_cols = [
            "ds", "actual", "consensus_pred",
            "panel_consensus_mean", "panel_consensus_median",
            "panel_consensus_count", "panel_consensus_std",
        ]
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

    champ_ov = overlap_df[["ds", "actual", "champion_pred"]].dropna()
    champ_base = full_metrics(
        champ_ov["actual"].values,
        champ_ov["champion_pred"].values,
        "Baseline_Champion",
        ds=champ_ov["ds"],
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
            panel_router_df, panel_router_metrics = build_panel_kalman_router(
                panel_results,
                kalman_df,
                min_history=24,
                objective="mae",
            )
            all_metrics.append(panel_router_metrics)
            logger.info(
                "  Panel/Kalman Router: MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                panel_router_metrics["MAE"], panel_router_metrics["RMSE"],
                panel_router_metrics["Acceleration_Accuracy"],
            )
            write_sandbox_output_bundle(
                results_df=panel_router_df,
                out_dir=out_dir / "panel_kalman_router",
                model_id="panel_kalman_router",
                diagnostics_label="PIT Panel/Kalman Router",
            )
            rule_counts = panel_router_df["selected_rule"].value_counts().to_dict()
            with open(out_dir / "panel_kalman_router" / "router_manifest.json", "w") as f:
                json.dump({
                    "objective": "mae",
                    "min_history": 24,
                    "rule_counts": rule_counts,
                    "pit_validation": "routing rule selected from prior actual months only",
                }, f, indent=2)
        except Exception as exc:
            logger.warning("Panel/Kalman router failed: %s", exc)

    # AccelOverride and Kalman+AccelPostFilter were removed (2026-05-11) because
    # both consistently underperformed the analyst Consensus baseline on the
    # 60-month backtest window. The Kalman_Fusion variant is the sole anchored
    # forecast we now emit alongside the raw Consensus baseline.

    # 4) Comparison metrics CSV
    metrics_df = pd.DataFrame(all_metrics).sort_values("MAE").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "comparison_metrics.csv", index=False)

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
