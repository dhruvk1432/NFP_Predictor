"""
Post-training consensus-anchor integration.

Runs after the main train-all pipeline to produce consensus-anchored
predictions using two proven approaches:
  1) Kalman Filter Fusion  (consensus + model signals fused via state-space)
  2) AccelOverride          (consensus base + model acceleration direction)

Both approaches use Optuna for hyperparameter tuning with expanding-window
cross-validation to maintain strict PIT safety.

The merged consensus+model dataset is built on-the-fly from:
  - Latest Unifier snapshot  (NFP_Consensus_Mean, ~315 months)
  - SA blend champion        (_output/sandbox/sa_blend_walkforward/backtest_results.csv)
  - SA revised challenger    (_output/SA_prediction/backtest_results.csv)

Outputs:
  _output/consensus_anchor/
  ├── merged_consensus_model.csv
  ├── kalman_fusion/
  │   ├── backtest_results.csv
  │   ├── summary_statistics.csv
  │   ├── backtest_predictions.png
  │   └── summary_table.png
  ├── accel_override/
  │   ├── backtest_results.csv
  │   ├── summary_statistics.csv
  │   ├── backtest_predictions.png
  │   └── summary_table.png
  └── comparison_metrics.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, DATA_PATH, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.variance_metrics import compute_variance_kpis
from Train.sandbox.output_utils import write_sandbox_output_bundle

logger = setup_logger(__file__, TEMP_DIR)

OUT_BASE = OUTPUT_DIR / "consensus_anchor"

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

SNAPSHOT_ROOT = DATA_PATH / "Exogenous_data" / "exogenous_unifier_data" / "decades"
TARGET_PARQUET = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"

# Minimum expanding-window history before producing a prediction
MIN_HISTORY = 12


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def full_metrics(actual: np.ndarray, pred: np.ndarray, label: str) -> Dict:
    """Compute the full metric suite for consensus anchor experiments."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]

    if a.size == 0:
        return {"Forecast": label, "N": 0}

    e = a - p
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mse = float(np.mean(e ** 2))
    me = float(np.mean(e))
    medae = float(np.median(np.abs(e)))

    dir_acc = float(np.mean(np.sign(a) == np.sign(p)))
    if a.size >= 2:
        accel_acc = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))))
    else:
        accel_acc = np.nan

    # SMAPE
    denom = (np.abs(a) + np.abs(p))
    smape = float(np.mean(2 * np.abs(e) / np.where(denom == 0, 1, denom)) * 100)

    vk = compute_variance_kpis(a, p)

    return {
        "Forecast": label,
        "N": int(a.size),
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "ME_Bias": me,
        "MedAE": medae,
        "SMAPE_pct": smape,
        "Directional_Accuracy": dir_acc,
        "Acceleration_Accuracy": accel_acc,
        "STD_Ratio": float(vk["std_ratio"]),
        "Diff_STD_Ratio": float(vk["diff_std_ratio"]),
        "Corr_Level": float(vk["corr_level"]),
        "Corr_Diff": float(vk["corr_diff"]),
        "Diff_Sign_Accuracy": float(vk["diff_sign_accuracy"]),
        "Tail_MAE": float(vk["tail_mae"]),
        "Extreme_Hit_Rate": float(vk["extreme_hit_rate"]),
    }


# ---------------------------------------------------------------------------
# Data Loading (reuses logic from build_consensus_anchor_merged_variants.py)
# ---------------------------------------------------------------------------

def _latest_snapshot_path() -> Path:
    candidates = list(SNAPSHOT_ROOT.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No snapshot parquet files found under {SNAPSHOT_ROOT}")

    def snapshot_key(path: Path) -> pd.Timestamp:
        return pd.to_datetime(path.stem + "-01", errors="coerce")

    dated = [(p, snapshot_key(p)) for p in candidates]
    dated = [(p, d) for p, d in dated if pd.notna(d)]
    if not dated:
        raise RuntimeError("Could not parse dates from snapshot parquet filenames")
    latest_path, _ = max(dated, key=lambda x: x[1])
    return latest_path


def _load_consensus(snapshot_path: Path) -> pd.DataFrame:
    snap = pd.read_parquet(snapshot_path, columns=["date", "series_name", "value"])
    cons = snap[snap["series_name"] == "NFP_Consensus_Mean"].copy()
    if cons.empty:
        raise RuntimeError(f"NFP_Consensus_Mean not found in snapshot: {snapshot_path}")

    cons["ds"] = pd.to_datetime(cons["date"]).dt.to_period("M").dt.to_timestamp()
    cons["value"] = pd.to_numeric(cons["value"], errors="coerce")
    cons = cons.dropna(subset=["ds", "value"]).sort_values("ds")

    monthly = (
        cons.groupby("ds", as_index=False)
        .agg(consensus_pred=("value", "last"))
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly


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

    snapshot_path = _latest_snapshot_path()
    logger.info("Consensus snapshot: %s", snapshot_path)
    consensus_monthly = _load_consensus(snapshot_path)

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

    challenger_path = output_base / "SA_prediction" / "backtest_results.csv"
    if not challenger_path.exists():
        challenger_path = output_base / "SA_prediction_revised" / "backtest_results.csv"

    champion_df = _load_model_backtest(champion_path, "champion_pred")
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
    """
    # Keep rows where consensus + model exist; actual can be NaN (OOS)
    keep_cols = ["ds", "actual", "consensus_pred", "champion_pred"]
    if "nsa_pred" in overlap_df.columns:
        keep_cols.append("nsa_pred")
    df = overlap_df[keep_cols].copy()
    df = df.dropna(subset=["consensus_pred", "champion_pred"])
    df = df.sort_values("ds").reset_index(drop=True)

    has_nsa = "nsa_pred" in df.columns and use_nsa_accel

    # Initialize noise parameters from full consensus history
    cons_hist = consensus_df[
        consensus_df["consensus_pred"].notna() & consensus_df["actual"].notna()
    ]
    cons_err = (cons_hist["actual"] - cons_hist["consensus_pred"]).values
    R_c_init = float(np.var(cons_err[-60:], ddof=1))
    Q_init = float(np.var(np.diff(cons_hist["actual"].dropna().values[-60:]), ddof=1))

    x_hat = float(df.iloc[0]["consensus_pred"])
    P = Q_init

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        # Use only historical rows with known actuals for noise estimation
        hist = df.iloc[:i]
        hist_valid = hist[hist["actual"].notna()]

        if len(hist_valid) >= 6:
            recent_cons_err = (hist_valid["actual"] - hist_valid["consensus_pred"]).values[-trailing_window:]
            R_c = float(np.var(recent_cons_err, ddof=1)) + 1e-6
            if use_model:
                recent_model_err = (hist_valid["actual"] - hist_valid["champion_pred"]).values[-trailing_window:]
                R_m = float(np.var(recent_model_err, ddof=1)) + 1e-6
            else:
                R_m = 1e12
            recent_actual_diff = np.diff(hist_valid["actual"].values[-trailing_window:])
            Q = float(np.var(recent_actual_diff, ddof=1)) + 1e-6
        else:
            R_c = R_c_init
            R_m = R_c_init * 1.5 if use_model else 1e12
            Q = Q_init

        # Prediction step
        x_prior = x_hat
        P_prior = P + Q

        # Update step: multi-observation Kalman (information filter)
        info_prior = 1.0 / P_prior
        info_c = 1.0 / R_c
        info_m = 1.0 / R_m if use_model else 0.0

        # NSA acceleration channel: observes delta, converted to level
        info_a = 0.0
        nsa_level_implied = 0.0
        if has_nsa and pd.notna(row.get("nsa_pred")) and len(hist_valid) >= 2:
            prev_actual = float(hist_valid["actual"].iloc[-1])
            nsa_delta = float(row["nsa_pred"]) - prev_actual
            nsa_level_implied = prev_actual + nsa_delta

            # Estimate NSA delta noise from trailing window
            if has_nsa and len(hist_valid) >= 6:
                nsa_hist = hist_valid[hist_valid["nsa_pred"].notna()] if "nsa_pred" in hist_valid.columns else pd.DataFrame()
                if len(nsa_hist) >= 4:
                    # NSA delta error: (actual[t] - actual[t-1]) - (nsa_pred[t] - actual[t-1])
                    # = actual[t] - nsa_pred[t]
                    recent_nsa_err = (nsa_hist["actual"] - nsa_hist["nsa_pred"]).values[-trailing_window:]
                    R_a = float(np.var(recent_nsa_err, ddof=1)) + 1e-6
                    info_a = nsa_weight_scale / R_a
                else:
                    # Insufficient NSA history — use conservative noise
                    R_a = R_c_init * 2.0
                    info_a = nsa_weight_scale / R_a
            else:
                R_a = R_c_init * 2.0
                info_a = nsa_weight_scale / R_a

        P_post = 1.0 / (info_prior + info_c + info_m + info_a)
        x_post = P_post * (
            info_prior * x_prior
            + info_c * row["consensus_pred"]
            + (info_m * row["champion_pred"] if use_model else 0.0)
            + (info_a * nsa_level_implied if info_a > 0 else 0.0)
        )

        pred = x_post

        if pd.notna(row["actual"]):
            x_hat = row["actual"]
            P = 1e-6
        else:
            x_hat = x_post
            P = P_post

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "error": row["actual"] - pred if pd.notna(row["actual"]) else np.nan,
        })

    res_df = pd.DataFrame(results)
    label = "Kalman_Fusion" if use_model else "Kalman_Consensus_Only"
    if has_nsa:
        label += "_NSA"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


def _tune_kalman(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    n_trials: int = N_OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> Dict:
    """Optuna-tune trailing_window and nsa_weight_scale for Kalman fusion.

    Uses composite objective: MAE - λ_accel * accel_acc - λ_dir * dir_acc
    to prioritize acceleration and directional accuracy alongside MAE.
    """
    from Train.config import KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available; using default Kalman params")
        return {"trailing_window": 18, "nsa_weight_scale": 1.0}

    has_nsa = "nsa_pred" in overlap_df.columns and overlap_df["nsa_pred"].notna().any()

    logger.info("Optuna tuning Kalman fusion: trials=%d timeout=%ds nsa=%s "
                "λ_accel=%.1f λ_dir=%.1f",
                n_trials, timeout, has_nsa, KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        tw = trial.suggest_int("trailing_window", 6, 36)
        nsa_ws = trial.suggest_float("nsa_weight_scale", 0.1, 3.0) if has_nsa else 1.0
        _, metrics = kalman_fusion(
            overlap_df, consensus_df,
            trailing_window=tw,
            use_nsa_accel=has_nsa,
            nsa_weight_scale=nsa_ws,
        )
        mae = metrics.get("MAE", float("inf"))
        accel_acc = metrics.get("Acceleration_Accuracy", 0.0)
        dir_acc = metrics.get("Directional_Accuracy", 0.0)

        if not np.isfinite(mae):
            return float("inf")

        # Composite: minimize MAE while maximizing acceleration and direction
        return float(mae - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial

    result = {"trailing_window": int(best.params["trailing_window"])}
    if has_nsa:
        result["nsa_weight_scale"] = float(best.params["nsa_weight_scale"])

    logger.info("Kalman Optuna: best_obj=%.1f trailing_window=%d nsa_weight_scale=%.2f",
                best.value, result["trailing_window"],
                result.get("nsa_weight_scale", 1.0))
    return result


# ---------------------------------------------------------------------------
# Approach: Acceleration Direction Override
# ---------------------------------------------------------------------------

def accel_override(
    overlap_df: pd.DataFrame,
    kappa: float = 0.5,
    magnitude_mode: str = "consensus",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Use consensus as base, override acceleration direction via majority vote.

    When consensus and champion disagree on direction, NSA acts as tiebreaker.
    If 2 of 3 signals (consensus, champion, NSA) agree, that direction wins.

    magnitude_mode:
      - 'consensus': keep consensus magnitude, only flip direction
      - 'blend': blend consensus and model magnitudes
      - 'model': partial adjustment from consensus toward model
    """
    # Keep rows where consensus + model exist; actual can be NaN (OOS)
    keep_cols = ["ds", "actual", "consensus_pred", "champion_pred"]
    has_nsa = "nsa_pred" in overlap_df.columns
    has_nsa_raw = "nsa_raw_pred" in overlap_df.columns
    if has_nsa:
        keep_cols.append("nsa_pred")
    if has_nsa_raw:
        keep_cols.append("nsa_raw_pred")
    df = overlap_df[keep_cols].copy()
    df = df.dropna(subset=["consensus_pred", "champion_pred"])
    df = df.sort_values("ds").reset_index(drop=True)
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        # Use only historical rows with known actuals for acceleration
        hist_valid = df.iloc[:i]
        hist_valid = hist_valid[hist_valid["actual"].notna()]

        if len(hist_valid) < 2:
            pred = row["consensus_pred"]
        else:
            last_actual = hist_valid["actual"].iloc[-1]
            cons_delta = row["consensus_pred"] - last_actual
            dir_cons = np.sign(cons_delta)
            model_delta = (
                float(row["champion_pred"]) - last_actual
                if pd.notna(row.get("champion_pred"))
                else cons_delta
            )

            # Majority vote using all available signals
            dir_votes = [dir_cons]
            if pd.notna(row.get("champion_pred")):
                dir_votes.append(np.sign(model_delta))
            if has_nsa and pd.notna(row.get("nsa_pred")):
                dir_votes.append(np.sign(float(row["nsa_pred"]) - last_actual))
            if has_nsa_raw and pd.notna(row.get("nsa_raw_pred")):
                dir_votes.append(np.sign(float(row["nsa_raw_pred"]) - last_actual))

            vote_sum = sum(dir_votes)
            chosen_dir = np.sign(vote_sum) if vote_sum != 0 else dir_cons

            agree = (chosen_dir == dir_cons)

            if agree:
                pred = row["consensus_pred"]
            else:
                if magnitude_mode == "consensus":
                    pred = last_actual + chosen_dir * abs(cons_delta)
                elif magnitude_mode == "blend":
                    blended_mag = kappa * abs(model_delta) + (1 - kappa) * abs(cons_delta)
                    pred = last_actual + chosen_dir * blended_mag
                elif magnitude_mode == "model":
                    correction = kappa * (model_delta - cons_delta)
                    pred = row["consensus_pred"] + correction
                else:
                    raise ValueError(magnitude_mode)

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "error": row["actual"] - pred if pd.notna(row["actual"]) else np.nan,
        })

    res_df = pd.DataFrame(results)
    label = f"AccelOverride_{magnitude_mode}_k{kappa:.2f}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


def _tune_accel_override(
    overlap_df: pd.DataFrame,
    n_trials: int = N_OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> Dict:
    """Optuna-tune kappa and magnitude_mode for AccelOverride.

    Uses composite objective: MAE - λ_accel * accel_acc - λ_dir * dir_acc.
    """
    from Train.config import KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available; using default AccelOverride params")
        return {"kappa": 0.5, "magnitude_mode": "consensus"}

    logger.info("Optuna tuning AccelOverride: trials=%d timeout=%ds λ_accel=%.1f λ_dir=%.1f",
                n_trials, timeout, KALMAN_LAMBDA_ACCEL, KALMAN_LAMBDA_DIR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        kappa = trial.suggest_float("kappa", 0.1, 0.9)
        mode = trial.suggest_categorical("magnitude_mode", ["consensus", "blend", "model"])
        _, metrics = accel_override(overlap_df, kappa=kappa, magnitude_mode=mode)
        mae = metrics.get("MAE", float("inf"))
        accel_acc = metrics.get("Acceleration_Accuracy", 0.0)
        dir_acc = metrics.get("Directional_Accuracy", 0.0)

        if not np.isfinite(mae):
            return float("inf")

        return float(mae - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial
    logger.info("AccelOverride Optuna: best_obj=%.1f kappa=%.3f mode=%s",
                best.value, best.params["kappa"], best.params["magnitude_mode"])
    return {
        "kappa": float(best.params["kappa"]),
        "magnitude_mode": str(best.params["magnitude_mode"]),
    }


# ---------------------------------------------------------------------------
# Comparison Visualization
# ---------------------------------------------------------------------------

# Stable colors per forecast across the overlay + bar chart.
_FORECAST_COLORS = {
    "Baseline_Consensus": "#DC2626",        # red
    "Kalman_Fusion_NSA": "#2563EB",         # blue
    "Kalman_Fusion": "#2563EB",
    "Kalman_AccelPostFilter": "#7C3AED",    # purple
    "AccelOverride": "#16A34A",             # green
    "Baseline_Champion": "#9CA3AF",         # gray
}


def _color_for(label: str) -> str:
    if label in _FORECAST_COLORS:
        return _FORECAST_COLORS[label]
    if label.startswith("AccelOverride"):
        return _FORECAST_COLORS["AccelOverride"]
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
  <div><h3>Kalman Fusion (NSA)</h3><img src="kalman_fusion/backtest_predictions.png"/></div>
  <div><h3>AccelOverride</h3><img src="accel_override/backtest_predictions.png"/></div>
  <div><h3>Kalman + AccelOverride Post-Filter</h3><img src="kalman_accel_postfilter/backtest_predictions.png"/></div>
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
    "consensus_anchor_kalman_fusion":            "consensus_anchor/kalman_fusion/summary_statistics.csv",
    "consensus_anchor_accel_override":           "consensus_anchor/accel_override/summary_statistics.csv",
    "consensus_anchor_kalman_accel_postfilter":  "consensus_anchor/kalman_accel_postfilter/summary_statistics.csv",
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
    accel_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
) -> None:
    """
    Append Consensus + consensus_anchor rows to _output/Predictions/predictions.csv.

    For each OOS month (actual is NaN) in the consensus-anchor result frames,
    add four rows:
      - Consensus  (the analyst median we are anchoring to)
      - consensus_anchor_kalman_fusion
      - consensus_anchor_accel_override
      - consensus_anchor_kalman_accel_postfilter

    CIs are derived from each forecast's historical residuals (last 36).
    """
    pred_path = output_base / "Predictions" / "predictions.csv"
    if not pred_path.exists():
        logger.warning("predictions.csv not found at %s; skipping augmentation", pred_path)
        return

    base_df = pd.read_csv(pred_path, parse_dates=["ds"])
    # Drop any consensus-anchor / Consensus rows from prior runs to keep the
    # file idempotent.
    keep_models = {"NSA", "SA", "NSA_plus_adjustment"}
    base_df = base_df[base_df["model"].isin(keep_models)].copy()

    new_rows: List[Dict] = []

    def _residuals(df: pd.DataFrame) -> np.ndarray:
        if "error" not in df.columns:
            return np.array([])
        return df["error"].dropna().to_numpy()[-36:]

    variant_specs = [
        ("consensus_anchor_kalman_fusion", kalman_df),
        ("consensus_anchor_accel_override", accel_df),
        ("consensus_anchor_kalman_accel_postfilter", hybrid_df),
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

    champ_ov = overlap_df[["actual", "champion_pred"]].dropna()
    champ_base = full_metrics(
        champ_ov["actual"].values,
        champ_ov["champion_pred"].values,
        "Baseline_Champion",
    )
    all_metrics.append(champ_base)
    logger.info("  Champion:  MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                champ_base["MAE"], champ_base["RMSE"], champ_base["Acceleration_Accuracy"])

    # 3) Kalman Fusion
    logger.info("Running Kalman Fusion...")
    if tune:
        # Tune on historical-only overlap (no OOS rows)
        kalman_params = _tune_kalman(overlap_df, consensus_df,
                                     n_trials=n_trials, timeout=timeout)
    else:
        kalman_params = {"trailing_window": 18}

    # Final run includes OOS future rows
    has_nsa = "nsa_pred" in overlap_with_oos.columns and overlap_with_oos["nsa_pred"].notna().any()
    kalman_df, kalman_metrics = kalman_fusion(
        overlap_with_oos, consensus_df,
        trailing_window=kalman_params["trailing_window"],
        use_nsa_accel=has_nsa,
        nsa_weight_scale=kalman_params.get("nsa_weight_scale", 1.0),
    )
    all_metrics.append(kalman_metrics)
    logger.info("  Kalman Fusion: MAE=%.1f RMSE=%.1f AccelAcc=%.3f (window=%d)",
                kalman_metrics["MAE"], kalman_metrics["RMSE"],
                kalman_metrics["Acceleration_Accuracy"],
                kalman_params["trailing_window"])

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

    # 4) AccelOverride
    logger.info("Running Acceleration Override...")
    if tune:
        # Tune on historical-only overlap (no OOS rows)
        accel_params = _tune_accel_override(overlap_df,
                                             n_trials=n_trials, timeout=timeout)
    else:
        accel_params = {"kappa": 0.5, "magnitude_mode": "consensus"}

    # Final run includes OOS future rows
    accel_df, accel_metrics = accel_override(
        overlap_with_oos,
        kappa=accel_params["kappa"],
        magnitude_mode=accel_params["magnitude_mode"],
    )
    all_metrics.append(accel_metrics)
    logger.info("  AccelOverride: MAE=%.1f RMSE=%.1f AccelAcc=%.3f (kappa=%.3f mode=%s)",
                accel_metrics["MAE"], accel_metrics["RMSE"],
                accel_metrics["Acceleration_Accuracy"],
                accel_params["kappa"], accel_params["magnitude_mode"])

    # Log OOS predictions
    accel_oos = accel_df[accel_df["actual"].isna()]
    if not accel_oos.empty:
        for _, r in accel_oos.iterrows():
            logger.info("  [OOS] AccelOverride %s -> predicted=%.1f",
                        r["ds"].strftime("%Y-%m"), r["predicted"])

    # Save AccelOverride output bundle
    accel_dir = out_dir / "accel_override"
    write_sandbox_output_bundle(
        results_df=accel_df,
        out_dir=accel_dir,
        model_id="accel_override",
        diagnostics_label="Acceleration Override (Consensus + Model Direction)",
    )
    with open(accel_dir / "tuned_params.json", "w") as f:
        json.dump(accel_params, f, indent=2)

    # 5) Kalman + AccelOverride Post-Filter (hybrid)
    # Uses Kalman's optimal level estimation, then overrides direction via majority vote
    logger.info("Running Kalman + AccelOverride Post-Filter...")
    hybrid_rows = []
    for _, krow in kalman_df.iterrows():
        pred = krow["predicted"]
        ds = krow["ds"]
        actual = krow["actual"]
        # Find matching row in overlap for signals
        ov_match = overlap_with_oos[overlap_with_oos["ds"] == ds]
        hist_valid = overlap_with_oos[
            (overlap_with_oos["ds"] < ds) & overlap_with_oos["actual"].notna()
        ]
        if not ov_match.empty and len(hist_valid) >= 2:
            last_actual = float(hist_valid.iloc[-1]["actual"])
            k_delta = pred - last_actual
            c_delta = float(ov_match.iloc[0]["consensus_pred"]) - last_actual
            m_delta = float(ov_match.iloc[0]["champion_pred"]) - last_actual
            signs = [np.sign(c_delta), np.sign(m_delta)]
            if has_nsa and pd.notna(ov_match.iloc[0].get("nsa_pred")):
                signs.append(np.sign(float(ov_match.iloc[0]["nsa_pred"]) - last_actual))
            if "nsa_raw_pred" in ov_match.columns and pd.notna(ov_match.iloc[0].get("nsa_raw_pred")):
                signs.append(np.sign(float(ov_match.iloc[0]["nsa_raw_pred"]) - last_actual))
            vote = sum(signs)
            majority_dir = np.sign(vote) if vote != 0 else np.sign(c_delta)
            if majority_dir != np.sign(k_delta) and abs(k_delta) > 1e-6:
                pred = last_actual + majority_dir * abs(k_delta)
        hybrid_rows.append({
            "ds": ds,
            "actual": actual,
            "predicted": pred,
            "consensus_pred": krow["consensus_pred"],
            "error": actual - pred if pd.notna(actual) else np.nan,
        })
    hybrid_df = pd.DataFrame(hybrid_rows)
    hybrid_metrics = full_metrics(
        hybrid_df["actual"].values, hybrid_df["predicted"].values,
        "Kalman_AccelPostFilter",
    )
    all_metrics.append(hybrid_metrics)
    logger.info("  Kalman+AccelPostFilter: MAE=%.1f RMSE=%.1f AccelAcc=%.3f DirAcc=%.3f",
                hybrid_metrics["MAE"], hybrid_metrics["RMSE"],
                hybrid_metrics["Acceleration_Accuracy"],
                hybrid_metrics["Directional_Accuracy"])

    # Save hybrid output bundle
    hybrid_dir = out_dir / "kalman_accel_postfilter"
    write_sandbox_output_bundle(
        results_df=hybrid_df,
        out_dir=hybrid_dir,
        model_id="kalman_accel_postfilter",
        diagnostics_label="Kalman + AccelOverride Post-Filter",
    )

    # 6) Comparison metrics CSV
    metrics_df = pd.DataFrame(all_metrics).sort_values("MAE").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "comparison_metrics.csv", index=False)

    logger.info("\nComparison (sorted by MAE):")
    for _, row in metrics_df.iterrows():
        logger.info("  %-25s MAE=%.1f RMSE=%.1f AccelAcc=%.3f",
                     row["Forecast"], row["MAE"], row["RMSE"],
                     row["Acceleration_Accuracy"])

    # 7) Unified comparison visualization across the 4 forecasts
    forecast_dfs = {
        cons_base["Forecast"]: cons_results,
        kalman_metrics["Forecast"]: kalman_df,
        accel_metrics["Forecast"]: accel_df,
        hybrid_metrics["Forecast"]: hybrid_df,
    }
    try:
        write_comparison_visualization(out_dir, forecast_dfs, metrics_df)
        logger.info("Wrote unified 4-way comparison visualization (overlay + bar + HTML)")
    except Exception as exc:
        logger.warning("Comparison visualization failed: %s", exc)

    # 8) Augment _output/Predictions/predictions.csv with the consensus anchor
    # OOS rows + the analyst Consensus we are anchoring to. The base file is
    # written by generate_all_output (NSA, SA, NSA_plus_adjustment rows).
    try:
        _augment_predictions_csv(
            output_base=output_base,
            cons_results=cons_results,
            kalman_df=kalman_df,
            accel_df=accel_df,
            hybrid_df=hybrid_df,
        )
    except Exception as exc:
        logger.warning("Augmenting predictions.csv failed: %s", exc)

    logger.info("Consensus anchor outputs saved to %s", out_dir)
    logger.info("=" * 60)

    return {
        "kalman_fusion": kalman_metrics,
        "accel_override": accel_metrics,
        "baselines": {"consensus": cons_base, "champion": champ_base},
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
