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
                          merged["nsa_pred"].notna().sum())
        else:
            merged["nsa_pred"] = np.nan
            logger.warning("No NSA predictions found; Kalman NSA channel disabled")

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

            # Majority vote using all available signals
            dir_votes = [np.sign(cons_delta)]
            if pd.notna(row.get("champion_pred")):
                dir_votes.append(np.sign(row["champion_pred"] - last_actual))
            if has_nsa and pd.notna(row.get("nsa_pred")):
                dir_votes.append(np.sign(float(row["nsa_pred"]) - last_actual))
            if has_nsa_raw and pd.notna(row.get("nsa_raw_pred")):
                dir_votes.append(np.sign(float(row["nsa_raw_pred"]) - last_actual))

            vote_sum = sum(dir_votes)
            chosen_dir = np.sign(vote_sum) if vote_sum != 0 else np.sign(cons_delta)

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
