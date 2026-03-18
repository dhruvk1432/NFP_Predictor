"""
Sandbox walk-forward blend:
  SA direct revised model + NSA + seasonal adjustment revised.

Uses PIT-safe predicted adjustment by default (--adj-source predicted).
The "perfect" adjustment option (--adj-source perfect) is retained for
diagnostic comparison only — it uses lookahead and must NOT be used
for production or consensus-anchor experiments.

This script does not retrain core models and does not change pipeline artifacts.
It consumes existing `_output/*_revised/backtest_results.csv` files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.variance_metrics import compute_variance_kpis, composite_objective_score
from Train.sandbox.output_utils import write_sandbox_output_bundle

logger = setup_logger(__file__, TEMP_DIR)
OUT_DIR = OUTPUT_DIR / "sandbox" / "sa_blend_walkforward"

try:
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    OPTUNA_AVAILABLE = True
except Exception as _optuna_err:
    OPTUNA_AVAILABLE = False
    _OPTUNA_IMPORT_ERROR = _optuna_err


@dataclass(frozen=True)
class BlendTuneOptions:
    enabled: bool = True
    n_trials: int = N_OPTUNA_TRIALS
    timeout: int = OPTUNA_TIMEOUT
    objective_mode: str = "composite"
    cv_splits: int = 4


def _objective(actual: np.ndarray, pred: np.ndarray) -> float:
    mae = float(np.mean(np.abs(actual - pred)))
    kpis = compute_variance_kpis(actual, pred)
    # Compute acceleration and directional accuracy for composite scoring
    accel_acc = 0.0
    if actual.size >= 3:
        da = np.diff(actual.astype(float))
        dp = np.diff(pred.astype(float))
        accel_acc = float(np.mean(np.sign(da) == np.sign(dp)))
    dir_acc = 0.0
    if actual.size >= 1:
        dir_acc = float(np.mean(np.sign(actual.astype(float)) == np.sign(pred.astype(float))))
    return composite_objective_score(
        mae=mae,
        std_ratio=float(kpis["std_ratio"]),
        diff_std_ratio=float(kpis["diff_std_ratio"]),
        tail_mae=float(kpis["tail_mae"]),
        corr_diff=float(kpis["corr_diff"]),
        diff_sign_accuracy=float(kpis["diff_sign_accuracy"]),
        lambda_std_ratio=18.0,
        lambda_diff_std_ratio=18.0,
        lambda_tail_mae=0.20,
        lambda_corr_diff=14.0,
        lambda_diff_sign=10.0,
        accel_accuracy=accel_acc,
        lambda_accel=15.0,
        dir_accuracy=dir_acc,
        lambda_dir=10.0,
    )


def _score(actual: np.ndarray, pred: np.ndarray, objective_mode: str) -> float:
    if actual.size == 0:
        return float("inf")
    if objective_mode == "mae":
        return float(np.mean(np.abs(actual - pred)))
    return _objective(actual, pred)


def _load_inputs(adj_source: str = "perfect") -> pd.DataFrame:
    sa_path = OUTPUT_DIR / "SA_prediction" / "backtest_results.csv"
    if adj_source == "predicted":
        adj_path = OUTPUT_DIR / "sandbox" / "nsa_predicted_adjustment_revised" / "backtest_results.csv"
    else:
        adj_path = OUTPUT_DIR / "NSA_plus_adjustment" / "backtest_results.csv"
    if not sa_path.exists() or not adj_path.exists():
        raise FileNotFoundError(
            "Missing input files. Expected:\n"
            f"  - {sa_path}\n"
            f"  - {adj_path}"
        )

    sa = pd.read_csv(sa_path, parse_dates=["ds"])
    adj = pd.read_csv(adj_path, parse_dates=["ds"])
    sa = sa.rename(columns={"predicted": "sa_predicted", "actual": "sa_actual"})
    adj = adj.rename(columns={"predicted": "adj_predicted", "actual": "adj_actual"})

    merged = pd.merge(
        sa[["ds", "sa_actual", "sa_predicted"]],
        adj[["ds", "adj_actual", "adj_predicted"]],
        on="ds",
        how="inner",
    )

    # Use SA actual as canonical target; fall back to adjustment file if needed.
    merged["actual"] = merged["sa_actual"].where(merged["sa_actual"].notna(), merged["adj_actual"])
    return merged.sort_values("ds").reset_index(drop=True)


def walkforward_blend(
    merged: pd.DataFrame,
    window: int = 18,
    min_history: int = 12,
    grid_step: float = 0.05,
    objective_mode: str = "composite",
) -> pd.DataFrame:
    grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
    rows = []

    for i, row in merged.iterrows():
        history = merged.iloc[:i].copy()
        history = history[history["actual"].notna()]
        if len(history) < min_history:
            w = 0.50
        else:
            hist = history.iloc[-window:].copy()
            actual = hist["actual"].values.astype(float)
            best_score = np.inf
            best_w = 0.50
            for cand_w in grid:
                pred = cand_w * hist["sa_predicted"].values + (1.0 - cand_w) * hist["adj_predicted"].values
                score = _score(actual, pred, objective_mode=objective_mode)
                if score < best_score:
                    best_score = score
                    best_w = float(cand_w)
            w = best_w

        blended_pred = float(w * row["sa_predicted"] + (1.0 - w) * row["adj_predicted"])
        actual_now = row["actual"]
        err = np.nan if pd.isna(actual_now) else float(actual_now - blended_pred)
        rows.append(
            {
                "ds": row["ds"],
                "actual": actual_now,
                "sa_predicted": float(row["sa_predicted"]),
                "adj_predicted": float(row["adj_predicted"]),
                "blend_weight_sa": float(w),
                "predicted": blended_pred,
                "error": err,
            }
        )

    return pd.DataFrame(rows)


def _predict_for_index_range(
    merged: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    window: int,
    min_history: int,
    grid_step: float,
    objective_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
    actual_vals = []
    pred_vals = []
    for i in range(start_idx, end_idx + 1):
        row = merged.iloc[i]
        history = merged.iloc[:i].copy()
        history = history[history["actual"].notna()]
        if len(history) < min_history:
            w = 0.50
        else:
            hist = history.iloc[-window:].copy()
            actual_hist = hist["actual"].values.astype(float)
            best_score = np.inf
            best_w = 0.50
            for cand_w in grid:
                pred_hist = (
                    cand_w * hist["sa_predicted"].values
                    + (1.0 - cand_w) * hist["adj_predicted"].values
                )
                score = _score(actual_hist, pred_hist, objective_mode=objective_mode)
                if score < best_score:
                    best_score = score
                    best_w = float(cand_w)
            w = best_w
        pred = float(w * row["sa_predicted"] + (1.0 - w) * row["adj_predicted"])
        if pd.notna(row["actual"]):
            actual_vals.append(float(row["actual"]))
            pred_vals.append(pred)
    return np.asarray(actual_vals, dtype=float), np.asarray(pred_vals, dtype=float)


def _tune_blend_params(
    merged: pd.DataFrame,
    tune_opts: BlendTuneOptions,
) -> Optional[Dict]:
    if not tune_opts.enabled:
        return None
    if not OPTUNA_AVAILABLE:
        raise RuntimeError(
            f"Optuna tuning requested for blend but unavailable: {_OPTUNA_IMPORT_ERROR}. "
            "Install with `pip install optuna`."
        )

    valid = merged[merged["actual"].notna()].copy().reset_index(drop=True)
    if len(valid) < 60:
        return None

    n_splits = min(max(2, int(tune_opts.cv_splits)), 6)
    if n_splits >= len(valid):
        n_splits = max(2, len(valid) // 20)
    if n_splits < 2:
        return None

    logger.info(
        "Optuna tuning (blend): objective=%s trials=%d timeout=%ss rows=%d splits=%d",
        tune_opts.objective_mode,
        int(tune_opts.n_trials),
        int(tune_opts.timeout),
        len(valid),
        int(n_splits),
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: "optuna.Trial") -> float:
        window = trial.suggest_int("window", 8, 36)
        min_history = trial.suggest_int("min_history", 8, min(24, window))
        grid_step = trial.suggest_categorical("grid_step", [0.01, 0.02, 0.05, 0.10])

        fold_scores = []
        for _, va_idx in tscv.split(valid):
            if len(va_idx) < 3:
                continue
            start_i = int(va_idx[0])
            end_i = int(va_idx[-1])
            y_true, y_pred = _predict_for_index_range(
                merged=valid,
                start_idx=start_i,
                end_idx=end_i,
                window=window,
                min_history=min_history,
                grid_step=grid_step,
                objective_mode=tune_opts.objective_mode,
            )
            if y_true.size == 0:
                continue
            fold_scores.append(
                _score(y_true, y_pred, objective_mode=tune_opts.objective_mode)
            )
        if not fold_scores:
            return float("inf")
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=8),
    )
    study.optimize(objective, n_trials=int(tune_opts.n_trials), timeout=int(tune_opts.timeout))
    best = study.best_trial
    logger.info(
        "Blend Optuna complete: best=%.3f window=%s min_history=%s grid_step=%s",
        float(best.value),
        best.params.get("window"),
        best.params.get("min_history"),
        best.params.get("grid_step"),
    )
    return {
        "window": int(best.params["window"]),
        "min_history": int(best.params["min_history"]),
        "grid_step": float(best.params["grid_step"]),
        "best_score": float(best.value),
    }


def _save_outputs(
    df: pd.DataFrame,
    tune_opts: BlendTuneOptions,
    blend_params: Dict,
) -> None:
    metrics = write_sandbox_output_bundle(
        results_df=df,
        out_dir=OUT_DIR,
        model_id="sa_blend_walkforward",
        diagnostics_label="Sandbox SA revised blend",
        n_features=0,
    )
    valid = df[df["error"].notna()].copy()
    if not valid.empty:
        metrics["Mean_Blend_Weight_SA"] = float(valid["blend_weight_sa"].mean())
        metrics["Blend_Window"] = int(blend_params["window"])
        metrics["Blend_Min_History"] = int(blend_params["min_history"])
        metrics["Blend_Grid_Step"] = float(blend_params["grid_step"])
        metrics["Blend_Tuning_Enabled"] = bool(tune_opts.enabled)
        metrics["Blend_Tuning_Objective"] = tune_opts.objective_mode if tune_opts.enabled else "none"
        metrics["Blend_Tuning_Trials"] = int(tune_opts.n_trials) if tune_opts.enabled else 0
        pd.DataFrame([metrics]).to_csv(OUT_DIR / "summary_statistics.csv", index=False)
    with open(OUT_DIR / "blend_config.json", "w") as f:
        json.dump(
            {"blend_params": blend_params, "tuning": asdict(tune_opts)},
            f,
            indent=2,
        )
    logger.info("Saved sandbox SA blend diagnostics bundle to %s", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox SA revised walk-forward blend.")
    parser.add_argument("--window", type=int, default=18)
    parser.add_argument("--min-history", type=int, default=12)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument(
        "--objective-mode",
        type=str,
        choices=["mae", "composite"],
        default="composite",
        help="Score used for per-step blend-weight search.",
    )
    parser.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        default=True,
        help="Enable Optuna tuning for blend hyperparameters (default: enabled).",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Disable Optuna tuning and use the provided blend hyperparameters as-is.",
    )
    parser.add_argument("--tune-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--tune-timeout", type=int, default=OPTUNA_TIMEOUT)
    parser.add_argument(
        "--tune-objective",
        type=str,
        choices=["mae", "composite"],
        default="composite",
        help="Objective used during Optuna tuning CV.",
    )
    parser.add_argument("--tune-cv-splits", type=int, default=4)
    parser.add_argument(
        "--adj-source",
        type=str,
        choices=["perfect", "predicted"],
        default="predicted",
        help="Source of seasonal adjustment: 'predicted' (PIT-safe, default) or 'perfect' (lookahead, diagnostic only).",
    )
    args = parser.parse_args()

    data = _load_inputs(adj_source=args.adj_source)
    logger.info("Adjustment source: %s", args.adj_source)
    tune_opts = BlendTuneOptions(
        enabled=bool(args.tune),
        n_trials=int(args.tune_trials),
        timeout=int(args.tune_timeout),
        objective_mode=str(args.tune_objective),
        cv_splits=max(2, int(args.tune_cv_splits)),
    )
    blend_params = {
        "window": int(args.window),
        "min_history": int(args.min_history),
        "grid_step": float(args.grid_step),
    }
    if tune_opts.enabled:
        tuned = _tune_blend_params(data, tune_opts=tune_opts)
        if tuned is not None:
            blend_params.update(
                {
                    "window": int(tuned["window"]),
                    "min_history": int(tuned["min_history"]),
                    "grid_step": float(tuned["grid_step"]),
                    "best_score": float(tuned["best_score"]),
                }
            )

    blended = walkforward_blend(
        data,
        window=int(blend_params["window"]),
        min_history=int(blend_params["min_history"]),
        grid_step=float(blend_params["grid_step"]),
        objective_mode=str(args.objective_mode),
    )
    _save_outputs(blended, tune_opts=tune_opts, blend_params=blend_params)
