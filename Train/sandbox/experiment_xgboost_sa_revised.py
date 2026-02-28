"""
Sandbox XGBoost backtest for SA revised target.

Standalone experiment. No core pipeline behavior changes.
Writes outputs under `_output/sandbox/xgboost_sa_revised`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Stable default for joblib-based feature build. Override with
# JOBLIB_MULTIPROCESSING=1 if your terminal supports it.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from settings import OUTPUT_DIR, BACKTEST_MONTHS, TEMP_DIR, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.data_loader import load_target_data
from Train.train_lightgbm_nfp import build_training_dataset, clean_features
from Train.model import calculate_sample_weights
from Train.variance_metrics import compute_variance_kpis, composite_objective_score
from Train.sandbox.output_utils import write_sandbox_output_bundle

logger = setup_logger(__file__, TEMP_DIR)
OUT_DIR = OUTPUT_DIR / "sandbox" / "xgboost_sa_revised"

try:
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    OPTUNA_AVAILABLE = True
except Exception as _optuna_err:
    OPTUNA_AVAILABLE = False
    _OPTUNA_IMPORT_ERROR = _optuna_err


@dataclass(frozen=True)
class TuneOptions:
    enabled: bool = False
    n_trials: int = N_OPTUNA_TRIALS
    timeout: int = OPTUNA_TIMEOUT
    objective_mode: str = "composite"
    tune_every_steps: int = 1
    use_huber_loss: bool = False


def _safe_feature_subset(X: pd.DataFrame, y: pd.Series) -> list[str]:
    cols = clean_features(X, y, min_non_nan=36)
    return [c for c in cols if c in X.columns and c != "ds"]


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, objective_mode: str) -> float:
    if y_true.size == 0:
        return float("inf")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if objective_mode == "mae":
        return mae

    kpis = compute_variance_kpis(y_true, y_pred)
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
    )


def _default_xgb_params(use_huber_loss: bool = True) -> Dict:
    return {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "objective": "reg:pseudohubererror" if use_huber_loss else "reg:absoluteerror",
        "random_state": 42,
        "n_jobs": 4,
    }


def _tune_xgboost_params(
    X_train_with_ds: pd.DataFrame,
    y_train: pd.Series,
    target_month: pd.Timestamp,
    tune_opts: TuneOptions,
) -> Optional[Dict]:
    if not tune_opts.enabled:
        return None
    if not OPTUNA_AVAILABLE:
        raise RuntimeError(
            f"Optuna tuning requested but unavailable: {_OPTUNA_IMPORT_ERROR}. "
            "Install with `pip install optuna`."
        )

    X_local = X_train_with_ds.copy().replace([np.inf, -np.inf], np.nan)
    y_local = pd.to_numeric(y_train, errors="coerce").astype(float)
    valid = y_local.notna()
    X_local = X_local.loc[valid].reset_index(drop=True)
    y_local = y_local.loc[valid].reset_index(drop=True)
    if len(X_local) < 100:
        return None

    X_feats = X_local.drop(columns=["ds"], errors="ignore")
    ds = pd.to_datetime(X_local["ds"]).reset_index(drop=True)
    n_splits = min(5, max(3, len(X_local) // 80))
    if n_splits >= len(X_local):
        n_splits = max(2, len(X_local) // 40)
    if n_splits < 2:
        return None

    logger.info(
        "Optuna tuning (XGBoost): month=%s objective=%s trials=%d timeout=%ss rows=%d features=%d splits=%d",
        target_month.strftime("%Y-%m"),
        tune_opts.objective_mode,
        int(tune_opts.n_trials),
        int(tune_opts.timeout),
        len(X_feats),
        len(X_feats.columns),
        int(n_splits),
    )

    from xgboost import XGBRegressor
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": (
                "reg:pseudohubererror" if tune_opts.use_huber_loss else "reg:absoluteerror"
            ),
            "random_state": 42,
            "n_jobs": 4,
        }
        half_life = trial.suggest_float("half_life_months", 6.0, 120.0)

        fold_scores = []
        for tr_idx, va_idx in tscv.split(X_feats):
            X_tr = X_feats.iloc[tr_idx]
            X_va = X_feats.iloc[va_idx]
            y_tr = y_local.iloc[tr_idx]
            y_va = y_local.iloc[va_idx]
            if len(X_va) < 3:
                continue

            fold_target = pd.Timestamp(ds.iloc[va_idx].max())
            w_tr = calculate_sample_weights(
                pd.DataFrame({"ds": ds.iloc[tr_idx].values}),
                target_month=fold_target,
                half_life_months=float(half_life),
            )
            model = XGBRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            pred = model.predict(X_va)
            fold_scores.append(
                _score_predictions(
                    y_true=y_va.values.astype(float),
                    y_pred=np.asarray(pred, dtype=float),
                    objective_mode=tune_opts.objective_mode,
                )
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
    best_params = dict(best.params)
    half_life = float(best_params.pop("half_life_months"))
    best_params["objective"] = (
        "reg:pseudohubererror" if tune_opts.use_huber_loss else "reg:absoluteerror"
    )
    best_params["random_state"] = 42
    best_params["n_jobs"] = 4

    logger.info(
        "XGBoost Optuna complete: best=%.3f lr=%.4f depth=%s trees=%s half_life=%.1f",
        float(best.value),
        float(best_params.get("learning_rate", 0.05)),
        best_params.get("max_depth"),
        best_params.get("n_estimators"),
        half_life,
    )
    return {
        "params": best_params,
        "half_life_months": half_life,
        "best_score": float(best.value),
    }


def run_sandbox_backtest(
    min_train_rows: int = 120,
    backtest_months: int = BACKTEST_MONTHS,
    tune_opts: Optional[TuneOptions] = None,
) -> pd.DataFrame:
    from xgboost import XGBRegressor

    if tune_opts is None:
        tune_opts = TuneOptions(enabled=False)

    target_df = load_target_data(
        target_type="sa", release_type="first", target_source="revised"
    )
    X_full, y_full = build_training_dataset(
        target_df=target_df,
        target_type="sa",
        release_type="first",
        target_source="revised",
        show_progress=False,
    )
    if X_full.empty:
        raise RuntimeError("Failed to build SA revised feature dataset.")

    backtest_targets = target_df.iloc[-backtest_months:]["ds"].tolist()
    date_to_idx = {d: i for i, d in enumerate(X_full["ds"])}
    rows = []
    cached_tune: Optional[Dict] = None

    for i, target_month in enumerate(backtest_targets, 1):
        target_idx = date_to_idx.get(target_month)
        if target_idx is None:
            continue

        train_mask = X_full["ds"] < target_month
        train_idx = X_full[train_mask].index.tolist()
        if len(train_idx) < min_train_rows:
            continue

        y_train = y_full.iloc[train_idx]
        valid = ~y_train.isna()
        if int(valid.sum()) < min_train_rows:
            continue

        X_train = X_full.iloc[[train_idx[j] for j in range(len(train_idx)) if valid.iloc[j]]].copy()
        y_train = y_train[valid].copy()
        X_pred = X_full.iloc[[target_idx]].copy()
        actual = y_full.iloc[target_idx]

        feature_cols = _safe_feature_subset(X_train, y_train)
        if not feature_cols:
            continue

        X_train_f = X_train[feature_cols]
        X_pred_f = X_pred[feature_cols]

        split = max(int(len(X_train_f) * 0.85), 24)
        if split >= len(X_train_f):
            split = len(X_train_f) - 1
        X_tr, X_val = X_train_f.iloc[:split], X_train_f.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        if tune_opts.enabled and ((i - 1) % max(1, int(tune_opts.tune_every_steps)) == 0):
            cached_tune = _tune_xgboost_params(
                X_train_with_ds=X_train[["ds"] + feature_cols].copy(),
                y_train=y_train,
                target_month=target_month,
                tune_opts=tune_opts,
            )

        if cached_tune is not None:
            model_params = dict(cached_tune.get("params", {}))
            half_life = float(cached_tune.get("half_life_months", 60.0))
            tuned = 1
        else:
            model_params = _default_xgb_params(use_huber_loss=tune_opts.use_huber_loss)
            half_life = 60.0
            tuned = 0

        all_weights = calculate_sample_weights(
            X_train[["ds"]].copy(),
            target_month=target_month,
            half_life_months=half_life,
        )
        w_tr = all_weights[:split]

        model = XGBRegressor(**model_params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = float(model.predict(X_pred_f)[0])
        err = np.nan if pd.isna(actual) else float(actual - pred)
        rows.append(
            {
                "ds": target_month,
                "actual": actual,
                "predicted": pred,
                "error": err,
                "n_features": len(feature_cols),
                "n_train_samples": len(X_train_f),
                "tuned": tuned,
                "tuned_half_life_months": float(half_life),
            }
        )
        logger.info(
            "[%d/%d] %s | actual=%s pred=%.1f | tuned=%s",
            i,
            len(backtest_targets),
            target_month.strftime("%Y-%m"),
            "nan" if pd.isna(actual) else f"{float(actual):.1f}",
            pred,
            "yes" if tuned else "no",
        )

    return pd.DataFrame(rows)


def _save_outputs(results: pd.DataFrame, tune_opts: TuneOptions) -> None:
    n_features = None
    if "n_features" in results.columns and not results["n_features"].dropna().empty:
        n_features = int(results["n_features"].dropna().median())
    metrics = write_sandbox_output_bundle(
        results_df=results,
        out_dir=OUT_DIR,
        model_id="xgboost_sa_revised",
        diagnostics_label="Sandbox XGBoost SA revised",
        n_features=n_features,
    )
    metrics["Tuning_Enabled"] = bool(tune_opts.enabled)
    metrics["Tuning_Objective"] = tune_opts.objective_mode if tune_opts.enabled else "none"
    metrics["Tuning_Trials"] = int(tune_opts.n_trials) if tune_opts.enabled else 0
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "summary_statistics.csv", index=False)
    with open(OUT_DIR / "tuning_config.json", "w") as f:
        import json

        json.dump(asdict(tune_opts), f, indent=2)
    logger.info("Saved sandbox XGBoost diagnostics bundle to %s", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox XGBoost SA revised backtest.")
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--backtest-months", type=int, default=BACKTEST_MONTHS)
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning mode.")
    parser.add_argument("--tune-trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--tune-timeout", type=int, default=OPTUNA_TIMEOUT)
    parser.add_argument(
        "--tune-objective",
        type=str,
        choices=["mae", "composite"],
        default="composite",
    )
    parser.add_argument(
        "--tune-every-steps",
        type=int,
        default=1,
        help="Re-tune every N walk-forward steps.",
    )
    parser.add_argument(
        "--tune-huber",
        action="store_true",
        help="Use pseudo-Huber objective in tuned/static XGBoost.",
    )
    args = parser.parse_args()

    tune_opts = TuneOptions(
        enabled=bool(args.tune),
        n_trials=int(args.tune_trials),
        timeout=int(args.tune_timeout),
        objective_mode=str(args.tune_objective),
        tune_every_steps=max(1, int(args.tune_every_steps)),
        use_huber_loss=bool(args.tune_huber),
    )
    out = run_sandbox_backtest(
        min_train_rows=int(args.min_train_rows),
        backtest_months=int(args.backtest_months),
        tune_opts=tune_opts,
    )
    _save_outputs(out, tune_opts=tune_opts)
