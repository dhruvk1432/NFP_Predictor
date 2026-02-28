"""
Sandbox CatBoost backtest for SA revised target.

This script is intentionally isolated from the core pipeline and writes only
under `_output/sandbox/catboost_sa_revised`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Stable default for joblib-based feature build. Override with
# JOBLIB_MULTIPROCESSING=1 if your terminal supports it.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from settings import OUTPUT_DIR, BACKTEST_MONTHS, TEMP_DIR, setup_logger
from Train.data_loader import load_target_data
from Train.train_lightgbm_nfp import build_training_dataset, clean_features
from Train.sandbox.output_utils import write_sandbox_output_bundle

logger = setup_logger(__file__, TEMP_DIR)
OUT_DIR = OUTPUT_DIR / "sandbox" / "catboost_sa_revised"


def _safe_feature_subset(X: pd.DataFrame, y: pd.Series) -> list[str]:
    cols = clean_features(X, y, min_non_nan=36)
    return [c for c in cols if c in X.columns and c != "ds"]


def run_sandbox_backtest(min_train_rows: int = 120) -> pd.DataFrame:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required for this sandbox experiment. "
            "Install with `pip install catboost`."
        ) from exc

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

    backtest_months = target_df.iloc[-BACKTEST_MONTHS:]["ds"].tolist()
    date_to_idx = {d: i for i, d in enumerate(X_full["ds"])}
    rows = []

    for i, target_month in enumerate(backtest_months, 1):
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

        # Last 15% as validation for early stopping (time-ordered).
        split = max(int(len(X_train_f) * 0.85), 24)
        if split >= len(X_train_f):
            split = len(X_train_f) - 1
        X_tr, X_val = X_train_f.iloc[:split], X_train_f.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        model = CatBoostRegressor(
            loss_function="MAE",
            depth=6,
            learning_rate=0.05,
            iterations=800,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
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
            }
        )
        logger.info(
            "[%d/%d] %s | actual=%s pred=%.1f",
            i,
            len(backtest_months),
            target_month.strftime("%Y-%m"),
            "nan" if pd.isna(actual) else f"{float(actual):.1f}",
            pred,
        )

    return pd.DataFrame(rows)


def _save_outputs(results: pd.DataFrame) -> None:
    n_features = None
    if "n_features" in results.columns and not results["n_features"].dropna().empty:
        n_features = int(results["n_features"].dropna().median())
    write_sandbox_output_bundle(
        results_df=results,
        out_dir=OUT_DIR,
        model_id="catboost_sa_revised",
        diagnostics_label="Sandbox CatBoost SA revised",
        n_features=n_features,
    )
    logger.info("Saved sandbox CatBoost diagnostics bundle to %s", OUT_DIR)


if __name__ == "__main__":
    out = run_sandbox_backtest()
    _save_outputs(out)
