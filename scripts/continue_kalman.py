#!/usr/bin/env python3
"""
Continuation of nsa_then_kalman.py after its post-NSA-training logger bug.
Picks up the freshly-saved NSA model + metrics JSON, reconstructs the NSA
backtest_results df, regenerates NSA_prediction + NSA_plus_adjustment
(using morning-archive SA), runs Kalman fusion, and archives.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from utils.transforms import winsorize_covid_period
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.data_loader import load_target_data
from Train.training_dataset_cache import load_cached_dataset
from Train.train_lightgbm_nfp import build_training_dataset, get_model_id
from Train.Output_code.generate_output import (
    _generate_prediction_folder,
    _generate_adjustment_folder,
    archive_outputs,
)
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)

ARCHIVE = OUTPUT_DIR / "Archive" / "2026-05-12_093623"


def load_full_dataset(target_type: str, release_type: str, target_source: str):
    target_df = load_target_data(
        target_type=target_type, release_type=release_type, target_source=target_source,
    )
    cached = load_cached_dataset(
        target_df, target_type, release_type, target_source,
        start_date=None, end_date=None,
    )
    if cached is None:
        logger.info(f"  Building dataset for {target_type}_{release_type}_{target_source}")
        X_full, y_full = build_training_dataset(
            target_df, target_type=target_type, release_type=release_type,
            target_source=target_source, show_progress=False,
        )
    else:
        logger.info(f"  Loaded cached dataset for {target_type}_{release_type}_{target_source}")
        X_full, y_full = cached

    x_indexed = X_full.set_index('ds')
    numeric = x_indexed.select_dtypes(include=[np.number]).columns
    x_indexed[numeric] = winsorize_covid_period(x_indexed[numeric])
    X_full = x_indexed.reset_index(names='ds')
    y_indexed = pd.Series(y_full.values, index=pd.to_datetime(X_full['ds']), name='y_mom')
    y_full = winsorize_covid_period(y_indexed).reset_index(drop=True)
    return X_full, y_full


def build_nsa_results(nsa_id: str, X_full: pd.DataFrame, model: lgb.Booster, metadata: dict) -> pd.DataFrame:
    metrics_path = OUTPUT_DIR / "backtest" / f"{nsa_id}_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    rows = []
    for entry in metrics["per_month"]:
        rows.append({
            "ds": pd.Timestamp(entry["ds"] + "-01"),
            "actual": entry["actual"],
            "predicted": entry["predicted"],
            "error": entry["error"],
            "dir_correct": entry.get("dir_correct"),
            "accel_correct": entry.get("accel_correct"),
        })
    df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    feature_cols = metadata["feature_cols"]
    X_full_ds = pd.to_datetime(X_full["ds"]).reset_index(drop=True)
    last_known = df["ds"].max()
    oos_candidates = X_full_ds[X_full_ds > last_known].sort_values()
    for oos_ds in oos_candidates:
        oos_idx = X_full_ds[X_full_ds == oos_ds].index[0]
        X_oos = X_full.iloc[[oos_idx]][feature_cols]
        oos_pred = float(model.predict(X_oos)[0])
        df = pd.concat([df, pd.DataFrame([{
            "ds": oos_ds, "actual": np.nan, "predicted": oos_pred,
            "error": np.nan, "dir_correct": None, "accel_correct": None,
        }])], ignore_index=True)
        logger.info(f"  OOS {oos_ds.strftime('%Y-%m')}: {oos_pred:.1f}")

    df = df.sort_values("ds").reset_index(drop=True)
    df.attrs["summary_metrics"] = metrics.get("overall", {})
    return df


def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("CONTINUE FROM NSA TRAINING → KALMAN FUSION")
    logger.info("=" * 70)

    nsa_id = get_model_id('nsa', 'first', 'revised')

    logger.info(f"\n[1/4] Loading freshly-trained NSA model from disk...")
    model_dir = OUTPUT_DIR / "models" / "lightgbm_nfp" / nsa_id
    nsa_model = lgb.Booster(model_file=str(model_dir / f"lightgbm_{nsa_id}_model.txt"))
    with open(model_dir / f"lightgbm_{nsa_id}_metadata.pkl", "rb") as f:
        nsa_metadata = pickle.load(f)
    logger.info(f"  NSA features: {len(nsa_metadata['feature_cols'])}")

    logger.info(f"\n[2/4] Loading NSA training dataset...")
    nsa_X_full, nsa_y_full = load_full_dataset("nsa", "first", "revised")

    logger.info(f"\n[3/4] Reconstructing NSA backtest_results + regenerating folders...")
    nsa_results = build_nsa_results(nsa_id, nsa_X_full, nsa_model, nsa_metadata)
    logger.info(f"  NSA rows: {len(nsa_results)}")
    summary = nsa_results.attrs.get("summary_metrics", {})
    if summary:
        rmse = summary.get('RMSE')
        mae = summary.get('MAE')
        logger.info(f"  Backtest summary: RMSE={rmse:.2f} MAE={mae:.2f}"
                    if rmse is not None and mae is not None
                    else f"  Backtest summary keys: {list(summary.keys())}")

    _generate_prediction_folder(
        nsa_results, nsa_model, nsa_metadata, nsa_X_full,
        OUTPUT_DIR / "NSA_prediction", "NSA",
    )

    sa_results = pd.read_csv(
        ARCHIVE / "SA_prediction" / "backtest_results.csv",
        parse_dates=["ds"],
    )
    logger.info(f"  Using morning-archive SA (rows={len(sa_results)})")
    _generate_adjustment_folder(
        nsa_results, sa_results,
        OUTPUT_DIR / "NSA_plus_adjustment",
    )

    logger.info(f"\n[4/4] Post-3 Kalman fusion + Post-4 archive...")
    run_consensus_anchor_pipeline(
        output_base=OUTPUT_DIR, tune=True,
        n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT,
    )
    archive_outputs(OUTPUT_DIR)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed/60:.1f} min")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
