#!/usr/bin/env python3
"""
Reconstruct NSA output bundle from on-disk artifacts + run Kalman fusion.

Used when an in-flight `--train-all` was killed AFTER the NSA branch
finished (model + metrics persisted) but BEFORE generate_all_output ran.
Skips fresh SA training — uses whatever SA artifacts are on disk.

Flow:
  1. Load cached NSA training dataset (X_full, y_full); winsorize.
  2. Load saved NSA model + metadata from _output/models/lightgbm_nfp/.
  3. Reconstruct nsa backtest_results from _output/backtest/*_metrics.json.
  4. Predict the OOS month with the saved model so forward predictions are fresh.
  5. Load stale SA artifacts from disk (model, metadata, backtest results).
  6. Run generate_all_output → writes NSA_prediction/, SA_prediction/,
     NSA_plus_adjustment/, Predictions/.
  7. Run Post-1 sandbox NSA predicted adjustment + Post-3 Kalman fusion
     + archive. Skips Post-2 SA blend entirely.
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
from Train.config import (
    N_OPTUNA_TRIALS,
    OPTUNA_TIMEOUT,
)
from Train.data_loader import load_target_data
from Train.training_dataset_cache import load_cached_dataset, save_cached_dataset
from Train.train_lightgbm_nfp import (
    build_training_dataset,
    get_model_id,
)
from utils.transforms import winsorize_covid_period
from Train.Output_code.generate_output import generate_all_output, archive_outputs
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)


def _load_full_dataset(target_type: str, release_type: str, target_source: str):
    target_df = load_target_data(
        target_type=target_type, release_type=release_type, target_source=target_source,
    )
    cached = load_cached_dataset(
        target_df, target_type, release_type, target_source,
        start_date=None, end_date=None,
    )
    if cached is not None:
        logger.info(f"  Loaded cached dataset for {target_type}_{release_type}_{target_source}")
        X_full, y_full = cached
    else:
        logger.info(f"  Building dataset (not cached) for {target_type}_{release_type}_{target_source}")
        X_full, y_full = build_training_dataset(
            target_df, target_type=target_type, release_type=release_type,
            target_source=target_source, show_progress=False,
        )
        if not X_full.empty:
            save_cached_dataset(
                X_full, y_full, target_df,
                target_type, release_type, target_source,
                start_date=None, end_date=None,
            )

    # Mirror train_and_evaluate winsorization so X_full/y_full match training.
    x_indexed = X_full.set_index('ds')
    numeric = x_indexed.select_dtypes(include=[np.number]).columns
    x_indexed[numeric] = winsorize_covid_period(x_indexed[numeric])
    X_full = x_indexed.reset_index(names='ds')

    y_indexed = pd.Series(
        y_full.values, index=pd.to_datetime(X_full['ds']), name='y_mom',
    )
    y_full = winsorize_covid_period(y_indexed).reset_index(drop=True)

    return X_full, y_full


def _load_model_bundle(model_id: str):
    model_dir = OUTPUT_DIR / "models" / "lightgbm_nfp" / model_id
    model = lgb.Booster(model_file=str(model_dir / f"lightgbm_{model_id}_model.txt"))
    with open(model_dir / f"lightgbm_{model_id}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, metadata


def _build_backtest_results(model_id: str, X_full: pd.DataFrame, y_full: pd.Series,
                             model: lgb.Booster, metadata: dict) -> pd.DataFrame:
    """Reconstruct a backtest_results DataFrame.

    Historical rows come from the per-month metrics JSON written by the
    aborted run.  The OOS row (latest target month with no actual yet) is
    re-predicted using the saved model so consensus-anchor / Kalman has a
    fresh forward forecast.
    """
    metrics_path = OUTPUT_DIR / "backtest" / f"{model_id}_metrics.json"
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

    # OOS row: predict for the latest target month present in X_full but
    # whose target has no released value yet (the row where y_full / actual
    # is NaN).
    feature_cols = metadata["feature_cols"]
    X_full_ds = pd.to_datetime(X_full["ds"]).reset_index(drop=True)
    last_known_actual_ds = df["ds"].max()
    oos_candidates = X_full_ds[X_full_ds > last_known_actual_ds].sort_values()
    if not oos_candidates.empty:
        oos_ds = oos_candidates.iloc[0]
        oos_idx = X_full_ds[X_full_ds == oos_ds].index[0]
        X_oos = X_full.iloc[[oos_idx]][feature_cols]
        oos_pred = float(model.predict(X_oos)[0])
        oos_row = pd.DataFrame([{
            "ds": oos_ds,
            "actual": np.nan,
            "predicted": oos_pred,
            "error": np.nan,
            "dir_correct": None,
            "accel_correct": None,
        }])
        df = pd.concat([df, oos_row], ignore_index=True).sort_values("ds").reset_index(drop=True)
        logger.info(f"  {model_id} OOS prediction {oos_ds.strftime('%Y-%m')}: {oos_pred:.1f}")

    # Carry summary metrics + gate metadata in attrs (matches train_and_evaluate).
    df.attrs["summary_metrics"] = metrics.get("overall", {})
    return df


def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("RECONSTRUCT NSA + RUN KALMAN FUSION (skip SA training)")
    logger.info("=" * 70)

    nsa_id = get_model_id("nsa", "first", "revised")
    sa_id = get_model_id("sa", "first", "revised")

    logger.info("\n[1/5] Loading NSA + SA full datasets...")
    nsa_X_full, nsa_y_full = _load_full_dataset("nsa", "first", "revised")
    sa_X_full, sa_y_full = _load_full_dataset("sa", "first", "revised")
    logger.info(f"  NSA dataset: {len(nsa_X_full)} rows × {nsa_X_full.shape[1]} cols")
    logger.info(f"  SA dataset:  {len(sa_X_full)} rows × {sa_X_full.shape[1]} cols")

    logger.info("\n[2/5] Loading saved model bundles...")
    nsa_model, nsa_metadata = _load_model_bundle(nsa_id)
    sa_model, sa_metadata = _load_model_bundle(sa_id)
    logger.info(f"  NSA features: {len(nsa_metadata['feature_cols'])}")
    logger.info(f"  SA features:  {len(sa_metadata['feature_cols'])}")

    logger.info("\n[3/5] Reconstructing backtest_results DataFrames...")
    nsa_results = _build_backtest_results(nsa_id, nsa_X_full, nsa_y_full,
                                          nsa_model, nsa_metadata)
    sa_results = _build_backtest_results(sa_id, sa_X_full, sa_y_full,
                                         sa_model, sa_metadata)
    logger.info(f"  NSA rows: {len(nsa_results)} | SA rows: {len(sa_results)}")

    logger.info("\n[4/5] Writing output bundle (NSA_prediction, SA_prediction, "
                "NSA_plus_adjustment, Predictions)...")
    generate_all_output(
        nsa_results=nsa_results,
        sa_results=sa_results,
        nsa_model=nsa_model,
        sa_model=sa_model,
        nsa_metadata=nsa_metadata,
        sa_metadata=sa_metadata,
        nsa_X_full=nsa_X_full,
        sa_X_full=sa_X_full,
        nsa_y_full=nsa_y_full,
        sa_y_full=sa_y_full,
        nsa_residuals=nsa_metadata.get("residuals", []),
        sa_residuals=sa_metadata.get("residuals", []),
        output_base=OUTPUT_DIR,
        suffix="",
        archive=False,
    )

    logger.info("\n[5/5] Post-training (Post-1 NSA seasonal adj, Post-3 Kalman fusion, "
                "Post-4 archive). Post-2 SA blend SKIPPED.")

    # Post-1: NSA predicted seasonal adjustment (sandbox)
    try:
        from Train.sandbox.experiment_predicted_adjustment import (
            load_adjustment_history,
            load_backtest_inputs,
            run_walkforward_backtest,
            evaluate_models,
            save_outputs,
            SARIMAPredictor,
            MonthlyAveragePredictor,
            TwelveMonthComplementPredictor,
            SameMonthLastYearPredictor,
            ExpWeightedMonthlyAvgPredictor,
            LinearRegressionPredictor,
        )
        logger.info("\n[Post-1] NSA predicted seasonal adjustment sandbox...")
        adj_history = load_adjustment_history()
        backtest_inputs = load_backtest_inputs()
        adj_models = [
            SARIMAPredictor(),
            MonthlyAveragePredictor(),
            TwelveMonthComplementPredictor(),
            SameMonthLastYearPredictor(),
            ExpWeightedMonthlyAvgPredictor(half_life_years=3.0),
            LinearRegressionPredictor(),
        ]
        model_results = run_walkforward_backtest(adj_history, backtest_inputs, adj_models)
        comparison = evaluate_models(model_results)
        best_name = comparison.iloc[0]["model_name"]
        save_outputs(best_name, model_results, comparison)
        logger.info(f"  Post-1 complete (best adjuster: {best_name})")
    except Exception as e:
        logger.warning(f"  Post-1 failed (non-fatal): {e}")

    # Post-3: Consensus anchor (Kalman fusion)
    logger.info("\n[Post-3] Consensus anchor (Kalman fusion)...")
    run_consensus_anchor_pipeline(
        output_base=OUTPUT_DIR,
        tune=True,
        n_trials=N_OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
    )
    logger.info("  Post-3 complete")

    # Post-4: Archive
    logger.info("\n[Post-4] Archiving final outputs...")
    archive_outputs(OUTPUT_DIR)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed/60:.1f} min")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
