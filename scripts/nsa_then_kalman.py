#!/usr/bin/env python3
"""
Reproduce morning's 88.9 MAE Kalman fusion result by:
  1. Restoring morning archive (_output/Archive/2026-05-12_093623/) to live _output/.
  2. Running NSA training fresh (deterministic w/ seed=42 + reverted code →
     should match morning's NSA backtest exactly).
  3. Regenerating NSA_prediction + NSA_plus_adjustment folders using fresh NSA
     + morning-archive SA (no SA retrain).
  4. Re-running Post-1 NSA seasonal adjustment sandbox, Post-3 consensus anchor
     (Kalman fusion), Post-4 archive.

The morning archive's SA_prediction/, Predictions/, and consensus_anchor/ are
used as the SA baseline so Kalman fusion sees the same SA inputs morning saw.
SA model training is intentionally skipped.
"""
from __future__ import annotations

import pickle
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.train_lightgbm_nfp import train_and_evaluate, get_model_id
from Train.Output_code.generate_output import (
    _generate_prediction_folder,
    _generate_adjustment_folder,
    archive_outputs,
)
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)

ARCHIVE = OUTPUT_DIR / "Archive" / "2026-05-12_093623"
RESTORE_FOLDERS = [
    "NSA_prediction",
    "SA_prediction",
    "NSA_plus_adjustment",
    "Predictions",
    "consensus_anchor",
    "sandbox",
]


def restore_morning_archive() -> None:
    logger.info("=" * 70)
    logger.info(f"[1/4] Restoring morning archive → live _output/")
    logger.info("=" * 70)
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Morning archive not found: {ARCHIVE}")
    for folder in RESTORE_FOLDERS:
        src = ARCHIVE / folder
        dst = OUTPUT_DIR / folder
        if not src.exists():
            logger.warning(f"  Missing in archive (skipping): {folder}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.info(f"  Restored {folder}")


def run_nsa_training():
    logger.info("=" * 70)
    logger.info("[2/4] Running NSA training (deterministic, seed=42)")
    logger.info("=" * 70)
    result = train_and_evaluate(
        target_type='nsa',
        release_type='first',
        target_source='revised',
        use_huber_loss=False,
        huber_delta=350.0,
        tune=True,
        nsa_backtest_results=None,
    )
    if result is None:
        raise RuntimeError("NSA train_and_evaluate returned None")
    model, feature_cols, residuals, backtest_results, X_full, y_full = result
    logger.info(f"NSA training done — backtest rows: {len(backtest_results)}, "
                f"features: {len(feature_cols)}")
    summary = backtest_results.attrs.get('summary_metrics', {})
    if summary:
        rmse = summary.get('RMSE')
        mae = summary.get('MAE')
        if rmse is not None and mae is not None:
            logger.info(f"  Backtest summary: RMSE={rmse:.2f} MAE={mae:.2f}")
        else:
            logger.info(f"  Backtest summary keys: {list(summary.keys())}")
    return model, feature_cols, residuals, backtest_results, X_full, y_full


def regenerate_nsa_folders(nsa_model, nsa_metadata, nsa_results, nsa_X_full):
    logger.info("=" * 70)
    logger.info("[3/4] Regenerating NSA_prediction + NSA_plus_adjustment "
                "(SA from morning archive)")
    logger.info("=" * 70)

    # NSA prediction folder — fresh model + fresh backtest_results
    _generate_prediction_folder(
        nsa_results, nsa_model, nsa_metadata, nsa_X_full,
        OUTPUT_DIR / "NSA_prediction", "NSA",
    )

    # NSA_plus_adjustment — fresh NSA + morning-archive SA backtest_results
    sa_results = pd.read_csv(
        ARCHIVE / "SA_prediction" / "backtest_results.csv",
        parse_dates=["ds"],
    )
    _generate_adjustment_folder(
        nsa_results, sa_results,
        OUTPUT_DIR / "NSA_plus_adjustment",
    )


def run_post_training():
    logger.info("=" * 70)
    logger.info("[4/4] Post-1 NSA sandbox + Post-3 Kalman fusion + Post-4 archive")
    logger.info("=" * 70)

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
    logger.info("  Post-4 complete")


def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("NSA-ONLY TRAINING + KALMAN FUSION (reproduce morning result)")
    logger.info("=" * 70)

    restore_morning_archive()

    nsa_model, nsa_features, nsa_residuals, nsa_results, nsa_X_full, nsa_y_full = (
        run_nsa_training()
    )

    # train_and_evaluate has already persisted model + metadata; reload metadata
    nsa_id = get_model_id('nsa', 'first', 'revised')
    model_dir = OUTPUT_DIR / "models" / "lightgbm_nfp" / nsa_id
    with open(model_dir / f"lightgbm_{nsa_id}_metadata.pkl", "rb") as f:
        nsa_metadata = pickle.load(f)

    regenerate_nsa_folders(nsa_model, nsa_metadata, nsa_results, nsa_X_full)

    run_post_training()

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed/60:.1f} min")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
