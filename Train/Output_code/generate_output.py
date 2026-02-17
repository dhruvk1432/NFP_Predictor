"""
Main output orchestrator.

Generates the full _Output/ folder structure after training both NSA and SA models:
  _Output/
  ├── NSA_prediction/   (5 items)
  ├── SA_prediction/    (5 items)
  ├── NSA_plus_adjustment/  (3 items)
  └── Archive/YYYY-MM-DD_HHMMSS/  (dated copy of the above 3 folders)
"""

import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import setup_logger, TEMP_DIR

from Train.Output_code.metrics import compute_metrics, save_metrics_csv
from Train.Output_code.plots import (
    plot_backtest_predictions,
    plot_shap_summary,
    render_summary_table,
)
from Train.Output_code.feature_importance import save_feature_importance_csv

logger = setup_logger(__file__, TEMP_DIR)


def _get_top_features(importance: Dict[str, float], n: int = 5) -> list:
    """Return top-n (feature_name, score) tuples sorted by importance descending."""
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]


def _generate_prediction_folder(
    results_df: pd.DataFrame,
    model,
    metadata: Dict,
    X_full: pd.DataFrame,
    folder: Path,
    label: str,
) -> None:
    """
    Generate the 5 outputs for an NSA or SA prediction folder.

    Args:
        results_df: Backtest results DataFrame.
        model: Trained LightGBM Booster.
        metadata: Model metadata dict (contains feature_cols, importance).
        X_full: Full training feature matrix (for SHAP).
        folder: Destination folder path.
        label: Human-readable label for titles (e.g. "NSA" or "SA").
    """
    folder.mkdir(parents=True, exist_ok=True)
    importance = metadata["importance"]
    feature_cols = metadata["feature_cols"]
    n_features = len(feature_cols)

    # (a) Backtest predictions plot
    logger.info(f"  Generating {label} predictions plot...")
    plot_backtest_predictions(
        results_df,
        title=f"{label} NFP MoM Change — Predicted vs Actual",
        save_path=folder / "backtest_predictions.png",
    )

    # (b) Summary statistics CSV
    metrics = compute_metrics(results_df)
    save_metrics_csv(metrics, folder / "summary_statistics.csv")
    logger.info(f"  {label} metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}")

    # (c) SHAP values plot
    logger.info(f"  Generating {label} SHAP plot (this may take a moment)...")
    try:
        plot_shap_summary(model, X_full, feature_cols, folder / "shap_values.png")
    except Exception as e:
        logger.warning(f"  SHAP plot failed for {label}: {e}")

    # (d) Feature importance CSV
    save_feature_importance_csv(importance, folder / "feature_importance.csv")

    # (e) Summary table image
    top_features = _get_top_features(importance, n=5)
    render_summary_table(metrics, top_features, n_features, folder / "summary_table.png")

    logger.info(f"  {label} output complete → {folder}")


def _generate_adjustment_folder(
    nsa_results: pd.DataFrame,
    sa_results: pd.DataFrame,
    folder: Path,
) -> None:
    """
    Generate the NSA + perfect seasonal adjustment folder (3 outputs).

    The perfect adjustment for each month is:
        adjustment = actual_SA_MoM - actual_NSA_MoM
        adjusted_pred = nsa_predicted + adjustment

    We then compare adjusted_pred to actual SA MoM.

    Args:
        nsa_results: NSA backtest results DataFrame.
        sa_results: SA backtest results DataFrame.
        folder: Destination folder path.
    """
    folder.mkdir(parents=True, exist_ok=True)

    # Merge NSA and SA results on date
    nsa = nsa_results[nsa_results["actual"].notna()][["ds", "predicted", "actual"]].copy()
    nsa = nsa.rename(columns={"predicted": "nsa_predicted", "actual": "nsa_actual"})

    sa = sa_results[sa_results["actual"].notna()][["ds", "actual"]].copy()
    sa = sa.rename(columns={"actual": "sa_actual"})

    merged = pd.merge(nsa, sa, on="ds", how="inner")

    # Compute perfect seasonal adjustment
    merged["adjustment"] = merged["sa_actual"] - merged["nsa_actual"]
    merged["adjusted_predicted"] = merged["nsa_predicted"] + merged["adjustment"]
    merged["error"] = merged["sa_actual"] - merged["adjusted_predicted"]

    # Build a results-like DataFrame for the plotting function
    adj_results = pd.DataFrame({
        "ds": merged["ds"],
        "actual": merged["sa_actual"],
        "predicted": merged["adjusted_predicted"],
        "error": merged["error"],
    })

    # (a) Predictions plot
    logger.info("  Generating NSA + adjustment predictions plot...")
    plot_backtest_predictions(
        adj_results,
        title="NSA Prediction + Perfect Seasonal Adjustment vs Actual SA MoM",
        save_path=folder / "backtest_predictions.png",
    )

    # (b) Summary statistics CSV
    metrics = compute_metrics(adj_results)
    save_metrics_csv(metrics, folder / "summary_statistics.csv")
    logger.info(f"  NSA+adj metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}")

    # (c) Summary table image
    render_summary_table(metrics, [], 0, folder / "summary_table.png")

    logger.info(f"  NSA + adjustment output complete → {folder}")


def generate_all_output(
    nsa_results: pd.DataFrame,
    sa_results: pd.DataFrame,
    nsa_model,
    sa_model,
    nsa_metadata: Dict,
    sa_metadata: Dict,
    nsa_X_full: pd.DataFrame,
    sa_X_full: pd.DataFrame,
    output_base: Optional[Path] = None,
) -> Path:
    """
    Generate the complete _Output/ folder structure.

    Args:
        nsa_results: NSA backtest results DataFrame.
        sa_results: SA backtest results DataFrame.
        nsa_model: Trained NSA LightGBM Booster.
        sa_model: Trained SA LightGBM Booster.
        nsa_metadata: NSA model metadata (feature_cols, importance, etc.).
        sa_metadata: SA model metadata.
        nsa_X_full: NSA full training feature matrix (for SHAP).
        sa_X_full: SA full training feature matrix (for SHAP).
        output_base: Base output directory. Defaults to project-level _Output/.

    Returns:
        Path to the output directory.
    """
    if output_base is None:
        output_base = Path(__file__).resolve().parent.parent.parent / "_Output"

    logger.info("=" * 60)
    logger.info("GENERATING OUTPUT")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 60)

    # 1) NSA prediction folder
    logger.info("\n[1/4] NSA Prediction")
    nsa_folder = output_base / "NSA_prediction"
    _generate_prediction_folder(
        nsa_results, nsa_model, nsa_metadata, nsa_X_full, nsa_folder, "NSA",
    )

    # 2) SA prediction folder
    logger.info("\n[2/4] SA Prediction")
    sa_folder = output_base / "SA_prediction"
    _generate_prediction_folder(
        sa_results, sa_model, sa_metadata, sa_X_full, sa_folder, "SA",
    )

    # 3) NSA + perfect seasonal adjustment folder
    logger.info("\n[3/4] NSA + Perfect Seasonal Adjustment")
    adj_folder = output_base / "NSA_plus_adjustment"
    _generate_adjustment_folder(nsa_results, sa_results, adj_folder)

    # 4) Archive: copy all three folders with timestamp
    logger.info("\n[4/4] Archiving output")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_folder = output_base / "Archive" / timestamp

    for src_name in ["NSA_prediction", "SA_prediction", "NSA_plus_adjustment"]:
        src = output_base / src_name
        dst = archive_folder / src_name
        shutil.copytree(src, dst)

    logger.info(f"  Archived to {archive_folder}")
    logger.info("\n" + "=" * 60)
    logger.info("OUTPUT GENERATION COMPLETE")
    logger.info("=" * 60)

    return output_base
