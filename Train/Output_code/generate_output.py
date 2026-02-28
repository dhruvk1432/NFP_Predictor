"""
Main output orchestrator.

Generates output artifacts after training:
  _output/
  ├── NSA_prediction/          (single-branch or combined)
  ├── SA_prediction/           (single-branch or combined)
  ├── NSA_plus_adjustment/     (combined only)
  ├── Predictions/             (combined only)
  └── Archive/YYYY-MM-DD_HHMMSS/
"""

import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import setup_logger, TEMP_DIR, OUTPUT_DIR

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

    # (b) Backtest results CSV (per-month predictions vs actuals)
    csv_cols = ["ds", "actual", "predicted", "error"]
    optional_cols = [
        "lower_80", "upper_80", "lower_50", "upper_50", "lower_95", "upper_95",
        "prediction_strategy", "strategy_selected_score", "strategy_applied_count",
        "n_features", "n_train_samples",
    ]
    csv_cols += [c for c in optional_cols if c in results_df.columns]
    results_df[csv_cols].to_csv(folder / "backtest_results.csv", index=False)
    logger.info(f"  Saved {label} backtest results CSV ({len(results_df)} rows)")

    # (c) Summary statistics CSV
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

    # (b) Backtest results CSV
    adj_results.to_csv(folder / "backtest_results.csv", index=False)
    logger.info(f"  Saved NSA+adj backtest results CSV ({len(adj_results)} rows)")

    # (c) Summary statistics CSV
    metrics = compute_metrics(adj_results)
    save_metrics_csv(metrics, folder / "summary_statistics.csv")
    logger.info(f"  NSA+adj metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}")

    # (c) Summary table image
    render_summary_table(metrics, [], 0, folder / "summary_table.png")

    logger.info(f"  NSA + adjustment output complete → {folder}")


def _generate_predictions_folder(
    nsa_model,
    sa_model,
    nsa_metadata: Dict,
    sa_metadata: Dict,
    nsa_X_full: pd.DataFrame,
    sa_X_full: pd.DataFrame,
    nsa_y_full: pd.Series,
    sa_y_full: pd.Series,
    nsa_residuals: List,
    sa_residuals: List,
    folder: Path,
) -> None:
    """
    Generate forward predictions using the production models for months
    where the target is NaN (unreleased data).

    Outputs:
        predictions.csv — one row per future month with point prediction + CIs
    """
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, model, metadata, X_full, y_full, residuals in [
        ("NSA", nsa_model, nsa_metadata, nsa_X_full, nsa_y_full, nsa_residuals),
        ("SA", sa_model, sa_metadata, sa_X_full, sa_y_full, sa_residuals),
    ]:
        feature_cols = metadata["feature_cols"]
        future_mask = y_full.isna()
        if not future_mask.any():
            logger.info(f"  No future months to predict for {label}")
            continue

        X_future = X_full[future_mask].copy()
        future_dates = X_future['ds'].tolist()
        X_pred = X_future[[c for c in feature_cols if c in X_future.columns]]
        preds = model.predict(X_pred)

        # Confidence intervals from OOS residuals
        if len(residuals) > 2:
            res = np.array(residuals[-36:])  # Use up to last 36 OOS residuals
            for j, (ds, pred) in enumerate(zip(future_dates, preds)):
                rows.append({
                    "model": label,
                    "ds": ds,
                    "predicted": pred,
                    "lower_50": pred + np.percentile(res, 25),
                    "upper_50": pred + np.percentile(res, 75),
                    "lower_80": pred + np.percentile(res, 10),
                    "upper_80": pred + np.percentile(res, 90),
                    "lower_95": pred + np.percentile(res, 2.5),
                    "upper_95": pred + np.percentile(res, 97.5),
                })
        else:
            # Fallback: no residual history
            for j, (ds, pred) in enumerate(zip(future_dates, preds)):
                rows.append({
                    "model": label,
                    "ds": ds,
                    "predicted": pred,
                    "lower_50": np.nan, "upper_50": np.nan,
                    "lower_80": np.nan, "upper_80": np.nan,
                    "lower_95": np.nan, "upper_95": np.nan,
                })

        for r in rows[-len(future_dates):]:
            logger.info(f"  {label} {r['ds'].strftime('%Y-%m')}: "
                        f"Pred={r['predicted']:.0f} "
                        f"[80% CI: {r['lower_80']:.0f}, {r['upper_80']:.0f}]")

    if rows:
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(folder / "predictions.csv", index=False)
        logger.info(f"  Saved {len(rows)} predictions to {folder / 'predictions.csv'}")
    else:
        logger.warning("  No future predictions to output")


def generate_all_output(
    nsa_results: pd.DataFrame,
    sa_results: pd.DataFrame,
    nsa_model,
    sa_model,
    nsa_metadata: Dict,
    sa_metadata: Dict,
    nsa_X_full: pd.DataFrame,
    sa_X_full: pd.DataFrame,
    nsa_y_full: pd.Series = None,
    sa_y_full: pd.Series = None,
    nsa_residuals: List = None,
    sa_residuals: List = None,
    output_base: Optional[Path] = None,
    suffix: str = '',
) -> Path:
    """
    Generate the complete output folder structure.

    Args:
        nsa_results: NSA backtest results DataFrame.
        sa_results: SA backtest results DataFrame.
        nsa_model: Trained NSA LightGBM Booster.
        sa_model: Trained SA LightGBM Booster.
        nsa_metadata: NSA model metadata (feature_cols, importance, etc.).
        sa_metadata: SA model metadata.
        nsa_X_full: NSA full training feature matrix (for SHAP).
        sa_X_full: SA full training feature matrix (for SHAP).
        nsa_y_full: NSA target series (to identify future months).
        sa_y_full: SA target series (to identify future months).
        nsa_residuals: NSA OOS residuals from backtest (for confidence intervals).
        sa_residuals: SA OOS residuals from backtest (for confidence intervals).
        output_base: Base output directory. Defaults to settings.OUTPUT_DIR (_output/).
        suffix: Suffix appended to output folder names (e.g., '_revised').

    Returns:
        Path to the output directory.
    """
    if output_base is None:
        output_base = OUTPUT_DIR

    logger.info("=" * 60)
    logger.info("GENERATING OUTPUT")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 60)

    # 1) NSA prediction folder
    logger.info(f"\n[1/5] NSA Prediction{suffix}")
    nsa_folder = output_base / f"NSA_prediction{suffix}"
    _generate_prediction_folder(
        nsa_results, nsa_model, nsa_metadata, nsa_X_full, nsa_folder, f"NSA{suffix}",
    )

    # 2) SA prediction folder
    logger.info(f"\n[2/5] SA Prediction{suffix}")
    sa_folder = output_base / f"SA_prediction{suffix}"
    _generate_prediction_folder(
        sa_results, sa_model, sa_metadata, sa_X_full, sa_folder, f"SA{suffix}",
    )

    # 3) NSA + perfect seasonal adjustment folder
    logger.info(f"\n[3/5] NSA + Perfect Seasonal Adjustment{suffix}")
    adj_folder = output_base / f"NSA_plus_adjustment{suffix}"
    _generate_adjustment_folder(nsa_results, sa_results, adj_folder)

    # 4) Forward predictions folder
    logger.info(f"\n[4/5] Forward Predictions{suffix}")
    pred_folder = output_base / f"Predictions{suffix}"
    if nsa_y_full is not None and sa_y_full is not None:
        _generate_predictions_folder(
            nsa_model, sa_model,
            nsa_metadata, sa_metadata,
            nsa_X_full, sa_X_full,
            nsa_y_full, sa_y_full,
            nsa_residuals or [], sa_residuals or [],
            pred_folder,
        )
    else:
        logger.warning("  Skipping predictions folder — y_full not provided")

    # 5) Archive: copy all folders with timestamp
    logger.info("\n[5/5] Archiving output")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_folder = output_base / "Archive" / timestamp

    for src_name in [f"NSA_prediction{suffix}", f"SA_prediction{suffix}",
                     f"NSA_plus_adjustment{suffix}", f"Predictions{suffix}"]:
        src = output_base / src_name
        if src.exists():
            dst = archive_folder / src_name
            shutil.copytree(src, dst)

    logger.info(f"  Archived to {archive_folder}")
    logger.info("\n" + "=" * 60)
    logger.info("OUTPUT GENERATION COMPLETE")
    logger.info("=" * 60)

    return output_base


def generate_single_branch_output(
    results_df: pd.DataFrame,
    model,
    metadata: Dict,
    X_full: pd.DataFrame,
    target_type: str,
    target_source: str = 'first_release',
    output_base: Optional[Path] = None,
    archive: bool = False,
) -> Path:
    """
    Generate visualization artifacts for a single trained branch.

    Writes:
      - _output/NSA_prediction{_revised}/... OR
      - _output/SA_prediction{_revised}/...
    """
    if output_base is None:
        output_base = OUTPUT_DIR

    suffix = '_revised' if target_source == 'revised' else ''
    target_norm = str(target_type).strip().lower()
    if target_norm not in {'nsa', 'sa'}:
        raise ValueError(f"target_type must be 'nsa' or 'sa', got: {target_type}")

    folder_name = f"{'NSA' if target_norm == 'nsa' else 'SA'}_prediction{suffix}"
    label = f"{'NSA' if target_norm == 'nsa' else 'SA'}{suffix}"
    branch_folder = output_base / folder_name

    logger.info("=" * 60)
    logger.info(f"GENERATING SINGLE-BRANCH OUTPUT: {folder_name}")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 60)

    _generate_prediction_folder(
        results_df=results_df,
        model=model,
        metadata=metadata,
        X_full=X_full,
        folder=branch_folder,
        label=label,
    )

    if archive:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        archive_folder = output_base / "Archive" / timestamp / folder_name
        archive_folder.parent.mkdir(parents=True, exist_ok=True)
        if archive_folder.exists():
            shutil.rmtree(archive_folder)
        shutil.copytree(branch_folder, archive_folder)
        logger.info(f"Archived single-branch output → {archive_folder}")

    return branch_folder
