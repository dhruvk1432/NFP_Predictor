"""
Main output orchestrator.

Generates output artifacts after training:
  _output/
  ├── NSA_prediction/          (single-branch or combined)
  ├── SA_prediction/           (single-branch or combined)
  ├── NSA_plus_adjustment/     (combined only)
  ├── Predictions/             (combined only)
  ├── consensus_anchor/        (final forecasts: kalman_fusion,
  │                             panel_kalman_router; diagnostics:
  │                             baseline_consensus, panel_consensus_mean)
  └── Archive/YYYY-MM-DD_HHMMSS/

Archiving is performed by `archive_outputs()` (defined below), which is
also called at the end of `--train-all` so the archive captures the full
final state including the post-training consensus-anchor outputs.
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


# Default folder list archived by `archive_outputs` when called at the end
# of `--train-all`. Paths are relative to `output_base`. Missing folders are
# skipped silently (e.g. sandbox outputs are optional).
DEFAULT_ARCHIVE_FOLDERS: List[str] = [
    "NSA_prediction",
    "SA_prediction",
    "NSA_plus_adjustment",
    "Predictions",
    # Post-training: consensus anchor (final forecasts + comparison artifacts)
    "consensus_anchor",
    # Post-training: sandbox experiments (optional)
    "sandbox/sa_blend_walkforward",
    "sandbox/nsa_predicted_adjustment_revised",
]


def archive_outputs(
    output_base: Path,
    folders: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Snapshot a set of output folders into `_output/Archive/<timestamp>/`.

    Each folder in `folders` is copied (recursively) under the archive
    directory, preserving its relative path. Missing folders are skipped
    with a warning so this is safe to call even when some optional
    post-training stages were disabled.

    Args:
        output_base: Base output dir (typically `settings.OUTPUT_DIR`).
        folders: List of folder paths relative to `output_base`. Defaults
            to `DEFAULT_ARCHIVE_FOLDERS`, which includes the model bundles
            plus the consensus anchor final-forecast scorecard.
        timestamp: Override the archive timestamp (mainly for tests).

    Returns:
        The archive directory path.
    """
    if folders is None:
        folders = DEFAULT_ARCHIVE_FOLDERS
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    archive_root = output_base / "Archive" / timestamp
    archive_root.mkdir(parents=True, exist_ok=True)

    archived: List[str] = []
    skipped: List[str] = []
    for rel in folders:
        src = output_base / rel
        if not src.exists():
            skipped.append(rel)
            continue
        dst = archive_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        archived.append(rel)

    logger.info(f"Archived {len(archived)} folder(s) to {archive_root}")
    for name in archived:
        logger.info(f"  + {name}")
    for name in skipped:
        logger.warning(f"  (skipped, not found) {name}")

    return archive_root


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
    half_life_years: float = 3.0,
) -> None:
    """
    Generate the NSA + PIT-safe predicted seasonal adjustment folder.

    Uses an exponentially-weighted median of same-calendar-month historical
    adjustments to predict the seasonal adjustment factor, then applies it
    to the NSA prediction.

    This is fully PIT-safe: for each target month, only adjustment data
    with operational_available_date < target_ds is used.

    Args:
        nsa_results: NSA backtest results DataFrame.
        sa_results: SA backtest results DataFrame (needs at minimum
            [ds, actual] for SA revised).
        folder: Destination folder path.
        half_life_years: Half-life (years) for the exponential decay of
            same-calendar-month weights. The default of 3.0 is a sane
            starting point; the consensus-anchor stage now tunes this
            against the Kalman fusion objective and re-invokes this
            function with the tuned value.
    """
    from Train.sandbox.experiment_predicted_adjustment import (
        ExpWeightedMedianCovidExcludedPredictor,
        load_adjustment_history,
    )

    folder.mkdir(parents=True, exist_ok=True)

    # Load full historical adjustment series (SA_MoM - NSA_MoM back to 1990)
    adj_history = load_adjustment_history()
    predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=half_life_years)

    # Merge NSA and SA results on date. Keep ALL NSA rows that have a
    # prediction (including OOS future months where actual is NaN); join on
    # SA actual when present so the consensus-anchor stage can pick up OOS
    # NSA+adjustment predictions as champion_pred.
    nsa = nsa_results[nsa_results["predicted"].notna()][["ds", "predicted", "actual"]].copy()
    nsa = nsa.rename(columns={"predicted": "nsa_predicted", "actual": "nsa_actual"})

    sa = sa_results[["ds", "actual"]].copy()
    sa = sa.rename(columns={"actual": "sa_actual"})

    merged = pd.merge(nsa, sa, on="ds", how="left").sort_values("ds").reset_index(drop=True)

    # For each backtest month, predict adjustment using PIT-safe expanding window
    predicted_adjustments = []
    for _, row in merged.iterrows():
        target_ds = row["ds"]

        # PIT filter: only use adjustments available before target month
        if "operational_available_date" in adj_history.columns:
            pit_mask = adj_history["operational_available_date"].notna() & (
                adj_history["operational_available_date"] < target_ds
            )
        else:
            pit_mask = adj_history["ds"] < target_ds
        avail_history = adj_history[pit_mask].copy()

        if avail_history.empty:
            predicted_adjustments.append(0.0)
        else:
            predicted_adjustments.append(predictor.fit_predict(avail_history, target_ds))

    merged["predicted_adjustment"] = predicted_adjustments
    merged["adjusted_predicted"] = merged["nsa_predicted"] + merged["predicted_adjustment"]
    merged["error"] = np.where(
        merged["sa_actual"].notna(),
        merged["sa_actual"] - merged["adjusted_predicted"],
        np.nan,
    )

    # Also compute perfect adjustment for diagnostic comparison (NaN for OOS)
    merged["perfect_adjustment"] = np.where(
        merged["sa_actual"].notna() & merged["nsa_actual"].notna(),
        merged["sa_actual"] - merged["nsa_actual"],
        np.nan,
    )

    # Build a results-like DataFrame for the plotting function
    adj_results = pd.DataFrame({
        "ds": merged["ds"],
        "actual": merged["sa_actual"],
        "predicted": merged["adjusted_predicted"],
        "error": merged["error"],
    })

    # (a) Predictions plot
    logger.info("  Generating NSA + predicted adjustment plot...")
    plot_backtest_predictions(
        adj_results,
        title="NSA Prediction + Exp-Weighted Seasonal Adjustment vs Actual SA MoM",
        save_path=folder / "backtest_predictions.png",
    )

    # (b) Backtest results CSV
    adj_results.to_csv(folder / "backtest_results.csv", index=False)
    logger.info(f"  Saved NSA+adj backtest results CSV ({len(adj_results)} rows)")

    # (c) Summary statistics CSV
    metrics = compute_metrics(adj_results)
    save_metrics_csv(metrics, folder / "summary_statistics.csv")
    logger.info(f"  NSA+adj metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}")

    # (d) Summary table image
    render_summary_table(metrics, [], 0, folder / "summary_table.png")

    logger.info(f"  NSA + predicted adjustment output complete → {folder}")


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
    output_base: Optional[Path] = None,
) -> None:
    """SA arguments may be None when the SA LightGBM branch is retired."""
    """
    Generate the forecast for the *next* unreleased NFP month using the
    production models.

    Predicts only the earliest future month (the next-to-release NFP), one
    row per model. We intentionally do not emit predictions for further-out
    OOS months because: (a) the analyst consensus survey doesn't exist yet
    for them, so the consensus-anchor variants can't be computed; (b) those
    forecasts would mix here with the next-release prediction and confuse
    downstream readers of `predictions.csv`.

    Outputs:
        predictions.csv — one row per model for the next unreleased month
        (NSA, SA, NSA_plus_adjustment, plus Consensus + consensus_anchor_*
        rows added later by the consensus-anchor stage).
    """
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    nsa_oos: Dict[pd.Timestamp, float] = {}
    _branches = [
        ("NSA", nsa_model, nsa_metadata, nsa_X_full, nsa_y_full, nsa_residuals),
    ]
    if sa_model is not None and sa_X_full is not None and sa_y_full is not None:
        _branches.append(("SA", sa_model, sa_metadata, sa_X_full, sa_y_full, sa_residuals))
    for label, model, metadata, X_full, y_full, residuals in _branches:
        feature_cols = metadata["feature_cols"]
        future_mask = y_full.isna()
        if not future_mask.any():
            logger.info(f"  No future months to predict for {label}")
            continue

        # Only predict the next unreleased month.
        X_future_all = X_full[future_mask].copy().sort_values('ds')
        X_future = X_future_all.iloc[[0]]
        future_dates = X_future['ds'].tolist()
        X_pred = X_future[[c for c in feature_cols if c in X_future.columns]]
        preds = model.predict(X_pred)

        if label == "NSA":
            nsa_oos = {pd.Timestamp(ds): float(p) for ds, p in zip(future_dates, preds)}

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

    # NSA + predicted seasonal adjustment for each OOS month.
    # PIT-safe: only uses adjustment history available before target_ds.
    if nsa_oos:
        try:
            from Train.sandbox.experiment_predicted_adjustment import (
                ExpWeightedMedianCovidExcludedPredictor,
                load_adjustment_history,
            )

            adj_history = load_adjustment_history()
            predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=3.0)

            # Residuals from the NSA_plus_adjustment backtest (written earlier
            # in the same generate_all_output run) drive CIs.
            adj_res: np.ndarray = np.array([])
            if output_base is not None:
                adj_path = output_base / "NSA_plus_adjustment" / "backtest_results.csv"
                if not adj_path.exists():
                    adj_path = output_base / "NSA_plus_adjustment_revised" / "backtest_results.csv"
                if adj_path.exists():
                    adj_bt = pd.read_csv(adj_path)
                    if "error" in adj_bt.columns:
                        adj_res = adj_bt["error"].dropna().to_numpy()[-36:]

            for ds, nsa_pred in nsa_oos.items():
                if "operational_available_date" in adj_history.columns:
                    pit_mask = adj_history["operational_available_date"].notna() & (
                        adj_history["operational_available_date"] < ds
                    )
                else:
                    pit_mask = adj_history["ds"] < ds
                avail = adj_history[pit_mask]
                adj_value = predictor.fit_predict(avail, ds) if not avail.empty else 0.0
                pred = nsa_pred + adj_value
                if adj_res.size > 2:
                    rows.append({
                        "model": "NSA_plus_adjustment",
                        "ds": ds,
                        "predicted": pred,
                        "lower_50": pred + np.percentile(adj_res, 25),
                        "upper_50": pred + np.percentile(adj_res, 75),
                        "lower_80": pred + np.percentile(adj_res, 10),
                        "upper_80": pred + np.percentile(adj_res, 90),
                        "lower_95": pred + np.percentile(adj_res, 2.5),
                        "upper_95": pred + np.percentile(adj_res, 97.5),
                    })
                else:
                    rows.append({
                        "model": "NSA_plus_adjustment",
                        "ds": ds,
                        "predicted": pred,
                        "lower_50": np.nan, "upper_50": np.nan,
                        "lower_80": np.nan, "upper_80": np.nan,
                        "lower_95": np.nan, "upper_95": np.nan,
                    })
                logger.info(
                    f"  NSA_plus_adjustment {pd.Timestamp(ds).strftime('%Y-%m')}: "
                    f"NSA={nsa_pred:.0f} + adj={adj_value:.0f} -> Pred={pred:.0f}"
                )
        except Exception as e:
            logger.warning(f"  Skipped NSA_plus_adjustment OOS prediction: {e}")

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
    sa_model=None,
    nsa_metadata: Optional[Dict] = None,
    sa_metadata: Optional[Dict] = None,
    nsa_X_full: Optional[pd.DataFrame] = None,
    sa_X_full: Optional[pd.DataFrame] = None,
    nsa_y_full: Optional[pd.Series] = None,
    sa_y_full: Optional[pd.Series] = None,
    nsa_residuals: Optional[List] = None,
    sa_residuals: Optional[List] = None,
    output_base: Optional[Path] = None,
    suffix: str = '',
    archive: bool = True,
) -> Path:
    """
    Generate the complete output folder structure.

    The SA LightGBM branch has been retired from the published pipeline.
    SA-related arguments are optional: pass them to render an SA diagnostic
    folder, omit them to skip SA-LightGBM artifacts entirely. The final
    SA-revised outputs are written later by the consensus-anchor layer
    (Kalman Fusion and Panel/Kalman Router). The NSA+adjustment folder always
    runs because it feeds that layer.

    Args:
        nsa_results: NSA backtest results DataFrame.
        sa_results: DataFrame providing at minimum [ds, actual] for SA revised
            — used by the NSA+adjustment folder as the actuals to compare
            against. If a full SA LGBM backtest is passed (with predicted/error),
            the SA prediction folder will also be rendered.
        nsa_model: Trained NSA LightGBM Booster.
        sa_model: Trained SA LightGBM Booster (optional; SA-pred folder skipped if None).
        nsa_metadata: NSA model metadata (feature_cols, importance, etc.).
        sa_metadata: SA model metadata (optional).
        nsa_X_full: NSA full training feature matrix (for SHAP).
        sa_X_full: SA full training feature matrix (optional).
        nsa_y_full: NSA target series (to identify future months).
        sa_y_full: SA target series (optional).
        nsa_residuals: NSA OOS residuals from backtest (for confidence intervals).
        sa_residuals: SA OOS residuals (optional).
        output_base: Base output directory. Defaults to settings.OUTPUT_DIR (_output/).
        suffix: Suffix appended to output folder names (e.g., '_revised').

    Returns:
        Path to the output directory.
    """
    if output_base is None:
        output_base = OUTPUT_DIR

    _sa_lgbm_enabled = sa_model is not None and sa_X_full is not None

    logger.info("=" * 60)
    logger.info("GENERATING OUTPUT")
    logger.info(f"Output directory: {output_base}")
    if not _sa_lgbm_enabled:
        logger.info("SA LightGBM branch is OFF — consensus-anchor layer writes final SA outputs")
    logger.info("=" * 60)

    # 1) NSA prediction folder
    logger.info(f"\n[1/5] NSA Prediction{suffix}")
    nsa_folder = output_base / f"NSA_prediction{suffix}"
    _generate_prediction_folder(
        nsa_results, nsa_model, nsa_metadata or {}, nsa_X_full, nsa_folder, f"NSA{suffix}",
    )

    # 2) SA prediction folder (diagnostic only — skipped when SA LightGBM is OFF)
    if _sa_lgbm_enabled:
        logger.info(f"\n[2/5] SA Prediction{suffix}")
        sa_folder = output_base / f"SA_prediction{suffix}"
        _generate_prediction_folder(
            sa_results, sa_model, sa_metadata or {}, sa_X_full, sa_folder, f"SA{suffix}",
        )
    else:
        logger.info(f"\n[2/5] SA Prediction{suffix} — SKIPPED (SA LGBM retired)")

    # 3) NSA + PIT-safe predicted seasonal adjustment folder
    logger.info(f"\n[3/5] NSA + Predicted Seasonal Adjustment{suffix}")
    adj_folder = output_base / f"NSA_plus_adjustment{suffix}"
    _generate_adjustment_folder(nsa_results, sa_results, adj_folder)

    # 4) Forward predictions folder
    logger.info(f"\n[4/5] Forward Predictions{suffix}")
    pred_folder = output_base / f"Predictions{suffix}"
    if nsa_y_full is not None:
        _generate_predictions_folder(
            nsa_model,
            sa_model if _sa_lgbm_enabled else None,
            nsa_metadata or {},
            (sa_metadata or {}) if _sa_lgbm_enabled else None,
            nsa_X_full,
            sa_X_full if _sa_lgbm_enabled else None,
            nsa_y_full,
            sa_y_full if _sa_lgbm_enabled else None,
            nsa_residuals or [],
            (sa_residuals or []) if _sa_lgbm_enabled else None,
            pred_folder,
            output_base=output_base,
        )
    else:
        logger.warning("  Skipping predictions folder — y_full not provided")

    # 5) Archive: copy folders into a timestamped snapshot.
    if archive:
        logger.info("\n[5/5] Archiving output")
        _archive_folders = [
            f"NSA_prediction{suffix}",
            f"NSA_plus_adjustment{suffix}",
            f"Predictions{suffix}",
        ]
        if _sa_lgbm_enabled:
            _archive_folders.insert(1, f"SA_prediction{suffix}")
        archive_outputs(output_base, folders=_archive_folders)
    else:
        logger.info("\n[5/5] Archiving deferred to end of pipeline")

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
    target_source: str = 'revised',
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

    suffix = ''  # No suffix needed — only revised models exist
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
