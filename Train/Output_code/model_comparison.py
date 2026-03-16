"""
Model Comparison Scorecard

Generates side-by-side comparison tables for all model variants
(NSA/SA revised) plus the SA+NSA revised blend champion
after `--train-all` completes.

Champion / Challenger labelling:
- The SA+NSA revised walk-forward blend is the champion.
- The best individual LightGBM model by MAE is the challenger.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from settings import OUTPUT_DIR, setup_logger, TEMP_DIR
from Train.variance_metrics import compute_variance_kpis

logger = setup_logger(__file__, TEMP_DIR)

COMPARISON_DIR = OUTPUT_DIR / "models" / "lightgbm_nfp"


BLEND_BACKTEST_PATH = OUTPUT_DIR / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv"
CHAMPION_MODEL_ID = "sa_blend_champion"


def _load_blend_result() -> dict | None:
    """Load the SA+NSA revised walk-forward blend backtest if it exists."""
    if not BLEND_BACKTEST_PATH.exists():
        logger.warning(
            "Blend backtest not found at %s — champion row will be absent. "
            "Run Train/sandbox/experiment_sa_blend.py first.",
            BLEND_BACKTEST_PATH,
        )
        return None
    bt = pd.read_csv(BLEND_BACKTEST_PATH, parse_dates=["ds"])
    if bt.empty:
        return None
    return {
        "backtest_results": bt,
        "n_features": 0,  # blend doesn't have its own features
        "n_train_obs": 0,
    }


def _compute_model_row(model_id: str, info: dict) -> dict | None:
    """Compute a single scorecard row from backtest results."""
    bt = info["backtest_results"]
    if bt.empty:
        logger.warning("[%s] Empty backtest results, skipping", model_id)
        return None

    bt_valid = bt[~bt["error"].isna()]
    if bt_valid.empty:
        logger.warning("[%s] No valid backtest predictions", model_id)
        return None

    errors = bt_valid["error"].values
    abs_errors = np.abs(errors)

    # Coverage — only available for full LightGBM models with CI columns
    ci_cols = {"actual", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"}
    interval_cols = {"in_50_interval", "in_80_interval", "in_95_interval"}
    if ci_cols.issubset(bt_valid.columns):
        c50 = float(((bt_valid["actual"] >= bt_valid["lower_50"]) & (bt_valid["actual"] <= bt_valid["upper_50"])).mean())
        c80 = float(((bt_valid["actual"] >= bt_valid["lower_80"]) & (bt_valid["actual"] <= bt_valid["upper_80"])).mean())
        c95 = float(((bt_valid["actual"] >= bt_valid["lower_95"]) & (bt_valid["actual"] <= bt_valid["upper_95"])).mean())
    elif interval_cols.issubset(bt_valid.columns):
        c50 = float(bt_valid["in_50_interval"].mean())
        c80 = float(bt_valid["in_80_interval"].mean())
        c95 = float(bt_valid["in_95_interval"].mean())
    else:
        c50 = c80 = c95 = np.nan

    row = {
        "Model": model_id,
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "MAE": np.mean(abs_errors),
        "Median_AE": np.median(abs_errors),
        "Max_AE": np.max(abs_errors),
        "Mean_Error": np.mean(errors),
        "50%_Coverage": c50 * 100.0 if not np.isnan(c50) else np.nan,
        "80%_Coverage": c80 * 100.0 if not np.isnan(c80) else np.nan,
        "95%_Coverage": c95 * 100.0 if not np.isnan(c95) else np.nan,
        "N_Features": info.get("n_features", 0),
        "N_Train_Obs": info.get("n_train_obs", 0),
        "N_Backtest_Months": len(bt_valid),
    }

    if {"actual", "predicted"}.issubset(bt_valid.columns):
        vk = compute_variance_kpis(bt_valid["actual"].values, bt_valid["predicted"].values)
        row.update({
            "STD_Ratio": float(vk["std_ratio"]),
            "Diff_STD_Ratio": float(vk["diff_std_ratio"]),
            "Corr_Diff": float(vk["corr_diff"]),
            "Diff_Sign_Accuracy": float(vk["diff_sign_accuracy"]),
            "Tail_MAE": float(vk["tail_mae"]),
            "Extreme_Hit_Rate": float(vk["extreme_hit_rate"]),
        })

    # Acceleration accuracy (for blend and any model with actual + predicted)
    if {"actual", "predicted"}.issubset(bt_valid.columns) and len(bt_valid) >= 2:
        actual_vals = bt_valid["actual"].values
        pred_vals = bt_valid["predicted"].values
        accel_acc = float(np.mean(np.sign(np.diff(actual_vals)) == np.sign(np.diff(pred_vals))))
        row["Acceleration_Accuracy"] = accel_acc

    return row


def generate_comparison_scorecard(
    all_results: dict,
    save_dir: Path = COMPARISON_DIR,
    include_blend: bool = True,
) -> pd.DataFrame:
    """
    Generate a comparative scorecard across all trained model variants.

    The SA+NSA revised walk-forward blend is automatically loaded as the
    champion model.  The best individual LightGBM variant by MAE is labelled
    the challenger.

    Args:
        all_results: Dict mapping model_id -> {
            'backtest_results': pd.DataFrame,
            'n_features': int,
            'n_train_obs': int,
        }
        save_dir: Directory to save outputs.
        include_blend: If True (default), load the blend backtest and
            include it as the champion row.

    Returns:
        DataFrame with one row per model and columns for each metric.
        The 'Role' column marks 'CHAMPION', 'CHALLENGER', or blank.
    """
    rows = []

    # --- Inject blend champion ---
    if include_blend:
        blend_info = _load_blend_result()
        if blend_info is not None:
            blend_row = _compute_model_row(CHAMPION_MODEL_ID, blend_info)
            if blend_row is not None:
                rows.append(blend_row)

    # --- Individual LightGBM models ---
    for model_id, info in all_results.items():
        model_row = _compute_model_row(model_id, info)
        if model_row is not None:
            rows.append(model_row)

    if not rows:
        logger.error("No valid results to compare")
        return pd.DataFrame()

    scorecard = pd.DataFrame(rows).set_index("Model")
    scorecard = scorecard.sort_values("MAE")

    # --- Assign champion / challenger roles ---
    scorecard["Role"] = ""
    if CHAMPION_MODEL_ID in scorecard.index:
        scorecard.loc[CHAMPION_MODEL_ID, "Role"] = "CHAMPION"
        # Challenger = best individual model (first non-blend row by MAE)
        individual_models = [m for m in scorecard.index if m != CHAMPION_MODEL_ID]
        if individual_models:
            challenger = individual_models[0]  # already sorted by MAE
            scorecard.loc[challenger, "Role"] = "CHALLENGER"
    else:
        # No blend available — best individual model is champion
        scorecard.iloc[0, scorecard.columns.get_loc("Role")] = "CHAMPION"
        if len(scorecard) > 1:
            scorecard.iloc[1, scorecard.columns.get_loc("Role")] = "CHALLENGER"

    # Print to log
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SCORECARD")
    logger.info("=" * 80)
    logger.info("\n%s", scorecard.to_string())

    champion_idx = scorecard[scorecard["Role"] == "CHAMPION"].index
    if not champion_idx.empty:
        champ = champion_idx[0]
        logger.info(
            "\n>>> CHAMPION: %s (MAE=%.1f)", champ, scorecard.loc[champ, "MAE"]
        )
    challenger_idx = scorecard[scorecard["Role"] == "CHALLENGER"].index
    if not challenger_idx.empty:
        chall = challenger_idx[0]
        logger.info(
            ">>> CHALLENGER: %s (MAE=%.1f)", chall, scorecard.loc[chall, "MAE"]
        )

    # Save outputs
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "model_comparison.csv"
    scorecard.to_csv(csv_path)
    logger.info("Saved comparison CSV: %s", csv_path)

    html_path = save_dir / "model_comparison.html"
    _save_styled_html(scorecard, html_path)
    logger.info("Saved comparison HTML: %s", html_path)

    return scorecard


def _save_styled_html(scorecard: pd.DataFrame, path: Path) -> None:
    """Save a styled HTML table with conditional formatting."""
    styler = scorecard.style

    # Highlight min values in error columns (lower is better)
    error_cols = ['RMSE', 'MAE', 'Median_AE', 'Max_AE', 'Tail_MAE']
    existing_error_cols = [c for c in error_cols if c in scorecard.columns]
    if existing_error_cols:
        styler = styler.highlight_min(subset=existing_error_cols, color='#90EE90')

    # Highlight max values in coverage columns (higher is better)
    coverage_cols = [
        '50%_Coverage', '80%_Coverage', '95%_Coverage',
        'STD_Ratio', 'Diff_STD_Ratio', 'Corr_Diff', 'Diff_Sign_Accuracy',
        'Acceleration_Accuracy', 'Extreme_Hit_Rate',
    ]
    existing_coverage_cols = [c for c in coverage_cols if c in scorecard.columns]
    if existing_coverage_cols:
        styler = styler.highlight_max(subset=existing_coverage_cols, color='#90EE90')

    # Format numbers
    format_dict = {
        'Role': '{}',
        'RMSE': '{:.1f}',
        'MAE': '{:.1f}',
        'Median_AE': '{:.1f}',
        'Max_AE': '{:.0f}',
        'Mean_Error': '{:+.1f}',
        '50%_Coverage': '{:.1f}%',
        '80%_Coverage': '{:.1f}%',
        '95%_Coverage': '{:.1f}%',
        'STD_Ratio': '{:.3f}',
        'Diff_STD_Ratio': '{:.3f}',
        'Corr_Diff': '{:.3f}',
        'Diff_Sign_Accuracy': '{:.1%}',
        'Acceleration_Accuracy': '{:.1%}',
        'Tail_MAE': '{:.1f}',
        'Extreme_Hit_Rate': '{:.1%}',
        'N_Features': '{:.0f}',
        'N_Train_Obs': '{:.0f}',
        'N_Backtest_Months': '{:.0f}',
    }
    existing_format = {k: v for k, v in format_dict.items() if k in scorecard.columns}
    styler = styler.format(existing_format)

    html = styler.set_caption("LightGBM NFP Model Comparison").to_html()

    # Wrap with minimal CSS for readability
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; background: #f5f5f5; }}
  table {{ border-collapse: collapse; margin: 20px auto; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th, td {{ padding: 10px 16px; border: 1px solid #ddd; text-align: right; }}
  th {{ background: #2c3e50; color: white; }}
  caption {{ font-size: 1.4em; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  tr:hover {{ background: #e8f4f8; }}
</style>
</head>
<body>
{html}
</body>
</html>"""

    path.write_text(full_html)
