"""
Model Comparison Scorecard

Generates side-by-side comparison tables for all 4 model variants
(NSA/SA × first_release/revised) after `--train-all` completes.
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


def generate_comparison_scorecard(
    all_results: dict,
    save_dir: Path = COMPARISON_DIR,
) -> pd.DataFrame:
    """
    Generate a comparative scorecard across all trained model variants.

    Args:
        all_results: Dict mapping model_id -> {
            'backtest_results': pd.DataFrame,
            'n_features': int,
            'n_train_obs': int,
        }
        save_dir: Directory to save outputs.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    rows = []

    for model_id, info in all_results.items():
        bt = info['backtest_results']
        if bt.empty:
            logger.warning(f"[{model_id}] Empty backtest results, skipping")
            continue

        # Filter to actual backtest rows (exclude future predictions)
        bt_valid = bt[~bt['error'].isna()]

        if bt_valid.empty:
            logger.warning(f"[{model_id}] No valid backtest predictions")
            continue

        errors = bt_valid['error'].values
        abs_errors = np.abs(errors)
        if {'actual', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_95', 'upper_95'}.issubset(bt_valid.columns):
            c50 = float(((bt_valid['actual'] >= bt_valid['lower_50']) & (bt_valid['actual'] <= bt_valid['upper_50'])).mean())
            c80 = float(((bt_valid['actual'] >= bt_valid['lower_80']) & (bt_valid['actual'] <= bt_valid['upper_80'])).mean())
            c95 = float(((bt_valid['actual'] >= bt_valid['lower_95']) & (bt_valid['actual'] <= bt_valid['upper_95'])).mean())
        else:
            c50 = float(bt_valid['in_50_interval'].mean())
            c80 = float(bt_valid['in_80_interval'].mean())
            c95 = float(bt_valid['in_95_interval'].mean())

        row = {
            'Model': model_id,
            'RMSE': np.sqrt(np.mean(errors ** 2)),
            'MAE': np.mean(abs_errors),
            'Median_AE': np.median(abs_errors),
            'Max_AE': np.max(abs_errors),
            'Mean_Error': np.mean(errors),  # bias
            '50%_Coverage': c50 * 100.0,
            '80%_Coverage': c80 * 100.0,
            '95%_Coverage': c95 * 100.0,
            'N_Features': info.get('n_features', 0),
            'N_Train_Obs': info.get('n_train_obs', 0),
            'N_Backtest_Months': len(bt_valid),
        }
        if {'actual', 'predicted'}.issubset(bt_valid.columns):
            vk = compute_variance_kpis(bt_valid['actual'].values, bt_valid['predicted'].values)
            row.update({
                'STD_Ratio': float(vk['std_ratio']),
                'Diff_STD_Ratio': float(vk['diff_std_ratio']),
                'Corr_Diff': float(vk['corr_diff']),
                'Diff_Sign_Accuracy': float(vk['diff_sign_accuracy']),
                'Tail_MAE': float(vk['tail_mae']),
                'Extreme_Hit_Rate': float(vk['extreme_hit_rate']),
            })
        rows.append(row)

    if not rows:
        logger.error("No valid results to compare")
        return pd.DataFrame()

    scorecard = pd.DataFrame(rows).set_index('Model')

    # Sort by MAE (best first)
    scorecard = scorecard.sort_values('MAE')

    # Print to log
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SCORECARD")
    logger.info("=" * 80)
    logger.info(f"\n{scorecard.to_string()}")

    # Highlight the best model
    best = scorecard.index[0]
    logger.info(f"\n>>> Best model by MAE: {best} (MAE={scorecard.loc[best, 'MAE']:.1f})")

    # Save outputs
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "model_comparison.csv"
    scorecard.to_csv(csv_path)
    logger.info(f"Saved comparison CSV: {csv_path}")

    html_path = save_dir / "model_comparison.html"
    _save_styled_html(scorecard, html_path)
    logger.info(f"Saved comparison HTML: {html_path}")

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
        'STD_Ratio', 'Diff_STD_Ratio', 'Corr_Diff', 'Diff_Sign_Accuracy', 'Extreme_Hit_Rate',
    ]
    existing_coverage_cols = [c for c in coverage_cols if c in scorecard.columns]
    if existing_coverage_cols:
        styler = styler.highlight_max(subset=existing_coverage_cols, color='#90EE90')

    # Format numbers
    format_dict = {
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
