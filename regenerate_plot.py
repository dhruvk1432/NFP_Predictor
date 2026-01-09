"""
Regenerate prediction plot with enhanced confidence interval shading.
Reads existing predictions.csv and creates updated visualization.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from Train.backtest_results import plot_predictions_vs_actual
from settings import OUTPUT_DIR, setup_logger, TEMP_DIR

logger = setup_logger(__file__, TEMP_DIR)

def main():
    """Regenerate prediction plots with confidence interval shading."""

    logger.info("="*60)
    logger.info("REGENERATING PREDICTION PLOTS WITH CI SHADING")
    logger.info("="*60)

    # Path to existing predictions
    predictions_path = OUTPUT_DIR / "backtest_results" / "nsa" / "backtest_results_nsa.csv"

    if not predictions_path.exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        return

    # Load predictions
    logger.info(f"\nLoading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)

    # Rename columns to match expected format
    df = df.rename(columns={
        'ds': 'date',
        'predicted': 'pred'
    })

    logger.info(f"Loaded {len(df)} predictions")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Check for confidence interval columns
    ci_cols = ['lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_95', 'upper_95']
    available_ci = [col for col in ci_cols if col in df.columns]
    logger.info(f"Available CI columns: {available_ci}")

    # Generate plot
    output_dir = OUTPUT_DIR / "backtest_results" / "nsa"
    logger.info(f"\nGenerating enhanced plot with confidence intervals...")

    plot_path = plot_predictions_vs_actual(
        predictions_df=df,
        output_dir=output_dir,
        title="NFP Non-Seasonally Adjusted MoM Change: Predictions vs Actuals"
    )

    if plot_path:
        logger.info(f"\n✓ Successfully saved plot to: {plot_path}")
        logger.info("\nThe plot now includes:")
        logger.info("  • Three confidence interval bands (50%, 80%, 95%)")
        logger.info("  • Gradient shading effect (darker = higher confidence)")
        logger.info("  • Future predictions highlighted in red")
        logger.info("  • Actual values in black, predictions in blue")
    else:
        logger.error("Failed to generate plot")

    logger.info("\n" + "="*60)

if __name__ == "__main__":
    main()
