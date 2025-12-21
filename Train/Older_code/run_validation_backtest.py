"""
Validation Backtest Runner

Runs the expanding window backtest using configuration from settings.
Backtest duration is controlled by BACKTEST_MONTHS in .env.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Train.run_expanding_backtest import run_expanding_window_backtest
from Train.evaluate_predictions import evaluate_backtest
from settings import setup_logger, TEMP_DIR, BACKTEST_MONTHS

logger = setup_logger(__file__, TEMP_DIR)

if __name__ == "__main__":
    logger.info("="*70)
    logger.info(f"RUNNING VALIDATION BACKTEST ({BACKTEST_MONTHS} MONTHS)")
    logger.info("="*70)
    
    # Run backtest
    logger.info(f"\nSTEP 1: Running expanding window backtest for last {BACKTEST_MONTHS} months...")
    results_df = run_expanding_window_backtest(
        backtest_months=BACKTEST_MONTHS,
        initial_training_months=120,
        save_results=True
    )
    
    # Evaluate results
    logger.info("\nSTEP 2: Evaluating predictions and generating visualizations...")
    metrics, _ = evaluate_backtest()
    
    logger.info("\n" + "="*70)
    logger.info("VALIDATION BACKTEST COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: _output/backtest/")
    logger.info(f"Visualizations saved to: _output/backtest/evaluation/")
    logger.info("\nKey Metrics:")
    if metrics:
        logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.1f}%")
        logger.info(f"  R²: {metrics.get('r_squared', 0):.3f}")
        logger.info(f"  RMSE: {metrics.get('rmse', 0):.1f} thousand")
    else:
        logger.warning("  No metrics computed (possibly no valid predictions)")
    logger.info("="*70)
