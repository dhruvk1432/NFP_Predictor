"""
Quick Runner: 36-Month Backtest

Runs the 36-month backtest and generates evaluation report.
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
    logger.info("RUNNING 36-MONTH BACKTEST")
    logger.info("="*70)
    
    # Run backtest (last 36 months)
    logger.info("\nSTEP 1: Running expanding window backtest...")
    results_df = run_expanding_window_backtest(
        backtest_months = BACKTEST_MONTHS,
        initial_training_months=120,
        save_results=True
    )
    
    # Evaluate results
    logger.info("\nSTEP 2: Evaluating predictions and generating visualizations...")
    metrics, _ = evaluate_backtest()
    
    logger.info("\n" + "="*70)
    logger.info("BACKTEST COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: _output/backtest/")
    logger.info(f"Visualizations saved to: _output/backtest/evaluation/")
    logger.info("\nKey Metrics:")
    logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    logger.info(f"  R²: {metrics['r_squared']:.3f}")
    logger.info(f"  RMSE: {metrics['rmse']:.1f} thousand")
    logger.info("="*70)
