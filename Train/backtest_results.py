"""
Backtest Results Presentation

Generates comprehensive backtest reports:
- Selected features table
- Error metrics table (RMSE, MAE, MAPE, etc.)
- Prediction vs actual visualizations with confidence intervals
- Summary statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, setup_logger, TEMP_DIR

logger = setup_logger(__file__, TEMP_DIR)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False


def calculate_error_metrics(predictions_df: pd.DataFrame, target_col: str = 'sa_mom_change') -> Dict:
    """
    Calculate comprehensive error metrics from predictions.

    Args:
        predictions_df: DataFrame with 'pred' and 'actual' columns
        target_col: Name of target variable for logging

    Returns:
        Dictionary of error metrics
    """
    pred = predictions_df['pred'].values
    actual = predictions_df['actual'].values

    # Remove NaN values
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred = pred[mask]
    actual = actual[mask]

    if len(pred) == 0:
        logger.warning("No valid predictions to evaluate")
        return {}

    # Calculate metrics
    errors = pred - actual
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / np.where(actual != 0, actual, 1)) * 100

    metrics = {
        # Basic errors
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE': np.mean(abs_errors),
        'MAPE': np.mean(pct_errors),
        'MedAPE': np.median(pct_errors),

        # Bias
        'Mean_Error': np.mean(errors),
        'Median_Error': np.median(errors),

        # Distribution
        'Std_Error': np.std(errors),
        'Max_Error': np.max(abs_errors),
        'Min_Error': np.min(abs_errors),

        # Percentiles
        'P90_Error': np.percentile(abs_errors, 90),
        'P95_Error': np.percentile(abs_errors, 95),

        # R-squared
        'R2': 1 - (np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2)),

        # Directional accuracy (sign match)
        'Direction_Accuracy': np.mean(np.sign(pred) == np.sign(actual)) * 100,

        # Sample size
        'N': len(pred),
    }

    return metrics


def create_features_table(selected_features: List[str], output_dir: Path = None) -> pd.DataFrame:
    """
    Create table of selected features with metadata.

    Args:
        selected_features: List of feature names selected after VIF/correlation/MI
        output_dir: Directory to save table

    Returns:
        DataFrame with feature information
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "backtest"

    # Categorize features
    feature_data = []

    for feat in selected_features:
        category = categorize_feature(feat)
        feature_data.append({
            'Feature': feat,
            'Category': category,
            'Index': len(feature_data) + 1
        })

    features_df = pd.DataFrame(feature_data)

    # Save to CSV
    features_path = output_dir / "selected_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info(f"✓ Saved selected features table: {features_path}")

    # Create summary by category
    summary = features_df['Category'].value_counts().reset_index()
    summary.columns = ['Category', 'Count']

    summary_path = output_dir / "features_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved features summary: {summary_path}")

    return features_df


def categorize_feature(feature_name: str) -> str:
    """
    Categorize feature by name pattern.

    Returns:
        Category string
    """
    feature_lower = feature_name.lower()

    if 'nfp' in feature_lower or 'target' in feature_lower:
        return 'NFP_Lagged'
    elif 'adp' in feature_lower:
        return 'ADP'
    elif 'jolts' in feature_lower:
        return 'JOLTS'
    elif 'claims' in feature_lower or 'icsa' in feature_lower or 'ccsa' in feature_lower:
        return 'Claims'
    elif 'noaa' in feature_lower:
        return 'NOAA'
    elif 'oil' in feature_lower:
        return 'Oil'
    elif 'credit' in feature_lower or 'spread' in feature_lower:
        return 'Credit'
    elif 'yield' in feature_lower:
        return 'Yield_Curve'
    elif 'ism' in feature_lower:
        return 'ISM'
    elif 'challenger' in feature_lower:
        return 'Challenger'
    elif 'consumer_confidence' in feature_lower or 'cb_' in feature_lower:
        return 'Consumer_Confidence'
    elif 'month' in feature_lower or 'quarter' in feature_lower or 'sin' in feature_lower or 'cos' in feature_lower:
        return 'Calendar'
    elif 'employment' in feature_lower or 'fred' in feature_lower:
        return 'Employment_Snapshot'
    else:
        return 'Other'


def create_metrics_table(metrics: Dict, output_dir: Path = None) -> pd.DataFrame:
    """
    Create formatted error metrics table.

    Args:
        metrics: Dictionary of error metrics
        output_dir: Directory to save table

    Returns:
        DataFrame with formatted metrics
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "backtest"

    # Format metrics for display
    formatted_metrics = []

    # Group 1: Primary Errors
    formatted_metrics.append({'Metric': 'Root Mean Squared Error (RMSE)', 'Value': f"{metrics.get('RMSE', np.nan):.2f}"})
    formatted_metrics.append({'Metric': 'Mean Absolute Error (MAE)', 'Value': f"{metrics.get('MAE', np.nan):.2f}"})
    formatted_metrics.append({'Metric': 'Mean Absolute Percentage Error (MAPE)', 'Value': f"{metrics.get('MAPE', np.nan):.2f}%"})
    formatted_metrics.append({'Metric': 'Median Absolute Percentage Error (MedAPE)', 'Value': f"{metrics.get('MedAPE', np.nan):.2f}%"})

    # Group 2: Bias
    formatted_metrics.append({'Metric': '---', 'Value': '---'})
    formatted_metrics.append({'Metric': 'Mean Error (Bias)', 'Value': f"{metrics.get('Mean_Error', np.nan):+.2f}"})
    formatted_metrics.append({'Metric': 'Median Error', 'Value': f"{metrics.get('Median_Error', np.nan):+.2f}"})

    # Group 3: Distribution
    formatted_metrics.append({'Metric': '---', 'Value': '---'})
    formatted_metrics.append({'Metric': 'Error Standard Deviation', 'Value': f"{metrics.get('Std_Error', np.nan):.2f}"})
    formatted_metrics.append({'Metric': 'Maximum Error', 'Value': f"{metrics.get('Max_Error', np.nan):.2f}"})
    formatted_metrics.append({'Metric': '90th Percentile Error', 'Value': f"{metrics.get('P90_Error', np.nan):.2f}"})
    formatted_metrics.append({'Metric': '95th Percentile Error', 'Value': f"{metrics.get('P95_Error', np.nan):.2f}"})

    # Group 4: Fit Quality
    formatted_metrics.append({'Metric': '---', 'Value': '---'})
    formatted_metrics.append({'Metric': 'R² Score', 'Value': f"{metrics.get('R2', np.nan):.4f}"})
    formatted_metrics.append({'Metric': 'Directional Accuracy', 'Value': f"{metrics.get('Direction_Accuracy', np.nan):.1f}%"})

    # Group 5: Sample Info
    formatted_metrics.append({'Metric': '---', 'Value': '---'})
    formatted_metrics.append({'Metric': 'Sample Size (N)', 'Value': f"{metrics.get('N', 0):.0f}"})

    metrics_df = pd.DataFrame(formatted_metrics)

    # Save to CSV
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Saved metrics table: {metrics_path}")

    # Also save raw metrics as JSON
    json_path = output_dir / "metrics_raw.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics_df


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    output_dir: Path = None,
    title: str = "NFP MoM Change: Predictions vs Actuals"
) -> Path:
    """
    Create shaded prediction graph with confidence intervals.

    Args:
        predictions_df: DataFrame with columns: date, pred, actual, [lower_bound, upper_bound]
        output_dir: Directory to save plot
        title: Plot title

    Returns:
        Path to saved plot
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available - skipping visualization")
        return None

    if output_dir is None:
        output_dir = OUTPUT_DIR / "backtest"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actual values
    ax.plot(predictions_df['date'], predictions_df['actual'],
            color='black', linewidth=2, label='Actual', marker='o', markersize=4)

    # Plot predictions
    ax.plot(predictions_df['date'], predictions_df['pred'],
            color='#2E86AB', linewidth=2, label='Predicted', marker='s', markersize=4, alpha=0.8)

    # Add confidence interval if available
    if 'lower_bound' in predictions_df.columns and 'upper_bound' in predictions_df.columns:
        ax.fill_between(predictions_df['date'],
                        predictions_df['lower_bound'],
                        predictions_df['upper_bound'],
                        color='#2E86AB', alpha=0.2, label='95% Confidence Interval')

    # Horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('MoM Change (thousands)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save
    plot_path = output_dir / "predictions_vs_actual.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved prediction plot: {plot_path}")
    return plot_path


def generate_backtest_report(
    predictions_df: pd.DataFrame,
    selected_features: List[str],
    output_dir: Path = None
) -> Dict:
    """
    Generate complete backtest report with all tables and visualizations.

    Args:
        predictions_df: DataFrame with predictions and actuals
        selected_features: List of selected feature names
        output_dir: Output directory

    Returns:
        Dictionary with paths to all generated files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "backtest"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("GENERATING BACKTEST REPORT")
    logger.info("="*60)

    report_files = {}

    # 1. Features table
    logger.info("\n1. Creating selected features table...")
    features_df = create_features_table(selected_features, output_dir)
    report_files['features_table'] = str(output_dir / "selected_features.csv")
    report_files['features_summary'] = str(output_dir / "features_summary.csv")

    # 2. Error metrics
    logger.info("\n2. Calculating error metrics...")
    metrics = calculate_error_metrics(predictions_df)
    metrics_df = create_metrics_table(metrics, output_dir)
    report_files['metrics_table'] = str(output_dir / "metrics_summary.csv")
    report_files['metrics_json'] = str(output_dir / "metrics_raw.json")

    # 3. Visualization
    logger.info("\n3. Creating prediction visualization...")
    plot_path = plot_predictions_vs_actual(predictions_df, output_dir)
    if plot_path:
        report_files['prediction_plot'] = str(plot_path)

    # 4. Save predictions
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    report_files['predictions'] = str(predictions_path)
    logger.info(f"✓ Saved predictions: {predictions_path}")

    # 5. Summary report
    summary = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'num_predictions': len(predictions_df),
        'num_features': len(selected_features),
        'metrics': metrics,
        'files': report_files
    }

    summary_path = output_dir / "report_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    report_files['summary'] = str(summary_path)

    logger.info("\n" + "="*60)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nGenerated {len(report_files)} files in: {output_dir}")
    logger.info("\nKey metrics:")
    logger.info(f"  RMSE: {metrics.get('RMSE', np.nan):.2f}")
    logger.info(f"  MAE: {metrics.get('MAE', np.nan):.2f}")
    logger.info(f"  MAPE: {metrics.get('MAPE', np.nan):.2f}%")
    logger.info(f"  R²: {metrics.get('R2', np.nan):.4f}")
    logger.info(f"  Direction Accuracy: {metrics.get('Direction_Accuracy', np.nan):.1f}%")

    return report_files


if __name__ == "__main__":
    # Test with dummy data
    logger.info("Testing backtest results module...")

    # Create dummy predictions
    dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
    np.random.seed(42)
    dummy_df = pd.DataFrame({
        'date': dates,
        'pred': np.random.randn(len(dates)) * 100 + 200,
        'actual': np.random.randn(len(dates)) * 100 + 200,
        'lower_bound': np.random.randn(len(dates)) * 100 + 150,
        'upper_bound': np.random.randn(len(dates)) * 100 + 250
    })

    # Dummy features
    dummy_features = [
        'ADP_actual', 'ICSA_monthly_avg', 'JOLTS_Layoffs',
        'Oil_Prices_mean', 'Credit_Spreads_avg',
        'NFP_lag1', 'NFP_lag2', 'month_sin', 'month_cos'
    ]

    # Generate report
    files = generate_backtest_report(dummy_df, dummy_features)

    logger.info("\n✓ Test complete!")
    logger.info(f"Generated files: {list(files.keys())}")
