"""
Generate Backtest Visualizations

Creates prediction vs actual graphs with confidence intervals for both NSA and SA models.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from settings import OUTPUT_DIR, setup_logger, TEMP_DIR

logger = setup_logger(__file__, TEMP_DIR)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.error("matplotlib not available. Install with: pip install matplotlib")
    PLOTTING_AVAILABLE = False
    sys.exit(1)


def create_backtest_visualization(
    backtest_csv_path: Path,
    output_dir: Path,
    model_name: str,
    title: str
):
    """
    Create prediction vs actual visualization with shaded confidence intervals.

    Args:
        backtest_csv_path: Path to backtest_results CSV
        output_dir: Directory to save plot
        model_name: Name of model (nsa or sa)
        title: Plot title
    """
    logger.info(f"Creating visualization for {model_name} model...")

    # Load backtest results
    df = pd.read_csv(backtest_csv_path)
    df['ds'] = pd.to_datetime(df['ds'])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual values
    ax.plot(df['ds'], df['actual'],
            color='black', linewidth=2.5, label='Actual NFP',
            marker='o', markersize=6, zorder=3)

    # Plot predictions
    ax.plot(df['ds'], df['predicted'],
            color='#2E86AB', linewidth=2.5, label='Predicted NFP',
            marker='s', markersize=6, alpha=0.9, zorder=2)

    # Add 95% confidence interval (shaded)
    if 'lower_95' in df.columns and 'upper_95' in df.columns:
        ax.fill_between(df['ds'],
                        df['lower_95'],
                        df['upper_95'],
                        color='#2E86AB', alpha=0.15,
                        label='95% Confidence Interval', zorder=1)

    # Add 80% confidence interval (slightly darker)
    if 'lower_80' in df.columns and 'upper_80' in df.columns:
        ax.fill_between(df['ds'],
                        df['lower_80'],
                        df['upper_80'],
                        color='#2E86AB', alpha=0.25,
                        label='80% Confidence Interval', zorder=1)

    # Add 50% confidence interval (darkest)
    if 'lower_50' in df.columns and 'upper_50' in df.columns:
        ax.fill_between(df['ds'],
                        df['lower_50'],
                        df['upper_50'],
                        color='#2E86AB', alpha=0.35,
                        label='50% Confidence Interval', zorder=1)

    # Horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    # Styling
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('NFP MoM Change (thousands)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, shadow=True)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save high-resolution plot
    plot_path = output_dir / f"predictions_vs_actual_{model_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"✓ Saved visualization: {plot_path}")

    # Also create error distribution plot
    create_error_distribution_plot(df, output_dir, model_name)

    return plot_path


def create_error_distribution_plot(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Create histogram of prediction errors."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate errors
    errors = df['error']

    # Plot 1: Error histogram
    ax1.hist(errors, bins=15, color='#A23B72', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error: {errors.mean():.1f}K')
    ax1.set_xlabel('Prediction Error (thousands)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error over time
    ax2.scatter(df['ds'], errors, alpha=0.6, s=80, c=abs(errors),
                cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Error (thousands)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    error_plot_path = output_dir / f"error_analysis_{model_name}.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"✓ Saved error analysis: {error_plot_path}")


def generate_all_visualizations():
    """Generate visualizations for all models."""

    logger.info("="*60)
    logger.info("GENERATING BACKTEST VISUALIZATIONS")
    logger.info("="*60)

    models = [
        {
            'name': 'nsa',
            'csv_path': OUTPUT_DIR / "backtest_results" / "nsa" / "backtest_results_nsa.csv",
            'output_dir': OUTPUT_DIR / "backtest_results" / "nsa",
            'title': 'NSA Model: NFP MoM Change Predictions vs Actuals'
        },
        {
            'name': 'sa',
            'csv_path': OUTPUT_DIR / "backtest_results" / "sa" / "backtest_results_sa.csv",
            'output_dir': OUTPUT_DIR / "backtest_results" / "sa",
            'title': 'SA Model: NFP MoM Change Predictions vs Actuals'
        }
    ]

    generated_files = []

    for model in models:
        if not model['csv_path'].exists():
            logger.warning(f"Backtest CSV not found: {model['csv_path']}")
            continue

        logger.info(f"\nGenerating visualizations for {model['name'].upper()} model...")

        try:
            plot_path = create_backtest_visualization(
                backtest_csv_path=model['csv_path'],
                output_dir=model['output_dir'],
                model_name=model['name'],
                title=model['title']
            )
            generated_files.append(plot_path)

        except Exception as e:
            logger.error(f"Failed to generate visualization for {model['name']}: {e}")
            continue

    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nGenerated {len(generated_files)} visualization files:")
    for f in generated_files:
        logger.info(f"  - {f}")

    return generated_files


if __name__ == "__main__":
    if not PLOTTING_AVAILABLE:
        logger.error("Cannot generate visualizations without matplotlib")
        sys.exit(1)

    generate_all_visualizations()
