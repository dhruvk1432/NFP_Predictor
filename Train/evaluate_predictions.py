"""
Evaluate NFP Forecast Predictions

Computes comprehensive metrics and generates visualizations for backtest results.

Metrics:
- Directional Accuracy: % correct sign of MoM change
- R² for MoM changes
- MAE, RMSE for MoM changes (in thousands)  
- Correlation with actual
- Bias analysis

Visualizations:
- Predicted vs Actual time series
- Scatter plot (predicted vs actual)
- Error distribution
- Metrics table

Usage:
    python evaluate_predictions.py  # Uses latest backtest results
    python evaluate_predictions.py --results path/to/results.parquet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """
    Compute comprehensive forecast metrics.
    
    Args:
        results_df: DataFrame with columns: sa_mom_pred, sa_mom_actual
        
    Returns:
        Dictionary of metrics
    """
    # Filter out NaNs
    valid = results_df.dropna(subset=['sa_mom_pred', 'sa_mom_actual'])
    
    if len(valid) == 0:
        logger.error("No valid predictions to evaluate!")
        return {}
    
    pred = valid['sa_mom_pred'].values
    actual = valid['sa_mom_actual'].values
    
    # 1. Directional Accuracy
    pred_sign = np.sign(pred)
    actual_sign = np.sign(actual)
    directional_accuracy = (pred_sign == actual_sign).mean() * 100
    
    # 2. R² (coefficient of determination)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 3. MAE, RMSE (in thousands)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    
    # 4. Correlation
    correlation = np.corrcoef(pred, actual)[0, 1]
    
    # 5. Bias (mean error)
    bias = np.mean(pred - actual)
    
    # 6. Hit rates by magnitude
    small_changes = np.abs(actual) < 100  # Small MoM changes (< 100k)
    large_changes = np.abs(actual) >= 100  # Large MoM changes (>= 100k)
    
    if small_changes.sum() > 0:
        small_directional_acc = (pred_sign[small_changes] == actual_sign[small_changes]).mean() * 100
    else:
        small_directional_acc = np.nan
        
    if large_changes.sum() > 0:
        large_directional_acc = (pred_sign[large_changes] == actual_sign[large_changes]).mean() * 100
    else:
        large_directional_acc = np.nan
    
    # 7. Mean Absolute Percentage Error (MAPE) - careful with zeros
    non_zero = actual != 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((actual[non_zero] - pred[non_zero]) / actual[non_zero])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'directional_accuracy': directional_accuracy,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'bias': bias,
        'mape': mape,
        'small_changes_directional_acc': small_directional_acc,
        'large_changes_directional_acc': large_directional_acc,
        'n_predictions': len(valid),
        'n_small_changes': small_changes.sum(),
        'n_large_changes': large_changes.sum()
    }
    
    return metrics


def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive visualizations.
    
    Args:
        results_df: Backtest results DataFrame
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid data
    valid = results_df.dropna(subset=['sa_mom_pred', 'sa_mom_actual']).copy()
    valid['error'] = valid['sa_mom_pred'] - valid['sa_mom_actual']
    valid['abs_error'] = np.abs(valid['error'])
    
    # Figure 1: Time Series - Predicted vs Actual MoM
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(valid['target_date'], valid['sa_mom_actual'], 
            marker='o', linewidth=2, markersize=5, label='Actual (First Release)', 
            color='#2E7D32', alpha=0.8)
    ax.plot(valid['target_date'], valid['sa_mom_pred'], 
            marker='s', linewidth=2, markersize=4, label='Predicted', 
            color='#1976D2', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Forecast Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('SA MoM Change (thousands)', fontsize=12, fontweight='bold')
    ax.set_title('NFP SA MoM Change: Predicted vs Actual (First Release)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'timeseries_mom_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(valid['sa_mom_actual'], valid['sa_mom_pred'], 
              alpha=0.6, s=80, color='#1976D2', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    lims = [
        min(valid['sa_mom_actual'].min(), valid['sa_mom_pred'].min()) - 50,
        max(valid['sa_mom_actual'].max(), valid['sa_mom_pred'].max()) + 50
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(valid['sa_mom_actual'], valid['sa_mom_pred'], 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), "r-", alpha=0.6, linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.0f}')
    
    ax.set_xlabel('Actual SA MoM Change (thousands)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted SA MoM Change (thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs Actual: Scatter Plot', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'scatter_mom_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Error Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(valid['error'], bins=20, color='#1976D2', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=valid['error'].mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean Error: {valid["error"].mean():.1f}')
    ax1.set_xlabel('Prediction Error (thousands)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Error over time
    ax2.plot(valid['target_date'], valid['error'], marker='o', linewidth=1.5, 
            markersize=4, color='#D32F2F', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(valid['target_date'], 0, valid['error'], 
                     where=(valid['error'] > 0), alpha=0.3, color='green', label='Over-prediction')
    ax2.fill_between(valid['target_date'], 0, valid['error'], 
                     where=(valid['error'] < 0), alpha=0.3, color='red', label='Under-prediction')
    ax2.set_xlabel('Forecast Month', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Prediction Error (thousands)', fontsize=11, fontweight='bold')
    ax2.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Directional Accuracy Over Time (Rolling)
    if len(valid) >= 12:
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Calculate rolling directional accuracy
        window = 12  # 12-month rolling window
        rolling_acc = []
        dates = []
        
        for i in range(window, len(valid)+1):
            window_data = valid.iloc[i-window:i]
            pred_sign = np.sign(window_data['sa_mom_pred'])
            actual_sign = np.sign(window_data['sa_mom_actual'])
            acc = (pred_sign == actual_sign).mean() * 100
            rolling_acc.append(acc)
            dates.append(window_data['target_date'].iloc[-1])
        
        ax.plot(dates, rolling_acc, linewidth=2.5, color='#1976D2', marker='o', markersize=4)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, 
                  label='Random (50%)', alpha=0.7)
        ax.axhline(y=np.mean(rolling_acc), color='green', linestyle='--', linewidth=1.5,
                  label=f'Average: {np.mean(rolling_acc):.1f}%', alpha=0.7)
        ax.set_xlabel('Forecast Month', fontsize=11, fontweight='bold')
        ax.set_ylabel('Directional Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Rolling {window}-Month Directional Accuracy', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        fig.savefig(output_dir / 'rolling_directional_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def create_metrics_table(metrics: dict, output_dir: Path):
    """Create a formatted metrics table and save as image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Format metrics for display
    table_data = [
        ['Metric', 'Value'],
        ['Directional Accuracy', f"{metrics['directional_accuracy']:.1f}%"],
        ['R² Score', f"{metrics['r_squared']:.3f}"],
        ['MAE (thousands)', f"{metrics['mae']:.1f}"],
        ['RMSE (thousands)', f"{metrics['rmse']:.1f}"],
        ['Correlation', f"{metrics['correlation']:.3f}"],
        ['Bias (thousands)', f"{metrics['bias']:.1f}"],
        ['MAPE', f"{metrics['mape']:.1f}%" if not np.isnan(metrics['mape']) else 'N/A'],
        ['', ''],
        ['Small Changes (<100k) Dir. Acc.', 
         f"{metrics['small_changes_directional_acc']:.1f}%" if not np.isnan(metrics['small_changes_directional_acc']) else 'N/A'],
        ['Large Changes (≥100k) Dir. Acc.', 
         f"{metrics['large_changes_directional_acc']:.1f}%" if not np.isnan(metrics['large_changes_directional_acc']) else 'N/A'],
        ['', ''],
        ['Total Predictions', f"{metrics['n_predictions']}"],
        ['Small Changes Count', f"{metrics['n_small_changes']}"],
        ['Large Changes Count', f"{metrics['n_large_changes']}"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#1976D2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style alternating rows
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            # Highlight section breaks
            if table_data[i][0] == '':
                table[(i, j)].set_facecolor('#E0E0E0')
    
    plt.title('Forecast Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics table to {output_dir}")


def evaluate_backtest(results_path: Path = None):
    """
    Main evaluation function.
    
    Args:
        results_path: Path to backtest results parquet file
    """
    # Load results
    if results_path is None:
        results_path = OUTPUT_DIR / "backtest" / "backtest_results.parquet"
    
    logger.info(f"Loading results from {results_path}")
    results_df = pd.read_parquet(results_path)
    
    logger.info(f"Loaded {len(results_df)} predictions")
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = compute_metrics(results_df)
    
    # Print metrics
    print("\n" + "="*70)
    print("FORECAST PERFORMANCE METRICS")
    print("="*70)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key or 'r_squared' in key or 'correlation' in key:
                print(f"{key:40s}: {value:.2f}")
            else:
                print(f"{key:40s}: {value:.1f}")
        else:
            print(f"{key:40s}: {value}")
    print("="*70)
    
    # Create visualizations
    output_dir = OUTPUT_DIR / "backtest" / "evaluation"
    logger.info(f"\nGenerating visualizations...")
    create_visualizations(results_df, output_dir)
    create_metrics_table(metrics, output_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    logger.info(f"Saved metrics to {output_dir / 'metrics.csv'}")
    
    return metrics, results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate NFP forecast predictions')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to backtest results parquet file')
    
    args = parser.parse_args()
    
    results_path = Path(args.results) if args.results else None
    metrics, results_df = evaluate_backtest(results_path)
