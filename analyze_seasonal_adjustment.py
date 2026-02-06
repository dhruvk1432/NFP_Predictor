"""
Seasonal Adjustment Analysis and SARIMA Prediction

This script:
1. Calculates the seasonal adjustment (SA - NSA) from target data
2. Compares SA actual vs NSA + seasonal adjustment
3. Fits a SARIMA model to predict future seasonal adjustments
4. Creates visualization of NSA predictions + adjustments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load target data and calculate seasonal adjustment
# =============================================================================

print("Loading target data...")
data_nsa = pd.read_parquet("data/NFP_target/total_nsa_first_release.parquet")
data_sa = pd.read_parquet("data/NFP_target/total_sa_first_release.parquet")

# Calculate seasonal adjustment (SA - NSA)
# This represents the adjustment applied to convert NSA to SA
adjustment = data_sa["y"] - data_nsa["y"]
adjustment.index = data_nsa["ds"]
adjustment.name = "seasonal_adjustment"

# Also calculate MoM change adjustment
adjustment_mom = data_sa["y_mom"] - data_nsa["y_mom"]
adjustment_mom.index = data_nsa["ds"]
adjustment_mom.name = "seasonal_adjustment_mom"

print(f"Seasonal adjustment calculated for {len(adjustment)} months")
print(f"Date range: {adjustment.index.min()} to {adjustment.index.max()}")

# =============================================================================
# 2. Load prediction data
# =============================================================================

print("\nLoading prediction data...")
nsa_predictions = pd.read_csv("_output/backtest_results/nsa_first/backtest_results_nsa_first.csv")
sa_predictions = pd.read_csv("_output/backtest_results/sa_first/predictions.csv")

# Rename column for consistency (sa_first uses 'date', nsa_first uses 'ds')
if 'date' in sa_predictions.columns:
    sa_predictions = sa_predictions.rename(columns={'date': 'ds'})
if 'pred' in sa_predictions.columns:
    sa_predictions = sa_predictions.rename(columns={'pred': 'predicted'})

# Convert dates
nsa_predictions['ds'] = pd.to_datetime(nsa_predictions['ds'])
sa_predictions['ds'] = pd.to_datetime(sa_predictions['ds'])

print(f"NSA predictions: {len(nsa_predictions)} months")
print(f"SA predictions: {len(sa_predictions)} months")

# =============================================================================
# 3. Compare SA actual vs NSA + seasonal adjustment
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: SA Actual vs NSA + Seasonal Adjustment")
print("=" * 70)

# Merge predictions with adjustment data
comparison = nsa_predictions[['ds', 'actual', 'predicted']].copy()
comparison = comparison.rename(columns={'actual': 'nsa_actual', 'predicted': 'nsa_predicted'})

# Get SA actual for comparison
sa_actual = sa_predictions[['ds', 'actual']].copy()
sa_actual = sa_actual.rename(columns={'actual': 'sa_actual'})
comparison = comparison.merge(sa_actual, on='ds', how='left')

# Get the seasonal adjustment (MoM change version) for each prediction date
# We need to align by date
adjustment_df = adjustment_mom.reset_index()
adjustment_df.columns = ['ds', 'adj_mom']
comparison = comparison.merge(adjustment_df, on='ds', how='left')

# Calculate NSA + adjustment (should approximate SA)
comparison['nsa_plus_adj'] = comparison['nsa_actual'] + comparison['adj_mom']

# Show comparison
print("\nSide-by-side comparison (MoM changes):")
print("-" * 70)
display_cols = ['ds', 'sa_actual', 'nsa_actual', 'adj_mom', 'nsa_plus_adj']
print(comparison[display_cols].to_string(index=False))

# Verify: nsa_plus_adj should equal sa_actual
comparison['diff'] = comparison['sa_actual'] - comparison['nsa_plus_adj']
print(f"\nValidation: SA_actual - (NSA_actual + adjustment) should be ~0:")
print(f"Mean diff: {comparison['diff'].mean():.4f}")
print(f"Max diff: {comparison['diff'].abs().max():.4f}")

# =============================================================================
# 4. Fit SARIMA model on seasonal adjustment
# =============================================================================

print("\n" + "=" * 70)
print("SARIMA MODEL FOR SEASONAL ADJUSTMENT PREDICTION")
print("=" * 70)

# Prepare adjustment series for SARIMA
adj_series = adjustment_mom.dropna()
adj_series.index = pd.to_datetime(adj_series.index)
adj_series = adj_series.asfreq('MS')  # Monthly start frequency

print(f"\nTraining SARIMA on {len(adj_series)} months of seasonal adjustment data")

# SARIMA with 12-month seasonality
# Order (p,d,q) x (P,D,Q,s)
# Using (1,0,1) x (1,1,1,12) for seasonal pattern
try:
    model = SARIMAX(
        adj_series,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_result = model.fit(disp=False)
    print("\nSARIMA Model Summary:")
    print(f"AIC: {sarima_result.aic:.2f}")
    print(f"BIC: {sarima_result.bic:.2f}")

    # Predict next 12 months
    last_date = adj_series.index.max()
    forecast_steps = 12
    forecast = sarima_result.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    print(f"\nForecast for next {forecast_steps} months:")
    forecast_df = pd.DataFrame({
        'date': forecast_mean.index,
        'predicted_adjustment': forecast_mean.values,
        'lower_95': forecast_ci.iloc[:, 0].values,
        'upper_95': forecast_ci.iloc[:, 1].values
    })
    print(forecast_df.to_string(index=False))

except Exception as e:
    print(f"SARIMA fitting error: {e}")
    print("Trying simpler model...")

    # Fallback to simpler seasonal model
    model = SARIMAX(
        adj_series,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False
    )
    sarima_result = model.fit(disp=False)
    forecast = sarima_result.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

# =============================================================================
# 5. Create visualization
# =============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# --- Plot 1: Historical Seasonal Adjustment Pattern ---
ax1 = axes[0]
adj_series_plot = adjustment_mom.dropna()
ax1.plot(adj_series_plot.index, adj_series_plot.values, 'b-', linewidth=1, alpha=0.7, label='Historical Adjustment')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Historical Seasonal Adjustment (SA - NSA MoM Change)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Adjustment (thousands)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: NSA Predictions + Actual Adjustments vs SA Actual ---
ax2 = axes[1]

# Filter to prediction period only
pred_dates = nsa_predictions['ds']
pred_nsa = nsa_predictions.set_index('ds')['predicted']
pred_sa = sa_predictions.set_index('ds')['predicted']

# Get actual adjustments for prediction dates
adj_for_pred = adjustment_mom.loc[adjustment_mom.index.isin(pred_dates)]

# NSA predicted + actual adjustment
nsa_plus_actual_adj = pred_nsa.loc[pred_nsa.index.isin(adj_for_pred.index)] + adj_for_pred

ax2.plot(pred_nsa.index, pred_nsa.values, 'b-o', markersize=4, label='NSA Predicted', alpha=0.7)
ax2.plot(pred_sa.index, pred_sa.values, 'g-s', markersize=4, label='SA Predicted', alpha=0.7)
ax2.plot(nsa_plus_actual_adj.index, nsa_plus_actual_adj.values, 'r-^', markersize=4,
         label='NSA Predicted + Actual Adj', alpha=0.7)

# Add actuals if available
nsa_actual_plot = nsa_predictions[nsa_predictions['actual'].notna()].set_index('ds')['actual']
sa_actual_plot = sa_predictions[sa_predictions['actual'].notna()].set_index('ds')['actual']

if not nsa_actual_plot.empty:
    ax2.scatter(nsa_actual_plot.index, nsa_actual_plot.values, color='blue', s=50,
                marker='x', label='NSA Actual', zorder=5)
if not sa_actual_plot.empty:
    ax2.scatter(sa_actual_plot.index, sa_actual_plot.values, color='green', s=50,
                marker='x', label='SA Actual', zorder=5)

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Prediction Period: NSA + Adjustments vs SA', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('MoM Change (thousands)')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Plot 3: NSA Predicted + SARIMA Predicted Adjustment ---
ax3 = axes[2]

# For historical predictions (where we have actual adjustment)
historical_mask = pred_nsa.index <= adj_series.index.max()
future_mask = pred_nsa.index > adj_series.index.max()

# Historical: NSA + actual adjustment
hist_dates = pred_nsa.index[historical_mask]
hist_nsa = pred_nsa[historical_mask]
hist_adj = adjustment_mom.reindex(hist_dates).fillna(0)
hist_combined = hist_nsa + hist_adj

# Future: NSA + SARIMA predicted adjustment
future_dates = pred_nsa.index[future_mask]
if len(future_dates) > 0:
    future_nsa = pred_nsa[future_mask]
    # Get SARIMA forecast for those dates
    future_adj = forecast_mean.reindex(future_dates).fillna(forecast_mean.mean())
    future_combined = future_nsa + future_adj
else:
    # If all predictions are in "historical" period, use in-sample fitted values
    # and extend to show forecast
    future_combined = pd.Series(dtype=float)

# Plot historical (NSA + actual adjustment)
ax3.plot(hist_dates, hist_combined.values, 'b-o', markersize=5,
         label='NSA Pred + Actual Adjustment', linewidth=2)

# Plot future (NSA + SARIMA predicted adjustment)
if len(future_dates) > 0:
    ax3.plot(future_dates, future_combined.values, 'r-^', markersize=5,
             label='NSA Pred + SARIMA Adjustment', linewidth=2)

# Also plot the SA predictions for reference
ax3.plot(pred_sa.index, pred_sa.values, 'g--s', markersize=4,
         label='SA Predicted (model)', alpha=0.7)

# Plot SA actuals
if not sa_actual_plot.empty:
    ax3.scatter(sa_actual_plot.index, sa_actual_plot.values, color='darkgreen', s=80,
                marker='*', label='SA Actual', zorder=5)

ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Combined Forecast: NSA Predictions + Seasonal Adjustment', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('MoM Change (thousands)')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('_output/seasonal_adjustment_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: _output/seasonal_adjustment_analysis.png")
plt.show()

# =============================================================================
# 6. Summary Table
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: PREDICTED VALUES WITH ADJUSTMENTS")
print("=" * 70)

summary = pd.DataFrame({
    'Date': pred_nsa.index,
    'NSA_Predicted': pred_nsa.values,
    'Actual_Adj': adjustment_mom.reindex(pred_nsa.index).values,
    'SARIMA_Adj': forecast_mean.reindex(pred_nsa.index).values if len(forecast_mean) > 0 else np.nan,
})

# Use actual adjustment where available, SARIMA where not
summary['Best_Adj'] = summary['Actual_Adj'].fillna(summary['SARIMA_Adj'])
summary['NSA_Plus_BestAdj'] = summary['NSA_Predicted'] + summary['Best_Adj']
summary['SA_Predicted'] = pred_sa.values
summary['SA_Actual'] = sa_actual_plot.reindex(pred_nsa.index).values if not sa_actual_plot.empty else np.nan

print(summary.to_string(index=False))

# Save summary to CSV
summary.to_csv('_output/seasonal_adjustment_summary.csv', index=False)
print("\nSummary saved to: _output/seasonal_adjustment_summary.csv")
