# NFP Forecasting Backtest System

## Overview

Comprehensive backtesting system for NFP month-over-month change predictions with historical archiving and detailed results presentation.

## Features

### 1. **Historical Archiving** (`backtest_archiver.py`)
- Automatically archives previous backtest results before new runs
- Creates timestamped snapshots: `backtest_historical/YYYY-MM-DD_HH-MM-SS/`
- Maintains clean separation between runs
- Preserves full history for model improvement tracking

### 2. **Comprehensive Reporting** (`backtest_results.py`)
- **Selected Features Table**: Lists all features used after VIF/correlation/MI selection
- **Error Metrics Table**: RMSE, MAE, MAPE, R², directional accuracy, etc.
- **Prediction Visualization**: Shaded graph with actual vs predicted and confidence intervals
- **Summary Statistics**: Complete performance breakdown

### 3. **Configuration** (`.env`)
```bash
BACKTEST_MONTHS=36  # Number of months to backtest (default: 36)
```

## Usage

### Run Backtest with Default Settings (36 months from .env)
```bash
python Train/run_expanding_backtest.py
```

This will:
1. Archive any existing results to `_output/backtest_historical/YYYY-MM-DD_HH-MM-SS/`
2. Clean `_output/backtest/` directory
3. Run backtest for last 36 months (as specified in .env)
4. Generate comprehensive report with:
   - `selected_features.csv` - Features used after selection
   - `features_summary.csv` - Feature counts by category
   - `metrics_summary.csv` - Formatted error metrics table
   - `metrics_raw.json` - Raw metrics dictionary
   - `predictions.csv` - All predictions and actuals
   - `predictions_vs_actual.png` - Shaded visualization
   - `report_summary.json` - Complete report metadata

### Custom Backtest Period
```bash
# Backtest last 24 months instead of 36
python Train/run_expanding_backtest.py --months 24

# Full history backtest
python Train/run_expanding_backtest.py --months 999
```

### Skip Archiving
```bash
# Don't archive previous results (for testing)
python Train/run_expanding_backtest.py --skip-archive
```

## Output Structure

```
_output/
├── backtest/                          # Current backtest results
│   ├── predictions.csv                # All predictions with actuals
│   ├── selected_features.csv          # Features after selection
│   ├── features_summary.csv           # Feature category counts
│   ├── metrics_summary.csv            # Formatted error metrics
│   ├── metrics_raw.json               # Raw metrics dictionary
│   ├── predictions_vs_actual.png      # Shaded prediction graph
│   ├── report_summary.json            # Report metadata
│   └── models/                        # Saved models (optional)
│
└── backtest_historical/               # Historical backtest archives
    ├── 2025-01-15_10-30-45/          # Archive from Jan 15, 2025 10:30:45
    │   └── [same structure as backtest/]
    ├── 2025-01-14_14-22-10/          # Archive from Jan 14, 2025 14:22:10
    │   └── [same structure as backtest/]
    └── ...
```

## Report Contents

### 1. Selected Features Table (`selected_features.csv`)
```csv
Feature,Category,Index
ADP_actual,ADP,1
ICSA_monthly_avg,Claims,2
JOLTS_Layoffs,JOLTS,3
...
```

Categories:
- NFP_Lagged: Historical NFP values as features
- ADP: ADP employment data
- JOLTS: Job openings and labor turnover
- Claims: Initial/continuing unemployment claims
- NOAA: Weather/disaster impacts
- Oil: Oil prices and volatility
- Credit: Credit spreads
- Yield_Curve: Treasury yield curve
- ISM: Manufacturing/services indices
- Challenger: Job cut announcements
- Consumer_Confidence: Consumer confidence index
- Calendar: Month/quarter cyclical features
- Employment_Snapshot: Employment subsector data
- Other: Miscellaneous features

### 2. Error Metrics Table (`metrics_summary.csv`)
```csv
Metric,Value
Root Mean Squared Error (RMSE),45.23
Mean Absolute Error (MAE),35.12
Mean Absolute Percentage Error (MAPE),12.34%
Median Absolute Percentage Error (MedAPE),10.45%
---,---
Mean Error (Bias),+2.15
Median Error,+1.80
---,---
Error Standard Deviation,42.10
Maximum Error,156.78
90th Percentile Error,68.90
95th Percentile Error,89.12
---,---
R² Score,0.6245
Directional Accuracy,75.5%
---,---
Sample Size (N),36
```

### 3. Prediction Visualization (`predictions_vs_actual.png`)
- Line plot with actual values (black) and predictions (blue)
- Shaded 95% confidence interval
- Horizontal line at zero for reference
- Grid and proper labeling

### 4. Predictions Data (`predictions.csv`)
```csv
date,pred,actual,lower_bound,upper_bound
2022-10-31,205.3,195.0,185.0,225.0
2022-11-30,180.5,190.0,160.5,200.5
...
```

## Tracking Model Improvements

### Compare with Previous Run
```python
from Train.backtest_archiver import compare_backtest_performance, list_historical_backtests

# List all historical runs
archives = list_historical_backtests()
print(archives)

# Compare current with most recent historical
comparison = compare_backtest_performance(current_metrics)
print(comparison)
```

### View Historical Archive
```bash
ls _output/backtest_historical/
# Output: 2025-01-15_10-30-45  2025-01-14_14-22-10  ...

# View specific archive
cat _output/backtest_historical/2025-01-15_10-30-45/metrics_summary.csv
```

## Feature Selection Process

The backtest uses multi-stage feature selection:

1. **Target Correlation Filter**: Remove features with |correlation| < 0.05
2. **High Correlation Removal**: Drop one of each pair with correlation > 0.95
3. **VIF Removal (Iterative)**: Remove features with VIF > 10.0 one at a time
4. **Final Selection**: Keep top MAX_FEATURES (80) by aggregated ranking

All selected features are documented in `selected_features.csv`.

## Error Metrics Explained

- **RMSE**: Root Mean Squared Error - penalizes large errors more
- **MAE**: Mean Absolute Error - average magnitude of errors
- **MAPE**: Mean Absolute Percentage Error - percentage terms
- **MedAPE**: Median Absolute Percentage Error - robust to outliers
- **Mean Error**: Average bias (positive = over-prediction)
- **R²**: Coefficient of determination (0-1, higher is better)
- **Directional Accuracy**: % of predictions with correct sign

## Best Practices

1. **Always Archive**: Let the system archive previous results automatically
2. **Track Changes**: Compare metrics across runs to measure improvements
3. **Review Features**: Check `selected_features.csv` to understand what drove predictions
4. **Validate Visually**: Always review `predictions_vs_actual.png` for patterns
5. **Monitor Bias**: Watch Mean Error for systematic over/under-prediction

## Troubleshooting

### Archive Failed
- Check write permissions on `_output/` directory
- Ensure enough disk space

### Plot Not Generated
- Install matplotlib/seaborn: `pip install matplotlib seaborn`

### Metrics Look Wrong
- Verify predictions data has 'pred' and 'actual' columns
- Check for NaN values in predictions

## Integration with Existing Code

The backtest system integrates seamlessly with:
- `train_lightgbm_nfp.py`: Feature selection and model training
- `forecast_pipeline.py`: NFP prediction generation
- `sa_mapper.py`: NSA to SA conversion
- `settings.py`: Configuration (BACKTEST_MONTHS)

## Future Enhancements

Potential additions:
- Automated comparison reports between runs
- Feature importance tracking over time
- Error distribution visualizations
- Subsector prediction accuracy breakdown
- Confidence interval calibration metrics
