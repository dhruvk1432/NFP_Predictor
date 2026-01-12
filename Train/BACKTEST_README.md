# NFP Forecasting Backtest System

## Overview

Comprehensive backtesting system for NFP month-over-month change predictions with historical archiving and detailed results presentation.

## Multi-Target Support

The system supports 4 target configurations (model variants):

| Model ID | Target Type | Release Type | Description |
|----------|-------------|--------------|-------------|
| `nsa_first` | NSA | First | Non-seasonally adjusted, initial release |
| `nsa_last` | NSA | Last | Non-seasonally adjusted, final revised |
| `sa_first` | SA | First | Seasonally adjusted, initial release |
| `sa_last` | SA | Last | Seasonally adjusted, final revised |

Each model variant has its own output directories for backtest results, feature importance, and saved models.

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

### Train All 4 Model Variants (Recommended)
```bash
python Train/train_lightgbm_nfp.py --train-all
```

This will:
1. Archive any existing results to `_output/backtest_historical/YYYY-MM-DD_HH-MM-SS/`
2. Train and backtest all 4 model variants in sequence
3. Generate comprehensive reports for each variant

### Train Single Model
```bash
# Train NSA first release (default)
python Train/train_lightgbm_nfp.py --train

# Train SA last release
python Train/train_lightgbm_nfp.py --train --target sa --release last
```

### Train Subsets
```bash
# Train both NSA and SA (same release type)
python Train/train_lightgbm_nfp.py --train-both --release first

# Train both release types (same target type)
python Train/train_lightgbm_nfp.py --train-both-releases --target nsa
```

### Custom Backtest Period
Configure in `.env`:
```bash
BACKTEST_MONTHS=24  # Backtest last 24 months instead of 36
```

## Output Structure

```
_output/
├── backtest_results/                  # Current backtest results (by model)
│   ├── nsa_first/
│   │   ├── backtest_results_nsa_first.parquet
│   │   ├── backtest_results_nsa_first.csv
│   │   ├── model_summary_nsa_first.csv
│   │   ├── predictions.csv
│   │   ├── selected_features.csv
│   │   ├── features_summary.csv
│   │   ├── metrics_summary.csv
│   │   ├── metrics_raw.json
│   │   ├── predictions_vs_actual.png
│   │   └── report_summary.json
│   ├── nsa_last/
│   │   └── [same structure]
│   ├── sa_first/
│   │   └── [same structure]
│   └── sa_last/
│       └── [same structure]
│
├── feature_importance/                # Feature importance (by model)
│   ├── nsa_first/
│   │   └── feature_importance_nsa_first.csv
│   ├── nsa_last/
│   ├── sa_first/
│   └── sa_last/
│
├── feature_selection/                 # Feature selection results (by model)
│   ├── nsa_first/
│   │   ├── feature_ranking_nsa.csv
│   │   ├── selected_features_nsa.csv
│   │   └── feature_selection_metadata_nsa.pkl
│   ├── nsa_last/
│   ├── sa_first/
│   └── sa_last/
│
├── models/lightgbm_nfp/               # Saved models (by model)
│   ├── nsa_first/
│   │   ├── lightgbm_nsa_first_model.txt
│   │   └── lightgbm_nsa_first_metadata.pkl
│   ├── nsa_last/
│   ├── sa_first/
│   └── sa_last/
│
└── backtest_historical/               # Historical backtest archives
    ├── 2025-01-15_10-30-45/
    │   └── [all model results at this timestamp]
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

# View specific archive for a model
cat _output/backtest_historical/2025-01-15_10-30-45/nsa_first/metrics_summary.csv
```

### List Available Models
```python
from Train.model import list_available_models

# List all trained models
models = list_available_models()
print(models)  # ['nsa_first', 'nsa_last', 'sa_first', 'sa_last']
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
- `train_lightgbm_nfp.py`: Feature selection and model training (multi-target)
- `config.py`: Target configuration and path utilities
- `data_loader.py`: Data loading with release_type support
- `model.py`: Model save/load with model_id support
- `settings.py`: Configuration (BACKTEST_MONTHS)

## CLI Reference

```bash
# Training Options
--train              # Train single model
--train-all          # Train all 4 model variants
--train-both         # Train NSA and SA (same release)
--train-both-releases  # Train first and last (same target)

# Target Selection
--target {nsa,sa}    # Target type (default: nsa)
--release {first,last}  # Release type (default: first)

# Loss Function
--huber-loss         # Use Huber loss (robust to outliers)
--huber-delta FLOAT  # Huber delta parameter (default: 1.0)

# Prediction
--predict YYYY-MM    # Predict for specific month
--latest             # Predict for latest available month

# Diagnostics
--diagnostics        # Run VIF and correlation analysis
```

## Future Enhancements

Potential additions:
- Automated comparison reports between runs
- Feature importance tracking over time
- Error distribution visualizations
- Subsector prediction accuracy breakdown
- Confidence interval calibration metrics
- Cross-model comparison reports
