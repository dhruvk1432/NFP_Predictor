# Train Module

Model training and evaluation layer for the NFP Predictor system. This module builds LightGBM models for predicting US Non-Farm Payrolls employment changes with regime-dependent weighting and confidence intervals.

## Overview

The Train module is responsible for:
- Loading and assembling training datasets from snapshots
- Engineering features from employment and exogenous data
- Training LightGBM models with time-series cross-validation
- Generating predictions with confidence intervals
- Managing model persistence and backtest results

## Key Features

- **Multi-target Support:** 4 model variants (NSA/SA × first/last release)
- **Regime-dependent Weighting:** 5x weight for panic periods (COVID-like events)
- **Linear Baseline:** Separate linear model for extreme event extrapolation
- **Confidence Intervals:** 50%, 80%, 95% intervals from historical residuals
- **Feature Selection:** Automated removal of highly correlated/low-signal features

## Directory Structure

```
Train/
├── __init__.py
├── README.md                 # This file
├── config.py                 # Centralized configuration
├── train_lightgbm_nfp.py     # Main training orchestrator
├── data_loader.py            # Data loading with caching
├── feature_engineering.py    # Feature creation
├── model.py                  # LightGBM training/prediction
├── evaluate_predictions.py   # Evaluation metrics
├── backtest_results.py       # Backtest reporting
├── backtest_archiver.py      # Results archiving
├── feature_importance.py     # Feature analysis
├── hyperopt.py               # Hyperparameter optimization
├── snapshot_loader.py        # Snapshot caching utility
└── test_pipeline.py          # Pipeline testing
```

## File Descriptions

### `config.py`
**Purpose:** Centralized configuration constants for the entire training system.

**Key Configurations:**

```python
# Target Configurations (4 model variants)
VALID_TARGET_TYPES = ('nsa', 'sa')
VALID_RELEASE_TYPES = ('first', 'last')

# Feature Selection Thresholds
MAX_FEATURES = 80
VIF_THRESHOLD = 10.0
CORR_THRESHOLD = 0.95
MIN_TARGET_CORR = 0.05

# LightGBM Hyperparameters
DEFAULT_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 6,
    ...
}

# Training Configuration
N_CV_SPLITS = 5
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
PANIC_REGIME_WEIGHT = 5.0

# Confidence Levels
CONFIDENCE_LEVELS = [0.50, 0.80, 0.95]
```

**Key Functions:**
| Function | Description |
|----------|-------------|
| `get_target_path(target_type, release_type)` | Get path to target parquet file |
| `get_model_id(target_type, release_type)` | Get model identifier (e.g., 'nsa_first') |
| `parse_model_id(model_id)` | Parse model ID back to components |

---

### `train_lightgbm_nfp.py`
**Purpose:** Main orchestrator for model training with CLI interface.

**Usage:**
```bash
# Train all 4 model variants
python Train/train_lightgbm_nfp.py --train-all

# Train single model
python Train/train_lightgbm_nfp.py --train --target nsa --release first

# Train both NSA and SA (same release)
python Train/train_lightgbm_nfp.py --train-both --release first

# Get prediction for specific month
python Train/train_lightgbm_nfp.py --predict 2024-12 --target sa --release first

# Get latest prediction
python Train/train_lightgbm_nfp.py --latest --target nsa --release first

# Run feature diagnostics
python Train/train_lightgbm_nfp.py --diagnostics --target nsa

# Train with Huber loss (robust to outliers)
python Train/train_lightgbm_nfp.py --train-all --huber-loss
```

**CLI Arguments:**
| Argument | Description |
|----------|-------------|
| `--train` | Train a single model |
| `--train-all` | Train all 4 model variants |
| `--train-both` | Train both NSA and SA |
| `--train-both-releases` | Train both first and last release |
| `--target` | Target type: 'nsa' or 'sa' |
| `--release` | Release type: 'first' or 'last' |
| `--predict` | Get prediction for YYYY-MM |
| `--latest` | Get latest available prediction |
| `--diagnostics` | Run feature diagnostics |
| `--huber-loss` | Use Huber loss (robust to outliers) |

---

### `data_loader.py`
**Purpose:** Data loading utilities with caching for efficient I/O.

**Key Functions:**
```python
def load_fred_snapshot(snapshot_date, use_cache=True):
    """Load FRED employment snapshot."""

def load_master_snapshot(snapshot_date, use_cache=True):
    """Load consolidated master snapshot."""

def load_target_data(target_type='nsa', release_type='first'):
    """Load target data with derived features."""

def get_lagged_target_features(target_df, target_month, prefix='nfp'):
    """Get lagged NFP features (lag1, lag12, rolling means)."""

def pivot_snapshot_to_wide(snapshot_df, target_month):
    """Convert long-format snapshot to wide format features."""
```

**Caching:**
- Module-level LRU cache for snapshots
- Avoids redundant I/O during training
- `clear_snapshot_cache()` to free memory

---

### `feature_engineering.py`
**Purpose:** Feature creation from employment and exogenous data.

**Calendar Features:**
```python
def add_calendar_features(df, target_month):
    """Add cyclical month encoding, survey intervals, seasonal flags."""
```

| Feature | Description |
|---------|-------------|
| `month_sin`, `month_cos` | Cyclical month encoding |
| `quarter_sin`, `quarter_cos` | Cyclical quarter encoding |
| `weeks_since_last_survey` | 4 vs 5 week survey interval |
| `is_5_week_month` | Binary flag for 5-week months |
| `is_summer`, `is_holiday_season` | Seasonal indicators |

**Employment Features:**
```python
def engineer_employment_features(fred_df, target_month, target_type='nsa'):
    """Create features from employment series."""
```

| Feature Pattern | Description |
|-----------------|-------------|
| `{series}_latest` | Most recent value |
| `{series}_mom` | Month-over-month change |
| `{series}_3m_chg` | 3-month change |
| `{series}_yoy` | Year-over-year change |
| `{series}_rolling_3m` | 3-month rolling mean |
| `{series}_volatility` | 6-month volatility |

**Sector Breadth:**
```python
def calculate_sector_breadth(fred_df, target_month, target_type='nsa'):
    """Calculate sector breadth (% of sectors expanding)."""
```

---

### `model.py`
**Purpose:** Core LightGBM training, prediction, and persistence.

**Training:**
```python
def train_lightgbm_model(X, y, n_splits=5, use_huber_loss=False):
    """Train LightGBM with time-series CV."""
    # Returns: (model, feature_importance, residuals)
```

**Sample Weighting:**
```python
def calculate_sample_weights(X):
    """Assign 5x weight to panic regime samples."""
```

Panic indicators checked:
- `VIX_panic_regime`: VIX > 50
- `SP500_crash_month`: Monthly return < -10%
- `VIX_high_regime`: VIX > 40
- `SP500_bear_market`: Drawdown < -20%
- `SP500_circuit_breaker`: Any day down > 5%

**Prediction with Intervals:**
```python
def predict_with_intervals(model, features, residuals, feature_cols):
    """Make prediction with 50%, 80%, 95% confidence intervals."""
```

**Model Persistence:**
```python
def save_model(model, feature_cols, residuals, importance, target_type, release_type):
    """Save model and metadata to disk."""

def load_model(target_type='nsa', release_type='first'):
    """Load trained model and metadata."""
```

**Linear Baseline:**
```python
def train_linear_baseline(X, y, predictor_cols):
    """Train OLS baseline for extreme event extrapolation."""
```

---

### `evaluate_predictions.py`
**Purpose:** Calculate evaluation metrics for model predictions.

**Metrics Calculated:**
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| Directional Accuracy | % correct direction predictions |
| Hit Rate | % within confidence intervals |

---

### `backtest_results.py`
**Purpose:** Generate and format backtest results reports.

**Output:**
- Per-month predictions vs actuals
- Rolling performance metrics
- Confidence interval coverage
- Feature importance rankings

---

### `backtest_archiver.py`
**Purpose:** Archive historical backtest results with timestamps.

**Output:** `_output/backtest_historical/{timestamp}/`

---

### `feature_importance.py`
**Purpose:** Analyze and visualize feature importance.

**Analysis:**
- LightGBM gain-based importance
- Permutation importance
- SHAP values (optional)

**Output:** `_output/feature_importance/{model_id}/`

---

### `hyperopt.py`
**Purpose:** Hyperparameter optimization using Optuna.

**Tunable Parameters:**
- `learning_rate`
- `num_leaves`
- `max_depth`
- `feature_fraction`
- `bagging_fraction`

---

### `snapshot_loader.py`
**Purpose:** Efficient snapshot loading with batch operations.

## Model Variants

| Model ID | Target | Release | Description |
|----------|--------|---------|-------------|
| `nsa_first` | NSA | First | Non-seasonally adjusted, initial release |
| `nsa_last` | NSA | Last | Non-seasonally adjusted, final revised |
| `sa_first` | SA | First | Seasonally adjusted, initial release |
| `sa_last` | SA | Last | Seasonally adjusted, final revised |

**When to use which:**
- **NSA models:** Captures raw seasonal patterns, useful for understanding true hiring dynamics
- **SA models:** More stable, comparable month-to-month
- **First release:** Predict what BLS will initially report
- **Last release:** Predict final revised values

## Feature Categories (86 Total)

### Financial Market Indicators
- VIX: mean, max, volatility, 30d spike, panic regime
- S&P 500: returns, drawdown, circuit breakers, crash months
- Oil: prices, 30d crash, volatility
- Credit spreads: average, changes, Z-scores
- Yield curve: slope, changes, inversion flags

### Employment Indicators
- Weekly claims: ICSA, CCSA levels and changes
- Employment lags: 1, 2, 3, 6, 12, 18, 24 months
- 40+ sector employment series with momentum features

### Regional/Real-Time
- Empire State Manufacturing Employment
- Philadelphia Fed Employment Diffusion
- Weekly Economic Index (WEI)

### NOAA Storm Events
- Deaths/injuries (employment-weighted)
- Property/crop damage (inflation-adjusted)

### Calendar Features
- Cyclical month/quarter encoding
- Survey week interval (4 vs 5 weeks)
- Seasonal indicators

### Protected Binary Flags (Never Removed)
- `VIX_panic_regime`
- `VIX_high_regime`
- `SP500_crash_month`
- `SP500_bear_market`
- `SP500_circuit_breaker`

## Training Pipeline

```
1. Load Data
   ├── Employment snapshots (fred_data_prepared/)
   ├── Master snapshots (master_snapshots/)
   └── Target data (NFP_target/)

2. Build Features
   ├── Pivot exogenous to wide format
   ├── Add calendar features
   ├── Add lagged NFP features
   └── Add employment sector features

3. Feature Selection
   ├── Remove highly correlated (>0.95)
   ├── Remove low target correlation (<0.05)
   └── Keep protected binary flags

4. Calculate Weights
   └── 5x weight for panic regime samples

5. Train Model
   ├── 5-fold time series CV
   ├── Early stopping (patience=50)
   └── Collect OOF residuals

6. Generate Output
   ├── Save model and metadata
   ├── Save feature importance
   └── Save backtest results
```

## Output Structure

```
_output/
├── models/lightgbm_nfp/
│   ├── nsa_first/
│   │   ├── lightgbm_nsa_first_model.txt
│   │   └── lightgbm_nsa_first_metadata.pkl
│   ├── nsa_last/
│   ├── sa_first/
│   └── sa_last/
├── backtest_results/
│   ├── nsa_first/
│   ├── nsa_last/
│   ├── sa_first/
│   └── sa_last/
├── backtest_historical/
│   └── {timestamp}/
├── feature_importance/
│   └── {model_id}/
└── feature_selection/
    └── {model_id}/
```

## Performance Metrics

Typical backtest results:
| Metric | Range | Notes |
|--------|-------|-------|
| Directional Accuracy | 65-70% | Correct sign prediction |
| MAE | 80-120K jobs | Mean absolute error |
| RMSE | 150-250K jobs | Higher due to COVID outliers |
| 80% CI Coverage | ~80% | Calibration check |

## Dependencies

- `lightgbm`: Gradient boosting model
- `scikit-learn`: CV, metrics, linear regression
- `pandas`, `numpy`: Data manipulation
- `optuna`: Hyperparameter optimization (optional)

## Related Modules

- **Load_Data:** Provides raw data
- **Prepare_Data:** Provides transformed features
- **utils/transforms.py:** SymLog inverse for predictions
- **utils/paths.py:** Centralized path generation
