# NFP Predictor

Machine learning system for predicting US Non-Farm Payrolls (NFP) employment changes using LightGBM with regime-dependent weighting and point-in-time data correctness.

## Overview

This project predicts monthly employment changes (Month-over-Month) for the US Bureau of Labor Statistics NFP report. Key features:

- **86 features** from FRED, ADP, Unifier, NOAA, and JOLTS data sources
- **LightGBM model** with regime-dependent weighting for extreme events
- **Point-in-time snapshots** ensuring no look-ahead bias
- **Extreme event detection** with COVID-specific features (VIX spikes, circuit breakers)
- **Confidence intervals** based on historical residuals
- **Multi-target support** for 4 model variants (NSA/SA x first/last release)

## Model Variants

The system supports 4 target configurations:

| Model ID | Target Type | Release Type | Description |
|----------|-------------|--------------|-------------|
| `nsa_first` | NSA | First | Non-seasonally adjusted, initial release |
| `nsa_last` | NSA | Last | Non-seasonally adjusted, final revised |
| `sa_first` | SA | First | Seasonally adjusted, initial release |
| `sa_last` | SA | Last | Seasonally adjusted, final revised |

**Target Types:**
- **NSA (Non-Seasonally Adjusted)**: Raw employment data with seasonal patterns
- **SA (Seasonally Adjusted)**: Employment data with seasonal adjustment applied

**Release Types:**
- **First Release**: Initial BLS estimate (~7 days after month-end)
- **Last Release**: Final revised values after all revisions complete

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (FRED_API_KEY, UNIFIER_TOKEN, etc.)

# 3. Fetch and prepare data
python Load_Data/fred_snapshots.py           # Employment data
python Load_Data/load_fred_exogenous.py      # Exogenous indicators
python Prepare_Data/create_master_snapshots.py  # Consolidate all data

# 4. Train all 4 model variants
python Train/train_lightgbm_nfp.py --train-all

# Or train a single model (default: nsa_first)
python Train/train_lightgbm_nfp.py --train --target nsa --release first
```

## Module Documentation

Each major module has detailed documentation:

| Module | Description | Documentation |
|--------|-------------|---------------|
| **Load_Data** | Data ingestion from external APIs | [Load_Data/README.md](Load_Data/README.md) |
| **Prepare_Data** | Data transformation and consolidation | [Prepare_Data/README.md](Prepare_Data/README.md) |
| **Train** | Model training and evaluation | [Train/README.md](Train/README.md) |

## Directory Structure

```
NFP_Predictor/
├── Load_Data/              # Data fetching scripts
│   ├── README.md               # Module documentation
│   ├── fred_snapshots.py       # FRED employment data (NSA/SA)
│   ├── load_fred_exogenous.py  # VIX, SP500, Oil, Credit Spreads, etc.
│   ├── load_unifier_data.py    # ISM, Consumer Confidence
│   ├── load_noaa_data.py       # Storm event data
│   ├── load_prosper_data.py    # Consumer sentiment surveys
│   └── load_ADP_Employment_change.py
│
├── Prepare_Data/           # Data preprocessing
│   ├── README.md               # Module documentation
│   ├── create_master_snapshots.py  # Consolidate all sources
│   ├── prepare_fred_snapshots.py   # MoM conversion, scaling
│   ├── create_noaa_weighted.py     # NOAA aggregation
│   └── create_adp_snapshots.py     # ADP alignment
│
├── Train/                  # Model training and evaluation
│   ├── README.md               # Module documentation
│   ├── train_lightgbm_nfp.py   # Main training script (multi-target)
│   ├── config.py               # Configuration constants
│   ├── data_loader.py          # Data loading with caching
│   ├── feature_engineering.py  # Feature creation
│   ├── model.py                # LightGBM training/prediction
│   ├── backtest_results.py     # Backtest reporting
│   └── backtest_archiver.py    # Results archiving
│
├── utils/                  # Shared utilities
│   ├── paths.py                # Centralized path generation
│   └── transforms.py           # SymLog, Log1p, Z-score functions
│
├── tests/                  # Unit tests
│   ├── conftest.py             # Shared fixtures
│   ├── test_config.py          # Config module tests
│   ├── test_data_loader.py     # Data loader tests
│   ├── test_feature_engineering.py  # Feature tests
│   ├── test_model.py           # Model tests
│   ├── test_paths.py           # Path utility tests
│   └── test_transforms.py      # Transform tests
│
├── data/                   # Data storage (gitignored)
│   ├── fred_data/              # Employment snapshots
│   ├── fred_data_prepared/     # Preprocessed employment data
│   ├── Exogenous_data/         # Master snapshots
│   └── NFP_target/             # Target files (4 variants)
│
├── _output/                # Model outputs and results
│   ├── backtest_results/       # Per-model backtest results
│   ├── backtest_historical/    # Timestamped archives
│   ├── feature_importance/     # Per-model feature importance
│   └── models/lightgbm_nfp/    # Saved models
│
├── settings.py             # Configuration and environment
├── pipeline_helpers.py     # Shared utilities
├── run_full_project.py     # Full pipeline orchestrator
└── requirements.txt
```

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
├─────────────────────────────────────────────────────────────────┤
│  FRED API     │  Unifier API  │  ADP Scraper  │  NOAA Data     │
│  - Employment │  - ISM        │  - Forecast   │  - Storm Events│
│  - VIX, SP500 │  - Confidence │  - Actual     │  - Damage      │
│  - Oil, Credit│  - JOLTS      │               │                │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOAD_DATA MODULE                              │
│  Raw data fetching with API rate limiting and error handling    │
│  Output: data/{source}/decades/{decade}/{year}/{YYYY-MM}.parquet│
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPARE_DATA MODULE                            │
│  - SymLog/Log1p transforms for extreme values                   │
│  - MoM change calculation                                       │
│  - NOAA employment-weighted aggregation                         │
│  - Master snapshot consolidation                                │
│  Output: data/Exogenous_data/master_snapshots/                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAIN MODULE                                │
│  - 86 engineered features (lags, rolling stats, calendar)       │
│  - Automated feature selection                                  │
│  - 5x regime-dependent weighting for panic months               │
│  - LightGBM with 5-fold time series CV                          │
│  - 50%, 80%, 95% confidence intervals                           │
│  Output: _output/models/lightgbm_nfp/{model_id}/                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Extreme Event Detection
- `VIX_panic_regime`: Binary flag when VIX > 50
- `VIX_high_regime`: Binary flag when VIX > 40
- `SP500_crash_month`: Monthly return < -10%
- `SP500_circuit_breaker`: Any day down > 5%
- `SP500_bear_market`: Drawdown < -20% from 52-week high

### Regime-Dependent Weighting
Training samples during extreme events (COVID crash) receive **5x weight** to ensure the model learns from rare but critical scenarios.

### Point-in-Time Correctness
All data is organized into monthly snapshots containing only information available as of each NFP release date, preventing look-ahead bias in backtesting.

### Data Transformations
- **SymLog**: `sign(x) * log1p(|x|)` for values with negative outliers
- **Log1p**: `log(1 + x)` for positive values with high skewness
- Both transforms are fully invertible for prediction recovery

## Training Commands

```bash
# Train all 4 model variants (recommended)
python Train/train_lightgbm_nfp.py --train-all

# Train a single model
python Train/train_lightgbm_nfp.py --train --target nsa --release first

# Train both NSA and SA (same release type)
python Train/train_lightgbm_nfp.py --train-both --release first

# Train both release types (same target type)
python Train/train_lightgbm_nfp.py --train-both-releases --target nsa

# Train with Huber loss (robust to outliers)
python Train/train_lightgbm_nfp.py --train-all --huber-loss

# Get prediction for a specific month
python Train/train_lightgbm_nfp.py --predict 2024-12 --target sa --release first

# Get latest prediction
python Train/train_lightgbm_nfp.py --latest --target nsa --release first

# Run feature diagnostics
python Train/train_lightgbm_nfp.py --diagnostics --target nsa
```

## Full Pipeline Execution

```bash
# Run complete pipeline with existing data
python run_full_project.py

# Run with fresh data (deletes and re-downloads)
python run_full_project.py --fresh

# Run specific stage only
python run_full_project.py --stage load
python run_full_project.py --stage prepare
python run_full_project.py --stage train

# Skip specific data sources
python run_full_project.py --skip noaa,prosper
```

## Environment Variables

Create a `.env` file with:

```env
FRED_API_KEY=your_fred_api_key
DATA_PATH=./data
START_DATE=1990-01-01
END_DATE=2025-12-31
BACKTEST_MONTHS=36
OUTPUT_DIR=./_output
TEMP_DIR=./_temp
UNIFIER_USER=your_username
UNIFIER_TOKEN=your_token
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=Train --cov=utils --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v
```

## Model Performance

Typical backtest metrics (varies by target configuration):
- **Directional Accuracy**: 65-70%
- **MAE**: 80-120K jobs
- **RMSE**: 150-250K jobs (higher due to COVID outliers)

Note: "Last release" models may show different metrics since they predict
the final revised values rather than initial estimates.

## Dependencies

- Python 3.10+
- pandas, numpy
- lightgbm
- scikit-learn
- fredapi
- requests

See `requirements.txt` for full list.

## License

Private/Internal Use

## Author

Dhruv Kohli
