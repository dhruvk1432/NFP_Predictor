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

## Directory Structure

```
NFP_Predictor/
├── Load_Data/              # Data fetching scripts
│   ├── fred_snapshots.py       # FRED employment data (NSA/SA)
│   ├── load_fred_exogenous.py  # VIX, SP500, Oil, Credit Spreads, etc.
│   ├── load_unifier_data.py    # ISM, Consumer Confidence
│   ├── load_noaa_data.py       # Storm event data
│   └── load_ADP_Employment_change.py
│
├── Prepare_Data/           # Data preprocessing
│   ├── create_master_snapshots.py  # Consolidate all sources
│   ├── prepare_fred_snapshots.py   # MoM conversion, scaling
│   └── create_noaa_weighted.py     # NOAA aggregation
│
├── Train/                  # Model training and evaluation
│   ├── train_lightgbm_nfp.py   # Main training script (multi-target)
│   ├── config.py               # Configuration constants
│   ├── data_loader.py          # Data loading with caching
│   ├── feature_engineering.py  # Feature creation
│   ├── model.py                # LightGBM training/prediction
│   ├── backtest_results.py     # Backtest reporting
│   └── backtest_archiver.py    # Results archiving
│
├── data/                   # Data storage (gitignored)
│   ├── fred_data/              # Employment snapshots
│   ├── fred_data_prepared/     # Preprocessed employment data
│   ├── Exogenous_data/         # Master snapshots
│   └── NFP_target/             # Target files (4 variants)
│       ├── y_nsa_first_release.parquet
│       ├── y_nsa_last_release.parquet
│       ├── y_sa_first_release.parquet
│       └── y_sa_last_release.parquet
│
├── _output/                # Model outputs and results
│   ├── backtest_results/       # Per-model backtest results
│   │   ├── nsa_first/
│   │   ├── nsa_last/
│   │   ├── sa_first/
│   │   └── sa_last/
│   ├── backtest_historical/    # Timestamped archives
│   ├── feature_importance/     # Per-model feature importance
│   ├── feature_selection/      # Per-model feature selection
│   └── models/lightgbm_nfp/    # Saved models
│       ├── nsa_first/
│       ├── nsa_last/
│       ├── sa_first/
│       └── sa_last/
│
├── settings.py             # Configuration and environment
├── pipeline_helpers.py     # Shared utilities
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
│                    MONTHLY SNAPSHOTS                             │
│  Point-in-time data available as of each NFP release date       │
│  Location: Data/Exogenous_data/master_snapshots/decades/        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│  - 86 exogenous features (VIX z-scores, crash indicators, etc.) │
│  - Employment sector lags and momentum                          │
│  - Calendar features (survey week intervals)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LIGHTGBM MODEL                                │
│  - Regime-dependent weighting (5x for panic months)             │
│  - Linear baseline + tree ensemble                              │
│  - Huber loss for robustness                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION OUTPUT                             │
│  - Point estimate (thousands of jobs)                           │
│  - 50%, 80%, 95% confidence intervals                           │
│  - Feature importance rankings                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Extreme Event Detection
- `VIX_max_5d_spike`: Rapid panic detection (5-day spike ratio)
- `SP500_days_circuit_breaker`: Count of >5% down days
- `Oil_worst_day_pct`: Single-day crash magnitude
- `VIX_panic_regime`: Binary flag when VIX > 50

### Regime-Dependent Weighting
Training samples during extreme events (COVID crash) receive **5x weight** to ensure the model learns from rare but critical scenarios.

### Point-in-Time Correctness
All data is organized into monthly snapshots containing only information available as of each NFP release date, preventing look-ahead bias in backtesting.

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
NBEATSX_MODEL_TYPE=generic
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
