# NFP Predictor

A robust, point-in-time machine learning system for predicting US Non-Farm Payrolls (NFP) employment changes. The pipeline leverages LightGBM, dynamic exponential decay weighting for regime adaptation, and strict anti-lookahead data architectures to forecast the most closely watched US economic indicator.

## Overview

This project predicts Month-over-Month (MoM) employment changes for the US Bureau of Labor Statistics NFP report. 

**Key Architectural Pillars:**
- **Strict Point-in-Time Data Construction**: Every historical snapshot is meticulously aligned relative to the NFP release calendar. We enforce rigorous `<` filtering on observation dates versus cutoff dates to guarantee **zero look-ahead bias**.
- **Dynamic Recency Weighting (Exponential Decay)**: Instead of hardcoded extreme regime weights, the model dynamically adapts to the current structural regime by applying exponentially decaying sample weights tuned by Optuna half-life bounds.
- **Multimodal Data Sources**: Ingests and processes data from FRED (Employment + Exogenous indicators), ADP, NOAA (Storm Damage), Unifier API (ISM, JOLTS, Confidence), and Prosper Trading.
- **Automated Hyperparameter Tuning**: Utilizes Optuna for optimal hyperparameter search, seamlessly integrated into an expanding window time-series cross-validation scheme.
- **Robust Feature Engineering**: Includes `SymLog` compression for extreme outliers, `Log1p` for skewness, Z-scores, standard lags, seasonal encodings (sin/cos), and proprietary `_symlog_pct_chg` calculations.

## Model Variants

The system supports target configurations primarily driven by the `MODEL_TYPE` environment variable (e.g., `univariate` yielding `total_` prefixes). 

**Target Types:**
- **NSA (Non-Seasonally Adjusted)**: Raw employment data containing seasonal patterns.
- **SA (Seasonally Adjusted)**: Employment data smoothed by BLS seasonal adjustments.

**Release Types:**
- **First Release**: Initial BLS estimate (~7 days after month-end), often highly reactive.
- **Last Release**: Final revised values after all subsequent historical revisions are completed.

---

## 🚀 Quick Start: Running the Pipeline

The **easiest and officially supported way** to run the pipeline is using the `run_full_project.py` orchestrator.

### 1. Installation & Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env to add your API keys: FRED_API_KEY, UNIFIER_USER, UNIFIER_TOKEN, etc.
```

### 2. Full Pipeline Orchestration

The `run_full_project.py` script manages the entire lifecycle: fetching data, creating point-in-time master snapshots, and training/tuning the LightGBM models.

```bash
# ---------------------------------------------------------
# MOST COMMON COMMANDS
# ---------------------------------------------------------

# Run the complete pipeline incrementally (reuses existing data if found)
python run_full_project.py

# Run with FRESH data (Deletes all local data/outputs and re-downloads everything)
python run_full_project.py --fresh
```

### 3. Granular Pipeline Execution (Stages & Skips)

You can run individual stages or skip slower data sources.

```bash
# Run only Data Collection + Preparation (no training)
python run_full_project.py --stage data

# Run only the Training stage (requires data to exist)
python run_full_project.py --stage train

# Run Training but skip Optuna tuning (much faster, uses default params)
python run_full_project.py --stage train --no-tune

# Run the full pipeline but skip specific slow/unnecessary API sources
python run_full_project.py --skip noaa,prosper
```

*To see all available individual steps that can be skipped, run:*
`python run_full_project.py --list-steps`

---

## Directory Structure

```text
NFP_Predictor/
├── Data_ETA_Pipeline/      # Centralized modern data ingestion and preprocessing
│   ├── adp_pipeline.py         # ADP employment processing
│   ├── create_master_snapshots.py # Consolidates all exogenous data into final grids
│   ├── fred_employment_pipeline.py # Core NFP target generation logic
│   ├── load_fred_exogenous.py  # High-frequency indicator pulling
│   ├── load_unifier_data.py    # Unifier API (ISM, Consumer Confidence, JOLTS)
│   ├── noaa_pipeline.py        # Storm events and state-level employment weighting
│   └── nfp_release_calendar.py # Source of truth for historical NFP release dates
│
├── Train/                  # Model training, validation, and prediction
│   ├── train_lightgbm_nfp.py   # Main ML execution script
│   ├── config.py               # Target configurations and hyperparameter bounds
│   ├── data_loader.py          # Secure data loading with point-in-time checks
│   ├── feature_engineering.py  # Cyclical encodings, survey week logic
│   ├── hyperparameter_tuning.py# Optuna optimization logic
│   └── model.py                # Core LightGBM training & prediction intervals
│
├── utils/                  # Shared utilities
│   ├── paths.py                # Centralized directory resolution
│   └── transforms.py           # SymLog, winsorization, Z-scores
│
├── tests/                  # Pytest Unit Test Suite
│   ├── test_config.py          # Validates config bounds and target paths
│   ├── test_model.py           # Validates exponential recency decay math
│   └── ...                     
│
├── scripts/                # Standalone Utilities & Diagnostics
│   ├── check_data_freshness.py # Generates a dashboard of API staleness
│   ├── directional_accuracy.py # Evaluates classification accuracy of continuous models
│   ├── predict_next_nfp.py     # Production inference script for the upcoming release
│   └── revision_analysis.py    # Analyzes historical BLS revision drift
│
├── data/                   # Data storage (gitignored)
├── _output/                # Outputs (models, backtests, plots)
│
├── run_full_project.py     # Unified Pipeline Orchestrator
├── settings.py             # Centralized environment variable & path manager
└── pyproject.toml / requirements.txt
```

## Anti-Lookahead Architecture

Predicting NFP accurately requires defending against the "Swiss Cheese" data problem (missing API initial release dates) and avoiding leakage from future revisions. 

1. **NFP-Relative Timing**: Exogenous data (like weekly jobless claims) is aggregated relative to the *exact NFP release window*, not standard calendar months.
2. **Median Lag Backfilling**: When APIs drop the `first_release_date`, the pipeline calculates empirical median delays to safely impute chronological availability without peering into the future.
3. **Decade Bucketing**: Target data is stored in decade-level folders to handle large timeframes while isolating the exact `realtime_start` values from FRED.

## Testing

The project maintains a rigorous `pytest` suite validating everything from path resolution to the mathematical correctness of the exponential decay sample weighting.

```bash
# Run all 42+ unit tests
pytest tests/ -v
```

## License
Private/Internal Use

## Author
Dhruv Kohli
