# NFP Predictor

A machine learning pipeline for forecasting U.S. Non-Farm Payrolls (NFP) month-over-month employment changes. It uses LightGBM, expanding-window walk-forward validation, and multimodal data sources (FRED, ADP, NOAA, Unifier, Prosper).

The architecture is explicitly designed around **point-in-time (PIT) correctness** and **regime-aware feature selection** to mathematically prevent lookahead bias and handle high-dimensional macroeconomic noise.

---

## Recent Updates (2026-02-25)

- **ETL pipeline consolidated into `Data_ETA_Pipeline/`**: All ingestion scripts (`load_fred_exogenous.py`, `adp_pipeline.py`, `noaa_pipeline.py`, `load_unifier_data.py`, `load_prosper_data.py`) now live here. The old `Load_Data/` folder is fully deprecated.
- **6-stage Feature Selection Engine added** (`Data_ETA_Pipeline/feature_selection_engine.py`): Regime-aware, per-source selection run as part of master snapshot generation.
- **Master snapshot pipeline (`create_master_snapshots.py`) is now quad-track**: Produces `{nsa,sa} × {first_release,revised}` snapshots; feature selection results are cached per-source to support incremental reruns.
- **All 4 model variants are now the default**: `run_full_project.py` trains `{nsa,sa} × {first_release,revised}` via `--train-all`, producing a comparative scorecard alongside per-variant diagnostics.
- **Performance profiling layer added** (`Data_ETA_Pipeline/perf_stats.py`): JSON timing dumps written to `_temp/perf/`.
- **`scripts/` folder expanded**: Added `check_data_freshness.py`, `benchmark_keep_rule.py`, `directional_accuracy.py`, `revision_analysis.py`.
- **Test suite expanded** to `tests/` with 13 test modules (pytest + ruff + mypy CI via `.github/workflows/test.yml`).
- **`utils/` module added**: `transforms.py` (SymLog, winsorization), `paths.py`, `benchmark_harness.py`.
- **Notebooks split into two folders**: `notebooks/` (NSA-track feature analysis) and `revised_notebooks/` (SA/revised-track feature analysis).

---

## 1. The NFP Challenge: Why This is Hard

Forecasting NFP is notoriously difficult for quantitative models due to four structural realities:
1. **Aggressive Revisions:** The initial Bureau of Labor Statistics (BLS) release is heavily revised in subsequent months. Models trained naively on "finalized" historical data will suffer severe lookahead bias by assuming they had clean data that didn't exist in real-time.
2. **Asynchronous Availability:** Economic indicators are published at mismatched frequencies (daily, weekly, monthly) with varying lag times (e.g., NOAA storm data arrives ~75 days late). Aligning these without peeking into the future requires rigorous data versioning.
3. **Regime Non-Stationarity:** The economy fundamentally breaks its own rules. Relationships that held during the "Great Moderation" (pre-2008) often decoupled or inverted during the Global Financial Crisis (GFC) or the 2020 COVID-19 shock.
4. **High-Dimensional Instability:** Thousands of macroeconomic series contain spurious or transient correlations. Without temporally-aware feature selection, complex models will catastrophically overfit to this noise.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph "Phase 1: Raw Ingestion"
        FRED[FRED APIs]
        ADP[ADP Payrolls]
        NOAA[NOAA Weather]
        Surveys[Survey Data]
    end

    subgraph "Phase 2: Time-Shielding"
        Snap[Monthly PIT Snapshots\nStrict Release-Date Cutoffs]
    end

    subgraph "Phase 3: Dimensionality Reduction"
        FeatSel[6-Stage Feature Selection\n(Per Regime & Source)]
        Master[Master Wide Snapshots]
    end

    subgraph "Phase 4: Walk-Forward Validation"
        Pool[Union Candidate Pool]
        Train[Expanding-Window Model\n+ Optuna Tuning]
    end

    subgraph "Phase 5: Analyst Output"
        Preds[Predictions & Intervals CSV]
        Metrics[RMSE / MAE Scorecard]
        Shap[SHAP Causality]
    end

    FRED --> Snap
    ADP --> Snap
    NOAA --> Snap
    Surveys --> Snap

    Snap --> FeatSel
    FeatSel --> Master
    Master --> Pool
    Pool --> Train
    Train --> Preds
    Train --> Metrics
    Train --> Shap
```

---

## 3. Quickstart

### Prerequisites

- Python 3.10+
- On macOS, LightGBM requires OpenMP: `brew install libomp`

### Install

```bash
pip install -r requirements.txt
```

For development tools (linting, pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

### Configure environment

Copy the template and fill in your credentials:

```bash
cp .env .env.local    # or edit .env directly — it is not committed to git
```

Required variables in `.env`:

| Variable | Description |
|---|---|
| `FRED_API_KEY` | FRED API key (required) |
| `UNIFIER_USER` | Unifier API username (required) |
| `UNIFIER_TOKEN` | Unifier API token (required) |
| `DATA_PATH` | Path to data directory (default: `./data`) |
| `START_DATE` | Training start date (e.g. `1990-01-01`) |
| `BACKTEST_MONTHS` | Number of backtest months (e.g. `36`) |

Optional: `END_DATE`, `MODEL_TYPE`, `TARGET_TYPE`, `OUTPUT_DIR` (default `_output`), `TEMP_DIR` (default `_temp`), `DEBUG`, `REFRESH_CACHE`.

### Smoke test

```bash
# Verify environment is configured and directories are accessible
python settings.py

# Run the test suite
pytest tests/ -v
```

---

## 4. Repository Structure

```
NFP_Predictor/
├── settings.py                          # Central config: env vars, paths, logger factory
├── run_full_project.py                  # Pipeline orchestrator (data → master snapshots → train)
│
├── Data_ETA_Pipeline/                   # Phase 1–3: Ingestion, PIT snapshots, feature selection
│   ├── fred_employment_pipeline.py      # FRED employment series (NSA + SA, ALFRED vintage)
│   ├── load_fred_exogenous.py           # FRED exogenous series (weekly jobless claims, etc.)
│   ├── adp_pipeline.py                  # ADP payroll snapshots
│   ├── noaa_pipeline.py                 # NOAA weather data (75-day lag modelling)
│   ├── load_unifier_data.py             # Unifier survey data (vendor timestamp repair)
│   ├── load_prosper_data.py             # Prosper consumer data
│   ├── feature_selection_engine.py      # 6-stage regime-aware feature selection
│   ├── create_master_snapshots.py       # Aggregates all sources → master parquets (quad-track)
│   ├── nfp_release_calendar.py          # BLS NFP release date calendar
│   ├── perf_stats.py                    # Performance profiling decorators + JSON dumps
│   └── utils.py                         # ETL utilities
│
├── Train/                               # Phase 4–5: Model training and output
│   ├── train_lightgbm_nfp.py            # Main entrypoint: expanding-window backtest
│   ├── config.py                        # Hyperparameters, paths, valid target configs
│   ├── data_loader.py                   # Master snapshot loading + pivot_snapshot_to_wide()
│   ├── feature_engineering.py           # Calendar, survey-week, employment lag features
│   ├── model.py                         # LightGBM fit/predict helpers (safe wrappers)
│   ├── hyperparameter_tuning.py         # Optuna tuning
│   ├── baselines.py                     # Naive baselines (prior month, VAR, AR1)
│   ├── candidate_pool.py                # Union of feature-selection survivors (cached)
│   ├── short_pass_selection.py          # Per-step top-K feature filter (walk-forward)
│   ├── revision_features.py             # Revision-specific feature construction
│   ├── selected_features/               # Per-source, per-track selected feature JSONs
│   └── Output_code/                     # Output generation modules
│       ├── model_comparison.py          # Multi-variant scorecard (CSV + styled HTML)
│       ├── generate_output.py           # Orchestrates all output artefacts
│       ├── metrics.py                   # RMSE, MAE, coverage calculations
│       ├── plots.py                     # Backtest, residual, and diagnostic plots
│       └── feature_importance.py        # Gain + SHAP importance analysis
│
├── scripts/                             # Utility / diagnostic scripts
│   ├── predict_next_nfp.py              # Production inference: next-month prediction + intervals
│   ├── check_data_freshness.py          # Verify data is up-to-date before release day
│   ├── benchmark_keep_rule.py           # Run keep-rule benchmark reports
│   ├── directional_accuracy.py          # Directional hit-rate analysis
│   └── revision_analysis.py            # NFP revision autocorrelation analysis
│
├── utils/                               # Shared utilities
│   ├── transforms.py                    # SymLog, COVID winsorization, Z-score helpers
│   ├── paths.py                         # Cross-platform path helpers
│   └── benchmark_harness.py            # A/B timing harness for algorithm comparisons
│
├── tests/                               # pytest test suite (13 modules)
│   ├── conftest.py
│   ├── test_feature_engineering.py
│   ├── test_data_loader.py
│   ├── test_model.py
│   ├── test_transforms.py
│   ├── test_paths.py
│   ├── test_config.py
│   ├── test_baselines.py
│   ├── test_candidate_pool.py
│   ├── test_short_pass_selection.py
│   ├── test_create_master_snapshots.py
│   ├── test_feature_selection_engine.py
│   ├── test_benchmark_harness.py
│   ├── test_global_lags.py
│   └── test_keep_rule_integration.py
│
├── notebooks/                           # Per-source feature analysis notebooks (NSA track)
├── revised_notebooks/                   # Per-source feature analysis notebooks (SA/revised track)
│
├── Prepare_Data/                        # Legacy (build_revised_targets.py only — deprecated)
│
├── data/                                # Data root (not committed; set via DATA_PATH)
│   ├── fred_data/decades/               # Raw FRED vintage snapshots
│   ├── fred_data_prepared_{nsa,sa}/     # Prepared FRED employment snapshots
│   ├── Exogenous_data/                  # Per-source parquets (ADP, NOAA, Unifier, Prosper, FRED exog)
│   ├── master_snapshots/{nsa,sa}/       # Feature-selected master snapshots (quad-track)
│   └── NFP_target/                      # Target parquets (y_nsa_first_release.parquet, etc.)
│
├── _output/                             # Pipeline output artefacts
├── _temp/                               # Logs and performance profiling JSON files
│
├── requirements.txt
├── pyproject.toml                       # Build config, ruff lint rules, pytest settings
└── .github/workflows/test.yml           # CI: pytest × Python 3.10–3.12, ruff, mypy
```

---

## 5. Point-In-Time (PIT) Data Integrity

**Target:** Month-over-month change in U.S. Non-Farm Payrolls (`y_mom`). First Release (initial, unrevised BLS estimate) is the primary ground-truth target.

**Constraint:** Every feature `X_i` mapped to prediction month `t` satisfies: `feature_release_date < target_release_date(t)`.

Source-level PIT fixes applied before merging:

- **FRED Employment:** Vintage ALFRED snapshots where available; missing pre-2009 release dates use heuristic imputation.
- **FRED Exogenous (weekly):** Weekly jobless claims are bucketed to NFP reference weeks via vintage-based backfills to prevent leakage.
- **NOAA (Weather):** Storm data publishes ~75 days late; modelled as `month-end + 75 days` to safely bypass recency validation.
- **Unifier (Surveys):** Vendor overwrites `first_release_date` with today's timestamp; repaired by computing historical median publication lag per series.

---

## 6. The Feature Selection Engine

Implemented in [`Data_ETA_Pipeline/feature_selection_engine.py`](Data_ETA_Pipeline/feature_selection_engine.py). Run independently per historical regime (Pre-GFC, GFC, Late-Cycle, COVID, Post-COVID) and per source. Six stages:

1. **Variance Filter:** Drops near-zero-variance series.
2. **Dual Filter:** Expanding-window purged correlation + Random Subspace LightGBM non-linear importances.
3. **Boruta:** Shadow-feature randomized permutation test (100 runs, binomial significance).
4. **Vintage Stability Check:** Exponential recency weighting to reject features that shift structurally over time.
5. **Cluster Redundancy:** NaN-aware Spearman hierarchical clustering to collapse collinear survivors.
6. **Sequential Forward Selection:** Walk-forward cross-validation with embargo rules to finalize the feature set.

Regime boundaries and feature cache versions are defined in [`Data_ETA_Pipeline/create_master_snapshots.py`](Data_ETA_Pipeline/create_master_snapshots.py).

---

## 7. Model Training & Validation

### How it works

The core simulation is an **Expanding Window Backtest** in [`Train/train_lightgbm_nfp.py`](Train/train_lightgbm_nfp.py):

- Marches forward one month at a time, strictly chronologically.
- At each step `t`: trains on `[0 … t-1]`, predicts `t`, records out-of-sample error.
- **Sample Weighting:** Exponential time-decay with a half-life tuned by Optuna.
- **Short-Pass Filter** (`Train/short_pass_selection.py`): At each step, selects the top-K features available at exactly that cutoff date, ensuring the matrix is dense.
- **Baselines** (`Train/baselines.py`): Prior-month carryover, VAR, AR1.
- **Keep Rule:** If trailing OOS error exceeds the naive baseline, a safety flag is logged and model deployment is blocked.

### Trained model variants

The pipeline trains **all 4 model variants** by default (via `--train-all`):

| Model ID | Target | Feature Source | Purpose |
|---|---|---|---|
| `nsa_first` | Non-Seasonally Adjusted | First Release | Real-time operational prediction |
| `nsa_first_revised` | Non-Seasonally Adjusted | Revised | Upper-bound benchmark (hindsight data) |
| `sa_first` | Seasonally Adjusted | First Release | Real-time, seasonal effects removed |
| `sa_first_revised` | Seasonally Adjusted | Revised | Upper-bound benchmark, seasonal effects removed |

First-release models are operationally deployable. Revised models quantify how much predictive power is lost to data noise vs. structural model weakness. All 4 are defined in [`Train/config.py`](Train/config.py) `ALL_TARGET_CONFIGS`.

---

## 8. Running the Pipeline

### Full pipeline (recommended)

```bash
# End-to-end: fetch data → feature selection → master snapshots → train all 4 variants
python run_full_project.py

# Fresh start: delete all local data and re-download from scratch
python run_full_project.py --fresh
```

### Individual stages

```bash
# Data collection + preparation only (no training)
python run_full_project.py --stage data

# Data loading only (fetch from external APIs)
python run_full_project.py --stage load

# Data preparation only (feature selection + build master snapshots; assumes load is complete)
python run_full_project.py --stage prepare

# Training only (master snapshots must already exist)
python run_full_project.py --stage train

# Training without Optuna tuning (faster, uses static defaults)
python run_full_project.py --stage train --no-tune

# Skip specific data sources (comma-separated)
python run_full_project.py --skip noaa,prosper

# List all pipeline steps
python run_full_project.py --list-steps
```

### Direct training script

```bash
# Train single model (default: nsa, first release)
python Train/train_lightgbm_nfp.py --train

# Train with specific target/release
python Train/train_lightgbm_nfp.py --train --target sa --release first

# Train all 4 variants and generate comparison scorecard
python Train/train_lightgbm_nfp.py --train-all

# Train without Optuna (faster for debugging)
python Train/train_lightgbm_nfp.py --train --no-tune

# Predict for a specific historical month
python Train/train_lightgbm_nfp.py --predict 2024-12 --target nsa

# Predict latest available month
python Train/train_lightgbm_nfp.py --latest --target nsa
```

### Production inference

```bash
# Generate next-month NFP prediction with confidence intervals
python scripts/predict_next_nfp.py --target nsa
python scripts/predict_next_nfp.py --target sa --output report.json
```

### Diagnostic utilities

```bash
# Check whether data sources are up-to-date
python scripts/check_data_freshness.py

# Run keep-rule benchmark reports
python scripts/benchmark_keep_rule.py

# Directional hit-rate analysis on backtest results
python scripts/directional_accuracy.py

# NFP revision autocorrelation analysis
python scripts/revision_analysis.py
```

---

## 9. Configuration

Primary configuration lives in `.env` and [`Train/config.py`](Train/config.py).

**Key `Train/config.py` constants:**

| Constant | Description |
|---|---|
| `DEFAULT_LGBM_PARAMS` | LightGBM baseline hyperparameters |
| `TUNE_EVERY_N_MONTHS` | Optuna tuning frequency (default: 12 months) |
| `HALF_LIFE_MIN_MONTHS` / `HALF_LIFE_MAX_MONTHS` | Optuna search range for sample decay half-life |
| `NUM_BOOST_ROUND` | Max boosting rounds |
| `EARLY_STOPPING_ROUNDS` | Early stopping patience |
| `USE_HUBER_LOSS_DEFAULT` / `HUBER_DELTA` | Huber loss config for outlier robustness |
| `ALL_TARGET_CONFIGS` | All 4 model variants (`nsa_first`, `nsa_first_revised`, `sa_first`, `sa_first_revised`) |
| `MASTER_SNAPSHOTS_BASE` | Root of the feature-selected master snapshot parquets |
| `MODEL_SAVE_DIR` | Where trained `.txt` model files are saved |

---

## 10. Output Artefacts

A completed training run deposits results under `_output/`:

```
_output/
├── NSA_prediction/
│   ├── backtest_results.csv         # OOS predictions vs actuals (expanding window)
│   ├── backtest_predictions.png     # Visual backtest overlay
│   ├── feature_importance.csv       # Gain-based feature rankings
│   ├── shap_values.png              # SHAP beeswarm plot
│   ├── summary_statistics.csv       # RMSE, MAE, coverage per window
│   └── summary_table.png
├── SA_prediction/                   # Same structure for SA model
├── NSA_plus_adjustment/             # NSA + seasonal adjustment diagnostics
├── Predictions/
│   └── predictions.csv              # Combined OOS predictions
├── benchmark_reports/               # Keep-rule JSON reports (itemN_keep_rule_report.json)
├── models/lightgbm_nfp/             # Saved LightGBM model files (.txt)
├── revision_analysis/               # Revision ACF/PACF and trend plots
├── Archive/                         # Timestamped snapshots of previous runs
├── directional_accuracy.jpg
└── revised_vs_predicted_mom.jpg
```

The default pipeline invokes `Train/Output_code/model_comparison.py` to write a multi-variant scorecard (CSV + styled HTML) to `_output/models/lightgbm_nfp/`, comparing all 4 variants side-by-side.

---

## 11. Reproducibility

- **Random seeds:** LightGBM uses a fixed `seed` in `DEFAULT_LGBM_PARAMS` (see [`Train/config.py`](Train/config.py)).
- **Determinism:** Expanding-window splits are strictly chronological with no shuffling.
- **Run tracking:** Performance profiling JSON files are written to `_temp/perf/` for each pipeline run (process ID + timestamp stamped). Archived output snapshots are saved to `_output/Archive/`.

---

## 12. Testing & Linting

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=Train --cov=utils --cov-report=term-missing

# Lint (ruff)
ruff check .

# Type check (non-blocking; known stubs missing for some dependencies)
mypy Train/ utils/ --ignore-missing-imports
```

CI is configured in [`.github/workflows/test.yml`](.github/workflows/test.yml) and runs pytest + ruff + mypy across Python 3.10, 3.11, and 3.12 on every push to `main`.

---

## 13. Economic Shock Handling

To maintain stability across massive dislocations (2008 GFC, 2020 COVID shock):

- **COVID Winsorization:** Spring 2020 extreme values are clipped to non-COVID distribution quantiles via `utils/transforms.py`. Preserves directionality without destroying loss geometry.
- **Symmetric Log Transforms:** Heavy-tailed features optionally undergo SymLog (`np.sign(x) * np.log1p(np.abs(x))`), compressing extreme kurtosis while handling negative bounds.
- **Post-1990 Anchor:** `load_snapshot_wide` enforces `X.index >= '1990-01-01'` to remove pre-1990 sparsity from peripheral survey metrics.

---

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: Missing required environment variable` | Set the variable in `.env`; see §3 for required variables |
| LightGBM OpenMP error on macOS | `brew install libomp` |
| `FileNotFoundError: No master snapshots found` | Run `python run_full_project.py --stage data` first |
| `optuna` import error | `pip install optuna` or use `--no-tune` to skip tuning |
| Empty `_output/` | Ensure training stage completes without errors; check `_temp/*.log` |

---

## License

Private / Internal Use

## Author

Dhruv Kohli
