# NFP Predictor

A point-in-time machine learning pipeline for predicting U.S. Non-Farm Payrolls (NFP) month-over-month employment changes. Uses LightGBM with expanding-window backtesting, multimodal data sources (FRED, ADP, NOAA, Unifier, Prosper), and a 6-stage feature selection engine. Strict anti-lookahead architecture ensures no future data leaks into historical predictions.

## Key Concepts

**Target variable:** Month-over-month change in U.S. Non-Farm Payrolls (`y_mom`), in two variants:
- **NSA** (Non-Seasonally Adjusted) and **SA** (Seasonally Adjusted)
- **First release** (initial BLS estimate) is the default training target in `Train/config.py`.
- **Revised** target branches are still built in ETL and can be trained explicitly for analysis, but they represent hindsight information.

**Point-in-time (as-of) correctness:** Every feature used for a given prediction month is filtered by the NFP release date for that month (strict `<` cutoff). Data is stored in monthly parquet snapshots aligned to the NFP release calendar. The model never sees data that wasn't publicly available before the target release date.

**NaN handling (current behavior):**
- LightGBM handles `NaN` natively in training.
- During snapshot-to-row conversion, the loader takes the latest available value as-of cutoff using within-snapshot carry-forward logic.
- NOAA is handled specially: carry-forward is capped to 6 months and explicit staleness features are added.

## Repository Structure

```
NFP_Predictor/
├── run_full_project.py              # Pipeline orchestrator (data + train)
├── settings.py                      # Env vars, paths, logger setup
├── .env                             # API keys and config (gitignored)
│
├── Data_ETA_Pipeline/               # Data ingestion & feature selection
│   ├── fred_employment_pipeline.py  # FRED employment series → snapshots + NFP target
│   ├── load_fred_exogenous.py       # FRED exogenous indicators (VIX, claims, etc.)
│   ├── adp_pipeline.py              # ADP employment → NFP-aligned snapshots
│   ├── noaa_pipeline.py             # NOAA storm data → state-weighted snapshots
│   ├── load_unifier_data.py         # Unifier API (ISM, JOLTS, Consumer Confidence)
│   ├── load_prosper_data.py         # Prosper Trading prediction market data
│   ├── nfp_release_calendar.py      # Historical NFP release dates
│   ├── create_master_snapshots.py   # Merges all sources → master wide-format parquets
│   └── feature_selection_engine.py  # 6-stage feature selection (see below)
│
├── Train/                           # Model training and evaluation
│   ├── train_lightgbm_nfp.py        # Main entrypoint: build dataset, backtest, train
│   ├── config.py                    # Target configs, LightGBM params, path definitions
│   ├── data_loader.py               # Snapshot loading, pivot to wide, feature sanitization
│   ├── feature_engineering.py       # Calendar features (survey week, cyclical month)
│   ├── revision_features.py         # Cross-snapshot revision features
│   ├── model.py                     # LightGBM train/predict, sample weights, model I/O
│   ├── hyperparameter_tuning.py     # Optuna tuning with inner TimeSeriesSplit CV
│   └── Output_code/
│       ├── generate_output.py       # Orchestrates output folder generation
│       ├── model_comparison.py      # Cross-model scorecard (RMSE, MAE, coverage)
│       ├── metrics.py               # RMSE, MAE, MSE computation
│       ├── plots.py                 # Backtest plots, SHAP summaries
│       └── feature_importance.py    # Gain + SHAP feature importance
│
├── scripts/                         # Standalone utilities
│   ├── predict_next_nfp.py          # Production inference for next month
│   ├── check_data_freshness.py      # Dashboard of API data staleness
│   ├── directional_accuracy.py      # Evaluates sign-accuracy of predictions
│   └── revision_analysis.py         # Analyzes BLS revision drift
│
├── utils/
│   ├── paths.py                     # Centralized directory resolution
│   └── transforms.py               # SymLog, inverse_symlog, winsorize_covid_period
│
├── tests/                           # Pytest suite
├── data/                            # Local data storage (gitignored)
├── _output/                         # Model outputs and artifacts (gitignored)
├── _temp/                           # Logs and temp files (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Data Pipeline

### Sources

All sources are defined in [create_master_snapshots.py](Data_ETA_Pipeline/create_master_snapshots.py) (`SOURCES` dict):

| Source | Script | Description |
|--------|--------|-------------|
| FRED Employment (NSA + SA) | `fred_employment_pipeline.py` | ~160 BLS employment series, stored as decade-bucketed parquet snapshots |
| FRED Exogenous | `load_fred_exogenous.py` | Macro indicators (VIX, jobless claims, etc.) |
| ADP | `adp_pipeline.py` | ADP private employment, aligned to NFP release windows |
| NOAA | `noaa_pipeline.py` | Storm damage events weighted by state-level employment |
| Unifier | `load_unifier_data.py` | ISM PMI, JOLTS, Consumer Confidence via Unifier API |
| Prosper | `load_prosper_data.py` | Prediction market data from Prosper Trading |

### Flow: Raw → Model-Ready

1. **Load** — Each source script fetches raw data from its API, cleans it, and writes monthly parquet snapshots to `data/` subdirectories (decade-bucketed).
2. **Prepare** — `create_master_snapshots.py` reads all source snapshots, runs feature selection per source (see below), and writes merged wide-format master snapshots to `data/master_snapshots/{nsa,sa}/{first_release,revised}/decades/`.
3. **Train** — `train_lightgbm_nfp.py` loads master snapshots via `data_loader.py`, pivots them to wide format per month using `pivot_snapshot_to_wide()`, adds calendar features and lagged target features, and produces the final `(X, y)` matrix.

### Data Source Quirks and Mitigations (Critical)

This is the operational contract for point-in-time correctness. If any source logic changes, this section should be updated with the code.

| Source | Key Quirks / Risks | Current Mitigation in Code | Residual Caveat |
|--------|---------------------|----------------------------|-----------------|
| **FRED Employment (NFP tree)** | Historical release metadata is uneven (especially pre-2009). Revisions can leak hindsight if not versioned. | Uses vintage snapshots; release-date imputation rules for older periods (`impute_target_release_date_simple/complex`); strict `< snapshot_date` filtering; shared NFP calendar utilities and hardcoded fallback releases for known edge months. | Earliest historical releases still rely on rule-based imputation, not perfect ground truth timestamps. |
| **FRED Exogenous (daily/weekly/monthly mix)** | Mixed frequency and release timing can cause silent lookahead (especially weekly claims). | Daily series treated as known at `T+1`; weekly claims use vintages + missing-release repair + NFP-window bucketing; monthly aggregates filtered with strict release-date checks; binary regime series excluded from pct-change transforms where mathematically invalid. | Some release-time assumptions are approximations when upstream metadata is incomplete. |
| **ADP** | API month labels + release timestamps require careful date mapping; revisions must not leak. | Reference-month reconstruction with year-boundary handling; strict `release_date < NFP_release_date`; dedupe to latest value known as-of snapshot per `(date, series_name)`. | Dependent on third-party API continuity/schema stability. |
| **NOAA** | Data is published with lag (~75 days after month-end), leaving trailing missing values near present; sparse/late availability can falsely drop useful features. | NOAA release date modeled as month-end + 75 days; strict `<` PIT filtering; feature selection uses source-aware recency (6 months for NOAA vs 3 months default); training adds `__staleness_months` features and caps NOAA carry-forward at 6 months. | Latest 3-4 months can legitimately remain NaN by design; very early snapshots may use best-effort employment-weight proxies if vintages predate ALFRED coverage. |
| **Unifier** | Missing `first_release_date` can be corrupted (`last_revision_date` defaults to timestamp), creating major lookahead bias. | Per-series median publication lag backfill when `first_release_date` missing; ignore corrupted revision timestamp in that case; strict `< snapshot` filtering; additional lookahead guard against month-M leakage; series-specific cleaning (e.g., Housing Starts scale normalization). | Backfilled release dates are estimated when metadata is missing. |
| **Prosper** | Survey schema drift, retired answer choices, and questionnaire redesign (legacy employed vs full/part-time) create unstable series definitions. | Filter unwanted legacy answers; merge full-time + part-time into continuous employed series; drop known erroneous first points; strict `< snapshot` by release timestamp; skip pct-change branch for percentage-based survey values. | Future vendor schema changes can require map updates. |

### Missingness and Carry-Forward Behavior

- Columns with fewer than a configurable threshold of non-NaN values are dropped ([train_lightgbm_nfp.py:`clean_features()`](Train/train_lightgbm_nfp.py)).
- In `pivot_snapshot_to_wide()`, features are converted to one-row as-of snapshots using latest-known values before cutoff.
- Non-NOAA features use uncapped carry-forward inside the snapshot window.
- NOAA features use capped carry-forward (`limit=6`) plus explicit `__staleness_months` features.
- Remaining `NaN`s are passed directly to LightGBM.
- `inf` values are replaced with `NaN` before training.

## Feature Selection

Implemented in [feature_selection_engine.py](Data_ETA_Pipeline/feature_selection_engine.py). Runs **per data source** during the master snapshot creation step. The 6 stages are:

1. **Group-wise Dual Filter** — Purged expanding-window correlation + Random Subspace LightGBM importance.
2. **Boruta** — Shadow-feature permutation test (100 runs, binomial significance).
3. **Vintage Stability** — Exponential recency weighting across historical snapshots to keep temporally stable features.
4. **Cluster Redundancy** — NaN-aware Spearman hierarchical clustering to remove highly correlated survivors.
5. **Interaction Rescue** — Two-phase (single-feature + split-pair) detection to recover dropped features with interaction value.
6. **Sequential Forward Selection** — Walk-forward CV with embargo to finalize the feature set.

Feature selection is run per `{target_type, target_source}` combination. Results are cached as JSON files at `data/master_snapshots/selected_features_{target_type}_{target_source}.json` and loaded at training time via [config.py:`load_selected_features()`](Train/config.py).

Selection uses leakage-safe hard-coded regime cutoffs in `create_master_snapshots.py`:
- `1998-01` Pre-GFC Great Moderation
- `2008-01` GFC Shock + Repair
- `2015-01` Late-Cycle Long Expansion
- `2020-03` COVID Shock + Great Resignation
- `2022-03` Inflation Tightening & Soft Landing
- `2025-02` AI / higher-volatility regime

Important implementation detail:
- Recency screening is source-aware: default 3 months, but NOAA features get 6 months to avoid false rejection from known publication lag.
- If a regime run returns zero features, master generation reuses the prior non-empty regime feature set instead of emitting an empty branch.

### Master Snapshot Orchestration Notes

- `create_master_snapshots.py` runs feature selection per source and per branch (`{nsa, sa} x {first_release, revised}`).
- For short verification windows, branch scope can auto-resolve to revised-only to reduce runtime; long windows run all branches.
- Heavy sources (`FRED_Employment_*`, `FRED_Exogenous`) are run sequentially to avoid RAM OOM, while smaller sources run in parallel workers.
- Feature selection results are cached at three levels: branch cache, per-regime cache, and per-source cache (all with TTL checks) to enable fast reruns and crash recovery.
- Master generation supports progress checkpointing by month so interrupted runs can resume without recomputing completed months.

## Model Training & Backtesting

### Model

- **Primary:** LightGBM (GBDT, MAE objective by default).
- **Hyperparameters:** Defined in [config.py](Train/config.py) (`DEFAULT_LGBM_PARAMS`). Optional Huber loss for outlier robustness (disabled by default since COVID outliers are handled by `winsorize_covid_period()`).
- **Sample weighting:** Exponential decay with configurable half-life (12–120 months), tuned by Optuna. Implemented in [model.py:`calculate_sample_weights()`](Train/model.py).

### Expanding Window Backtest

Implemented in [train_lightgbm_nfp.py:`run_expanding_window_backtest()`](Train/train_lightgbm_nfp.py):

- Marches forward one month at a time (strictly chronological, no K-Fold).
- At each step, the model is fully retrained from scratch using only data before the target month.
- Hyperparameters are re-tuned via Optuna every `TUNE_EVERY_N_MONTHS` (default: 12) months using inner `TimeSeriesSplit` CV on the training partition only.
- Minimum 24 months of training data required before first prediction.
- Backtest window size is controlled by `BACKTEST_MONTHS` (set in `.env`, default: 36).

### Evaluation Metrics

Computed in [metrics.py](Train/Output_code/metrics.py): **RMSE**, **MAE**, **MSE**.

The multi-model comparison scorecard ([model_comparison.py](Train/Output_code/model_comparison.py)) adds coverage statistics and is generated when `--train-all` is used.

### Confidence Intervals

Empirical non-parametric prediction intervals at 50%, 80%, and 95% levels, based on historical out-of-sample residuals. Implemented in [model.py:`predict_with_intervals()`](Train/model.py).

## How to Run

### 1. Environment Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Create .env from the template and fill in your API keys
cp .env.example .env
```

Required environment variables in `.env`:
| Variable | Description |
|----------|-------------|
| `FRED_API_KEY` | FRED API key (required) |
| `UNIFIER_USER` | Unifier API username (required) |
| `UNIFIER_TOKEN` | Unifier API token (required) |
| `DATA_PATH` | Path to data directory (default: `./data`) |
| `START_DATE` | Training start date, e.g. `1990-01-01` |
| `BACKTEST_MONTHS` | Number of months for backtest window (e.g. `36`) |
| `MODEL_TYPE` | `univariate` or `multivariate` |

Optional: `END_DATE`, `OUTPUT_DIR`, `TEMP_DIR`, `DEBUG`, `REFRESH_CACHE`.

### 2. Full Pipeline (recommended)

```bash
# Run everything: fetch data → build master snapshots → train models
python run_full_project.py

# Fresh start (deletes all local data and re-downloads)
python run_full_project.py --fresh
```

### 3. Individual Stages

```bash
# Data collection + preparation only
python run_full_project.py --stage data

# Training only (data must already exist)
python run_full_project.py --stage train

# Training without Optuna tuning (faster)
python run_full_project.py --stage train --no-tune

# Skip slow/optional data sources
python run_full_project.py --skip noaa,prosper

# List all available pipeline steps
python run_full_project.py --list-steps
```

### 4. Direct Training Script

```bash
# Train single model (default: nsa_first)
python Train/train_lightgbm_nfp.py --train

# Train with specific target
python Train/train_lightgbm_nfp.py --train --target sa --release first

# Train all model variants and generate comparison scorecard
python Train/train_lightgbm_nfp.py --train-all

# Train on revised target (uses M+1 FRED snapshot)
python Train/train_lightgbm_nfp.py --train --revised

# Skip Optuna tuning
python Train/train_lightgbm_nfp.py --train --no-tune

# Predict for a specific month
python Train/train_lightgbm_nfp.py --predict 2024-12 --target nsa

# Predict latest available month
python Train/train_lightgbm_nfp.py --latest --target nsa
```

### 5. Production Inference

```bash
python scripts/predict_next_nfp.py --target nsa
python scripts/predict_next_nfp.py --target sa --output report.json
```

## Configuration

Primary configuration is split between `.env` (runtime settings) and [Train/config.py](Train/config.py) (model constants):

**`Train/config.py` key knobs:**
- `ALL_TARGET_CONFIGS` — Which `(target_type, release_type)` combos to train (currently `nsa_first` and `sa_first` only).
- `DEFAULT_LGBM_PARAMS` — LightGBM hyperparameters (learning rate, num_leaves, max_depth, etc.).
- `N_CV_SPLITS` — Inner CV folds for Optuna tuning (default: 5).
- `NUM_BOOST_ROUND` / `EARLY_STOPPING_ROUNDS` — Boosting rounds and early stopping patience.
- `N_OPTUNA_TRIALS` / `OPTUNA_TIMEOUT` — Optuna trial budget (25 trials, 300s timeout).
- `TUNE_EVERY_N_MONTHS` — How often to re-tune during backtest (default: 12).
- `HALF_LIFE_MIN_MONTHS` / `HALF_LIFE_MAX_MONTHS` — Bounds for exponential decay half-life (12–120 months).
- `CONFIDENCE_LEVELS` — Prediction interval levels (`[0.50, 0.80, 0.95]`).

**`settings.py` key knobs:**
- `START_DATE` / `END_DATE` — Date range for data and training.
- `BACKTEST_MONTHS` — Expanding window backtest length.
- `MODEL_TYPE` — `univariate` or `multivariate` (affects target file prefix).

## Outputs & Artifacts

All outputs are written to `_output/` (gitignored). Structure after a full run:

```
_output/
├── models/lightgbm_nfp/         # Saved models (.pkl) + comparison scorecard
├── NSA_prediction/              # NSA backtest results, metrics, plots, SHAP
├── SA_prediction/               # SA backtest results, metrics, plots, SHAP
├── NSA_plus_adjustment/         # NSA with seasonal adjustment overlay
├── Predictions/                 # Forward predictions with confidence intervals
├── benchmark_reports/           # Benchmark analysis outputs
├── revision_analysis/           # BLS revision drift analysis
└── Archive/YYYY-MM-DD_HHMMSS/  # Timestamped archive of each run
```

Per-model output includes:
- Backtest predictions CSV (actual vs. predicted, error, confidence intervals)
- Metrics CSV (RMSE, MAE, MSE)
- Backtest prediction plot
- SHAP summary plot
- Feature importance CSV (gain-based + SHAP)

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=Train --cov=utils
```

Test files cover: config validation, data loading, feature engineering, model math (exponential decay), path resolution, transforms, feature selection engine, and master snapshot creation.

## Troubleshooting

- **Missing `.env` variables:** The pipeline will raise `RuntimeError` on startup if `FRED_API_KEY`, `DATA_PATH`, `UNIFIER_USER`, or `UNIFIER_TOKEN` are not set. Copy `.env.example` to `.env` and fill them in.
- **No `.env.example` file:** Create `.env` manually with the required variables listed in the Configuration section above.
- **Empty `data/` directory:** Run `python run_full_project.py --stage data` first before training.
- **Optuna not installed:** Training will fall back to static `DEFAULT_LGBM_PARAMS`. Install with `pip install optuna`.
- **LightGBM import errors:** Ensure `lightgbm>=4.6` is installed. On macOS, you may need `brew install libomp` first.
- **Feature selection cache miss:** If `selected_features_*.json` is not found, `create_master_snapshots.py` will automatically run the 6-stage feature selection engine (this can be slow on first run).

## License

Private / Internal Use

## Author

Dhruv Kohli
