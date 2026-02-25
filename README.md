# NFP Predictor

A machine learning pipeline for forecasting U.S. Non-Farm Payrolls (NFP) month-over-month employment changes. It uses LightGBM, expanding-window walk-forward validation, and multimodal data sources (FRED, ADP, NOAA, Unifier, Prosper). 

The architecture is explicitly designed around **point-in-time (PIT) correctness** and **regime-aware feature selection** to mathematically prevent lookahead bias and handle high-dimensional macroeconomic noise.

## 1. The NFP Challenge: Why This is Hard

Forecasting NFP is notoriously difficult for quantitative models due to four structural realities:
1. **Aggressive Revisions:** The initial Bureau of Labor Statistics (BLS) release is heavily revised in subsequent months. Models trained naively on "finalized" historical data will suffer severe lookahead bias by assuming they had clean data that didn't exist in real-time.
2. **Asynchronous Availability:** Economic indicators are published at mismatched frequencies (daily, weekly, monthly) with varying lag times (e.g., NOAA storm data arrives ~75 days late). Aligning these without peeking into the future requires rigorous data versioning.
3. **Regime Non-Stationarity:** The economy fundamentally breaks its own rules. Relationships that held during the "Great Moderation" (pre-2008) often decoupled or inverted during the Global Financial Crisis (GFC) or the 2020 COVID-19 shock. 
4. **High-Dimensional Instability:** Thousands of macroeconomic series contain spurious or transient correlations. Without temporally-aware feature selection, complex models will catastrophically overfit to this noise.

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

## 3. How the Pipeline Works (Conceptual Flow)

The pipeline operates as a rigid, time-aware simulation of a quantitative trader predicting real-time employment changes. At each forecast month $t$, it executes the following logic:

1. **Point-In-Time (PIT) Shielding:** The system artificially steps back to month $t$ and completely blinds the data loaders to any information, backfills, or revisions published *after* $t$. All available indicators are saved into a single "Master Snapshot".
2. **Regime-Aware Feature Selection:** To combat high-dimensional noise, the system runs a 6-stage feature selection tournament independently across distinct historical eras. It mathematically purges indicators that are collinear, structurally unstable, or fundamentally useless.
3. **Expanding-Window Validation:** Starting from 1990, the model trains on $[0 ... t-1]$, predicts month $t$, records its out-of-sample error, and seamlessly rolls the window forward one month. It continuously retrains and self-tunes its hyperparameters via inner-fold cross-validation.
4. **Real-time Diagnostics & Keep-Rules:** The system evaluates the model's trailing out-of-sample error against naive baselines (e.g., repeating the last value). If the ML degrades below this "dumb" baseline, the system throws a safety flag and stops deployment.

---

## 4. Point-In-Time (PIT) Data Integrity

**Target Space:** The pipeline models the month-over-month change in U.S. Non-Farm Payrolls (`y_mom`). The *First Release* (the initial, unrevised BLS estimate) is the critical ground-truth target for live trading simulation.

**As-of Mathematical Constraint:** Every individual feature $X_i$ mapped to a prediction month $t$ is strictly filtered by the inequality: `feature_release_date < target_release_date(t)`. 

### Expected Quirk Resolution
Because vendor APIs are deeply flawed for temporal ML, the pipeline implements source-level fixes before merging data into the monthly Master Snapshot:

- **FRED Employment:** Employs vintage ALFRED snapshots where available. Missing pre-2009 release dates use carefully constructed release-date heuristic imputation logic.
- **FRED Exogenous (Weekly):** Weekly jobless claims cause enormous leakage if not handled. Mapped using vintage-based backfills and NFP-reference-week bucketing logic.
- **NOAA (Weather):** Destructive sparsity (storm data publishes ~75 days late). Modeled formally as `month-end + 75 days`, allowing NOAA variables to safely bypass standard recency validation filters.
- **Unifier (Surveys):** Vendor actively overwrites expected `first_release_date` with today's API execution timestamp. Repaired dynamically by computing the historical median publication lag per series and back-projecting the true timestamp.

---

## 5. Handling Economic Shocks & Training Scope

To maintain mathematical stability across massive economic dislocations (e.g., 2008 GFC, 2020 COVID shock), the pipeline explicitly transforms inputs:

- **COVID Winsorization:** Extreme outliers during the Spring 2020 COVID shock are explicitly winsorized (`utils/transforms.py`) to their non-COVID distribution quantiles. This preserves the directionality and sequence of the shock without obliterating the MSE loss function geometry with scale destruction.
- **Symmetric Log Transforms:** Heavy-tailed features optionally undergo a SymLog transform (`np.sign(x) * np.log1p(np.abs(x))`), compressing extreme kurtosis while gracefully handling negative bounds.
- **Dynamic Z-Scores:** High-velocity indicators utilize 3-month and 12-month rolling Z-scores (`Load_Data/load_fred_exogenous.py`), capturing acceleration and relative momentum rather than relying on brittle absolute levels.

**Post-1990 Anchor:** During `load_snapshot_wide`, the matrix enforces an explicit `X.index >= '1990-01-01'` boundary. While older macro data exists, pre-1990 datasets exhibit massive NaN sparsity across peripheral survey metrics; anchoring to 1990 removes empty cycles and ensures matrix density.

---

## 6. The Feature Selection Engine

Because regimes break structurally over 30 years, feature selection is run independently per historical era (e.g., pre-GFC, GFC, post-COVID). 

Implemented in `Data_ETA_Pipeline/feature_selection_engine.py`, the massive feature space is pruned via 6 mathematical stages:

1. **Group-wise Dual Filter:** Fast purge via expanding-window correlation combined with Random Subspace LightGBM non-linear importances.
2. **Boruta Generation:** A shadow-feature randomized permutation test executed over 100 runs using binomial significance to isolate true predictive power from noise.
3. **Vintage Stability Check:** Exponential recency weighting applied across historical snapshots to reject features that shift conceptually or statistically over time. 
4. **Cluster Redundancy:** NaN-aware Spearman hierarchical clustering to algorithmically drop colinear survivor branches, collapsing redundancy.
5. **Interaction Rescue:** Two-phase structural scans (single-feature + split-pair impact) to rescue dropped features that the base tree algorithms rely heavily on in paired branching.
6. **Sequential Forward Selection:** Walk-forward cross-validation with embargo rules to hone the final mathematical set.

---

## 7. Model Training & Validation

The core simulation is an **Expanding Window Validation** (`Train/train_lightgbm_nfp.py`). Marches forward one month at a time chronologically, strictly simulating real-time decisions:

- **Sample Weighting Base:** Exponential time-decay with a configurable half-life dynamically tuned by Optuna, ensuring recent structural economic realities explicitly outweigh distant memory.
- **Union Candidate Pool:** Feature survivors from the ETL processes are united inside a cache layer (`Train/candidate_pool.py`).
- **Short-Pass Filtering:** During the walk-forward simulation, an ultra-fast secondary heuristic (`Train/short_pass_selection.py`) isolates the top $K$ (e.g., 60) most valuable features available *exactly* at that time step, guaranteeing the final model matrix is dense and optimal.

---

## 8. End-User Output Artifacts

A fully completed pipeline produces **exactly 4 independent model variants**, each with complete diagnostics and performance metrics:

### Multi-Model Output Structure

The pipeline trains and validates all four target/release combinations in parallel:

| Model Variant | Target | Release | Description |
|---------------|--------|---------|-------------|
| **NSA first_release** | Not Seasonally Adjusted | First (Initial BLS Estimate) | Real-time operational prediction; highest deployment relevance |
| **NSA revised** | Not Seasonally Adjusted | Revised (Final BLS Revision) | Ultimate predictability ceiling; used for performance benchmarking |
| **SA first_release** | Seasonally Adjusted | First (Initial BLS Estimate) | Cleaned seasonal effects, real-time |
| **SA revised** | Seasonally Adjusted | Revised (Final BLS Revision) | Cleaned seasonal effects, ultimate benchmark |

Each variant is trained with its own expanding-window backtest, hyperparameter tuning, and baseline comparisons, producing isolated artifacts.

### Output Directory Structure

A fully completed pipeline deposits actionable quant metrics into `_output/`:

1. **Predictions Dataset (`Predictions/`):** Four CSVs (one per variant), each containing:
   - True vs. out-of-sample predicted NFP numbers
   - Comparative naive baseline predictions (e.g., prior month carryover, VAR, AR1)
   - Empirical non-parametric confidence bounds (50%, 80%, 95%) anchored to rolling residual dispersion
   - Per-month walk-forward backtest dates and expanding window boundaries

2. **Metrics Scorecard (`models/lightgbm_nfp/`):** Four scorecard sets including:
   - Out-of-sample RMSE, MAE, MAPE per variant
   - Prediction interval coverage ratios (50%, 80%, 95% bands)
   - Keep-rule diagnostics and baseline comparison rankings
   - Optuna hyperparameter optimization history

3. **Diagnostics & Feature Causality (`NSA_prediction/` / `SA_prediction/`):** Per-variant subdirectories containing:
   - Visual diagnostic plots (residual distributions, rolling error regimes, forecast horizon drift)
   - Structured SHAP (SHapley Additive exPlanations) datasets highlighting which macroeconomic indicators mathematically triggered each monthly inference
   - Feature importance rankings and stability across time
   - Residual ACF/PACF plots for autocorrelation diagnostics

---

## 9. Operational Mechanics
*(Engineering usage details for installation and execution)*

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

### 2. Full Pipeline (Recommended)

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

# Predict latest available month
python Train/train_lightgbm_nfp.py --latest --target nsa
```

### 5. Production Inference

```bash
python scripts/predict_next_nfp.py --target nsa
python scripts/predict_next_nfp.py --target sa --output report.json
```

## Configuration Knobs

Primary configuration is set in `.env` and `Train/config.py`:
- `DEFAULT_LGBM_PARAMS` — LightGBM baseline hyperparameters.
- `TUNE_EVERY_N_MONTHS` — Optuna backtest execution pacing (default: 12).
- `HALF_LIFE_MIN_MONTHS` / `HALF_LIFE_MAX_MONTHS` — Sample decay weight constraints.
- `KEEP_RULE_ENABLED` — Abort saving production models if baseline tests fail.
- `BACKTEST_MONTHS` — Backtesting horizon depth.

## Troubleshooting

- **Missing `.env` variables:** Pipeline fails on startup if required API keys are missing.
- **Optuna errors:** Ensure `optuna` is installed. Pipeline defaults to static weights otherwise.
- **LightGBM missing libs:** On macOS, `brew install libomp` may be necessary. 
- **Empty `data/` directory:** Ensure `python run_full_project.py --stage data` is executed to build foundational shards before machine learning execution begins.

## License

Private / Internal Use

## Author

Dhruv Kohli
