# NFP Predictor: Claude Guidelines

## Project Context
This is an institutional-grade macroeconomic forecasting pipeline designed to predict the U.S. Non-Farm Payrolls (NFP) month-over-month (MoM) change.
- **Core Challenge:** Predicting a highly scrutinized economic indicator using massive, uneven datasets (e.g., FRED has 49k features, others have 2k-8k).
- **Critical Requirement:** **Strict Point-In-Time (PIT) correctness.** Lookahead bias is the ultimate enemy. Data must *only* be used if it was publicly available prior to the NFP release date for the target month.

## Architecture & Data Flow
The pipeline has a distinct lifecycle:
1.  **Ingestion & Structuring (`Data_ETA_Pipeline/`):** Raw data from various sources (FRED, ADP, NOAA, Unifier, Prosper) is loaded, cleaned, transformed, and organized into monthly parquet snapshots. Legacy folders like `Load_Data/` or `Prepare_Data/` are deprecated.
2.  **Feature Selection (`Data_ETA_Pipeline/feature_selection_engine.py`):** A rigorous 6-stage engine (Variance Filter -> Dual Filter [Purged Corr + Random Subspace LightGBM] -> Boruta -> Vintage Stability -> Cluster Redundancy -> Interaction Rescue -> Sequential Forward Selection).
3.  **Master Aggregation (`Data_ETA_Pipeline/create_master_snapshots.py`):** Combines the surviving features from all sources into a single wide-format `.parquet` file per month. Operates in a quad-track mode: `{nsa, sa} x {first_release, revised}`.
4.  **Training & Evaluation (`Train/`):**
    - **Expanding Window Backtest:** We use strictly chronological expanding windows (not K-Fold CV) to evaluate our model's true, out-of-sample trading performance. The model is retrained from scratch each month using only historical data prior to that month.
    - **Native `NaN` Handling:** The primary model is LightGBM because it mathematically handles the `NaN`s in our staggered historical wide-format datasets without requiring imputation.
    - **Baselines:** We track performance against raw `NFP_Prior_Month` and several smart linear baselines (VAR, AR1, and specific macroeconomic shock identifiers).
    - **Multi-Model Scorecard:** `train_lightgbm_nfp.py --train-all` is the command to run all 4 branches sequentially. It feeds the results to `Output_code/model_comparison.py`, generating an extensive comparative scorecard (RMSE, MAE, coverage) in CSV and styled HTML formats.

## Coding Conventions
- **Language/Frameworks:** Python 3.x, pandas, numpy, scikit-learn, LightGBM, Optuna.
- **Data Handling:** 
    - Rely on LightGBM's native `NaN` handling. Never forward-fill (`ffill()`) or impute `NaN`s unless strictly mathematically justified (e.g., within cluster redundancy checks).
    - Features must be sanitized (e.g., stripping JSON-forbidden characters) before being passed to LightGBM. Use existing `_safe_lgb_fit` and `_safe_lgb_predict` helpers.
    - Be incredibly careful with `DatetimeIndex` vs. `date` column assumptions when merging dataframes.
- **Efficiency:** We routinely deal with $p \gg n$ datasets (49k features vs. 1k rows). All new algorithms must be highly vectorized, computationally bounded (e.g., using random subspaces or tournament chunking), and fail gracefully.
- **Logs:** Use the existing `logger` configuration from `settings.py`. Do not use basic `print()`.

## Interaction Style
- **Proactive & Architectural:** When asked to implement a feature or fix a bug, don't just patch the immediate issue. Consider the architectural implications, especially concerning lookahead bias, execution time (efficiency), and multi-processing stability.
- **Data Integrity Above All:** If a proposed change introduces even a slight risk of forward-looking data leakage, flag it immediately and propose a safer alternative.
- **Clear Explanations:** When discussing financial/macro modeling concepts (e.g., VIF, Boruta, Symlog), provide clear, expert-level explanations of *why* we are doing it, not just *how*.
- **Comprehensive Edits:** Provide full, copy-pasteable blocks of code for modifications, complete with context lines, to avoid editing errors. Use the `/slash` commands when applicable if working within an IDE. 
