# Changes since commit b9832e3 (`ship Phase 1 backtest speedups + gated Tier-A scaffolding`, 2026-05-11 22:56:03 -0500)

This document inventories every code change between [b9832e3](.) and the current HEAD that touches `Data_ETA_Pipeline/`, `Train/`, or the helper/configuration files used by the pipeline (`utils/transforms.py`, `settings.py`, `.env`, `.gitignore`), plus the new scripts and tests added under `scripts/` and `tests/`. Intermediate commits in the range:

- `e80a5c5` — best run so far (88 RMSE)
- `4ab1d5c` — current v2 (MAE 90)
- `4deab97` — gitignore: never commit *.zip bundles

Recurring theme across the changes: a PIT-leakage audit (documented in [leakage.md](leakage.md)) closed 11 leakage issues in the ETL + Train pipeline, COVID handling was centralized and stratified through every stage, the consensus-anchor pipeline was simplified to Kalman-only with nested walk-forward CV, and the LightGBM training paths gained a full determinism block.

2026-05-17 addition: `NFP_ENABLE_PANEL_REPLACES_CONSENSUS_KALMAN=1` now emits an experiment-only `consensus_anchor/panel_replaces_consensus_kalman/` bundle during train-all. It uses a dynamic PIT rolling panel from full economist history with configurable default knobs (`8m/top8/80% coverage/median`, Kalman `trailing_window=18`, `nsa_weight_scale=0.40`) while preserving production defaults. The rolling panel is rebuilt with strict current-forecast and prior-actual cutoffs, writes `panel_replacement_pit_audit.csv`, and is listed under `experimental_outputs` in `main_models.json`.

2026-05-17 PIT audit addition: snapshot-wise lags passed strict cutoff checks, but branch-target `nfp_nsa_*` lag/rolling features were not snapshot-derived and needed their own operational-availability mask. `Train/data_loader.py` now masks revised target `y`/`y_mom` unless `operational_available_date < cutoff_date` before lag construction in both batch training and prediction paths. `scripts/audit_current_pipeline_pit.py` writes current-pipeline PIT artifacts; the first audit found 2,157 selected NSA target-derived feature-month values changed from legacy non-null to PIT-missing, so local model/fusion artifacts from before this fix are stale.

---

## 1. `Data_ETA_Pipeline/`

### 1.1 [Data_ETA_Pipeline/adp_pipeline.py](Data_ETA_Pipeline/adp_pipeline.py)

Only the snapshot validator changed; ingestion logic is unchanged.

```diff
-            if max_release <= snap_date:
-                logger.info(f"  Point-in-time: OK (all releases <= snapshot)")
+            if max_release < snap_date:
+                logger.info(f"  Point-in-time: OK (all releases < snapshot)")
             else:
-                logger.error(f"  Point-in-time VIOLATION: {max_release} > {snap_date}")
+                logger.error(f"  Point-in-time VIOLATION: {max_release} >= {snap_date}")
```

**Impact:** Tightens the PIT invariant from `<= snap_date` to strict `< snap_date`. Any same-day release that previously logged as "OK" now correctly logs as a VIOLATION. This is purely diagnostic — the snapshot generator already filters with strict `<` — but it brings the validator into agreement with the actual filter and prevents same-day leakage being silently accepted in future audits.

---

### 1.2 [Data_ETA_Pipeline/feature_selection_engine.py](Data_ETA_Pipeline/feature_selection_engine.py)

```diff
-      * macOS (Darwin) → 1 (the original deadlock fix: LightGBM + ProcessPoolExecutor …)
+      * macOS (Darwin) → 5 (empirically determined sweet spot via
+        ``Train/sandbox/probe_fs_lgbm_njobs.py``: 2.12× speedup vs n_jobs=1
+        on this hardware; n_jobs=6+ regresses due to thread contention.
+        Confirmed not to deadlock — the original ``n_jobs=-1`` issue was
+        specifically about LightGBM running inside ProcessPoolExecutor, not
+        Boruta-internal threading).
…
-    return 1 if platform.system() == "Darwin" else -1
+    return 5 if platform.system() == "Darwin" else -1
```

**Impact:** macOS LightGBM `n_jobs` inside Boruta / random-subspace selection moves from the pessimistic default of 1 to an empirically-tested 5. ~2× speedup of every Stage-2/3 selection pass on the local dev box without re-introducing the historic ProcessPool+LightGBM deadlock (which was specific to nested process pools, not internal LightGBM threading). Linux/CI behavior (`-1`) is unchanged. Override env var `NFP_FS_LGBM_NJOBS` still works.

---

### 1.3 [Data_ETA_Pipeline/load_fred_exogenous.py](Data_ETA_Pipeline/load_fred_exogenous.py)

The "late-start" branch of `_fetch_single_series` (the path used for weekly initial / continued claims series CCNSA, CCSA, ICNSA, ICSA) was rewritten.

**Old logic (single threshold):** any vintage observation whose first realtime_start exceeded `obs_date + 30 days` was treated as a "late start" and replaced with a synthetic obs+7 release date. This collapsed *two* distinct populations into one synthetic value.

**New logic (catalog-gap vs publication-delay disambiguation):**

```python
catalog_start = vintage_df['realtime_start'].min()
catalog_gap = earliest_vintage - earliest_vintage.index
in_catalog_window = earliest_vintage <= (catalog_start + pd.Timedelta(days=14))
gap_is_large = catalog_gap > pd.Timedelta(days=30)
late_start_set = set(earliest_vintage.index[in_catalog_window & gap_is_large])
```

Plus a long comment block explaining the two populations:
- **Population A (pre-2009 ALFRED catalog gap):** uses synthetic obs+7. The value comes from the latest-revised series (`current_df`), which is acknowledged as a small ~1.4–7% bias vs the true first-release value; the alternative would be dropping 42 years (1967–2009) of claims history.
- **Population B (real publication delay — e.g. 2025 government shutdown, 33–61 day delays):** vintage well after `catalog_start` keeps its true late realtime_start; the prior code's blanket obs+7 substitution was forward-looking and is the bug fix.

**Impact:** Closes leakage Issue 4 in [leakage.md](leakage.md). Real publication-delayed claims releases (most visibly the 2025 shutdown weeks) are no longer back-dated to obs+7 in master snapshots; the model can no longer "see" claims weeks earlier than the DOL actually published them. Pre-2009 history retained as before with explicit Population-A trade-off documented.

---

### 1.4 [Data_ETA_Pipeline/noaa_pipeline.py](Data_ETA_Pipeline/noaa_pipeline.py)

Three logically distinct changes in this file.

**(a) Inflation adjustment removed.** The entire `load_cpi_series` helper (≈50 lines) and the CPIAUCSL deflation block in `aggregate_to_state_monthly` were deleted. Damage columns are now nominal USD throughout. `_real` column names retained for backward compatibility with downstream consumers but they no longer carry CPI adjustment. The comment explains:

> Using a single "today's CPI" deflator across all historical rows would embed future CPI knowledge into every snapshot (PIT violation). Tree models split on absolute thresholds and do not require real-dollar normalization; the downstream log1p in `create_weighted_national_aggregates` also compresses long-run scale drift.

**Impact:** Closes a leakage path — every historical snapshot previously inherited "today's" CPI as a multiplier, which is the textbook lookahead-via-shared-deflator pattern. The release-date logic (75-day NOAA lag) is preserved; only the damage scaling changes.

**(b) FRED state-employment download gains exponential-backoff retry.** `download_state_employment_vintages` previously did a single `fred.get_series_as_of_date` per state; a transient 5xx silently dropped that state from the cache. New code retries up to 4 times with 5s/10s/20s/40s backoff:

```python
MAX_ATTEMPTS = 4
BACKOFF_BASE = 5
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        vintage_df = fred.get_series_as_of_date(code, as_of_date=as_of_str)
        break
    except Exception as e:
        if attempt == MAX_ATTEMPTS:
            logger.error(...)
        else:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(wait)
```

After the loop a hard `ValueError` is raised if any state is missing:

```python
got_states = {df['state_code'].iloc[0] for df in all_vintages}
missing = set(fred_codes.keys()) - got_states
if missing:
    raise ValueError(f"FRED state employment download incomplete after retries: …")
```

**Impact:** Eliminates a silent data-quality bug — previously, a single failed state would degrade every downstream weighted national aggregate from a 51-state to a (e.g.) 50-state weighted mean without any warning. The hard fail forces a re-run so the cache is always complete.

**(c) `get_state_employment_weights` rewritten for the ALFRED pre-coverage period.**

Old logic: if `snap_date < earliest_vintage`, fall back to all rows where `realtime_start == earliest_vintage` (using the *global* min across all states). This silently dropped the 44 states whose own earliest vintage was later than the global min (2007-06-19 vs 2005-06-17), producing a 7-state "national" aggregate masquerading as a 51-state one for snap_dates in 2005-06 → 2007-06.

New logic: per-state proxy fallback.

```python
strict_pit = vintages_df[vintages_df['realtime_start'] < snap_date]
pit_states = set(strict_pit['state_code'].unique())
missing_states = expected_states - pit_states

if not missing_states:
    known_df = strict_pit.copy()
else:
    earliest_per_state = vintages_df.groupby('state_code')['realtime_start'].min()
    proxy_mask = (
        vintages_df['state_code'].isin(missing_states)
        & (vintages_df['realtime_start']
           == vintages_df['state_code'].map(earliest_per_state))
    )
    proxy_rows = vintages_df[proxy_mask]
    known_df = pd.concat([strict_pit, proxy_rows], ignore_index=True)
```

A multi-paragraph docstring explains the trade-off (median share drift 0.003 pp, p95 0.025 pp on the data ALFRED does cover) and points at [leakage.md](leakage.md) Issue 5.

**Impact:** Closes leakage Issue 5. NOAA weighted national snapshots for 1990-2007 now use all 51 states rather than a 7-state subset; magnitudes/signs of the weighted storm-damage features change materially for that era. The PIT property is preserved everywhere strict-PIT data is available; the small-bias proxy is documented and bounded.

**(d) Snapshot path keyed by observation month, not release month.** In `create_noaa_weighted_snapshots`:

```diff
-        save_path = get_snapshot_path(NOAA_SNAPSHOTS_DIR, snap_date)
+        save_path = get_snapshot_path(NOAA_SNAPSHOTS_DIR, event_date)
```

with the comment "File is keyed by the OBSERVATION MONTH (matches every other source's convention …)". `snap_date` is still used as the PIT filter for what data goes inside the file; only the filename changes.

**Impact:** Aligns NOAA snapshot filenames with the convention used by every other source (`create_master_snapshots._snapshot_path` looks up by observation month). Previous mismatch silently dropped NOAA features from master snapshots when the loader couldn't find a file under the expected observation-month name. After regeneration, NOAA features participate correctly in feature selection and master aggregation.

---

## 2. `Train/`

### 2.1 [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) (largest change, ~770 lines diffed)

Headline: this file went from "Kalman + AccelOverride + Kalman+AccelPostFilter" to a cleaner final layer centered on Kalman. A later 2026-05 update promoted the PIT Panel/Kalman Router as the second main output; the consensus loader / Optuna tuner / metric stratifier were all rewritten.

#### 2.1.1 Removed approaches

Removed: `accel_override(...)`, `_tune_accel_override(...)`, and the inline Kalman+AccelOverride post-filter loop in `run_consensus_anchor_pipeline`. The associated output bundles (`accel_override/`, `kalman_accel_postfilter/`) are no longer written and the `predictions.csv` augmentation no longer emits their rows. Reasoning recorded in the module docstring:

> AccelOverride and Kalman+AccelPostFilter were removed (2026-05-11) because both consistently underperformed the Consensus baseline on the 60-month backtest window.

**Impact:** ~200 lines of dead inference code deleted. The output bundle contract now centers on `kalman_fusion/` and `panel_kalman_router/`, with `baseline_consensus/` and `panel_consensus_mean/` kept as diagnostics. Downstream `predictions.csv` remains idempotent across this trim — `_augment_predictions_csv` drops any lingering `consensus_anchor_accel_override` / `consensus_anchor_kalman_accel_postfilter` rows from earlier runs.

#### 2.1.2 Consensus loader rewritten — `_latest_snapshot_path` → `_load_consensus_pit`

Old: read the latest Unifier decade-snapshot parquet and `groupby('ds', 'last')` over `NFP_Consensus_Mean`. This carried no PIT enforcement and would have silently leaked if the upstream consensus ever back-updated.

New: walk the master-snapshot directory tree and, for each observation month M, read only the row `date == M` from the M-keyed snapshot. Each master snapshot was already built at ETL time with `release_date < M's NFP release`, so any row inside it is PIT-correct by construction. The reader skips snapshots that don't contain the `NFP_Consensus_Mean` column.

After loading, COVID winsorization is applied:

```python
out_indexed["consensus_pred"] = winsorize_covid_period(
    out_indexed[["consensus_pred"]]
)["consensus_pred"]
```

with an explanation: raw consensus for Apr 2020 was -14,448 vs winsorized SA actual -537, producing an artificial +13,911 residual that inflates the Consensus baseline's MAE and contaminates Kalman `R_c` estimation.

**Impact:** Closes leakage Issue 8 (consensus loader was using a single mutable "latest" Unifier snapshot rather than per-target PIT snapshots) and makes the Consensus-baseline error metric apples-to-apples with the SA actuals that were already pre-winsorized at parquet-write time. Reported Consensus MAE drops materially because the April-2020 outlier is no longer counted at full magnitude.

#### 2.1.3 Kalman noise init + per-step variance estimate are now COVID-clean

The initial-noise prior previously used `cons_hist[-60:]` over *all* consensus history including April 2020:

```diff
-    cons_err = (cons_hist["actual"] - cons_hist["consensus_pred"]).values
-    R_c_init = float(np.var(cons_err[-60:], ddof=1))
-    Q_init = float(np.var(np.diff(cons_hist["actual"].dropna().values[-60:]), ddof=1))
+    first_backtest_ds = df.iloc[0]["ds"] if not df.empty else pd.Timestamp.max
+    prior_cons = consensus_df[
+        consensus_df["consensus_pred"].notna()
+        & consensus_df["actual"].notna()
+        & (consensus_df["ds"] < first_backtest_ds)
+    ]
+    prior_cons_err = (prior_cons["actual"] - prior_cons["consensus_pred"]).values
+    R_c_init = float(np.var(prior_cons_err[-60:], ddof=1)) if len(prior_cons_err) >= 2 else 1.0
+    prior_actuals = prior_cons["actual"].dropna().values[-60:]
+    Q_init = float(np.var(np.diff(prior_actuals), ddof=1)) if len(prior_actuals) >= 2 else 1.0
```

Per-step variance estimation inside the Kalman loop now filters out the COVID months from the trailing window:

```python
if len(hist_valid) >= 6:
    hist_clean = hist_valid[~is_covid_month(hist_valid["ds"])]
    if len(hist_clean) >= 4:
        recent_cons_err = (hist_clean["actual"] - hist_clean["consensus_pred"]).values[-trailing_window:]
        R_c = float(np.var(recent_cons_err, ddof=1)) + 1e-6
        …
    # else: keep R_c, R_m, Q at the most recent estimates (or inits)
```

R_c, R_m, Q, R_a are now stateful across the loop (initialized to the prior-history defaults and updated only when a COVID-clean window of length ≥ 4 is available). The NSA delta-noise branch picked up the same COVID-clean treatment.

**Impact:** Two fixes in one place:
1. Init prior no longer peeks at backtest months (closes a small leakage path where R_c was effectively trained on data the Kalman would later evaluate).
2. Trailing-window noise estimate stops collapsing toward the winsorized COVID values (which sit on a flat plateau and shrink `var(...)` toward zero). The post-COVID Kalman gains are now realistic instead of being clamped by an artificially-small `R_c`.

#### 2.1.4 Optuna tuner gains nested expanding-window CV

Added `_walkforward_cv_score`, `_kalman_fold_runner`, `_composite_kalman_accel_objective`. The Optuna `objective` no longer scores against a single fit of the full overlap; it scores against the mean composite score across `n_splits=5` chronological folds. Each fold's score is computed only over its strictly-future evaluation rows.

**Impact:** Closes leakage Issue 9 — the Optuna "meta-leak" where `trailing_window` and `nsa_weight_scale` were chosen by optimizing against the same actuals they would later be tested on. Selected hyperparameters are now genuine OOS choices. Tuning cost is ~5× higher (one fold ≈ one full Kalman pass over a prefix), partially offset by the smaller search.

#### 2.1.5 Metric stratification (full_metrics + helper)

`full_metrics` now optionally takes a `ds` argument and emits unprefixed/all + `NonCovid_*` + `CovidOnly_*` blocks plus `N_NonCovid` / `N_Covid`. The metric-keys list is fixed and NaN-filled for empty strata so the CSV column set is stable across forecasts.

**Impact:** `comparison_metrics.csv` becomes a true stratified scorecard — non-COVID columns reflect post-winsor model skill, COVID-only columns reflect performance on the three winsorized months in isolation. Lets readers see whether headline gains come from the bulk of the sample or from compressed-residual COVID rows.

---

### 2.2 [Train/Output_code/generate_output.py](Train/Output_code/generate_output.py)

```diff
-        ExpWeightedMonthlyAvgPredictor,
+        ExpWeightedMedianCovidExcludedPredictor,
         load_adjustment_history,
…
-    predictor = ExpWeightedMonthlyAvgPredictor(half_life_years=3.0)
+    predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=3.0)
```

(Two call sites: the `_generate_adjustment_folder` walk-forward backtest and the `_generate_predictions_folder` next-month prediction.)

**Impact:** The seasonal-adjustment delta added to NSA predictions to produce the `NSA_plus_adjustment` forecast now uses an exponentially-weighted *median* (rather than mean) of same-calendar-month adjustments, with Mar–May 2020 COVID rows excluded from history. The median is robust to the residual COVID artifacts and to occasional individual-month outliers; the COVID exclusion stops Apr-2020's +2,433 winsor-artifact from biasing every future April prediction.

---

### 2.3 [Train/Output_code/metrics.py](Train/Output_code/metrics.py)

`compute_metrics` refactored into a private `_metric_block(df, prefix)` helper that returns RMSE/MAE/MSE + variance KPIs over an arbitrary stratum, and the public function now produces three blocks: unprefixed (all), `NonCovid_*`, `CovidOnly_*` — plus N, N_NonCovid, N_Covid counts. NaN-filled blocks for empty strata so the column set is stable.

**Impact:** Mirrors the change in `consensus_anchor_runner.full_metrics`. Every `summary_statistics.csv` from `Output_code/` now carries the same stratified schema; downstream consumers (model_comparison.py / HTML scorecard) can show non-COVID metrics alongside the all-period headline.

---

### 2.4 [Train/hyperparameter_tuning.py](Train/hyperparameter_tuning.py)

```diff
+    LGBM_DETERMINISM,
…
         'verbosity': -1,
-        'random_state': 42,
         'n_jobs': -1,
+        **LGBM_DETERMINISM,
```

Replaces a single `random_state=42` with the full LightGBM determinism block from `config.LGBM_DETERMINISM` (already imported elsewhere). Applied at the two LightGBM-params dicts in the file (Optuna trial-side and final-best refit).

**Impact:** Makes Optuna trials and the final refit byte-identically reproducible across runs (random_state alone is not sufficient for LightGBM; `bagging_seed`, `feature_fraction_seed`, `data_random_seed`, `extra_seed`, `objective_seed`, `deterministic`, `force_col_wise` all need to be set). Eliminates a low-grade source of run-to-run variance in tuned params.

---

### 2.5 [Train/nsa_acceleration.py](Train/nsa_acceleration.py)

Both target loaders now pull the `operational_available_date` column from the parquet:

```diff
-    df = pd.read_parquet(path, columns=["ds", "y_mom", "acceleration"])
+    df = pd.read_parquet(
+        path, columns=["ds", "y_mom", "acceleration", "operational_available_date"]
+    )
…
+    df["operational_available_date"] = pd.to_datetime(
+        df["operational_available_date"], errors="coerce"
+    )
```

`compute_nsa_acceleration_features` gains an optional `cutoff_date` parameter; when provided, revised actuals are filtered by `operational_available_date < cutoff_date` rather than the legacy `ds < target_month`:

```python
if (
    cutoff_date is not None
    and "operational_available_date" in nsa_target.columns
    and "operational_available_date" in sa_target.columns
):
    cutoff_ts = pd.Timestamp(cutoff_date)
    nsa_hist = nsa_target[
        nsa_target["operational_available_date"].notna()
        & (nsa_target["operational_available_date"] < cutoff_ts)
    ].sort_values("ds")
    sa_hist = sa_target[
        sa_target["operational_available_date"].notna()
        & (sa_target["operational_available_date"] < cutoff_ts)
    ].sort_values("ds")
else:
    nsa_hist = nsa_target[nsa_target["ds"] < target_month].sort_values("ds")
    sa_hist = sa_target[sa_target["ds"] < target_month].sort_values("ds")
```

`build_nsa_features_for_training` gains an optional `cutoff_dates: Dict[ds, cutoff]` map and forwards it per row.

**Impact:** Closes leakage Issue 10. Revised NSA / SA actuals were previously filtered by observation date; this allowed the SA training step for target month M to "see" revised values whose revision-release timestamp equalled M's NFP release date (same-day leak). With the operational-availability filter the SA branch only sees revisions that landed strictly before its own cutoff. The legacy filter is preserved as fallback for callers that don't supply `cutoff_dates`.

---

### 2.6 [Train/reduce_features.py](Train/reduce_features.py)

```diff
+    'deterministic': True,
+    'force_col_wise': True,
+    'bagging_seed': 42,
+    'feature_fraction_seed': 42,
+    'data_random_seed': 42,
+    'extra_seed': 42,
+    'objective_seed': 42,
```

added to `BORUTA_LGB_PARAMS`. New comment explains that `random_state` / `seed` are still placeholders (callers override them per iteration for shadow-feature shuffles).

**Impact:** Boruta passes become reproducible up to the per-iteration shuffle seed. Previously the Boruta-internal bagging / column-fraction / data-shuffle seeds were unset, so the same shadow features could still produce different importance rankings across runs on the same iteration.

---

### 2.7 [Train/rerun_post_train_adj_and_consensus.py](Train/rerun_post_train_adj_and_consensus.py)

Two changes:
1. Docstring updated to reflect the new responsibilities (it now refreshes `predictions.csv` for `NSA_plus_adjustment` in addition to rerunning the consensus anchor).
2. New private helper `_refresh_nsa_plus_adjustment_in_predictions_csv` (≈55 lines): reads the freshly regenerated `NSA_plus_adjustment/backtest_results.csv`, computes RMSE + 50/80/95% CI bounds from the last 36 residuals, swaps the `NSA_plus_adjustment` row in `_output/Predictions/predictions.csv` while leaving every other model row untouched.

The `main()` flow gains a step 2 between adjustment regeneration and consensus rerun:
```python
# 2) Refresh the NSA_plus_adjustment row in predictions.csv from the new CSV
_refresh_nsa_plus_adjustment_in_predictions_csv()
```

**Impact:** When iterating on the adjustment predictor without retraining (the common case after Output_code changes 2.2 above), `predictions.csv` no longer gets stuck with the old `NSA_plus_adjustment` row + CI bounds. Both the OOS predicted value and the historical-residual-derived CIs reflect the freshly regenerated backtest.

---

### 2.8 [Train/sandbox/experiment_predicted_adjustment.py](Train/sandbox/experiment_predicted_adjustment.py)

The largest change in the sandbox layer. Structurally:

**(a) Base class `AdjustmentPredictor` now Template-Method.** `fit_predict` is no longer abstract; it strips COVID rows (configurable via `exclude_covid=True`) and delegates to a new abstract `_fit_predict_impl`. All five existing predictors (SARIMA, MonthlyAverage, TwelveMonthComplement, SameMonthLastYear, ExpWeightedMonthlyAvg, LinearRegression) were retrofitted: they pass `exclude_covid` through `super().__init__()` and rename their `fit_predict` to `_fit_predict_impl`.

**(b) New predictor — `ExpWeightedMedianCovidExcludedPredictor`.** Same family as `ExpWeightedMonthlyAvgPredictor` but uses a weight-aware *median* (via `_weighted_median`) instead of a weighted mean. The "_covid_excl" suffix is kept on the class name for backward compatibility with `comparison_metrics.csv`'s Forecast column; the actual COVID exclusion is now done by the base class so the internal `~same_month["ds"].isin(COVID_EXCLUDE_MONTHS)` filter is defensive only.

**(c) New predictor — `LightGBMAdjustmentPredictor`.** Full LightGBM regression on engineered features (`_build_features`): 12 month dummies, linear trend (months since history start), NSA level proxies (`nsa_lvl_12m_mean`, `nsa_lag_1`), same-calendar-month lagged adjustments (lag-1/2/3), and linear lags (adj_lag_1, adj_lag_12). `_fit_predict_impl` builds the feature matrix on (history + target row), trains LightGBM on PIT-filtered rows, predicts the target row; optional `winsorize_covid` clips train-y to non-COVID [1, 99] percentiles before fitting. Falls back to `_monthly_avg_fallback` on any exception or when lag features can't be computed.

**(d) COVID constants moved to `utils.transforms`.** `COVID_EXCLUDE_MONTHS` and `is_covid_month` are now imported rather than defined locally:

```python
from utils.transforms import COVID_EXCLUDE_MONTHS, is_covid_month  # noqa: E402,F401
```

**(e) `main()` updated** to register the two new predictors in the walk-forward backtest.

**Impact:** Closes leakage Issue 6 — Apr 2020's winsor artifact (+2,433 vs raw +50) was previously embedded in every same-month-average adjustment estimate. With the base-class filter, every existing predictor opts in to the COVID-clean history by default; pass `exclude_covid=False` to reproduce the leaky behaviour for an A/B. Production output (Section 2.2) now consumes the median variant, which combined with COVID exclusion was the contributor to the lower RMSE in the morning's run.

---

### 2.9 [Train/short_pass_selection.py](Train/short_pass_selection.py)

```diff
         'random_state': seed,
+        'seed': seed,
+        'bagging_seed': seed,
+        'feature_fraction_seed': seed,
+        'data_random_seed': seed,
+        'extra_seed': seed,
+        'objective_seed': seed,
+        'deterministic': True,
+        'force_col_wise': True,
```

Full determinism block added to `short_pass_lgbm_gain`'s LightGBM params.

**Impact:** Mirrors the change in 2.4 / 2.6 — short-pass gain selection now produces identical feature rankings across runs given the same input seed. Eliminates run-to-run variance in the per-step gain ranking that feeds the candidate pool.

---

### 2.10 [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py)

Two related changes — both make **per-window replay mode take precedence over all-features mode** at both backtest time and production-fit time.

Backtest path:
```diff
-    _per_window_mode = USE_PER_WINDOW_FEATURES and not _all_features_mode
+    _per_window_mode = USE_PER_WINDOW_FEATURES
…
-        if _all_features_mode:
+        if _per_window_mode:
+            _trigger_reselection = False
+        elif _all_features_mode:
             _trigger_reselection = (...)
```

Production-fit path: same logic — the `_prod_per_window_mode` branch is moved ahead of `_prod_all_features_mode` in the `if/elif` ladder, and `_prod_per_window_mode = USE_PER_WINDOW_FEATURES` (no longer ANDed with `not _prod_all_features_mode`). A log line announces the override (`"(overriding ALL-FEATURES mode)"`) when both flags are set.

**Impact:** With `.env`'s new `USE_PER_WINDOW_FEATURES=True` (see 4.1 below), the production / backtest pipeline replays a user-curated per-window feature schedule from disk — even when `RESELECT_EVERY_N_MONTHS` is also non-zero. The schedule from disk is treated as a deliberate, authoritative choice that bypasses both fresh reselection and all-features fall-through. Backstops: if no per-window JSONs are found on disk the flag silently falls back to whatever was previously configured.

---

## 3. Helper / configuration files

### 3.1 [.env](.env)

```diff
-# Dynamic feature re-selection interval (months) during OOS backtest.
-# Set to 0 to disable dynamic reselection.
-RESELECT_EVERY_N_MONTHS=36
+RESELECT_EVERY_N_MONTHS=0
+
+USE_PER_WINDOW_FEATURES=True
```

**Impact:** Dynamic reselection during backtest is now off by default; the pipeline replays a per-window feature schedule from disk (see 2.10 and 3.3 below). Faster backtests (no Boruta / Sequential-Forward-Selection per window) and reproducible feature sets across runs.

---

### 3.2 [.gitignore](.gitignore)

```diff
+
+# Archive bundles — never commit large zips
+*.zip
```

**Impact:** Prevents accidental commit of `_output/Archive/*.zip` bundles.

---

### 3.3 [settings.py](settings.py)

```diff
 RESELECT_EVERY_N_MONTHS = _to_int(_get_env("RESELECT_EVERY_N_MONTHS", "6"))
+USE_PER_WINDOW_FEATURES = _to_bool(_get_env("USE_PER_WINDOW_FEATURES", "False"))
```

**Impact:** Plumbs the new `.env` flag through to a Python constant that `train_lightgbm_nfp.py` reads. Default is `False`, so consumers without the new `.env` retain legacy behaviour.

---

### 3.4 [utils/transforms.py](utils/transforms.py)

```diff
+# Centralized COVID-shock window. The Mar-May 2020 lockdown produced 5+σ outliers
+# in NFP MoM (raw NSA April 2020 = -19,733K; raw SA = -20,471K) …
+COVID_START_DEFAULT = '2020-03-01'
+COVID_END_DEFAULT = '2020-05-01'
+COVID_EXCLUDE_MONTHS = pd.to_datetime([
+    '2020-03-01', '2020-04-01', '2020-05-01',
+])
+
+
+def is_covid_month(ds) -> "pd.Series | np.ndarray":
+    """Return a boolean mask: True where ds is in the COVID exclusion window. …"""
+    ds_dt = pd.to_datetime(ds)
+    if isinstance(ds, pd.Series):
+        return ds_dt.isin(COVID_EXCLUDE_MONTHS)
+    return ds_dt.isin(COVID_EXCLUDE_MONTHS)

 def winsorize_covid_period(
     data: pd.DataFrame | pd.Series,
-    covid_start: str = '2020-03-01',
-    covid_end: str = '2020-05-01',
+    covid_start: str = COVID_START_DEFAULT,
+    covid_end: str = COVID_END_DEFAULT,
```

**Impact:** Single source of truth for the COVID window. Every downstream consumer (`consensus_anchor_runner`, `Output_code/metrics`, `experiment_predicted_adjustment`, the new test) imports `is_covid_month` / `COVID_EXCLUDE_MONTHS` from here, so a future widening / narrowing of the COVID window touches exactly one file.

---

## 4. New files added since b9832e3

### 4.1 [leakage.md](leakage.md) (1864 lines, committed)
The full PIT leakage audit. Documents all 12 issues identified during the audit (7 in ETL, 5 in Train), their status (closed / accepted-with-rationale / verified-not-a-leak), and the empirical bias bounds for the two accepted PIT trade-offs (Population A in load_fred_exogenous, pre-coverage proxy in NOAA weights). Read this for the rationale behind the changes in §1.3, §1.4, §2.1.2-§2.1.4, §2.5, §2.8.

### 4.2 [scripts/reconstruct_nsa_and_kalman.py](scripts/reconstruct_nsa_and_kalman.py) (259 lines, committed)
Recovery script. Used when `--train-all` is interrupted after the NSA branch finishes (model + metrics persisted) but before `generate_all_output` runs. Loads cached NSA training dataset, restores the saved NSA model, reconstructs `nsa backtest_results` from `_output/backtest/*_metrics.json`, predicts the OOS month from the saved model, reuses on-disk SA artifacts unchanged, runs `generate_all_output` and the post-train sandbox + Kalman fusion + archive. Skips SA training entirely.

### 4.3 [tests/test_covid_winsorization.py](tests/test_covid_winsorization.py) (403 lines, committed)
New test module covering the COVID winsorization invariants:
1. centralized constants in `utils.transforms`,
2. `_load_consensus_pit` (post-train consensus loader),
3. Kalman fusion noise estimator's COVID-clean window logic,
4. `compute_metrics` / `full_metrics` stratified output,
5. `AdjustmentPredictor` base-class COVID filter,
6. `X_full` upfront-winsorization symmetry.

Regression suite for the COVID-handling changes scattered through §1.4, §2.1.3, §2.3, §2.8, §3.4.

### 4.4 Untracked scripts under `scripts/`
These three exist on disk but are not committed:

- **scripts/kalman_only.py** — runs only the consensus-anchor (Kalman fusion) stage against whatever's currently in `_output/`, plus archive. Useful for re-running just the Kalman tuner after a tuning hyperparameter change.
- **scripts/nsa_then_kalman.py** — restores the morning archive (`_output/Archive/2026-05-12_093623/`), trains NSA fresh (deterministic with seed=42), regenerates `NSA_prediction/` + `NSA_plus_adjustment/` using fresh NSA + morning-archive SA (no SA retrain), runs the post-train sandbox + Kalman fusion + archive. Intended to reproduce the morning's 88.9 MAE result.
- **scripts/continue_kalman.py** — continuation of `nsa_then_kalman.py` past a logger bug it hit; picks up the freshly-saved NSA model + metrics JSON, reconstructs the NSA backtest df, regenerates `NSA_prediction/` + `NSA_plus_adjustment/` (using morning-archive SA), runs Kalman fusion, archives.

### 4.5 Untracked data folder
- **Best_features_selected/** — feature-set artifacts on disk; the per-window replay schedule that `train_lightgbm_nfp.py` reads when `USE_PER_WINDOW_FEATURES=True` (see §2.10, §3.1, §3.3).

---

## Summary of expected end-to-end impact

| Area | Net effect |
|---|---|
| **Leakage closure** | 4 distinct PIT leaks closed in ETL (claims late-start, NOAA CPI deflator, NOAA weighted-state pre-coverage, NOAA snapshot filename mismatch); 4 in Train (consensus loader, Kalman init prior, Optuna meta-leak, NSA-acceleration same-day revision leak); plus 1 leak fix in the adjustment sandbox (Apr-2020 winsor artifact embedded in same-month means). All documented in [leakage.md](leakage.md). |
| **COVID handling** | Centralized via `utils.transforms.{COVID_EXCLUDE_MONTHS,is_covid_month}`. Consumed by consensus loader (winsorize), Kalman noise estimator (filter), metrics (stratify), adjustment predictors (filter). Headline MAE/RMSE drops because Apr-2020 winsor artifacts are no longer counted at full magnitude. New stratified `NonCovid_*` / `CovidOnly_*` columns in every `summary_statistics.csv`. |
| **Reproducibility** | Full LightGBM determinism block (`deterministic`, `force_col_wise`, plus all five secondary seeds) applied in `hyperparameter_tuning`, `reduce_features`, `short_pass_selection`, and the new `LightGBMAdjustmentPredictor`. Eliminates a known source of run-to-run variance. |
| **Consensus-anchor pipeline** | AccelOverride and Kalman+AccelPostFilter removed. The two main final outputs are now `Kalman_Fusion` and `Panel_Kalman_Router`, with Baseline Consensus and Panel Consensus Mean retained as diagnostics. Optuna uses nested expanding-window CV (5 folds) for Kalman. |
| **Adjustment predictor** | Default production predictor swapped from `ExpWeightedMonthlyAvgPredictor` to `ExpWeightedMedianCovidExcludedPredictor` — median + COVID-clean history. New `LightGBMAdjustmentPredictor` available in the sandbox for further A/B testing. |
| **Feature selection** | macOS `FS_LGBM_NJOBS` default 1 → 5 (~2× faster). Per-window replay mode (driven by `USE_PER_WINDOW_FEATURES=True` in `.env` + `Best_features_selected/` on disk) bypasses fresh reselection entirely in both backtest and production paths, while logging "(overriding ALL-FEATURES mode)" when both flags conflict. |
| **Operational robustness** | NOAA FRED state-employment download now retries with exponential backoff and hard-fails if any state is missing — kills a silent N<51 weighted-aggregate bug. `rerun_post_train_adj_and_consensus.py` now refreshes `predictions.csv` for `NSA_plus_adjustment` so post-train iteration doesn't leave stale OOS rows. |

---

## 2026-05-17 PIT follow-up: final-layer operational actual availability

The final Kalman/panel layer now treats revised actual history the same way the
LightGBM branch-target lag path does: a prior actual is usable only when
`actual_available_date < target_release_date`.

Changed files:

- [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py): carries target release/actual availability dates into the merged final-layer frame; filters Kalman history, adaptive-grid selection, and panel-router rule scoring by strict operational availability; writes `history_available_n` and `latest_available_actual_ds` audit columns.
- [experiments/sidecars/feature_matrix.py](experiments/sidecars/feature_matrix.py): builds sidecar target dynamics from availability-filtered revised target history instead of raw `.shift(1)` when metadata exists.
- [experiments/sidecars/acceleration_classifier_sidecar.py](experiments/sidecars/acceleration_classifier_sidecar.py): uses the PIT-built `*_accel_lag1` feature for target momentum/acceleration composites instead of `actual_accel.shift(1)`.
- [experiments/sidecars/economist_panel_sidecar.py](experiments/sidecars/economist_panel_sidecar.py): filters economist track records to actuals available before the target release cutoff.
- [Train/Output_code/sa_consensus_anchor_runner.py](Train/Output_code/sa_consensus_anchor_runner.py): applies the same operational-availability history filter to the isolated SA challenger fusion.
- [Train/training_dataset_cache.py](Train/training_dataset_cache.py): bumps cache schema to `2` so old pre-fix training matrices are not reused.
- [scripts/audit_current_pipeline_pit.py](scripts/audit_current_pipeline_pit.py): expands the audit to operational Kalman history, sidecar target dynamics, and cache schema.

Verification:

- Focused PIT regression suite: 55 passed.
- Current audit artifacts: `_output_pairing_baseline_pitfix/pit_audit_current_pipeline/`.
- Non-destructive runtime Kalman validation: 59 rows, 0 operational-history violations.
- Official model/fusion CSVs in `_output_pairing_baseline_pitfix` are still stale relative to this code and must be regenerated before their headline metrics are trusted.
