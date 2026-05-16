# Sandbox experiments

Standalone experiments that **do not** modify the production training pipeline ([`Train/train_lightgbm_nfp.py`](../train_lightgbm_nfp.py)) or any artefacts under [`_output/`](../../_output/) outside of [`_output/sandbox/`](../../_output/sandbox/).

These scripts test alternative models, blends, predicted-adjustment formulations, and consensus-anchor variants without disturbing the main walk-forward. The best ideas from this folder graduate into the production [`Train/Output_code/consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py).

> **Context as of 2026-05.** The production SA-revised forecast is the Kalman fusion (consensus + NSA + adjustment + NSA-acceleration channel). A standalone SA LightGBM is no longer trained — see [`../../README.md`](../../README.md) §10 for the architecture. Several SA-focused scripts in this folder (`compare_sa_revised_models.py`, `freeze_sa_revised_baseline.py`, `experiment_*_sa_revised*.py`) are kept for historical comparison and ablation studies but are not part of the production critical path. **`experiment_predicted_adjustment.py` is still on the critical path** because it provides the half-life predictor that the Kalman fusion tunes inside its joint Optuna search.

> **Multiprocessing note.** Sandbox feature-building defaults to `JOBLIB_MULTIPROCESSING=0` for stability on macOS. If your terminal is stable with multiprocessing, override per run, e.g.:
>
> ```bash
> JOBLIB_MULTIPROCESSING=1 python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all
> ```

---

## Inventory

### Critical-path (still consumed by production)

| Script | Purpose |
|---|---|
| [`experiment_predicted_adjustment.py`](experiment_predicted_adjustment.py) | Walk-forward PIT-safe predictor for the seasonal-adjustment factor `SA_MoM − NSA_MoM`. Tests SARIMA, monthly average, 12-month complement, same-month-last-year, exponentially-weighted monthly average, and linear regression with month dummies. The `ExpWeightedMedianCovidExcludedPredictor` class is imported by the Kalman fusion's joint Optuna tune ([`consensus_anchor_runner.py:_tune_kalman`](../Output_code/consensus_anchor_runner.py)). |
| [`output_utils.py`](output_utils.py) | Shared `write_sandbox_output_bundle` writer used by every sandbox script for consistent diagnostic bundles (backtest CSV, summary stats, ACF/PACF, plot, table). |

### Research / ablation (SA-focused; SA LightGBM retired)

| Script | Purpose |
|---|---|
| [`experiment_catboost_sa_revised.py`](experiment_catboost_sa_revised.py) | CatBoost walk-forward backtest for `sa_first_revised`. |
| [`experiment_xgboost_sa_revised.py`](experiment_xgboost_sa_revised.py) | XGBoost walk-forward backtest, optional Optuna tuning. |
| [`experiment_lgbm_sa_revised_variants.py`](experiment_lgbm_sa_revised_variants.py) | Four LightGBM variants (`v1`–`v4`) on the SA-revised feature set. |
| [`experiment_sa_blend.py`](experiment_sa_blend.py) | Walk-forward dynamic blend of SA Direct + NSA+Adjustment. Used to be a Kalman channel; now kept as the fallback champion in [`consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py). |
| [`explore_anchor_plus_direction.py`](explore_anchor_plus_direction.py) | Research only: SA anchor + NSA direction nudge. |
| [`explore_consensus_anchor.py`](explore_consensus_anchor.py) | Research only: replace SA anchor with the LSEG consensus poll. The seed of the production fusion. |
| [`compare_sa_revised_models.py`](compare_sa_revised_models.py) | Unified ranking table across main + sandbox + archive runs. |
| [`run_sa_revised_sandbox_suite.py`](run_sa_revised_sandbox_suite.py) | One-command runner: every sandbox experiment, then the comparison. |
| [`freeze_sa_revised_baseline.py`](freeze_sa_revised_baseline.py) | Champion / challenger baseline freezer (immutable `_output/frozen_baselines/` snapshots). |

### Engineering / diagnostics

| Script | Purpose |
|---|---|
| [`probe_fs_lgbm_njobs.py`](probe_fs_lgbm_njobs.py) | Sweeps `n_jobs` candidates for the FS-engine Boruta loop in a subprocess (so deadlocks surface as timeouts). Helps choose a safe `FS_LGBM_NJOBS` default per host. |
| [`validate_tiered_reselection.py`](validate_tiered_reselection.py) | Validates that the Tier-A cached universe (`USE_UNIVERSE_CACHE=True`) produces feature sets close enough to the full-reselection baseline to ship. Enforces the PIT invariant `universe_asof ≤ step_date` with hard asserts. Gated until median Jaccard ≥ 0.70 and median MAE-delta-proxy ≤ 5%. |

Each experiment writes its full diagnostic bundle (backtest CSV, summary stats, summary metrics JSON, ACF/PACF tables, prediction plot, summary table image, ACF/PACF diagnostic plot) under `_output/sandbox/<experiment_id>/`.

---

## 1. Predicted seasonal adjustment (critical path)

```bash
python Train/sandbox/experiment_predicted_adjustment.py
```

Walk-forwards multiple adjustment models against `SA_MoM − NSA_MoM` with no lookahead, then combines NSA predictions with the predicted adjustment to produce SA-space forecasts. Models tested:

1. SARIMA(1,0,1)(1,1,1)<sub>12</sub>
2. Monthly average (same calendar month, all prior years)
3. 12-month complement (last 11 months summed, negated)
4. Same-month last year (naive seasonal)
5. **Exponentially-weighted monthly average / median** with COVID exclusion — the production class consumed by the Kalman fusion's joint Optuna tune
6. Linear regression with month dummies + trend

Output: `_output/sandbox/nsa_predicted_adjustment_revised/`.

**Why this matters.** The Kalman fusion's `half_life_years` parameter (tuned ∈ [0.5, 8.0] years) drives the recency weighting in `ExpWeightedMedianCovidExcludedPredictor` — see [`consensus_anchor_runner.py:_compute_adjustment_series`](../Output_code/consensus_anchor_runner.py). Changing the class here changes what the fusion can express.

---

## 2. CatBoost / XGBoost SA-revised backtest

```bash
python Train/sandbox/experiment_catboost_sa_revised.py
# or
python Train/sandbox/experiment_xgboost_sa_revised.py
```

Optuna tuning for XGBoost against the composite objective:

```bash
python Train/sandbox/experiment_xgboost_sa_revised.py \
  --tune --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

If CatBoost is not installed, the XGBoost script is the recommended fallback.

**Output bundle (per experiment, e.g. `xgboost_sa_revised`).**

```
_output/sandbox/<experiment_id>/
├── backtest_results.csv
├── summary_statistics.csv
├── summary_metrics.json
├── backtest_predictions.png
├── summary_table.png
├── acf_sa_revised.csv         pacf_sa_revised.csv
├── acf_error_sa_revised.csv   pacf_error_sa_revised.csv
└── acf_pacf_diagnostics.png
```

---

## 3. SA-blend walk-forward (SA Direct + NSA+Adjustment)

Builds a walk-forward dynamic blend from existing main-pipeline backtest CSVs — no retraining.

```bash
python Train/sandbox/experiment_sa_blend.py
```

Tune blend hyperparameters (`window`, `min_history`, `grid_step`):

```bash
python Train/sandbox/experiment_sa_blend.py \
  --tune --tune-objective composite --tune-trials 25 --tune-cv-splits 4
```

Output: `_output/sandbox/sa_blend_walkforward/` (same diagnostics bundle).

**Production note.** Used to be a Kalman channel; now superseded by NSA+Adjustment directly. Still kept as the **fallback champion** in [`consensus_anchor_runner.py:build_merged_dataset`](../Output_code/consensus_anchor_runner.py) when the NSA+Adjustment CSV isn't available.

---

## 4. LightGBM SA-revised variant suite

Four sandbox-only LightGBM variants on the current SA-revised feature set:

- `v1_legacy_level_only`
- `v2_snapshot_plus_branch`
- `v3_dynamics_tail`
- `v4_delta_amp_stack`

```bash
# Run all four with default hyperparameters
python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all

# Full Optuna tuning per walk-forward step, per variant
python Train/sandbox/experiment_lgbm_sa_revised_variants.py \
  --variants all --tune --tune-objective composite \
  --tune-trials 25 --tune-every-steps 1
```

**Outputs:**

```
_output/sandbox/sa_revised_variants/
├── <variant_id>/...           # Per-variant diagnostics bundle
├── suite_summary.csv
└── suite_summary.json
```

These are kept for ablation analysis even though the SA LightGBM is no longer in production — useful when investigating whether a particular feature subset or amplitude calibration would have helped.

---

## 5. Consensus / anchor explorations

Research-only scripts that informed the production [`consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py). They reuse existing backtest CSVs and the latest Unifier consensus snapshot — no retraining.

```bash
python Train/sandbox/explore_anchor_plus_direction.py   # SA anchor + NSA direction nudge
python Train/sandbox/explore_consensus_anchor.py        # Consensus poll as anchor (seed of production)
```

Both print metrics to stdout; neither writes to `_output/` (they are intentionally side-effect-free).

---

## 6. Unified SA-revised comparison

Builds one ranking table and overlay plots across main, sandbox, and archived SA-revised runs:

```bash
python Train/sandbox/compare_sa_revised_models.py \
  --include-archive --archive-limit 8 --min-backtest-rows 12
```

**Outputs:**

```
_output/sandbox/sa_revised_comparison/
├── model_metrics_raw.csv
├── model_metrics_ranked.csv
├── comparison_summary.txt
├── compare_error_metrics.png
├── compare_variance_metrics.png
├── compare_scores.png
└── compare_predictions_overlay.png
```

---

## 7. One-command suite runner

Run every sandbox experiment, then the comparison table:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py
```

With tuned LightGBM variants:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py \
  --tune-variants --tune-objective composite \
  --tune-trials 25 --tune-every-steps 1
```

Full tuning across XGBoost + blend + LightGBM variants:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py \
  --tune-xgboost --xgb-tune-objective composite \
  --xgb-tune-trials 25 --xgb-tune-every-steps 1 \
  --tune-blend --blend-tune-objective composite --blend-tune-trials 25 \
  --tune-variants --tune-objective composite \
  --tune-trials 25 --tune-every-steps 1
```

Run the full suite, then freeze the SA-revised baseline pair:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py \
  --freeze-baseline --freeze-version v1
```

---

## 8. Freeze baseline (champion / challenger)

Freezes the current SA-revised baseline pair to immutable snapshot folders:

- `champion_v2+` → sandbox SA blend (`_output/sandbox/sa_blend_walkforward`)
- `challenger_v2+` → raw SA-revised LightGBM (`_output/SA_prediction_revised`, falling back to `_output/SA_prediction`)

```bash
python Train/sandbox/freeze_sa_revised_baseline.py --version v1
```

**Outputs:**

```
_output/frozen_baselines/sa_revised/champion_v1/
_output/frozen_baselines/sa_revised/challenger_v1/
_output/model_registry/sa_revised/sa_revised_baseline_v1.json
_output/model_registry/sa_revised/sa_revised_baseline_latest.json
```

> Frozen baselines snapshot 14 standard files (backtest_results.csv, summary_*, feature_importance.csv, blend_config.json, plots, ACF/PACF tables and diagnostics) with SHA-256 hashes recorded in the registry JSON.

---

## 9. Engineering probes

### Probe FS LightGBM `n_jobs`

```bash
python Train/sandbox/probe_fs_lgbm_njobs.py
python Train/sandbox/probe_fs_lgbm_njobs.py --candidates 1 2 4 8
python Train/sandbox/probe_fs_lgbm_njobs.py --timeout 300 --runs 3
```

[`Data_ETA_Pipeline/feature_selection_engine.py:LGB_PARAMS`](../../Data_ETA_Pipeline/feature_selection_engine.py) historically pins `n_jobs=1` because LightGBM + `ProcessPoolExecutor` + `n_jobs=-1` deadlocked on macOS. This probe runs a representative Boruta workload in a subprocess (so deadlocks surface as timeouts) and records wall-clock + peak RSS. Output: `_output/sandbox/fs_njobs_probe/report.csv`.

### Validate tiered reselection

```bash
python Train/sandbox/validate_tiered_reselection.py --target nsa
python Train/sandbox/validate_tiered_reselection.py --target nsa \
  --step-dates 2023-06,2023-12,2024-06,2024-12 \
  --universe-asof 2023-01
python Train/sandbox/validate_tiered_reselection.py --target nsa --quick
```

Validates that Tier-A (cached Pass-1 universe) + Tier-B (fresh Pass-2) produces feature sets close enough to baseline full-reselection to ship.

- **Promotion rule.** Median Jaccard ≥ 0.70 across step_dates AND median |MAE-delta-proxy| ≤ 5%.
- **PIT invariant.** Every Tier-A cache used at step_date `t` must have `universe_asof ≤ t`. Hard `AssertionError` if violated, so we cannot accidentally validate a leaking scheme.
- **Gate status.** [`USE_UNIVERSE_CACHE`](../config.py) stays `False` until this sandbox confirms parity vs baseline.

Output: `_output/sandbox/tiered_reselection/{target}_{source}_report.csv`.

---

## Output bundle conventions

Every sandbox experiment that produces results uses `output_utils.write_sandbox_output_bundle`, which writes:

```
_output/sandbox/<experiment_id>/
├── backtest_results.csv         # ds, actual, predicted, error (+ extras)
├── summary_statistics.csv       # MAE, RMSE, MSE, Corr, std ratios, AccelAcc, DirAcc, …
├── summary_metrics.json         # Same metrics as a flat dict (for programmatic consumption)
├── backtest_predictions.png     # Actual vs predicted overlay with confidence shading
├── summary_table.png            # Rendered metrics table image
├── acf_<id>.csv  pacf_<id>.csv  # ACF/PACF of the actual series
├── acf_error_<id>.csv pacf_error_<id>.csv   # ACF/PACF of the residuals
└── acf_pacf_diagnostics.png     # 2×2 diagnostic plot
```

This is the same bundle shape used by [`Train/Output_code/consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py) for its `kalman_fusion/` and `baseline_consensus/` directories — so cross-comparison via `compare_sa_revised_models.py` works uniformly across main, sandbox, and archive runs.
