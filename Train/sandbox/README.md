# Sandbox Experiments (No Core Pipeline Changes)

These scripts are intentionally isolated from the production training pipeline.
They do not modify model artifacts under `Train/train_lightgbm_nfp.py`.

## 1) CatBoost SA Revised Backtest

Runs a standalone walk-forward backtest for `sa_first_revised` using CatBoost.

```bash
python Train/sandbox/experiment_catboost_sa_revised.py
```

Outputs:

- `_output/sandbox/catboost_sa_revised/backtest_results.csv`
- `_output/sandbox/catboost_sa_revised/summary_statistics.csv`
- `_output/sandbox/catboost_sa_revised/summary_metrics.json`
- `_output/sandbox/catboost_sa_revised/backtest_predictions.png`
- `_output/sandbox/catboost_sa_revised/summary_table.png`
- `_output/sandbox/catboost_sa_revised/acf_sa_revised.csv`
- `_output/sandbox/catboost_sa_revised/pacf_sa_revised.csv`
- `_output/sandbox/catboost_sa_revised/acf_error_sa_revised.csv`
- `_output/sandbox/catboost_sa_revised/pacf_error_sa_revised.csv`
- `_output/sandbox/catboost_sa_revised/acf_pacf_diagnostics.png`

If CatBoost is not installed, run XGBoost sandbox instead:

```bash
python Train/sandbox/experiment_xgboost_sa_revised.py
```

Enable Optuna tuning for XGBoost walk-forward:

```bash
python Train/sandbox/experiment_xgboost_sa_revised.py --tune --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

Outputs:

- `_output/sandbox/xgboost_sa_revised/backtest_results.csv`
- `_output/sandbox/xgboost_sa_revised/summary_statistics.csv`
- `_output/sandbox/xgboost_sa_revised/summary_metrics.json`
- `_output/sandbox/xgboost_sa_revised/backtest_predictions.png`
- `_output/sandbox/xgboost_sa_revised/summary_table.png`
- `_output/sandbox/xgboost_sa_revised/acf_sa_revised.csv`
- `_output/sandbox/xgboost_sa_revised/pacf_sa_revised.csv`
- `_output/sandbox/xgboost_sa_revised/acf_error_sa_revised.csv`
- `_output/sandbox/xgboost_sa_revised/pacf_error_sa_revised.csv`
- `_output/sandbox/xgboost_sa_revised/acf_pacf_diagnostics.png`

## 2) SA Blend Walk-Forward (SA Direct + NSA+Perfect Adjustment)

Builds a walk-forward dynamic blend using existing output CSVs.

```bash
python Train/sandbox/experiment_sa_blend.py
```

Enable Optuna tuning for blend hyperparameters (`window`, `min_history`, `grid_step`):

```bash
python Train/sandbox/experiment_sa_blend.py --tune --tune-objective composite --tune-trials 25 --tune-cv-splits 4
```

Outputs:

- `_output/sandbox/sa_blend_walkforward/backtest_results.csv`
- `_output/sandbox/sa_blend_walkforward/summary_statistics.csv`
- `_output/sandbox/sa_blend_walkforward/summary_metrics.json`
- `_output/sandbox/sa_blend_walkforward/backtest_predictions.png`
- `_output/sandbox/sa_blend_walkforward/summary_table.png`
- `_output/sandbox/sa_blend_walkforward/acf_sa_revised.csv`
- `_output/sandbox/sa_blend_walkforward/pacf_sa_revised.csv`
- `_output/sandbox/sa_blend_walkforward/acf_error_sa_revised.csv`
- `_output/sandbox/sa_blend_walkforward/pacf_error_sa_revised.csv`
- `_output/sandbox/sa_blend_walkforward/acf_pacf_diagnostics.png`

## 3) LightGBM SA Revised Version Suite (Past-Style -> Current-Style)

Runs sandbox-only SA revised variants on the current feature dataset:

- `v1_legacy_level_only`
- `v2_snapshot_plus_branch`
- `v3_dynamics_tail`
- `v4_delta_amp_stack`

```bash
python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all
```

To enable full Optuna tuning for each walk-forward step of each variant:

```bash
python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all --tune --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

Outputs:

- `_output/sandbox/sa_revised_variants/<variant_id>/...` (same diagnostics bundle)
- `_output/sandbox/sa_revised_variants/suite_summary.csv`
- `_output/sandbox/sa_revised_variants/suite_summary.json`

Note:

- Sandbox feature-building defaults to `JOBLIB_MULTIPROCESSING=0` for stability.
- If your terminal is stable with multiprocessing, override per run:
  `JOBLIB_MULTIPROCESSING=1 python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all`

## 4) Unified SA Revised Comparison (Main + Sandbox + Archive)

Builds one ranking table and comparison plots for all discovered SA revised runs.

```bash
python Train/sandbox/compare_sa_revised_models.py --include-archive --archive-limit 8 --min-backtest-rows 12
```

Outputs:

- `_output/sandbox/sa_revised_comparison/model_metrics_raw.csv`
- `_output/sandbox/sa_revised_comparison/model_metrics_ranked.csv`
- `_output/sandbox/sa_revised_comparison/comparison_summary.txt`
- `_output/sandbox/sa_revised_comparison/compare_error_metrics.png`
- `_output/sandbox/sa_revised_comparison/compare_variance_metrics.png`
- `_output/sandbox/sa_revised_comparison/compare_scores.png`
- `_output/sandbox/sa_revised_comparison/compare_predictions_overlay.png`

## 5) One-Command Suite Runner

Runs all sandbox experiments, then comparison:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py
```

Suite runner with tuned LightGBM variants:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py --tune-variants --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

Suite runner with tuning on XGBoost + blend + LightGBM variants:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py --tune-xgboost --xgb-tune-objective composite --xgb-tune-trials 25 --xgb-tune-every-steps 1 --tune-blend --blend-tune-objective composite --blend-tune-trials 25 --tune-variants --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

## 6) Freeze Baseline Step 1 (Champion/Challenger)

Freeze the current SA revised baseline pair:

- `champion_v1`: sandbox SA blend (`_output/sandbox/sa_blend_walkforward`)
- `challenger_v1`: main SA revised (`_output/SA_prediction_revised`)

```bash
python Train/sandbox/freeze_sa_revised_baseline.py --version v1
```

Outputs:

- `_output/frozen_baselines/sa_revised/champion_v1/`
- `_output/frozen_baselines/sa_revised/challenger_v1/`
- `_output/model_registry/sa_revised/sa_revised_baseline_v1.json`
- `_output/model_registry/sa_revised/sa_revised_baseline_latest.json`

To run full suite then freeze in one command:

```bash
python Train/sandbox/run_sa_revised_sandbox_suite.py --freeze-baseline --freeze-version v1
```
