# Sandbox Experiments

Standalone experiments that **do not** modify the production training pipeline ([`Train/train_lightgbm_nfp.py`](../train_lightgbm_nfp.py)) or any artefacts under [`_output/`](../../_output/) outside of [`_output/sandbox/`](../../_output/sandbox/).

These scripts exist to test alternative models, blends, and consensus-anchor formulations against the SA revised target without risking the main backtest. The best ideas from this folder graduate into the production [`Train/Output_code/consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py).

> **Multiprocessing note:** sandbox feature-building defaults to `JOBLIB_MULTIPROCESSING=0` for stability on macOS. If your terminal is stable with multiprocessing, override per run, e.g.:
> ```bash
> JOBLIB_MULTIPROCESSING=1 python Train/sandbox/experiment_lgbm_sa_revised_variants.py --variants all
> ```

---

## Inventory

| Script | Purpose |
|---|---|
| [`experiment_catboost_sa_revised.py`](experiment_catboost_sa_revised.py) | CatBoost walk-forward backtest for `sa_first_revised` |
| [`experiment_xgboost_sa_revised.py`](experiment_xgboost_sa_revised.py) | XGBoost walk-forward backtest, optional Optuna tuning |
| [`experiment_lgbm_sa_revised_variants.py`](experiment_lgbm_sa_revised_variants.py) | Four LightGBM variants (`v1`–`v4`) on the SA-revised feature set |
| [`experiment_sa_blend.py`](experiment_sa_blend.py) | Walk-forward dynamic blend of SA Direct + NSA+Adjustment |
| [`experiment_predicted_adjustment.py`](experiment_predicted_adjustment.py) | Predict `SA_MoM − NSA_MoM` (seasonal adjustment factor) via SARIMA, monthly avg, exp-weighted, etc. |
| [`explore_anchor_plus_direction.py`](explore_anchor_plus_direction.py) | Use SA prediction as anchor, NSA-derived direction as nudge — research only |
| [`explore_consensus_anchor.py`](explore_consensus_anchor.py) | Replace SA anchor with the LSEG consensus poll — research only |
| [`compare_sa_revised_models.py`](compare_sa_revised_models.py) | Unified ranking table across main + sandbox + archive runs |
| [`run_sa_revised_sandbox_suite.py`](run_sa_revised_sandbox_suite.py) | One-command runner: all sandbox experiments, then comparison |
| [`freeze_sa_revised_baseline.py`](freeze_sa_revised_baseline.py) | Champion / challenger baseline freezer |
| [`output_utils.py`](output_utils.py) | Shared sandbox diagnostics writer (`write_sandbox_output_bundle`) — used by other scripts |

Each experiment writes a full diagnostic bundle (backtest CSV, summary stats, ACF/PACF, prediction plot, summary table) under `_output/sandbox/<experiment_id>/`.

---

## 1. CatBoost / XGBoost SA Revised Backtest

```bash
python Train/sandbox/experiment_catboost_sa_revised.py
# or
python Train/sandbox/experiment_xgboost_sa_revised.py
```

Enable Optuna tuning for XGBoost (composite objective):

```bash
python Train/sandbox/experiment_xgboost_sa_revised.py \
  --tune --tune-objective composite --tune-trials 25 --tune-every-steps 1
```

If CatBoost is not installed, the XGBoost script is the recommended fallback.

**Outputs (per experiment, e.g. `xgboost_sa_revised`):**
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

## 2. SA Blend Walk-Forward (SA Direct + NSA+Adjustment)

Builds a walk-forward dynamic blend from existing main-pipeline backtest CSVs — no retraining.

```bash
python Train/sandbox/experiment_sa_blend.py
```

Tune blend hyperparameters (`window`, `min_history`, `grid_step`):

```bash
python Train/sandbox/experiment_sa_blend.py \
  --tune --tune-objective composite --tune-trials 25 --tune-cv-splits 4
```

Output: `_output/sandbox/sa_blend_walkforward/` (same diagnostics bundle as above). This blend used to be a Kalman-fusion channel and is still kept as the fallback champion in [`consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py).

---

## 3. LightGBM SA Revised Variant Suite

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

---

## 4. Predicted Seasonal Adjustment

Backtests several models for the seasonal adjustment factor (`SA_MoM − NSA_MoM`) without lookahead, then combines NSA predictions with the predicted adjustment to produce SA-space forecasts. Models tested: SARIMA, monthly average, 12-month complement, same-month last year, exponentially-weighted monthly average, linear regression with month dummies.

```bash
python Train/sandbox/experiment_predicted_adjustment.py
```

Output: `_output/sandbox/nsa_predicted_adjustment_revised/`.

---

## 5. Consensus / Anchor Explorations

Research-only scripts that informed the production [`consensus_anchor_runner.py`](../Output_code/consensus_anchor_runner.py). They reuse existing backtest CSVs and the latest Unifier consensus snapshot — no retraining.

```bash
python Train/sandbox/explore_anchor_plus_direction.py   # SA anchor + NSA direction
python Train/sandbox/explore_consensus_anchor.py        # Consensus poll as anchor
```

Both print metrics to stdout; neither writes to `_output/` (they are intentionally side-effect-free explorations).

---

## 6. Unified SA Revised Comparison

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

## 7. One-Command Suite Runner

Run all sandbox experiments, then the comparison table:

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

## 8. Freeze Baseline (Champion / Challenger)

Freezes the current SA-revised baseline pair:

- `champion_v1` — sandbox SA blend (`_output/sandbox/sa_blend_walkforward`)
- `challenger_v1` — main SA-revised (`_output/SA_prediction`, falling back to `_output/SA_prediction_revised`)

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
