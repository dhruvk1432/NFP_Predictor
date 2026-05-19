# HMM Regime Reselection Study

Contained diagnostics for the PIT HMM regime trigger. These scripts read cached
training matrices and backtest outputs, then write all study artifacts under
`_output_hmm_regime_reselection_study/`.

## Run

```bash
python experiments/hmm_regime_reselection_study/run_regime_audit.py \
  --output-dir _output_hmm_regime_reselection_study \
  --training-cache _output_pairing_baseline_pitfix/cache/training_dataset \
  --backtest-dir _output_pairing_baseline_pitfix \
  --target nsa \
  --source revised \
  --start 2000-01 \
  --n-components 3 \
  --max-features 32 \
  --emission-profile seasonal_resid \
  --surprise-quantile 0.95
```

Repeat the audit across the small trigger-shape grid before any model training:

```text
n_components: 3, 4, 5
emission_profile: seasonal_resid, hybrid, macro_only
surprise_quantile: 0.90, 0.95, 0.975
```

Evaluate whether a trigger policy is sparse enough for feature-selection dry run:

```bash
python experiments/hmm_regime_reselection_study/evaluate_trigger_policy.py \
  --input _output_hmm_regime_reselection_study/hmm_regime_monthly.csv
```

Only after the trigger policy passes sparsity/event checks, run feature-selection
dry run without model training:

```bash
python experiments/hmm_regime_reselection_study/dry_run_feature_reselection.py \
  --hmm-audit _output_hmm_regime_reselection_study/hmm_regime_monthly.csv \
  --output-dir _output_hmm_regime_reselection_study/feature_reselection_dry_run \
  --training-cache _output_pairing_baseline_pitfix/cache/training_dataset \
  --target nsa \
  --source revised
```

```bash
python experiments/hmm_regime_reselection_study/join_backtest.py \
  --hmm-dir _output_hmm_regime_reselection_study \
  --backtest-dir _output_pairing_baseline_pitfix
```

```bash
python experiments/hmm_regime_reselection_study/summarize_results.py \
  --input _output_hmm_regime_reselection_study/hmm_joined_backtests.csv
```

## Outputs

- `hmm_regime_monthly.csv`: walk-forward HMM decisions.
- `hmm_regime_monthly.json`: config and full event payload.
- `trigger_policy_eval/`: trigger dates, clusters, event precision/recall,
  seasonal false-positive diagnostics, and acceptance summary.
- `feature_reselection_dry_run/`: selected feature sets and Jaccard stability
  diagnostics without fitting the forecast stack.
- `hmm_joined_backtests.csv`: HMM regimes joined to NSA/Kalman/router backtests.
- `summary.md`: regime, trigger, event-window, stress-month, churn, and bias-correction diagnostics.

## Acceptance Focus

Use the HMM as a regime sensor. A production change should improve final
`panel_kalman_router` or `kalman_fusion` metrics without raising feature churn
or weakening PIT stress windows.

`--max-gap-months` defaults to `0` in this study so the HMM is evaluated
without a fixed-cadence fallback. Set it explicitly only when testing an
operational stale-model safeguard.

Do not run `Train/train_lightgbm_nfp.py --train-all` until:

- full-period triggers are at most 20;
- last-60-month triggers are at most 8;
- January/July triggers are at most 35% of all triggers;
- cluster-level event precision is at least 0.50;
- major-event recall is at least 0.75;
- dry-run median feature Jaccard is at least 0.50.
