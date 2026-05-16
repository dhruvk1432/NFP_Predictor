# NFP Predictor Research and Improvement Roadmap

Last updated: 2026-05-16

This document consolidates the current repo state, the recoverable May 12 pushed
baseline, the forked project ideas, the Advanced Time Series Analysis materials,
and the current diagnostic findings. It is meant to guide the next sequence of
experiments without rewriting the production pipeline by accident.

The current practical benchmark is the recoverable pushed archive
`2026-05-12_165541`, not the unrecoverable May 12 morning run. The morning run
reached about 88.9 MAE but its exact feature set and settings were not uploaded.
The recoverable pushed baseline reached about 90.4 MAE and should be treated as
the reference point. The latest local/AWS-synced run,
`2026-05-16_124429`, reached **92.66 MAE**, which materially changes the
diagnosis: the post-baseline changes are no longer just "promising but
regressing." They have now recovered most of the gap to the recoverable May 12
push while improving acceleration behavior.

## Executive Summary

The model should remain a consensus-anchored Kalman fusion system. The best
recoverable setup was not a standalone SA model and not a standalone NSA model.
It was a fusion of:

- PIT consensus forecast.
- NSA model plus predicted seasonal adjustment.
- Kalman filter/fusion logic.
- Conservative dynamic feature selection.
- Leakage/COVID controls added around May 12.

The main current problem is not that the post-baseline work was bad. The latest
92.66 run shows the opposite: the recent step-5/step-6 style changes are a real
bump from the deteriorated 98-100 MAE region and are now competitive with the
recoverable May 12 push. The issue is attribution. Several additions after the
recoverable baseline are conceptually strong and now visibly entering the model:
the individual economist panel, continuous daily futures features, PIT-safe
SA-NSA gap features, richer feature generation, and joint NSA/Kalman tuning.

The latest run is best interpreted as: **keep the new data/features and keep
joint fusion-aware tuning, but continue constraining the noisy NSA plus
adjustment channel.** The NSA branch is still weak as a standalone forecast;
Kalman is extracting useful conditional information from it by assigning it a
small effective weight rather than trusting its raw level.

The highest-value next work is:

1. Use `2026-05-12_165541` as the reference baseline.
2. Archive every local/AWS run before overwrites.
3. Treat `2026-05-16_124429` as the new current-head checkpoint and compare it
   to both `2026-05-12_165541` and consensus.
4. Use the AWS grid to choose dynamic feature-selection frequency and tuning
   settings, but score runs against both the recoverable baseline and the new
   92.66 checkpoint.
5. Preserve post-baseline data additions and test them one by one: economist
   panel, futures, SA-NSA gap features, and new feature-generation blocks.
6. Keep standalone SA out of production unless it proves itself; use SA as
   features, gap signals, acceleration signals, or a high-noise fusion input.
7. Import forked direct-acceleration ideas as a sidecar/router/uncertainty
   signal, not as a hard replacement for the level forecast.
8. Add model averaging over the best few stable configurations rather than
   relying on one fragile dynamic-FS winner.
9. Add formal paired forecast tests before accepting small gains.

## Current Evidence

Artifact-only diagnostics live in:

- `experiments/deep_dive_diagnostics.py`
- `_output/deep_dive_diagnostics/SUMMARY.md`
- `_output/deep_dive_diagnostics/archive_scoreboard.csv`
- `_output/deep_dive_diagnostics/current_vs_baseline_month_deltas.csv`
- `_output/deep_dive_diagnostics/static_blend_grid.csv`
- `_output/deep_dive_diagnostics/rolling_blend_grid.csv`

The diagnostic script reads existing artifacts and the fork's acceleration
outputs. It does not import or mutate the training pipeline. It defaults to the
recoverable baseline:

```bash
python experiments/deep_dive_diagnostics.py
```

To re-anchor to another archive after an AWS or local run:

```bash
BASELINE_ARCHIVE_NAME=2026-05-XX_HHMMSS python experiments/deep_dive_diagnostics.py
```

Current key numbers from the artifact diagnostics after the latest run:

| Forecast | MAE | RMSE | Notes |
|---|---:|---:|---|
| Recoverable baseline Kalman, `2026-05-12_165541` | 90.42 | 128.89 | Practical target to recover/exceed |
| Latest Kalman, `2026-05-16_124429` | 92.66 | 130.89 | New current checkpoint; big recovery from the 98-100 MAE region |
| Consensus | 101.28 | 144.03 | Strong baseline and anchor |
| Baseline SA LightGBM | 141.88 | 209.99 | Too weak as production peer |
| Current SA LightGBM | 148.69 | 210.59 | Still too weak; variance-collapsed |
| Baseline NSA plus adjustment | 173.80 | 218.96 | Weak standalone but useful through fusion |
| Latest NSA plus adjustment | 192.13 | 248.13 | Still noisy; improved from the worst run but weaker than baseline standalone |

Additional latest-run facts:

- Official production scorecard acceleration accuracy improved from **56.1%**
  in `2026-05-12_165541` to **70.2%** in `2026-05-16_124429`.
- Directional accuracy stayed at **96.6%**.
- Tail MAE is **139.1**, not as strong as the recoverable baseline's **131.3**
  but much better than several recent post-baseline runs.
- Latest MAE is only **2.25 points worse** than the recoverable baseline, while
  materially better than consensus by **8.61 points**.
- Versus the worse `2026-05-15_145054` region, the latest run recovered about
  **6.76 MAE**.
- Versus the stronger `2026-05-15_163708` archive, the latest run still gained
  about **1.79 MAE**.

The biggest latest losses versus the recoverable baseline are concentrated in:

- `2026-03`
- `2022-07`
- `2022-12`
- `2021-06`
- `2022-11`
- `2023-01`
- `2024-04`
- `2023-09`
- `2023-06`
- `2023-10`

These months should become a named stress panel in every future report.

The biggest latest improvements versus the recoverable baseline are:

- `2023-12`
- `2024-01`
- `2021-12`
- `2022-05`
- `2022-02`
- `2023-04`
- `2021-11`
- `2024-02`
- `2024-07`
- `2024-03`
- `2025-02`
- `2026-01`

## What The Latest 92.66 Run Says

### 1. The New Sources Are Showing Up In Useful Places

The latest NSA feature-importance table remains anchored by the target-history,
claims, and FRED-employment spine, which is good. But the post-baseline sources
are no longer invisible:

- `NFP_Consensus_Mean` is a top-10 final NSA feature.
- Individual economist features appear in dynamic-selection cohorts, including
  `NFP_Forecast_CONTINUUM_ECON`, `NFP_Forecast_DANSKE_BK`, and top-4 panel
  transforms.
- Futures/market features appear in selected cohorts and final importance:
  Fed Funds, Treasury/Yield, NatGas, JPY/USD volatility, VIX/WTI in some
  cohorts, and `SP500_max_5d_drop_chg_3m_lag_3m`.
- SA-NSA gap is entering the final model through `sanagap_adj_lag1`, which
  ranked in the latest top-30 final NSA importances.

This supports keeping these data additions. They should be attributed and
regularized, not removed.

### 2. The Bump Is Mostly Fusion-Aware Tuning, Not Standalone NSA Quality

The latest joint tune selected:

- `half_life_years = 3.2337`
- `trailing_window = 28`
- `nsa_weight_scale = 0.1064`
- `best_score = 102.58`
- `n_trials_run = 25`

That `nsa_weight_scale` is the key. The model did **not** improve because the
NSA plus adjustment branch became a good standalone level forecast. It improved
because fusion learned to treat that branch as a low-weight, high-information
observation. The branch is still volatile:

- Baseline NSA plus adjustment MAE: **173.80**
- Latest NSA plus adjustment MAE: **192.13**
- Latest NSA plus adjustment Diff STD ratio: **2.37**

So the current mechanism is: keep the NSA branch for nonlinear/acceleration and
seasonal-adjustment information, but let Kalman/consensus dominate level
placement when NSA becomes too explosive.

### 3. Small Positive Direction/Acceleration Penalties Help Objective Curvature

The latest config reintroduced small `KALMAN_LAMBDA_ACCEL = 5` and
`KALMAN_LAMBDA_DIR = 5`. This is materially different from the earlier
pure-MAE mode. The practical effect is not just "optimize acceleration." It
helps identify a better half-life/window/weight surface for fusion. The result
is higher acceleration accuracy without a large MAE penalty.

This is worth keeping unless the AWS grid proves a better nearby setting.

### 4. Dynamic Feature Selection Is Better, But The Live Working Directory Still Needs Hygiene

The current final feature count is 80, but the live
`_output/dynamic_selection/nsa_revised` directory still contains mixed feature
counts from multiple cohorts: 25, 80, 114, and 120. This does not invalidate
the latest archive, but it does create replay risk. Before any replay or
"preserve best features" step, wipe the live dynamic-selection directory and
copy only one known cohort.

### 5. Remaining Failure Modes Are Specific

The latest run still loses badly versus the recoverable baseline in `2026-03`,
`2022-07`, `2022-12`, `2021-06`, and `2022-11`. These failures have a common
shape: the fusion moves too close to the low consensus/low-NSA view in a month
where the actual is stronger, or it underreacts to a turning point. The next
diagnostic should split these months by:

- Consensus miss direction.
- NSA plus adjustment disagreement direction.
- Selected futures/economist/gap features active in that month.
- Kalman implied weight/noise state.
- Whether direct acceleration sidecars would have corrected the sign.

## What Is Wrong With The Current State

The word "wrong" here does not mean "new changes should be reverted." It means
"currently unresolved or insufficiently attributed." The project has promising
new data sources and feature blocks, but the current end-to-end score does not
yet tell us which pieces helped and which pieces hurt.

### 1. Reproducibility Is Still Too Fragile

The best unrecoverable run cannot be reconstructed because exact features and
settings were not uploaded. That cannot happen again.

Every meaningful run should save:

- Full config values.
- Git commit hash and dirty diff summary.
- Dynamic feature-selection JSONs.
- Final selected features per window.
- Optuna trials and winning params.
- Kalman/fusion params.
- Environment variables.
- Model artifacts.
- Predictions and comparison metrics.
- Run notes describing what changed from the previous run.

The benchmark should not be "whatever `_output` currently contains." `_output`
is a working area. Every run that matters needs a durable archive.

### 2. Dynamic Feature Selection Appears Unstable

Current `_output/dynamic_selection/nsa_revised` contains mixed/stale-looking
cohorts with feature counts such as 25, 80, 114, and 120. The latest archived
run itself is coherent enough to keep, but the live working directory is not
clean enough to replay blindly. That distinction matters: the problem is run
hygiene and stability testing, not proof that dynamic-FS failed.

The AWS grid over dynamic-FS frequency is therefore highly relevant. However,
the winning criterion should not be raw MAE alone. Prefer a cadence that is:

- Close to or better than 90.4 MAE.
- Stable across 2022, 2025, and 2026 stress months.
- Stable in selected feature count.
- Stable in source composition.
- Not dependent on a single lucky month.
- Not forcing Kalman to downweight NSA almost completely.

High-frequency reselection is likely to overfit. Very stale feature selection
will miss real changes. The prior should be "medium cadence wins" unless the AWS
grid proves otherwise.

### 3. Too Many Knobs Are Coupled

Recent work introduced or modified several interacting pieces:

- Individual economist panel features.
- Continuous daily futures features.
- New feature-generation blocks.
- Dynamic-FS frequency.
- Pass-2 max feature cap.
- Reselection stages.
- Joint Optuna.
- Kalman half-life/window tuning.
- Kalman direction/acceleration penalties.
- NSA plus adjustment construction.
- Possible SA/NSA gap features.

This makes it hard to tell which change helped or hurt. The next experiments
should isolate one axis at a time, then combine only the winners.

The important implication: do not treat a small aggregate regression as evidence
that the economist panel, futures data, or SA-NSA gap features are bad. Treat it
as evidence that attribution is missing.

### 4. Level Accuracy And Acceleration Accuracy Are Being Traded Off

Some later configurations improved acceleration-like scores while worsening MAE.
The model is ultimately judged on SA revised level forecast quality. Acceleration
is useful only if it improves the level forecast or the risk/uncertainty logic.

The latest run is a good version of this tradeoff: acceleration accuracy rose
to 70.2% while MAE stayed near the recoverable baseline. Future runs should
preserve that shape. Do not accept a model that gains acceleration accuracy but
loses materially on MAE, RMSE, or stress-month behavior.

### 5. The Old Standalone SA Model Is Weak, But A Modern Direct SA Model Is Still Promising

The current and baseline SA LightGBM forecasts are both too weak to be a
production peer. Static blend diagnostics assign SA zero optimal weight when
blending current Kalman/consensus/NSA/SA on the full current panel.

This should not be interpreted as evidence that direct SA prediction is a dead
end. The old SA branch is outdated relative to the current repo: it does not
fully exploit the newer individual economist panel, continuous futures, SA-NSA
gap features, direct feature-generation blocks, current leakage fixes, current
dynamic-FS machinery, or fusion-aware tuning. We may be losing many features
that are weak for NSA level prediction but highly relevant for the SA revised
target directly.

Feature families that may be better suited to direct SA than NSA:

- Consensus level, consensus revisions, economist-panel dispersion, and
  individual forecaster disagreement. These are already expressed in SA terms
  by market/economist participants, so forcing them through NSA plus adjustment
  can throw away information.
- Daily futures/rates/equity/FX/volatility windows immediately before the NFP
  release. These may capture market-implied expectations or risk regimes around
  the SA headline rather than the raw NSA payroll count.
- SA target-history transforms: lagged SA revised MoM, SA momentum, SA
  acceleration, rolling volatility, trend breaks, and low-payroll/negative-print
  regime flags.
- Consensus-residual features: variables that predict `SA_revised - consensus`
  may have little value for raw NSA prediction but large value as a direct
  consensus correction.
- SA-NSA gap features: gap level, seasonal strength, volatility, and trend may
  be useful in a direct SA model as indicators of when seasonal adjustment is
  unusually unstable.
- Release-calendar and survey-window features: timing of the reference week,
  weekday placement, pre-release claims/ADP/ISM windows, and weather or strike
  shocks may affect the headline SA surprise more than the NSA level.

The correct use of the current old SA model is probably:

- SA target-history features.
- SA momentum/acceleration features.
- SA-NSA gap and gap volatility features.
- Direct acceleration classification.
- High-noise optional Kalman observation.
- Diagnostic model for failure attribution.

The wrong use is likely:

- Reactivating the old standalone SA LightGBM unchanged and giving it equal
  production status.

The promising use is:

- Build a new direct-SA sidecar from the current feature universe, current PIT
  fixes, and current tuning/evaluation discipline, then let the fusion/model
  averaging layer decide how much trust it deserves.

### 6. NSA Plus Adjustment Is Too Volatile

The latest NSA plus adjustment branch is still worse than the recoverable
baseline branch as a standalone forecast, though it improved from the worst
post-baseline state. This does not mean the branch is useless; the latest 92.66
run is evidence that Kalman fusion can extract useful information from a weak
standalone channel. But it does mean the branch must be monitored as a noisy
input.

Track for each run:

- NSA plus adjustment MAE/RMSE.
- NSA plus adjustment bias.
- NSA plus adjustment variance ratio.
- Difference standard deviation ratio.
- Effective fusion/Kalman weight.
- Months where NSA plus adjustment creates large disagreement versus consensus.

### 7. Evaluation Is Not Yet Formal Enough

The backtest window is small. A one- or two-point MAE improvement can be noise.
Every finalist should be evaluated with paired errors versus the recoverable
baseline and current consensus:

- Paired absolute-error differences.
- Paired squared-error differences.
- HAC standard error on loss differences.
- Moving-block bootstrap confidence intervals.
- Stress-panel error deltas.
- Tail-MAE deltas.
- Direction/acceleration deltas.

## What To Keep

### Keep The PIT And Leakage Fixes

The May 12 leakage and COVID work is structurally valuable:

- PIT consensus loading.
- COVID-winsorized consensus handling.
- Pre-backtest Kalman noise priors.
- Revised actual leakage prevention.
- Adjustment-history fixes.
- Nested tuning rather than full-overlap tuning.
- Deterministic training settings.

Do not roll these back to chase a historical score.

### Keep And Isolate The Post-Baseline Data Additions

Several additions after the recoverable 90.4-MAE baseline are worth protecting.
The latest 92.66 run now makes this stronger: they are not merely theoretical
additions, they appear in selected cohorts and final feature importance.

#### Individual Economist Panel

Current repo asset:

- `Data_ETA_Pipeline/load_economist_panel.py`
- Raw inputs under `economist_panel/`
- Output snapshots under `Exogenous_data/exogenous_economist_data`
- Diagnostics under `_output/economist_panel/`

The current implementation is promising because it is deterministic and
PIT-aware. It uses a hand-curated top-4 panel selected on shared historical
accuracy, emits each economist's forecast as a feature, and emits a top-4
equal-weight ensemble with release timing set to the latest constituent
forecast. That is the right shape: individual forecasts can contain useful
cross-sectional disagreement that the aggregate consensus loses.

Tests to run:

- No economist features versus top-4 individual features.
- Top-4 mean only versus individual top-4 plus mean.
- All available economists with strong regularization versus hard top-4.
- Rolling top-k economist selection using only historical data available before
  each month.
- Accuracy-weighted top-k ensemble versus equal-weight top-4.
- Dispersion, range, and disagreement features if the panel has enough members.
- Forecast-revision features if multiple forecast timestamps exist for the same
  event month.

Metrics to inspect:

- Incremental MAE versus consensus and recoverable baseline.
- Stress-panel performance.
- Whether the economist features reduce consensus residual error.
- Whether economist disagreement predicts channel uncertainty.
- Feature-selection stability: do selected economist features persist or churn?

Failure modes:

- Top-4 selection may be overfit to the 2022-2025 shared sample.
- Individual forecasters may stop filing or file too late.
- The ensemble may duplicate consensus unless individual disagreement is used.
- If release timing is mishandled, the feature can become subtly leaky.

#### Continuous Daily Futures Features

Current repo asset:

- `Data_ETA_Pipeline/load_futures_data.py`
- Raw daily continuous futures under `continuous_futures/`
- Output snapshots under `Exogenous_data/exogenous_futures_data`

The current implementation is also promising. It includes rate expectations,
Treasury curve contracts, equity sentiment, FX, VIX futures, industrial
commodities, and precious metals. It distinguishes plain series for levels from
back-adjusted continuous series for returns/volatility, which is important
because back-adjustment can corrupt level interpretation while plain contracts
can create roll artifacts in returns.

Tests to run:

- No futures versus all futures.
- Futures by group: rates, Treasury curve, equity, FX, volatility, industrial
  commodities, energy, precious metals.
- Level-only versus return/volatility/range features.
- Pre-NFP daily window features: last 1d, 3d, 5d, 10d, 21d before release.
- Event-window features around ADP, claims, ISM, JOLTS, and FOMC events.
- Futures features only in uncertainty/error model versus in level model.
- Futures features as regime filters for Kalman observation noise.

Metrics to inspect:

- Do futures reduce tail misses or only add noise?
- Do rate futures help consensus residuals?
- Do equity/credit/vol futures help stress regimes?
- Do commodities help only in goods/construction regimes?
- Do features selected from futures persist across dynamic-FS windows?

Failure modes:

- Monthly aggregation can wash out pre-release information.
- Some contracts are more about post-release reaction than pre-release signal.
- Futures may help uncertainty and regimes more than point levels.
- Roll handling must be kept separate for levels versus returns.

#### SA-NSA Gap Features

Current repo asset:

- `Train/feature_engineering_sa_nsa_gap.py`
- Integration hooks in `Data_ETA_Pipeline/create_master_snapshots.py`

These features are promising because the production target is SA revised while
one core model channel is NSA plus predicted seasonal adjustment. The gap itself
can carry seasonal regime information, benchmark effects, and adjustment
instability.

Tests to run:

- Gap features off versus on.
- Gap level, lag, rolling mean, rolling volatility, and z-score separately.
- Gap features in NSA model only versus in fusion/SA sidecar only.
- Gap volatility as Kalman observation-noise input.

Failure modes:

- If adjustment history is sparse, gap features can be unstable.
- Gap features should not become a backdoor for revised-target leakage.
- Gap features may help only specific months and hurt average performance.

#### Modern Direct SA Sidecar

Current repo issue:

- The historical SA branch is stale. Its poor MAE says the old implementation
  is not production-ready; it does not prove that direct SA prediction cannot
  add value.
- Current dynamic selection is optimized around the NSA/fusion path, so it may
  discard features whose strongest relationship is to the SA revised headline
  or to the consensus residual.
- Several newer feature families are naturally SA-facing: economist forecasts,
  consensus microstructure, market-implied futures moves, SA-NSA gap behavior,
  and headline-surprise regimes.

Goal:

- Build a fresh direct-SA model as a **sidecar source of truth**, not as an
  immediate production replacement.
- Use it to predict either `SA_revised` directly or `SA_revised - consensus`.
- Feed its output into fusion/model averaging only after it proves
  out-of-sample value.

Model targets to test:

- Direct level: `SA_revised`.
- Consensus residual: `SA_revised - consensus`.
- Directional residual: sign and magnitude bucket of `SA_revised - consensus`.
- Acceleration: sign of `SA_revised[t] - SA_revised[t-1]`.
- Distributional target: quantiles around `SA_revised` or residual error.

Feature sets to test:

- SA target-history block: SA lags, momentum, acceleration, rolling volatility,
  rolling mean, and regime flags.
- Consensus/economist block: consensus level, top-4 mean, individual
  economist forecasts, disagreement, dispersion, forecaster revisions, and
  top-k rolling accuracy weights.
- Futures block: pre-release rates/curve/equity/FX/vol windows at 1d, 3d, 5d,
  10d, and 21d horizons.
- Event-surprise block: ADP, claims, ISM employment, JOLTS, Challenger, NFIB,
  consumer-confidence labor differential, and other release-calendar surprises
  where timestamps are clean.
- Gap block: `sanagap_*` features, gap volatility, gap trend, and seasonal
  strength.
- Current NSA/fusion features: only those that survive a direct-SA audit, not
  the entire NSA-selected feature set by default.

How to let the main model use it:

- As a fourth Kalman observation with high initial measurement noise and
  rolling noise updates.
- As a consensus-residual correction: `final = consensus + bounded_direct_sa_residual`.
- As a model-average candidate with constrained prequential weights.
- As a router: increase/decrease NSA channel weight when direct-SA confidence
  disagrees with consensus in a historically reliable regime.
- As an uncertainty signal: widen intervals or raise tail-risk flags when
  direct-SA, consensus, and NSA disagree.
- As a feature into the Kalman fusion state update, not necessarily as a raw
  level forecast.

Acceptance criteria:

- It must beat consensus on residual MAE or improve the fused forecast
  prequentially.
- It must receive nonzero rolling weight without full-sample tuning leakage.
- It must improve at least one of: full MAE, stress-panel MAE, tail MAE,
  2026-03/late-2022 failure cases, or uncertainty calibration.
- If it only improves acceleration/direction but hurts level MAE, use it as a
  router or uncertainty input, not as a point-forecast peer.

Implementation constraints:

- Do not reuse the old SA branch as-is.
- Start with an artifact-only or sandbox direct-SA experiment.
- Reuse the current PIT loaders and master snapshots.
- Give direct SA its own feature-selection target; do not inherit the NSA
  selected set.
- Save selected features/source counts so we can see which "lost" features the
  SA sidecar recovers.

#### New Feature Generation Blocks

Post-baseline feature generation may be useful even when the aggregate run
regresses. Evaluate generated features by source and role:

- Level/trend features.
- Momentum features.
- Acceleration features.
- Volatility/uncertainty features.
- Consensus-residual features.
- Regime features.

The right question is not "did all new features help?" The right question is
"which feature blocks should feed the level model, which should feed the
acceleration model, and which should feed Kalman uncertainty?"

### Keep Consensus As The Anchor

Consensus is strong and stable. Current consensus is around 101 MAE, which is
hard to beat consistently. It should remain a core input and a floor.

Useful consensus-derived features:

- Consensus level.
- Consensus surprise versus target-history nowcast.
- Consensus dispersion if available from external data.
- Consensus revision from prior survey snapshots.
- Consensus z-score against recent consensus history.
- Consensus versus NSA plus adjustment disagreement.

### Keep Kalman Fusion

Kalman fusion is the right shape for this problem. The next improvement is not
"replace Kalman," but make it more signal-aware:

- Observation noise varies by source quality and regime.
- NSA plus adjustment gets downweighted when seasonal gap volatility is high.
- Consensus gets downweighted when consensus dispersion is high.
- Direct acceleration probability adjusts transition/level prior rather than
  replacing the level forecast.
- COVID/extreme-month behavior is handled explicitly.

### Keep Dynamic Feature Selection, But Make It Auditable

Dynamic-FS is valuable if it adapts without thrashing. It needs stricter audit
outputs:

- Selected features per window.
- Feature source counts per window.
- Overlap with prior window.
- Overlap with recoverable baseline feature sets when available.
- Number of features entering/exiting.
- Downstream MAE by cohort.
- Feature stability score.

### Keep The Fork's Direct-Acceleration Thread

The fork's `acceleration_maxxing` work is one of the most relevant additions.
Its direct acceleration models show materially higher acceleration accuracy than
level-derived standalone models over many windows.

Promising pieces:

- XGBoost direct acceleration classifier.
- LightGBM direct acceleration classifier.
- VARMA/XGB hybrid.
- Feature selection around SA diff/z-score, target states, claims, ADP, and
  composite features.
- Probability outputs as confidence signals.

Avoid:

- Hard replacement of level forecasts.
- Overcomplicated MR/rule ensembles unless they beat simple XGB/LGBM.
- Direction files that measure a different target than SA revised acceleration.

### Keep The Fork's Exact-Feature Audit

The fork's feature audit is useful for pruning and prioritization.

High-priority zones:

- Claims.
- ADP.
- Consensus.
- Top-level employment aggregates.
- Target-history states.
- Selected hours/activity measures from Unifier.
- FRED Employment NSA aggregates and construction/goods components.
- FRED Employment SA diff/z-score transforms for acceleration.

Lower priority:

- NOAA, unless a specific weather shock hypothesis is being tested.
- Calendar as alpha source; keep it mostly as a structural control.
- Deep wrapper/lag expansions that create lots of dead exact features.

## Course And Literature Ideas To Use

### 1. Model Averaging Instead Of Winner-Take-All

The Advanced Time Series notes point out that if the goal is forecasting, we do
not necessarily need to select one model. Bayesian model averaging and
information-criterion weighting are natural alternatives.

Repo translation:

- Do not pick one AWS grid winner by raw MAE alone.
- Keep the top stable configurations.
- Build prequential weights using recent out-of-sample loss.
- Constrain weights to avoid overreacting.
- Include consensus as a persistent model in the pool.

Candidate averaging methods:

- Exponential weighted average over recent MAE.
- Constrained nonnegative least squares on rolling windows.
- BIC/AIC-like weights for simple time-series sidecars.
- Stacking with strong regularization.
- Bayesian model averaging approximation using validation likelihood.

### 2. Hierarchical/Shrinkage BVAR Sidecar

The GLP/BVAR materials are relevant because this project has a large feature
pool and a small effective sample. A flat VAR would overfit. A shrinkage VAR or
hierarchical BVAR can provide a different source of truth.

Do not start with a huge BVAR. Start with a compact monthly state vector:

- SA revised target history.
- Consensus.
- NSA plus adjustment.
- Direct acceleration probability.
- Claims factor.
- ADP factor.
- Employment aggregate factor.
- Financial stress factor.
- Labor state factor.
- SA-NSA gap factor.

Uses:

- Standalone sidecar forecast.
- Kalman observation.
- Model-average candidate.
- Regime diagnostic.
- Density/uncertainty estimate.

Expected value:

- More stable than LightGBM in small samples.
- Different failure modes from tree models.
- Useful for uncertainty and shrinkage.

### 3. Factor-Augmented Forecasting

The GLP appendix and Stock-Watson-style factor forecasting map well to the repo.
Instead of feeding thousands of raw transformed features, extract source-level
factors and combine them with target lags.

Suggested factor groups:

- Claims.
- ADP/private payroll proxies.
- FRED Employment NSA.
- FRED Employment SA.
- Unifier activity/hours.
- Financial stress and market conditions.
- Labor state.
- Consensus/survey data.
- SA-NSA gap.

Models to test:

- Ridge regression on factors.
- ElasticNet on factors.
- Local projection for one-month-ahead level and acceleration.
- Small BVAR on factors.
- Factor-augmented Kalman observation.

### 4. State-Space Upgrade

The course state-space/Kalman content supports a richer version of the current
fusion system.

Current fusion likely behaves too much like a tuned weighted blend. A stronger
version would model:

- Latent true labor-market momentum.
- Consensus as a noisy observation.
- NSA plus adjustment as a noisy observation.
- Direct acceleration probability as a state-transition cue.
- SA-NSA gap volatility as measurement-noise input.
- Regime-specific observation variances.

Do this incrementally:

1. Add time-varying observation variance to existing Kalman fusion.
2. Add source-quality covariates.
3. Add direct acceleration probability as a soft transition adjustment.
4. Only then consider a more formal latent-state model.

### 5. Spectral And Coherency Diagnostics

The spectral notes are useful for understanding which features help at which
frequencies.

Repo translation:

- Features that explain low-frequency cycles help level/trend.
- Features that explain high-frequency cycles help release shocks and
  acceleration.
- Some features should be routed to level models, others to acceleration models.

Possible diagnostic:

- Compute rolling/coarse cross-spectral coherence between candidate features and
  SA revised level changes/acceleration.
- Bucket features as trend, business-cycle, high-frequency, or noise.
- Use this as a feature-prior or quota in dynamic FS.

This should be diagnostic first, not production first.

### 6. HAC And Block-Bootstrap Forecast Comparisons

The GLP evaluation discussion uses HAC uncertainty for forecast score
comparisons. Monthly forecast errors are serially dependent, so iid tests are
too optimistic.

For every finalist:

- Compute paired loss difference versus `2026-05-12_165541`.
- HAC standard error of mean loss difference.
- Moving-block bootstrap interval for MAE delta.
- Stress-panel delta.
- Tail-MAE delta.

Accept a run only if it is better overall or clearly better in targeted
failure modes without unacceptable degradation elsewhere.

### 7. Local Projections For Direct Acceleration

The course covers VARs/local projections conceptually. For this project, local
projection-style direct models are useful because the acceleration target is not
the same as the level target.

Test:

- Direct binary acceleration model.
- Direct signed magnitude model.
- Direct quantile model for level surprise.
- Separate models for high-confidence acceleration months.

Use outputs as:

- Kalman transition adjustment.
- Model-average feature.
- Regime router.
- Uncertainty widening/narrowing signal.

## Additional Different Ideas Worth Trying

These are intentionally different from the current LightGBM plus Kalman path.
They are not all equally likely to win, but they can provide independent
evidence for model averaging.

### 1. Survey-Disagreement Model

Build a model whose target is not SA revised directly, but the residual:

```text
SA_revised - consensus
```

Features:

- Consensus level.
- Consensus revision.
- Consensus dispersion if available.
- ADP/claims surprise.
- Prior NFP revision history.
- Financial stress.
- Direct acceleration probability.

Why it may help:

Consensus is hard to beat in levels. Modeling the correction to consensus is
often easier than modeling the whole number.

### 2. Revision-Aware Target Model

NFP first print and revised values can differ materially. Build a sidecar for:

```text
SA_revised - SA_first
```

or, if first-print timing is the production target:

```text
revision risk conditional on first-print environment
```

This could inform intervals and bias correction even if it does not improve the
point forecast.

### 3. Regime-Conditional Error Model

Instead of only forecasting the target, forecast the expected absolute error of
each channel:

- Consensus expected error.
- NSA plus adjustment expected error.
- Kalman expected error.
- Direct acceleration expected error.

Use predicted channel error as fusion weights.

Features:

- Recent channel errors.
- Channel disagreement.
- Consensus dispersion.
- Claims volatility.
- SA-NSA gap volatility.
- Financial stress.
- VIX/rates/credit stress.
- Release calendar structure.

### 4. Robust Median/Trimmed Ensemble

Build a small candidate set and use robust aggregation:

- Consensus.
- Current Kalman.
- Best stable AWS dynamic-FS Kalman.
- NSA plus adjustment with bounded correction.
- Factor model.
- BVAR/shrinkage model.
- Direct acceleration adjusted forecast.

Aggregate by:

- Trimmed mean.
- Median.
- Huber-weighted mean.
- Weighted median.

This can reduce catastrophic errors from a single overfit channel.

### 5. Quantile And Distributional Forecasting

Build quantile forecasts for SA revised:

- 10th, 25th, 50th, 75th, 90th percentiles.
- Use median as robust point forecast.
- Use width as uncertainty input for fusion.

Models:

- Quantile LightGBM.
- Quantile regression on factors.
- Conformal intervals over backtest residuals.

This helps because the model has large tail misses in specific regimes.

### 6. Change-Point / Break Detector

Use a simple break detector to decide when dynamic-FS should refresh or when
Kalman noise should widen.

Signals:

- Claims volatility.
- Consensus error volatility.
- SA-NSA gap volatility.
- Payroll component dispersion.
- Financial stress.
- Large revisions.

This may outperform fixed reselection frequency.

### 7. Data Revision Real-Time Vintage Model

If vintage histories are available, model how signals looked at the time rather
than as currently revised. This is especially important for macro series whose
revisions alter apparent predictability.

Priority:

- ALFRED/FRED vintage where possible.
- BLS historical releases.
- WRDS/LSEG timestamped releases.
- Internal point-in-time snapshot validation.

## WRDS And LSEG Data Wishlist

The guiding rule: download data that is timely, point-in-time, and plausibly
available before the NFP release. Avoid data that only looks predictive because
it is revised or released too late.

Some of this work has already started. The repo now has an individual economist
panel and continuous daily futures pipeline. The data wishlist below should
therefore be read as "extend and enrich the current sources" rather than "start
from scratch."

### Highest Priority External Data

#### 1. Survey And Consensus Microdata

From LSEG/Refinitiv, Bloomberg-like feeds if available, or other survey sources:

- Additional individual economist NFP forecasts beyond the current top-4 panel.
- Forecast timestamps.
- Consensus mean/median.
- Dispersion/stdev.
- Min/max/range.
- Number of forecasters.
- Forecast revisions between survey snapshots.
- Forecaster-level historical accuracy.

Why:

Consensus is already strong. Dispersion and revision dynamics should improve
Kalman observation noise and consensus-residual modeling.

Specific extension to current repo:

- Preserve the hardcoded top-4 as the stable baseline panel.
- Add an "all economists" experimental panel gated behind a feature flag.
- Add rolling historical accuracy weights.
- Add disagreement/dispersion features.
- Add forecaster freshness and number-of-forecasters features.
- Add panel revision features when multiple forecast vintages exist.

#### 2. ADP And Payroll Proxy History

If LSEG provides vintage/timestamped ADP and payroll-related private data:

- ADP headline.
- ADP industry breakdowns.
- ADP revisions.
- Timestamped release history.

Why:

The fork audit consistently points to ADP/private payroll proxies as useful.

#### 3. Initial And Continuing Claims Microstructure

From LSEG, FRED, or other PIT feeds:

- Initial claims.
- Continuing claims.
- Insured unemployment rate.
- State-level claims if available.
- Claims release surprises versus consensus.
- Four-week average and weekly-to-monthly aggregation available before NFP.

Why:

Claims are one of the most consistently useful labor-market nowcast sources.

#### 4. Market-Implied Labor/Rates Reaction Signals

From LSEG:

- Fed funds futures.
- SOFR futures.
- Treasury yields around labor-related releases.
- OIS curve.
- STIR futures.
- Equity index futures.
- Dollar index.
- VIX.
- Credit spreads.

Use carefully:

- Only values before NFP release.
- Prefer pre-release levels and changes around ADP/claims/ISM releases.
- Avoid using post-NFP market reaction.

Why:

Markets aggregate information and can help with consensus error, stress regimes,
and uncertainty.

Specific extension to current repo:

- The current `load_futures_data.py` already covers continuous futures. Extend
  it with pre-release daily windows and event-window features rather than only
  month-end aggregation.
- Add curve-slope and curve-butterfly features from rate/Treasury futures.
- Add cross-asset risk-on/risk-off composites.
- Add futures disagreement with macro consensus, e.g. rates pricing easing while
  economist panel expects strong payrolls.

#### 5. News/Event Calendar Data

From LSEG:

- Economic release calendar.
- Expected/actual values for labor-related releases.
- Timestamped surprise values.
- Revision fields.

Useful releases:

- ADP.
- ISM employment components.
- Conference Board labor differential.
- NFIB hiring plans.
- Challenger layoffs.
- JOLTS.
- Consumer confidence labor questions.
- Initial/continuing claims.
- Regional Fed employment components.

Why:

This is a natural source for consensus-residual and acceleration models.

### WRDS-Specific Ideas

WRDS is less directly labor-nowcast focused, but useful for financial stress,
firm behavior, and cross-sectional labor sensitivity.

#### 1. CRSP Market Stress Features

Download:

- Daily market returns.
- Small-cap returns.
- Sector returns.
- Market volatility proxies.
- Drawdown features.
- Pre-NFP monthly aggregation.

Use:

- Stress regime feature.
- Kalman observation-noise feature.
- Tail-risk indicator.

#### 2. Compustat Employment/Capex/Layoff Proxy Features

Quarterly Compustat is slower, but can help regime context:

- Aggregate employment where available.
- Industry-level employment changes.
- Capex and sales growth.
- Margins/profit stress.

Use:

- Slow-moving labor demand factor.
- Not a high-frequency nowcast signal.

#### 3. IBES Analyst Forecast Revisions

If available:

- Aggregate earnings forecast revisions.
- Sector-level revisions.
- Breadth of negative revisions.

Use:

- Business-cycle/labor-demand regime factor.

#### 4. RavenPack/News Analytics If Available Through WRDS

If the subscription exists:

- Layoff news intensity.
- Hiring news intensity.
- Macro labor sentiment.
- Firm-level employment news.

Use:

- Direct acceleration model.
- Extreme/tail risk model.

#### 5. TRACE / Credit Market Stress

If available:

- Corporate bond spreads.
- Spread changes.
- Distress indicators.

Use:

- Tail and regime features.
- Error-volatility model.

### LSEG-Specific Ideas

LSEG is likely more valuable than WRDS for this project because of real-time
macro and survey data.

Priority LSEG downloads:

1. NFP consensus microdata and snapshots.
2. ADP vintages and survey expectations.
3. Claims expectations/actuals and revisions.
4. ISM/S&P PMI employment components and expectations.
5. JOLTS expectations/actuals/revisions.
6. Challenger layoffs.
7. Consumer confidence labor differential.
8. NFIB jobs/hiring plans.
9. Fed funds/SOFR futures pre-release levels.
10. Treasury curve and equity/credit stress features before NFP.

For every LSEG series, store:

- Release timestamp.
- Survey timestamp.
- Value timestamp.
- First available date.
- Revision timestamp if revised.
- Source metadata.

## Experiment Order

### Phase 0: Preservation And Run Hygiene

Goal: Never lose another useful run.

Run before changing modeling code:

1. Preserve `2026-05-16_124429` as the new current-head checkpoint.
2. Archive AWS grid outputs with full config and selected features.
3. Add a run manifest for every experiment.
4. Make diagnostic script point to the run archive.

Required artifact checklist:

- `config_snapshot.json`
- `env_snapshot.txt`
- `git_status.txt`
- `git_diff.patch`
- `selected_features/`
- `dynamic_selection/`
- `optuna_trials.csv`
- `kalman_params.json`
- `predictions.csv`
- `comparison_metrics.csv`
- `stress_panel_metrics.csv`
- `run_notes.md`

### Phase 1: Rebuild The Recoverable Baseline Comparison

Goal: Compare everything to both `2026-05-12_165541` and
`2026-05-16_124429`.

Tests:

- Latest versus baseline month deltas.
- Latest versus baseline component deltas.
- Latest versus baseline feature source counts.
- Latest versus baseline Kalman/fusion params.
- Latest versus baseline NSA plus adjustment behavior.
- Latest versus deteriorated post-baseline runs (`2026-05-15_145054`,
  `2026-05-14_235212`, or any other AWS grid loser) to identify what the
  step-5/step-6 changes fixed.

Decision rule:

- A change that improves versus the 92.66 checkpoint is a promotion candidate.
- A change that does not beat 92.66 can still be useful if it explains or fixes
  a targeted failure month versus the 90.4 baseline.
- A change that worsens both 92.66 and the targeted stress panel is low
  priority.

### Phase 2: AWS Grid Triage

Goal: Pick dynamic-FS/tuning settings that are stable, not just lucky, using
92.66 as the new current-head bar.

For each AWS run compute:

- Full MAE/RMSE/MedAE.
- Tail_MAE.
- Stress-panel MAE.
- 2022-only MAE.
- 2025-2026 MAE.
- Consensus delta.
- Recoverable-baseline delta.
- Latest-checkpoint delta (`2026-05-16_124429`).
- NSA plus adjustment metrics.
- Kalman effective source weights.
- Feature-count stability.
- Feature-source stability.
- Runtime.

Promote at most 3-5 finalist configs.

Reject if:

- Feature count thrashes.
- Improvement is isolated to one month.
- NSA weight collapses to near zero **and** the run does not improve MAE or
  stress behavior. A small NSA weight is acceptable if it is the mechanism that
  safely extracts nonlinear signal from a noisy branch, as in the 92.66 run.
- Stress months worsen materially.
- Runtime is not worth the gain.

### Phase 2B: Post-Baseline Source Attribution

Goal: separate promising data additions from harmful tuning interactions. The
latest run makes this a refinement exercise, not a rollback exercise.

Run these as close as possible to the same training/tuning settings:

| Variant | Economist Panel | Futures | SA-NSA Gap | New Feature Gen | Purpose |
|---|---:|---:|---:|---:|---|
| A | Off | Off | Off | Baseline-compatible | Control |
| B | On | Off | Off | Baseline-compatible | Economist panel value |
| C | Off | On | Off | Baseline-compatible | Futures value |
| D | Off | Off | On | Baseline-compatible | Gap feature value |
| E | On | On | Off | Baseline-compatible | Survey plus market interaction |
| F | On | Off | On | Baseline-compatible | Survey plus seasonal gap |
| G | Off | On | On | Baseline-compatible | Market plus seasonal gap |
| H | On | On | On | Baseline-compatible | All new sources, constrained tuning |
| I | On | On | On | Current feature gen | All new sources plus new feature blocks |

For each variant, report:

- Full MAE/RMSE/MedAE.
- Stress-panel MAE.
- Consensus-residual MAE.
- NSA plus adjustment metrics.
- Feature source counts.
- Selected economist/futures/gap features.
- Kalman source weights/noise.
- Runtime.

Decision rule:

- Keep a source if it helps full MAE, stress MAE, consensus-residual MAE, or
  uncertainty/fusion weighting without causing large instability.
- If a source helps only when tuning is constrained, the source is good and the
  tuning layer needs restraint.
- If a source is never selected and never improves residual/stress metrics, move
  it to low priority.

### Phase 3: One-Axis Ablations

Goal: Identify what matters.

Run controlled ablations around the 92.66 checkpoint and the best AWS finalist:

1. Dynamic-FS frequency only.
2. Feature cap only.
3. Reselection stages only.
4. Joint Optuna on/off.
5. Kalman acceleration/direction penalties on/off.
6. SA-NSA gap features on/off.
7. Economist panel feature variants.
8. Futures feature variants.
9. New feature-generation blocks.
10. NSA plus adjustment variants.
11. Consensus residual target variant.
12. Small Kalman lambda grid around `(5, 5)`: `(0, 0)`, `(2.5, 2.5)`,
    `(5, 5)`, `(10, 10)`, and asymmetric accel-heavy/direction-heavy variants.

Do not combine winners until single-axis behavior is known.

### Phase 4: SA Complement Tests

Goal: rebuild SA as a modern sidecar and decide whether it should enter as
features, a residual correction, a high-noise observation, a router, or a
model-average candidate.

Tests:

1. Old SA LightGBM unchanged, only as a stale baseline.
2. Fresh direct-SA LightGBM on the current feature universe with a direct-SA
   feature-selection target.
3. Direct consensus-residual model: predict `SA_revised - consensus`.
4. SA-derived target-history features only.
5. Consensus/economist/futures/gap feature blocks in direct SA, one at a time.
6. Direct SA acceleration classifier.
7. Direct SA quantile/error-width model.
8. Direct SA as high-noise Kalman observation.
9. Direct SA residual correction with bounded shrinkage.
10. Direct SA as model-average candidate with rolling constrained weights.

Accept SA only if:

- It improves full MAE or stress-panel MAE after prequential validation.
- It receives nontrivial rolling weight out-of-sample.
- It improves uncertainty/failure detection even if not point MAE.
- It recovers feature families that are plausibly SA-relevant and not merely
  noisy duplicates of consensus.

### Phase 5: Fork Acceleration Integration

Goal: Use acceleration signal without corrupting level forecast.

Steps:

1. Reproduce fork XGB/LGBM direct acceleration on current PIT data.
2. Confirm target definition matches operational SA revised acceleration.
3. Export monthly probability and confidence.
4. Test as:
   - Feature into Kalman/fusion.
   - Transition adjustment.
   - Observation-noise adjustment.
   - Regime router.
   - Model-average input.

Reject hard overrides unless they improve point forecast and stress-panel MAE.

### Phase 6: Model Averaging

Goal: Use several stable sources of truth.

Candidate pool:

- Consensus.
- Best AWS Kalman.
- Latest 92.66 Kalman.
- Recoverable-style 90.4 Kalman.
- NSA plus adjustment with bounded correction.
- Consensus-residual model.
- Direct acceleration adjusted model.
- Factor model.
- BVAR/shrinkage model.

Methods:

- Static constrained weights.
- Rolling constrained weights.
- Exponential loss weights.
- Weighted median.
- Huber ensemble.
- Stacking with strong ridge penalty.

Evaluation:

- Prequential only.
- No full-sample fit for reported performance.
- Compare to recoverable baseline and consensus.

### Phase 7: External Data Addition

Goal: Add only data with a credible timestamp and model role.

Order:

1. LSEG consensus microdata.
2. LSEG event-calendar surprises.
3. LSEG market-implied stress/rates.
4. WRDS CRSP stress features.
5. WRDS IBES/Compustat slow labor-demand factors.
6. WRDS/RavenPack/news if available.

Every new data source must pass:

- Timestamp validation.
- PIT join validation.
- Missingness report.
- Single-source feature audit.
- Incremental value test.
- Stress-panel test.

## Metrics To Report For Every Serious Run

Core:

- MAE.
- RMSE.
- MedAE.
- Bias.
- Tail_MAE.
- Directional accuracy.
- Production acceleration accuracy / operational accel-sign accuracy.
- Predicted-diff sign accuracy if separately reported.
- STD ratio.
- Diff STD ratio.

Relative:

- Delta versus consensus.
- Delta versus `2026-05-12_165541`.
- Delta versus `2026-05-16_124429`.
- Delta versus current head at time of run.
- Paired absolute-error difference.
- Paired squared-error difference.

Stability:

- Feature count by window.
- Feature source count by window.
- Feature overlap between windows.
- Kalman/fusion weights.
- Optuna param stability.
- Runtime.

Stress:

- Late 2022 panel.
- 2025-2026 panel.
- Negative/low-payroll months.
- High-consensus-disagreement months.
- High-claims-volatility months.
- High-SA-NSA-gap-volatility months.

Uncertainty:

- HAC standard error for loss delta.
- Moving-block bootstrap interval.
- Conformal interval coverage if quantiles are used.

## What We Want

Minimum acceptable next win:

- Beat `92.66` MAE or materially reduce stress-panel error with no major full
  MAE degradation.

Strong win:

- Full MAE below `90.4` while keeping acceleration accuracy near or above the
  latest 70.2%.
- RMSE no worse than recoverable baseline.
- Tail_MAE improved.
- 2022 stress months improved.
- Stable feature count/source composition.
- No reliance on unrecoverable artifacts.

Excellent win:

- Full MAE below `88`.
- Better than consensus in most regimes.
- Direct acceleration signal improves level forecast, not just acceleration.
- External data adds measurable, PIT-safe incremental value.
- Model averaging reduces catastrophic misses.

## Immediate Next Actions

1. Treat `_output/archive/2026-05-16_124429` as the current checkpoint.
2. Preserve the current code state in git so the 92.66 setup is recoverable.
3. Add a compact run manifest for `2026-05-16_124429`: config, selected
   features, joint-tuned params, and known caveats.
4. Triage AWS grid finalists using both baselines: `2026-05-12_165541` and
   `2026-05-16_124429`.
5. Select 3-5 finalists, not one.
6. Run post-baseline source attribution: economist panel on/off, futures on/off,
   SA-NSA gap on/off, new feature generation on/off.
7. Run one-axis ablations around the 92.66 checkpoint and best AWS finalist.
8. Build a sandbox modern direct-SA sidecar with its own SA feature-selection
   target and compare direct level, consensus-residual, and acceleration
   targets.
9. Reproduce fork direct-acceleration on current PIT data.
10. Build a small model-average layer over consensus and finalists, including
    direct SA only if it earns nonzero prequential weight.
11. Extend LSEG/WRDS data only after the current new-source attribution is done,
    starting with richer economist-panel metadata, event-calendar surprises, and
    pre-release futures windows.
