# Why higher reselection cadence wins (paradox investigation)

**Status:** open. Findings from grid search on 2026-05-16 → 17 (in progress at write time; expected ~64 cells, finishes Mon 18 ~05:00 UTC).

**TL;DR:** at `cadence=60` (select features once at start of 60-month backtest, never re-select) the model achieves MAE=87.71 (cell_07: cap=120, t=50). At `cadence=48` the best is 92.57 (cell_09: cap=60, t=50) — strictly worse. The intuition "more frequent reselection = adapts better to regime change" is wrong here. Dominant cause is **variance injected by each reselection compounding with the Kalman fusion's trailing-window weighting.**

---

## Empirical evidence

| Tier | Best cell | Best MAE | Best cap | MAE range across tier |
|---|---|---|---|---|
| cad=60 | cell_07 cap=120 t=50 | **87.71** | 120 | 87.71 → 98.41 (10.7 K) |
| cad=48 | cell_09 cap=60 t=50 | 92.57 | 60 | 92.57 → 108.00 (**15.4 K**) |
| cad=36 / 30 / 24 / 18 / 12 / 6 | (in progress) | | | expected to widen further |

Two patterns to watch as the grid finishes:
1. **MAE distribution spread grows tier-by-tier.** That's the smoking gun for variance injection.
2. **Optimal `cap` drops as cadence drops** (120 → 60 already; expect 20-40 at cad=6 if pattern holds).

---

## Diagnosis: three independent stochastic processes fire on every reselection

### 1. Boruta + dual-filter (`Data_ETA_Pipeline/feature_selection_engine.py`)
Prescreen is intentionally randomized (random-subspace LightGBM, shadow features). Re-running on overlapping data with slightly more rows produces a *different* top-N each time, even when the underlying truth is unchanged.

### 2. Pass-2 cap selection (`Train/train_lightgbm_nfp.py:_dynamic_reselection`)
Capping at e.g. 120 of ~1,500 candidates means small score differences (ties / near-ties) cause feature swap-in/swap-out across cadences. A feature that ranked 119 last cycle and 121 this cycle quietly enters/exits with no underlying signal change.

### 3. Optuna hyperparameter tuning (`Train/hyperparameter_tuning.py`, `Train/joint_tuning.py`)
Each reselection forces a *fresh* Optuna sweep because the feature space changed. With fixed `trials=25` budget, each sweep is shallow. At cad=6 you do 10 sweeps × 25 trials = 250 trials total, **but each one starts from scratch on a different feature space → no convergence carry-over.** Sweeps don't accumulate knowledge.

---

## Why this hurts the Kalman fusion specifically

The Kalman fusion (`Train/Output_code/consensus_anchor_runner.py`, configured via `joint_tuning.py`) weights the LightGBM channel against consensus using a **28-month trailing window** (`trailing_window=28` in cell_07's tuned params). It learns "how reliable is the model's deviation from consensus, given recent error structure?"

When features change every 6 months, the LightGBM's prediction characteristics (bias direction, error magnitude, variance) shift discretely. The Kalman is fitting **non-stationary residuals over a 28-month window that spans multiple feature regimes**. Two consequences:

- **Kalman defensively downweights the model channel.** `nsa_weight_scale=0.13` for the leader is small precisely because the optimizer sees noisy LightGBM contributions.
- **The little signal the model does have gets averaged out.** Months where the model was right get washed by months where it was right with *different features* — Kalman can't tell them apart.

At `cad=60` you have one stable LightGBM model. Its bias/variance is fixed. Kalman can learn a clean weight. Even if the static features are slightly suboptimal in absolute terms, the **predictability of their contribution** is much higher.

---

## Why optimal `cap` drops with cadence (corollary)

At `cad=60` (one selection), cap=120 is affordable — large feature set, but **selected once**, so over-parameterization is regularized by the single-point estimate. At `cad=48`, you're picking 120 features *twice* on overlapping data. The marginal 70 features (ranks 50-120) churn the most between cycles → noise dominates signal. Smaller cap (60) keeps only the high-confidence features that survive both selections. **Cap acts as a noise filter when reselection variance is high.**

---

## What this is NOT

- **Not** that dynamic FS is fundamentally bad — it's the **interaction between FS variance and the Kalman trailing-window assumption** that's bad.
- **Not** the per-tuning Optuna budget being too small (cad=60 also uses 25 trials and wins). It's that more tunings *don't accumulate knowledge*.
- **Not** look-ahead bias — PIT discipline is intact. This is genuinely "more updates = more noise injection."

---

## Mitigations worth trying (ranked by expected impact)

### 1. Sticky feature selection (highest leverage)
Require new features to beat the existing slot by a margin (e.g. 10% higher importance score) before swap. Converts random churn into evolutionary updates. Implementation:
- In `Train/train_lightgbm_nfp.py:_dynamic_reselection`, after computing new survivor set, intersect with previous survivors and only swap features whose score difference exceeds a threshold.
- Suggested threshold: 1 SD of historical score variance, or a simple 10-20% relative margin.

### 2. Deterministic FS seeds across reselections
Pin the RNG state used in random-subspace draws + Boruta so that churn comes from new data only, not from RNG. Implementation:
- Add `random_state=42 + cycle_idx * 1000` to the LightGBM calls inside `feature_selection_engine.py`'s random-subspace loop, so each cycle gets a deterministic-but-different seed.
- Quantify residual variance after this fix to know how much of the current variance is "pure RNG noise."

### 3. Warm-start Optuna
Pass previous best hyperparameters as a prior to the next tuning so you don't re-pay convergence cost on each reselection. Implementation:
- In `hyperparameter_tuning.py` / `joint_tuning.py`, accept an optional `warm_start_params` arg and use `study.enqueue_trial(warm_start_params)` before `optimize()`.
- Carry the previous cycle's best params through the orchestrator in `train_lightgbm_nfp.py`.

### 4. Cadence-aware Kalman reset
Currently `trailing_window=28` is fixed. When features change, partially reset the Kalman state (e.g. inflate the process noise covariance for one observation) so it doesn't try to extrapolate stale residual structure into the new regime. Implementation: in the Kalman update step, on reselection months, scale Q by a "reset factor" (3-10x).

---

## Suggested follow-up grids (after current 64-cell grid finishes)

### Grid A: stickiness × cap (test mitigation 1)
- `CADENCE_VALUES = [12, 24]` (the cadences that are currently losing badly)
- `STICKINESS_MARGIN = [0.0, 0.10, 0.20, 0.50]` (new env var)
- `CAP_VALUES = [60, 80]`
- 16 cells, expect ~25h on m7i.4xlarge with 3-parallel
- Success criterion: at least one cell with cad ≤ 24 cracks 90 MAE

### Grid B: deterministic seeds (test mitigation 2)
- Same `CAP × CADENCE × TRIALS` as current grid but with `NFP_FS_DETERMINISTIC=1` env
- Quantifies how much variance was pure RNG vs structural
- Re-run the same 8 cad=60 + 8 cad=48 cells with deterministic seeds, compare MAE spread

### Grid C: warm-start Optuna (test mitigation 3)
- Implement warm-start in `hyperparameter_tuning.py` first
- Re-run cad=12 and cad=6 tiers with/without warm start
- Hypothesis: warm-start cuts the per-tuning cost ~50% in convergence terms, reducing the noise per cycle

---

## What to do FIRST in the new conversation

1. **Wait for current grid to finish (~Mon morning UTC)**, then read the full `_output_grid_archive/grid_results.csv` from local laptop after `./aws/pull_grid_archive.sh`. Confirm or refute these predictions:
   - MAE spread per tier grows monotonically from cad=60 → cad=6
   - Optimal cap drops monotonically as cadence drops
2. **Run a controlled experiment locally first**: pick one cell config (e.g. cap=60, cad=12, t=25) and re-run it 5 times with identical config — observe MAE variance from pure stochasticity. That sets the noise floor.
3. **Implement mitigation 2 (deterministic seeds) first** — it's smallest code change and directly tests the variance hypothesis. If MAE variance drops sharply, the diagnosis is confirmed.
4. **Then mitigation 1 (sticky selection)** as the substantive fix.

---

## Key file references for the new session

- Grid search orchestrator: [Train/grid_search.py](Train/grid_search.py)
- Main training script (dynamic reselection lives here): [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py) — search `_dynamic_reselection`
- Feature selection engine: [Data_ETA_Pipeline/feature_selection_engine.py](Data_ETA_Pipeline/feature_selection_engine.py)
- Hyperparameter tuning (per-cycle Optuna): [Train/hyperparameter_tuning.py](Train/hyperparameter_tuning.py)
- Joint Optuna for Kalman fusion: [Train/joint_tuning.py](Train/joint_tuning.py)
- Kalman fusion consumer: [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py)
- AWS toolkit (for re-running grids on the EC2 box): [aws/README.md](aws/README.md)
- Grid archive on laptop (downloaded from S3): `_output_grid_archive/`
- S3 master copy: `s3://nfp-predictor-989571801493/_output_grid/`

---

## Current model context (cell_07, leader as of 2026-05-17)

- `MAE = 87.71` over 58-month backtest
- vs consensus mean (101.28) — model edge **+13.4%** all-time
- vs consensus median (100.10) — model edge **+12.4%** all-time
- **In last 36 months: model edge only +4.2% vs mean, +5.2% vs median**
- Model beats consensus median in **only 47%** of recent 36 months — wins fewer rounds but wins bigger when right
- Kalman tuned params: `trailing_window=28`, `nsa_weight_scale=0.133`, `half_life_years=2.41`
- Implication: most of model's edge is historical (COVID-recovery period when consensus was wildly wrong); in normal regimes the model barely differentiates

**The deeper question this analysis sets up:** can we get the model to differentiate from consensus more aggressively (the cell_07 design heavily anchors on consensus), AND keep that differentiation reliable? Right now the LightGBM channel is downweighted to 13% — fix the variance issue and you can credibly upweight it.
