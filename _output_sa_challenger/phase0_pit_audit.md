# Phase 0 — PIT Audit (Consolidated)

Audit date: 2026-05-16
Scope: every surface the SA challenger will inherit — data ingestion, target build, feature engineering, feature selection engine, tuning loop, Kalman fusion, sidecar contract + producers, distribution/interval creation.

Three independent audits ran via the `point-in-time-auditor` agent (data/target/FE; FS+tuning+Kalman+intervals; sidecars). This document consolidates findings, ranks severity, and recommends what to fix **before** Phase 1 begins.

---

## Findings — severity-ordered

### HIGH — Global COVID winsorization computes quantiles over the full series

**Files**:
- [utils/transforms.py:724-746](utils/transforms.py#L724-L746) (definition)
- [Train/train_lightgbm_nfp.py:1946-1955](Train/train_lightgbm_nfp.py#L1946-L1955) (`X_full`, `y_full` winsorized *before* the expanding-window loop)
- [Train/data_loader.py:420-423](Train/data_loader.py#L420-L423) (`load_target_data` winsorizes raw y)
- [Train/data_loader.py:598-601](Train/data_loader.py#L598-L601) (`build_revised_target` winsorizes revised y)

**Mechanism**: `winsorize_covid_period` calls `non_covid.quantile(0.01/0.99)` over the **entire** series (1948→present), then clips the 3 COVID months (2020-03..2020-05) to those bounds. The clip is applied to `X_full`/`y_full` *before* the per-step backtest loop. For any backtest month `M ≥ 2020-06`, the training tape's COVID rows have been pre-clipped using bounds derived from data that includes months `≥ M` — i.e., the future relative to M.

**Affected rows**: ~3 per step (the COVID months in training), bounds at 1/99% so the clip is mild. Practical impact on a 92.66-MAE backtest is small. But it is a **strict PIT violation** and exactly the kind of subtle global-statistic leak that erodes credibility under audit.

**Fix**: move winsorization inside the expanding-window loop. At each step, compute quantiles only on `X_train` (rows with `ds < cutoff_date`), then clip the COVID rows of that same training slice. Same for the target in `load_target_data` / `build_revised_target` — defer winsorization to train-time or compute per-fold bounds.

**Blocks Phase 1**: YES. Every new SA sidecar inherits this leak via the shared target/feature loaders.

---

### MAJOR — HMM grid selection uses whole-history out-of-sample backtest

**File**: [experiments/sidecars/hmm_acceleration_sidecar.py:214-248](experiments/sidecars/hmm_acceleration_sidecar.py#L214-L248) (`run_hmm_grid`)

**Mechanism**: iterates `n_components ∈ {2,3,4}` × `covariance_type ∈ {diag, full}`, runs the **full expanding-window backtest for each**, writes `model_selection_report.csv` sorted by acceleration accuracy. If a downstream user picks the winner from this report and promotes it, the choice was made using out-of-sample but **whole-history** performance — overfit-to-test risk.

**Fix**: split the grid into an inner pre-2020 window for grid selection, freeze the winner, then evaluate on post-2020 unseen window. Document explicitly in the grid runner.

**Blocks Phase 1**: PARTIAL. We will lift this sidecar into SA, but we can defer the grid hardening to Phase 2 if we commit to a fixed HMM config now (the current production HMM config is presumably already a committed choice).

---

### MAJOR — Sidecar promotion gate computed on training/tuning data

**File**: [experiments/sidecars/common.py:227-249](experiments/sidecars/common.py#L227-L249) (`promotion_gate_passed` and `_base_metrics`)

**Mechanism**: the gate is purely a function of each sidecar's own expanding-window prediction CSV — no held-out fold. Combined with the HMM grid pattern (#2), a sidecar that overfits its expanding window will simultaneously produce favorable gate metrics and pass the gate. There is no holdout the gate has not seen.

**Fix**: compute gate metrics on the last N months only (e.g., trailing 24m), with the sidecar retrained on `< T-24m` for that window. Or split into a "tuning" run (pre-2022) and a "promotion" run (2022→).

**Blocks Phase 1**: PARTIAL. The SA challenger's promotion gate (Phase 5) must use a proper holdout — at minimum, hold out 2024-01→2026-03 from any tuning that feeds the gate decision.

---

### MINOR — Dead code in `_load_fusion_selection_target` masks broken convergence test

**File**: [Train/train_lightgbm_nfp.py:768-892](Train/train_lightgbm_nfp.py#L768-L892)

**Mechanism**: the dynamic-FS-HL feedback path was reverted (early return at line 783, comment dated 2026-05-15). The block at lines 785-892 is unreachable. `dynamic_fs_selection_hl.json` is never written. The `--iterate-fusion-tune` convergence test at lines 3915-3942 compares against this file, which is structurally broken (either missing → no convergence, or stale → spurious convergence).

**Why this matters for SA**: per memory, the user prefers `--iterate-fusion-tune` over single-pass `--train-all` for fusion tuning. The SA challenger should not inherit a broken convergence loop.

**Fix**: delete the dead block (lines 785-892), remove dangling references in [consensus_anchor_runner.py:1693-1725](Train/Output_code/consensus_anchor_runner.py#L1693-L1725) and [train_lightgbm_nfp.py:3838-3911](Train/train_lightgbm_nfp.py#L3838-L3911), and either retire the convergence test or wire it to a metric that *is* updated each pass (e.g., NSA Optuna best-trial half-life).

**Blocks Phase 1**: NO, but should be cleaned up early to keep `--iterate-fusion-tune --target sa` reliable.

---

### MINOR — `acceleration_classifier` z-score expanding window includes current row

**File**: [experiments/sidecars/acceleration_classifier_sidecar.py:148-149](experiments/sidecars/acceleration_classifier_sidecar.py#L148-L149)

```python
z = (z - z.expanding(min_periods=12).mean()) / z.expanding(min_periods=12).std()...
```

`pandas.Series.expanding()` is **inclusive of the current row**. So `composite_*_mean_z[ds=t]` is normalized using stats that include `ds=t`. With ~12-point windows in early years, the row contributes ~8% to its own z-score. The normalizer is a feature (not the label), so net information leak about the target is zero — but it's not strictly causal.

**Fix**: `z.shift(1).expanding(min_periods=12)...` or `rolling(window=N, closed='left')`.

**Blocks Phase 1**: NO.

---

### MINOR — `direct_sa_residual_sidecar` uses `y_sa_revised` for training labels (vintage mismatch)

**File**: [experiments/sidecars/direct_sa_residual_sidecar.py:35,148,157](experiments/sidecars/direct_sa_residual_sidecar.py#L35)

**Mechanism**: labels are `actual_mom - consensus_pred` where `actual_mom = y_sa_revised[ds=t]`. The final-revision NFP for recent months is **not knowable** at NFP release time `t` (revisions trickle in for ~5 years). Using the final-revision value for trailing-24m training labels leaks revision information.

**Architectural status**: known tradeoff per project CLAUDE.md. The pipeline targets `sa_revised` directly because that's the market-moving number, and the team has accepted that the most recent ~24 months of training labels are vintage-mixed.

**Fix for SA challenger**: use `y_sa_first_release` for the trailing-24m of training labels (rolling vintage merge). Bigger architectural decision — defer to Phase 1 design conversation.

**Blocks Phase 1**: PARTIAL — needs an explicit user call on whether to fix.

---

### LOW — `TimeSeriesSplit(n_splits=k, gap=0)` in Optuna inner CV

**Files**: [Train/hyperparameter_tuning.py:250](Train/hyperparameter_tuning.py#L250), [Train/joint_tuning.py:366](Train/joint_tuning.py#L366)

**Mechanism**: train-end abuts val-start by one month. Defense-in-depth issue only — no rolling-window features overlap across the boundary today, so the risk is zero in practice.

**Fix**: `TimeSeriesSplit(n_splits=k, gap=3)` to match the embargo used in SFS ([feature_selection_engine.py:2026-2061](Data_ETA_Pipeline/feature_selection_engine.py#L2026-L2061)).

**Blocks Phase 1**: NO.

---

### LOW — `release_date.bfill()` in NOAA pipeline

**File**: [Data_ETA_Pipeline/noaa_pipeline.py:513](Data_ETA_Pipeline/noaa_pipeline.py#L513)

**Mechanism**: `full["release_date"] = full["release_date"].ffill().bfill()` only fires when the left-merge produced NaT. The lag function is monotone in month, so `bfill` cannot map a row to an earlier date than its own month produces in practice. But the pattern is a footgun.

**Fix**: drop the `.bfill()`, log a warning if any release_date remains NaT.

**Blocks Phase 1**: NO.

---

### LOW — Uncapped `ffill` for non-NOAA columns in `pivot_snapshot_to_wide`

**Files**: [Train/data_loader.py:1009-1019](Train/data_loader.py#L1009-L1019), [Train/data_loader.py:1186-1195](Train/data_loader.py#L1186-L1195)

**Mechanism**: `wide_df[other_cols].ffill()` carries the last observed value forward indefinitely. PIT-safe (only uses rows with `date < cutoff`), but a discontinued series will quietly contribute a stale value forever — data-integrity drift rather than lookahead.

**Fix**: cap with `limit=3-6` for monthly cadence, or emit a staleness column.

**Blocks Phase 1**: NO.

---

### SUSPECTED — `NFP_Consensus_*` date semantics

**File**: [Data_ETA_Pipeline/load_unifier_data.py:295-298](Data_ETA_Pipeline/load_unifier_data.py#L295-L298)

**Question**: rows are stored with `date = month-in-which-the-poll-was-filed` and `release_date = MonthEnd(date)`. Downstream ([consensus_anchor_runner.py:228-254](Train/Output_code/consensus_anchor_runner.py#L228-L254)) joins by `obs_month == row['date']`. Is the consensus published in month M the forecast for month M's NFP (released early M+1), or for M+1's NFP?

- If "forecast for M's NFP": current alignment is correct. PIT-clean.
- If "forecast for M+1's NFP": still PIT-safe (uses snapshot < release date), but the variable is labeled at the wrong calendar slot and the consensus-anchor sidecar is one month off.

**Action**: confirm with the Unifier vendor / inspect a couple of raw records before promoting the challenger.

**Blocks Phase 1**: NO, but answer before Phase 5 promotion.

---

## Verified clean (OK)

These surfaces were audited and found PIT-tight:

- **Master snapshot construction**: [create_master_snapshots.py](Data_ETA_Pipeline/create_master_snapshots.py) — every snapshot at `snapshot_date` contains only data with `release_date < snapshot_date`. Per-source loaders (`load_fred_exogenous.py`, `load_unifier_data.py`, `load_economist_panel.py`, `adp_pipeline.py`, `load_prosper_data.py`) all enforce strict `< snap_date` on release dates.
- **Revised target construction**: [Train/data_loader.py:480-653](Train/data_loader.py#L480-L653) — `build_revised_target` uses the M+1 NFP snapshot, sets `operational_available_date = M+1 release date`. Strict pre-cutoff snapshot selection.
- **NFP release calendar**: pure calendar arithmetic, no data reads.
- **Feature engineering**: [Train/feature_engineering.py](Train/feature_engineering.py) and [Train/feature_engineering_sa_nsa_gap.py](Train/feature_engineering_sa_nsa_gap.py) — `_pit_filter` double-gates on `operational_available_date < snapshot_date` AND `ds < d`. All rolling/shift operations are strictly backward-looking. Calendar-month seasonal anchors use `.shift(1)` correctly.
- **Pivot to wide**: [Train/data_loader.py:906-1204](Train/data_loader.py#L906-L1204) — strict `wide_df.index < cutoff`, never reads future snapshots.
- **Universe cache**: [Train/universe_cache.py](Train/universe_cache.py) — triple-layered PIT enforcement (filename regex, glob filter, asof reassertion in `load_latest_universe`, `is_cache_fresh` rejects future-asof).
- **Feature selection engine** (all 6 stages): [Data_ETA_Pipeline/feature_selection_engine.py](Data_ETA_Pipeline/feature_selection_engine.py)
  - Stage 0 prescreen: ranks over bounded inputs only.
  - Stage 1 dual filter (purged correlation): 3-month embargo, bounded inputs.
  - Stage 2 Boruta: in-sample-only column shuffles.
  - Stage 3 vintage stability: `/dev/null` snapshots_dir keeps it Latest-only.
  - Stage 4 cluster redundancy: target excluded from Spearman matrix.
  - Stage 5 interaction rescue: bounded inputs.
  - Stage 6 SFS: `gap=3` embargo, bounded inputs.
- **Kalman fusion**: [Train/Output_code/consensus_anchor_runner.py:382-700](Train/Output_code/consensus_anchor_runner.py#L382-L700)
  - Trailing-window R_c / R_m / Q estimation strictly uses `df.iloc[:i]` (pre-step). Init values from pre-backtest slice only.
  - `_load_consensus_pit` / `_load_economist_panel_pit`: each `ds=t` reads only its own snapshot.
  - `_tune_kalman` walk-forward CV: nested expanding, no fold sees future data.
- **Prediction intervals**: [Train/train_lightgbm_nfp.py:2580-2607](Train/train_lightgbm_nfp.py#L2580-L2607) and [Train/Output_code/generate_output.py:356-381](Train/Output_code/generate_output.py#L356-L381) — quantiles over strictly past OOS residuals; current step's residual appended *after* interval is computed.
- **Iterate-fusion-tune**: [Train/train_lightgbm_nfp.py:3832-3957](Train/train_lightgbm_nfp.py#L3832-L3957) — each pass clears `UNIVERSE_CACHE_DIR`, subprocesses `--train-all` to recompute FS from scratch reading the prior pass's `tuned_params.json`. Intended convergence dynamic, not a leak.
- **Sidecar contract**: [experiments/sidecars/common.py:295-304](experiments/sidecars/common.py#L295-L304) — `validate_pit_predictions` hard-rejects `trained_through >= ds`. Loader at [integration.py:126](experiments/sidecars/integration.py#L126) re-applies the same filter. `_add_sidecar_meta_features` is purely per-row (axis=1).
- **All four current sidecar producers** (`acceleration_classifier`, `hmm_acceleration`, `bvar_prior`, `direct_sa_residual`): each refits per step on `design.iloc[:idx]`. `trained_through = train["ds"].max() < ds` by construction. HMM `predict_proba` on a 1-row sequence is forward-only (not Baum-Welch smoothed) — clean.

---

## Recommended remediation order

| # | Finding | Block Phase 1? | Effort | Order |
|---|---|---|---|---|
| 1 | Global COVID winsorization | **YES** | ~30 LOC | **Fix first** |
| 3 | Promotion gate uses train/tune data | PARTIAL | ~50 LOC | Fix before Phase 5 |
| 2 | HMM grid uses whole-history | PARTIAL | ~80 LOC | Fix before lifting HMM into SA (Phase 1) |
| 4 | Dead `--iterate-fusion-tune` code | NO | ~100 LOC delete | Clean opportunistically |
| 6 | `direct_sa_residual` vintage mismatch | DECISION | Architectural | Discuss before Phase 1 |
| 5 | `acceleration_classifier` expanding window | NO | 1 line | Cosmetic |
| 7 | Optuna `gap=0` | NO | 2 lines | Defense-in-depth |
| 8 | NOAA `bfill` | NO | 1 line | Cosmetic |
| 9 | Uncapped `ffill` | NO | 1 line | Cosmetic |
| 10 | Consensus date semantics (suspected) | DECISION | Verify | Confirm before Phase 5 |

---

## Bottom line

**One real strict PIT violation found** (global COVID winsorization). Practical impact on the 92.66-MAE backtest is small but non-zero. Fixing it is a ~30-LOC change and must happen before Phase 1 begins, both to clean the inheritance for SA and to harden the production NSA fusion at the same time.

**Two model-selection-bias surfaces flagged MAJOR** (HMM grid and promotion gate). Neither is strict lookahead, but both can produce optimistic backtest numbers via overfit-to-test. The HMM grid should be hardened before we lift the HMM sidecar into SA in Phase 1; the promotion gate should be hardened before Phase 5.

Everything else is cosmetic, deferrable, or already accepted as an architectural tradeoff. The pipeline's PIT discipline overall is strong: feature engineering, target construction, the 6-stage FS engine, the Kalman fusion, prediction intervals, and the sidecar contract are all clean.
