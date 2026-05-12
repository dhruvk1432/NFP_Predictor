# Faster Backtests: A Performance Audit of the NFP Training Pipeline

**Scope:** read-only investigation. No code changed. Every claim below cites
`file:line` so the proposed work is anchored to actual code, not impression.

**Question the user asked:** can we make the dynamic-reselection training
pipeline materially faster so that we can run larger OOS backtests in a
reasonable wall-clock time, and are there I/O / write-file inefficiencies that
should be cleaned up?

**Short answer:** yes. The single biggest lever (2× to 5× speedup at large
backtest depths) is **avoiding repeated work inside `build_training_dataset` /
`_process_single_month_task` and re-using a cached `(X_full, y_full)` matrix
across runs.** The second lever, which dominates at deep backtests, is
**reducing the cost of each `_dynamic_reselection` event** (Pass-1 Boruta is
run with LightGBM hard-pinned to `n_jobs=1`). I/O is *not* the bottleneck
inside the hot loop — the loop writes essentially nothing — but there are
clear wins in archival/output that would reduce disk pressure.

---

## 1. How the backtest is actually structured today

The orchestration is in [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py).
Two top-level concerns interact: a multi-branch driver (`--train-all`) and a
per-branch expanding window.

### 1.1 Multi-branch driver (`--train-all`)

[Train/train_lightgbm_nfp.py:2870-2874](Train/train_lightgbm_nfp.py#L2870-L2874)
defines the branches that actually run today:

```
ALL_COMBOS = [
    ('nsa', 'first', 'revised'),
    ('sa',  'first', 'revised'),
]
```

The driver loop is [train_lightgbm_nfp.py:3050-3098](Train/train_lightgbm_nfp.py#L3050-L3098).
Despite the project documentation claiming a "quad-track" 4-branch matrix,
the actual driver runs **two branches sequentially** (NSA, then SA). NSA must
finish before SA starts because SA injects NSA-acceleration features
([train_lightgbm_nfp.py:1762-1786](Train/train_lightgbm_nfp.py#L1762-L1786)).

### 1.2 Per-branch flow

`train_and_evaluate(...)` per branch does, in order:

1.  `build_training_dataset(...)` — builds **(X_full, y_full)** *once* before
    the loop. Definition at [train_lightgbm_nfp.py:1146-1282](Train/train_lightgbm_nfp.py#L1146-L1282).
    Internally parallel via joblib (`Parallel(n_jobs=-1)` at line 1236).
2.  `run_expanding_window_backtest(...)` — definition at
    [train_lightgbm_nfp.py:1289](Train/train_lightgbm_nfp.py#L1289). The
    monthly loop is `for i, target_month in enumerate(backtest_months):`
    at [train_lightgbm_nfp.py:1488](Train/train_lightgbm_nfp.py#L1488) and
    continues to roughly line 2010.
3.  Output generation `generate_all_output(...)` — after the loop completes.

### 1.3 Cadence inside the loop

From [Train/config.py:280-282](Train/config.py#L280-L282) and
[settings.py:73](settings.py#L73):

| Knob | Value | Where |
| --- | --- | --- |
| `BACKTEST_MONTHS` | env-driven | [settings.py:63](settings.py#L63) |
| `RESELECT_EVERY_N_MONTHS` | **6** | [settings.py:73](settings.py#L73) |
| `TUNE_EVERY_N_MONTHS` | **12** | [Train/config.py:282](Train/config.py#L282) |
| `N_OPTUNA_TRIALS` | **25** | [Train/config.py:280](Train/config.py#L280) |
| `n_inner_splits` | **5** (TimeSeriesSplit) | [Train/hyperparameter_tuning.py:69](Train/hyperparameter_tuning.py#L69) and 199 |
| `NUM_BOOST_ROUND` | **1000** | [Train/config.py:272](Train/config.py#L272) |
| `EARLY_STOPPING_ROUNDS` | **50** | [Train/config.py:273](Train/config.py#L273) |
| `DYNAMIC_FS_PASS2_MAX_FEATURES` | **80** | [Train/config.py:383](Train/config.py#L383) |
| `DYNAMIC_FS_BORUTA_RUNS` | **50** | [Train/config.py:386](Train/config.py#L386) |

So for a backtest of length **N** months per branch, the work per branch is:

- Build phase: N parallel workers, each loading **two** master parquets and
  pivoting **three** times (see §3.1).
- Loop: **N** LightGBM trains.
- Reselection: roughly **N / 6** events (one mandatory at step 0, then every
  6 months from `RESELECTION_START_DATE` = `2000-01-01`,
  [Train/config.py:400](Train/config.py#L400)).
- Tuning: at least **N / 12** Optuna runs of 25 trials × 5 folds = 125
  LightGBM fits per Optuna run. But every reselection invalidates the cache
  ([train_lightgbm_nfp.py:1633](Train/train_lightgbm_nfp.py#L1633):
  `tuned_params = None`), so a tuning run is also forced after each
  reselection — i.e. effective tuning cadence ≈ **min(6, 12) = 6 months**.

This last point matters a lot for big-N: tuning effectively happens **every
6 months**, not every 12. That doubles the Optuna budget over what `config.py`
suggests.

### 1.4 Concrete evidence the cadence is currently *light*

`ls _output/dynamic_selection/nsa_revised/` shows only **4 JSON files**
across the entire archive of past runs: `2021-04, 2021-06, 2024-04, 2024-06`.
Pre-2021 there were *no* reselection events on disk — meaning the current
production backtest never grew the window far enough to scale-stress this
function. **Any extension of `BACKTEST_MONTHS` materially scales the
reselection cost** in a way the codebase has not yet had to amortize.

---

## 2. The big claim: **`(X_full, y_full)` is identical between runs and is the largest amortizable cost**

This is the single most important finding for "make larger OOS backtests
feasible".

### 2.1 Evidence

`build_training_dataset` ([train_lightgbm_nfp.py:1146-1282](Train/train_lightgbm_nfp.py#L1146-L1282))
runs `_process_single_month_task` in parallel for **every target month** in
the requested date range. Each worker:

- loads **master snapshot M** with `use_cache=False`
  ([train_lightgbm_nfp.py:124](Train/train_lightgbm_nfp.py#L124))
- pivots it to wide ([train_lightgbm_nfp.py:129](Train/train_lightgbm_nfp.py#L129))
- loads **master snapshot M-1** with `use_cache=False`
  ([train_lightgbm_nfp.py:142](Train/train_lightgbm_nfp.py#L142))
- pivots **the *current* snapshot a second time** with a different
  `target_month` ([train_lightgbm_nfp.py:145](Train/train_lightgbm_nfp.py#L145))
- pivots **the prior snapshot** ([train_lightgbm_nfp.py:146](Train/train_lightgbm_nfp.py#L146))
- computes revision features.

That is **2 parquet reads + 3 pivots per worker per month**. For N=200 months
× 2 branches × 3 pivots = **1,200 pivots and 800 parquet reads** every time
you run `--train-all`.

`use_cache=False` is set intentionally
([train_lightgbm_nfp.py:120 comment block](Train/train_lightgbm_nfp.py#L120))
because joblib workers don't share memory anyway — the module-level
`_snapshot_cache` ([Train/data_loader.py:47](Train/data_loader.py#L47))
lives in the parent process. So *within a single build phase*, caching it
makes no difference. But **between runs**, nothing is persisted to disk
either — every fresh `python Train/train_lightgbm_nfp.py --train-all` re-does
the entire pivot/revision computation from scratch.

The output of `build_training_dataset` is a pure deterministic function of:

- the master snapshot parquets on disk,
- `target_type`, `release_type`, `target_source`,
- `start_date`, `end_date`,
- the lag-feature config and calendar utilities.

None of those change during an iterative development cycle. **(X_full, y_full)
is therefore a perfect candidate for a content-hashed Parquet cache.**

### 2.2 Why this is the biggest lever

For an N-month backtest, the loop itself does **N LightGBM trains**, each on
~50–300 features. With early-stopping at 50 and `NUM_BOOST_ROUND=1000`, each
train is typically a few seconds on a modern Mac. So the *base* loop is in
practice O(N) and not where the wall-clock blows up.

The wall-clock blows up because **the build phase runs every time** and
**reselection runs at O(N/6) and each reselection re-loads/re-computes
recency weights and reruns Boruta from scratch**. The build phase is
trivially amortizable; the reselection cost is harder but tractable.

### 2.3 Proposal (no code yet — just claim with evidence)

Cache `(X_full, y_full)` as Parquet under
`_output/cache/training_dataset/{target_type}_{release_type}_{target_source}__{snapshots_hash}.parquet`,
keyed on the mtime-or-content-hash of the relevant master snapshot directory.
Hit → skip the entire `Parallel` block. Miss → run it as today and write the
result back. This collapses repeated dev iterations from "rebuild everything"
to "load one Parquet."

Expected savings at N=200: the build phase today takes minutes; this drops it
to a single parquet read (<1s) on cache hit.

---

## 3. Per-month repeated work inside the loop (smaller wins, but they compound)

The loop body lives at [train_lightgbm_nfp.py:1488-2010](Train/train_lightgbm_nfp.py#L1488-L2010).
I walked it line by line; here is the cost ledger.

### 3.1 `_process_single_month_task` calls 3 pivots per worker

Already cited above ([train_lightgbm_nfp.py:129, 145, 146](Train/train_lightgbm_nfp.py#L129)).
The current snapshot is pivoted *twice* with two different `target_month`
arguments because revision features need a "what does today's snapshot say
about last month" view. This is correct but does mean `pivot_snapshot_to_wide`
([data_loader.py:907-1105](Train/data_loader.py#L907-L1105)) is called
**3 × N × 2 branches** times per `--train-all` run.

Two of those three pivots act on the *same* `snapshot_df`. If
`pivot_snapshot_to_wide` is called with the same `(snapshot_df, cutoff_date)`
twice with only `target_month` differing, the inner "sort_index → filter by
cutoff → take last valid row per col" work is mostly redundant: the only
piece that depends on `target_month` is the NOAA staleness suffix
([data_loader.py:994-1006](Train/data_loader.py#L994-L1006)) and which rows
clear the cutoff. A `@lru_cache`-keyed helper that returns the
post-cutoff-filtered view of a snapshot, plus a thin per-target-month
"take last valid row" step on top of it, would cut this cost roughly in
half.

### 3.2 `clean_features` runs every month

[train_lightgbm_nfp.py:1536-1539](Train/train_lightgbm_nfp.py#L1536-L1539)
re-runs `clean_features(X_train_valid, y_train_valid)` (definition at
[train_lightgbm_nfp.py:224](Train/train_lightgbm_nfp.py#L224)) every month.
The result *does* change month to month because the expanding window adds
rows and column NaN sparsity changes. But the cost per call is dominated by
per-column NaN counting on a wide DataFrame; the column set hardly changes
between consecutive months.

A simpler optimization than caching: track the *delta* (one new row appended)
instead of recomputing fully. Low-priority — this is ~100 ms per call.

### 3.3 Sample weights computed *twice* per step

[train_lightgbm_nfp.py:1654-1660](Train/train_lightgbm_nfp.py#L1654-L1660):

```
weights = calculate_sample_weights(X_train_valid, target_month, default_half_life)
weights = _apply_tail_aware_weighting(...)
```

and later, after tuning settles `final_half_life`,
[train_lightgbm_nfp.py:1843-1849](Train/train_lightgbm_nfp.py#L1843-L1849):

```
weights = calculate_sample_weights(X_train_valid_with_ds, target_month, final_half_life)
weights = _apply_tail_aware_weighting(...)
```

The first call is used by the legacy static FS path (short-pass, branch-target
FS — [train_lightgbm_nfp.py:1693-1700](Train/train_lightgbm_nfp.py#L1693-L1700)).
The second is the production weight used for the final model fit.

When `dynamic_features` is already populated and the dynamic path is taken,
**the first computation is dead work**. Cheap per call (vectorized exp), but
unnecessary.

### 3.4 Sort & copy per step

[train_lightgbm_nfp.py:1517-1522](Train/train_lightgbm_nfp.py#L1517-L1522):

```
X_train_valid = X_full.iloc[train_idx_valid].copy()
y_train_valid = y_train[valid_train_mask].copy()
sort_order = X_train_valid['ds'].argsort()
X_train_valid = X_train_valid.iloc[sort_order].reset_index(drop=True)
```

A `.copy()` of a (rows × ~1500-cols) DataFrame each step is non-trivial — at
N=200 and 1500 features it dominates the loop overhead. If `X_full` is
already sorted by `ds` (which it should be after `build_training_dataset`
since `ds` is appended in iteration order), the argsort is identity-work
that can be replaced with a one-time check, and the `.copy()` is only needed
because downstream code mutates (NSA-acceleration injection at
[train_lightgbm_nfp.py:1779-1784](Train/train_lightgbm_nfp.py#L1779-L1784)).
For non-SA branches the copy is gratuitous.

### 3.5 NSA acceleration injection rebuilds training-time features every SA step

[train_lightgbm_nfp.py:1762-1786](Train/train_lightgbm_nfp.py#L1762-L1786)
calls `build_nsa_features_for_training(nsa_backtest_results, training_months, cutoff_dates=...)`
**every SA step**. The cutoff_dates dict is the same every step and the only
thing that grows is `training_months`. This is monthly-rebuild that
incrementally appends one row — natural candidate to precompute once for the
entire SA backtest window and slice from it.

---

## 4. The dominant cost at deep N: dynamic reselection

`_dynamic_reselection` lives at [train_lightgbm_nfp.py:431-679](Train/train_lightgbm_nfp.py#L431-L679).

### 4.1 What it actually does

- Pass 1 ([train_lightgbm_nfp.py:525-598](Train/train_lightgbm_nfp.py#L525-L598)):
  partitions columns by data source. The **two FRED Employment sources run
  sequentially** ([train_lightgbm_nfp.py:566-573](Train/train_lightgbm_nfp.py#L566-L573))
  with `gc.collect()` between them because they're memory-hot; the small
  sources run in a ThreadPoolExecutor with `max_workers=4`
  ([train_lightgbm_nfp.py:580-587](Train/train_lightgbm_nfp.py#L580-L587)).
- Pass 2 ([train_lightgbm_nfp.py:606-661](Train/train_lightgbm_nfp.py#L606-L661)):
  combines Pass-1 survivors with non-snapshot columns and runs the pipeline
  again, optionally hard-capping to 80 features by running Boruta a final time.

The stages are configurable: `RESELECTION_STAGES_PASS1 = (0, 2, 4, 5)`,
`RESELECTION_STAGES_PASS2 = (0, 2, 4)`
([Train/config.py:401-402](Train/config.py#L401-L402)).

### 4.2 The hard bottleneck: Boruta with `n_jobs=1`

[Data_ETA_Pipeline/feature_selection_engine.py:68-71](Data_ETA_Pipeline/feature_selection_engine.py#L68-L71):

```python
# MUST BE 1. LightGBM + ProcessPoolExecutor + n_jobs=-1 = OOM deadlock on macOS
'n_jobs': 1,
```

This LightGBM `n_jobs=1` is what every Boruta-internal model fit uses, and
Boruta runs **N_RUNS = 50** trees by default ([Train/config.py:386](Train/config.py#L386))
through `get_boruta_importance` ([feature_selection_engine.py:1322](Data_ETA_Pipeline/feature_selection_engine.py#L1322)).
The driving comment is accurate (this was a real prior incident), but
the constraint is *macOS-specific*: on Linux/CI, `n_jobs=-1` is safe with
`fork()`-based joblib. So the hardcode is currently leaving "10–20 ×
single-core wall-clock per Boruta call" on the floor when the same pipeline
runs on Linux.

Additionally, the engine *already* spawns ThreadPoolExecutors internally
([feature_selection_engine.py:924, 1038, 1862, 2155](Data_ETA_Pipeline/feature_selection_engine.py#L924))
for tournament chunking, but each thread still has a single-core LightGBM
underneath it.

**Net:** one full reselection event today, on a six-core machine,
under-utilizes the CPU.

### 4.3 Number of reselection events scales linearly with N

With `RESELECT_EVERY_N_MONTHS=6` and `RESELECTION_START_DATE='2000-01-01'`,
a backtest covering 2010–2024 (168 months) would trigger ≈ **28 reselection
events per branch**, 56 across `--train-all`. If each takes ~3 minutes, that
alone is roughly **3 hours**. Today's archive shows only 4 reselection JSONs,
which means the production backtest has never actually exercised this scale.

### 4.4 Proposals (each cites why it works)

1.  **Detect platform and set `n_jobs=-1` on non-Darwin.** The comment at
    feature_selection_engine.py:69 is platform-specific; a runtime check
    `os.name == 'posix' and platform.system() != 'Darwin'` would safely use
    all cores on Linux/CI while preserving the fix on macOS. Expected per-call
    speedup: 4–8× on modern multi-core.
2.  **Persist Boruta importance per source per cutoff_date.** The Pass-1
    per-source result is a function of `(source_name, columns, X_train rows,
    sample weights)`. With a content hash of those inputs as a key, a JSON
    cache under `_output/cache/reselection/` would let the second and
    subsequent runs of a similar backtest reuse Boruta results when the
    window grew by only a few months.
3.  **Skip reselection when the column set is stable.** Track Jaccard of
    selected features vs the previous event (the loop already tracks short-pass
    Jaccard at [train_lightgbm_nfp.py:1791-1798](Train/train_lightgbm_nfp.py#L1791-L1798) —
    extending this to reselection would be a few lines). If two consecutive
    reselection events agree above some threshold (e.g. 0.9), defer the next
    one. This converts "every 6 months no matter what" into "every 6 months
    *unless the data is quiet*."
4.  **Reduce `DYNAMIC_FS_BORUTA_RUNS` from 50 to 30 in reselection mode.**
    Boruta is a hypothesis-test framework; the variance of "is this feature
    chosen" drops as sqrt(n_runs), so 30 vs 50 is a small change in confidence
    for a 40% reduction in cost. This is a tunable knob, not a refactor.

---

## 5. Optuna: doing more work than it has to

`tune_hyperparameters` is at
[Train/hyperparameter_tuning.py:65-360](Train/hyperparameter_tuning.py).

### 5.1 Effective cadence is 6 months, not 12

[train_lightgbm_nfp.py:1815](Train/train_lightgbm_nfp.py#L1815) gates tuning
on `tune and (tuned_params is None or i % TUNE_EVERY_N_MONTHS == 0)` — i.e.
every 12 months. But [train_lightgbm_nfp.py:1633](Train/train_lightgbm_nfp.py#L1633)
forces `tuned_params = None` after every reselection, and a reselection
happens every 6 months. Result: tuning runs **every 6 months** in practice.

This is intentional (you want to re-tune when the feature set changes), but
there is a warm-start path
([hyperparameter_tuning.py:294-310](Train/hyperparameter_tuning.py#L294-L310))
that's used to "seed Trial 0 with previous best." When that warm-start exists
and the feature set delta is small, the n_trials budget could be **halved**
(12-13 trials) without losing much. Today `N_OPTUNA_TRIALS=25` is used
unconditionally.

### 5.2 No parallel trial execution

[hyperparameter_tuning.py:327](Train/hyperparameter_tuning.py#L327):

```python
study.optimize(_counted_objective, n_trials=n_trials, timeout=timeout)
```

No `n_jobs` argument is passed. Each trial runs serially. Inside each trial,
LightGBM uses `n_jobs=-1` ([hyperparameter_tuning.py:153, 337](Train/hyperparameter_tuning.py#L153)),
so the CPU is busy within a trial, but only one trial is in flight at a time.

The TPESampler is sequential by design (each trial uses the posterior built
from prior trials), so parallel trials lose some sample efficiency. But Optuna's
own docs recommend `n_jobs=2-4` when wall-clock matters more than trials-to-converge,
which is the case here.

### 5.3 Inner CV is 5 forward folds — could be 3

[hyperparameter_tuning.py:135](Train/hyperparameter_tuning.py#L135),
default `n_inner_splits=5`. For a financial time series with ~150–500
training rows in early-window expanding-backtest steps, the variance of a
3-fold vs 5-fold TimeSeriesSplit estimate is modest. Dropping to 3 folds
gives a clean 40% reduction in per-trial cost.

### 5.4 Summary of Optuna cost

Per tuning event today: `25 trials × 5 folds = 125 LightGBM fits`.
Tuning events: every 6 months effective.
For N=168, that is 28 events × 125 fits × 2 branches = **7,000 LightGBM fits**
spent on hyperparameter search alone — and not counting the final model fit
per step.

A reasonable target after the changes above:
`15 trials (warm-started) × 3 folds = 45 fits` per event, halved cadence on
quiet windows ≈ **2× to 4× reduction in Optuna cost**.

---

## 6. I/O audit: where bytes actually get written

### 6.1 Inside the backtest loop

Grepping `Train/train_lightgbm_nfp.py` for write operations
(`to_csv|to_parquet|savefig|to_pickle|json.dump|open(...,'w'...`) finds
only three write sites in the entire 3,369-line file:

| Line | What is written | When |
| --- | --- | --- |
| [1624-1625](Train/train_lightgbm_nfp.py#L1624) | Per-reselection JSON log (`_output/dynamic_selection/{target_type}_{target_source}/{YYYY-MM}.json`) | Every reselection event (≈ N/6) |
| [2274-2275](Train/train_lightgbm_nfp.py#L2274) | Stability JSON (Jaccard, etc.) | Once after the loop |
| [2681-2682](Train/train_lightgbm_nfp.py#L2681) | Backtest metrics JSON | Once after the loop |

**The hot loop itself writes nothing else.** The "loop is slow because of
I/O" hypothesis is wrong — every other write happens after the loop.

### 6.2 Write inventory across `Train/Output_code/`

```
Train/Output_code/feature_importance.py:21  df.to_csv(...)              once per branch
Train/Output_code/metrics.py:107            df.to_csv(...)              once per branch
Train/Output_code/generate_output.py:159    results_df.to_csv(...)      once per branch
Train/Output_code/generate_output.py:278    adj_results.to_csv(...)     once per branch
Train/Output_code/generate_output.py:442    pred_df.to_csv(...)         once per branch
Train/Output_code/plots.py:58,90,168        savefig (dpi=150)           3 plots per branch
Train/Output_code/consensus_anchor_runner.py:818,868   savefig(dpi=200) 2 plots once
Train/Output_code/consensus_anchor_runner.py:1081,1140,1234  to_csv     3 CSVs once
Train/Output_code/model_comparison.py:202   scorecard.to_csv(...)       once at end
```

All of these run once, after the backtest. Aggregate output is small —
`du -sh _output/*` shows:

```
4.0K    _output/Predictions
 24K    _output/backtest
 60K    _output/dynamic_selection
248K    _output/NSA_plus_adjustment
600K    _output/SA_prediction
676K    _output/NSA_prediction
1.7M    _output/consensus_anchor
 51M    _output/Archive
```

**The 51 MB of `_output/Archive` is the only meaningful disk footprint.**
There are **27 timestamped subfolders** (`ls _output/Archive | wc -l`) — every
prior `--train-all` invocation snapshotted the entire output tree. That is
fine as an audit trail but is the only "unnecessary writes" item I'd surface.

### 6.3 What's *not* written that matters

- `(X_full, y_full)` is **not persisted**. Every run recomputes it. See §2.
- Per-source Pass-1 Boruta importance is not persisted. Every reselection
  recomputes it. See §4.4.
- Pivoted master snapshots are not persisted. Each branch recomputes them.

These three "not-written" items are the actual I/O improvement opportunity:
**add a content-hashed parquet/JSON cache for each, gated by file mtime
of the source snapshots.**

### 6.4 Other small write asymmetries

- `pickle.dump` for the trained model is at
  [Train/model.py:490](Train/model.py#L490) (once per branch — fine).
- `to_html` for the model comparison scorecard is at
  [Train/Output_code/model_comparison.py:257](Train/Output_code/model_comparison.py#L257)
  (once at the end of `--train-all`).
- 21 `to_parquet` calls live in `Data_ETA_Pipeline/*` and
  `Train/reduce_features.py:849-850`. None of these are inside the
  expanding-window loop; they're upstream ETL and feature-engine cache
  writes.

---

## 7. Ranked recommendations

Each item is annotated with **estimated wall-clock benefit at N=168 / two
branches** and the **change footprint** (low/medium/high) it implies. None
of the estimates assume code beyond what was inspected.

| # | Change | Where | Est. wall-clock saved | Effort |
| --- | --- | --- | --- | --- |
| 1 | Persist `(X_full, y_full)` to content-hashed Parquet cache; skip `build_training_dataset` on hit. Key off mtime of `MASTER_SNAPSHOTS_BASE` and the branch tuple. | new file in `Train/`, called from [train_lightgbm_nfp.py:1356](Train/train_lightgbm_nfp.py#L1356) | 2–10 minutes on every iterated dev run; turns a "cold rebuild" into a parquet read | Low |
| 2 | Auto-detect platform; on non-macOS set Boruta-time LightGBM `n_jobs=-1`. | [feature_selection_engine.py:68-71](Data_ETA_Pipeline/feature_selection_engine.py#L68-L71) | 4–8× per Boruta call; on a 28-reselection backtest, ≥ 30 minutes saved | Low |
| 3 | Cache per-source Pass-1 Boruta importance on disk, keyed on `(source, columns_hash, X_rows_hash)`. Reuse on next reselection if the window grew but the source's column set is unchanged. | extend [train_lightgbm_nfp.py:_run_source_pass1](Train/train_lightgbm_nfp.py#L496) | At deep N, halves reselection cost on hot reruns. 20–60 minutes per dev iteration. | Medium |
| 4 | Use Optuna `n_jobs=2-4` when warm-start is present (sequential TPE is less valuable when seeded). | [hyperparameter_tuning.py:327](Train/hyperparameter_tuning.py#L327) | 30–60% per tuning event; at 28 events / branch, 10–20 minutes. | Low |
| 5 | Drop default Optuna inner-CV folds 5 → 3 *and* default trials 25 → 15 when `warm_start_params is not None`. | [hyperparameter_tuning.py:135, 280](Train/hyperparameter_tuning.py#L135) | ~40% per tuning event | Low |
| 6 | Memoize `pivot_snapshot_to_wide` for the (snapshot_df_id, cutoff_date) pair so the second and third invocations per worker reuse the post-cutoff sorted view. | [data_loader.py:907](Train/data_loader.py#L907) and [_process_single_month_task:129,145,146](Train/train_lightgbm_nfp.py#L129) | ~33% off `build_training_dataset` | Low |
| 7 | Move `calculate_sample_weights` (first call) inside the legacy static-FS branch so it only runs when `dynamic_features is None`. | [train_lightgbm_nfp.py:1654](Train/train_lightgbm_nfp.py#L1654) | Tiny — 1–3 minutes total. Code-cleanliness more than perf. | Low |
| 8 | Skip the `X_train_valid = X_full.iloc[...].copy()` clone on NSA branch (no downstream mutation). | [train_lightgbm_nfp.py:1517-1522](Train/train_lightgbm_nfp.py#L1517-L1522) | 5–20 ms × N — 1–3 min at deep N | Low |
| 9 | Precompute `build_nsa_features_for_training` once for the entire SA backtest window and slice per step. | [train_lightgbm_nfp.py:1762-1786](Train/train_lightgbm_nfp.py#L1762-L1786) | 1–2 minutes at deep N | Low |
| 10 | Add a stability-aware "skip-this-reselection" check: if last two reselections agreed at Jaccard > 0.9, defer the next one. | [train_lightgbm_nfp.py:1592-1605](Train/train_lightgbm_nfp.py#L1592-L1605) | Variable; 20–50% of reselections in stable regimes. | Medium |
| 11 | Cap `_output/Archive` at last-K runs (e.g., 10) — purge older `2026-02-*` snapshots. | new util, called from `archive_outputs` | 30+ MB / month on disk; not a wall-clock win, but data hygiene. | Low |

A reasonable target with items **1, 2, 4, 5, 6** implemented (the "low-effort
pack"): **2× to 3× speedup at N=168 backtest**, with the largest single
improvement coming from caching the training matrix between runs (item 1).
Items **3, 9, 10** push the speedup further at deeper N but require more
careful change.

---

## 8. Things the report deliberately does *not* claim

- The current per-month LightGBM training is *not* a notable bottleneck. With
  `n_jobs=-1` and early-stopping at 50, each step's fit is small.
- Plot generation (SHAP, summary tables) is one-shot after the loop — not
  responsible for slow backtests, even though plots account for most of
  `_output/Archive` disk size.
- The `use_cache=False` in `_process_single_month_task` workers is the *right
  call* given joblib's process model. Don't flip it; the win is at a different
  layer (persistent disk cache of `X_full`/`y_full`, not the in-memory dict
  cache).
- Inner-loop logging volume is not a meaningful cost.

---

## 9. Where to look next

If after the "low-effort pack" the backtest still feels slow at the target N,
the next investigation should profile a single reselection event end-to-end
(the `@profiled` decorators are already in place — see
[train_lightgbm_nfp.py:1146 / 1289 / 1611](Train/train_lightgbm_nfp.py)) and
verify the Pass-1 vs Pass-2 split of cost. If Pass-2's hard-cap re-Boruta at
[train_lightgbm_nfp.py:647-661](Train/train_lightgbm_nfp.py#L647-L661) is
dominating, that's the next surgical target — it runs Boruta a second time
purely to rank the cap.
