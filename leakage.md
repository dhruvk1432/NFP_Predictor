# Point-In-Time (PIT) Leakage Audit

**Original audit scope (Issues 1-7):** All files in `Data_ETA_Pipeline/` that ingest raw
data and produce per-month snapshots used downstream by `Train/`.

**Extended audit scope (Issues 8-14, added 2026-05-11 and 2026-05-17):** All files in `Train/` that consume
the snapshots, train the model, run the backtest, and produce final outputs. This covers the
training pipeline (`train_lightgbm_nfp.py`, `model.py`, `data_loader.py`,
`feature_engineering.py`, `revision_features.py`, `hyperparameter_tuning.py`,
`short_pass_selection.py`, `branch_target_selection.py`, `candidate_pool.py`, `baselines.py`,
`nsa_acceleration.py`), the post-train consensus-anchor stage
(`Output_code/consensus_anchor_runner.py`, `Output_code/generate_output.py`,
`rerun_post_train_adj_and_consensus.py`), and the predicted-adjustment sandbox
(`sandbox/experiment_predicted_adjustment.py`).

**Audit date:** 2026-05-10
**Last updated:** 2026-05-11 — **All 12 issues now closed.** Original 7 ETL issues all
closed (1, 2, 4-PopB, 5, 6 fixed in code; 3 and 4-PopA accepted as designed; 7 verified
empirically as not-a-leak). Extended audit added 5 issues (8-12) in the Train/ pipeline;
8, 9, 10, 11 fixed in code (2026-05-11); 12 verified by code-path tracing as not-a-leak.
On-disk artifacts (NOAA per-state parquets, NOAA snapshots, FRED exogenous snapshots,
master snapshots, `selected_features_*.json`) must be regenerated for the ETL-stage code
fixes to take effect (Issues 6 and 7 are no-op for data; no regeneration needed for
them). For Train/ stage fixes (8-11): consensus_anchor must be re-run (Issues 8, 9, 11)
and the SA branch must be re-trained (Issue 10) — see per-issue Status sections.

**Methodology:** Every snapshot file the pipeline writes is supposed to contain only data
that was *publicly available strictly before* the NFP release date for the observation month.
For each source we verified:

1. Where `release_date` (publication timestamp) comes from.
2. Whether the snapshot filter is `release_date < snap_date` (strict).
3. Whether the saved file is keyed correctly (so the master loader actually reads the right
   month).
4. Whether any value-side transform (CPI deflation, scaling, imputation, "current value"
   fallback) injects information from the future into a historical row.

**Severity legend:**
- **CRITICAL** — Future information measurably influences the value of a feature in a
  historical snapshot. Will bias the backtest.
- **HIGH** — Likely future information leak whose magnitude depends on a publication-lag
  assumption that is documented but unverified.
- **MEDIUM** — Not strictly a leak (data is *missing* rather than from the future), but
  the snapshot does not contain what it claims to contain. Still corrupts the backtest.
- **LOW** — Bounded edge case (e.g. pre-1990 only) or cosmetic issue (validation log
  inconsistent with actual filter).

---

## SUMMARY OF FINDINGS

| # | Status | Severity | File | Issue |
|---|--------|----------|------|-------|
| 1 | **FIXED** | ~~CRITICAL~~ | `noaa_pipeline.py` | ~~Damages inflated to *today's* CPI for every historical row~~ — switched to nominal USD |
| 2 | **FIXED** | ~~MEDIUM~~ | `noaa_pipeline.py` + `create_master_snapshots.py` | ~~NOAA snapshots are keyed by NFP release date, master expects observation month~~ — save key changed to `event_date` |
| 3 | **ACCEPTED** | ~~HIGH~~ | `load_prosper_data.py` | `release_date` set equal to the observation date. Confirmed 2026-05-10: the `prosper_v2` Unifier endpoint does not carry a publication timestamp — `asof_date`, `timestamp`, `date`, and `muts` all encode the same survey-period date. Treating data as available on the given day is the best we can do. |
| 4 | **CLOSED** | ~~HIGH~~ | `load_fred_exogenous.py` | Population B (5 shutdown-era weeks per series, Sep-Oct 2025, claimed available 26-54 days too early) **fixed in code**. Population A (pre-2009 catalog gap, 2,223 weeks using latest revised value as synthetic first release) **accepted as designed** — value-side bias only, median revision 0.0%, p99 ~7%; documented in source comment at [load_fred_exogenous.py:969-984](Data_ETA_Pipeline/load_fred_exogenous.py#L969-L984). |
| 5 | **FIXED** | ~~MEDIUM~~ | `noaa_pipeline.py` | ~~Pre-2007 NFP snapshots fell back to the earliest employment vintage and silently dropped 44 of 51 states~~ — fallback replaced with per-state earliest-vintage proxy; all 51 states now represented for all 437 production snapshots, max share-bias 0.18 pp |
| 6 | **FIXED** | ~~LOW~~ | `adp_pipeline.py` | ~~Validation message said `<=` but actual filter is `<`~~ — log message updated to match the filter |
| 7 | **CLOSED** | ~~NOTE~~ | `fred_employment_pipeline.py` | ~~Sub-component release dates forced to the `total_nsa` calendar — theoretical leak risk if a sub-component were ever later than `total_nsa`~~ — verified empirically (2026-05-11): 0 of 429,407 substitutions are leak-direction; 94% no-op, 6% anti-leak |
| 8 | **FIXED** | ~~HIGH~~ | `Train/Output_code/consensus_anchor_runner.py` | ~~`_load_consensus` reads only `[date, series_name, value]` (drops `release_date`) and aggregates per ds with `("value", "last")`~~ — replaced with `_load_consensus_pit` that reads `NFP_Consensus_Mean` from the master snapshot at each target month (PIT-correct by ETL contract). Verified: 317 rows / 0 value diff vs old loader on current data. |
| 9 | **FIXED** | ~~HIGH~~ | `Train/Output_code/consensus_anchor_runner.py` | ~~Optuna tunes Kalman / AccelOverride hyperparameters on the same `overlap_df` used to compute the published `comparison_metrics.csv`~~ — both `_tune_kalman` and `_tune_accel_override` now use `_walkforward_cv_score` (n_splits=5 nested expanding-window CV); each trial's score averages over chronological folds where each fold is strictly future relative to all earlier folds. Verified non-overlapping eval windows. |
| 10 | **FIXED** | ~~MEDIUM~~ | `Train/nsa_acceleration.py` | ~~`compute_nsa_acceleration_features` filters revised NSA / SA actuals by `ds < target_month`, not by `release_date < cutoff_date`~~ — added `cutoff_date` parameter (and `cutoff_dates` map for the training-month wrapper); when provided, filters revised actuals by `operational_available_date < cutoff_date`. SA backtest loop now passes the SA-target release date for both the prediction call and the training-month build. Verified: same-day-released revised M-1 row is correctly excluded. |
| 11 | **FIXED** | ~~LOW~~ | `Train/Output_code/consensus_anchor_runner.py` | ~~Kalman noise initialization (`R_c_init`, `Q_init`) uses the last 60 entries of the FULL consensus history~~ — restricted to `consensus_df[ds < first_backtest_ds]`. Init noise prior cannot peek into months that will later be evaluated. |
| 12 | **CLOSED** | **NOTE** | `Train/sandbox/experiment_predicted_adjustment.py` + `Train/Output_code/generate_output.py` | Sandbox `evaluate_models` selects "best" predictor by full-backtest composite — but production `_generate_adjustment_folder` hard-codes `ExpWeightedMedianCovidExcludedPredictor(half_life_years=3.0)`, so the sandbox selection is informational only. Not a leak in the production pipeline. |
| 13 | **FIXED** | **HIGH** | `utils/transforms.py`, `Train/train_lightgbm_nfp.py`, `Train/Output_code/{consensus_anchor_runner,metrics}.py`, `Train/sandbox/experiment_predicted_adjustment.py` | COVID winsorization was applied inconsistently: training X_train was winsorized per-step but X_pred at COVID target months was raw; consensus values were raw downstream of the model in the consensus_anchor stage; all sandbox adjustment predictors except the production default fitted on artificial winsorized adjustment values; metrics CSVs had no COVID stratification. **Six fixes applied (2026-05-11):** (1) centralized COVID constants in `utils/transforms`, (2) winsorize X_full once upfront in `run_expanding_window_backtest` (closes train/predict asymmetry + fixes production model), (3) winsorize consensus in `_load_consensus_pit` (Apr 2020 −14,448 → −525), (4) exclude COVID rows from Kalman R_c/R_m/Q noise estimation, (5) stratified `NonCovid_*`/`CovidOnly_*` columns in `compute_metrics` + `full_metrics`, (6) COVID-aware base class for `AdjustmentPredictor` (all 8 sandbox predictors honor `exclude_covid=True` by default). 11 unit tests in `tests/test_covid_winsorization.py` verify each fix. |
| 14 | **FIXED** | **HIGH** | `Train/data_loader.py`, `Train/train_lightgbm_nfp.py` | Branch-target `nfp_nsa_*` lag/rolling features were computed from the revised target table with observation-month `.shift()` only. These are not master-snapshot features. For strict release-time replay, M-1 revised target values released on the same NFP release date as month M must be unavailable. Fixed by masking revised target `y`/`y_mom` unless `operational_available_date < cutoff_date` before lag/rolling construction, in both backtest batch and prediction paths. Audit found 2,157 selected feature-month values changed from legacy non-null to PIT-missing. |

**All 14 issues are now closed.** Issues 1, 2, 4-PopB, 5, 6 fixed in code (ETL stage).
Issues 3, 4-PopA accepted as designed. Issues 7, 12 verified empirically/by code-path
tracing as not-a-leak. Issues 8, 9, 10, 11 fixed in code (Train/ stage; 2026-05-11).
Issue 13 (COVID winsorization consistency) fixed in code with 11 unit tests (2026-05-11).
Issue 14 (branch-target revised lag availability) fixed in code with unit tests and
`scripts/audit_current_pipeline_pit.py` artifacts (2026-05-17). The post-train
consensus_anchor stage must be re-run to regenerate `_output/consensus_anchor/`
artifacts (Issues 8, 9, 11, 13), and the full backtest must be re-run to incorporate
the upfront X_full winsorization (Issue 13), the strict NSA-acceleration cutoff
(Issue 10), and strict branch-target revised lag masking (Issue 14). Issues 1
(CPI leak), 2 (NOAA filename),
4-Population-B (FRED Exogenous shutdown-era backfill), 5 (NOAA pre-coverage fallback),
and 6 (ADP validation log) were fixed in code. Issue 3 (Prosper) was accepted because
the Unifier API does not carry a publication timestamp. Issue 4-Population-A was
accepted with a source-level documentation comment because the empirical revision
magnitude is small (median 0.0%, p99 ~7%) and the alternative would drop 42 years of
historical coverage. Issue 7 (FRED Employment total_nsa calendar enforcement) was
verified empirically as not-a-leak: 0 of 429,407 substitutions move a sub-component
to an earlier `realtime_start` than its true value; the function only ever no-ops
(94%) or pushes substitutions later (6% — anti-leak). The pipeline must be re-run to
regenerate downstream artifacts for Issues 1, 2, 4-PopB, and 5 — see the per-issue
Status sections below for the exact regeneration steps.

---

## ISSUE 1 — ~~CRITICAL~~ FIXED: NOAA damages are inflated to "today's" CPI

**Status:** FIXED 2026-05-10. Inflation logic removed from `aggregate_to_state_monthly`;
the `_real` damage columns now hold nominal USD (column names retained for downstream
schema compatibility). `load_cpi_series` deleted. `inflation_factor` removed from the
state grid, the per-state file ordering, the US-aggregate `agg(...)`, and the
`create_noaa_master` exclusion list. See diff at
[noaa_pipeline.py:539-590](Data_ETA_Pipeline/noaa_pipeline.py#L539-L590) and
[noaa_pipeline.py:679-693](Data_ETA_Pipeline/noaa_pipeline.py#L679-L693).

**Required regeneration to take effect:**

```
rm -rf data/Exogenous_data/NOAA_data/                 # 53 per-state parquets, US_NOAA_data.parquet, NOAA_master.parquet
rm -rf data/Exogenous_data/exogenous_noaa_snapshots/  # 432 PIT snapshots
rm -rf data/master_snapshots/{nsa,sa,_unified,source_caches,regime_caches,progress}
rm  data/master_snapshots/selected_features_*.json
python -m Data_ETA_Pipeline.noaa_pipeline
python -m Data_ETA_Pipeline.create_master_snapshots
```

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/noaa_pipeline.py](Data_ETA_Pipeline/noaa_pipeline.py)
**Lines (pre-fix):** 552–576 (computation), 681–683 (US aggregate), 980–984 (downstream
feature filter that *kept* the contaminated `*_real_weighted_log` columns).

### The code

```python
# Lines 552–558
cpi_today = cpi_df["cpi"].max()           # latest CPI in sample (approx "today")
cpi_df["inflation_factor"] = cpi_today / cpi_df["cpi"]
# NOAA-specific: Use documented 75-day lag from month-end
cpi_df["release_date"] = cpi_df["month"].apply(
    lambda m: calculate_noaa_release_date(m, lag_days=75)
)

# Lines 574–576
full["total_property_damage_real"] = full["total_property_damage"] * full["inflation_factor"]
full["total_crop_damage_real"]     = full["total_crop_damage"]     * full["inflation_factor"]
full["total_damage_real"]          = full["total_damage"]          * full["inflation_factor"]
```

The CPI series itself comes from `load_cpi_series(...)` ([noaa_pipeline.py:272-314](Data_ETA_Pipeline/noaa_pipeline.py#L272-L314)),
which fetches the **latest revised** CPI from the FRED v1 observations endpoint — no
vintage information.

Then `cpi_today = cpi_df["cpi"].max()` resolves to the highest CPI value in the entire
loaded series — i.e. the most recent month's CPI (~2025/2026 by the time the pipeline
runs).

### Why this is a PIT violation

`total_property_damage_real`, `total_crop_damage_real`, and `total_damage_real` are the
columns that flow downstream into the `*_weighted_log` features actually used by the model
([noaa_pipeline.py:1052-1058](Data_ETA_Pipeline/noaa_pipeline.py#L1052-L1058)).
Concretely, for an event in **January 1995**:

```
real_damage[1995-01] = nominal_damage[1995-01] × (CPI_2025 / CPI_1995)
```

`CPI_2025` was not knowable in 1995. Yet the snapshot built for the **1995-02 NFP release**
(predicting Jan 1995) would store this number, and the model would train on it.

It does not help that the per-snapshot release filter on line 1027 (`release_date <
snap_date`) is correct: that filter only gates *which rows* of NOAA data appear in the
snapshot. The *value* of every row that does pass the filter has already been multiplied
by `CPI_today / CPI_event_month`, where `CPI_today` is a future quantity from the
perspective of every historical snapshot.

### Why this matters for tree models

LightGBM is monotone-invariant within a single column **only if the same monotone transform
is applied to every value**. Here the multiplier `CPI_today / CPI_event_month` is *different
for every event_month*, because the denominator depends on the event month. So:

- The relative scale between damage in 1995 and damage in 2010 is not what was knowable in
  2010 (in 2010, damage_2010 / damage_1995 was computed using `CPI_2010 / CPI_2010 = 1`
  vs `CPI_2010 / CPI_1995`, not `CPI_2025 / CPI_2010` vs `CPI_2025 / CPI_1995`). The
  numeric ratio happens to be the same algebraically, but only because we assume CPI
  itself is not revised. **CPIAUCSL is in fact revised** (FRED stores vintages for it),
  so even that algebraic identity does not strictly hold.
- Splits learned on `*_real_weighted_log` are split on a quantity that uses future
  information.

### Fix applied

Adopted Option 1 (use nominal damages). The CPI fetch and `inflation_factor` plumbing are
gone; `*_real` columns are now `=` their nominal counterparts. This keeps every downstream
column name (`total_property_damage_real`, `total_crop_damage_real`,
`total_damage_real`, and the derived `*_weighted_log` features in
`create_weighted_national_aggregates`) intact, so no model-side or feature-selection JSON
schema changes are required — only the *values* change.

The per-snapshot vintage CPI alternative (Option 2 in the original analysis) was rejected:
it adds a second vintage dependency (CPIAUCSL with realtime_start filtering) for marginal
benefit, since the downstream `np.log1p` at
[noaa_pipeline.py:1019-1025](Data_ETA_Pipeline/noaa_pipeline.py#L1019-L1025) already
compresses long-run scale drift in the weighted aggregate.

**Tradeoff retained for the record:** "$50M damage" means something different in 1995 vs
2024 in nominal terms, so feature thresholds the model learns on recent data may not
transfer cleanly to early data. This is a statistical efficiency loss only — no leakage.

---

## ISSUE 2 — ~~MEDIUM~~ FIXED: NOAA snapshots are saved under the wrong month key

**Status:** FIXED 2026-05-10. The save key in `create_noaa_weighted_snapshots` was
changed from `snap_date` (NFP release date — i.e. the *next* month) to `event_date`
(observation month), matching the convention used by every other source and assumed by
`create_master_snapshots._snapshot_path`. The PIT row-level filter is unchanged: it still
uses `snap_date` to gate which NOAA rows enter the snapshot, so the 75-day NOAA
publication lag is fully respected. Only the filename labelling was wrong. See diff at
[noaa_pipeline.py:1092-1106](Data_ETA_Pipeline/noaa_pipeline.py#L1092-L1106).

**Required regeneration to take effect:** the per-state and master files do *not* change,
but the existing NOAA snapshot files are now labelled wrong. They must be deleted and
rewritten:

```
rm -rf data/Exogenous_data/exogenous_noaa_snapshots/decades/
rm -rf data/master_snapshots/{nsa,sa,_unified,source_caches,regime_caches,progress}
python -m Data_ETA_Pipeline.noaa_pipeline      # regenerates with correct filenames
python -m Data_ETA_Pipeline.create_master_snapshots
```

If you have already followed the Issue 1 cleanup commands above, you can skip this step
— Issue 1 deletes the same directories.

The original analysis follows for historical reference.

---

**Files:**
- Save side (pre-fix): [Data_ETA_Pipeline/noaa_pipeline.py:1150-1155](Data_ETA_Pipeline/noaa_pipeline.py#L1150-L1155)
- Path utility: [Data_ETA_Pipeline/utils.py:13-41](Data_ETA_Pipeline/utils.py#L13-L41)
- Load side: [Data_ETA_Pipeline/create_master_snapshots.py:640-643](Data_ETA_Pipeline/create_master_snapshots.py#L640-L643), called from [create_master_snapshots.py:973](Data_ETA_Pipeline/create_master_snapshots.py#L973) and [create_master_snapshots.py:1184](Data_ETA_Pipeline/create_master_snapshots.py#L1184).

### The mismatch

NOAA writes its snapshot using the **NFP release date** (which falls in the *next* month),
not the observation month:

```python
# noaa_pipeline.py, lines 1150–1155
for i, (idx, row) in enumerate(nfp_releases.iterrows(), 1):
    event_date = row['ds']
    snap_date = row['release_date']  # NFP release date

    # Create directory using utility
    save_path = get_snapshot_path(NOAA_SNAPSHOTS_DIR, snap_date)   # <-- snap_date, not event_date
```

`get_snapshot_path` keys the file by month-of-the-second-arg:

```python
# utils.py, lines 31–41
month = obs_month.strftime('%Y-%m')
save_dir = base_dir / "decades" / decade / year
return save_dir / f"{month}.parquet"
```

So for the **January 2024** observation (NFP released **Feb 2 2024**), NOAA writes
`exogenous_noaa_snapshots/decades/2020s/2024/2024-02.parquet`.

But every other source — ADP ([adp_pipeline.py:284-292](Data_ETA_Pipeline/adp_pipeline.py#L284-L292)),
Prosper ([load_prosper_data.py:383](Data_ETA_Pipeline/load_prosper_data.py#L383)),
Unifier ([load_unifier_data.py:417](Data_ETA_Pipeline/load_unifier_data.py#L417)),
FRED Employment ([fred_employment_pipeline.py:1076-1079](Data_ETA_Pipeline/fred_employment_pipeline.py#L1076-L1079)),
FRED Exogenous ([load_fred_exogenous.py:1111](Data_ETA_Pipeline/load_fred_exogenous.py#L1111)) —
keys its snapshot by the **observation month**, e.g. `2024-01.parquet`.

The master loader assumes the observation-month convention:

```python
# create_master_snapshots.py, lines 640–643
def _snapshot_path(base_dir: Path, date_ts: pd.Timestamp) -> Path:
    decade = f"{date_ts.year // 10 * 10}s"
    year = str(date_ts.year)
    return base_dir / decade / year / f"{date_ts.strftime('%Y-%m')}.parquet"
```

It is called with `obs_month` from the NFP release map (e.g. [create_master_snapshots.py:1184](Data_ETA_Pipeline/create_master_snapshots.py#L1184)),
so for the Jan 2024 master it asks for `…/2024/2024-01.parquet` from every source.

### What the master actually loads for NOAA

For the Jan 2024 observation (snap_date = Feb 2 2024) the master asks NOAA for
`2024-01.parquet`. That file does exist in the NOAA directory — but it was written by
the previous iteration (December 2023 observation, snap_date = Jan 5 2024). That
December snapshot was filtered with `release_date < Jan 5, 2024`.

So the Jan 2024 master's NOAA contribution **silently drops every NOAA storm row whose
publication date falls between Jan 5 and Feb 2 2024**, even though that data was
publicly available at the Feb 2 NFP release.

### Why this is "MEDIUM" not "CRITICAL"

This loses data rather than injects future data — i.e. it is *anti*-leakage for any given
feature value, but it makes the master snapshot a structurally wrong representation of
"what was known on snap_date." It also makes the NOAA contribution roughly 30 days stale
vs every other source. For a model that already has many other (more frequently revised)
NOAA-adjacent features, the realised loss may be modest, but the bug undermines the very
PIT contract the pipeline is documented as enforcing.

### Fix applied

`save_path = get_snapshot_path(NOAA_SNAPSHOTS_DIR, event_date)` (was `snap_date`). The
"skip if exists" log message was also updated to print `event_date.strftime('%Y-%m')` so
log output matches the new filename convention.

---

## ISSUE 3 — ~~HIGH~~ ACCEPTED: Prosper `release_date` set equal to the observation date

**Status:** ACCEPTED 2026-05-10. Live inspection of `unifier.get_dataframe(name="prosper_v2",
key=...)` confirmed the schema returns these columns and nothing else:

```
['asof_date', 'timestamp', 'symbol', 'segment', 'source', 'question_id',
 'question_text', 'answer_id', 'answer_text', 'value', 'muts', 'date']
```

Numerically `asof_date == timestamp == date`, and `muts` is the same value re-encoded as
microseconds since epoch (verified for three sample rows: `1746057600000000` →
`2025-05-01`, `1677628800000000` → `2023-03-01`, `1727740800000000` → `2024-10-01`). There
is no publication-date field. Without an external lag table, `release_date = date` is the
only available signal, so the pipeline retains its current behavior. Code unchanged at
[load_prosper_data.py:124-131](Data_ETA_Pipeline/load_prosper_data.py#L124-L131).

If a per-question publication-lag table is sourced later (e.g. directly from Prosper),
revisit by setting `release_date = date + lag_for(question)` and re-running the snapshot
loop.

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/load_prosper_data.py:124-131](Data_ETA_Pipeline/load_prosper_data.py#L124-L131)

```python
out_df = pd.DataFrame({
    'date':         pd.to_datetime(prosper_df['date']).dt.to_period('M').dt.to_timestamp(),
    'release_date': pd.to_datetime(prosper_df['date']),     # <-- equals 'date'
    'value':        prosper_df['value'].values,
    'series_name':  create_series_name(question, answer, symbol),
    'series_code':  create_series_code(symbol, answer_id),
})
```

`prosper_df['date']` is the date from the Unifier `prosper_v2` endpoint, which is the
**survey period date** (typically the start or middle of the month the survey covers),
not the date the survey results were published.

Down the snapshot loop, the standard PIT filter is then applied
([load_prosper_data.py:386](Data_ETA_Pipeline/load_prosper_data.py#L386)):

```python
snap_data = combined_df[combined_df['release_date'] < snap_date].copy()
```

Because `release_date` *is* the observation date, this filter implicitly assumes survey
results are public on the day the survey period starts. Prosper monthly surveys are
typically released with a multi-week lag. If the actual release lag is, say, 30 days,
then a survey collected during March 2024 (date = 2024-03-01) is in fact published in
late March/early April 2024 — but the snapshot for the March 2024 NFP (released early
April) treats it as available from March 1 onwards, and the snapshot for the
**February 2024 NFP** (released early March) would also treat the March-period survey
as available before snap_date because `2024-03-01 < 2024-03-08` and so on.

### Why this is "HIGH" not "CRITICAL"

The leak exists, but its magnitude depends on Prosper's actual publication lag, which we
have not verified empirically. Some Prosper series may legitimately be published at or
near observation time (e.g. forward-looking forecasts collected in real time). For the
lagged ones, the model is trained to use survey responses one cycle earlier than they
were actually available, which is exactly the kind of lookahead the rest of the pipeline
takes pains to avoid.

### Resolution

Option 1 (pull publication timestamp from Unifier) was investigated: **the timestamp
does not exist in the API response**. Schema dump shows `asof_date`, `timestamp`,
`date`, and `muts` are all the same survey-period date.

Option 2 (apply a per-question publication-lag offset) requires a lag table we do not
have. Until such a table is sourced from Prosper or another reference, the pipeline
proceeds with `release_date = date` and the issue is closed as accepted.

---

## ISSUE 4 — ~~HIGH~~ CLOSED: FRED Exogenous "current series" backfill

**Status:** CLOSED 2026-05-10. Issue split into two distinct populations after empirical
investigation; both are now resolved.

- **Population B (forward-looking time leak): FIXED in code.** The `late_start_set`
  discriminator at
  [load_fred_exogenous.py:936-953](Data_ETA_Pipeline/load_fred_exogenous.py#L936-L953)
  was tightened from a single `lag > 30 days` rule to a **two-condition** rule:

```python
catalog_start = vintage_df['realtime_start'].min()
catalog_gap = earliest_vintage - earliest_vintage.index
in_catalog_window = earliest_vintage <= (catalog_start + pd.Timedelta(days=14))
gap_is_large = catalog_gap > pd.Timedelta(days=30)
late_start_set = set(earliest_vintage.index[in_catalog_window & gap_is_large])
```

A date is now eligible for the `obs+7` synthetic-release backfill **only if both**:

(a) its earliest stored vintage sits at the ALFRED catalog start (≈2009-09-10 for
    CCSA/CCNSA), indicating an actual catalog gap rather than a real publication
    delay; AND
(b) the catalog gap is large (>30 days) — so the synthetic obs+7 timestamp is
    materially earlier than what the catalog would say, and reflects DOL's true
    historical ~7-day publication cadence.

**What this fixes (Population B):** the 5 weekly observations in Sep-Oct 2025
whose actual vintages lagged 33-61 days (consistent with a government-shutdown
delay). Previously the `>30d` rule classified these as "data quality issues" and
overwrote them with synthetic `obs+7`, claiming the values were knowable 26-54
days **before** they actually became public. The new condition (a) excludes them
because their `earliest_vintage` (2025-11-20) is far from the catalog start.
They now retain their true late timestamps.

- **Population A (value-side bias from latest-revised proxy): ACCEPTED as designed.**
  Pre-2009 weekly observations whose only ALFRED vintage sits at the catalog start
  remain backfilled to `obs+7`, with the **value** taken from the latest-revised
  `current_series`. The synthetic obs+7 release date is approximately correct for
  DOL's actual ~7-day publication cadence and needs no fix. The value-side bias is
  small enough to live with for three reasons:

  1. **Empirical magnitude is small.** Comparing first-release vs latest-revised on
     the post-2009 universe (where both are observable) gives a direct lower bound on
     Population A's value error: median 0.0%, mean 0.7%, p99 ~7% for CCSA; median 0.0%,
     mean 0.12%, p99 ~1.4% for CCNSA. Tree models tolerate this level of noise.

  2. **The natural "fix" — using the catalog-start vintage value (2009-09-10) instead
     of `current_series`** — moves the needle by almost nothing. For obs in 1990, the
     value at `realtime_start=2009-09-10` already incorporates ~19 years of revisions;
     the additional 17 years between 2009 and today contribute essentially zero further
     revision because revision activity decays exponentially with time-since-observation.
     ~6 lines of code change for a cosmetic improvement.

  3. **The clean alternative — dropping pre-2009 weekly claims entirely** — would lose
     42 years of historical coverage, including the recession-era levels of `CCSA_*` /
     `CCNSA_*` features the model needs to learn from. Net negative.

  The trade-off is now documented in source at
  [load_fred_exogenous.py:969-984](Data_ETA_Pipeline/load_fred_exogenous.py#L969-L984)
  so future maintainers understand why `current_series` is used as the value fallback.
  If a true first-release archive becomes available later (e.g. directly from DOL),
  revisit by replacing the `current_df` fallback there.

**Sandbox verification (2026-05-10):** identical-input parity test on live
CCSA + CCNSA data confirmed exactly 5 dates per series shifted forward
(+26 to +54 days, all to 2025-11-20), zero dates shifted earlier, and 10
representative pre-vintage-era control dates (1967-2009) unchanged.

**Required regeneration:** the existing FRED exogenous snapshots embed the
leaky obs+7 timestamps for the 5 shutdown weeks. To take effect:

```
rm -rf data/Exogenous_data/exogenous_fred_data/decades/
rm -rf data/master_snapshots/{nsa,sa,_unified,source_caches,regime_caches,progress}
rm  data/master_snapshots/selected_features_*.json
python -m Data_ETA_Pipeline.load_fred_exogenous
python -m Data_ETA_Pipeline.create_master_snapshots
```

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/load_fred_exogenous.py:923-963](Data_ETA_Pipeline/load_fred_exogenous.py#L923-L963)

```python
# Lines 923–934
elif name in claims_series:
    vintage_df = _rate_limited_fetch(fred.get_series_as_of_date, code, as_of_date='2100-01-01')
    ...
    current_series = _rate_limited_fetch(fred.get_series, code)        # <-- LATEST revised values
    current_df = current_series.to_frame(name='value').reset_index()
    ...

# Lines 936–963 — synthesize "first releases" from current series
earliest_vintage = vintage_df.groupby('date')['realtime_start'].min()
late_start_dates = earliest_vintage[
    earliest_vintage > (earliest_vintage.index + pd.Timedelta(days=30))
].index
late_start_set = set(late_start_dates)

if late_start_set:
    df = vintage_df[~vintage_df['date'].isin(late_start_set)].copy()
else:
    df = vintage_df.copy()

# Add synthetic first releases for dates with late vintages
missing_dates = (
    set(current_df['date']) -
    set(df['date'])
)

if missing_dates:
    missing_df = current_df[current_df['date'].isin(missing_dates)].copy()
    # LESS PESSIMISTIC: approximate weekly claims lag as 7 days
    missing_df['realtime_start'] = missing_df['date'] + pd.Timedelta(days=7)
    df = pd.concat([df, missing_df], ignore_index=True)
```

### Why this is a leak

`fred.get_series(code)` returns the **latest revised value** for each date — there are no
vintages on this endpoint. The code then takes those revised values and inserts them as
synthetic "first releases" with a 7-day lag.

For any date that is in `missing_dates` (i.e. `late_start_dates` plus any date entirely
absent from vintages), the row that ends up in the snapshot is:

- **Value:** the *latest revision* of that week's claims figure, often from years later.
- **`release_date`:** the original observation date + 7 days, i.e. *before* the actual
  first publication of the value.

The PIT filter at [load_fred_exogenous.py:1117](Data_ETA_Pipeline/load_fred_exogenous.py#L1117)
(`valid = df[df['realtime_start'] < snap_date]`) then happily lets that row through for
any snapshot at least ~8 days after the observation week. The model sees the most-revised
value at a time when the original first-release value (which would have differed) was the
only thing actually known.

This is the canonical "current value used as if it were the first release" leak.

### Scope

- Affects only `CCNSA` and `CCSA` (`claims_series`).
- The `late_start_dates` filter targets dates whose first vintage is more than 30 days
  after the observation date — i.e. exactly the dates where vintage data is most
  unreliable.
- For dates with reliable vintages, the vintage path runs normally and there is no leak.

### Suggested fix

Either:
1. **Stop synthesizing first releases.** If a date has no vintage and only a
   forward-revised value, drop it (the LightGBM model handles NaN natively). This
   removes the leak at the cost of slightly more NaNs in early data.
2. **Use the observation date itself as `realtime_start`** for synthesized rows, then
   shift it well past the actual publication. A 7-day approximation is too generous; for
   weekly claims released on Thursdays, `next Thursday after observation week end` is the
   correct convention (see `clean_weekly_release_dates` already in this file at lines
   201–248).

The cleanest fix is to delete lines 952–963 entirely and let LightGBM handle the missing
weeks as NaN. The weekly-claims feature pipeline already aggregates by NFP release window
([aggregate_weekly_to_monthly_nfp_based](Data_ETA_Pipeline/load_fred_exogenous.py#L266-L369)),
so missing weeks would become missing monthly aggregates, which the model already handles.

---

## ISSUE 5 — ~~LOW~~ MEDIUM, FIXED: NOAA pre-coverage employment-weight fallback

**Status:** FIXED 2026-05-11. The fallback at
[noaa_pipeline.py:806-870](Data_ETA_Pipeline/noaa_pipeline.py#L806-L870) was replaced
with a **per-state earliest-vintage proxy**: `get_state_employment_weights` now first
attempts the strict-PIT path (`realtime_start < snap_date`) for every state; for any
state without a strict-PIT vintage, it falls back to that state's earliest stored
vintage's value of the relevant observation month. This produces a full 51-state
weighting for all 437 production snapshots, with a small empirical bias on the relative
shares (the actual quantity used).

The original fallback path used `realtime_start == earliest_vintage` with the GLOBAL min
earliest vintage, which had two defects: (a) the **value** of every observation in that
vintage already incorporated years of revisions made *after* `snap_date` (classical PIT
leak), and (b) the row-filter silently **dropped 44 of 51 states** because their earliest
vintage is later than the global min — the resulting "national" weighted aggregate was
actually a 7-state regional aggregate (AR, IL, IN, KY, MO, MS, TN) with no representation
of CA, TX, NY, FL, etc. Severity raised from LOW to MEDIUM in light of (b).

**Why a proxy and not a hard skip:** the empirical bias on the proxy is tiny on the
quantity we actually use (relative employment shares):

| Obs date probed | Median \|Δ share\| | p95 | Max |
|---|---|---|---|
| 1990-06-01 | 0.0023 pp | 0.0225 pp | 0.032 pp |
| 1995-01-01 | 0.0034 pp | 0.0247 pp | 0.037 pp |
| 2000-12-01 | 0.0031 pp | 0.0272 pp | 0.067 pp |
| 2005-06-01 | 0.0041 pp | 0.0325 pp | 0.18 pp (CA) |

These are *lower bounds* on the true revision magnitude (computed as drift between the
2007-06-19 vintage and the latest revised value, since no earlier vintage exists to
compare). For a state with a ~12% share, max bias is ~0.3% relative — well below the
noise floor of the log-transformed weighted storm-damage features that consume these
weights. The trade-off is analogous to Issue 4 Population A's acceptance of
latest-revised as a first-release proxy. The skip-only alternative would lose the NOAA
storm-damage signal for 17 years of training data — net negative.

**Empirical scope (with `START_DATE=1990-01-01`):** of 437 NFP snapshots from 1990
onward, the new code's two paths split as:

- **Pre-coverage proxy path** fires for ~209 snapshots (1990-02 → 2007-06):
  - 185 in 1990-02 → 2005-06: 0/51 states have strict-PIT vintage; 51/51 use proxy
  - ~24 in 2005-07 → 2007-06: 7/51 states have strict-PIT vintage; 44/51 use proxy
- **Strict-PIT path** fires for 228 snapshots from 2007-07 onward (51/51 strict-PIT).

**Sandbox verification (2026-05-11)** on live cache `state_employment_vintages_nsa_2026-05-06.parquet`:

| snap_date | path | top-3 weights (correct historical structure) |
|---|---|---|
| 1990-02-02 | proxy 51/51 | CA=0.115, NY=0.075, TX=0.064 (NY > TX in 1990 ✓) |
| 1995-02-03 | proxy 51/51 | CA=0.107, TX=0.069, NY=0.068 (TX overtaking NY ✓) |
| 2003-04-04 | proxy 51/51 | CA=0.111, TX=0.072, NY=0.065 |
| 2005-06-17 | proxy 51/51 | CA=0.111, TX=0.072, NY=0.064 |
| 2006-01-06 | proxy 44 + PIT 7 | CA=0.111, TX=0.074, NY=0.063 |
| 2007-06-01 | proxy 44 + PIT 7 | CA=0.111, TX=0.075, NY=0.063 |
| 2007-07-06 | strict PIT 51/51 | CA=0.111, TX=0.075, NY=0.063 (matches 2007-06 — proxy ≈ PIT at boundary) |
| 2010-01-08 | strict PIT 51/51 | CA=0.108, TX=0.079, NY=0.066 |
| 2024-12-06 | strict PIT 51/51 | CA=0.114, TX=0.090, FL=0.063 (FL > NY in 2020s ✓) |

The cross-boundary continuity (2007-06-01 → 2007-07-06 weights virtually identical)
demonstrates the proxy and strict-PIT paths produce equivalent results where they
overlap.

**Why we cannot just pull more historical vintages.** Verified directly against ALFRED's
`vintagedates` endpoint (2026-05-11): for every state employment series we use
(`{state}NAN`), ALFRED's earliest stored vintage is 2007-06-19 — except for the 7
Mississippi-basin states whose archive starts 2005-06-17. Alternative variants tested
(`CANA` SA, `CAUR` unemployment) have the same hard floor at 2007-06-19. The data is
absent on FRED's side; this is not a pull-side fix.

The **observation data**, however, goes back to 1939 (in the 7 Mississippi-basin states'
2005 vintage) or 1990 (in the 44 other states' 2007 vintage). The proxy path exploits
this: for a 1995 snapshot, every state has at least one stored vintage that contains
a 1995 observation — that observation is just from a 2005 or 2007 vintage rather than
a true 1995 vintage. The bias is the cumulative revision history baked into the
post-1995 → 2007 portion of the vintage.

**Required regeneration:** the existing NOAA snapshots embed the old 7-state
weighted aggregates for ~209 historical months. To take effect:

```
rm -rf data/Exogenous_data/exogenous_noaa_snapshots/decades/
rm -rf data/master_snapshots/{nsa,sa,_unified,source_caches,regime_caches,progress}
rm  data/master_snapshots/selected_features_*.json
python -m Data_ETA_Pipeline.noaa_pipeline
python -m Data_ETA_Pipeline.create_master_snapshots
```

If you have already followed the Issue 1/2/4 cleanup commands above, you can skip this
step — those instructions delete the same directories.

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/noaa_pipeline.py:864-896](Data_ETA_Pipeline/noaa_pipeline.py#L864-L896)

```python
# Lines 877–887 (pre-fix)
earliest_vintage = vintages_df['realtime_start'].min()

if snap_date < earliest_vintage:
    # Fallback: Use earliest vintage
    logger.warning(f"Snapshot {snap_date.date()} predates earliest vintage "
                   f"({earliest_vintage.date()}). Using earliest vintage as proxy.")
    known_df = vintages_df[vintages_df['realtime_start'] == earliest_vintage].copy()
else:
    # Normal case: Filter to vintages known BEFORE snap_date (strict <)
    known_df = vintages_df[vintages_df['realtime_start'] < snap_date].copy()
```

For a snapshot whose `snap_date` predates the earliest available state-employment
vintage, the function fell back to using the *earliest available* vintage — which is
*from the future* relative to that snapshot. The original audit framed this as a
"pre-1990" issue based on a stated assumption that ALFRED state-employment vintages
"mostly start in the early 1990s." The empirical record contradicts this:

- ALFRED's earliest stored vintage is **2005-06-17** (7 states) and **2007-06-19**
  (44 states). Confirmed by direct query of FRED's `vintagedates` endpoint
  (2026-05-11).
- With `START_DATE=1990-01-01`, this fallback fires for ≈185 production snapshots,
  not "pre-1990 only."

### The two distinct defects

**(A) Value-side PIT leak.** The 2005-06-17 vintage value of, e.g., 1995-01 employment
already reflects ~10 years of revisions; it is not what was knowable in 1995. Magnitude
on the 7 fallback states (1990-2005 obs): median |Δ| 0.20%, p95 0.85%, p99 1.0%, max 1.2%.
Modest by itself.

**(B) Structural state-coverage bug** (not in the original audit writeup). The fallback
expression `realtime_start == earliest_vintage` keeps only rows from the 2005-06-17
vintage — i.e. only the 7 Mississippi-basin states. Texas, California, NY, Florida,
Pennsylvania, Ohio — all dropped. The resulting "national weighted average" of NOAA
storm damages over 1990-2005 is in fact a 7-state-only weighted average, in which
Mississippi gets >5% of the weight (vs its true ~1%) and the entire West Coast
contributes 0%. The same defect carries through ≈24 snapshots in the 2005-07 → 2007-06
transition window even though the explicit fallback isn't triggered there — there just
aren't enough vintages yet.

### Why MEDIUM, not LOW

The audit's LOW rating rested on two assumptions that don't hold in this codebase:
"pre-vintage fires only before ~1990" (wrong — fires for 209 of 437 post-1990 snapshots
in production) and "production usage typically focuses on the post-2000 era" (`.env`
has `START_DATE=1990-01-01`). Defect (B) corrupts a feature the model thinks is
national-level for ~48% of the months it trains on.

### Fix applied

`get_state_employment_weights` now uses two paths in sequence:

1. **Strict-PIT** — for each state, take the latest vintage with `realtime_start <
   snap_date`. Used for snap_dates after full vintage coverage (≈2007-07-01 onward).
2. **Per-state earliest-vintage proxy** — for any state with no strict-PIT vintage,
   use that state's earliest stored vintage (2005-06-17 for the 7 Mississippi-basin
   states; 2007-06-19 for the other 44).

Both paths produce a full 51-state weighting. The proxy path carries a small
revision-bias on the values used (bounded above by ~0.18 pp on shares for any state),
which is well below the noise floor of the log-transformed weighted storm-damage
features. This is materially better than the alternatives:

- Old fallback (`realtime_start == GLOBAL min`) → silently dropped 44 states for 209
  snapshots; "national" aggregate was a 7-state regional aggregate.
- Hard skip (`raise ValueError`) → loses 17 years of NOAA storm signal in training.
- Per-state earliest proxy (current) → all 51 states represented, max share-bias
  0.18 pp, NOAA storm signal preserved across full backtest.

The empirical justification for accepting the proxy bias is the same form as Issue 4
Population A's acceptance of latest-revised as a first-release proxy: the bias is small
relative to the noise floor of downstream features, and the alternative is much
worse.

---

## ISSUE 6 — ~~LOW~~ FIXED: ADP validation message inconsistent with filter

**Status:** FIXED 2026-05-11. Validation log message at
[adp_pipeline.py:342-345](Data_ETA_Pipeline/adp_pipeline.py#L342-L345) updated to use
`<` (matching the actual filter at line 264). No data regeneration needed — log-only
change.

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/adp_pipeline.py:264](Data_ETA_Pipeline/adp_pipeline.py#L264) (filter), [adp_pipeline.py:339-345](Data_ETA_Pipeline/adp_pipeline.py#L339-L345) (validation)

The actual snapshot filter is correct:

```python
# Line 264 — strict less-than
snapshot_df = df[df['release_date'] < snap_date].copy()
```

But the validation log message uses `<=`:

```python
# Lines 342–345
if max_release <= snap_date:
    logger.info(f"  Point-in-time: OK (all releases <= snapshot)")
else:
    logger.error(f"  Point-in-time VIOLATION: {max_release} > {snap_date}")
```

This is purely cosmetic — `max_release` will always be `< snap_date` because of the
strict filter on line 264, so the `<=` check is satisfied. But the log statement is
misleading: it claims `<=` is the contract, when the contract is actually `<`. A future
maintainer who reads only the validation function could reasonably infer that same-day
data is allowed.

### Suggested fix

Change the validation to `if max_release < snap_date:` and the message to `"all releases
< snapshot"`.

---

## ISSUE 7 — ~~NOTE~~ CLOSED: Total NSA calendar propagation is provably anti-leak

**Status:** CLOSED 2026-05-11. Verified empirically against the live FRED Employment
audit (`data/fred_data/_audit_asof_2026-05-06.parquet`, 546,634 rows, 258 series): the
calendar substitution at
[fred_employment_pipeline.py:2155-2179](Data_ETA_Pipeline/fred_employment_pipeline.py#L2155-L2179)
**never moves a sub-component to an earlier `realtime_start`** than its true value.
Out of 429,407 substitutions where a sub-component vintage matches a `total_nsa`
vintage by `(ds, vintage_idx)`:

| Direction | Count | % | Effect |
|---|---|---|---|
| `delta == 0` (no-op) | 405,451 | 94.42% | Sub-component already matched total_nsa |
| `delta < 0` (sub earlier than total) | 23,956 | 5.58% | Substitution pushes realtime_start *later* — anti-leak |
| `delta > 0` (sub later than total) | **0** | **0.00%** | Would-be leak case — empirically does not occur |

Maximum delta across the entire dataset is exactly 0.0 days. The `combine_first` overwrite
is bidirectional in principle, but in practice ALFRED never records a sub-component vintage
with a `realtime_start` strictly later than `total_nsa` for the same vintage_idx — the
BLS Employment Situation Report ships the entire table B-1 atomically, and ALFRED's
per-series ingestion timing only ever puts sub-components at-or-before total_nsa. So the
calendar enforcement only ever tightens the PIT contract; it never relaxes it.

Severity downgraded from NOTE to CLOSED. The original concern was theoretical; the
empirical record refutes it. The function is purely defensive.

The original analysis follows for historical reference.

---

**File:** [Data_ETA_Pipeline/fred_employment_pipeline.py:2155-2179](Data_ETA_Pipeline/fred_employment_pipeline.py#L2155-L2179)

`collapse_latest_asof()` aligns every sub-component series's `realtime_start` to the
`total_nsa` series's release calendar:

```python
# Lines 2155–2179 (excerpt)
total_uid = "total_nsa" if "total_nsa" in aligned["unique_id"].values else None
if total_uid:
    total = aligned[aligned["unique_id"] == total_uid].copy()
    total = total.sort_values(["ds", "realtime_start"])
    total["vintage_idx"] = total.groupby("ds").cumcount()
    ...
    aligned = aligned.merge(calendar, on=["ds", "vintage_idx"], how="left")
    aligned["realtime_start"] = aligned["total_realtime_start"].combine_first(aligned["realtime_start"])
```

In practice all sub-components are released alongside `total_nsa` (the BLS Employment
Situation Report releases the whole table B-1 at once). But ALFRED's per-series
`realtime_start` is sometimes a day or two off from the official release date due to FRED's
own ingestion timing. This step forces them all to a consistent calendar.

The original audit flagged a theoretical concern: if a sub-component's true `realtime_start`
were *later* than `total_nsa`'s, this code would make the row appear to be available
earlier than it really was — a small leak. The empirical check above (2026-05-11) shows
this case never occurs in the actual data: ALFRED's per-series ordering always has
`sub_realtime_start <= total_realtime_start`, so the substitution is always either a no-op
(94.4%) or anti-leak (5.6%).

---

## VERIFICATIONS — code paths that look risky but are correct

The following were checked in detail and are PIT-correct:

### V1 — Strict `<` on every snapshot filter
Confirmed in:
- ADP: [adp_pipeline.py:264](Data_ETA_Pipeline/adp_pipeline.py#L264)
- Prosper: [load_prosper_data.py:386](Data_ETA_Pipeline/load_prosper_data.py#L386)
- Unifier (vectorized): [load_unifier_data.py:233](Data_ETA_Pipeline/load_unifier_data.py#L233)
- Unifier (single-row): [load_unifier_data.py:156](Data_ETA_Pipeline/load_unifier_data.py#L156)
- NOAA filter: [noaa_pipeline.py:1027](Data_ETA_Pipeline/noaa_pipeline.py#L1027)
- NOAA employment weights: [noaa_pipeline.py:887](Data_ETA_Pipeline/noaa_pipeline.py#L887)
- FRED Employment audit collapse: [fred_employment_pipeline.py:2182](Data_ETA_Pipeline/fred_employment_pipeline.py#L2182)
- FRED Exogenous: [load_fred_exogenous.py:1117](Data_ETA_Pipeline/load_fred_exogenous.py#L1117)
- FRED Exogenous final per-snapshot filter: [load_fred_exogenous.py:1314](Data_ETA_Pipeline/load_fred_exogenous.py#L1314) (`>= snap_date` is dropped)
- FRED Exogenous daily features cutoff: [load_fred_exogenous.py:1135](Data_ETA_Pipeline/load_fred_exogenous.py#L1135) (`snap_date - 1 day`, conservative because daily series get `realtime_start = date + 1 day`)

### V2 — Unifier missing-`first_release_date` backfill is conservative
The Unifier API sets `last_revision_date = timestamp` whenever `first_release_date`
is missing — i.e. it forges a same-day "publication." The pipeline correctly **rejects
this trap** in
[get_effective_release_and_value_vectorized](Data_ETA_Pipeline/load_unifier_data.py#L165-L238)
and instead backfills the release date from the *empirical median lag for that series*
([load_unifier_data.py:222](Data_ETA_Pipeline/load_unifier_data.py#L222)). For rows with a
real `first_release_date`, the chosen value is the most recent one whose
`last_revision_date < snap_date` — a textbook PIT join.

### V3 — `compute_features_wide` and `compute_all_features` are PIT-correct *within a snapshot*
File: [utils/transforms.py](utils/transforms.py)

All within-series operations (`diff`, `pct_change`, `rolling`, `expanding`, `shift`) are
strictly backward-looking. They are applied **after** the snapshot has been PIT-filtered,
so they cannot inject future data into a row that was not in the snapshot. The lag
operations at [transforms.py:316-337](utils/transforms.py#L316-L337) shift values forward
in time (a row at date T sees the value from T-lag), which is the correct direction.

### V4 — FRED Exogenous expanding z-scores are PIT-correct
[load_fred_exogenous.py:770-774, 834-837](Data_ETA_Pipeline/load_fred_exogenous.py#L770-L774)

`pd.Series.expanding()` is strictly backward-looking. The expanding mean / std at date D
is computed only from values at dates ≤ D. Pre-computing once on the full history before
slicing to the snapshot does not change this — the expanding statistic for each date is
self-contained.

### V5 — FRED Exogenous SP500 52-week-high drawdown is PIT-correct
[load_fred_exogenous.py:673](Data_ETA_Pipeline/load_fred_exogenous.py#L673)

`rolling(window=252)` is backward-looking. The 252-day high at date D uses only D-251
through D.

### V6 — Feature selection uses `<= asof_month` for historical regimes
[create_master_snapshots.py:361-373](Data_ETA_Pipeline/create_master_snapshots.py#L361-L373) and
[create_master_snapshots.py:1804-1825](Data_ETA_Pipeline/create_master_snapshots.py#L1804-L1825).

When `--with-selection` is used, regime cutoffs are the hardcoded list at
[create_master_snapshots.py:79-86](Data_ETA_Pipeline/create_master_snapshots.py#L79-L86)
(1998, 2008, 2015, 2020, 2022, 2025). Each regime selects features using the snapshot at
or before that cutoff, then those features are used for predictions until the next
cutoff. The *target* used for selection is loaded as the full target series, but it is
intersected with the snapshot's date index (which goes only up to `asof_month`), so
target values from the future of the cutoff do not enter the correlation / Boruta
calculations. **PIT-correct for all months ≥ first cutoff (1998).** For months *before*
1998, features are selected from the 1998 regime — which used post-1995 target data
relative to e.g. a 1995 prediction. This is feature-selection leakage for backfilled
months pre-1998, but production runs typically begin well after 1998.

The default mode (`_run_unified_no_selection`) stores all features and defers selection
to backtest time, sidestepping this concern entirely.

### V7 — VIX / SP500 / Credit / Yield / Oil monthly aggregation is PIT-correct
The monthly aggregator filters daily data with `cutoff_date = snap_date - 1 day`
([load_fred_exogenous.py:1135](Data_ETA_Pipeline/load_fred_exogenous.py#L1135) etc.) and
sets `release_date = end-of-month` ([load_fred_exogenous.py:651, 814, 868](Data_ETA_Pipeline/load_fred_exogenous.py#L651)).
Final per-snapshot filter at line 1314 then drops any aggregate with `release_date >=
snap_date`. The combined effect is that the obs-month aggregate (the one being predicted)
is correctly *excluded* from the snapshot for that prediction, and prior-month aggregates
are included only if their end-of-month is strictly before snap_date. Both ends of the
inequality are handled.

---

## RECOMMENDED REMEDIATION ORDER

1. ~~**Issue 1 (CPI inflation).**~~ **DONE 2026-05-10** — switched `*_real` damages to
   nominal, deleted `load_cpi_series`, removed `inflation_factor` from state grid, US
   aggregate, and `create_noaa_master` exclusion list.
2. ~~**Issue 2 (NOAA filename).**~~ **DONE 2026-05-10** — `snap_date` → `event_date` at
   the save call; "skip if exists" log message updated.
3. ~~**Issue 4 (FRED Exogenous claims backfill).**~~ **CLOSED 2026-05-10** —
   Population B (shutdown-era forward-looking leak) eliminated in code by adding a
   catalog-window check to the `late_start_set` discriminator. Population A
   (pre-2009 catalog gap, value-side bias only) accepted as designed — empirical
   revision magnitude is small (median 0.0%, p99 ~7%), the alternative would drop
   42 years of historical coverage, and the trade-off is now documented in the source
   file. Revisit only if a first-release archive becomes available externally.
4. ~~**Issue 3 (Prosper release_date).**~~ **CLOSED 2026-05-10** — Unifier `prosper_v2`
   endpoint does not carry a publication timestamp; `release_date = date` is the best we
   can do with available data. Revisit if a per-question lag table is sourced later.
5. ~~**Issue 5 (NOAA pre-coverage fallback).**~~ **DONE 2026-05-11** — replaced the
   global-min `realtime_start == earliest_vintage` fallback (which silently dropped
   44 of 51 states) with a per-state earliest-vintage proxy. Strict-PIT path runs
   for snap_dates with full 51-state coverage (~2007-07 onward, 228 snapshots);
   proxy path runs for the remaining 209 production snapshots, with empirical
   share-bias bounded at 0.18 pp (justified analogously to Issue 4 Population A).
   All 51 states represented for every snapshot. Sandbox-verified on live cache.
6. ~~**Issue 6 (ADP validation log).**~~ **DONE 2026-05-11** — log message and operator
   updated to `<` to match the actual filter. No regeneration needed (log-only).

**Action required for Issues 1, 2, 4-Population-B, and 5 to take effect:** the on-disk
NOAA per-state files, NOAA snapshots, FRED exogenous snapshots, master snapshots, and
`selected_features_*.json` caches were all built with the old (leaky) values, old
(mis-keyed) filenames, old (shutdown-era leaky) FRED claims timestamps, and old
(7-state-only) NOAA weighted aggregates for ~209 historical months. They must be
deleted and regenerated end-to-end:

```
rm -rf data/Exogenous_data/NOAA_data/
rm -rf data/Exogenous_data/exogenous_noaa_snapshots/
rm -rf data/Exogenous_data/exogenous_fred_data/
rm -rf data/master_snapshots/{nsa,sa,_unified,source_caches,regime_caches,progress}
rm  data/master_snapshots/selected_features_*.json
python -m Data_ETA_Pipeline.noaa_pipeline
python -m Data_ETA_Pipeline.load_fred_exogenous
python -m Data_ETA_Pipeline.create_master_snapshots
```

This was *not* run as part of the code fix because deleting on-disk artifacts is a
hard-to-reverse action. Confirm before executing.

---

# EXTENDED AUDIT — `Train/` Pipeline (added 2026-05-11)

The original audit covered the ETL stage (Issues 1-7). The following extended audit (Issues
8-12) covers the `Train/` pipeline: training, backtest, prediction, post-train consensus
anchor, and predicted seasonal adjustment. Files audited:

- [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py) (3,332 lines) — main entry point,
  expanding-window backtest, dynamic re-selection, variance-enhancement stack, production
  model fit, `predict_nfp_mom`.
- [Train/data_loader.py](Train/data_loader.py) — snapshot loaders, target loaders,
  `pivot_snapshot_to_wide`, `batch_lagged_target_features`.
- [Train/feature_engineering.py](Train/feature_engineering.py) — calendar features.
- [Train/model.py](Train/model.py) — LightGBM training, sample weights, prediction intervals.
- [Train/baselines.py](Train/baselines.py) — `last_y` and `rolling_mean` baselines.
- [Train/revision_features.py](Train/revision_features.py) — cross-snapshot revision diffs.
- [Train/hyperparameter_tuning.py](Train/hyperparameter_tuning.py) — Optuna inner-CV tuning.
- [Train/short_pass_selection.py](Train/short_pass_selection.py) — per-step LGBM gain ranker.
- [Train/branch_target_selection.py](Train/branch_target_selection.py) — branch-target FS.
- [Train/candidate_pool.py](Train/candidate_pool.py) — union-pool builder.
- [Train/nsa_acceleration.py](Train/nsa_acceleration.py) — NSA features injected into the SA
  branch.
- [Train/Output_code/generate_output.py](Train/Output_code/generate_output.py) — final output
  artifacts and the predicted-adjustment glue.
- [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py)
  (1,271 lines) — Kalman fusion, AccelOverride, hybrid post-filter, predictions.csv
  augmentation.
- [Train/rerun_post_train_adj_and_consensus.py](Train/rerun_post_train_adj_and_consensus.py).
- [Train/sandbox/experiment_predicted_adjustment.py](Train/sandbox/experiment_predicted_adjustment.py)
  — `ExpWeightedMedianCovidExcludedPredictor` and `load_adjustment_history`.

The audit traces every release-date filter, snapshot cutoff, sample-weight anchor, and
hyperparameter search across the train/predict/post-train pipeline.

---

## ISSUE 8 — ~~HIGH~~ FIXED: Analyst consensus loaded with no PIT filter and `.last()` aggregator

**Status:** FIXED 2026-05-11. Replaced `_load_consensus(snapshot_path)` and the upstream
`_latest_snapshot_path()` helper with a new `_load_consensus_pit(target_type, target_source)`
that walks the master-snapshot tree and reads `NFP_Consensus_Mean` from the master
snapshot at each target month. Master snapshots are filtered at ETL time with strict
`release_date < snap_date(M's NFP release)`, so consensus is now PIT-correct *by
construction* — no per-row release-date filter needed at the consumer. See
[consensus_anchor_runner.py:141-204](Train/Output_code/consensus_anchor_runner.py#L141-L204).

**Cross-check (2026-05-11):** new loader produces 317 rows (1999-04 → 2026-04) vs old
loader's 317 rows; 0/317 value differences. Consensus is not historically revised in the
current dataset, so the value impact today is zero — but the PIT contract is now enforced
in code rather than relying on upstream Unifier behavior.

**Required regeneration:** re-run `python -m Train.rerun_post_train_adj_and_consensus`
to regenerate `_output/consensus_anchor/` with the new (PIT-correct-by-construction)
consensus values.

The original analysis follows for historical reference.

---

**File (pre-fix):** [Train/Output_code/consensus_anchor_runner.py:141-157](Train/Output_code/consensus_anchor_runner.py#L141-L157)

### The code

```python
def _load_consensus(snapshot_path: Path) -> pd.DataFrame:
    snap = pd.read_parquet(snapshot_path, columns=["date", "series_name", "value"])
    cons = snap[snap["series_name"] == "NFP_Consensus_Mean"].copy()
    if cons.empty:
        raise RuntimeError(f"NFP_Consensus_Mean not found in snapshot: {snapshot_path}")

    cons["ds"] = pd.to_datetime(cons["date"]).dt.to_period("M").dt.to_timestamp()
    cons["value"] = pd.to_numeric(cons["value"], errors="coerce")
    cons = cons.dropna(subset=["ds", "value"]).sort_values("ds")

    monthly = (
        cons.groupby("ds", as_index=False)
        .agg(consensus_pred=("value", "last"))
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly
```

The on-disk Unifier parquet schema (verified) includes a `release_date` column for every
series. `_load_consensus` deliberately omits it from `pd.read_parquet(..., columns=[...])`,
then groups by target month `ds` and takes `.last()` of `value` per group. The function is
called from `_latest_snapshot_path()` (lines 125-138), which **always returns the most
recent Unifier snapshot** — not a PIT-vintage view.

### Why it is a leak (architecturally)

The consensus row's *value* is whatever Unifier currently has stored for that target month.
Two structural risks:

1. **Post-release consensus updates leak silently.** If the Bloomberg/Reuters survey ever
   re-publishes a "true median" after the NFP release (e.g. corrected respondent count, late
   submissions), and Unifier carries that update under `series_name == "NFP_Consensus_Mean"`,
   `.last()` will pick it up. The historical "consensus" stored in `merged_consensus_model.csv`
   then would not be the survey median that was *publicly available before the NFP release*
   for that month — it would be a post-release value. Every Kalman / AccelOverride /
   AccelPostFilter prediction built off that consensus inherits the same future-knowledge
   error.

2. **`.last()` is unordered for ties.** Without sorting by `release_date` first, the "last"
   row per ds is order-of-iteration dependent. If multiple `release_date` values exist for
   the same ds (e.g. a re-survey published later), `.last()` is non-deterministic in a
   strict reading; pandas' `groupby(...).agg("last")` actually preserves the row order from
   the prior `sort_values("ds")`, so it picks the last row by *parquet row order* within
   that ds — typically the latest insertion, not necessarily the earliest publication.

3. **No release-date filter at the per-snapshot consumer level.** Every other source in the
   pipeline (ADP, FRED, Unifier prosper, NOAA) applies `release_date < snap_date` either at
   ETL time or at consumption time. This loader applies neither — it trusts the upstream
   data without enforcement.

### Impact magnitude

In the *current* state of the live Unifier snapshot, `NFP_Consensus_Mean` appears to carry a
single pre-release survey median per target month (Bloomberg/Reuters consensus is finalized
before NFP release ~6 business days into the next month and is not historically revised), so
the realised value-side error today is probably zero. The PIT contract is nonetheless broken
at the loader: a future change in Unifier source behaviour (revised consensus, late
respondents, schema update) will silently corrupt every consensus-anchor backtest with no
audit trail.

This matters more in `consensus_anchor_runner.py` than in the rest of the pipeline because:

- Every post-train consensus-anchor variant (Kalman fusion, AccelOverride, hybrid) uses the
  consensus value as a primary observation channel (lines 286-419 for Kalman, 486-572 for
  AccelOverride). The Kalman gain at line 389-395 directly multiplies `info_c * row["consensus_pred"]`
  into the posterior estimate. A future-corrupted `consensus_pred` propagates 1:1 into the
  posterior.

- The `comparison_metrics.csv` ranking that drives the `_augment_predictions_csv` sort
  (line 951: `augmented = augmented.sort_values(["rmse", "model"], ...)`) treats the
  baseline `Consensus` row as a valid OOS comparator. If the consensus has any post-release
  knowledge, the consensus row's MAE is artificially low, and downstream forecasts that
  anchor to it inherit the same artificial accuracy.

### Suggested fix

```python
def _load_consensus(snapshot_path: Path,
                    nfp_release_dates: pd.Series  # ds -> NFP release date
                   ) -> pd.DataFrame:
    snap = pd.read_parquet(
        snapshot_path,
        columns=["date", "series_name", "value", "release_date"],   # ADD release_date
    )
    cons = snap[snap["series_name"] == "NFP_Consensus_Mean"].copy()
    cons["ds"] = pd.to_datetime(cons["date"]).dt.to_period("M").dt.to_timestamp()
    cons["release_date"] = pd.to_datetime(cons["release_date"], errors="coerce")
    cons["value"] = pd.to_numeric(cons["value"], errors="coerce")
    cons = cons.dropna(subset=["ds", "value", "release_date"])

    # Per-ds: keep only consensus values whose release_date < the NFP release date
    # for that ds, then take the latest such value.
    cons = cons.merge(
        nfp_release_dates.rename("nfp_release"),
        left_on="ds", right_index=True, how="left",
    )
    cons = cons[cons["release_date"] < cons["nfp_release"]]
    monthly = (
        cons.sort_values(["ds", "release_date"])
            .groupby("ds", as_index=False)
            .agg(consensus_pred=("value", "last"))
    )
    return monthly
```

`nfp_release_dates` is already available in `data_loader.load_target_data` output (the
`release_date` column on the first-release target). Plumbing it through `build_merged_dataset`
adds two lines and closes the contract.

### Severity rationale

**HIGH** because the safety guarantee is delegated entirely to a third-party data source
with no enforcement. The current realised impact may be zero, but the risk surface is wide
(any Unifier behavioural change), the propagation path is fully unblocked (consensus value →
Kalman posterior → ranked predictions.csv), and the consensus is the single most influential
input to every post-train anchor variant.

---

## ISSUE 9 — ~~HIGH~~ FIXED: Optuna tunes Kalman / AccelOverride hyperparameters on the same OOS span the metrics report on

**Status:** FIXED 2026-05-11. Both `_tune_kalman` and `_tune_accel_override` now wrap
their per-trial scoring with a new `_walkforward_cv_score` helper that runs the
underlying `kalman_fusion` / `accel_override` on K=5 chronological folds; each fold's
composite score is computed only on its own eval window (rows
`overlap_df.iloc[eval_start:eval_end]`). Each eval window is strictly future relative
to all earlier folds and never overlaps with future folds. The trial's score is the
mean across folds, so winning hyperparameters are chosen by performance on data the
trial has not "trained on" via earlier-fold actuals.

The final `kalman_fusion` / `accel_override` runs (and their `comparison_metrics.csv`
metrics) still cover the full `overlap_with_oos`, but those metrics are now honest
because the params used were selected via nested OOS CV. See
[consensus_anchor_runner.py:468-548](Train/Output_code/consensus_anchor_runner.py#L468-L548)
(`_kalman_fold_runner`, `_accel_fold_runner`, `_walkforward_cv_score`,
`_composite_kalman_accel_objective`).

**Cross-check (2026-05-11):** synthetic 250-row test with n_splits=5 produces fold eval
windows `[60:98], [98:136], [136:174], [174:212], [212:250]` — non-overlapping and
strictly chronological.

**Required regeneration:** re-run `python -m Train.rerun_post_train_adj_and_consensus`
to regenerate `_output/consensus_anchor/comparison_metrics.csv` with honest OOS
hyperparameters. Tuned params will likely shift; reported metrics may move but are no
longer in-sample biased.

The original analysis follows for historical reference.

---



**File:** [Train/Output_code/consensus_anchor_runner.py:1066-1126](Train/Output_code/consensus_anchor_runner.py#L1066-L1126), with tuner internals at [422-479](Train/Output_code/consensus_anchor_runner.py#L422-L479) (Kalman) and [575-618](Train/Output_code/consensus_anchor_runner.py#L575-L618) (AccelOverride).

### The code

Tuner driver:

```python
# Line 1068-1071
if tune:
    # Tune on historical-only overlap (no OOS rows)
    kalman_params = _tune_kalman(overlap_df, consensus_df,
                                 n_trials=n_trials, timeout=timeout)
...
# Line 1077-1082 — the SAME spans get the report
kalman_df, kalman_metrics = kalman_fusion(
    overlap_with_oos, consensus_df,
    trailing_window=kalman_params["trailing_window"],
    use_nsa_accel=has_nsa,
    nsa_weight_scale=kalman_params.get("nsa_weight_scale", 1.0),
)
```

`_tune_kalman` body (excerpted):

```python
# Line 446-463
def objective(trial: "optuna.Trial") -> float:
    tw = trial.suggest_int("trailing_window", 6, 36)
    nsa_ws = trial.suggest_float("nsa_weight_scale", 0.1, 3.0) if has_nsa else 1.0
    _, metrics = kalman_fusion(
        overlap_df, consensus_df,
        trailing_window=tw,
        use_nsa_accel=has_nsa,
        nsa_weight_scale=nsa_ws,
    )
    mae = metrics.get("MAE", float("inf"))
    accel_acc = metrics.get("Acceleration_Accuracy", 0.0)
    dir_acc = metrics.get("Directional_Accuracy", 0.0)
    ...
    return float(mae - KALMAN_LAMBDA_ACCEL * accel_acc - KALMAN_LAMBDA_DIR * dir_acc)
```

`overlap_df` is defined at [Line 269-279](Train/Output_code/consensus_anchor_runner.py#L269-L279):

```python
overlap = df[
    df["consensus_pred"].notna()
    & df["actual"].notna()
    & df["champion_pred"].notna()
].copy()
overlap_with_oos = df[
    df["consensus_pred"].notna()
    & df["champion_pred"].notna()
].copy()
```

`overlap_df` is "every historical month with consensus + champion + actual" — i.e. the
*entire* span over which the comparison metrics are reported. `overlap_with_oos` adds the
forward OOS rows, but the OOS rows have `actual==NaN` and so contribute nothing to MAE /
DirAcc / AccelAcc anyway.

### Why this is a leak

This is meta-leakage / OOS-honesty failure, not feature leakage:

1. The Optuna search picks `trailing_window` (and `nsa_weight_scale` if NSA is present) by
   minimizing the composite objective over `overlap_df`.
2. The same `kalman_fusion(overlap_with_oos, ...)` is then run with the winning params, and
   the resulting `kalman_metrics` are written into both `comparison_metrics.csv` and
   (indirectly) into the rmse-driven sort in `_augment_predictions_csv`.
3. The metric is "walk-forward" in the sense that each Kalman step uses only `df.iloc[:i]`
   for noise estimation, but the *hyperparameter that drives the step* is chosen with full
   knowledge of every step's outcome. So the published metric is in-sample with respect to
   `trailing_window` selection.
4. Identical structure for `_tune_accel_override` (lines 575-618), with `kappa` and
   `magnitude_mode` chosen the same way.

The downstream `_augment_predictions_csv` sort:

```python
# Line 951
augmented = augmented.sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)
```

uses each model's `summary_statistics.csv` RMSE — which for the consensus-anchor variants is
the in-sample-tuned RMSE. So the "best forecast" surfaced first in `predictions.csv` may be
ranked above models that would actually outperform it in genuinely OOS use.

### Magnitude

Hard to bound without empirical re-runs, but the search space is non-trivial:
`trailing_window ∈ [6, 36]` (31 integer values), `nsa_weight_scale ∈ [0.1, 3.0]` (continuous),
`kappa ∈ [0.1, 0.9]` (continuous), `magnitude_mode ∈ {consensus, blend, model}`. With 25
TPE trials, even a modest fraction of search is enough to find a configuration that fits the
backtest's idiosyncratic noise. Empirical inflation of 5-15% on AccelAcc is plausible.

The `KALMAN_LAMBDA_ACCEL = 50.0` / `KALMAN_LAMBDA_DIR = 30.0` weights in
[Train/config.py:452-453](Train/config.py#L452-L453) make the search aggressive — the
optimizer is willing to trade ~50 MAE points for one fractional unit of acceleration
accuracy. A search that aggressive over the same span the metrics live on is a textbook
overfitting setup.

### Suggested fix

Two options, in order of preference:

1. **Nested expanding-window inside the tuner.** Inside `_tune_kalman`, split `overlap_df`
   into K chronological folds. For each fold's tail months, run `kalman_fusion` with the
   trial's params using only earlier folds; aggregate fold scores. This keeps every metric
   strictly OOS w.r.t. its own params.

2. **Holdout slice.** Tune on `overlap_df.iloc[:-K]` (the "in-sample tuning span") and
   report metrics on `overlap_df.iloc[-K:]` (true holdout). Simpler but wastes K months of
   reportable history.

The current "use full history for Kalman noise estimation but only trailing window for
each step" pattern is already PIT-correct at the per-step level — it's only the
hyperparameter selection layer that breaks the contract.

### Severity rationale

**HIGH** because the metrics are explicitly used to *rank forecasts* in the final
`predictions.csv` deliverable. A non-PIT tuning loop biases the very number a downstream
consumer would use to decide "which forecast to trust." This is exactly the failure mode
walk-forward backtests are supposed to prevent.

---

## ISSUE 10 — ~~MEDIUM~~ FIXED: NSA-acceleration features for the SA branch use revised target values stamped only by observation date

**Status:** FIXED 2026-05-11. `_load_nsa_revised_target` and `_load_sa_revised_target`
now load `operational_available_date` alongside `[ds, y_mom, acceleration]`.
`compute_nsa_acceleration_features` accepts a new `cutoff_date` parameter; when
provided, revised actuals are filtered by `operational_available_date < cutoff_date`
instead of `ds < target_month`. `build_nsa_features_for_training` accepts a
`cutoff_dates: Dict[ds, cutoff]` map and forwards each month's cutoff to the
per-month call. The SA backtest loop in `run_expanding_window_backtest` now builds
`target_release_date_map` once (outside the loop, from `target_df['release_date']`)
and passes the SA target's first-release date as the cutoff for both the prediction
and the training-month build. See
[Train/nsa_acceleration.py:31-298](Train/nsa_acceleration.py#L31-L298) and
[Train/train_lightgbm_nfp.py:1342-1357, 1727-1764](Train/train_lightgbm_nfp.py#L1727-L1764).

**Cross-check (2026-05-11):** for SA target_month = 2024-02-01 (cutoff = 2024-03-08),
the OLD `ds < target_month` filter included the revised value of NSA Jan 2024
(op_avail = 2024-03-08 — same day as the cutoff, leak). The NEW filter
(`op_avail < cutoff`) correctly excludes that row; the most recent admitted ds is
2023-12-01 (op_avail = 2024-02-02, well before cutoff). 1 leak row excluded per backtest
step. The downstream `nsa_actual_accel` feature value changes accordingly.

**Required regeneration:** re-run the SA branch only, then post-train consensus_anchor:
```
python Train/train_lightgbm_nfp.py --train --target sa --release first
python -m Train.rerun_post_train_adj_and_consensus
```
Issue 10 itself does not affect the NSA branch; Issue 14 is separate and does require
rerunning NSA artifacts.

The original analysis follows for historical reference.

---



**File:** [Train/nsa_acceleration.py:115-145](Train/nsa_acceleration.py#L115-L145)

### The code

```python
# Lines 115-118
df = nsa_backtest_df.copy()
df["ds"] = pd.to_datetime(df["ds"])
hist = df[df["ds"] < target_month].sort_values("ds").reset_index(drop=True)

# Lines 130-135 — REVISED actuals filtered by observation date only
nsa_hist = nsa_target[nsa_target["ds"] < target_month].sort_values("ds")
nsa_hist = nsa_hist[nsa_hist["y_mom"].notna()]

sa_hist = sa_target[sa_target["ds"] < target_month].sort_values("ds")
sa_hist = sa_hist[sa_hist["y_mom"].notna()]

# Lines 140-150 — these revised values flow directly into features
last_nsa_actual = float(nsa_hist.iloc[-1]["y_mom"])
nsa_pred_delta = nsa_pred_now - last_nsa_actual
features["nsa_pred_delta"] = nsa_pred_delta
if len(nsa_hist) >= 2:
    prev_nsa_actual = float(nsa_hist.iloc[-2]["y_mom"])
    actual_delta_prev = last_nsa_actual - prev_nsa_actual
    features["nsa_pred_accel"] = nsa_pred_delta - actual_delta_prev
```

`nsa_target` and `sa_target` are loaded from `data/NFP_target/y_nsa_revised.parquet` and
`y_sa_revised.parquet` — these store **once-revised** values (the level reported at the
M+1 NFP release). For obs month M-1, the revised value is published on M's first NFP
release date.

The function is called from [train_lightgbm_nfp.py:1727-1750](Train/train_lightgbm_nfp.py#L1727-L1750):

```python
if target_type == 'sa' and nsa_backtest_results is not None:
    from Train.nsa_acceleration import compute_nsa_acceleration_features, build_nsa_features_for_training
    nsa_feats_pred = compute_nsa_acceleration_features(
        nsa_backtest_results, target_month
    )
    ...
    nsa_train_feats = build_nsa_features_for_training(
        nsa_backtest_results, training_months
    )
    for col in _nsa_accel_cols:
        if col in nsa_train_feats.columns:
            X_train_valid[col] = nsa_train_feats[col].reindex(training_months).values
        ...
```

### Why this is a leak

The cutoff for SA training features is set in `build_training_dataset` to **M's first NFP
release date** (strict `<`):

```python
# train_lightgbm_nfp.py:1219-1231
release_date_map = dict(zip(
    target_ref.loc[valid_mask, 'ds'],
    target_ref.loc[valid_mask, 'release_date'],
))
...
tasks.append((
    tm, target_values[i],
    release_date_map.get(tm, tm),         # cutoff_date = release_date of tm (M)
    ...
))
```

For SA target_month = M (e.g., 2024-02-01):

- `cutoff_date` = release_date of M's first NFP release (e.g., 2024-03-08)
- Strict `<` cutoff means features with `release_date == 2024-03-08` are EXCLUDED
- The revised value of NSA[M-1] (released on M's first NFP release date, 2024-03-08)
  should therefore be excluded
- But `compute_nsa_acceleration_features` filters `nsa_target` by `ds < target_month`
  (i.e., `ds < 2024-02-01`), which **includes** ds=2024-01-01 with its revised value
- The revised y_mom for ds=2024-01-01 is the value released on 2024-03-08, which violates
  the strict `< 2024-03-08` cutoff
- This is a **same-day data leak** — borderline but technically a violation of the
  pipeline's own contract

The same issue applies to `sa_hist` (revised SA actuals for ds=M-1 also released on M's
first release date) and propagates into `nsa_actual_accel`, `nsa_sa_accel_corr_12m`, and
`nsa_sa_gap_delta`.

The NSA backtest predictions in `hist = df[df["ds"] < target_month]` are *not* affected
because those predictions are themselves OOS values produced by the NSA model and were
known at the target_month-1's prediction time, well before target_month.

### Two interpretations of the operational target

The leak severity depends on what "the SA-revised model is operationally predicting":

- **Interpretation A — predict at M's first release time:** The cutoff in
  `build_training_dataset` (M's first release) suggests the model is asked to make a
  prediction of the revised SA value at the moment the first-release SA value is being
  published. Under this interpretation, the revised NSA[M-1] (released on the same day) is
  a leak — features for SA M cannot include data released at exactly M's release time.

- **Interpretation B — predict at M+1's release time (when revised target is known):**
  The revised target is only operationally available after the M+1 NFP release. If the
  prediction is meant to be made at M+1's release, then revised NSA[M-1] (released a month
  earlier at M's release) is squarely fair game.

The operational use of these models is unclear from the code; under Interpretation B this
is a non-leak. Under Interpretation A — which the cutoff actually enforces — it is a
same-day leak on one feature axis (`y_mom` of the revised target for ds=M-1).

### Magnitude

- Affects 1 month per backtest step (M-1's revised value), used in 4 features
  (`nsa_pred_delta`, `nsa_pred_accel`, `nsa_actual_accel`, `nsa_sa_accel_corr_12m`,
  `nsa_sa_gap_delta`).
- Same-day data: the leak is the *revision* in the M+1 release vs the (unknown-to-the-model)
  first-release value of M-1. NSA first-release vs once-revised typical magnitude on
  monthly MoM is small (median |Δ| ~ 5-15K, p95 ~ 30-50K relative to MoM scale of ~150K).
- Tree splits on `nsa_pred_accel` could pick up the small revision signal as if it were
  predictive content.

### Suggested fix

Change the filter in `compute_nsa_acceleration_features` from observation-date to
release-date filtering, mirroring the rest of the pipeline:

```python
# Replace the current ds-based filter with operational_available_date filter
nsa_hist = nsa_target[
    nsa_target["operational_available_date"].notna()
    & (nsa_target["operational_available_date"] < cutoff_date)  # NEW arg
    & nsa_target["y_mom"].notna()
].sort_values("ds")
```

This requires plumbing `cutoff_date` into the function (currently it only takes
`target_month`). The caller (`train_lightgbm_nfp.py:1732-1750`) already has access to the
NSA training cutoff via `release_date_map`, so the change is mechanical.

`y_nsa_revised.parquet` already carries the `operational_available_date` column (added at
[Train/data_loader.py:603-616](Train/data_loader.py#L603-L616)) precisely to support this
filter.

### Severity rationale

**MEDIUM** because:
- The leak is structural (always same-day, always one month) but bounded in magnitude;
- The interpretation argument (B vs A) gives the function a defensible reading; and
- Only the SA branch uses these features (`if target_type == 'sa' and nsa_backtest_results
  is not None:` gate at train_lightgbm_nfp.py:1727), so the NSA branch and all consensus-
  anchor stages are unaffected.

It is *not* LOW because the leak is in features the model can split on, not in a one-time
init parameter.

---

## ISSUE 11 — ~~LOW~~ FIXED: Kalman noise initialization uses full-history future data

**Status:** FIXED 2026-05-11. `R_c_init` and `Q_init` in `kalman_fusion` are now
computed from `consensus_df[ds < first_backtest_ds]` (where `first_backtest_ds` is the
first month being evaluated by the kalman_fusion call) rather than from the last 60
entries of the full consensus history. The init noise prior cannot peek into months
that will later be evaluated. Falls back to `1.0` (sane defaults) when no prior
history exists. See
[consensus_anchor_runner.py:322-336](Train/Output_code/consensus_anchor_runner.py#L322-L336).

**Required regeneration:** re-run `python -m Train.rerun_post_train_adj_and_consensus`
to regenerate `_output/consensus_anchor/` with PIT-correct init noise. Impact is
bounded to the first ~6 backtest months only.

The original analysis follows for historical reference.

---



**File:** [Train/Output_code/consensus_anchor_runner.py:322-328, 381-387](Train/Output_code/consensus_anchor_runner.py#L322-L328)

### The code

```python
# Lines 322-328
cons_hist = consensus_df[
    consensus_df["consensus_pred"].notna() & consensus_df["actual"].notna()
]
cons_err = (cons_hist["actual"] - cons_hist["consensus_pred"]).values
R_c_init = float(np.var(cons_err[-60:], ddof=1))
Q_init = float(np.var(np.diff(cons_hist["actual"].dropna().values[-60:]), ddof=1))

# Lines 350-353 — these inits are used as fallbacks for the first ~6 backtest months
else:
    R_c = R_c_init
    R_m = R_c_init * 1.5 if use_model else 1e12
    Q = Q_init

# Lines 381-387 — and again for NSA noise scaling
else:
    R_a = R_c_init * 2.0
    info_a = nsa_weight_scale / R_a
```

`consensus_df` is the FULL consensus history including months from **after** the early
backtest months. For backtest step `i` near the start (e.g., the 13th step), if
`len(hist_valid) < 6` (the trailing-window guard at line 340), the Kalman filter falls
back to `R_c_init` / `Q_init` — both computed from the last 60 entries of the full history,
which by definition includes future months relative to step `i`.

Once enough history accumulates (~6 backtest months in), the per-step `R_c, R_m, Q` are
correctly computed from `hist.iloc[:i]` only (lines 340-349), so the leak is bounded in time.

### Why this is a leak

For the first ~6 backtest months, the Kalman covariance scaling uses noise variance
estimates derived from a 60-month window that overlaps with the backtest's own future
months. The Kalman gain at lines 389-395 is a function of `info_prior + info_c + info_m +
info_a` — if `R_c_init` is a future-knowledge estimate of consensus error variance, the
weight given to the consensus observation in those early months reflects future regime
information.

This is a form of "look-ahead in the prior" — not feature leakage, but the prior's strength
is informed by future observations.

### Magnitude

Affects only the first ~6 backtest months. With ~250 months in the typical backtest, this
is ~2.4% of the reported MAE. The variance of `R_c_init` itself (last-60 vs strictly-prior
windows) is modest because consensus error variance is fairly stationary outside of COVID.
Likely impact on tail metrics: < 1 MAE point, probably negligible.

### Suggested fix

```python
# Compute init from data strictly prior to the first backtest month
first_backtest_ds = df.iloc[0]["ds"] if len(df) > 0 else pd.Timestamp.max
prior_cons_hist = consensus_df[
    consensus_df["consensus_pred"].notna()
    & consensus_df["actual"].notna()
    & (consensus_df["ds"] < first_backtest_ds)
]
prior_cons_err = (prior_cons_hist["actual"] - prior_cons_hist["consensus_pred"]).values
R_c_init = float(np.var(prior_cons_err[-60:], ddof=1)) if len(prior_cons_err) >= 2 else 1.0
Q_init = float(np.var(np.diff(prior_cons_hist["actual"].dropna().values[-60:]), ddof=1)) \
         if len(prior_cons_hist) >= 2 else 1.0
```

### Severity rationale

**LOW** because:
- The window of impact is tiny (~6 / ~250 backtest months ≈ 2.4%);
- Consensus error variance is approximately stationary outside COVID, so the future-vs-
  prior delta is small;
- Once `len(hist_valid) >= 6` (achieved by month 7 of the backtest), the per-step path
  takes over and the leak vanishes;
- It does not affect any `comparison_metrics.csv` row computed over the full backtest by
  more than ~1 MAE point.

---

## ISSUE 12 — CLOSED, **NOTE**: Sandbox predictor selection does not feed production output

**File:** [Train/sandbox/experiment_predicted_adjustment.py:537-642](Train/sandbox/experiment_predicted_adjustment.py#L537-L642),
verified non-leak via [Train/Output_code/generate_output.py:212-213, 388-389](Train/Output_code/generate_output.py#L212-L213).

### The concern (initial)

`run_walkforward_backtest` evaluates 8 candidate adjustment predictors on the full
backtest. `evaluate_models` (line 490-539) ranks them by `sa_composite` over the full
span, and `save_outputs` writes the "winner" to disk. This is a model-selection-on-OOS-
metrics pattern identical in shape to Issue 9.

### Why it is NOT a leak in the production pipeline

The production output stage in `generate_output.py` does **not** read the sandbox winner.
Instead, it hard-codes the predictor:

```python
# generate_output.py:204-213 (the production adjustment folder generator)
from Train.sandbox.experiment_predicted_adjustment import (
    ExpWeightedMedianCovidExcludedPredictor,
    load_adjustment_history,
)

folder.mkdir(parents=True, exist_ok=True)
adj_history = load_adjustment_history()
predictor = ExpWeightedMedianCovidExcludedPredictor(half_life_years=3.0)
```

And again at [generate_output.py:387-388](Train/Output_code/generate_output.py#L387-L388)
for the OOS prediction path. The sandbox's "best model" selection is informational only —
it produces `_output/sandbox/nsa_predicted_adjustment_revised/` artifacts but does not
mutate which predictor the production output uses.

The chosen `half_life_years=3.0` is a one-time human-set constant, not iterative meta-leak.
The PIT filter inside `run_walkforward_backtest` (lines 423-430:
`pit_mask = adj_history["operational_available_date"] < target_ds`) is correct and
strictly historical relative to each target month.

### Why it is also "conservative-not-leak"

The PIT filter `operational_available_date < target_ds` uses `target_ds` = first-of-target-
month (e.g., 2024-02-01 for predicting Feb 2024). The actual SA prediction is made at SA
Feb 2024's release date (~Mar 8 2024) or M+1's release for revised target (~Apr 5 2024).
So the filter is **far more restrictive than necessary** — a strictly historical filter that
discards months of legitimately-available data. Not a leak.

### Severity rationale

**NOTE / closed** — verified that the leaky-shaped `evaluate_models` selection does not
reach production. The hard-coded production predictor + correctly-conservative PIT filter
together close the contract.

---

# VERIFICATIONS — Train/ paths checked and confirmed PIT-correct

The following Train/ paths were audited in detail and are PIT-correct:

### V8 — Expanding window training mask is strict `<`
[train_lightgbm_nfp.py:1465](Train/train_lightgbm_nfp.py#L1465) — `train_mask = X_full['ds'] < target_month`. Strict less-than ensures no training row equals or exceeds the target month. The valid-target filter at line 1475 then drops NaN-y rows from the training set.

### V9 — Sample weights cap distance at zero (no future-pulling weight)
[model.py:107-110](Train/model.py#L107-L110):
```python
distance_days = (target_month - pd.to_datetime(X['ds'])).dt.days
distance_months = np.maximum(0, distance_days / 30.436875)
```
The `np.maximum(0, ...)` clamp protects against any row inadvertently in the future (which would otherwise produce a weight > 1.0). Combined with V8's strict-< train mask, this is doubly safe.

### V10 — Inner Optuna CV uses TimeSeriesSplit
[hyperparameter_tuning.py:135, 187](Train/hyperparameter_tuning.py#L135) — `inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)` and `for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X)):`. TimeSeriesSplit produces strictly chronological folds: each validation fold is purely future relative to its training fold. Per-fold sample weights at line 197 use `fold_target_month = X.iloc[val_idx]['ds'].max()` so the weight anchor is internal to the fold and doesn't peek beyond.

The hyperparameter selection IS PIT-correct at the per-fold level — the Optuna meta-leak (Issue 9) only applies to the post-train consensus-anchor tuners, NOT to the LightGBM hyperparameter tuner inside the main backtest loop.

### V11 — Dynamic re-selection uses only training data
[train_lightgbm_nfp.py:1561-1572](Train/train_lightgbm_nfp.py#L1561-L1572) — `_dynamic_reselection(X_train=X_train_valid, y_train=y_train_valid, ...)`. `X_train_valid` is built from the `train_mask = X_full['ds'] < target_month` filter at line 1465, so the reselection sees no future months. The recency weights at lines 517-523 use `step_date - dates` (positive distances only) and `RESELECTION_HALF_LIFE_MONTHS=9999` makes them effectively uniform.

### V12 — `clean_features` post-2010 NaN-rate evaluation respects the training window
[train_lightgbm_nfp.py:283-291](Train/train_lightgbm_nfp.py#L283-L291) — `modern_mask = X['ds'] >= eval_ts` and `nan_rates = X_modern.isna().sum() / n_modern`. The function is called inside the backtest loop with `X_train_valid` (data strictly < target_month), so the post-2010 evaluation window is `[2010-01-01, target_month)` — strictly historical relative to the prediction.

### V13 — Snapshot lags are PIT-safe; branch-target revised lags also require availability masking
[data_loader.py:683-771](Train/data_loader.py#L683-L771) — `_build_lagged_target_feature_frame` uses `mom.shift(1)`, `mom.rolling(N, min_periods=N).mean().shift(1)`, etc. The lag math is backward-looking in observation-month order; the `.shift(1)` after `.rolling()` ensures the rolling stat at month t uses [t-N, t-1] (no t itself). Same-month seasonal features at lines 760-764 use `month_groups.shift(1)` over groupby-month, also strictly prior-year.

This is sufficient for features already inside the master snapshot, because the snapshot itself has already been filtered with strict `release_date < cutoff`. It is not sufficient for the branch-target `nfp_nsa_*` features injected from `data/NFP_target/y_nsa_revised.parquet`: those values are revised target actuals, not snapshot rows, and the prior month's revised value is often released on the same NFP release date as the current target.

**Status:** fixed 2026-05-17. `get_lagged_target_features(..., cutoff_date=...)` now masks revised target `y` and `y_mom` unless `operational_available_date < cutoff_date` before calling `_build_lagged_target_feature_frame`; `batch_lagged_target_features(..., cutoff_dates=...)` uses the same per-month cutoff map during training; and `predict_nfp_mom` passes the live cutoff date for inference. Audit artifacts live under `_output_pairing_baseline_pitfix/pit_audit_current_pipeline/`.

### V14 — Production model uses `target_month=X.max()` as weight anchor
[train_lightgbm_nfp.py:2324](Train/train_lightgbm_nfp.py#L2324) — `final_target_month = pd.to_datetime(X_full_valid['ds'].max())`. Production model trains on all historical valid data with weights anchored to the latest available month. Distances are non-negative (V9), and the model is then used to predict future months whose `ds > final_target_month` — no leakage, just decayed weighting toward recent regimes.

### V15 — `predict_nfp_mom` uses release_date as cutoff, not target_month
[train_lightgbm_nfp.py:2741-2750](Train/train_lightgbm_nfp.py#L2741-L2750) — `cutoff_date = match['release_date'].iloc[0]` (when available), passed to `pivot_snapshot_to_wide(snapshot_df, target_month, cutoff_date=cutoff_date)`. The pivot at `data_loader.py:958` enforces `wide_df = wide_df[wide_df.index < cutoff]` (strict <).

### V16 — Revised-model timing guard at inference
[train_lightgbm_nfp.py:2700-2710](Train/train_lightgbm_nfp.py#L2700-L2710) — `predict_nfp_mom` for `target_source='revised'` raises `RuntimeError` if called before the revised target's `operational_available_date` has passed. Prevents calling the model when the operational target is not yet observable.

### V17 — Revision features compute strictly historical diffs
[revision_features.py:144-167](Train/revision_features.py#L144-L167) and [train_lightgbm_nfp.py:138-149](Train/train_lightgbm_nfp.py#L138-L149) — `view_curr` and `view_prev` are both pivoted with the same `prev_month` cutoff. Both snapshots were already filtered at ETL time to `release_date < snap_date`. The diff is "what changed between M-1 and M for the M-1 observation", which is purely a backward-looking revision signal.

### V18 — NOAA staleness signal is captured but ffill is bounded to 6 months
[data_loader.py:990-1018](Train/data_loader.py#L990-L1018) — NOAA columns get `.ffill(limit=NOAA_MAX_FFILL_MONTHS=6)` to handle publication lag, with explicit `__staleness_months` features so the model can discount stale NOAA values. Non-NOAA columns get unlimited ffill, which is PIT-safe (just carries the most recent value forward; no future data injected). The `last_obs.idxmax()` calculation at line 997 is the last-true-observation timestamp, not a forward-look.

### V19 — Predicted-adjustment uses `operational_available_date < target_ds` filter (conservative)
[generate_output.py:233-238](Train/Output_code/generate_output.py#L233-L238), [experiment_predicted_adjustment.py:423-430](Train/sandbox/experiment_predicted_adjustment.py#L423-L430) — strictly historical filter. `target_ds` is the first-of-target-month, well before the actual SA prediction time, so the filter is over-restrictive (loses some legitimately-available data). Conservative, not a leak.

### V20 — `kalman_fusion` per-step noise estimates use `df.iloc[:i]` only
[consensus_anchor_runner.py:336-353](Train/Output_code/consensus_anchor_runner.py#L336-L353) — once `len(hist_valid) >= 6`, per-step `R_c, R_m, Q` are computed from `hist = df.iloc[:i]` filtered to `actual.notna()`. Strictly historical. The leak in Issue 11 only applies to the *initialization* (first ~6 months); after that, the loop is clean.

### V21 — `accel_override` uses `df.iloc[:i]` historical actuals only
[consensus_anchor_runner.py:518-519](Train/Output_code/consensus_anchor_runner.py#L518-L519) — `hist_valid = df.iloc[:i]` filtered to `actual.notna()`. Each step's direction vote uses only past actuals. PIT-correct.

### V22 — Hybrid (Kalman + AccelOverride post-filter) loop is PIT-safe
[consensus_anchor_runner.py:1148-1180](Train/Output_code/consensus_anchor_runner.py#L1148-L1180) — `hist_valid = overlap_with_oos[(ds<curr) & actual.notna()]` correctly truncates. The Kalman predictions already came from a PIT-ordered loop (V20). No additional leak introduced by the post-filter.

---

# RECOMMENDED REMEDIATION ORDER (Train/ pipeline, Issues 8-14)

1. ~~**Issue 8 (consensus PIT filter).**~~ **DONE 2026-05-11** — replaced `_load_consensus`
   with `_load_consensus_pit` that reads `NFP_Consensus_Mean` directly from the
   master-snapshot tree (which already enforces strict `release_date < snap_date` at ETL
   time). PIT-correct by construction; no per-row release-date filter needed. Verified
   identical 317 rows / 0 value diff vs the old loader on current data.

2. ~~**Issue 9 (Optuna meta-leak in consensus tuners).**~~ **DONE 2026-05-11** — added
   `_walkforward_cv_score` helper with K=5 chronological folds; both `_tune_kalman` and
   `_tune_accel_override` now average each trial's composite score over folds where each
   eval window is strictly future relative to all earlier folds. Hyperparameter selection
   is now OOS-honest. Final `comparison_metrics.csv` metrics are still over the full
   `overlap_with_oos`, but the tuned params used were chosen via nested OOS CV.

3. ~~**Issue 10 (NSA acceleration features same-day leak).**~~ **DONE 2026-05-11** —
   plumbed `cutoff_date` (and `cutoff_dates` map for the training-month wrapper) into
   `compute_nsa_acceleration_features` / `build_nsa_features_for_training`. SA backtest
   loop builds `target_release_date_map` once and passes the SA target's first-release
   date as the cutoff. Filter switched from `ds < target_month` to
   `operational_available_date < cutoff_date`. Verified 1 leak row excluded per backtest
   step.

4. ~~**Issue 11 (Kalman noise init).**~~ **DONE 2026-05-11** — restricted `R_c_init` /
   `Q_init` source to `consensus_df[ds < first_backtest_ds]`. Init noise prior cannot
   peek into months that will later be evaluated.

5. **Issue 12.** Already closed (verified non-leak in production).

6. ~~**Issue 14 (branch-target revised target lags).**~~ **DONE 2026-05-17** — NSA
   branch-target lag/rolling features now mask revised target values by
   `operational_available_date < cutoff_date` before shift/rolling construction.

**Action required for the fixes to take effect on disk:**

- **Issues 8, 9, 11:** post-train consensus_anchor only — re-run
  `python -m Train.rerun_post_train_adj_and_consensus`. This regenerates
  `_output/consensus_anchor/{baseline_consensus,kalman_fusion,accel_override,kalman_accel_postfilter}/`
  plus `comparison_metrics.csv`. Fast (~minutes).
- **Issue 10:** the SA branch's training features change; re-run
  `python Train/train_lightgbm_nfp.py --train --target sa --release first` (or the full
  `--train-all`), then `python -m Train.rerun_post_train_adj_and_consensus`. Issue 10
  itself does not affect the NSA branch; Issue 14 below does.
- **Issue 14:** the NSA branch's target-derived lag features change; re-run
  `python Train/train_lightgbm_nfp.py --train-all`, then the consensus-anchor post-train
  stage. Existing local NSA and final-layer metrics from before 2026-05-17 are stale.

No on-disk parquet artifacts under `data/` need regeneration for Issues 8-14 —
these issues are all in the training/post-train consumption layer, not the ETL layer.

---

## ISSUE 13 — FIXED 2026-05-11: COVID winsorization consistency across the pipeline

**Status:** FIXED 2026-05-11. Six concrete fixes plus 11 unit tests in
[tests/test_covid_winsorization.py](tests/test_covid_winsorization.py).

### Diagnosis

A focused audit of COVID handling found three real asymmetries:

1. **`NFP_Consensus_Mean` was winsorized as a LightGBM training feature** (per-step
   winsorize at `train_lightgbm_nfp.py:1502-1508`) **but raw at prediction time and
   raw downstream of the model.** For SA target_month = 2020-04-01, the model was
   trained on Mar 2020 consensus clipped to ~+56 but `X_pred` carried raw consensus
   −14,448 (because `X_pred = X_full.iloc[[target_idx]]` and X_full was never
   winsorized). Same in the consensus_anchor stage: `_load_consensus_pit` returned
   raw consensus, which compared against winsorized SA actuals produced an
   artificial +13,911 error for Apr 2020 — inflating the Consensus baseline's
   MAE/RMSE in `comparison_metrics.csv` and biasing Kalman's adaptive
   `R_c = var(actual − consensus_pred)` estimator.

2. **All sandbox adjustment predictors except the production default
   (`ExpWeightedMedianCovidExcludedPredictor`) were exposed to artificial
   winsorized adjustment values** during COVID. Apr 2020 adjustment from
   `load_adjustment_history` is +2,433 (winsorized SA −537 minus winsorized NSA
   −2,971), but the real raw April 2020 adjustment was approximately +50.

3. **`compute_metrics` and `full_metrics` did not stratify by COVID** when writing
   `summary_statistics.csv` / `comparison_metrics.csv`. Console-only stratification
   in the train backtest was not persisted to disk.

### Fixes applied

| # | File | Change |
|---|---|---|
| 1 | [utils/transforms.py](utils/transforms.py) | Added `COVID_START_DEFAULT`, `COVID_END_DEFAULT`, `COVID_EXCLUDE_MONTHS`, `is_covid_month()` as the single source of truth. |
| 2 | [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py) | Replaced per-step winsorize on `X_train_valid` with one upfront winsorize on `X_full` and `y_full` immediately after `build_training_dataset`. Training, prediction (`X_pred`), and the production model fit (which reuses `X_full` via `train_and_evaluate`) now all see consistent winsorized values. |
| 3 | [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | Wrapped `_load_consensus_pit` return through `winsorize_covid_period`. Verified: Apr 2020 consensus −14,448 → −525.48 (clipped to non-COVID 1st pct). |
| 4 | [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | `kalman_fusion`'s per-step `R_c`/`R_m`/`Q` and the NSA-channel `R_a` now exclude COVID rows from `hist_valid` before variance estimation. Falls back to most-recent estimates when COVID-clean trailing window is too small. |
| 5 | [Train/Output_code/metrics.py](Train/Output_code/metrics.py) + [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | `compute_metrics` and `full_metrics` now emit `NonCovid_*` and `CovidOnly_*` prefixed metric blocks plus `N_NonCovid` / `N_Covid` counts inline alongside the existing schema. All 4 `full_metrics` call sites pass `ds=` for stratification. |
| 6 | [Train/sandbox/experiment_predicted_adjustment.py](Train/sandbox/experiment_predicted_adjustment.py) | `AdjustmentPredictor` base class now strips COVID rows in `fit_predict` (default `exclude_covid=True`); each concrete predictor renamed `fit_predict` → `_fit_predict_impl`. All 8 predictors honor the flag. Sandbox now imports `COVID_EXCLUDE_MONTHS` from `utils.transforms` rather than redefining it. |

### Unit tests

11 tests in [tests/test_covid_winsorization.py](tests/test_covid_winsorization.py),
all passing:

| Test | Verifies |
|---|---|
| `test_covid_constants_centralized` | `COVID_START_DEFAULT`/`COVID_END_DEFAULT`/`COVID_EXCLUDE_MONTHS`/`is_covid_month` exported from `utils.transforms`. |
| `test_sandbox_constant_matches_central` | Sandbox `COVID_EXCLUDE_MONTHS` equals the centralized constant. |
| `test_winsorize_clips_extreme_consensus` | Apr 2020 raw −14,448 clips to non-COVID 1st pct (~+56 in synthetic 50-250 range). |
| `test_kalman_noise_excludes_covid` | Post-COVID Kalman predictions agree within 10K between full and COVID-removed datasets — proves R_c/Q estimation is COVID-clean. |
| `test_compute_metrics_stratified` | Synthetic 13-month dataset emits `N=13, N_NonCovid=10, N_Covid=3`; `CovidOnly_MAE > 5× NonCovid_MAE`; aggregate MAE lies between strata. |
| `test_compute_metrics_handles_no_covid_data` | Pure non-COVID input produces `N_Covid=0`, `CovidOnly_MAE=NaN`, `NonCovid_MAE == MAE`. |
| `test_full_metrics_stratified_when_ds_provided` | Without `ds=`, legacy schema only; with `ds=`, `NonCovid_*` / `CovidOnly_*` present. |
| `test_sandbox_predictor_default_strips_covid` | All 5 concrete predictors with `exclude_covid=True` (default) do NOT echo a +2,434 Apr 2020 spike into a 2024-Apr prediction. |
| `test_sandbox_predictor_opt_out_works` | `exclude_covid=False` reproduces the legacy code path (API works). |
| `test_load_consensus_pit_winsorizes_covid_months` | Live master-snapshot read for Apr 2020 returns clipped magnitude (not raw −14,448); Aug 2020 unchanged. |
| `test_xfull_winsorize_symmetric` | Synthetic X_full with raw Apr 2020 consensus → upfront winsorize → X_pred at April is clipped (no longer −14,448). |

### Required regeneration

```
# Post-train consensus_anchor only (regenerates Baseline_Consensus + Kalman + AccelOverride
# + hybrid CSVs/plots with winsorized consensus and COVID-clean Kalman noise):
python -m Train.rerun_post_train_adj_and_consensus

# Full pipeline regen (for upfront X_full winsorize + stratified summary_statistics.csv):
python Train/train_lightgbm_nfp.py --train-all
python -m Train.rerun_post_train_adj_and_consensus
```

After regen, expect:
- `Baseline_Consensus` MAE in `comparison_metrics.csv` to drop noticeably (the Apr 2020
  +13,911 artificial error is removed).
- A new `NonCovid_*` block in every `summary_statistics.csv` and
  `comparison_metrics.csv` row.
- The model's predictions at COVID target months now use winsorized inputs (by design;
  the training distribution and prediction distribution now match).

---

## ISSUE 14 — FIXED 2026-05-17: Branch-target revised lags needed operational availability cutoffs

**Status:** FIXED 2026-05-17. The lag formulas themselves were backward-looking, and
master-snapshot-derived lags remain PIT-safe by snapshot construction. The leak was in
a different path: branch-target `nfp_nsa_*` features injected directly from
`data/NFP_target/y_nsa_revised.parquet` before the master snapshot is joined.

For a month M prediction made at the M NFP release, the M-1 revised value is often
released in that same report. A simple observation-month `.shift(1)` therefore treated
same-day M-1 revised values as known before the M release, which is not strict PIT.

### Fixes applied

| # | File | Change |
|---|---|---|
| 1 | [Train/data_loader.py](Train/data_loader.py) | Added `_mask_unavailable_revised_targets()` and `cutoff_date` support in `get_lagged_target_features()`. Revised target `y` and `y_mom` are blanked unless `operational_available_date < cutoff_date` before shift/rolling features are built. |
| 2 | [Train/data_loader.py](Train/data_loader.py) | Added `cutoff_dates` support to `batch_lagged_target_features()` so the training batch path uses the same per-month strict cutoff as prediction. |
| 3 | [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py) | `build_training_dataset()` now builds `release_date_map` before branch-target lag precomputation and passes it into `batch_lagged_target_features()`. |
| 4 | [Train/train_lightgbm_nfp.py](Train/train_lightgbm_nfp.py) | `predict_nfp_mom()` now passes the live prediction `cutoff_date` into `get_lagged_target_features()`. |
| 5 | [scripts/audit_current_pipeline_pit.py](scripts/audit_current_pipeline_pit.py) | Added a reproducible current-pipeline PIT audit for NSA branch-target lags, rolling panel replacement, Kalman history use, snapshot cutoffs, and stale artifacts. |

### Audit result

`scripts/audit_current_pipeline_pit.py --output-base _output_pairing_baseline_pitfix`
wrote `_output_pairing_baseline_pitfix/pit_audit_current_pipeline/` and found:

- `snapshot_cutoffs`: PASS, 24 latest master snapshots, 0 cutoff violations.
- `nsa_branch_target_lags`: PASS_AFTER_CODE_FIX, 2,157 selected feature-month values
  changed from legacy non-null to PIT-missing.
- Changed selected NSA target-derived features:
  `nfp_nsa_accel_rolling_3m`, `nfp_nsa_accel_vol_3m`,
  `nfp_nsa_mom_abs_lag1`, `nfp_nsa_mom_abs_rolling_6m`,
  `nfp_nsa_mom_vol_3m`.
- `panel_replacement`: PASS, 0 trained-through leaks and 0 forecast-release leaks.
- `kalman_fusion`: PASS, 0 history order violations.
- `artifact_staleness`: STALE_ARTIFACTS, because the latest local model/fusion files
  predate this code fix and must be rerun before their metrics are trusted.

### Unit tests

`tests/test_data_loader.py` includes:

- `test_revised_lag_excludes_same_day_available_value`
- `test_batch_revised_lags_respect_cutoff_map`

Both verify that same-day operational availability is excluded under strict
`operational_available_date < cutoff_date`.

### Required regeneration

```
python Train/train_lightgbm_nfp.py --train-all
```

The output root should then be audited again with:

```
python scripts/audit_current_pipeline_pit.py --output-base <new_output_root>
```

---

## ISSUE 15 — FIXED 2026-05-17: Final-layer Kalman/router and sidecar target dynamics needed operational actual availability

**Status:** FIXED 2026-05-17. A second same-day revised-actual issue existed outside
the LightGBM feature matrix. The post-train Kalman fusion, adaptive Kalman grid, and
Panel/Kalman Router used chronological prior actual rows (`ds < current ds`) for
noise estimation, state resets, and router rule scoring. For revised NFP targets,
the immediately prior month's revised actual is commonly operationally available on
the same release date as the current forecast, so chronology alone is not strict PIT.

Sidecar target-dynamics helpers had the same issue when they used `shift(1)` on the
revised target table to build `prev_mom`, target lags, and acceleration labels.

### Fixes applied

| # | File | Change |
|---|---|---|
| 1 | [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | `build_merged_dataset()` now carries `target_release_date` and `actual_available_date` from the SA revised target parquet into the final-layer dataset. |
| 2 | [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | `kalman_fusion()` now filters history to `actual_available_date < target_release_date`, resets state only to actuals known before the current release, and scales random-walk process noise by the month gap from the latest known actual. |
| 3 | [Train/Output_code/consensus_anchor_runner.py](Train/Output_code/consensus_anchor_runner.py) | `pit_adaptive_kalman_fusion()` and `build_panel_kalman_router()` now score candidate params/rules only on prior operationally available actuals. |
| 4 | [experiments/sidecars/feature_matrix.py](experiments/sidecars/feature_matrix.py) | Sidecar target dynamics use a PIT history window when release and operational availability columns exist; plain shift is retained only for synthetic/legacy frames without availability metadata. |
| 5 | [experiments/sidecars/economist_panel_sidecar.py](experiments/sidecars/economist_panel_sidecar.py) | Economist sidecar track records now use actuals whose `operational_available_date` is strictly before the target release cutoff. |
| 6 | [experiments/sidecars/acceleration_classifier_sidecar.py](experiments/sidecars/acceleration_classifier_sidecar.py) | Acceleration-classifier composite features use PIT-built `*_accel_lag1` rather than `actual_accel.shift(1)`. |
| 7 | [Train/Output_code/sa_consensus_anchor_runner.py](Train/Output_code/sa_consensus_anchor_runner.py) | Isolated SA challenger fusion now uses the same operational-availability filter for R/Q/state updates. |
| 8 | [Train/training_dataset_cache.py](Train/training_dataset_cache.py) | Training dataset cache schema bumped to `2` so old cached matrices from pre-fix target-lag semantics are not reused. |
| 9 | [scripts/audit_current_pipeline_pit.py](scripts/audit_current_pipeline_pit.py) | Audit now checks operational-history fields, cache schema version, and sidecar target-dynamics line coverage. |

### Audit result

The refreshed audit report lives under
`_output_pairing_baseline_pitfix/pit_audit_current_pipeline/`.

- Static line audit: PASS, 14 PIT checks.
- NSA branch-target lags: PASS_AFTER_CODE_FIX, same 2,157 selected feature-month
  values corrected to PIT-missing.
- Panel replacement: PASS, 0 trained-through leaks and 0 forecast-release leaks.
- Training dataset cache schema: PASS, `SCHEMA_VERSION=2`.
- Snapshot cutoffs: PASS, 24 latest snapshots checked.
- Existing official Kalman CSVs: `MISSING_OPERATIONAL_FIELDS` / stale, because they
  predate this fix and do not yet contain `target_release_date`,
  `actual_available_date`, or `history_available_n`.
- Runtime non-destructive Kalman check:
  `_output_pairing_baseline_pitfix/pit_audit_current_pipeline/kalman_fusion_runtime_pit_validation.csv`
  has 59 rows and 0 operational-history violations.

### Unit tests

Added regression tests for:

- Kalman fusion excluding same-day prior revised actuals.
- Adaptive Kalman grid selection excluding same-day prior revised actuals.
- Panel/Kalman Router excluding same-day prior revised actuals.
- Sidecar target dynamics excluding same-day revised target shifts.
- Economist panel sidecar track records excluding actuals not operationally available.

Focused command:

```
FRED_API_KEY=dummy DATA_PATH=data START_DATE=1990-01-01 BACKTEST_MONTHS=60 UNIFIER_USER=dummy UNIFIER_TOKEN=dummy pytest tests/test_consensus_panel_anchor.py tests/test_kalman_optuna_pit.py tests/test_acceleration_classifier_sidecar.py tests/test_economist_panel_sidecar.py tests/test_data_loader.py tests/test_sidecar_integration.py
```

Result: 55 passed.

---

## Master-snapshot infrastructure leverage (architectural takeaway)

A direct benefit of Fix 1 (Issue 8) is that **the master-snapshot infrastructure is now
the single source of truth for consensus** — the same `NFP_Consensus_Mean` value the
LightGBM models split on is the one the consensus_anchor stage consumes. Three
properties this gives the pipeline:

1. **PIT-correct by construction.** The ETL contract (`release_date < snap_date`)
   propagates automatically to every consumer that loads from the master snapshot.
   No per-consumer PIT logic to validate.

2. **Schema-resilient.** Future Unifier behavioural changes (post-release consensus
   updates, late respondents, schema renames) are absorbed by the ETL once and do not
   reach the post-train consumers as silent leaks.

3. **Pattern for future post-train features.** Any new variant that needs to consume a
   PIT-correct exogenous series should follow the same pattern: lift the series into
   the master snapshot at ETL time, then read it via `_load_*_pit(target_type,
   target_source)` style loaders that walk the master-snapshot tree.

The remaining post-train consumers (`load_adjustment_history`, `build_revised_target`)
already enforce PIT correctly via `operational_available_date` filters on the target
parquets — no further architectural changes are needed there.
