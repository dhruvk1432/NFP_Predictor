# `Best_features_selected/` — preserved dynamic-FS feature schedules

`Best_features_selected/` is a **tracked-in-git** store of the best-known dynamic-reselection JSON cohorts produced by the walk-forward pipeline. It exists because `_output/dynamic_selection/` is gitignored and is overwritten by every fresh reselection run, so a lucky non-deterministic selection cohort would otherwise be lost the next time the pipeline ran.

## Why it exists

The feature-selection engine ([`Data_ETA_Pipeline/feature_selection_engine.py`](../Data_ETA_Pipeline/feature_selection_engine.py)) intentionally **does not pin determinism** inside its inner LightGBM / Boruta `LGB_PARAMS`:

```python
LGB_PARAMS = {
    'objective': 'regression', 'metric': 'l2',
    'n_estimators': 100, 'learning_rate': 0.05,
    'num_leaves': 31, 'n_jobs': 1,
    'random_state': 42,
    # NOTE: deterministic=True and force_col_wise=True are deliberately omitted
}
```

This means that two fresh runs against the same data and the same hyperparameters can produce **different** Stage-2 (Boruta) survivor sets — Boruta's shadow features are permuted by NumPy's RNG inside multi-process workers, and without `deterministic=True` LightGBM's split-finding is non-deterministic across threads. Empirically, the resulting per-step JSONs have Jaccard ~0.85 across reruns at fixed seed, which is enough variation that a fresh rerun rarely recovers the best-known cohort.

The training entrypoint, in contrast, is fully deterministic (`Train/config.py:LGBM_DETERMINISM`) — the **model** is reproducible; the **feature selection** is not.

So when a particular walk-forward run produces a fusion result that materially beats the rolling best — e.g. the **88.9 MAE morning-of-2026-05-12 Kalman fusion** — we copy the entire reselection cohort here so it can be replayed verbatim on any machine.

## Expected layout

```
Best_features_selected/
├── README.md           (this file)
├── nsa_revised/
│   ├── 2021-06.json
│   ├── 2022-05.json
│   ├── 2023-05.json
│   ├── 2024-05.json
│   └── 2026-05.json
└── sa_revised/         (kept for reference; SA LightGBM is retired)
    ├── 2021-04.json
    └── …
```

Each JSON matches the schema written by [`train_lightgbm_nfp.py:_save_per_window_features`](../Train/train_lightgbm_nfp.py):

```json
{
  "target_type":   "nsa",
  "target_source": "revised",
  "step_date":     "2026-05",
  "step_index":    59,
  "n_features":    80,
  "features":      [ "ADP_actual_pct_chg_rolling_mean_3m", "…" ]
}
```

`n_features` is capped by [`DYNAMIC_FS_PASS2_MAX_FEATURES = 80`](../Train/config.py). The filename `{YYYY-MM}.json` is the `step_date` — the cutoff month for the expanding window that produced this selection.

> The directory ships with `.gitkeep` placeholders; populate it manually when a run produces a result worth preserving.

## How to use (replay a preserved cohort)

```bash
# 1) Wipe the live dynamic-selection directory so the replay loader
#    sees only the cohort you want.
rm -f _output/dynamic_selection/nsa_revised/*.json
rm -f _output/dynamic_selection/sa_revised/*.json

# 2) Copy the canonical cohort into the live directory.
cp Best_features_selected/nsa_revised/*.json _output/dynamic_selection/nsa_revised/
cp Best_features_selected/sa_revised/*.json  _output/dynamic_selection/sa_revised/

# 3) Touch the mtimes into a single "cohort window" so the replay loader
#    picks them up as one run (it filters to JSONs whose mtime is within
#    6 hours of the latest — see _load_per_window_feature_sets).
touch _output/dynamic_selection/nsa_revised/*.json
touch _output/dynamic_selection/sa_revised/*.json

# 4) Set replay mode and run.
USE_PER_WINDOW_FEATURES=True python scripts/nsa_then_kalman.py
#   …or…
USE_PER_WINDOW_FEATURES=True python Train/train_lightgbm_nfp.py --train-all
```

When `USE_PER_WINDOW_FEATURES=True` (env var loaded by [`settings.py`](../settings.py)), the training pipeline calls [`_load_per_window_feature_sets()`](../Train/train_lightgbm_nfp.py) instead of re-running `_dynamic_reselection()`. It loads every JSON in `_output/dynamic_selection/{target_type}_{target_source}/`, filters to the most-recent mtime cohort, and replays the saved feature schedule at each walk-forward step — completely skipping the slow Boruta / clustering passes.

## How to use (preserve a new "best known" cohort)

After a `--train-all` run produces a result you want to keep:

```bash
# Snapshot the live cohort into the preserved folder
cp _output/dynamic_selection/nsa_revised/*.json Best_features_selected/nsa_revised/
git add Best_features_selected/nsa_revised/
git commit -m "preserve NSA reselection cohort: MAE 88.9 Kalman fusion (2026-05-12)"
```

## What NOT to do

- **Do not** add `deterministic=True` / `force_col_wise=True` to `feature_selection_engine.py:LGB_PARAMS`. The non-determinism is intentional — it gives the dynamic FS multiple "rolls of the dice" across reselections, and the best of those rolls is what we preserve here. Pinning would lock the pipeline into a single (sometimes worse) feature schedule forever.
- **Do not** check in `_output/dynamic_selection/` itself. That directory is the *working* selection state; this folder is the *archive* of best-known states.
- **Do not** mix mtime cohorts. The replay loader filters by mtime window, so leaving stale `.json`s with significantly older mtimes alongside your fresh ones produces silent partial replays. Always wipe the live directory before copying.

## Related code

- [`Train/train_lightgbm_nfp.py:_load_per_window_feature_sets()`](../Train/train_lightgbm_nfp.py) — replay loader.
- [`Train/train_lightgbm_nfp.py:_save_per_window_features()`](../Train/train_lightgbm_nfp.py) — writer (called inside `_dynamic_reselection`).
- [`scripts/nsa_then_kalman.py`](../scripts/nsa_then_kalman.py) — re-run NSA training (with replay if env set) + Kalman fusion.
- [`scripts/reconstruct_nsa_and_kalman.py`](../scripts/reconstruct_nsa_and_kalman.py) — rebuild from a preserved cohort end-to-end.
