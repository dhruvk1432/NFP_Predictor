# Best_features_selected/

Tracked-in-git store of the **best known dynamic-reselection JSON files** —
the ones that produced morning sanity's 88.9 MAE Kalman fusion result on
2026-05-12.

Why this exists: `_output/dynamic_selection/` is .gitignored, and any fresh
reselection run overwrites the JSONs there. To preserve the best-known
feature schedule across runs (and across machines), copy the canonical
JSONs into the appropriate subdirectory here.

## Expected layout

```
Best_features_selected/
├── nsa_revised/
│   ├── 2021-04.json   (morning, 09:56:03 mtime)
│   ├── 2021-06.json   (morning, 09:56:03 mtime — the lost one to be restored)
│   ├── 2024-04.json   (morning, 09:56:03 mtime)
│   └── 2024-06.json   (morning, 09:56:03 mtime)
└── sa_revised/
    ├── 2021-04.json
    ├── 2021-06.json
    ├── 2024-04.json
    └── 2024-06.json
```

## To use (replay morning)

```bash
# Empty live dir, copy from canonical, touch mtimes into a single cohort
rm _output/dynamic_selection/nsa_revised/*.json
rm _output/dynamic_selection/sa_revised/*.json
cp Best_features_selected/nsa_revised/*.json _output/dynamic_selection/nsa_revised/
cp Best_features_selected/sa_revised/*.json _output/dynamic_selection/sa_revised/
touch _output/dynamic_selection/nsa_revised/*.json _output/dynamic_selection/sa_revised/*.json

# Then run nsa_then_kalman.py (per-window mode active, will replay these)
python scripts/nsa_then_kalman.py
```

## Notes

- These JSONs were produced by a NON-deterministic LightGBM/Boruta run.
  Re-running fresh reselection will not reproduce them. Treat as a
  preserved snapshot.
- `LGB_PARAMS` in `Data_ETA_Pipeline/feature_selection_engine.py`
  intentionally lacks `deterministic=True` and `force_col_wise=True` —
  morning's run was non-deterministic and we preserve that behavior so any
  *future* lucky run can be saved here too.
