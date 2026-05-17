# Phase 0 — Branch isolation audit + locked baseline

Run date: 2026-05-16
Backtest window: 2021-06 → 2026-03 (58 observations, 60-month walk-forward setting)
Target: `y_sa_revised.y_mom` (SA revised MoM change, the market-moving number)

---

## 1. Branch-isolation audit (PASS)

The existing infrastructure is already target-type aware. No code changes are required before lifting SA-specific upgrades.

| Surface | Evidence | Status |
|---|---|---|
| CLI | `Train/train_lightgbm_nfp.py --help` exposes `--target {nsa,sa}` and `--release {first,last}`. | PASS |
| Output paths | [Train/train_lightgbm_nfp.py:535](Train/train_lightgbm_nfp.py#L535) writes to `OUTPUT_DIR / "dynamic_selection" / f"{target_type}_{target_source}"`. Model files land in `_output/models/lightgbm_nfp/{target_type}_first_revised/`. | PASS |
| Universe cache | [Train/universe_cache.py:44](Train/universe_cache.py#L44) regex enforces filename pattern `universe_(nsa\|sa)_(revised)_(YYYY-MM).json`. [Train/universe_cache.py:72-76](Train/universe_cache.py#L72-L76) globs by `target_type` and double-checks the regex group. Save also writes `target_type` to payload ([line 149](Train/universe_cache.py#L149)). | PASS |
| FS scorer cache | [Data_ETA_Pipeline/feature_selection_engine.py:900-935](Data_ETA_Pipeline/feature_selection_engine.py#L900-L935) `_feature_set_cache_key` is a per-call memoization keyed on feature-name tuple — lives within a single FS run, not across target_type boundaries. No leakage path. | PASS |
| Branch-specific tuning logic | [Train/train_lightgbm_nfp.py:1226-1248](Train/train_lightgbm_nfp.py#L1226-L1248) routes objective mode + FS method + top_k per `(target_type, target_source)`. SA branch can be configured independently. | PASS |
| Existing SA artifacts | `_output/models/lightgbm_nfp/sa_first_revised/` and `_output/backtest/sa_first_revised_metrics.json` exist alongside NSA artifacts. No cross-contamination observed in current `_output/`. | PASS |

**Conclusion**: We can build SA sidecars + SA Kalman fusion in `_output_sa_challenger/` without forking pipelines or modifying the production NSA runner. Re-running `--train-all` continues to produce both `nsa_first_revised` and `sa_first_revised` artifacts independently.

---

## 2. Locked baseline (60-month walk-forward, 58 obs)

Numbers below are extracted from existing artifacts — no re-run needed. These are the targets the SA challenger must beat.

| Model | RMSE | MAE | DirAcc | AccelAcc | STD_Ratio | Source |
|---|---:|---:|---:|---:|---:|---|
| **SA-LightGBM (sa_first_revised)** | 209.99 | 141.88 | 96.55% | 40.35% | 0.228 | `_output/backtest/sa_first_revised_metrics.json` (run 2026-05-12) |
| **SA Blend (legacy "champion")** | 176.84 | 125.63 | 93.10% | 66.67% | 0.783 | `_output/sandbox/sa_blend_walkforward/summary_metrics.json` |
| **Consensus baseline (`NFP_Consensus_Mean`)** | 144.03 | 101.28 | 96.55% | 66.67% | 0.816 | `_output/consensus_anchor/comparison_metrics.csv` |
| **NSA Kalman fusion (live production)** | 130.89 | **92.66** | 96.55% | **70.18%** | 0.813 | `_output/consensus_anchor/comparison_metrics.csv` |
| **Baseline "Champion" (SA blend in fusion runner)** | 249.10 | 191.74 | 81.03% | 64.91% | 1.459 | same |

**Key reading**:
- SA-LightGBM has the highest DirAcc (96.55%) but the worst AccelAcc (40.35%) — it's heavily shrunken (STD_Ratio 0.23), always predicts modest positive payrolls, so it gets direction right by default but cannot pick up MoM acceleration.
- The retired SA blend lifts MAE to 125.63 by fixing the variance shortfall (STD_Ratio 0.78), but DirAcc drops to 93.10% — it adds variance but also adds errors on direction.
- The live NSA Kalman fusion (the actual current production system) sits at MAE 92.66 / AccelAcc 70.18% — that's the bar the SA challenger must clear.
- Consensus `NFP_Consensus_Mean` at MAE 101.28 is the public benchmark we ultimately want to beat by ≥ 12 points on `sa_revised`.

**Live Kalman tuned params** (`_output/consensus_anchor/kalman_fusion/tuned_params.json`):
- `trailing_window = 28`
- `nsa_weight_scale = 0.1064`
- `half_life_years = 3.234`

---

## 3. Success bar for SA challenger (from plan)

The SA challenger pipeline (LightGBM + state-space + DFM + BVAR-GLP fused via Kalman) must **all four** dominate the live NSA Kalman fusion on the same 58-obs window:

| Metric | NSA fusion (champion) | SA challenger target | Beat by |
|---|---:|---:|---:|
| RMSE | 130.89 | ≤ 128.0 | ≥ 2.9 |
| MAE | 92.66 | ≤ 90.0 | ≥ 2.7 |
| DirAcc | 96.55% | ≥ 97% | ≥ 0.45pp |
| AccelAcc | 70.18% | ≥ 72% | ≥ 1.82pp |

Additionally, against the public benchmark:
- MAE vs `NFP_Consensus_Mean` (101.28) must beat by ≥ 12.0 points → SA fusion MAE ≤ 89.3.

If the challenger passes the gate, promote to production via Phase 5 (update `consensus_anchor_runner.py` to call `sa_consensus_anchor_runner.py` and combine NSA + SA fusion). If it does not, keep running as challenger — the density forecasts (state-space variance, BVAR posterior) still have diagnostic value for interval calibration.

---

## 4. Phase 0 outcome

- Audit: PASS. No remediation needed before Phase 1.
- Baseline: LOCKED in this file. Will be referenced by `_output_sa_challenger/comparison_metrics.csv` (Phase 4).
- **No new training run was required.** We deliberately extracted baseline metrics from artifacts already produced by the most recent NSA-champion training cycle (`_output/`) rather than launching a fresh `--iterate-fusion-tune --target sa --train-all` job, which would have been a multi-hour run with no new information — the existing SA artifacts in `_output/models/lightgbm_nfp/sa_first_revised/` were generated by exactly that pipeline on 2026-05-12 and remain current.

**Ready to start Phase 1**: generalize sidecar producers (`integration.py`, `bvar_prior_sidecar.py`, `acceleration_classifier_sidecar.py`, `hmm_acceleration_sidecar.py`) on `target_type`, and promote `direct_sa_residual_sidecar.py` to a first-class SA producer.
