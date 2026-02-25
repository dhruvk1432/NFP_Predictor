"""
Stage 0 Spearman Pre-Screen — Isolation Test
=============================================
Measures how much the new _vectorized_target_prescreen() prunes
FRED_Employment_NSA before Stage 1 sees it.

Replicates the exact pre-processing of _process_source_features():
  1. Load latest NSA snapshot
  2. Apply source min-history filter (96 obs)
  3. Apply zero-variance filter
  4. Run Stage 0 variance filter   → baseline
  5. Run Stage 0 + Spearman prescreen → new

Usage:
    python scripts/test_stage0_prescreen.py
    python scripts/test_stage0_prescreen.py --fdr-alpha 0.20
    python scripts/test_stage0_prescreen.py --fdr-alpha 0.40
    python scripts/test_stage0_prescreen.py --show-corr-dist
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger
from Data_ETA_Pipeline.feature_selection_engine import (
    load_snapshot_wide,
    _variance_filter,
    _vectorized_target_prescreen,
)
from utils.transforms import winsorize_covid_period

logger = setup_logger(__file__, TEMP_DIR)

# ---------------------------------------------------------------------------
# Constants (mirror create_master_snapshots.py)
# ---------------------------------------------------------------------------
SOURCE_NAME = 'FRED_Employment_NSA'
SOURCE_DIR = DATA_PATH / "fred_data_prepared_nsa" / "decades"
TARGET_PATH = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
MIN_VALID_OBS = 96  # Same as SOURCE_MIN_VALID_OBS for FRED_Employment_NSA
PRESCREEN_THRESHOLD = 5000
PRESCREEN_MIN_SURVIVORS = 500


def _latest_snapshot(source_dir: Path) -> Path | None:
    """Return the most recent .parquet file under source_dir."""
    files = sorted(source_dir.rglob("*.parquet"))
    return files[-1] if files else None


def _load_target(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    # Normalize: the target files use a 'ds' date column and 'y' value column
    date_col = next((c for c in ('ds', 'date') if c in df.columns), None)
    if date_col is not None:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    # Prefer 'y' column; fall back to first numeric column
    if 'y' in df.columns:
        return df['y'].dropna()
    col = df.select_dtypes(include=[np.number]).columns[0]
    return df[col].dropna()


def _print_separator(char='─', width=70):
    print(char * width)


def _corr_distribution(r_vals: np.ndarray, label: str, bins=10):
    """Print a text histogram of |corr| values."""
    abs_r = np.abs(r_vals)
    if len(abs_r) == 0:
        print(f"\n  {label} |ρ| distribution: (empty)")
        return
    edges = np.linspace(0, abs_r.max() + 1e-6, bins + 1)
    counts, _ = np.histogram(abs_r, bins=edges)
    print(f"\n  {label} |ρ| distribution (n={len(abs_r)}):")
    max_cnt = max(counts.max(), 1)
    for i, cnt in enumerate(counts):
        bar = '█' * min(cnt * 40 // max_cnt, 40)
        print(f"    [{edges[i]:.3f}-{edges[i+1]:.3f}]  {bar} {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Test Stage 0 Spearman pre-screen")
    parser.add_argument('--fdr-alpha', type=float, default=0.30,
                        help='BH-FDR alpha for pre-screen (default: 0.30)')
    parser.add_argument('--show-corr-dist', action='store_true',
                        help='Print |corr| distribution before/after pre-screen')
    args = parser.parse_args()

    _print_separator('═')
    print(f"  Stage 0 Pre-Screen Isolation Test — {SOURCE_NAME}")
    print(f"  fdr_alpha = {args.fdr_alpha}")
    _print_separator('═')

    # ------------------------------------------------------------------
    # 1. Load latest snapshot (same as _process_source_features step 1)
    # ------------------------------------------------------------------
    print("\n[1/5] Loading latest snapshot...")
    t_load = time.time()
    latest = _latest_snapshot(SOURCE_DIR)
    if latest is None:
        print("ERROR: No snapshot files found.")
        sys.exit(1)
    snap_wide = load_snapshot_wide(latest)
    print(f"      Loaded {latest.name}: {snap_wide.shape[0]} rows × {snap_wide.shape[1]} cols  "
          f"({time.time() - t_load:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Source min-history filter (same as _process_source_features step 2)
    # ------------------------------------------------------------------
    print(f"\n[2/5] Applying min-history filter ({MIN_VALID_OBS} obs)...")
    valid_counts = snap_wide.count()
    short_features = valid_counts[valid_counts < MIN_VALID_OBS].index
    if len(short_features):
        snap_wide = snap_wide.drop(columns=short_features)
        print(f"      Dropped {len(short_features)} short-history features "
              f"→ {snap_wide.shape[1]} remaining")
    zero_var = snap_wide.std() == 0
    if zero_var.any():
        snap_wide = snap_wide.loc[:, ~zero_var]
        print(f"      Dropped {int(zero_var.sum())} zero-variance features "
              f"→ {snap_wide.shape[1]} remaining")
    n_after_source_filter = snap_wide.shape[1]
    print(f"      After source filters: {n_after_source_filter} features")

    # ------------------------------------------------------------------
    # 3. Load & align target
    # ------------------------------------------------------------------
    print("\n[3/5] Loading target...")
    y_raw = _load_target(TARGET_PATH)
    y_target = winsorize_covid_period(y_raw)
    common_all = snap_wide.index.intersection(y_target.dropna().index)
    print(f"      Target: {len(y_target)} obs  |  Common dates with snap_wide: {len(common_all)}")

    # ------------------------------------------------------------------
    # 4. Stage 0 — variance filter (baseline, no pre-screen)
    # ------------------------------------------------------------------
    print("\n[4/5] Stage 0 — variance filter (baseline)...")
    t0 = time.time()
    snap_var = _variance_filter(snap_wide.copy())
    t_var = time.time() - t0
    n_after_variance = snap_var.shape[1]
    print(f"      {n_after_source_filter} → {n_after_variance} features  ({t_var:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Stage 0 — variance filter + Spearman pre-screen (new)
    # ------------------------------------------------------------------
    print(f"\n[5/5] Stage 0 — Spearman pre-screen (fdr_alpha={args.fdr_alpha})...")
    if n_after_variance <= PRESCREEN_THRESHOLD:
        print(f"      NOTE: {n_after_variance} cols ≤ threshold {PRESCREEN_THRESHOLD}. "
              f"Pre-screen would be skipped in pipeline. Running anyway for measurement.")

    t_pre = time.time()
    # Internal call — no winsorize here; y_target is already winsorized above
    survivors = _vectorized_target_prescreen(
        snap_var, y_target,
        fdr_alpha=args.fdr_alpha,
        min_corr_obs=30,
    )
    t_prescreen = time.time() - t_pre

    n_survivors = len(survivors)
    n_dropped = n_after_variance - n_survivors
    pct_dropped = 100 * n_dropped / max(n_after_variance, 1)

    # ------------------------------------------------------------------
    # Optional: correlation distribution
    # ------------------------------------------------------------------
    if args.show_corr_dist:
        y_valid_idx = y_target.dropna().index
        common = snap_var.index.intersection(y_valid_idx)
        X_ranked = snap_var.loc[common].rank(pct=True, na_option='keep')
        y_ranked = y_target.loc[common].rank(pct=True)
        corrs = X_ranked.corrwith(y_ranked)
        r_all = corrs.fillna(0.0).values
        r_surv = corrs.reindex(survivors).fillna(0.0).values
        r_drop = corrs.drop(index=survivors, errors='ignore').fillna(0.0).values
        _corr_distribution(r_all, "All features")
        _corr_distribution(r_surv, "Survivors")
        _corr_distribution(r_drop, "Dropped")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_separator()
    print("\n  RESULTS SUMMARY")
    _print_separator()
    print(f"  {'Step':<40} {'Features':>10}  {'Time':>8}")
    _print_separator('-')
    print(f"  {'After load + source filters':<40} {n_after_source_filter:>10}")
    print(f"  {'After variance filter (Stage 0 baseline)':<40} {n_after_variance:>10}  {t_var:>7.1f}s")
    print(f"  {'After Spearman pre-screen (Stage 0 new)':<40} {n_survivors:>10}  {t_prescreen:>7.1f}s")
    _print_separator()
    print(f"\n  Reduction:  {n_after_variance} → {n_survivors}  "
          f"({n_dropped} dropped, {pct_dropped:.1f}%)")
    print(f"  Pre-screen safe to apply: "
          f"{'YES' if n_survivors >= PRESCREEN_MIN_SURVIVORS else 'NO (below floor)'}")
    print(f"\n  Stage 1 will now operate on {n_survivors} features instead of "
          f"{n_after_variance}")

    # Rough speedup estimate (Stage 1 scales approximately linearly with p)
    speedup = n_after_variance / max(n_survivors, 1)
    print(f"  Estimated Stage 1 speedup: ~{speedup:.1f}x")
    _print_separator('═')


if __name__ == '__main__':
    main()
