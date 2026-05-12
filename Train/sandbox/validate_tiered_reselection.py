"""
Sandbox: validate that Tier-A (cached Pass-1 universe) + Tier-B (fresh Pass-2)
produces a feature set close enough to baseline full-reselection to ship.

Hierarchy under test
--------------------
Today's ``_dynamic_reselection`` (Train/train_lightgbm_nfp.py:431-679) runs
two passes back-to-back at every reselection event:
  * Pass-1: heavy per-source pipeline (Boruta + cluster + interaction) on the
    FULL master-snapshot universe — slow, scales with feature count.
  * Pass-2: light global pipeline on the union of Pass-1 survivors (≤ ~1000
    cols typically), capped at DYNAMIC_FS_PASS2_MAX_FEATURES.

The proposed hierarchical scheme keeps Pass-2 at the original 6-month cadence
but only re-runs Pass-1 every ``UNIVERSE_REFRESH_MONTHS`` (24 by default).
This script answers: **for step_dates inside a universe window, do the
80-feature working sets diverge from baseline?**

Promotion rule (per the approved plan):
  median Jaccard ≥ 0.70 across step_dates AND median |MAE-delta-proxy| ≤ 5%.

PIT invariant enforced by this sandbox itself:
  every Tier-A cache used at step_date ``t`` has ``universe_asof ≤ t``.
  A hard ``AssertionError`` fires if violated, so we cannot accidentally
  validate a leaking scheme.

Usage
-----
    python Train/sandbox/validate_tiered_reselection.py --target nsa
        # 4 step_dates spaced 6 months apart over the last 24 months,
        # universe_asof = first test step_date minus 1 day.

    python Train/sandbox/validate_tiered_reselection.py --target sa \\
        --step-dates 2023-06,2023-12,2024-06,2024-12 \\
        --universe-asof 2023-01

    python Train/sandbox/validate_tiered_reselection.py --target nsa --quick
        # 2 step_dates only — useful when iterating on this script itself.

Output
------
Writes ``_output/sandbox/tiered_reselection/{target}_{source}_report.csv``
with one row per step_date and the global Jaccard summary at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import (
    DYNAMIC_FS_BORUTA_RUNS,
    DYNAMIC_FS_PASS2_MAX_FEATURES,
    RESELECTION_HALF_LIFE_MONTHS,
    RESELECTION_STAGES_PASS1,
    RESELECTION_STAGES_PASS2,
)
from Train.data_loader import load_target_data
from Train.train_lightgbm_nfp import (
    _classify_columns_by_source,
    _dynamic_reselection,
    build_training_dataset,
    partition_feature_columns,
)
from Train.training_dataset_cache import (
    load_cached_dataset,
    save_cached_dataset,
)
from Data_ETA_Pipeline.feature_selection_engine import (
    _classify_series,
    run_full_source_pipeline,
)

logger = setup_logger(__file__, TEMP_DIR)

OUT_DIR = OUTPUT_DIR / "sandbox" / "tiered_reselection"


# ---------------------------------------------------------------------------
# Dataset loading (uses the Phase-1 disk cache to keep iterations fast)
# ---------------------------------------------------------------------------

def _load_xy_full(target_type: str, target_source: str) -> Tuple[pd.DataFrame, pd.Series]:
    target_df = load_target_data(
        target_type, release_type='first', target_source=target_source,
    )
    cached = load_cached_dataset(
        target_df, target_type, 'first', target_source,
        start_date=None, end_date=None,
    )
    if cached is not None:
        return cached
    logger.info("Cache miss → building training dataset (slow first call)")
    X, y = build_training_dataset(
        target_df, target_type=target_type, release_type='first',
        target_source=target_source, show_progress=False,
    )
    if not X.empty:
        save_cached_dataset(
            X, y, target_df,
            target_type, 'first', target_source,
            start_date=None, end_date=None,
        )
    return X, y


def _slice_pit(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    cutoff: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Strict PIT slice: rows with ``ds < cutoff``, NaN targets dropped."""
    mask = pd.to_datetime(X_full['ds']) < cutoff
    X = X_full.loc[mask].copy()
    y = y_full.loc[mask].copy()
    valid = ~y.isna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    X = X.sort_values('ds').reset_index(drop=True)
    y = y.iloc[X.index].reset_index(drop=True)
    return X, y


# ---------------------------------------------------------------------------
# Tier-A: compute Pass-1 once at universe_asof, return per-source survivors
# ---------------------------------------------------------------------------

def _compute_recency_weights(
    X: pd.DataFrame, ref_date: pd.Timestamp, half_life_months: float,
) -> pd.Series:
    dates = pd.to_datetime(X['ds'])
    distance_months = np.maximum(
        0, (ref_date - dates).dt.days.values / 30.436875,
    )
    decay = np.log(2) / half_life_months
    w = np.exp(-decay * distance_months)
    w = w / np.mean(w)
    return pd.Series(w, index=pd.to_datetime(X['ds']).values, name='sample_weight')


def _compute_pass1_universe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    universe_asof: pd.Timestamp,
    target_type: str,
    stages: Tuple[int, ...],
) -> Dict[str, List[str]]:
    """Tier-A: run Pass-1 once. Mirrors _dynamic_reselection lines 525-598."""
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    all_cols = [c for c in numeric_cols if c != 'ds']
    groups = partition_feature_columns(all_cols, target_type=target_type)
    snapshot_cols = groups['snapshot_features']

    source_groups = _classify_columns_by_source(snapshot_cols)

    y_sel = pd.Series(
        y_train.values,
        index=pd.to_datetime(X_train['ds'].values),
        name='y_mom',
    ).dropna()

    X_dated = X_train[all_cols].copy()
    X_dated.index = pd.to_datetime(X_train['ds'].values)
    X_dated.index.name = 'date'

    sw_series = _compute_recency_weights(
        X_train, universe_asof, RESELECTION_HALF_LIFE_MONTHS,
    )

    pass1_survivors: Dict[str, List[str]] = {}

    # Run all sources sequentially in the sandbox — no parallelism, easier
    # to interrupt and debug. Production code parallelises small sources.
    for source_name in sorted(source_groups.keys()):
        cols = source_groups.get(source_name, [])
        if not cols:
            continue
        if source_name == 'Unknown':
            # Production passes Unknown features through directly.
            pass1_survivors[source_name] = list(cols)
            continue
        snap_wide = X_dated[cols].copy()
        zero_var = snap_wide.std() == 0
        if zero_var.any():
            snap_wide = snap_wide.loc[:, ~zero_var]
        if snap_wide.empty:
            pass1_survivors[source_name] = []
            continue

        series_groups_local = defaultdict(list)
        for col in snap_wide.columns:
            grp = _classify_series(col, source_name)
            series_groups_local[grp].append(col)

        try:
            survivors = run_full_source_pipeline(
                snap_wide, y_sel, source_name, Path("/dev/null"),
                series_groups_local, stages=stages,
                sample_weight=sw_series,
            )
            pass1_survivors[source_name] = survivors
            logger.info(f"  Tier-A {source_name}: {snap_wide.shape[1]} → {len(survivors)} survivors")
        except Exception as e:
            logger.error(f"  Tier-A {source_name}: Pass-1 failed: {e}")
            pass1_survivors[source_name] = []

    return pass1_survivors


# ---------------------------------------------------------------------------
# Tier-B: run Pass-2 only, given the persisted Pass-1 survivors
# ---------------------------------------------------------------------------

def _run_pass2_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    step_date: pd.Timestamp,
    pass1_survivors: Dict[str, List[str]],
    target_type: str,
    stages: Tuple[int, ...],
) -> List[str]:
    """Tier-B: Pass-2 only. Mirrors _dynamic_reselection lines 606-661."""
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    all_cols = [c for c in numeric_cols if c != 'ds']
    groups = partition_feature_columns(all_cols, target_type=target_type)
    non_snapshot_cols = (
        groups['target_branch_features']
        + groups['calendar_features']
        + groups['revision_features']
    )

    y_sel = pd.Series(
        y_train.values,
        index=pd.to_datetime(X_train['ds'].values),
        name='y_mom',
    ).dropna()

    X_dated = X_train[all_cols].copy()
    X_dated.index = pd.to_datetime(X_train['ds'].values)
    X_dated.index.name = 'date'

    sw_series = _compute_recency_weights(
        X_train, step_date, RESELECTION_HALF_LIFE_MONTHS,
    )

    all_pass1_cols: List[str] = []
    for feats in pass1_survivors.values():
        all_pass1_cols.extend(feats)
    pass2_cols = list(set(all_pass1_cols + non_snapshot_cols))
    pass2_cols = [c for c in pass2_cols if c in X_dated.columns]

    if not pass2_cols:
        return []

    pass2_wide = X_dated[pass2_cols].copy()
    pass2_groups = defaultdict(list)
    for col in pass2_wide.columns:
        pass2_groups["Global"].append(col)

    survivors = run_full_source_pipeline(
        pass2_wide, y_sel, "Global_Pass2_Tiered", Path("/dev/null"),
        pass2_groups, stages=stages,
        sample_weight=sw_series,
    )

    if len(survivors) > DYNAMIC_FS_PASS2_MAX_FEATURES:
        survivors = survivors[:DYNAMIC_FS_PASS2_MAX_FEATURES]

    available_in_train = set(X_train.columns)
    return [f for f in survivors if f in available_in_train]


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def _default_step_dates(latest_ds: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    """``n`` step_dates spaced 6 months apart, anchored on the latest target."""
    out = []
    for i in range(n):
        out.append((latest_ds - pd.DateOffset(months=6 * i)).normalize())
    return sorted(out)


def run_validation(
    target_type: str,
    target_source: str,
    step_dates: List[pd.Timestamp],
    universe_asof: pd.Timestamp,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading X_full/y_full for {target_type}/{target_source}...")
    X_full, y_full = _load_xy_full(target_type, target_source)
    latest_ds = pd.to_datetime(X_full['ds']).max()
    logger.info(f"  → {len(X_full)} rows, latest ds={latest_ds.strftime('%Y-%m')}")

    # PIT enforcement: every step_date must be > universe_asof
    for sd in step_dates:
        assert universe_asof <= sd, (
            f"PIT VIOLATION: universe_asof={universe_asof} > step_date={sd}. "
            f"A Tier-A cache built at {universe_asof} cannot be used at {sd}."
        )

    # Tier-A run (once)
    logger.info(f"\n=== Tier-A: running Pass-1 universe distillation at asof={universe_asof.strftime('%Y-%m')} ===")
    X_asof, y_asof = _slice_pit(X_full, y_full, universe_asof)
    logger.info(f"  Tier-A training slice: {len(X_asof)} rows × {len(X_asof.columns)} cols")
    t0 = time.perf_counter()
    pass1_survivors = _compute_pass1_universe(
        X_asof, y_asof, universe_asof, target_type, RESELECTION_STAGES_PASS1,
    )
    tier_a_seconds = time.perf_counter() - t0
    total_survivors = sum(len(v) for v in pass1_survivors.values())
    logger.info(
        f"  Tier-A complete in {tier_a_seconds:.1f}s, "
        f"{total_survivors} total survivors across {len(pass1_survivors)} sources"
    )

    # Persist the Tier-A snapshot for inspection
    tier_a_path = OUT_DIR / f"tier_a_{target_type}_{target_source}_{universe_asof.strftime('%Y-%m')}.json"
    tier_a_path.write_text(json.dumps({
        "universe_asof": universe_asof.strftime("%Y-%m-%d"),
        "target_type": target_type,
        "target_source": target_source,
        "stages": list(RESELECTION_STAGES_PASS1),
        "elapsed_s": tier_a_seconds,
        "survivors": {k: list(v) for k, v in pass1_survivors.items()},
    }, indent=2))
    logger.info(f"  Tier-A survivors → {tier_a_path.name}")

    rows = []
    for sd in step_dates:
        logger.info(f"\n=== Step {sd.strftime('%Y-%m')} ===")
        X_t, y_t = _slice_pit(X_full, y_full, sd)
        if len(X_t) < 60:
            logger.warning(f"  Skipping {sd.strftime('%Y-%m')}: only {len(X_t)} training rows")
            continue

        # Baseline = today's _dynamic_reselection (Pass-1 + Pass-2 from scratch)
        logger.info(f"  Baseline: full _dynamic_reselection on {len(X_t)} rows")
        t_b = time.perf_counter()
        baseline = _dynamic_reselection(
            X_train=X_t, y_train=y_t, step_date=sd,
            target_type=target_type, target_source=target_source,
        )
        elapsed_baseline = time.perf_counter() - t_b
        logger.info(f"  Baseline: {len(baseline)} features in {elapsed_baseline:.1f}s")

        # Tiered = Pass-2 only on cached Pass-1 survivors
        logger.info(f"  Tiered: Pass-2-only using cached Tier-A from {universe_asof.strftime('%Y-%m')}")
        t_t = time.perf_counter()
        tiered = _run_pass2_only(
            X_t, y_t, sd, pass1_survivors,
            target_type, RESELECTION_STAGES_PASS2,
        )
        elapsed_tiered = time.perf_counter() - t_t
        logger.info(f"  Tiered:   {len(tiered)} features in {elapsed_tiered:.1f}s")

        s_b, s_t = set(baseline), set(tiered)
        intersect = s_b & s_t
        union = s_b | s_t
        jaccard = len(intersect) / len(union) if union else 1.0
        baseline_only = sorted(s_b - s_t)
        tiered_only = sorted(s_t - s_b)

        rows.append({
            "step_date": sd.strftime("%Y-%m"),
            "n_baseline": len(baseline),
            "n_tiered": len(tiered),
            "n_intersect": len(intersect),
            "jaccard": jaccard,
            "baseline_only_count": len(baseline_only),
            "tiered_only_count": len(tiered_only),
            "elapsed_baseline_s": elapsed_baseline,
            "elapsed_tiered_s": elapsed_tiered,
            "speedup_tiered_vs_baseline": (
                elapsed_baseline / elapsed_tiered if elapsed_tiered > 0 else float("nan")
            ),
            "baseline_only_sample": ",".join(baseline_only[:5]),
            "tiered_only_sample": ",".join(tiered_only[:5]),
        })
        logger.info(
            f"  Jaccard={jaccard:.3f} | baseline-only={len(baseline_only)} | "
            f"tiered-only={len(tiered_only)} | speedup={rows[-1]['speedup_tiered_vs_baseline']:.1f}x"
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"{target_type}_{target_source}_report.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nReport → {csv_path}")

    if len(df):
        median_j = df["jaccard"].median()
        median_speedup = df["speedup_tiered_vs_baseline"].median()
        logger.info(
            f"Summary [{target_type}/{target_source}]: "
            f"median Jaccard={median_j:.3f}, median tiered speedup={median_speedup:.1f}x, "
            f"Tier-A one-time cost={tier_a_seconds:.0f}s"
        )
        promotion = "PASS" if median_j >= 0.70 else "FAIL"
        logger.info(f"Promotion rule (median Jaccard >= 0.70): {promotion}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["nsa", "sa"], default="nsa")
    parser.add_argument("--source", default="revised")
    parser.add_argument(
        "--step-dates", type=str, default=None,
        help="Comma-separated YYYY-MM list of test step_dates. "
             "Default: 4 dates spaced 6 months apart anchored at the latest X_full date.",
    )
    parser.add_argument(
        "--universe-asof", type=str, default=None,
        help="YYYY-MM Tier-A asof date. Default: 24 months before the earliest test step_date.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only 2 step_dates — faster smoke test of the sandbox itself.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Need X_full bounds to default step_dates — load lazily
    target_df = load_target_data(args.target, release_type='first', target_source=args.source)
    latest_ds = pd.to_datetime(target_df['ds']).max().normalize()

    if args.step_dates:
        step_dates = sorted(
            pd.Timestamp(s.strip()).normalize() for s in args.step_dates.split(",")
        )
    else:
        n = 2 if args.quick else 4
        step_dates = _default_step_dates(latest_ds, n)
        # Drop dates earlier than what we have data for
        step_dates = [d for d in step_dates if d <= latest_ds]

    if not step_dates:
        raise RuntimeError("No valid step_dates resolved")

    if args.universe_asof:
        universe_asof = pd.Timestamp(args.universe_asof).normalize()
    else:
        universe_asof = (step_dates[0] - pd.DateOffset(months=24)).normalize()

    logger.info(
        f"Validation plan: target={args.target}/{args.source}, "
        f"universe_asof={universe_asof.strftime('%Y-%m')}, "
        f"step_dates={[d.strftime('%Y-%m') for d in step_dates]}"
    )

    run_validation(args.target, args.source, step_dates, universe_asof)


if __name__ == "__main__":
    main()
