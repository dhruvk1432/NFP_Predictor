"""
Data-loading smoke check.

Runs the full training dataset build for both NSA and SA branches without
any backtesting/training, then reports:
  - target series coverage
  - X_full shape, NaN rates per source bucket, future-month count

Usage:
    python -m Train.data_load_check
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import setup_logger, TEMP_DIR
from Train.data_loader import load_target_data
from Train.train_lightgbm_nfp import (
    build_training_dataset,
    _classify_columns_by_source,
)

logger = setup_logger(__file__, TEMP_DIR)


def _bucket_report(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    feat_cols = [c for c in X.columns if c != "ds"]
    groups = _classify_columns_by_source(feat_cols)
    report: Dict[str, Dict[str, float]] = {}
    for src, cols in groups.items():
        if not cols:
            continue
        sub = X[cols]
        nan_rate = float(sub.isna().to_numpy().mean())
        all_nan_cols = int(sub.isna().all().sum())
        zero_var_cols = int((sub.nunique(dropna=True) <= 1).sum())
        report[src] = {
            "n_features": len(cols),
            "nan_rate_overall": round(nan_rate, 4),
            "all_nan_cols": all_nan_cols,
            "zero_variance_cols": zero_var_cols,
        }
    return report


def _check_branch(target_type: str, release_type: str, target_source: str) -> None:
    label = f"{target_type}/{release_type}/{target_source}"
    logger.info("=" * 70)
    logger.info(f"BRANCH: {label}")
    logger.info("=" * 70)

    t0 = time.time()
    target_df = load_target_data(
        target_type, release_type=release_type, target_source=target_source
    )
    logger.info(
        f"Target: {len(target_df)} rows ({target_df['ds'].min().date()} → "
        f"{target_df['ds'].max().date()}); "
        f"y_mom NaN = {int(target_df['y_mom'].isna().sum())} (future months)"
    )

    X, y = build_training_dataset(
        target_df,
        target_type=target_type,
        release_type=release_type,
        target_source=target_source,
        show_progress=False,
    )
    elapsed = time.time() - t0

    n_future = int(y.isna().sum())
    n_train = int(y.notna().sum())
    feat_cols = [c for c in X.columns if c != "ds"]
    overall_nan = float(X[feat_cols].isna().to_numpy().mean()) if feat_cols else 0.0

    logger.info(
        f"X_full: {X.shape[0]} rows × {X.shape[1]} cols  "
        f"(features={len(feat_cols)}, train_rows={n_train}, future_rows={n_future})"
    )
    logger.info(
        f"Date span: {X['ds'].min().date()} → {X['ds'].max().date()}  "
        f"overall_NaN_rate={overall_nan:.3f}  build_time={elapsed:.1f}s"
    )

    bucket = _bucket_report(X)
    logger.info("Per-source coverage:")
    logger.info(
        f"  {'source':24s} {'n_features':>11s} {'nan_rate':>10s} "
        f"{'all_nan':>8s} {'zero_var':>9s}"
    )
    for src, stats in bucket.items():
        logger.info(
            f"  {src:24s} {stats['n_features']:>11d} "
            f"{stats['nan_rate_overall']:>10.3f} {stats['all_nan_cols']:>8d} "
            f"{stats['zero_variance_cols']:>9d}"
        )

    # Quick sanity: every backtest month should have at least some non-NaN
    # features beyond the calendar set.
    rows_all_nan = int(X[feat_cols].isna().all(axis=1).sum())
    if rows_all_nan:
        logger.warning(f"  WARNING: {rows_all_nan} rows have ALL feature columns NaN")
    else:
        logger.info("  All rows have at least one non-NaN feature.")


def main() -> None:
    branches = [
        ("nsa", "first", "revised"),
        ("sa", "first", "revised"),
    ]
    for target_type, release_type, target_source in branches:
        _check_branch(target_type, release_type, target_source)


if __name__ == "__main__":
    main()
