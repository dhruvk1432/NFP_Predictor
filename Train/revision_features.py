"""
Cross-Snapshot Revision Features for LightGBM NFP Model

Macroeconomic data is frequently revised months after initial publication.
This module computes aggregate revision statistics by comparing what two consecutive
snapshots say about the exactly identical historical period. This captures the magnitude 
and direction of data revisions, late releases, and benchmark adjustments.

For predicting target month M's NFP value:
- We load the data snapshot published at the end of M, and the snapshot published at M-1.
- We apply a strict cutoff (e.g. the NFP release date) to both to prevent looking into the future.
- Diff = The revisions released to the public *between* M-1 and M.

By design, this has NO look-ahead bias: both snapshots are pre-filtered at ETL time
(release_date < snapshot_date), and the pivot strictly enforces obs_date < cutoff.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.data_loader import (
    load_master_snapshot,
    pivot_snapshot_to_wide,
)

logger = setup_logger(__file__, TEMP_DIR)


def compute_revision_features(
    features_current: pd.DataFrame,
    features_prev: pd.DataFrame,
    prefix: str = 'rev',
    per_series: bool = False,
    per_series_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute revision statistics from two snapshot views of the same period.

    Args:
        features_current: Single-row DataFrame from pivot_snapshot_to_wide()
            applied to snapshot M with prev_month cutoff.
        features_prev: Single-row DataFrame from pivot_snapshot_to_wide()
            applied to snapshot M-1 with prev_month cutoff.
        prefix: Prefix for output feature names.
        per_series: If True, also emit per-series revision diffs.
        per_series_cols: If provided, only emit per-series revisions for these
            columns. If None and per_series=True, emit for all overlapping columns.

    Returns:
        Single-row DataFrame with 8 aggregate features, plus per-series
        revision columns if per_series=True.
    """
    if features_current.empty and features_prev.empty:
        return pd.DataFrame()

    current_cols = set(features_current.columns) if not features_current.empty else set()
    prev_cols = set(features_prev.columns) if not features_prev.empty else set()

    # Overlapping series for revision computation (vectorized subtraction)
    overlap = sorted(current_cols & prev_cols)

    if overlap:
        curr_vals = features_current[overlap].iloc[0]
        prev_vals = features_prev[overlap].iloc[0]
        diffs = curr_vals - prev_vals
        diffs = diffs.dropna()
        revisions = diffs.to_dict()
    else:
        revisions = {}

    rev_values = np.array(list(revisions.values())) if revisions else np.array([])

    # Aggregate statistics
    result = {}

    if len(rev_values) > 0:
        nonzero_revisions = rev_values[rev_values != 0]
        result[f'{prefix}_mean'] = float(np.mean(rev_values))
        result[f'{prefix}_abs_mean'] = float(np.mean(np.abs(rev_values)))
        result[f'{prefix}_positive_ratio'] = float(np.mean(rev_values > 0)) if len(nonzero_revisions) > 0 else 0.5
        result[f'{prefix}_count'] = int(len(nonzero_revisions))
        result[f'{prefix}_max_abs'] = float(np.max(np.abs(rev_values)))
        result[f'{prefix}_std'] = float(np.std(rev_values)) if len(rev_values) > 1 else 0.0
    else:
        result[f'{prefix}_mean'] = np.nan
        result[f'{prefix}_abs_mean'] = np.nan
        result[f'{prefix}_positive_ratio'] = np.nan
        result[f'{prefix}_count'] = 0
        result[f'{prefix}_max_abs'] = np.nan
        result[f'{prefix}_std'] = np.nan

    # New and dropped series counts
    result[f'{prefix}_n_new'] = int(len(current_cols - prev_cols))
    result[f'{prefix}_n_dropped'] = int(len(prev_cols - current_cols))

    # Per-series revision diffs (e.g., for FRED employment series)
    if per_series and revisions:
        target_cols = set(per_series_cols) if per_series_cols else set(revisions.keys())
        for col, rev_val in revisions.items():
            if col in target_cols:
                result[f'{col}_rev'] = rev_val

    return pd.DataFrame([result])


def get_revision_features_for_month(
    target_month: pd.Timestamp,
    prev_cutoff: Optional[pd.Timestamp] = None,
    target_type: str = 'nsa',
    target_source: str = 'first_release',
) -> pd.DataFrame:
    """
    Load two consecutive master snapshots and compute revision statistics.

    For a target prediction month M (e.g., May), this function compares the data world as it
    existed at the end of May vs the end of April, using *April* as the observation cutoff
    for both pivots. Any subsequent changes in the May snapshot for data points occurring
    in April or earlier represent "revisions".

    These revisions are aggregated into statistical features (mean magnitude, standard
    deviation, positivity ratio) which give the LightGBM model an understanding of whether
    macroeconomic data is broadly being revised upwards or downwards entering the release.

    Master snapshots already contain all data sources (FRED employment + exogenous) and
    are pre-filtered to only selected features, so no additional filtering is needed.

    Args:
        target_month: The month being predicted (YYYY-MM-01).
        prev_cutoff: Cutoff date for the previous month's data window.
            Typically the M-1 NFP release date. Defaults to prev_month.
        target_type: 'nsa' or 'sa' - determines which master snapshot variant to load.
        target_source: 'first_release' or 'revised' - determines which master snapshot variant.

    Returns:
        pd.DataFrame: A single-row DataFrame containing 8 aggregate revision features.
        Returns an empty DataFrame if the M-1 snapshot is missing.
    """
    prev_month = target_month - pd.DateOffset(months=1)
    if prev_cutoff is None:
        prev_cutoff = prev_month

    # Snapshot dates (month-end)
    snapshot_date_m = target_month + pd.offsets.MonthEnd(0)
    snapshot_date_prev = prev_month + pd.offsets.MonthEnd(0)

    # Load master snapshots (already contain all data sources, pre-filtered to selected features)
    master_m = load_master_snapshot(
        snapshot_date_m, target_type=target_type, target_source=target_source
    )
    master_prev = load_master_snapshot(
        snapshot_date_prev, target_type=target_type, target_source=target_source
    )

    if master_prev is None or master_prev.empty:
        logger.debug(f"No previous master snapshot for {prev_month.strftime('%Y-%m')}, skipping revisions")
        return pd.DataFrame()

    view_current = pivot_snapshot_to_wide(master_m, prev_month, cutoff_date=prev_cutoff) if master_m is not None else pd.DataFrame()
    view_prev = pivot_snapshot_to_wide(master_prev, prev_month, cutoff_date=prev_cutoff)

    revisions = compute_revision_features(view_current, view_prev, prefix='rev_master')
    return revisions if not revisions.empty else pd.DataFrame()
