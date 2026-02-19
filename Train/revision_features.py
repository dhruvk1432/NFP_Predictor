"""
Cross-Snapshot Revision Features for LightGBM NFP Model

Computes aggregate revision statistics by comparing what two consecutive
snapshots say about the same historical period. Captures data revisions,
late releases, and benchmark adjustments.

For target month M:
- Load snapshot M and snapshot M-1
- Apply prev_month cutoff to both (same historical window)
- Diff = revisions released between M-1 and M NFP dates

No look-ahead bias: both snapshots are pre-filtered at ETL time
(release_date < snapshot_date), and pivot applies obs_date < cutoff.
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
    load_fred_snapshot,
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


def _filter_to_selected(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Filter a single-row DataFrame to only columns matching selected features."""
    if df.empty:
        return df
    keep = [c for c in df.columns if c in set(selected_features)]
    return df[keep] if keep else pd.DataFrame()


def get_revision_features_for_month(
    target_month: pd.Timestamp,
    prev_cutoff: Optional[pd.Timestamp] = None,
    selected_features: Optional[List[str]] = None,
    include_fred: bool = True,
) -> pd.DataFrame:
    """
    Load two consecutive snapshots and compute revision features for
    ONLY the pre-selected series.

    For target month M, compares snapshot M vs snapshot M-1 using
    prev_month as the observation cutoff for both pivots. Pivoted
    DataFrames are filtered to selected_features before computing
    revisions, so only curated series are touched.

    Args:
        target_month: The month being predicted (YYYY-MM-01).
        prev_cutoff: Cutoff date for the previous month's data window.
            Typically the M-1 NFP release date. Defaults to prev_month.
        selected_features: Pre-selected feature names. Both aggregates and
            per-series revisions are computed only over these columns.
            If None, uses all columns (not recommended).
        include_fred: Whether to also compute FRED revision features.

    Returns:
        Single-row DataFrame with 16 aggregate revision features
        (8 master + 8 FRED) plus per-series FRED employment revision diffs
        (one {series}_rev column per selected FRED series).
        Empty DataFrame if M-1 snapshot unavailable.
    """
    prev_month = target_month - pd.DateOffset(months=1)
    if prev_cutoff is None:
        prev_cutoff = prev_month

    # Snapshot dates (month-end)
    snapshot_date_m = target_month + pd.offsets.MonthEnd(0)
    snapshot_date_prev = prev_month + pd.offsets.MonthEnd(0)

    all_revision_features = pd.DataFrame()

    # --- Master snapshot revisions (aggregates only, selected features only) ---
    master_m = load_master_snapshot(snapshot_date_m)
    master_prev = load_master_snapshot(snapshot_date_prev)

    if master_prev is not None and not master_prev.empty:
        view_current = pivot_snapshot_to_wide(master_m, prev_month, cutoff_date=prev_cutoff) if master_m is not None else pd.DataFrame()
        view_prev = pivot_snapshot_to_wide(master_prev, prev_month, cutoff_date=prev_cutoff)

        # Filter to only selected features before computing revisions
        if selected_features is not None:
            view_current = _filter_to_selected(view_current, selected_features)
            view_prev = _filter_to_selected(view_prev, selected_features)

        master_revisions = compute_revision_features(view_current, view_prev, prefix='rev_master')
        if not master_revisions.empty:
            all_revision_features = master_revisions
    else:
        logger.debug(f"No previous master snapshot for {prev_month.strftime('%Y-%m')}, skipping master revisions")

    # --- FRED employment revisions (aggregates + per-series, selected features only) ---
    if include_fred:
        fred_m = load_fred_snapshot(snapshot_date_m)
        fred_prev = load_fred_snapshot(snapshot_date_prev)

        if fred_prev is not None and not fred_prev.empty:
            fred_view_current = pivot_snapshot_to_wide(fred_m, prev_month, cutoff_date=prev_cutoff) if fred_m is not None else pd.DataFrame()
            fred_view_prev = pivot_snapshot_to_wide(fred_prev, prev_month, cutoff_date=prev_cutoff)

            # Filter to only selected features before computing revisions
            if selected_features is not None:
                fred_view_current = _filter_to_selected(fred_view_current, selected_features)
                fred_view_prev = _filter_to_selected(fred_view_prev, selected_features)

            # Per-series revisions for selected FRED employment series
            fred_revisions = compute_revision_features(
                fred_view_current, fred_view_prev,
                prefix='rev_fred', per_series=True,
            )
            if not fred_revisions.empty:
                if all_revision_features.empty:
                    all_revision_features = fred_revisions
                else:
                    for col in fred_revisions.columns:
                        all_revision_features[col] = fred_revisions[col].iloc[0]
        else:
            logger.debug(f"No previous FRED snapshot for {prev_month.strftime('%Y-%m')}, skipping FRED revisions")

    return all_revision_features
