"""
Pipeline Helper Functions for NBEATSx NFP Forecasting

This module contains adapted versions of key functions from prediction_pipeline.py,
modified to work with the new data structure:
- Employment data from data/fred_data/ snapshots
- Exogenous data from features.py (master_snapshots, NOAA, ADP)

Functions adapted from prediction_pipeline.py:
- make_time_features(): Create lags, rolling stats, Fourier features
- build_hierarchy_structure(): Build parent-child hierarchy relationships
- static_flags(): Create binary industry/sector flags
- add_derived_columns(): Create derived/residual series

All functions maintain the same logic as the originals but work with the new
data format where series_name is used instead of unique_id (with underscores).
"""

import pandas as pd
import numpy as np
import pytimetk as tk
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import logging

from settings import TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# Delimiter for hierarchical series names
DELIM = "." 


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create unified time-based features for a series table.
    
    Builds lagged values, rolling-window statistics, Fourier seasonality features,
    and one-hot encodings of categorical columns. Also prunes highly-correlated
    numeric columns to reduce redundancy.
    
    Adapted from prediction_pipeline.py to work with new data structure.
    
    Args:
        df: DataFrame with columns ['unique_id', 'ds', 'y', ...]. May include
            additional categorical columns to be one-hot encoded.
    
    Returns:
        A new DataFrame with augmented features and no missing values
        (rows with NA after augmentation are dropped).
    
    Notes:
        - Uses pytimetk for lags/rolling/Fourier augmentation
        - Ensures unique_id remains available for merging
        - Handles both single and multi-series DataFrames
    """
    if len(df) == 0:
        return df 
    
    out = df.copy()
    
    # Create lag features
    feat_df = tk.augment_lags(
        data=out, 
        date_column="ds", 
        value_column="y", 
        lags=[1, 2, 3]
    )
    
    # Create rolling statistics
    feat_df = tk.augment_rolling(
        data=feat_df,
        date_column="ds",
        value_column="y_lag_1",
        window=[3, 6, 9, 12],
        window_func=[("mean", lambda x: x.mean()), ("std", lambda x: x.std())]
    )
    
    # Create Fourier seasonal features
    feat_df = tk.augment_fourier(
        data=feat_df, 
        date_column="ds", 
        periods=12, 
        max_order=3
    )

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_cols = feat_df.select_dtypes(include=["object", "category"]).columns.to_list()
    if "unique_id" in cat_cols:
        cat_cols.remove("unique_id")
    
    if cat_cols:
        encoded = encoder.fit_transform(feat_df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(cat_cols), 
            index=feat_df.index
        )
        feat_df_encoded = pd.concat([feat_df.drop(columns=cat_cols), encoded_df], axis=1)
    else:
        feat_df_encoded = feat_df
    
    # Drop highly correlated features to reduce redundancy
    def drop_highly_correlated(df, threshold=0.8, exclude_cols=["ds", "y", "unique_id"]):
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        keep_cols = set(exclude_cols)
        corr_cols = [col for col in numeric_cols if col not in keep_cols]
        
        if len(corr_cols) < 2:
            return df
            
        corr_matrix = df[corr_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)

    feat_df_encoded = drop_highly_correlated(feat_df_encoded)
    
    return feat_df_encoded.dropna()


def static_flags(series_name: str) -> Dict[str, int]:
    """Derive coarse-grained static binary indicators from a hierarchical series_name.
    
    Adapted from prediction_pipeline.py to work with underscore-separated series
    names from data/fred_data/ instead of dot-separated unique_ids.
    
    Args:
        series_name: Hierarchical series identifier using underscore notation,
                    e.g. "private_goods_manufacturing_nsa" or 
                    "government_federal" or "private_services_trade_transportation_utilities_nsa"
    
    Returns:
        Dict of 0/1 integer flags covering ownership (private/government),
        super-sector (goods/services), and major industry bubuckets.
    
    Notes:
        - These flags are used as static exogenous variables for NBEATSx
        - Handles both SA and NSA series (strips _nsa suffix first)
    
    Examples:
        >>> static_flags("private_goods_manufacturing_nsa")
        {'flag_private': 1, 'flag_government': 0, 'flag_goods': 1, ...}
        
        >>> static_flags("government_federal")
        {'flag_private': 0, 'flag_government': 1, ...}
    """
    # Remove _nsa suffix if present for consistent parsing
    clean_name = series_name.replace("_nsa", "").replace("_sa", "")
    parts = clean_name.split(DELIM)
    
    # Determine ownership level by checking first parts
    if parts[0] == "government":
        own = "government"
        kind = "government_sector"
        branch = "government"
    elif parts[0] == "private":
        own = "private"
        # Determine if goods or services
        kind = parts[1] if len(parts) > 1 else "unknown"
        # Determine specific industry branch
        # For complex names like trade_transportation_utilities, we need to look for the full segment
        if len(parts) > 2:
            # Join remaining parts in case of multi-word industries
            branch = "_".join(parts[2:])
        else:
            branch = "unknown"
    else:
        # Handle 'total' or other top-level series
        own = "total"
        kind = "total"
        branch = "total"
    
    return {
        # Ownership flags
        "flag_private": int(own == "private"),
        "flag_government": int(own == "government"),
        
        # Super-sector flags
        "flag_goods": int(kind == "goods"),
        "flag_services": int(kind == "services"),
        
        # Major industry flags (choose ONE for private, fallback to gov)
        "ind_mining": int(branch == "mining_logging"),
        "ind_construction": int(branch == "construction"),
        "ind_mfg": int(branch == "manufacturing"),
        "ind_trade_trans": int(branch == "trade_transportation_utilities"),
        "ind_info": int(branch == "information"),
        "ind_financial": int(branch == "financial"),
        "ind_prof_bus": int(branch == "professional_business"),
        "ind_ed_health": int(branch == "education_health"),
        "ind_leisure": int(branch == "leisure_hospitality"),
        "ind_other_srv": int(branch == "other"),
        "ind_government": int(own == "government"),
    }


def build_hierarchy_structure(
    series_list: List[str],
    include_nsa: bool = True
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Construct parent→children adjacency and identify bottom series using
    dot-separated hierarchical ids from FRED_EMPLOYMENT_CODES.

    Example ids:
        'total'
        'total.private'
        'total.private.goods'
        'total.private.goods_nsa'
        'total.private.goods.mining_logging_nsa'
    """

    # 1. Filter to NSA or SA
    if include_nsa:
        filtered = [s for s in series_list if s.endswith("_nsa")]
    else:
        filtered = [s for s in series_list if not s.endswith("_nsa")]

    series_set = set(filtered)

    def strip_suffix(s: str) -> str:
        if s.endswith("_nsa"):
            return s[:-4]
        if s.endswith("_sa"):
            return s[:-3]
        return s

    def add_suffix(base: str, suffix_from_child: str) -> str:
        return base + suffix_from_child if suffix_from_child else base

    hierarchy: Dict[str, List[str]] = {}

    # 2. Build parent→children dict using dot prefixes
    for s in filtered:
        base = strip_suffix(s)
        # Top-level nodes like 'total_nsa' or 'total' have no '.'
        if "." not in base:
            continue

        parent_base = base.rsplit(".", 1)[0]
        suffix = "_nsa" if s.endswith("_nsa") else "_sa" if s.endswith("_sa") else ""
        parent = add_suffix(parent_base, suffix)

        if parent in series_set:
            hierarchy.setdefault(parent, []).append(s)

    # 3. Determine roots and bottom series
    child_set = {child for children in hierarchy.values() for child in children}
    roots = [s for s in series_set if s not in child_set]

    ordered: List[str] = []

    def dfs(node: str) -> None:
        if node in ordered:
            return
        ordered.append(node)
        for child in sorted(hierarchy.get(node, [])):
            dfs(child)

    for r in sorted(roots):
        dfs(r)

    # Add any disconnected series (should be rare)
    for s in sorted(series_set):
        if s not in ordered:
            dfs(s)

    bottom_series = [s for s in filtered if s not in hierarchy]

    return hierarchy, ordered, bottom_series


def _strip_suffix(series_name: str) -> tuple[str, str]:
    """
    Split a series_name into (base, suffix), where suffix is '_nsa' or '_sa' if present.
    Example:
        'total.private.goods.mining_logging_nsa' -> ('total.private.goods.mining_logging', '_nsa')
        'total_sa'                               -> ('total', '_sa')
        'total.private'                          -> ('total.private', '')
    """
    if series_name.endswith("_nsa"):
        return series_name[:-4], "_nsa"
    if series_name.endswith("_sa"):
        return series_name[:-3], "_sa"
    return series_name, ""


def _parent_name_dot(series_name: str) -> Optional[str]:
    """
    Given a dot-hierarchical id with optional '_nsa'/'_sa' suffix, return its parent name.

    Examples:
        'total.private.goods.mining_logging_nsa' -> 'total.private.goods_nsa'
        'total.private.goods_nsa'                -> 'total.private_nsa'
        'total.private_nsa'                      -> 'total_nsa'
        'total_nsa'                              -> None  (no parent with dot)
    """
    base, suffix = _strip_suffix(series_name)

    # No dot = top-level, no parent
    if "." not in base:
        return None

    parent_base = base.rsplit(".", 1)[0]
    return parent_base + suffix


def add_derived_columns(
    df: pd.DataFrame, 
    series_list: List[str],
    hierarchy: Dict[str, List[str]]
) -> pd.DataFrame:
    """Create derived series for nodes without direct data via residual calculation.
    
    For series that don't have direct data but are in the hierarchy, calculates
    their values as:
        derived_value = parent_value - sum(observed_sibling_values)
    
    Args:
        df: Long-format DataFrame with columns ['series_name', 'ds', 'value']
        series_list: List of all series names that should exist (some may need derivation)
        hierarchy: Dict of parent -> [children] relationships (using dot-based ids)
    
    Returns:
        Augmented DataFrame containing both original and derived series
    """
    available_series = set(df['series_name'].unique())
    missing_series = [s for s in series_list if s not in available_series]
    
    if not missing_series:
        logger.info("All series have data, no derivation needed")
        return df
    
    logger.info(f"Need to derive {len(missing_series)} series")
    
    for series_name in missing_series:
        # --- NEW LOGIC: find parent using dot-hierarchy + suffix ---
        parent_name = _parent_name_dot(series_name)
        if parent_name is None:
            logger.warning(f"Cannot derive {series_name}: no parent (top-level)")
            continue
        
        if parent_name not in available_series:
            logger.warning(f"Cannot derive {series_name}: parent {parent_name} not available")
            continue
        # --- END NEW LOGIC ---
        
        # Get all children of this parent
        siblings = hierarchy.get(parent_name, [])
        observed_siblings = [s for s in siblings if s in available_series and s != series_name]
        
        if not observed_siblings:
            logger.warning(f"Cannot derive {series_name}: no observed siblings")
            continue
        
        logger.info(f"Deriving {series_name} from parent {parent_name} minus {len(observed_siblings)} siblings")
        
        # Calculate residual: parent - sum(observed siblings)
        parent_data = df[df['series_name'] == parent_name].set_index('ds')['value']
        sibling_data = df[df['series_name'].isin(observed_siblings)]
        sibling_sum = sibling_data.pivot_table(
            index='ds', 
            columns='series_name', 
            values='value'
        ).sum(axis=1)
        
        # Align dates
        common_dates = parent_data.index.intersection(sibling_sum.index)
        if len(common_dates) == 0:
            logger.warning(f"No common dates for deriving {series_name}")
            continue
        
        derived_values = parent_data.loc[common_dates] - sibling_sum.loc[common_dates]
        
        # Create derived series dataframe
        derived_df = pd.DataFrame({
            'ds': common_dates,
            'series_name': series_name,
            'value': derived_values.values
        })
        
        # Check for negative values
        if (derived_values < 0).any():
            logger.warning(f"Derived series {series_name} has {(derived_values < 0).sum()} negative values")
        
        # Append to main dataframe
        df = pd.concat([df, derived_df], ignore_index=True)
        available_series.add(series_name)
        
        logger.info(f"Successfully derived {series_name}: {len(derived_df)} observations")
    
    return df.sort_values(['series_name', 'ds']).reset_index(drop=True)



if __name__ == "__main__":
    # Run basic tests
    logger.info("Testing pipeline_helpers.py functions...")
    
    # Test 1: make_time_features
    logger.info("\n=== Test 1: make_time_features ===")
    test_df = pd.DataFrame({
        'unique_id': ['series1'] * 50,
        'ds': pd.date_range('2020-01-01', periods=50, freq='MS'),
        'y': np.random.randn(50).cumsum() + 100
    })
    
    result = make_time_features(test_df)
    logger.info(f"Input shape: {test_df.shape}, Output shape: {result.shape}")
    logger.info(f"Created {len(result.columns) - 3} new features")
    logger.info(f"Sample features: {[c for c in result.columns if 'lag' in c or 'roll' in c or 'sin' in c][:10]}")
    
    # Test 2: static_flags
    logger.info("\n=== Test 2: static_flags ===")
    test_series = [
        "private_goods_manufacturing_nsa",
        "private_services_trade_transportation_utilities_nsa",
        "government_federal_nsa",
        "total_nsa"
    ]
    
    for series in test_series:
        flags = static_flags(series)
        logger.info(f"{series}: private={flags['flag_private']}, goods={flags['flag_goods']}, "
                   f"mfg={flags['ind_mfg']}, govt={flags['flag_government']}")
    
    # Test 3: build_hierarchy_structure
    logger.info("\n=== Test 3: build_hierarchy_structure ===")
    test_hierarchy_series = [
        'total_nsa',
        'private_nsa',
        'government_nsa',
        'private_goods_nsa',
        'private_services_nsa',
        'private_goods_manufacturing_nsa',
        'private_goods_construction_nsa'
    ]
    
    hierarchy, ordered, bottom = build_hierarchy_structure(test_hierarchy_series, include_nsa=True)
    logger.info(f"Hierarchy structure: {hierarchy}")
    logger.info(f"Ordered series: {ordered}")
    logger.info(f"Bottom series: {bottom}")
    
    logger.info("\nAll basic tests completed!")
