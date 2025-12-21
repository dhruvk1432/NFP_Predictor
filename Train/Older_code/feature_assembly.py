"""
Feature Assembly for NBEATSx Training

Handles:
1. Long-to-wide format conversion with variable-length series
2. Splitting targets (leaf nodes) from exogenous features (parent nodes + external)
3. Categorizing features into futr_exog and hist_exog based on data availability
4. Scaling with RobustScaler (compatible with NBEATSx scaler_type='robust')
5. Static flags and calendar features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from sklearn.preprocessing import RobustScaler
import copy

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, setup_logger
from pipeline_helpers import build_hierarchy_structure, make_time_features, static_flags
from Train.snapshot_loader import load_snapshot_data

logger = setup_logger(__file__, TEMP_DIR)


def identify_futr_exog_features(
    exog_df: pd.DataFrame,
    snapshot_date: pd.Timestamp
) -> List[str]:
    """
    Identify which exog features can be used as future exog (known at forecast time).
    
    Per NBEATSx design:
    - futr_exog: Variables known/available for the forecast period
    - hist_exog: Historical values (includes PAST values of futr_exog + slow-release vars)
    
    Categories:
    1. **Fourier seasonality**: Always available (calendar-based)
    2. **Daily market data**: Oil, Credit, Yields - available by month end
    3. **Fast-release indicators**: ADP, Claims, ISM, Challenger - available within days
    4. **Slow-release (LAGGED in futr_exog)**: JOLTS, CB Confidence, IURSA, NOAA - use lag1 only
    
    Args:
        exog_df: Exogenous features DataFrame  
        snapshot_date: Current snapshot date
        
    Returns:
        List of feature names for futr_exog_list
    """
    # Fourier seasonality features (always generated fresh)
    fourier_features = ['sin', 'cos']
    
    # Daily market data (available by month end for forecast month)
    daily_market_patterns = [
        'Oil_Prices',       # Daily oil prices aggregated monthly
        'Credit_Spreads',   # Daily credit spreads aggregated monthly  
        'Yield_Curve'       # Daily yield curve aggregated monthly
    ]
    
    # Fast-release economic indicators (available within days of month end)
    # These come out BEFORE NFP so we can use them for forecast month
    fast_release_patterns = [
        'ADP_',             # ADP ~2 days before NFP (Tuesday)
        'ICSA',             # Weekly initial claims (every Thursday)
        'CCSA',             # Weekly continuing claims (every Thursday)
        'ISM',              # ISM released 1st business day of month
        'Challenger'        # Challenger job cuts (available early)
    ]
    
    # Slow-release data (comes AFTER NFP or with significant lag)
    # These should NEVER be in futr_exog, not even lag1!
    # Example: JOLTS has ~6 week release lag, so JOLTS for month M-1 
    # is not available when forecasting month M+1 at end of month M
    # These go to hist_exog ONLY for learning historical patterns
    slow_release_patterns = [
        'JOLTS',            # JOLTS released ~6 weeks after reference month
        'CB_Consumer_Confidence',  # CB Confidence mid-month but for reference period
        'IURSA',            # Insured unemployment (state data, lagged)
        'deaths_',          # NOAA storm deaths (slow aggregation)
        'injuries_',        # NOAA storm injuries (slow aggregation)
        'storm_count',      # NOAA storm counts (slow aggregation)
        'crop_damage',      # NOAA crop damage (slow aggregation)  
        'property_damage'   # NOAA property damage (slow aggregation)
    ]
    
    futr_series = []
    forecast_month_start = snapshot_date.replace(day=1)
    
    logger.info(f"Categorizing features for forecast month: {forecast_month_start.date()}")
    
    for series_name in exog_df['series_name'].unique():
        # Always include Fourier terms
        if any(f in series_name for f in fourier_features):
            futr_series.append(series_name)
            continue
        
        # Daily market data - check availability for forecast month
        if any(pattern in series_name for pattern in daily_market_patterns):
            # Only include if NOT a lag3+ or rolling_mean_12+
            # (those are historical features)
            if '_lag3' in series_name or '_lag6' in series_name or '_lag12' in series_name \
               or '_lag18' in series_name or '_lag24' in series_name:
                continue  # Skip long lags, use in hist_exog only
            if '_rolling_mean_12' in series_name or '_rolling_mean_24' in series_name:
                continue  # Skip long rolling means, use in hist_exog only
                
            series_data = exog_df[exog_df['series_name'] == series_name]
            max_date = series_data['date'].max()
            
            if pd.notna(max_date) and max_date >= forecast_month_start:
                futr_series.append(series_name)
            continue
        
        # Fast-release indicators - check availability
        if any(pattern in series_name for pattern in fast_release_patterns):
            # Exclude long lags (3+) and long rolling means (6+)
            if '_lag3' in series_name or '_lag6' in series_name:
                continue
            if '_rolling_mean_6' in series_name or '_rolling_mean_12' in series_name \
               or '_rolling_mean_24' in series_name:
                continue
                
            series_data = exog_df[exog_df['series_name'] == series_name]
            max_date = series_data['date'].max()
            
            if pd.notna(max_date) and max_date >= forecast_month_start:
                futr_series.append(series_name)
            continue
        
        # Slow-release indicators - SKIP ENTIRELY for futr_exog
        # These are NOT available at forecast time, not even lag1
        # They go to hist_exog only for learning patterns
        if any(pattern in series_name for pattern in slow_release_patterns):
            # Skip all slow-release features for futr_exog
            # They will be included in hist_exog automatically
            continue
    
    logger.info(f"Identified {len(futr_series)} future exogenous features")
    
    # Debug: categorize what we found
    fourier_count = sum(1 for s in futr_series if any(f in s for f in fourier_features))
    market_count = sum(1 for s in futr_series if any(p in s for p in daily_market_patterns))
    fast_count = sum(1 for s in futr_series if any(p in s for p in fast_release_patterns))
    
    logger.info(f"  Fourier: {fourier_count}, Daily Market: {market_count}, "
                f"Fast-release: {fast_count}")
    
    return futr_series


def long_to_wide_with_padding(
    df_long: pd.DataFrame,
    id_col: str = 'series_name',
    date_col: str = 'date',
    value_col: str = 'value'
) -> pd.DataFrame:
    """
    Convert long format to wide, handling variable-length series with forward fill.
    
    Args:
        df_long: Long format DataFrame
        id_col: Column name for series identifier
        date_col: Column name for date
        value_col: Column name for values
        
    Returns:
        Wide format DataFrame with date index and series as columns
        
    Notes:
        - Series starting late will have NaN for early dates
        - We forward-fill these NaNs to handle missing early history
        - This is acceptable for exogenous features (better than zeros)
    """
    wide = df_long.pivot_table(
        index=date_col,
        columns=id_col,
        values=value_col,
        aggfunc='first'  # Handle any duplicates
    )
    
    # Forward fill to handle series that start late
    # This carries forward the first available value backwards
    wide = wide.fillna(method='bfill').fillna(method='ffill')
    
    # Any remaining NaNs (series with no data) → 0
    wide = wide.fillna(0.0)
    
    return wide.reset_index()


def prepare_nbeats_training_data(
    snapshot_date: pd.Timestamp,
    lookback_months: int = 120,
    include_parent_features: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Prepare NBEATSx training dataset for a single snapshot.
    
    This is the main data preparation function that:
    1. Loads snapshot data
    2. Separates leaf nodes (targets) from parent nodes (features)
    3. Engineers parent employment features
    4. Merges external exogenous features
    5. Adds static flags and Fourier seasonality
    6. Categorizes features into hist/futr/static
    
    Args:
        snapshot_date: Month-end timestamp for snapshot
        lookback_months: Number of months of history to include
        include_parent_features: Whether to use parent nodes as exogenous
        
    Returns:
        Tuple of:
            - train_df: Wide format [unique_id, ds, y, ...features]
            - feature_lists: Dict with 'hist_exog_list', 'futr_exog_list', 'stat_exog_list'
    """
    from pipeline_helpers import DELIM
    
    logger.info(f"Preparing training data for snapshot {snapshot_date.date()}")
    
    # 1. Load snapshot data
    data = load_snapshot_data(snapshot_date)
    endo_df = data['endogenous'].copy()
    exog_df = data['exogenous'].copy()
    
    # Rename columns for consistency
    endo_df = endo_df.rename(columns={'series_name': 'unique_id'})
    
    # 2. Identify hierarchy
    all_series = endo_df['unique_id'].unique()
    hierarchy, ordered, bottom_series = build_hierarchy_structure(
        series_list=all_series,
        include_nsa=True
    )
    
    # Filter to NSA bottom series only
    nsa_bottom = [s for s in bottom_series if s.endswith('_nsa')]
    logger.info(f"Identified {len(nsa_bottom)} NSA leaf nodes as targets")
    
    # 3. Prepare targets (leaf nodes) in long format
    target_df = endo_df[endo_df['unique_id'].isin(nsa_bottom)].copy()
    target_df = target_df[['date', 'unique_id', 'value']].rename(columns={'value': 'y', 'date': 'ds'})
    target_df['ds'] = pd.to_datetime(target_df['ds'])
    
    # Apply lookback window
    cutoff_date = snapshot_date - pd.DateOffset(months=lookback_months)
    target_df = target_df[target_df['ds'] >= cutoff_date].copy()
    
    logger.info(f"Target data: {len(target_df)} rows, {target_df['unique_id'].nunique()} series")
    
    # 4. Engineer parent employment features (if enabled)
    parent_features_wide = None
    if include_parent_features:
        parent_series = [s for s in all_series if s.endswith('_nsa') and s not in nsa_bottom]
        logger.info(f"Engineering features from {len(parent_series)} parent employment series")
        
        parent_df = endo_df[endo_df['unique_id'].isin(parent_series)].copy()
        parent_df = parent_df[['date', 'unique_id', 'value']].rename(columns={'value': 'y', 'date': 'ds'})
        parent_df['ds'] = pd.to_datetime(parent_df['ds'])
        
        # Apply time features to each parent series + additional smoothness indicators
        parent_df = parent_df[parent_df['ds'] >= cutoff_date].copy()
        
        # Group by unique_id and apply comprehensive feature engineering
        parent_dfs = []
        for uid, group in parent_df.groupby('unique_id'):
            group = group.sort_values('ds').copy()
            
            # 1. Base time features (lags, rolling means)
            group_featured = make_time_features(group)
            
            # 2. YoY changes (smooths seasonal noise)
            group_featured['yoy_change'] = group_featured['y'] - group_featured['y'].shift(12)
            group_featured['yoy_pct'] = (group_featured['y'] / group_featured['y'].shift(12) - 1) * 100
            
            # 3. Trend indicators (6-month linear slope)
            def calc_trend(series, window=6):
                if len(series) < window:
                    return np.nan
                x = np.arange(window)
                y = series.values[-window:]
                if np.all(np.isnan(y)):
                    return np.nan
                slope = np.polyfit(x, y, 1)[0] if not np.any(np.isnan(y)) else np.nan
                return slope
            
            group_featured['trend_6m'] = group_featured['y'].rolling(6).apply(calc_trend, raw=False)
            
            # 4. Volatility (6-month std of MoM changes)
            group_featured['volatility_6m'] = group_featured['y'].diff().rolling(6).std()
            
            parent_dfs.append(group_featured)
        
        parent_featured = pd.concat(parent_dfs, ignore_index=True)
        
        # Convert to wide format for merging
        # CRITICAL: Only use lagged/rolling/derived features, NOT raw 'y' value!
        # Including 'y' would leak parent information (Parent = Sum(Children))
        feature_cols = [c for c in parent_featured.columns if c not in ['unique_id', 'ds', 'y']]
        
        # Create one wide dataframe per feature type
        parent_wide_list = []
        for col in feature_cols:
            if col in parent_featured.columns:
                temp = parent_featured.pivot_table(
                    index='ds',
                    columns='unique_id',
                    values=col,
                    aggfunc='first'
                )
                # Rename columns to include feature name
                temp.columns = [f"{uid}_{col}" for uid in temp.columns]
                parent_wide_list.append(temp)
        
        if parent_wide_list:
            parent_features_wide = pd.concat(parent_wide_list, axis=1).reset_index()
            logger.info(f"Parent features: {parent_features_wide.shape[1]-1} columns (including YoY, trends, volatility)")
    
    # 5. Prepare external exogenous features
    exog_df['date'] = pd.to_datetime(exog_df['date'])
    exog_df = exog_df[exog_df['date'] >= cutoff_date].copy()
    
    # Identify futr vs hist exog
    futr_series = identify_futr_exog_features(exog_df, snapshot_date)
    
    # Convert to wide
    exog_wide = long_to_wide_with_padding(exog_df, id_col='series_name', date_col='date', value_col='value')
    exog_wide = exog_wide.rename(columns={'date': 'ds'})
    
    logger.info(f"External exog: {exog_wide.shape[1]-1} features")
    
    # 6. Add Fourier seasonality features (calendar-based, always futr_exog)
    import pytimetk as tk
    
    # Create base df with dates
    date_range = pd.date_range(start=cutoff_date, end=snapshot_date, freq='MS')
    fourier_df = pd.DataFrame({'ds': date_range})
    fourier_df = tk.augment_fourier(fourier_df, date_column='ds', periods=12, max_order=3)
    
    # Extract Fourier columns
    fourier_cols = [c for c in fourier_df.columns if 'sin' in c or 'cos' in c]
    fourier_df = fourier_df[['ds'] + fourier_cols]
    
    logger.info(f"Added {len(fourier_cols)} Fourier features")
    
    # 7. Merge all features into target dataframe
    # Start with targets in long format
    train_df = target_df.copy()
    
    # Merge Fourier (calendar features)
    train_df = train_df.merge(fourier_df, on='ds', how='left')
    
    # Merge external exog
    train_df = train_df.merge(exog_wide, on='ds', how='left')
    
    # Merge parent features
    if parent_features_wide is not None:
        train_df = train_df.merge(parent_features_wide, on='ds', how='left')
    
    # 8. Add static flags
    static_df = pd.DataFrame([
        {**static_flags(uid), 'unique_id': uid}
        for uid in nsa_bottom
    ])
    
    train_df = train_df.merge(static_df, on='unique_id', how='left')
    
    # 9. Fill any remaining NaNs
    train_df = train_df.fillna(0.0)
    
    logger.info(f"Final training data: {train_df.shape}")
    
    # 10. Categorize features per NBEATSx design:
    # - stat_exog: Static/time-invariant features (sector flags, etc.)
    # - futr_exog: Features known at forecast time (Fourier, fast-release, market data)
    # - hist_exog: ALL historical features INCLUDING past values of futr_exog
    #              (parent features + ALL exog including slow-release)
    
    stat_exog_list = [c for c in static_df.columns if c != 'unique_id']
    
    # Futr exog: Fourier + fast-release + daily market (known at forecast time)
    futr_exog_list = fourier_cols + [c for c in exog_wide.columns if any(s in c for s in futr_series)]
    futr_exog_list = [c for c in futr_exog_list if c in train_df.columns and c != 'ds']
    
    # Hist exog: ALL exogenous features (not just non-futr ones!)
    # Per NBEATSx docs: hist_exog includes historical values of futr_exog + slow-release vars
    # This allows the model to learn from the PAST of fast-release indicators
    all_exog_cols = [c for c in exog_wide.columns if c != 'ds']
    parent_cols = [c for c in parent_features_wide.columns if c != 'ds'] if parent_features_wide is not None else []
    
    hist_exog_list = all_exog_cols + parent_cols
    hist_exog_list = [c for c in hist_exog_list if c in train_df.columns]
    
    feature_lists = {
        'hist_exog_list': hist_exog_list,
        'futr_exog_list': futr_exog_list,
        'stat_exog_list': stat_exog_list
    }
    
    logger.info(f"Feature categorization (NBEATSx design):")
    logger.info(f"  hist_exog: {len(hist_exog_list)} features (includes history of all exog + parents)")
    logger.info(f"  futr_exog: {len(futr_exog_list)} features (known at forecast time)")
    logger.info(f"  stat_exog: {len(stat_exog_list)} features (time-invariant)")
    
    return train_df, feature_lists


if __name__ == "__main__":
    # Test data preparation
    test_date = pd.Timestamp('2020-01-31')
    logger.info(f"Testing data preparation for {test_date.date()}")
    
    train_df, feature_lists = prepare_nbeats_training_data(
        snapshot_date=test_date,
        lookback_months=24  # Just 2 years for testing
    )
    
    logger.info(f"\nTraining data shape: {train_df.shape}")
    logger.info(f"Unique series: {train_df['unique_id'].nunique()}")
    logger.info(f"Date range: {train_df['ds'].min()} to {train_df['ds'].max()}")
    
    logger.info(f"\nSample features:")
    logger.info(f"  hist_exog (first 10): {feature_lists['hist_exog_list'][:10]}")
    logger.info(f"  futr_exog (first 10): {feature_lists['futr_exog_list'][:10]}")
    logger.info(f"  stat_exog: {feature_lists['stat_exog_list']}")
    
    logger.info(f"\nData sample:\n{train_df.head()}")
