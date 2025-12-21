"""
NBEATSx Prediction Module

Generates forecasts from trained NBEATSx model for all NSA leaf series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger
from Train.feature_assembly import prepare_nbeats_training_data
from Train.train_nbeats import scale_features

logger = setup_logger(__file__, TEMP_DIR)

try:
    from neuralforecast import NeuralForecast
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    logger.warning("NeuralForecast not available")
    NEURALFORECAST_AVAILABLE = False


def generate_nbeats_predictions(
    model: NeuralForecast,
    snapshot_date: pd.Timestamp,
    lookback_months: int = 120,
    scalers: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate predictions from trained NBEATSx model.
    
    Args:
        model: Trained NeuralForecast model
        snapshot_date: Current snapshot date
        lookback_months: Months of history for prediction
        scalers: Dict of scalers used in training
        
    Returns:
        DataFrame with predictions [unique_id, ds, NBEATSx]
    """
    logger.info(f"Generating NBEATSx predictions for {snapshot_date.date()}")
    
    # 1. Prepare data (same as training but for prediction)
    train_df, feature_lists = prepare_nbeats_training_data(
        snapshot_date=snapshot_date,
        lookback_months=lookback_months
    )
    
    # 2. Scale features (using same scalers as training)
    if scalers:
        if 'hist_exog' in scalers and feature_lists['hist_exog_list']:
            hist_cols = [c for c in feature_lists['hist_exog_list'] if c in train_df.columns]
            train_df[hist_cols] = scalers['hist_exog'].transform(train_df[hist_cols])
        
        if 'futr_exog' in scalers and feature_lists['futr_exog_list']:
            futr_cols = [c for c in feature_lists['futr_exog_list'] if c in train_df.columns]
            train_df[futr_cols] = scalers['futr_exog'].transform(train_df[futr_cols])
    
    # 3. Prepare future dataframe (current month relative to snapshot)
    # If snapshot is Feb 28, we predict Feb data (h=1 from Jan data)
    next_month = snapshot_date
    next_month_start = next_month.replace(day=1)
    
    # Create future df with futr_exog values for next month
    unique_ids = train_df['unique_id'].unique()
    
    # Get last observation for each series (to continue from)
    last_obs = train_df.groupby('unique_id').tail(1)[['unique_id', 'ds']].copy()
    
    # Build future exog dataframe
    import pytimetk as tk
    futr_df = pd.DataFrame({
        'unique_id': unique_ids,
        'ds': next_month_start
    })
    
    # Add Fourier features for next month (manual calculation to avoid pytimetk single-row error)
    # periods=12, max_order=3
    month_idx = next_month_start.month - 1  # 0-11
    
    for k in range(1, 4):  # orders 1, 2, 3
        futr_df[f'ds_sin_{k}_12'] = np.sin(2 * np.pi * k * (month_idx + 1) / 12)
        futr_df[f'ds_cos_{k}_12'] = np.cos(2 * np.pi * k * (month_idx + 1) / 12)
    
    fourier_cols = [f'ds_sin_{k}_12' for k in range(1, 4)] + [f'ds_cos_{k}_12' for k in range(1, 4)]
    
    # Add other futr_exog features (assume same as last month for now)
    # In production, you'd load actual values for next month
    other_futr = [c for c in feature_lists['futr_exog_list'] if c not in fourier_cols]
    for col in other_futr:
        if col in train_df.columns:
            # Use last available value per series
            last_values = train_df.groupby('unique_id')[col].last()
            futr_df[col] = futr_df['unique_id'].map(last_values)
            
    # Scale future features if needed
    if scalers and 'futr_exog' in scalers:
        futr_cols_to_scale = [c for c in feature_lists['futr_exog_list'] if c in futr_df.columns]
        if futr_cols_to_scale:
            futr_df[futr_cols_to_scale] = scalers['futr_exog'].transform(futr_df[futr_cols_to_scale])
    
    # 4. Generate predictions
    logger.info(f"Predicting for {len(unique_ids)} series")
    
    # Extract static features for prediction
    static_cols = ['unique_id'] + feature_lists['stat_exog_list']
    static_df = train_df[static_cols].groupby('unique_id').first().reset_index()
    
    # Debug: Check IDs using get_missing_future
    try:
        missing_combinations = model.get_missing_future(futr_df)
        if not missing_combinations.empty:
            logger.error(f"Missing combinations in futr_df: {len(missing_combinations)} rows")
            logger.error(f"Sample missing: \n{missing_combinations.head()}")
            
            # Fix: Append missing combinations to futr_df
            # We need to fill features for these rows
            # Use the first row of futr_df as a template for features
            template_row = futr_df.iloc[0].copy()
            
            new_rows = []
            for _, row in missing_combinations.iterrows():
                new_row = template_row.copy()
                new_row['unique_id'] = row['unique_id']
                new_row['ds'] = row['ds']
                new_rows.append(new_row)
            
            if new_rows:
                futr_df = pd.concat([futr_df, pd.DataFrame(new_rows)], ignore_index=True)
                logger.info(f"Added {len(new_rows)} missing rows to futr_df")
                
    except Exception as e:
        logger.warning(f"Could not check missing future: {e}")

    predictions = model.predict(futr_df=futr_df)
    
    # 5. Format predictions
    predictions = predictions.reset_index()
    predictions = predictions.rename(columns={'NBEATSx': 'forecast'})
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions


def aggregate_to_total_nsa(
    predictions: pd.DataFrame,
    last_actuals: pd.DataFrame
) -> Tuple[float, float, pd.DataFrame]:
    """
    Aggregate bottom-up NSA predictions to total NSA.
    
    Args:
        predictions: NBEATSx predictions [unique_id, ds, forecast]
        last_actuals: Last month actuals [unique_id, ds, y]
        
    Returns:
        Tuple of (total_nsa_forecast, mom_change, series_contributions)
    """
    # Sum all leaf predictions
    total_nsa_forecast = predictions['forecast'].sum()
    
    # Get last month total
    last_month_total = last_actuals['y'].sum()
    
    # Calculate MoM change
    mom_change = total_nsa_forecast - last_month_total
    
    # Calculate each series contribution to change
    series_contributions = predictions.copy()
    series_contributions = series_contributions.merge(
        last_actuals[['unique_id', 'y']],
        on='unique_id',
        how='left'
    )
    series_contributions['contribution_to_change'] = (
        series_contributions['forecast'] - series_contributions['y']
    )
    
    logger.info(f"Total NSA forecast: {total_nsa_forecast:,.0f}")
    logger.info(f"Last month NSA: {last_month_total:,.0f}")
    logger.info(f"MoM change: {mom_change:+,.0f}")
    
    return total_nsa_forecast, mom_change, series_contributions


if __name__ == "__main__":
    # Test prediction generation
    test_date = pd.Timestamp('2020-01-31')
    
    logger.info("Note: This requires a trained model to test predictions")
    logger.info("Run train_nbeats.py first to create a model")
