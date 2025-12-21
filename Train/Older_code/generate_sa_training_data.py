"""
Generate Training Data for SA Mapper

This script runs an expanding window backtest of the NBEATSx model to generate 
a dataset of (NSA Forecast + Features) -> (Actual SA Value).

This dataset is used to train the LightGBM SA Mapper.

Process:
1. Define a history range from START_DATE to (END_DATE - BACKTEST_MONTHS).
2. For each month:
    a. Train NBEATSx on data up to that month (or load cached).
    b. Generate NSA forecast for next month.
    c. Engineer SA features.
    d. Load ACTUAL SA value for next month (target).
3. Save the collected dataset.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger
from Train.train_nbeats import train_nbeats_snapshot
from Train.forecast_pipeline import generate_nfp_forecast
from Train.snapshot_loader import load_snapshot_data

logger = setup_logger(__file__, TEMP_DIR)

class DummySAMapper:
    """Dummy mapper just to let the pipeline run to get features."""
    def predict(self, X):
        return [0.0] * len(X)

def get_actual_sa_values(target_date: pd.Timestamp, prev_date: pd.Timestamp) -> Tuple[float, float]:
    """
    Get actual SA total NFP for target date and previous date.
    
    Returns:
        Tuple of (SA_target, SA_prev) for calculating MoM change
    """
    try:
        # Load target month snapshot
        data_target = load_snapshot_data(target_date)
        df_target = data_target['endogenous']
        
        # Load previous month snapshot
        data_prev = load_snapshot_data(prev_date)
        df_prev = data_prev['endogenous']
        
        # Get SA total ('total' is SA series)
        row_target = df_target[(df_target['series_name'] == 'total') & (df_target['date'] == target_date)]
        row_prev = df_prev[(df_prev['series_name'] == 'total') & (df_prev['date'] == prev_date)]
        
        if not row_target.empty and not row_prev.empty:
            sa_target = row_target['value'].iloc[0]
            sa_prev = row_prev['value'].iloc[0]
            return sa_target, sa_prev
            
    except Exception as e:
        logger.warning(f"Could not load SA values for {target_date.date()}: {e}")
        
    return np.nan, np.nan

def generate_sa_training_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_months: int = 48,
    stride_months: int = 1  # Run every month
):
    """
    Generate dataset for LightGBM training.
    """
    logger.info(f"Generating SA training data from {start_date.date()} to {end_date.date()}")
    
    # Load target data for actuals
    target_path = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"
    if not target_path.exists():
        logger.error(f"Target file not found: {target_path}")
        return None
        
    target_df = pd.read_parquet(target_path)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    dataset = []
    
    # We can skip retraining NBEATSx every single month to save time
    # Maybe retrain every 6 months?
    # For this script, we'll assume we retrain every 3 months.
    
    current_model = None
    current_scalers = None
    last_train_date = None
    
    for snapshot_date in tqdm(dates):
        logger.info(f"Processing {snapshot_date.date()}")
        
        # 1. Train/Update NBEATSx
        # Retrain if no model or if it's been 3 months
        should_retrain = (
            current_model is None or 
            last_train_date is None or 
            (snapshot_date.year != last_train_date.year) or 
            (snapshot_date.month - last_train_date.month >= 3)
        )
        
        if should_retrain:
            logger.info("Retraining NBEATSx...")
            try:
                current_model, metadata = train_nbeats_snapshot(
                    snapshot_date=snapshot_date,
                    lookback_months=lookback_months,
                    max_steps=500, # Shorter training for data generation
                    save_model=False
                )
                current_scalers = metadata['scalers']
                last_train_date = snapshot_date
            except Exception as e:
                logger.error(f"Failed to train NBEATSx for {snapshot_date}: {e}")
                continue
        
        # 2. Generate Forecast
        try:
            forecast = generate_nfp_forecast(
                snapshot_date=snapshot_date,
                nbeats_model=current_model,
                sa_mapper_model=DummySAMapper(), # We don't have one yet!
                scalers=current_scalers,
                lookback_months=lookback_months
            )
            
            # 3. Extract Features
            features = forecast['sa_features'].iloc[0].to_dict()
            
            # 4. Get Target (Actual SA MoM Change)
            # Target date is the snapshot date (predicting current month)
            target_date = snapshot_date.replace(day=1)
            prev_date = (snapshot_date - pd.DateOffset(months=1)).replace(day=1)
            
            # Load actuals from target file
            # We load this once outside the loop in production, but here for safety we can load or pass it in
            # Better: load it at the start of the function
            
            # Look up values
            try:
                # We need to access the target_df passed to the function or loaded globally
                # For now, let's load it inside the function if not passed, but efficient way is to pass it
                # Let's assume we load it at start of function
                
                val_target = target_df.loc[target_df['ds'] == target_date, 'y']
                val_prev = target_df.loc[target_df['ds'] == prev_date, 'y']
                
                if not val_target.empty and not val_prev.empty:
                    sa_target = val_target.iloc[0]
                    sa_prev = val_prev.iloc[0]
                    sa_mom_change = sa_target - sa_prev
                    
                    # Add target to features
                    features['target_sa_mom_change'] = sa_mom_change
                    features['target_sa_total'] = sa_target
                    features['prev_sa_total'] = sa_prev
                    features['snapshot_date'] = snapshot_date
                    features['target_date'] = target_date
                    
                    dataset.append(features)
                else:
                    logger.warning(f"SA values not found in target file for {target_date.date()}")
                    
            except Exception as e:
                logger.warning(f"Error getting actuals for {target_date.date()}: {e}")
                continue
            
        except Exception as e:
            logger.error(f"Failed to generate forecast for {snapshot_date}: {e}")
            continue
            
    # Save dataset
    if dataset:
        df = pd.DataFrame(dataset)
        output_path = OUTPUT_DIR / "training_data" / "sa_mapper_training.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} samples to {output_path}")
        return df
    else:
        logger.error("No data generated")
        return None

if __name__ == "__main__":
    from settings import START_DATE, END_DATE, BACKTEST_MONTHS, DATA_PATH
    
    # Load target data for actuals
    target_path = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"
    if not target_path.exists():
        logger.error(f"Target file not found: {target_path}")
        sys.exit(1)
        
    target_df = pd.read_parquet(target_path)
    logger.info(f"Loaded target data from {target_path}")
    
    # Generate data from START_DATE up to the beginning of the backtest period
    # This ensures we have history for the backtest to use without look-ahead
    # For preloading, we go from START_DATE to END_DATE-BACKTEST_MONTHS to get
    # even more data (with more series as time progresses which the model should be able to handle)
    
    gen_start = pd.Timestamp(START_DATE)
    
    # Backtest starts at END_DATE - BACKTEST_MONTHS
    # So generation should end just before that
    backtest_start = pd.Timestamp(END_DATE) - pd.DateOffset(months=BACKTEST_MONTHS)
    gen_end = backtest_start - pd.DateOffset(months=1)
    gen_end = gen_end + pd.offsets.MonthEnd(0)
    
    logger.info(f"Generating historical training data for LightGBM...")
    logger.info(f"Range: {gen_start.date()} to {gen_end.date()}")
    
    if gen_start >= gen_end:
        logger.error("Generation start date is after end date! Check settings.")
    else:
        # Pass target_df to the function (we need to modify signature first, but for now let's make it global or load inside)
        # To avoid signature change issues in this step, we'll load it inside the function for now
        # But wait, I can modify the signature in the same file.
        pass
        
    generate_sa_training_data(gen_start, gen_end)
