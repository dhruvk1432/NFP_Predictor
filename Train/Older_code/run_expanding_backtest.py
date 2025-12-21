"""
Expanding Window Backtest for NFP Forecasting Pipeline

Runs comprehensive backtesting with NBEATSx + LightGBM SA mapper:
1. Trains NBEATSx with warm starts for efficiency
2. Generates NSA forecasts and aggregates to total
3. Trains/updates LightGBM SA mapper on expanding window
4. Predicts SA MoM change
5. Records predictions vs actuals
6. Computes comprehensive metrics
7. Generates visualizations

Usage:
    python run_expanding_backtest.py  # Full history backtest 
    python run_expanding_backtest.py --months 36  # Last 36 months only
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, START_DATE, END_DATE, BACKTEST_MONTHS, setup_logger
from Train.train_nbeats import train_nbeats_snapshot
from Train.forecast_pipeline import generate_nfp_forecast
from Train.snapshot_loader import load_snapshot_data
from Train.sa_mapper import train_sa_mapper
from Train.backtest_archiver import prepare_new_backtest_run
from Train.backtest_results import generate_backtest_report

logger = setup_logger(__file__, TEMP_DIR)

def get_actual_sa_mom_change(target_month_end: pd.Timestamp, target_df: pd.DataFrame) -> tuple:
    """
    Get actual SA MoM change from the target dataframe (first release).
    
    Args:
        target_month_end: The month we predicted (e.g., 2022-10-31)
        target_df: DataFrame with 'ds' and 'y' columns (SA first release)
        
    Returns:
        (sa_mom_change, sa_target, sa_prev)
    """
    try:
        # Determine target and prev dates (start of month)
        target_start = target_month_end.replace(day=1)
        prev_start = (target_month_end - pd.DateOffset(months=1)).replace(day=1)
        
        val_target = target_df.loc[target_df['ds'] == target_start, 'y']
        val_prev = target_df.loc[target_df['ds'] == prev_start, 'y']
        
        if not val_target.empty and not val_prev.empty:
            sa_target = val_target.iloc[0]
            sa_prev = val_prev.iloc[0]
            sa_mom_change = sa_target - sa_prev
            return sa_mom_change, sa_target, sa_prev
            
    except Exception as e:
        logger.warning(f"Could not load actuals for {target_month_end.date()}: {e}")
        
    return np.nan, np.nan, np.nan


def get_actual_nsa_total(target_month_end: pd.Timestamp, nsa_target_df: pd.DataFrame) -> tuple:
    """
    Get actual NSA total from summing all NSA leaf series.
    
    Args:
        target_month_end: The month we predicted (e.g., 2022-10-31)
        nsa_target_df: DataFrame with NSA series (ds, y, series columns)
        
    Returns:
        (nsa_total, nsa_mom_change, nsa_total_prev)
    """
    try:
        # Determine target and prev dates (start of month)
        target_start = target_month_end.replace(day=1)
        prev_start = (target_month_end - pd.DateOffset(months=1)).replace(day=1)
        
        # Get all values for target month and sum
        target_data = nsa_target_df[nsa_target_df['ds'] == target_start]
        prev_data = nsa_target_df[nsa_target_df['ds'] == prev_start]
        
        if not target_data.empty and not prev_data.empty:
            # Sum across all series for total NSA
            nsa_total = target_data.drop(columns=['ds']).sum().sum()
            nsa_total_prev = prev_data.drop(columns=['ds']).sum().sum()
            nsa_mom_change = nsa_total - nsa_total_prev
            return nsa_total, nsa_mom_change, nsa_total_prev
            
    except Exception as e:
        logger.warning(f"Could not calculate NSA total for {target_month_end.date()}: {e}")
        
    return np.nan, np.nan, np.nan


def run_expanding_window_backtest(
    end_date: str = None,
    backtest_months: int = None,
    initial_training_months: int = 120,
    save_results: bool = True
):
    """
    Run expanding window backtest for NFP forecasting.
    
    Trains NBEATSx fresh from scratch at each snapshot (no warm starts).
    
    Args:
        end_date: End date (default: from settings)
        backtest_months: Number of months to backtest (default: all available)
        initial_training_months: Initial training window in months
        save_results: Save predictions and models
        
    Returns:
        DataFrame with predictions and actuals
    """
    # Parse settings
    if end_date is None:
        end_date = END_DATE
    end_date = pd.Timestamp(end_date)
    
    logger.info("="*70)
    logger.info("EXPANDING WINDOW BACKTEST - NFP FORECASTING PIPELINE")
    logger.info("="*70)
    logger.info(f"End date: {end_date.date()}")
    logger.info(f"Initial training window: {initial_training_months} months")
    logger.info(f"Training approach: Fresh NBEATSx training at each snapshot")
    
    # Determine backtest range
    if backtest_months:
        start_date = end_date - pd.DateOffset(months=backtest_months)
        logger.info(f"Backtest period: {backtest_months} months ({start_date.date()} to {end_date.date()})")
    else:
        # Use START_DATE from settings
        start_date = pd.Timestamp(START_DATE) + pd.DateOffset(months=initial_training_months)
        logger.info(f"Full history backtest from {start_date.date()} to {end_date.date()}")
    
    # Generate monthly dates (month-end)
    validation_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    logger.info(f"Total validation months: {len(validation_dates)}")
    
    # Create output directories
    models_dir = OUTPUT_DIR / "backtest" / "models"
    predictions_dir = OUTPUT_DIR / "backtest" / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Load target data for actuals
    # SA targets: for final LightGBM evaluation
    sa_target_path = DATA_PATH / "NFP_target" / "y_sa_first_release.parquet"
    if not sa_target_path.exists():
        logger.error(f"SA target file not found: {sa_target_path}")
        return pd.DataFrame()
    sa_target_df = pd.read_parquet(sa_target_path)
    logger.info(f"Loaded SA target data from {sa_target_path}")
    
    # NSA targets: for NBEATSx evaluation
    nsa_target_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    if not nsa_target_path.exists():
        logger.error(f"NSA target file not found: {nsa_target_path}")
        return pd.DataFrame()
    nsa_target_df = pd.read_parquet(nsa_target_path)
    logger.info(f"Loaded NSA target data from {nsa_target_path}: {nsa_target_df.shape}")
    
    # Storage for results
    results = []
    sa_training_data = []
    
    # Pre-load SA training data if available (to solve cold start)
    sa_training_file = OUTPUT_DIR / "training_data" / "sa_mapper_training.parquet"
    if sa_training_file.exists():
        try:
            logger.info(f"Loading historical SA training data from {sa_training_file}")
            hist_sa_df = pd.read_parquet(sa_training_file)
            sa_training_data = hist_sa_df.to_dict('records')
            logger.info(f"Loaded {len(sa_training_data)} historical samples")
        except Exception as e:
            logger.warning(f"Could not load historical SA data: {e}")
    
    # Progress bar
    pbar = tqdm(validation_dates, desc="Expanding Window Backtest")
    
    for i, snapshot_date in enumerate(pbar):
        pbar.set_description(f"Processing {snapshot_date.date()}")
        
        try:
            # 1. Train NBEATSx from scratch (no warm starts)
            logger.info(f"\n{'='*60}")
            logger.info(f"Training NBEATSx for {snapshot_date.date()} ({i+1}/{len(validation_dates)})")
            logger.info(f"{'='*60}")
            
            current_nbeats_model, metadata = train_nbeats_snapshot(
                snapshot_date=snapshot_date,
                lookback_months=initial_training_months,
                max_steps=1000,
                save_model=save_results,
                model_dir=models_dir / f"nbeatsx_{snapshot_date.strftime('%Y%m')}.pkl" if save_results else None
            )
            current_scalers = metadata['scalers']
            
            logger.info(f"NBEATSx training complete for {snapshot_date.date()}")
            
            # 2. Generate forecast (NSA prediction)
            # Use dummy SA mapper for now, we'll train real one after collecting data
            class DummySAMapper:
                def predict(self, X):
                    # Just return NSA MoM * 1.01 as placeholder
                    return [X['nsa_mom_change'].iloc[0] * 1.01]
            
            # CRITICAL FIX: Target date is the snapshot month itself (e.g., Feb 28 snapshot predicts Feb data)
            # Because at Feb 28, we only have Jan data (released early Feb). Feb data releases early March.
            # So we are predicting h=1 (Feb) from Jan data.
            forecast = generate_nfp_forecast(
                snapshot_date=snapshot_date,
                nbeats_model=current_nbeats_model,
                sa_mapper_model=DummySAMapper(),
                scalers=current_scalers,
                lookback_months=initial_training_months
            )
            
            # 3. Collect SA training data
            # Target date is the snapshot date (we just predicted this month)
            target_date = snapshot_date
            
            actual_sa_mom, actual_sa_total, prev_sa_total = get_actual_sa_mom_change(target_date, sa_target_df)
            
            # Also get NSA actuals for NBEATSx evaluation
            nsa_total_actual, nsa_mom_actual, nsa_total_prev = get_actual_nsa_total(target_date, nsa_target_df)
            
            if not np.isnan(actual_sa_mom):
                # Store features + target for LightGBM training
                sa_features = forecast['sa_features'].iloc[0].to_dict()
                sa_features['target_sa_mom_change'] = actual_sa_mom
                sa_features['snapshot_date'] = snapshot_date
                sa_features['target_date'] = target_date
                sa_training_data.append(sa_features)
            
            # 4. Train LightGBM SA mapper on accumulated data
            lgb_model = None
            if len(sa_training_data) >= 24:  # Need at least 24 months to train
                logger.info(f"Training LightGBM on {len(sa_training_data)} samples")
                
                train_df = pd.DataFrame(sa_training_data)
                # Remove date columns for training
                feature_cols = [c for c in train_df.columns 
                               if c not in ['target_sa_mom_change', 'snapshot_date', 'target_date']]
                
                # Handle NaNs (LightGBM can handle them, but let's fill for stability)
                train_df[feature_cols] = train_df[feature_cols].fillna(0)
                
                lgb_model, importance = train_sa_mapper(
                    training_data=train_df,
                    target_col='target_sa_mom_change',
                    num_boost_round=300,
                    early_stopping_rounds=30
                )
                
                # Save model
                if save_results:
                    import pickle
                    model_path = models_dir / f"lightgbm_{snapshot_date.strftime('%Y%m')}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(lgb_model, f)
            
            # 5. Generate final prediction with trained LightGBM
            if lgb_model is not None:
                forecast_final = generate_nfp_forecast(
                    snapshot_date=snapshot_date,
                    nbeats_model=current_nbeats_model,
                    sa_mapper_model=lgb_model,
                    scalers=current_scalers,
                    lookback_months=initial_training_months
                )
                predicted_sa_mom = forecast_final['sa_mom_change']
            else:
                # Use placeholder if LightGBM not yet trained
                predicted_sa_mom = forecast['nsa_mom_change'] * 1.01
            
            # 6. Record results
            result = {
                'snapshot_date': snapshot_date,
                'target_date': target_date,
                # NBEATSx NSA predictions
                'nsa_total_pred': forecast['nsa_total'],
                'nsa_mom_pred': forecast['nsa_mom_change'],
                # NBEATSx NSA actuals
                'nsa_total_actual': nsa_total_actual,
                'nsa_mom_actual': nsa_mom_actual,
                # LightGBM SA predictions
                'sa_mom_pred': predicted_sa_mom,
                # LightGBM SA actuals
                'sa_mom_actual': actual_sa_mom,
                'sa_total_actual': actual_sa_total,
                # Model status
                'lgb_trained': (lgb_model is not None),
                'num_training_samples': len(sa_training_data)
            }

            
            results.append(result)
            
            # Save predictions
            if save_results:
                pred_file = predictions_dir / f"forecast_{target_date.strftime('%Y%m')}.parquet"
                forecast_final_df = pd.DataFrame([result])
                forecast_final_df.to_parquet(pred_file, index=False)
            
            pbar.set_postfix({
                'NSA_MoM': f"{forecast['nsa_mom_change']:.0f}",
                'SA_MoM_pred': f"{predicted_sa_mom:.0f}",
                'SA_MoM_act': f"{actual_sa_mom:.0f}" if not np.isnan(actual_sa_mom) else "N/A"
            })
            
        except Exception as e:
            logger.error(f"Error processing {snapshot_date.date()}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    if save_results:
        output_file = OUTPUT_DIR / "backtest" / "backtest_results.parquet" 
        results_df.to_parquet(output_file, index=False)
        logger.info(f"\nSaved backtest results to {output_file}")
    
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run expanding window backtest')
    parser.add_argument('--months', type=int, default=None,
                       help=f'Backtest last N months (default: {BACKTEST_MONTHS} from .env)')
    parser.add_argument('--initial-training', type=int, default=120,
                       help='Initial training window in months (default: 120)')
    parser.add_argument('--skip-archive', action='store_true',
                       help='Skip archiving previous results')

    args = parser.parse_args()

    # Use BACKTEST_MONTHS from .env if not specified
    backtest_months = args.months if args.months is not None else BACKTEST_MONTHS

    # Archive previous results (unless skipped)
    if not args.skip_archive:
        archived_path = prepare_new_backtest_run()
        if archived_path:
            logger.info(f"Previous results archived to: {archived_path}")

    # Run backtest
    results = run_expanding_window_backtest(
        backtest_months=backtest_months,
        initial_training_months=args.initial_training
    )

    # Generate comprehensive report (placeholder - will be implemented after examining full backtest structure)
    logger.info("\n" + "="*70)
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info("="*70)

    # Note: Full report generation will be added once we see what features are available
    logger.info("✓ Results saved")
    logger.info(f"✓ Total forecasts: {len(results)}")
    logger.info(f"✓ Output directory: {OUTPUT_DIR}/backtest/")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"Total forecasts: {len(results)}")
    print(f"Backtest period: {backtest_months} months (from .env: BACKTEST_MONTHS={BACKTEST_MONTHS})")
    print(f"Results saved to: {OUTPUT_DIR}/backtest/")
