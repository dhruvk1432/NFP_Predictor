"""
Complete NFP Forecasting Pipeline

Orchestrates the full prediction workflow:
1. NBEATSx: Predict 95 NSA leaf series
2. Reconciliation: Sum to NSA total + calculate MoM change
3. SA Mapper: LightGBM predicts SA total from NSA + features

This is the main entry point for generating NFP forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger
from Train.snapshot_loader import load_snapshot_data
from Train.feature_assembly import prepare_nbeats_training_data
from Train.predict_nbeats import generate_nbeats_predictions, aggregate_to_total_nsa
from Train.sa_mapper import engineer_sa_mapper_features, predict_sa_mom_change, load_sa_mapper

logger = setup_logger(__file__, TEMP_DIR)

try:
    from neuralforecast import NeuralForecast
    import lightgbm as lgb
    MODELS_AVAILABLE = True
except ImportError:
    logger.warning("NeuralForecast or LightGBM not available")
    MODELS_AVAILABLE = False


def generate_nfp_forecast(
    snapshot_date: pd.Timestamp,
    nbeats_model: NeuralForecast,
    sa_mapper_model: lgb.Booster,
    scalers: Optional[Dict] = None,
    lookback_months: int = 120
) -> Dict:
    """
    Generate complete NFP forecast (NSA and SA).
    
    This is the main forecasting function that orchestrates the entire pipeline.
    
    Args:
        snapshot_date: Current snapshot date (e.g., 2020-01-31)
        nbeats_model: Trained NBEATSx model
        sa_mapper_model: Trained LightGBM SA mapper (predicts MoM change)
        scalers: Feature scalers from NBEATSx training
        lookback_months: Months of history to use
        
    Returns:
        Dictionary with:
            - nsa_total: Bottom-up NSA total forecast
            - sa_total: SA total level (converted from MoM change)
            - sa_mom_change: SA MoM change prediction
            - nsa_mom_change: NSA MoM change
            - series_forecasts: Individual series predictions
            - features: Features used for SA mapper
            - metadata: Additional info
    """
    logger.info(f"="*60)
    logger.info(f"Generating NFP Forecast for {snapshot_date.date()}")
    logger.info(f"="*60)
    
    # 1. Generate NBEATSx predictions for all 95 leaf series
    logger.info("\n1. NBEATSx Prediction (95 NSA leaf series)")
    logger.info("-" * 40)
    
    predictions = generate_nbeats_predictions(
        model=nbeats_model,
        snapshot_date=snapshot_date,
        lookback_months=lookback_months,
        scalers=scalers
    )
    
    # 2. Get last month actuals for MoM calculation
    logger.info("\n2. Loading Historical Data")
    logger.info("-" * 40)
    
    train_df, _ = prepare_nbeats_training_data(
        snapshot_date=snapshot_date,
        lookback_months=lookback_months
    )
    
    # Get last observation for each series
    last_actuals = train_df.groupby('unique_id').tail(1)[['unique_id', 'ds', 'y']].copy()
    
    # Also get historical NSA totals
    historical_nsa = train_df.groupby('ds')['y'].sum().reset_index()
    historical_nsa = historical_nsa.sort_values('ds')
    
    # Get last month's SA actual (for converting MoM back to level)
    # We need SA series 'total' from the snapshot
    data = load_snapshot_data(snapshot_date)
    endo_df = data['endogenous']
    
    # Get last SA actual (series 'total' is SA, 'total_nsa' is NSA)
    sa_series = endo_df[endo_df['series_name'] == 'total'].sort_values('date')
    if not sa_series.empty:
        last_sa_actual = sa_series['value'].iloc[-1]
        last_sa_date = sa_series['date'].iloc[-1]
    else:
        # Fallback: use NSA if SA not available
        logger.warning("SA series 'total' not found, using NSA as fallback")
        last_sa_actual = historical_nsa['y'].iloc[-1]
        last_sa_date = historical_nsa['ds'].iloc[-1]
    
    logger.info(f"Last SA actual ({last_sa_date.date()}): {last_sa_actual:,.0f}")
    
    # 3. Aggregate to total NSA
    logger.info("\n3. Bottom-Up Aggregation to NSA Total")
    logger.info("-" * 40)
    
    nsa_total, nsa_mom_change, series_contributions = aggregate_to_total_nsa(
        predictions=predictions,
        last_actuals=last_actuals
    )
    
    # 4. Load exogenous features for SA mapper
    logger.info("\n4. Engineering Features for SA Mapper")
    logger.info("-" * 40)
    
    exog_df = data['exogenous']
    
    # Pivot exog to wide for easy access
    exog_wide = exog_df.pivot_table(
        index='date',
        columns='series_name',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Engineer features
    sa_features = engineer_sa_mapper_features(
        nsa_total_forecast=nsa_total,
        mom_change=nsa_mom_change,
        series_contributions=series_contributions,
        snapshot_date=snapshot_date,
        exog_features=exog_wide,
        historical_nsa=historical_nsa
    )
    
    logger.info(f"Created {len(sa_features.columns)} features for SA mapper")
    
    # 5. Predict SA MoM change
    logger.info("\n5. LightGBM SA MoM Change Prediction")
    logger.info("-" * 40)
    
    sa_mom_change = predict_sa_mom_change(
        model=sa_mapper_model,
        features=sa_features
    )
    
    # 6. Convert MoM change to level
    sa_total = last_sa_actual + sa_mom_change
    
    # 7. Compile results
    logger.info("\n" + "="*60)
    logger.info("FINAL FORECAST SUMMARY")
    logger.info("="*60)
    logger.info(f"NSA Total (bottom-up):    {nsa_total:>12,.0f}")
    logger.info(f"NSA MoM Change:           {nsa_mom_change:>12,.0f}")
    logger.info(f"SA MoM Change (LightGBM): {sa_mom_change:>12,.0f}")
    logger.info(f"SA Total (level):         {sa_total:>12,.0f}")
    logger.info(f"Forecast Month:           {snapshot_date.strftime('%Y-%m')}")
    logger.info("="*60)
    
    # Calculate some additional metrics
    sa_nsa_diff = sa_total - nsa_total
    
    return {
        'nsa_total': nsa_total,
        'nsa_mom_change': nsa_mom_change,
        'sa_total': sa_total,
        'sa_mom_change': sa_mom_change,
        'sa_nsa_diff': sa_nsa_diff,
        'series_forecasts': predictions,
        'series_contributions': series_contributions,
        'sa_features': sa_features,
        'metadata': {
            'snapshot_date': snapshot_date,
            'forecast_month': snapshot_date,
            'num_series': len(predictions),
            'last_sa_actual': last_sa_actual,
            'last_sa_date': last_sa_date,
            'num_features': len(sa_features.columns)
        }
    }


def save_forecast_results(
    forecast: Dict,
    output_dir: Path = None
):
    """
    Save forecast results to disk.
    
    Args:
        forecast: Forecast dictionary from generate_nfp_forecast()
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "forecasts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_date = forecast['metadata']['snapshot_date']
    forecast_month = forecast['metadata']['forecast_month']
    
    # Save summary
    summary = {
        'forecast_month': forecast_month.strftime('%Y-%m'),
        'snapshot_date': snapshot_date.strftime('%Y-%m-%d'),
        'nsa_total': forecast['nsa_total'],
        'nsa_mom_change': forecast['nsa_mom_change'],
        'sa_total': forecast['sa_total'],
        'sa_mom_change': forecast['sa_mom_change'],
        'sa_nsa_diff': forecast['sa_nsa_diff'],
        'num_series': forecast['metadata']['num_series'],
        'num_features': forecast['metadata']['num_features']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / f"forecast_{forecast_month.strftime('%Y%m')}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Save detailed series forecasts
    series_file = output_dir / f"forecast_{forecast_month.strftime('%Y%m')}_series.parquet"
    forecast['series_forecasts'].to_parquet(series_file, index=False)
    
    # Save contributions
    contrib_file = output_dir / f"forecast_{forecast_month.strftime('%Y%m')}_contributions.parquet"
    forecast['series_contributions'].to_parquet(contrib_file, index=False)
    
    # Save SA features
    features_file = output_dir / f"forecast_{forecast_month.strftime('%Y%m')}_sa_features.parquet"
    forecast['sa_features'].to_parquet(features_file, index=False)
    
    logger.info(f"\nSaved forecast results to {output_dir}")
    logger.info(f"  - Summary: {summary_file.name}")
    logger.info(f"  - Series forecasts: {series_file.name}")
    logger.info(f"  - Contributions: {contrib_file.name}")
    logger.info(f"  - SA Features: {features_file.name}")


if __name__ == "__main__":
    logger.info("NFP Forecasting Pipeline - Main Orchestrator")
    logger.info("\nThis module coordinates:")
    logger.info("  1. NBEATSx prediction (95 NSA series)")
    logger.info("  2. Bottom-up aggregation (NSA total)")
    logger.info("  3. LightGBM SA mapping (final prediction)")
    logger.info("\nTo use: Load trained models and call generate_nfp_forecast()")
