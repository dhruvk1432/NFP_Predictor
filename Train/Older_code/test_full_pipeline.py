"""
Test Full Pipeline End-to-End

Runs the complete workflow on a few recent snapshots to verify integration.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.train_nbeats import train_nbeats_snapshot
from Train.forecast_pipeline import generate_nfp_forecast

logger = setup_logger(__file__, TEMP_DIR)

class DummySAMapper:
    """Dummy model to test pipeline flow without trained LightGBM."""
    def predict(self, X):
        # Return dummy SA MoM change (small % of NSA forecast)
        # Assume SA MoM is similar to NSA MoM (with small adjustment)
        nsa_mom = X['nsa_mom_change'].iloc[0] if 'nsa_mom_change' in X.columns else 0
        # Add small random variation
        return [nsa_mom * 1.01]  # SA MoM slightly higher than NSA MoM

def run_pipeline_test():
    test_dates = [
        pd.Timestamp('2023-12-31'),
        pd.Timestamp('2024-03-31')
    ]
    
    logger.info("="*60)
    logger.info("STARTING FULL PIPELINE VERIFICATION")
    logger.info("="*60)
    
    results = []
    
    for snapshot_date in test_dates:
        logger.info(f"\n\n>>> TESTING SNAPSHOT: {snapshot_date.date()} <<<")
        
        try:
            # 1. Train NBEATSx (Quick run)
            logger.info("\n1. Training NBEATSx (Fast Mode)...")
            model, metadata = train_nbeats_snapshot(
                snapshot_date=snapshot_date,
                lookback_months=48,  # Increased to 48 to satisfy input_size=24 requirement
                max_steps=10,        # Very short training for test
                batch_size=16,
                save_model=False     # Don't clutter disk
            )
            
            # 2. Generate Forecast
            logger.info("\n2. Generating Forecast...")
            
            # Use dummy SA mapper
            dummy_sa_model = DummySAMapper()
            
            forecast = generate_nfp_forecast(
                snapshot_date=snapshot_date,
                nbeats_model=model,
                sa_mapper_model=dummy_sa_model,
                scalers=metadata['scalers'],
                lookback_months=48  # Match training lookback
            )
            
            # 3. Validation
            nsa_total = forecast['nsa_total']
            nsa_mom_change = forecast['nsa_mom_change']
            sa_mom_change = forecast['sa_mom_change']
            sa_features = forecast['sa_features']
            
            logger.info("\n3. Validation Results:")
            logger.info(f"   ✓ NSA Total: {nsa_total:,.0f}")
            logger.info(f"   ✓ NSA MoM Change: {nsa_mom_change:+,.0f}")
            logger.info(f"   ✓ SA MoM Change: {sa_mom_change:+,.0f}")
            logger.info(f"   ✓ SA Features Generated: {len(sa_features.columns)}")
            
            # Check for NaNs in features
            if sa_features.isna().any().any():
                logger.error("   ✗ NaN values found in SA features!")
                logger.error(sa_features.columns[sa_features.isna().any()].tolist())
            else:
                logger.info("   ✓ SA Features clean (no NaNs)")
                
            results.append({
                'date': snapshot_date,
                'status': 'SUCCESS',
                'nsa_total': nsa_total,
                'nsa_mom_change': nsa_mom_change,
                'sa_mom_change': sa_mom_change
            })
            
        except Exception as e:
            logger.error(f"   ✗ FAILED: {str(e)}", exc_info=True)
            results.append({
                'date': snapshot_date,
                'status': 'FAILED',
                'error': str(e)
            })
            
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for res in results:
        status_icon = "✅" if res['status'] == 'SUCCESS' else "❌"
        if res['status'] == 'SUCCESS':
            logger.info(f"{status_icon} {res['date'].date()}: NSA={res['nsa_total']:,.0f}, NSA_MoM={res['nsa_mom_change']:+,.0f}, SA_MoM={res['sa_mom_change']:+,.0f}")
        else:
            logger.info(f"{status_icon} {res['date'].date()}: {res['error']}")

if __name__ == "__main__":
    run_pipeline_test()
