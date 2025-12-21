"""
Test Runner for Training Pipeline

Quick validation script to test the entire training flow on a small dataset.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.snapshot_loader import load_snapshot_data
from Train.feature_assembly import prepare_nbeats_training_data

logger = setup_logger(__file__, TEMP_DIR)


def test_data_pipeline():
    """Test the data preparation pipeline."""
    logger.info("="*60)
    logger.info("Testing Training Data Pipeline")
    logger.info("="*60)
    
    # Test snapshot
    test_date = pd.Timestamp('2020-01-31')
    
    # 1. Test snapshot loading
    logger.info(f"\n1. Testing snapshot loader for {test_date.date()}")
    try:
        data = load_snapshot_data(test_date)
        logger.info(f"   ✓ Loaded {data['endogenous'].shape[0]} endogenous rows")
        logger.info(f"   ✓ Loaded {data['exogenous'].shape[0]} exogenous rows")
        logger.info(f"   ✓ Endogenous series: {data['endogenous']['series_name'].nunique()}")
        logger.info(f"   ✓ Exogenous series: {data['exogenous']['series_name'].nunique()}")
    except Exception as e:
        logger.error(f"   ✗ Snapshot loading failed: {e}")
        return False
    
    # 2. Test feature assembly
    logger.info(f"\n2. Testing feature assembly")
    try:
        train_df, feature_lists = prepare_nbeats_training_data(
            snapshot_date=test_date,
            lookback_months=24  # Just 2 years for quick test
        )
        logger.info(f"   ✓ Training data shape: {train_df.shape}")
        logger.info(f"   ✓ Unique series: {train_df['unique_id'].nunique()}")
        logger.info(f"   ✓ Date range: {train_df['ds'].min().date()} to {train_df['ds'].max().date()}")
        logger.info(f"   ✓ Feature counts:")
        logger.info(f"      - hist_exog: {len(feature_lists['hist_exog_list'])}")
        logger.info(f"      - futr_exog: {len(feature_lists['futr_exog_list'])}")
        logger.info(f"      - stat_exog: {len(feature_lists['stat_exog_list'])}")
        
        # Check for NaNs
        nan_count = train_df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"   ⚠ Found {nan_count} NaN values in training data")
        else:
            logger.info(f"   ✓ No NaN values in training data")
        
        # Check data types
        logger.info(f"\n   Data types:")
        logger.info(f"      - unique_id: {train_df['unique_id'].dtype}")
        logger.info(f"      - ds: {train_df['ds'].dtype}")
        logger.info(f"      - y: {train_df['y'].dtype}")
        
    except Exception as e:
        logger.error(f"   ✗ Feature assembly failed: {e}", exc_info=True)
        return False
    
    # 3. Sample data check
    logger.info(f"\n3. Sample data validation")
    try:
        # Check a specific series
        sample_series = train_df[train_df['unique_id'] == train_df['unique_id'].iloc[0]].copy()
        logger.info(f"   ✓ Sample series: {sample_series['unique_id'].iloc[0]}")
        logger.info(f"   ✓ Observations: {len(sample_series)}")
        logger.info(f"   ✓ Target range: [{sample_series['y'].min():.0f}, {sample_series['y'].max():.0f}]")
        
        # Show first few rows
        logger.info(f"\n   First 3 observations:")
        display_cols = ['unique_id', 'ds', 'y'] + feature_lists['futr_exog_list'][:3]
        logger.info(f"\n{sample_series[display_cols].head(3).to_string()}")
        
    except Exception as e:
        logger.error(f"   ✗ Sample validation failed: {e}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ All pipeline tests passed!")
    logger.info(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    success = test_data_pipeline()
    
    if success:
        logger.info("\n✅ Training pipeline is ready for use")
        logger.info("\nNext steps:")
        logger.info("  1. Run a small training test: python Train/train_nbeats.py")
        logger.info("  2. Implement expanding window backtest")
        logger.info("  3. Run full historical validation")
    else:
        logger.error("\n❌ Pipeline tests failed - check logs above")
        sys.exit(1)
