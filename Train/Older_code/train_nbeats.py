"""
NBEATSx Model Training

Handles training of multi-series NBEATSx model with:
- Proper scaling (RobustScaler for compatibility with scaler_type='robust')
- Warm-start optimization
- Sample weighting (exponential decay)
- Model checkpointing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger
from Train.feature_assembly import prepare_nbeats_training_data

logger = setup_logger(__file__, TEMP_DIR)

# Import neuralforecast
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    logger.warning("NeuralForecast not available. Install with: pip install neuralforecast")
    NEURALFORECAST_AVAILABLE = False


def compute_sample_weights(dates: pd.Series, decay_rate: float = 0.95) -> np.ndarray:
    """
    Compute exponential sample weights (recent data weighted higher).
    
    Args:
        dates: Series of dates
        decay_rate: Decay rate (0-1), higher = more weight on recent
        
    Returns:
        Array of weights
    """
    # Sort dates and get time differences from most recent
    sorted_dates = pd.to_datetime(dates).sort_values()
    max_date = sorted_dates.max()
    
    # Compute months from most recent
    months_from_end = (max_date.year - sorted_dates.dt.year) * 12 + (max_date.month - sorted_dates.dt.month)
    
    # Exponential decay
    weights = decay_rate ** months_from_end.values
    
    # Normalize to mean = 1
    weights = weights / weights.mean()
    
    return weights


def scale_features(
    train_df: pd.DataFrame,
    feature_lists: Dict[str, List[str]],
    scaler_type: str = 'robust'
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Scale exogenous features for NBEATSx.
    
    Args:
        train_df: Training dataframe
        feature_lists: Dict with hist_exog_list, futr_exog_list, stat_exog_list
        scaler_type: 'robust' or 'minmax'
        
    Returns:
        Tuple of (scaled_df, scalers_dict)
        
    Notes:
        - NBEATSx.scaler_type='robust' expects features pre-scaled or will scale y only
        - We pre-scale exogenous features for better convergence
        - Static features (binary) are not scaled
    """
    df_scaled = train_df.copy()
    scalers = {}
    
    # Scale historical exogenous features
    if feature_lists['hist_exog_list']:
        if scaler_type == 'robust':
            scaler_hist = RobustScaler()
        else:
            scaler_hist = MinMaxScaler()
        
        hist_cols = [c for c in feature_lists['hist_exog_list'] if c in df_scaled.columns]
        if hist_cols:
            df_scaled[hist_cols] = scaler_hist.fit_transform(df_scaled[hist_cols])
            scalers['hist_exog'] = scaler_hist
            logger.info(f"Scaled {len(hist_cols)} historical exog features with {scaler_type}")
    
    # Scale future exogenous features (same scaler type for consistency)
    if feature_lists['futr_exog_list']:
        if scaler_type == 'robust':
            scaler_futr = RobustScaler()
        else:
            scaler_futr = MinMaxScaler()
        
        futr_cols = [c for c in feature_lists['futr_exog_list'] if c in df_scaled.columns]
        if futr_cols:
            df_scaled[futr_cols] = scaler_futr.fit_transform(df_scaled[futr_cols])
            scalers['futr_exog'] = scaler_futr
            logger.info(f"Scaled {len(futr_cols)} future exog features with {scaler_type}")
    
    # Static features are NOT scaled (they're binary flags)
    logger.info(f"Static features ({len(feature_lists['stat_exog_list'])}) left unscaled (binary)")
    
    return df_scaled, scalers


def train_nbeats_snapshot(
    snapshot_date: pd.Timestamp,
    lookback_months: int = 120,
    max_steps: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    input_size: int = 24,  # Changed from 12 to 24 for year-over-year seasonality
    warm_start_model: Optional[NeuralForecast] = None,
    sample_weight_decay: float = 0.95,
    scaler_type: str = 'robust',
    save_model: bool = True,
    model_dir: Optional[Path] = None
) -> Tuple[NeuralForecast, Dict]:
    """
    Train NBEATSx model for a single snapshot.
    
    Args:
        snapshot_date: Snapshot month-end
        lookback_months: Months of training history
        max_steps: Training iterations
        learning_rate: Learning rate
        batch_size: Batch size
        input_size: Input window size (months)
        warm_start_model: Previous model for weight initialization
        sample_weight_decay: Exponential decay for sample weights
        scaler_type: 'robust' or 'minmax'
        save_model: Whether to save trained model
        
    Returns:
        Tuple of (trained_model, metadata_dict)
    """
    if not NEURALFORECAST_AVAILABLE:
        raise ImportError("NeuralForecast is required. Install with: pip install neuralforecast")
    
    logger.info(f"Training NBEATSx for snapshot {snapshot_date.date()}")
    
    # 1. Prepare training data
    train_df, feature_lists = prepare_nbeats_training_data(
        snapshot_date=snapshot_date,
        lookback_months=lookback_months
    )
    
    # 2. Scale features
    train_df_scaled, scalers = scale_features(train_df, feature_lists, scaler_type=scaler_type)
    
    # 3. Add sample weights
    # Group by unique_id and compute weights per series
    train_df_scaled['sample_weight'] = 1.0
    for uid in train_df_scaled['unique_id'].unique():
        mask = train_df_scaled['unique_id'] == uid
        series_dates = train_df_scaled.loc[mask, 'ds']
        weights = compute_sample_weights(series_dates, decay_rate=sample_weight_decay)
        train_df_scaled.loc[mask, 'sample_weight'] = weights
    
    logger.info(f"Sample weight range: [{train_df_scaled['sample_weight'].min():.3f}, {train_df_scaled['sample_weight'].max():.3f}]")
    
    # 4. Define NBEATSx model
    nbeats_model = NBEATSx(
        h=1,  # 1-month ahead forecast
        input_size=input_size,
        
        # Exogenous variables
        hist_exog_list=feature_lists['hist_exog_list'],
        futr_exog_list=feature_lists['futr_exog_list'],
        stat_exog_list=feature_lists['stat_exog_list'],
        
        # Architecture
        # Use identity blocks for h=1 (seasonality stack requires larger h)
        # We provide explicit seasonality via Fourier features anyway
        stack_types=['identity', 'identity', 'identity'],
        n_blocks=[3, 3, 3],
        mlp_units=[[512, 512], [512, 512], [512, 512]],
        
        # Training
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        windows_batch_size=256,
        
        # Regularization
        dropout_prob_theta=0.1,
        
        # Scaling (NBEATSx will apply robust scaling to y)
        scaler_type=scaler_type,
        
        # Random seed for reproducibility
        random_seed=42
    )
    
    # 5. Create NeuralForecast wrapper
    nf = NeuralForecast(models=[nbeats_model], freq='MS')
    
    # 6. Warm start (if provided)
    if warm_start_model is not None:
        try:
            # Copy weights from previous model
            nf.models[0].load_state_dict(warm_start_model.models[0].state_dict(), strict=False)
            logger.info("Initialized from warm start weights")
        except Exception as e:
            logger.warning(f"Could not load warm start weights: {e}")
    
    # 7. Train
    logger.info(f"Training with {len(train_df_scaled)} observations across {train_df_scaled['unique_id'].nunique()} series")
    logger.info(f"Features: {len(feature_lists['hist_exog_list'])} hist, {len(feature_lists['futr_exog_list'])} futr, {len(feature_lists['stat_exog_list'])} static")
    
    # Debug: Check for static columns
    missing_static = [c for c in feature_lists['stat_exog_list'] if c not in train_df_scaled.columns]
    if missing_static:
        logger.error(f"Missing static columns in dataframe: {missing_static}")
    else:
        logger.info("All static columns present in dataframe")

    # Extract static features to separate dataframe for NeuralForecast
    # Group by unique_id and take first value (they are static)
    static_cols = ['unique_id'] + feature_lists['stat_exog_list']
    static_df = train_df_scaled[static_cols].groupby('unique_id').first().reset_index()
    
    # Drop static columns from temporal dataframe to avoid confusion
    # (Though NeuralForecast might handle it, explicit separation is safer)
    train_df_temporal = train_df_scaled.drop(columns=feature_lists['stat_exog_list'])

    nf.fit(df=train_df_temporal, static_df=static_df)
    
    logger.info(f"Training complete for {snapshot_date.date()}")
    
    # 8. Save model
    if save_model:
        if model_dir is None:
            model_dir = OUTPUT_DIR / "models" / "nbeats"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"nbeats_{snapshot_date.strftime('%Y%m')}"
        nf.save(path=str(model_path), model_index=None, overwrite=True, save_dataset=False)
        
        # Save scalers separately
        scaler_path = model_dir / f"scalers_{snapshot_date.strftime('%Y%m')}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        logger.info(f"Saved model to {model_path}")
    
    # 9. Return metadata
    metadata = {
        'snapshot_date': snapshot_date,
        'n_series': train_df_scaled['unique_id'].nunique(),
        'n_observations': len(train_df_scaled),
        'date_range': (train_df_scaled['ds'].min(), train_df_scaled['ds'].max()),
        'feature_counts': {
            'hist_exog': len(feature_lists['hist_exog_list']),
            'futr_exog': len(feature_lists['futr_exog_list']),
            'stat_exog': len(feature_lists['stat_exog_list'])
        },
        'scalers': scalers
    }
    
    return nf, metadata


if __name__ == "__main__":
    # Test training
    test_date = pd.Timestamp('2020-01-31')
    logger.info(f"Testing NBEATSx training for {test_date.date()}")
    
    if NEURALFORECAST_AVAILABLE:
        try:
            model, metadata = train_nbeats_snapshot(
                snapshot_date=test_date,
                lookback_months=24,  # 2 years for quick test
                max_steps=100,  # Fewer steps for testing
                save_model=True
            )
            
            logger.info(f"\nTraining successful!")
            logger.info(f"Metadata: {metadata}")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
    else:
        logger.error("Cannot test: NeuralForecast not installed")
