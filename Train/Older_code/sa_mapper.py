"""
LightGBM SA Mapper

Maps NSA total forecast (and rich features) to SA total NFP prediction.

This is the final layer that converts hierarchically-reconciled NSA forecasts
to the seasonally-adjusted total NFP number.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, TEMP_DIR, OUTPUT_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


def engineer_sa_mapper_features(
    nsa_total_forecast: float,
    mom_change: float,
    series_contributions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    exog_features: pd.DataFrame,
    historical_nsa: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineer rich feature set for LightGBM SA mapper.
    
    Args:
        nsa_total_forecast: Bottom-up NSA total prediction
        mom_change: MoM change in NSA
        series_contributions: Per-series contribution to change
        snapshot_date: Current snapshot date
        exog_features: External exogenous features (from snapshot)
        historical_nsa: Historical NSA totals (for momentum, trends)
        
    Returns:
        DataFrame with one row of features for LightGBM
    """
    features = {}
    
    # 1. Core NSA forecasts
    features['nsa_forecast'] = nsa_total_forecast
    features['nsa_mom_change'] = mom_change
    features['nsa_mom_change_pct'] = (mom_change / (nsa_total_forecast - mom_change)) * 100
    
    # 2. Calendar features
    forecast_month = snapshot_date + pd.DateOffset(months=1)
    features['month'] = forecast_month.month
    features['quarter'] = forecast_month.quarter
    features['is_jan'] = int(forecast_month.month == 1)
    features['is_july'] = int(forecast_month.month == 7)  # Seasonal adjustment updates
    features['is_dec'] = int(forecast_month.month == 12)
    features['is_q1'] = int(forecast_month.quarter == 1)
    features['is_q4'] = int(forecast_month.quarter == 4)
    
    # 3. Series composition (sector momentum)
    if not series_contributions.empty:
        # Separate by sector
        private_series = series_contributions[
            series_contributions['unique_id'].str.contains('private')
        ]
        govt_series = series_contributions[
            series_contributions['unique_id'].str.contains('government')
        ]
        goods_series = series_contributions[
            series_contributions['unique_id'].str.contains('goods')
        ]
        services_series = series_contributions[
            series_contributions['unique_id'].str.contains('services')
        ]
        
        features['private_contribution'] = private_series['contribution_to_change'].sum()
        features['govt_contribution'] = govt_series['contribution_to_change'].sum()
        features['goods_contribution'] = goods_series['contribution_to_change'].sum()
        features['services_contribution'] = services_series['contribution_to_change'].sum()
        
        # Momentum indicators (positive vs negative changes)
        features['pct_series_positive'] = (
            (series_contributions['contribution_to_change'] > 0).sum() / 
            len(series_contributions) * 100
        )
        features['pct_series_negative'] = (
            (series_contributions['contribution_to_change'] < 0).sum() / 
            len(series_contributions) * 100
        )
        
        # Concentration of change
        abs_contributions = series_contributions['contribution_to_change'].abs()
        total_abs_change = abs_contributions.sum()
        if total_abs_change > 0:
            # Top 10 series account for what % of total change?
            top10_contribution = abs_contributions.nlargest(10).sum()
            features['change_concentration_top10'] = (top10_contribution / total_abs_change) * 100
        else:
            features['change_concentration_top10'] = 0.0
    
    # 4. Historical momentum (from historical_nsa)
    if not historical_nsa.empty and len(historical_nsa) >= 12:
        # Grab up to 13 months to allow for 12-month change calculation
        recent_nsa = historical_nsa['y'].iloc[-13:].values
        
        # Historical NSA momentum
        # Need at least N points for N-1 month change (current + N-1 prior)
        if len(recent_nsa) >= 4: # For 3-month change (current - 3 months ago)
            features['nsa_3m_change'] = recent_nsa[-1] - recent_nsa[-4]
        else:
            features['nsa_3m_change'] = 0.0
            
        if len(recent_nsa) >= 7: # For 6-month change (current - 6 months ago)
            features['nsa_6m_change'] = recent_nsa[-1] - recent_nsa[-7]
        else:
            features['nsa_6m_change'] = 0.0
            
        if len(recent_nsa) >= 13: # For 12-month change (current - 12 months ago)
            features['nsa_12m_change'] = recent_nsa[-1] - recent_nsa[-13]
        elif len(recent_nsa) >= 12: # Fallback to 11-month change
            features['nsa_12m_change'] = recent_nsa[-1] - recent_nsa[-12]
        else:
            features['nsa_12m_change'] = 0.0
        
        # Trend: linear regression slope over last 6 months
        if len(recent_nsa) >= 6:
            x = np.arange(6)
            y = recent_nsa[-6:]
            slope = np.polyfit(x, y, 1)[0]
            features['nsa_6m_trend'] = slope
        
        # Volatility
        if len(recent_nsa) >= 6:
            recent_changes = np.diff(recent_nsa[-7:])
            features['nsa_volatility_6m'] = np.std(recent_changes)
    
    # 5. Year-over-year for seasonal context
    if not historical_nsa.empty and len(historical_nsa) >= 12:
        yoy_idx = -13  # 12 months ago
        if abs(yoy_idx) <= len(historical_nsa):
            last_year_value = historical_nsa['y'].iloc[yoy_idx]
            features['nsa_yoy_change'] = nsa_total_forecast - last_year_value
            features['nsa_yoy_change_pct'] = (
                (nsa_total_forecast - last_year_value) / last_year_value * 100
            )
    
    
    # 6. Comprehensive exogenous indicators (use all 212 available features)
    if not exog_features.empty:
        # Get latest values for ALL key indicators
        comprehensive_indicators = {
            # Labor market (fast-release)
            'adp': ['ADP_actual', 'ADP_forecast', 'ADP_revision', 'ADP_previous'],
            'claims': ['ICSA_monthly_avg', 'CCSA_monthly_avg'],  
            'challenger': ['Challenger_Job_Cuts'],
            
            # Labor market (slow-release - use lag1 which is known)
            'jolts': ['JOLTS_Hires_log_lag1', 'JOLTS_Quits_log_lag1', 
                     'JOLTS_Layoffs_log_lag1', 'JOLTS_Openings_log_lag1'],
            'iursa': ['IURSA_monthly_avg_log_lag1'],
            
            # Business surveys
            'ism_mfg': ['ISM_Manufacturing_Index', 'ISM_Manufacturing_Index_lag1'],
            'ism_svc': ['ISM_NonManufacturing_Index'],
            'cb_confidence': ['CB_Consumer_Confidence', 'CB_Consumer_Confidence_lag1'],
            
            # Financial markets
            'oil': ['Oil_Prices_end_of_month', 'Oil_Prices_volatility', 
                   'Oil_Prices_mean', 'Oil_Prices_max', 'Oil_Prices_min'],
            'credit': ['Credit_Spreads_avg', 'Credit_Spreads_last', 
                      'Credit_Spreads_monthly_chg', 'Credit_Spreads_vol_of_changes'],
            'yields': ['Yield_Curve_avg', 'Yield_Curve_last', 'Yield_Curve_monthly_chg'],
            
            # Weather/disaster impacts (use lag1 as it's slow-release)
            'noaa_deaths': ['deaths_direct_weighted_log_lag1', 'deaths_indirect_weighted_log_lag1'],
            'noaa_injuries': ['injuries_direct_weighted_log_lag1', 'injuries_indirect_weighted_log_lag1'],
            'noaa_damage': ['total_property_damage_real_weighted_log_lag1', 
                           'total_crop_damage_real_weighted_log_lag1'],
            'noaa_storms': ['storm_count_weighted_log_lag1']
        }
        
        # Extract available features
        for category, patterns in comprehensive_indicators.items():
            for pattern in patterns:
                matching_cols = [c for c in exog_features.columns if pattern in c]
                if matching_cols:
                    # Use first match (prefer base, then lag1)
                    col = sorted(matching_cols, key=lambda x: ('_lag' not in x, x))[0]
                    features[f'exog_{category}_{pattern}'] = exog_features[col].iloc[-1]
        
        # Add rolling means for key indicators (3-month smoothing)
        for category, patterns in [
            ('adp_rolling', ['ADP_actual', 'rolling_mean_3']),
            ('claims_rolling', ['ICSA', 'rolling_mean_3']),
            ('jolts_rolling', ['JOLTS_Hires', 'rolling_mean_3']),
            ('oil_rolling', ['Oil_Prices_end_of_month', 'rolling_mean_12']),
            ('credit_rolling', ['Credit_Spreads_avg', 'rolling_mean_6'])
        ]:
            pattern_str = '_'.join(patterns)
            matching_cols = [c for c in exog_features.columns if all(p in c for p in patterns)]
            if matching_cols:
                col = matching_cols[0]
                features[f'{category}'] = exog_features[col].iloc[-1]
    
    # 7. Parent employment features (smoothness indicators, no look-ahead bias)
    # Add parent sector momentum from historical_nsa if series contributions available
    if not series_contributions.empty:
        # Calculate parent-level aggregates (smooth signal)
        private_parents = ['private_nsa', 'private_goods_nsa', 'private_services_nsa']
        govt_parents = ['government_nsa', 'federal_nsa', 'state_local_nsa']
        
        # If we have access to historical parent data, use it for parent trends
        # (This would come from the snapshot's endogenous data)
        # For now, use the series contributions as a proxy for parent momentum
        
        # Parent smoothness ratio: how volatile are children vs implied parent?
        total_child_volatility = series_contributions['contribution_to_change'].std()
        if total_child_volatility > 0 and 'nsa_volatility_6m' in features:
            features['parent_child_volatility_ratio'] = features.get('nsa_volatility_6m', 0) / total_child_volatility
    
    # 8. Interaction features
    if 'nsa_mom_change' in features and 'month' in features:
        # Seasonal interaction: how does MoM change interact with month?
        features['mom_x_month'] = features['nsa_mom_change'] * features['month']
        
    if 'nsa_yoy_change_pct' in features and 'nsa_mom_change_pct' in features:
        # Momentum alignment
        features['yoy_mom_alignment'] = (
            features['nsa_yoy_change_pct'] * features['nsa_mom_change_pct']
        )
    
    # 9. Statistical aggregations
    features['total_features'] = len(features)
    
    return pd.DataFrame([features])


def train_sa_mapper(
    training_data: pd.DataFrame,
    target_col: str = 'sa_mom_change',
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50
) -> Tuple[lgb.Booster, Dict]:
    """
    Train LightGBM model to map NSA → SA.
    
    Args:
        training_data: DataFrame with features + target
        target_col: Name of target column (SA total)
        num_boost_round: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Tuple of (trained_model, feature_importance_dict)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required")
    
    logger.info(f"Training LightGBM SA mapper on {len(training_data)} samples")
    
    # Separate features and target
    feature_cols = [c for c in training_data.columns if c != target_col]
    X = training_data[feature_cols]
    y = training_data[target_col]
    
    # Create train/validation split (temporal)
    train_size = int(len(training_data) * 0.8)
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Train
    logger.info(f"Training with {len(feature_cols)} features")
    
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50)
        ],
        evals_result=evals_result
    )
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type='gain')))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"\nTop 10 most important features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
        logger.info(f"  {i}. {feat}: {imp:.1f}")
    
    # Validation metrics
    train_rmse = evals_result['train']['rmse'][-1]
    val_rmse = evals_result['valid']['rmse'][-1]
    logger.info(f"\nFinal RMSE - Train: {train_rmse:.2f}, Validation: {val_rmse:.2f}")
    
    return model, importance


def predict_sa_mom_change(
    model: lgb.Booster,
    features: pd.DataFrame
) -> float:
    """
    Predict SA MoM change from features.
    
    Args:
        model: Trained LightGBM model
        features: Feature DataFrame (single row)
        
    Returns:
        SA MoM change prediction
    """
    prediction = model.predict(features)[0]
    logger.info(f"SA MoM change prediction: {prediction:,.0f}")
    return prediction


def save_sa_mapper(
    model: lgb.Booster,
    save_dir: Path,
    model_name: str = "sa_mapper"
):
    """Save LightGBM SA mapper model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}.txt"
    model.save_model(str(model_path))
    logger.info(f"Saved SA mapper to {model_path}")


def load_sa_mapper(
    save_dir: Path,
    model_name: str = "sa_mapper"
) -> Optional[lgb.Booster]:
    """Load LightGBM SA mapper model."""
    model_path = save_dir / f"{model_name}.txt"
    
    if not model_path.exists():
        logger.warning(f"SA mapper not found: {model_path}")
        return None
    
    model = lgb.Booster(model_file=str(model_path))
    logger.info(f"Loaded SA mapper from {model_path}")
    return model


if __name__ == "__main__":
    logger.info("LightGBM SA Mapper Module")
    logger.info("This module maps NSA forecasts to SA total predictions")
    
    if LIGHTGBM_AVAILABLE:
        logger.info("✓ LightGBM is available")
    else:
        logger.error("✗ LightGBM not installed - run: pip install lightgbm")
