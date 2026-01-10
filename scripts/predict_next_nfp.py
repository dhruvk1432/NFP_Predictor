#!/usr/bin/env python3
"""
Predict Next NFP

Production script to generate NFP prediction for the upcoming month.
Loads latest model, fetches fresh data, and outputs formatted prediction.

Usage:
    python scripts/predict_next_nfp.py --target nsa
    python scripts/predict_next_nfp.py --target sa --output report.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import MODEL_SAVE_DIR
from Train.model import load_model, predict_with_intervals

logger = setup_logger(__file__, TEMP_DIR)


def get_latest_snapshot_date() -> pd.Timestamp:
    """Find the most recent master snapshot available."""
    from Train.config import MASTER_SNAPSHOTS_DIR
    
    all_files = list(MASTER_SNAPSHOTS_DIR.rglob("*.parquet"))
    
    if not all_files:
        raise FileNotFoundError("No master snapshots found")
    
    # Extract dates from filenames
    dates = []
    for f in all_files:
        try:
            date_str = f.stem  # e.g., "2024-11"
            date = pd.to_datetime(date_str + "-01")
            dates.append(date)
        except:
            continue
    
    if not dates:
        raise ValueError("Could not parse snapshot dates")
    
    return max(dates)


def load_latest_snapshot(snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Load the most recent master snapshot."""
    from Train.data_loader import load_master_snapshot
    
    snapshot = load_master_snapshot(snapshot_date)
    
    if snapshot is None:
        raise FileNotFoundError(f"Snapshot not found for {snapshot_date}")
    
    return snapshot


def build_prediction_features(
    snapshot_df: pd.DataFrame,
    target_month: pd.Timestamp,
    target_type: str = 'nsa'
) -> pd.DataFrame:
    """
    Build feature row for prediction from snapshot data.
    
    Args:
        snapshot_df: Master snapshot DataFrame
        target_month: Month to predict
        target_type: 'nsa' or 'sa'
        
    Returns:
        Single-row DataFrame with all features
    """
    from Train.data_loader import (
        pivot_snapshot_to_wide,
        get_lagged_target_features,
        load_target_data
    )
    from Train.feature_engineering import add_calendar_features
    
    # Pivot snapshot to wide format
    features = {}
    
    # Get latest value for each series
    for series_name in snapshot_df['series_name'].unique():
        series_data = snapshot_df[snapshot_df['series_name'] == series_name]
        if not series_data.empty:
            features[f"{series_name}_latest"] = series_data['value'].iloc[-1]
    
    # Add calendar features
    calendar_features = add_calendar_features(None, target_month)
    features.update(calendar_features)
    
    # Add lagged target features
    try:
        target_df = load_target_data(target_type)
        prefix = f'nfp_{target_type}'
        lagged = get_lagged_target_features(target_df, target_month, prefix)
        features.update(lagged)
    except Exception as e:
        logger.warning(f"Could not load target data for lags: {e}")
    
    features['ds'] = target_month
    
    return pd.DataFrame([features])


def generate_prediction(
    target_type: str = 'nsa',
    output_path: Path = None
) -> dict:
    """
    Generate NFP prediction for the upcoming month.
    
    Args:
        target_type: 'nsa' or 'sa'
        output_path: Optional path to save JSON report
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Generating {target_type.upper()} prediction...")
    
    # Load model
    try:
        model, metadata = load_model(target_type=target_type)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"No trained model found for {target_type}")
        return {'error': 'Model not found. Train a model first.'}
    
    feature_cols = metadata.get('feature_cols', [])
    residuals = metadata.get('residuals', [])
    
    # Get latest snapshot
    snapshot_date = get_latest_snapshot_date()
    logger.info(f"Latest snapshot: {snapshot_date.strftime('%Y-%m')}")
    
    snapshot_df = load_latest_snapshot(snapshot_date)
    
    # Target month is the snapshot month (we predict current month's NFP)
    target_month = snapshot_date
    
    # Build features
    features_df = build_prediction_features(snapshot_df, target_month, target_type)
    
    # Check for missing features
    available_cols = [c for c in feature_cols if c in features_df.columns]
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} features, using NaN")
        for col in missing_cols:
            features_df[col] = np.nan
    
    # Make prediction
    X = features_df[feature_cols].values.reshape(1, -1)
    
    # Handle NaN with median imputation
    X = np.nan_to_num(X, nan=0)
    
    prediction = model.predict(X)[0]
    
    # Calculate intervals
    from Train.model import calculate_prediction_intervals
    intervals = calculate_prediction_intervals(residuals, prediction)
    
    # Build result
    result = {
        'timestamp': datetime.now().isoformat(),
        'target_type': target_type.upper(),
        'target_month': target_month.strftime('%Y-%m'),
        'prediction': {
            'point_estimate': round(prediction, 0),
            'unit': 'thousands of jobs (MoM change)',
            'confidence_intervals': {
                '50%': [round(intervals[0.50][0], 0), round(intervals[0.50][1], 0)],
                '80%': [round(intervals[0.80][0], 0), round(intervals[0.80][1], 0)],
                '95%': [round(intervals[0.95][0], 0), round(intervals[0.95][1], 0)],
            }
        },
        'model_info': {
            'features_used': len(available_cols),
            'features_missing': len(missing_cols),
            'residuals_count': len(residuals),
        },
        'snapshot_date': snapshot_date.strftime('%Y-%m')
    }
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved report to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"NFP PREDICTION: {target_month.strftime('%B %Y')}")
    print(f"{'='*60}")
    print(f"Type: {target_type.upper()}")
    print(f"Point Estimate: {prediction:+,.0f}K jobs")
    print(f"\nConfidence Intervals:")
    print(f"  50%: [{intervals[0.50][0]:+,.0f}K, {intervals[0.50][1]:+,.0f}K]")
    print(f"  80%: [{intervals[0.80][0]:+,.0f}K, {intervals[0.80][1]:+,.0f}K]")
    print(f"  95%: [{intervals[0.95][0]:+,.0f}K, {intervals[0.95][1]:+,.0f}K]")
    print(f"\nGenerated: {result['timestamp']}")
    print(f"{'='*60}")
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate NFP prediction for upcoming month"
    )
    parser.add_argument(
        '--target', type=str, default='nsa',
        choices=['nsa', 'sa'],
        help="Target type (default: nsa)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output path for JSON report"
    )
    
    args = parser.parse_args()
    
    result = generate_prediction(
        target_type=args.target,
        output_path=args.output
    )
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
