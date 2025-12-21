"""
Next Month NFP Prediction
=========================

Predicts the next month's NFP MoM change given END_DATE in .env

This script:
1. Loads the most recent trained models for NSA and SA targets
2. Loads the snapshot for the prediction month (END_DATE)
3. Generates features using the same feature engineering as training
4. Makes predictions with confidence intervals
5. Saves predictions to output file

Usage:
    python Predict/predict_next_month.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger, END_DATE
from Train.snapshot_loader import load_target_data
from Train.train_lightgbm_nfp import (
    load_model,
    build_training_dataset,
    MODEL_SAVE_DIR
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.error("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False
    sys.exit(1)


def get_prediction_month():
    """
    Determine the prediction month based on most recent available target data.

    Returns:
        prediction_month as pd.Timestamp
    """
    # Load target data to find the most recent month
    target_data = load_target_data(target_type='nsa')
    most_recent_month = target_data['ds'].max()

    # The prediction month is the month after the most recent data
    prediction_month = (most_recent_month + pd.offsets.MonthBegin(1)).replace(day=1)

    logger.info(f"Most recent target data: {most_recent_month.strftime('%Y-%m')}")
    logger.info(f"Prediction target: {prediction_month.strftime('%Y-%m')}")
    logger.info(f"(Snapshot for {prediction_month.strftime('%Y-%m')} must exist)")

    return prediction_month


def build_prediction_features(
    prediction_month: pd.Timestamp,
    target_type: str = 'nsa'
) -> pd.DataFrame:
    """
    Build features for the prediction month using build_training_dataset.

    Args:
        prediction_month: Month we're predicting (e.g., 2025-12-01)
        target_type: 'nsa' or 'sa'

    Returns:
        DataFrame with one row containing all features
    """
    logger.info(f"Building features for {prediction_month.strftime('%Y-%m')}...")

    # Load target data (we need a dummy target value since we're predicting)
    target_data = load_target_data(target_type=target_type)

    # Create a dummy target row for the prediction month
    # This is needed so build_training_dataset processes it, but the y value doesn't matter
    dummy_target = pd.DataFrame({
        'ds': [prediction_month],
        'y_mom': [0.0]  # Dummy value, not used
    })

    # Build features using the same function as training
    # This ensures consistency in feature engineering
    X_features, _ = build_training_dataset(
        target_df=dummy_target,
        target_type=target_type,
        start_date=prediction_month,
        end_date=prediction_month
    )

    if X_features.empty:
        raise ValueError(f"Failed to build features for {prediction_month.strftime('%Y-%m')}")

    logger.info(f"Built {len(X_features.columns)} features")

    return X_features


def predict_with_model(
    model: lgb.Booster,
    metadata: dict,
    features: pd.DataFrame,
    prediction_month: pd.Timestamp,
    target_type: str
) -> dict:
    """
    Make prediction using the trained model.

    Args:
        model: Trained LightGBM model
        metadata: Model metadata with feature columns and residuals
        features: Feature DataFrame for prediction
        prediction_month: Month being predicted
        target_type: 'nsa' or 'sa'

    Returns:
        Dictionary with prediction results
    """
    # Get feature columns from model metadata
    feature_cols = metadata['feature_cols']

    # Check for missing features
    missing_features = [f for f in feature_cols if f not in features.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features, will be imputed as 0")
        for feat in missing_features:
            features[feat] = 0

    # Select and order features to match model
    X_pred = features[feature_cols]

    # Make prediction
    prediction = model.predict(X_pred)[0]

    # Calculate prediction intervals using historical residuals
    residuals = metadata.get('residuals', [])
    if len(residuals) > 10:
        residual_array = np.array(residuals[-36:])  # Use last 36 months
        lower_50 = prediction + np.percentile(residual_array, 25)
        upper_50 = prediction + np.percentile(residual_array, 75)
        lower_80 = prediction + np.percentile(residual_array, 10)
        upper_80 = prediction + np.percentile(residual_array, 90)
        lower_95 = prediction + np.percentile(residual_array, 2.5)
        upper_95 = prediction + np.percentile(residual_array, 97.5)
    else:
        # Fallback to standard deviation estimate
        std_residual = metadata.get('std_residual', 200)
        lower_50, upper_50 = prediction - 0.67*std_residual, prediction + 0.67*std_residual
        lower_80, upper_80 = prediction - 1.28*std_residual, prediction + 1.28*std_residual
        lower_95, upper_95 = prediction - 1.96*std_residual, prediction + 1.96*std_residual

    result = {
        'prediction_month': prediction_month.strftime('%Y-%m'),
        'target_type': target_type,
        'prediction': round(prediction, 1),
        'lower_50': round(lower_50, 1),
        'upper_50': round(upper_50, 1),
        'lower_80': round(lower_80, 1),
        'upper_80': round(upper_80, 1),
        'lower_95': round(lower_95, 1),
        'upper_95': round(upper_95, 1),
        'model_trained_at': metadata.get('trained_at', 'unknown'),
        'n_features': len(feature_cols)
    }

    return result


def save_prediction(results: dict, output_dir: Path):
    """
    Save prediction results to CSV and JSON files.

    Args:
        results: Dictionary with NSA and SA predictions
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df_results = pd.DataFrame([
        {
            'prediction_month': results['nsa']['prediction_month'],
            'target_type': 'NSA',
            'prediction': results['nsa']['prediction'],
            'lower_50': results['nsa']['lower_50'],
            'upper_50': results['nsa']['upper_50'],
            'lower_80': results['nsa']['lower_80'],
            'upper_80': results['nsa']['upper_80'],
            'lower_95': results['nsa']['lower_95'],
            'upper_95': results['nsa']['upper_95'],
            'n_features': results['nsa']['n_features']
        },
        {
            'prediction_month': results['sa']['prediction_month'],
            'target_type': 'SA',
            'prediction': results['sa']['prediction'],
            'lower_50': results['sa']['lower_50'],
            'upper_50': results['sa']['upper_50'],
            'lower_80': results['sa']['lower_80'],
            'upper_80': results['sa']['upper_80'],
            'lower_95': results['sa']['lower_95'],
            'upper_95': results['sa']['upper_95'],
            'n_features': results['sa']['n_features']
        }
    ])

    # Save CSV
    csv_path = output_dir / f"next_month_prediction_{results['nsa']['prediction_month']}.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions to {csv_path}")

    # Save detailed JSON with metadata
    import json
    json_path = output_dir / f"next_month_prediction_{results['nsa']['prediction_month']}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {json_path}")

    return csv_path


def main():
    """Main prediction workflow."""
    logger.info("=" * 80)
    logger.info("NFP NEXT MONTH PREDICTION")
    logger.info("=" * 80)

    # Get prediction month
    prediction_month = get_prediction_month()

    results = {}

    # Predict for both NSA and SA targets
    for target_type in ['nsa', 'sa']:
        logger.info("\n" + "=" * 80)
        logger.info(f"PREDICTING {target_type.upper()} TARGET")
        logger.info("=" * 80)

        try:
            # Load model
            logger.info(f"Loading {target_type.upper()} model...")
            model, metadata = load_model(save_dir=MODEL_SAVE_DIR, target_type=target_type)
            logger.info(f"Model trained at: {metadata.get('trained_at', 'unknown')}")
            logger.info(f"Model uses {len(metadata['feature_cols'])} features")

            # Build features
            features = build_prediction_features(
                prediction_month,
                target_type=target_type
            )

            # Make prediction
            result = predict_with_model(
                model,
                metadata,
                features,
                prediction_month,
                target_type
            )

            results[target_type] = result

            # Display results
            logger.info("\n" + "-" * 40)
            logger.info(f"{target_type.upper()} PREDICTION RESULTS")
            logger.info("-" * 40)
            logger.info(f"Prediction month: {result['prediction_month']}")
            logger.info(f"Point prediction: {result['prediction']:,.0f}K jobs")
            logger.info(f"50% confidence: [{result['lower_50']:,.0f}, {result['upper_50']:,.0f}]")
            logger.info(f"80% confidence: [{result['lower_80']:,.0f}, {result['upper_80']:,.0f}]")
            logger.info(f"95% confidence: [{result['lower_95']:,.0f}, {result['upper_95']:,.0f}]")

        except Exception as e:
            logger.error(f"Error predicting {target_type.upper()}: {e}", exc_info=True)
            raise

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    output_dir = OUTPUT_DIR / "predictions"
    csv_path = save_prediction(results, output_dir)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Prediction month: {prediction_month.strftime('%Y-%m')}")
    logger.info(f"NSA prediction: {results['nsa']['prediction']:,.0f}K jobs")
    logger.info(f"SA prediction:  {results['sa']['prediction']:,.0f}K jobs")
    logger.info(f"\nResults saved to: {csv_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
