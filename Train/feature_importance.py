"""
Feature Importance Analysis for NFP Predictor

Analyzes feature importance across trained models to:
1. Identify most predictive features (LightGBM gain + SHAP)
2. Generate feature importance reports
3. Summarize importance by category
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import (
    MODEL_SAVE_DIR,
    PROTECTED_BINARY_FLAGS,
    LINEAR_BASELINE_PREDICTORS,
)

logger = setup_logger(__file__, TEMP_DIR)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def load_trained_model(
    model_dir: Path = MODEL_SAVE_DIR,
    target_type: str = 'nsa'
) -> Tuple[Optional[lgb.Booster], Optional[Dict]]:
    """
    Load trained model and metadata.

    Args:
        model_dir: Directory containing saved model
        target_type: 'nsa' or 'sa'

    Returns:
        Tuple of (model, metadata) or (None, None) if not found
    """
    model_path = model_dir / f"lightgbm_{target_type}_model.txt"
    metadata_path = model_dir / f"lightgbm_{target_type}_metadata.pkl"

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None, None

    import pickle

    model = lgb.Booster(model_file=str(model_path))

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata


def get_feature_importance(
    model: lgb.Booster,
    feature_names: List[str],
    importance_type: str = 'gain'
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature column names
        importance_type: 'gain' (default) or 'split'

    Returns:
        DataFrame with feature names and importance scores
    """
    importance = model.feature_importance(importance_type=importance_type)

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Normalize to percentages
    total = df['importance'].sum()
    if total > 0:
        df['importance_pct'] = (df['importance'] / total) * 100
    else:
        df['importance_pct'] = 0

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_shap_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    feature_names: List[str]
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP-based feature importance.

    Args:
        model: Trained LightGBM Booster
        X: Feature DataFrame for computing SHAP values
        feature_names: List of feature column names

    Returns:
        DataFrame with mean |SHAP| per feature, or None if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available for importance analysis")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        X_features = X[feature_names] if feature_names else X
        shap_values = explainer.shap_values(X_features)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        return shap_df

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def categorize_features(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature category based on naming patterns.

    Categories:
    - nfp_*: Past NFP values
    - emp_*: Employment sector data
    - VIX_*, SP500_*, etc.: Market indicators
    - CCSA_*: Claims data
    - calendar: Month, quarter, survey weeks
    """
    def get_category(feature: str) -> str:
        feature_lower = feature.lower()

        if feature_lower.startswith('nfp_'):
            return 'NFP_History'
        elif feature_lower.startswith('emp_'):
            return 'Employment_Sectors'
        elif feature_lower.startswith('vix_'):
            return 'VIX_Volatility'
        elif feature_lower.startswith('sp500_'):
            return 'SP500_Market'
        elif feature_lower.startswith('ccsa_'):
            return 'Weekly_Claims'
        elif feature_lower.startswith('oil_'):
            return 'Oil_Prices'
        elif feature_lower.startswith('yield_'):
            return 'Yield_Curve'
        elif feature_lower.startswith('credit_'):
            return 'Credit_Spreads'
        elif any(x in feature_lower for x in ['month', 'quarter', 'survey', 'january', 'december']):
            return 'Calendar'
        elif feature_lower.startswith('adp_'):
            return 'ADP_Forecast'
        elif feature_lower.startswith('ism_'):
            return 'ISM_Surveys'
        elif feature_lower.startswith('noaa_'):
            return 'NOAA_Weather'
        else:
            return 'Other'

    importance_df['category'] = importance_df['feature'].apply(get_category)
    return importance_df


def summarize_by_category(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize feature importance by category.

    Returns:
        DataFrame with category-level importance statistics
    """
    summary = importance_df.groupby('category').agg({
        'feature': 'count',
        'importance': 'sum',
        'importance_pct': 'sum'
    }).rename(columns={
        'feature': 'num_features',
        'importance': 'total_importance',
        'importance_pct': 'total_pct'
    })

    summary = summary.sort_values('total_importance', ascending=False)
    return summary


def generate_importance_report(
    target_type: str = 'nsa',
    save_dir: Optional[Path] = None
) -> Dict:
    """
    Generate comprehensive feature importance report.

    Includes LightGBM gain importance and SHAP importance if available.

    Args:
        target_type: 'nsa' or 'sa'
        save_dir: Directory to save report (default: OUTPUT_DIR/reports)

    Returns:
        Dictionary with report content
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating feature importance report for {target_type.upper()} model")

    # Load model
    model, metadata = load_trained_model(target_type=target_type)

    if model is None:
        logger.error("Could not load model. Train a model first.")
        return {'error': 'Model not found'}

    feature_cols = metadata.get('feature_cols', [])

    # Get importance from model
    importance_df = get_feature_importance(model, feature_cols)
    importance_df = categorize_features(importance_df)

    # Category summary
    category_summary = summarize_by_category(importance_df)

    # Try to load SHAP importance if previously computed
    shap_path = OUTPUT_DIR / "shap_analysis" / f"{target_type}_first" / f"shap_importance_{target_type}_first.csv"
    shap_info = {}
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        importance_df = importance_df.merge(shap_df[['feature', 'mean_abs_shap']], on='feature', how='left')
        shap_info = {
            'shap_top_20': shap_df.head(20)[['feature', 'mean_abs_shap']].to_dict('records'),
            'shap_available': True
        }
        logger.info(f"Merged SHAP importance from {shap_path}")
    else:
        shap_info = {'shap_available': False}

    # Build report
    report = {
        'target_type': target_type,
        'total_features': len(feature_cols),
        'top_20_features': importance_df.head(20)[['feature', 'importance_pct', 'category']].to_dict('records'),
        'category_summary': category_summary.to_dict(),
        **shap_info,
    }

    # Save detailed CSV
    csv_path = save_dir / f"feature_importance_{target_type}.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed importance to {csv_path}")

    # Save summary JSON
    json_path = save_dir / f"feature_importance_summary_{target_type}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved summary to {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Feature Importance Report: {target_type.upper()}")
    print(f"{'='*60}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nTop 10 Features:")
    print(importance_df.head(10)[['feature', 'importance_pct', 'category']].to_string(index=False))
    print(f"\nCategory Summary:")
    print(category_summary.to_string())
    print(f"\nReports saved to {save_dir}")

    return report


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run feature importance analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze feature importance for NFP models")
    parser.add_argument('--target', type=str, default='nsa', choices=['nsa', 'sa'],
                        help="Target type (default: nsa)")
    parser.add_argument('--output', type=str, default=None,
                        help="Output directory for reports")

    args = parser.parse_args()

    save_dir = Path(args.output) if args.output else None
    report = generate_importance_report(target_type=args.target, save_dir=save_dir)

    if 'error' in report:
        print(f"\nError: {report['error']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
