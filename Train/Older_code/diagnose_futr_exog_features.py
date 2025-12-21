"""
Comprehensive Feature Diagnostics for futr_exog

Analyzes all 149 futr_exog features to identify:
- Missing value patterns
- Start/end dates
- Statistical properties (mean, std, skew, kurtosis)
- Outliers and data quality issues
- Potential issues for LightGBM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.feature_assembly import prepare_nbeats_training_data

logger = setup_logger(__file__, TEMP_DIR)

def compute_feature_diagnostics(train_df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Compute comprehensive diagnostics for each feature.
    
    Args:
        train_df: Training data DataFrame
        feature_list: List of feature column names
        
    Returns:
        DataFrame with diagnostic statistics per feature
    """
    diagnostics = []
    
    for feature in feature_list:
        if feature not in train_df.columns:
            logger.warning(f"Feature {feature} not found in data")
            continue
            
        # Get feature data (all series combined)
        data = train_df[feature].values
        
        # Basic stats
        diag = {
            'feature_name': feature,
            'total_values': len(data),
            'missing_count': np.isnan(data).sum(),
            'missing_pct': (np.isnan(data).sum() / len(data)) * 100,
            'non_zero_count': np.count_nonzero(~np.isnan(data) & (data != 0)),
            'zero_count': np.count_nonzero(data == 0),
            'zero_pct': (np.count_nonzero(data == 0) / len(data)) * 100
        }
        
        # Valid (non-NaN) data
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            diag['mean'] = np.mean(valid_data)
            diag['std'] = np.std(valid_data)
            diag['min'] = np.min(valid_data)
            diag['max'] = np.max(valid_data)
            diag['median'] = np.median(valid_data)
            diag['q25'] = np.percentile(valid_data, 25)
            diag['q75'] = np.percentile(valid_data, 75)
            diag['iqr'] = diag['q75'] - diag['q25']
            
            # Skewness and kurtosis
            if len(valid_data) > 2:
                diag['skewness'] = stats.skew(valid_data)
                diag['kurtosis'] = stats.kurtosis(valid_data)
            else:
                diag['skewness'] = np.nan
                diag['kurtosis'] = np.nan
            
            # Outliers (using IQR method)
            if diag['iqr'] > 0:
                lower_bound = diag['q25'] - 1.5 * diag['iqr']
                upper_bound = diag['q75'] + 1.5 * diag['iqr']
                outliers = (valid_data < lower_bound) | (valid_data > upper_bound)
                diag['outlier_count'] = np.sum(outliers)
                diag['outlier_pct'] = (np.sum(outliers) / len(valid_data)) * 100
            else:
                diag['outlier_count'] = 0
                diag['outlier_pct'] = 0.0
            
            # Coefficient of variation (relative variability)
            if diag['mean'] != 0:
                diag['coef_variation'] = (diag['std'] / abs(diag['mean'])) * 100
            else:
                diag['coef_variation'] = np.nan
                
        else:
            # All NaN
            for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 
                       'iqr', 'skewness', 'kurtosis', 'outlier_count', 
                       'outlier_pct', 'coef_variation']:
                diag[key] = np.nan
        
        # Determine feature type and category
        if 'sin' in feature or 'cos' in feature:
            diag['category'] = 'Fourier'
        elif any(p in feature for p in ['Oil', 'Credit', 'Yield']):
            diag['category'] = 'Daily Market'
        elif any(p in feature for p in ['ADP', 'ICSA', 'CCSA', 'ISM', 'Challenger']):
            diag['category'] = 'Fast Release'
        elif '_lag1' in feature:
            diag['category'] = 'Slow Release (lag1)'
        else:
            diag['category'] = 'Other'
        
        # Data quality flags
        diag['has_high_missing'] = diag['missing_pct'] > 10.0
        diag['has_high_zeros'] = diag['zero_pct'] > 50.0
        diag['has_extreme_skew'] = abs(diag['skewness']) > 2.0 if not np.isnan(diag['skewness']) else False
        diag['has_high_kurtosis'] = abs(diag['kurtosis']) > 7.0 if not np.isnan(diag['kurtosis']) else False
        diag['has_many_outliers'] = diag['outlier_pct'] > 5.0
        
        diagnostics.append(diag)
    
    return pd.DataFrame(diagnostics)


def analyze_temporal_availability(train_df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Analyze when each feature starts being available.
    
    Args:
        train_df: Training data with 'ds' column
        feature_list: List of features
        
    Returns:
        DataFrame with temporal availability info
    """
    temporal_info = []
    
    for feature in feature_list:
        if feature not in train_df.columns:
            continue
            
        # Group by date to see availability over time
        by_date = train_df.groupby('ds')[feature].agg([
            ('non_null_count', lambda x: x.notna().sum()),
            ('total_count', 'count'),
            ('mean_value', 'mean')
        ]).reset_index()
        
        # Find first and last non-null dates
        non_null_dates = by_date[by_date['non_null_count'] > 0]
        
        if len(non_null_dates) > 0:
            info = {
                'feature_name': feature,
                'first_available': non_null_dates['ds'].min(),
                'last_available': non_null_dates['ds'].max(),
                'total_months': len(non_null_dates),
                'months_with_data': len(non_null_dates[non_null_dates['non_null_count'] > 0]),
                'coverage_pct': (len(non_null_dates[non_null_dates['non_null_count'] > 0]) / len(by_date)) * 100
            }
        else:
            info = {
                'feature_name': feature,
                'first_available': pd.NaT,
                'last_available': pd.NaT,
                'total_months': 0,
                'months_with_data': 0,
                'coverage_pct': 0.0
            }
        
        temporal_info.append(info)
    
    return pd.DataFrame(temporal_info)


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("COMPREHENSIVE FEATURE DIAGNOSTICS FOR FUTR_EXOG")
    logger.info("="*70)
    
    # Use a recent snapshot with good data coverage
    test_date = pd.Timestamp('2024-01-31')
    logger.info(f"\nAnalyzing features from snapshot: {test_date.date()}")
    
    # Load training data
    logger.info("\nStep 1: Loading training data...")
    train_df, feature_lists = prepare_nbeats_training_data(
        snapshot_date=test_date,
        lookback_months=120  # 10 years of data
    )
    
    futr_exog_list = feature_lists['futr_exog_list']
    logger.info(f"Found {len(futr_exog_list)} futr_exog features")
    
    # Compute statistical diagnostics
    logger.info("\nStep 2: Computing statistical diagnostics...")
    stat_diagnostics = compute_feature_diagnostics(train_df, futr_exog_list)
    
    # Compute temporal availability
    logger.info("\nStep 3: Analyzing temporal availability...")
    temporal_diagnostics = analyze_temporal_availability(train_df, futr_exog_list)
    
    # Merge diagnostics
    full_diagnostics = stat_diagnostics.merge(
        temporal_diagnostics, 
        on='feature_name', 
        how='left'
    )
    
    # Sort by category and feature name
    full_diagnostics = full_diagnostics.sort_values(['category', 'feature_name'])
    
    # Save to CSV
    output_dir = OUTPUT_DIR / "feature_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "futr_exog_diagnostics.csv"
    full_diagnostics.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved full diagnostics to: {output_file}")
    
    # Create summary by category
    category_summary = full_diagnostics.groupby('category').agg({
        'feature_name': 'count',
        'missing_pct': 'mean',
        'zero_pct': 'mean',
        'skewness': lambda x: x.abs().mean(),
        'kurtosis': lambda x: x.abs().mean(),
        'has_high_missing': 'sum',
        'has_extreme_skew': 'sum',
        'has_high_kurtosis': 'sum',
        'has_many_outliers': 'sum'
    }).round(2)
    
    category_summary.columns = [
        'feature_count', 
        'avg_missing_pct', 
        'avg_zero_pct',
        'avg_abs_skew',
        'avg_abs_kurtosis',
        'num_high_missing',
        'num_extreme_skew',
        'num_high_kurtosis',
        'num_many_outliers'
    ]
    
    summary_file = output_dir / "futr_exog_category_summary.csv"
    category_summary.to_csv(summary_file)
    logger.info(f"✓ Saved category summary to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("FEATURE CATEGORY SUMMARY")
    print("="*70)
    print(category_summary.to_string())
    
    # Identify problematic features
    print("\n" + "="*70)
    print("POTENTIAL DATA QUALITY ISSUES")
    print("="*70)
    
    high_missing = full_diagnostics[full_diagnostics['has_high_missing']]
    if len(high_missing) > 0:
        print(f"\n⚠ Features with >10% missing values ({len(high_missing)}):")
        print(high_missing[['feature_name', 'missing_pct', 'category']].to_string(index=False))
    
    extreme_skew = full_diagnostics[full_diagnostics['has_extreme_skew']]
    if len(extreme_skew) > 0:
        print(f"\n⚠ Features with extreme skewness (|skew| > 2) ({len(extreme_skew)}):")
        print(extreme_skew[['feature_name', 'skewness', 'category']].to_string(index=False))
    
    high_kurtosis = full_diagnostics[full_diagnostics['has_high_kurtosis']]
    if len(high_kurtosis) > 0:
        print(f"\n⚠ Features with high kurtosis (|kurt| > 7) ({len(high_kurtosis)}):")
        print(high_kurtosis[['feature_name', 'kurtosis', 'category']].to_string(index=False))
    
    # Print all 149 features in compact format
    print("\n" + "="*70)
    print("ALL 149 FUTR_EXOG FEATURES")
    print("="*70)
    
    for category in ['Fourier', 'Daily Market', 'Fast Release', 'Slow Release (lag1)', 'Other']:
        cat_features = full_diagnostics[full_diagnostics['category'] == category]
        if len(cat_features) > 0:
            print(f"\n{category} ({len(cat_features)} features):")
            print("-" * 70)
            for _, row in cat_features.iterrows():
                print(f"  {row['feature_name']}")
    
    logger.info(f"\n✓ Complete! Check {output_dir} for full reports")
