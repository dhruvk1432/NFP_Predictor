"""
Analyze NOAA Weighted Aggregation Vintage Availability

Downloads state employment vintages and analyzes the earliest available dates
to understand the impact of the fallback logic in create_noaa_weighted.py.

This script helps answer: How much historical data uses the fallback to earliest
vintage instead of point-in-time correct weights?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fredapi import Fred
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, FRED_API_KEY, START_DATE, END_DATE, TEMP_DIR, setup_logger

logger = setup_logger(__file__, TEMP_DIR)

# State name to FRED code mapping (same as in create_noaa_weighted.py)
STATE_NAME_TO_CODE = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI',
    'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX',
    'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
    'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
    'DISTRICT OF COLUMBIA': 'DC'
}


def download_state_employment_vintages(fred: Fred, end_date: str = END_DATE):
    """
    Download ALL vintages for all state employment series.
    Identical logic to create_noaa_weighted.py to ensure consistency.
    """
    logger.info(f"Downloading state employment vintages as of {end_date}...")
    
    all_vintages = []
    
    for state_name, state_code in STATE_NAME_TO_CODE.items():
        series_id = f"{state_code}NA"  # e.g., ALNA, AKNA, etc.
        
        try:
            # Download all vintages for this series
            df = fred.get_series_all_releases(series_id)
            
            if df.empty:
                logger.warning(f"  {state_code}: No data available")
                continue
            
            df = df.reset_index()
            # The fredapi get_series_all_releases returns columns: 'date', 'realtime_start', 'realtime_end', 'value'
            # We only need 'date', 'realtime_start', 'value'
            df.columns = ['date', 'realtime_start', 'realtime_end', 'value']
            df = df[['date', 'realtime_start', 'value']] # Select only the desired columns
            
            df['state_code'] = state_code
            df['state_name'] = state_name
            
            # Convert to timestamps
            df['date'] = pd.to_datetime(df['date'])
            df['realtime_start'] = pd.to_datetime(df['realtime_start'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce') # Ensure value is numeric
            
            all_vintages.append(df)
            logger.info(f"  {state_code}: {len(df)} vintage records")
            
        except Exception as e:
            logger.error(f"  {state_code}: Error - {e}")
            continue
    
    if not all_vintages:
        raise RuntimeError("No state employment data downloaded!")
    
    vintages_df = pd.concat(all_vintages, ignore_index=True)
    logger.info(f"\nTotal vintage records downloaded: {len(vintages_df):,}")
    
    return vintages_df


def analyze_earliest_vintages(vintages_df: pd.DataFrame):
    """
    Analyze the earliest available vintage date for each state.
    """
    logger.info("\n" + "="*70)
    logger.info("EARLIEST VINTAGE ANALYSIS BY STATE")
    logger.info("="*70)
    
    # Find earliest vintage date for each state
    earliest_by_state = vintages_df.groupby('state_code')['realtime_start'].min().reset_index()
    earliest_by_state.columns = ['state_code', 'earliest_vintage']
    
    # Sort by earliest date
    earliest_by_state = earliest_by_state.sort_values('earliest_vintage')
    
    logger.info(f"\n{'State':<6} {'Earliest Vintage':<20} {'Days Since 1990-01-01'}")
    logger.info("-" * 60)
    
    reference_date = pd.Timestamp('1990-01-01')
    
    for _, row in earliest_by_state.iterrows():
        state = row['state_code']
        earliest = row['earliest_vintage']
        days_since_1990 = (earliest - reference_date).days
        
        logger.info(f"{state:<6} {earliest.date()!s:<20} {days_since_1990:>6,} days")
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    
    overall_earliest = earliest_by_state['earliest_vintage'].min()
    overall_latest = earliest_by_state['earliest_vintage'].max()
    median_earliest = earliest_by_state['earliest_vintage'].median()
    
    logger.info(f"Overall earliest vintage: {overall_earliest.date()}")
    logger.info(f"Overall latest (among earliest): {overall_latest.date()}")
    logger.info(f"Median earliest vintage: {median_earliest.date()}")
    
    # Count states by vintage era
    pre_2000 = (earliest_by_state['earliest_vintage'] < '2000-01-01').sum()
    pre_2005 = (earliest_by_state['earliest_vintage'] < '2005-01-01').sum()
    pre_2010 = (earliest_by_state['earliest_vintage'] < '2010-01-01').sum()
    
    logger.info(f"\nStates with vintages before 2000: {pre_2000}/52")
    logger.info(f"States with vintages before 2005: {pre_2005}/52")
    logger.info(f"States with vintages before 2010: {pre_2010}/52")
    
    return earliest_by_state


def analyze_fallback_impact(vintages_df: pd.DataFrame, earliest_by_state: pd.DataFrame):
    """
    Analyze how many snapshots would be affected by fallback logic.
    
    Fallback logic in create_noaa_weighted.py:
    - If snap_date < earliest vintage → use earliest vintage (LOOKAHEAD BIAS)
    """
    logger.info("\n" + "="*70)
    logger.info("FALLBACK IMPACT ANALYSIS")
    logger.info("="*70)
    
    # Create monthly snapshot dates from START_DATE to END_DATE
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    
    snapshot_dates = pd.date_range(start, end, freq='MS')
    
    logger.info(f"\nAnalyzing {len(snapshot_dates)} monthly snapshots")
    logger.info(f"Date range: {snapshot_dates[0].date()} to {snapshot_dates[-1].date()}")
    
    # For each state, count how many snapshots would trigger fallback
    fallback_counts = []
    
    for _, row in earliest_by_state.iterrows():
        state = row['state_code']
        earliest_vintage = row['earliest_vintage']
        
        # Count snapshots before earliest vintage
        affected_snapshots = (snapshot_dates < earliest_vintage).sum()
        pct_affected = (affected_snapshots / len(snapshot_dates)) * 100
        
        fallback_counts.append({
            'state_code': state,
            'earliest_vintage': earliest_vintage,
            'affected_snapshots': affected_snapshots,
            'total_snapshots': len(snapshot_dates),
            'pct_affected': pct_affected
        })
    
    fallback_df = pd.DataFrame(fallback_counts)
    fallback_df = fallback_df.sort_values('pct_affected', ascending=False)
    
    # Display states with highest fallback impact
    logger.info(f"\nStates with HIGHEST fallback impact:")
    logger.info(f"{'State':<6} {'Earliest':<12} {'Affected':<10} {'Total':<8} {'% Affected'}")
    logger.info("-" * 60)
    
    for _, row in fallback_df.head(10).iterrows():
        logger.info(
            f"{row['state_code']:<6} "
            f"{row['earliest_vintage'].date()!s:<12} "
            f"{row['affected_snapshots']:<10} "
            f"{row['total_snapshots']:<8} "
            f"{row['pct_affected']:>6.1f}%"
        )
    
    # Overall impact
    logger.info("\n" + "="*70)
    logger.info("OVERALL IMPACT")
    logger.info("="*70)
    
    # Find the worst case (state with latest earliest vintage)
    worst_state = fallback_df.iloc[0]
    
    logger.info(f"\nWorst case state: {worst_state['state_code']}")
    logger.info(f"  Earliest vintage: {worst_state['earliest_vintage'].date()}")
    logger.info(f"  Affected snapshots: {worst_state['affected_snapshots']}/{worst_state['total_snapshots']}")
    logger.info(f"  Percentage: {worst_state['pct_affected']:.1f}%")
    
    # Average impact across all states
    avg_pct = fallback_df['pct_affected'].mean()
    logger.info(f"\nAverage % of snapshots affected (across all states): {avg_pct:.1f}%")
    
    # Timeline analysis
    logger.info("\nSnapshot periods with MOST fallback usage:")
    
    # For each snapshot, count how many states would use fallback
    snapshot_analysis = []
    for snap_date in snapshot_dates[:60]:  # First 5 years only for readability
        states_using_fallback = (earliest_by_state['earliest_vintage'] > snap_date).sum()
        pct_states = (states_using_fallback / len(earliest_by_state)) * 100
        
        snapshot_analysis.append({
            'snapshot_date': snap_date,
            'states_using_fallback': states_using_fallback,
            'pct_states': pct_states
        })
    
    snap_df = pd.DataFrame(snapshot_analysis)
    snap_df = snap_df.sort_values('pct_states', ascending=False)
    
    logger.info(f"{'Snapshot Date':<15} {'States Using Fallback':<25} {'% of States'}")
    logger.info("-" * 60)
    
    for _, row in snap_df.head(10).iterrows():
        logger.info(
            f"{row['snapshot_date'].date()!s:<15} "
            f"{row['states_using_fallback']:<25} "
            f"{row['pct_states']:>6.1f}%"
        )
    
    return fallback_df


def main():
    """Main analysis workflow."""
    logger.info("="*70)
    logger.info("NOAA WEIGHT VINTAGE AVAILABILITY ANALYSIS")
    logger.info("="*70)
    
    # Initialize FRED API
    fred = Fred(api_key=FRED_API_KEY)
    
    # 1. Download vintages
    vintages_df = download_state_employment_vintages(fred, end_date=END_DATE)
    
    # 2. Analyze earliest vintages by state
    earliest_by_state = analyze_earliest_vintages(vintages_df)
    
    # 3. Analyze fallback impact
    fallback_df = analyze_fallback_impact(vintages_df, earliest_by_state)
    
    # 4. Save results
    output_dir = DATA_PATH / "_temp"
    output_dir.mkdir(exist_ok=True)
    
    vintages_file = output_dir / "state_employment_vintages.parquet"
    earliest_file = output_dir / "state_earliest_vintages.csv"
    fallback_file = output_dir / "fallback_impact_analysis.csv"
    
    vintages_df.to_parquet(vintages_file, index=False)
    earliest_by_state.to_csv(earliest_file, index=False)
    fallback_df.to_csv(fallback_file, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS SAVED")
    logger.info("="*70)
    logger.info(f"Vintages data: {vintages_file}")
    logger.info(f"Earliest vintages: {earliest_file}")
    logger.info(f"Fallback analysis: {fallback_file}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
