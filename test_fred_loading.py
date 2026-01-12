"""
Unit tests to diagnose FRED exogenous data loading issues.
Investigates why release_date > snapshot_date filtering occurs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from settings import DATA_PATH, START_DATE, END_DATE

def test_nfp_release_schedule():
    """Test 1: Verify NFP release schedule is loaded correctly."""
    print("=" * 80)
    print("TEST 1: NFP Release Schedule")
    print("=" * 80)

    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    if not nfp_path.exists():
        print(f"❌ FAILED: NFP release schedule not found at {nfp_path}")
        return None

    nfp = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp.columns = ['data_month', 'nfp_release_date']
    nfp['data_month'] = pd.to_datetime(nfp['data_month'])
    nfp['nfp_release_date'] = pd.to_datetime(nfp['nfp_release_date'])
    nfp = nfp.sort_values('nfp_release_date')

    print(f"✓ Loaded {len(nfp)} NFP release dates")
    print(f"\nFirst 5 entries:")
    print(nfp.head())
    print(f"\nLast 5 entries:")
    print(nfp.tail())

    # Check for any issues
    print(f"\nData month range: {nfp['data_month'].min()} to {nfp['data_month'].max()}")
    print(f"Release date range: {nfp['nfp_release_date'].min()} to {nfp['nfp_release_date'].max()}")

    return nfp

def test_recent_snapshot_filtering():
    """Test 2: Examine recent snapshot to see what's being filtered."""
    print("\n" + "=" * 80)
    print("TEST 2: Recent Snapshot Filtering Analysis")
    print("=" * 80)

    # Find most recent snapshot
    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"

    # Get all parquet files
    all_snapshots = list(base_dir.glob("**/*.parquet"))
    if not all_snapshots:
        print("❌ FAILED: No snapshots found")
        return

    # Sort by modification time to get most recent
    all_snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent = all_snapshots[0]

    print(f"Analyzing most recent snapshot: {most_recent}")

    # Extract observation month from filename
    obs_month = most_recent.stem  # e.g., "2024-12"
    print(f"Observation month: {obs_month}")

    # Load snapshot
    snap = pd.read_parquet(most_recent)

    print(f"\nSnapshot shape: {snap.shape}")
    print(f"Columns: {snap.columns.tolist()}")

    # Get the snapshot_date used
    if 'snapshot_date' in snap.columns:
        snapshot_date = pd.to_datetime(snap['snapshot_date'].iloc[0])
        print(f"\nSnapshot date used: {snapshot_date}")
    else:
        print("❌ WARNING: snapshot_date column not found")
        return

    # Check release_date distribution
    snap['release_date'] = pd.to_datetime(snap['release_date'])

    print(f"\nRelease date statistics:")
    print(f"  Min: {snap['release_date'].min()}")
    print(f"  Max: {snap['release_date'].max()}")
    print(f"  Median: {snap['release_date'].median()}")

    # Count how many would be filtered
    future_releases = snap[snap['release_date'] >= snapshot_date]
    valid_releases = snap[snap['release_date'] < snapshot_date]

    print(f"\n📊 Filtering results:")
    print(f"  Valid (release < snapshot): {len(valid_releases)} rows")
    print(f"  Would be filtered (release >= snapshot): {len(future_releases)} rows")

    if len(future_releases) > 0:
        print(f"\n⚠️  ISSUE DETECTED: {len(future_releases)} rows have release_date >= snapshot_date")
        print(f"\nBreakdown by series:")
        print(future_releases.groupby('series_name').size().sort_values(ascending=False))

        print(f"\nSample of problematic rows:")
        sample = future_releases[['date', 'series_name', 'release_date', 'snapshot_date']].head(10)
        print(sample)

        # Check if these are daily series with forced 1-day lag
        daily_series = future_releases[future_releases['series_name'].str.contains('VIX|SP500|Oil|Credit|Yield', na=False)]
        if not daily_series.empty:
            print(f"\n🔍 Daily series found in filtered data: {len(daily_series)} rows")
            print("This suggests the 1-day lag logic might be causing issues")

            # Show example
            example = daily_series.head(3)
            for idx, row in example.iterrows():
                print(f"\n  Series: {row['series_name']}")
                print(f"  Date: {row['date']}")
                print(f"  Release: {row['release_date']}")
                print(f"  Snapshot: {row['snapshot_date']}")
                print(f"  Difference: {(pd.to_datetime(row['release_date']) - pd.to_datetime(row['snapshot_date'])).days} days")
    else:
        print("✓ No filtering issues detected")

    return snap

def test_daily_series_logic():
    """Test 3: Verify daily series release date logic."""
    print("\n" + "=" * 80)
    print("TEST 3: Daily Series Release Date Logic")
    print("=" * 80)

    # Load NFP schedule
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    nfp = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp.columns = ['data_month', 'nfp_release_date']
    nfp['data_month'] = pd.to_datetime(nfp['data_month'])
    nfp['nfp_release_date'] = pd.to_datetime(nfp['nfp_release_date'])

    # Take a recent example: December 2024 data released in early January 2025
    example_month = pd.Timestamp('2024-12-01')

    # Find NFP release date for this month
    nfp_release_row = nfp[nfp['data_month'] == example_month]
    if nfp_release_row.empty:
        print(f"No NFP release found for {example_month}")
        return

    nfp_release_date = nfp_release_row.iloc[0]['nfp_release_date']

    print(f"Example: December 2024 NFP data")
    print(f"  Observation month: {example_month.date()}")
    print(f"  NFP release date: {nfp_release_date.date()}")

    # Simulate daily data for the last day of the month
    last_day_of_month = pd.Timestamp('2024-12-31')

    # Current logic: realtime_start = date + 1 day
    forced_release = last_day_of_month + pd.Timedelta(days=1)

    print(f"\n📅 Last trading day scenario:")
    print(f"  Observation date: {last_day_of_month.date()}")
    print(f"  Forced release (date + 1): {forced_release.date()}")
    print(f"  NFP snapshot date: {nfp_release_date.date()}")

    # After monthly resampling, this becomes month-start
    month_start = pd.Timestamp('2024-12-01')
    month_end_release = month_start + pd.offsets.MonthEnd(0)  # 2024-12-31

    print(f"\n📊 After monthly aggregation:")
    print(f"  Month: {month_start.date()}")
    print(f"  Release date assigned: {month_end_release.date()}")
    print(f"  NFP snapshot date: {nfp_release_date.date()}")
    print(f"  Would be filtered? {month_end_release >= nfp_release_date} (release >= snapshot)")

    # The issue: For daily data aggregated to monthly, the release_date is set to month-end
    # But NFP release might be BEFORE month-end (e.g., Jan 3rd for December data)
    if month_end_release >= nfp_release_date:
        print("\n⚠️  PROBLEM IDENTIFIED:")
        print("  Daily data is assigned release_date = month-end")
        print("  But NFP snapshot date might be EARLIER in the month")
        print("  This causes valid data to be filtered out!")
        print("\n💡 SOLUTION:")
        print("  Daily data should use release_date = data_date + 1 day")
        print("  NOT month-end after aggregation")

def test_weekly_series_bucketing():
    """Test 4: Verify weekly claims bucketing logic."""
    print("\n" + "=" * 80)
    print("TEST 4: Weekly Claims NFP Bucketing")
    print("=" * 80)

    # Check a recent snapshot
    base_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data" / "decades"
    all_snapshots = list(base_dir.glob("**/*.parquet"))

    if not all_snapshots:
        print("❌ No snapshots found")
        return

    # Get a recent one
    all_snapshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    snap = pd.read_parquet(all_snapshots[0])

    # Filter to weekly series
    weekly_series = snap[snap['series_name'].str.contains('CCSA|ICSA|IURSA|Financial_Stress|Weekly_Econ', na=False)]

    if weekly_series.empty:
        print("No weekly series found in snapshot")
        return

    print(f"Found {len(weekly_series)} weekly series rows")
    print(f"\nSeries breakdown:")
    print(weekly_series.groupby('series_name').size())

    # Check release dates
    weekly_series['release_date'] = pd.to_datetime(weekly_series['release_date'])
    weekly_series['snapshot_date'] = pd.to_datetime(weekly_series['snapshot_date'])

    # Count future releases
    future = weekly_series[weekly_series['release_date'] >= weekly_series['snapshot_date']]

    if not future.empty:
        print(f"\n⚠️  Found {len(future)} weekly rows with future release dates")
        print("\nBreakdown by series:")
        print(future.groupby('series_name').size())

        print("\nSample:")
        print(future[['date', 'series_name', 'release_date', 'snapshot_date']].head())

def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("FRED EXOGENOUS DATA LOADING DIAGNOSTICS")
    print("=" * 80)

    # Test 1: NFP schedule
    nfp = test_nfp_release_schedule()

    if nfp is not None:
        # Test 2: Recent snapshot
        snap = test_recent_snapshot_filtering()

        # Test 3: Daily series logic
        test_daily_series_logic()

        # Test 4: Weekly series logic
        test_weekly_series_bucketing()

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
