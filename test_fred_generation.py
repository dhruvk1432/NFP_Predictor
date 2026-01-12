"""
Test the snapshot GENERATION process to see where filtering happens.
Simulates a single snapshot creation to debug the filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from settings import DATA_PATH, FRED_API_KEY
from fredapi import Fred

def simulate_snapshot_generation():
    """Simulate generating a snapshot to see what gets filtered."""
    print("=" * 80)
    print("SIMULATING SNAPSHOT GENERATION")
    print("=" * 80)

    # Load NFP schedule
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    nfp_schedule = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp_schedule.columns = ['data_month', 'nfp_release_date']
    nfp_schedule['data_month'] = pd.to_datetime(nfp_schedule['data_month'])
    nfp_schedule['nfp_release_date'] = pd.to_datetime(nfp_schedule['nfp_release_date'])

    # Pick a recent month: December 2024
    obs_month = pd.Timestamp('2024-12-01')
    nfp_release_row = nfp_schedule[nfp_schedule['data_month'] == obs_month]

    if nfp_release_row.empty:
        print(f"No NFP release for {obs_month}")
        return

    snap_date = pd.Timestamp(nfp_release_row.iloc[0]['nfp_release_date'])

    print(f"\nObservation month: {obs_month.date()}")
    print(f"NFP release date (snapshot date): {snap_date.date()}")

    # Initialize FRED
    fred = Fred(api_key=FRED_API_KEY)

    # Test with a daily series (VIX)
    print("\n" + "-" * 80)
    print("Testing DAILY series: VIX")
    print("-" * 80)

    # Fetch VIX data (no revisions, forced 1-day lag)
    series = fred.get_series('VIXCLS')
    df = series.to_frame(name='value')
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])

    # FORCE 1-DAY LAG (as per load_fred_exogenous.py line 425)
    df['realtime_start'] = df['date'] + pd.Timedelta(days=1)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    print(f"\nFetched {len(df)} daily VIX observations")

    # Apply snapshot filtering (line 568 in load_fred_exogenous.py)
    # Changed from <= to strict < to prevent same-day data leakage
    valid = df[df['realtime_start'] < snap_date].copy()

    print(f"After filtering (realtime_start < snap_date): {len(valid)} rows")

    if valid.empty:
        print("❌ No valid data!")
        return

    # Get latest vintage per date
    valid = valid.sort_values(['date', 'realtime_start'])
    latest = valid.drop_duplicates(subset=['date'], keep='last')

    print(f"After deduplication: {len(latest)} rows")

    # Show last few dates
    print(f"\nLast 5 observations:")
    print(latest[['date', 'value', 'realtime_start']].tail())

    # Now do the VIX transformations and monthly aggregation
    sub_df = latest[['date', 'value']].set_index('date').sort_index()

    # Calculate daily changes
    sub_df['daily_chg'] = sub_df['value'].diff()

    # Resample to monthly (line 605)
    monthly_agg = sub_df.resample('MS').agg({
        'value': ['mean', 'max'],
        'daily_chg': 'std'
    })

    if isinstance(monthly_agg.columns, pd.MultiIndex):
        monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

    # Create temp_df with features
    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['VIX_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['VIX_max'] = monthly_agg.get('value_max', np.nan)

    # Melt to long format
    sub_df = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )

    # CRITICAL LINE 671: release_date = date + MonthEnd
    sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)

    print(f"\nAfter monthly aggregation: {len(sub_df)} rows")

    # Show December 2024 row
    dec_2024_rows = sub_df[sub_df['date'] == obs_month]

    if not dec_2024_rows.empty:
        print(f"\nDecember 2024 aggregated data:")
        print(dec_2024_rows[['date', 'series_name', 'release_date']])

        release_date = pd.to_datetime(dec_2024_rows.iloc[0]['release_date'])
        print(f"\n📅 Release date assigned: {release_date.date()}")
        print(f"📅 Snapshot date: {snap_date.date()}")
        print(f"📅 Comparison: release_date < snap_date? {release_date < snap_date}")

        # CRITICAL FILTER (line 1059)
        would_be_filtered = release_date >= snap_date

        if would_be_filtered:
            print(f"\n⚠️  WOULD BE FILTERED OUT!")
            print(f"   Reason: release_date ({release_date.date()}) >= snap_date ({snap_date.date()})")
        else:
            print(f"\n✓ Would be kept (release_date < snap_date)")

    # Now test with a monthly series (JOLTS)
    print("\n" + "-" * 80)
    print("Testing MONTHLY series: JOLTS Quits")
    print("-" * 80)

    # Fetch JOLTS data
    vintage_df = fred.get_series_as_of_date('JTSQUR', as_of_date='2100-01-01')
    vintage_df['date'] = pd.to_datetime(vintage_df['date'])
    vintage_df['realtime_start'] = pd.to_datetime(vintage_df['realtime_start'])
    vintage_df['value'] = pd.to_numeric(vintage_df['value'], errors='coerce')

    print(f"\nFetched {len(vintage_df)} JOLTS vintage observations")

    # Filter to snapshot
    valid = vintage_df[vintage_df['realtime_start'] < snap_date].copy()
    print(f"After filtering (realtime_start < snap_date): {len(valid)} rows")

    # Get latest per month
    valid = valid.sort_values(['date', 'realtime_start'])
    latest = valid.drop_duplicates(subset=['date'], keep='last')

    print(f"After deduplication: {len(latest)} rows")

    # Show last few
    print(f"\nLast 5 observations:")
    print(latest[['date', 'value', 'realtime_start']].tail())

    # Monthly resampling (line 1022)
    sub_df = latest[['date', 'value']].set_index('date').sort_index()
    sub_df = sub_df.resample('MS').last().reset_index()
    sub_df['series_name'] = 'JOLTS_Quits'

    # Release date mapping (line 1026-1034)
    latest_with_month = latest.copy()
    latest_with_month['obs_month'] = latest_with_month['date'].dt.to_period('M').dt.to_timestamp()
    month_release_map = (
        latest_with_month.groupby('obs_month')['realtime_start']
        .min()
        .to_dict()
    )

    sub_df['release_date'] = sub_df['date'].map(month_release_map)

    # Fill missing with 2-month lag
    missing_mask = sub_df['release_date'].isna()
    if missing_mask.any():
        sub_df.loc[missing_mask, 'release_date'] = (
            sub_df.loc[missing_mask, 'date'] + pd.DateOffset(months=2)
        )

    print(f"\nAfter monthly processing: {len(sub_df)} rows")

    # Check December 2024
    dec_rows = sub_df[sub_df['date'] == obs_month]

    if not dec_rows.empty:
        print(f"\nDecember 2024 JOLTS data:")
        print(dec_rows[['date', 'series_name', 'release_date']])

        release_date = pd.to_datetime(dec_rows.iloc[0]['release_date'])
        print(f"\n📅 Release date: {release_date.date()}")
        print(f"📅 Snapshot date: {snap_date.date()}")
        print(f"📅 Comparison: release_date < snap_date? {release_date < snap_date}")

        would_be_filtered = release_date >= snap_date

        if would_be_filtered:
            print(f"\n⚠️  WOULD BE FILTERED OUT!")
        else:
            print(f"\n✓ Would be kept")

if __name__ == "__main__":
    simulate_snapshot_generation()
