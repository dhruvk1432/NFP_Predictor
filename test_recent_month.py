"""
Test snapshot generation for the most recent month (2025-12).
This is where the 71 rows are being filtered.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from settings import DATA_PATH, FRED_API_KEY
from fredapi import Fred

def test_december_2025():
    """Test December 2025 snapshot generation."""
    print("=" * 80)
    print("TESTING DECEMBER 2025 SNAPSHOT (Most Recent)")
    print("=" * 80)

    # Load NFP schedule
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    nfp_schedule = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp_schedule.columns = ['data_month', 'nfp_release_date']
    nfp_schedule['data_month'] = pd.to_datetime(nfp_schedule['data_month'])
    nfp_schedule['nfp_release_date'] = pd.to_datetime(nfp_schedule['nfp_release_date'])

    # December 2025
    obs_month = pd.Timestamp('2025-12-01')
    nfp_release_row = nfp_schedule[nfp_schedule['data_month'] == obs_month]

    if nfp_release_row.empty:
        print(f"❌ No NFP release for {obs_month}")
        return

    snap_date = pd.Timestamp(nfp_release_row.iloc[0]['nfp_release_date'])

    print(f"\n📅 Observation month: {obs_month.date()}")
    print(f"📅 NFP release date (snapshot date): {snap_date.date()}")
    print(f"📅 Today's date: {pd.Timestamp.now().date()}")

    # Initialize FRED
    fred = Fred(api_key=FRED_API_KEY)

    print("\n" + "-" * 80)
    print("Testing VIX (daily series)")
    print("-" * 80)

    # Fetch VIX
    series = fred.get_series('VIXCLS')
    df = series.to_frame(name='value')
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df['realtime_start'] = df['date'] + pd.Timedelta(days=1)

    print(f"\nTotal VIX observations: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Filter to snapshot date
    valid = df[df['realtime_start'] < snap_date].copy()

    print(f"\nAfter filtering (realtime_start < {snap_date.date()}): {len(valid)} rows")

    if valid.empty:
        print("⚠️  NO VALID DATA after filtering!")
        print(f"\nLast 5 observations before filtering:")
        print(df[['date', 'realtime_start']].tail())
        return

    # Get latest per date
    valid = valid.sort_values(['date', 'realtime_start'])
    latest = valid.drop_duplicates(subset=['date'], keep='last')

    print(f"After deduplication: {len(latest)} rows")
    print(f"\nLast observation date: {latest['date'].max().date()}")

    # Monthly aggregation
    sub_df = latest[['date', 'value']].set_index('date').sort_index()
    sub_df['daily_chg'] = sub_df['value'].diff()

    monthly_agg = sub_df.resample('MS').agg({
        'value': ['mean', 'max'],
        'daily_chg': 'std'
    })

    if isinstance(monthly_agg.columns, pd.MultiIndex):
        monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]

    temp_df = pd.DataFrame(index=monthly_agg.index)
    temp_df['VIX_mean'] = monthly_agg.get('value_mean', np.nan)
    temp_df['VIX_max'] = monthly_agg.get('value_max', np.nan)

    sub_df = temp_df.reset_index().melt(
        id_vars=['date'],
        var_name='series_name',
        value_name='value'
    )

    # CRITICAL: release_date assignment
    sub_df['release_date'] = sub_df['date'] + pd.offsets.MonthEnd(0)
    sub_df['snapshot_date'] = snap_date

    print(f"\nAfter monthly aggregation: {len(sub_df)} rows")

    # Check December 2025
    dec_2025_rows = sub_df[sub_df['date'] == obs_month]

    if not dec_2025_rows.empty:
        print(f"\n{'='*80}")
        print(f"DECEMBER 2025 VIX DATA")
        print(f"{'='*80}")
        print(dec_2025_rows[['date', 'series_name', 'value', 'release_date', 'snapshot_date']])

        release_date = pd.to_datetime(dec_2025_rows.iloc[0]['release_date'])

        print(f"\n📅 Release date: {release_date.date()}")
        print(f"📅 Snapshot date: {snap_date.date()}")
        print(f"📅 Days difference: {(release_date - snap_date).days} days")
        print(f"📅 Would be filtered? {release_date >= snap_date}")

        if release_date >= snap_date:
            print(f"\n⚠️  ⚠️  ⚠️  PROBLEM FOUND! ⚠️  ⚠️  ⚠️")
            print(f"\nThis is the issue:")
            print(f"  1. VIX is a DAILY series")
            print(f"  2. After monthly aggregation, release_date = month-end = {release_date.date()}")
            print(f"  3. But NFP snapshot date = {snap_date.date()} (EARLIER in the month!)")
            print(f"  4. Filter: release_date >= snap_date → {release_date.date()} >= {snap_date.date()} → TRUE")
            print(f"  5. Result: FILTERED OUT despite having valid daily data")
    else:
        print(f"\n⚠️  No December 2025 data found")

    # Count all rows that would be filtered
    before_filter = len(sub_df)
    after_filter = len(sub_df[sub_df['release_date'] < snap_date])
    filtered_out = before_filter - after_filter

    print(f"\n{'='*80}")
    print(f"FILTERING SUMMARY FOR VIX")
    print(f"{'='*80}")
    print(f"  Before filter: {before_filter} rows")
    print(f"  After filter: {after_filter} rows")
    print(f"  Filtered out: {filtered_out} rows")

    if filtered_out > 0:
        filtered_rows = sub_df[sub_df['release_date'] >= snap_date]
        print(f"\nFiltered months:")
        print(filtered_rows['date'].unique())

def test_all_daily_series():
    """Test all daily series to count total filtered rows."""
    print("\n" + "=" * 80)
    print("TESTING ALL DAILY SERIES")
    print("=" * 80)

    # Load NFP schedule
    nfp_path = DATA_PATH / "NFP_target" / "y_nsa_first_release.parquet"
    nfp_schedule = pd.read_parquet(nfp_path)[['ds', 'release_date']].copy()
    nfp_schedule.columns = ['data_month', 'nfp_release_date']
    nfp_schedule['data_month'] = pd.to_datetime(nfp_schedule['data_month'])
    nfp_schedule['nfp_release_date'] = pd.to_datetime(nfp_schedule['nfp_release_date'])

    # December 2025
    obs_month = pd.Timestamp('2025-12-01')
    nfp_release_row = nfp_schedule[nfp_schedule['data_month'] == obs_month]
    snap_date = pd.Timestamp(nfp_release_row.iloc[0]['nfp_release_date'])

    print(f"\nSnapshot date: {snap_date.date()}")

    fred = Fred(api_key=FRED_API_KEY)

    # Daily series
    DAILY_SERIES = {
        "VIX": "VIXCLS",
        "SP500": "SP500",
        "Credit_Spreads": "BAMLH0A0HYM2",
        "Yield_Curve": "T10Y3M",
        "Oil_Prices": "DCOILWTICO"
    }

    total_filtered = 0

    for name, code in DAILY_SERIES.items():
        try:
            print(f"\n{'-'*80}")
            print(f"Testing {name}")
            print(f"{'-'*80}")

            series = fred.get_series(code)
            df = series.to_frame(name='value')
            df.index.name = 'date'
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
            df['realtime_start'] = df['date'] + pd.Timedelta(days=1)

            # Filter
            valid = df[df['realtime_start'] < snap_date].copy()

            if valid.empty:
                print("  No valid data")
                continue

            # Get latest
            valid = valid.sort_values(['date', 'realtime_start'])
            latest = valid.drop_duplicates(subset=['date'], keep='last')

            # Monthly aggregation (simplified - just count features)
            sub_df = latest[['date', 'value']].set_index('date').sort_index()
            monthly_agg = sub_df.resample('MS').agg({'value': 'mean'})

            # For VIX and SP500, there are ~15 features per month
            # For others, there are ~7 features per month
            if name in ["VIX", "SP500"]:
                features_per_month = 15
            else:
                features_per_month = 7

            num_months = len(monthly_agg)

            # Create mock sub_df
            mock_rows = []
            for month in monthly_agg.index:
                for i in range(features_per_month):
                    mock_rows.append({
                        'date': month,
                        'release_date': month + pd.offsets.MonthEnd(0),
                        'snapshot_date': snap_date
                    })

            mock_df = pd.DataFrame(mock_rows)
            mock_df['release_date'] = pd.to_datetime(mock_df['release_date'])

            # Filter
            before = len(mock_df)
            after = len(mock_df[mock_df['release_date'] < snap_date])
            filtered = before - after

            print(f"  Total feature rows: {before}")
            print(f"  Filtered out: {filtered}")

            total_filtered += filtered

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*80}")
    print(f"TOTAL FILTERED ACROSS ALL DAILY SERIES: {total_filtered}")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_december_2025()
    test_all_daily_series()
