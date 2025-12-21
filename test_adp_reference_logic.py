"""
Test script to verify ADP reference month logic
"""
import pandas as pd
import re
from datetime import datetime

def test_reference_month_logic():
    """Test the reference month determination logic with provided test cases"""

    test_cases = [
        # (Release Date String, Expected Reference Month)
        ("Aug 30, 2023  (Aug)", "2023-08-01"),  # Explicit format
        ("Aug 02, 2023  (Jul)", "2023-07-01"),  # Explicit format
        ("Apr 30, 2008", "2008-04-01"),  # Late release (day 30 >= 20)
        ("Apr 02, 2008", "2008-03-01"),  # Early release (day 2 <= 10)
        ("Mar 05, 2008", "2008-02-01"),  # Early release (day 5 <= 10)
    ]

    print("Testing ADP Reference Month Logic")
    print("=" * 70)

    all_passed = True

    for release_date_str, expected_ref_month in test_cases:
        # Extract reference month from (Month) suffix if present
        month_match = re.search(r'\(([A-Za-z]+)\)\s*$', release_date_str)

        # Clean the release date string (remove suffix)
        release_date_clean = re.sub(r'\s*\([A-Za-z]+\)\s*$', '', release_date_str).strip()

        try:
            release_dt = pd.to_datetime(release_date_clean, format='%b %d, %Y')
        except:
            release_dt = pd.to_datetime(release_date_clean)

        # Determine reference date using priority logic
        if month_match:
            # Explicit format
            ref_month_str = month_match.group(1)
            release_month = release_dt.month
            release_year = release_dt.year

            ref_month_dt = pd.to_datetime(ref_month_str, format='%b')
            ref_month_num = ref_month_dt.month

            # Handle year boundary
            if release_month <= 2 and ref_month_num >= 11:
                ref_year = release_year - 1
            else:
                ref_year = release_year

            reference_date = pd.Timestamp(year=ref_year, month=ref_month_num, day=1)
        else:
            # Implicit format: use day-based rule
            release_day = release_dt.day

            if release_day <= 10:
                # Early release: previous month
                reference_date = (release_dt - pd.DateOffset(months=1)).replace(day=1)
            elif release_day >= 20:
                # Late release: current month
                reference_date = release_dt.replace(day=1)
            else:
                # Days 11-19: default to previous month
                reference_date = (release_dt - pd.DateOffset(months=1)).replace(day=1)

        # Check result
        result_str = reference_date.strftime('%Y-%m-%d')
        passed = result_str == expected_ref_month
        status = "✓ PASS" if passed else "✗ FAIL"

        if not passed:
            all_passed = False

        print(f"{status} | Release: {release_date_str:25} | Expected: {expected_ref_month} | Got: {result_str}")

    print("=" * 70)
    if all_passed:
        print("✓ All test cases passed!")
    else:
        print("✗ Some test cases failed")

    return all_passed

if __name__ == "__main__":
    test_reference_month_logic()