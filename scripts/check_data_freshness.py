#!/usr/bin/env python3
"""
Check Data Freshness

Monitor script to verify data pipelines are current.
Checks FRED data, snapshots, and model files.

Usage:
    python scripts/check_data_freshness.py
    python scripts/check_data_freshness.py --alert-days 7
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import DATA_PATH, OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import MASTER_SNAPSHOTS_DIR, MODEL_SAVE_DIR

logger = setup_logger(__file__, TEMP_DIR)


def get_latest_file_time(directory: Path, pattern: str = "*.parquet") -> datetime:
    """Get modification time of most recent file in directory."""
    files = list(directory.rglob(pattern))
    
    if not files:
        return None
    
    latest = max(files, key=lambda x: x.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime)


def get_latest_snapshot_month(directory: Path) -> pd.Timestamp:
    """Get the most recent snapshot month from filenames."""
    files = list(directory.rglob("*.parquet"))
    
    if not files:
        return None
    
    dates = []
    for f in files:
        try:
            date_str = f.stem
            date = pd.to_datetime(date_str + "-01")
            dates.append(date)
        except:
            continue
    
    return max(dates) if dates else None


def check_data_sources() -> dict:
    """Check freshness of all data sources."""
    checks = {}
    now = datetime.now()
    
    # Check FRED employment data
    fred_dir = DATA_PATH / "fred_data"
    if fred_dir.exists():
        fred_time = get_latest_file_time(fred_dir)
        checks['fred_employment'] = {
            'exists': True,
            'last_modified': fred_time.isoformat() if fred_time else None,
            'days_old': (now - fred_time).days if fred_time else None
        }
    else:
        checks['fred_employment'] = {'exists': False}
    
    # Check FRED exogenous data
    exog_dir = DATA_PATH / "Exogenous_data" / "exogenous_fred_data"
    if exog_dir.exists():
        exog_time = get_latest_file_time(exog_dir)
        checks['fred_exogenous'] = {
            'exists': True,
            'last_modified': exog_time.isoformat() if exog_time else None,
            'days_old': (now - exog_time).days if exog_time else None
        }
    else:
        checks['fred_exogenous'] = {'exists': False}
    
    # Check master snapshots
    if MASTER_SNAPSHOTS_DIR.exists():
        snap_time = get_latest_file_time(MASTER_SNAPSHOTS_DIR)
        snap_month = get_latest_snapshot_month(MASTER_SNAPSHOTS_DIR)
        checks['master_snapshots'] = {
            'exists': True,
            'last_modified': snap_time.isoformat() if snap_time else None,
            'days_old': (now - snap_time).days if snap_time else None,
            'latest_month': snap_month.strftime('%Y-%m') if snap_month else None
        }
    else:
        checks['master_snapshots'] = {'exists': False}
    
    # Check trained models
    for target_type in ['nsa', 'sa']:
        model_path = MODEL_SAVE_DIR / f"lightgbm_{target_type}_model.txt"
        if model_path.exists():
            model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            checks[f'model_{target_type}'] = {
                'exists': True,
                'last_modified': model_time.isoformat(),
                'days_old': (now - model_time).days
            }
        else:
            checks[f'model_{target_type}'] = {'exists': False}
    
    return checks


def generate_report(alert_days: int = 7) -> dict:
    """
    Generate data freshness report.
    
    Args:
        alert_days: Days after which to flag as stale
        
    Returns:
        Dictionary with freshness report
    """
    checks = check_data_sources()
    now = datetime.now()
    
    # Determine overall status
    alerts = []
    for source, info in checks.items():
        if not info.get('exists', False):
            alerts.append(f"MISSING: {source}")
        elif info.get('days_old') and info['days_old'] > alert_days:
            alerts.append(f"STALE: {source} ({info['days_old']} days old)")
    
    status = 'OK' if not alerts else 'WARNING'
    
    report = {
        'timestamp': now.isoformat(),
        'status': status,
        'alert_threshold_days': alert_days,
        'sources': checks,
        'alerts': alerts
    }
    
    return report


def print_report(report: dict) -> None:
    """Print formatted freshness report."""
    print(f"\n{'='*60}")
    print("DATA FRESHNESS REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Status: {report['status']}")
    print(f"Alert threshold: {report['alert_threshold_days']} days")
    print()
    
    print("Data Sources:")
    for source, info in report['sources'].items():
        if info.get('exists'):
            days = info.get('days_old', '?')
            status = '✓' if days and days <= report['alert_threshold_days'] else '⚠'
            print(f"  {status} {source}: {days} days old")
            if 'latest_month' in info:
                print(f"      Latest: {info['latest_month']}")
        else:
            print(f"  ✗ {source}: NOT FOUND")
    
    if report['alerts']:
        print(f"\nAlerts ({len(report['alerts'])}):")
        for alert in report['alerts']:
            print(f"  ⚠ {alert}")
    else:
        print("\nNo alerts - all data sources are fresh!")
    
    print(f"{'='*60}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check data freshness for NFP Predictor"
    )
    parser.add_argument(
        '--alert-days', type=int, default=7,
        help="Days after which to flag as stale (default: 7)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output path for JSON report"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Only output if there are alerts"
    )
    
    args = parser.parse_args()
    
    report = generate_report(alert_days=args.alert_days)
    
    # Print unless quiet and no alerts
    if not (args.quiet and report['status'] == 'OK'):
        print_report(report)
    
    # Save if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {output_path}")
    
    # Return non-zero if alerts
    return 1 if report['alerts'] else 0


if __name__ == "__main__":
    exit(main())
