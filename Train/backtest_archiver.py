"""
Backtest Results Archiver

Manages historical backtest outputs:
- Archives current _output/backtest before new run
- Creates timestamped historical snapshots
- Maintains clean separation between runs
"""

import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, setup_logger, TEMP_DIR

logger = setup_logger(__file__, TEMP_DIR)

BACKTEST_DIR = OUTPUT_DIR / "backtest_results"
HISTORICAL_DIR = OUTPUT_DIR / "backtest_historical"


def archive_current_backtest() -> Optional[Path]:
    """
    Archive current backtest results to historical directory.

    Creates timestamped archive: backtest_historical/YYYY-MM-DD_HH-MM-SS/

    Returns:
        Path to archived directory, or None if no backtest exists
    """
    if not BACKTEST_DIR.exists() or not any(BACKTEST_DIR.iterdir()):
        logger.info("No existing backtest results to archive")
        return None

    # Create timestamp for archive
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = HISTORICAL_DIR / timestamp

    logger.info(f"Archiving current backtest results to: {archive_path}")

    try:
        # Create historical directory
        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

        # Copy entire backtest directory
        shutil.copytree(BACKTEST_DIR, archive_path, dirs_exist_ok=False)

        # Create metadata file
        metadata = {
            "archived_at": timestamp,
            "archived_from": str(BACKTEST_DIR),
            "contains": [p.name for p in archive_path.iterdir() if p.is_dir() or p.suffix in ['.csv', '.parquet', '.json', '.pkl']]
        }

        with open(archive_path / "archive_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Archived {len(metadata['contains'])} items")
        return archive_path

    except Exception as e:
        logger.error(f"Failed to archive backtest: {e}")
        return None


def clean_current_backtest():
    """
    Remove current backtest directory to prepare for new run.

    Should be called AFTER archiving.
    """
    if not BACKTEST_DIR.exists():
        logger.info("No backtest directory to clean")
        return

    logger.info(f"Cleaning current backtest directory: {BACKTEST_DIR}")

    try:
        shutil.rmtree(BACKTEST_DIR)
        logger.info("✓ Current backtest directory cleaned")
    except Exception as e:
        logger.error(f"Failed to clean backtest directory: {e}")


def list_historical_backtests() -> pd.DataFrame:
    """
    List all historical backtests with metadata.

    Returns:
        DataFrame with columns: timestamp, path, size_mb, num_files
    """
    if not HISTORICAL_DIR.exists():
        return pd.DataFrame(columns=['timestamp', 'path', 'size_mb', 'num_files'])

    records = []

    for archive_dir in sorted(HISTORICAL_DIR.iterdir()):
        if not archive_dir.is_dir():
            continue

        # Get directory size
        total_size = sum(f.stat().st_size for f in archive_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)

        # Count files
        num_files = len([f for f in archive_dir.rglob('*') if f.is_file()])

        records.append({
            'timestamp': archive_dir.name,
            'path': str(archive_dir),
            'size_mb': round(size_mb, 2),
            'num_files': num_files
        })

    return pd.DataFrame(records)


def compare_backtest_performance(current_metrics: Dict, archive_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Compare current backtest performance with historical runs.

    Args:
        current_metrics: Dictionary with current run metrics
        archive_path: Path to specific historical run (default: most recent)

    Returns:
        DataFrame comparing metrics
    """
    if archive_path is None:
        # Get most recent historical run
        historical = list_historical_backtests()
        if historical.empty:
            logger.warning("No historical backtests found for comparison")
            return pd.DataFrame()

        archive_path = Path(historical.iloc[-1]['path'])

    # Load historical metrics
    hist_metrics_path = archive_path / "metrics_summary.csv"
    if not hist_metrics_path.exists():
        logger.warning(f"No metrics found in {archive_path}")
        return pd.DataFrame()

    hist_metrics = pd.read_csv(hist_metrics_path)

    # Create comparison
    comparison = []
    for metric in current_metrics.keys():
        if metric in hist_metrics.columns:
            comparison.append({
                'metric': metric,
                'current': current_metrics[metric],
                'previous': hist_metrics[metric].iloc[0] if not hist_metrics.empty else None,
                'change': current_metrics[metric] - (hist_metrics[metric].iloc[0] if not hist_metrics.empty else 0)
            })

    return pd.DataFrame(comparison)


def prepare_new_backtest_run() -> Path:
    """
    Prepare for new backtest run:
    1. Archive current results (if any)
    2. Clean current backtest directory
    3. Create fresh backtest directory

    Returns:
        Path to archived results (or None if nothing archived)
    """
    logger.info("="*60)
    logger.info("PREPARING NEW BACKTEST RUN")
    logger.info("="*60)

    # Step 1: Archive
    archive_path = archive_current_backtest()

    if archive_path:
        logger.info(f"✓ Previous results archived to: {archive_path.name}")

    # Step 2: Clean
    clean_current_backtest()

    # Step 3: Create fresh directory
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Fresh backtest directory created: {BACKTEST_DIR}")

    logger.info("="*60)

    return archive_path


if __name__ == "__main__":
    # Test archiving
    logger.info("Testing backtest archiver...")

    # List existing archives
    archives = list_historical_backtests()
    logger.info(f"\nFound {len(archives)} historical backtests:")
    if not archives.empty:
        print(archives)

    # Prepare new run (will archive if needed)
    archived = prepare_new_backtest_run()

    if archived:
        logger.info(f"\n✓ Archived to: {archived}")
    else:
        logger.info("\nNo previous results to archive")
