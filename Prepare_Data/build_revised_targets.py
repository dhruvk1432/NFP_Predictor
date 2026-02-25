"""
Pre-Build Revised Target Parquets
==================================
Materializes revised (once-revised) target data for NSA and SA into
parquet files at the same location as first-release targets:

    data/NFP_target/y_nsa_revised.parquet
    data/NFP_target/y_sa_revised.parquet

This eliminates the Train-time cost of rebuilding revised targets from
435 individual FRED employment snapshots on every train run (~1.2s → ~3ms).

Parity: calls the exact same build_revised_target() function used by the
legacy Train path, then writes the DataFrame to parquet.  When Train
detects these files it loads them directly; otherwise it falls back to
the legacy build_revised_target() path (no behavior change).

Usage:
    python -m Prepare_Data.build_revised_targets          # build both NSA + SA
    python -m Prepare_Data.build_revised_targets --type nsa   # NSA only
    python -m Prepare_Data.build_revised_targets --type sa    # SA only
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.data_loader import build_revised_target, clear_snapshot_cache, _target_cache
from Train.config import NFP_TARGET_DIR

logger = setup_logger(__file__, TEMP_DIR)


def get_revised_target_path(target_type: str) -> Path:
    """Deterministic path for cached revised target parquet."""
    return NFP_TARGET_DIR / f"y_{target_type}_revised.parquet"


def build_and_save(target_type: str) -> Path:
    """
    Build revised target for a given type and save to parquet.

    Uses the exact same build_revised_target() function from Train/data_loader.py
    to ensure parity with the legacy in-process build path.

    Returns:
        Path to the saved parquet file.
    """
    out_path = get_revised_target_path(target_type)
    logger.info(f"Building revised {target_type.upper()} target...")

    # Clear caches to ensure clean build
    clear_snapshot_cache()
    _target_cache.clear()

    t0 = time.perf_counter()
    df = build_revised_target(target_type)
    elapsed = time.perf_counter() - t0

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info(
        f"Saved revised {target_type.upper()} target: "
        f"{len(df)} rows, {df['y_mom'].notna().sum()} valid MoM, "
        f"{out_path.stat().st_size / 1024:.1f} KB, "
        f"built in {elapsed:.1f}s → {out_path.name}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build revised target parquets for Train cache."
    )
    parser.add_argument(
        '--type',
        choices=['nsa', 'sa', 'both'],
        default='both',
        help='Which target type(s) to build (default: both)',
    )
    args = parser.parse_args()

    types = ['nsa', 'sa'] if args.type == 'both' else [args.type]

    for tt in types:
        build_and_save(tt)

    logger.info("Revised target prebuild complete.")


if __name__ == '__main__':
    main()
