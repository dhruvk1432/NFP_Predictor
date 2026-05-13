#!/usr/bin/env python3
"""Run Kalman fusion + archive on whatever's currently in _output/."""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.config import N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.Output_code.generate_output import archive_outputs
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)


def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("KALMAN FUSION ONLY (against current _output/ state)")
    logger.info("=" * 70)

    logger.info("\n[Post-3] Consensus anchor (Kalman fusion)...")
    run_consensus_anchor_pipeline(
        output_base=OUTPUT_DIR,
        tune=True,
        n_trials=N_OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
    )
    logger.info("  Post-3 complete")

    logger.info("\n[Post-4] Archiving final outputs...")
    archive_outputs(OUTPUT_DIR)
    logger.info("  Post-4 complete")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed/60:.1f} min")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
