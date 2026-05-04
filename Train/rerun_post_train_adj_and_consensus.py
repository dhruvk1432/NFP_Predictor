"""
Re-run only the post-training stages affected by the OOS-row fix in
`_generate_adjustment_folder`:

  1) Regenerate `_output/NSA_plus_adjustment/backtest_results.csv` (now
     including OOS future months).
  2) Re-run the consensus anchor pipeline (which reads the file from step 1
     and now sees April + May 2026 as champion_pred), which also re-augments
     `_output/Predictions/predictions.csv` with Consensus + consensus_anchor
     rows.

This avoids re-running the 90-minute --train-all when only the post-training
output stage needs to be redone.

Usage:
    python -m Train.rerun_post_train_adj_and_consensus
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.Output_code.generate_output import _generate_adjustment_folder
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)


def main() -> None:
    nsa_path = OUTPUT_DIR / "NSA_prediction" / "backtest_results.csv"
    sa_path = OUTPUT_DIR / "SA_prediction" / "backtest_results.csv"

    if not (nsa_path.exists() and sa_path.exists()):
        raise FileNotFoundError(
            f"Missing backtest CSVs: {nsa_path} / {sa_path}. "
            "Run --train-all first."
        )

    logger.info("Loading NSA backtest from %s", nsa_path)
    nsa_results = pd.read_csv(nsa_path, parse_dates=["ds"])
    logger.info("Loading SA backtest from %s", sa_path)
    sa_results = pd.read_csv(sa_path, parse_dates=["ds"])

    logger.info(
        "NSA: %d rows (%d with actual, %d OOS)",
        len(nsa_results),
        int(nsa_results["actual"].notna().sum()),
        int(nsa_results["actual"].isna().sum()),
    )
    logger.info(
        "SA:  %d rows (%d with actual, %d OOS)",
        len(sa_results),
        int(sa_results["actual"].notna().sum()),
        int(sa_results["actual"].isna().sum()),
    )

    # 1) Regenerate NSA_plus_adjustment folder (now with OOS rows)
    adj_folder = OUTPUT_DIR / "NSA_plus_adjustment"
    logger.info("Regenerating %s ...", adj_folder)
    _generate_adjustment_folder(nsa_results, sa_results, adj_folder)

    # 2) Re-run consensus anchor pipeline (will augment predictions.csv at the end)
    logger.info("Re-running consensus anchor pipeline ...")
    run_consensus_anchor_pipeline(output_base=OUTPUT_DIR, tune=True)

    logger.info("Done. Inspect _output/Predictions/predictions.csv and _output/consensus_anchor/.")


if __name__ == "__main__":
    main()
