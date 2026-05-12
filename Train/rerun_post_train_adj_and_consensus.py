"""
Re-run only the post-training stages affected by adjustment-pipeline changes:

  1) Regenerate `_output/NSA_plus_adjustment/backtest_results.csv` using the
     current adjustment predictor (set in `Output_code/generate_output.py`).
  2) Refresh the NSA_plus_adjustment row in `_output/Predictions/predictions.csv`
     using the regenerated backtest CSV (its `predicted` column for the OOS
     month and its residuals for the CIs and RMSE).
  3) Re-run the consensus anchor pipeline (which re-augments predictions.csv
     with Consensus + consensus_anchor rows).

This avoids re-running the 90-minute --train-all when only the post-training
output stage needs to be redone.

Usage:
    python -m Train.rerun_post_train_adj_and_consensus
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.Output_code.generate_output import _generate_adjustment_folder
from Train.Output_code.consensus_anchor_runner import run_consensus_anchor_pipeline

logger = setup_logger(__file__, TEMP_DIR)


def _refresh_nsa_plus_adjustment_in_predictions_csv() -> None:
    """Rewrite the NSA_plus_adjustment OOS row in predictions.csv from the
    regenerated NSA_plus_adjustment/backtest_results.csv.

    Keeps every other model row untouched. Recomputes RMSE and 50/80/95% CIs
    from the new in-sample residuals (last 36).
    """
    pred_path = OUTPUT_DIR / "Predictions" / "predictions.csv"
    adj_path = OUTPUT_DIR / "NSA_plus_adjustment" / "backtest_results.csv"
    if not pred_path.exists() or not adj_path.exists():
        logger.warning("Skipped predictions.csv refresh: %s or %s missing",
                       pred_path, adj_path)
        return

    preds = pd.read_csv(pred_path, parse_dates=["ds"])
    adj_bt = pd.read_csv(adj_path, parse_dates=["ds"])
    res = adj_bt["error"].dropna().to_numpy()[-36:]
    rmse = float(np.sqrt(np.mean(res ** 2))) if res.size > 0 else float("nan")

    oos = adj_bt[adj_bt["actual"].isna()][["ds", "predicted"]].copy().sort_values("ds")
    if oos.empty:
        logger.warning("No OOS rows in NSA_plus_adjustment to refresh.")
        return

    other = preds[preds["model"] != "NSA_plus_adjustment"].copy()
    next_row = oos.iloc[0]
    ds, pred = next_row["ds"], float(next_row["predicted"])

    new_row = {
        "model": "NSA_plus_adjustment", "ds": ds, "predicted": pred,
        "rmse": rmse,
    }
    if res.size > 2:
        new_row.update({
            "lower_50": pred + np.percentile(res, 25),
            "upper_50": pred + np.percentile(res, 75),
            "lower_80": pred + np.percentile(res, 10),
            "upper_80": pred + np.percentile(res, 90),
            "lower_95": pred + np.percentile(res, 2.5),
            "upper_95": pred + np.percentile(res, 97.5),
        })
    else:
        for k in ("lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"):
            new_row[k] = np.nan

    refreshed = pd.concat([other, pd.DataFrame([new_row])], ignore_index=True)
    refreshed = refreshed[preds.columns]
    refreshed.to_csv(pred_path, index=False)
    logger.info(
        "Refreshed NSA_plus_adjustment row in %s: ds=%s predicted=%.2f rmse=%.2f",
        pred_path, pd.Timestamp(ds).strftime("%Y-%m"), pred, rmse,
    )


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

    # 2) Refresh the NSA_plus_adjustment row in predictions.csv from the new CSV
    logger.info("Refreshing NSA_plus_adjustment row in predictions.csv ...")
    _refresh_nsa_plus_adjustment_in_predictions_csv()

    # 3) Re-run consensus anchor pipeline (will augment predictions.csv at the end)
    logger.info("Re-running consensus anchor pipeline ...")
    run_consensus_anchor_pipeline(output_base=OUTPUT_DIR, tune=True)

    logger.info("Done. Inspect _output/Predictions/predictions.csv and _output/consensus_anchor/.")


if __name__ == "__main__":
    main()
