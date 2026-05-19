#!/usr/bin/env python
"""Replay-only consensus hit-rate evaluator.

Reads a long monthly prediction table with columns:
``run, model, ds, actual, predicted`` and writes consensus-relative metrics by
run/model/period. This script does not rebuild data or train models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Train.Output_code.metrics import add_consensus_hit_rate_metrics
from utils.transforms import is_covid_month


PERIODS: Dict[str, Tuple[str | None, str | int | None]] = {
    "full": (None, None),
    "2010_to_2015": ("2010-01-01", "2015-12-01"),
    "2016_to_2020_02": ("2016-01-01", "2020-02-01"),
    "2020_06_to_2021_12": ("2020-06-01", "2021-12-01"),
    "2022_present": ("2022-01-01", None),
    "last_36": ("last", 36),
    "last_60": ("last", 60),
    "last_120": ("last", 120),
    "last_180": ("last", 180),
}


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    frame = frame.dropna(subset=["actual", "predicted"]).copy()
    if frame.empty:
        return {"N": 0}
    err = frame["actual"] - frame["predicted"]
    out: Dict[str, float] = {
        "N": int(len(frame)),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "MAE": float(np.mean(np.abs(err))),
    }
    non_covid = frame[~is_covid_month(frame["ds"])]
    if not non_covid.empty:
        nc_err = non_covid["actual"] - non_covid["predicted"]
        out["NonCovid_RMSE"] = float(np.sqrt(np.mean(np.square(nc_err))))
        out["NonCovid_MAE"] = float(np.mean(np.abs(nc_err)))
    return add_consensus_hit_rate_metrics(out, frame)


def _period_frame(df: pd.DataFrame, start: str | None, end: str | int | None) -> pd.DataFrame:
    if start == "last":
        months = sorted(df["ds"].dropna().unique())[-int(end):]
        return df[df["ds"].isin(months)].copy()
    out = df.copy()
    if start is not None:
        out = out[out["ds"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["ds"] <= pd.Timestamp(str(end))]
    return out


def evaluate(input_path: Path, output_path: Path) -> pd.DataFrame:
    long = pd.read_csv(input_path, parse_dates=["ds"])
    refs = (
        long[long["model"].isin(["Consensus_Mean", "Consensus_Median"])]
        .pivot_table(index=["run", "ds"], columns="model", values="predicted", aggfunc="first")
        .rename(columns={
            "Consensus_Mean": "consensus_pred",
            "Consensus_Median": "consensus_median_pred",
        })
        .reset_index()
    )
    rows = []
    for (run, model), group in long.groupby(["run", "model"], dropna=False):
        merged = group.merge(refs, on=["run", "ds"], how="left")
        for period, (start, end) in PERIODS.items():
            sub = _period_frame(merged, start, end)
            metrics = _metrics(sub)
            rows.append({"period": period, "run": run, "model": model, **metrics})

    out = pd.DataFrame(rows).sort_values(["period", "run", "MAE"], na_position="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("_output_panel_grid_report_20260519/monthly_predictions_long.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_output_panel_grid_report_20260519/consensus_hit_rates_by_period.csv"),
    )
    args = parser.parse_args()
    out = evaluate(args.input, args.output)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
