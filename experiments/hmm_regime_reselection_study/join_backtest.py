"""Join HMM regime audit rows to model backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_SPECS = {
    "nsa_raw": "NSA_prediction/backtest_results.csv",
    "kalman_fusion": "consensus_anchor/kalman_fusion/backtest_results.csv",
    "panel_kalman_router": "consensus_anchor/panel_kalman_router/backtest_results.csv",
}


def _load_hmm(hmm_dir: Path) -> pd.DataFrame:
    path = hmm_dir / "hmm_regime_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "step_date" not in df.columns:
        raise ValueError(f"{path} is missing step_date")
    df["ds"] = pd.to_datetime(df["step_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df.dropna(subset=["ds"]).sort_values("ds")


def _load_backtest(path: Path, model_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    required = {"ds", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["model_id"] = model_id
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["predicted"] = pd.to_numeric(out["predicted"], errors="coerce")
    out["actual_minus_pred"] = out["actual"] - out["predicted"]
    out["abs_error"] = out["actual_minus_pred"].abs()
    return out.dropna(subset=["ds"])


def run(args: argparse.Namespace) -> None:
    hmm_dir = Path(args.hmm_dir)
    backtest_dir = Path(args.backtest_dir)
    hmm = _load_hmm(hmm_dir)
    frames: list[pd.DataFrame] = []
    for model_id, rel_path in BACKTEST_SPECS.items():
        path = backtest_dir / rel_path
        if not path.exists():
            print(f"skip {model_id}: missing {path}")
            continue
        bt = _load_backtest(path, model_id)
        joined = bt.merge(hmm, on="ds", how="left", suffixes=("", "_hmm"))
        joined["has_hmm_regime"] = joined["hmm_available"].fillna(False).astype(bool) if "hmm_available" in joined.columns else False
        frames.append(joined)

    if not frames:
        raise RuntimeError(f"No backtest files found under {backtest_dir}")
    out = pd.concat(frames, ignore_index=True).sort_values(["model_id", "ds"])
    out_path = hmm_dir / "hmm_joined_backtests.csv"
    out.to_csv(out_path, index=False)
    summary = (
        out.dropna(subset=["actual_minus_pred"])
        .groupby("model_id")
        .agg(
            n=("actual_minus_pred", "size"),
            mae=("abs_error", "mean"),
            rmse=("actual_minus_pred", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("actual_minus_pred", "mean"),
        )
        .round(3)
    )
    summary.to_csv(hmm_dir / "joined_model_summary.csv")
    print(f"Wrote {len(out)} joined rows to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmm-dir", default="_output_hmm_regime_reselection_study")
    parser.add_argument("--backtest-dir", default="_output_pairing_baseline_pitfix")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
