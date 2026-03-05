"""
Build first-release consensus merged datasets for consensus-anchor experiments.

This script avoids live Reuters/Unifier API calls by reading the latest local
Unifier snapshot parquet and extracting `NFP_Consensus_Mean`, which is stored
as the first-release consensus series in this pipeline.

Outputs:
  - _output/consensus_comparison/consensus_model_merged_first_release_blend.csv
  - _output/consensus_comparison/consensus_model_merged_first_release_nsa_adjustment.csv
  - _output/consensus_comparison/consensus_model_merged_first_release_sa_revised.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_ROOT = REPO_ROOT / "data" / "Exogenous_data" / "exogenous_unifier_data" / "decades"
OUT_DIR = REPO_ROOT / "_output" / "consensus_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_snapshot_path() -> Path:
    candidates = list(SNAPSHOT_ROOT.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No snapshot parquet files found under {SNAPSHOT_ROOT}")

    def snapshot_key(path: Path) -> pd.Timestamp:
        # Files are named YYYY-MM.parquet
        return pd.to_datetime(path.stem + "-01", errors="coerce")

    dated = [(p, snapshot_key(p)) for p in candidates]
    dated = [(p, d) for p, d in dated if pd.notna(d)]
    if not dated:
        raise RuntimeError("Could not parse dates from snapshot parquet filenames")

    latest_path, _ = max(dated, key=lambda x: x[1])
    return latest_path


def _load_first_release_consensus(snapshot_path: Path) -> pd.DataFrame:
    snap = pd.read_parquet(snapshot_path, columns=["date", "series_name", "value"])
    cons = snap[snap["series_name"] == "NFP_Consensus_Mean"].copy()
    if cons.empty:
        raise RuntimeError(
            f"NFP_Consensus_Mean not found in snapshot: {snapshot_path}"
        )

    cons["ds"] = pd.to_datetime(cons["date"]).dt.to_period("M").dt.to_timestamp()
    cons["value"] = pd.to_numeric(cons["value"], errors="coerce")
    cons = cons.dropna(subset=["ds", "value"]).sort_values("ds")

    monthly = (
        cons.groupby("ds", as_index=False)
        .agg(
            consensus_pred=("value", "last"),
            consensus_mean=("value", "mean"),
            consensus_median=("value", "median"),
            consensus_std=("value", "std"),
            consensus_rows=("value", "size"),
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly


def _load_model_backtest(path: Path, pred_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["ds"])
    missing = [c for c in ["ds", "actual", "predicted"] if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")

    out = df[["ds", "actual", "predicted"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["predicted"] = pd.to_numeric(out["predicted"], errors="coerce")
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = out.rename(
        columns={
            "actual": f"actual_{pred_name}",
            "predicted": pred_name,
        }
    )
    return out


def _build_merged(
    consensus_monthly: pd.DataFrame,
    champion_path: Path,
    challenger_path: Path,
) -> pd.DataFrame:
    champion_df = _load_model_backtest(champion_path, "champion_pred")
    challenger_df = _load_model_backtest(challenger_path, "challenger_pred")

    merged = (
        consensus_monthly[
            ["ds", "consensus_pred", "consensus_mean", "consensus_median", "consensus_std", "consensus_rows"]
        ]
        .merge(champion_df, on="ds", how="outer")
        .merge(challenger_df, on="ds", how="outer")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    merged["actual"] = merged["actual_champion_pred"].combine_first(
        merged["actual_challenger_pred"]
    )
    return merged


def main() -> None:
    snapshot_path = _latest_snapshot_path()
    consensus_monthly = _load_first_release_consensus(snapshot_path)

    challenger_path = REPO_ROOT / "_output" / "SA_prediction_revised" / "backtest_results.csv"

    variants: Dict[str, Path] = {
        "blend": REPO_ROOT / "_output" / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv",
        "nsa_adjustment": REPO_ROOT / "_output" / "sandbox" / "nsa_predicted_adjustment_revised" / "backtest_results.csv",
        "sa_revised": REPO_ROOT / "_output" / "SA_prediction_revised" / "backtest_results.csv",
    }

    print(f"Using snapshot: {snapshot_path}")
    print(
        "First-release consensus rows:",
        len(consensus_monthly),
        f"({consensus_monthly['ds'].min().date()} to {consensus_monthly['ds'].max().date()})",
    )

    for name, champion_path in variants.items():
        merged = _build_merged(
            consensus_monthly=consensus_monthly,
            champion_path=champion_path,
            challenger_path=challenger_path,
        )

        out_path = OUT_DIR / f"consensus_model_merged_first_release_{name}.csv"
        merged.to_csv(out_path, index=False)

        overlap = merged[
            merged["consensus_pred"].notna()
            & merged["champion_pred"].notna()
            & merged["actual"].notna()
        ]
        print(
            f"[{name}] wrote {out_path} | rows={len(merged)} | overlap={len(overlap)}"
        )


if __name__ == "__main__":
    main()
