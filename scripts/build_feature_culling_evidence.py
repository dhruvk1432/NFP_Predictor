"""Summarize evidence used by generation-time feature culling.

This reads dynamic-selection JSONs and archived feature-importance CSVs.  It
does not choose model features; it produces stability evidence for allow/protect
and deny-review decisions across acceleration, HMM/regime, NSA, Kalman/fusion,
SA, and other local time-series uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.feature_generation_policy import RETENTION_USE_CASES, feature_transform_metadata


DEFAULT_OUTPUT_DIR = REPO_ROOT / "_output" / "feature_culling_evidence"


def classify_feature_source(feature: str) -> str:
    if feature.startswith("total_"):
        return "FRED_Employment_NSA" if "_nsa" in feature else "FRED_Employment_SA"
    if feature.startswith(("Treasury_", "FedFunds_", "SOFR_", "WTI_Crude_", "NatGas_", "Gold_", "Copper_", "DollarIndex_", "EuroFX_", "YenFX_")):
        return "Futures"
    if feature.startswith(("NFP_Forecast_", "Economist_")):
        return "EconomistPanel"
    if feature.startswith("sanagap_"):
        return "SA_NSA_Gap"
    if feature.startswith(("is_", "month_", "quarter_", "year", "weeks_since_", "nfp_", "rev_master_")):
        return "DerivedControls"
    if feature.startswith(("NOAA_", "storm_", "hurricane_")):
        return "NOAA"
    if feature.startswith(("Prosper_", "Consumer_Mood", "Consumer_Spending")):
        return "Prosper"
    if feature.startswith(("ADP_", "adp_")):
        return "ADP"
    if feature.startswith(("AHE_", "AWH_", "CB_", "Challenger_", "Empire_", "Housing_", "ISM_", "Industrial_", "Retail_", "UMich_")):
        return "Unifier"
    if feature.startswith(("VIX", "SP500", "CCNSA", "CCSA", "Financial_Stress", "Yield_Curve", "Oil_Prices", "Weekly_Econ_Index", "Credit_Spreads")):
        return "FRED_Exogenous"
    return "Unknown"


def _read_dynamic_selection(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "_output" / "dynamic_selection").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        features = payload.get("selected_features") or payload.get("features") or []
        target = path.parent.name
        window = path.stem
        for rank, feature in enumerate(features, start=1):
            feature = str(feature)
            rows.append(
                {
                    "evidence_type": "dynamic_selection",
                    "run": "current",
                    "target": target,
                    "window": window,
                    "feature": feature,
                    "rank": rank,
                    "importance": pd.NA,
                    "source": classify_feature_source(feature),
                    "retention_use_cases": "|".join(sorted(RETENTION_USE_CASES)),
                    **feature_transform_metadata(feature),
                }
            )
    return pd.DataFrame(rows)


def _read_archive_importance(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "_output" / "Archive").glob("*/**/feature_importance.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        feature_col = "feature" if "feature" in frame.columns else frame.columns[0]
        importance_col = "importance" if "importance" in frame.columns else None
        run = path.parts[path.parts.index("Archive") + 1] if "Archive" in path.parts else "archive"
        target = path.parent.name
        for rank, row in enumerate(frame.to_dict("records"), start=1):
            feature = str(row.get(feature_col))
            importance = row.get(importance_col, pd.NA) if importance_col else pd.NA
            rows.append(
                {
                    "evidence_type": "archive_importance",
                    "run": run,
                    "target": target,
                    "window": "",
                    "feature": feature,
                    "rank": rank,
                    "importance": importance,
                    "source": classify_feature_source(feature),
                    "retention_use_cases": "|".join(sorted(RETENTION_USE_CASES)),
                    **feature_transform_metadata(feature),
                }
            )
    return pd.DataFrame(rows)


def build_evidence(root: Path = REPO_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.concat(
        [_read_dynamic_selection(root), _read_archive_importance(root)],
        ignore_index=True,
        sort=False,
    )
    if detail.empty:
        return detail, pd.DataFrame()

    summary = (
        detail.groupby(["feature", "source", "base_series", "transform_family", "lag"], dropna=False)
        .agg(
            evidence_rows=("feature", "size"),
            dynamic_windows=("window", lambda s: int((s.astype(str) != "").sum())),
            archive_runs=("run", "nunique"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(["evidence_rows", "archive_runs", "dynamic_windows", "best_rank"], ascending=[False, False, False, True])
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detail, summary = build_evidence(REPO_ROOT)
    detail.to_csv(out_dir / "feature_culling_evidence_detail.csv", index=False)
    summary.to_csv(out_dir / "feature_culling_evidence_summary.csv", index=False)
    print(f"Wrote {len(detail):,} evidence rows and {len(summary):,} feature summaries to {out_dir}")


if __name__ == "__main__":
    main()
