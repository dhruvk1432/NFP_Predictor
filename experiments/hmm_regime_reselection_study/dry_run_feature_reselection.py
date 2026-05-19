"""Dry-run dynamic feature reselection on HMM trigger dates without model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hmm_regime_reselection_study.run_regime_audit import _find_cache_pair, _load_y
from Train.train_lightgbm_nfp import _dynamic_reselection


def _trigger_mask(df: pd.DataFrame) -> pd.Series:
    raw = df.get("final_reselect", False)
    if isinstance(raw, bool):
        return pd.Series(raw, index=df.index)
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return np.nan
    return float(len(a & b) / len(a | b))


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = _find_cache_pair(Path(args.training_cache), args.target, args.source)
    X = pd.read_parquet(x_path).copy()
    y = _load_y(y_path)
    X["ds"] = pd.to_datetime(X["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    X = X.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    y = y.iloc[: len(X)].reset_index(drop=True)

    audit = pd.read_csv(args.hmm_audit)
    audit["ds"] = pd.to_datetime(audit["step_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    triggers = audit[_trigger_mask(audit)].sort_values("ds").reset_index(drop=True)
    if args.max_triggers > 0:
        triggers = triggers.head(int(args.max_triggers))

    incumbent: list[str] | None = None
    if args.initial_features_json:
        raw = json.loads(Path(args.initial_features_json).read_text())
        incumbent = raw.get("features") or raw.get("selected_features") or raw.get("final_features")
        if not isinstance(incumbent, list):
            incumbent = None

    rows: list[dict[str, Any]] = []
    for _, trig in triggers.iterrows():
        step_date = pd.Timestamp(trig["ds"])
        train_mask = X["ds"] < step_date
        X_train = X.loc[train_mask].reset_index(drop=True)
        y_train = y.loc[train_mask].reset_index(drop=True)
        hmm_log = {
            key: trig.get(key)
            for key in [
                "hmm_regime_label",
                "hmm_trigger_class",
                "hmm_trigger_reason",
                "hmm_surprise_ratio",
                "hmm_transition_risk",
            ]
            if key in trig.index
        }
        features, meta = _dynamic_reselection(
            X_train=X_train,
            y_train=y_train,
            step_date=step_date,
            target_type=args.target,
            target_source=args.source,
            incumbent_features=incumbent,
            hmm_regime_log=hmm_log,
            return_metadata=True,
        )
        jaccard = _jaccard(incumbent or [], features)
        rows.append({
            "step_date": step_date.strftime("%Y-%m"),
            "hmm_regime_label": trig.get("hmm_regime_label"),
            "hmm_trigger_class": trig.get("hmm_trigger_class"),
            "hmm_trigger_reason": trig.get("hmm_trigger_reason"),
            "n_features": len(features),
            "jaccard_vs_previous": jaccard,
            "replacement_share": 1.0 - jaccard if np.isfinite(jaccard) else np.nan,
            "selection_meta": json.dumps(meta, default=str),
            "features": json.dumps(features),
        })
        incumbent = features

    out = pd.DataFrame(rows)
    out_path = output_dir / "feature_reselection_dry_run.csv"
    out.to_csv(out_path, index=False)
    summary = {
        "n_trigger_runs": int(len(out)),
        "median_jaccard": float(out["jaccard_vs_previous"].median(skipna=True)) if not out.empty else np.nan,
        "min_non_crash_jaccard": float(
            out.loc[~out["hmm_regime_label"].isin(["crash", "volatile_down"]), "jaccard_vs_previous"].min(skipna=True)
        )
        if not out.empty
        else np.nan,
        "eligible_for_training_experiment": bool(
            not out.empty
            and out["jaccard_vs_previous"].median(skipna=True) >= 0.50
            and (
                out.loc[~out["hmm_regime_label"].isin(["crash", "volatile_down"]), "jaccard_vs_previous"]
                .dropna()
                .ge(0.30)
                .all()
            )
        ),
    }
    (output_dir / "feature_reselection_dry_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote dry-run feature reselection diagnostics to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmm-audit", default="_output_hmm_regime_reselection_study/hmm_regime_monthly.csv")
    parser.add_argument("--output-dir", default="_output_hmm_regime_reselection_study/feature_reselection_dry_run")
    parser.add_argument("--training-cache", default="_output_pairing_baseline_pitfix/cache/training_dataset")
    parser.add_argument("--target", default="nsa")
    parser.add_argument("--source", default="revised")
    parser.add_argument("--initial-features-json", default="")
    parser.add_argument("--max-triggers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
