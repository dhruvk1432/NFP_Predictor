"""Compact VAR/BVAR-style prior sidecar.

This is intentionally low-dimensional and shrinkage-heavy.  It uses target
history plus a compact set of existing consensus, economist, labor-cycle, and
futures/stress signals to emit a prior forecast and acceleration signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sidecars.common import feature_audit_from_frame, write_sidecar_artifacts
from experiments.sidecars.feature_matrix import (
    build_sidecar_design,
    rank_features_by_correlation,
    select_numeric_feature_cols,
    source_map_for_columns,
)

try:
    from settings import DATA_PATH, OUTPUT_DIR
except RuntimeError:
    DATA_PATH = Path("data")
    OUTPUT_DIR = Path("_output")


DEFAULT_TARGET_PATH = DATA_PATH / "NFP_target" / "y_sa_revised.parquet"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "compact_bvar_prior"
DEFAULT_MODEL_ID = "compact_bvar_prior"


def _compact_columns(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    priority_prefixes = (
        "sa_mom_lag",
        "sa_accel_lag",
        "sa_mom_mean",
        "sa_mom_std",
        "NFP_Consensus",
        "rev_master_",
        "NFP_Forecast_",
        "Economist_",
        "Treasury_",
        "FedFunds_",
        "SOFR_",
        "Financial_",
        "VIX_",
        "Challenger_",
        "ISM_",
        "UMich_",
        "CB_",
        "sanagap_",
    )
    out = [c for c in feature_cols if c.startswith(priority_prefixes)]
    if len(out) < 6:
        out.extend([c for c in feature_cols if c not in out][: 20 - len(out)])
    return out[:80]


def _add_var_lags(frame: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    generated: list[str] = []
    for col in cols:
        for lag in lags:
            name = f"{col}_varlag{lag}"
            out[name] = out[col].shift(lag)
            generated.append(name)
    return out, generated


def run_bvar_prior_sidecar(
    *,
    target_path: Path = DEFAULT_TARGET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start: str = "2000-01",
    min_train: int = 72,
    target_space: str = "sa_revised",
    include_snapshots: bool = True,
    max_snapshot_columns: int = 120,
    top_features: int = 35,
    ridge_alpha: float = 50.0,
    lags: tuple[int, ...] = (1, 2),
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    design = build_sidecar_design(
        target_space=target_space,
        target_path=target_path,
        include_snapshots=include_snapshots,
        snapshot_blocks=("consensus", "economist", "futures", "unifier", "stress", "gap"),
        max_snapshot_columns=max_snapshot_columns,
    )
    design = design.dropna(subset=["prev_mom", "actual_mom", "actual_accel"]).reset_index(drop=True)
    base_features = _compact_columns(design, select_numeric_feature_cols(design))
    design, lagged_features = _add_var_lags(design, base_features, lags)
    feature_cols = [c for c in base_features + lagged_features if c in design.columns]
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()

    rows: list[dict[str, object]] = []
    used_features: set[str] = set()
    for idx, row in design.iterrows():
        ds = pd.Timestamp(row["ds"])
        if ds < start_ts:
            continue
        train = design.iloc[:idx].dropna(subset=["actual_mom", "prev_mom"]).copy()
        if len(train) < int(min_train):
            continue
        ranked = rank_features_by_correlation(
            train,
            feature_cols,
            "actual_mom",
            min_non_nan=max(18, min(int(min_train), 48)),
            top_n=top_features,
        )
        if len(ranked) < 3:
            continue
        used_features.update(ranked)
        med = train[ranked].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_train = train[ranked].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        y_train = train["actual_mom"].to_numpy(dtype=float)
        X_pred_raw = pd.DataFrame([row[ranked].to_dict()])[ranked]
        X_pred = X_pred_raw.replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        xp = scaler.transform(X_pred)
        model = Ridge(alpha=float(ridge_alpha))
        model.fit(Xs, y_train)
        pred = float(model.predict(xp)[0])
        train_scale = float(train["actual_mom"].tail(60).abs().median())
        train_scale = train_scale if np.isfinite(train_scale) and train_scale > 0 else 100.0
        pred = float(np.clip(pred, -3.0 * train_scale, 3.0 * train_scale))
        predicted_accel = float(pred - row["prev_mom"])
        accel_scale = float(train["actual_accel"].tail(36).abs().median())
        accel_scale = accel_scale if np.isfinite(accel_scale) and accel_scale > 0 else train_scale
        proba_up = float(1.0 / (1.0 + np.exp(-predicted_accel / max(accel_scale, 1e-6))))
        rows.append(
            {
                "ds": ds,
                "trained_through": pd.Timestamp(train["ds"].max()),
                "predicted_mom": pred,
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": float(np.sign(predicted_accel)),
                "predicted_accel_proba_up": proba_up,
                "confidence": float(abs(proba_up - 0.5) * 2.0),
                "uncertainty": float(1.0 - abs(proba_up - 0.5) * 2.0),
                "actual_mom": float(row["actual_mom"]),
                "actual_accel": float(row["actual_accel"]),
                "prev_mom": float(row["prev_mom"]),
                "n_features": int(len(ranked)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Compact BVAR prior sidecar produced no predictions")
    audit_cols = sorted(used_features) or feature_cols
    audit = feature_audit_from_frame(
        design,
        audit_cols,
        source_map=source_map_for_columns(audit_cols),
    )
    _, metrics = write_sidecar_artifacts(
        output_dir=output_dir,
        model_id=model_id,
        target_space=target_space,
        predictions=results,
        feature_audit=audit,
        config={
            "start": start,
            "min_train": int(min_train),
            "target_space": target_space,
            "include_snapshots": bool(include_snapshots),
            "max_snapshot_columns": int(max_snapshot_columns),
            "top_features": int(top_features),
            "ridge_alpha": float(ridge_alpha),
            "lags": list(lags),
        },
        extra_metrics={
            "ridge_alpha": float(ridge_alpha),
            "mean_confidence": float(results["confidence"].mean()),
        },
        data_paths={"target_path": str(target_path)},
    )
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2000-01")
    parser.add_argument("--min-train", type=int, default=72)
    parser.add_argument("--target-space", choices=["sa_revised", "nsa_revised"], default="sa_revised")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--max-snapshot-columns", type=int, default=120)
    parser.add_argument("--top-features", type=int, default=35)
    parser.add_argument("--ridge-alpha", type=float, default=50.0)
    args = parser.parse_args()
    _, metrics = run_bvar_prior_sidecar(
        target_path=args.target_path,
        output_dir=args.output_dir,
        start=args.start,
        min_train=args.min_train,
        target_space=args.target_space,
        include_snapshots=not args.no_snapshots,
        max_snapshot_columns=args.max_snapshot_columns,
        top_features=args.top_features,
        ridge_alpha=args.ridge_alpha,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
