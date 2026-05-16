"""Direct-SA residual sidecar.

This model keeps consensus as the level prior and asks whether existing PIT
features explain consensus misses.  It predicts SA revised MoM residuals
(`actual_mom - consensus_pred`) and emits the implied level/acceleration signal
using the shared sidecar artifact contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
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
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "direct_sa_residual"
DEFAULT_MODEL_ID = "direct_sa_residual"


def _load_consensus_pit_frame() -> pd.DataFrame:
    try:
        from Train.Output_code.consensus_anchor_runner import _load_consensus_pit

        out = _load_consensus_pit(target_type="sa", target_source="revised")
        return out[["ds", "consensus_pred"]].copy()
    except Exception:
        return pd.DataFrame(columns=["ds", "consensus_pred"])


def _attach_consensus(design: pd.DataFrame) -> pd.DataFrame:
    out = design.copy()
    candidates = [c for c in out.columns if c == "NFP_Consensus_Mean" or c.startswith("NFP_Consensus_Mean")]
    if candidates:
        out["consensus_pred"] = pd.to_numeric(out[candidates[0]], errors="coerce")
        return out
    cons = _load_consensus_pit_frame()
    if cons.empty:
        out["consensus_pred"] = np.nan
        return out
    cons["ds"] = pd.to_datetime(cons["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out.merge(cons, on="ds", how="left")


def _optional_regressor(kind: str) -> Any | None:
    if kind == "xgb":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=120,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=8.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
        except Exception:
            return None
    if kind == "lgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                objective="regression",
                n_estimators=140,
                learning_rate=0.035,
                num_leaves=7,
                min_child_samples=12,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )
        except Exception:
            return None
    return None


def _fit_predict(kind: str, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> tuple[float, str]:
    if kind == "ridge":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        xp = scaler.transform(X_pred)
        model = Ridge(alpha=25.0)
        model.fit(Xs, y_train)
        return float(model.predict(xp)[0]), "ridge"
    if kind == "elasticnet":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        xp = scaler.transform(X_pred)
        model = ElasticNet(alpha=0.05, l1_ratio=0.15, random_state=42, max_iter=5000)
        model.fit(Xs, y_train)
        return float(model.predict(xp)[0]), "elasticnet"
    model = _optional_regressor(kind)
    if model is not None:
        try:
            model.fit(X_train, y_train)
            return float(model.predict(X_pred)[0]), kind
        except Exception:
            pass
    return _fit_predict("ridge", X_train, y_train, X_pred)


def run_direct_sa_residual_sidecar(
    *,
    target_path: Path = DEFAULT_TARGET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start: str = "2000-01",
    min_train: int = 72,
    include_snapshots: bool = True,
    max_snapshot_columns: int = 250,
    top_features: int = 60,
    model_kind: str = "ridge",
    model_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model_kind not in {"ridge", "elasticnet", "xgb", "lgbm"}:
        raise ValueError(f"Unsupported model_kind={model_kind!r}")
    model_id = model_id or f"{DEFAULT_MODEL_ID}_{model_kind}"
    design = build_sidecar_design(
        target_space="sa_revised",
        target_path=target_path,
        include_snapshots=include_snapshots,
        max_snapshot_columns=max_snapshot_columns,
    )
    design = _attach_consensus(design)
    design = design.dropna(subset=["prev_mom", "actual_mom", "actual_accel", "consensus_pred"]).reset_index(drop=True)
    if design.empty:
        raise RuntimeError("Direct-SA residual sidecar has no rows with consensus_pred")
    design["consensus_residual"] = design["actual_mom"] - design["consensus_pred"]
    feature_cols = select_numeric_feature_cols(
        design,
        exclude=("ds", "y_mom", "actual_mom", "actual_accel", "consensus_residual"),
    )
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()

    rows: list[dict[str, object]] = []
    used_features: set[str] = set()
    for idx, row in design.iterrows():
        ds = pd.Timestamp(row["ds"])
        if ds < start_ts:
            continue
        train = design.iloc[:idx].dropna(subset=["consensus_residual", "actual_mom", "prev_mom"]).copy()
        if len(train) < int(min_train):
            continue
        ranked = rank_features_by_correlation(
            train,
            feature_cols,
            "consensus_residual",
            min_non_nan=max(18, min(int(min_train), 48)),
            top_n=top_features,
        )
        if len(ranked) < 3:
            continue
        used_features.update(ranked)
        med = train[ranked].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_train = train[ranked].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        y_train = train["consensus_residual"].to_numpy(dtype=float)
        X_pred_raw = pd.DataFrame([row[ranked].to_dict()])[ranked]
        X_pred = X_pred_raw.replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        residual_pred, fitted_kind = _fit_predict(model_kind, X_train, y_train, X_pred)
        residual_scale = float(pd.Series(y_train).tail(36).abs().median())
        if not np.isfinite(residual_scale) or residual_scale <= 1e-6:
            residual_scale = float(pd.Series(y_train).abs().median())
        residual_scale = residual_scale if np.isfinite(residual_scale) and residual_scale > 0 else 75.0
        residual_pred = float(np.clip(residual_pred, -2.0 * residual_scale, 2.0 * residual_scale))
        predicted_mom = float(row["consensus_pred"] + residual_pred)
        predicted_accel = float(predicted_mom - row["prev_mom"])
        accel_scale = float(train["actual_accel"].tail(36).abs().median())
        accel_scale = accel_scale if np.isfinite(accel_scale) and accel_scale > 0 else residual_scale
        proba_up = float(1.0 / (1.0 + np.exp(-predicted_accel / max(accel_scale, 1e-6))))
        rows.append(
            {
                "ds": ds,
                "trained_through": pd.Timestamp(train["ds"].max()),
                "predicted_mom": predicted_mom,
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": float(np.sign(predicted_accel)),
                "predicted_accel_proba_up": proba_up,
                "confidence": float(abs(proba_up - 0.5) * 2.0),
                "uncertainty": float(1.0 - abs(proba_up - 0.5) * 2.0),
                "suggested_nudge": residual_pred,
                "actual_mom": float(row["actual_mom"]),
                "actual_accel": float(row["actual_accel"]),
                "prev_mom": float(row["prev_mom"]),
                "consensus_pred": float(row["consensus_pred"]),
                "residual_pred": residual_pred,
                "n_features": int(len(ranked)),
                "fitted_kind": fitted_kind,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Direct-SA residual sidecar produced no predictions")
    audit_cols = sorted(used_features) or feature_cols
    audit = feature_audit_from_frame(
        design,
        audit_cols,
        source_map=source_map_for_columns(audit_cols),
    )
    _, metrics = write_sidecar_artifacts(
        output_dir=output_dir,
        model_id=model_id,
        target_space="sa_revised",
        predictions=results,
        feature_audit=audit,
        config={
            "start": start,
            "min_train": int(min_train),
            "include_snapshots": bool(include_snapshots),
            "max_snapshot_columns": int(max_snapshot_columns),
            "top_features": int(top_features),
            "model_kind": model_kind,
        },
        extra_metrics={
            "model_kind": model_kind,
            "mean_abs_residual_pred": float(results["residual_pred"].abs().mean()),
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
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--max-snapshot-columns", type=int, default=250)
    parser.add_argument("--top-features", type=int, default=60)
    parser.add_argument("--model-kind", choices=["ridge", "elasticnet", "xgb", "lgbm"], default="ridge")
    args = parser.parse_args()
    _, metrics = run_direct_sa_residual_sidecar(
        target_path=args.target_path,
        output_dir=args.output_dir,
        start=args.start,
        min_train=args.min_train,
        include_snapshots=not args.no_snapshots,
        max_snapshot_columns=args.max_snapshot_columns,
        top_features=args.top_features,
        model_kind=args.model_kind,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
