"""PIT-safe acceleration classifier sidecar.

This ports the fork's acceleration-first idea as an isolated signal generator.
It predicts acceleration direction and a bounded acceleration nudge from the
existing cleaned feature universe; it does not replace the production forecast.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sidecars.common import (
    feature_audit_from_frame,
    sidecar_branch_root,
    write_sidecar_artifacts,
)
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


DEFAULT_TARGET_TYPE = "sa"
DEFAULT_MODEL_ID = "acceleration_classifier"
# Legacy flat fallback for NSA when no explicit --output-dir is given.
LEGACY_DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "acceleration_classifier"
)


def _default_target_path(target_type: str) -> Path:
    if target_type == "nsa":
        return DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    return DATA_PATH / "NFP_target" / "y_sa_revised.parquet"


def _default_target_space(target_type: str) -> str:
    return "nsa_revised" if target_type == "nsa" else "sa_revised"


def _resolve_output_dir(explicit: Path | None, target_type: str, run_id: str) -> Path:
    """SA always lands under sidecars/sa/<run_id>/acceleration_classifier/.
    NSA keeps the legacy flat path unless an explicit dir is supplied.
    """
    if explicit is not None:
        return explicit
    target_type = str(target_type).strip().lower()
    if target_type == "sa":
        return sidecar_branch_root(OUTPUT_DIR, "sa") / run_id / "acceleration_classifier"
    return LEGACY_DEFAULT_OUTPUT_DIR


# Back-compat aliases (some external scripts import these names).
DEFAULT_TARGET_PATH = _default_target_path(DEFAULT_TARGET_TYPE)
DEFAULT_OUTPUT_DIR = LEGACY_DEFAULT_OUTPUT_DIR


def _optional_model(kind: str) -> Any | None:
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            )
        except Exception:
            return None
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="binary",
                n_estimators=100,
                learning_rate=0.04,
                num_leaves=7,
                min_child_samples=12,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )
        except Exception:
            return None
    return None


def _logistic_probability(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> float:
    if len(np.unique(y_train)) < 2:
        return float(np.mean(y_train))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    xp = scaler.transform(X_pred)
    model = LogisticRegression(
        C=0.35,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(Xs, y_train)
    return float(model.predict_proba(xp)[0, 1])


def _tree_probability(kind: str, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> float | None:
    model = _optional_model(kind)
    if model is None or len(np.unique(y_train)) < 2:
        return None
    try:
        model.fit(X_train, y_train)
        return float(model.predict_proba(X_pred)[0, 1])
    except Exception:
        return None


def _candidate_probabilities(
    *,
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
) -> dict[str, float]:
    probs: dict[str, float] = {}
    if kind in {"logistic", "ensemble"}:
        probs["logistic"] = _logistic_probability(X_train, y_train, X_pred)
    if kind in {"xgb", "ensemble"}:
        p = _tree_probability("xgb", X_train, y_train, X_pred)
        if p is not None:
            probs["xgb"] = p
    if kind in {"lgbm", "ensemble"}:
        p = _tree_probability("lgbm", X_train, y_train, X_pred)
        if p is not None:
            probs["lgbm"] = p
    if not probs:
        probs["base_rate"] = float(np.mean(y_train))
    return probs


def _add_composites(design: pd.DataFrame) -> pd.DataFrame:
    out = design.copy()
    groups = {
        "labor_breadth": ("total_",),
        "consensus_economist": ("NFP_Consensus", "NFP_Forecast_", "Economist_", "economist_"),
        "futures_stress": ("Treasury_", "FedFunds_", "SOFR_", "Financial_", "VIX_", "Yield_"),
        "claims_ism_challenger": ("Challenger_", "ISM_"),
    }
    for name, prefixes in groups.items():
        cols = [
            c for c in out.columns
            if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(out[c])
        ][:25]
        if not cols:
            continue
        z = out[cols].replace([np.inf, -np.inf], np.nan)
        z = (z - z.expanding(min_periods=12).mean()) / z.expanding(min_periods=12).std().replace(0.0, np.nan)
        out[f"composite_{name}_mean_z"] = z.mean(axis=1)
        out[f"composite_{name}_dispersion"] = z.std(axis=1)
    accel_lag1_cols = [c for c in out.columns if c.endswith("_accel_lag1")]
    if "prev_mom" in out.columns and accel_lag1_cols:
        # Use the target-dynamics helper's PIT-built acceleration lag. A raw
        # actual_accel.shift(1) can leak a same-day previous revised actual.
        out["composite_target_mom_accel_interaction"] = out["prev_mom"] * out[accel_lag1_cols[0]]
    return out


def run_acceleration_classifier_sidecar(
    *,
    target_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2000-01",
    min_train: int = 72,
    target_space: str | None = None,
    include_snapshots: bool = True,
    max_snapshot_columns: int = 250,
    top_features: int = 50,
    model_kind: str = "ensemble",
    model_id: str | None = None,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = "local_sidecar_once",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model_kind not in {"logistic", "xgb", "lgbm", "ensemble"}:
        raise ValueError(f"Unsupported model_kind={model_kind!r}")
    target_type = str(target_type).strip().lower()
    if target_type not in {"nsa", "sa"}:
        raise ValueError(f"target_type must be 'nsa' or 'sa'; got {target_type!r}")
    target_path = target_path or _default_target_path(target_type)
    target_space = target_space or _default_target_space(target_type)
    output_dir = _resolve_output_dir(output_dir, target_type, run_id)
    model_id = model_id or f"{DEFAULT_MODEL_ID}_{model_kind}"
    design = build_sidecar_design(
        target_space=target_space,
        target_path=target_path,
        include_snapshots=include_snapshots,
        max_snapshot_columns=max_snapshot_columns,
    )
    design = _add_composites(design)
    design = design.dropna(subset=["prev_mom", "actual_mom", "actual_accel"]).reset_index(drop=True)
    feature_cols = select_numeric_feature_cols(design)
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()

    rows: list[dict[str, object]] = []
    used_features: set[str] = set()
    for idx, row in design.iterrows():
        ds = pd.Timestamp(row["ds"])
        if ds < start_ts:
            continue
        train = design.iloc[:idx].dropna(subset=["actual_accel", "actual_mom", "prev_mom"]).copy()
        if len(train) < int(min_train):
            continue
        train = train.copy()
        train["accel_up"] = (train["actual_accel"] > 0).astype(int)
        if train["accel_up"].nunique() < 2:
            continue
        ranked = rank_features_by_correlation(
            train,
            feature_cols,
            "actual_accel",
            min_non_nan=max(18, min(int(min_train), 48)),
            top_n=top_features,
        )
        if len(ranked) < 3:
            continue
        used_features.update(ranked)
        med = train[ranked].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_train = train[ranked].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        y_train = train["accel_up"].to_numpy(dtype=int)
        X_pred_raw = pd.DataFrame([row[ranked].to_dict()])[ranked]
        X_pred = X_pred_raw.replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
        probs = _candidate_probabilities(
            kind=model_kind,
            X_train=X_train,
            y_train=y_train,
            X_pred=X_pred,
        )
        proba_up = float(np.mean(list(probs.values())))
        signed_margin = (proba_up - 0.5) * 2.0
        accel_scale = float(train["actual_accel"].abs().tail(36).median())
        if not np.isfinite(accel_scale) or accel_scale <= 1e-6:
            accel_scale = float(train["actual_accel"].abs().median())
        accel_scale = accel_scale if np.isfinite(accel_scale) and accel_scale > 0 else 50.0
        predicted_accel = float(np.clip(signed_margin * accel_scale, -1.5 * accel_scale, 1.5 * accel_scale))
        suggested_nudge = float(np.clip(predicted_accel, -0.35 * accel_scale, 0.35 * accel_scale))
        rows.append(
            {
                "ds": ds,
                "trained_through": pd.Timestamp(train["ds"].max()),
                "predicted_mom": float(row["prev_mom"] + predicted_accel),
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": float(np.sign(predicted_accel)),
                "predicted_accel_proba_up": proba_up,
                "confidence": float(abs(signed_margin)),
                "uncertainty": float(1.0 - abs(signed_margin)),
                "suggested_nudge": suggested_nudge,
                "actual_mom": float(row["actual_mom"]),
                "actual_accel": float(row["actual_accel"]),
                "prev_mom": float(row["prev_mom"]),
                "n_features": int(len(ranked)),
                "available_models": ",".join(sorted(probs)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Acceleration sidecar produced no predictions; lower --min-train or check data")
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
            "model_kind": model_kind,
        },
        extra_metrics={
            "model_kind": model_kind,
            "mean_confidence": float(results["confidence"].mean()),
            "mean_abs_suggested_nudge": float(results["suggested_nudge"].abs().mean()),
        },
        data_paths={"target_path": str(target_path)},
    )
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=None,
                        help="Explicit target parquet; if omitted, picks "
                             "y_<target_type>_revised.parquet by --target-type.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Explicit output dir. SA falls back to "
                             "sidecars/sa/<run_id>/acceleration_classifier; "
                             "NSA falls back to the legacy flat path.")
    parser.add_argument("--target-type", default=DEFAULT_TARGET_TYPE,
                        choices=["nsa", "sa"],
                        help="Branch subtree the artifact lands under.")
    parser.add_argument("--run-id", default="local_sidecar_once",
                        help="Sidecar run-id directory under the branch subtree.")
    parser.add_argument("--start", default="2000-01")
    parser.add_argument("--min-train", type=int, default=72)
    parser.add_argument("--target-space", choices=["sa_revised", "nsa_revised"], default=None,
                        help="Target space override; defaults from --target-type.")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--max-snapshot-columns", type=int, default=250)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--model-kind", choices=["logistic", "xgb", "lgbm", "ensemble"], default="ensemble")
    args = parser.parse_args()
    _, metrics = run_acceleration_classifier_sidecar(
        target_path=args.target_path,
        output_dir=args.output_dir,
        start=args.start,
        min_train=args.min_train,
        target_space=args.target_space,
        include_snapshots=not args.no_snapshots,
        max_snapshot_columns=args.max_snapshot_columns,
        top_features=args.top_features,
        model_kind=args.model_kind,
        target_type=args.target_type,
        run_id=args.run_id,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
