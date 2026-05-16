"""Shared utilities for PIT-safe sidecar signal artifacts.

Sidecars are deliberately separate from the production pipeline.  This module
defines a small, stable artifact contract that lets experiments write signals
which can later be replayed as features, fusion observations, or routers behind
explicit flags.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SIDECAR_SCHEMA_VERSION = "2026-05-16-sidecar-signal-v1"
REQUIRED_PREDICTION_COLUMNS = {
    "ds",
    "model_id",
    "target_space",
    "predicted_mom",
    "predicted_accel",
    "predicted_accel_sign",
    "predicted_accel_proba_up",
    "confidence",
    "uncertainty",
    "trained_through",
}
STRESS_PANEL_MONTHS = {
    "2026-03",
    "2022-07",
    "2022-12",
    "2021-06",
    "2022-11",
    "2023-01",
    "2024-04",
    "2023-09",
    "2023-06",
    "2023-10",
}


def month_start(values: Any) -> Any:
    converted = pd.to_datetime(values, errors="coerce")
    if isinstance(converted, pd.Timestamp):
        return converted.to_period("M").to_timestamp()
    return converted.dt.to_period("M").dt.to_timestamp()


def repo_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sanitize_model_id(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in {"-", ".", " "}:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "sidecar"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _base_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    df = predictions.copy()
    actual_col = "actual_mom" if "actual_mom" in df.columns else "actual"
    metrics: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "n_predictions": int(len(df)),
    }
    if actual_col not in df.columns:
        return metrics

    scored = df[df[actual_col].notna() & df["predicted_mom"].notna()].copy()
    metrics["n_scored"] = int(len(scored))
    if scored.empty:
        return metrics

    actual = scored[actual_col].to_numpy(dtype=float)
    pred = scored["predicted_mom"].to_numpy(dtype=float)
    err = pred - actual
    metrics.update(
        {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "bias": float(np.mean(err)),
            "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        }
    )
    if "actual_accel" in scored.columns and "predicted_accel" in scored.columns:
        aa = scored["actual_accel"].to_numpy(dtype=float)
        pa = scored["predicted_accel"].to_numpy(dtype=float)
        mask = np.isfinite(aa) & np.isfinite(pa)
        metrics["acceleration_accuracy"] = (
            float(np.mean(np.sign(aa[mask]) == np.sign(pa[mask])))
            if mask.any()
            else None
        )

    abs_err = np.abs(err)
    if len(abs_err) >= 5:
        threshold = np.quantile(np.abs(actual), 0.80)
        tail_mask = np.abs(actual) >= threshold
        metrics["tail_mae"] = float(np.mean(abs_err[tail_mask])) if tail_mask.any() else None
    ds_month = scored["ds"].dt.strftime("%Y-%m")
    stress_mask = ds_month.isin(STRESS_PANEL_MONTHS).to_numpy()
    metrics["stress_panel_mae"] = (
        float(np.mean(abs_err[stress_mask])) if stress_mask.any() else None
    )
    metrics["n_stress_panel"] = int(stress_mask.sum())
    return metrics


def _load_reference(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["ds"])
    except Exception:
        return None
    if "predicted" not in df.columns or "actual" not in df.columns:
        return None
    out = df[["ds", "actual", "predicted"]].copy()
    out["ds"] = month_start(out["ds"])
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["predicted"] = pd.to_numeric(out["predicted"], errors="coerce")
    return out.dropna(subset=["ds"])


def _paired_reference_metrics(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    actual_col = "actual_mom" if "actual_mom" in predictions.columns else "actual"
    if actual_col not in predictions.columns:
        return {}
    pred = predictions[["ds", actual_col, "predicted_mom"]].copy()
    pred = pred.rename(columns={actual_col: "actual_sidecar", "predicted_mom": "sidecar"})
    merged = pred.merge(reference, on="ds", how="inner", suffixes=("", "_ref"))
    merged = merged[
        merged["sidecar"].notna()
        & merged["predicted"].notna()
        & merged["actual"].notna()
    ].copy()
    if merged.empty:
        return {f"paired_{label}_n": 0}

    actual = merged["actual"].to_numpy(dtype=float)
    side = merged["sidecar"].to_numpy(dtype=float)
    ref = merged["predicted"].to_numpy(dtype=float)
    side_abs = np.abs(side - actual)
    ref_abs = np.abs(ref - actual)
    side_sq = np.square(side - actual)
    ref_sq = np.square(ref - actual)
    out = {
        f"paired_{label}_n": int(len(merged)),
        f"paired_{label}_mae": float(np.mean(ref_abs)),
        f"paired_{label}_sidecar_mae_delta": float(np.mean(side_abs - ref_abs)),
        f"paired_{label}_rmse": float(np.sqrt(np.mean(ref_sq))),
        f"paired_{label}_sidecar_mse_delta": float(np.mean(side_sq - ref_sq)),
    }
    if "predicted_accel" in predictions.columns:
        p2 = predictions[["ds", "predicted_accel"]].copy()
        tmp = merged[["ds", "actual"]].merge(p2, on="ds", how="left")
        actual_acc = np.diff(tmp["actual"].to_numpy(dtype=float))
        side_acc = tmp["predicted_accel"].to_numpy(dtype=float)[1:]
        ref_acc = np.diff(tmp["predicted"].to_numpy(dtype=float))
        mask = np.isfinite(actual_acc) & np.isfinite(side_acc) & np.isfinite(ref_acc)
        if mask.any():
            out[f"paired_{label}_sidecar_accel_acc_delta"] = float(
                np.mean(np.sign(side_acc[mask]) == np.sign(actual_acc[mask]))
                - np.mean(np.sign(ref_acc[mask]) == np.sign(actual_acc[mask]))
            )
    return out


def add_reference_deltas(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    baseline_archive: str = "2026-05-12_165541",
) -> dict[str, Any]:
    refs = {
        "current_kalman": output_dir / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv",
        "consensus": output_dir / "consensus_anchor" / "baseline_consensus" / "backtest_results.csv",
        "baseline_2026_05_12_165541": output_dir / "Archive" / baseline_archive / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv",
    }
    out: dict[str, Any] = {}
    for label, path in refs.items():
        ref = _load_reference(path)
        if ref is not None:
            out.update(_paired_reference_metrics(predictions, ref, label))
    return out


def promotion_gate_passed(metrics: Mapping[str, Any]) -> bool:
    delta = metrics.get("paired_current_kalman_sidecar_mae_delta")
    accel_delta = metrics.get("paired_current_kalman_sidecar_accel_acc_delta")
    stress = metrics.get("stress_panel_mae")
    current_mae = metrics.get("paired_current_kalman_mae")
    if isinstance(delta, (int, float)) and delta <= 0:
        return True
    if (
        isinstance(delta, (int, float))
        and isinstance(accel_delta, (int, float))
        and abs(delta) <= 5.0
        and accel_delta >= 0.05
    ):
        return True
    if (
        isinstance(delta, (int, float))
        and isinstance(stress, (int, float))
        and isinstance(current_mae, (int, float))
        and delta <= 10.0
        and stress <= current_mae
    ):
        return True
    return False


def standardize_predictions(
    predictions: pd.DataFrame,
    *,
    model_id: str,
    target_space: str,
) -> pd.DataFrame:
    df = predictions.copy()
    df["ds"] = month_start(df["ds"])
    if "trained_through" not in df.columns:
        df["trained_through"] = df["ds"] - pd.offsets.MonthBegin(1)
    df["trained_through"] = month_start(df["trained_through"])
    df["model_id"] = model_id
    df["target_space"] = target_space
    if "predicted_accel" not in df.columns:
        df["predicted_accel"] = np.nan
    if "predicted_accel_sign" not in df.columns:
        df["predicted_accel_sign"] = np.sign(pd.to_numeric(df["predicted_accel"], errors="coerce"))
    if "predicted_accel_proba_up" not in df.columns:
        df["predicted_accel_proba_up"] = np.where(df["predicted_accel_sign"] > 0, 1.0, 0.0)
    if "confidence" not in df.columns:
        proba = pd.to_numeric(df["predicted_accel_proba_up"], errors="coerce")
        df["confidence"] = (proba - 0.5).abs() * 2.0
    if "uncertainty" not in df.columns:
        df["uncertainty"] = 1.0 - pd.to_numeric(df["confidence"], errors="coerce").clip(0, 1)
    for col in REQUIRED_PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered = [
        "ds",
        "model_id",
        "target_space",
        "predicted_mom",
        "predicted_accel",
        "predicted_accel_sign",
        "predicted_accel_proba_up",
        "confidence",
        "uncertainty",
        "trained_through",
    ]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra].sort_values("ds").reset_index(drop=True)


def validate_pit_predictions(predictions: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Sidecar predictions missing required columns: {sorted(missing)}")
    ds = pd.to_datetime(predictions["ds"], errors="coerce")
    trained = pd.to_datetime(predictions["trained_through"], errors="coerce")
    bad = predictions[trained >= ds]
    if not bad.empty:
        sample = bad[["ds", "trained_through"]].head(5).to_dict(orient="records")
        raise ValueError(f"trained_through must be strictly before ds; sample={sample}")


def write_sidecar_artifacts(
    *,
    output_dir: Path,
    model_id: str,
    target_space: str,
    predictions: pd.DataFrame,
    feature_audit: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
    data_paths: Mapping[str, Any] | None = None,
    write_legacy: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = standardize_predictions(predictions, model_id=model_id, target_space=target_space)
    validate_pit_predictions(preds)

    metrics = _base_metrics(preds)
    try:
        from settings import OUTPUT_DIR

        metrics.update(add_reference_deltas(preds, output_dir=OUTPUT_DIR))
    except Exception:
        pass
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics["promotion_gate_passed"] = promotion_gate_passed(metrics)

    audit = feature_audit.copy() if feature_audit is not None else pd.DataFrame()
    if audit.empty:
        audit = pd.DataFrame(
            [{"feature": "unknown", "source": "unknown", "missing_rate": None, "pit_cutoff": "trained_through"}]
        )

    manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "model_id": model_id,
        "target_space": target_space,
        "repo_version": repo_version(),
        "config": dict(config or {}),
        "data_paths": dict(data_paths or {}),
        "environment": {
            "NFP_SIDECAR_MODE": os.getenv("NFP_SIDECAR_MODE", "off"),
        },
        "n_predictions": int(len(preds)),
        "pit_validation": "trained_through < ds",
    }

    preds.to_csv(output_dir / "predictions.csv", index=False)
    audit.to_csv(output_dir / "feature_audit.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))

    for legacy_name, kind in (write_legacy or {}).items():
        if kind == "predictions":
            preds.to_csv(output_dir / legacy_name, index=False)
        elif kind == "metrics":
            (output_dir / legacy_name).write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))

    return preds, _json_safe(metrics)


def feature_audit_from_frame(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    source_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    source_map = source_map or {}
    for col in feature_cols:
        rows.append(
            {
                "feature": col,
                "source": source_map.get(col, "sidecar_design"),
                "missing_rate": float(frame[col].isna().mean()) if col in frame else None,
                "pit_cutoff": "trained_through",
            }
        )
    return pd.DataFrame(rows)
