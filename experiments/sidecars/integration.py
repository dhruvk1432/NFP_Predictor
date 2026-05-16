"""Gated sidecar integration helpers for training and Kalman replay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.sidecars.common import sanitize_model_id


VALID_SIDECAR_MODES = {"off", "features", "fusion", "router", "all"}


def sidecar_mode() -> str:
    mode = os.getenv("NFP_SIDECAR_MODE", "off").strip().lower() or "off"
    if mode not in VALID_SIDECAR_MODES:
        return "off"
    return mode


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "t", "y"}


def sidecar_features_enabled() -> bool:
    mode = sidecar_mode()
    return mode in {"features", "all"} or _env_bool("NFP_ENABLE_SIDECAR_FEATURES", False)


def sidecar_fusion_enabled() -> bool:
    mode = sidecar_mode()
    return mode in {"fusion", "all"} or _env_bool("NFP_ENABLE_SIDECAR_FUSION", False)


def sidecar_router_enabled() -> bool:
    return sidecar_mode() in {"router", "all"}


def sidecar_require_passing_gate() -> bool:
    return _env_bool("NFP_SIDECAR_REQUIRE_PASSING_GATE", True)


def sidecar_max_precision_share() -> float:
    raw = os.getenv("NFP_SIDECAR_MAX_PRECISION_SHARE", "0.15").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.15


def _resolve_run_dir(output_dir: Path) -> Path | None:
    run_id = os.getenv("NFP_SIDECAR_RUN_ID", "").strip()
    base = output_dir / "sidecars"
    if run_id:
        path = base / run_id
        return path if path.exists() else None
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _prediction_files(output_dir: Path) -> list[Path]:
    run_dir = _resolve_run_dir(output_dir)
    if run_dir is None:
        return []
    return sorted(run_dir.glob("*/predictions.csv"))


def _metrics_pass(path: Path, require_gate: bool) -> bool:
    if not require_gate:
        return True
    metrics_path = path.parent / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        return False
    return bool(metrics.get("promotion_gate_passed", False))


def load_sidecar_feature_frame(
    *,
    output_dir: Path,
    require_passing_gate: bool | None = None,
    include_model_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load standardized sidecar predictions as a wide ds-indexed feature frame."""
    require_gate = sidecar_require_passing_gate() if require_passing_gate is None else require_passing_gate
    include = {sanitize_model_id(x) for x in include_model_ids} if include_model_ids else None
    frames: list[pd.DataFrame] = []
    value_cols = [
        "predicted_mom",
        "predicted_accel",
        "predicted_accel_sign",
        "predicted_accel_proba_up",
        "confidence",
        "uncertainty",
        "suggested_nudge",
    ]
    for path in _prediction_files(output_dir):
        if not _metrics_pass(path, require_gate):
            continue
        try:
            df = pd.read_csv(path, parse_dates=["ds", "trained_through"])
        except Exception:
            continue
        if df.empty or "model_id" not in df.columns:
            continue
        model_id = sanitize_model_id(str(df["model_id"].dropna().iloc[0]))
        if include is not None and model_id not in include:
            continue
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df["trained_through"] = pd.to_datetime(df["trained_through"], errors="coerce")
        df = df[df["trained_through"] < df["ds"]].copy()
        if df.empty:
            continue
        keep_cols = [c for c in value_cols if c in df.columns]
        keep_cols.extend([c for c in df.columns if c.startswith("regime_")])
        if not keep_cols:
            continue
        out = df[["ds"] + keep_cols].copy()
        rename = {c: f"sidecar_{model_id}__{c}" for c in keep_cols}
        frames.append(out.rename(columns=rename))
    if not frames:
        return pd.DataFrame(columns=["ds"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="ds", how="outer")
    return merged.sort_values("ds").reset_index(drop=True)


def attach_sidecar_features(
    X: pd.DataFrame,
    *,
    output_dir: Path,
    logger=None,
) -> pd.DataFrame:
    if not sidecar_features_enabled():
        return X
    sidecars = load_sidecar_feature_frame(output_dir=output_dir)
    if sidecars.empty or len(sidecars.columns) <= 1:
        if logger is not None:
            logger.warning("Sidecar feature mode enabled, but no passing sidecar features were loaded")
        return X
    out = X.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out = out.merge(sidecars, on="ds", how="left")
    if logger is not None:
        logger.info("Attached %d gated sidecar feature columns", len(sidecars.columns) - 1)
    return out


def merge_sidecar_observations(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    logger=None,
) -> pd.DataFrame:
    if not (sidecar_fusion_enabled() or sidecar_router_enabled()):
        return df
    sidecars = load_sidecar_feature_frame(output_dir=output_dir)
    if sidecars.empty or len(sidecars.columns) <= 1:
        if logger is not None:
            logger.warning("Sidecar fusion/router enabled, but no passing sidecar observations were loaded")
        return df
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out = out.merge(sidecars, on="ds", how="left")
    if logger is not None:
        logger.info("Merged %d gated sidecar observation/router columns", len(sidecars.columns) - 1)
    return out


def sidecar_observation_columns(df: pd.DataFrame) -> list[str]:
    if not sidecar_fusion_enabled():
        return []
    return [c for c in df.columns if c.startswith("sidecar_") and c.endswith("__predicted_mom")]


def sidecar_router_values(row: pd.Series) -> tuple[float, float]:
    if not sidecar_router_enabled():
        return 0.0, 0.0
    risks = []
    confidences = []
    for col, value in row.items():
        if not str(col).startswith("sidecar_"):
            continue
        if str(col).endswith("__regime_transition_risk") or str(col).endswith("__uncertainty"):
            if pd.notna(value):
                risks.append(float(np.clip(value, 0.0, 1.0)))
        if str(col).endswith("__confidence") and pd.notna(value):
            confidences.append(float(np.clip(value, 0.0, 1.0)))
    risk = float(np.mean(risks)) if risks else 0.0
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return risk, confidence
