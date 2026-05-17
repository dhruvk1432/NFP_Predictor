"""Gated sidecar integration helpers for training and Kalman replay.

Sidecar artifacts live under
``<output_dir>/sidecars/<target_type>/<run_id>/<sidecar_name>/`` for the SA
challenger and any new branch-specific sidecars. The legacy NSA pipeline
wrote sidecars to ``<output_dir>/sidecars/<run_id>/<sidecar_name>/`` (no
target_type subdir). ``_resolve_run_dir`` honors both layouts: when
``target_type='nsa'`` (the default), it first looks under the new
``sidecars/nsa/`` subtree and falls back to the legacy flat layout if
that path is empty. This keeps the live NSA production runner working
without a one-shot migration step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from experiments.sidecars.common import sanitize_model_id


VALID_SIDECAR_MODES = {"off", "features", "fusion", "router", "all"}
VALID_TARGET_TYPES = {"nsa", "sa"}


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


def _normalize_target_type(target_type: Optional[str]) -> str:
    """Return a lowercase, validated target_type. Defaults to env var or 'nsa'."""
    if target_type is None:
        target_type = os.getenv("NFP_SIDECAR_TARGET_TYPE", "nsa").strip().lower() or "nsa"
    target_type = str(target_type).strip().lower()
    if target_type not in VALID_TARGET_TYPES:
        return "nsa"
    return target_type


def _candidate_run_bases(output_dir: Path, target_type: str) -> list[Path]:
    """Return run-base directories to search, in priority order.

    For ``target_type='nsa'`` we look first under the new
    ``sidecars/nsa/`` subtree and fall back to the legacy flat
    ``sidecars/`` layout (which is where the live NSA production runner
    has been writing). For ``target_type='sa'`` we only look under
    ``sidecars/sa/`` so the SA challenger artifacts are strictly
    isolated.
    """
    branch_base = output_dir / "sidecars" / target_type
    if target_type == "nsa":
        return [branch_base, output_dir / "sidecars"]
    return [branch_base]


def _resolve_run_dir(
    output_dir: Path,
    target_type: Optional[str] = None,
) -> Path | None:
    target = _normalize_target_type(target_type)
    run_id = os.getenv("NFP_SIDECAR_RUN_ID", "").strip()
    for base in _candidate_run_bases(output_dir, target):
        if run_id:
            path = base / run_id
            if path.exists():
                return path
            continue
        if not base.exists():
            continue
        # The base directory may contain either run_id subdirs (new layout
        # under sidecars/<target_type>/) or — for the legacy NSA flat
        # layout — also a per-target-type subdir like sidecars/sa/. Filter
        # out any directory whose name happens to be a known target_type
        # so legacy NSA discovery doesn't grab the SA branch.
        candidates = [
            p for p in base.iterdir()
            if p.is_dir() and p.name not in VALID_TARGET_TYPES
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _prediction_files(
    output_dir: Path,
    target_type: Optional[str] = None,
) -> list[Path]:
    run_dir = _resolve_run_dir(output_dir, target_type=target_type)
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
    target_type: Optional[str] = None,
) -> pd.DataFrame:
    """Load standardized sidecar predictions as a wide ds-indexed feature frame.

    ``target_type`` selects the branch subtree (``sidecars/<target_type>/``).
    Defaults to the ``NFP_SIDECAR_TARGET_TYPE`` env var, falling back to
    ``'nsa'``. The 'nsa' variant also falls back to the legacy flat
    ``sidecars/`` layout if the new subtree is empty.
    """
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
    for path in _prediction_files(output_dir, target_type=target_type):
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
    merged = _add_sidecar_meta_features(merged)
    return merged.sort_values("ds").reset_index(drop=True)


def _sidecar_cols(frame: pd.DataFrame, suffix: str) -> list[str]:
    return [c for c in frame.columns if c.startswith("sidecar_") and c.endswith(suffix)]


def _add_sidecar_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact ensemble features for downstream LightGBM pairing.

    Raw sidecar columns are still exposed, but these meta columns give the main
    model stable, low-dimensional summaries: mean acceleration prior, ensemble
    probability, disagreement, and confidence-weighted signed acceleration.
    """
    out = frame.copy()
    accel_cols = _sidecar_cols(out, "__predicted_accel")
    mom_cols = _sidecar_cols(out, "__predicted_mom")
    proba_cols = _sidecar_cols(out, "__predicted_accel_proba_up")
    conf_cols = _sidecar_cols(out, "__confidence")
    sign_cols = _sidecar_cols(out, "__predicted_accel_sign")
    nudge_cols = _sidecar_cols(out, "__suggested_nudge")
    risk_cols = _sidecar_cols(out, "__regime_transition_risk") + _sidecar_cols(out, "__uncertainty")

    if accel_cols:
        accel = out[accel_cols].apply(pd.to_numeric, errors="coerce")
        out["sidecar_meta__predicted_accel_mean"] = accel.mean(axis=1)
        out["sidecar_meta__predicted_accel_std"] = accel.std(axis=1)
        out["sidecar_meta__predicted_accel_abs_max"] = accel.abs().max(axis=1)
    if mom_cols:
        mom = out[mom_cols].apply(pd.to_numeric, errors="coerce")
        out["sidecar_meta__predicted_mom_mean"] = mom.mean(axis=1)
        out["sidecar_meta__predicted_mom_std"] = mom.std(axis=1)
    if proba_cols:
        proba = out[proba_cols].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
        out["sidecar_meta__accel_proba_up_mean"] = proba.mean(axis=1)
        out["sidecar_meta__accel_proba_up_std"] = proba.std(axis=1)
    if conf_cols:
        conf = out[conf_cols].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
        out["sidecar_meta__confidence_mean"] = conf.mean(axis=1)
        out["sidecar_meta__confidence_max"] = conf.max(axis=1)
    if sign_cols:
        signs = out[sign_cols].apply(pd.to_numeric, errors="coerce")
        out["sidecar_meta__accel_sign_vote"] = np.sign(signs.mean(axis=1))
        out["sidecar_meta__accel_sign_disagreement"] = 1.0 - signs.abs().mean(axis=1).clip(0.0, 1.0)
    if accel_cols and conf_cols:
        accel = out[accel_cols].apply(pd.to_numeric, errors="coerce")
        conf = out[conf_cols].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
        conf = conf.rename(columns={
            c: c.replace("__confidence", "__predicted_accel") for c in conf.columns
        })
        common = [c for c in accel.columns if c in conf.columns]
        if common:
            numerator = (accel[common] * conf[common]).sum(axis=1, min_count=1)
            denom = conf[common].sum(axis=1).replace(0.0, np.nan)
            out["sidecar_meta__confidence_weighted_accel"] = numerator / denom
    if nudge_cols:
        nudges = out[nudge_cols].apply(pd.to_numeric, errors="coerce")
        out["sidecar_meta__suggested_nudge_mean"] = nudges.mean(axis=1)
        out["sidecar_meta__suggested_nudge_abs_max"] = nudges.abs().max(axis=1)
    if risk_cols:
        risk = out[risk_cols].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
        out["sidecar_meta__risk_mean"] = risk.mean(axis=1)
        out["sidecar_meta__risk_max"] = risk.max(axis=1)
    return out


def attach_sidecar_features(
    X: pd.DataFrame,
    *,
    output_dir: Path,
    logger=None,
    target_type: Optional[str] = None,
) -> pd.DataFrame:
    if not sidecar_features_enabled():
        return X
    sidecars = load_sidecar_feature_frame(output_dir=output_dir, target_type=target_type)
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
    target_type: Optional[str] = None,
) -> pd.DataFrame:
    if not (sidecar_fusion_enabled() or sidecar_router_enabled()):
        return df
    sidecars = load_sidecar_feature_frame(output_dir=output_dir, target_type=target_type)
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
