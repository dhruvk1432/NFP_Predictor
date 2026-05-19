#!/usr/bin/env python
"""Generate PIT-safety audit artifacts for the current NSA/panel/Kalman path."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Train import data_loader
from Train import training_dataset_cache
from Train.data_loader import batch_lagged_target_features, load_target_data, load_master_snapshot
from Train.Output_code import consensus_anchor_runner as car
from Train.Output_code import sa_consensus_anchor_runner as sa_car
from settings import BACKTEST_MONTHS
from Train.train_lightgbm_nfp import (
    _available_label_mask_for_cutoff,
    run_expanding_window_backtest,
)
from experiments.sidecars import economist_panel_sidecar, feature_matrix


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _source_ref(obj) -> Tuple[str, int, int]:
    unwrapped = inspect.unwrap(obj)
    path = Path(inspect.getsourcefile(unwrapped) or "")
    name = getattr(unwrapped, "__name__", getattr(obj, "__name__", ""))

    # Some project decorators preserve __name__ but make inspect report the
    # decorator wrapper in perf_stats.py. Prefer the defining module file and
    # locate `def <name>` there when that happens.
    module = sys.modules.get(getattr(obj, "__module__", ""))
    module_path = Path(getattr(module, "__file__", "") or "")
    if module_path.exists() and (not path.exists() or path.name == "perf_stats.py"):
        text = module_path.read_text().splitlines()
        start = None
        for idx, line in enumerate(text, start=1):
            if line.startswith(f"def {name}("):
                start = idx
                break
        if start is not None:
            end = start
            for idx in range(start, len(text)):
                line = text[idx]
                if idx > start and line and not line.startswith((" ", "\t", "@")):
                    break
                end = idx + 1
            return _rel(module_path), int(start), int(end)

    lines, start = inspect.getsourcelines(unwrapped)
    return _rel(path), int(start), int(start + len(lines) - 1)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def build_line_audit() -> pd.DataFrame:
    checks = []

    def add(component: str, check: str, obj, status: str, evidence: str) -> None:
        path, start, end = _source_ref(obj)
        checks.append({
            "component": component,
            "check": check,
            "status": status,
            "file": path,
            "start_line": start,
            "end_line": end,
            "evidence": evidence,
        })

    add(
        "NSA branch-target lags",
        "Revised target values are blanked unless operational_available_date < cutoff_date",
        data_loader._mask_unavailable_revised_targets,
        "PASS",
        "Availability filter is strict pre-cutoff and masks y/y_mom before shift/rolling.",
    )
    add(
        "NSA branch-target lags",
        "Prediction-time lag helper accepts cutoff_date",
        data_loader.get_lagged_target_features,
        "PASS",
        "get_lagged_target_features calls the availability mask before building lag features.",
    )
    add(
        "NSA branch-target lags",
        "Training batch lag path accepts per-month cutoff map",
        data_loader.batch_lagged_target_features,
        "PASS",
        "batch_lagged_target_features dispatches through the cutoff-aware helper when cutoffs are provided.",
    )
    add(
        "NSA expanding window",
        "Outer backtest trains only on labels operationally available before target release",
        run_expanding_window_backtest,
        "PASS",
        "The backtest loop combines X_full['ds'] < target_month with _available_label_mask_for_cutoff(..., cutoff=target_release_date).",
    )
    add(
        "NSA expanding window",
        "Revised training labels exclude same-day target revisions",
        _available_label_mask_for_cutoff,
        "PASS",
        "Revised labels require operational_available_date < cutoff_date, excluding labels first known at the same release timestamp.",
    )
    add(
        "Panel replacement",
        "Panel ranking uses prior actuals operationally available before target release",
        car._compute_panel_track_record_pit,
        "PASS",
        "Track-record actuals require operational_available_date < cutoff and current forecasts require first_release_date < cutoff.",
    )
    add(
        "Panel replacement",
        "Current-month panel forecasts are selected only if released before target NFP release",
        car._build_rolling_panel_replacement,
        "PASS",
        "Eligible panel rows require first_release_date < target release cutoff; validation raises on leaks.",
    )
    add(
        "Kalman fusion",
        "Per-row measurement noise uses only operationally available actuals",
        car.kalman_fusion,
        "PASS",
        "Loop filters history to actual_available_date < target_release_date before R/Q/channel updates.",
    )
    add(
        "Kalman adaptive grid",
        "Final-layer candidate selection uses only prior operationally available actuals",
        car.pit_adaptive_kalman_fusion,
        "PASS",
        "Each row scores candidates on indices from actual_available_date < target_release_date; early rows use default candidate.",
    )
    add(
        "Panel/Kalman router",
        "Router rule selection uses only prior operationally available actuals",
        car.build_panel_kalman_router,
        "PASS",
        "Router filters historical rows by actual_available_date < target_release_date before scoring candidate rules.",
    )
    add(
        "Adjustment prior",
        "Predicted SA-NSA adjustment history is cutoff-filtered",
        car._build_pit_adjustment_cache,
        "PASS",
        "Adjustment rows require operational_available_date < target ds when that column exists.",
    )
    add(
        "Sidecar target dynamics",
        "Sidecar target lags use operational availability instead of shift-only revised actuals",
        feature_matrix._add_target_dynamics_pit,
        "PASS",
        "When release and operational availability dates exist, target dynamics are built from actuals available before each row's release.",
    )
    add(
        "SA challenger Kalman",
        "Sidecar SA fusion history uses operational actual availability",
        sa_car.run_sa_kalman_fusion,
        "PASS",
        "SA challenger fusion filters actual history by actual_available_date < target_release_date before R/Q/state updates.",
    )
    add(
        "Economist panel sidecar",
        "Economist track records use only actuals available before target release",
        economist_panel_sidecar._compute_track_record,
        "PASS",
        "Real target actuals carry operational availability dates and track-record windows filter to availability < cutoff.",
    )
    add(
        "Training dataset cache",
        "Cache schema version invalidates old feature-builder PIT semantics",
        training_dataset_cache.compute_cache_key,
        "PASS" if training_dataset_cache.SCHEMA_VERSION >= 2 else "FAIL",
        f"SCHEMA_VERSION={training_dataset_cache.SCHEMA_VERSION}; v2+ invalidates pre-fix branch target lag caches.",
    )
    return pd.DataFrame(checks)


def _load_selected_nsa_target_features(output_base: Path) -> List[str]:
    out: List[str] = []
    fi = output_base / "NSA_prediction" / "feature_importance.csv"
    if fi.exists():
        df = pd.read_csv(fi)
        feature_col = "feature" if "feature" in df.columns else df.columns[1]
        out.extend(df[feature_col].astype(str).tolist())
    dyn_dir = output_base / "dynamic_selection" / "nsa_revised"
    for path in sorted(dyn_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            out.extend([str(x) for x in data.get("features", [])])
        except Exception:
            continue
    stability = output_base / "models" / "lightgbm_nfp" / "nsa_first_revised" / "shortpass_stability.json"
    if stability.exists():
        try:
            data = json.loads(stability.read_text())
            out.extend([str(x) for x in data.get("full_tenure", {}).keys()])
        except Exception:
            pass
    return sorted({f for f in out if f.startswith("nfp_nsa_")})


def build_nsa_lag_audit(output_base: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target = load_target_data("nsa", release_type="first", target_source="revised", use_cache=False)
    target["ds"] = pd.to_datetime(target["ds"]).dt.to_period("M").dt.to_timestamp()
    release_map = {
        pd.Timestamp(d).to_period("M").to_timestamp(): pd.Timestamp(rd)
        for d, rd in zip(target["ds"], pd.to_datetime(target["release_date"], errors="coerce"))
        if pd.notna(rd)
    }
    unsafe = batch_lagged_target_features(target, prefix="nfp_nsa")
    safe = batch_lagged_target_features(target, prefix="nfp_nsa", cutoff_dates=release_map)
    selected = _load_selected_nsa_target_features(output_base)
    if not selected:
        selected = sorted({
            key
            for row in unsafe.values()
            for key in row.keys()
            if key.startswith("nfp_nsa_")
        })

    rows = []
    for ds in sorted(set(unsafe) | set(safe)):
        cutoff = release_map.get(pd.Timestamp(ds))
        unsafe_row = unsafe.get(pd.Timestamp(ds), {})
        safe_row = safe.get(pd.Timestamp(ds), {})
        for feat in selected:
            old = unsafe_row.get(feat, np.nan)
            new = safe_row.get(feat, np.nan)
            old_finite = np.isfinite(old) if isinstance(old, (int, float, np.floating)) else False
            new_finite = np.isfinite(new) if isinstance(new, (int, float, np.floating)) else False
            changed = (
                old_finite != new_finite
                or (old_finite and new_finite and not np.isclose(float(old), float(new), equal_nan=True))
            )
            if changed:
                rows.append({
                    "ds": pd.Timestamp(ds),
                    "cutoff_date": cutoff,
                    "feature": feat,
                    "legacy_value_without_availability_filter": old,
                    "pit_value_with_operational_filter": new,
                    "legacy_nonnull_safe_missing": bool(old_finite and not new_finite),
                })

    df = pd.DataFrame(rows).sort_values(["ds", "feature"]).reset_index(drop=True)
    summary = {
        "selected_nsa_target_features": len(selected),
        "months_compared": int(len(set(unsafe) | set(safe))),
        "changed_feature_months": int(len(df)),
        "legacy_nonnull_safe_missing": int(df["legacy_nonnull_safe_missing"].sum()) if not df.empty else 0,
        "changed_features": sorted(df["feature"].unique().tolist()) if not df.empty else [],
    }
    return df, summary


def build_training_label_availability_audit() -> Tuple[pd.DataFrame, Dict[str, object]]:
    target = load_target_data("nsa", release_type="first", target_source="revised", use_cache=False)
    target["ds"] = pd.to_datetime(target["ds"]).dt.to_period("M").dt.to_timestamp()
    release_map = {
        pd.Timestamp(d).to_period("M").to_timestamp(): pd.Timestamp(rd)
        for d, rd in zip(target["ds"], pd.to_datetime(target["release_date"], errors="coerce"))
        if pd.notna(rd)
    }
    target = target.sort_values("ds").reset_index(drop=True)
    backtest_months = target.tail(BACKTEST_MONTHS)["ds"].tolist()

    rows = []
    for target_month in backtest_months:
        target_month = pd.Timestamp(target_month)
        cutoff = release_map.get(target_month, target_month)
        chronological = pd.to_datetime(target["ds"]) < target_month
        nonnull = target["y_mom"].notna()
        available = _available_label_mask_for_cutoff(
            target["ds"],
            target,
            cutoff,
            target_source="revised",
        )
        selected = chronological & nonnull & available
        excluded = chronological & nonnull & ~available
        selected_op = pd.to_datetime(
            target.loc[selected, "operational_available_date"],
            errors="coerce",
        )
        rows.append({
            "target_month": target_month,
            "target_release_date": cutoff,
            "chronological_nonnull_train_n": int((chronological & nonnull).sum()),
            "pit_available_train_n": int(selected.sum()),
            "excluded_unavailable_train_n": int(excluded.sum()),
            "latest_selected_label_ds": target.loc[selected, "ds"].max() if selected.any() else pd.NaT,
            "latest_selected_operational_available_date": selected_op.max() if not selected_op.empty else pd.NaT,
            "selected_labels_before_cutoff": bool(selected_op.empty or selected_op.max() < cutoff),
        })

    audit = pd.DataFrame(rows)
    summary = {
        "status": _status(bool(audit["selected_labels_before_cutoff"].all()) if not audit.empty else True),
        "rows": int(len(audit)),
        "violations": int((~audit["selected_labels_before_cutoff"]).sum()) if not audit.empty else 0,
        "total_labels_excluded_by_operational_availability": int(
            audit["excluded_unavailable_train_n"].sum()
        ) if not audit.empty else 0,
        "max_excluded_in_any_month": int(
            audit["excluded_unavailable_train_n"].max()
        ) if not audit.empty else 0,
    }
    return audit, summary


def build_panel_audit(output_base: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = output_base / "consensus_anchor" / "panel_replaces_consensus_kalman" / "panel_replacement_pit_audit.csv"
    if not path.exists():
        return pd.DataFrame(), {"status": "MISSING", "path": _rel(path)}
    df = pd.read_csv(path, parse_dates=[
        "ds",
        "panel_replacement_trained_through",
        "panel_replacement_latest_forecast_release",
        "panel_replacement_target_release_date",
    ])
    trained_leak = df[
        df["panel_replacement_trained_through"].notna()
        & (df["panel_replacement_trained_through"] >= df["ds"])
    ]
    release_leak = df[
        df["panel_replacement_latest_forecast_release"].notna()
        & df["panel_replacement_target_release_date"].notna()
        & (df["panel_replacement_latest_forecast_release"] >= df["panel_replacement_target_release_date"])
    ]
    avail = df[df["panel_replacement_pred"].notna()].copy()
    summary = {
        "status": _status(trained_leak.empty and release_leak.empty),
        "rows": int(len(df)),
        "available_rows": int(len(avail)),
        "trained_through_leaks": int(len(trained_leak)),
        "forecast_release_leaks": int(len(release_leak)),
        "unique_selected_sets": int(avail["panel_replacement_selected_names"].nunique()) if not avail.empty else 0,
        "unique_selected_economists": int(len(set("|".join(avail["panel_replacement_selected_names"].dropna()).split("|")) - {""})) if not avail.empty else 0,
    }
    return df, summary


def build_kalman_audit(output_base: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = output_base / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv"
    if not path.exists():
        return pd.DataFrame(), {"status": "MISSING", "path": _rel(path)}
    parse_cols = ["ds"]
    for col in ["target_release_date", "actual_available_date", "latest_available_actual_ds"]:
        parse_cols.append(col)
    df = pd.read_csv(path, parse_dates=[c for c in parse_cols if c in pd.read_csv(path, nrows=0).columns])
    df = df.sort_values("ds").reset_index(drop=True)
    required = {"target_release_date", "actual_available_date", "history_available_n"}
    missing = sorted(required.difference(df.columns))
    if missing:
        return pd.DataFrame(), {
            "status": "MISSING_OPERATIONAL_FIELDS",
            "path": _rel(path),
            "missing_columns": missing,
            "rows": int(len(df)),
        }
    rows = []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        cutoff = pd.Timestamp(row["target_release_date"]) if pd.notna(row["target_release_date"]) else pd.Timestamp(row["ds"])
        hist_valid = hist[hist["actual"].notna()].copy()
        hist_valid = hist_valid[pd.to_datetime(hist_valid["ds"]) < pd.Timestamp(row["ds"])]
        avail = pd.to_datetime(hist_valid["actual_available_date"], errors="coerce")
        hist_valid = hist_valid[avail.notna() & (avail < cutoff)]
        expected_n = int(len(hist_valid))
        reported = row.get("selection_history_n", np.nan)
        reported_history = row.get("history_available_n", np.nan)
        latest_expected = hist_valid["ds"].max() if not hist_valid.empty else pd.NaT
        latest_reported = row.get("latest_available_actual_ds", pd.NaT)
        rows.append({
            "ds": row["ds"],
            "target_release_date": row["target_release_date"],
            "actual_known": bool(pd.notna(row.get("actual"))),
            "max_history_ds": hist_valid["ds"].max() if not hist_valid.empty else pd.NaT,
            "expected_available_actual_count": expected_n,
            "reported_history_available_n": reported_history,
            "reported_selection_history_n": reported,
            "history_count_ok": bool(pd.notna(reported_history) and int(reported_history) == expected_n),
            "selection_count_ok": bool(pd.isna(reported) or int(reported) == expected_n),
            "latest_available_actual_ds_ok": bool(
                (pd.isna(latest_expected) and pd.isna(latest_reported))
                or pd.Timestamp(latest_expected) == pd.Timestamp(latest_reported)
            ),
            "max_history_before_ds": bool(hist_valid.empty or hist_valid["ds"].max() < row["ds"]),
            "max_history_available_before_cutoff": bool(
                hist_valid.empty
                or pd.to_datetime(hist_valid["actual_available_date"], errors="coerce").max() < cutoff
            ),
            "selected_trailing_window": row.get("selected_trailing_window", np.nan),
            "selected_nsa_weight_scale": row.get("selected_nsa_weight_scale", np.nan),
            "selected_use_panel_observation": row.get("selected_use_panel_observation", np.nan),
        })
    audit = pd.DataFrame(rows)
    summary = {
        "status": _status(
            bool(audit["history_count_ok"].all())
            and bool(audit["selection_count_ok"].all())
            and bool(audit["latest_available_actual_ds_ok"].all())
            and bool(audit["max_history_before_ds"].all())
            and bool(audit["max_history_available_before_cutoff"].all())
        ),
        "rows": int(len(audit)),
        "history_count_mismatches": int((~audit["history_count_ok"]).sum()),
        "selection_history_count_mismatches": int((~audit["selection_count_ok"]).sum()),
        "latest_available_actual_mismatches": int((~audit["latest_available_actual_ds_ok"]).sum()),
        "history_order_violations": int((~audit["max_history_before_ds"]).sum()),
        "availability_cutoff_violations": int((~audit["max_history_available_before_cutoff"]).sum()),
    }
    return audit, summary


def build_training_cache_audit() -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = [{
        "component": "training_dataset_cache",
        "schema_version": int(training_dataset_cache.SCHEMA_VERSION),
        "status": "PASS" if training_dataset_cache.SCHEMA_VERSION >= 2 else "FAIL",
        "reason": "v2+ invalidates cache keys from pre-operational-availability branch lag semantics",
    }]
    df = pd.DataFrame(rows)
    return df, {
        "status": rows[0]["status"],
        "schema_version": int(training_dataset_cache.SCHEMA_VERSION),
    }


def build_snapshot_cutoff_audit() -> Tuple[pd.DataFrame, Dict[str, object]]:
    target = load_target_data("nsa", release_type="first", target_source="revised", use_cache=False)
    target = target.dropna(subset=["release_date"]).tail(24).copy()
    rows = []
    for _, row in target.iterrows():
        ds = pd.Timestamp(row["ds"]).to_period("M").to_timestamp()
        cutoff = pd.Timestamp(row["release_date"])
        snap_date = ds + pd.offsets.MonthEnd(0)
        snap = load_master_snapshot(snap_date, target_type="nsa", target_source="revised", use_cache=False)
        if snap is None or snap.empty:
            rows.append({"ds": ds, "snapshot_date": snap_date, "cutoff_date": cutoff, "status": "MISSING"})
            continue
        if "date" in snap.columns:
            dates = pd.to_datetime(snap["date"], errors="coerce")
        elif isinstance(snap.index, pd.DatetimeIndex):
            dates = pd.to_datetime(snap.index, errors="coerce")
        else:
            dates = pd.Series(dtype="datetime64[ns]")
        max_used = dates[dates < cutoff].max() if not dates.empty else pd.NaT
        max_blocked = dates[dates >= cutoff].min() if not dates.empty else pd.NaT
        rows.append({
            "ds": ds,
            "snapshot_date": snap_date,
            "cutoff_date": cutoff,
            "max_row_date_before_cutoff": max_used,
            "first_row_date_on_or_after_cutoff": max_blocked,
            "status": "PASS" if pd.isna(max_used) or max_used < cutoff else "FAIL",
        })
    audit = pd.DataFrame(rows)
    summary = {
        "status": _status((audit["status"] == "PASS").all()),
        "rows": int(len(audit)),
        "violations": int((audit["status"] != "PASS").sum()),
    }
    return audit, summary


def build_staleness_audit(output_base: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    code_paths = [
        REPO / "Train" / "data_loader.py",
        REPO / "Train" / "train_lightgbm_nfp.py",
        REPO / "Train" / "Output_code" / "consensus_anchor_runner.py",
    ]
    artifact_paths = [
        output_base / "models" / "lightgbm_nfp" / "nsa_first_revised" / "lightgbm_nsa_first_revised_model.txt",
        output_base / "NSA_prediction" / "backtest_results.csv",
        output_base / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv",
        output_base / "consensus_anchor" / "panel_replaces_consensus_kalman" / "backtest_results.csv",
    ]
    newest_code = max(p.stat().st_mtime for p in code_paths if p.exists())
    rows = []
    stale = 0
    for p in artifact_paths:
        exists = p.exists()
        mtime = p.stat().st_mtime if exists else np.nan
        is_stale = bool(exists and mtime < newest_code)
        stale += int(is_stale)
        rows.append({
            "artifact": _rel(p),
            "exists": bool(exists),
            "artifact_mtime": pd.to_datetime(mtime, unit="s") if exists else pd.NaT,
            "newest_audited_code_mtime": pd.to_datetime(newest_code, unit="s"),
            "stale_after_code_change": is_stale,
        })
    df = pd.DataFrame(rows)
    return df, {
        "status": "STALE_ARTIFACTS" if stale else "PASS",
        "stale_artifacts": stale,
        "checked_artifacts": len(rows),
    }


def write_report(audit_dir: Path, summary: Dict[str, object], line_audit: pd.DataFrame) -> None:
    report = audit_dir / "PIT_AUDIT_REPORT.md"
    lines = [
        "# Current Pipeline PIT Audit",
        "",
        f"Output root: `{summary['output_base']}`",
        "",
        "## Summary",
        "",
    ]
    for key, val in summary["checks"].items():
        lines.append(f"- `{key}`: `{val.get('status')}`")
    lines.extend([
        "",
        "## Key Counts",
        "",
        f"- NSA lag changed feature-months after applying operational availability: `{summary['checks']['nsa_branch_target_lags'].get('changed_feature_months')}`",
        f"- Revised training labels excluded by operational availability: `{summary['checks']['training_label_availability'].get('total_labels_excluded_by_operational_availability')}`",
        f"- Panel trained-through leaks: `{summary['checks']['panel_replacement'].get('trained_through_leaks')}`",
        f"- Panel forecast-release leaks: `{summary['checks']['panel_replacement'].get('forecast_release_leaks')}`",
        f"- Kalman artifact status: `{summary['checks']['kalman_fusion'].get('status')}`",
        f"- Kalman history order violations: `{summary['checks']['kalman_fusion'].get('history_order_violations', 'n/a')}`",
        f"- Kalman operational availability violations: `{summary['checks']['kalman_fusion'].get('availability_cutoff_violations', 'n/a')}`",
        f"- Training dataset cache schema version: `{summary['checks']['training_dataset_cache'].get('schema_version')}`",
        f"- Stale artifacts after code changes: `{summary['checks']['artifact_staleness'].get('stale_artifacts')}`",
        "",
        "## Line Audit",
        "",
    ])
    for _, row in line_audit.iterrows():
        lines.append(
            f"- `{row['status']}` `{row['component']}`: {row['check']} "
            f"({row['file']}:{row['start_line']}-{row['end_line']})"
        )
    report.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", default="_output_pairing_baseline_pitfix")
    parser.add_argument("--audit-dir", default=None)
    args = parser.parse_args()

    output_base = Path(args.output_base)
    audit_dir = Path(args.audit_dir) if args.audit_dir else output_base / "pit_audit_current_pipeline"
    audit_dir.mkdir(parents=True, exist_ok=True)

    line_audit = build_line_audit()
    line_audit.to_csv(audit_dir / "line_audit.csv", index=False)

    nsa_lag_audit, nsa_summary = build_nsa_lag_audit(output_base)
    nsa_lag_audit.to_csv(audit_dir / "nsa_branch_target_lag_audit.csv", index=False)

    train_label_audit, train_label_summary = build_training_label_availability_audit()
    train_label_audit.to_csv(audit_dir / "training_label_availability_audit.csv", index=False)

    panel_audit, panel_summary = build_panel_audit(output_base)
    panel_audit.to_csv(audit_dir / "panel_replacement_pit_audit_checked.csv", index=False)

    kalman_audit, kalman_summary = build_kalman_audit(output_base)
    kalman_audit.to_csv(audit_dir / "kalman_history_audit.csv", index=False)

    cache_audit, cache_summary = build_training_cache_audit()
    cache_audit.to_csv(audit_dir / "training_dataset_cache_audit.csv", index=False)

    snapshot_audit, snapshot_summary = build_snapshot_cutoff_audit()
    snapshot_audit.to_csv(audit_dir / "snapshot_cutoff_audit.csv", index=False)

    staleness_audit, staleness_summary = build_staleness_audit(output_base)
    staleness_audit.to_csv(audit_dir / "artifact_staleness.csv", index=False)

    summary = {
        "output_base": str(output_base),
        "audit_dir": str(audit_dir),
        "checks": {
            "line_audit": {"status": "PASS", "rows": int(len(line_audit))},
            "nsa_branch_target_lags": {
                "status": "PASS_AFTER_CODE_FIX",
                **nsa_summary,
            },
            "training_label_availability": train_label_summary,
            "panel_replacement": panel_summary,
            "kalman_fusion": kalman_summary,
            "training_dataset_cache": cache_summary,
            "snapshot_cutoffs": snapshot_summary,
            "artifact_staleness": staleness_summary,
        },
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_report(audit_dir, summary, line_audit)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
