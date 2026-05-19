"""Evaluate HMM reselection trigger sparsity and event alignment."""

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

from Train.hmm_regime_reselection.trigger import KNOWN_ECONOMIC_WINDOWS


MAJOR_EVENT_LABELS = {
    "dotcom_recession",
    "global_financial_crisis",
    "covid_crash",
    "covid_reopening_recovery",
    "inflation_tightening",
}


def _month_index(ts: pd.Timestamp) -> int:
    return int(ts.year) * 12 + int(ts.month)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "step_date" not in df.columns:
        raise ValueError(f"{path} is missing step_date")
    df["ds"] = pd.to_datetime(df["step_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if "final_reselect" not in df.columns:
        df["final_reselect"] = df.get("hmm_should_reselect", False)
    for col in (
        "hmm_surprise_ratio",
        "hmm_transition_risk",
        "hmm_expected_duration",
        "hmm_state_support_n",
        "hmm_state_support_share",
        "hmm_trigger_score",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hmm_month_of_year" not in df.columns:
        df["hmm_month_of_year"] = df["ds"].dt.month
    df["event_window"] = df.get("event_window", pd.Series(index=df.index, dtype=object)).replace("", np.nan)
    return df


def _trigger_mask(df: pd.DataFrame) -> pd.Series:
    raw = df["final_reselect"]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _cluster_triggers(triggers: pd.DataFrame, gap_months: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current: list[pd.Series] = []
    prev_ds: pd.Timestamp | None = None
    for _, row in triggers.sort_values("ds").iterrows():
        ds = pd.Timestamp(row["ds"])
        if prev_ds is not None and _month_index(ds) - _month_index(prev_ds) > int(gap_months):
            rows.append(_summarize_cluster(current))
            current = []
        current.append(row)
        prev_ds = ds
    if current:
        rows.append(_summarize_cluster(current))
    return pd.DataFrame(rows)


def _summarize_cluster(rows: list[pd.Series]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    events = sorted(str(v) for v in frame["event_window"].dropna().unique())
    return {
        "cluster_start": pd.Timestamp(frame["ds"].min()).strftime("%Y-%m"),
        "cluster_end": pd.Timestamp(frame["ds"].max()).strftime("%Y-%m"),
        "n_triggers": int(len(frame)),
        "trigger_dates": ",".join(pd.to_datetime(frame["ds"]).dt.strftime("%Y-%m")),
        "regime_labels": ",".join(sorted(str(v) for v in frame["hmm_regime_label"].dropna().unique())),
        "trigger_classes": ",".join(sorted(str(v) for v in frame["hmm_trigger_class"].dropna().unique())),
        "event_windows": ",".join(events),
        "overlaps_event": bool(events),
        "max_surprise_ratio": float(frame["hmm_surprise_ratio"].max(skipna=True))
        if "hmm_surprise_ratio" in frame.columns
        else np.nan,
        "max_trigger_score": float(frame["hmm_trigger_score"].max(skipna=True))
        if "hmm_trigger_score" in frame.columns
        else np.nan,
    }


def _event_recall(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_events = clusters["event_windows"].fillna("").astype(str) if not clusters.empty else pd.Series(dtype=str)
    for start, end, label in KNOWN_ECONOMIC_WINDOWS:
        if label not in MAJOR_EVENT_LABELS:
            continue
        hit = bool(cluster_events.str.contains(label, regex=False).any())
        rows.append({"event_window": label, "start": start, "end": end, "hit": hit})
    return pd.DataFrame(rows)


def _semiannual_crash_repeats(triggers: pd.DataFrame) -> pd.DataFrame:
    part = triggers[
        triggers["ds"].dt.year.between(2011, 2020)
        & triggers["ds"].dt.month.isin([1, 7])
        & triggers["hmm_regime_label"].eq("crash")
    ].copy()
    if part.empty:
        return pd.DataFrame(columns=["year", "jan_trigger", "jul_trigger"])
    part["year"] = part["ds"].dt.year
    part["month"] = part["ds"].dt.month
    out = (
        part.assign(value=True)
        .pivot_table(index="year", columns="month", values="value", aggfunc="any", fill_value=False)
        .rename(columns={1: "jan_trigger", 7: "jul_trigger"})
        .reset_index()
    )
    for col in ("jan_trigger", "jul_trigger"):
        if col not in out.columns:
            out[col] = False
    return out[["year", "jan_trigger", "jul_trigger"]]


def evaluate(df: pd.DataFrame, *, cluster_gap_months: int, last_n_months: int) -> dict[str, Any]:
    mask = _trigger_mask(df)
    triggers = df[mask].copy()
    clusters = _cluster_triggers(triggers, cluster_gap_months)
    event_recall = _event_recall(clusters)
    jan_jul = triggers["ds"].dt.month.isin([1, 7])
    last_cutoff = df["ds"].max() - pd.DateOffset(months=int(last_n_months) - 1)
    last_triggers = triggers[triggers["ds"] >= last_cutoff]
    event_clusters = int(clusters["overlaps_event"].sum()) if not clusters.empty else 0
    precision = event_clusters / len(clusters) if len(clusters) else 0.0
    recall = float(event_recall["hit"].mean()) if not event_recall.empty else 0.0
    semiannual = _semiannual_crash_repeats(triggers)
    repeated_semiannual = int((semiannual["jan_trigger"] & semiannual["jul_trigger"]).sum()) if not semiannual.empty else 0
    jan_jul_share = float(jan_jul.mean()) if len(triggers) else 0.0
    summary = {
        "n_rows": int(len(df)),
        "n_triggers": int(len(triggers)),
        "n_clusters": int(len(clusters)),
        "last_period_start": pd.Timestamp(last_cutoff).strftime("%Y-%m"),
        "last_period_triggers": int(len(last_triggers)),
        "jan_jul_triggers": int(jan_jul.sum()),
        "jan_jul_share": jan_jul_share,
        "cluster_event_precision": float(precision),
        "major_event_recall": recall,
        "repeated_semiannual_crash_years_2011_2020": repeated_semiannual,
        "eligible_for_feature_dry_run": bool(
            len(triggers) <= 20
            and len(last_triggers) <= 8
            and precision >= 0.50
            and recall >= 0.75
            and jan_jul_share <= 0.35
            and repeated_semiannual == 0
        ),
    }
    return {
        "summary": summary,
        "triggers": triggers,
        "clusters": clusters,
        "event_recall": event_recall,
        "semiannual_crash_repeats": semiannual,
    }


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["triggers"].to_csv(output_dir / "trigger_dates.csv", index=False)
    result["clusters"].to_csv(output_dir / "trigger_clusters.csv", index=False)
    result["event_recall"].to_csv(output_dir / "event_recall.csv", index=False)
    result["semiannual_crash_repeats"].to_csv(output_dir / "semiannual_crash_repeats.csv", index=False)

    triggers = result["triggers"]
    by_year = triggers.assign(year=triggers["ds"].dt.year).groupby("year").size().rename("n").reset_index()
    by_month = triggers.assign(month_of_year=triggers["ds"].dt.month).groupby("month_of_year").size().rename("n").reset_index()
    by_year.to_csv(output_dir / "trigger_count_by_year.csv", index=False)
    by_month.to_csv(output_dir / "trigger_count_by_month_of_year.csv", index=False)

    false_positive = triggers[triggers["event_window"].isna()].copy()
    false_positive.to_csv(output_dir / "false_positive_candidates.csv", index=False)
    missed_high_surprise = result.get("missed_high_surprise")
    if isinstance(missed_high_surprise, pd.DataFrame):
        missed_high_surprise.to_csv(output_dir / "missed_high_surprise_months.csv", index=False)

    state_cols = [
        c for c in [
            "step_date",
            "hmm_regime_label",
            "hmm_state_support_n",
            "hmm_state_support_share",
            "hmm_expected_duration",
            "hmm_surprise_ratio",
            "hmm_trigger_score",
            "hmm_structural_gate_reason",
            "final_reselect",
        ]
        if c in result["all_rows"].columns
    ]
    result["all_rows"][state_cols].to_csv(output_dir / "state_support_duration_diagnostics.csv", index=False)

    (output_dir / "trigger_policy_summary.json").write_text(
        json.dumps(result["summary"], indent=2, default=str)
    )
    lines = ["# HMM Trigger Policy Evaluation", ""]
    lines.extend(f"- `{k}`: {v}" for k, v in result["summary"].items())
    lines.append("")
    lines.append("## Trigger Dates")
    dates = result["triggers"]["ds"].dt.strftime("%Y-%m").tolist() if not result["triggers"].empty else []
    lines.append(", ".join(dates) if dates else "No trigger dates.")
    lines.append("")
    lines.append("## Clusters")
    if result["clusters"].empty:
        lines.append("No clusters.")
    else:
        lines.append(result["clusters"].to_markdown(index=False))
    (output_dir / "trigger_policy_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    df = _load(input_path)
    result = evaluate(
        df,
        cluster_gap_months=int(args.cluster_gap_months),
        last_n_months=int(args.last_n_months),
    )
    high_surprise = df[
        (~_trigger_mask(df))
        & pd.to_numeric(df.get("hmm_surprise_ratio", np.nan), errors="coerce").ge(float(args.high_surprise_ratio))
    ].copy()
    result["missed_high_surprise"] = high_surprise
    result["all_rows"] = df
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "trigger_policy_eval"
    _write_outputs(result, output_dir)
    print(json.dumps(result["summary"], indent=2, default=str))
    print(f"Wrote trigger policy evaluation to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="_output_hmm_regime_reselection_study/hmm_regime_monthly.csv")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cluster-gap-months", type=int, default=6)
    parser.add_argument("--last-n-months", type=int, default=60)
    parser.add_argument("--high-surprise-ratio", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
