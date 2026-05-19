"""Summarize HMM regime/backtest diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STRESS_MONTHS = {
    "2021-06",
    "2022-07",
    "2022-10",
    "2022-12",
    "2023-01",
    "2026-03",
}


def _metric_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df.columns:
        return pd.DataFrame()
    valid = df.dropna(subset=["actual_minus_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(by, dropna=False)
        .agg(
            n=("actual_minus_pred", "size"),
            mae=("abs_error", "mean"),
            rmse=("actual_minus_pred", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("actual_minus_pred", "mean"),
            transition_risk=("hmm_transition_risk", "mean"),
            surprise=("hmm_surprise", "mean"),
        )
        .sort_values(["mae", "n"], ascending=[False, False])
        .round(3)
    )


def _feature_jaccards(hmm_rows: pd.DataFrame) -> pd.DataFrame:
    if "hmm_features" not in hmm_rows.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    prev: set[str] | None = None
    for _, row in hmm_rows.sort_values("ds").iterrows():
        raw = row.get("hmm_features")
        try:
            features = set(json.loads(raw)) if isinstance(raw, str) else set()
        except Exception:
            features = set()
        if prev is not None and (features or prev):
            union = features | prev
            rows.append({
                "ds": row["ds"],
                "hmm_regime_label": row.get("hmm_regime_label"),
                "final_reselect": row.get("final_reselect"),
                "feature_jaccard": len(features & prev) / len(union) if union else np.nan,
                "n_features": len(features),
            })
        prev = features
    return pd.DataFrame(rows)


def _bias_correction(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, part in df.sort_values("ds").groupby("model_id"):
        residual_by_regime: dict[str, list[float]] = {}
        corrected_errors: list[float] = []
        raw_errors: list[float] = []
        for _, row in part.iterrows():
            actual = row.get("actual")
            pred = row.get("predicted")
            regime = str(row.get("hmm_regime_label") or "unknown")
            if pd.isna(actual) or pd.isna(pred):
                continue
            hist = residual_by_regime.get(regime, [])
            correction = 0.0
            if len(hist) >= 12:
                bias = float(np.mean(hist))
                shrink = len(hist) / (len(hist) + 12.0)
                correction = shrink * bias
            raw_error = float(actual - pred)
            corrected_error = float(actual - (pred + correction))
            raw_errors.append(raw_error)
            corrected_errors.append(corrected_error)
            residual_by_regime.setdefault(regime, []).append(raw_error)
        if raw_errors:
            raw = np.asarray(raw_errors, dtype=float)
            corr = np.asarray(corrected_errors, dtype=float)
            rows.append({
                "model_id": model_id,
                "n": int(len(raw)),
                "raw_mae": float(np.mean(np.abs(raw))),
                "bias_corrected_mae": float(np.mean(np.abs(corr))),
                "delta_mae": float(np.mean(np.abs(corr)) - np.mean(np.abs(raw))),
            })
    return pd.DataFrame(rows).round(3)


def _write_table(lines: list[str], title: str, table: pd.DataFrame) -> None:
    lines.append(f"## {title}")
    if table.empty:
        lines.append("No data.")
    else:
        try:
            lines.append(table.to_markdown())
        except Exception:
            lines.append(table.to_string())
    lines.append("")


def _save_tables(output_dir: Path, tables: Iterable[tuple[str, pd.DataFrame]]) -> None:
    for name, table in tables:
        if not table.empty:
            table.to_csv(output_dir / f"{name}.csv")


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, parse_dates=["ds"])
    if "abs_error" not in df.columns:
        df["actual_minus_pred"] = pd.to_numeric(df["actual"], errors="coerce") - pd.to_numeric(df["predicted"], errors="coerce")
        df["abs_error"] = df["actual_minus_pred"].abs()
    df["month"] = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m")

    tables = [
        ("by_model", _metric_table(df, "model_id")),
        ("by_regime", _metric_table(df, "hmm_regime_label")),
        ("by_trigger_class", _metric_table(df, "hmm_trigger_class")),
        ("by_reselect", _metric_table(df, "final_reselect")),
        ("by_event_window", _metric_table(df, "event_window")),
    ]
    stress = df[df["month"].isin(STRESS_MONTHS)].copy()
    stress_table = (
        stress.dropna(subset=["actual_minus_pred"])
        .sort_values("abs_error", ascending=False)
        [["model_id", "month", "actual", "predicted", "actual_minus_pred", "abs_error", "hmm_regime_label", "hmm_trigger_class"]]
    )
    cooldown = pd.DataFrame()
    if "hmm_trigger_class" in df.columns:
        cooldown = df[df["hmm_trigger_class"].eq("cooldown_suppressed")].copy()
        cooldown = cooldown.sort_values("abs_error", ascending=False).head(20)
    jaccard = _feature_jaccards(df.drop_duplicates("ds"))
    bias = _bias_correction(df)

    _save_tables(
        output_dir,
        tables + [
            ("stress_months", stress_table),
            ("cooldown_suppressed_large_misses", cooldown),
            ("feature_jaccard", jaccard),
            ("regime_bias_correction_probe", bias),
        ],
    )

    lines = ["# HMM Regime Reselection Study Summary", ""]
    for title, table in (
        ("By Model", tables[0][1]),
        ("By HMM Regime", tables[1][1]),
        ("By Trigger Class", tables[2][1]),
        ("By Reselection Flag", tables[3][1]),
        ("By Economic Event Window", tables[4][1]),
        ("Stress Months", stress_table.head(24)),
        ("Cooldown-Suppressed Large Misses", cooldown[[
            c for c in ["model_id", "month", "abs_error", "hmm_regime_label", "hmm_trigger_reason"]
            if c in cooldown.columns
        ]].head(20) if not cooldown.empty else cooldown),
        ("Feature Jaccard", jaccard.describe().round(3) if not jaccard.empty else jaccard),
        ("Regime Bias Correction Probe", bias),
    ):
        _write_table(lines, title, table)
    out_path = output_dir / "summary.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote summary to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="_output_hmm_regime_reselection_study/hmm_joined_backtests.csv")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
