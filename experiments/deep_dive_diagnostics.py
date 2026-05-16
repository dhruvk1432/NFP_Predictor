#!/usr/bin/env python3
"""Artifact-only diagnostics for the NFP predictor deep dive.

This script intentionally does not import or mutate the production pipeline.
It reads existing backtest outputs, archived outputs, and the forked
acceleration experiments, then writes small CSV/Markdown summaries under
_output/deep_dive_diagnostics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_output" / "deep_dive_diagnostics"
ARCHIVE = ROOT / "_output" / "Archive"
# The true May 12 morning setup was not fully recoverable because its exact
# feature set/settings were not uploaded. Use the recoverable pushed archive by
# default, and allow quick re-anchoring without editing this file.
BASELINE_ARCHIVE_NAME = os.environ.get("BASELINE_ARCHIVE_NAME", "2026-05-12_165541")
BASELINE_ARCHIVE = ARCHIVE / BASELINE_ARCHIVE_NAME
FORK = Path(
    "/Users/dhruvkohli/Desktop/"
    "NFP_Predictor-d9c711d13b1fbd2e9dc1f7cdf006e2ce4e3ce037"
)


def read_forecast(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ds", "actual", name])
    df = pd.read_csv(path)
    if not {"ds", "actual", "predicted"}.issubset(df.columns):
        return pd.DataFrame(columns=["ds", "actual", name])
    return (
        df.loc[:, ["ds", "actual", "predicted"]]
        .assign(ds=lambda x: pd.to_datetime(x["ds"]))
        .rename(columns={"predicted": name})
    )


def merge_forecasts(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    wide = frames[0].copy()
    for frame in frames[1:]:
        wide = wide.merge(frame, on=["ds", "actual"], how="outer")
    return wide.sort_values("ds").reset_index(drop=True)


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    d = pd.DataFrame({"actual": y, "pred": pred}).dropna()
    if d.empty:
        return {"N": 0}
    err = d["actual"] - d["pred"]
    out = {
        "N": float(len(d)),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "Bias": float(err.mean()),
        "MedAE": float(err.abs().median()),
        "STD_Ratio": float(d["pred"].std(ddof=0) / d["actual"].std(ddof=0))
        if d["actual"].std(ddof=0)
        else np.nan,
    }
    if len(d) > 1:
        actual_delta = d["actual"].diff().iloc[1:]
        pred_delta = d["pred"].diff().iloc[1:]
        operational_delta = (d["pred"] - d["actual"].shift()).iloc[1:]
        diff_sign_acc = float(
            (np.sign(actual_delta) == np.sign(pred_delta)).mean()
        )
        operational_acc = float(
            (np.sign(actual_delta) == np.sign(operational_delta)).mean()
        )
        # Production scorecards define acceleration operationally: did the
        # point forecast land above/below the previously known actual in the
        # same direction as the next actual change? Keep predicted-diff sign
        # separately because it is still useful for smoothness diagnostics.
        out["Forecast_Diff_Accuracy"] = diff_sign_acc
        out["Predicted_Diff_Sign_Accuracy"] = diff_sign_acc
        out["Acceleration_Accuracy"] = operational_acc
        out["Operational_Acceleration_Accuracy"] = operational_acc
        out["Diff_STD_Ratio"] = float(
            pred_delta.std(ddof=0) / actual_delta.std(ddof=0)
        ) if actual_delta.std(ddof=0) else np.nan
    else:
        out["Forecast_Diff_Accuracy"] = np.nan
        out["Acceleration_Accuracy"] = np.nan
        out["Operational_Acceleration_Accuracy"] = np.nan
        out["Diff_STD_Ratio"] = np.nan
    return out


def simplex_weights(n: int, step: float) -> Iterable[tuple[float, ...]]:
    units = int(round(1.0 / step))

    def rec(k: int, remaining: int) -> Iterable[tuple[int, ...]]:
        if k == 1:
            yield (remaining,)
            return
        for value in range(remaining + 1):
            for rest in rec(k - 1, remaining - value):
                yield (value,) + rest

    for weights in rec(n, units):
        yield tuple(v / units for v in weights)


def static_blend_search(
    wide: pd.DataFrame, cols: list[str], step: float = 0.05
) -> dict[str, object]:
    d = wide.loc[:, ["actual", *cols]].dropna()
    if len(d) < 10:
        return {"N": len(d), "cols": "|".join(cols), "error": "too few rows"}
    best: dict[str, object] | None = None
    preds = d.loc[:, cols].to_numpy(float)
    actual = d["actual"].to_numpy(float)
    for weights in simplex_weights(len(cols), step):
        pred = preds @ np.array(weights)
        m = metrics(pd.Series(actual), pd.Series(pred))
        row: dict[str, object] = {"cols": "|".join(cols), "weights": json.dumps(dict(zip(cols, weights))), **m}
        if best is None or float(row["MAE"]) < float(best["MAE"]):
            best = row
    return best or {"N": 0, "cols": "|".join(cols)}


def rolling_blend_search(
    wide: pd.DataFrame, cols: list[str], window: int, step: float = 0.10
) -> dict[str, object]:
    d = wide.loc[:, ["ds", "actual", *cols]].dropna().sort_values("ds").reset_index(drop=True)
    if len(d) <= window + 5:
        return {"N": len(d), "cols": "|".join(cols), "window": window, "error": "too few rows"}
    rows = []
    weights_seen = []
    weight_grid = list(simplex_weights(len(cols), step))
    for i in range(window, len(d)):
        train = d.iloc[i - window : i]
        test = d.iloc[i]
        train_actual = train["actual"].to_numpy(float)
        train_preds = train.loc[:, cols].to_numpy(float)
        best_w = None
        best_loss = float("inf")
        for weights in weight_grid:
            pred = train_preds @ np.array(weights)
            loss = float(np.mean(np.abs(train_actual - pred)))
            if loss < best_loss:
                best_loss = loss
                best_w = weights
        assert best_w is not None
        rows.append(
            {
                "ds": test["ds"],
                "actual": float(test["actual"]),
                "predicted": float(test.loc[cols].to_numpy(float) @ np.array(best_w)),
            }
        )
        weights_seen.append(best_w)
    pred_df = pd.DataFrame(rows)
    m = metrics(pred_df["actual"], pred_df["predicted"])
    avg_weights = np.array(weights_seen).mean(axis=0)
    return {
        "cols": "|".join(cols),
        "window": window,
        "weights_avg": json.dumps(dict(zip(cols, avg_weights.round(4)))),
        **m,
    }


def summarize_archives() -> pd.DataFrame:
    rows = []
    candidates = [(ROOT / "_output", "CURRENT")]
    candidates.extend((p, p.name) for p in sorted(ARCHIVE.iterdir()) if p.is_dir())
    for base, name in candidates:
        comp = base / "consensus_anchor" / "comparison_metrics.csv"
        if comp.exists():
            df = pd.read_csv(comp)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "archive": name,
                        "forecast": row.get("Forecast"),
                        "N": row.get("N"),
                        "MAE": row.get("MAE"),
                        "RMSE": row.get("RMSE"),
                        "Acceleration_Accuracy": row.get("Acceleration_Accuracy"),
                        "STD_Ratio": row.get("STD_Ratio"),
                        "Diff_STD_Ratio": row.get("Diff_STD_Ratio"),
                        "Tail_MAE": row.get("Tail_MAE"),
                    }
                )
        for model in ["SA_prediction", "NSA_plus_adjustment", "NSA_prediction"]:
            stats = base / model / "summary_statistics.csv"
            if stats.exists():
                df = pd.read_csv(stats)
                if not df.empty:
                    row = df.iloc[0]
                    rows.append(
                        {
                            "archive": name,
                            "forecast": model,
                            "N": row.get("N"),
                            "MAE": row.get("MAE"),
                            "RMSE": row.get("RMSE"),
                            "Acceleration_Accuracy": row.get("Diff_Sign_Accuracy"),
                            "STD_Ratio": row.get("STD_Ratio"),
                            "Diff_STD_Ratio": row.get("Diff_STD_Ratio"),
                            "Tail_MAE": row.get("Tail_MAE"),
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["forecast", "MAE"], na_position="last")
    return out


def feature_source(feature: str) -> str:
    if "__" in feature:
        return feature.split("__", 1)[0]
    if feature.startswith("nfp_"):
        return "DerivedControls"
    if feature.startswith(("rev_master_", "is_", "month_", "quarter_", "year", "weeks_since_")):
        return "DerivedControls"
    if feature.startswith("sanagap_"):
        return "SA_NSA_Gap"
    if "consensus" in feature.lower():
        return "Consensus"
    if feature.startswith(("Treasury_", "FedFunds_", "SOFR_", "WTI_Crude_", "NatGas_", "Gold_", "Copper_", "DollarIndex_", "EuroFX_", "YenFX_")):
        return "Futures"
    if feature.startswith(("NFP_Forecast_", "Economist_")):
        return "EconomistPanel"
    if feature.startswith(("NOAA_", "storm_", "hurricane_")):
        return "NOAA"
    if feature.startswith(("Prosper_", "Consumer_Mood", "Consumer_Spending")):
        return "Prosper"
    if feature.startswith(("CC", "IC", "AW", "CES", "total", "private")):
        return "FRED_Employment"
    return feature.split("_", 1)[0] if "_" in feature else "Other"


def summarize_dynamic_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    source_rows = []
    for path in sorted((ROOT / "_output" / "dynamic_selection").glob("*/*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        features = payload.get("selected_features") or payload.get("features") or []
        rows.append(
            {
                "target": path.parent.name,
                "file": str(path.relative_to(ROOT)),
                "mtime": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "n_features": len(features),
            }
        )
        counts = pd.Series([feature_source(str(f)) for f in features]).value_counts()
        for source, count in counts.items():
            source_rows.append(
                {
                    "target": path.parent.name,
                    "file": str(path.relative_to(ROOT)),
                    "source": source,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(source_rows)


def summarize_fork_acceleration(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    results_dir = FORK / "acceleration_maxxing" / "results"
    rows = []
    if not results_dir.exists():
        return pd.DataFrame()
    for path in sorted(results_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"ds", "method", "direction_correct", "actual_accel"}
        if not required.issubset(df.columns):
            continue
        df = df.assign(ds=pd.to_datetime(df["ds"], errors="coerce")).dropna(subset=["ds"])
        window = df[(df["ds"] >= start) & (df["ds"] <= end)]
        for method, group in window.groupby("method"):
            if len(group) < 24:
                continue
            rows.append(
                {
                    "file": path.name,
                    "method": method,
                    "N": len(group),
                    "Direction_Accuracy": float(group["direction_correct"].mean()),
                    "MAE_accel": float(group.get("abs_error", pd.Series(dtype=float)).mean())
                    if "abs_error" in group
                    else np.nan,
                    "start": group["ds"].min().date().isoformat(),
                    "end": group["ds"].max().date().isoformat(),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Direction_Accuracy", "N"], ascending=[False, False])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    current = merge_forecasts(
        [
            read_forecast(ROOT / "_output" / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv", "current_kalman"),
            read_forecast(ROOT / "_output" / "consensus_anchor" / "baseline_consensus" / "backtest_results.csv", "consensus"),
            read_forecast(ROOT / "_output" / "NSA_plus_adjustment" / "backtest_results.csv", "current_nsa_adj"),
            read_forecast(ROOT / "_output" / "SA_prediction" / "backtest_results.csv", "current_sa_lgbm"),
        ]
    )
    best = merge_forecasts(
        [
            read_forecast(BASELINE_ARCHIVE / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv", "baseline_kalman"),
            read_forecast(BASELINE_ARCHIVE / "NSA_plus_adjustment" / "backtest_results.csv", "baseline_nsa_adj"),
            read_forecast(BASELINE_ARCHIVE / "SA_prediction" / "backtest_results.csv", "baseline_sa_lgbm"),
        ]
    )
    wide = merge_forecasts([current, best])
    wide.to_csv(OUT / "forecast_panel_current_vs_baseline.csv", index=False)

    delta = wide.dropna(subset=["current_kalman", "baseline_kalman"]).copy()
    delta["current_abs_error"] = (delta["actual"] - delta["current_kalman"]).abs()
    delta["baseline_abs_error"] = (delta["actual"] - delta["baseline_kalman"]).abs()
    delta["current_minus_baseline_abs_error"] = (
        delta["current_abs_error"] - delta["baseline_abs_error"]
    )
    delta.sort_values("current_minus_baseline_abs_error", ascending=False).to_csv(
        OUT / "current_vs_baseline_month_deltas.csv", index=False
    )

    archive_scores = summarize_archives()
    archive_scores.to_csv(OUT / "archive_scoreboard.csv", index=False)

    static_specs = [
        ["current_kalman", "consensus"],
        ["current_kalman", "current_sa_lgbm"],
        ["consensus", "current_nsa_adj"],
        ["consensus", "current_nsa_adj", "current_sa_lgbm"],
        ["current_kalman", "consensus", "current_sa_lgbm"],
        ["current_kalman", "consensus", "current_nsa_adj", "current_sa_lgbm"],
    ]
    static_rows = [static_blend_search(wide, cols) for cols in static_specs]
    pd.DataFrame(static_rows).sort_values("MAE").to_csv(OUT / "static_blend_grid.csv", index=False)

    rolling_rows = []
    for cols in static_specs:
        for window in [12, 24, 36]:
            rolling_rows.append(rolling_blend_search(wide, cols, window))
    pd.DataFrame(rolling_rows).sort_values("MAE").to_csv(
        OUT / "rolling_blend_grid.csv", index=False
    )

    single_rows = []
    for col in [
        "current_kalman",
        "consensus",
        "current_nsa_adj",
        "current_sa_lgbm",
        "baseline_kalman",
        "baseline_nsa_adj",
        "baseline_sa_lgbm",
    ]:
        if col in wide.columns:
            single_rows.append({"forecast": col, **metrics(wide["actual"], wide[col])})
    pd.DataFrame(single_rows).sort_values("MAE").to_csv(OUT / "single_forecast_metrics.csv", index=False)

    dyn, dyn_sources = summarize_dynamic_selection()
    dyn.to_csv(OUT / "dynamic_selection_summary.csv", index=False)
    dyn_sources.to_csv(OUT / "dynamic_selection_source_counts.csv", index=False)

    if not current.empty:
        fork_accel = summarize_fork_acceleration(current["ds"].min(), current["ds"].max())
        fork_accel.to_csv(OUT / "fork_acceleration_alignment.csv", index=False)
    else:
        fork_accel = pd.DataFrame()

    best_archives = (
        archive_scores[
            (archive_scores["forecast"] == "Kalman_Fusion_NSA")
            & archive_scores["MAE"].notna()
        ]
        .sort_values("MAE")
        .head(10)
    )
    top_static = pd.read_csv(OUT / "static_blend_grid.csv").head(6)
    top_rolling = pd.read_csv(OUT / "rolling_blend_grid.csv").head(6)
    top_fork = fork_accel.head(10)
    top_delta = pd.read_csv(OUT / "current_vs_baseline_month_deltas.csv").head(10)

    summary = [
        "# Deep Dive Diagnostics",
        "",
        "All numbers below are artifact-only diagnostics; no pipeline code was imported or changed.",
        f"Recoverable baseline archive: `{BASELINE_ARCHIVE_NAME}`.",
        "",
        "## Best Kalman Archives",
        best_archives.to_markdown(index=False),
        "",
        "## Current/Best Single Forecasts",
        pd.read_csv(OUT / "single_forecast_metrics.csv").to_markdown(index=False),
        "",
        "## Static Blend Search",
        top_static.to_markdown(index=False),
        "",
        "## Rolling Blend Search",
        top_rolling.to_markdown(index=False),
        "",
        "## Months Where Current Kalman Lost Most vs Recoverable Baseline",
        top_delta[
            [
                "ds",
                "actual",
                "current_kalman",
                "baseline_kalman",
                "current_abs_error",
                "baseline_abs_error",
                "current_minus_baseline_abs_error",
            ]
        ].to_markdown(index=False),
        "",
        "## Fork Acceleration Alignment",
        top_fork.to_markdown(index=False) if not top_fork.empty else "No aligned fork acceleration rows found.",
        "",
        "## Dynamic Selection Cohorts",
        dyn.sort_values(["target", "mtime", "file"]).to_markdown(index=False)
        if not dyn.empty
        else "No dynamic selection JSONs found.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary))
    print(f"Wrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
