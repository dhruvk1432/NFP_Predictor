"""Walk-forward top-k economist panel ranking grid.

Builds simple equal-weight panel forecasts from the refreshed Reuters/LSEG
contributor cache:

* rank economists by trailing MAE or RMSE over the previous ``w`` months;
* require at least ``min_coverage_pct`` coverage in that trailing window;
* average the current-month forecasts from the selected top-k economists;
* score the resulting derived features against consensus mean/median over
  trailing evaluation windows.

The script is intentionally panel-only. It does not run Kalman fusion or alter
master snapshots.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.transforms import winsorize_covid_period  # noqa: E402
PANEL_DIR = PROJECT_ROOT / "economist_panel" / "by_economist"
TARGET_PATH = PROJECT_ROOT / "data" / "NFP_target" / "y_sa_revised.parquet"
FIRST_RELEASE_PATH = PROJECT_ROOT / "data" / "NFP_target" / "y_sa_first_release.parquet"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "master_snapshots" / "sa" / "revised"
OUT_DIR = PROJECT_ROOT / "_output_economist_topk_grid"


@dataclass(frozen=True)
class Spec:
    rank_metric: str
    rank_window: int
    top_k: int

    @property
    def model_id(self) -> str:
        return f"top{self.top_k}_rank_{self.rank_metric}_w{self.rank_window}_cov70_mean"


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values:
        raise ValueError(f"Expected comma-separated integers, got {raw!r}")
    return values


def _parse_strings(raw: str) -> tuple[str, ...]:
    values = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError(f"Expected comma-separated strings, got {raw!r}")
    bad = sorted(set(values).difference({"mae", "rmse"}))
    if bad:
        raise ValueError(f"Unsupported rank metric(s): {bad}")
    return values


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    err = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else float("nan")


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    err = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    err = err[np.isfinite(err)]
    return float(np.mean(np.abs(err))) if len(err) else float("nan")


def _load_release_map() -> pd.Series:
    release = pd.read_parquet(FIRST_RELEASE_PATH, columns=["ds", "release_date"]).dropna()
    release["ds"] = pd.to_datetime(release["ds"]).dt.to_period("M").dt.to_timestamp()
    release["release_date"] = pd.to_datetime(release["release_date"])
    return release.drop_duplicates("ds").set_index("ds")["release_date"].sort_index()


def _load_actuals() -> pd.DataFrame:
    actuals = pd.read_parquet(
        TARGET_PATH,
        columns=["ds", "y_mom", "release_date", "operational_available_date"],
    )
    actuals["ds"] = pd.to_datetime(actuals["ds"]).dt.to_period("M").dt.to_timestamp()
    actual_raw = pd.to_numeric(actuals["y_mom"], errors="coerce")
    actuals["actual_raw"] = actual_raw
    actuals["actual"] = winsorize_covid_period(
        pd.Series(actual_raw.to_numpy(), index=actuals["ds"], name="actual")
    ).to_numpy()
    actuals["target_release_date"] = pd.to_datetime(actuals["release_date"], errors="coerce")
    actuals["actual_available_date"] = pd.to_datetime(
        actuals["operational_available_date"],
        errors="coerce",
    )
    return (
        actuals[["ds", "actual", "target_release_date", "actual_available_date"]]
        .drop_duplicates("ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _load_panel() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(PANEL_DIR.glob("*.parquet")):
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "ident",
                    "name",
                    "timestamp",
                    "first_release_value",
                    "first_release_date",
                ],
            )
        except Exception:
            continue
        frame = frame.dropna(subset=["ident", "timestamp", "first_release_value", "first_release_date"])
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ds"] = pd.to_datetime(frame["timestamp"]).dt.to_period("M").dt.to_timestamp()
        frame["forecast"] = pd.to_numeric(frame["first_release_value"], errors="coerce")
        frame["first_release_date"] = pd.to_datetime(frame["first_release_date"], errors="coerce")
        rows.append(frame[["ds", "ident", "name", "forecast", "first_release_date"]])
    if not rows:
        raise RuntimeError(f"No usable economist panel parquet files under {PANEL_DIR}")
    out = pd.concat(rows, ignore_index=True).dropna(subset=["forecast", "first_release_date"])
    out = out.sort_values(["ident", "ds", "first_release_date"], kind="mergesort")
    out = out.drop_duplicates(["ident", "ds"], keep="last").reset_index(drop=True)
    out["forecast"] = out["forecast"].astype(float)
    for ident, idx in out.groupby("ident", sort=False).groups.items():
        series = out.loc[idx, ["ds", "forecast"]].set_index("ds").sort_index()["forecast"]
        clipped = winsorize_covid_period(series)
        out.loc[idx, "forecast"] = out.loc[idx, "ds"].map(clipped).to_numpy()
    return out


def _latest_available(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    return (
        panel.sort_values(["ident", "ds", "first_release_date"], kind="mergesort")
        .drop_duplicates(["ident", "ds"], keep="last")
        .reset_index(drop=True)
    )


def _load_consensus_from_snapshots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SNAPSHOT_DIR.glob("**/*.parquet")):
        try:
            obs_month = pd.Timestamp(path.stem + "-01")
        except ValueError:
            continue
        try:
            snap = pd.read_parquet(
                path,
                columns=["date", "NFP_Consensus_Mean", "NFP_Consensus_Median"],
            )
        except Exception:
            continue
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        match = snap[snap["date"] == obs_month]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "ds": obs_month,
                "consensus_mean": pd.to_numeric(row.get("NFP_Consensus_Mean"), errors="coerce"),
                "consensus_median": pd.to_numeric(row.get("NFP_Consensus_Median"), errors="coerce"),
            }
        )
    if not rows:
        raise RuntimeError(f"No consensus mean/median values found under {SNAPSHOT_DIR}")
    out = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    indexed = out.set_index("ds")
    indexed[["consensus_mean", "consensus_median"]] = winsorize_covid_period(
        indexed[["consensus_mean", "consensus_median"]]
    )
    return indexed.reset_index()


def _winsorize_panel_predictions(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    if not cols:
        return out
    for model_id, idx in out.groupby("model_id", sort=False).groups.items():
        group = out.loc[idx, ["ds", *cols]].drop_duplicates("ds").set_index("ds").sort_index()
        clipped = winsorize_covid_period(group[cols])
        for col in cols:
            out.loc[idx, col] = out.loc[idx, "ds"].map(clipped[col]).to_numpy()
    return out


def _track_record(
    *,
    panel_valid: pd.DataFrame,
    actuals: pd.DataFrame,
    target_month: pd.Timestamp,
    target_release: pd.Timestamp,
    window: int,
    min_coverage_pct: float,
) -> tuple[pd.DataFrame, int]:
    window_months = pd.date_range(
        target_month - pd.DateOffset(months=window),
        target_month - pd.DateOffset(months=1),
        freq="MS",
    )
    actual_window = actuals[
        actuals["ds"].isin(window_months)
        & actuals["actual"].notna()
        & actuals["actual_available_date"].notna()
        & (actuals["actual_available_date"] < target_release)
    ][["ds", "actual"]].copy()
    n_scorable = int(actual_window["ds"].nunique())
    columns = ["ident", "name", "mae", "rmse", "bias", "n", "coverage"]
    if n_scorable == 0:
        return pd.DataFrame(columns=columns), 0

    hist = panel_valid[panel_valid["ds"].isin(actual_window["ds"])].copy()
    hist = _latest_available(hist)
    if hist.empty:
        return pd.DataFrame(columns=columns), n_scorable
    hist = hist.merge(actual_window, on="ds", how="inner").dropna(subset=["forecast", "actual"])
    if hist.empty:
        return pd.DataFrame(columns=columns), n_scorable
    hist["err"] = hist["forecast"].astype(float) - hist["actual"].astype(float)
    score = (
        hist.groupby(["ident", "name"], as_index=False)
        .agg(
            mae=("err", lambda s: float(np.mean(np.abs(s)))),
            rmse=("err", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("err", lambda s: float(np.mean(s))),
            n=("err", "size"),
        )
    )
    score["n"] = score["n"].astype(int)
    score["coverage"] = score["n"].astype(float) / float(n_scorable)
    score = score[score["coverage"] >= float(min_coverage_pct)].copy()
    return score, n_scorable


def _build_predictions(
    *,
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    release_map: pd.Series,
    specs: Iterable[Spec],
    min_coverage_pct: float,
) -> pd.DataFrame:
    actual_lookup = actuals.set_index("ds")["actual"]
    target_months = sorted(set(actuals["ds"]).union(set(release_map.index)))
    release_by_row = panel["ds"].map(release_map)
    panel_valid = panel[release_by_row.notna() & (panel["first_release_date"] < release_by_row)].copy()

    rows: list[dict[str, object]] = []
    specs_by_window: dict[int, list[Spec]] = {}
    for spec in specs:
        specs_by_window.setdefault(spec.rank_window, []).append(spec)

    for target_month in target_months:
        target_month = pd.Timestamp(target_month)
        target_release = release_map.get(target_month, pd.NaT)
        if pd.isna(target_release):
            continue
        target_release = pd.Timestamp(target_release)
        eligible = panel_valid[panel_valid["ds"] == target_month][
            ["ds", "ident", "name", "forecast", "first_release_date"]
        ].copy()
        eligible = _latest_available(eligible)

        for window, window_specs in specs_by_window.items():
            track, n_scorable = _track_record(
                panel_valid=panel_valid,
                actuals=actuals,
                target_month=target_month,
                target_release=target_release,
                window=window,
                min_coverage_pct=min_coverage_pct,
            )
            if eligible.empty:
                missing_reason = "no_current_forecasts_before_release"
                ranked = pd.DataFrame()
            elif track.empty:
                missing_reason = "no_rankable_forecasters_pass_coverage"
                ranked = pd.DataFrame()
            else:
                active = track[track["ident"].isin(eligible["ident"])].copy()
                if active.empty:
                    missing_reason = "no_active_ranked_forecasters"
                    ranked = pd.DataFrame()
                else:
                    missing_reason = ""
                    ranked = active.merge(
                        eligible[["ident", "forecast", "first_release_date"]],
                        on="ident",
                        how="left",
                        validate="one_to_one",
                    )

            for spec in window_specs:
                if not ranked.empty:
                    selected = ranked.sort_values(
                        [spec.rank_metric, "coverage", "n", "ident"],
                        ascending=[True, False, False, True],
                    ).head(spec.top_k)
                else:
                    selected = pd.DataFrame()
                pred = float(selected["forecast"].mean()) if not selected.empty else float("nan")
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "rank_metric": spec.rank_metric,
                        "rank_window": spec.rank_window,
                        "top_k": spec.top_k,
                        "ds": target_month,
                        "target_release_date": target_release,
                        "actual": actual_lookup.get(target_month, np.nan),
                        "predicted": pred,
                        "selected_count": int(len(selected)),
                        "eligible_count": int(len(eligible)),
                        "rankable_count": int(len(ranked)),
                        "n_scorable_months": int(n_scorable),
                        "selected_mean_coverage": (
                            float(selected["coverage"].mean()) if not selected.empty else np.nan
                        ),
                        "selected_mean_mae": float(selected["mae"].mean()) if not selected.empty else np.nan,
                        "selected_mean_rmse": float(selected["rmse"].mean()) if not selected.empty else np.nan,
                        "selected_idents": "|".join(selected["ident"].astype(str)) if not selected.empty else "",
                        "selected_names": "|".join(selected["name"].astype(str)) if not selected.empty else "",
                        "missing_reason": missing_reason if selected.empty else "",
                    }
                )
    return pd.DataFrame(rows).sort_values(["model_id", "ds"]).reset_index(drop=True)


def _score_model_windows(
    frame: pd.DataFrame,
    eval_windows: Iterable[int],
    eval_end_ds: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual_months = frame[frame["actual"].notna()].sort_values("ds")["ds"].drop_duplicates()
    if eval_end_ds is not None:
        actual_months = actual_months[actual_months <= eval_end_ds]
    for model_id, group in frame.groupby("model_id", sort=False):
        meta = group.iloc[0][["rank_metric", "rank_window", "top_k"]].to_dict()
        for n in eval_windows:
            months = set(actual_months.tail(int(n)))
            sub = group[group["ds"].isin(months)].copy()
            pred_mask = sub["actual"].notna() & sub["predicted"].notna()
            panel_mae = _mae(sub.loc[pred_mask, "actual"], sub.loc[pred_mask, "predicted"])
            panel_rmse = _rmse(sub.loc[pred_mask, "actual"], sub.loc[pred_mask, "predicted"])
            row = {
                "model_id": model_id,
                **meta,
                "eval_window_months": int(n),
                "scored_months": int(pred_mask.sum()),
                "missing_months": int((sub["actual"].notna() & sub["predicted"].isna()).sum()),
                "MAE": panel_mae,
                "RMSE": panel_rmse,
                "coverage": float(pred_mask.sum() / max(1, sub["actual"].notna().sum())),
            }
            for baseline in ("consensus_mean", "consensus_median"):
                common = pred_mask & sub[baseline].notna()
                b_mae = _mae(sub.loc[common, "actual"], sub.loc[common, baseline])
                b_rmse = _rmse(sub.loc[common, "actual"], sub.loc[common, baseline])
                row[f"{baseline}_common_MAE"] = b_mae
                row[f"{baseline}_common_RMSE"] = b_rmse
                row[f"delta_MAE_vs_{baseline}_common"] = b_mae - panel_mae
                row[f"delta_RMSE_vs_{baseline}_common"] = b_rmse - panel_rmse
            rows.append(row)
    return pd.DataFrame(rows)


def _score_baselines(
    frame: pd.DataFrame,
    eval_windows: Iterable[int],
    eval_end_ds: pd.Timestamp | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual_months = frame[frame["actual"].notna()].sort_values("ds")["ds"].drop_duplicates()
    if eval_end_ds is not None:
        actual_months = actual_months[actual_months <= eval_end_ds]
    for baseline in ("consensus_mean", "consensus_median"):
        for n in eval_windows:
            months = set(actual_months.tail(int(n)))
            sub = frame[frame["ds"].isin(months)].copy()
            mask = sub["actual"].notna() & sub[baseline].notna()
            rows.append(
                {
                    "model_id": baseline,
                    "rank_metric": "baseline",
                    "rank_window": np.nan,
                    "top_k": np.nan,
                    "eval_window_months": int(n),
                    "scored_months": int(mask.sum()),
                    "missing_months": int((sub["actual"].notna() & sub[baseline].isna()).sum()),
                    "MAE": _mae(sub.loc[mask, "actual"], sub.loc[mask, baseline]),
                    "RMSE": _rmse(sub.loc[mask, "actual"], sub.loc[mask, baseline]),
                    "coverage": float(mask.sum() / max(1, sub["actual"].notna().sum())),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-windows", default="6,8,12,18,24,36,60")
    parser.add_argument("--top-ks", default="1,2,3,4,5,6,8,10,15,20")
    parser.add_argument("--rank-metrics", default="mae,rmse")
    parser.add_argument("--eval-windows", default="36,60,120,180")
    parser.add_argument(
        "--eval-end-ds",
        default=None,
        help="Optional YYYY-MM evaluation anchor. Scores trailing windows ending at this month.",
    )
    parser.add_argument("--min-coverage-pct", type=float, default=0.70)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--no-covid-winsor",
        action="store_true",
        help="Disable the repo-standard COVID-period clipping on forecast features.",
    )
    args = parser.parse_args()

    rank_windows = _parse_ints(args.rank_windows)
    top_ks = _parse_ints(args.top_ks)
    rank_metrics = _parse_strings(args.rank_metrics)
    eval_windows = _parse_ints(args.eval_windows)
    eval_end_ds = (
        pd.Timestamp(str(args.eval_end_ds) + "-01").to_period("M").to_timestamp()
        if args.eval_end_ds
        else None
    )
    specs = [
        Spec(rank_metric=metric, rank_window=window, top_k=top_k)
        for metric in rank_metrics
        for window in rank_windows
        for top_k in top_ks
    ]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    release_map = _load_release_map()
    actuals = _load_actuals()
    panel = _load_panel()
    consensus = _load_consensus_from_snapshots()

    predictions = _build_predictions(
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        specs=specs,
        min_coverage_pct=float(args.min_coverage_pct),
    )
    merged = predictions.merge(consensus, on="ds", how="left")
    if not args.no_covid_winsor:
        merged = _winsorize_panel_predictions(merged, ["predicted"])

    grid = _score_model_windows(merged, eval_windows, eval_end_ds)
    baseline_frame = actuals[["ds", "actual"]].merge(consensus, on="ds", how="left")
    baselines = _score_baselines(baseline_frame, eval_windows, eval_end_ds)
    all_metrics = pd.concat([baselines, grid], ignore_index=True)

    missing_scope = merged["actual"].notna() & merged["predicted"].isna()
    if eval_end_ds is not None:
        missing_scope &= merged["ds"] <= eval_end_ds
    missing = (
        merged[missing_scope]
        .groupby(["model_id", "missing_reason"], dropna=False)
        .agg(
            missing_months=("ds", "nunique"),
            first_missing=("ds", "min"),
            last_missing=("ds", "max"),
        )
        .reset_index()
        .sort_values(["model_id", "missing_reason"])
    )
    recent_missing = (
        merged[missing_scope]
        [["model_id", "rank_metric", "rank_window", "top_k", "ds", "missing_reason"]]
        .sort_values(["ds", "model_id"])
    )

    merged.to_csv(out_dir / "monthly_predictions.csv", index=False)
    all_metrics.to_csv(out_dir / "metrics_grid.csv", index=False)
    missing.to_csv(out_dir / "missing_summary.csv", index=False)
    recent_missing.to_csv(out_dir / "missing_months.csv", index=False)

    manifest = {
        "panel_rows": int(len(panel)),
        "panel_economists": int(panel["ident"].nunique()),
        "panel_min_ds": str(panel["ds"].min().date()),
        "panel_max_ds": str(panel["ds"].max().date()),
        "eval_end_ds": str(eval_end_ds.date()) if eval_end_ds is not None else None,
        "actual_last_available_ds": str(actuals[actuals["actual"].notna()]["ds"].max().date()),
        "actual_last_scored_ds": str(
            (
                actuals[actuals["actual"].notna() & (actuals["ds"] <= eval_end_ds)]["ds"].max()
                if eval_end_ds is not None
                else actuals[actuals["actual"].notna()]["ds"].max()
            ).date()
        ),
        "rank_windows": list(rank_windows),
        "top_ks": list(top_ks),
        "rank_metrics": list(rank_metrics),
        "eval_windows": list(eval_windows),
        "min_coverage_pct": float(args.min_coverage_pct),
        "covid_winsorized": not args.no_covid_winsor,
        "outputs": [
            "monthly_predictions.csv",
            "metrics_grid.csv",
            "missing_summary.csv",
            "missing_months.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    best = (
        all_metrics[all_metrics["rank_metric"].isin(["mae", "rmse"])]
        .sort_values(["eval_window_months", "RMSE", "MAE"])
        .groupby("eval_window_months")
        .head(5)
    )
    baseline_short = all_metrics[all_metrics["rank_metric"].eq("baseline")]
    print("Wrote", out_dir)
    print("\nBaselines:")
    print(baseline_short.to_string(index=False))
    print("\nTop 5 panel configs by RMSE per evaluation window:")
    print(
        best[
            [
                "eval_window_months",
                "model_id",
                "rank_metric",
                "rank_window",
                "top_k",
                "scored_months",
                "missing_months",
                "MAE",
                "RMSE",
                "coverage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
