"""
Compare SA revised model outputs across main + sandbox + archives.

Produces clean tables and plots under:
  `_output/sandbox/sa_revised_comparison`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from settings import OUTPUT_DIR, TEMP_DIR, setup_logger
from Train.sandbox.output_utils import load_summary_for_comparison

logger = setup_logger(__file__, TEMP_DIR)
OUT_DIR = OUTPUT_DIR / "sandbox" / "sa_revised_comparison"


def _read_backtest(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    return df


def _safe_float(val) -> float:
    try:
        out = float(val)
        return out
    except Exception:
        return float("nan")


def _compute_dir_accel(backtest: pd.DataFrame) -> Tuple[float, float]:
    if backtest.empty or not {"actual", "predicted"}.issubset(backtest.columns):
        return float("nan"), float("nan")
    valid = backtest[backtest["actual"].notna() & backtest["predicted"].notna()].copy()
    if valid.empty:
        return float("nan"), float("nan")

    actual = valid["actual"].values.astype(float)
    pred = valid["predicted"].values.astype(float)
    dir_acc = float(np.mean(np.sign(actual) == np.sign(pred)))
    if len(valid) >= 2:
        accel_acc = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))))
    else:
        accel_acc = float("nan")
    return dir_acc, accel_acc


def _extract_metrics(folder: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    summary_json = folder / "summary_metrics.json"
    summary_csv = folder / "summary_statistics.csv"
    payload = load_summary_for_comparison(summary_json) if summary_json.exists() else None
    if payload is not None:
        overall = payload.get("overall", {})
        if isinstance(overall, dict):
            for k, v in overall.items():
                metrics[k] = _safe_float(v)
    elif summary_csv.exists():
        try:
            row = pd.read_csv(summary_csv).iloc[0].to_dict()
            for k, v in row.items():
                metrics[k] = _safe_float(v)
        except Exception:
            pass

    return metrics


def _normalize_record(raw: Dict[str, float], backtest: pd.DataFrame) -> Dict[str, float]:
    metrics = dict(raw)
    dir_acc, accel_acc = _compute_dir_accel(backtest)

    # Normalize key aliases across old/new formats.
    alias_map = {
        "RMSE": ["RMSE"],
        "MAE": ["MAE"],
        "MSE": ["MSE"],
        "STD_Ratio": ["STD_Ratio", "std_ratio"],
        "Diff_STD_Ratio": ["Diff_STD_Ratio", "diff_std_ratio"],
        "Corr_Diff": ["Corr_Diff", "corr_diff"],
        "Diff_Sign_Accuracy": ["Diff_Sign_Accuracy", "diff_sign_accuracy"],
        "Tail_MAE": ["Tail_MAE", "tail_mae"],
        "Extreme_Hit_Rate": ["Extreme_Hit_Rate", "extreme_hit_rate"],
        "Directional_Accuracy": ["Directional_Accuracy", "dir_accuracy"],
        "Acceleration_Accuracy": ["Acceleration_Accuracy", "accel_accuracy"],
        "N_Backtest": ["N_Backtest"],
    }

    out: Dict[str, float] = {}
    for canonical, aliases in alias_map.items():
        val = float("nan")
        for key in aliases:
            if key in metrics and np.isfinite(_safe_float(metrics[key])):
                val = _safe_float(metrics[key])
                break
        out[canonical] = val

    if not np.isfinite(out["Directional_Accuracy"]):
        out["Directional_Accuracy"] = dir_acc
    if not np.isfinite(out["Acceleration_Accuracy"]):
        out["Acceleration_Accuracy"] = accel_acc
    if not np.isfinite(out["N_Backtest"]):
        valid = backtest[backtest["error"].notna()].copy() if "error" in backtest.columns else backtest
        out["N_Backtest"] = float(len(valid))
    return out


def _minmax_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if float(mx - mn) < 1e-12:
        out = pd.Series(1.0, index=series.index, dtype=float)
    else:
        out = (s - mn) / (mx - mn)
    if invert:
        out = 1.0 - out
    return out


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rmse_score"] = _minmax_scale(out["RMSE"], invert=True).fillna(0.0)
    out["mae_score"] = _minmax_scale(out["MAE"], invert=True).fillna(0.0)

    std_close = 1.0 - (np.abs(out["STD_Ratio"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0)
    diff_std_close = 1.0 - (np.abs(out["Diff_STD_Ratio"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0)
    corr_scaled = ((out["Corr_Diff"].clip(lower=-1.0, upper=1.0) + 1.0) / 2.0)
    diff_sign = out["Diff_Sign_Accuracy"].clip(lower=0.0, upper=1.0)
    accel = out["Acceleration_Accuracy"].clip(lower=0.0, upper=1.0)

    out["variance_score"] = (
        100.0
        * (
            0.30 * std_close.fillna(0.0)
            + 0.30 * diff_std_close.fillna(0.0)
            + 0.20 * corr_scaled.fillna(0.0)
            + 0.10 * diff_sign.fillna(0.0)
            + 0.10 * accel.fillna(0.0)
        )
    )
    out["error_score"] = 100.0 * (0.60 * out["rmse_score"] + 0.40 * out["mae_score"])
    out["overall_score"] = 0.35 * out["error_score"] + 0.65 * out["variance_score"]
    return out


def _plot_error_metrics(df: pd.DataFrame, save_path: Path) -> None:
    ranked = df.sort_values("RMSE", ascending=True).copy()
    labels = ranked["model_label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.45 * len(labels))))
    axes[0].barh(labels, ranked["RMSE"], color="#4C78A8")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].grid(alpha=0.25, axis="x")

    axes[1].barh(labels, ranked["MAE"], color="#F58518")
    axes[1].set_title("MAE (lower is better)")
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_variance_metrics(df: pd.DataFrame, save_path: Path) -> None:
    ranked = df.sort_values("overall_score", ascending=False).copy()
    labels = ranked["model_label"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    panels = [
        ("STD_Ratio", 1.0, "STD Ratio"),
        ("Diff_STD_Ratio", 1.0, "Diff STD Ratio"),
        ("Corr_Diff", 0.0, "Corr Diff"),
        ("Diff_Sign_Accuracy", 0.5, "Diff Sign Acc"),
        ("Directional_Accuracy", 0.5, "Directional Acc"),
        ("Acceleration_Accuracy", 0.5, "Acceleration Acc"),
    ]

    for ax, (col, ref, title) in zip(axes.ravel(), panels):
        ax.bar(x, ranked[col], color="#54A24B")
        ax.axhline(ref, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(title)
        ax.grid(alpha=0.20, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scores(df: pd.DataFrame, save_path: Path) -> None:
    ranked = df.sort_values("overall_score", ascending=False).copy()
    labels = ranked["model_label"].tolist()
    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(16, max(6, 0.45 * len(labels))))
    ax.bar(x - w, ranked["overall_score"], width=w, label="Overall", color="#4C78A8")
    ax.bar(x, ranked["variance_score"], width=w, label="Variance", color="#54A24B")
    ax.bar(x + w, ranked["error_score"], width=w, label="Error", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Model Scores")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_overlay(
    model_backtests: Dict[str, pd.DataFrame],
    ranking: pd.DataFrame,
    save_path: Path,
    top_n: int = 6,
) -> None:
    ranked_ids = ranking["model_id"].tolist()[:top_n]

    best_actual = None
    best_count = -1
    for mid, df in model_backtests.items():
        if "actual" not in df.columns or "ds" not in df.columns:
            continue
        valid = df[df["actual"].notna()][["ds", "actual"]].drop_duplicates("ds")
        if len(valid) > best_count:
            best_actual = valid.copy()
            best_count = len(valid)
    if best_actual is None or best_actual.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(best_actual["ds"], best_actual["actual"], color="black", linewidth=2.0, label="Actual")

    for mid in ranked_ids:
        df = model_backtests.get(mid)
        if df is None or "predicted" not in df.columns:
            continue
        pred = df[["ds", "predicted"]].dropna().drop_duplicates("ds").sort_values("ds")
        label_row = ranking[ranking["model_id"] == mid]
        label = label_row["model_label"].iloc[0] if not label_row.empty else mid
        ax.plot(pred["ds"], pred["predicted"], linewidth=1.4, alpha=0.8, label=label)

    ax.set_title(f"SA Revised Predictions Overlay (Top {min(top_n, len(ranked_ids))})")
    ax.set_xlabel("Date")
    ax.set_ylabel("MoM change")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _discover_model_folders(include_archive: bool, archive_limit: int) -> List[Tuple[str, str, Path]]:
    discovered: List[Tuple[str, str, Path]] = []

    # Main current pipeline SA revised.
    discovered.append(("main_current", "main", OUTPUT_DIR / "SA_prediction_revised"))

    # Existing sandbox experiments.
    discovered.extend(
        [
            ("catboost_sa_revised", "sandbox", OUTPUT_DIR / "sandbox" / "catboost_sa_revised"),
            ("xgboost_sa_revised", "sandbox", OUTPUT_DIR / "sandbox" / "xgboost_sa_revised"),
            ("sa_blend_walkforward", "sandbox", OUTPUT_DIR / "sandbox" / "sa_blend_walkforward"),
            ("nsa_predicted_adj", "sandbox", OUTPUT_DIR / "sandbox" / "nsa_predicted_adjustment_revised"),
            ("blend_predicted_adj", "sandbox", OUTPUT_DIR / "sandbox" / "sa_blend_predicted_adj_walkforward"),
        ]
    )

    # Variant suite.
    variant_root = OUTPUT_DIR / "sandbox" / "sa_revised_variants"
    if variant_root.exists():
        for variant_dir in sorted([p for p in variant_root.iterdir() if p.is_dir()]):
            discovered.append((variant_dir.name, "sandbox_variant", variant_dir))

    if include_archive:
        archive_root = OUTPUT_DIR / "Archive"
        if archive_root.exists():
            archives = sorted([p for p in archive_root.iterdir() if p.is_dir()], reverse=True)
            count = 0
            for ts_dir in archives:
                folder = ts_dir / "SA_prediction_revised"
                if folder.exists():
                    discovered.append((f"archive_{ts_dir.name}", "archive", folder))
                    count += 1
                    if count >= archive_limit:
                        break

    return discovered


def build_comparison(
    include_archive: bool,
    archive_limit: int,
    overlay_top_n: int,
    min_backtest_rows: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_entries = _discover_model_folders(include_archive=include_archive, archive_limit=archive_limit)

    rows: List[Dict] = []
    model_backtests: Dict[str, pd.DataFrame] = {}

    for model_id, source, folder in model_entries:
        backtest_path = folder / "backtest_results.csv"
        backtest = _read_backtest(backtest_path)
        if backtest is None:
            continue
        raw_metrics = _extract_metrics(folder)
        metrics = _normalize_record(raw=raw_metrics, backtest=backtest)

        model_backtests[model_id] = backtest
        row = {
            "model_id": model_id,
            "model_label": model_id.replace("_", " "),
            "source": source,
            "folder": str(folder),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError("No comparable SA revised model outputs found.")

    raw_df = pd.DataFrame(rows)
    if min_backtest_rows > 0:
        raw_df = raw_df[
            pd.to_numeric(raw_df["N_Backtest"], errors="coerce").fillna(0) >= float(min_backtest_rows)
        ].copy()
    if raw_df.empty:
        raise RuntimeError(
            f"No models met minimum backtest-row requirement (min_backtest_rows={min_backtest_rows})."
        )
    scored = _compute_scores(raw_df)
    ranked = scored.sort_values("overall_score", ascending=False).reset_index(drop=True)

    raw_df.to_csv(OUT_DIR / "model_metrics_raw.csv", index=False)
    ranked.to_csv(OUT_DIR / "model_metrics_ranked.csv", index=False)

    _plot_error_metrics(ranked, OUT_DIR / "compare_error_metrics.png")
    _plot_variance_metrics(ranked, OUT_DIR / "compare_variance_metrics.png")
    _plot_scores(ranked, OUT_DIR / "compare_scores.png")
    _plot_prediction_overlay(
        model_backtests=model_backtests,
        ranking=ranked,
        save_path=OUT_DIR / "compare_predictions_overlay.png",
        top_n=overlay_top_n,
    )

    top_cols = [
        "model_id",
        "source",
        "RMSE",
        "MAE",
        "STD_Ratio",
        "Diff_STD_Ratio",
        "Corr_Diff",
        "Diff_Sign_Accuracy",
        "Directional_Accuracy",
        "Acceleration_Accuracy",
        "overall_score",
        "variance_score",
        "error_score",
    ]
    display_df = ranked[top_cols].copy()
    with open(OUT_DIR / "comparison_summary.txt", "w") as f:
        f.write("SA Revised Model Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"Models compared: {len(display_df)}\n\n")
        f.write(display_df.to_string(index=False))
        f.write("\n")

    logger.info("Saved SA revised comparison outputs to %s", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SA revised models (main + sandbox + archives).")
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include historical _output/Archive/*/SA_prediction_revised runs.",
    )
    parser.add_argument(
        "--archive-limit",
        type=int,
        default=8,
        help="Maximum number of archive runs to include (latest first).",
    )
    parser.add_argument(
        "--overlay-top-n",
        type=int,
        default=6,
        help="Number of top-ranked models to include in prediction overlay.",
    )
    parser.add_argument(
        "--min-backtest-rows",
        type=int,
        default=12,
        help="Exclude models with fewer than this many valid backtest rows.",
    )
    args = parser.parse_args()
    build_comparison(
        include_archive=bool(args.include_archive),
        archive_limit=int(args.archive_limit),
        overlay_top_n=int(args.overlay_top_n),
        min_backtest_rows=int(args.min_backtest_rows),
    )


if __name__ == "__main__":
    main()
