"""
Sandbox output utilities for SA revised model experiments.

Provides production-style diagnostics for sandbox runs without touching
core pipeline artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import TEMP_DIR, setup_logger
from Train.Output_code.metrics import compute_metrics, save_metrics_csv
from Train.Output_code.plots import plot_backtest_predictions, render_summary_table

logger = setup_logger(__file__, TEMP_DIR)

try:
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


def _compute_directional_metrics(backtest: pd.DataFrame) -> Dict[str, float]:
    """Directional and acceleration accuracy on valid backtest rows."""
    if backtest.empty or not {"actual", "predicted"}.issubset(backtest.columns):
        return {
            "Directional_Accuracy": float("nan"),
            "Acceleration_Accuracy": float("nan"),
        }

    actual = backtest["actual"].values.astype(float)
    pred = backtest["predicted"].values.astype(float)
    dir_acc = float(np.mean(np.sign(actual) == np.sign(pred)))

    if len(backtest) >= 2:
        accel_actual = np.diff(actual)
        accel_pred = np.diff(pred)
        accel_acc = float(np.mean(np.sign(accel_actual) == np.sign(accel_pred)))
    else:
        accel_acc = float("nan")

    return {
        "Directional_Accuracy": dir_acc,
        "Acceleration_Accuracy": accel_acc,
    }


def _acf_pacf_df(series: pd.Series, max_lag: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (acf_df, pacf_df) with columns [lag, value]."""
    s = pd.Series(series, copy=False).astype(float).dropna()
    if s.empty:
        empty = pd.DataFrame({"lag": [], "value": []})
        return empty, empty

    n = len(s)
    max_acf_lag = int(max(1, min(max_lag, n - 1)))
    max_pacf_lag = int(max(1, min(max_lag, (n // 2) - 1, n - 1)))

    if not _HAS_STATSMODELS:
        idx = np.arange(0, max_acf_lag + 1)
        acf_df = pd.DataFrame({"lag": idx, "value": np.nan})
        pacf_df = pd.DataFrame({"lag": idx[: max_pacf_lag + 1], "value": np.nan})
        return acf_df, pacf_df

    acf_vals = sm_acf(s.values, nlags=max_acf_lag, fft=True, missing="drop")
    try:
        pacf_vals = sm_pacf(s.values, nlags=max_pacf_lag, method="ywm")
    except Exception:
        pacf_vals = np.full(max_pacf_lag + 1, np.nan, dtype=float)

    acf_df = pd.DataFrame({"lag": np.arange(len(acf_vals)), "value": acf_vals})
    pacf_df = pd.DataFrame({"lag": np.arange(len(pacf_vals)), "value": pacf_vals})
    return acf_df, pacf_df


def _plot_acf_pacf(
    acf_level: pd.DataFrame,
    pacf_level: pd.DataFrame,
    acf_error: pd.DataFrame,
    pacf_error: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    """Create a 2x2 diagnostics plot for ACF/PACF of level and residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()

    panels = [
        (acf_level, "ACF (Actual SA Revised)"),
        (pacf_level, "PACF (Actual SA Revised)"),
        (acf_error, "ACF (Residual Error)"),
        (pacf_error, "PACF (Residual Error)"),
    ]

    for ax, (df, ptitle) in zip(axes, panels):
        if df.empty:
            ax.set_title(f"{ptitle} [empty]")
            ax.axis("off")
            continue
        ax.bar(df["lag"], df["value"], color="#4C78A8", alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(ptitle)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_per_month(backtest: pd.DataFrame) -> list[Dict]:
    rows = []
    actual = backtest["actual"].values.astype(float)
    pred = backtest["predicted"].values.astype(float)

    dir_correct = (np.sign(actual) == np.sign(pred)).astype(int)
    accel_correct = np.full(len(backtest), np.nan)
    if len(backtest) >= 2:
        accel_correct[1:] = (
            np.sign(np.diff(actual)) == np.sign(np.diff(pred))
        ).astype(int)

    for i, (_, row) in enumerate(backtest.iterrows()):
        rows.append(
            {
                "ds": pd.Timestamp(row["ds"]).strftime("%Y-%m"),
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "error": float(row["error"]),
                "dir_correct": int(dir_correct[i]),
                "accel_correct": None if np.isnan(accel_correct[i]) else int(accel_correct[i]),
            }
        )
    return rows


def write_sandbox_output_bundle(
    results_df: pd.DataFrame,
    out_dir: Path,
    model_id: str,
    diagnostics_label: str = "SA revised",
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    """
    Write rich sandbox diagnostics bundle:
    - backtest_results.csv
    - summary_statistics.csv
    - summary_metrics.json
    - backtest_predictions.png
    - summary_table.png
    - acf/pacf CSVs for level + residuals
    - acf_pacf_diagnostics.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_df.copy()
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    df.to_csv(out_dir / "backtest_results.csv", index=False)
    backtest = df[df["error"].notna()].copy()

    metrics = compute_metrics(df)
    metrics.update(_compute_directional_metrics(backtest))
    metrics["N_Backtest"] = int(len(backtest))
    save_metrics_csv(metrics, out_dir / "summary_statistics.csv")

    if not backtest.empty:
        plot_backtest_predictions(
            df,
            title=f"{diagnostics_label} — Predicted vs Actual",
            save_path=out_dir / "backtest_predictions.png",
        )
    else:
        logger.warning("[%s] No backtest rows to plot", model_id)

    # ACF/PACF diagnostics
    acf_level, pacf_level = _acf_pacf_df(backtest["actual"] if not backtest.empty else pd.Series(dtype=float))
    acf_error, pacf_error = _acf_pacf_df(backtest["error"] if not backtest.empty else pd.Series(dtype=float))

    acf_level.to_csv(out_dir / "acf_sa_revised.csv", index=False)
    pacf_level.to_csv(out_dir / "pacf_sa_revised.csv", index=False)
    acf_error.to_csv(out_dir / "acf_error_sa_revised.csv", index=False)
    pacf_error.to_csv(out_dir / "pacf_error_sa_revised.csv", index=False)

    _plot_acf_pacf(
        acf_level=acf_level,
        pacf_level=pacf_level,
        acf_error=acf_error,
        pacf_error=pacf_error,
        save_path=out_dir / "acf_pacf_diagnostics.png",
        title=f"{diagnostics_label} ACF/PACF Diagnostics",
    )

    # Summary table image (no feature-importance dependency in sandbox)
    render_summary_table(
        metrics=metrics,
        top_features=[],
        n_features=int(n_features) if n_features is not None else 0,
        save_path=out_dir / "summary_table.png",
    )

    summary_payload = {
        "model_id": model_id,
        "overall": metrics,
        "per_month": _to_per_month(backtest) if not backtest.empty else [],
    }
    with open(out_dir / "summary_metrics.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("Saved sandbox diagnostics bundle for %s -> %s", model_id, out_dir)
    return metrics


def load_summary_for_comparison(summary_path: Path) -> Optional[Dict]:
    """Load sandbox/main summary JSON in either flat or nested format."""
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    if "overall" in data and isinstance(data["overall"], dict):
        return data
    # Backward-compat: old sandbox JSON stored flat metrics only.
    return {"model_id": summary_path.parent.name, "overall": data}
