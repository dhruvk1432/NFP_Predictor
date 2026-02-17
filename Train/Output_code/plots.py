"""
Plotting functions for backtest output.

All plots are saved as PNG at 150 DPI.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, List, Tuple


def plot_backtest_predictions(
    results_df: pd.DataFrame,
    title: str,
    save_path: Path,
) -> None:
    """
    Line chart of predicted vs actual MoM change over the backtest period.

    Shades 80% prediction interval. Filters out future rows (NaN actuals).

    Args:
        results_df: Backtest DataFrame with columns ds, actual, predicted,
                    lower_80, upper_80.
        title: Plot title.
        save_path: Where to save the PNG.
    """
    df = results_df[results_df["actual"].notna()].copy()
    df = df.sort_values("ds")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["ds"], df["actual"], label="Actual", color="#1f77b4", linewidth=1.8)
    ax.plot(df["ds"], df["predicted"], label="Predicted", color="#ff7f0e", linewidth=1.8, linestyle="--")

    if "lower_80" in df.columns and "upper_80" in df.columns:
        ax.fill_between(
            df["ds"], df["lower_80"], df["upper_80"],
            alpha=0.15, color="#ff7f0e", label="80% CI",
        )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("MoM Change (thousands)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(
    model,
    X_sample: pd.DataFrame,
    feature_cols: List[str],
    save_path: Path,
) -> None:
    """
    Generate a SHAP beeswarm summary plot for the final production model.

    Args:
        model: Trained LightGBM Booster.
        X_sample: Feature matrix (training data) used as background.
        feature_cols: Feature column names the model was trained on.
        save_path: Where to save the PNG.
    """
    import shap

    available_cols = [c for c in feature_cols if c in X_sample.columns]
    X = X_sample[available_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def render_summary_table(
    metrics: Dict[str, float],
    top_features: List[Tuple[str, float]],
    n_features: int,
    save_path: Path,
) -> None:
    """
    Render a clean summary table as a PNG image.

    Shows summary statistics, top 5 features, and total feature count.

    Args:
        metrics: Dict with RMSE, MAE, MSE.
        top_features: List of (feature_name, importance) tuples, already sorted.
        n_features: Total number of features used by the model.
        save_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # Build table data
    rows = []
    rows.append(["RMSE", f"{metrics['RMSE']:.2f}"])
    rows.append(["MAE", f"{metrics['MAE']:.2f}"])
    rows.append(["MSE", f"{metrics['MSE']:.2f}"])
    rows.append(["", ""])  # spacer
    rows.append(["Total Features", str(n_features)])
    rows.append(["", ""])  # spacer
    rows.append(["Top Features", "Importance"])
    for i, (feat, imp) in enumerate(top_features[:5], 1):
        # Truncate long feature names
        display_name = feat if len(feat) <= 40 else feat[:37] + "..."
        rows.append([f"  {i}. {display_name}", f"{imp:.1f}"])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Style spacer and sub-header rows
    for i, row in enumerate(rows, start=1):
        if row == ["", ""]:
            for j in range(2):
                table[i, j].set_facecolor("#f0f0f0")
                table[i, j].set_edgecolor("#f0f0f0")
        elif row == ["Top Features", "Importance"]:
            for j in range(2):
                table[i, j].set_facecolor("#D6E4F0")
                table[i, j].set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
