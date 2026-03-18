#!/usr/bin/env python3
"""
Generate all plots and data for the NFP Predictor presentation.
Outputs are saved to _output/presentation/
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_output"
PRES = OUT / "presentation"
PRES.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})

# ---------------------------------------------------------------------------
# Helper: compute metrics from per-month data
# ---------------------------------------------------------------------------
def compute_metrics(df, actual_col="actual", pred_col="predicted"):
    """Compute MAE, RMSE, DirAcc, AccelAcc from a DataFrame."""
    err = df[actual_col] - df[pred_col]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    # Directional accuracy: both predicted and actual same sign
    dir_acc = ((df[actual_col] > 0) == (df[pred_col] > 0)).mean()

    # Acceleration accuracy
    actual_diff = df[actual_col].diff()
    pred_diff = df[pred_col].diff()
    valid = actual_diff.notna() & pred_diff.notna()
    if valid.sum() > 0:
        accel_acc = ((actual_diff[valid] > 0) == (pred_diff[valid] > 0)).mean()
    else:
        accel_acc = np.nan
    return {"MAE": mae, "RMSE": rmse, "DirAcc": dir_acc, "AccelAcc": accel_acc}


# ===================================================================
# 1. Kalman Fusion vs Consensus vs Actual
# ===================================================================
print("=" * 60)
print("1. Kalman Fusion vs Consensus vs Actual Plot")
print("=" * 60)

kalman_bt = pd.read_csv(OUT / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv")
kalman_bt["ds"] = pd.to_datetime(kalman_bt["ds"])

fig, ax = plt.subplots(figsize=(14, 6))

# Shade last 36 months
cutoff_36m = kalman_bt["ds"].max() - pd.DateOffset(months=35)
ax.axvspan(cutoff_36m, kalman_bt["ds"].max() + pd.DateOffset(days=15),
           alpha=0.10, color="gray", label="Last 36 months")

ax.plot(kalman_bt["ds"], kalman_bt["actual"], color="black", linewidth=1.8,
        marker="o", markersize=3, label="Actual")
ax.plot(kalman_bt["ds"], kalman_bt["predicted"], color="#2563EB", linewidth=1.5,
        marker="s", markersize=2.5, label="Kalman Fusion")
ax.plot(kalman_bt["ds"], kalman_bt["consensus_pred"], color="#DC2626",
        linewidth=1.5, linestyle="--", marker="^", markersize=2.5, label="Consensus")

ax.set_title("Kalman Fusion vs Consensus vs Actual (SA Revised MoM)", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("NFP MoM Change (thousands)")
ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
fig.savefig(PRES / "kalman_vs_consensus_vs_actual.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {PRES / 'kalman_vs_consensus_vs_actual.png'}")


# ===================================================================
# 2. NSA and SA Backtest Predictions
# ===================================================================
print("\n" + "=" * 60)
print("2. NSA and SA Backtest Prediction Plots")
print("=" * 60)

for label, folder in [("NSA", "NSA_prediction"), ("SA", "SA_prediction")]:
    bt = pd.read_csv(OUT / folder / "backtest_results.csv")
    bt["ds"] = pd.to_datetime(bt["ds"])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(bt["ds"], bt["actual"], color="black", linewidth=1.8,
            marker="o", markersize=3, label="Actual")
    ax.plot(bt["ds"], bt["predicted"], color="#2563EB", linewidth=1.5,
            marker="s", markersize=2.5, label=f"{label} LightGBM Predicted")

    # 80% CI shading
    if "lower_80" in bt.columns and "upper_80" in bt.columns:
        ax.fill_between(bt["ds"], bt["lower_80"], bt["upper_80"],
                        alpha=0.18, color="#2563EB", label="80% CI")

    ax.set_title(f"{label} First Release -- Backtest: Actual vs Predicted", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NFP MoM Change (thousands)")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()

    fname = f"{label.lower()}_backtest.png"
    fig.savefig(PRES / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PRES / fname}")


# ===================================================================
# 3. Metrics Comparison Bar Chart
# ===================================================================
print("\n" + "=" * 60)
print("3. Metrics Comparison Bar Chart (Full 59-month)")
print("=" * 60)

# Load the comparison metrics csv which has full-period numbers
comp = pd.read_csv(OUT / "consensus_anchor" / "comparison_metrics.csv")

# Extract rows for our three models
consensus_row = comp[comp["Forecast"].str.contains("Baseline_Consensus")].iloc[0]
kalman_row = comp[comp["Forecast"].str.contains("Kalman_Fusion")].iloc[0]
accel_row = comp[comp["Forecast"].str.contains("AccelOverride")].iloc[0]

metrics = ["MAE", "RMSE", "Acceleration_Accuracy", "Directional_Accuracy"]
display_names = ["MAE", "RMSE", "AccelAcc", "DirAcc"]

consensus_vals = [consensus_row[m] for m in metrics]
kalman_vals = [kalman_row[m] for m in metrics]
accel_vals = [accel_row[m] for m in metrics]

x = np.arange(len(display_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, consensus_vals, width, label="Consensus", color="#DC2626", alpha=0.85)
bars2 = ax.bar(x, kalman_vals, width, label="Kalman Fusion", color="#2563EB", alpha=0.85)
bars3 = ax.bar(x + width, accel_vals, width, label="AccelOverride", color="#16A34A", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(display_names)
ax.set_title("Model Comparison: Full 59-Month Backtest Period", fontweight="bold")
ax.legend(frameon=True, fancybox=True, shadow=True)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        fmt = f"{h:.1f}" if h > 2 else f"{h:.3f}"
        ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(PRES / "metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {PRES / 'metrics_comparison.png'}")


# ===================================================================
# 4. Feature Stability Heatmap (dynamic selection)
# ===================================================================
print("\n" + "=" * 60)
print("4. Feature Stability Heatmap")
print("=" * 60)

dyn_dir = OUT / "dynamic_selection" / "sa_revised"
if dyn_dir.exists():
    json_files = sorted(dyn_dir.glob("*.json"))
    if len(json_files) >= 2:
        all_features_by_date = {}
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            all_features_by_date[data["step_date"]] = set(data["features"])

        dates = sorted(all_features_by_date.keys())
        all_feats = sorted(set().union(*all_features_by_date.values()))

        # Build presence matrix
        mat = np.zeros((len(all_feats), len(dates)))
        for j, d in enumerate(dates):
            for i, feat in enumerate(all_feats):
                if feat in all_features_by_date[d]:
                    mat[i, j] = 1.0

        # Only show features present in at least 1 snapshot
        row_sums = mat.sum(axis=1)
        keep = row_sums > 0
        mat = mat[keep]
        feats_show = [f for f, k in zip(all_feats, keep) if k]

        fig, ax = plt.subplots(figsize=(max(8, len(dates) * 2), max(10, len(feats_show) * 0.22)))
        cax = ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(feats_show)))
        ax.set_yticklabels(feats_show, fontsize=6)
        ax.set_title("Dynamic Feature Selection Stability (SA Revised)", fontweight="bold")
        ax.set_xlabel("Reselection Window")
        fig.tight_layout()
        fig.savefig(PRES / "feature_stability_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {PRES / 'feature_stability_heatmap.png'}")

        # Jaccard stability
        jaccard_pairs = []
        for i in range(len(dates) - 1):
            s1 = all_features_by_date[dates[i]]
            s2 = all_features_by_date[dates[i + 1]]
            jacc = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
            jaccard_pairs.append(jacc)
        mean_jaccard = np.mean(jaccard_pairs)
        print(f"  Jaccard stability (consecutive windows): {jaccard_pairs}")
        print(f"  Mean Jaccard: {mean_jaccard:.4f}")
    else:
        print("  Only 1 dynamic selection JSON found; need >= 2 for heatmap.")
else:
    print("  No dynamic_selection/sa_revised/ directory found; skipping.")


# ===================================================================
# 5. Print ALL Numbers for Presentation
# ===================================================================
print("\n" + "=" * 60)
print("5. ALL NUMBERS FOR PRESENTATION")
print("=" * 60)

# --- Load all backtest results ---
models = {}

# Consensus (from Kalman backtest CSV -- consensus_pred column)
models["Consensus"] = kalman_bt[["ds", "actual", "consensus_pred"]].rename(
    columns={"consensus_pred": "predicted"})

# Kalman Fusion
models["Kalman Fusion"] = kalman_bt[["ds", "actual", "predicted"]]

# AccelOverride
accel_bt = pd.read_csv(OUT / "consensus_anchor" / "accel_override" / "backtest_results.csv")
accel_bt["ds"] = pd.to_datetime(accel_bt["ds"])
models["AccelOverride"] = accel_bt[["ds", "actual", "predicted"]]

# SA Direct (LightGBM)
sa_bt = pd.read_csv(OUT / "SA_prediction" / "backtest_results.csv")
sa_bt["ds"] = pd.to_datetime(sa_bt["ds"])
models["SA Direct (LightGBM)"] = sa_bt[["ds", "actual", "predicted"]]

# NSA + Adjustment
nsaadj_bt = pd.read_csv(OUT / "NSA_plus_adjustment" / "backtest_results.csv")
nsaadj_bt["ds"] = pd.to_datetime(nsaadj_bt["ds"])
models["NSA+Adj"] = nsaadj_bt[["ds", "actual", "predicted"]]

# SA Blend
sa_blend_json = json.load(open(OUT / "sandbox" / "sa_blend_walkforward" / "summary_metrics.json"))
sa_blend_pm = pd.DataFrame(sa_blend_json["per_month"])
sa_blend_pm["ds"] = pd.to_datetime(sa_blend_pm["ds"])
models["SA Blend"] = sa_blend_pm[["ds", "actual", "predicted"]]

# --- Compute full period + last 36m metrics ---
print("\n--- FULL PERIOD (59 months) METRICS ---")
print(f"{'Model':<25s} {'MAE':>8s} {'RMSE':>8s} {'DirAcc':>8s} {'AccelAcc':>9s}")
print("-" * 60)

full_results = {}
for name, df in models.items():
    m = compute_metrics(df)
    full_results[name] = m
    print(f"{name:<25s} {m['MAE']:8.1f} {m['RMSE']:8.1f} {m['DirAcc']:8.3f} {m['AccelAcc']:9.3f}")

# Last 36 months
cutoff_36 = models["Kalman Fusion"]["ds"].max() - pd.DateOffset(months=35)
print(f"\n--- LAST 36 MONTHS (from {cutoff_36.strftime('%Y-%m')}) METRICS ---")
print(f"{'Model':<25s} {'MAE':>8s} {'RMSE':>8s} {'DirAcc':>8s} {'AccelAcc':>9s}")
print("-" * 60)

last36_results = {}
for name, df in models.items():
    df_36 = df[df["ds"] >= cutoff_36].copy()
    if len(df_36) > 0:
        m = compute_metrics(df_36)
        last36_results[name] = m
        print(f"{name:<25s} {m['MAE']:8.1f} {m['RMSE']:8.1f} {m['DirAcc']:8.3f} {m['AccelAcc']:9.3f}")
    else:
        print(f"{name:<25s}  (no data in last 36m window)")

# --- Kalman Tuned Params ---
print("\n--- KALMAN FUSION TUNED PARAMS ---")
with open(OUT / "consensus_anchor" / "kalman_fusion" / "tuned_params.json") as f:
    kp = json.load(f)
for k, v in kp.items():
    print(f"  {k}: {v}")

# --- AccelOverride Tuned Params ---
print("\n--- ACCEL OVERRIDE TUNED PARAMS ---")
with open(OUT / "consensus_anchor" / "accel_override" / "tuned_params.json") as f:
    ap = json.load(f)
for k, v in ap.items():
    print(f"  {k}: {v}")

# --- SA Blend Weights ---
print("\n--- SA BLEND CONFIG ---")
with open(OUT / "sandbox" / "sa_blend_walkforward" / "blend_config.json") as f:
    bc = json.load(f)
for section, vals in bc.items():
    print(f"  [{section}]")
    if isinstance(vals, dict):
        for k, v in vals.items():
            print(f"    {k}: {v}")
    else:
        print(f"    {vals}")

# Mean SA blend weight from summary_statistics
sa_blend_stats = pd.read_csv(OUT / "sandbox" / "sa_blend_walkforward" / "summary_statistics.csv")
if "Mean_Blend_Weight_SA" in sa_blend_stats.columns:
    print(f"  Mean SA Blend Weight: {sa_blend_stats['Mean_Blend_Weight_SA'].iloc[0]:.4f}")

# --- Feature Counts ---
print("\n--- FEATURE SELECTION SUMMARY ---")
for branch in ["nsa_first_release", "sa_first_release"]:
    summary_path = OUT / "feature_reduction" / branch / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
        print(f"\n  [{branch}]")
        print(f"    Original features:   {s['original_json_count']}")
        print(f"    After Stage 1 (clusters): {s['stage1_output']}")
        print(f"    After Stage 2 (Boruta):   {s['stage2_output']}")
        print(f"    Final count:         {s['final_count']} (target: {s['target_n']})")

# --- Dynamic selection Jaccard ---
if dyn_dir.exists() and len(json_files) >= 2:
    print(f"\n--- DYNAMIC RESELECTION JACCARD (SA Revised) ---")
    for i, j_val in enumerate(jaccard_pairs):
        print(f"  {dates[i]} -> {dates[i+1]}: {j_val:.4f}")
    print(f"  Mean Jaccard: {mean_jaccard:.4f}")

# --- Top 10 Features ---
print("\n--- NSA TOP 10 FEATURES (by importance) ---")
nsa_fi = pd.read_csv(OUT / "NSA_prediction" / "feature_importance.csv")
for _, row in nsa_fi.head(10).iterrows():
    print(f"  {int(row['rank']):2d}. {row['feature_name']:<70s} {row['importance_score']:>14.1f}")

print("\n--- SA TOP 10 FEATURES (by importance) ---")
sa_fi = pd.read_csv(OUT / "SA_prediction" / "feature_importance.csv")
for _, row in sa_fi.head(10).iterrows():
    print(f"  {int(row['rank']):2d}. {row['feature_name']:<70s} {row['importance_score']:>14.1f}")

# --- Comparison metrics from the CSV ---
print("\n--- FULL COMPARISON METRICS (from comparison_metrics.csv) ---")
print(comp.to_string(index=False))

print("\n" + "=" * 60)
print("DONE. All presentation assets saved to:")
print(f"  {PRES}")
print("=" * 60)
