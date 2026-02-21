import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "_output"
FRED_DIR = BASE_DIR / "data" / "fred_data" / "decades"

models = {
    "NSA Prediction": {
        "path": OUTPUT_DIR / "NSA_prediction" / "backtest_results.csv",
        "series": "total_nsa",
    },
    "SA Prediction": {
        "path": OUTPUT_DIR / "SA_prediction" / "backtest_results.csv",
        "series": "total",
    },
    "NSA + Adjustment": {
        "path": OUTPUT_DIR / "NSA_plus_adjustment" / "backtest_results.csv",
        "series": "total",
    },
}

# Add revised model paths if output exists
_revised = {
    "NSA Revised": {
        "path": OUTPUT_DIR / "NSA_prediction_revised" / "backtest_results.csv",
        "series": "total_nsa",
    },
    "SA Revised": {
        "path": OUTPUT_DIR / "SA_prediction_revised" / "backtest_results.csv",
        "series": "total",
    },
}
for _name, _cfg in _revised.items():
    if _cfg["path"].exists():
        models[_name] = _cfg


def get_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime("%Y-%m")
    return FRED_DIR / decade / year / f"{month_str}.parquet"


def load_revised_levels(snapshot_date: pd.Timestamp, series_name: str) -> pd.Series:
    """Load employment levels for a series from a given snapshot, indexed by date."""
    path = get_snapshot_path(snapshot_date)
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    mask = df["series_name"] == series_name
    sub = df.loc[mask, ["date", "value"]].drop_duplicates("date").set_index("date")
    return sub["value"].sort_index()


def get_revised_mom_and_accel(backtest_dates, series_name):
    """
    For each backtest month M, load the M+1 snapshot and compute:
      - revised_mom:   level[M] - level[M-1]   (direction)
      - revised_accel: mom[M] - mom[M-1]        (acceleration, using M-2 level)
    Returns DataFrames indexed by backtest month.
    """
    revised_mom = {}
    revised_accel = {}

    for m in backtest_dates:
        m_ts = pd.Timestamp(m)
        # Snapshot one month after M
        snapshot_ts = m_ts + pd.DateOffset(months=1)
        levels = load_revised_levels(snapshot_ts, series_name)
        if levels.empty:
            continue

        # Need levels for M, M-1, M-2
        m_minus1 = m_ts - pd.DateOffset(months=1)
        m_minus2 = m_ts - pd.DateOffset(months=2)

        if m_ts in levels.index and m_minus1 in levels.index:
            mom_m = levels[m_ts] - levels[m_minus1]
            revised_mom[m] = mom_m

            if m_minus2 in levels.index:
                mom_m1 = levels[m_minus1] - levels[m_minus2]
                revised_accel[m] = mom_m - mom_m1

    return pd.Series(revised_mom), pd.Series(revised_accel)


rows = []
# Store data for comparison charts: {model_name: {dates, revised_mom, predicted, first_release}}
chart_data = {}

for name, cfg in models.items():
    df = pd.read_csv(cfg["path"])
    df = df.dropna(subset=["actual", "predicted"]).reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"])

    series = cfg["series"]
    rev_mom, rev_accel = get_revised_mom_and_accel(df["ds"], series)

    # --- Direction accuracy (revised MoM vs predicted) ---
    dir_dates = rev_mom.index.intersection(df["ds"])
    pred_for_dir = df.set_index("ds").loc[dir_dates, "predicted"]
    first_release = df.set_index("ds").loc[dir_dates, "actual"]
    dir_acc = (np.sign(rev_mom[dir_dates].values) == np.sign(pred_for_dir.values)).mean() * 100
    n_dir = len(dir_dates)

    # Save for charts (only NSA and SA, not NSA+Adjustment which shares SA revised)
    if name in ("NSA Prediction", "SA Prediction"):
        chart_data[name] = {
            "dates": dir_dates,
            "revised_mom": rev_mom[dir_dates],
            "predicted": pred_for_dir,
            "first_release": first_release,
        }

    # --- Acceleration accuracy ---
    accel_dates = rev_accel.index.intersection(df["ds"])
    pred_series = df.set_index("ds")["predicted"]

    accel1_correct = 0
    accel2_correct = 0
    n_accel = 0

    for t in accel_dates:
        t_minus1 = t - pd.DateOffset(months=1)

        actual_accel = rev_accel[t]  # mom[t] - mom[t-1] from revised data
        pred_t = pred_series.get(t)

        # Type 1: sign(actual_accel) vs sign(pred[t] - pred[t-1])
        pred_tm1 = pred_series.get(t_minus1)
        if pred_tm1 is not None and pred_t is not None:
            pred_accel_1 = pred_t - pred_tm1
            if np.sign(actual_accel) == np.sign(pred_accel_1):
                accel1_correct += 1

        # Type 2: sign(actual_accel) vs sign(pred[t] - revised_mom[t-1])
        rev_mom_tm1 = rev_mom.get(t_minus1)
        if rev_mom_tm1 is not None and pred_t is not None:
            pred_accel_2 = pred_t - rev_mom_tm1
            if np.sign(actual_accel) == np.sign(pred_accel_2):
                accel2_correct += 1

        n_accel += 1

    accel1_acc = (accel1_correct / n_accel * 100) if n_accel > 0 else float("nan")
    accel2_acc = (accel2_correct / n_accel * 100) if n_accel > 0 else float("nan")

    rows.append({
        "Model": name,
        "N": n_dir,
        "Direction": f"{dir_acc:.1f}%",
        "Accel (P vs P)": f"{accel1_acc:.1f}%",
        "Accel (P vs A)": f"{accel2_acc:.1f}%",
    })

result = pd.DataFrame(rows)

# ── Table figure ──
fig_table, ax = plt.subplots(figsize=(7, 0.5 + 0.4 * len(result)))
ax.axis("off")
table = ax.table(
    cellText=result.values,
    colLabels=result.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

for j in range(len(result.columns)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

for i in range(1, len(result) + 1):
    color = "#f2f2f2" if i % 2 == 0 else "white"
    for j in range(len(result.columns)):
        table[i, j].set_facecolor(color)

fig_table.tight_layout()
table_path = OUTPUT_DIR / "directional_accuracy.jpg"
fig_table.savefig(table_path, dpi=200, bbox_inches="tight", format="jpeg")
plt.close(fig_table)

# ── Comparison chart: Revised MoM vs Predicted vs First Release (NSA & SA) ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

panel_info = [
    ("NSA Prediction", "total_nsa (PAYNSA)"),
    ("SA Prediction", "total (PAYEMS)"),
]

for ax, (model_name, series_label) in zip(axes, panel_info):
    d = chart_data[model_name]
    dates = d["dates"]
    x = pd.to_datetime(dates)

    ax.bar(x, d["revised_mom"].values, width=20, alpha=0.3, color="#2c3e50",
           label="Revised MoM (M+1 snapshot)")
    ax.plot(x, d["first_release"].values, "o-", color="#e74c3c", markersize=4,
            linewidth=1.2, label="First Release MoM")
    ax.plot(x, d["predicted"].values, "s--", color="#2980b9", markersize=4,
            linewidth=1.2, label="Predicted MoM")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="-")
    ax.set_ylabel("MoM Change (thousands)")
    ax.set_title(f"{model_name}  —  {series_label}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Revised vs First Release vs Predicted  (OOS Backtest)", fontsize=13,
             fontweight="bold", y=1.01)
fig.tight_layout()
chart_path = OUTPUT_DIR / "revised_vs_predicted_mom.jpg"
fig.savefig(chart_path, dpi=200, bbox_inches="tight", format="jpeg")
plt.close(fig)

print(f"Table saved to {table_path}")
print(f"Chart saved to {chart_path}")
print()
print(result.to_string(index=False))
print()
print("Using revised actuals from M+1 FRED snapshot (levels -> MoM diff)")
print("Direction      = sign(revised_mom[M]) vs sign(predicted[M])")
print("Accel (P vs P) = sign(revised_accel[M]) vs sign(predicted[M] - predicted[M-1])")
print("Accel (P vs A) = sign(revised_accel[M]) vs sign(predicted[M] - revised_mom[M-1])")
