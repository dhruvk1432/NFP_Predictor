"""
Phase 0: Revision Analysis
Understand the statistical properties of revision deltas before building models.

revision_delta[M] = revised_mom[M] - first_release_mom[M]
where revised_mom uses levels from the M+1 FRED snapshot.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "_output"
FRED_DIR = BASE_DIR / "data" / "fred_data" / "decades"
ANALYSIS_DIR = OUTPUT_DIR / "revision_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def get_snapshot_path(snapshot_date: pd.Timestamp) -> Path:
    decade = f"{snapshot_date.year // 10 * 10}s"
    year = str(snapshot_date.year)
    month_str = snapshot_date.strftime("%Y-%m")
    return FRED_DIR / decade / year / f"{month_str}.parquet"


def load_revised_levels(snapshot_date: pd.Timestamp, series_name: str) -> pd.Series:
    path = get_snapshot_path(snapshot_date)
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    mask = df["series_name"] == series_name
    sub = df.loc[mask, ["date", "value"]].drop_duplicates("date").set_index("date")
    return sub["value"].sort_index()


def build_revision_data(backtest_df, series_name):
    """
    For each backtest month M, compute:
      - first_release_mom: the 'actual' from backtest results (first release)
      - revised_mom: level[M] - level[M-1] from M+1 snapshot
      - revision_delta: revised_mom - first_release_mom
    """
    records = []
    for _, row in backtest_df.iterrows():
        m = pd.Timestamp(row["ds"])
        first_release = row["actual"]

        snapshot_ts = m + pd.DateOffset(months=1)
        levels = load_revised_levels(snapshot_ts, series_name)
        if levels.empty:
            continue

        m_minus1 = m - pd.DateOffset(months=1)
        if m not in levels.index or m_minus1 not in levels.index:
            continue

        revised_mom = levels[m] - levels[m_minus1]
        records.append({
            "ds": m,
            "month": m.month,
            "first_release_mom": first_release,
            "revised_mom": revised_mom,
            "revision_delta": revised_mom - first_release,
        })

    return pd.DataFrame(records).set_index("ds")


def analyze_series(name, series_name, backtest_path):
    """Run full revision analysis for one model/series."""
    bt = pd.read_csv(backtest_path)
    bt = bt.dropna(subset=["actual", "predicted"])
    bt["ds"] = pd.to_datetime(bt["ds"])

    df = build_revision_data(bt, series_name)
    if df.empty:
        print(f"  No data for {name}")
        return None

    delta = df["revision_delta"]

    print(f"\n{'='*60}")
    print(f"  {name}  ({series_name})  —  {len(delta)} months")
    print(f"{'='*60}")

    # ── 1. Descriptive stats ──
    print(f"\n  Revision Delta Stats (revised_mom - first_release_mom):")
    print(f"    Mean:     {delta.mean():>8.1f}")
    print(f"    Median:   {delta.median():>8.1f}")
    print(f"    Std:      {delta.std():>8.1f}")
    print(f"    Min:      {delta.min():>8.1f}")
    print(f"    Max:      {delta.max():>8.1f}")
    print(f"    Skew:     {delta.skew():>8.2f}")
    print(f"    Kurtosis: {delta.kurtosis():>8.2f}")

    # ── 2. Sign-flip rate ──
    sign_first = np.sign(df["first_release_mom"])
    sign_revised = np.sign(df["revised_mom"])
    sign_flips = (sign_first != sign_revised).sum()
    print(f"\n  Sign flips (revision changes MoM direction): {sign_flips}/{len(delta)} "
          f"({sign_flips/len(delta)*100:.1f}%)")

    # List the months where sign flipped
    flip_mask = sign_first != sign_revised
    if flip_mask.any():
        print(f"    Months: {', '.join(df.index[flip_mask].strftime('%Y-%m'))}")
        for idx in df.index[flip_mask]:
            row = df.loc[idx]
            print(f"      {idx.strftime('%Y-%m')}: first_release={row['first_release_mom']:.0f} → revised={row['revised_mom']:.0f} (delta={row['revision_delta']:.0f})")

    # ── 3. Autocorrelation ──
    max_lags = min(12, len(delta) - 2)
    if max_lags >= 1:
        acf_vals = acf(delta.values, nlags=max_lags, fft=False)
        print(f"\n  Autocorrelation of revision_delta:")
        for lag in range(1, max_lags + 1):
            bar = "█" * int(abs(acf_vals[lag]) * 20)
            sign = "+" if acf_vals[lag] >= 0 else "-"
            print(f"    Lag {lag:>2}: {acf_vals[lag]:>7.3f}  {sign}{bar}")

    # ── 4. Seasonal pattern ──
    monthly = df.groupby("month")["revision_delta"].agg(["mean", "std", "count"])
    print(f"\n  Seasonal pattern (revision_delta by calendar month):")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m_idx, row in monthly.iterrows():
        bar = "█" * int(abs(row["mean"]) / max(abs(monthly["mean"].max()), abs(monthly["mean"].min()), 1) * 15)
        sign = "+" if row["mean"] >= 0 else "-"
        print(f"    {month_names[m_idx-1]:>3}: mean={row['mean']:>7.1f}  std={row['std']:>7.1f}  n={int(row['count'])}  {sign}{bar}")

    # ── 5. Correlation with first_release_mom ──
    corr_fr = delta.corr(df["first_release_mom"])
    print(f"\n  Correlation(revision_delta, first_release_mom): {corr_fr:.3f}")

    # ── 6. Correlation with revision features (if available from backtest) ──
    # Load the existing revision features from a training run if they exist
    # For now, just note what we'd check

    # ── 7. Revision as % of first release ──
    pct = (delta / df["first_release_mom"].replace(0, np.nan)).dropna()
    print(f"\n  Revision as % of first release:")
    print(f"    Mean:   {pct.mean()*100:>7.1f}%")
    print(f"    Median: {pct.median()*100:>7.1f}%")
    print(f"    Abs mean: {pct.abs().mean()*100:>7.1f}%")

    return df


def plot_all(data_dict):
    """Create a multi-panel summary figure."""
    n = len(data_dict)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (name, df) in enumerate(data_dict.items()):
        delta = df["revision_delta"]

        # Panel 1: Time series of revision delta
        ax1 = axes[i, 0]
        colors = ["#e74c3c" if v < 0 else "#2980b9" for v in delta.values]
        ax1.bar(df.index, delta.values, width=20, color=colors, alpha=0.7)
        ax1.axhline(0, color="grey", linewidth=0.8)
        ax1.set_title(f"{name}\nRevision Delta Over Time", fontweight="bold", fontsize=10)
        ax1.set_ylabel("Revised - First Release")
        ax1.tick_params(axis="x", rotation=45)

        # Panel 2: Distribution
        ax2 = axes[i, 1]
        ax2.hist(delta.values, bins=15, color="#2c3e50", alpha=0.7, edgecolor="white")
        ax2.axvline(delta.mean(), color="#e74c3c", linewidth=2, linestyle="--",
                    label=f"Mean={delta.mean():.1f}")
        ax2.axvline(0, color="grey", linewidth=1)
        ax2.set_title("Distribution of Revision Delta", fontweight="bold", fontsize=10)
        ax2.set_xlabel("Revision Delta")
        ax2.legend(fontsize=8)

        # Panel 3: Seasonal boxplot
        ax3 = axes[i, 2]
        month_data = [df.loc[df["month"] == m, "revision_delta"].values for m in range(1, 13)]
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        bp = ax3.boxplot(month_data, labels=month_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#2980b9")
            patch.set_alpha(0.5)
        ax3.axhline(0, color="grey", linewidth=0.8)
        ax3.set_title("Revision Delta by Month", fontweight="bold", fontsize=10)
        ax3.set_xlabel("Calendar Month")

    fig.suptitle("Revision Analysis: revised_mom - first_release_mom", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    out = ANALYSIS_DIR / "revision_analysis.jpg"
    fig.savefig(out, dpi=200, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"\nFigure saved to {out}")


def plot_autocorrelation(data_dict):
    """Plot ACF for each series."""
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, data_dict.items()):
        delta = df["revision_delta"]
        max_lags = min(12, len(delta) - 2)
        acf_vals = acf(delta.values, nlags=max_lags, fft=False)
        conf = 1.96 / np.sqrt(len(delta))

        ax.bar(range(1, max_lags + 1), acf_vals[1:], color="#2c3e50", alpha=0.7)
        ax.axhline(conf, color="#e74c3c", linestyle="--", linewidth=1, label=f"95% CI (±{conf:.2f})")
        ax.axhline(-conf, color="#e74c3c", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{name}\nAutocorrelation of Revision Delta", fontweight="bold", fontsize=10)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("ACF")
        ax.set_xticks(range(1, max_lags + 1))
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = ANALYSIS_DIR / "revision_acf.jpg"
    fig.savefig(out, dpi=200, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"ACF figure saved to {out}")


# ── Main ──
configs = {
    "NSA": {
        "path": OUTPUT_DIR / "NSA_prediction" / "backtest_results.csv",
        "series": "total_nsa",
    },
    "SA": {
        "path": OUTPUT_DIR / "SA_prediction" / "backtest_results.csv",
        "series": "total",
    },
}

all_data = {}
for name, cfg in configs.items():
    df = analyze_series(name, cfg["series"], cfg["path"])
    if df is not None:
        all_data[name] = df

if all_data:
    plot_all(all_data)
    plot_autocorrelation(all_data)

print("\n" + "=" * 60)
print("  SUMMARY: Key questions for deciding next steps")
print("=" * 60)
print("  1. Is revision_delta noise or structured?")
print("     → Check autocorrelation (any significant lags?)")
print("     → Check seasonal pattern (Jan/Jul benchmark revisions?)")
print("  2. How often do revisions flip the MoM sign?")
print("     → If rarely, direction accuracy is already near-ceiling")
print("  3. Is revision_delta correlated with first_release_mom?")
print("     → If yes, simple linear correction may help")
print("  4. Is revision_delta large enough to matter for acceleration?")
print("     → Compare delta std to typical MoM magnitude")
