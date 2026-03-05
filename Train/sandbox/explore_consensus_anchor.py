"""
Exploration: Consensus first_release_value as anchor + NSA direction adjustment.

Theory: Replace our SA revised prediction as anchor with the economists'
consensus first_release_value forecast, then apply NSA-derived directional
adjustments. If consensus is a better level anchor than our SA model,
the combined prediction should improve.

Uses existing backtest results + Unifier poll data.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from Train.variance_metrics import compute_variance_kpis

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "_output"

# ── Load unifier consensus ───────────────────────────────────────────────
UNIFIER_USER = os.getenv("UNIFIER_USER", "demo")
UNIFIER_TOKEN = os.getenv("UNIFIER_TOKEN", "qZy7KpE+L5Fg2MmjFqC/1xmweo7pgnnACCcK3JNEAec=")

from unifier import unifier
unifier.user = UNIFIER_USER
unifier.token = UNIFIER_TOKEN

poll_df = unifier.get_dataframe(
    name="lseg_us_reuters_polls", key="US&NFAREO", asof_date=None
)
print(f"Pulled {len(poll_df):,} rows from Reuters poll")
print(f"Columns: {list(poll_df.columns)}")

# Show which value columns are available
value_cols = [c for c in poll_df.columns if 'value' in c.lower() or 'median' in c.lower()
              or 'mean' in c.lower() or 'forecast' in c.lower() or 'estimate' in c.lower()]
print(f"Value-related columns: {value_cols}")
print()

# ── Prepare consensus monthly (using first_release_value) ────────────────
date_col = "timestamp"
# Try first_release_value first, fall back to other columns
for candidate_col in ["first_release_value", "latest_revised_actual_value"]:
    if candidate_col in poll_df.columns:
        value_col = candidate_col
        break
else:
    raise KeyError(f"No suitable value column found. Available: {list(poll_df.columns)}")

print(f"Using consensus column: {value_col}")

x = poll_df.copy()
x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
x = x.dropna(subset=[date_col, value_col]).copy()
x["ds"] = x[date_col].dt.to_period("M").dt.to_timestamp()
x = x.sort_values(date_col)

consensus = (
    x.groupby("ds", as_index=False)
     .agg(consensus_pred=(value_col, "last"))
)

# Also load the latest_revised version for comparison
if "latest_revised_actual_value" in poll_df.columns and value_col != "latest_revised_actual_value":
    x2 = poll_df.copy()
    x2[date_col] = pd.to_datetime(x2[date_col], errors="coerce")
    x2["latest_revised_actual_value"] = pd.to_numeric(x2["latest_revised_actual_value"], errors="coerce")
    x2 = x2.dropna(subset=[date_col, "latest_revised_actual_value"]).copy()
    x2["ds"] = x2[date_col].dt.to_period("M").dt.to_timestamp()
    x2 = x2.sort_values(date_col)
    consensus_revised = (
        x2.groupby("ds", as_index=False)
          .agg(consensus_revised=(("latest_revised_actual_value"), "last"))
    )
    consensus = consensus.merge(consensus_revised, on="ds", how="left")

print(f"Consensus monthly rows: {len(consensus)}")

# ── Load model backtests ─────────────────────────────────────────────────
blend_raw = pd.read_csv(
    OUTPUT_DIR / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv",
    parse_dates=["ds"],
)
sa_bt = pd.read_csv(
    OUTPUT_DIR / "SA_prediction_revised" / "backtest_results.csv",
    parse_dates=["ds"],
)

# Merge everything
df = blend_raw[["ds", "actual", "sa_predicted", "adj_predicted"]].copy()
df = df.merge(consensus, on="ds", how="left")
df = df[df["actual"].notna()].reset_index(drop=True)

# Check overlap
has_consensus = df["consensus_pred"].notna().sum()
print(f"Backtest months with consensus: {has_consensus} / {len(df)}")
if has_consensus < len(df):
    missing = df[df["consensus_pred"].isna()]["ds"].tolist()
    print(f"  Missing consensus for: {missing}")

# Drop rows without consensus for fair comparison
df = df[df["consensus_pred"].notna()].reset_index(drop=True)
n = len(df)

actual = df["actual"].values
sa_pred = df["sa_predicted"].values
adj_pred = df["adj_predicted"].values
cons_pred = df["consensus_pred"].values

print(f"\nFinal comparison window: {n} months ({df['ds'].min()} to {df['ds'].max()})")
print()

# ── Evaluation helper ────────────────────────────────────────────────────
def evaluate(name: str, pred: np.ndarray, actual: np.ndarray = actual):
    err = actual - pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    kpis = compute_variance_kpis(actual, pred)
    n_acc = 0
    n_acc_correct = 0
    for i in range(1, len(actual)):
        actual_change = actual[i] - actual[i - 1]
        pred_change = pred[i] - pred[i - 1]
        n_acc += 1
        if np.sign(actual_change) == np.sign(pred_change):
            n_acc_correct += 1
    accel_acc = n_acc_correct / n_acc if n_acc > 0 else 0
    dir_acc = np.mean(np.sign(pred) == np.sign(actual))
    print(f"  {name:55s}  MAE={mae:6.1f}  RMSE={rmse:6.1f}  "
          f"Dir={dir_acc:.1%}  Accel={accel_acc:.1%}  "
          f"STD_R={float(kpis['std_ratio']):.2f}  "
          f"Corr_D={float(kpis['corr_diff']):.2f}  "
          f"DiffSign={float(kpis['diff_sign_accuracy']):.1%}  "
          f"ExtHit={float(kpis['extreme_hit_rate']):.0%}")
    return mae, accel_acc

# ── Baselines ────────────────────────────────────────────────────────────
print("=" * 130)
print("BASELINES (all on same {n} months)")
print("=" * 130)
evaluate("SA revised model", sa_pred)
evaluate("NSA adjusted model", adj_pred)
evaluate("Consensus first_release", cons_pred)
if "consensus_revised" in df.columns:
    evaluate("Consensus latest_revised", df["consensus_revised"].values)

# Current blend for reference
blend_pred = 0.5057 * sa_pred + (1 - 0.5057) * adj_pred
evaluate("Current walk-forward blend (approx)", blend_pred)

# Best SA-anchor approach from previous exploration
sa_anchor_02 = sa_pred + 0.2 * (adj_pred - sa_pred)
evaluate("SA anchor + 0.2*(adj-SA)  [prev best]", sa_anchor_02)

# ── Approach A: Consensus anchor + alpha*(adj - consensus) ───────────────
print("\n" + "=" * 130)
print("APPROACH A: consensus + alpha * (adj_pred - consensus)")
print("  Use consensus as level anchor, move toward NSA-adjusted")
print("=" * 130)

for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    pred = cons_pred + alpha * (adj_pred - cons_pred)
    evaluate(f"alpha={alpha:.2f}", pred)

# ── Approach B: Consensus anchor + alpha*(SA - consensus) ────────────────
print("\n" + "=" * 130)
print("APPROACH B: consensus + alpha * (SA_pred - consensus)")
print("  Use consensus as anchor, move toward SA model")
print("=" * 130)

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    pred = cons_pred + alpha * (sa_pred - cons_pred)
    evaluate(f"alpha={alpha:.1f}", pred)

# ── Approach C: Consensus + alpha*(adj-cons) + beta*(SA-cons) ────────────
print("\n" + "=" * 130)
print("APPROACH C: consensus + alpha*(adj-cons) + beta*(SA-cons)")
print("  Three-way: consensus anchor, blended model corrections")
print("=" * 130)

best_mae_c = np.inf
best_params_c = (0, 0)
for alpha in np.arange(0.0, 0.51, 0.05):
    for beta in np.arange(0.0, 0.51, 0.05):
        pred = cons_pred + alpha * (adj_pred - cons_pred) + beta * (sa_pred - cons_pred)
        mae = np.mean(np.abs(actual - pred))
        if mae < best_mae_c:
            best_mae_c = mae
            best_params_c = (alpha, beta)

alpha, beta = best_params_c
pred = cons_pred + alpha * (adj_pred - cons_pred) + beta * (sa_pred - cons_pred)
print(f"  Best: alpha={alpha:.2f} (adj), beta={beta:.2f} (SA)")
evaluate(f"alpha={alpha:.2f}, beta={beta:.2f}", pred)

# Also show a few nearby points
for a in [best_params_c[0] - 0.05, best_params_c[0], best_params_c[0] + 0.05]:
    for b in [best_params_c[1] - 0.05, best_params_c[1], best_params_c[1] + 0.05]:
        if a < 0 or b < 0:
            continue
        pred = cons_pred + a * (adj_pred - cons_pred) + b * (sa_pred - cons_pred)
        evaluate(f"a={a:.2f}, b={b:.2f}", pred)

# ── Approach D: Walk-forward consensus + alpha*(adj - cons) ──────────────
print("\n" + "=" * 130)
print("APPROACH D: Walk-forward consensus + alpha*(adj-cons)")
print("  Alpha optimized on trailing window to minimize MAE")
print("=" * 130)

for window in [6, 9, 12, 18]:
    pred = np.copy(cons_pred)
    alphas_used = []
    for i in range(n):
        if i < window:
            best_alpha = 0.0
        else:
            hist_actual = actual[i - window:i]
            hist_cons = cons_pred[i - window:i]
            hist_adj = adj_pred[i - window:i]
            best_mae_inner = np.inf
            best_alpha = 0.0
            for a in np.arange(0.0, 0.61, 0.02):
                cand = hist_cons + a * (hist_adj - hist_cons)
                m = np.mean(np.abs(hist_actual - cand))
                if m < best_mae_inner:
                    best_mae_inner = m
                    best_alpha = a
        pred[i] = cons_pred[i] + best_alpha * (adj_pred[i] - cons_pred[i])
        alphas_used.append(best_alpha)
    evaluate(f"window={window:2d}", pred)
    print(f"    Mean alpha: {np.mean(alphas_used):.3f}, "
          f"last 12: {np.mean(alphas_used[-12:]):.3f}")

# ── Approach E: Walk-forward 3-way (cons + adj + SA) ─────────────────────
print("\n" + "=" * 130)
print("APPROACH E: Walk-forward consensus + alpha*(adj-cons) + beta*(SA-cons)")
print("  Both alpha and beta optimized on trailing window")
print("=" * 130)

for window in [9, 12, 18]:
    pred = np.copy(cons_pred)
    for i in range(n):
        if i < window:
            best_a, best_b = 0.0, 0.0
        else:
            hist_actual = actual[i - window:i]
            hist_cons = cons_pred[i - window:i]
            hist_adj = adj_pred[i - window:i]
            hist_sa = sa_pred[i - window:i]
            best_mae_inner = np.inf
            best_a, best_b = 0.0, 0.0
            for a in np.arange(0.0, 0.51, 0.05):
                for b in np.arange(0.0, 0.51, 0.05):
                    cand = hist_cons + a * (hist_adj - hist_cons) + b * (hist_sa - hist_cons)
                    m = np.mean(np.abs(hist_actual - cand))
                    if m < best_mae_inner:
                        best_mae_inner = m
                        best_a, best_b = a, b
        pred[i] = cons_pred[i] + best_a * (adj_pred[i] - cons_pred[i]) + best_b * (sa_pred[i] - cons_pred[i])
    evaluate(f"window={window:2d}", pred)

# ── Per-month comparison table ───────────────────────────────────────────
print("\n" + "=" * 130)
print("PER-MONTH COMPARISON: Consensus anchor (alpha=0.15) vs SA anchor (alpha=0.2) vs Current blend")
print("=" * 130)

cons_blend = cons_pred + 0.15 * (adj_pred - cons_pred)
sa_blend = sa_pred + 0.2 * (adj_pred - sa_pred)
wf_blend = 0.5057 * sa_pred + (1 - 0.5057) * adj_pred

print(f"{'Month':>10s}  {'Actual':>7s}  {'Cons':>7s}  {'ConsBlend':>9s}  {'SA':>7s}  {'SABlend':>9s}  {'WFBlend':>9s}  "
      f"{'|ConsB|':>7s}  {'|SAB|':>7s}  {'|WFB|':>7s}  {'Winner':>8s}")
print("-" * 130)

wins = {"ConsBlend": 0, "SABlend": 0, "WFBlend": 0}
for i in range(n):
    ds = df["ds"].iloc[i].strftime("%Y-%m")
    a = actual[i]
    c = cons_pred[i]
    cb = cons_blend[i]
    s = sa_pred[i]
    sb = sa_blend[i]
    wb = wf_blend[i]
    e_cb = abs(a - cb)
    e_sb = abs(a - sb)
    e_wb = abs(a - wb)
    winner = min([("ConsBlend", e_cb), ("SABlend", e_sb), ("WFBlend", e_wb)], key=lambda x: x[1])[0]
    wins[winner] += 1
    print(f"{ds:>10s}  {a:7.0f}  {c:7.0f}  {cb:9.1f}  {s:7.0f}  {sb:9.1f}  {wb:9.1f}  "
          f"{e_cb:7.1f}  {e_sb:7.1f}  {e_wb:7.1f}  {winner:>8s}")

print(f"\nWin counts: {wins}")
