"""
Exploration: SA-anchor + NSA-direction adjustment model.

Theory: Use SA revised prediction as a fixed anchor (weight=1),
then derive a directional/acceleration adjustment from NSA revised
to nudge the prediction in the right direction without destroying
the SA model's low MAE.

This script uses ONLY existing backtest results — no retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from Train.variance_metrics import compute_variance_kpis


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "_output"

# ── Load data ────────────────────────────────────────────────────────────
sa = pd.read_csv(
    OUTPUT_DIR / "SA_prediction_revised" / "backtest_results.csv",
    parse_dates=["ds"],
)
blend_raw = pd.read_csv(
    OUTPUT_DIR / "sandbox" / "sa_blend_walkforward" / "backtest_results.csv",
    parse_dates=["ds"],
)

# adj_predicted is the NSA-adjusted prediction (NSA model → seasonal adjustment → SA space)
df = blend_raw[["ds", "actual", "sa_predicted", "adj_predicted"]].copy()
df = df[df["actual"].notna()].reset_index(drop=True)

actual = df["actual"].values
sa_pred = df["sa_predicted"].values
adj_pred = df["adj_predicted"].values

n = len(actual)
print(f"Backtest months: {n}\n")

# ── Baselines ────────────────────────────────────────────────────────────
def evaluate(name: str, pred: np.ndarray, actual: np.ndarray = actual):
    err = actual - pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    kpis = compute_variance_kpis(actual, pred)
    # Acceleration accuracy
    n_acc = 0
    n_acc_correct = 0
    for i in range(1, len(actual)):
        actual_change = actual[i] - actual[i - 1]
        pred_change = pred[i] - pred[i - 1]
        n_acc += 1
        if np.sign(actual_change) == np.sign(pred_change):
            n_acc_correct += 1
    accel_acc = n_acc_correct / n_acc if n_acc > 0 else 0
    # Directional accuracy (sign of prediction matches sign of actual)
    dir_acc = np.mean(np.sign(pred) == np.sign(actual))
    print(f"  {name:45s}  MAE={mae:6.1f}  RMSE={rmse:6.1f}  "
          f"Dir={dir_acc:.1%}  Accel={accel_acc:.1%}  "
          f"STD_R={float(kpis['std_ratio']):.2f}  "
          f"Corr_D={float(kpis['corr_diff']):.2f}  "
          f"DiffSign={float(kpis['diff_sign_accuracy']):.1%}  "
          f"ExtHit={float(kpis['extreme_hit_rate']):.0%}")
    return mae, rmse, accel_acc

print("=" * 120)
print("BASELINES")
print("=" * 120)
evaluate("SA revised (standalone)", sa_pred)
evaluate("NSA adjusted (standalone)", adj_pred)

# Current walk-forward blend
blend_pred = df["sa_predicted"].values * 0.5057 + df["adj_predicted"].values * (1 - 0.5057)
evaluate("Current walk-forward blend (approx)", blend_pred)

# ── Approach 1: SA + scaled NSA MoM delta ────────────────────────────────
# The NSA-adjusted model captures month-over-month direction well.
# Extract the MoM change implied by NSA-adjusted and add a fraction to SA.
print("\n" + "=" * 120)
print("APPROACH 1: SA_pred + alpha * (adj_pred[t] - adj_pred[t-1])")
print("  Use the month-over-month change from NSA-adjusted as a directional nudge")
print("=" * 120)

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred = np.copy(sa_pred)
    for i in range(1, n):
        nsa_delta = adj_pred[i] - adj_pred[i - 1]
        pred[i] = sa_pred[i] + alpha * nsa_delta
    evaluate(f"alpha={alpha:.1f}", pred)

# ── Approach 2: SA + scaled (adj - SA) gap ───────────────────────────────
# The NSA-adjusted prediction often knows the direction better.
# Move SA some fraction toward it.
print("\n" + "=" * 120)
print("APPROACH 2: SA_pred + alpha * (adj_pred - SA_pred)")
print("  Move SA prediction a fraction toward NSA-adjusted")
print("=" * 120)

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    pred = sa_pred + alpha * (adj_pred - sa_pred)
    evaluate(f"alpha={alpha:.1f}", pred)

# ── Approach 3: Walk-forward adaptive alpha on (adj - SA) gap ────────────
# Same as Approach 2, but alpha is chosen per-month using trailing history.
print("\n" + "=" * 120)
print("APPROACH 3: Walk-forward SA + alpha*(adj-SA), alpha chosen on trailing window")
print("  Alpha optimized to minimize MAE on trailing 12-month window")
print("=" * 120)

for window in [6, 9, 12, 18]:
    pred = np.copy(sa_pred)
    alphas_used = []
    for i in range(n):
        if i < window:
            best_alpha = 0.0  # No history, use pure SA
        else:
            # Find best alpha on trailing window
            hist_actual = actual[i - window:i]
            hist_sa = sa_pred[i - window:i]
            hist_adj = adj_pred[i - window:i]
            best_mae = np.inf
            best_alpha = 0.0
            for a in np.arange(0.0, 0.81, 0.05):
                cand = hist_sa + a * (hist_adj - hist_sa)
                m = np.mean(np.abs(hist_actual - cand))
                if m < best_mae:
                    best_mae = m
                    best_alpha = a
        pred[i] = sa_pred[i] + best_alpha * (adj_pred[i] - sa_pred[i])
        alphas_used.append(best_alpha)
    evaluate(f"window={window:2d}", pred)
    print(f"    Mean alpha: {np.mean(alphas_used):.3f}, "
          f"last 12 alpha: {np.mean(alphas_used[-12:]):.3f}")

# ── Approach 4: SA + scaled NSA MoM delta, walk-forward alpha ────────────
print("\n" + "=" * 120)
print("APPROACH 4: Walk-forward SA + alpha*(adj_delta), alpha on trailing window")
print("  SA anchor + walk-forward optimized directional nudge from NSA MoM changes")
print("=" * 120)

for window in [6, 9, 12, 18]:
    pred = np.copy(sa_pred)
    alphas_used = []
    for i in range(n):
        if i < max(1, window):
            best_alpha = 0.0
        else:
            hist_actual = actual[max(1, i - window):i]
            hist_sa = sa_pred[max(1, i - window):i]
            hist_adj_delta = adj_pred[max(1, i - window):i] - adj_pred[max(1, i - window) - 1:i - 1]
            best_mae = np.inf
            best_alpha = 0.0
            for a in np.arange(0.0, 1.51, 0.05):
                cand = hist_sa + a * hist_adj_delta
                m = np.mean(np.abs(hist_actual - cand))
                if m < best_mae:
                    best_mae = m
                    best_alpha = a
            best_alpha = best_alpha
        if i >= 1:
            nsa_delta = adj_pred[i] - adj_pred[i - 1]
            pred[i] = sa_pred[i] + best_alpha * nsa_delta
        alphas_used.append(best_alpha)
    evaluate(f"window={window:2d}", pred)
    print(f"    Mean alpha: {np.mean(alphas_used):.3f}")

# ── Approach 5: Directional override ─────────────────────────────────────
# Keep SA magnitude, but if NSA-adjusted disagrees on MoM direction,
# flip the SA prediction's direction.
print("\n" + "=" * 120)
print("APPROACH 5: Directional override")
print("  Use SA magnitude but override MoM direction with NSA signal")
print("=" * 120)

# 5a: If NSA says MoM is opposite to SA's MoM, adjust SA toward NSA
pred5a = np.copy(sa_pred)
for i in range(1, n):
    sa_mom = sa_pred[i] - sa_pred[i - 1]
    nsa_mom = adj_pred[i] - adj_pred[i - 1]
    if np.sign(sa_mom) != np.sign(nsa_mom):
        # SA and NSA disagree on direction — nudge SA in NSA's direction
        # by flipping SA's delta sign but keeping its magnitude
        pred5a[i] = sa_pred[i - 1] + np.sign(nsa_mom) * abs(sa_mom)
evaluate("5a: flip SA delta when NSA disagrees", pred5a)

# 5b: Use SA level but add a fraction of NSA's MoM magnitude in NSA's direction
for frac in [0.3, 0.5, 0.7]:
    pred5b = np.copy(sa_pred)
    for i in range(1, n):
        nsa_mom = adj_pred[i] - adj_pred[i - 1]
        sa_mom = sa_pred[i] - sa_pred[i - 1]
        # Blend: keep SA's level, adjust by nsa_direction * blended_magnitude
        blended_mag = (1 - frac) * abs(sa_mom) + frac * abs(nsa_mom)
        pred5b[i] = sa_pred[i - 1] + np.sign(nsa_mom) * blended_mag
    evaluate(f"5b: NSA direction, blended magnitude (frac={frac})", pred5b)

# ── Approach 6: Two-component: SA for level, NSA for acceleration ────────
print("\n" + "=" * 120)
print("APPROACH 6: SA_pred + beta * nsa_acceleration")
print("  SA anchor + a fraction of the NSA-implied acceleration (second difference)")
print("=" * 120)

for beta in [0.1, 0.2, 0.3, 0.4, 0.5]:
    pred6 = np.copy(sa_pred)
    for i in range(2, n):
        nsa_accel = (adj_pred[i] - adj_pred[i - 1]) - (adj_pred[i - 1] - adj_pred[i - 2])
        pred6[i] = sa_pred[i] + beta * nsa_accel
    evaluate(f"beta={beta:.1f}", pred6)

print("\n" + "=" * 120)
print("SUMMARY: Best from each approach")
print("=" * 120)
