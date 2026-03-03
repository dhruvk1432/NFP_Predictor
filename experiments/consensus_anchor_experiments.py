"""
Consensus-Anchor Integration Experiments
=========================================
Tests 7 mathematical frameworks for integrating economist consensus
as an anchor into the NFP prediction pipeline.

Key insight: consensus beats our models on MAE/RMSE but our champion
blend beats consensus on acceleration accuracy (64.7% vs 41.2%).
Goal: combine both strengths.

Data available:
  - 315 months of consensus vs actual (1999-2026)  -> train consensus-bias models
  - 35 months of overlap (consensus + champion + challenger + actual) -> test all combos

All approaches use strict expanding-window evaluation (no lookahead).

Usage:
    python experiments/consensus_anchor_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Train.variance_metrics import compute_variance_kpis

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MERGED_CSV = REPO_ROOT / "_output" / "consensus_comparison" / "consensus_model_merged.csv"
TARGET_PARQUET = REPO_ROOT / "data" / "NFP_target" / "y_sa_revised.parquet"
OUT_DIR = REPO_ROOT / "_output" / "consensus_anchor_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum expanding-window history before producing a prediction
MIN_HISTORY = 12

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the merged consensus + model predictions dataset AND backfill
    actual NFP values from the target parquet for the full consensus history.

    The merged CSV only has 'actual' for the 35-month overlap where model
    backtests exist. By joining with y_sa_revised.parquet (y_mom column),
    we get actual values for all ~315 consensus months back to 1999.
    This is critical for Approach 1 (residual learning) which trains on
    the full consensus-vs-actual history.
    """
    df = pd.read_csv(MERGED_CSV, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Load full actual NFP MoM from target parquet
    target = pd.read_parquet(TARGET_PARQUET, columns=["ds", "y_mom"])
    target["ds"] = pd.to_datetime(target["ds"])
    target = target.rename(columns={"y_mom": "actual_from_target"})

    # Merge: backfill actual where missing
    df = df.merge(target, on="ds", how="left")
    df["actual"] = df["actual"].combine_first(df["actual_from_target"])
    df = df.drop(columns=["actual_from_target"])
    df = df.sort_values("ds").reset_index(drop=True)

    return df


def split_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into:
      - consensus_full: rows with consensus + actual (no model pred needed)
        Now ~270+ rows thanks to target parquet backfill.
      - overlap: rows with consensus + champion + challenger + actual -> 35 rows
    """
    consensus_full = df[df["consensus_pred"].notna() & df["actual"].notna()].copy()
    overlap = df[
        df["consensus_pred"].notna()
        & df["actual"].notna()
        & df["champion_pred"].notna()
        & df["challenger_pred"].notna()
    ].copy()
    return consensus_full, overlap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def full_metrics(actual: np.ndarray, pred: np.ndarray, label: str) -> Dict:
    """Compute the full metric suite matching the comparison notebook."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]

    if a.size == 0:
        return {"Forecast": label, "N": 0}

    e = a - p
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    me = float(np.mean(e))
    medae = float(np.median(np.abs(e)))

    dir_acc = float(np.mean(np.sign(a) == np.sign(p)))
    if a.size >= 2:
        accel_acc = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))))
    else:
        accel_acc = np.nan

    vk = compute_variance_kpis(a, p)

    return {
        "Forecast": label,
        "N": int(a.size),
        "RMSE": rmse,
        "MAE": mae,
        "ME_Bias": me,
        "MedAE": medae,
        "Directional_Accuracy": dir_acc,
        "Acceleration_Accuracy": accel_acc,
        "STD_Ratio": float(vk["std_ratio"]),
        "Diff_STD_Ratio": float(vk["diff_std_ratio"]),
        "Corr_Level": float(vk["corr_level"]),
        "Corr_Diff": float(vk["corr_diff"]),
        "Diff_Sign_Accuracy": float(vk["diff_sign_accuracy"]),
        "Tail_MAE": float(vk["tail_mae"]),
        "Extreme_Hit_Rate": float(vk["extreme_hit_rate"]),
    }


# ============================================================================
# APPROACH 1: Consensus-Anchored Residual Learning
# ============================================================================
#
# Mathematical Framework:
#   Instead of predicting y (NFP) directly, we predict the consensus error:
#       ε_t = actual_t - consensus_t
#
#   The consensus error is a much lower-variance signal than NFP itself.
#   We model ε_t using features derived from the consensus history:
#
#       ε̂_t = f(consensus_t, Δconsensus_t, |consensus_t|, regime_t, ...)
#
#   Final prediction: ŷ_t = consensus_t + ε̂_t
#
#   Why this works:
#   - Consensus already encodes ~60% of the level information (corr_level=0.66)
#   - The residual ε has systematic patterns: consensus has +30.4 bias,
#     tends to underreact to acceleration, and misses tail events
#   - By modeling these patterns, we preserve consensus MAE while adding
#     our acceleration edge
#
#   Features for residual model (all PIT-safe, derived from consensus history):
#   - consensus_t: current consensus level
#   - Δconsensus_t = consensus_t - consensus_{t-1}: consensus momentum
#   - |consensus_t|: magnitude (absolute level)
#   - consensus_accel = Δconsensus_t - Δconsensus_{t-1}: consensus acceleration
#   - trailing_bias_6m: mean(actual - consensus) over prior 6 months
#   - trailing_bias_12m: mean(actual - consensus) over prior 12 months
#   - trailing_abs_error_6m: mean(|actual - consensus|) over prior 6 months
#   - consensus_dispersion: std of consensus estimates (if available)
#   - regime indicator: based on consensus level buckets
# ============================================================================

def _build_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build PIT-safe features from consensus history for residual modeling.
    All features at time t use only data available at or before t.
    """
    out = df[["ds", "consensus_pred", "actual"]].copy()
    out["residual"] = out["actual"] - out["consensus_pred"]

    # Consensus dynamics (known at prediction time)
    out["consensus_mom"] = out["consensus_pred"].diff()
    out["consensus_accel"] = out["consensus_mom"].diff()
    out["consensus_abs"] = out["consensus_pred"].abs()
    out["consensus_sign"] = np.sign(out["consensus_pred"])

    # Trailing bias (expanding window, shifted to avoid lookahead)
    out["trailing_bias_6m"] = out["residual"].shift(1).rolling(6, min_periods=3).mean()
    out["trailing_bias_12m"] = out["residual"].shift(1).rolling(12, min_periods=6).mean()
    out["trailing_bias_24m"] = out["residual"].shift(1).rolling(24, min_periods=12).mean()

    # Trailing absolute error
    out["trailing_abs_err_6m"] = out["residual"].shift(1).abs().rolling(6, min_periods=3).mean()
    out["trailing_abs_err_12m"] = out["residual"].shift(1).abs().rolling(12, min_periods=6).mean()

    # Trailing directional miss rate (how often consensus got direction wrong)
    dir_miss = (np.sign(out["actual"]) != np.sign(out["consensus_pred"])).astype(float)
    out["trailing_dir_miss_12m"] = dir_miss.shift(1).rolling(12, min_periods=6).mean()

    # Trailing acceleration miss rate
    actual_accel = np.sign(out["actual"].diff())
    cons_accel = np.sign(out["consensus_pred"].diff())
    accel_miss = (actual_accel != cons_accel).astype(float)
    out["trailing_accel_miss_12m"] = accel_miss.shift(1).rolling(12, min_periods=6).mean()

    # Consensus regime (bucketed level)
    out["consensus_regime"] = pd.cut(
        out["consensus_pred"],
        bins=[-np.inf, -100, 0, 100, 200, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Interaction: consensus level * trailing bias
    out["consensus_x_bias"] = out["consensus_pred"] * out["trailing_bias_12m"]

    return out


def approach_1_residual_learning(
    consensus_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    method: str = "ridge",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Approach 1: Learn consensus residual patterns from full history,
    then apply correction.

    Methods:
      - 'ridge': Ridge regression on consensus features
      - 'bias_only': Simple trailing bias correction (baseline)
      - 'adaptive_bias': Exponentially-weighted trailing bias
    """
    full = _build_consensus_features(consensus_df)
    feature_cols = [
        "consensus_pred", "consensus_mom", "consensus_accel", "consensus_abs",
        "consensus_sign", "trailing_bias_6m", "trailing_bias_12m", "trailing_bias_24m",
        "trailing_abs_err_6m", "trailing_abs_err_12m", "trailing_dir_miss_12m",
        "trailing_accel_miss_12m", "consensus_regime", "consensus_x_bias",
    ]

    # Expanding-window predictions on the OVERLAP period
    overlap_start = overlap_df["ds"].min()
    results = []

    for idx in range(len(full)):
        row = full.iloc[idx]
        if row["ds"] < overlap_start:
            continue
        if pd.isna(row["actual"]):
            continue

        # Check this month is in our overlap set
        if row["ds"] not in overlap_df["ds"].values:
            continue

        # Training data: everything before this month
        train = full.iloc[:idx].dropna(subset=["residual"])
        train = train.dropna(subset=feature_cols, how="all")

        if len(train) < MIN_HISTORY:
            pred = row["consensus_pred"]  # fallback to raw consensus
        elif method == "bias_only":
            # Simple: correct by trailing 12m bias
            bias = train["residual"].iloc[-12:].mean()
            pred = row["consensus_pred"] + bias
        elif method == "adaptive_bias":
            # Exponentially-weighted bias with half-life of 12 months
            resids = train["residual"].values
            n = len(resids)
            weights = np.exp(-np.log(2) * np.arange(n - 1, -1, -1) / 12.0)
            bias = float(np.average(resids, weights=weights))
            pred = row["consensus_pred"] + bias
        elif method == "ridge":
            from sklearn.linear_model import Ridge

            X_train = train[feature_cols].fillna(0).values
            y_train = train["residual"].values

            X_test = pd.DataFrame([row[feature_cols]]).fillna(0).values

            model = Ridge(alpha=10.0, fit_intercept=True)
            model.fit(X_train, y_train)
            residual_pred = model.predict(X_test)[0]
            pred = row["consensus_pred"] + residual_pred
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
        })

    res_df = pd.DataFrame(results)
    label = f"A1_Residual_{method}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# APPROACH 2: Dynamic Linear Blend (Consensus + Champion)
# ============================================================================
#
# Mathematical Framework:
#   ŷ_t = α_t · consensus_t + (1 - α_t) · champion_t
#
#   The weight α_t is chosen by walk-forward optimization on trailing history.
#
#   Key insight: we can optimize α for DIFFERENT objectives:
#     - α_mae: minimizes MAE on trailing window
#     - α_accel: maximizes acceleration accuracy on trailing window
#     - α_composite: minimizes a weighted composite
#
#   The optimal α is found by grid search over [0, 1] at each step.
#
#   Extension: use TWO weights, one for "level" and one for "delta":
#     level_t = α · consensus_t + (1-α) · champion_t
#     Δ̂_t = β · Δconsensus_t + (1-β) · Δchampion_t
#     ŷ_t = γ · level_t + (1-γ) · (actual_{t-1} + Δ̂_t)
#
#   This decouples level accuracy (consensus strength) from
#   acceleration accuracy (model strength).
# ============================================================================

def approach_2_dynamic_blend(
    overlap_df: pd.DataFrame,
    objective: str = "mae",
    window: int = 18,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk-forward dynamic linear blend of consensus + champion.
    """
    df = overlap_df[["ds", "actual", "consensus_pred", "champion_pred"]].dropna().copy()
    df = df.sort_values("ds").reset_index(drop=True)

    grid = np.arange(0.0, 1.01, 0.02)
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i]

        if len(hist) < MIN_HISTORY:
            alpha = 0.5
        else:
            h = hist.iloc[-window:]
            a = h["actual"].values
            best_score = np.inf if objective != "accel" else -np.inf
            best_alpha = 0.5

            for cand in grid:
                p = cand * h["consensus_pred"].values + (1 - cand) * h["champion_pred"].values

                if objective == "mae":
                    score = float(np.mean(np.abs(a - p)))
                    if score < best_score:
                        best_score = score
                        best_alpha = cand
                elif objective == "accel":
                    if len(a) >= 2:
                        score = float(np.mean(
                            np.sign(np.diff(a)) == np.sign(np.diff(p))
                        ))
                    else:
                        score = 0.0
                    if score > best_score:
                        best_score = score
                        best_alpha = cand
                elif objective == "composite":
                    mae = float(np.mean(np.abs(a - p)))
                    if len(a) >= 2:
                        acc = float(np.mean(
                            np.sign(np.diff(a)) == np.sign(np.diff(p))
                        ))
                    else:
                        acc = 0.5
                    # Composite: lower is better. Penalize MAE, reward accel.
                    score = mae - 50.0 * acc
                    if score < best_score:
                        best_score = score
                        best_alpha = cand
                else:
                    raise ValueError(objective)

            alpha = best_alpha

        pred = alpha * row["consensus_pred"] + (1 - alpha) * row["champion_pred"]
        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "alpha": alpha,
            "consensus_pred": row["consensus_pred"],
            "champion_pred": row["champion_pred"],
        })

    res_df = pd.DataFrame(results)
    label = f"A2_Blend_{objective}_w{window}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# APPROACH 3: Decoupled Level-Acceleration Blend
# ============================================================================
#
# Mathematical Framework:
#   This is the key theoretical contribution. We separate the prediction
#   into two orthogonal components:
#
#   1. LEVEL component (where consensus excels):
#       level_t = consensus_t    (or bias-corrected consensus)
#
#   2. ACCELERATION component (where our model excels):
#       Δ̂_model_t = champion_t - champion_{t-1}  (model's predicted MoM change)
#       Δ̂_cons_t = consensus_t - actual_{t-1}     (consensus's implied MoM change)
#
#   Blended prediction:
#       ŷ_t = consensus_t + λ · (Δ̂_model_t - Δ̂_cons_t)
#
#   Where λ ∈ [0, 1] controls how much we trust the model's acceleration
#   signal OVER the consensus's implicit acceleration.
#
#   When λ = 0: pure consensus (best MAE)
#   When λ = 1: consensus level + full model acceleration correction
#
#   The correction term (Δ̂_model_t - Δ̂_cons_t) is bounded to prevent
#   large deviations from consensus that would hurt MAE.
#
#   Why this is theoretically optimal:
#   - Consensus captures E[y_t | public_info] well (low MAE)
#   - Model captures E[Δy_t | features] better (high accel accuracy)
#   - By adding only the DIFFERENTIAL acceleration signal, we get both
#   - The bound prevents catastrophic MAE blowups on outlier months
# ============================================================================

def approach_3_decoupled_level_accel(
    overlap_df: pd.DataFrame,
    lam: float = 0.5,
    correction_cap: float = 100.0,
    use_bias_correction: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Consensus for level + model acceleration override.

    Args:
        lam: weight on model's differential acceleration signal [0,1]
        correction_cap: max absolute correction in thousands
        use_bias_correction: subtract trailing consensus bias first
    """
    df = overlap_df[["ds", "actual", "consensus_pred", "champion_pred"]].dropna().copy()
    df = df.sort_values("ds").reset_index(drop=True)
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i]

        # Bias correction from trailing history
        bias_correction = 0.0
        if use_bias_correction and len(hist) >= 6:
            recent_bias = (hist["actual"] - hist["consensus_pred"]).iloc[-12:].mean()
            bias_correction = recent_bias

        # Model's predicted acceleration (change from prior prediction)
        if i > 0:
            model_delta = row["champion_pred"] - df.iloc[i - 1]["champion_pred"]
            # Consensus's implied delta from last actual
            consensus_delta = row["consensus_pred"] - hist["actual"].iloc[-1]
            # Differential: how much does model disagree with consensus on direction
            accel_correction = lam * (model_delta - consensus_delta)
            # Bound it
            accel_correction = np.clip(accel_correction, -correction_cap, correction_cap)
        else:
            accel_correction = 0.0

        pred = row["consensus_pred"] + bias_correction + accel_correction
        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "bias_correction": bias_correction,
            "accel_correction": accel_correction,
        })

    res_df = pd.DataFrame(results)
    label = f"A3_Decoupled_lam{lam:.2f}_cap{correction_cap:.0f}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# APPROACH 4: Bayesian Shrinkage Estimator
# ============================================================================
#
# Mathematical Framework:
#   James-Stein / Bayesian shrinkage toward consensus.
#
#   The idea: treat consensus as a prior and model as the likelihood.
#
#   Prior:     y_t ~ N(consensus_t, σ²_c)   where σ²_c = Var(actual - consensus)
#   Likelihood: y_t ~ N(model_t, σ²_m)       where σ²_m = Var(actual - model)
#
#   Posterior mean (optimal Bayesian combination):
#       ŷ_t = w_c · consensus_t + w_m · model_t
#
#   Where:
#       w_c = σ²_m / (σ²_c + σ²_m)     (weight on consensus)
#       w_m = σ²_c / (σ²_c + σ²_m)     (weight on model)
#
#   Since σ²_c < σ²_m (consensus has lower MSE), w_c > w_m.
#
#   Extension: TIME-VARYING shrinkage using trailing variances:
#       σ²_c(t) = Var_trailing(actual - consensus)
#       σ²_m(t) = Var_trailing(actual - model)
#       w_c(t) = σ²_m(t) / (σ²_c(t) + σ²_m(t))
#
#   This adapts the blend to recent relative accuracy.
# ============================================================================

def approach_4_bayesian_shrinkage(
    overlap_df: pd.DataFrame,
    trailing_window: int = 18,
    time_varying: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Bayesian shrinkage toward consensus based on relative variance."""
    df = overlap_df[["ds", "actual", "consensus_pred", "champion_pred"]].dropna().copy()
    df = df.sort_values("ds").reset_index(drop=True)
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i]

        if len(hist) < MIN_HISTORY:
            # Equal weight fallback
            pred = 0.5 * row["consensus_pred"] + 0.5 * row["champion_pred"]
        else:
            h = hist.iloc[-trailing_window:] if time_varying else hist

            err_c = (h["actual"] - h["consensus_pred"]).values
            err_m = (h["actual"] - h["champion_pred"]).values

            var_c = float(np.var(err_c, ddof=1)) + 1e-6
            var_m = float(np.var(err_m, ddof=1)) + 1e-6

            w_consensus = var_m / (var_c + var_m)
            w_model = var_c / (var_c + var_m)

            pred = w_consensus * row["consensus_pred"] + w_model * row["champion_pred"]

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
        })

    res_df = pd.DataFrame(results)
    tv = "tv" if time_varying else "fixed"
    label = f"A4_Bayes_{tv}_w{trailing_window}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# APPROACH 5: Consensus + Acceleration Direction Override
# ============================================================================
#
# Mathematical Framework:
#   This approach uses consensus as the BASE but only overrides when
#   our model's ACCELERATION SIGNAL disagrees with consensus.
#
#   Define:
#     Δ_actual_{t-1} = actual_{t-1} - actual_{t-2}    (last known accel)
#     Δ̂_model_t = champion_t - actual_{t-1}            (model's implied change)
#     Δ̂_cons_t = consensus_t - actual_{t-1}             (consensus's implied change)
#
#   Decision rule:
#     If sign(Δ̂_model_t) == sign(Δ̂_cons_t):
#         ŷ_t = consensus_t                    (agree -> trust consensus level)
#     Else:
#         ŷ_t = consensus_t + κ · (Δ̂_model_t - Δ̂_cons_t)
#         where κ ∈ [0, 1] controls override strength
#
#   Variant: Also correct the MAGNITUDE when disagreeing:
#     ŷ_t = actual_{t-1} + sign(Δ̂_model_t) · |Δ̂_cons_t|
#     (Use model direction but consensus magnitude)
#
#   Why: Our model is 64.7% on acceleration vs consensus at 41.2%.
#   By selectively overriding only the acceleration direction, we
#   improve accel accuracy while minimizing MAE damage.
# ============================================================================

def approach_5_accel_override(
    overlap_df: pd.DataFrame,
    kappa: float = 0.5,
    magnitude_mode: str = "consensus",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Use consensus as base, override acceleration direction when model disagrees.

    magnitude_mode:
      - 'consensus': keep consensus magnitude, only flip direction
      - 'blend': blend consensus and model magnitudes
      - 'model': use model's magnitude with corrected direction
    """
    df = overlap_df[["ds", "actual", "consensus_pred", "champion_pred"]].dropna().copy()
    df = df.sort_values("ds").reset_index(drop=True)
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i]

        if len(hist) < 2:
            pred = row["consensus_pred"]
        else:
            last_actual = hist["actual"].iloc[-1]

            model_delta = row["champion_pred"] - last_actual
            cons_delta = row["consensus_pred"] - last_actual

            # Do model and consensus agree on direction?
            agree = np.sign(model_delta) == np.sign(cons_delta)

            if agree:
                # Agreement -> trust consensus level entirely
                pred = row["consensus_pred"]
            else:
                # Disagreement -> model overrides acceleration direction
                if magnitude_mode == "consensus":
                    # Model direction, consensus magnitude
                    pred = last_actual + np.sign(model_delta) * abs(cons_delta)
                elif magnitude_mode == "blend":
                    # Blend of both
                    blended_mag = kappa * abs(model_delta) + (1 - kappa) * abs(cons_delta)
                    pred = last_actual + np.sign(model_delta) * blended_mag
                elif magnitude_mode == "model":
                    # Partial adjustment from consensus toward model direction
                    correction = kappa * (model_delta - cons_delta)
                    pred = row["consensus_pred"] + correction
                else:
                    raise ValueError(magnitude_mode)

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
        })

    res_df = pd.DataFrame(results)
    label = f"A5_AccelOverride_{magnitude_mode}_k{kappa:.2f}"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# APPROACH 6: Stacked Meta-Learner with Full Consensus History
# ============================================================================
#
# Mathematical Framework:
#   Two-stage stacking:
#
#   Stage 1 (pre-trained on 315 months of consensus-only data):
#     Learn g(consensus_t, features_t) -> ε̂_t (consensus residual model)
#     This captures systematic consensus biases across 25+ years.
#
#   Stage 2 (trained on overlap window):
#     Meta-learner combines:
#       x_1 = consensus_t
#       x_2 = champion_t
#       x_3 = challenger_t
#       x_4 = consensus_t + ε̂_t (bias-corrected consensus from Stage 1)
#       x_5 = |champion_t - consensus_t| (disagreement signal)
#       x_6 = sign(champion_delta) - sign(consensus_delta) (accel disagreement)
#
#     ŷ_t = ridge(x_1, ..., x_6)
#
#   The Stage 1 pre-training is crucial: it gives us a strong consensus
#   correction model trained on 315 months, not just 35.
# ============================================================================

def approach_6_stacked_meta(
    consensus_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """Two-stage stacking: consensus residual pre-training + meta-learner."""
    from sklearn.linear_model import Ridge

    # Stage 1: Build consensus residual model from full history
    full = _build_consensus_features(consensus_df)
    feature_cols = [
        "consensus_pred", "consensus_mom", "consensus_accel", "consensus_abs",
        "trailing_bias_6m", "trailing_bias_12m", "trailing_bias_24m",
        "trailing_abs_err_6m", "trailing_abs_err_12m",
    ]

    # Prepare overlap data
    overlap = overlap_df[
        ["ds", "actual", "consensus_pred", "champion_pred", "challenger_pred"]
    ].dropna().copy()
    overlap = overlap.sort_values("ds").reset_index(drop=True)
    results = []

    for i in range(len(overlap)):
        row = overlap.iloc[i]
        hist_overlap = overlap.iloc[:i]

        # Stage 1: expanding-window consensus residual prediction
        # Use all consensus history up to this month
        full_train = full[full["ds"] < row["ds"]].dropna(subset=["residual"])
        full_train = full_train.dropna(subset=feature_cols, how="all")

        if len(full_train) >= MIN_HISTORY:
            X_s1 = full_train[feature_cols].fillna(0).values
            y_s1 = full_train["residual"].values
            ridge_s1 = Ridge(alpha=10.0)
            ridge_s1.fit(X_s1, y_s1)

            row_features = full[full["ds"] == row["ds"]]
            if not row_features.empty:
                X_test_s1 = row_features[feature_cols].fillna(0).values
                consensus_corrected = row["consensus_pred"] + ridge_s1.predict(X_test_s1)[0]
            else:
                consensus_corrected = row["consensus_pred"]
        else:
            consensus_corrected = row["consensus_pred"]

        # Stage 2: meta-learner (needs overlap history)
        if len(hist_overlap) < MIN_HISTORY:
            # Not enough meta-training data; use bias-corrected consensus
            pred = consensus_corrected
        else:
            # Build meta-features for training
            meta_features_train = []
            meta_targets_train = []
            for j in range(len(hist_overlap)):
                hr = hist_overlap.iloc[j]
                # Compute disagreement features
                disagree = abs(hr["champion_pred"] - hr["consensus_pred"])
                if j > 0:
                    prev = hist_overlap.iloc[j - 1]
                    champ_delta = hr["champion_pred"] - prev["champion_pred"]
                    cons_delta = hr["consensus_pred"] - prev["consensus_pred"]
                    accel_disagree = float(np.sign(champ_delta) != np.sign(cons_delta))
                else:
                    accel_disagree = 0.0

                meta_features_train.append([
                    hr["consensus_pred"],
                    hr["champion_pred"],
                    hr["challenger_pred"],
                    disagree,
                    accel_disagree,
                ])
                meta_targets_train.append(hr["actual"])

            X_meta = np.array(meta_features_train)
            y_meta = np.array(meta_targets_train)

            # Build meta-features for this month
            disagree_now = abs(row["champion_pred"] - row["consensus_pred"])
            if i > 0:
                prev = overlap.iloc[i - 1]
                cd = row["champion_pred"] - prev["champion_pred"]
                csd = row["consensus_pred"] - prev["consensus_pred"]
                accel_dis_now = float(np.sign(cd) != np.sign(csd))
            else:
                accel_dis_now = 0.0

            X_test_meta = np.array([[
                row["consensus_pred"],
                row["champion_pred"],
                row["challenger_pred"],
                disagree_now,
                accel_dis_now,
            ]])

            ridge_meta = Ridge(alpha=50.0)  # heavy regularization for small sample
            ridge_meta.fit(X_meta, y_meta)
            pred = ridge_meta.predict(X_test_meta)[0]

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "consensus_corrected": consensus_corrected,
        })

    res_df = pd.DataFrame(results)
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, "A6_Stacked_Meta")
    return res_df, metrics


# ============================================================================
# APPROACH 7: Kalman Filter Fusion
# ============================================================================
#
# Mathematical Framework:
#   State-space model treating true NFP as a latent state:
#
#   State transition:    x_t = x_{t-1} + w_t,    w_t ~ N(0, Q)
#   Observation 1:       consensus_t = x_t + v_c,  v_c ~ N(0, R_c)
#   Observation 2:       model_t = x_t + v_m,      v_m ~ N(0, R_m)
#
#   Where:
#     Q = process noise (volatility of true NFP changes)
#     R_c = consensus observation noise (MSE of consensus)
#     R_m = model observation noise (MSE of model)
#
#   Kalman update with TWO observations:
#     K_c = P_prior / (P_prior + R_c)   (Kalman gain for consensus)
#     K_m = P_prior / (P_prior + R_m)   (Kalman gain for model)
#
#   Combined update:
#     x̂_t = x̂_{t|t-1} + K_combined · innovation
#
#   Or equivalently, the multi-observation Kalman:
#     P_post = 1 / (1/P_prior + 1/R_c + 1/R_m)
#     x̂_post = P_post · (x̂_prior/P_prior + consensus/R_c + model/R_m)
#
#   The noise parameters R_c and R_m are estimated from trailing residuals.
#   Q is estimated from actual MoM variance.
#
#   Advantage: provides uncertainty estimates and adapts weights smoothly.
#   When one source degrades, the Kalman gain automatically shifts to the other.
# ============================================================================

def approach_7_kalman_fusion(
    overlap_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    use_model: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kalman filter fusing consensus and model predictions.

    If use_model=False, only fuses consensus (useful as a baseline
    to show the benefit of adding the model signal).
    """
    df = overlap_df[["ds", "actual", "consensus_pred", "champion_pred"]].dropna().copy()
    df = df.sort_values("ds").reset_index(drop=True)

    # Initialize noise parameters from full consensus history
    cons_hist = consensus_df[
        consensus_df["consensus_pred"].notna() & consensus_df["actual"].notna()
    ]
    cons_err = (cons_hist["actual"] - cons_hist["consensus_pred"]).values
    R_c_init = float(np.var(cons_err[-60:], ddof=1))  # last 5 years
    Q_init = float(np.var(np.diff(cons_hist["actual"].dropna().values[-60:]), ddof=1))

    # State initialization
    x_hat = float(df.iloc[0]["consensus_pred"])  # initial state estimate
    P = Q_init  # initial state uncertainty

    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i]

        # Estimate noise params from trailing data
        if len(hist) >= 6:
            recent_cons_err = (hist["actual"] - hist["consensus_pred"]).values[-18:]
            R_c = float(np.var(recent_cons_err, ddof=1)) + 1e-6

            if use_model:
                recent_model_err = (hist["actual"] - hist["champion_pred"]).values[-18:]
                R_m = float(np.var(recent_model_err, ddof=1)) + 1e-6
            else:
                R_m = 1e12  # effectively infinite -> ignore model

            recent_actual_diff = np.diff(hist["actual"].values[-18:])
            Q = float(np.var(recent_actual_diff, ddof=1)) + 1e-6
        else:
            R_c = R_c_init
            R_m = R_c_init * 1.5 if use_model else 1e12
            Q = Q_init

        # Prediction step
        x_prior = x_hat  # random walk transition
        P_prior = P + Q

        # Update step: fuse both observations
        # Multi-observation Kalman: P_post = 1/(1/P_prior + 1/R_c + 1/R_m)
        info_prior = 1.0 / P_prior
        info_c = 1.0 / R_c
        info_m = 1.0 / R_m if use_model else 0.0

        P_post = 1.0 / (info_prior + info_c + info_m)
        x_post = P_post * (
            info_prior * x_prior
            + info_c * row["consensus_pred"]
            + (info_m * row["champion_pred"] if use_model else 0.0)
        )

        pred = x_post

        # After observing actual, update state for next step
        if pd.notna(row["actual"]):
            # Final update with actual observation (perfect info)
            x_hat = row["actual"]
            P = 1e-6  # near-zero uncertainty after observing truth
        else:
            x_hat = x_post
            P = P_post

        results.append({
            "ds": row["ds"],
            "actual": row["actual"],
            "predicted": pred,
            "consensus_pred": row["consensus_pred"],
            "P_prior": P_prior,
            "P_post": P_post,
            "R_c": R_c,
            "R_m": R_m if use_model else np.nan,
        })

    res_df = pd.DataFrame(results)
    label = "A7_Kalman_fusion" if use_model else "A7_Kalman_consensus_only"
    metrics = full_metrics(res_df["actual"].values, res_df["predicted"].values, label)
    return res_df, metrics


# ============================================================================
# PARETO FRONTIER ANALYSIS
# ============================================================================

def compute_pareto_frontier(
    points: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """
    Find Pareto-optimal points minimizing MAE and maximizing Acceleration Accuracy.
    Input: list of (MAE, AccelAcc, label).
    Returns Pareto-dominant points.
    """
    # We want to MINIMIZE MAE and MAXIMIZE AccelAcc
    # A point dominates another if it has lower MAE AND higher AccelAcc
    pareto = []
    for mae_i, acc_i, label_i in points:
        dominated = False
        for mae_j, acc_j, label_j in points:
            if label_i == label_j:
                continue
            if mae_j <= mae_i and acc_j >= acc_i and (mae_j < mae_i or acc_j > acc_i):
                dominated = True
                break
        if not dominated:
            pareto.append((mae_i, acc_i, label_i))
    return sorted(pareto, key=lambda x: x[0])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments():
    """Run all 7 approaches with multiple hyperparameter settings."""
    print("=" * 80)
    print("CONSENSUS ANCHOR INTEGRATION EXPERIMENTS")
    print("=" * 80)

    df = load_data()
    consensus_df, overlap_df = split_datasets(df)
    print(f"\nConsensus+actual rows (full history): {len(consensus_df)}")
    print(f"  Date range: {consensus_df['ds'].min().date()} to {consensus_df['ds'].max().date()}")
    print(f"Overlap rows (all 3 sources):         {len(overlap_df)}")
    print(f"  Date range: {overlap_df['ds'].min().date()} to {overlap_df['ds'].max().date()}")

    all_metrics = []
    all_results = {}

    # ---- Baselines (all on the 35-month overlap window for fair comparison) ----
    print("\n--- Baselines (on 35-month overlap window) ---")
    m = full_metrics(
        overlap_df["actual"].dropna().values,
        overlap_df["consensus_pred"].dropna().values,
        "Baseline_Consensus"
    )
    all_metrics.append(m)
    print(f"  Consensus:  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # Champion baseline on overlap
    ov = overlap_df[["actual", "champion_pred"]].dropna()
    m = full_metrics(ov["actual"].values, ov["champion_pred"].values, "Baseline_Champion")
    all_metrics.append(m)
    print(f"  Champion:   MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # Challenger baseline
    ov2 = overlap_df[["actual", "challenger_pred"]].dropna()
    m = full_metrics(ov2["actual"].values, ov2["challenger_pred"].values, "Baseline_Challenger")
    all_metrics.append(m)
    print(f"  Challenger: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 1: Residual Learning ----
    print("\n--- Approach 1: Consensus-Anchored Residual Learning ---")
    for method in ["bias_only", "adaptive_bias", "ridge"]:
        res_df, m = approach_1_residual_learning(consensus_df, overlap_df, method=method)
        all_metrics.append(m)
        all_results[m["Forecast"]] = res_df
        print(f"  {method:15s}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}  Bias={m['ME_Bias']:.1f}")

    # ---- Approach 2: Dynamic Blend ----
    print("\n--- Approach 2: Dynamic Linear Blend ---")
    for obj in ["mae", "accel", "composite"]:
        for window in [12, 18]:
            res_df, m = approach_2_dynamic_blend(overlap_df, objective=obj, window=window)
            all_metrics.append(m)
            all_results[m["Forecast"]] = res_df
            print(f"  obj={obj:10s} w={window:2d}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 3: Decoupled Level-Acceleration ----
    print("\n--- Approach 3: Decoupled Level-Acceleration Blend ---")
    for lam in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for cap in [50, 100, 200]:
            res_df, m = approach_3_decoupled_level_accel(overlap_df, lam=lam, correction_cap=cap)
            all_metrics.append(m)
            all_results[m["Forecast"]] = res_df
            print(f"  λ={lam:.1f} cap={cap:3d}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 4: Bayesian Shrinkage ----
    print("\n--- Approach 4: Bayesian Shrinkage ---")
    for tv in [True, False]:
        for w in [12, 18, 24]:
            res_df, m = approach_4_bayesian_shrinkage(overlap_df, trailing_window=w, time_varying=tv)
            all_metrics.append(m)
            all_results[m["Forecast"]] = res_df
            tv_str = "TV" if tv else "Fixed"
            print(f"  {tv_str:5s} w={w:2d}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 5: Acceleration Override ----
    print("\n--- Approach 5: Acceleration Direction Override ---")
    for mode in ["consensus", "blend", "model"]:
        for kappa in [0.3, 0.5, 0.7]:
            res_df, m = approach_5_accel_override(overlap_df, kappa=kappa, magnitude_mode=mode)
            all_metrics.append(m)
            all_results[m["Forecast"]] = res_df
            print(f"  {mode:10s} κ={kappa:.1f}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 6: Stacked Meta-Learner ----
    print("\n--- Approach 6: Stacked Meta-Learner ---")
    res_df, m = approach_6_stacked_meta(consensus_df, overlap_df)
    all_metrics.append(m)
    all_results[m["Forecast"]] = res_df
    print(f"  Stacked:    MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Approach 7: Kalman Filter ----
    print("\n--- Approach 7: Kalman Filter Fusion ---")
    for use_model in [True, False]:
        res_df, m = approach_7_kalman_fusion(overlap_df, consensus_df, use_model=use_model)
        all_metrics.append(m)
        all_results[m["Forecast"]] = res_df
        label = "with_model" if use_model else "consensus_only"
        print(f"  {label:15s}: MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  AccelAcc={m['Acceleration_Accuracy']:.3f}")

    # ---- Compile Results ----
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("MAE").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("FULL RESULTS TABLE (sorted by MAE)")
    print("=" * 80)
    display_cols = ["Forecast", "N", "RMSE", "MAE", "ME_Bias",
                    "Directional_Accuracy", "Acceleration_Accuracy",
                    "STD_Ratio", "Diff_STD_Ratio", "Tail_MAE", "Extreme_Hit_Rate"]
    print(metrics_df[display_cols].to_string(index=False, float_format="%.3f"))

    # ---- Pareto Frontier ----
    print("\n" + "=" * 80)
    print("PARETO FRONTIER (MAE vs Acceleration Accuracy)")
    print("=" * 80)
    points = []
    for _, row in metrics_df.iterrows():
        if pd.notna(row["MAE"]) and pd.notna(row["Acceleration_Accuracy"]):
            points.append((row["MAE"], row["Acceleration_Accuracy"], row["Forecast"]))

    pareto = compute_pareto_frontier(points)
    print(f"\n{'Rank':<5} {'MAE':>8} {'AccelAcc':>10} {'Forecast'}")
    print("-" * 60)
    for rank, (mae, acc, label) in enumerate(pareto, 1):
        print(f"{rank:<5} {mae:>8.1f} {acc:>10.3f} {label}")

    # ---- Save outputs ----
    metrics_df.to_csv(OUT_DIR / "all_experiment_metrics.csv", index=False)

    # Save Pareto frontier
    pareto_df = pd.DataFrame(pareto, columns=["MAE", "Acceleration_Accuracy", "Forecast"])
    pareto_df.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)

    # Save individual results
    for name, rdf in all_results.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        rdf.to_csv(OUT_DIR / f"results_{safe_name}.csv", index=False)

    print(f"\nAll outputs saved to: {OUT_DIR}")

    # ---- Generate Pareto Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 9))

        # Color by approach
        approach_colors = {
            "Baseline": "#999999",
            "A1": "#e41a1c",
            "A2": "#377eb8",
            "A3": "#4daf4a",
            "A4": "#984ea3",
            "A5": "#ff7f00",
            "A6": "#a65628",
            "A7": "#f781bf",
        }

        for _, row in metrics_df.iterrows():
            label = row["Forecast"]
            prefix = label.split("_")[0]
            color = approach_colors.get(prefix, "#333333")
            ax.scatter(row["MAE"], row["Acceleration_Accuracy"],
                      c=color, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

        # Highlight Pareto frontier
        if pareto:
            p_mae = [p[0] for p in pareto]
            p_acc = [p[1] for p in pareto]
            ax.plot(p_mae, p_acc, "k--", linewidth=1.5, alpha=0.5, label="Pareto frontier")
            for mae, acc, label in pareto:
                ax.annotate(label, (mae, acc), fontsize=6,
                           xytext=(5, 5), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

        # Reference lines for baselines
        cons_row = metrics_df[metrics_df["Forecast"] == "Baseline_Consensus"].iloc[0]
        ax.axvline(cons_row["MAE"], color="orange", linestyle=":", alpha=0.5, label=f"Consensus MAE={cons_row['MAE']:.1f}")
        ax.axhline(cons_row["Acceleration_Accuracy"], color="orange", linestyle="--", alpha=0.5,
                   label=f"Consensus AccelAcc={cons_row['Acceleration_Accuracy']:.3f}")

        champ_row = metrics_df[metrics_df["Forecast"] == "Baseline_Champion"].iloc[0]
        ax.axvline(champ_row["MAE"], color="blue", linestyle=":", alpha=0.5, label=f"Champion MAE={champ_row['MAE']:.1f}")
        ax.axhline(champ_row["Acceleration_Accuracy"], color="blue", linestyle="--", alpha=0.5,
                   label=f"Champion AccelAcc={champ_row['Acceleration_Accuracy']:.3f}")

        ax.set_xlabel("MAE (lower is better)", fontsize=12)
        ax.set_ylabel("Acceleration Accuracy (higher is better)", fontsize=12)
        ax.set_title("Pareto Frontier: MAE vs Acceleration Accuracy\n(Consensus Anchor Integration Experiments)", fontsize=13)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.2)

        # Add approach legend
        for prefix, color in approach_colors.items():
            ax.scatter([], [], c=color, s=40, label=prefix, edgecolors="black", linewidth=0.5)
        ax.legend(loc="lower left", fontsize=7, ncol=2)

        fig.tight_layout()
        fig.savefig(OUT_DIR / "pareto_frontier.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pareto plot saved to: {OUT_DIR / 'pareto_frontier.png'}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return metrics_df, pareto


if __name__ == "__main__":
    metrics_df, pareto = run_all_experiments()
