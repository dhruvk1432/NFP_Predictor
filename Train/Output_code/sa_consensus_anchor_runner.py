"""SA Kalman fusion runner — the SA-branch champion-challenger.

Fuses the SA-revised NFP MoM forecast across N independent channels via a
walk-forward, multi-observation Kalman filter in information-filter form.
Default channel inventory (any can be turned off via CLI):

    consensus           — NFP_Consensus_Mean (PIT) from master snapshots
    economist_panel     — auto-selected top-N economist ensemble
    dfm_factor          — DynamicFactor over labor observables
    bvar_glp            — Minnesota-prior BVAR (closed-form NIW posterior)
    sa_state_space      — UnobservedComponents (local-linear-trend + seasonal)

The math is the textbook scalar information-filter update — same form as the
production NSA runner's ``kalman_fusion`` (Train/Output_code/consensus_anchor_runner.py),
but factored to take a generic list of channels rather than the NSA-specific
``champion_pred`` / ``nsa_pred`` / ``panel_consensus_mean`` hardcoded names.

PIT contract:
* Each channel's predictions.csv carries ``ds`` and ``predicted_mom`` produced
  walk-forward by a PIT-audited sidecar (Phase 1/2 audits passed).
* At step t the filter uses only actual rows with
  ``actual_available_date < target_release_date(t)`` to estimate per-channel R,
  process noise Q, and state resets. Chronological ``ds <= t-1`` is not enough
  for revised NFP because the prior month's revised actual is commonly released
  on the same day as the current forecast.
* COVID months (2020-03..2020-12) are excluded from R/Q estimation only
  (winsorization-friendly), exactly mirroring the production NSA runner.
* Output lands under ``_output_sa_challenger/kalman_fusion/`` — fully isolated
  from the live ``_output/consensus_anchor/`` production tree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import OUTPUT_DIR  # noqa: E402
from Train.Output_code.consensus_anchor_runner import (  # noqa: E402
    _load_consensus_pit,
    full_metrics,
)
from utils.transforms import is_covid_month, winsorize_covid_period  # noqa: E402


SA_CHALLENGER_ROOT = PROJECT_ROOT / "_output_sa_challenger"
DEFAULT_RUN_ID = "phase1_final"

# The default channel inventory. Each entry: (channel_id, sidecar_dir_name,
# predictions_filename, prediction_column). predictions.csv lives at
# _output/sidecars/sa/<run_id>/<sidecar_dir_name>/predictions.csv.
#
# Channels match the plan's 5-channel fusion (Notes/SA challenger plan):
#   1. SA-LightGBM (the production SA model, walk-forward sidecar-shaped).
#   2. economist_panel (note: known PIT leakage; excluded from default).
#   3. dfm_factor.
#   4. bvar_glp.
#   5. sa_state_space (note: weak MoM signal; excluded from default; reachable
#      via --channels override).
DEFAULT_CHANNELS: List[Tuple[str, str, str]] = [
    ("sa_lightgbm",            "sa_lightgbm",            "predicted_mom"),
    ("consensus_median",       "consensus_median",       "predicted_mom"),
    ("dynamic_top6_k12_cov90", "dynamic_top6_k12_cov90", "predicted_mom"),
    ("dfm_factor",             "dfm_factor",             "predicted_mom"),
    ("bvar_glp",               "bvar_glp",               "predicted_mom"),
]

# Fully-extended channel list (for --channels overrides). Use with care:
#   - sa_state_space alone on MoM is weak (~500 MAE).
EXTENDED_CHANNELS: List[Tuple[str, str, str]] = [
    ("sa_lightgbm",            "sa_lightgbm",            "predicted_mom"),
    ("consensus_median",       "consensus_median",       "predicted_mom"),
    ("dynamic_top6_k12_cov90", "dynamic_top6_k12_cov90", "predicted_mom"),
    ("dfm_factor",             "dfm_factor",             "predicted_mom"),
    ("bvar_glp",               "bvar_glp",               "predicted_mom"),
    ("sa_state_space",         "sa_state_space",         "predicted_mom"),
    ("economist_panel",        "economist_panel",        "predicted_mom"),
]


@dataclass
class FusionConfig:
    run_id: str = DEFAULT_RUN_ID
    target_type: str = "sa"
    target_source: str = "revised"
    output_dir: Path = SA_CHALLENGER_ROOT / "kalman_fusion"
    trailing_window: int = 18
    channels: List[Tuple[str, str, str]] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    channel_weight_scales: Dict[str, float] = field(default_factory=dict)
    include_consensus: bool = True
    use_winsorized_actuals: bool = True
    max_precision_share_per_channel: float = 0.65
    # Per-channel bias correction: when True, at step t subtract the trailing
    # COVID-clean mean error (actual - pred) from the channel's R estimate base
    # and ADD it to the raw observation, so the fused state is unbiased even
    # when individual channels carry persistent biases (e.g., DFM −18, etc.).
    bias_correct: bool = True
    bias_correct_min_history: int = 6
    # Fail-safe: when the strongest channel is missing at row t, fall back to
    # a designated safer channel rather than letting the noisier channels
    # average together into a confident-but-wrong forecast. Default targets
    # the dynamic_top6 → consensus_median substitution, which is the planned
    # behavior during economist-panel-data outages.
    fallback_enabled: bool = True
    fallback_primary_channel: str = "dynamic_top6_k12_cov90"
    fallback_to_channel: str = "consensus_median"


# --------------------------------------------------------------------------- #
# Channel loaders
# --------------------------------------------------------------------------- #

def _load_sa_actuals(use_winsorized: bool = True) -> pd.DataFrame:
    """Load (ds, actual_mom) from y_sa_revised, optionally COVID-winsorized
    using the same window that the rest of the pipeline winsorizes.
    """
    path = PROJECT_ROOT / "data" / "NFP_target" / "y_sa_revised.parquet"
    raw = pd.read_parquet(path)
    keep = ["ds", "y_mom"]
    for col in ("release_date", "operational_available_date"):
        if col in raw.columns:
            keep.append(col)
    df = raw[keep].copy()
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["y_mom"]).sort_values("ds").reset_index(drop=True)
    if use_winsorized:
        idx = df.set_index("ds")
        idx["y_mom"] = winsorize_covid_period(idx[["y_mom"]])["y_mom"]
        df = idx.reset_index()
    df = df.rename(columns={"y_mom": "actual_mom"})
    if "release_date" in df.columns:
        df["target_release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "operational_available_date" in df.columns:
        df["actual_available_date"] = pd.to_datetime(
            df["operational_available_date"],
            errors="coerce",
        )
    df = df.drop(columns=["release_date", "operational_available_date"], errors="ignore")
    return df


def _row_cutoff(row: pd.Series) -> pd.Timestamp:
    cutoff = row.get("target_release_date", pd.NaT)
    if pd.notna(cutoff):
        return pd.Timestamp(cutoff)
    return pd.Timestamp(row["ds"])


def _actual_history_available_before(hist: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if hist.empty:
        return hist.copy()
    out = hist[hist["actual_mom"].notna()].copy()
    if out.empty:
        return out
    out = out[pd.to_datetime(out["ds"]) < pd.Timestamp(row["ds"])]
    if "actual_available_date" in out.columns:
        cutoff = _row_cutoff(row)
        avail = pd.to_datetime(out["actual_available_date"], errors="coerce")
        out = out[avail.notna() & (avail < cutoff)]
    return out


def _month_gap(later, earlier) -> int:
    if pd.isna(later) or pd.isna(earlier):
        return 1
    later_p = pd.Timestamp(later).to_period("M")
    earlier_p = pd.Timestamp(earlier).to_period("M")
    return max(1, (later_p.year - earlier_p.year) * 12 + (later_p.month - earlier_p.month))


def _load_sidecar_channel(run_id: str, sidecar_name: str, pred_col: str) -> pd.DataFrame:
    """Read predictions.csv for one sidecar and return (ds, <channel>_pred)."""
    path = (
        OUTPUT_DIR
        / "sidecars"
        / "sa"
        / run_id
        / sidecar_name
        / "predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Sidecar artifact missing: {path}")
    df = pd.read_csv(path, parse_dates=["ds"])
    if pred_col not in df.columns:
        raise KeyError(f"{path} missing required column {pred_col}")
    out = df[["ds", pred_col]].copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.to_period("M").dt.to_timestamp()
    out = out.dropna(subset=[pred_col])
    out = out.rename(columns={pred_col: f"{sidecar_name}_pred"})
    return out


def build_fusion_frame(cfg: FusionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Assemble the per-month wide-form frame with one column per channel.

    Returns:
        df: per-month frame with ds, actual_mom, and one *_pred column per
            available channel; ordered by ds ascending.
        consensus_pit: the full consensus PIT frame (for R-init priors).
        active_channels: list of channel_ids actually present.
    """
    actuals = _load_sa_actuals(use_winsorized=cfg.use_winsorized_actuals)
    frame = actuals.copy()

    active: List[str] = []
    consensus_pit = pd.DataFrame()

    if cfg.include_consensus:
        consensus_pit = _load_consensus_pit(
            target_type=cfg.target_type, target_source=cfg.target_source,
        )
        consensus_pit["ds"] = pd.to_datetime(consensus_pit["ds"]).dt.to_period("M").dt.to_timestamp()
        # Attach as a fusion channel under the name 'consensus'.
        c = consensus_pit.rename(columns={"consensus_pred": "consensus_pred"}).copy()
        frame = frame.merge(c, on="ds", how="left")
        active.append("consensus")

    for channel_id, sidecar_name, pred_col in cfg.channels:
        try:
            c = _load_sidecar_channel(cfg.run_id, sidecar_name, pred_col)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  (skip) {channel_id}: {exc}")
            continue
        frame = frame.merge(
            c.rename(columns={f"{sidecar_name}_pred": f"{channel_id}_pred"}),
            on="ds", how="left",
        )
        active.append(channel_id)

    # Require at least one channel observation per scored row.
    pred_cols = [f"{c}_pred" for c in active]
    frame = frame.sort_values("ds").reset_index(drop=True)
    has_any_pred = frame[pred_cols].notna().any(axis=1) if pred_cols else pd.Series(False, index=frame.index)
    frame = frame[has_any_pred].reset_index(drop=True)

    return frame, consensus_pit, active


# --------------------------------------------------------------------------- #
# Multi-observation Kalman fusion (information-filter form)
# --------------------------------------------------------------------------- #

def _trailing_r_and_bias_for_channel(
    hist_clean: pd.DataFrame,
    pred_col: str,
    trailing_window: int,
    min_history_for_bias: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Trailing-window variance + mean error (bias) for one channel.

    R = var(actual - pred) over last trailing_window COVID-clean rows.
    bias = mean(actual - pred) over the same window (added back to forecasts
    for bias correction).

    Returns (R, bias). Either entry may be None when there isn't enough
    history — caller falls back to most-recent or skips bias correction.
    """
    pres = hist_clean[hist_clean[pred_col].notna()]
    if len(pres) < 4:
        return None, None
    err = (pres["actual_mom"] - pres[pred_col]).values[-trailing_window:]
    if len(err) < 2:
        return None, None
    R = float(np.var(err, ddof=1)) + 1e-6
    bias = float(np.mean(err)) if len(err) >= int(min_history_for_bias) else None
    return R, bias


def _trailing_r_for_channel(
    hist_clean: pd.DataFrame, pred_col: str, trailing_window: int,
) -> Optional[float]:
    R, _ = _trailing_r_and_bias_for_channel(
        hist_clean, pred_col, trailing_window, min_history_for_bias=10**9,
    )
    return R


def _trailing_q(hist_clean: pd.DataFrame, trailing_window: int) -> Optional[float]:
    """Trailing-window process noise — var of diff(actual_mom)."""
    actuals = hist_clean["actual_mom"].values[-trailing_window:]
    if len(actuals) < 3:
        return None
    return float(np.var(np.diff(actuals), ddof=1)) + 1e-6


def run_sa_kalman_fusion(
    cfg: FusionConfig,
) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """Walk-forward multi-observation Kalman fusion. Returns (results, metrics, active_channels).
    """
    frame, consensus_pit, active = build_fusion_frame(cfg)
    if not active:
        raise RuntimeError("No fusion channels available; cannot run fusion.")

    pred_cols = [f"{c}_pred" for c in active]

    # Initialize Q and per-channel R from history strictly BEFORE the first row.
    first_ds = frame["ds"].iloc[0] if not frame.empty else pd.Timestamp.max
    first_row_for_cutoff = frame.iloc[0] if not frame.empty else pd.Series({"ds": first_ds})
    consensus_for_init = consensus_pit if not consensus_pit.empty else pd.DataFrame(columns=["ds", "consensus_pred"])
    prior = pd.DataFrame()
    if not consensus_for_init.empty:
        cons_with_actual = consensus_for_init.merge(
            frame[[
                c for c in ["ds", "actual_mom", "target_release_date", "actual_available_date"]
                if c in frame.columns
            ]].drop_duplicates("ds"),
            on="ds", how="inner",
        )
        prior = _actual_history_available_before(cons_with_actual, first_row_for_cutoff)
        prior = prior[~is_covid_month(prior["ds"])]

    R_init = 1.0
    if not prior.empty:
        err = (prior["actual_mom"] - prior["consensus_pred"]).values[-60:]
        if len(err) >= 2:
            R_init = float(np.var(err, ddof=1)) + 1e-6
    Q_init = 1.0
    if not prior.empty and len(prior) >= 3:
        Q_init = float(np.var(np.diff(prior["actual_mom"].values[-60:]), ddof=1)) + 1e-6

    # Per-channel R + bias caches — fall back to most-recent estimate when window short.
    R_cache: Dict[str, float] = {c: R_init * (1.0 if c == "consensus" else 1.5) for c in active}
    bias_cache: Dict[str, float] = {c: 0.0 for c in active}
    Q = Q_init

    # Seed state from available predictions. Actuals are only anchored after
    # the operational-availability filter admits them.
    first_row = frame.iloc[0]
    seed_preds = [first_row[c] for c in pred_cols if pd.notna(first_row.get(c))]
    x_hat = float(np.mean(seed_preds)) if seed_preds else 0.0
    P = R_init

    rows: List[Dict] = []
    last_state_actual_ds: Optional[pd.Timestamp] = None
    for i in range(len(frame)):
        row = frame.iloc[i]
        current_ds = pd.Timestamp(row["ds"])
        hist = frame.iloc[:i]
        hist_valid = _actual_history_available_before(hist, row)
        if not hist_valid.empty:
            latest = hist_valid.sort_values("ds").iloc[-1]
            latest_ds = pd.Timestamp(latest["ds"])
            if last_state_actual_ds is None or latest_ds > last_state_actual_ds:
                x_hat = float(latest["actual_mom"])
                P = 1e-6
                last_state_actual_ds = latest_ds
        state_gap_months = _month_gap(current_ds, last_state_actual_ds) if last_state_actual_ds is not None else 1
        hist_clean = hist_valid[~is_covid_month(hist_valid["ds"])] if not hist_valid.empty else hist_valid

        # Refresh Q from COVID-clean window when feasible.
        if len(hist_clean) >= 4:
            new_Q = _trailing_q(hist_clean, cfg.trailing_window)
            if new_Q is not None:
                Q = new_Q

        # Refresh each channel's R + bias when its COVID-clean history is large enough.
        for c in active:
            new_R, new_bias = _trailing_r_and_bias_for_channel(
                hist_clean, f"{c}_pred", cfg.trailing_window,
                min_history_for_bias=cfg.bias_correct_min_history,
            )
            if new_R is not None:
                R_cache[c] = new_R
            if cfg.bias_correct and new_bias is not None:
                bias_cache[c] = new_bias

        # Prediction step (random walk).
        x_prior = x_hat
        P_prior = P + Q * float(state_gap_months)
        info_prior = 1.0 / max(P_prior, 1e-9)

        # Channel terms (information contributions) — only channels with an
        # observed value at this row contribute. Bias-corrected obs: we add
        # the cached trailing mean error (actual - pred) so each channel is
        # debiased relative to its own track record before being precision-
        # weighted into the fused posterior.
        channel_infos: List[Tuple[str, float, float]] = []
        for c in active:
            obs = row.get(f"{c}_pred")
            if pd.isna(obs):
                continue
            R = max(R_cache.get(c, R_init), 1e-6)
            w = cfg.channel_weight_scales.get(c, 1.0)
            info_c = max(w, 0.0) / R
            obs_corrected = float(obs) + (bias_cache.get(c, 0.0) if cfg.bias_correct else 0.0)
            channel_infos.append((c, info_c, obs_corrected))

        # Optional cap: no single channel can dominate posterior precision
        # beyond `max_precision_share_per_channel`. Acts as a guardrail
        # against a single sidecar's R collapsing to near-zero.
        total_channel_info = sum(info for _, info, _ in channel_infos)
        base_for_cap = info_prior + total_channel_info
        cap_share = float(np.clip(cfg.max_precision_share_per_channel, 0.0, 0.99))
        if cap_share < 1.0 and total_channel_info > 0 and base_for_cap > 0:
            capped: List[Tuple[str, float, float]] = []
            for cname, info_c, obs in channel_infos:
                cap_info = (cap_share / max(1.0 - cap_share, 1e-6)) * (base_for_cap - info_c)
                capped.append((cname, min(info_c, cap_info) if cap_info > 0 else info_c, obs))
            channel_infos = capped
            total_channel_info = sum(info for _, info, _ in channel_infos)

        # Posterior.
        total_info = info_prior + total_channel_info
        if total_info <= 0:
            x_post = x_prior
            P_post = P_prior
        else:
            P_post = 1.0 / total_info
            weighted = info_prior * x_prior + sum(info_c * obs for _, info_c, obs in channel_infos)
            x_post = P_post * weighted

        pred = x_post
        fallback_used = False
        if (
            cfg.fallback_enabled
            and cfg.fallback_primary_channel
            and cfg.fallback_to_channel
        ):
            primary_col = f"{cfg.fallback_primary_channel}_pred"
            fallback_col = f"{cfg.fallback_to_channel}_pred"
            if (
                primary_col in frame.columns
                and pd.isna(row.get(primary_col))
                and fallback_col in frame.columns
                and pd.notna(row.get(fallback_col))
            ):
                pred = float(row[fallback_col])
                fallback_used = True

        # Step the state: if actual known, "anchor"; else keep posterior.
        actual_available_now = (
            pd.notna(row["actual_mom"])
            and (
                "actual_available_date" not in frame.columns
                or (
                    pd.notna(row.get("actual_available_date"))
                    and pd.Timestamp(row["actual_available_date"]) < _row_cutoff(row)
                )
            )
        )
        if actual_available_now:
            x_hat = float(row["actual_mom"])
            P = 1e-6
            last_state_actual_ds = current_ds
        else:
            x_hat = x_post
            P = P_post

        out_row = {
            "ds": row["ds"],
            "actual": row["actual_mom"],
            "target_release_date": row.get("target_release_date", pd.NaT),
            "actual_available_date": row.get("actual_available_date", pd.NaT),
            "history_available_n": int(len(hist_valid)),
            "latest_available_actual_ds": (
                hist_valid["ds"].max() if not hist_valid.empty else pd.NaT
            ),
            "state_gap_months": int(state_gap_months),
            "predicted": pred,
            "predicted_kalman_raw": x_post,  # un-overridden Kalman posterior
            "fallback_used": bool(fallback_used),
            "error": (row["actual_mom"] - pred) if pd.notna(row["actual_mom"]) else np.nan,
            "P_prior": P_prior,
            "P_post": P_post,
            "n_active_channels": int(len(channel_infos)),
        }
        for c, info_c, obs in channel_infos:
            out_row[f"{c}_pred"] = obs  # bias-corrected obs that the filter saw
            out_row[f"{c}_precision_share"] = float(info_c / total_info) if total_info > 0 else 0.0
            out_row[f"{c}_R"] = R_cache.get(c, R_init)
            out_row[f"{c}_bias"] = bias_cache.get(c, 0.0)
        rows.append(out_row)

    res_df = pd.DataFrame(rows)
    metrics = full_metrics(
        res_df["actual"].values,
        res_df["predicted"].values,
        label="SA_Kalman_Fusion",
        ds=res_df["ds"],
    )
    return res_df, metrics, active


# --------------------------------------------------------------------------- #
# Per-channel baselines for comparison
# --------------------------------------------------------------------------- #

def _per_channel_baselines(
    cfg: FusionConfig,
) -> pd.DataFrame:
    """Compute standalone metrics for each channel (no fusion).

    Returns a DataFrame with columns: channel, n_scored, mae, rmse, dir_acc, accel_acc.
    """
    frame, _, active = build_fusion_frame(cfg)
    rows: List[Dict] = []
    for c in active:
        col = f"{c}_pred"
        scored = frame.dropna(subset=[col, "actual_mom"]).copy()
        if scored.empty:
            rows.append({"channel": c, "n_scored": 0})
            continue
        err = scored[col].values - scored["actual_mom"].values
        a = scored["actual_mom"].values
        p = scored[col].values
        # canonical acceleration accuracy = sign(pred[m] - actual[m-1]) == sign(actual[m] - actual[m-1])
        prev = scored["actual_mom"].shift(1).values
        mask = ~np.isnan(prev)
        aa = (
            float(np.mean(np.sign(p[mask] - prev[mask]) == np.sign(a[mask] - prev[mask])))
            if mask.any() else float("nan")
        )
        rows.append({
            "channel": c,
            "n_scored": int(len(scored)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "dir_acc": float(np.mean(np.sign(p) == np.sign(a))),
            "accel_acc": aa,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Artifact writing
# --------------------------------------------------------------------------- #

def _write_artifacts(
    res_df: pd.DataFrame,
    metrics: Dict,
    cfg: FusionConfig,
    active_channels: List[str],
    baselines: pd.DataFrame,
) -> Dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths["backtest"] = cfg.output_dir / "backtest_results.csv"
    res_df.to_csv(paths["backtest"], index=False)
    paths["metrics"] = cfg.output_dir / "metrics.json"
    paths["metrics"].write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    paths["baselines"] = cfg.output_dir / "per_channel_baselines.csv"
    baselines.to_csv(paths["baselines"], index=False)
    manifest = {
        "run_id": cfg.run_id,
        "target_type": cfg.target_type,
        "target_source": cfg.target_source,
        "trailing_window": cfg.trailing_window,
        "include_consensus": cfg.include_consensus,
        "use_winsorized_actuals": cfg.use_winsorized_actuals,
        "max_precision_share_per_channel": cfg.max_precision_share_per_channel,
        "channels_requested": [c[0] for c in cfg.channels],
        "channels_active": active_channels,
        "channel_weight_scales": cfg.channel_weight_scales,
        "fallback_enabled": cfg.fallback_enabled,
        "fallback_primary_channel": cfg.fallback_primary_channel,
        "fallback_to_channel": cfg.fallback_to_channel,
    }
    paths["manifest"] = cfg.output_dir / "manifest.json"
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return paths


def run_pipeline(cfg: FusionConfig) -> Dict:
    res_df, metrics, active = run_sa_kalman_fusion(cfg)
    baselines = _per_channel_baselines(cfg)
    paths = _write_artifacts(res_df, metrics, cfg, active, baselines)
    return {
        "metrics": metrics,
        "n_rows": int(len(res_df)),
        "active_channels": active,
        "paths": {k: str(v) for k, v in paths.items()},
        "baselines": baselines.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID,
                        help="Sidecar run-id (under _output/sidecars/sa/<run_id>).")
    parser.add_argument("--output-dir", type=Path, default=SA_CHALLENGER_ROOT / "kalman_fusion")
    parser.add_argument("--trailing-window", type=int, default=18)
    parser.add_argument("--no-consensus", action="store_true",
                        help="Exclude the consensus channel.")
    parser.add_argument("--no-winsor-actuals", action="store_true",
                        help="Disable COVID-period winsorization of actuals.")
    parser.add_argument("--max-precision-share", type=float, default=0.65,
                        help="Per-channel precision-share cap.")
    parser.add_argument("--no-bias-correct", action="store_true",
                        help="Disable per-channel trailing-bias correction.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable the missing-primary-channel fallback.")
    parser.add_argument("--fallback-primary-channel", default="dynamic_top6_k12_cov90",
                        help="When this channel is NaN at row t, override the "
                             "fused forecast with the fallback channel.")
    parser.add_argument("--fallback-to-channel", default="consensus_median",
                        help="Channel id to use when the primary is missing.")
    parser.add_argument("--channels", default=None,
                        help="Comma-separated channel ids to include (default: all).")
    args = parser.parse_args()

    # When user passes --channels, search the EXTENDED list so they can
    # opt in to econ_panel / sa_state_space; otherwise use the safe default.
    if args.channels:
        wanted = {x.strip() for x in args.channels.split(",") if x.strip()}
        chans = [c for c in EXTENDED_CHANNELS if c[0] in wanted]
    else:
        chans = list(DEFAULT_CHANNELS)
    cfg = FusionConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        trailing_window=int(args.trailing_window),
        channels=chans,
        include_consensus=not args.no_consensus,
        use_winsorized_actuals=not args.no_winsor_actuals,
        max_precision_share_per_channel=float(args.max_precision_share),
        bias_correct=not args.no_bias_correct,
        fallback_enabled=not args.no_fallback,
        fallback_primary_channel=args.fallback_primary_channel,
        fallback_to_channel=args.fallback_to_channel,
    )
    out = run_pipeline(cfg)
    print(json.dumps({
        "metrics": out["metrics"],
        "active_channels": out["active_channels"],
        "n_rows": out["n_rows"],
        "baselines": out["baselines"],
        "paths": out["paths"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
