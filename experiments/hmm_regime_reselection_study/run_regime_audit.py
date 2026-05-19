"""Walk-forward HMM regime audit over cached training matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Train.hmm_regime_reselection import HMMRegimeConfig, evaluate_hmm_reselection_trigger


def _find_cache_pair(cache_dir: Path, target: str, source: str) -> tuple[Path, Path]:
    stem_hint = f"{target}_first_{source}"
    x_files = sorted(cache_dir.glob(f"{stem_hint}*.X.parquet"))
    if not x_files:
        x_files = sorted(cache_dir.glob("*.X.parquet"))
    for x_path in x_files:
        y_path = x_path.with_name(x_path.name.replace(".X.parquet", ".y.parquet"))
        if y_path.exists():
            return x_path, y_path
    raise FileNotFoundError(f"No X/y parquet cache pair found under {cache_dir}")


def _load_y(path: Path) -> pd.Series:
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.Series):
        return pd.to_numeric(obj, errors="coerce").reset_index(drop=True)
    numeric = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
    if "y_mom" in obj.columns:
        return pd.to_numeric(obj["y_mom"], errors="coerce").reset_index(drop=True)
    if numeric:
        return pd.to_numeric(obj[numeric[0]], errors="coerce").reset_index(drop=True)
    raise ValueError(f"Could not infer target column from {path}")


def _jsonable_event(event: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in event.items():
        if isinstance(value, pd.Timestamp):
            out[key] = value.strftime("%Y-%m-%d")
        else:
            out[key] = value
    return out


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = _find_cache_pair(Path(args.training_cache), args.target, args.source)
    X = pd.read_parquet(x_path).copy()
    y = _load_y(y_path)
    X["ds"] = pd.to_datetime(X["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    X = X.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    y = y.iloc[: len(X)].reset_index(drop=True)

    config = HMMRegimeConfig(
        min_train_months=int(args.min_train_months),
        n_components=int(args.n_components),
        covariance_type=str(args.covariance_type),
        max_features=int(args.max_features),
        min_features=int(args.min_features),
        min_non_nan=int(args.min_non_nan),
        min_gap_months=int(args.min_gap_months),
        max_gap_months=int(args.max_gap_months),
        surprise_quantile=float(args.surprise_quantile),
        force_reselect_risk=float(args.force_reselect_risk),
        emission_profile=str(args.emission_profile),
        trigger_score_threshold=float(args.trigger_score_threshold),
        surprise_low_quantile=float(args.surprise_low_quantile),
        surprise_high_quantile=float(args.surprise_high_quantile),
        severe_surprise_ratio=float(args.severe_surprise_ratio),
        downside_surprise_ratio=float(args.downside_surprise_ratio),
        seasonal_penalty_surprise_ratio=float(args.seasonal_penalty_surprise_ratio),
        min_state_support_n=int(args.min_state_support_n),
        min_state_support_share=float(args.min_state_support_share),
        min_expected_duration=float(args.min_expected_duration),
        episode_suppression_months=int(args.episode_suppression_months),
        episode_clear_months=int(args.episode_clear_months),
    )

    previous_snapshot = None
    last_reselection_date = None
    active_episode_start = None
    low_signal_streak = 0
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(args.start).to_period("M").to_timestamp()
    numeric_features = [
        c for c in X.columns
        if c != "ds" and pd.api.types.is_numeric_dtype(X[c])
    ]

    for pos, step_date in enumerate(X["ds"]):
        step_date = pd.Timestamp(step_date)
        if step_date < start:
            continue
        train = X.iloc[:pos].reset_index(drop=True)
        y_train = y.iloc[:pos].reset_index(drop=True)
        if len(train) < int(args.min_train_months):
            continue
        decision = evaluate_hmm_reselection_trigger(
            X_train=train,
            y_train=y_train,
            step_date=step_date,
            cleaned_features=numeric_features,
            previous_snapshot=previous_snapshot,
            last_reselection_date=last_reselection_date,
            config=config,
            X_current=X.iloc[[pos]],
        )
        log = decision.to_log_dict()
        bootstrap = previous_snapshot is None and decision.snapshot is not None
        raw_reselect = bool(decision.should_reselect or bootstrap)
        if bootstrap:
            log["hmm_trigger_class"] = "bootstrap"
            log["hmm_trigger_reason"] = "bootstrap"
        final_reselect = raw_reselect
        if raw_reselect and not bootstrap and active_episode_start is not None:
            months_in_episode = step_date.year * 12 + step_date.month - (
                active_episode_start.year * 12 + active_episode_start.month
            )
            severe = (
                pd.notna(log.get("hmm_surprise_ratio"))
                and float(log["hmm_surprise_ratio"]) >= float(config.severe_surprise_ratio)
                and str(log.get("hmm_regime_label")) in set(config.force_reselect_labels)
            )
            if months_in_episode < int(config.episode_suppression_months) and not severe:
                final_reselect = False
                log["hmm_episode_suppressed"] = True
                log["hmm_trigger_class"] = "episode_suppressed"
                log["hmm_trigger_reason"] = f"episode_suppressed:{log.get('hmm_trigger_reason')}"
        if final_reselect:
            last_reselection_date = step_date
            active_episode_start = step_date
            low_signal_streak = 0
        else:
            low_signal = (
                pd.notna(log.get("hmm_transition_risk"))
                and float(log["hmm_transition_risk"]) < float(config.transition_risk_threshold)
                and (
                    pd.isna(log.get("hmm_surprise_ratio"))
                    or float(log["hmm_surprise_ratio"]) < 1.0
                )
            )
            if active_episode_start is not None and low_signal:
                low_signal_streak += 1
                if low_signal_streak >= int(config.episode_clear_months):
                    active_episode_start = None
                    low_signal_streak = 0
            elif raw_reselect:
                low_signal_streak = 0
        if decision.snapshot is not None:
            previous_snapshot = decision.snapshot
        log.update({
            "step_index": int(pos),
            "target_type": args.target,
            "target_source": args.source,
            "raw_reselect": raw_reselect,
            "final_reselect": final_reselect,
        })
        rows.append(_jsonable_event(log))

    out = pd.DataFrame(rows)
    csv_path = output_dir / "hmm_regime_monthly.csv"
    if "hmm_features" in out.columns:
        out["hmm_features"] = out["hmm_features"].apply(json.dumps)
    if "hmm_state_stats" in out.columns:
        out["hmm_state_stats"] = out["hmm_state_stats"].apply(json.dumps)
    out.to_csv(csv_path, index=False)
    payload = {
        "x_path": str(x_path),
        "y_path": str(y_path),
        "config": config.__dict__,
        "events": rows,
    }
    (output_dir / "hmm_regime_monthly.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    print(f"Wrote {len(out)} HMM audit rows to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="_output_hmm_regime_reselection_study")
    parser.add_argument("--training-cache", default="_output_pairing_baseline_pitfix/cache/training_dataset")
    parser.add_argument("--backtest-dir", default="_output_pairing_baseline_pitfix")
    parser.add_argument("--target", default="nsa")
    parser.add_argument("--source", default="revised")
    parser.add_argument("--start", default="2000-01")
    parser.add_argument("--min-train-months", type=int, default=96)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--covariance-type", default="diag")
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--min-features", type=int, default=4)
    parser.add_argument("--min-non-nan", type=int, default=72)
    parser.add_argument("--min-gap-months", type=int, default=9)
    parser.add_argument("--max-gap-months", type=int, default=0)
    parser.add_argument("--surprise-quantile", type=float, default=0.95)
    parser.add_argument("--force-reselect-risk", type=float, default=0.60)
    parser.add_argument(
        "--emission-profile",
        choices=["raw", "seasonal_resid", "hybrid", "macro_only"],
        default="seasonal_resid",
    )
    parser.add_argument("--trigger-score-threshold", type=float, default=2.0)
    parser.add_argument("--surprise-low-quantile", type=float, default=0.90)
    parser.add_argument("--surprise-high-quantile", type=float, default=0.975)
    parser.add_argument("--severe-surprise-ratio", type=float, default=2.0)
    parser.add_argument("--downside-surprise-ratio", type=float, default=1.05)
    parser.add_argument("--seasonal-penalty-surprise-ratio", type=float, default=1.50)
    parser.add_argument("--min-state-support-n", type=int, default=12)
    parser.add_argument("--min-state-support-share", type=float, default=0.05)
    parser.add_argument("--min-expected-duration", type=float, default=2.0)
    parser.add_argument("--episode-suppression-months", type=int, default=6)
    parser.add_argument("--episode-clear-months", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
