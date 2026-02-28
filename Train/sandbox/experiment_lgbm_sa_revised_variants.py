"""
Sandbox LightGBM variant suite for SA revised.

Goal:
- Re-run "past-style" and "current-style" SA-revised model variants on the
  current feature dataset.
- Keep all artifacts fully sandboxed under `_output/sandbox/sa_revised_variants`.
- Produce production-style diagnostics (ACF/PACF, directional/acceleration,
  plots, tables) for each variant.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Default to stable non-multiprocessing mode for joblib feature building.
# Override by exporting JOBLIB_MULTIPROCESSING=1 before running.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from settings import OUTPUT_DIR, BACKTEST_MONTHS, TEMP_DIR, setup_logger
from Train.branch_target_selection import (
    partition_feature_columns,
    select_branch_target_features_for_step,
)
from Train.config import load_selected_features, N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Train.data_loader import load_target_data
from Train.hyperparameter_tuning import tune_hyperparameters
from Train.model import calculate_sample_weights
from Train.sandbox.output_utils import write_sandbox_output_bundle
from Train.short_pass_selection import select_features_for_step
from Train.train_lightgbm_nfp import build_training_dataset, clean_features

logger = setup_logger(__file__, TEMP_DIR)
OUT_ROOT = OUTPUT_DIR / "sandbox" / "sa_revised_variants"


@dataclass(frozen=True)
class VariantConfig:
    name: str
    description: str
    min_non_nan: int = 36
    half_life_months: float = 60.0
    use_shortpass: bool = False
    shortpass_topk: int = 60
    shortpass_method: str = "lgbm_gain"
    include_branch_target: bool = False
    use_branch_fs: bool = False
    branch_topk: int = 8
    branch_method: str = "weighted_corr"
    branch_corr_threshold: float = 0.90
    branch_min_overlap: int = 24
    tail_weighting: bool = False
    tail_level_q: float = 0.80
    tail_diff_q: float = 0.80
    tail_level_boost: float = 1.35
    tail_diff_boost: float = 1.35
    tail_max_mult: float = 2.50
    amplitude_calibration: bool = False
    amp_slope_min: float = 0.50
    amp_slope_max: float = 3.00
    delta_blend: float = 0.0
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    num_boost_round: int = 600
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class TuneOptions:
    enabled: bool = False
    n_trials: int = N_OPTUNA_TRIALS
    timeout: int = OPTUNA_TIMEOUT
    objective_mode: str = "composite"
    tune_every_steps: int = 1
    use_huber_loss: bool = True


VARIANTS: Dict[str, VariantConfig] = {
    "v1_legacy_level_only": VariantConfig(
        name="v1_legacy_level_only",
        description="Legacy-style direct level model: master snapshot features only.",
        use_shortpass=False,
        include_branch_target=False,
        use_branch_fs=False,
        tail_weighting=False,
        amplitude_calibration=False,
        delta_blend=0.0,
    ),
    "v2_snapshot_plus_branch": VariantConfig(
        name="v2_snapshot_plus_branch",
        description="Adds short-pass snapshot extras + weighted-corr branch-target selection.",
        use_shortpass=True,
        include_branch_target=True,
        use_branch_fs=True,
        branch_topk=8,
        branch_method="weighted_corr",
        tail_weighting=False,
        amplitude_calibration=False,
        delta_blend=0.0,
    ),
    "v3_dynamics_tail": VariantConfig(
        name="v3_dynamics_tail",
        description="Variance-aware branch dynamics selection + tail-aware weighting.",
        use_shortpass=True,
        include_branch_target=True,
        use_branch_fs=True,
        branch_topk=16,
        branch_method="dynamics_composite",
        tail_weighting=True,
        amplitude_calibration=False,
        delta_blend=0.0,
    ),
    "v4_delta_amp_stack": VariantConfig(
        name="v4_delta_amp_stack",
        description="Adds amplitude calibration and delta-head blend on top of v3.",
        use_shortpass=True,
        include_branch_target=True,
        use_branch_fs=True,
        branch_topk=16,
        branch_method="dynamics_composite",
        tail_weighting=True,
        amplitude_calibration=True,
        delta_blend=0.70,
    ),
}


def _merge_unique(*groups: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for feat in group:
            if feat not in seen:
                seen.add(feat)
                out.append(feat)
    return out


def _apply_tail_weighting(
    base_weights: np.ndarray,
    y_values: np.ndarray,
    cfg: VariantConfig,
) -> np.ndarray:
    if not cfg.tail_weighting:
        return base_weights

    y = np.asarray(y_values, dtype=float)
    w = np.asarray(base_weights, dtype=float).copy()
    if y.size == 0 or y.size != w.size:
        return w

    abs_y = np.abs(y)
    level_thr = float(np.quantile(abs_y, cfg.tail_level_q))
    mult = np.ones_like(y, dtype=float)
    mult[abs_y >= level_thr] *= cfg.tail_level_boost

    abs_dy = np.abs(np.diff(y, prepend=y[0]))
    diff_thr = float(np.quantile(abs_dy, cfg.tail_diff_q))
    mult[abs_dy >= diff_thr] *= cfg.tail_diff_boost

    mult = np.clip(mult, 1.0, cfg.tail_max_mult)
    out = w * mult
    mean_w = float(np.mean(out))
    if mean_w > 0:
        out = out / mean_w
    return out


def _fit_amplitude_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slope_min: float,
    slope_max: float,
) -> Optional[tuple[float, float]]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 12:
        return None
    x = y_pred[mask].astype(float)
    y = y_true[mask].astype(float)
    if float(np.std(x)) < 1e-10:
        return None

    # OLS: y = intercept + slope * x
    slope, intercept = np.polyfit(x, y, deg=1)
    slope = float(np.clip(float(slope), slope_min, slope_max))
    return float(intercept), slope


def _fit_lgb_model(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_tr: np.ndarray,
    cfg: VariantConfig,
    params_override: Optional[Dict] = None,
):
    import lightgbm as lgb

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params_override:
        tuned = dict(params_override)
        tuned.pop("half_life_months", None)
        params.update(tuned)

    train_data = lgb.Dataset(X_tr, label=y_tr.values, weight=weights_tr)
    valid_data = lgb.Dataset(X_val, label=y_val.values, reference=train_data)
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _select_features_for_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_month: pd.Timestamp,
    cfg: VariantConfig,
    snapshot_master_features: List[str],
) -> List[str]:
    cleaned = clean_features(X_train, y_train, min_non_nan=cfg.min_non_nan)
    cleaned = [c for c in cleaned if c in X_train.columns and c != "ds"]
    if not cleaned:
        return []

    groups = partition_feature_columns(cleaned, target_type="sa")

    snapshot_candidates = [c for c in groups["snapshot_features"] if c in X_train.columns]
    master_base = [c for c in snapshot_master_features if c in snapshot_candidates]
    if not master_base:
        master_base = snapshot_candidates.copy()
    master_base_set = set(master_base)
    snapshot_extra = [c for c in snapshot_candidates if c not in master_base_set]

    fs_weights = calculate_sample_weights(
        X_train[["ds"]].copy(),
        target_month=target_month,
        half_life_months=cfg.half_life_months,
    )

    snapshot_selected = master_base
    if cfg.use_shortpass and snapshot_extra:
        top_k = min(cfg.shortpass_topk, len(snapshot_extra))
        if len(snapshot_extra) > top_k:
            snapshot_extra_selected = select_features_for_step(
                X_train=X_train[snapshot_extra],
                y_train=y_train,
                candidate_features=snapshot_extra,
                top_k=top_k,
                method=cfg.shortpass_method,
                sample_weights=fs_weights,
            )
        else:
            snapshot_extra_selected = snapshot_extra
        snapshot_selected = _merge_unique(master_base, snapshot_extra_selected)

    branch_candidates = groups["target_branch_features"] if cfg.include_branch_target else []
    if cfg.use_branch_fs and branch_candidates:
        branch_selected = select_branch_target_features_for_step(
            X_train=X_train,
            y_train=y_train,
            target_type="sa",
            candidate_features=branch_candidates,
            top_k=cfg.branch_topk,
            method=cfg.branch_method,
            corr_threshold=cfg.branch_corr_threshold,
            min_overlap=cfg.branch_min_overlap,
            sample_weights=fs_weights,
        )
    else:
        branch_selected = branch_candidates

    always_keep = _merge_unique(groups["calendar_features"], groups["revision_features"])
    features = _merge_unique(snapshot_selected, branch_selected, always_keep)
    return [c for c in features if c in X_train.columns and c != "ds"]


def _tune_params_for_step(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    target_month: pd.Timestamp,
    cfg: VariantConfig,
    tune_opts: TuneOptions,
) -> Optional[Dict]:
    if not tune_opts.enabled:
        return None
    if not feature_cols:
        return None

    X_tune = X_train_raw[["ds"] + feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    logger.info(
        "Optuna tuning: month=%s objective=%s trials=%d timeout=%ss features=%d rows=%d",
        target_month.strftime("%Y-%m"),
        tune_opts.objective_mode,
        int(tune_opts.n_trials),
        int(tune_opts.timeout),
        len(feature_cols),
        len(X_tune),
    )
    tuned = tune_hyperparameters(
        X=X_tune,
        y=y_train,
        target_month=target_month,
        n_trials=int(tune_opts.n_trials),
        timeout=int(tune_opts.timeout),
        num_boost_round=cfg.num_boost_round,
        early_stopping_rounds=cfg.early_stopping_rounds,
        objective_mode=tune_opts.objective_mode,
        use_huber_loss=bool(tune_opts.use_huber_loss),
    )
    return tuned


def run_variant_backtest(
    cfg: VariantConfig,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    target_df: pd.DataFrame,
    snapshot_master_features: List[str],
    min_train_rows: int,
    backtest_months: int,
    tune_opts: Optional[TuneOptions] = None,
) -> pd.DataFrame:
    if tune_opts is None:
        tune_opts = TuneOptions(enabled=False)

    backtest_targets = target_df.iloc[-backtest_months:]["ds"].tolist()
    date_to_idx = {d: i for i, d in enumerate(X_full["ds"])}
    rows: List[Dict] = []

    for step_idx, target_month in enumerate(backtest_targets, 1):
        target_idx = date_to_idx.get(target_month)
        if target_idx is None:
            continue

        train_mask = X_full["ds"] < target_month
        train_idx = X_full[train_mask].index.tolist()
        if len(train_idx) < min_train_rows:
            continue

        y_train = y_full.iloc[train_idx]
        valid_mask = ~y_train.isna()
        if int(valid_mask.sum()) < min_train_rows:
            continue

        valid_train_idx = [train_idx[j] for j in range(len(train_idx)) if bool(valid_mask.iloc[j])]
        X_train_raw = X_full.iloc[valid_train_idx].copy()
        y_train = y_train[valid_mask].copy()
        X_pred_raw = X_full.iloc[[target_idx]].copy()
        actual = y_full.iloc[target_idx]

        feature_cols = _select_features_for_step(
            X_train=X_train_raw,
            y_train=y_train,
            target_month=target_month,
            cfg=cfg,
            snapshot_master_features=snapshot_master_features,
        )
        if not feature_cols:
            continue

        X_train = X_train_raw[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_pred = X_pred_raw[feature_cols].replace([np.inf, -np.inf], np.nan)

        split = max(int(len(X_train) * 0.85), 24)
        if split >= len(X_train):
            split = len(X_train) - 1
        if split <= 0:
            continue

        X_tr = X_train.iloc[:split]
        X_val = X_train.iloc[split:]
        y_tr = y_train.iloc[:split]
        y_val = y_train.iloc[split:]
        if len(X_val) < 3:
            continue

        tuned_params: Optional[Dict] = None
        if tune_opts.enabled and ((step_idx - 1) % max(1, int(tune_opts.tune_every_steps)) == 0):
            tuned_params = _tune_params_for_step(
                X_train_raw=X_train_raw,
                y_train=y_train,
                feature_cols=feature_cols,
                target_month=target_month,
                cfg=cfg,
                tune_opts=tune_opts,
            )

        half_life = cfg.half_life_months
        if tuned_params is not None and "half_life_months" in tuned_params:
            try:
                half_life = float(tuned_params["half_life_months"])
            except Exception:
                half_life = cfg.half_life_months

        all_weights = calculate_sample_weights(
            X_train_raw[["ds"]].copy(),
            target_month=target_month,
            half_life_months=half_life,
        )
        all_weights = _apply_tail_weighting(
            base_weights=all_weights,
            y_values=y_train.values.astype(float),
            cfg=cfg,
        )
        w_tr = all_weights[:split]

        level_model = _fit_lgb_model(
            X_tr, y_tr, X_val, y_val, w_tr, cfg, params_override=tuned_params
        )
        best_iter = level_model.best_iteration or cfg.num_boost_round
        pred_level = float(level_model.predict(X_pred, num_iteration=best_iter)[0])

        if cfg.amplitude_calibration:
            val_pred = level_model.predict(X_val, num_iteration=best_iter)
            calib = _fit_amplitude_calibrator(
                y_true=y_val.values.astype(float),
                y_pred=np.asarray(val_pred, dtype=float),
                slope_min=cfg.amp_slope_min,
                slope_max=cfg.amp_slope_max,
            )
            if calib is not None:
                intercept, slope = calib
                pred_level = float(intercept + slope * pred_level)

        pred_final = pred_level

        if cfg.delta_blend > 0.0 and len(y_train) >= 30:
            y_delta = y_train.diff()
            delta_mask = y_delta.notna()
            if int(delta_mask.sum()) >= 24:
                X_delta = X_train.loc[delta_mask.values]
                y_delta_train = y_delta[delta_mask].astype(float)
                split_d = max(int(len(X_delta) * 0.85), 18)
                if split_d >= len(X_delta):
                    split_d = len(X_delta) - 1
                if split_d > 0 and len(X_delta.iloc[split_d:]) >= 3:
                    X_d_tr = X_delta.iloc[:split_d]
                    X_d_val = X_delta.iloc[split_d:]
                    y_d_tr = y_delta_train.iloc[:split_d]
                    y_d_val = y_delta_train.iloc[split_d:]
                    w_delta_full = all_weights[delta_mask.values]
                    w_d_tr = w_delta_full[:split_d]
                    delta_model = _fit_lgb_model(
                        X_d_tr, y_d_tr, X_d_val, y_d_val, w_d_tr, cfg, params_override=tuned_params
                    )
                    delta_iter = delta_model.best_iteration or cfg.num_boost_round
                    pred_delta_model = float(delta_model.predict(X_pred, num_iteration=delta_iter)[0])
                    prev_actual = float(y_train.iloc[-1])
                    base_delta = float(pred_level - prev_actual)
                    blended_delta = (1.0 - cfg.delta_blend) * base_delta + cfg.delta_blend * pred_delta_model
                    pred_final = float(prev_actual + blended_delta)

        err = np.nan if pd.isna(actual) else float(actual - pred_final)
        rows.append(
            {
                "ds": target_month,
                "actual": actual,
                "predicted": pred_final,
                "error": err,
                "n_features": int(len(feature_cols)),
                "n_train_samples": int(len(X_train)),
                "tuned": int(bool(tuned_params)),
                "tuned_half_life_months": float(half_life),
            }
        )
        logger.info(
            "[%s][%d/%d] %s | actual=%s pred=%.1f | n_features=%d | tuned=%s",
            cfg.name,
            step_idx,
            len(backtest_targets),
            target_month.strftime("%Y-%m"),
            "nan" if pd.isna(actual) else f"{float(actual):.1f}",
            pred_final,
            len(feature_cols),
            "yes" if tuned_params else "no",
        )

    return pd.DataFrame(rows)


def _parse_variants_arg(raw: str) -> List[str]:
    val = (raw or "all").strip().lower()
    if val in {"all", "*"}:
        return list(VARIANTS.keys())
    selected = [v.strip() for v in val.split(",") if v.strip()]
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {list(VARIANTS.keys())}")
    return selected


def run_suite(
    variant_ids: List[str],
    min_train_rows: int,
    backtest_months: int,
    tune_opts: Optional[TuneOptions] = None,
) -> None:
    if tune_opts is None:
        tune_opts = TuneOptions(enabled=False)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    target_df = load_target_data(
        target_type="sa",
        release_type="first",
        target_source="revised",
    )
    X_full, y_full = build_training_dataset(
        target_df=target_df,
        target_type="sa",
        release_type="first",
        target_source="revised",
        show_progress=False,
    )
    if X_full.empty:
        raise RuntimeError("Failed to build SA revised training dataset.")

    snapshot_master_features = load_selected_features("sa", "revised")
    manifest_rows: List[Dict] = []

    for idx, variant_id in enumerate(variant_ids, 1):
        cfg = VARIANTS[variant_id]
        logger.info(
            "\n%s\n[%d/%d] Running variant: %s\n%s",
            "=" * 72,
            idx,
            len(variant_ids),
            cfg.name,
            cfg.description,
        )
        variant_dir = OUT_ROOT / cfg.name
        results = run_variant_backtest(
            cfg=cfg,
            X_full=X_full,
            y_full=y_full,
            target_df=target_df,
            snapshot_master_features=snapshot_master_features,
            min_train_rows=min_train_rows,
            backtest_months=backtest_months,
            tune_opts=tune_opts,
        )

        n_features = None
        if "n_features" in results.columns and not results["n_features"].dropna().empty:
            n_features = int(results["n_features"].dropna().median())
        metrics = write_sandbox_output_bundle(
            results_df=results,
            out_dir=variant_dir,
            model_id=cfg.name,
            diagnostics_label=f"Sandbox {cfg.name}",
            n_features=n_features,
        )

        with open(variant_dir / "variant_config.json", "w") as f:
            json.dump(
                {
                    "variant_config": asdict(cfg),
                    "tuning": asdict(tune_opts),
                },
                f,
                indent=2,
            )

        row = {"model_id": cfg.name, "description": cfg.description}
        row.update(metrics)
        row["tuning_enabled"] = bool(tune_opts.enabled)
        row["tuning_objective"] = tune_opts.objective_mode if tune_opts.enabled else "none"
        row["tuning_trials"] = int(tune_opts.n_trials) if tune_opts.enabled else 0
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_ROOT / "suite_summary.csv", index=False)
    with open(OUT_ROOT / "suite_summary.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
    logger.info("Saved SA revised variant suite summary -> %s", OUT_ROOT / "suite_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sandbox LightGBM SA-revised variant suite (no main pipeline side effects)."
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant ids or 'all'.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=120,
        help="Minimum usable rows before each walk-forward step.",
    )
    parser.add_argument(
        "--backtest-months",
        type=int,
        default=BACKTEST_MONTHS,
        help="Number of most-recent months for walk-forward backtest.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna tuning for sandbox LightGBM variants.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=N_OPTUNA_TRIALS,
        help=f"Optuna trials per tuning run (default: {N_OPTUNA_TRIALS}).",
    )
    parser.add_argument(
        "--tune-timeout",
        type=int,
        default=OPTUNA_TIMEOUT,
        help=f"Optuna timeout (seconds) per tuning run (default: {OPTUNA_TIMEOUT}).",
    )
    parser.add_argument(
        "--tune-objective",
        type=str,
        choices=["mae", "composite"],
        default="composite",
        help="Optuna objective for sandbox variants.",
    )
    parser.add_argument(
        "--tune-every-steps",
        type=int,
        default=1,
        help="Re-tune every N walk-forward steps (1 = full tuning each step).",
    )
    parser.add_argument(
        "--no-tune-huber",
        action="store_true",
        help="Disable Huber objective during Optuna tuning.",
    )
    args = parser.parse_args()

    variant_ids = _parse_variants_arg(args.variants)
    tune_opts = TuneOptions(
        enabled=bool(args.tune),
        n_trials=int(args.tune_trials),
        timeout=int(args.tune_timeout),
        objective_mode=str(args.tune_objective),
        tune_every_steps=max(1, int(args.tune_every_steps)),
        use_huber_loss=not bool(args.no_tune_huber),
    )
    run_suite(
        variant_ids=variant_ids,
        min_train_rows=int(args.min_train_rows),
        backtest_months=int(args.backtest_months),
        tune_opts=tune_opts,
    )


if __name__ == "__main__":
    main()
