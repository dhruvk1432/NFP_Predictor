"""Dynamic Factor Model sidecar.

PIT-safe walk-forward DFM (`statsmodels.tsa.statespace.DynamicFactor`) over the
target's NFP MoM plus a small curated panel of employment / activity
observables drawn from the PIT-safe master snapshots. Emits:

* Direct one-step-ahead forecast for the NFP MoM (joint endog modeling).
* Latent factor estimate s_t and its first difference Δs_t.
* Factor 3-month slope as a low-noise momentum diagnostic.
* Per-step posterior forecast variance (Kalman filter native).

Decomposition (Notes/State_Space_Models_1.pdf §1.2):

    y_{t,i} = λ_i · s_t + u_{t,i},        u ~ N(0, diag(σ²))
    s_t     = ρ · s_{t-1} + ε_t,          ε ~ N(0, σ²_ε)

We include y_mom as one of the y_{t,i} so the model's 1-step-ahead Kalman
prediction for that row is the natural NFP forecast. This avoids a separate
factor → MoM regression step and keeps everything under one MLE.

PIT contract:
* The endog panel is built from the PIT-safe master-snapshot matrix
  (one row per snapshot, only the snapshot's release-time data is exposed).
* For target month M we train on rows ds < M and forecast ds = M.
* ``trained_through`` is the last training row's ds, strictly < M.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sidecars.common import (  # noqa: E402
    feature_audit_from_frame,
    sidecar_branch_root,
    write_sidecar_artifacts,
)
from experiments.sidecars.feature_matrix import (  # noqa: E402
    build_sidecar_design,
    select_numeric_feature_cols,
    source_map_for_columns,
)

try:
    from settings import DATA_PATH, OUTPUT_DIR  # noqa: E402
except RuntimeError:
    DATA_PATH = Path("data")
    OUTPUT_DIR = Path("_output")

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_TARGET_TYPE = "sa"
DEFAULT_MODEL_ID = "dfm_factor"
LEGACY_DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "dfm_factor"
)

# Curated observable prefixes — these are the labor / activity blocks most
# directly tied to the labor cycle factor. Pulled from the unifier + stress
# master snapshot blocks (build_sidecar_design exposes these).
LABOR_FLAVOR_PREFIXES: tuple[str, ...] = (
    "ISM_",
    "Challenger_",
    "AHE_",
    "AWH_",
    "UMich_",
    "CB_",
    "Retail_",
    "Industrial_",
)


def _default_target_path(target_type: str) -> Path:
    if target_type == "nsa":
        return DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    return DATA_PATH / "NFP_target" / "y_sa_revised.parquet"


def _default_target_space(target_type: str) -> str:
    return "nsa_revised" if target_type == "nsa" else "sa_revised"


def _resolve_output_dir(explicit: Path | None, target_type: str, run_id: str) -> Path:
    if explicit is not None:
        return explicit
    target_type = str(target_type).strip().lower()
    if target_type == "sa":
        return sidecar_branch_root(OUTPUT_DIR, "sa") / run_id / "dfm_factor"
    return LEGACY_DEFAULT_OUTPUT_DIR


DEFAULT_TARGET_PATH = _default_target_path(DEFAULT_TARGET_TYPE)
DEFAULT_OUTPUT_DIR = LEGACY_DEFAULT_OUTPUT_DIR


def _pick_observable_columns(
    design: pd.DataFrame,
    max_observables: int,
    min_non_nan: int,
) -> list[str]:
    """Pick the K most-populated labor-flavored snapshot columns.

    Falls back to broadly-populated numeric snapshot columns if no
    labor-prefix matches survive.
    """
    snapshot_cols = [
        c for c in design.columns
        if c.startswith(LABOR_FLAVOR_PREFIXES) and pd.api.types.is_numeric_dtype(design[c])
    ]
    if not snapshot_cols:
        snapshot_cols = [
            c for c in select_numeric_feature_cols(design)
            if c != "y" and not c.startswith(("sa_", "nsa_", "prev_", "actual_", "target_"))
        ]
    scored: list[tuple[float, float, str]] = []
    for c in snapshot_cols:
        s = pd.to_numeric(design[c], errors="coerce")
        non_nan = int(s.notna().sum())
        if non_nan < int(min_non_nan):
            continue
        var = float(s.var(skipna=True))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        scored.append((non_nan, var, c))
    scored.sort(reverse=True)
    return [c for _, _, c in scored[: int(max_observables)]]


def _standardize_block(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score each column using its own column moments. DFM is scale-sensitive."""
    mu = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, skipna=True).replace(0.0, np.nan)
    return (df - mu) / sd, mu, sd


def _fit_predict_one_step(
    endog_train: pd.DataFrame,
    *,
    k_factors: int,
    factor_order: int,
) -> dict[str, float | int]:
    """Fit DFM on standardized endog_train and 1-step-ahead forecast each column.

    Returns dict with predicted_std (per column), factor_estimate (last filtered
    state), factor_diff (last - prior smoothed state), factor_slope_3m.
    """
    out: dict[str, Any] = {
        "predicted_std_y_mom": float("nan"),
        "predicted_var_y_mom": float("nan"),
        "factor_estimate": float("nan"),
        "factor_diff": float("nan"),
        "factor_slope_3m": float("nan"),
        "fit_succeeded": False,
    }
    if endog_train.shape[0] < 36 or endog_train.shape[1] < 2:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        try:
            model = sm.tsa.DynamicFactor(
                endog=endog_train.to_numpy(dtype=float),
                k_factors=int(k_factors),
                factor_order=int(factor_order),
                error_order=0,
            )
            res = model.fit(method="lbfgs", disp=False, maxiter=200)
            fc = res.get_forecast(steps=1)
            mean_std = float(fc.predicted_mean[0, 0])
            var = fc.var_pred_mean
            var_std = float(var[0, 0, 0]) if var.ndim == 3 else float(var[0, 0])
            smoothed = res.smoothed_state
            # smoothed_state shape: (n_states, T)
            factor_path = smoothed[0]
            factor_last = float(factor_path[-1])
            factor_diff = (
                float(factor_path[-1] - factor_path[-2]) if len(factor_path) >= 2 else float("nan")
            )
            factor_slope_3m = (
                float((factor_path[-1] - factor_path[-4]) / 3.0)
                if len(factor_path) >= 4 else float("nan")
            )
            out.update(
                {
                    "predicted_std_y_mom": mean_std,
                    "predicted_var_y_mom": var_std,
                    "factor_estimate": factor_last,
                    "factor_diff": factor_diff,
                    "factor_slope_3m": factor_slope_3m,
                    "fit_succeeded": True,
                }
            )
        except Exception:
            pass
    return out


def run_dfm_factor_sidecar(
    *,
    target_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2010-01",
    min_train: int = 84,
    target_space: str | None = None,
    model_id: str | None = None,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = "local_sidecar_once",
    include_snapshots: bool = True,
    max_snapshot_columns: int = 200,
    max_observables: int = 8,
    k_factors: int = 1,
    factor_order: int = 1,
    min_observable_non_nan: int = 60,
    max_target_month: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_type = str(target_type).strip().lower()
    if target_type not in {"nsa", "sa"}:
        raise ValueError(f"target_type must be 'nsa' or 'sa'; got {target_type!r}")
    target_path = target_path or _default_target_path(target_type)
    target_space = target_space or _default_target_space(target_type)
    output_dir = _resolve_output_dir(output_dir, target_type, run_id)
    model_id = model_id or DEFAULT_MODEL_ID

    design = build_sidecar_design(
        target_space=target_space,
        target_path=target_path,
        include_snapshots=include_snapshots,
        snapshot_blocks=("unifier", "stress", "labor"),
        max_snapshot_columns=max_snapshot_columns,
    )
    design = design.dropna(subset=["y_mom"]).reset_index(drop=True)
    if design.empty:
        raise RuntimeError(f"No target rows; check {target_path}")

    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()
    # Universe selection is a one-shot prior — pick observable columns using
    # only data prior to the first scored target month so it's PIT-safe at
    # every step. If the slice is empty (no pre-start rows yet), fall back to
    # the full design as a degenerate-case escape hatch.
    pre_start = design[design["ds"] < start_ts]
    universe_slice = pre_start if not pre_start.empty else design
    observable_cols = _pick_observable_columns(
        universe_slice, max_observables, min_observable_non_nan,
    )
    if len(observable_cols) < 1:
        raise RuntimeError(
            "DFM sidecar: no labor-flavored observables met the coverage threshold; "
            "lower --min-observable-non-nan or pass include_snapshots=True with broader blocks."
        )
    endog_cols = ["y_mom"] + observable_cols
    max_ts = (
        pd.Timestamp(max_target_month).to_period("M").to_timestamp()
        if max_target_month
        else design["ds"].max()
    )

    rows: list[dict[str, object]] = []
    n_fit_failed = 0
    for i in range(len(design)):
        target_ds = design.loc[i, "ds"]
        if target_ds < start_ts or target_ds > max_ts:
            continue
        if i < min_train:
            continue
        endog_full = design.loc[: i - 1, endog_cols].copy()
        # Require every endog column to have ≥ min_observable_non_nan observations
        # in the training window — else drop that column for this step.
        keep_cols = [
            c for c in endog_cols
            if int(endog_full[c].notna().sum()) >= int(min_observable_non_nan)
        ]
        if "y_mom" not in keep_cols or len(keep_cols) < 2:
            continue
        endog_train = endog_full[keep_cols].dropna(how="any").copy()
        if endog_train.shape[0] < int(min_train) // 2:
            continue
        endog_std, mu, sd = _standardize_block(endog_train)
        diag = _fit_predict_one_step(
            endog_std, k_factors=k_factors, factor_order=factor_order,
        )
        if not diag["fit_succeeded"]:
            n_fit_failed += 1
        predicted_mom_std = diag["predicted_std_y_mom"]
        # Un-standardize the y_mom forecast: pred = std·σ + μ
        predicted_mom = (
            float(predicted_mom_std * sd["y_mom"] + mu["y_mom"])
            if np.isfinite(predicted_mom_std) else float("nan")
        )
        predicted_mom_var = (
            float(diag["predicted_var_y_mom"] * (sd["y_mom"] ** 2))
            if np.isfinite(diag["predicted_var_y_mom"]) else float("nan")
        )
        last_y_mom = float(design.loc[i - 1, "y_mom"])
        actual_mom = float(design.loc[i, "y_mom"]) if np.isfinite(design.loc[i, "y_mom"]) else float("nan")
        prev_mom = float(design.loc[i - 2, "y_mom"]) if i >= 2 and np.isfinite(design.loc[i - 2, "y_mom"]) else float("nan")
        actual_accel = (
            actual_mom - last_y_mom
            if np.isfinite(actual_mom) and np.isfinite(last_y_mom)
            else float("nan")
        )
        predicted_accel = (
            predicted_mom - last_y_mom
            if np.isfinite(predicted_mom) and np.isfinite(last_y_mom)
            else float("nan")
        )
        if np.isfinite(predicted_mom_var) and predicted_mom_var > 0 and np.isfinite(predicted_mom):
            confidence = float(
                np.clip(1.0 - min(1.0, np.sqrt(predicted_mom_var) / max(abs(predicted_mom), 1.0)), 0.0, 1.0)
            )
        else:
            confidence = float("nan")
        proba_up = (
            float(1.0 / (1.0 + np.exp(-predicted_accel / max(abs(last_y_mom) + 1.0, 50.0))))
            if np.isfinite(predicted_accel)
            else float("nan")
        )
        rows.append(
            {
                "ds": target_ds,
                "trained_through": design.loc[i - 1, "ds"],
                "predicted_mom": predicted_mom,
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": (
                    float(np.sign(predicted_accel)) if np.isfinite(predicted_accel) else float("nan")
                ),
                "predicted_accel_proba_up": proba_up,
                "confidence": confidence,
                "uncertainty": (1.0 - confidence) if np.isfinite(confidence) else float("nan"),
                "predicted_mom_var": predicted_mom_var,
                "factor_estimate": diag["factor_estimate"],
                "factor_diff": diag["factor_diff"],
                "factor_slope_3m": diag["factor_slope_3m"],
                "actual_mom": actual_mom,
                "actual_accel": (actual_mom - last_y_mom)
                if (np.isfinite(actual_mom) and np.isfinite(last_y_mom))
                else float("nan"),
                "prev_mom": last_y_mom,
                "n_train": int(endog_train.shape[0]),
                "n_observables": int(endog_train.shape[1]),
                "fit_succeeded": int(bool(diag["fit_succeeded"])),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("DFM factor sidecar produced no predictions; lower --min-train or check observable coverage")

    feature_audit = feature_audit_from_frame(
        design,
        endog_cols,
        source_map=source_map_for_columns(endog_cols),
    )

    _, metrics = write_sidecar_artifacts(
        output_dir=output_dir,
        model_id=model_id,
        target_space=target_space,
        predictions=results,
        feature_audit=feature_audit,
        config={
            "start": start,
            "min_train": int(min_train),
            "target_space": target_space,
            "max_observables": int(max_observables),
            "k_factors": int(k_factors),
            "factor_order": int(factor_order),
            "min_observable_non_nan": int(min_observable_non_nan),
            "include_snapshots": bool(include_snapshots),
            "max_snapshot_columns": int(max_snapshot_columns),
            "observable_cols": observable_cols,
            "max_target_month": max_target_month,
        },
        extra_metrics={
            "n_fit_failed": int(n_fit_failed),
            "fit_success_rate": float(1.0 - n_fit_failed / max(len(results), 1)),
            "observable_cols": ",".join(observable_cols),
            "n_observables_curated": int(len(observable_cols)),
        },
        data_paths={"target_path": str(target_path)},
    )
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=None,
                        help="Explicit target parquet; defaults from --target-type.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Explicit output dir; SA falls back to "
                             "sidecars/sa/<run_id>/dfm_factor.")
    parser.add_argument("--target-type", default=DEFAULT_TARGET_TYPE,
                        choices=["nsa", "sa"],
                        help="Branch subtree the artifact lands under.")
    parser.add_argument("--run-id", default="local_sidecar_once",
                        help="Sidecar run-id directory under the branch subtree.")
    parser.add_argument("--start", default="2010-01")
    parser.add_argument("--min-train", type=int, default=84)
    parser.add_argument("--target-space", choices=["sa_revised", "nsa_revised"], default=None)
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--max-snapshot-columns", type=int, default=200)
    parser.add_argument("--max-observables", type=int, default=8)
    parser.add_argument("--k-factors", type=int, default=1)
    parser.add_argument("--factor-order", type=int, default=1)
    parser.add_argument("--min-observable-non-nan", type=int, default=60)
    parser.add_argument("--max-target-month", default=None)
    args = parser.parse_args()
    _, metrics = run_dfm_factor_sidecar(
        target_path=args.target_path,
        output_dir=args.output_dir,
        start=args.start,
        min_train=args.min_train,
        target_space=args.target_space,
        target_type=args.target_type,
        run_id=args.run_id,
        include_snapshots=not args.no_snapshots,
        max_snapshot_columns=args.max_snapshot_columns,
        max_observables=args.max_observables,
        k_factors=args.k_factors,
        factor_order=args.factor_order,
        min_observable_non_nan=args.min_observable_non_nan,
        max_target_month=args.max_target_month,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
