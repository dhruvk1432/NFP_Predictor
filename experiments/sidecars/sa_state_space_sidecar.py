"""State-space SA sidecar.

PIT-safe structural time-series decomposition of total nonfarm payrolls (SA),
fit by maximum likelihood via ``statsmodels.tsa.statespace.UnobservedComponents``
on an expanding window per backtest step. Emits one-step-ahead Kalman level
forecasts (differenced to MoM) plus filter posterior variance and seasonal /
trend diagnostics.

Decomposition (notation follows Notes/State_Space_Models_1.pdf §1.1, §1.5):

    level_t   = level_{t-1} + trend_{t-1} + η_level,t
    trend_t   = trend_{t-1} + η_trend,t
    season_t  = -Σ_{j=1..s-1} season_{t-j} + η_season,t
    y_t       = level_t + season_t + β·D_covid_t + ε_t

PIT contract: at forecast step for target month M we train on y[ds < M],
i.e. ``trained_through = M - 1 month``. y[M-1] was released strictly before
y[M]'s release date, so the training set is PIT-safe.

The COVID indicator is a known exogenous (calendar) variable, not a
look-forward — at any historical month the model knows whether that month is
in 2020-03..2020-12. We pass the dummy as an exog regressor with a learned
coefficient, which lets MLE absorb the structural break instead of letting
the smoother propagate it into the level/seasonal components.
"""

from __future__ import annotations

import argparse
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

try:
    from settings import DATA_PATH, OUTPUT_DIR  # noqa: E402
except RuntimeError:
    DATA_PATH = Path("data")
    OUTPUT_DIR = Path("_output")

import warnings

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_TARGET_TYPE = "sa"
DEFAULT_MODEL_ID = "sa_state_space"
LEGACY_DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "sa_state_space"
)

COVID_DUMMY_START = pd.Timestamp("2020-03-01")
COVID_DUMMY_END = pd.Timestamp("2020-12-01")


def _default_target_path(target_type: str) -> Path:
    if target_type == "nsa":
        return DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    return DATA_PATH / "NFP_target" / "y_sa_revised.parquet"


def _default_target_space(target_type: str) -> str:
    return "nsa_revised" if target_type == "nsa" else "sa_revised"


def _resolve_output_dir(explicit: Path | None, target_type: str, run_id: str) -> Path:
    """SA always lands under sidecars/sa/<run_id>/sa_state_space/.
    NSA preserves the legacy flat path unless an explicit dir is supplied.
    """
    if explicit is not None:
        return explicit
    target_type = str(target_type).strip().lower()
    if target_type == "sa":
        return sidecar_branch_root(OUTPUT_DIR, "sa") / run_id / "sa_state_space"
    return LEGACY_DEFAULT_OUTPUT_DIR


DEFAULT_TARGET_PATH = _default_target_path(DEFAULT_TARGET_TYPE)
DEFAULT_OUTPUT_DIR = LEGACY_DEFAULT_OUTPUT_DIR


def _load_level_series(target_path: Path) -> pd.DataFrame:
    """Load (ds, y) where y is the level series (total nonfarm payrolls)."""
    df = pd.read_parquet(target_path)
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Target parquet {target_path} must have ['ds','y']; got {list(df.columns)}"
        )
    out = df[["ds", "y"]].copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.to_period("M").dt.to_timestamp()
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.sort_values("ds").reset_index(drop=True)


def _covid_dummy(ds: pd.Series) -> np.ndarray:
    """Return a 0/1 indicator for months inside the COVID winsorize window.

    Calendar-only — uses no future information; safe to pass as exog.
    """
    ts = pd.to_datetime(ds)
    mask = (ts >= COVID_DUMMY_START) & (ts <= COVID_DUMMY_END)
    return mask.astype(float).to_numpy().reshape(-1, 1)


def _fit_predict_one_step(
    y_train: np.ndarray,
    exog_train: np.ndarray,
    exog_forecast: np.ndarray,
) -> dict[str, float]:
    """Fit UC on training window and return one-step-ahead forecast diagnostics.

    Returns dict with predicted_level, predicted_level_var, seasonal_std,
    level_slope, fit_succeeded.
    """
    out = {
        "predicted_level": float("nan"),
        "predicted_level_var": float("nan"),
        "seasonal_std": float("nan"),
        "level_slope": float("nan"),
        "fit_succeeded": False,
    }
    if len(y_train) < 36 or not np.all(np.isfinite(y_train)):
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        try:
            model = sm.tsa.UnobservedComponents(
                endog=y_train,
                exog=exog_train,
                level="local linear trend",
                seasonal=12,
                stochastic_seasonal=True,
                irregular=True,
            )
            res = model.fit(method="lbfgs", disp=False, maxiter=200)
            fc = res.get_forecast(steps=1, exog=exog_forecast.reshape(1, -1))
            mean = float(fc.predicted_mean[0])
            var = float(fc.var_pred_mean[0]) if hasattr(fc, "var_pred_mean") else float("nan")
            smoothed = res.smoothed_state
            # Row 0 = level, row 1 = trend in local-linear-trend specification.
            seasonal_rows = smoothed[2:14] if smoothed.shape[0] >= 14 else None
            seasonal_std = (
                float(np.nanstd(seasonal_rows[0])) if seasonal_rows is not None else float("nan")
            )
            level_slope = float(smoothed[1, -1]) if smoothed.shape[0] >= 2 else float("nan")
            out.update(
                {
                    "predicted_level": mean,
                    "predicted_level_var": var,
                    "seasonal_std": seasonal_std,
                    "level_slope": level_slope,
                    "fit_succeeded": True,
                }
            )
        except Exception:
            pass
    return out


def run_sa_state_space_sidecar(
    *,
    target_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2010-01",
    min_train: int = 84,
    target_space: str | None = None,
    model_id: str | None = None,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = "local_sidecar_once",
    use_covid_dummy: bool = True,
    max_target_month: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Walk-forward State-space SA sidecar.

    For each target month ds >= start with at least min_train history, fits the
    UC model on y[ds_train < ds] and emits a one-step-ahead level forecast.
    PIT contract: trained_through is set to the last training month, which is
    strictly < ds.
    """
    target_type = str(target_type).strip().lower()
    if target_type not in {"nsa", "sa"}:
        raise ValueError(f"target_type must be 'nsa' or 'sa'; got {target_type!r}")
    target_path = target_path or _default_target_path(target_type)
    target_space = target_space or _default_target_space(target_type)
    output_dir = _resolve_output_dir(output_dir, target_type, run_id)
    model_id = model_id or DEFAULT_MODEL_ID

    series = _load_level_series(target_path)
    series = series.dropna(subset=["y"]).reset_index(drop=True)
    if series.empty:
        raise RuntimeError(f"No observations in {target_path}")

    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()
    max_ts = (
        pd.Timestamp(max_target_month).to_period("M").to_timestamp()
        if max_target_month
        else series["ds"].max()
    )

    exog_all = _covid_dummy(series["ds"]) if use_covid_dummy else None

    rows: list[dict[str, object]] = []
    n_fit_failed = 0
    for i in range(len(series)):
        target_ds = series.loc[i, "ds"]
        if target_ds < start_ts or target_ds > max_ts:
            continue
        if i < min_train:
            continue
        y_train = series.loc[: i - 1, "y"].to_numpy(dtype=float)
        if use_covid_dummy:
            exog_train = exog_all[:i]
            exog_forecast = exog_all[i]
        else:
            exog_train = None
            exog_forecast = np.zeros(1)
        diag = _fit_predict_one_step(y_train, exog_train, exog_forecast)
        if not diag["fit_succeeded"]:
            n_fit_failed += 1
        last_y = float(series.loc[i - 1, "y"])
        actual_y = float(series.loc[i, "y"]) if np.isfinite(series.loc[i, "y"]) else float("nan")
        predicted_level = diag["predicted_level"]
        predicted_mom = predicted_level - last_y if np.isfinite(predicted_level) else float("nan")
        actual_mom = actual_y - last_y if np.isfinite(actual_y) else float("nan")
        if i >= 2:
            prev_y = float(series.loc[i - 2, "y"])
            prev_mom = last_y - prev_y if np.isfinite(prev_y) else float("nan")
            actual_accel = (
                actual_mom - prev_mom
                if np.isfinite(actual_mom) and np.isfinite(prev_mom)
                else float("nan")
            )
            predicted_accel = (
                predicted_mom - prev_mom
                if np.isfinite(predicted_mom) and np.isfinite(prev_mom)
                else float("nan")
            )
        else:
            prev_mom = float("nan")
            actual_accel = float("nan")
            predicted_accel = float("nan")
        # Confidence proxy from filter variance: high variance => low confidence.
        pred_var = diag["predicted_level_var"]
        if np.isfinite(pred_var) and pred_var > 0 and np.isfinite(predicted_mom):
            confidence = float(np.clip(1.0 - min(1.0, np.sqrt(pred_var) / max(abs(predicted_mom), 1.0)), 0.0, 1.0))
        else:
            confidence = float("nan")
        proba_up = (
            float(1.0 / (1.0 + np.exp(-predicted_accel / max(abs(prev_mom) + 1.0, 50.0))))
            if np.isfinite(predicted_accel)
            else float("nan")
        )
        rows.append(
            {
                "ds": target_ds,
                "trained_through": series.loc[i - 1, "ds"],
                "predicted_mom": predicted_mom,
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": (
                    float(np.sign(predicted_accel)) if np.isfinite(predicted_accel) else float("nan")
                ),
                "predicted_accel_proba_up": proba_up,
                "confidence": confidence,
                "uncertainty": (1.0 - confidence) if np.isfinite(confidence) else float("nan"),
                "predicted_level": predicted_level,
                "predicted_mom_var": pred_var,
                "seasonal_strength": diag["seasonal_std"],
                "level_slope": diag["level_slope"],
                "actual_mom": actual_mom,
                "actual_accel": actual_accel,
                "prev_mom": prev_mom,
                "n_train": int(len(y_train)),
                "fit_succeeded": int(bool(diag["fit_succeeded"])),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("State-space SA sidecar produced no predictions; lower --start or --min-train")

    feature_audit = feature_audit_from_frame(
        series.rename(columns={"y": "endog_level"}),
        ["endog_level"],
        source_map={"endog_level": str(target_path)},
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
            "use_covid_dummy": bool(use_covid_dummy),
            "max_target_month": max_target_month,
        },
        extra_metrics={
            "n_fit_failed": int(n_fit_failed),
            "fit_success_rate": float(1.0 - n_fit_failed / max(len(results), 1)),
            "mean_predicted_mom_var": float(
                np.nanmean(pd.to_numeric(results["predicted_mom_var"], errors="coerce"))
            ),
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
                             "sidecars/sa/<run_id>/sa_state_space.")
    parser.add_argument("--target-type", default=DEFAULT_TARGET_TYPE,
                        choices=["nsa", "sa"],
                        help="Branch subtree the artifact lands under.")
    parser.add_argument("--run-id", default="local_sidecar_once",
                        help="Sidecar run-id directory under the branch subtree.")
    parser.add_argument("--start", default="2010-01")
    parser.add_argument("--min-train", type=int, default=84)
    parser.add_argument("--target-space", choices=["sa_revised", "nsa_revised"], default=None,
                        help="Target space override; defaults from --target-type.")
    parser.add_argument("--no-covid-dummy", action="store_true")
    parser.add_argument("--max-target-month", default=None,
                        help="Optional upper bound on target ds (YYYY-MM).")
    args = parser.parse_args()
    _, metrics = run_sa_state_space_sidecar(
        target_path=args.target_path,
        output_dir=args.output_dir,
        start=args.start,
        min_train=args.min_train,
        target_space=args.target_space,
        target_type=args.target_type,
        run_id=args.run_id,
        use_covid_dummy=not args.no_covid_dummy,
        max_target_month=args.max_target_month,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
