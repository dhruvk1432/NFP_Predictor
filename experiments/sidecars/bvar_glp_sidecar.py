"""BVAR-GLP sidecar — Minnesota prior on a compact employment VAR.

Implements the Giannone-Lenza-Primiceri (2015) closed-form conjugate
Normal-Inverse-Wishart Bayesian VAR via the **dummy-observation** stacking of
Banbura et al. (2010). The full hierarchical empirical-Bayes selection over
γ = (λ, μ, δ) is approximated by a small grid search on the marginal
log-likelihood per backtest step (refit every K months).

Notation (matches Notes/GLP_2015.pdf §III):

    y_t = c + B_1 y_{t-1} + ... + B_p y_{t-p} + ε_t,   ε_t ~ N(0, Σ)

Prior:
    vec(B) | Σ ~ N(b_0, Σ ⊗ Ω_0),  Σ ~ IW(Ψ_0, d_0)

The Minnesota prior centers diagonal AR(1) elements at δ_i (≈1 for I(1)
series, 0 for stationary), off-diagonals at 0, with column-i scale
proportional to σ_i (univariate AR(p) residual std) and shrinkage controlled
by λ. We implement it by appending dummy rows to (Y, X) so the joint OLS on
the stacked system has the form of the conjugate posterior mean.

PIT contract: for each target month M, we train on the full pre-M panel and
forecast ds=M. ``trained_through`` is the last training month (< M).
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


DEFAULT_TARGET_TYPE = "sa"
DEFAULT_MODEL_ID = "bvar_glp"
LEGACY_DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "bvar_glp"
)

# Same labor-flavored block as DFM. The BVAR's endogenous block must be
# compact (n=4..8 with ~250 obs) for hyperparameter estimation to be stable.
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
        return sidecar_branch_root(OUTPUT_DIR, "sa") / run_id / "bvar_glp"
    return LEGACY_DEFAULT_OUTPUT_DIR


DEFAULT_TARGET_PATH = _default_target_path(DEFAULT_TARGET_TYPE)
DEFAULT_OUTPUT_DIR = LEGACY_DEFAULT_OUTPUT_DIR


# --------------------------------------------------------------------------- #
# Minnesota prior — dummy-observation construction (Banbura et al. 2010 §2.1)
# --------------------------------------------------------------------------- #

def _ar_marginal_stds(Y: np.ndarray, p: int) -> np.ndarray:
    """Per-column univariate AR(p) residual std, the canonical scale for the
    Minnesota prior's column-i tightness."""
    T, n = Y.shape
    sigmas = np.zeros(n)
    for i in range(n):
        y = Y[:, i]
        if T <= p + 1:
            sigmas[i] = float(np.nanstd(y))
            continue
        X = np.column_stack([y[p - k - 1 : T - k - 1] for k in range(p)] + [np.ones(T - p)])
        try:
            beta, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
            res = y[p:] - X @ beta
            sigma = float(np.std(res, ddof=p + 1))
            sigmas[i] = sigma if sigma > 0 else float(np.nanstd(y))
        except Exception:
            sigmas[i] = float(np.nanstd(y))
    return sigmas


def _minnesota_dummies(
    Y: np.ndarray,
    p: int,
    *,
    lam: float,
    mu: float,
    delta: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y_d, X_d) dummy-observation pair encoding the Minnesota prior.

    Encodes three priors via stacking:
      (1) own- and cross-lag shrinkage with tightness λ.
      (2) sum-of-coefficients ("no-cointegration") with tightness μ.
      (3) initial-observation dummy with the same tightness μ.

    Reference: Banbura, Giannone & Reichlin (2010), eqs. (12)-(14).
    """
    T, n = Y.shape
    K = n * p + 1  # regressors = constant + np lags

    # (1) Coefficient prior dummies — covers own/cross lags and intercept.
    # Construct Y_d1 (n*p+1, n) and X_d1 (n*p+1, K).
    Y_d1 = np.zeros((n * p + 1, n))
    X_d1 = np.zeros((n * p + 1, K))
    # Own-coef dummies: lag-1 of variable i centered at δ_i.
    Y_d1[:n, :] = np.diag(delta * sigma) / lam
    # X side: each block lag-h has 1/(lam·h) on the diagonal scaled by σ.
    for h in range(1, p + 1):
        start = (h - 1) * n
        # X cols are stacked as [lag1_vars, lag2_vars, ..., const]
        X_d1[: n, start : start + n] = np.diag(sigma) / (lam * h) if h == 1 else np.zeros((n, n))
    # Cross/higher-lag tightness: penalize all lag-h coefficients with h^d scaling.
    # Add separate dummy rows for higher lags so we control h-decay.
    extra_rows = []
    extra_Yd = []
    for h in range(2, p + 1):
        Mk = np.zeros((n, K))
        Mk[:, (h - 1) * n : h * n] = np.diag(sigma) / (lam * h)
        extra_rows.append(Mk)
        extra_Yd.append(np.zeros((n, n)))
    if extra_rows:
        X_d1 = np.vstack([X_d1[:n], *extra_rows, X_d1[n:]])
        Y_d1 = np.vstack([Y_d1[:n], *extra_Yd, Y_d1[n:]])

    # Constant prior — very loose by default (small contribution).
    # The last row of Y_d1 / X_d1 is for the intercept.
    eps_c = 1e-3
    X_d1[-1, -1] = eps_c
    Y_d1[-1, :] = 0.0

    # (2) Sum-of-coefficients dummies (Doan, Litterman & Sims 1984).
    y_bar = np.nanmean(Y[: min(p, T), :], axis=0) if p > 0 else np.nanmean(Y, axis=0)
    Y_d2 = np.diag(delta * y_bar) / mu
    X_d2 = np.zeros((n, K))
    for h in range(1, p + 1):
        X_d2[:, (h - 1) * n : h * n] = np.diag(delta * y_bar) / mu

    # (3) Dummy-initial-observation (Sims 1993).
    Y_d3 = (y_bar / mu).reshape(1, -1)
    X_d3 = np.zeros((1, K))
    for h in range(1, p + 1):
        X_d3[0, (h - 1) * n : h * n] = y_bar / mu
    X_d3[0, -1] = 1.0 / mu

    Y_d = np.vstack([Y_d1, Y_d2, Y_d3])
    X_d = np.vstack([X_d1, X_d2, X_d3])
    return Y_d, X_d


def _stack_var_regressors(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y_reg, X_reg) for OLS form of VAR(p).

    Y_reg: (T-p, n), X_reg: (T-p, n*p+1) with constant in last column.
    """
    T, n = Y.shape
    if T <= p:
        return np.zeros((0, n)), np.zeros((0, n * p + 1))
    Y_reg = Y[p:]
    X_blocks = [Y[p - h : T - h] for h in range(1, p + 1)]
    X_reg = np.column_stack(X_blocks + [np.ones(T - p)])
    return Y_reg, X_reg


def _bvar_posterior_predict(
    Y_train: np.ndarray,
    *,
    p: int,
    lam: float,
    mu: float,
    delta: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit BVAR via dummy-obs Minnesota prior and return 1-step-ahead forecast.

    Returns dict with predicted_mean (n,), predicted_var (n,), B_hat
    ((np+1, n)), Σ_hat (n, n), success flag, marginal_loglik (float).
    """
    T, n = Y_train.shape
    if T <= p + 3:
        return {"success": False}
    sigmas = _ar_marginal_stds(Y_train, p)
    if delta is None:
        # Use 1 for series that look I(1) (low first-difference variance ratio),
        # 0 otherwise. Pragmatic default: 1 for all (random-walk-ish prior).
        delta = np.ones(n)
    try:
        Y_d, X_d = _minnesota_dummies(
            Y_train, p, lam=lam, mu=mu, delta=delta, sigma=sigmas,
        )
        Y_reg, X_reg = _stack_var_regressors(Y_train, p)
        Y_star = np.vstack([Y_reg, Y_d])
        X_star = np.vstack([X_reg, X_d])
        # Posterior mean of B is OLS on the stacked system.
        XtX = X_star.T @ X_star
        try:
            XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(XtX.shape[0]))
        except np.linalg.LinAlgError:
            return {"success": False}
        B_hat = XtX_inv @ X_star.T @ Y_star
        resid = Y_star - X_star @ B_hat
        T_star = Y_star.shape[0]
        df = max(T_star - X_star.shape[1], 1)
        Sigma_hat = (resid.T @ resid) / df
        # 1-step-ahead forecast: x_T = [y_T, y_{T-1}, ..., y_{T-p+1}, 1]
        if T < p:
            return {"success": False}
        x_T = np.concatenate(
            [Y_train[T - h] for h in range(1, p + 1)] + [np.array([1.0])]
        )
        y_pred = x_T @ B_hat
        # Posterior predictive variance: Σ̂ * (1 + x_T'(X*'X*)^-1 x_T)
        scale = 1.0 + float(x_T @ XtX_inv @ x_T)
        Sigma_pred = Sigma_hat * scale
        # Marginal log-likelihood (proportional, ignoring constants):
        #   -T/2 log |Σ̂| - (T-K)/2 log(scale_factor)
        # For γ selection we use proxy: -sum(log diag(Σ̂_orig)) — penalizes
        # overfit but cheap to compute.
        try:
            sign, logdet = np.linalg.slogdet(Sigma_hat)
            marginal_loglik = float(-0.5 * Y_reg.shape[0] * logdet) if sign > 0 else float("-inf")
        except Exception:
            marginal_loglik = float("nan")
        return {
            "success": True,
            "predicted_mean": y_pred,
            "predicted_var": np.diag(Sigma_pred),
            "B_hat": B_hat,
            "Sigma_hat": Sigma_hat,
            "marginal_loglik": marginal_loglik,
            "x_T": x_T,
        }
    except Exception:
        return {"success": False}


def _select_lambda_by_marginal_loglik(
    Y_train: np.ndarray,
    *,
    p: int,
    lambda_grid: tuple[float, ...],
    mu: float,
) -> float:
    """Pick λ that maximizes the marginal log-likelihood proxy on the grid."""
    best_lam = lambda_grid[len(lambda_grid) // 2]
    best_ll = float("-inf")
    for lam in lambda_grid:
        out = _bvar_posterior_predict(Y_train, p=p, lam=lam, mu=mu)
        if not out.get("success"):
            continue
        ll = out.get("marginal_loglik", float("-inf"))
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_lam = lam
    return float(best_lam)


def _pick_observable_columns(
    design: pd.DataFrame,
    max_observables: int,
    min_non_nan: int,
) -> list[str]:
    snapshot_cols = [
        c for c in design.columns
        if c.startswith(LABOR_FLAVOR_PREFIXES) and pd.api.types.is_numeric_dtype(design[c])
    ]
    if not snapshot_cols:
        snapshot_cols = [
            c for c in select_numeric_feature_cols(design)
            if not c.startswith(("sa_", "nsa_", "prev_", "actual_", "target_"))
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


def run_bvar_glp_sidecar(
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
    max_observables: int = 6,
    var_order: int = 2,
    lambda_grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5),
    mu: float = 5.0,
    refit_every: int = 6,
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
    # One-shot universe-selection prior on pre-start rows only — keeps the
    # endogenous block stable across backtest steps while remaining PIT-safe.
    pre_start = design[design["ds"] < start_ts]
    universe_slice = pre_start if not pre_start.empty else design
    observable_cols = _pick_observable_columns(
        universe_slice, max_observables, min_observable_non_nan,
    )
    if len(observable_cols) < 1:
        raise RuntimeError(
            "BVAR-GLP sidecar: no labor-flavored observables met the coverage threshold."
        )
    endog_cols = ["y_mom"] + observable_cols
    max_ts = (
        pd.Timestamp(max_target_month).to_period("M").to_timestamp()
        if max_target_month
        else design["ds"].max()
    )

    rows: list[dict[str, object]] = []
    last_lambda_select_at = -10**9
    current_lambda = lambda_grid[len(lambda_grid) // 2]
    n_fit_failed = 0
    for i in range(len(design)):
        target_ds = design.loc[i, "ds"]
        if target_ds < start_ts or target_ds > max_ts:
            continue
        if i < min_train:
            continue
        endog_full = design.loc[: i - 1, endog_cols].copy()
        keep_cols = [
            c for c in endog_cols
            if int(endog_full[c].notna().sum()) >= int(min_observable_non_nan)
        ]
        if "y_mom" not in keep_cols or len(keep_cols) < 2:
            continue
        endog_train = endog_full[keep_cols].dropna(how="any").to_numpy(dtype=float)
        if endog_train.shape[0] < int(min_train) // 2 or endog_train.shape[0] <= var_order + 3:
            continue
        # Re-tune λ every refit_every months on the marginal-likelihood proxy.
        if i - last_lambda_select_at >= int(refit_every):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                current_lambda = _select_lambda_by_marginal_loglik(
                    endog_train, p=var_order, lambda_grid=lambda_grid, mu=mu,
                )
            last_lambda_select_at = i

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = _bvar_posterior_predict(
                endog_train, p=var_order, lam=current_lambda, mu=mu,
            )
        if not fit.get("success"):
            n_fit_failed += 1
            continue
        y_pred = fit["predicted_mean"]
        y_var = fit["predicted_var"]
        predicted_mom = float(y_pred[0])  # y_mom is the first column
        predicted_mom_var = float(y_var[0])
        last_y_mom = float(design.loc[i - 1, "y_mom"])
        actual_mom = float(design.loc[i, "y_mom"]) if np.isfinite(design.loc[i, "y_mom"]) else float("nan")
        actual_accel = (
            actual_mom - last_y_mom
            if np.isfinite(actual_mom) else float("nan")
        )
        predicted_accel = predicted_mom - last_y_mom
        if predicted_mom_var > 0:
            confidence = float(np.clip(
                1.0 - min(1.0, np.sqrt(predicted_mom_var) / max(abs(predicted_mom), 1.0)),
                0.0, 1.0,
            ))
        else:
            confidence = float("nan")
        proba_up = float(1.0 / (1.0 + np.exp(-predicted_accel / max(abs(last_y_mom) + 1.0, 50.0))))
        rows.append(
            {
                "ds": target_ds,
                "trained_through": design.loc[i - 1, "ds"],
                "predicted_mom": predicted_mom,
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": float(np.sign(predicted_accel)),
                "predicted_accel_proba_up": proba_up,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence if np.isfinite(confidence) else float("nan"),
                "predicted_mom_var": predicted_mom_var,
                "lambda_selected": current_lambda,
                "var_order": int(var_order),
                "actual_mom": actual_mom,
                "actual_accel": actual_accel,
                "prev_mom": last_y_mom,
                "n_train": int(endog_train.shape[0]),
                "n_endog": int(endog_train.shape[1]),
                "marginal_loglik_proxy": float(fit.get("marginal_loglik", float("nan"))),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("BVAR-GLP sidecar produced no predictions")

    feature_audit = feature_audit_from_frame(
        design, endog_cols, source_map=source_map_for_columns(endog_cols),
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
            "var_order": int(var_order),
            "lambda_grid": list(lambda_grid),
            "mu": float(mu),
            "refit_every": int(refit_every),
            "min_observable_non_nan": int(min_observable_non_nan),
            "include_snapshots": bool(include_snapshots),
            "observable_cols": observable_cols,
        },
        extra_metrics={
            "n_fit_failed": int(n_fit_failed),
            "fit_success_rate": float(1.0 - n_fit_failed / max(len(results) + n_fit_failed, 1)),
            "observable_cols": ",".join(observable_cols),
            "n_endog": int(1 + len(observable_cols)),
            "mean_lambda_selected": float(results["lambda_selected"].mean()),
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
                             "sidecars/sa/<run_id>/bvar_glp.")
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
    parser.add_argument("--max-observables", type=int, default=6)
    parser.add_argument("--var-order", type=int, default=2)
    parser.add_argument("--mu", type=float, default=5.0)
    parser.add_argument("--refit-every", type=int, default=6)
    parser.add_argument("--min-observable-non-nan", type=int, default=60)
    parser.add_argument("--max-target-month", default=None)
    args = parser.parse_args()
    _, metrics = run_bvar_glp_sidecar(
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
        var_order=args.var_order,
        mu=args.mu,
        refit_every=args.refit_every,
        min_observable_non_nan=args.min_observable_non_nan,
        max_target_month=args.max_target_month,
    )
    import json

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
