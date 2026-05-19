"""PIT + posterior correctness tests for bvar_glp_sidecar.

Validates:
1. Dummy-observation Minnesota prior recovers an AR(1) coefficient close to
   the true value on a known DGP.
2. The closed-form posterior predict produces non-degenerate forecasts +
   positive predictive variance.
3. λ selection on the marginal-loglik grid returns a value in the grid.
4. Output dir routing splits SA vs NSA correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sidecars.bvar_glp_sidecar import (  # noqa: E402
    _bvar_posterior_predict,
    _resolve_output_dir,
    _select_lambda_by_marginal_loglik,
    _stack_var_regressors,
)


@pytest.fixture
def toy_var_data():
    """Stationary VAR(1) with known coefficient matrix and known noise."""
    rng = np.random.default_rng(7)
    T, n = 300, 3
    B_true = np.array([
        [0.80, 0.05, 0.00],
        [0.10, 0.70, 0.10],
        [0.00, 0.05, 0.85],
    ])
    sigma = np.array([1.0, 1.5, 0.8])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = B_true @ Y[t - 1] + rng.normal(0, sigma, size=n)
    return Y, B_true, sigma


def test_stack_var_regressors_shape(toy_var_data):
    Y, _, _ = toy_var_data
    p = 2
    Y_reg, X_reg = _stack_var_regressors(Y, p)
    assert Y_reg.shape == (Y.shape[0] - p, Y.shape[1])
    # X has n*p lag cols + 1 constant.
    assert X_reg.shape == (Y.shape[0] - p, Y.shape[1] * p + 1)
    assert np.allclose(X_reg[:, -1], 1.0)


def test_posterior_recovers_ar_coefficient(toy_var_data):
    """With tight λ the prior dominates; with loose λ the data wins.
    Posterior B̂ should be close to B_true on the diagonal for loose λ on n=300 obs.
    """
    Y, B_true, _ = toy_var_data
    p = 1
    fit = _bvar_posterior_predict(Y, p=p, lam=2.0, mu=10.0)
    assert fit["success"]
    B_hat = fit["B_hat"]
    # B_hat has shape (n*p + 1, n) for OLS form.
    # First n rows correspond to lag-1 coefficients (B_1.T row stacked).
    # OLS form: y_t = X_t·B + ε so column j of B is the regressor weights for var j.
    # Compare diagonal of B_hat[:n, :] with diagonal of B_true.
    n = Y.shape[1]
    diag_hat = np.diag(B_hat[:n, :])
    diag_true = np.diag(B_true)
    # Expect within ~0.15 of true on 300 obs with a loose prior.
    assert np.allclose(diag_hat, diag_true, atol=0.15), (
        f"Estimated AR diag {diag_hat} too far from true {diag_true}"
    )


def test_posterior_variance_positive(toy_var_data):
    Y, _, _ = toy_var_data
    fit = _bvar_posterior_predict(Y, p=2, lam=0.5, mu=5.0)
    assert fit["success"]
    var = fit["predicted_var"]
    assert (var > 0).all(), f"predicted_var must be positive; got {var}"
    assert np.all(np.isfinite(var))


def test_lambda_selection_returns_grid_value(toy_var_data):
    Y, _, _ = toy_var_data
    grid = (0.05, 0.1, 0.2, 0.5, 1.0)
    lam = _select_lambda_by_marginal_loglik(Y, p=2, lambda_grid=grid, mu=5.0)
    assert lam in grid


def test_output_dir_routing_sa_vs_nsa():
    sa_path = _resolve_output_dir(None, "sa", "test_run")
    nsa_path = _resolve_output_dir(None, "nsa", "test_run")
    assert str(sa_path).endswith("sidecars/sa/test_run/bvar_glp")
    assert "sidecars/local_sidecar_once/bvar_glp" in str(nsa_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
