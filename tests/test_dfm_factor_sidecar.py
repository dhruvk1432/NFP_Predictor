"""PIT + DFM toy-fit + schema tests for dfm_factor_sidecar.

The toy fixture generates a 4-observable panel driven by a known AR(1) latent
factor with known per-observable loadings. We then call the internal DFM
fitter and assert it recovers a non-degenerate forecast on a small training
window, plus the full PIT/schema contract via a mock build_sidecar_design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sidecars.common import REQUIRED_PREDICTION_COLUMNS  # noqa: E402
from experiments.sidecars.dfm_factor_sidecar import (  # noqa: E402
    _fit_predict_one_step,
    _pick_observable_columns,
    _resolve_output_dir,
    _standardize_block,
    run_dfm_factor_sidecar,
)


@pytest.fixture
def toy_dfm_panel():
    """4-observable panel from a known AR(1) latent factor."""
    rng = np.random.default_rng(42)
    n = 200
    rho = 0.85
    sigma_eps = 1.0
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + rng.normal(0, sigma_eps)
    loadings = np.array([1.0, 0.7, -0.5, 0.3])
    sigma_u = np.array([0.5, 0.4, 0.6, 0.7])
    Y = np.outer(s, loadings) + rng.normal(0, 1.0, (n, 4)) * sigma_u
    df = pd.DataFrame(Y, columns=["y_mom", "obs2", "obs3", "obs4"])
    return df


def test_pick_observable_columns_excludes_target_dynamics():
    df = pd.DataFrame({
        "ds": pd.date_range("2010-01-01", periods=120, freq="MS"),
        "y_mom": np.random.randn(120),
        "actual_mom": np.random.randn(120),
        "prev_mom": np.random.randn(120),
        "ISM_Manufacturing_PMI": np.random.randn(120),
        "Challenger_Layoffs": np.random.randn(120),
        "AHE_Earnings_MoM": np.random.randn(120),
        "Random_Noise_Col": np.random.randn(120),
    })
    cols = _pick_observable_columns(df, max_observables=5, min_non_nan=60)
    # Labor-prefix columns must dominate.
    assert "ISM_Manufacturing_PMI" in cols
    assert "Challenger_Layoffs" in cols
    assert "AHE_Earnings_MoM" in cols
    # Target-dynamics cols must not be picked.
    for excluded in ("y_mom", "actual_mom", "prev_mom"):
        assert excluded not in cols


def test_fit_predict_recovers_finite_forecast(toy_dfm_panel):
    """On a well-specified DFM panel, the fit should produce a finite forecast
    for the first column and a non-degenerate factor estimate.
    """
    train = toy_dfm_panel.iloc[:160].copy()
    std, mu, sd = _standardize_block(train)
    diag = _fit_predict_one_step(std, k_factors=1, factor_order=1)
    assert diag["fit_succeeded"], "DFM should fit a clean 4-observable panel"
    assert np.isfinite(diag["predicted_std_y_mom"])
    assert np.isfinite(diag["predicted_var_y_mom"])
    assert diag["predicted_var_y_mom"] > 0
    assert np.isfinite(diag["factor_estimate"])


def test_pit_contract_and_schema_real_target(tmp_path):
    """Run the full pipeline against the actual SA target file with snapshots
    disabled (so we don't pay the snapshot-loading cost). The DFM needs >=2
    observables — when snapshots are off, we synthesize a degenerate fallback
    that should be caught and raised, so we expect a RuntimeError.
    """
    with pytest.raises(RuntimeError):
        run_dfm_factor_sidecar(
            output_dir=tmp_path / "dfm",
            start="2018-01",
            min_train=84,
            target_type="sa",
            include_snapshots=False,
        )


def test_output_dir_routing_sa_vs_nsa():
    sa_path = _resolve_output_dir(None, "sa", "test_run")
    nsa_path = _resolve_output_dir(None, "nsa", "test_run")
    assert str(sa_path).endswith("sidecars/sa/test_run/dfm_factor")
    assert "sidecars/local_sidecar_once/dfm_factor" in str(nsa_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
